import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_RAG_PARQUET,
    TRAIN_RAG_PARQUET,
    VAL_RAG_PARQUET,
)
from src.drift import compute_drift_metrics
from src.evaluate import (
    compute_bootstrap_auc_ci,
    compute_ece,
    compute_paired_bootstrap_auc_diff,
    compute_precision_at_k,
)
from src.explain import ModelExplainer
from src.fairness import audit_subgroup_metrics, train_blinded_fairness_model
from src.train import prepare_lgbm_data

FROZEN_JSON_PATH = MODELS_DIR / "FROZEN_SELECTION.json"
LOCK_FILE_PATH = METRICS_DIR / "test_eval.lock"
TEST_RESULTS_JSON = METRICS_DIR / "test_results.json"
COHORT_DEMO_DIR = REPORTS_DIR / "demo"
COHORT_SCORED_PARQUET = COHORT_DEMO_DIR / "cohort_scored.parquet"


def run_final_test_evaluation(force_ignore_lock: bool = False) -> Dict:
    """Execute final test split evaluation exactly once, locked against repetition."""
    print("=== STAGE 2: Final Test Evaluation ===")

    # 1. Guard against un-frozen evaluation
    if not FROZEN_JSON_PATH.exists():
        raise RuntimeError(
            "TEST EVALUATION BLOCKED: models/FROZEN_SELECTION.json is missing! "
            "You must complete Stage 1 model freeze on validation before touching test data."
        )

    # 2. Guard against repeated evaluation
    if LOCK_FILE_PATH.exists() and not force_ignore_lock:
        with open(LOCK_FILE_PATH, "r") as f:
            lock_info = f.read()
        raise RuntimeError(
            f"TEST EVALUATION BLOCKED (LOCKED): Final test split has already been evaluated!\n"
            f"Lock details:\n{lock_info}\n"
            f"Repeated evaluation is prohibited to avoid p-hacking and statistical leakage."
        )

    # 3. Create lock file
    now_utc = datetime.now(timezone.utc).isoformat()
    with open(FROZEN_JSON_PATH, "r", encoding="utf-8") as f:
        frozen = json.load(f)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    COHORT_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOCK_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("TEST EVALUATION LOCK ACQUIRED\n")
        f.write(f"Timestamp: {now_utc}\n")
        f.write(f"Git Commit: {frozen.get('git_commit_hash', 'unknown')}\n")
        f.write(f"Model: {frozen.get('model_name')}\n")

    print(f"Acquired test evaluation lock at {LOCK_FILE_PATH}")

    # 4. Load models and feature schema
    final_dir = MODELS_DIR / "final"
    model = joblib.load(final_dir / "model.joblib")
    calibrator = joblib.load(final_dir / "calibrator.joblib")
    base_model = joblib.load(MODELS_DIR / "lgbm_base.joblib")

    with open(final_dir / "feature_list.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    rag_num_cols = meta["numeric_features"]
    cat_cols = meta["categorical_features"]
    feature_cols = meta["all_features"]
    base_num_cols = [c for c in rag_num_cols if c != "rag_struct_rate_k20"]

    # 5. Load TEST split
    print(f"Loading held-out TEST dataset ({TEST_RAG_PARQUET})...")
    test_df = pd.read_parquet(TEST_RAG_PARQUET)
    y_test = test_df["readmitted_30d"].values.astype(int)
    n_test = len(test_df)
    print(
        f"Test cohort: {n_test:,} encounters, {y_test.sum():,} positive readmissions ({y_test.mean():.2%})"
    )

    # 6. Predict and Calibrate
    X_test_rag = prepare_lgbm_data(test_df, rag_num_cols, cat_cols)
    X_test_base = prepare_lgbm_data(test_df, base_num_cols, cat_cols)

    raw_test_probs = model.predict_proba(X_test_rag)[:, 1]
    cal_test_probs = calibrator.transform(raw_test_probs)

    raw_base_probs = base_model.predict_proba(X_test_base)[:, 1]
    cal_base_probs = calibrator.transform(raw_base_probs)

    # 7. Metrics at Frozen Decision Threshold
    f1_th = frozen["thresholds"]["optimal_f1_decision_threshold"]
    th_5 = frozen["thresholds"]["capacity_top_5pct_threshold"]
    th_10 = frozen["thresholds"]["capacity_top_10pct_threshold"]
    th_20 = frozen["thresholds"]["capacity_top_20pct_threshold"]

    y_pred_frozen = (cal_test_probs >= f1_th).astype(int)

    test_auc = float(roc_auc_score(y_test, cal_test_probs))
    ci_low, ci_high = compute_bootstrap_auc_ci(
        y_test, cal_test_probs, n_bootstraps=1000, seed=RANDOM_SEED
    )

    test_f1 = float(f1_score(y_test, y_pred_frozen, zero_division=0))
    test_prec = float(precision_score(y_test, y_pred_frozen, zero_division=0))
    test_rec = float(recall_score(y_test, y_pred_frozen, zero_division=0))
    test_brier = float(brier_score_loss(y_test, cal_test_probs))
    test_ece = compute_ece(y_test, cal_test_probs)

    # Capacity Top-K% Metrics
    prec_5, lift_5, rec_5 = compute_precision_at_k(y_test, cal_test_probs, top_k_ratio=0.05)
    prec_10, lift_10, rec_10 = compute_precision_at_k(y_test, cal_test_probs, top_k_ratio=0.10)
    prec_20, lift_20, rec_20 = compute_precision_at_k(y_test, cal_test_probs, top_k_ratio=0.20)

    # Paired Bootstrap Difference (With-RAG vs Without-RAG) on TEST
    paired_diff_info = compute_paired_bootstrap_auc_diff(
        y_test, cal_base_probs, cal_test_probs, n_bootstraps=1000, seed=RANDOM_SEED
    )

    print("\n--- TEST SPLIT EVALUATION RESULTS ---")
    print(f"AUC-ROC: {test_auc:.4f} (95% Bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}])")
    print(
        f"At Frozen F1 Threshold ({f1_th:.4f}): F1={test_f1:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}"
    )
    print(f"Capacity Top 5%:  Prec={prec_5:.4f}, Lift={lift_5:.2f}x, Rec={rec_5:.4f}")
    print(f"Capacity Top 10%: Prec={prec_10:.4f}, Lift={lift_10:.2f}x, Rec={rec_10:.4f}")
    print(f"Capacity Top 20%: Prec={prec_20:.4f}, Lift={lift_20:.2f}x, Rec={rec_20:.4f}")
    print(f"Calibration: Brier={test_brier:.4f}, ECE={test_ece:.4f}")
    print(
        f"Paired Delta AUC (RAG - Base): {paired_diff_info['observed_diff']:+.4f} (95% CI: {paired_diff_info['ci_95']}, Crosses 0: {paired_diff_info['crosses_zero']})"
    )

    # 8. Diagnostic Figures on TEST
    # ROC and PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fpr, tpr, _ = roc_curve(y_test, cal_test_probs)
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"LightGBM + RAG (AUC = {test_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance (AUC = 0.50)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("TEST Split ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    precs_pr, recs_pr, _ = precision_recall_curve(y_test, cal_test_probs)
    ax2.plot(
        recs_pr,
        precs_pr,
        color="#2b8a3e",
        lw=2,
        label=f"LightGBM + RAG (PR-AUC = {auc(recs_pr, precs_pr):.4f})",
    )
    ax2.axhline(
        y_test.mean(), color="gray", linestyle="--", label=f"Prevalence ({y_test.mean():.2%})"
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("TEST Split Precision-Recall Curve")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "test_roc_pr_curve.png", dpi=300)
    plt.close(fig)

    # Test Calibration Curve
    fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
    pt_uncal, pp_uncal = calibration_curve(y_test, raw_test_probs, n_bins=10)
    pt_cal, pp_cal = calibration_curve(y_test, cal_test_probs, n_bins=10)
    ax_cal.plot([0, 1], [0, 1], "k--", label="Ideal Calibration")
    ax_cal.plot(
        pp_uncal,
        pt_uncal,
        "s-",
        color="#d9534f",
        label=f"Raw Probabilities (Brier: {brier_score_loss(y_test, raw_test_probs):.4f})",
    )
    ax_cal.plot(
        pp_cal,
        pt_cal,
        "o-",
        color="#2b8a3e",
        label=f"Isotonic Calibrated (Brier: {test_brier:.4f})",
    )
    ax_cal.set_xlabel("Predicted Readmission Probability")
    ax_cal.set_ylabel("Observed Readmission Rate")
    ax_cal.set_title("TEST Split Reliability Diagram")
    ax_cal.legend(loc="upper left")
    ax_cal.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_cal.savefig(FIGURES_DIR / "test_calibration_curve.png", dpi=300)
    plt.close(fig_cal)

    # 9. Create reports/demo/cohort_scored.parquet for /cohort replay API
    print("Generating demo cohort scored parquet...")
    # Classify Risk Tiers
    risk_tiers = np.where(
        cal_test_probs >= th_5,
        "Critical",
        np.where(
            cal_test_probs >= th_10,
            "High",
            np.where(cal_test_probs >= th_20, "Medium", "Low"),
        ),
    )

    cohort_df = (
        pd.DataFrame(
            {
                "encounter_id": test_df["encounter_id"].values,
                "patient_nbr": test_df["patient_nbr"].values,
                "age": (
                    test_df["age"].values
                    if "age" in test_df.columns
                    else test_df["age_num"].astype(str).values
                ),
                "gender": test_df["gender"].values,
                "race": test_df["race"].values,
                "time_in_hospital": test_df["time_in_hospital"].values,
                "number_inpatient": test_df["number_inpatient"].values,
                "number_emergency": test_df["number_emergency"].values,
                "diag_1_group": test_df["diag_1_group"].values,
                "readmitted_30d": y_test,
                "risk_score": np.round(cal_test_probs, 4),
                "risk_tier": risk_tiers,
            }
        )
        .sort_values(by="risk_score", ascending=False)
        .reset_index(drop=True)
    )

    cohort_df.to_parquet(COHORT_SCORED_PARQUET, index=False)
    print(f"Saved scored test cohort ({len(cohort_df):,} rows) to {COHORT_SCORED_PARQUET}")

    # 10. Run Explainability & Global SHAP on Validation
    print("Running SHAP explainability audit...")
    val_df = pd.read_parquet(VAL_RAG_PARQUET)
    explainer = ModelExplainer()
    explainer.generate_global_summary_plot(val_df, FIGURES_DIR / "shap_summary.png")

    sample_explanations = []
    for idx in [0, 10, 50, 100, 200]:
        pat_row = test_df.iloc[idx].to_dict()
        expl = explainer.explain_patient(pat_row, top_k=3)
        sample_explanations.append(
            {
                "encounter_id": int(pat_row.get("encounter_id", idx)),
                "risk_score": round(float(cal_test_probs[idx]), 4),
                "actual_outcome": int(y_test[idx]),
                "top_risk_increasing": expl["top_risk_increasing"],
                "top_risk_reducing": expl["top_risk_reducing"],
            }
        )
    with open(METRICS_DIR / "sample_explanations.json", "w", encoding="utf-8") as f:
        json.dump(sample_explanations, f, indent=2, default=str)

    # 11. Run Fairness Audit on TEST
    print("Running final fairness audit on TEST split...")
    train_df = pd.read_parquet(
        TRAIN_RAG_PARQUET,
        columns=list(set(feature_cols + ["race", "gender", "age_num", "readmitted_30d"])),
    )

    # Age bracket
    test_df_copy = test_df.copy()
    test_df_copy["age_bracket"] = pd.cut(
        test_df_copy["age_num"],
        bins=[0, 45, 65, 80, 110],
        labels=["<45", "45-64", "65-79", "80+"],
        right=False,
    ).astype(str)

    audit_race = audit_subgroup_metrics(
        test_df_copy, "race", y_test, cal_test_probs, th_10, reference_group="Caucasian"
    )
    audit_gender = audit_subgroup_metrics(
        test_df_copy, "gender", y_test, cal_test_probs, th_10, reference_group="Female"
    )
    audit_age = audit_subgroup_metrics(
        test_df_copy, "age_bracket", y_test, cal_test_probs, th_10, reference_group="65-79"
    )

    # Blinded benchmark model
    blinded_res = train_blinded_fairness_model(
        train_df, val_df, test_df, feature_cols, rag_num_cols, cat_cols
    )
    blind_test_probs = blinded_res["test_probs"]
    audit_race_blind = audit_subgroup_metrics(
        test_df_copy, "race", y_test, blind_test_probs, th_10, reference_group="Caucasian"
    )
    audit_gender_blind = audit_subgroup_metrics(
        test_df_copy, "gender", y_test, blind_test_probs, th_10, reference_group="Female"
    )

    fairness_report = {
        "evaluation_split": "test",
        "sample_size": n_test,
        "capacity_threshold_evaluated": th_10,
        "primary_model_test_auc": round(test_auc, 4),
        "blinded_model_test_auc": round(blinded_res["test_auc"], 4),
        "auc_delta_blinded_vs_primary": round(blinded_res["test_auc"] - test_auc, 4),
        "audits": {
            "race": audit_race,
            "gender": audit_gender,
            "age": audit_age,
        },
        "blinded_audits": {
            "race": audit_race_blind,
            "gender": audit_gender_blind,
        },
    }
    with open(METRICS_DIR / "fairness_audit.json", "w", encoding="utf-8") as f:
        json.dump(fairness_report, f, indent=2, default=str)

    # 12. Run Drift Monitoring
    print("Running drift monitoring analysis...")
    drift_report = compute_drift_metrics(
        train_df,
        val_df,
        test_df,
        meta["all_features"],
        train_scores=None,
        val_scores=None,
        test_scores=cal_test_probs,
    )
    with open(METRICS_DIR / "drift_metrics.json", "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2, default=str)

    # 13. Assemble and save test_results.json
    results = {
        "model_name": frozen["model_name"],
        "eval_timestamp": now_utc,
        "git_commit_hash": frozen["git_commit_hash"],
        "test_cohort_size": n_test,
        "test_positive_prevalence": round(float(y_test.mean()), 4),
        "auc_roc": round(test_auc, 4),
        "auc_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "frozen_decision_threshold": round(f1_th, 4),
        "f1": round(test_f1, 4),
        "precision": round(test_prec, 4),
        "recall": round(test_rec, 4),
        "brier_score": round(test_brier, 4),
        "ece": round(test_ece, 4),
        "capacity_metrics": {
            "top_5pct": {
                "threshold": round(th_5, 4),
                "precision": prec_5,
                "lift": lift_5,
                "recall": rec_5,
            },
            "top_10pct": {
                "threshold": round(th_10, 4),
                "precision": prec_10,
                "lift": lift_10,
                "recall": rec_10,
            },
            "top_20pct": {
                "threshold": round(th_20, 4),
                "precision": prec_20,
                "lift": lift_20,
                "recall": rec_20,
            },
        },
        "paired_rag_difference_test": {
            "observed_delta_auc": paired_diff_info["observed_diff"],
            "ci_95": paired_diff_info["ci_95"],
            "crosses_zero": paired_diff_info["crosses_zero"],
        },
        "validation_comparison": {
            "val_auc": frozen["validation_performance"]["auc_roc"],
            "val_auc_ci": frozen["validation_performance"]["auc_ci_95"],
            "auc_generalization_gap": round(
                test_auc - frozen["validation_performance"]["auc_roc"], 4
            ),
        },
    }

    with open(TEST_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Final test evaluation artifacts successfully saved to {TEST_RESULTS_JSON}")
    print("STAGE 2 COMPLETE.")
    return results


if __name__ == "__main__":
    run_final_test_evaluation(force_ignore_lock=False)
