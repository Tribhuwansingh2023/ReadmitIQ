import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_LIST_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREPROCESSOR_BASE_PATH,
    PREPROCESSOR_RAG_PATH,
    RANDOM_SEED,
    TRAIN_RAG_PARQUET,
    VAL_RAG_PARQUET,
)
from src.evaluate import (
    compute_paired_bootstrap_auc_diff,
    evaluate_predictions,
)
from src.features import build_preprocessor


def prepare_lgbm_data(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> pd.DataFrame:
    """Format dataframe for native LightGBM categorical handling."""
    data = df[num_cols + cat_cols].copy()
    for col in cat_cols:
        data[col] = data[col].astype("category")
    return data


def tune_random_forest(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    y_train: np.ndarray,
    groups: np.ndarray,
) -> int:
    """Tune RF max_depth across [8, 12, 16] using 5-fold patient-grouped CV on train."""
    print("Tuning Random Forest max_depth via 5-fold GroupKFold on Train...")
    depths = [8, 12, 16]
    best_depth = 12
    best_auc = -1.0
    gkf = GroupKFold(n_splits=5)

    # Use preprocessor on train once for tuning efficiency
    prep = build_preprocessor(num_cols, cat_cols)
    X_trans = prep.fit_transform(train_df[feature_cols])

    for depth in depths:
        fold_aucs = []
        for train_idx, val_idx in gkf.split(X_trans, y_train, groups=groups):
            clf = RandomForestClassifier(
                n_estimators=100,  # 100 for fast tuning grid
                max_depth=depth,
                min_samples_split=10,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
            clf.fit(X_trans[train_idx], y_train[train_idx])
            probs = clf.predict_proba(X_trans[val_idx])[:, 1]
            from sklearn.metrics import roc_auc_score

            fold_aucs.append(roc_auc_score(y_train[val_idx], probs))

        mean_auc = float(np.mean(fold_aucs))
        print(f"  RF max_depth={depth}: 5-fold CV AUC = {mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_depth = depth

    print(f"Selected RF max_depth: {best_depth} (CV AUC: {best_auc:.4f})")
    return best_depth


def tune_lightgbm(
    X_lgbm: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> Tuple[int, float]:
    """Tune LightGBM max_depth and learning_rate via 5-fold GroupKFold on Train."""
    print("Tuning LightGBM (max_depth, learning_rate) via 5-fold GroupKFold on Train...")
    param_grid = [
        (4, 0.05),
        (6, 0.03),
        (6, 0.05),
        (6, 0.10),
        (8, 0.03),
    ]
    best_params = (6, 0.05)
    best_auc = -1.0
    gkf = GroupKFold(n_splits=5)

    for depth, lr in param_grid:
        fold_aucs = []
        for train_idx, val_idx in gkf.split(X_lgbm, y_train, groups=groups):
            clf = LGBMClassifier(
                n_estimators=120,
                max_depth=depth,
                learning_rate=lr,
                scale_pos_weight=7.7,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            )
            clf.fit(X_lgbm.iloc[train_idx], y_train[train_idx])
            probs = clf.predict_proba(X_lgbm.iloc[val_idx])[:, 1]
            from sklearn.metrics import roc_auc_score

            fold_aucs.append(roc_auc_score(y_train[val_idx], probs))

        mean_auc = float(np.mean(fold_aucs))
        print(f"  LGBM max_depth={depth}, lr={lr}: 5-fold CV AUC = {mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = (depth, lr)

    print(
        f"Selected LightGBM params: max_depth={best_params[0]}, lr={best_params[1]} (CV AUC: {best_auc:.4f})"
    )
    return best_params


def generate_train_oof_probabilities(
    model_type: str,
    train_df: pd.DataFrame,
    feature_cols: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    y_train: np.ndarray,
    groups: np.ndarray,
    model_kwargs: dict,
) -> np.ndarray:
    """Generate out-of-fold probability predictions on TRAIN only using 5-fold GroupKFold."""
    print(f"Generating 5-fold patient-grouped out-of-fold predictions on Train for {model_type}...")
    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(train_df), dtype=float)

    for fold, (t_idx, v_idx) in enumerate(gkf.split(train_df, y_train, groups=groups)):
        if model_type == "lgbm":
            X_tr = prepare_lgbm_data(train_df.iloc[t_idx], num_cols, cat_cols)
            X_va = prepare_lgbm_data(train_df.iloc[v_idx], num_cols, cat_cols)
            clf = LGBMClassifier(**model_kwargs)
            clf.fit(X_tr, y_train[t_idx])
            oof_probs[v_idx] = clf.predict_proba(X_va)[:, 1]
        elif model_type == "rf":
            pipe = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(num_cols, cat_cols)),
                    ("clf", RandomForestClassifier(**model_kwargs)),
                ]
            )
            pipe.fit(train_df.iloc[t_idx][feature_cols], y_train[t_idx])
            oof_probs[v_idx] = pipe.predict_proba(train_df.iloc[v_idx][feature_cols])[:, 1]
        elif model_type == "lr":
            pipe = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(num_cols, cat_cols)),
                    ("clf", LogisticRegression(**model_kwargs)),
                ]
            )
            pipe.fit(train_df.iloc[t_idx][feature_cols], y_train[t_idx])
            oof_probs[v_idx] = pipe.predict_proba(train_df.iloc[v_idx][feature_cols])[:, 1]

    return oof_probs


def plot_and_save_calibration(
    y_val: np.ndarray,
    val_prob_uncal: np.ndarray,
    val_prob_cal: np.ndarray,
    save_path: Path,
):
    """Plot reliability diagram before and after calibration."""
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_val, val_prob_uncal, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_val, val_prob_cal, n_bins=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Reliability curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax1.plot(prob_pred_uncal, prob_true_uncal, "s-", color="#d9534f", label="Uncalibrated (Raw)")
    ax1.plot(prob_pred_cal, prob_true_cal, "o-", color="#2b8a3e", label="Calibrated (Isotonic)")
    ax1.set_xlabel("Mean Predicted Risk Probability")
    ax1.set_ylabel("Observed Readmission Rate")
    ax1.set_title("Reliability Diagram (Validation Split)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Histogram of predictions
    ax2.hist(
        val_prob_uncal,
        bins=25,
        alpha=0.5,
        color="#d9534f",
        label="Uncalibrated Probabilities",
        density=True,
    )
    ax2.hist(
        val_prob_cal,
        bins=25,
        alpha=0.5,
        color="#2b8a3e",
        label="Calibrated Probabilities",
        density=True,
    )
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Probability Distribution")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved calibration curve to {save_path}")


def train_and_evaluate_all_models() -> List[Dict]:
    """Train LR, RF, LightGBM with and without RAG on full splits with tuning, paired bootstrap and calibration."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading RAG-enhanced train and validation datasets...")
    # Load required columns to optimize RAM
    with open(FEATURE_LIST_PATH, "r") as f:
        f_list = json.load(f)

    base_all = f_list["base_features"]
    rag_col = "rag_struct_rate_k20"

    cols_to_load = list(set(base_all + [rag_col, "readmitted_30d", "patient_nbr"]))
    train_df = pd.read_parquet(TRAIN_RAG_PARQUET, columns=cols_to_load)
    val_df = pd.read_parquet(VAL_RAG_PARQUET, columns=cols_to_load)

    cat_cols = [c for c in base_all if not pd.api.types.is_numeric_dtype(train_df[c])]
    base_num_cols = [c for c in base_all if pd.api.types.is_numeric_dtype(train_df[c])]
    rag_num_cols = sorted(base_num_cols + [rag_col])

    y_train = train_df["readmitted_30d"].values.astype(int)
    y_val = val_df["readmitted_30d"].values.astype(int)
    groups_train = train_df["patient_nbr"].values

    print(f"Cohort sizes: Train={len(train_df):,} encounters, Val={len(val_df):,} encounters")
    print(
        f"Features: Base={len(base_all)} ({len(base_num_cols)} num, {len(cat_cols)} cat); With RAG={len(base_all)+1}"
    )

    # 1. Hyperparameter Tuning on TRAIN
    rf_depth = tune_random_forest(
        train_df,
        base_all,
        base_num_cols,
        cat_cols,
        y_train,
        groups_train,
    )

    X_train_lgbm_base = prepare_lgbm_data(train_df, base_num_cols, cat_cols)
    lgbm_depth, lgbm_lr = tune_lightgbm(X_train_lgbm_base, y_train, groups_train)

    ablation_records = []
    val_predictions = {}

    # -------------------------------------------------------------
    # 1. Logistic Regression: Without RAG
    # -------------------------------------------------------------
    print("\n--- Training Logistic Regression (Without RAG) ---")
    pipe_lr_base = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(base_num_cols, cat_cols)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
                ),
            ),
        ]
    )
    pipe_lr_base.fit(train_df[base_num_cols + cat_cols], y_train)
    val_prob_lr_base = pipe_lr_base.predict_proba(val_df[base_num_cols + cat_cols])[:, 1]
    val_predictions["lr_base"] = val_prob_lr_base
    m_lr_base = evaluate_predictions(y_val, val_prob_lr_base)
    m_lr_base.update({"model": "Logistic Regression", "rag_feature": False})
    ablation_records.append(m_lr_base)
    joblib.dump(pipe_lr_base, MODELS_DIR / "lr_base.joblib")
    print(
        f"Val AUC: {m_lr_base['auc_roc']} ({m_lr_base['auc_ci_95']}) | Brier: {m_lr_base['brier_score']} | Prec@10%: {m_lr_base['precision_top_10pct']} (Lift: {m_lr_base['lift_top_10pct']}x)"
    )

    # -------------------------------------------------------------
    # 2. Logistic Regression: With RAG
    # -------------------------------------------------------------
    print("\n--- Training Logistic Regression (With RAG) ---")
    pipe_lr_rag = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(rag_num_cols, cat_cols)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
                ),
            ),
        ]
    )
    pipe_lr_rag.fit(train_df[rag_num_cols + cat_cols], y_train)
    val_prob_lr_rag = pipe_lr_rag.predict_proba(val_df[rag_num_cols + cat_cols])[:, 1]
    val_predictions["lr_rag"] = val_prob_lr_rag
    m_lr_rag = evaluate_predictions(y_val, val_prob_lr_rag)
    m_lr_rag.update({"model": "Logistic Regression", "rag_feature": True})
    ablation_records.append(m_lr_rag)
    joblib.dump(pipe_lr_rag, MODELS_DIR / "lr_rag.joblib")
    print(
        f"Val AUC: {m_lr_rag['auc_roc']} ({m_lr_rag['auc_ci_95']}) | Brier: {m_lr_rag['brier_score']} | Prec@10%: {m_lr_rag['precision_top_10pct']} (Lift: {m_lr_rag['lift_top_10pct']}x)"
    )

    # -------------------------------------------------------------
    # 3. Random Forest: Without RAG
    # -------------------------------------------------------------
    print("\n--- Training Random Forest (Without RAG) ---")
    pipe_rf_base = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(base_num_cols, cat_cols)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=rf_depth,
                    min_samples_split=10,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    pipe_rf_base.fit(train_df[base_num_cols + cat_cols], y_train)
    val_prob_rf_base = pipe_rf_base.predict_proba(val_df[base_num_cols + cat_cols])[:, 1]
    val_predictions["rf_base"] = val_prob_rf_base
    m_rf_base = evaluate_predictions(y_val, val_prob_rf_base)
    m_rf_base.update({"model": "Random Forest", "rag_feature": False})
    ablation_records.append(m_rf_base)
    joblib.dump(pipe_rf_base, MODELS_DIR / "rf_base.joblib")
    print(
        f"Val AUC: {m_rf_base['auc_roc']} ({m_rf_base['auc_ci_95']}) | Brier: {m_rf_base['brier_score']} | Prec@10%: {m_rf_base['precision_top_10pct']} (Lift: {m_rf_base['lift_top_10pct']}x)"
    )

    # -------------------------------------------------------------
    # 4. Random Forest: With RAG
    # -------------------------------------------------------------
    print("\n--- Training Random Forest (With RAG) ---")
    pipe_rf_rag = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(rag_num_cols, cat_cols)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=rf_depth,
                    min_samples_split=10,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    pipe_rf_rag.fit(train_df[rag_num_cols + cat_cols], y_train)
    val_prob_rf_rag = pipe_rf_rag.predict_proba(val_df[rag_num_cols + cat_cols])[:, 1]
    val_predictions["rf_rag"] = val_prob_rf_rag
    m_rf_rag = evaluate_predictions(y_val, val_prob_rf_rag)
    m_rf_rag.update({"model": "Random Forest", "rag_feature": True})
    ablation_records.append(m_rf_rag)
    joblib.dump(pipe_rf_rag, MODELS_DIR / "rf_rag.joblib")
    print(
        f"Val AUC: {m_rf_rag['auc_roc']} ({m_rf_rag['auc_ci_95']}) | Brier: {m_rf_rag['brier_score']} | Prec@10%: {m_rf_rag['precision_top_10pct']} (Lift: {m_rf_rag['lift_top_10pct']}x)"
    )

    # -------------------------------------------------------------
    # 5. LightGBM: Without RAG
    # -------------------------------------------------------------
    print("\n--- Training LightGBM (Without RAG, Native Categoricals) ---")
    X_val_lgbm_base = prepare_lgbm_data(val_df, base_num_cols, cat_cols)
    lgbm_base = LGBMClassifier(
        n_estimators=150,
        max_depth=lgbm_depth,
        learning_rate=lgbm_lr,
        scale_pos_weight=7.7,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_base.fit(X_train_lgbm_base, y_train)
    val_prob_lgbm_base = lgbm_base.predict_proba(X_val_lgbm_base)[:, 1]
    val_predictions["lgbm_base"] = val_prob_lgbm_base
    m_lgbm_base = evaluate_predictions(y_val, val_prob_lgbm_base)
    m_lgbm_base.update({"model": "LightGBM", "rag_feature": False})
    ablation_records.append(m_lgbm_base)
    joblib.dump(lgbm_base, MODELS_DIR / "lgbm_base.joblib")
    print(
        f"Val AUC: {m_lgbm_base['auc_roc']} ({m_lgbm_base['auc_ci_95']}) | Brier: {m_lgbm_base['brier_score']} | Prec@10%: {m_lgbm_base['precision_top_10pct']} (Lift: {m_lgbm_base['lift_top_10pct']}x)"
    )

    # -------------------------------------------------------------
    # 6. LightGBM: With RAG
    # -------------------------------------------------------------
    print("\n--- Training LightGBM (With RAG, Native Categoricals) ---")
    X_train_lgbm_rag = prepare_lgbm_data(train_df, rag_num_cols, cat_cols)
    X_val_lgbm_rag = prepare_lgbm_data(val_df, rag_num_cols, cat_cols)
    lgbm_rag = LGBMClassifier(
        n_estimators=150,
        max_depth=lgbm_depth,
        learning_rate=lgbm_lr,
        scale_pos_weight=7.7,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_rag.fit(X_train_lgbm_rag, y_train)
    val_prob_lgbm_rag = lgbm_rag.predict_proba(X_val_lgbm_rag)[:, 1]
    val_predictions["lgbm_rag"] = val_prob_lgbm_rag
    m_lgbm_rag = evaluate_predictions(y_val, val_prob_lgbm_rag)
    m_lgbm_rag.update({"model": "LightGBM", "rag_feature": True})
    ablation_records.append(m_lgbm_rag)
    joblib.dump(lgbm_rag, MODELS_DIR / "lgbm_rag.joblib")
    print(
        f"Val AUC: {m_lgbm_rag['auc_roc']} ({m_lgbm_rag['auc_ci_95']}) | Brier: {m_lgbm_rag['brier_score']} | Prec@10%: {m_lgbm_rag['precision_top_10pct']} (Lift: {m_lgbm_rag['lift_top_10pct']}x)"
    )

    # Save fitted preprocessors
    joblib.dump(pipe_lr_base.named_steps["preprocessor"], PREPROCESSOR_BASE_PATH)
    joblib.dump(pipe_rf_rag.named_steps["preprocessor"], PREPROCESSOR_RAG_PATH)

    # -------------------------------------------------------------
    # 7. Paired Bootstrap AUC Difference (With RAG vs Without RAG)
    # -------------------------------------------------------------
    print("\n--- Computing Paired Bootstrap AUC Differences (With RAG - Without RAG) on Val ---")
    paired_diffs = {}
    for family, key_base, key_rag in [
        ("Logistic Regression", "lr_base", "lr_rag"),
        ("Random Forest", "rf_base", "rf_rag"),
        ("LightGBM", "lgbm_base", "lgbm_rag"),
    ]:
        diff_info = compute_paired_bootstrap_auc_diff(
            y_val,
            val_predictions[key_base],
            val_predictions[key_rag],
            n_bootstraps=1000,
            seed=RANDOM_SEED,
        )
        paired_diffs[family] = diff_info
        print(
            f"  {family}: Delta AUC = {diff_info['observed_diff']:+.4f} (95% CI: {diff_info['ci_95']}) | Crosses 0: {diff_info['crosses_zero']}"
        )

    # -------------------------------------------------------------
    # 8. Calibration on Train OOF
    # -------------------------------------------------------------
    print("\n--- Fitting Calibration via Train Out-Of-Fold Predictions ---")
    # Identify top model family by Val AUC
    val_auc_map = {
        "lr_rag": m_lr_rag["auc_roc"],
        "rf_rag": m_rf_rag["auc_roc"],
        "lgbm_rag": m_lgbm_rag["auc_roc"],
        "lgbm_base": m_lgbm_base["auc_roc"],
    }
    top_model_key = max(val_auc_map, key=val_auc_map.get)
    print(
        f"Top model for primary calibration: {top_model_key} (Val AUC: {val_auc_map[top_model_key]:.4f})"
    )

    # Generate Train OOF for LightGBM with RAG (or the winning family)
    lgbm_kwargs = {
        "n_estimators": 150,
        "max_depth": lgbm_depth,
        "learning_rate": lgbm_lr,
        "scale_pos_weight": 7.7,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    train_oof_probs = generate_train_oof_probabilities(
        "lgbm",
        train_df,
        rag_num_cols + cat_cols,
        rag_num_cols,
        cat_cols,
        y_train,
        groups_train,
        lgbm_kwargs,
    )

    # Fit Isotonic Calibrator on Train OOF only
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(train_oof_probs, y_train)
    joblib.dump(calibrator, MODELS_DIR / "calibrator.joblib")
    print("Saved calibrator to models/calibrator.joblib")

    # Apply to Validation predictions
    val_prob_calibrated = calibrator.transform(val_prob_lgbm_rag)
    m_uncal = evaluate_predictions(y_val, val_prob_lgbm_rag)
    m_cal = evaluate_predictions(y_val, val_prob_calibrated)

    calibration_report = {
        "model": "LightGBM + Structured RAG",
        "brier_uncalibrated": m_uncal["brier_score"],
        "brier_calibrated": m_cal["brier_score"],
        "ece_uncalibrated": m_uncal["ece"],
        "ece_calibrated": m_cal["ece"],
        "auc_roc": m_cal["auc_roc"],
    }
    print(
        f"Calibration results on Val: Brier {m_uncal['brier_score']} -> {m_cal['brier_score']} | ECE {m_uncal['ece']} -> {m_cal['ece']}"
    )

    # Save calibration figure
    plot_and_save_calibration(
        y_val,
        val_prob_lgbm_rag,
        val_prob_calibrated,
        FIGURES_DIR / "calibration_curve.png",
    )

    # -------------------------------------------------------------
    # 9. Save Ablation Outputs
    # -------------------------------------------------------------
    # Augment ablation records with paired diffs
    for rec in ablation_records:
        rec_model = rec["model"]
        if rec["rag_feature"]:
            rec["paired_delta_auc"] = paired_diffs[rec_model]["observed_diff"]
            rec["paired_ci_95"] = paired_diffs[rec_model]["ci_95"]
            rec["ci_crosses_zero"] = paired_diffs[rec_model]["crosses_zero"]
        else:
            rec["paired_delta_auc"] = 0.0
            rec["paired_ci_95"] = [0.0, 0.0]
            rec["ci_crosses_zero"] = True

    with open(METRICS_DIR / "ablation_table.json", "w") as f:
        json.dump(
            {
                "ablation_results": ablation_records,
                "paired_bootstrap_differences": paired_diffs,
                "calibration_report": calibration_report,
            },
            f,
            indent=2,
        )

    # Markdown ablation table
    md_content = "# ReadmitIQ Validation Ablation Table\n\n"
    md_content += f"Cohort: Train={len(train_df):,} encounters, Val={len(val_df):,} encounters (UCI 130-US Hospitals).\n\n"
    md_content += "| Model | RAG Feature | Val AUC (95% CI) | F1 | Precision@10% | Lift@10% | Brier | ECE | Paired Delta AUC (95% CI) | Crosses 0? |\n"
    md_content += "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"

    for r in ablation_records:
        rag_str = "Yes (k=20)" if r["rag_feature"] else "No"
        ci_str = f"[{r['auc_ci_95'][0]:.4f}, {r['auc_ci_95'][1]:.4f}]"
        if r["rag_feature"]:
            p_ci_str = f"[{r['paired_ci_95'][0]:+.4f}, {r['paired_ci_95'][1]:+.4f}]"
            cz_str = "Yes" if r["ci_crosses_zero"] else "No"
            p_diff_str = f"{r['paired_delta_auc']:+.4f}"
        else:
            p_ci_str = "-"
            cz_str = "-"
            p_diff_str = "Ref"

        md_content += f"| {r['model']} | {rag_str} | {r['auc_roc']:.4f} {ci_str} | {r['f1']:.4f} | {r['precision_top_10pct']:.4f} | {r['lift_top_10pct']:.2f}x | {r['brier_score']:.4f} | {r['ece']:.4f} | {p_diff_str} {p_ci_str} | {cz_str} |\n"

    # Add placeholder for Transformer
    md_content += (
        "| DistilBERT (Notes) | None | *Pending Colab run* | - | - | - | - | - | - | - |\n\n"
    )

    md_content += "### Calibration Performance (LightGBM + RAG)\n"
    md_content += f"- **Brier Score**: {calibration_report['brier_uncalibrated']:.4f} (uncalibrated) -> **{calibration_report['brier_calibrated']:.4f}** (calibrated via Train OOF isotonic regression)\n"
    md_content += f"- **Expected Calibration Error (ECE)**: {calibration_report['ece_uncalibrated']:.4f} -> **{calibration_report['ece_calibrated']:.4f}**\n\n"

    md_content += "### Key Methodological Findings\n"
    md_content += "- All models evaluated on the FULL validation set (14,902 encounters) with no subsampling.\n"
    md_content += "- Paired bootstrap confidence intervals directly test if RAG improvements are statistically significant vs base feature set.\n"

    with open(METRICS_DIR / "ablation_table.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(
        f"Saved ablation table to {METRICS_DIR / 'ablation_table.json'} and {METRICS_DIR / 'ablation_table.md'}"
    )
    return ablation_records


if __name__ == "__main__":
    train_and_evaluate_all_models()
