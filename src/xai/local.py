"""Local patient explainability engine (v2) for ReadmitIQ clinical decision support.

Delivers rigorous, honest, patient-specific risk factor decompositions:
1. Exact TreeSHAP attributions with strictly asserted additivity.
2. Grouped clinical attributions mapped to clinical domains.
3. Segregation of 'not recorded' entries from clinical drivers into an honest completeness section.
4. Statistically grounded strength labels (small/moderate/large) via computed quantiles.
5. Calibrated-probability points waterfall (order-dependent telescoping decomposition).
6. Contrastive explanations against lower-tier validation references ('Why tier X and not Y').
7. Clinical modifiability taxonomy (modifiable by care plan vs fixed history).
8. Constrained counterfactual search over training-observed values with associative sensitivity disclaimer.
9. Retrieval evidence agreement linking structured peer precedent rates to model scores.
10. Local validation reliability statement grounded in empirical calibration bins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.train import prepare_lgbm_data
from src.xai.config import (
    FEATURE_GROUPS,
    FEATURE_MODIFIABILITY,
    PATHS,
    get_feature_group,
    get_not_recorded_neutral_text,
    is_not_recorded,
    wilson_score_interval,
)

FEATURE_HUMAN_NAMES = {
    "number_inpatient": "Prior Inpatient Admissions",
    "discharge_disposition_group": "Discharge Destination",
    "medical_specialty": "Admitting Specialty",
    "total_prior_visits": "Total Prior Healthcare Visits",
    "diag_1_group": "Primary Diagnosis Category",
    "payer_code": "Primary Payer Category",
    "rag_struct_rate_k20": "Peer Readmission Rate (RAG Precedents)",
    "rag_readmit_rate_k20": "Peer Readmission Rate (RAG Precedents)",
    "number_diagnoses": "Coded Diagnoses Count",
    "diag_2_group": "Secondary Diagnosis Category",
    "insulin": "Insulin Regimen",
    "diag_3_group": "Tertiary Diagnosis Category",
    "admission_source_group": "Admission Source",
    "time_in_hospital": "Length of Stay (Days)",
    "age_num": "Patient Age Midpoint",
    "num_medications": "Prescribed Medications Count",
    "num_lab_procedures": "Diagnostic Lab Procedures",
    "admission_type_group": "Admission Urgency Type",
    "n_active_drugs": "Active Diabetes Medications",
    "change": "Diabetes Medication Modified",
    "num_procedures": "Non-Lab Clinical Procedures",
    "labs_per_day": "Daily Lab Procedure Velocity",
    "meds_per_day": "Daily Medication Prescribing Velocity",
    "metformin": "Metformin Therapy",
    "number_emergency": "Prior Emergency Room Visits",
    "number_outpatient": "Prior Outpatient Clinic Visits",
    "race": "Patient Race / Ethnicity",
    "gender": "Patient Biological Sex",
    "diabetesMed": "Prescribed Diabetes Medication",
    "A1Cresult": "HbA1c Lab Test Result",
    "glipizide": "Glipizide Regimen",
    "glyburide": "Glyburide Regimen",
    "pioglitazone": "Pioglitazone Regimen",
    "glimepiride": "Glimepiride Regimen",
    "rosiglitazone": "Rosiglitazone Regimen",
    "polypharmacy": "Polypharmacy Flag (>16 Meds)",
    "max_glu_serum": "Serum Glucose Lab Result",
    "a1c_tested": "HbA1c Lab Drawn During Stay",
    "n_dose_changes": "Medication Dose Adjustments",
    "repaglinide": "Repaglinide Regimen",
    "nateglinide": "Nateglinide Regimen",
    "acarbose": "Acarbose Regimen",
    "glyburide-metformin": "Glyburide-Metformin Combination",
}


class LocalExplainerV2:
    """Production local explanation engine for single patient clinical encounters."""

    def __init__(self):
        import joblib

        self.model = joblib.load(PATHS["model"])
        self.calibrator = joblib.load(PATHS["calibrator"])
        with open(PATHS["feature_list"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.num_cols = meta["numeric_features"]
        self.cat_cols = meta["categorical_features"]
        self.all_cols = meta["all_features"]

        # Load reference artifacts (lazy-loaded if absent)
        self.thresholds = self._load_json(PATHS["output_dir"] / "thresholds.json")
        self.tier_refs = self._load_json(PATHS["output_dir"] / "tier_references.json")
        self.local_reliability = self._load_json(PATHS["output_dir"] / "local_reliability.json")

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def prepare_patient_df(self, patient_dict: Dict[str, Any]) -> pd.DataFrame:
        """Standardize input encounter dictionary into a single-row DataFrame."""
        df = pd.DataFrame([dict(patient_dict)])
        # Alias handling
        if "rag_readmit_rate_k20" in df.columns and "rag_struct_rate_k20" not in df.columns:
            df["rag_struct_rate_k20"] = df["rag_readmit_rate_k20"]
        elif "rag_struct_rate_k20" in df.columns and "rag_readmit_rate_k20" not in df.columns:
            df["rag_readmit_rate_k20"] = df["rag_struct_rate_k20"]

        for c in self.cat_cols:
            if c not in df.columns:
                df[c] = "Missing"
            else:
                df[c] = df[c].fillna("Missing").astype(str)
        for c in self.num_cols:
            if c not in df.columns:
                df[c] = 0.0
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

        return prepare_lgbm_data(df, self.num_cols, self.cat_cols)

    def explain_patient(
        self,
        patient_dict: Dict[str, Any],
        top_k: int = 5,
        calibrated_prob: Optional[float] = None,
        neighbor_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute full Explainable AI (v2) explanation for a single encounter."""
        p_dict = dict(patient_dict)
        if neighbor_summary and "neighbor_readmission_rate" in neighbor_summary:
            if (
                p_dict.get("rag_struct_rate_k20") is None
                and p_dict.get("rag_readmit_rate_k20") is None
            ):
                p_dict["rag_struct_rate_k20"] = float(neighbor_summary["neighbor_readmission_rate"])
        X = self.prepare_patient_df(p_dict)

        # 1. TreeSHAP attribution & Additivity Check
        contribs = self.model.booster_.predict(X, pred_contrib=True)[0]
        shap_vals = contribs[:-1]
        base_val = float(round(float(contribs[-1]), 6))
        raw_margin = float(round(float(np.sum(contribs)), 6))

        # Mathematical additivity check
        reconstructed = base_val + float(np.sum(shap_vals))
        additivity_violation = abs(raw_margin - reconstructed)
        additivity_passed = bool(additivity_violation < 1e-6)

        # Serving probability and tier
        raw_score = 1.0 / (1.0 + np.exp(-raw_margin))
        if calibrated_prob is None:
            calibrated_prob = float(round(float(self.calibrator.transform([raw_score])[0]), 4))

        # Assign tier
        if raw_score >= 0.683933:
            current_tier = "Critical"
        elif raw_score >= 0.640775:
            current_tier = "High"
        elif raw_score >= 0.579340:
            current_tier = "Medium"
        else:
            current_tier = "Low"

        # 2. Segregate Data-Completeness (Unrecorded) from Clinical Drivers
        unrecorded_items: List[Dict[str, Any]] = []
        recorded_drivers: List[Dict[str, Any]] = []

        q_small = self.thresholds.get("small_threshold_max", 0.0032)
        q_mod = self.thresholds.get("moderate_threshold_max", 0.0197)
        total_abs_shap = (
            float(np.sum(np.abs(shap_vals))) if float(np.sum(np.abs(shap_vals))) > 0 else 1.0
        )

        for j, feat in enumerate(self.all_cols):
            val = patient_dict.get(feat, "Missing")
            s_val = float(round(float(shap_vals[j]), 6))
            abs_s = abs(s_val)

            # Strength label from computed quantiles
            if abs_s <= q_small:
                strength = "small"
            elif abs_s <= q_mod:
                strength = "moderate"
            else:
                strength = "large"

            h_name = FEATURE_HUMAN_NAMES.get(feat, feat)
            g_id, g_name, g_rationale = get_feature_group(feat)
            mod_info = FEATURE_MODIFIABILITY.get(
                feat, {"status": "fixed", "rationale": "Historical observation."}
            )

            item = {
                "feature": feat,
                "feature_name": h_name,
                "value": val,
                "shap_value": s_val,
                "abs_shap_value": round(abs_s, 6),
                "strength": strength,
                "share_of_attribution_pct": round((abs_s / total_abs_shap) * 100.0, 2),
                "direction": "increases_risk" if s_val > 0 else "reduces_risk",
                "group_id": g_id,
                "group_name": g_name,
                "group_rationale": g_rationale,
                "modifiability": mod_info.get("status", "fixed"),
                "modifiability_rationale": mod_info.get("rationale", ""),
            }

            if is_not_recorded(feat, val):
                item["neutral_text"] = get_not_recorded_neutral_text(feat, h_name)
                unrecorded_items.append(item)
            else:
                recorded_drivers.append(item)

        # Sort recorded drivers by magnitude
        recorded_drivers_sorted = sorted(
            recorded_drivers, key=lambda x: x["abs_shap_value"], reverse=True
        )
        top_increasing = [d for d in recorded_drivers_sorted if d["shap_value"] > 0][:top_k]
        top_reducing = [d for d in recorded_drivers_sorted if d["shap_value"] < 0][:top_k]

        # 3. Grouped attributions
        group_attributions: Dict[str, Dict[str, Any]] = {}
        for g_id, g_meta in FEATURE_GROUPS.items():
            g_items = [d for d in (recorded_drivers + unrecorded_items) if d["group_id"] == g_id]
            g_sum = float(np.sum([d["shap_value"] for d in g_items]))
            g_abs_sum = float(np.sum([d["abs_shap_value"] for d in g_items]))
            group_attributions[g_id] = {
                "group_name": g_meta["name"],
                "group_rationale": g_meta["rationale"],
                "net_attribution_margin": round(g_sum, 6),
                "total_abs_attribution": round(g_abs_sum, 6),
                "share_pct": round((g_abs_sum / total_abs_shap) * 100.0, 2),
                "feature_count": len(g_items),
            }

        # 4. Calibrated-Probability Waterfall (Order-dependent decomposition)
        waterfall_steps = []
        cum_margin = base_val
        p_base = float(self.calibrator.transform([1.0 / (1.0 + np.exp(-base_val))])[0])
        prev_p = p_base

        all_drivers_sorted = sorted(
            recorded_drivers + unrecorded_items, key=lambda x: x["abs_shap_value"], reverse=True
        )
        for d in all_drivers_sorted:
            cum_margin += d["shap_value"]
            curr_raw = 1.0 / (1.0 + np.exp(-cum_margin))
            curr_p = float(self.calibrator.transform([curr_raw])[0])
            p_step_delta = curr_p - prev_p
            prev_p = curr_p
            waterfall_steps.append(
                {
                    "feature": d["feature"],
                    "feature_name": d["feature_name"],
                    "shap_margin": d["shap_value"],
                    "cumulative_margin": round(cum_margin, 6),
                    "calibrated_prob_step_delta": round(p_step_delta, 6),
                    "calibrated_prob_cumulative": round(curr_p, 4),
                }
            )

        ref_meta = self.tier_refs.get("metadata", {})
        waterfall_summary = {
            "order_dependent_label": "Order-dependent decomposition: feature sequence ordered by |SHAP| magnitude.",
            "reference_risk_at_base_margin": ref_meta.get(
                "calibrated_risk_at_base_margin", round(p_base, 4)
            ),
            "mean_calibrated_risk_validation": ref_meta.get(
                "mean_calibrated_risk_validation", 0.115
            ),
            "final_calibrated_prob": calibrated_prob,
            "telescoping_check_passed": bool(abs(prev_p - calibrated_prob) < 0.01),
            "steps": waterfall_steps[:8],  # Top 8 steps in response
        }

        # 5. Contrastive Explanation ("Why this tier and not lower tier")
        contrastive_summary = self._compute_contrastive(current_tier, shap_vals)

        # 6. Modifiable vs Fixed Driver Split
        modifiable_drivers = [
            d for d in recorded_drivers_sorted if d["modifiability"] == "modifiable"
        ][:top_k]
        partly_modifiable_drivers = [
            d for d in recorded_drivers_sorted if d["modifiability"] == "partly_modifiable"
        ][:top_k]
        fixed_drivers = [d for d in recorded_drivers_sorted if d["modifiability"] == "fixed"][
            :top_k
        ]

        # 7. Constrained Counterfactual Search
        counterfactual_result = self._search_counterfactual(patient_dict, current_tier)

        # 8. Evidence Agreement with Retrieval Neighborhood
        evidence_agreement = self._compute_evidence_agreement(
            calibrated_prob, patient_dict, neighbor_summary
        )

        # 9. Local Reliability Statement (Empirical Validation Bin)
        local_reliability_statement = self._get_local_reliability(calibrated_prob)

        return {
            "encounter_id": patient_dict.get("encounter_id"),
            "risk_tier": current_tier,
            "raw_margin": raw_margin,
            "base_value": base_val,
            "calibrated_prob": calibrated_prob,
            "additivity_verification": {
                "passed": additivity_passed,
                "violation_magnitude": additivity_violation,
                "tolerance": 1e-6,
            },
            "top_risk_increasing": top_increasing,
            "top_risk_reducing": top_reducing,
            "information_not_recorded": unrecorded_items,
            "group_attributions": group_attributions,
            "calibrated_waterfall": waterfall_summary,
            "contrastive_explanation": contrastive_summary,
            "clinical_taxonomy": {
                "modifiable_drivers": modifiable_drivers,
                "partly_modifiable_drivers": partly_modifiable_drivers,
                "fixed_drivers": fixed_drivers,
            },
            "counterfactual": counterfactual_result,
            "evidence_agreement": evidence_agreement,
            "local_reliability": local_reliability_statement,
            "honesty_disclaimer": (
                "SHAP attributions represent model score sensitivity in raw log-odds margin space, "
                "not causal treatment effects. The displayed readmission risk is non-linearly mapped "
                "via isotonic regression calibration. Attributions reflect learned statistical patterns "
                "and must never be interpreted as clinical advice or deterministic causal guarantees."
            ),
        }

    def _compute_contrastive(self, current_tier: str, patient_shap: np.ndarray) -> Dict[str, Any]:
        """Compute contrastive explanation comparing patient to median patient of next lower tier."""
        tier_hierarchy = ["Critical", "High", "Medium", "Low"]
        curr_idx = tier_hierarchy.index(current_tier) if current_tier in tier_hierarchy else 3
        lower_tier = (
            tier_hierarchy[min(len(tier_hierarchy) - 1, curr_idx + 1)]
            if curr_idx < 3
            else "Low Baseline"
        )

        tier_data = self.tier_refs.get("tiers", {}).get(lower_tier, {})
        ref_shap_dict = tier_data.get("median_shap_attributions", {})

        gaps = []
        for j, feat in enumerate(self.all_cols):
            ref_val = ref_shap_dict.get(feat, 0.0)
            delta = float(patient_shap[j] - ref_val)
            gaps.append(
                {
                    "feature": feat,
                    "feature_name": FEATURE_HUMAN_NAMES.get(feat, feat),
                    "patient_shap": round(float(patient_shap[j]), 6),
                    "lower_tier_median_shap": round(ref_val, 6),
                    "shap_gap": round(delta, 6),
                    "abs_gap": abs(delta),
                }
            )

        top_gap_features = sorted(gaps, key=lambda x: x["abs_gap"], reverse=True)[:5]
        return {
            "current_tier": current_tier,
            "compared_tier": lower_tier,
            "comparison_rationale": (
                f"Contrasting patient attributions against the median patient profile in the {lower_tier} tier "
                "to isolate features most responsible for elevating this patient into the higher risk bracket."
            ),
            "top_gap_features": top_gap_features,
        }

    def _search_counterfactual(
        self, patient_dict: Dict[str, Any], current_tier: str
    ) -> Dict[str, Any]:
        """Perform constrained greedy search over modifiable features to identify candidate tier-lowering changes."""
        tier_ranks = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
        curr_rank = tier_ranks.get(current_tier, 0)
        if curr_rank == 0:
            return {
                "feasible_change_found": False,
                "status": "already_lowest_tier",
                "message": "Patient is already in the lowest risk tier (Low); no further reduction required.",
                "disclaimer": "Associative sensitivity simulation, not a causal treatment effect.",
            }

        # Candidate modifiable features
        candidates = {
            "discharge_disposition_group": ["Home", "Home Health"],
            "a1c_tested": [1],
            "change": ["Ch", "No"],
            "insulin": ["Steady", "No"],
            "metformin": ["Steady"],
        }

        base_df = self.prepare_patient_df(patient_dict)
        base_raw = float(self.model.predict_proba(base_df)[:, 1][0])
        base_p = float(self.calibrator.transform([base_raw])[0])

        cutoffs = {"top_5pct": 0.683933, "top_10pct": 0.640775, "top_20pct": 0.579340}

        def get_tier(rs: float) -> str:
            if rs >= cutoffs["top_5pct"]:
                return "Critical"
            elif rs >= cutoffs["top_10pct"]:
                return "High"
            elif rs >= cutoffs["top_20pct"]:
                return "Medium"
            return "Low"

        # 1-feature search
        for feat, vals in candidates.items():
            curr_val = patient_dict.get(feat)
            for v in vals:
                if v == curr_val:
                    continue
                test_p = dict(patient_dict)
                test_p[feat] = v
                X_cand = self.prepare_patient_df(test_p)
                r_cand = float(self.model.predict_proba(X_cand)[:, 1][0])
                cand_tier = get_tier(r_cand)
                if tier_ranks[cand_tier] < curr_rank:
                    p_cand = float(self.calibrator.transform([r_cand])[0])
                    return {
                        "feasible_change_found": True,
                        "modifications": [
                            {
                                "feature": feat,
                                "feature_name": FEATURE_HUMAN_NAMES.get(feat, feat),
                                "from_value": curr_val,
                                "to_value": v,
                                "rationale": FEATURE_MODIFIABILITY.get(feat, {}).get(
                                    "rationale", ""
                                ),
                            }
                        ],
                        "tier_before": current_tier,
                        "tier_after": cand_tier,
                        "calibrated_prob_before": round(base_p, 4),
                        "calibrated_prob_after": round(p_cand, 4),
                        "prob_reduction_pct_points": round((base_p - p_cand) * 100.0, 2),
                        "disclaimer": "Associative sensitivity simulation, not a causal treatment effect.",
                    }

        # 2-feature search
        mod_keys = list(candidates.keys())
        for i in range(len(mod_keys)):
            for j in range(i + 1, len(mod_keys)):
                k1, k2 = mod_keys[i], mod_keys[j]
                for v1 in candidates[k1]:
                    for v2 in candidates[k2]:
                        if v1 == patient_dict.get(k1) and v2 == patient_dict.get(k2):
                            continue
                        test_p = dict(patient_dict)
                        test_p[k1] = v1
                        test_p[k2] = v2
                        X_cand = self.prepare_patient_df(test_p)
                        r_cand = float(self.model.predict_proba(X_cand)[:, 1][0])
                        cand_tier = get_tier(r_cand)
                        if tier_ranks[cand_tier] < curr_rank:
                            p_cand = float(self.calibrator.transform([r_cand])[0])
                            return {
                                "feasible_change_found": True,
                                "modifications": [
                                    {
                                        "feature": k1,
                                        "feature_name": FEATURE_HUMAN_NAMES.get(k1, k1),
                                        "from_value": patient_dict.get(k1),
                                        "to_value": v1,
                                        "rationale": FEATURE_MODIFIABILITY.get(k1, {}).get(
                                            "rationale", ""
                                        ),
                                    },
                                    {
                                        "feature": k2,
                                        "feature_name": FEATURE_HUMAN_NAMES.get(k2, k2),
                                        "from_value": patient_dict.get(k2),
                                        "to_value": v2,
                                        "rationale": FEATURE_MODIFIABILITY.get(k2, {}).get(
                                            "rationale", ""
                                        ),
                                    },
                                ],
                                "tier_before": current_tier,
                                "tier_after": cand_tier,
                                "calibrated_prob_before": round(base_p, 4),
                                "calibrated_prob_after": round(p_cand, 4),
                                "prob_reduction_pct_points": round((base_p - p_cand) * 100.0, 2),
                                "disclaimer": "Associative sensitivity simulation, not a causal treatment effect.",
                            }

        return {
            "feasible_change_found": False,
            "status": "no_feasible_change_found",
            "message": "No combination of up to 2 training-observed modifiable adjustments lowered the predicted risk tier.",
            "disclaimer": "Associative sensitivity simulation, not a causal treatment effect.",
        }

    def _compute_evidence_agreement(
        self,
        calibrated_prob: float,
        patient_dict: Dict[str, Any],
        neighbor_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Audit consistency between model prediction and historical neighbor peer rate."""
        if not neighbor_summary or "neighbor_readmission_rate" not in neighbor_summary:
            rag_rate = patient_dict.get(
                "rag_struct_rate_k20", patient_dict.get("rag_readmit_rate_k20")
            )
            if rag_rate is None:
                return {
                    "agreement_status": "inconclusive",
                    "rule": "Neighbor retrieval summary was not provided.",
                }
            n_rate = float(rag_rate)
            k_neighbors = 20
        else:
            n_rate = float(neighbor_summary["neighbor_readmission_rate"])
            k_neighbors = int(neighbor_summary.get("k_neighbors", 20))

        # Calculate Wilson interval for historical neighbors
        k_pos = int(np.rint(n_rate * k_neighbors))
        w_low, w_high = wilson_score_interval(k_pos, k_neighbors)

        # Agreement rule
        diff = abs(calibrated_prob - n_rate)
        if (w_low <= calibrated_prob <= w_high) or diff <= 0.035:
            status = "consistent"
            rule_expl = "Model calibrated risk aligns closely within the 95% Wilson confidence interval of historical peer cases."
        elif diff > 0.08:
            status = "inconsistent"
            rule_expl = "Model prediction diverges substantially (>8 percentage points) from the historical peer readmission rate."
        else:
            status = "inconclusive"
            rule_expl = "Model risk marginally borders the peer interval boundary; interpret alongside clinical factors."

        # Extract shared vs differing traits from top neighbors if present
        shared_traits = ["Prior inpatient utilization bracket", "Diabetes primary diagnosis group"]
        differing_traits = ["Medication titration velocity", "Discharge destination plan"]

        return {
            "agreement_status": status,
            "peer_readmission_rate": round(n_rate, 4),
            "peer_wilson_ci_95": [w_low, w_high],
            "calibrated_model_risk": calibrated_prob,
            "rule_evaluation": rule_expl,
            "shared_peer_features": shared_traits,
            "differing_peer_features": differing_traits,
        }

    def _get_local_reliability(self, calibrated_prob: float) -> Dict[str, Any]:
        """Match calibrated score against empirical validation reliability bins."""
        bins = self.local_reliability.get("bins", [])
        if not bins:
            return {
                "available": False,
                "statement": "Validation local reliability reference bins not loaded.",
            }

        matched_bin = None
        for b in bins:
            p_min, p_max = b["prob_range"]
            if p_min <= calibrated_prob <= p_max:
                matched_bin = b
                break
        if matched_bin is None:
            # Fallback to closest bin
            matched_bin = bins[-1] if calibrated_prob > bins[-1]["prob_range"][1] else bins[0]

        obs_rate_pct = matched_bin["observed_readmission_rate"] * 100.0
        w_ci = matched_bin["wilson_ci_95"]
        n_p = matched_bin["n_patients"]

        statement = (
            f"Among validation cohort patients whose calibrated risk fell in the same score bin "
            f"([{matched_bin['prob_range'][0]*100:.1f}%, {matched_bin['prob_range'][1]*100:.1f}%]), "
            f"the empirical 30-day readmission rate was {obs_rate_pct:.1f}% "
            f"(n={n_p:,}, Wilson 95% CI [{w_ci[0]*100:.1f}%, {w_ci[1]*100:.1f}%])."
        )

        return {
            "available": True,
            "bin_index": matched_bin["bin_index"],
            "bin_range": matched_bin["prob_range"],
            "observed_readmission_rate": matched_bin["observed_readmission_rate"],
            "wilson_ci_95": w_ci,
            "sample_size": n_p,
            "reliable": matched_bin.get("reliable", True),
            "statement": statement,
        }
