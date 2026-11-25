from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

BASE = Path("results")
OUT = BASE / "spec_scenario"
CODE = 17
BATCH = 789

provenance = {}


# 1. Data loading and provenance
def load(name, folder=OUT):
    path = folder / name

    if not path.exists():
        raise FileNotFoundError(path)

    provenance[str(path)] = {
        "sha256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    }

    return pd.read_csv(path)


def records(df):
    return json.loads(
        df.to_json(
            orient="records",
            date_format="iso",
            double_precision=10
        )
    )


def one(df):
    assert len(df) == 1
    return records(df)[0]


def select_batch(df):
    return df[
        (df["code"] == CODE) &
        (df["batch"] == BATCH)
    ].copy()


# 2. Batch information
target = select_batch(
    load("rca_target_batch.csv")
)

reference = load("rca_reference_batches.csv")

assert len(target) == 1
assert len(reference) == 114
assert reference["code"].eq(CODE).all()

assert (
    pd.to_datetime(reference["end_ts"])
    <= pd.to_datetime(target["start_ts"].iloc[0])
).all()

actual = float(target["tbl_yield"].iloc[0])

batch_info = {
    "code": CODE,
    "batch": BATCH,
    "campaign": 155,
    "start": str(target["start_ts"].iloc[0]),
    "end": str(target["end_ts"].iloc[0]),
    "actual_yield_pct": actual,
    "analysis_phase": "post_batch"
}


# 3. Specification and SPC
spec_path = Path(
    "config/yield_specifications.json"
)

spec_config = json.loads(
    spec_path.read_text(encoding="utf-8")
)

provenance[str(spec_path)] = {
    "sha256": hashlib.sha256(
        spec_path.read_bytes()
    ).hexdigest()
}

spec_row = one(
    select_batch(load("spec_detection_all.csv"))
)

lsl = float(spec_row["lsl"])
usl = float(spec_row["usl"])

specification = {
    "status": "research_poc_only",
    "approved_manufacturing_spec": False,
    "configuration": spec_config,
    "lsl_pct": lsl,
    "usl_pct": usl,
    "actual_pct": actual,
    "spec_outside": bool(
        actual < lsl or actual > usl
    ),
    "gap_below_lsl_pp": max(lsl - actual, 0),
    "actual_gmp_oos": None,
    "detection_record": spec_row
}

spc = {
    "signals": one(target),
    "limits": records(
        load(
            "spc_summary.csv",
            BASE / "followup"
        ).query("code == @CODE")
    ),
    "interpretation": "statistical_signal_not_root_cause"
}


# 4. Historical reference
reference_info = {
    "selection": (
        "Same product, historical training batches "
        "within hypothetical yield limits"
    ),
    "count": len(reference),
    "code": CODE,
    "same_campaign_reference_count": 0,
    "yield_mean_pct": float(
        reference["tbl_yield"].mean()
    ),
    "yield_median_pct": float(
        reference["tbl_yield"].median()
    ),
    "yield_std_pp": float(
        reference["tbl_yield"].std()
    ),
    "yield_min_pct": float(
        reference["tbl_yield"].min()
    ),
    "yield_max_pct": float(
        reference["tbl_yield"].max()
    ),
    "approved_normal_status_verified": False
}


# 5. Feature comparison
fc = load("rca_feature_comparison.csv")

outside = fc[
    fc["outside_reference_range"]
    .astype(str).str.lower()
    .isin(["true", "1"])
]

assert len(fc) == 58
assert len(outside) == 12

feature_comparison = {
    "compared_features": len(fc),
    "outside_reference_range_count": len(outside),
    "group_summary": records(
        load("rca_group_summary.csv")
    ),
    "all_features": records(fc),
    "outside_reference_range": records(outside),
    "excluded_features": [
        {
            "feature": "api_l_impurity",
            "reason": "target_missing",
            "reference_valid_count": 50
        }
    ],
    "warning": (
        "Historical range is not a manufacturing "
        "specification. Very small MAD may inflate "
        "robust z-scores."
    )
}


# 6. Time series evidence
ts = load("rca_batch789_validated_timeseries.csv")
stops = load("rca_internal_stop_segments.csv")
speed = load("rca_speed_segments.csv")
events = load("rca_waste_events.csv")
windows = load("rca_30min_window_summary.csv")

assert len(ts) == 2292
assert len(stops) == 8
assert len(events) == 71

waste_total = float(events["waste_increase"].sum())

peak_windows = windows[
    windows["window_start_min"].isin([120, 150])
]

peak_waste = float(
    peak_windows["waste_increase"].sum()
)

stable_300 = speed[
    speed["speed"].eq(300)
].sort_values(
    "estimated_minutes",
    ascending=False
).head(1)

timeseries = {
    "timestamp_resolution": "minute",
    "nominal_sampling_seconds": 10,
    "target_rows": len(ts),
    "missing_minutes": 0,
    "duration_estimation": "record_count_x_10_seconds",
    "operating_phase_summary": records(
        load("rca_operating_phase_summary.csv")
    ),
    "internal_stop_count": len(stops),
    "estimated_internal_stop_minutes": float(
        stops["estimated_minutes"].sum()
    ),
    "internal_stop_segments": records(stops),
    "major_speed_segments": records(
        speed[speed["rows"] >= 6]
    ),
    "waste_counter_total_increase": waste_total,
    "waste_increase_event_count": len(events),
    "priority_window": {
        "start": "2020-10-27 16:01:00",
        "end_exclusive": "2020-10-27 17:01:00",
        "waste_counter_increase": peak_waste,
        "share_pct": peak_waste / waste_total * 100
    },
    "waste_events": records(events),
    "window_summary": records(windows),
    "counterexample_same_speed": records(stable_300),
    "early_comparison": records(
        load("rca_timeseries_early_comparison.csv")
    ),
    "full_comparison": records(
        load("rca_timeseries_full_comparison.csv")
    ),
    "stop_comparison": records(
        load("rca_timeseries_stop_comparison.csv")
    ),
    "counter_decreases": records(
        load("rca_batch789_counter_decreases.csv")
    ),
    "counter_warning": (
        "Counter scales and reconciliation remain "
        "unverified. Do not reconstruct actual yield."
    )
}


# 7. ML prediction
pred = select_batch(
    load("rca_batch789_model_comparison.csv")
)

assert len(pred) == 6

rf = one(
    pred[
        pred["mode"].eq("early_30m") &
        pred["model"].eq("RandomForest")
    ]
)

ml_prediction = {
    "intended_use": "advisory_only",
    "models": records(pred),
    "selected_model": "rf_full_train.joblib",
    "selected_prediction_pct": rf["predicted"],
    "absolute_error_pp": rf["absolute_error"],
    "actual_spec_outside": specification["spec_outside"],
    "predicted_spec_outside": rf["predicted_spec_outside"],
    "prediction_failed_for_this_case": True,
    "uses_post_30min_data": False
}


# 8. SHAP evidence
sh = select_batch(
    load("rca_batch789_shap_ranked.csv")
)

assert len(sh) == 51
assert sh["feature"].nunique() == 51

baseline = float(sh["baseline"].iloc[0])
shap_sum = float(sh["shap"].sum())
prediction = float(sh["prediction"].iloc[0])

error = abs(baseline + shap_sum - prediction)

assert error < 1e-5
assert np.isclose(prediction, rf["predicted"])

shap_data = {
    "model": "rf_full_train.joblib",
    "feature_count": len(sh),
    "baseline_pct": baseline,
    "prediction_pct": prediction,
    "net_contribution_pp": shap_sum,
    "additivity_error": error,
    "local_all_features": records(sh),
    "top_negative": records(
        sh[sh["shap"] < 0]
        .sort_values("shap").head(10)
    ),
    "top_positive": records(
        sh[sh["shap"] > 0]
        .sort_values("shap", ascending=False)
        .head(10)
    ),
    "group_contributions": records(
        load("rca_batch789_shap_groups.csv")
    ),
    "global_importance": records(
        load(
            "shap_global.csv",
            BASE / "followup"
        )
    ),
    "imputed_features": [
        {
            "feature": "api_l_impurity",
            "observed_value": None,
            "model_input_value": float(
                sh.loc[
                    sh["feature"].eq("api_l_impurity"),
                    "input_value"
                ].iloc[0]
            ),
            "method": "training_median"
        }
    ],
    "interpretation": "model_explanation_not_causality"
}


# 9. Separate uncertainty model
uncertainty = one(
    select_batch(
        load("rca_batch789_uncertainty.csv")
    )
)

uncertainty["model"] = "rf_early_fixed.joblib"
uncertainty["nominal_coverage"] = 0.90
uncertainty["applies_to_full_train_model"] = False
uncertainty["range_screen"] = (
    "Exploratory per-product inner-training range"
)
uncertainty["formal_drift_detection"] = False


# 10. Verified literature references
references = [
    {
        "id": "REF01",
        "authors": "Zagar and Mihelic",
        "year": 2022,
        "doi": "10.1038/s41597-022-01203-x",
        "purpose": "Original dataset and definitions"
    },
    {
        "id": "REF02",
        "authors": "Narang et al.",
        "year": 2010,
        "doi": "10.1016/j.ijpharm.2010.08.027",
        "purpose": "Feeder conditions and tablet strength"
    },
    {
        "id": "REF03",
        "authors": "Sarkar et al.",
        "year": 2015,
        "doi": "10.1002/jps.24328",
        "purpose": "Compression conditions and capping"
    },
    {
        "id": "REF04",
        "authors": "Uzondu et al.",
        "year": 2018,
        "doi": "10.1016/j.ijpharm.2018.03.064",
        "purpose": "Ejection force and lubrication"
    },
    {
        "id": "REF05",
        "authors": "Honti et al.",
        "year": 2024,
        "doi": "10.1016/j.ijpharm.2024.124509",
        "purpose": "Explainable pharmaceutical process AI"
    }
]


# 11. Human-authored investigation hypotheses
# Case-specific: not generated by a validated RCA model
hypotheses = [
    {
        "id": "H01",
        "priority": 1,
        "title": "Speed change and restart conditions",
        "status": "unconfirmed",
        "evidence_paths": [
            "timeseries.priority_window",
            "timeseries.major_speed_segments",
            "timeseries.counterexample_same_speed"
        ],
        "counterevidence": (
            "Later operation at speed 300 showed "
            "no additional waste counter increase."
        ),
        "required_checks": [
            "Recipe and speed setpoint history",
            "Restart procedure and equipment alarms",
            "Tablet rejection reason records"
        ],
        "literature_ids": ["REF01", "REF02"]
    },
    {
        "id": "H02",
        "priority": 2,
        "title": "Main compression settings and control",
        "status": "unconfirmed",
        "evidence_paths": [
            "timeseries.early_comparison",
            "timeseries.full_comparison",
            "timeseries.waste_events"
        ],
        "counterevidence": (
            "Initial compression variability was low."
        ),
        "required_checks": [
            "Compression setpoints",
            "Approved tolerance limits",
            "Compression force alarms"
        ],
        "literature_ids": ["REF01", "REF03"]
    },
    {
        "id": "H03",
        "priority": 3,
        "title": "Ejection, tooling and lubrication",
        "status": "unconfirmed",
        "evidence_paths": [
            "timeseries.early_comparison",
            "timeseries.full_comparison",
            "shap.top_negative"
        ],
        "counterevidence": (
            "No direct tooling or defect records "
            "confirm a lubrication problem."
        ),
        "required_checks": [
            "Punch and die inspection records",
            "Lubrication and blending history",
            "Tablet defect inspection"
        ],
        "literature_ids": ["REF04"]
    },
    {
        "id": "H04",
        "priority": 4,
        "title": "Long internal stops",
        "status": "unconfirmed",
        "evidence_paths": [
            "timeseries.internal_stop_segments",
            "timeseries.stop_comparison"
        ],
        "counterevidence": (
            "Stop count was within historical range. "
            "Stops are not confirmed equipment failures."
        ),
        "required_checks": [
            "Equipment alarm history",
            "Operator log",
            "Planned versus unplanned stops"
        ],
        "literature_ids": ["REF01"]
    },
    {
        "id": "H05",
        "priority": 5,
        "title": "Feeding and filling conditions",
        "status": "unconfirmed",
        "evidence_paths": [
            "feature_comparison.all_features",
            "timeseries.full_comparison"
        ],
        "counterevidence": (
            "Mean filling depth was within "
            "historical range."
        ),
        "required_checks": [
            "Feeder settings",
            "Powder flow and filling conditions"
        ],
        "literature_ids": ["REF02"]
    },
    {
        "id": "H06",
        "priority": 6,
        "title": "Raw material and blend properties",
        "status": "unconfirmed",
        "evidence_paths": [
            "feature_comparison.group_summary",
            "feature_comparison.excluded_features"
        ],
        "counterevidence": (
            "No range violation among 19 comparable "
            "material and batch features."
        ),
        "required_checks": [
            "Raw material lot records",
            "Blend properties",
            "Missing impurity measurement"
        ],
        "literature_ids": ["REF01", "REF04"]
    }
]


# 12. Safety and limitations
limitations = {
    "scope": "Single-case retrospective research PoC",
    "approved_specifications_available": False,
    "confirmed_root_cause": None,
    "causality_established": False,
    "human_review_required": True,
    "autonomous_recipe_change_allowed": False,
    "autonomous_batch_release_allowed": False,
    "automatic_capa_execution_allowed": False,
    "warnings": [
        "Hypothetical yield limits are not GMP limits.",
        "Historical comparison does not prove causality.",
        "Same campaign reference is unavailable.",
        "Early operating stages may not be equivalent.",
        "Exact stop reasons and rejection causes are unknown.",
        "Counter units and yield reconciliation need review.",
        "SHAP explains prediction, not actual root cause.",
        "Initial ML prediction missed the severe yield loss.",
        "Input screening is not formal drift validation.",
        "Literature supports mechanisms, not case confirmation."
    ]
}


# 13. Assemble RCA evidence payload
payload = {
    "schema_version": "0.1.0",
    "case_id": f"code{CODE}_batch{BATCH}",
    "scenario_type": "retrospective_demo",
    "batch_info": batch_info,
    "specification": specification,
    "spc": spc,
    "reference": reference_info,
    "feature_comparison": feature_comparison,
    "timeseries": timeseries,
    "ml_prediction": ml_prediction,
    "shap": shap_data,
    "uncertainty": uncertainty,
    "hypotheses": hypotheses,
    "references": references,
    "limitations": limitations,
    "provenance": provenance
}


# 14. Validate key evidence
assert np.isclose(actual, 87.973)
assert np.isclose(lsl, 97.0)
assert specification["spec_outside"]
assert np.isclose(waste_total, 43707)
assert np.isclose(peak_waste, 30246)
assert np.isclose(
    timeseries["estimated_internal_stop_minutes"],
    176.3333333333
)

assert uncertainty["review_required"]
assert not uncertainty["covered"]


# 15. Save standard JSON
path = OUT / "rca_agent_batch789.json"

with path.open("w", encoding="utf-8") as f:
    json.dump(
        payload,
        f,
        ensure_ascii=False,
        indent=2,
        allow_nan=False
    )

with path.open(encoding="utf-8") as f:
    restored = json.load(f)

assert restored["case_id"] == payload["case_id"]

print("\n=== RCA AGENT EXPORT ===")
print("Case ID:", restored["case_id"])
print("Schema version:", restored["schema_version"])
print("Top-level keys:", list(restored.keys()))

print("\n=== KEY EVIDENCE ===")
print("Actual yield:", actual)
print("LSL:", lsl)
print("Spec outside:", specification["spec_outside"])
print("Reference batches:", len(reference))
print("Compared features:", len(fc))
print("Outside range:", len(outside))
print("Internal stops:", len(stops))
print("Waste counter increase:", waste_total)
print("Priority window share:", peak_waste / waste_total * 100)
print("ML prediction:", prediction)
print("SHAP features:", len(sh))
print("SHAP additivity error:", error)
print("Human review required:", uncertainty["review_required"])
print("Investigation hypotheses:", len(hypotheses))
print("Confirmed root cause:", limitations["confirmed_root_cause"])
print("Provenance files:", len(provenance))

print("\n=== SAVED FILE ===")
print(path)
print("Size:", path.stat().st_size, "bytes")

print("\n=== VALIDATION PASSED ===")
