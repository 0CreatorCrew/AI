from pathlib import Path
import json
import hashlib
import math

PATH = Path(
    "results/spec_scenario/rca_agent_batch789.json"
)

with PATH.open(encoding="utf-8") as f:
    data = json.load(f)


def close(a, b):
    return math.isclose(
        float(a), float(b),
        rel_tol=0, abs_tol=1e-6
    )


# 1. Required fields
required = {
    "schema_version", "case_id", "scenario_type",
    "batch_info", "specification", "spc",
    "reference", "feature_comparison",
    "timeseries", "ml_prediction", "shap",
    "uncertainty", "hypotheses", "references",
    "limitations", "provenance"
}

assert required.issubset(data.keys())

print("[PASS] Required JSON fields")


# 2. Batch identity
batch = data["batch_info"]

assert batch["code"] == 17
assert batch["batch"] == 789
assert data["case_id"] == "code17_batch789"

print("[PASS] Batch identity")


# 3. Specification consistency
spec = data["specification"]
actual = batch["actual_yield_pct"]

assert close(actual, spec["actual_pct"])

assert spec["spec_outside"] == (
    actual < spec["lsl_pct"]
    or actual > spec["usl_pct"]
)

assert close(
    spec["gap_below_lsl_pp"],
    max(spec["lsl_pct"] - actual, 0)
)

assert spec["approved_manufacturing_spec"] is False
assert spec["actual_gmp_oos"] is None

print("[PASS] Hypothetical specification consistency")


# 4. SPC signals
signals = data["spc"]["signals"]

assert signals["r1_low"]
assert signals["mr_high"]
assert close(signals["tbl_yield"], actual)

print("[PASS] SPC evidence")


# 5. Reference and feature comparison
ref = data["reference"]
fc = data["feature_comparison"]

assert ref["count"] == 114
assert ref["approved_normal_status_verified"] is False

assert len(fc["all_features"]) == 58
assert len(fc["outside_reference_range"]) == 12

assert sum(
    g["total"] for g in fc["group_summary"]
) == 58

assert sum(
    g["outside"] for g in fc["group_summary"]
) == 12

print("[PASS] Reference and feature comparison")


# 6. Time series consistency
ts = data["timeseries"]

assert len(ts["internal_stop_segments"]) == (
    ts["internal_stop_count"]
)

assert close(
    sum(
        s["estimated_minutes"]
        for s in ts["internal_stop_segments"]
    ),
    ts["estimated_internal_stop_minutes"]
)

waste = sum(
    e["waste_increase"]
    for e in ts["waste_events"]
)

assert close(
    waste,
    ts["waste_counter_total_increase"]
)

assert close(
    waste,
    sum(
        w["waste_increase"]
        for w in ts["window_summary"]
    )
)

assert close(
    waste,
    sum(
        p["waste_increase"]
        for p in ts["operating_phase_summary"]
    )
)

peak = ts["priority_window"]

assert close(
    peak["share_pct"],
    peak["waste_counter_increase"] / waste * 100
)

print("[PASS] Time series and waste consistency")


# 7. ML and SHAP consistency
ml = data["ml_prediction"]
sh = data["shap"]

assert close(
    ml["absolute_error_pp"],
    abs(ml["selected_prediction_pct"] - actual)
)

assert len(sh["local_all_features"]) == 51

assert len({
    f["feature"] for f in sh["local_all_features"]
}) == 51

shap_sum = sum(
    f["shap"]
    for f in sh["local_all_features"]
)

assert close(
    sh["baseline_pct"] + shap_sum,
    sh["prediction_pct"]
)

assert close(
    sh["prediction_pct"],
    ml["selected_prediction_pct"]
)

assert sh["model"] == ml["selected_model"]

print("[PASS] ML prediction and SHAP additivity")


# 8. Uncertainty model separation
u = data["uncertainty"]

assert u["model"] != ml["selected_model"]
assert u["applies_to_full_train_model"] is False

assert close(
    (u["lower"] + u["upper"]) / 2,
    u["prediction"]
)

assert u["covered"] == (
    u["lower"] <= actual <= u["upper"]
)

assert u["review_required"] == (
    u["outside_count"] > 0
    or u["missing_count"] > 0
)

print("[PASS] Separate uncertainty model")


# 9. Hypothesis evidence links
reference_ids = {
    r["id"] for r in data["references"]
}

for h in data["hypotheses"]:

    assert h["status"] == "unconfirmed"

    assert h["required_checks"]
    assert h["counterevidence"]

    for ref_id in h["literature_ids"]:
        assert ref_id in reference_ids

    for evidence_path in h["evidence_paths"]:

        current = data

        for key in evidence_path.split("."):
            assert key in current, evidence_path
            current = current[key]

print("[PASS] Hypothesis evidence links")


# 10. Human review and safety constraints
limits = data["limitations"]

assert limits["confirmed_root_cause"] is None
assert limits["causality_established"] is False
assert limits["human_review_required"] is True

for key in [
    "autonomous_recipe_change_allowed",
    "autonomous_batch_release_allowed",
    "automatic_capa_execution_allowed"
]:
    assert limits[key] is False

print("[PASS] Human review constraints")


# 11. Source file integrity
for name, info in data["provenance"].items():

    source = Path(name)

    assert source.exists(), name

    actual_hash = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    assert actual_hash == info["sha256"], name

print("[PASS] Source file SHA-256 integrity")


# 12. Standard JSON serialization
json.dumps(data, allow_nan=False)

print("[PASS] Standard JSON compatibility")


# Final summary
print("\n=== FINAL VALIDATION SUMMARY ===")
print("Case ID:", data["case_id"])
print("Schema version:", data["schema_version"])
print("Validation categories: 12")
print("Evidence files:", len(data["provenance"]))
print("Hypotheses:", len(data["hypotheses"]))
print("Human review required:", limits["human_review_required"])
print("Confirmed root cause:", limits["confirmed_root_cause"])
print("\nALL VALIDATION CHECKS PASSED")
