from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import pandas as pd

OUT = Path("results/spec_all_products_q10")

CONFIG = Path(
    "config/yield_specifications_all_products_q10.json"
)

SCENARIOS = [
    (17, 789, 107, 99, 99),
    (15, 799, 38, 37, 33)
]


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("[PASS]", message)


def close(a, b):
    return math.isclose(
        float(a), float(b),
        rel_tol=1e-9,
        abs_tol=1e-8
    )


def read_json(path):

    def reject_constant(value):
        raise ValueError(
            f"Invalid JSON constant: {value}"
        )

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant
    )


config = read_json(CONFIG)

spec = pd.read_csv(
    OUT / "all_product_specifications_q10.csv"
)

candidates = pd.read_csv(
    OUT / "scenario_candidates_q10.csv"
)

# 1. Validate product configuration

print("\n=== 1. PRODUCT SPECIFICATION VALIDATION ===")

check(config["version"] == "0.2.0",
      "Configuration version")

check(len(config["products"]) == 25,
      "All 25 products")

check(spec["train_n"].sum() == 772,
      "772 training batches")

check(spec["test_n"].sum() == 232,
      "232 evaluation batches")

groups = Counter(
    p["data_group"]
    for p in config["products"].values()
)

check(groups == {"A": 7, "B": 9, "C": 9},
      "Product group distribution")

check(
    sum(
        p["lsl"] is not None
        for p in config["products"].values()
    ) == 16,
    "16 products with PoC specifications"
)

check(
    config["approved_manufacturing_spec"] is False,
    "Not an approved manufacturing specification"
)

for _, row in spec.iterrows():

    product = config["products"][str(int(row["code"]))]

    assert close(
        product["candidate_lsl"], row["lsl_q10"]
    )

    assert close(
        product["candidate_usl"], row["usl_candidate"]
    )

    assert product["data_group"] == row["data_group"]

    assert product["production_use_allowed"] is False
    assert product["automatic_alert_allowed"] is False

    if product["data_group"] == "C":
        assert product["lsl"] is None
        assert product["usl"] is None
    else:
        assert close(product["lsl"], row["poc_lsl"])
        assert close(product["usl"], row["poc_usl"])

print("[PASS] All product settings match CSV")


# 2. Validate detection results

print("\n=== 2. DETECTION VALIDATION ===")

check(len(candidates) == 21,
      "21 outside-specification batches")

check(candidates["outside_q10"].all(),
      "All candidates outside q10 specifications")

check(candidates["below_q10"].all(),
      "All 21 candidates below LSL")

check(candidates["newly_detected"].sum() == 17,
      "17 newly detected batches compared with q025")


# 3. Validate each scenario

for code, batch, ref_n, ts_n, waste_n in SCENARIOS:

    print(f"\n=== CODE {code} / BATCH {batch} ===")

    path = OUT / (
        f"rca_agent_code{code}_batch{batch}.json"
    )

    case = read_json(path)

    product = config["products"][str(code)]

    detection = candidates[
        (candidates["code"] == code) &
        (candidates["batch"] == batch)
    ]

    check(len(detection) == 1,
          "Unique scenario detection record")

    detection = detection.iloc[0]

    actual = case["batch_info"]["actual_yield_pct"]
    s = case["specification"]

    check(case["schema_version"] == "0.2.0",
          "Scenario schema version")

    check(close(actual, detection["tbl_yield"]),
          "Actual yield matches detection CSV")

    check(close(s["lsl_pct"], product["lsl"]),
          "LSL matches configuration")

    check(close(s["usl_pct"], product["usl"]),
          "USL matches configuration")

    check(close(s["gap_below_lsl_pp"],
                product["lsl"] - actual),
          "LSL violation magnitude")

    check(actual < product["lsl"],
          "Confirmed PoC lower-limit violation")

    # SPC

    spc = case["spc"]

    check(
        actual < spc["limits"]["i_lcl"]
        and spc["signals"]["i_low"],
        "I-chart lower-limit signal"
    )

    check(
        spc["observations"]["mr"] >
        spc["limits"]["mr_ucl"]
        and spc["signals"]["mr_signal"],
        "MR-chart upper-limit signal"
    )

    z = (
        actual - spc["observations"]["training_mean"]
    ) / spc["observations"]["training_std"]

    check(close(z, spc["observations"]["yield_z_score"]),
          "Yield Z-score recalculation")

    # Reference batches

    ref = pd.read_csv(
        OUT / f"reference_code_{code}.csv"
    )

    target = pd.read_csv(
        OUT / f"target_code_{code}_batch_{batch}.csv"
    )

    check(len(ref) == ref_n,
          "Reference batch count")

    check(
        (ref["code"] == code).all()
        and ref["tbl_yield"].between(
            product["lsl"], product["usl"]
        ).all(),
        "Same-product references within PoC limits"
    )

    check(batch not in set(ref["batch"]),
          "Target excluded from historical references")

    check(
        case["reference"]["count"] == ref_n,
        "Reference count matches scenario JSON"
    )

    # Recalculate feature statistics

    features = case[
        "feature_comparison"
    ]["all_features"]

    check(len(features) == 32,
          "32 compared numeric features")

    for feature, stats in features.items():

        values = pd.to_numeric(
            ref[feature], errors="coerce"
        ).dropna()

        value = float(target[feature].iloc[0])

        assert len(values) == stats["reference_n"]
        assert close(value, stats["target"])
        assert close(values.mean(), stats["mean"])
        assert close(values.median(), stats["median"])
        assert close(values.std(), stats["std"])
        assert close(values.min(), stats["minimum"])
        assert close(values.max(), stats["maximum"])

        assert close(
            values.quantile(0.25), stats["q1"]
        )

        assert close(
            values.quantile(0.75), stats["q3"]
        )

        if stats["ratio"] is not None:
            assert close(
                value / values.median(),
                stats["ratio"]
            )

        if stats["z_score"] is not None:
            assert close(
                (value - values.mean()) / values.std(),
                stats["z_score"]
            )

        outside = (
            value < values.min()
            or value > values.max()
        )

        assert bool(outside) == stats["outside_range"]

    print("[PASS] All feature statistics recalculated")

    process = [
        f for f in features.values()
        if f["category"] == "process"
    ]

    check(len(process) == 24,
          "24 process features")

    check(
        sum(f["outside_range"] for f in process)
        == case["feature_comparison"][
            "outside_reference_range_count"
        ],
        "Process range-violation count"
    )

    # Timeseries and waste

    ts = case["timeseries"]

    raw = pd.read_csv(
        OUT / f"waste_events_code_{code}.csv"
    ).sort_values("record_no")

    increments = (
        raw["waste"].diff().fillna(0).clip(lower=0)
    )

    total = increments.sum()

    check(
        close(
            total,
            ts["waste_summary"][
                "total_positive_increments"
            ]
        ),
        "Waste counter increments recalculated"
    )

    window = ts["priority_window"]

    focus = raw[
        (raw["elapsed_min"] >=
         window["start_elapsed_min"]) &
        (raw["elapsed_min"] <
         window["end_elapsed_min_exclusive"])
    ]

    focus_waste = increments.loc[focus.index].sum()

    check(close(
        focus_waste, window["waste_increment"]
    ), "Focus window waste increments")

    check(close(
        focus_waste / total * 100,
        window["waste_share_pct"]
    ), "Focus window waste share")

    # Historical timeseries

    hist = pd.read_csv(
        OUT / f"reference_timeseries_code_{code}.csv"
    )

    hist = hist[hist["group"] == "reference"]

    check(len(hist) == ts_n,
          "Historical timeseries reference count")

    check(hist["counter_valid"].sum() == waste_n,
          "Valid historical waste counter count")

    for metric, stats in ts[
        "historical_comparison"
    ].items():

        values = hist[metric].dropna()

        assert len(values) == stats["reference_n"]

        assert close(
            values.median(),
            stats["reference_median"]
        )

        assert close(
            values.quantile(0.95),
            stats["reference_q95"]
        )

        assert close(
            values.max(),
            stats["reference_max"]
        )

    print("[PASS] Historical timeseries statistics")

    # Hypothesis evidence

    check(len(case["hypotheses"]) == 5,
          "Five investigation hypotheses")

    for hypothesis in case["hypotheses"]:

        assert hypothesis["status"] == "unconfirmed"

        for pointer in hypothesis["evidence_paths"]:

            node = case

            for key in pointer.strip("/").split("/"):
                node = node[key]

    print("[PASS] All hypothesis evidence paths")

    # Safety and limitations

    limits = case["limitations"]

    check(
        limits["confirmed_root_cause"] is None
        and limits["causality_established"] is False,
        "No unsupported causal conclusion"
    )

    check(
        limits["human_review_required"] is True,
        "Human review required"
    )

    check(
        not limits["autonomous_recipe_change_allowed"]
        and not limits["autonomous_batch_release_allowed"]
        and not limits["automatic_capa_execution_allowed"],
        "Autonomous manufacturing actions prohibited"
    )

    check(
        all(
            case[key]["included"] is False
            for key in [
                "ml_prediction", "shap", "uncertainty"
            ]
        ),
        "No unvalidated ML results included"
    )

    # Graphs and provenance

    check(len(case["artifacts"]["graphs"]) == 4,
          "Four analysis graphs")

    for graph in case["artifacts"]["graphs"]:

        graph_path = Path(graph)

        assert graph_path.exists()

        with graph_path.open("rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"

    print("[PASS] All PNG graph files")

    for filename, metadata in case["provenance"].items():

        source = Path(filename)

        assert source.exists(), filename

        actual_hash = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()

        assert actual_hash == metadata["sha256"], filename

    print(
        "[PASS] All provenance hashes:",
        len(case["provenance"])
    )

    print(f"\nCODE {code} / BATCH {batch}: PASSED")


print("\n========================================")
print("ALL FINAL VALIDATIONS PASSED")
print("========================================")
