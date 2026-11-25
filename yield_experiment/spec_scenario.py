import json
from pathlib import Path

import numpy as np
import pandas as pd


SPEC_PATH = Path("config/yield_specifications.json")
DATA_PATH = Path("results/followup/spc_signals.csv")
OUT = Path("results/spec_scenario")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with SPEC_PATH.open(encoding="utf-8") as f:
        config = json.load(f)

    specs = config["products"]

    df = pd.read_csv(DATA_PATH)
    test = df[df["phase"] == "test"].copy()

    assert test.groupby("code").size().to_dict() == {
        13: 60, 17: 88, 23: 61
    }
    assert test["tbl_yield"].notna().all()
    assert not test.duplicated(["code", "batch"]).any()

    # 1. Load product-specific specification limits
    test["lsl"] = test["code"].map(
        lambda c: specs[str(c)]["lsl"]
    )

    test["usl"] = test["code"].map(
        lambda c: specs[str(c)]["usl"]
    )

    assert (test["lsl"] < test["usl"]).all()

    # 2. Specification check
    test["below_lsl"] = (
        test["tbl_yield"] < test["lsl"]
    )

    test["above_usl"] = (
        test["tbl_yield"] > test["usl"]
    )

    test["spec_outside"] = (
        test["below_lsl"] | test["above_usl"]
    )

    test["spec_status"] = np.select(
        [test["below_lsl"], test["above_usl"]],
        ["below_lsl", "above_usl"],
        default="within_spec"
    )

    # 3. Magnitude of specification violation
    test["below_gap_pp"] = (
        test["lsl"] - test["tbl_yield"]
    ).clip(lower=0)

    test["above_gap_pp"] = (
        test["tbl_yield"] - test["usl"]
    ).clip(lower=0)

    test["spec_gap_pp"] = (
        test["below_gap_pp"] +
        test["above_gap_pp"]
    )

    # 4. Compare with existing SPC signals
    test["spc_signal"] = test["any_signal"].map(
        lambda v: str(v).strip().lower()
    ).map({
        "true": True, "false": False,
        "1": True, "0": False,
        "1.0": True, "0.0": False
    })

    assert test["spc_signal"].notna().all()
    test["spc_signal"] = test["spc_signal"].astype(bool)

    test["signal_type"] = np.select(
        [
            test["spec_outside"] & test["spc_signal"],
            test["spec_outside"] & ~test["spc_signal"],
            ~test["spec_outside"] & test["spc_signal"]
        ],
        [
            "A_both",
            "B_spec_only",
            "C_spc_only"
        ],
        default="D_neither"
    )

    test["spec_version"] = config["version"]

    # 5. Product-level summary
    summary = test.groupby("code").agg(
        total=("batch", "size"),
        below_lsl=("below_lsl", "sum"),
        above_usl=("above_usl", "sum"),
        spec_outside=("spec_outside", "sum"),
        spc_signal=("spc_signal", "sum")
    )

    summary["within_spec"] = (
        summary["total"] -
        summary["spec_outside"]
    )

    summary["spec_outside_rate_pct"] = (
        summary["spec_outside"] /
        summary["total"] * 100
    )

    # 6. Signal overlap
    signal_types = [
        "A_both",
        "B_spec_only",
        "C_spc_only",
        "D_neither"
    ]

    overlap = pd.crosstab(
        test["code"],
        test["signal_type"]
    ).reindex(columns=signal_types, fill_value=0)

    overlap["total"] = overlap.sum(axis=1)
    overlap.loc["Total"] = overlap.sum(axis=0)

    # 7. List specification violations
    outside = test[test["spec_outside"]].copy()

    outside = outside.sort_values(
        "spec_gap_pp", ascending=False
    )

    columns = [
        "code", "batch", "tbl_yield",
        "lsl", "usl", "spec_status",
        "spec_gap_pp",
        "r1_low", "r1_high",
        "r2_low", "r2_high",
        "r3_low", "r3_high",
        "mr_high", "signal_type"
    ]

    # 8. Save results
    test.to_csv(
        OUT / "spec_detection_all.csv",
        index=False
    )

    summary.to_csv(
        OUT / "spec_detection_summary.csv"
    )

    outside.to_csv(
        OUT / "spec_outside_batches.csv",
        index=False
    )

    overlap.to_csv(
        OUT / "spec_spc_overlap.csv"
    )

    # 9. Display results
    print("\n=== PRODUCT SPEC DETECTION ===")
    print(summary.round(4).to_string())

    print("\n=== SPEC AND SPC OVERLAP ===")
    print(overlap.to_string())

    print("\n=== OUTSIDE SPEC BATCHES ===")
    print(
        outside[columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== TOTAL ===")
    print("Evaluation batches:", len(test))
    print("Outside spec:", int(test["spec_outside"].sum()))
    print("SPC signals:", int(test["spc_signal"].sum()))

    print("\n=== SAVED FILES ===")

    for name in [
        "spec_detection_all.csv",
        "spec_detection_summary.csv",
        "spec_outside_batches.csv",
        "spec_spc_overlap.csv"
    ]:
        print(OUT / name)


if __name__ == "__main__":
    main()
