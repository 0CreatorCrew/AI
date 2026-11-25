from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/spec_all_products_q10")

SCENARIOS = [(17, 789), (15, 799)]

QUALITY_COLS = [
    "Drug release average (%)",
    "Drug release min (%)",
    "Residual solvent",
    "Total impurities",
    "Impurity O",
    "Impurity L"
]

LOSS_COLS = [
    "total_waste",
    "startup_waste"
]

EXCLUDE_COLS = [
    "code",
    "batch",
    "tbl_yield",
    "weekend"
]

all_results = []

for code, batch in SCENARIOS:

    ref = pd.read_csv(
        OUT / f"reference_code_{code}.csv"
    )

    target = pd.read_csv(
        OUT / f"target_code_{code}_batch_{batch}.csv"
    )

    assert len(target) == 1
    assert (ref["code"] == code).all()
    assert target["code"].iloc[0] == code
    assert target["batch"].iloc[0] == batch

    features = [
        col for col in ref.columns
        if col not in EXCLUDE_COLS
    ]

    rows = []

    for col in features:

        values = pd.to_numeric(
            ref[col], errors="coerce"
        ).dropna()

        value = pd.to_numeric(
            target[col], errors="coerce"
        ).iloc[0]

        if col in QUALITY_COLS:
            category = "quality"
        elif col in LOSS_COLS:
            category = "production_loss"
        else:
            category = "process"

        if len(values) < 2 or pd.isna(value):
            print("Insufficient data:", code, col)
            continue

        mean = values.mean()
        median = values.median()
        std = values.std()
        minimum = values.min()
        maximum = values.max()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr

        z_score = (
            (value - mean) / std
            if std > 0 else np.nan
        )

        ratio = (
            value / median
            if median != 0 else np.nan
        )

        outside_range = (
            value < minimum or value > maximum
        )

        iqr_outlier = (
            value < iqr_lower or value > iqr_upper
        ) if iqr > 0 else None

        rows.append({
            "code": code,
            "batch": batch,
            "category": category,
            "feature": col,
            "reference_n": len(values),
            "reference_missing": len(ref) - len(values),
            "mean": mean,
            "median": median,
            "std": std,
            "minimum": minimum,
            "maximum": maximum,
            "target": value,
            "difference": value - median,
            "ratio": ratio,
            "z_score": z_score,
            "abs_z_score": abs(z_score),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "iqr_lower": iqr_lower,
            "iqr_upper": iqr_upper,
            "outside_range": outside_range,
            "iqr_outlier": iqr_outlier
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["outside_range", "abs_z_score"],
        ascending=[False, False]
    )

    process = result[
        result["category"] == "process"
    ]

    loss = result[
        result["category"] == "production_loss"
    ]

    quality = result[
        result["category"] == "quality"
    ]

    cols = [
        "feature", "median", "target",
        "ratio", "z_score",
        "outside_range", "iqr_outlier"
    ]

    print(f"\n=== CODE {code} / BATCH {batch} ===")
    print("Reference batches:", len(ref))

    print("\n--- PROCESS VARIABLE COMPARISON ---")
    print(process[cols].round(4).to_string(index=False))

    print("\n--- PRODUCTION LOSS COMPARISON ---")
    print(loss[cols].round(4).to_string(index=False))

    print("\n--- QUALITY VARIABLE COMPARISON ---")
    print(quality[cols].round(4).to_string(index=False))

    print("\n--- PROCESS VARIABLES OUTSIDE RANGE ---")

    outside = process[process["outside_range"]]

    print(
        outside[[
            "feature", "minimum",
            "maximum", "target", "z_score"
        ]].round(4).to_string(index=False)
    )

    print("\n--- SUMMARY ---")
    print("Process variables:", len(process))
    print("Outside historical range:", len(outside))
    print(
        "IQR outliers:",
        (process["iqr_outlier"] == True).sum()
    )
    print(
        "Absolute Z-score >= 3:",
        (process["abs_z_score"] >= 3).sum()
    )

    result.to_csv(
        OUT / f"rca_comparison_code_{code}.csv",
        index=False
    )

    all_results.append(result)

pd.concat(
    all_results,
    ignore_index=True
).to_csv(
    OUT / "rca_comparison_all.csv",
    index=False
)

print("\nALL COMPARISONS COMPLETED")
