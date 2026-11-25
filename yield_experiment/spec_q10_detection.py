from pathlib import Path
import pandas as pd

SRC = Path("results/spec_all_products")
OUT = Path("results/spec_all_products_q10")

df = pd.read_csv(SRC / "all_product_batch_split.csv")
spec = pd.read_csv(OUT / "all_product_specifications_q10.csv")

test = df[df["phase"] == "test"].copy()

test = test.merge(
    spec[[
        "code", "train_n", "data_group",
        "lsl_q025", "lsl_q05", "lsl_q10",
        "usl_candidate"
    ]],
    on="code",
    how="left",
    validate="many_to_one"
)

assert len(test) == 232
assert (test["data_group"] == "A").all()

# 1. Detect specification violations

for q in ["q025", "q05", "q10"]:
    test[f"below_{q}"] = (
        test["tbl_yield"] < test[f"lsl_{q}"]
    )

    test[f"above_{q}"] = (
        test["tbl_yield"] > test["usl_candidate"]
    )

    test[f"outside_{q}"] = (
        test[f"below_{q}"] | test[f"above_{q}"]
    )

# 2. Compare detection results

print("\n=== OVERALL DETECTION COMPARISON ===")

rows = []

for q in ["q025", "q05", "q10"]:
    below = int(test[f"below_{q}"].sum())
    above = int(test[f"above_{q}"].sum())
    outside = int(test[f"outside_{q}"].sum())

    rows.append({
        "criterion": q,
        "test_n": len(test),
        "below_lsl": below,
        "above_usl": above,
        "outside": outside,
        "outside_rate_pct": outside / len(test) * 100
    })

summary = pd.DataFrame(rows)

print(summary.round(4).to_string(index=False))

# 3. Product-level comparison

print("\n=== PRODUCT LEVEL COMPARISON ===")

product_summary = test.groupby("code").agg(
    test_n=("tbl_yield", "size"),
    outside_q025=("outside_q025", "sum"),
    outside_q05=("outside_q05", "sum"),
    outside_q10=("outside_q10", "sum")
).reset_index()

product_summary["additional_q10"] = (
    product_summary["outside_q10"]
    - product_summary["outside_q025"]
)

print(product_summary.to_string(index=False))

# 4. Extract q10 violations

outside = test[test["outside_q10"]].copy()

outside["lsl_gap"] = (
    outside["lsl_q10"] - outside["tbl_yield"]
).clip(lower=0)

outside["usl_gap"] = (
    outside["tbl_yield"] - outside["usl_candidate"]
).clip(lower=0)

outside["violation_gap"] = (
    outside["lsl_gap"] + outside["usl_gap"]
)

outside["newly_detected"] = (
    ~outside["outside_q025"]
)

outside = outside.sort_values(
    "violation_gap",
    ascending=False
)

cols = [
    "code", "batch", "tbl_yield",
    "lsl_q025", "lsl_q10", "usl_candidate",
    "violation_gap", "newly_detected"
]

print("\n=== Q10 OUTSIDE BATCHES ===")
print(outside[cols].round(4).to_string(index=False))

print("\n=== NEWLY DETECTED BATCHES ===")

new = outside[outside["newly_detected"]]

print(new[cols].round(4).to_string(index=False))
print("Additional batches:", len(new))

# 5. Validate and save

assert (
    ~test["outside_q025"] | test["outside_q05"]
).all()

assert (
    ~test["outside_q05"] | test["outside_q10"]
).all()

summary.to_csv(
    OUT / "detection_comparison.csv",
    index=False
)

product_summary.to_csv(
    OUT / "product_detection_summary.csv",
    index=False
)

outside.to_csv(
    OUT / "q10_outside_batches.csv",
    index=False
)

test.to_csv(
    OUT / "all_evaluation_batches.csv",
    index=False
)

print("\nALL CHECKS PASSED")
