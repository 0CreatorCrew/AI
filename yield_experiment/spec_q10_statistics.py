from pathlib import Path
import pandas as pd

SRC = Path("results/spec_all_products")
OUT = Path("results/spec_all_products_q10")

df = pd.read_csv(
    SRC / "all_product_batch_split.csv",
    parse_dates=["start_ts", "end_ts"]
)

outside = pd.read_csv(
    OUT / "q10_outside_batches.csv"
)

limits = []
records = []

# 1. Calculate product-level statistics

for code in [13, 15, 17, 23]:

    train = df[
        (df["code"] == code) &
        (df["phase"] == "train")
    ].sort_values(["end_ts", "batch"]).copy()

    test = df[
        (df["code"] == code) &
        (df["phase"] == "test")
    ].sort_values(["end_ts", "batch"]).copy()

    y = train["tbl_yield"]

    mean = y.mean()
    std = y.std()

    # Moving range
    mr = y.diff().abs()
    mr_bar = mr.mean()

    # I-MR control limits
    sigma = mr_bar / 1.128

    i_ucl = mean + 3 * sigma
    i_lcl = mean - 3 * sigma
    mr_ucl = 3.267 * mr_bar

    limits.append({
        "code": code,
        "train_n": len(train),
        "mean": mean,
        "std": std,
        "i_ucl": i_ucl,
        "i_lcl": i_lcl,
        "mr_bar": mr_bar,
        "mr_ucl": mr_ucl
    })

    # 2. Evaluate later batches

    combined = pd.concat(
        [train, test],
        ignore_index=True
    )

    combined["mr"] = (
        combined["tbl_yield"].diff().abs()
    )

    t = combined[
        combined["phase"] == "test"
    ].copy()

    t["mean"] = mean
    t["std"] = std

    t["z_score"] = (
        (t["tbl_yield"] - mean) / std
    )

    t["i_lcl"] = i_lcl
    t["i_ucl"] = i_ucl
    t["mr_ucl"] = mr_ucl

    t["i_low"] = t["tbl_yield"] < i_lcl
    t["i_high"] = t["tbl_yield"] > i_ucl

    t["mr_signal"] = t["mr"] > mr_ucl

    t["imr_signal"] = (
        t["i_low"] |
        t["i_high"] |
        t["mr_signal"]
    )

    records.append(t)

# 3. Combine statistics with q10 violations

control = pd.DataFrame(limits)
stats = pd.concat(records, ignore_index=True)

assert len(stats) == 232

candidates = outside.merge(
    stats[[
        "code", "batch",
        "mean", "std", "z_score",
        "i_lcl", "i_ucl",
        "mr", "mr_ucl",
        "i_low", "i_high",
        "mr_signal", "imr_signal"
    ]],
    on=["code", "batch"],
    how="left",
    validate="one_to_one"
)

assert len(candidates) == 21
assert candidates["z_score"].notna().all()

candidates = candidates.sort_values("z_score")

# 4. Print results

print("\n=== PRODUCT CONTROL LIMITS ===")

print(
    control.round(4).to_string(index=False)
)

print("\n=== Q10 SCENARIO CANDIDATES ===")

cols = [
    "code", "batch", "tbl_yield",
    "lsl_q10", "violation_gap",
    "z_score", "i_low", "mr_signal",
    "newly_detected"
]

print(
    candidates[cols]
    .round(4)
    .to_string(index=False)
)

print("\n=== SPEC AND I-MR OVERLAP ===")

print("Specification violations:", len(candidates))

print(
    "I-chart lower signals:",
    candidates["i_low"].sum()
)

print(
    "MR-chart signals:",
    candidates["mr_signal"].sum()
)

print(
    "Specification and I-MR signals:",
    candidates["imr_signal"].sum()
)

# 5. Save results

control.to_csv(
    OUT / "control_limits_q10.csv",
    index=False
)

stats.to_csv(
    OUT / "evaluation_statistics_q10.csv",
    index=False
)

candidates.to_csv(
    OUT / "scenario_candidates_q10.csv",
    index=False
)

print("\nALL CHECKS PASSED")
