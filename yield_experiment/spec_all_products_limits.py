from pathlib import Path
import math
import pandas as pd

OUT = Path("results/spec_all_products")

df = pd.read_csv(
    OUT / "all_product_batch_split.csv",
    parse_dates=["start_ts", "end_ts"]
)

CUTOFF = pd.Timestamp("2020-10-19 02:16:00")

train = df[df["phase"] == "train"].copy()
test = df[df["phase"] == "test"].copy()

# 1. Validate temporal split
assert len(df) == 1005
assert len(train) == 772
assert len(test) == 232
assert (df["phase"] == "boundary").sum() == 1

assert (train["end_ts"] < CUTOFF).all()
assert (test["start_ts"] >= CUTOFF).all()

# 2. Rounding functions
def round_half(value):
    return round(float(value) * 2) / 2

def ceil_half(value):
    return math.ceil(float(value) * 2) / 2

# 3. Calculate product-level specifications
rows = []

for code in sorted(df["code"].unique()):

    g = train[train["code"] == code]
    t = test[test["code"] == code]

    y = g["tbl_yield"]

    n = len(y)

    assert n > 0
    assert y.notna().all()

    q01 = y.quantile(0.01)
    q025 = y.quantile(0.025)
    q05 = y.quantile(0.05)

    lsl_a = round_half(q01)
    lsl_b = round_half(q025)
    lsl_c = round_half(q05)

    usl = ceil_half(y.max())

    if n >= 40:
        group = "A"
        status = "poc_candidate"
    elif n >= 10:
        group = "B"
        status = "provisional_review"
    else:
        group = "C"
        status = "insufficient_data_hold"

    usable = (
        group != "C"
        and lsl_b < usl
    )

    rows.append({
        "code": int(code),
        "total_n": int((df["code"] == code).sum()),
        "train_n": n,
        "test_n": len(t),
        "boundary_n": int(
            (
                (df["code"] == code) &
                (df["phase"] == "boundary")
            ).sum()
        ),
        "mean": y.mean(),
        "median": y.median(),
        "std": y.std(),
        "minimum": y.min(),
        "maximum": y.max(),
        "q01": q01,
        "q025": q025,
        "q05": q05,
        "lsl_a": lsl_a,
        "lsl_b": lsl_b,
        "lsl_c": lsl_c,
        "usl_candidate": usl,
        "poc_lsl": lsl_b if usable else None,
        "poc_usl": usl if usable else None,
        "train_outside_a": int(
            ((y < lsl_a) | (y > usl)).sum()
        ),
        "train_outside_b": int(
            ((y < lsl_b) | (y > usl)).sum()
        ),
        "train_outside_c": int(
            ((y < lsl_c) | (y > usl)).sum()
        ),
        "data_group": group,
        "status": status,
        "test_available": len(t) > 0
    })

result = pd.DataFrame(rows)

assert len(result) == 25
assert result["train_n"].sum() == 772

# 4. Product statistics
print("\n=== TRAINING YIELD STATISTICS ===")

print(
    result[
        [
            "code", "train_n", "test_n",
            "mean", "median", "std",
            "minimum", "maximum",
            "q01", "q025", "q05"
        ]
    ].round(4).to_string(index=False)
)

# 5. Specification candidates
print("\n=== ALL 25 PRODUCT SPECIFICATIONS ===")

print(
    result[
        [
            "code", "train_n",
            "lsl_a", "lsl_b", "lsl_c",
            "usl_candidate",
            "data_group", "status"
        ]
    ].to_string(index=False)
)

# 6. Recommended PoC specification table
print("\n=== POC SPECIFICATION TABLE ===")

print(
    result[
        [
            "code", "train_n", "test_n",
            "poc_lsl", "poc_usl", "status"
        ]
    ].to_string(index=False)
)

# 7. Previous specification consistency
print("\n=== PREVIOUS SPECIFICATION CHECK ===")

expected = {
    13: (70, 60, 97.0, 100.0),
    17: (119, 88, 97.0, 100.0),
    23: (126, 61, 96.5, 100.0)
}

for code, values in expected.items():

    r = result[result["code"] == code].iloc[0]

    observed = (
        int(r["train_n"]),
        int(r["test_n"]),
        float(r["lsl_b"]),
        float(r["usl_candidate"])
    )

    print(
        f"Code {code}:",
        "expected =", values,
        "| observed =", observed,
        "| match =", observed == values
    )

    assert observed == values

# 8. Candidate comparison
print("\n=== CANDIDATE COMPARISON ===")

for candidate in ["a", "b", "c"]:

    outside = result[
        f"train_outside_{candidate}"
    ].sum()

    print(
        f"Candidate {candidate.upper()}:",
        "outside =", outside,
        "| rate =", round(outside / len(train) * 100, 4)
    )

# 9. Data availability
print("\n=== DATA GROUP SUMMARY ===")
print(result["data_group"].value_counts().to_string())

print("\n=== INSUFFICIENT DATA PRODUCTS ===")

print(
    result[result["data_group"] == "C"][
        ["code", "train_n", "lsl_b", "usl_candidate"]
    ].to_string(index=False)
)

print("\n=== PRODUCTS WITH TEST DATA ===")

print(
    result[result["test_available"]][
        ["code", "train_n", "test_n"]
    ].to_string(index=False)
)

# 10. Save results
result.to_csv(
    OUT / "all_product_specifications.csv",
    index=False
)

result[
    [
        "code", "train_n", "test_n",
        "q025", "lsl_b", "usl_candidate",
        "poc_lsl", "poc_usl",
        "data_group", "status"
    ]
].to_csv(
    OUT / "all_product_spec_summary.csv",
    index=False
)

print("\n=== SAVED FILES ===")
print(OUT / "all_product_specifications.csv")
print(OUT / "all_product_spec_summary.csv")

print("\nALL CHECKS PASSED")
