from pathlib import Path
import math
import pandas as pd

SRC = Path("results/spec_all_products")
OUT = Path("results/spec_all_products_q10")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC / "all_product_batch_split.csv")

train = df[df["phase"] == "train"].copy()
test = df[df["phase"] == "test"].copy()

assert len(df) == 1005
assert len(train) == 772
assert len(test) == 232

def round_half(value):
    return round(float(value) * 2) / 2

def ceil_half(value):
    return math.ceil(float(value) * 2) / 2

rows = []

for code, g in train.groupby("code"):
    y = g["tbl_yield"]
    n = len(y)

    q025 = y.quantile(0.025)
    q05 = y.quantile(0.05)
    q10 = y.quantile(0.10)

    lsl_q025 = round_half(q025)
    lsl_q05 = round_half(q05)
    lsl_q10 = round_half(q10)
    usl = ceil_half(y.max())

    if n >= 40:
        group = "A"
    elif n >= 10:
        group = "B"
    else:
        group = "C"

    usable = group != "C" and lsl_q10 < usl

    rows.append({
        "code": int(code),
        "train_n": n,
        "test_n": len(test[test["code"] == code]),
        "mean": y.mean(),
        "std": y.std(),
        "q025": q025,
        "q05": q05,
        "q10": q10,
        "lsl_q025": lsl_q025,
        "lsl_q05": lsl_q05,
        "lsl_q10": lsl_q10,
        "lsl_increase": lsl_q10 - lsl_q025,
        "usl_candidate": usl,
        "poc_lsl": lsl_q10 if usable else None,
        "poc_usl": usl if usable else None,
        "data_group": group
    })

result = pd.DataFrame(rows)

assert len(result) == 25
assert (result["lsl_q10"] >= result["lsl_q025"]).all()

print("\n=== DATA SUMMARY ===")
print("Training batches:", len(train))
print("Evaluation batches:", len(test))

print("\n=== LSL COMPARISON ===")
print(
    result[[
        "code", "train_n", "test_n",
        "lsl_q025", "lsl_q05", "lsl_q10",
        "lsl_increase", "usl_candidate",
        "data_group"
    ]].round(4).to_string(index=False)
)

print("\n=== GROUP SUMMARY ===")
print(result["data_group"].value_counts())

result.to_csv(
    OUT / "all_product_specifications_q10.csv",
    index=False
)

print("\n=== SAVED FILE ===")
print(OUT / "all_product_specifications_q10.csv")

print("\nALL CHECKS PASSED")
