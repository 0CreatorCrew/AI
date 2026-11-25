from pathlib import Path
from collections import Counter
import json
import pandas as pd

CONFIG = Path("config")
OUT = Path("results/spec_all_products_q10")

OLD_PATH = (
    CONFIG / "yield_specifications_all_products.json"
)

NEW_PATH = (
    CONFIG / "yield_specifications_all_products_q10.json"
)

# 1. Load existing configuration

old = json.loads(
    OLD_PATH.read_text(encoding="utf-8")
)

spec = pd.read_csv(
    OUT / "all_product_specifications_q10.csv"
)

candidates = pd.read_csv(
    OUT / "scenario_candidates_q10.csv"
)

assert len(spec) == 25
assert spec["code"].nunique() == 25
assert spec["train_n"].sum() == 772
assert spec["test_n"].sum() == 232
assert len(candidates) == 21

assert old["approved_manufacturing_spec"] is False

# 2. Create updated configuration

new = {
    "version": "0.2.0",
    "status": "research_poc_only",
    "target": old["target"],
    "unit": old["unit"],
    "reference_cutoff": old["reference_cutoff"],
    "approved_manufacturing_spec": False,
    "method": {
        **old["method"],
        "lsl": (
            "10th percentile of historical training yields, "
            "rounded to the nearest 0.5 percentage point"
        )
    },
    "products": {}
}

# Convert pandas missing values to JSON null

def nullable_number(value):
    if pd.isna(value):
        return None
    return float(value)

# 3. Update all 25 products

for _, row in spec.iterrows():

    code = str(int(row["code"]))
    group = str(row["data_group"])

    old_product = old["products"][code]

    # Ensure that the existing data group is unchanged
    assert old_product["data_group"] == group

    candidate_lsl = float(row["lsl_q10"])
    candidate_usl = float(row["usl_candidate"])

    lsl = nullable_number(row["poc_lsl"])
    usl = nullable_number(row["poc_usl"])

    new["products"][code] = {
        "train_n": int(row["train_n"]),
        "test_n": int(row["test_n"]),

        "q025": float(row["q025"]),
        "q10": float(row["q10"]),

        "candidate_lsl": candidate_lsl,
        "candidate_usl": candidate_usl,

        "lsl": lsl,
        "usl": usl,

        "data_group": group,
        "status": old_product["status"],

        "evaluation_data_available":
            int(row["test_n"]) > 0,

        "production_use_allowed": False,
        "automatic_alert_allowed": False
    }

# 4. Validate product specifications

products = new["products"]

assert len(products) == 25

groups = Counter(
    p["data_group"] for p in products.values()
)

assert groups == {"A": 7, "B": 9, "C": 9}

for code, product in products.items():

    assert product["production_use_allowed"] is False
    assert product["automatic_alert_allowed"] is False

    if product["data_group"] == "C":
        assert product["lsl"] is None
        assert product["usl"] is None
    else:
        assert product["lsl"] is not None
        assert product["usl"] is not None
        assert product["lsl"] <= product["usl"]

    assert (
        product["candidate_usl"]
        == old["products"][code]["candidate_usl"]
    )

# Validate the two scenario products

assert products["17"]["lsl"] == 97.5
assert products["17"]["usl"] == 100.0

assert products["15"]["lsl"] == 97.0
assert products["15"]["usl"] == 101.0

# 5. Save JSON

CONFIG.mkdir(parents=True, exist_ok=True)

NEW_PATH.write_text(
    json.dumps(
        new,
        indent=2,
        ensure_ascii=False,
        allow_nan=False
    ),
    encoding="utf-8"
)

# 6. Reload and validate saved file

saved = json.loads(
    NEW_PATH.read_text(encoding="utf-8")
)

assert saved == new

print("\n=== CONFIGURATION SUMMARY ===")

print("Previous version:", old["version"])
print("New version:", saved["version"])

print("Products:", len(products))
print("Training batches:", spec["train_n"].sum())
print("Evaluation batches:", spec["test_n"].sum())
print("Detected outside batches:", len(candidates))

print("Groups:", dict(groups))

print(
    "Products with PoC specifications:",
    sum(
        p["lsl"] is not None
        for p in products.values()
    )
)

print(
    "Products without PoC specifications:",
    sum(
        p["lsl"] is None
        for p in products.values()
    )
)

print("\n=== PRODUCT SPECIFICATION COMPARISON ===")

comparison = []

for code, product in products.items():

    comparison.append({
        "code": int(code),
        "group": product["data_group"],
        "train_n": product["train_n"],
        "old_lsl": old["products"][code]["lsl"],
        "new_lsl": product["lsl"],
        "usl": product["usl"]
    })

print(
    pd.DataFrame(comparison)
    .sort_values("code")
    .to_string(index=False)
)

print("\n=== SCENARIO PRODUCT: CODE 17 ===")
print(json.dumps(products["17"], indent=2))

print("\n=== SCENARIO PRODUCT: CODE 15 ===")
print(json.dumps(products["15"], indent=2))

print("\n=== GROUP C EXAMPLE: CODE 16 ===")
print(json.dumps(products["16"], indent=2))

print("\nSaved:", NEW_PATH)
print("Original configuration preserved:", OLD_PATH)

print("\nALL JSON VALIDATIONS PASSED")
