from pathlib import Path
import json
import pandas as pd

OUT = Path("results/spec_all_products")

spec = pd.read_csv(
    OUT / "all_product_specifications.csv"
)

data = pd.read_csv(
    OUT / "all_product_batch_split.csv"
)

assert len(spec) == 25
assert set(spec["code"]) == set(range(1, 26))
assert spec["train_n"].sum() == 772
assert spec["test_n"].sum() == 232

# 1. Product specification configuration
products = {}

for _, r in spec.iterrows():

    code = str(int(r["code"]))

    usable = (
        pd.notna(r["poc_lsl"])
        and pd.notna(r["poc_usl"])
    )

    products[code] = {
        "train_n": int(r["train_n"]),
        "test_n": int(r["test_n"]),
        "q025": float(r["q025"]),
        "candidate_lsl": float(r["lsl_b"]),
        "candidate_usl": float(r["usl_candidate"]),
        "lsl": float(r["poc_lsl"]) if usable else None,
        "usl": float(r["poc_usl"]) if usable else None,
        "data_group": r["data_group"],
        "status": r["status"],
        "evaluation_data_available": bool(
            r["test_n"] > 0
        ),
        "production_use_allowed": False,
        "automatic_alert_allowed": False
    }

    if usable:
        assert r["poc_lsl"] < r["poc_usl"]

config = {
    "version": "0.1.0",
    "status": "research_poc_only",
    "target": "tbl_yield",
    "unit": "percent",
    "reference_cutoff": "2020-10-19 02:16:00",
    "approved_manufacturing_spec": False,
    "method": {
        "lsl": "2.5 percentile rounded to nearest 0.5",
        "usl": "training maximum rounded up to 0.5",
        "group_a": "train_n >= 40",
        "group_b": "10 <= train_n < 40",
        "group_c": "train_n < 10"
    },
    "products": products
}

path = Path(
    "config/yield_specifications_all_products.json"
)

path.parent.mkdir(parents=True, exist_ok=True)

path.write_text(
    json.dumps(
        config,
        indent=2,
        ensure_ascii=False,
        allow_nan=False
    ),
    encoding="utf-8"
)

# 2. Reload and validate JSON
loaded = json.loads(
    path.read_text(encoding="utf-8")
)

assert len(loaded["products"]) == 25
assert loaded == config

for code, p in loaded["products"].items():

    if p["data_group"] == "C":
        assert p["lsl"] is None
        assert p["usl"] is None

    assert not p["production_use_allowed"]
    assert not p["automatic_alert_allowed"]

print("\n=== CONFIGURATION VALIDATION ===")
print("Total products:", len(products))
print("Group A:", sum(
    p["data_group"] == "A"
    for p in products.values()
))
print("Group B:", sum(
    p["data_group"] == "B"
    for p in products.values()
))
print("Group C:", sum(
    p["data_group"] == "C"
    for p in products.values()
))

# 3. Evaluation data
test = data[data["phase"] == "test"].copy()

test = test.merge(
    spec[
        ["code", "poc_lsl", "poc_usl", "data_group"]
    ],
    on="code",
    how="left",
    validate="many_to_one"
)

assert len(test) == 232
assert test["poc_lsl"].notna().all()
assert test["poc_usl"].notna().all()

test["below_lsl"] = (
    test["tbl_yield"] < test["poc_lsl"]
)

test["above_usl"] = (
    test["tbl_yield"] > test["poc_usl"]
)

test["spec_outside"] = (
    test["below_lsl"] | test["above_usl"]
)

# 4. Product-level evaluation summary
summary = test.groupby(
    "code",
    as_index=False
).agg(
    test_n=("batch", "size"),
    below_lsl=("below_lsl", "sum"),
    above_usl=("above_usl", "sum"),
    outside=("spec_outside", "sum")
)

summary["within"] = (
    summary["test_n"] - summary["outside"]
)

summary["outside_rate_pct"] = (
    summary["outside"] / summary["test_n"] * 100
)

print("\n=== EVALUATION SUMMARY ===")
print(summary.round(4).to_string(index=False))

outside = test[test["spec_outside"]].copy()

print("\n=== SPEC OUTSIDE BATCHES ===")
print(
    outside[
        [
            "code", "batch", "tbl_yield",
            "poc_lsl", "poc_usl",
            "below_lsl", "above_usl"
        ]
    ].to_string(index=False)
)

# 5. Verify previous three-product results
previous = test[
    test["code"].isin([13, 17, 23])
]

assert len(previous) == 209

previous_outside = previous[
    previous["spec_outside"]
]

observed = set(
    zip(
        previous_outside["code"],
        previous_outside["batch"]
    )
)

expected = {
    (13, 851),
    (17, 789),
    (23, 903)
}

assert observed == expected
assert previous_outside["below_lsl"].all()

print("\n=== PREVIOUS RESULT CHECK ===")
print("Previous evaluation batches:", len(previous))
print("Previous outside batches:", len(previous_outside))
print("Previous result match:", observed == expected)

# 6. Final summary
print("\n=== OVERALL SUMMARY ===")
print("Evaluation products:", test["code"].nunique())
print("Evaluation batches:", len(test))
print("Below LSL:", test["below_lsl"].sum())
print("Above USL:", test["above_usl"].sum())
print("Spec outside:", len(outside))
print("Within spec:", len(test) - len(outside))
print("Outside rate:", round(
    len(outside) / len(test) * 100, 4
))
print("Products without evaluation data:", sum(
    not p["evaluation_data_available"]
    for p in products.values()
))

# 7. Save results
test.to_csv(
    OUT / "all_product_spec_detection.csv",
    index=False
)

summary.to_csv(
    OUT / "all_product_evaluation_summary.csv",
    index=False
)

outside.to_csv(
    OUT / "all_product_outside_batches.csv",
    index=False
)

print("\n=== SAVED FILES ===")
print(path)
print(OUT / "all_product_spec_detection.csv")
print(OUT / "all_product_evaluation_summary.csv")
print(OUT / "all_product_outside_batches.csv")

print("\nALL VALIDATION CHECKS PASSED")
