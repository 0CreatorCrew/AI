from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("results/spec_scenario")
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load source data
lab = pd.read_csv(
    "data/Laboratory.csv", sep=";"
)

process = pd.read_csv(
    "data/Process.csv", sep=";"
)

lab.columns = lab.columns.str.strip()
process.columns = process.columns.str.strip()

reference = pd.read_csv(
    OUT / "rca_reference_batches.csv"
)

target = pd.read_csv(
    OUT / "rca_target_batch.csv"
)

keys = ["code", "batch"]

assert len(reference) == 114
assert len(target) == 1

assert not lab.duplicated(keys).any()
assert not process.duplicated(keys).any()

print("\n=== DATA SHAPES ===")
print("Laboratory:", lab.shape)
print("Process:", process.shape)

# 2. Variable groups
raw_features = [
    "size",
    "api_water",
    "api_total_impurities",
    "api_l_impurity",
    "api_content",
    "api_ps01",
    "api_ps05",
    "api_ps09",
    "lactose_water",
    "lactose_sieve0045",
    "lactose_sieve015",
    "lactose_sieve025",
    "smcc_water",
    "smcc_td",
    "smcc_bd",
    "smcc_ps01",
    "smcc_ps05",
    "smcc_ps09",
    "starch_ph",
    "starch_water"
]

process_features = [
    "tbl_speed_mean",
    "tbl_speed_change",
    "fom_mean",
    "fom_change",
    "SREL_startup_mean",
    "SREL_production_mean",
    "SREL_production_max",
    "main_CompForce mean",
    "main_CompForce_sd",
    "main_CompForce_median",
    "pre_CompForce_mean",
    "tbl_fill_mean",
    "tbl_fill_sd",
    "cyl_height_mean",
    "stiffness_mean",
    "stiffness_max",
    "stiffness_min",
    "ejection_mean",
    "ejection_max",
    "ejection_min",
    "Startup_tbl_fill_maxDifference",
    "Startup_main_CompForce_mean",
    "Startup_tbl_fill_mean"
]

loss_features = [
    "tbl_speed_0_duration",
    "total_waste",
    "startup_waste"
]

quality_features = [
    "tbl_min_thickness",
    "tbl_max_thickness",
    "tbl_min_weight",
    "tbl_max_weight",
    "tbl_rsd_weight",
    "tbl_min_hardness",
    "tbl_max_hardness",
    "tbl_av_hardness",
    "tbl_max_diameter",
    "tbl_tensile",
    "dissolution_av",
    "dissolution_min",
    "batch_yield"
]


# 3. Match batches
def get_matched(source, batches):
    merged = batches[keys].merge(
        source,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True
    )

    assert merged["_merge"].eq("both").all()

    return merged.drop(columns="_merge")


ref_lab = get_matched(lab, reference)
target_lab = get_matched(lab, target)

ref_process = get_matched(process, reference)
target_process = get_matched(process, target)

print("\n=== MATCHED DATA ===")
print("Reference Laboratory:", len(ref_lab))
print("Reference Process:", len(ref_process))
print("Target Laboratory:", len(target_lab))
print("Target Process:", len(target_process))


# 4. Numeric conversion
def numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )


# 5. Feature comparison
def compare(ref, tgt, features, group):
    rows = []

    for feature in features:
        if feature not in ref.columns:
            print("Missing column:", feature)
            continue

        x = numeric(ref[feature]).dropna()
        v = numeric(tgt[feature]).iloc[0]

        if len(x) < 2 or pd.isna(v):
            print("Insufficient values:", feature)
            continue

        mean = x.mean()
        median = x.median()
        std = x.std()

        mad = (x - median).abs().median()

        z = (
            (v - mean) / std
            if std > 0 else np.nan
        )

        robust_z = (
            0.67448975 * (v - median) / mad
            if mad > 0 else np.nan
        )

        change_pct = (
            (v - median) / median * 100
            if median != 0 else np.nan
        )

        ratio = (
            v / median
            if median != 0 else np.nan
        )

        outside = (
            v < x.min() or v > x.max()
        )

        rows.append({
            "group": group,
            "feature": feature,
            "reference_n": len(x),
            "reference_missing": len(ref) - len(x),
            "reference_mean": mean,
            "reference_median": median,
            "reference_std": std,
            "reference_mad": mad,
            "reference_min": x.min(),
            "reference_q25": x.quantile(0.25),
            "reference_q75": x.quantile(0.75),
            "reference_max": x.max(),
            "target_value": v,
            "difference_from_median": v - median,
            "change_pct": change_pct,
            "ratio_to_median": ratio,
            "z_score": z,
            "robust_z_score": robust_z,
            "abs_robust_z": abs(robust_z),
            "outside_reference_range": outside
        })

    return pd.DataFrame(rows)


results = pd.concat([
    compare(
        ref_lab, target_lab,
        raw_features, "raw_material"
    ),
    compare(
        ref_process, target_process,
        process_features, "process"
    ),
    compare(
        ref_process, target_process,
        loss_features, "production_loss"
    ),
    compare(
        ref_lab, target_lab,
        quality_features, "quality_outcome"
    )
], ignore_index=True)

results = results.sort_values(
    "abs_robust_z",
    ascending=False,
    na_position="last"
)

# 6. Save comparison results
results.to_csv(
    OUT / "rca_feature_comparison.csv",
    index=False
)

display_cols = [
    "feature",
    "reference_n",
    "reference_median",
    "target_value",
    "ratio_to_median",
    "z_score",
    "robust_z_score",
    "outside_reference_range"
]

for group in [
    "raw_material",
    "process",
    "production_loss",
    "quality_outcome"
]:
    subset = results[
        results["group"] == group
    ]

    subset.to_csv(
        OUT / f"rca_{group}_comparison.csv",
        index=False
    )

    print(f"\n=== {group.upper()} ===")
    print(
        subset[display_cols]
        .round(4)
        .to_string(index=False)
    )

print("\n=== OUTSIDE REFERENCE RANGE ===")

print(
    results.loc[
        results["outside_reference_range"],
        ["group"] + display_cols
    ].round(4).to_string(index=False)
)

print("\n=== SUMMARY ===")
print("Compared features:", len(results))
print(
    "Outside reference range:",
    results["outside_reference_range"].sum()
)

print("\n=== SAVED FILES ===")

for p in sorted(OUT.glob("rca_*comparison.csv")):
    print(p)
