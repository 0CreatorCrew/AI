from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

BASE = Path("results")
FOLLOWUP = BASE / "followup"
OUT = BASE / "spec_scenario"

OUT.mkdir(parents=True, exist_ok=True)

# 1. Load existing results
local = pd.read_csv(
    FOLLOWUP / "shap_local.csv"
)

global_shap = pd.read_csv(
    FOLLOWUP / "shap_global.csv"
)

predictions = pd.read_csv(
    BASE / "predictions.csv"
)

uncertainty = pd.read_csv(
    FOLLOWUP / "uncertainty_review.csv"
)

with open(
    FOLLOWUP / "models" / "metadata.json",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

# 2. Extract Batch 789
g = local[
    (local["batch"] == 789) &
    (local["code"] == 17)
].copy()

assert len(g) == 51
assert not g["feature"].duplicated().any()
assert set(g["feature"]) == set(metadata["features"])
assert g["prediction"].nunique() == 1
assert g["baseline"].nunique() == 1
assert g["actual"].nunique() == 1

g = g.set_index("feature").loc[
    metadata["features"]
].reset_index()

actual = float(g["actual"].iloc[0])
prediction = float(g["prediction"].iloc[0])
baseline = float(g["baseline"].iloc[0])

# 3. SHAP additivity check
shap_sum = g["shap"].sum()
reconstructed = baseline + shap_sum

additivity_error = abs(
    reconstructed - prediction
)

print("\n=== SHAP ADDITIVITY CHECK ===")
print("Actual yield:", actual)
print("Baseline:", baseline)
print("SHAP sum:", shap_sum)
print("Reconstructed prediction:", reconstructed)
print("Saved prediction:", prediction)
print("Additivity error:", additivity_error)

assert np.isclose(
    reconstructed,
    prediction,
    atol=1e-5,
    rtol=0
)

# 4. Rank local contributions
g["abs_shap"] = g["shap"].abs()

g["direction"] = np.select(
    [g["shap"] < 0, g["shap"] > 0],
    ["decrease", "increase"],
    default="neutral"
)

ranked = g.sort_values(
    "abs_shap",
    ascending=False
)

ranked.to_csv(
    OUT / "rca_batch789_shap_ranked.csv",
    index=False
)

cols = [
    "feature", "input_value",
    "shap", "abs_shap", "direction"
]

print("\n=== TOP 15 LOCAL SHAP FEATURES ===")
print(
    ranked[cols].head(15)
    .round(6).to_string(index=False)
)

print("\n=== TOP 10 NEGATIVE CONTRIBUTIONS ===")
print(
    g[g["shap"] < 0]
    .sort_values("shap")[cols]
    .head(10).round(6)
    .to_string(index=False)
)

print("\n=== TOP 10 POSITIVE CONTRIBUTIONS ===")
print(
    g[g["shap"] > 0]
    .sort_values("shap", ascending=False)[cols]
    .head(10).round(6)
    .to_string(index=False)
)

print("\n=== ALL EARLY PROCESS FEATURES ===")
print(
    ranked[
        ranked["feature"].str.startswith("early_")
    ][cols].round(6).to_string(index=False)
)

# 5. Feature-group contributions
def feature_group(name):

    if name.startswith("code_"):
        return "product_code"

    if name.startswith("lag"):
        return "historical_yield"

    if name.startswith("early_"):
        return "early_process"

    return "raw_material_and_batch"


g["group"] = g["feature"].map(feature_group)

groups = g.groupby("group").agg(
    features=("feature", "size"),
    net_shap=("shap", "sum"),
    sum_abs_shap=("abs_shap", "sum")
)

print("\n=== SHAP CONTRIBUTION BY GROUP ===")
print(groups.round(6).to_string())

groups.to_csv(
    OUT / "rca_batch789_shap_groups.csv"
)

# 6. SHAP waterfall plot
explanation = shap.Explanation(
    values=g["shap"].to_numpy(dtype=float),
    base_values=baseline,
    data=g["input_value"].to_numpy(dtype=float),
    feature_names=g["feature"].tolist()
)

shap.plots.waterfall(
    explanation,
    max_display=15,
    show=False
)

fig = plt.gcf()

fig.suptitle(
    "Code 17 / Batch 789: Local SHAP Explanation",
    fontsize=14
)

fig.savefig(
    OUT / "rca_batch789_shap_waterfall.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

# 7. Local SHAP bar chart
top = ranked.head(15).sort_values("abs_shap")

colors = [
    "#527CA7" if v < 0 else "#C65D5D"
    for v in top["shap"]
]

fig, ax = plt.subplots(figsize=(11, 8))

bars = ax.barh(
    top["feature"],
    top["shap"],
    color=colors
)

ax.axvline(0, color="gray", linewidth=1)

ax.bar_label(
    bars,
    fmt="%+.4f",
    padding=4,
    fontsize=9
)

ax.margins(x=0.2)

ax.set_xlabel(
    "SHAP Contribution (percentage points)"
)

ax.set_title(
    "Batch 789: Top 15 Local SHAP Contributions"
)

ax.grid(axis="x", alpha=0.2)

fig.tight_layout()

fig.savefig(
    OUT / "rca_batch789_shap_local_bar.png",
    dpi=200
)

plt.close(fig)

# 8. Global SHAP importance
top_global = global_shap.nlargest(
    15, "mean_abs_shap"
).sort_values("mean_abs_shap")

fig, ax = plt.subplots(figsize=(11, 8))

bars = ax.barh(
    top_global["feature"],
    top_global["mean_abs_shap"],
    color="#527CA7"
)

ax.bar_label(
    bars,
    fmt="%.4f",
    padding=4,
    fontsize=9
)

ax.margins(x=0.15)
ax.set_xlabel("Mean Absolute SHAP Value")
ax.set_title("Random Forest: Global SHAP Importance")
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()

fig.savefig(
    OUT / "rca_shap_global_bar.png",
    dpi=200
)

plt.close(fig)

# 9. Model prediction comparison
p = predictions[
    (predictions["batch"] == 789) &
    (predictions["code"] == 17)
].copy()

assert len(p) == 6
assert np.allclose(p["actual"], actual)

p["absolute_error"] = (
    p["predicted"] - p["actual"]
).abs()

p["predicted_spec_outside"] = (
    (p["predicted"] < 97.0) |
    (p["predicted"] > 100.0)
)

p.to_csv(
    OUT / "rca_batch789_model_comparison.csv",
    index=False
)

print("\n=== MODEL PREDICTION COMPARISON ===")

print(
    p[
        [
            "mode", "model", "actual",
            "predicted", "absolute_error",
            "predicted_spec_outside"
        ]
    ].round(6).to_string(index=False)
)

names = [
    f"{mode}\n{model}"
    for mode, model in zip(
        p["mode"], p["model"]
    )
]

fig, ax = plt.subplots(figsize=(13, 6))

ax.scatter(
    range(len(p)),
    p["predicted"],
    color="#527CA7",
    s=90,
    zorder=4,
    label="Predicted Yield"
)

for i, value in enumerate(p["predicted"]):
    ax.annotate(
        f"{value:.3f}%",
        (i, value),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

ax.axhline(
    actual,
    color="#C65D5D",
    linewidth=2,
    label=f"Actual: {actual:.3f}%"
)

ax.axhline(
    97.0,
    color="#D29A52",
    linestyle="--",
    label="Hypothetical LSL: 97%"
)

ax.axhline(
    100.0,
    color="#D29A52",
    linestyle="--",
    label="Hypothetical USL: 100%"
)

ax.set_ylim(85, 102)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Tableting Yield (%)")
ax.set_title("Batch 789: Actual vs Predicted Yield")
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="lower right")

fig.tight_layout()

fig.savefig(
    OUT / "rca_batch789_prediction_comparison.png",
    dpi=200
)

plt.close(fig)

# 10. Separate uncertainty model results
u = uncertainty[
    (uncertainty["batch"] == 789) &
    (uncertainty["code"] == 17)
].copy()

assert len(u) == 1

print("\n=== SEPARATE UNCERTAINTY MODEL ===")
print(u.round(6).to_string(index=False))

u.to_csv(
    OUT / "rca_batch789_uncertainty.csv",
    index=False
)

# 11. Saved figures
print("\n=== GENERATED FIGURES ===")

for name in [
    "rca_batch789_shap_waterfall.png",
    "rca_batch789_shap_local_bar.png",
    "rca_shap_global_bar.png",
    "rca_batch789_prediction_comparison.png"
]:
    print(OUT / name)

print("\n=== ANALYSIS COMPLETE ===")
