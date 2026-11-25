from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/spec_scenario")

reference = pd.read_csv(
    OUT / "rca_reference_timeseries.csv"
)

target = pd.read_csv(
    OUT / "rca_batch789_timeseries.csv"
)

features = [
    "tbl_speed",
    "main_comp",
    "pre_comp",
    "ejection",
    "tbl_fill",
    "stiffness",
    "SREL",
    "fom"
]


# 1. Calculate statistics for each batch
def summarize(g):

    g = g.copy()

    ts = pd.to_datetime(
        g["timestamp"],
        format="%d%m%Y %H:%M",
        errors="coerce"
    )

    assert ts.notna().all()
    assert ts.is_monotonic_increasing

    elapsed = (
        ts - ts.iloc[0]
    ).dt.total_seconds() / 60

    early = g[elapsed < 30]

    running = g[g["tbl_speed"] > 0]

    early_running = early[
        early["tbl_speed"] > 0
    ]

    row = {
        "batch": int(g["batch"].iloc[0]),
        "campaign": int(g["campaign"].iloc[0]),
        "total_rows": len(g),
        "early_rows": len(early),
        "early_running_rows": len(early_running),
        "running_rows": len(running),
        "early_minutes_covered":
            ts[elapsed < 30].nunique(),
        "missing_speed_rows":
            g["tbl_speed"].isna().sum(),
        "zero_speed_pct":
            (g["tbl_speed"] == 0).mean() * 100
    }

    # Internal stop statistics
    speed = g["tbl_speed"].reset_index(drop=True)

    active = np.flatnonzero(
        speed.gt(0).to_numpy()
    )

    if len(active) and speed.notna().all():

        internal = speed.iloc[
            active[0]:active[-1] + 1
        ].eq(0)

        row["internal_stop_count"] = int(
            (
                internal &
                ~internal.shift(fill_value=False)
            ).sum()
        )

        row["internal_stop_min"] = (
            internal.sum() * 10 / 60
        )

        row["internal_stop_pct"] = (
            internal.sum() / len(g) * 100
        )

    else:
        row["internal_stop_count"] = np.nan
        row["internal_stop_min"] = np.nan
        row["internal_stop_pct"] = np.nan

    # Compare equivalent operating conditions
    for col in features:

        row[f"early_{col}_mean"] = (
            early_running[col].mean()
        )

        row[f"early_{col}_std"] = (
            early_running[col].std()
        )

        row[f"full_{col}_mean"] = (
            running[col].mean()
        )

        row[f"full_{col}_std"] = (
            running[col].std()
        )

    return row


ref_stats = pd.DataFrame([
    summarize(g)
    for _, g in reference.groupby(
        "batch", sort=False
    )
])

target_stats = pd.DataFrame([
    summarize(target)
])

assert len(ref_stats) == 114

ref_stats.to_csv(
    OUT / "rca_reference_batch_timeseries_stats.csv",
    index=False
)

target_stats.to_csv(
    OUT / "rca_target_timeseries_stats.csv",
    index=False
)

print("\n=== REFERENCE COVERAGE ===")

print(
    ref_stats[
        [
            "early_rows",
            "early_running_rows",
            "early_minutes_covered",
            "missing_speed_rows"
        ]
    ].describe().round(4).to_string()
)

print("\nReference batches:", len(ref_stats))

# Require full minute coverage for early comparison
early_ref = ref_stats[
    (ref_stats["early_minutes_covered"] == 30) &
    (ref_stats["early_running_rows"] >= 2)
].copy()

print(
    "Eligible early reference batches:",
    len(early_ref)
)

# 2. Compare target against reference distribution
def compare(columns, ref_data):

    rows = []

    for col in columns:

        x = ref_data[col].dropna()
        v = target_stats[col].iloc[0]

        if len(x) < 2 or pd.isna(v):
            continue

        med = x.median()
        std = x.std()

        rows.append({
            "feature": col,
            "reference_n": len(x),
            "reference_median": med,
            "reference_min": x.min(),
            "reference_q05": x.quantile(0.05),
            "reference_q95": x.quantile(0.95),
            "reference_max": x.max(),
            "target": v,
            "ratio": v / med if med != 0 else np.nan,
            "z_score":
                (v - x.mean()) / std
                if std > 0 else np.nan,
            "outside_range":
                bool(v < x.min() or v > x.max())
        })

    return pd.DataFrame(rows)


early_result = compare(
    [
        f"early_{f}_{s}"
        for f in features
        for s in ["mean", "std"]
    ],
    early_ref
)

full_result = compare(
    [
        f"full_{f}_{s}"
        for f in features
        for s in ["mean", "std"]
    ],
    ref_stats
)

stop_result = compare(
    [
        "zero_speed_pct",
        "internal_stop_count",
        "internal_stop_min",
        "internal_stop_pct"
    ],
    ref_stats
)

display_cols = [
    "feature",
    "reference_n",
    "reference_median",
    "reference_q05",
    "reference_q95",
    "target",
    "ratio",
    "outside_range"
]

for name, result in [
    ("early", early_result),
    ("full", full_result),
    ("stop", stop_result)
]:

    print(f"\n=== {name.upper()} COMPARISON ===")

    print(
        result[display_cols]
        .round(4).to_string(index=False)
    )

    result.to_csv(
        OUT / f"rca_timeseries_{name}_comparison.csv",
        index=False
    )


# 3. Early operating condition visualization
plot_features = [
    ("early_tbl_speed_mean", "Early Mean Speed"),
    ("early_main_comp_mean", "Early Mean Main Compression"),
    ("early_main_comp_std", "Early Main Compression SD"),
    ("early_ejection_mean", "Early Mean Ejection Force"),
    ("early_ejection_std", "Early Ejection Force SD"),
    ("early_tbl_fill_mean", "Early Mean Fill Depth")
]

fig, axes = plt.subplots(
    2, 3, figsize=(15, 9)
)

for ax, (col, title) in zip(
    axes.flat, plot_features
):

    x = early_ref[col].dropna()
    v = target_stats[col].iloc[0]

    bp = ax.boxplot(
        x,
        positions=[1],
        widths=0.35,
        patch_artist=True,
        showmeans=True
    )

    bp["boxes"][0].set_facecolor("#8FA8BF")

    ax.scatter(
        1.35, v,
        color="#C65D5D",
        s=100, zorder=5,
        label=f"Batch 789: {v:.3f}"
    )

    ax.set_xticks([1, 1.35])

    ax.set_xticklabels(
        ["Reference", "Batch 789"]
    )

    ax.set_title(title)
    ax.set_xlim(0.65, 1.75)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=9)

fig.suptitle(
    "Initial 30-Minute Operating Condition Comparison",
    fontsize=17
)

fig.tight_layout()

fig.savefig(
    OUT / "rca_early30_comparison.png",
    dpi=200
)

plt.close(fig)


# 4. Internal stop comparison
stop_features = [
    ("internal_stop_count", "Internal Stop Count"),
    ("internal_stop_min", "Estimated Internal Stop Minutes"),
    ("internal_stop_pct", "Internal Stop Record Percentage")
]

fig, axes = plt.subplots(
    1, 3, figsize=(14, 5)
)

for ax, (col, title) in zip(
    axes, stop_features
):

    x = ref_stats[col].dropna()
    v = target_stats[col].iloc[0]

    bp = ax.boxplot(
        x,
        positions=[1],
        widths=0.35,
        patch_artist=True
    )

    bp["boxes"][0].set_facecolor("#8FA8BF")

    ax.scatter(
        1.35, v,
        color="#C65D5D",
        s=100, zorder=5,
        label=f"Batch 789: {v:.3f}"
    )

    ax.set_xticks([1, 1.35])

    ax.set_xticklabels(
        ["Reference", "Batch 789"]
    )

    ax.set_xlim(0.65, 1.75)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=9)

fig.tight_layout()

fig.savefig(
    OUT / "rca_internal_stop_comparison.png",
    dpi=200
)

plt.close(fig)

print("\n=== GENERATED FIGURES ===")
print(OUT / "rca_early30_comparison.png")
print(OUT / "rca_internal_stop_comparison.png")
