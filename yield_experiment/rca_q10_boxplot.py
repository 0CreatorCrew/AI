from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/spec_all_products_q10")

SCENARIOS = [
    {
        "code": 17,
        "batch": 789,
        "features": {
            "main_CompForce mean": "Mean Main Compression Force",
            "tbl_speed_mean": "Mean Tablet Speed",
            "ejection_mean": "Mean Ejection Force",
            "main_CompForce_sd": "Main Compression Force Std",
            "tbl_speed_0_duration": "Zero-speed Duration Metric",
            "total_waste": "Total Waste Metric"
        }
    },
    {
        "code": 15,
        "batch": 799,
        "features": {
            "tbl_speed_change": "Tablet Speed Change Metric",
            "main_CompForce_sd": "Main Compression Force Std",
            "tbl_speed_0_duration": "Zero-speed Duration Metric",
            "tbl_speed_mean": "Mean Tablet Speed",
            "startup_waste": "Startup Waste",
            "total_waste": "Total Waste Metric"
        }
    }
]

for scenario in SCENARIOS:

    code = scenario["code"]
    batch = scenario["batch"]

    ref = pd.read_csv(
        OUT / f"reference_code_{code}.csv"
    )

    target = pd.read_csv(
        OUT / f"target_code_{code}_batch_{batch}.csv"
    )

    stats = pd.read_csv(
        OUT / f"rca_comparison_code_{code}.csv"
    ).set_index("feature")

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 10),
        constrained_layout=True
    )

    for ax, (col, title) in zip(
        axes.flat,
        scenario["features"].items()
    ):

        values = ref[col].dropna()
        value = target[col].iloc[0]
        median = values.median()

        z = stats.loc[col, "z_score"]
        ratio = stats.loc[col, "ratio"]

        ax.boxplot(
            values,
            positions=[1],
            widths=0.45,
            patch_artist=True,
            boxprops={
                "facecolor": "#AEC9E0",
                "edgecolor": "#416D94"
            },
            medianprops={
                "color": "#174A75",
                "linewidth": 2
            },
            flierprops={
                "markersize": 3,
                "alpha": 0.5
            }
        )

        ax.scatter(
            1.25,
            value,
            color="red",
            s=75,
            zorder=5,
            label=f"Batch {batch}"
        )

        ax.axhline(
            median,
            color="gray",
            linestyle=":",
            linewidth=1
        )

        ax.set_title(title, fontsize=11)

        ax.set_xticks([1, 1.25])
        ax.set_xticklabels(
            ["Reference", "Target"]
        )

        ax.set_xlim(0.6, 1.65)

        ax.text(
            0.03, 0.97,
            f"Median: {median:.3f}\n"
            f"Target: {value:.3f}\n"
            f"Ratio: {ratio:.2f}x\n"
            f"Z-score: {z:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "lightgray"
            }
        )

        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Code {code} / Batch {batch}\n"
        f"Process comparison with "
        f"{len(ref)} historical reference batches",
        fontsize=16
    )

    path = OUT / f"rca_boxplot_code_{code}.png"

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\n=== CODE {code} ===")
    print("Reference batches:", len(ref))
    print("Target batch:", batch)
    print("Variables plotted:",
          len(scenario["features"]))
    print("Saved:", path)

print("\nALL BOXPLOTS GENERATED")
