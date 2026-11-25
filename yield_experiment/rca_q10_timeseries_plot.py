from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/spec_all_products_q10")

SCENARIOS = [
    (17, 789, 120, 180),
    (15, 799, 0, 90)
]

for code, batch, focus_start, focus_end in SCENARIOS:

    df = pd.read_csv(
        OUT / f"waste_events_code_{code}.csv",
        parse_dates=["timestamp"]
    ).sort_values("record_no").reset_index(drop=True)

    df["cumulative_waste_increment"] = (
        df["waste_increment"].cumsum()
    )

    minute_waste = df.groupby(
        "elapsed_min"
    )["waste_increment"].sum()

    total = df["waste_increment"].sum()

    focus_waste = df.loc[
        (df["elapsed_min"] >= focus_start) &
        (df["elapsed_min"] < focus_end),
        "waste_increment"
    ].sum()

    focus_share = focus_waste / total * 100

    for mode in ["full", "focus"]:

        fig, axes = plt.subplots(
            5, 1,
            figsize=(16, 12),
            sharex=True,
            constrained_layout=True
        )

        features = [
            ("tbl_speed", "Tablet Speed"),
            ("main_comp", "Main Compression Force"),
            ("ejection", "Ejection Force"),
            (
                "cumulative_waste_increment",
                "Cumulative Waste Counter Increase"
            )
        ]

        for ax, (col, label) in zip(
            axes[:4], features
        ):

            ax.plot(
                df["elapsed_min"],
                df[col],
                linewidth=1,
                color="#416D94"
            )

            ax.set_ylabel(label)
            ax.grid(alpha=0.2)

        axes[4].bar(
            minute_waste.index,
            minute_waste.values,
            width=0.8,
            color="#BE7468"
        )

        axes[4].set_ylabel(
            "Waste Increase / Minute"
        )

        axes[4].set_xlabel(
            "Elapsed Time (minutes)"
        )

        axes[4].grid(alpha=0.2)

        for ax in axes:

            ax.axvspan(
                focus_start,
                focus_end,
                color="orange",
                alpha=0.15
            )

        if mode == "full":
            axes[-1].set_xlim(
                0, df["elapsed_min"].max()
            )
            subtitle = "Full Batch Timeline"

        else:
            axes[-1].set_xlim(
                max(0, focus_start - 5),
                focus_end + 5
            )
            subtitle = "Focus Window"

            # Scale each graph to the visible period
            visible = df[
                (df["elapsed_min"] >=
                 max(0, focus_start - 5)) &
                (df["elapsed_min"] <= focus_end + 5)
            ]

            for ax, (col, _) in zip(
                axes[:4], features
            ):
                lo = visible[col].min()
                hi = visible[col].max()
                pad = max((hi - lo) * 0.08, 0.01)

                ax.set_ylim(lo - pad, hi + pad)

            visible_waste = minute_waste[
                (minute_waste.index >=
                 max(0, focus_start - 5)) &
                (minute_waste.index <= focus_end + 5)
            ]

            axes[4].set_ylim(
                0, max(visible_waste.max() * 1.1, 1)
            )

        fig.suptitle(
            f"Code {code} / Batch {batch}\n"
            f"{subtitle} | "
            f"Focus Waste Share: {focus_share:.2f}%",
            fontsize=16
        )

        path = OUT / (
            f"rca_timeseries_code_{code}_{mode}.png"
        )

        fig.savefig(
            path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Saved:", path)

    print(f"\n=== CODE {code} ===")
    print("Total waste increments:", total)
    print("Focus waste increments:", focus_waste)
    print("Focus waste share (%):",
          round(focus_share, 4))

print("\nALL TIMESERIES CHARTS GENERATED")
