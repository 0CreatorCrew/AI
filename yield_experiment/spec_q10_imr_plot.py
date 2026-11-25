from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SRC = Path("results/spec_all_products")
OUT = Path("results/spec_all_products_q10")

df = pd.read_csv(
    SRC / "all_product_batch_split.csv",
    parse_dates=["start_ts", "end_ts"]
)

spec = pd.read_csv(
    OUT / "all_product_specifications_q10.csv"
)

control = pd.read_csv(
    OUT / "control_limits_q10.csv"
)

candidates = pd.read_csv(
    OUT / "scenario_candidates_q10.csv"
)

SCENARIOS = [(17, 789), (15, 799)]

for code, batch in SCENARIOS:

    train = df[
        (df["code"] == code) &
        (df["phase"] == "train")
    ].sort_values(["end_ts", "batch"])

    test = df[
        (df["code"] == code) &
        (df["phase"] == "test")
    ].sort_values(["end_ts", "batch"])

    g = pd.concat(
        [train, test],
        ignore_index=True
    )

    g["sequence"] = range(1, len(g) + 1)
    g["mr"] = g["tbl_yield"].diff().abs()

    s = spec[spec["code"] == code].iloc[0]
    c = control[control["code"] == code].iloc[0]

    lsl = s["lsl_q10"]
    usl = s["usl_candidate"]

    mean = c["mean"]
    i_ucl = c["i_ucl"]
    i_lcl = c["i_lcl"]
    mr_bar = c["mr_bar"]
    mr_ucl = c["mr_ucl"]

    target = g[g["batch"] == batch].iloc[0]

    assert target["phase"] == "test"

    x = target["sequence"]
    y = target["tbl_yield"]
    mr = target["mr"]

    previous = g.iloc[int(x) - 2]

    # Create I-MR charts

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(15, 9),
        sharex=True,
        constrained_layout=True
    )

    # Individuals chart

    ax1.plot(
        g["sequence"],
        g["tbl_yield"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Batch yield"
    )

    ax1.axhline(
        mean, color="gray",
        label=f"CL = {mean:.3f}"
    )

    ax1.axhline(
        i_ucl, color="red", linestyle="--",
        label=f"UCL = {i_ucl:.3f}"
    )

    ax1.axhline(
        i_lcl, color="red", linestyle="--",
        label=f"LCL = {i_lcl:.3f}"
    )

    ax1.axhline(
        lsl, color="orange", linestyle=":",
        linewidth=2,
        label=f"LSL = {lsl:.1f}"
    )

    ax1.axhline(
        usl, color="orange", linestyle=":",
        linewidth=2,
        label=f"USL = {usl:.1f}"
    )

    ax1.scatter(
        x, y, color="red",
        s=90, zorder=5
    )

    ax1.annotate(
        f"Batch {batch}: {y:.3f}%",
        xy=(x, y),
        xytext=(15, 15),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"}
    )

    ax1.set_title(
        f"Code {code} - Individuals Chart"
    )

    ax1.set_ylabel("Yield (%)")
    ax1.legend(loc="best", ncol=3)

    # Moving range chart

    ax2.plot(
        g["sequence"],
        g["mr"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Moving range"
    )

    ax2.axhline(
        mr_bar, color="gray",
        label=f"CL = {mr_bar:.3f}"
    )

    ax2.axhline(
        mr_ucl, color="red", linestyle="--",
        label=f"UCL = {mr_ucl:.3f}"
    )

    ax2.axhline(
        0, color="gray", linestyle=":",
        label="LCL = 0"
    )

    ax2.scatter(
        x, mr, color="red",
        s=90, zorder=5
    )

    ax2.annotate(
        f"Batch {batch}: MR = {mr:.3f}",
        xy=(x, mr),
        xytext=(15, 15),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"}
    )

    ax2.set_title(
        f"Code {code} - Moving Range Chart"
    )

    ax2.set_ylabel("Moving range (%p)")
    ax2.set_xlabel("Chronological batch sequence")
    ax2.legend(loc="best")

    # Separate training and evaluation periods

    for ax in [ax1, ax2]:

        ax.axvspan(
            0.5, len(train) + 0.5,
            color="gray", alpha=0.08
        )

        ax.axvline(
            len(train) + 0.5,
            color="black",
            linestyle="--",
            linewidth=1
        )

        ax.set_xlim(0.5, len(g) + 0.5)
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"Code {code} / Batch {batch} - I-MR Analysis"
    )

    path = OUT / f"imr_code_{code}_batch_{batch}.png"

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Print interpretation data

    print(f"\n=== CODE {code} / BATCH {batch} ===")
    print("Training batches:", len(train))
    print("Evaluation batches:", len(test))
    print("Historical mean:", round(mean, 4))
    print("Actual yield:", round(y, 4))
    print("LSL:", lsl)
    print("I-chart LCL:", round(i_lcl, 4))
    print("Previous batch:", previous["batch"])
    print("Previous yield:", previous["tbl_yield"])
    print("Actual MR:", round(mr, 4))
    print("MR UCL:", round(mr_ucl, 4))
    print("Specification violation:", y < lsl)
    print("I-chart lower signal:", y < i_lcl)
    print("MR-chart signal:", mr > mr_ucl)
    print("Saved:", path)

print("\nALL CHARTS GENERATED")
