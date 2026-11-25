from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/spec_scenario")

df = pd.read_csv(
    OUT / "rca_batch789_validated_timeseries.csv"
)

df = df.sort_values("row_order").reset_index(drop=True)

df["ts"] = pd.to_datetime(df["ts"])

assert len(df) == 2292
assert df["tbl_speed"].notna().all()
assert df["ts"].is_monotonic_increasing

# 1. Operating state
df["running"] = df["tbl_speed"] > 0

first_run = df.index[df["running"]][0]
last_run = df.index[df["running"]][-1]

df["phase"] = np.select(
    [
        df.index < first_run,
        df.index > last_run,
        df["running"]
    ],
    [
        "before_first_run",
        "after_last_run",
        "running"
    ],
    default="internal_stop"
)

# 2. Counter increments
for col in ["produced", "waste"]:
    delta = df[col].diff()

    df[f"{col}_delta"] = delta

    df[f"{col}_increase"] = (
        delta.clip(lower=0).fillna(0)
    )

    df[f"{col}_reset"] = delta < 0

# 3. Operating segments
df["segment_id"] = (
    df["phase"] != df["phase"].shift()
).cumsum()

segments = df.groupby("segment_id").agg(
    phase=("phase", "first"),
    start=("ts", "first"),
    end=("ts", "last"),
    start_row=("row_order", "first"),
    end_row=("row_order", "last"),
    rows=("row_order", "size"),
    speed_mean=("tbl_speed", "mean"),
    main_comp_mean=("main_comp", "mean"),
    ejection_mean=("ejection", "mean"),
    waste_increase=("waste_increase", "sum"),
    produced_increase=("produced_increase", "sum")
)

segments["estimated_minutes"] = (
    segments["rows"] * 10 / 60
)

segments.to_csv(
    OUT / "rca_operating_segments.csv"
)

print("\n=== OPERATING PHASE SUMMARY ===")

phase_summary = df.groupby("phase").agg(
    rows=("row_order", "size"),
    speed_mean=("tbl_speed", "mean"),
    main_comp_mean=("main_comp", "mean"),
    ejection_mean=("ejection", "mean"),
    waste_increase=("waste_increase", "sum"),
    produced_increase=("produced_increase", "sum")
)

phase_summary["estimated_minutes"] = (
    phase_summary["rows"] * 10 / 60
)

print(phase_summary.round(4).to_string())

phase_summary.to_csv(
    OUT / "rca_operating_phase_summary.csv"
)

print("\n=== INTERNAL STOP SEGMENTS ===")

stops = segments[
    segments["phase"] == "internal_stop"
]

print("Internal stop count:", len(stops))

print(
    "Estimated total stop minutes:",
    round(stops["estimated_minutes"].sum(), 4)
)

print(
    "Longest estimated stop minutes:",
    round(stops["estimated_minutes"].max(), 4)
)

print(
    stops.sort_values(
        "estimated_minutes", ascending=False
    ).head(15).round(4).to_string()
)

stops.to_csv(
    OUT / "rca_internal_stop_segments.csv"
)

# 4. Speed level changes
df["speed_segment"] = (
    df["tbl_speed"] != df["tbl_speed"].shift()
).cumsum()

speed_segments = df.groupby("speed_segment").agg(
    speed=("tbl_speed", "first"),
    start=("ts", "first"),
    end=("ts", "last"),
    rows=("row_order", "size"),
    main_comp_mean=("main_comp", "mean"),
    ejection_mean=("ejection", "mean"),
    waste_increase=("waste_increase", "sum")
)

speed_segments["estimated_minutes"] = (
    speed_segments["rows"] * 10 / 60
)

speed_segments.to_csv(
    OUT / "rca_speed_segments.csv"
)

print("\n=== MAJOR SPEED SEGMENTS ===")

major = speed_segments[
    speed_segments["rows"] >= 6
]

print(
    major.round(4).to_string()
)

# 5. Waste increase events
events = df.loc[
    df["waste_increase"] > 0,
    [
        "row_order", "ts", "elapsed_min",
        "phase", "tbl_speed",
        "main_comp", "ejection",
        "waste", "waste_increase"
    ]
].copy()

total_waste = events["waste_increase"].sum()

events["share_pct"] = (
    events["waste_increase"] / total_waste * 100
)

events.to_csv(
    OUT / "rca_waste_events.csv",
    index=False
)

print("\n=== WASTE EVENT SUMMARY ===")
print("Increase events:", len(events))
print("Total positive increase:", total_waste)

top_events = events.sort_values(
    "waste_increase", ascending=False
)

print("\nTop 10 waste increase events:")
print(
    top_events.head(10)
    .round(4).to_string(index=False)
)

print(
    "\nTop 10 event share (%):",
    round(top_events.head(10)["share_pct"].sum(), 4)
)

# 6. Waste distribution by 30-minute window
df["window_start_min"] = (
    df["elapsed_min"] // 30 * 30
).astype(int)

windows = df.groupby("window_start_min").agg(
    rows=("row_order", "size"),
    running_rows=("running", "sum"),
    waste_increase=("waste_increase", "sum"),
    produced_increase=("produced_increase", "sum")
)

windows["running_pct"] = (
    windows["running_rows"] /
    windows["rows"] * 100
)

windows["waste_share_pct"] = (
    windows["waste_increase"] /
    total_waste * 100
)

windows.to_csv(
    OUT / "rca_30min_window_summary.csv"
)

print("\n=== 30-MINUTE WINDOW SUMMARY ===")
print(windows.round(4).to_string())

# 7. Visualize full process timeline
x = df["elapsed_min"]

fig, axes = plt.subplots(
    6, 1,
    figsize=(16, 16),
    sharex=True
)

plots = [
    ("tbl_speed", "Tableting Speed", "#527CA7"),
    ("main_comp", "Main Compression Force", "#527CA7"),
    ("ejection", "Ejection Force", "#527CA7"),
    ("tbl_fill", "Fill Depth", "#527CA7"),
    ("waste_increase", "Waste Counter Increase", "#C65D5D"),
    ("produced", "Produced Counter", "#689B86")
]

for ax, (col, label, color) in zip(axes, plots):

    if col == "waste_increase":
        ax.vlines(
            x, 0, df[col],
            color=color, linewidth=0.9
        )
    else:
        ax.plot(
            x, df[col],
            color=color, linewidth=0.9
        )

    ax.axvspan(
        0, 30,
        color="#739AC0", alpha=0.12
    )

    for _, stop in stops.iterrows():
        start_min = (
            stop["start"] - df["ts"].iloc[0]
        ).total_seconds() / 60

        end_min = (
            stop["end"] - df["ts"].iloc[0]
        ).total_seconds() / 60

        ax.axvspan(
            start_min, end_min + 1,
            color="#D79B65", alpha=0.15
        )

    ax.set_ylabel(label)
    ax.grid(alpha=0.2)

axes[-1].set_xlabel(
    "Elapsed Time (minutes, recorded minute resolution)"
)

fig.suptitle(
    "Code 17 / Batch 789: Full Process Timeline",
    fontsize=18
)

fig.tight_layout()

fig.savefig(
    OUT / "rca_batch789_full_timeline.png",
    dpi=200
)

plt.close(fig)

# 8. Cumulative counter visualization
fig, axes = plt.subplots(
    2, 1,
    figsize=(14, 8),
    sharex=True
)

for ax, col in zip(
    axes, ["produced", "waste"]
):
    valid = ~df[f"{col}_reset"]

    ax.plot(
        x[valid],
        df.loc[valid, col],
        linewidth=1.4,
        color="#527CA7"
    )

    ax.set_ylabel(f"{col} counter")
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Elapsed Time (minutes)")

fig.suptitle(
    "Production and Waste Counters Before Final Reset",
    fontsize=16
)

fig.tight_layout()

fig.savefig(
    OUT / "rca_batch789_counters.png",
    dpi=200
)

plt.close(fig)

# 9. Waste concentration by time window
fig, ax = plt.subplots(figsize=(13, 5))

ax.bar(
    windows.index.astype(str),
    windows["waste_increase"],
    color="#C65D5D"
)

ax.set_xlabel("30-Minute Window Start")
ax.set_ylabel("Waste Counter Increase")
ax.set_title("Waste Increase by Process Time Window")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

fig.savefig(
    OUT / "rca_batch789_waste_windows.png",
    dpi=200
)

plt.close(fig)

print("\n=== GENERATED FIGURES ===")

for p in sorted(OUT.glob("rca_batch789_*.png")):
    print(p)
