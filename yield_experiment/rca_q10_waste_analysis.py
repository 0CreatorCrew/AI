from pathlib import Path
import pandas as pd

OUT = Path("results/spec_all_products_q10")
SCENARIOS = [(17, 789), (15, 799)]

summary = []

for code, batch in SCENARIOS:

    df = pd.read_csv(
        OUT / f"timeseries_code_{code}_batch_{batch}.csv",
        parse_dates=["timestamp"]
    )

    df = df.sort_values("record_no").reset_index(drop=True)

    assert df["timestamp"].is_monotonic_increasing
    assert df["waste"].notna().all()
    assert df["waste"].iloc[0] == 0

    # 1. Calculate waste counter increments

    df["waste_diff"] = (
        df["waste"].diff().fillna(0)
    )

    df["waste_increment"] = (
        df["waste_diff"].clip(lower=0)
    )

    resets = df[df["waste_diff"] < 0]

    total = df["waste_increment"].sum()

    assert total > 0

    print(f"\n=== CODE {code} / BATCH {batch} ===")

    print("Maximum waste counter:", df["waste"].max())
    print("Total positive increments:", total)
    print("Increase records:",
          (df["waste_increment"] > 0).sum())
    print("Counter decrease records:", len(resets))

    print("\n--- COUNTER DECREASE RECORDS ---")

    print(
        resets[[
            "timestamp", "record_no",
            "waste", "waste_diff"
        ]].to_string(index=False)
    )

    # 2. Aggregate by 30-minute windows

    df["window_start_min"] = (
        df["elapsed_min"] // 30 * 30
    )

    windows = df.groupby(
        "window_start_min"
    ).agg(
        waste_increment=("waste_increment", "sum"),
        records=("record_no", "size")
    ).reset_index()

    start = df["timestamp"].min()

    windows["start_time"] = (
        start + pd.to_timedelta(
            windows["window_start_min"], unit="m"
        )
    )

    windows["end_time"] = (
        windows["start_time"] +
        pd.Timedelta(minutes=30)
    )

    windows["share_pct"] = (
        windows["waste_increment"] / total * 100
    )

    assert abs(
        windows["waste_increment"].sum() - total
    ) < 0.0001

    top = windows.sort_values(
        "waste_increment", ascending=False
    ).head(10)

    print("\n--- TOP 10 WASTE WINDOWS ---")

    print(
        top[[
            "start_time", "end_time",
            "waste_increment", "share_pct"
        ]].round(4).to_string(index=False)
    )

    # 3. Compare early production periods

    first_running = df.loc[
        df["tbl_speed"] > 0, "timestamp"
    ].min()

    assert pd.notna(first_running)

    early90 = df.loc[
        df["elapsed_min"] < 90,
        "waste_increment"
    ].sum()

    before_running = df.loc[
        df["timestamp"] < first_running,
        "waste_increment"
    ].sum()

    running90 = df.loc[
        (df["timestamp"] >= first_running) &
        (df["timestamp"] <
         first_running + pd.Timedelta(minutes=90)),
        "waste_increment"
    ].sum()

    print("\n--- EARLY PERIOD ANALYSIS ---")
    print("First running timestamp:", first_running)
    print("Waste increments before first running minute:",
          before_running)
    print("Waste increments in first 90 min:", early90)
    print("First 90 min share (%):",
          round(early90 / total * 100, 4))
    print("Waste increments in 90 min from first running:",
          running90)
    print("Share (%):",
          round(running90 / total * 100, 4))

    # 4. Compare operating states

    running_waste = df.loc[
        df["tbl_speed"] > 0,
        "waste_increment"
    ].sum()

    stopped_waste = df.loc[
        df["tbl_speed"] == 0,
        "waste_increment"
    ].sum()

    print("\n--- OPERATING STATE COMPARISON ---")
    print("Running waste increments:", running_waste)
    print("Running share (%):",
          round(running_waste / total * 100, 4))
    print("Zero-speed waste increments:", stopped_waste)
    print("Zero-speed share (%):",
          round(stopped_waste / total * 100, 4))

    # 5. Save results

    df.to_csv(
        OUT / f"waste_events_code_{code}.csv",
        index=False
    )

    windows.to_csv(
        OUT / f"waste_windows_code_{code}.csv",
        index=False
    )

    summary.append({
        "code": code,
        "batch": batch,
        "total_positive_increments": total,
        "counter_decrease_records": len(resets),
        "first_90min_waste": early90,
        "first_90min_share_pct": early90 / total * 100,
        "first_running_90min_waste": running90,
        "running_waste": running_waste,
        "zero_speed_waste": stopped_waste
    })

pd.DataFrame(summary).to_csv(
    OUT / "waste_analysis_summary.csv",
    index=False
)

print("\nALL WASTE ANALYSES COMPLETED")
