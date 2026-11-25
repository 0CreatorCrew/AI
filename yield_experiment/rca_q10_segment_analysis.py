from pathlib import Path
import pandas as pd

OUT = Path("results/spec_all_products_q10")

SCENARIOS = [
    (17, 789, 120, 180),
    (15, 799, 0, 90)
]

for code, batch, start_min, end_min in SCENARIOS:

    df = pd.read_csv(
        OUT / f"waste_events_code_{code}.csv",
        parse_dates=["timestamp"]
    ).sort_values("record_no").reset_index(drop=True)

    total_waste = df["waste_increment"].sum()

    df["running"] = df["tbl_speed"] > 0

    df["focus"] = (
        (df["elapsed_min"] >= start_min) &
        (df["elapsed_min"] < end_min)
    )

    print(f"\n=== CODE {code} / BATCH {batch} ===")

    # 1. Compare focus window with remaining period

    rows = []

    for label, group in [
        ("Focus window", df[df["focus"]]),
        ("Other periods", df[~df["focus"]])
    ]:

        running = group[group["running"]]
        waste = group["waste_increment"].sum()

        rows.append({
            "period": label,
            "records": len(group),
            "running_records": len(running),
            "zero_speed_pct":
                (group["tbl_speed"] == 0).mean() * 100,
            "speed_mean": running["tbl_speed"].mean(),
            "speed_std": running["tbl_speed"].std(),
            "main_comp_mean": running["main_comp"].mean(),
            "main_comp_std": running["main_comp"].std(),
            "ejection_mean": running["ejection"].mean(),
            "ejection_std": running["ejection"].std(),
            "waste_increment": waste,
            "waste_share_pct": waste / total_waste * 100
        })

    comparison = pd.DataFrame(rows)

    print("\n--- FOCUS WINDOW COMPARISON ---")
    print(comparison.T.round(4).to_string())

    # 2. Identify continuous running and stop segments

    df["segment_id"] = (
        df["running"] != df["running"].shift()
    ).cumsum()

    segments = df.groupby("segment_id").agg(
        running=("running", "first"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        records=("record_no", "size"),
        speed_mean=("tbl_speed", "mean"),
        speed_median=("tbl_speed", "median"),
        speed_std=("tbl_speed", "std"),
        main_comp_mean=("main_comp", "mean"),
        main_comp_std=("main_comp", "std"),
        ejection_mean=("ejection", "mean"),
        waste_increment=("waste_increment", "sum")
    ).reset_index()

    segments["observed_span_min"] = (
        segments["end_time"] - segments["start_time"]
    ).dt.total_seconds() / 60

    segments["waste_share_pct"] = (
        segments["waste_increment"] / total_waste * 100
    )

    assert abs(
        segments["waste_increment"].sum() - total_waste
    ) < 0.0001

    running_segments = segments[segments["running"]]

    print("\n--- RUNNING SEGMENTS WITH MOST WASTE ---")

    cols = [
        "segment_id", "start_time", "end_time",
        "observed_span_min", "speed_median",
        "main_comp_mean", "main_comp_std",
        "waste_increment", "waste_share_pct"
    ]

    print(
        running_segments.sort_values(
            "waste_increment", ascending=False
        )[cols].head(10).round(4).to_string(index=False)
    )

    # 3. Identify internal stop periods

    first_run = running_segments["segment_id"].min()
    last_run = running_segments["segment_id"].max()

    stops = segments[
        (~segments["running"]) &
        (segments["segment_id"] > first_run) &
        (segments["segment_id"] < last_run)
    ]

    print("\n--- LONGEST INTERNAL STOPS ---")
    print("Internal stop segments:", len(stops))

    print(
        stops.sort_values(
            "observed_span_min", ascending=False
        )[[
            "segment_id", "start_time", "end_time",
            "records", "observed_span_min",
            "waste_increment"
        ]].head(10).to_string(index=False)
    )

    # 4. Waste increments near running starts

    running_rows = df[df["running"]].copy()

    running_rows["segment_start"] = (
        running_rows.groupby("segment_id")["timestamp"]
        .transform("min")
    )

    running_rows["minutes_from_start"] = (
        running_rows["timestamp"]
        - running_rows["segment_start"]
    ).dt.total_seconds() / 60

    early = running_rows[
        running_rows["minutes_from_start"] < 5
    ]

    early_waste = early["waste_increment"].sum()

    print("\n--- START AND RESTART ANALYSIS ---")
    print("Running segments:", len(running_segments))
    print("First 5-min running records:", len(early))
    print("Total running records:", len(running_rows))

    print(
        "First 5-min running record share (%):",
        round(len(early) / len(running_rows) * 100, 4)
    )

    print("Waste increments in first 5 min:", early_waste)

    print(
        "Share of running waste (%):",
        round(
            early_waste
            / running_rows["waste_increment"].sum()
            * 100, 4
        )
    )

    # 5. Save results

    comparison.to_csv(
        OUT / f"focus_comparison_code_{code}.csv",
        index=False
    )

    segments.to_csv(
        OUT / f"operating_segments_code_{code}.csv",
        index=False
    )

print("\nALL SEGMENT ANALYSES COMPLETED")
