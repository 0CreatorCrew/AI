from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

OUT = Path("results/spec_all_products_q10")

SCENARIOS = [(17, 789), (15, 799)]


def analyze_batch(g):

    g = g.sort_values(
        ["timestamp", "original_order"]
    ).reset_index(drop=True).copy()

    running = g["tbl_speed"] > 0

    g["running"] = running

    g["segment_id"] = (
        running != running.shift()
    ).cumsum()

    g["waste_diff"] = (
        g["waste"].diff().fillna(0)
    )

    g["waste_increment"] = (
        g["waste_diff"].clip(lower=0)
    )

    segments = g.groupby("segment_id").agg(
        running=("running", "first"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        records=("timestamp", "size")
    ).reset_index()

    segments["span_min"] = (
        segments["end"] - segments["start"]
    ).dt.total_seconds() / 60

    runs = segments[segments["running"]]

    if runs.empty:
        return None

    stops = segments[
        (~segments["running"]) &
        (segments["segment_id"] >
         runs["segment_id"].min()) &
        (segments["segment_id"] <
         runs["segment_id"].max())
    ]

    r = g[g["running"]].copy()

    r["segment_start"] = (
        r.groupby("segment_id")["timestamp"]
        .transform("min")
    )

    r["minutes_from_start"] = (
        r["timestamp"] - r["segment_start"]
    ).dt.total_seconds() / 60

    early = r[r["minutes_from_start"] < 5]

    running_waste = r["waste_increment"].sum()
    early_waste = early["waste_increment"].sum()
    total_waste = g["waste_increment"].sum()

    # Check counter reliability
    decreases = g[g["waste_diff"] < 0]

    terminal_reset = (
        len(decreases) == 1
        and decreases.index[0] == len(g) - 1
        and decreases["waste"].iloc[0] == 0
    )

    counter_valid = (
        g["waste"].iloc[0] == 0
        and (len(decreases) == 0 or terminal_reset)
    )

    elapsed = (
        g["timestamp"] - g["timestamp"].min()
    ).dt.total_seconds() / 60

    first90_waste = g.loc[
        elapsed < 90, "waste_increment"
    ].sum()

    return {
        "records": len(g),
        "running_segments": len(runs),
        "internal_stop_count": len(stops),
        "longest_stop_span_min":
            stops["span_min"].max()
            if len(stops) else 0,
        "zero_speed_record_pct":
            (g["tbl_speed"] == 0).mean() * 100,
        "early5_running_record_pct":
            len(early) / len(r) * 100,
        "total_waste_increment":
            total_waste if counter_valid else np.nan,
        "first90_waste_share_pct":
            first90_waste / total_waste * 100
            if counter_valid and total_waste > 0
            else np.nan,
        "early5_running_waste_share_pct":
            early_waste / running_waste * 100
            if counter_valid and running_waste > 0
            else np.nan,
        "counter_decreases": len(decreases),
        "counter_valid": counter_valid
    }


METRICS = [
    "running_segments",
    "internal_stop_count",
    "longest_stop_span_min",
    "zero_speed_record_pct",
    "early5_running_record_pct",
    "total_waste_increment",
    "first90_waste_share_pct",
    "early5_running_waste_share_pct"
]

all_comparisons = []

with ZipFile("data/Process.zip") as z:

    for code, batch in SCENARIOS:

        reference = pd.read_csv(
            OUT / f"reference_code_{code}.csv"
        )

        ref_ids = set(reference["batch"])

        with z.open(f"Process/{code}.csv") as f:
            raw = pd.read_csv(
                f,
                sep=";",
                usecols=[
                    "timestamp", "code", "batch",
                    "tbl_speed", "waste"
                ]
            )

        raw = raw[
            (raw["code"] == code) &
            (raw["batch"].isin(ref_ids | {batch}))
        ].copy()

        raw["original_order"] = raw.index

        raw["timestamp"] = pd.to_datetime(
            raw["timestamp"].astype(str).str.strip(),
            format="%d%m%Y %H:%M",
            errors="raise"
        )

        rows = []

        for batch_id, g in raw.groupby("batch"):

            if g[["tbl_speed", "waste"]].isna().any().any():
                print("Missing data:", code, batch_id)
                continue

            result = analyze_batch(g)

            if result is None:
                continue

            result["code"] = code
            result["batch"] = batch_id
            result["group"] = (
                "target" if batch_id == batch
                else "reference"
            )

            rows.append(result)

        result = pd.DataFrame(rows)

        ref = result[
            result["group"] == "reference"
        ]

        target = result[
            result["group"] == "target"
        ]

        assert len(target) == 1

        print(f"\n=== CODE {code} / BATCH {batch} ===")
        print("Expected reference batches:", len(ref_ids))
        print("Analyzed reference batches:", len(ref))
        print("Valid reference waste counters:",
              ref["counter_valid"].sum())

        print("\n--- TARGET STATISTICS ---")
        print(target.T.to_string())

        comparison = []

        for metric in METRICS:

            values = ref[metric].dropna()
            value = target[metric].iloc[0]

            if values.empty or pd.isna(value):
                continue

            median = values.median()

            comparison.append({
                "code": code,
                "batch": batch,
                "metric": metric,
                "reference_n": len(values),
                "reference_median": median,
                "reference_q95": values.quantile(0.95),
                "reference_max": values.max(),
                "target": value,
                "ratio":
                    value / median
                    if median != 0 else np.nan,
                "above_q95":
                    value > values.quantile(0.95),
                "above_max":
                    value > values.max()
            })

        comparison = pd.DataFrame(comparison)

        print("\n--- HISTORICAL COMPARISON ---")
        print(comparison.round(4).to_string(index=False))

        result.to_csv(
            OUT / f"reference_timeseries_code_{code}.csv",
            index=False
        )

        comparison.to_csv(
            OUT / f"reference_ts_comparison_code_{code}.csv",
            index=False
        )

        all_comparisons.append(comparison)

pd.concat(
    all_comparisons,
    ignore_index=True
).to_csv(
    OUT / "reference_ts_comparison_all.csv",
    index=False
)

print("\nALL REFERENCE TIMESERIES ANALYSES COMPLETED")
