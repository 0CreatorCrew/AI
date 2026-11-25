from pathlib import Path
from zipfile import ZipFile
import pandas as pd

OUT = Path("results/spec_all_products_q10")
SCENARIOS = [(17, 789), (15, 799)]

FEATURES = [
    "tbl_speed",
    "main_comp",
    "pre_comp",
    "tbl_fill",
    "ejection",
    "produced",
    "waste"
]

summary = []

with ZipFile("data/Process.zip") as z:

    for code, batch in SCENARIOS:

        path = f"Process/{code}.csv"

        with z.open(path) as f:
            raw = pd.read_csv(f, sep=";")

        target = raw[
            (raw["code"] == code) &
            (raw["batch"] == batch)
        ].copy()

        assert len(target) > 0

        # Preserve original record order
        target["original_order"] = target.index

        # Convert timestamp
        target["timestamp"] = pd.to_datetime(
            target["timestamp"].astype(str).str.strip(),
            format="%d%m%Y %H:%M",
            errors="raise"
        )

        target = target.sort_values(
            ["timestamp", "original_order"]
        ).reset_index(drop=True)

        target["record_no"] = range(
            1, len(target) + 1
        )

        start = target["timestamp"].min()
        end = target["timestamp"].max()

        target["elapsed_min"] = (
            target["timestamp"] - start
        ).dt.total_seconds() / 60

        duration = (
            end - start
        ).total_seconds() / 60

        # Separate operating and zero-speed records
        target["running"] = target["tbl_speed"] > 0

        running = target[target["running"]]
        zero_speed = target[target["tbl_speed"] == 0]

        print(f"\n=== CODE {code} / BATCH {batch} ===")
        print("Raw product records:", len(raw))
        print("Target records:", len(target))
        print("Campaign:", target["campaign"].unique())
        print("Start:", start)
        print("End:", end)
        print("Duration (min):", duration)

        print("\n--- TIMESTAMP QUALITY ---")
        print(
            "Unique timestamps:",
            target["timestamp"].nunique()
        )
        print(
            "Repeated timestamp records:",
            target["timestamp"].duplicated().sum()
        )

        print("\n--- MISSING VALUES ---")
        print(target[FEATURES].isna().sum())

        print("\n--- OPERATING STATUS ---")
        print("Running records:", len(running))
        print("Zero-speed records:", len(zero_speed))
        print(
            "Zero-speed record ratio (%):",
            round(len(zero_speed) / len(target) * 100, 4)
        )

        print("\n--- FULL BATCH STATISTICS ---")
        print(
            target[FEATURES]
            .describe()
            .round(4)
            .to_string()
        )

        print("\n--- RUNNING PERIOD STATISTICS ---")
        print(
            running[FEATURES]
            .describe()
            .round(4)
            .to_string()
        )

        print("\n--- FIRST 5 RECORDS ---")
        print(
            target[["timestamp"] + FEATURES]
            .head()
            .to_string(index=False)
        )

        print("\n--- LAST 5 RECORDS ---")
        print(
            target[["timestamp"] + FEATURES]
            .tail()
            .to_string(index=False)
        )

        save_path = (
            OUT / f"timeseries_code_{code}_batch_{batch}.csv"
        )

        target.to_csv(save_path, index=False)

        summary.append({
            "code": code,
            "batch": batch,
            "records": len(target),
            "start": start,
            "end": end,
            "duration_min": duration,
            "unique_timestamps": target["timestamp"].nunique(),
            "running_records": len(running),
            "zero_speed_records": len(zero_speed),
            "zero_speed_record_ratio_pct":
                len(zero_speed) / len(target) * 100
        })

        print("\nSaved:", save_path)

pd.DataFrame(summary).to_csv(
    OUT / "timeseries_summary.csv",
    index=False
)

print("\nALL TIMESERIES EXTRACTED")
