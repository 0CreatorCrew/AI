from pathlib import Path
from zipfile import ZipFile
import csv

import pandas as pd
import numpy as np

OUT = Path("results/spec_all_products")
OUT.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2020-10-19 02:16:00")

# 1. Laboratory data
lab = pd.read_csv(
    "data/Laboratory.csv",
    sep=";"
)

lab = lab[
    ["code", "batch", "start", "tbl_yield"]
].copy()

lab["tbl_yield"] = pd.to_numeric(
    lab["tbl_yield"],
    errors="coerce"
)

assert not lab.duplicated(["code", "batch"]).any()
assert lab["code"].nunique() == 25
assert len(lab) == 1005
assert lab["tbl_yield"].notna().all()


# 2. Timestamp parsing
def parse_timestamps(series):

    raw = series.astype("string").str.strip()

    # Existing format: DDMMYYYY HH:MM
    first = pd.to_datetime(
        raw,
        format="%d%m%Y %H:%M",
        errors="coerce"
    )

    # ISO format: YYYY-MM-DD HH:MM:SS
    second = pd.to_datetime(
        raw.where(first.isna()),
        format="ISO8601",
        errors="coerce"
    )

    parsed = first.fillna(second)

    return parsed, first.notna().sum(), second.notna().sum()


# 3. Extract batch timestamps
parts = []
quality = []

with ZipFile("data/Process.zip") as z:

    names = sorted(
        n for n in z.namelist()
        if n.lower().endswith(".csv")
        and not n.startswith("__MACOSX")
        and not Path(n).name.startswith("._")
    )

    print("\n=== PROCESS FILES ===")
    print("CSV files:", len(names))

    for name in names:

        with z.open(name) as f:
            header = f.readline().decode("utf-8-sig")

        sep = csv.Sniffer().sniff(
            header,
            delimiters=";,\t"
        ).delimiter

        with z.open(name) as f:
            df = pd.read_csv(
                f,
                sep=sep,
                usecols=["code", "batch", "timestamp"]
            )

        df["ts"], legacy_n, iso_n = parse_timestamps(
            df["timestamp"]
        )

        invalid = int(df["ts"].isna().sum())

        quality.append({
            "file": name,
            "total_rows": len(df),
            "legacy_format_rows": int(legacy_n),
            "iso_format_rows": int(iso_n),
            "invalid_timestamps": invalid,
            "missing_timestamps": int(
                df["timestamp"].isna().sum()
            )
        })

        print(
            name,
            "| rows:", len(df),
            "| legacy:", legacy_n,
            "| ISO:", iso_n,
            "| invalid:", invalid
        )

        if invalid:

            print("\nINVALID EXAMPLES")

            print(
                df.loc[
                    df["ts"].isna(),
                    ["code", "batch", "timestamp"]
                ].head(10).to_string(index=False)
            )

            raise ValueError(
                f"Unparsed timestamps: {name}"
            )

        stats = df.groupby(
            ["code", "batch"],
            as_index=False
        ).agg(
            start_ts=("ts", "min"),
            end_ts=("ts", "max"),
            log_rows=("ts", "size")
        )

        parts.append(stats)


# 4. Combine batch timestamps
timestamps = pd.concat(
    parts,
    ignore_index=True
).groupby(
    ["code", "batch"],
    as_index=False
).agg(
    start_ts=("start_ts", "min"),
    end_ts=("end_ts", "max"),
    log_rows=("log_rows", "sum")
)

print("\n=== TIMESTAMP COVERAGE ===")
print("Timestamp batches:", len(timestamps))
print("First timestamp:", timestamps["start_ts"].min())
print("Last timestamp:", timestamps["end_ts"].max())

assert (
    timestamps["end_ts"] >= timestamps["start_ts"]
).all()


# 5. Merge Laboratory and Process
data = lab.merge(
    timestamps,
    on=["code", "batch"],
    how="left",
    validate="one_to_one"
)

print("\n=== MERGE VALIDATION ===")
print("Laboratory batches:", len(lab))
print("Merged batches:", len(data))
print("Missing start:", data["start_ts"].isna().sum())
print("Missing end:", data["end_ts"].isna().sum())


# 6. Temporal split
data["phase"] = np.select(
    [
        data["start_ts"].isna()
        | data["end_ts"].isna(),
        data["end_ts"] < CUTOFF,
        data["start_ts"] >= CUTOFF
    ],
    [
        "missing",
        "train",
        "test"
    ],
    default="boundary"
)

print("\n=== OVERALL SPLIT ===")
print("Cutoff:", CUTOFF)
print(data["phase"].value_counts().to_string())


# 7. Product-level split summary
summary = pd.crosstab(
    data["code"],
    data["phase"]
).reindex(
    columns=[
        "train", "test",
        "boundary", "missing"
    ],
    fill_value=0
)

summary["total"] = summary.sum(axis=1)

print("\n=== PRODUCT SPLIT SUMMARY ===")
print(summary.to_string())

assert len(summary) == 25
assert summary["total"].sum() == 1005


# 8. Previous experiment consistency
print("\n=== PREVIOUS EXPERIMENT CHECK ===")

expected = {
    13: (70, 60, 1),
    17: (119, 88, 0),
    23: (126, 61, 0)
}

all_match = True

for code, values in expected.items():

    observed = tuple(
        int(summary.loc[code, col])
        for col in ["train", "test", "boundary"]
    )

    match = observed == values
    all_match &= match

    print(
        f"Code {code}:",
        "expected =", values,
        "| observed =", observed,
        "| match =", match
    )


# 9. Training data availability
print("\n=== TRAINING DATA AVAILABILITY ===")

print(
    "40 or more:",
    (summary["train"] >= 40).sum()
)

print(
    "10 to 39:",
    summary["train"].between(10, 39).sum()
)

print(
    "1 to 9:",
    summary["train"].between(1, 9).sum()
)

print(
    "No training data:",
    (summary["train"] == 0).sum()
)

print("\n=== PRODUCTS WITHOUT TRAINING DATA ===")

print(
    summary.index[
        summary["train"] == 0
    ].tolist()
)

print("\n=== PRODUCTS WITHOUT TEST DATA ===")

print(
    summary.index[
        summary["test"] == 0
    ].tolist()
)


# 10. Save timestamp quality report
pd.DataFrame(quality).to_csv(
    OUT / "all_product_timestamp_quality.csv",
    index=False
)

# 11. Save split results
data.to_csv(
    OUT / "all_product_batch_split.csv",
    index=False
)

summary.to_csv(
    OUT / "all_product_split_summary.csv"
)

print("\n=== SAVED FILES ===")
print(OUT / "all_product_timestamp_quality.csv")
print(OUT / "all_product_batch_split.csv")
print(OUT / "all_product_split_summary.csv")

print("\nPrevious split consistency:", all_match)
