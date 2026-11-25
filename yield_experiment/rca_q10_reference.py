from pathlib import Path
import pandas as pd

SRC = Path("results/spec_all_products")
OUT = Path("results/spec_all_products_q10")

# Load data
split = pd.read_csv(
    SRC / "all_product_batch_split.csv"
)

spec = pd.read_csv(
    OUT / "all_product_specifications_q10.csv"
)

process = pd.read_csv(
    "data/Process.csv", sep=";"
)

assert len(process) == 1005
assert not process.duplicated(["code", "batch"]).any()

SCENARIOS = [(17, 789), (15, 799)]

summary = []

for code, batch in SCENARIOS:

    s = spec[spec["code"] == code].iloc[0]

    lsl = s["lsl_q10"]
    usl = s["usl_candidate"]

    # Select historical batches of the same product
    train = split[
        (split["code"] == code) &
        (split["phase"] == "train")
    ].copy()

    # Select batches within hypothetical specifications
    reference = train[
        train["tbl_yield"].between(lsl, usl)
    ].copy()

    excluded = train[
        ~train["tbl_yield"].between(lsl, usl)
    ].copy()

    # Select the target batch
    target = split[
        (split["code"] == code) &
        (split["batch"] == batch) &
        (split["phase"] == "test")
    ].copy()

    assert len(target) == 1
    assert target["tbl_yield"].iloc[0] < lsl

    # Join yield records with process data
    ref_process = reference[
        ["code", "batch", "tbl_yield"]
    ].merge(
        process,
        on=["code", "batch"],
        how="left",
        validate="one_to_one",
        indicator=True
    )

    target_process = target[
        ["code", "batch", "tbl_yield"]
    ].merge(
        process,
        on=["code", "batch"],
        how="left",
        validate="one_to_one",
        indicator=True
    )

    assert (ref_process["_merge"] == "both").all()
    assert (target_process["_merge"] == "both").all()

    ref_process = ref_process.drop(columns="_merge")
    target_process = target_process.drop(columns="_merge")

    # Reference yield statistics
    y = reference["tbl_yield"]

    print(f"\n=== CODE {code} / BATCH {batch} ===")

    print("LSL:", lsl)
    print("USL:", usl)
    print("Historical batches:", len(train))
    print("Reference batches:", len(reference))
    print("Excluded batches:", len(excluded))

    print("\n--- REFERENCE YIELD STATISTICS ---")
    print("Mean:", round(y.mean(), 4))
    print("Median:", round(y.median(), 4))
    print("Std:", round(y.std(), 4))
    print("Min:", round(y.min(), 4))
    print("Max:", round(y.max(), 4))

    print("\n--- TARGET YIELD ---")
    print("Actual yield:", target["tbl_yield"].iloc[0])
    print(
        "Gap from reference median:",
        round(
            y.median() - target["tbl_yield"].iloc[0],
            4
        )
    )

    print("\n--- EXCLUDED HISTORICAL BATCHES ---")
    print(
        excluded[["batch", "tbl_yield"]]
        .to_string(index=False)
    )

    ref_process.to_csv(
        OUT / f"reference_code_{code}.csv",
        index=False
    )

    target_process.to_csv(
        OUT / f"target_code_{code}_batch_{batch}.csv",
        index=False
    )

    summary.append({
        "code": code,
        "batch": batch,
        "train_n": len(train),
        "reference_n": len(reference),
        "excluded_n": len(excluded),
        "lsl": lsl,
        "usl": usl,
        "reference_yield_mean": y.mean(),
        "reference_yield_median": y.median(),
        "reference_yield_std": y.std(),
        "target_yield": target["tbl_yield"].iloc[0]
    })

pd.DataFrame(summary).to_csv(
    OUT / "rca_reference_summary.csv",
    index=False
)

print("\nALL CHECKS PASSED")
