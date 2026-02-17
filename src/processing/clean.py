"""
Raw data cleaning pipeline.

Reads the 2.3GB workcomp_extract.csv in chunks, applies cleaning rules,
filters to primary waste streams (Refuse, Paper, MGP), and outputs a
compact parquet file.

Usage:
    python -m src.processing.clean
"""

import sys

import pandas as pd
from tqdm import tqdm

from src.config import (
    DATA_PROCESSED,
    DEV_MODE,
    DEV_SAMPLE_FRAC,
    DROP_SECTIONS,
    PRIMARY_STREAM_CODES,
    PRIMARY_STREAMS,
    RAW_COLUMNS,
    RAW_CSV,
    RAW_DTYPES,
    STREAM_COL_NAMES,
    clean_district_code,
    extract_borough,
    setup_logging,
)

logger = setup_logging("clean")

CHUNK_SIZE = 500_000  # rows per chunk — balances memory and speed


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules to a single chunk."""
    n_start = len(chunk)

    # Strip whitespace from string columns
    str_cols = chunk.select_dtypes(include="object").columns
    for col in str_cols:
        chunk[col] = chunk[col].str.strip()

    # Remove placeholder truck IDs
    chunk = chunk[chunk["Truck_ID"] != "9999999"]

    # Drop rows with missing timestamps
    chunk = chunk.dropna(subset=["Collection_Date"])

    # Parse dates
    chunk["Collection_Date"] = pd.to_datetime(
        chunk["Collection_Date"], errors="coerce"
    )
    chunk = chunk.dropna(subset=["Collection_Date"])

    # Remove non-positive tonnages
    chunk = chunk[chunk["Tons_Collected"] > 0]

    # Filter to primary waste streams
    chunk["Material_Type_Code"] = pd.to_numeric(
        chunk["Material_Type_Code"], errors="coerce"
    )
    chunk = chunk.dropna(subset=["Material_Type_Code"])
    chunk["Material_Type_Code"] = chunk["Material_Type_Code"].astype(int)
    chunk = chunk[chunk["Material_Type_Code"].isin(PRIMARY_STREAM_CODES)]

    # Clean district and section codes
    chunk["District_Code"] = chunk["District_Code"].apply(clean_district_code)
    chunk["Section_Code"] = chunk["Section_Code"].apply(clean_district_code)

    # Drop non-geographic sections
    chunk = chunk[~chunk["Section_Code"].isin(DROP_SECTIONS)]

    # Add derived columns
    chunk["borough"] = chunk["District_Code"].apply(extract_borough)
    chunk["material_name"] = chunk["Material_Type_Code"].map(PRIMARY_STREAMS)
    chunk["year"] = chunk["Collection_Date"].dt.year
    chunk["month"] = chunk["Collection_Date"].dt.month
    chunk["week"] = chunk["Collection_Date"].dt.isocalendar().week.astype(int)
    chunk["day_of_week"] = chunk["Collection_Date"].dt.dayofweek

    n_end = len(chunk)
    return chunk


def main():
    logger.info("=" * 60)
    logger.info("Starting raw data cleaning pipeline")
    logger.info(f"Input: {RAW_CSV}")
    logger.info(f"Dev mode: {DEV_MODE}")

    if not RAW_CSV.exists():
        logger.error(f"Raw CSV not found at {RAW_CSV}")
        sys.exit(1)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "waste_clean.parquet"

    # Count total rows for progress bar (fast line count)
    logger.info("Counting rows...")
    total_rows = sum(1 for _ in open(RAW_CSV, "r")) - 1  # subtract header
    logger.info(f"Total rows in CSV: {total_rows:,}")

    # Process in chunks
    chunks = []
    rows_read = 0
    rows_kept = 0

    reader = pd.read_csv(
        RAW_CSV,
        usecols=RAW_COLUMNS,
        dtype={k: v for k, v in RAW_DTYPES.items() if k in RAW_COLUMNS},
        chunksize=CHUNK_SIZE,
        on_bad_lines="skip",
    )

    with tqdm(total=total_rows, desc="Cleaning", unit="rows") as pbar:
        for chunk in reader:
            rows_read += len(chunk)

            if DEV_MODE:
                chunk = chunk.sample(
                    frac=DEV_SAMPLE_FRAC, random_state=42
                )

            cleaned = clean_chunk(chunk)
            rows_kept += len(cleaned)
            chunks.append(cleaned)

            pbar.update(len(chunk) if not DEV_MODE else int(len(chunk)))

    logger.info(f"Rows read: {rows_read:,}")
    logger.info(f"Rows kept after cleaning: {rows_kept:,}")
    logger.info(f"Drop rate: {1 - rows_kept / rows_read:.1%}")

    # Combine and save
    logger.info("Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)

    # Final dtype optimization
    df["Shift_ID"] = df["Shift_ID"].astype("category")
    df["District_Code"] = df["District_Code"].astype("category")
    df["Section_Code"] = df["Section_Code"].astype("category")
    df["borough"] = df["borough"].astype("category")
    df["material_name"] = df["material_name"].astype("category")
    df["Material_Type_Code"] = df["Material_Type_Code"].astype("int16")

    # Sort by date for efficient time-based queries
    df = df.sort_values("Collection_Date").reset_index(drop=True)

    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Date range: {df['Collection_Date'].min()} to {df['Collection_Date'].max()}")
    logger.info(f"Districts: {df['District_Code'].nunique()}")
    logger.info(f"Sections: {df['Section_Code'].nunique()}")
    logger.info(f"Material breakdown:\n{df['material_name'].value_counts().to_string()}")

    # Save
    df.to_parquet(output_path, index=False, engine="pyarrow")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved to {output_path} ({size_mb:.1f} MB)")
    logger.info("Cleaning pipeline complete")


if __name__ == "__main__":
    main()
