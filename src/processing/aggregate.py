"""
Weekly aggregation pipeline.

Reads the cleaned parquet and produces weekly tonnage summaries at both
district and section levels, pivoted by waste stream.

Usage:
    python -m src.processing.aggregate
"""

import sys

import pandas as pd
from tqdm import tqdm

from src.config import (
    DATA_PROCESSED,
    PRIMARY_STREAMS,
    STREAM_COL_NAMES,
    setup_logging,
)

logger = setup_logging("aggregate")


def build_weekly_agg(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Aggregate tonnages by (group_col, iso_year, iso_week, material) and
    pivot waste streams into separate columns.

    Parameters
    ----------
    df : DataFrame with Collection_Date, Material_Type_Code, Tons_Collected
    group_col : 'District_Code' or 'Section_Code'

    Returns
    -------
    DataFrame indexed by (group_col, week_start) with columns:
        tons_refuse, tons_paper, tons_mgp, tons_total,
        plus rolling averages.
    """
    # Create ISO year-week and a week_start date (Monday)
    df = df.copy()
    iso = df["Collection_Date"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)

    # Aggregate by group + year-week + material
    agg = (
        df.groupby([group_col, "iso_year", "iso_week", "Material_Type_Code"])["Tons_Collected"]
        .sum()
        .reset_index()
    )

    # Pivot material types into columns
    agg["stream_col"] = agg["Material_Type_Code"].map(STREAM_COL_NAMES)
    pivoted = agg.pivot_table(
        index=[group_col, "iso_year", "iso_week"],
        columns="stream_col",
        values="Tons_Collected",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # Flatten column index
    pivoted.columns.name = None

    # Ensure all stream columns exist
    for col_name in STREAM_COL_NAMES.values():
        if col_name not in pivoted.columns:
            pivoted[col_name] = 0.0

    # Total across streams
    stream_cols = list(STREAM_COL_NAMES.values())
    pivoted["tons_total"] = pivoted[stream_cols].sum(axis=1)

    # Create a proper date column (Monday of each ISO week)
    pivoted["week_start"] = pd.to_datetime(
        pivoted["iso_year"].astype(str) + "-W"
        + pivoted["iso_week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
    )

    # Sort for rolling calculations
    pivoted = pivoted.sort_values([group_col, "week_start"]).reset_index(drop=True)

    # Rolling averages per group
    logger.info(f"Computing rolling averages for {group_col}...")
    rolling_windows = {"4wk": 4, "13wk": 13, "52wk": 52}

    for suffix, window in rolling_windows.items():
        for stream_col in stream_cols + ["tons_total"]:
            new_col = f"{stream_col}_ma_{suffix}"
            pivoted[new_col] = (
                pivoted.groupby(group_col)[stream_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

    # Drop intermediate columns
    pivoted = pivoted.drop(columns=["iso_year", "iso_week"])

    return pivoted


def main():
    logger.info("=" * 60)
    logger.info("Starting weekly aggregation pipeline")

    clean_path = DATA_PROCESSED / "waste_clean.parquet"
    if not clean_path.exists():
        logger.error(f"Cleaned data not found at {clean_path}. Run clean.py first.")
        sys.exit(1)

    logger.info(f"Reading {clean_path}...")
    df = pd.read_parquet(clean_path)
    logger.info(f"Loaded {len(df):,} rows")

    # --- District-level aggregation ---
    logger.info("Aggregating at district level...")
    weekly_district = build_weekly_agg(df, "District_Code")
    out_district = DATA_PROCESSED / "weekly_district.parquet"
    weekly_district.to_parquet(out_district, index=False)
    logger.info(
        f"District: {weekly_district.shape[0]:,} rows, "
        f"{weekly_district['District_Code'].nunique()} districts, "
        f"saved to {out_district}"
    )

    # --- Section-level aggregation ---
    logger.info("Aggregating at section level...")
    weekly_section = build_weekly_agg(df, "Section_Code")
    out_section = DATA_PROCESSED / "weekly_section.parquet"
    weekly_section.to_parquet(out_section, index=False)
    logger.info(
        f"Section: {weekly_section.shape[0]:,} rows, "
        f"{weekly_section['Section_Code'].nunique()} sections, "
        f"saved to {out_section}"
    )

    # Summary stats
    for level, data in [("District", weekly_district), ("Section", weekly_section)]:
        logger.info(f"\n--- {level} Summary ---")
        logger.info(f"  Date range: {data['week_start'].min()} to {data['week_start'].max()}")
        for col in ["tons_refuse", "tons_paper", "tons_mgp", "tons_total"]:
            logger.info(f"  {col}: mean={data[col].mean():.1f}, std={data[col].std():.1f}")

    logger.info("Aggregation pipeline complete")


if __name__ == "__main__":
    main()
