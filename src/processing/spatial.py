"""
Spatial data processing pipeline.

Loads DSNY shapefiles, fetches Census population data, performs spatial
joins, and produces enriched GeoDataFrames.

Usage:
    python -m src.processing.spatial
"""

import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DATA_SPATIAL,
    FREQUENCIES_SHP,
    NYC_COUNTY_FIPS,
    SECTIONS_SHP,
    clean_district_code,
    setup_logging,
)

logger = setup_logging("spatial")


def load_dsny_sections() -> gpd.GeoDataFrame:
    """Load and clean DSNY sections shapefile."""
    logger.info(f"Loading sections shapefile: {SECTIONS_SHP}")
    gdf = gpd.read_file(SECTIONS_SHP)
    logger.info(f"  Loaded {len(gdf)} section polygons, CRS: {gdf.crs}")

    # Inspect column names
    logger.info(f"  Columns: {list(gdf.columns)}")

    # Identify the section code column (varies by shapefile version)
    code_col = None
    for candidate in ["Sections", "DSNY_Secti", "Section", "SEC_CODE", "SECTION", "DSECT"]:
        if candidate in gdf.columns:
            code_col = candidate
            break

    if code_col is None:
        # Use the first non-geometry column that looks like a code
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype == object:
                code_col = col
                break

    if code_col is None:
        logger.error("Could not identify section code column in shapefile")
        sys.exit(1)

    logger.info(f"  Using '{code_col}' as section identifier")
    gdf = gdf.rename(columns={code_col: "Section_Code"})
    gdf["Section_Code"] = gdf["Section_Code"].astype(str).str.strip()

    # Reproject to EPSG:4326 for Census compatibility, keep a projected copy
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:2263")  # NY State Plane (common for NYC data)
    gdf_proj = gdf.to_crs("EPSG:2263")  # projected for area calculations
    gdf = gdf.to_crs("EPSG:4326")       # geographic for Census join

    # Store area in sq meters from projected CRS
    gdf["area_sqm"] = gdf_proj.geometry.area

    return gdf


def load_collection_frequencies() -> gpd.GeoDataFrame:
    """Load citywide collection frequencies shapefile."""
    if not FREQUENCIES_SHP.exists():
        logger.warning(f"Frequencies shapefile not found: {FREQUENCIES_SHP}")
        return None

    logger.info(f"Loading collection frequencies: {FREQUENCIES_SHP}")
    gdf = gpd.read_file(FREQUENCIES_SHP)
    logger.info(f"  Loaded {len(gdf)} features, columns: {list(gdf.columns)}")
    return gdf


def fetch_census_population() -> gpd.GeoDataFrame:
    """
    Fetch Census tract-level population for NYC using cenpy.

    Falls back to a simpler approach if cenpy has issues.
    """
    logger.info("Fetching Census population data for NYC...")

    try:
        from cenpy import products

        acs = products.ACS(2019)
        nyc_tracts = acs.from_county(
            state="36",
            county=["005", "047", "061", "081", "085"],
            level="tract",
            variables=["B01001_001E"],  # Total population
        )
        nyc_tracts = nyc_tracts.rename(columns={"B01001_001E": "population"})
        nyc_tracts["population"] = pd.to_numeric(nyc_tracts["population"], errors="coerce").fillna(0)
        nyc_tracts = nyc_tracts.to_crs("EPSG:4326")
        logger.info(f"  Fetched {len(nyc_tracts)} Census tracts")
        return nyc_tracts

    except Exception as e:
        logger.warning(f"cenpy fetch failed: {e}")
        logger.info("Attempting fallback Census data fetch...")

        try:
            import requests

            # Use Census API directly
            base_url = "https://api.census.gov/data/2019/acs/acs5"
            tracts_all = []

            for county_fips in ["005", "047", "061", "081", "085"]:
                params = {
                    "get": "B01001_001E,NAME",
                    "for": "tract:*",
                    "in": f"state:36 county:{county_fips}",
                }
                resp = requests.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                header = data[0]
                rows = data[1:]
                df = pd.DataFrame(rows, columns=header)
                tracts_all.append(df)

            tracts_df = pd.concat(tracts_all, ignore_index=True)
            tracts_df["population"] = pd.to_numeric(tracts_df["B01001_001E"], errors="coerce").fillna(0)
            tracts_df["GEOID"] = tracts_df["state"] + tracts_df["county"] + tracts_df["tract"]

            # Fetch tract geometries from Census TIGER
            tiger_url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2019/MapServer/8/query"
            geom_frames = []
            for county_fips in ["005", "047", "061", "081", "085"]:
                params = {
                    "where": f"STATE='36' AND COUNTY='{county_fips}'",
                    "outFields": "GEOID,BASENAME",
                    "outSR": "4326",
                    "f": "geojson",
                    "returnGeometry": "true",
                }
                resp = requests.get(tiger_url, params=params, timeout=60)
                resp.raise_for_status()
                gdf = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
                geom_frames.append(gdf)

            tracts_geo = pd.concat(geom_frames, ignore_index=True)
            tracts_geo = tracts_geo.merge(tracts_df[["GEOID", "population"]], on="GEOID", how="left")
            logger.info(f"  Fetched {len(tracts_geo)} Census tracts via API")
            return tracts_geo

        except Exception as e2:
            logger.error(f"Census data fetch failed entirely: {e2}")
            logger.info("Proceeding without population data")
            return None


def spatial_join_population(
    sections: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Area-weighted interpolation of tract population into DSNY sections.

    For each section, finds overlapping tracts and allocates population
    proportional to the overlap area.
    """
    logger.info("Performing area-weighted population interpolation...")

    # Work in projected CRS for accurate area calculations
    sections_proj = sections.to_crs("EPSG:2263")
    tracts_proj = tracts.to_crs("EPSG:2263")

    # Compute tract areas
    tracts_proj["tract_area"] = tracts_proj.geometry.area

    # Overlay (intersection)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlay = gpd.overlay(sections_proj, tracts_proj, how="intersection")

    overlay["overlap_area"] = overlay.geometry.area
    overlay["area_fraction"] = overlay["overlap_area"] / overlay["tract_area"]
    overlay["pop_allocated"] = overlay["population"] * overlay["area_fraction"]

    # Sum allocated population per section
    pop_by_section = (
        overlay.groupby("Section_Code")["pop_allocated"]
        .sum()
        .reset_index()
        .rename(columns={"pop_allocated": "population"})
    )
    pop_by_section["population"] = pop_by_section["population"].round(0).astype(int)

    # Merge back
    sections = sections.merge(pop_by_section, on="Section_Code", how="left")
    sections["population"] = sections["population"].fillna(0).astype(int)

    logger.info(f"  Population assigned to {(sections['population'] > 0).sum()} / {len(sections)} sections")
    logger.info(f"  Total population: {sections['population'].sum():,}")

    return sections


def main():
    logger.info("=" * 60)
    logger.info("Starting spatial data processing pipeline")

    DATA_SPATIAL.mkdir(parents=True, exist_ok=True)

    # Load shapefiles
    sections = load_dsny_sections()

    frequencies = load_collection_frequencies()
    if frequencies is not None:
        freq_path = DATA_SPATIAL / "collection_frequencies.parquet"
        frequencies.to_parquet(freq_path)
        logger.info(f"Saved collection frequencies to {freq_path}")

    # Save base sections
    base_path = DATA_SPATIAL / "dsny_sections.parquet"
    sections.to_parquet(base_path)
    logger.info(f"Saved base sections to {base_path}")

    # Fetch Census data and join
    tracts = fetch_census_population()
    if tracts is not None:
        sections_pop = spatial_join_population(sections, tracts)
        pop_path = DATA_SPATIAL / "dsny_sections_with_pop.parquet"
        sections_pop.to_parquet(pop_path)
        logger.info(f"Saved sections with population to {pop_path}")
    else:
        logger.warning("Skipping population join — Census data unavailable")

    logger.info("Spatial processing pipeline complete")


if __name__ == "__main__":
    main()
