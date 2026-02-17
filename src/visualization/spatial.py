"""
Spatial visualization module.

Produces static choropleth maps (matplotlib/geopandas) and interactive
Folium maps for exploring cluster assignments and tonnage patterns.

Usage:
    python -m src.visualization.spatial
"""

import warnings

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    DATA_PROCESSED,
    DATA_SPATIAL,
    FIGURES_DIR,
    INTERACTIVE_DIR,
    RESULTS_DIR,
    setup_logging,
)

logger = setup_logging("viz_spatial")

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150


def _load_sections() -> gpd.GeoDataFrame | None:
    """Load DSNY sections geodataframe."""
    for path in [
        DATA_SPATIAL / "dsny_sections_with_pop.parquet",
        DATA_SPATIAL / "dsny_sections.parquet",
    ]:
        if path.exists():
            gdf = gpd.read_parquet(path)
            logger.info(f"Loaded sections: {len(gdf)} polygons from {path.name}")
            return gdf

    logger.warning("No spatial data found — run spatial.py first")
    return None


# ---------------------------------------------------------------------------
# Static maps (matplotlib)
# ---------------------------------------------------------------------------

def plot_tonnage_choropleth(sections: gpd.GeoDataFrame):
    """Average weekly refuse tonnage by section."""
    logger.info("Plotting tonnage choropleth...")

    section_path = DATA_PROCESSED / "weekly_section.parquet"
    if not section_path.exists():
        logger.warning("Section data not found")
        return

    df = pd.read_parquet(section_path)
    avg_tons = df.groupby("Section_Code")["tons_refuse"].mean().reset_index()
    avg_tons.columns = ["Section_Code", "avg_tons_refuse"]

    merged = sections.merge(avg_tons, on="Section_Code", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    merged.plot(
        column="avg_tons_refuse",
        cmap="YlOrRd",
        legend=True,
        legend_kwds={"label": "Avg Weekly Tons (Refuse)", "shrink": 0.6},
        missing_kwds={"color": "lightgray", "label": "No data"},
        ax=ax,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title("Average Weekly Refuse Tonnage by DSNY Section", fontsize=14)
    ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "choropleth_tonnage.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_maps(sections: gpd.GeoDataFrame, level: str = "section"):
    """Static maps showing cluster assignments from each method."""
    logger.info(f"Plotting cluster maps ({level})...")

    group_col = "Section_Code" if level == "section" else "District_Code"
    assign_path = RESULTS_DIR / f"cluster_assignments_{level}.parquet"
    if not assign_path.exists():
        logger.warning(f"No cluster assignments for {level}")
        return

    assignments = pd.read_parquet(assign_path)
    cluster_cols = [c for c in assignments.columns if c.startswith("cluster_")]

    if level == "district":
        # Dissolve sections to district level
        sections = sections.copy()
        sections["District_Code"] = sections["Section_Code"].str[:4]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sections = sections.dissolve(by="District_Code", as_index=False)

    n_methods = len(cluster_cols)
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 14))
    if n_methods == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("Set3", 15)

    for ax, col in zip(axes, cluster_cols):
        merged = sections.merge(assignments[[group_col, col]], on=group_col, how="left")
        merged[col] = merged[col].fillna(-1).astype(int)

        merged.plot(
            column=col,
            cmap=cmap,
            legend=True,
            legend_kwds={"loc": "lower left", "fontsize": 8},
            missing_kwds={"color": "lightgray"},
            ax=ax,
            edgecolor="black",
            linewidth=0.3,
            categorical=True,
        )
        method_name = col.replace("cluster_", "").replace("_", " ").title()
        ax.set_title(method_name, fontsize=12)
        ax.set_axis_off()

    plt.suptitle(f"Cluster Assignments — {level.title()} Level", y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"cluster_maps_{level}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_error_map(sections: gpd.GeoDataFrame):
    """Map of prediction MAPE by district."""
    logger.info("Plotting prediction error map...")

    metrics_path = RESULTS_DIR / "prediction_by_district.csv"
    if not metrics_path.exists():
        logger.warning("No per-district prediction metrics")
        return

    metrics = pd.read_csv(metrics_path)

    # Dissolve to district
    sec = sections.copy()
    sec["District_Code"] = sec["Section_Code"].str[:4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        districts_geo = sec.dissolve(by="District_Code", as_index=False)

    merged = districts_geo.merge(metrics, on="District_Code", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    merged.plot(
        column="mape",
        cmap="RdYlGn_r",
        legend=True,
        legend_kwds={"label": "MAPE", "shrink": 0.6},
        missing_kwds={"color": "lightgray"},
        ax=ax,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title("Prediction Error (MAPE) by District — Best Model", fontsize=14)
    ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "prediction_error_map.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive maps (Folium)
# ---------------------------------------------------------------------------

def create_interactive_cluster_map(sections: gpd.GeoDataFrame, level: str = "section"):
    """Interactive Folium map with cluster assignments and layer toggle."""
    logger.info(f"Creating interactive cluster map ({level})...")

    group_col = "Section_Code" if level == "section" else "District_Code"
    assign_path = RESULTS_DIR / f"cluster_assignments_{level}.parquet"
    if not assign_path.exists():
        logger.warning(f"No cluster assignments for {level}")
        return

    assignments = pd.read_parquet(assign_path)

    geo = sections.copy()
    if level == "district":
        geo["District_Code"] = geo["Section_Code"].str[:4]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geo = geo.dissolve(by="District_Code", as_index=False)

    geo = geo.merge(assignments, on=group_col, how="left")

    # Center on NYC
    center = [40.7128, -74.0060]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    # Color palette
    colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
        "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
    ]

    cluster_cols = [c for c in geo.columns if c.startswith("cluster_")]

    for col in cluster_cols:
        method_name = col.replace("cluster_", "").replace("_", " ").title()
        layer = folium.FeatureGroup(name=method_name)

        for _, row in geo.iterrows():
            cluster_id = int(row[col]) if pd.notna(row[col]) else -1
            color = colors[cluster_id % len(colors)] if cluster_id >= 0 else "#cccccc"

            popup_text = f"<b>{row[group_col]}</b><br>Cluster: {cluster_id}"
            if "population" in row.index and pd.notna(row.get("population")):
                popup_text += f"<br>Pop: {int(row['population']):,}"

            if row.geometry is not None:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, c=color: {
                        "fillColor": c,
                        "color": "black",
                        "weight": 0.5,
                        "fillOpacity": 0.6,
                    },
                    popup=folium.Popup(popup_text, max_width=200),
                ).add_to(layer)

        layer.add_to(m)

    folium.LayerControl().add_to(m)

    out_path = INTERACTIVE_DIR / f"cluster_map_{level}.html"
    m.save(str(out_path))
    logger.info(f"Saved interactive map to {out_path}")


def create_interactive_tonnage_map(sections: gpd.GeoDataFrame):
    """Interactive Folium map with tonnage data."""
    logger.info("Creating interactive tonnage map...")

    section_path = DATA_PROCESSED / "weekly_section.parquet"
    if not section_path.exists():
        logger.warning("Section data not found")
        return

    df = pd.read_parquet(section_path)
    avg_tons = df.groupby("Section_Code")["tons_refuse"].mean().reset_index()
    avg_tons.columns = ["Section_Code", "avg_tons"]

    geo = sections.merge(avg_tons, on="Section_Code", how="left")

    center = [40.7128, -74.0060]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    # Choropleth via GeoJson
    max_tons = geo["avg_tons"].quantile(0.95)

    for _, row in geo.iterrows():
        tons = row.get("avg_tons", 0)
        if pd.isna(tons):
            tons = 0
        intensity = min(tons / max_tons, 1.0) if max_tons > 0 else 0

        # Red intensity scale
        r = int(255)
        g = int(255 * (1 - intensity))
        b = int(255 * (1 - intensity))
        color = f"#{r:02x}{g:02x}{b:02x}"

        popup_text = f"<b>{row['Section_Code']}</b><br>Avg tons/week: {tons:.1f}"

        if row.geometry is not None:
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color: {
                    "fillColor": c,
                    "color": "black",
                    "weight": 0.5,
                    "fillOpacity": 0.6,
                },
                popup=folium.Popup(popup_text, max_width=200),
            ).add_to(m)

    out_path = INTERACTIVE_DIR / "tonnage_map.html"
    m.save(str(out_path))
    logger.info(f"Saved interactive tonnage map to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Generating spatial visualizations")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    sections = _load_sections()
    if sections is None:
        logger.error("Cannot generate spatial visualizations without section data")
        return

    # Static maps
    plot_tonnage_choropleth(sections)
    for level in ["district", "section"]:
        plot_cluster_maps(sections, level)
    plot_prediction_error_map(sections)

    # Interactive maps
    for level in ["district", "section"]:
        create_interactive_cluster_map(sections, level)
    create_interactive_tonnage_map(sections)

    logger.info("Spatial visualization complete")


if __name__ == "__main__":
    main()
