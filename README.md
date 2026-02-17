# Urban Waste Dynamics of NYC

Reproducible analytical pipeline for NYC DSNY (Department of Sanitation) truck-level waste collection data (~7.1M rows, 2003-2015). The project performs temporal clustering to identify areas with similar waste generation patterns and ensemble forecasting to predict future tonnages.

## Research Objectives

1. **Temporal Clustering** - Identify DSNY districts and sections with similar weekly waste generation patterns using multiple clustering methods (K-Means, DTW K-Means, Hierarchical, DBSCAN)
2. **Tonnage Forecasting** - Predict weekly waste tonnages per district using an ensemble of classical, ML, and deep learning models (SARIMA, Prophet, XGBoost, LightGBM, LSTM)
3. **Spatial Analysis** - Map waste generation intensity and cluster assignments across NYC, enriched with Census population data for per-capita normalization

## Data

The pipeline analyzes the `workcomp_extract.csv` dataset containing individual truck load records with tonnage, district/section codes, material type, and timestamps. Three primary waste streams are tracked:

| Code | Stream | Description |
|------|--------|-------------|
| 01 | Refuse | Curbside household collection |
| 31 | Paper | Newspaper / magazines / corrugated |
| 33 | MGP | Metal / glass / plastic recycling |

Data is aggregated to weekly totals by district and section, with rolling averages (4-week, 13-week, 52-week) computed for trend analysis.

## Project Structure

```
.
├── Makefile                          # Pipeline orchestration
├── requirements.txt                  # Pinned dependencies
├── src/
│   ├── config.py                     # Paths, material codes, constants
│   ├── processing/
│   │   ├── clean.py                  # Raw CSV → cleaned parquet
│   │   ├── aggregate.py              # Weekly aggregation by district/section
│   │   └── spatial.py                # Shapefile processing + Census population
│   ├── analysis/
│   │   ├── clustering.py             # K-Means, DTW, Hierarchical, DBSCAN
│   │   ├── prediction.py             # SARIMA, Prophet, XGBoost, LightGBM, LSTM
│   │   └── evaluation.py             # Shared metrics (silhouette, RMSE, etc.)
│   └── visualization/
│       ├── timeseries.py             # Time series and seasonal plots
│       ├── spatial.py                # Choropleth maps + interactive Folium
│       └── comparison.py             # Model/cluster comparison charts
├── notebooks/
│   ├── 01_eda_raw_data.ipynb         # Raw data profiling
│   ├── 02_eda_processed.ipynb        # Processed data exploration
│   ├── 03_clustering_exploration.ipynb
│   ├── 04_prediction_exploration.ipynb
│   └── 05_spatial_exploration.ipynb
├── data/
│   ├── processed/                    # Cleaned parquet files (generated)
│   └── spatial/                      # GeoDataFrames with population (generated)
├── outputs/
│   ├── figures/                      # Static plots
│   ├── interactive/                  # Folium HTML maps
│   ├── models/                       # Saved model checkpoints
│   └── results/                      # Cluster assignments, metrics, predictions
├── resources/                        # Source data and reference materials
│   └── DSNY/                         # Raw CSV, shapefiles, material codes
└── logs/                             # Pipeline logs
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the DSNY source data in `resources/DSNY/`:
- `workcomp_extract.csv` (raw truck load data)
- `Material_codes.xlsx` (material type code mapping)
- `DSNY_Sections/` (section boundary shapefile)
- `DSNY_Citywide_collection_frequencies/` (collection schedule shapefile)

## Usage

The pipeline is driven by Make targets with file-based dependency tracking:

```bash
make all          # Run full pipeline (process → spatial → cluster → predict → visualize)
make dev          # Run full pipeline on 10% sample (fast iteration)
make process      # Clean raw CSV and aggregate weekly
make spatial      # Process shapefiles and fetch Census population
make cluster      # Run all clustering methods
make predict      # Run ensemble forecasting models
make visualize    # Generate all figures and interactive maps
make clean        # Remove all generated files
```

### Pipeline Stages

1. **Process** - Reads 2.3GB CSV in chunks, strips whitespace, removes invalid records (placeholder truck IDs, null dates, non-positive tonnage), filters to primary streams, normalizes district/section codes, and saves as parquet. Aggregates to weekly totals at district and section levels.

2. **Spatial** - Loads DSNY section boundary shapefiles, fetches Census ACS 2019 population at tract level, and performs area-weighted interpolation to assign population to each DSNY section.

3. **Cluster** - Builds feature matrices from weekly time series (z-score normalized), runs four clustering methods at both district and section levels, and evaluates with silhouette, Calinski-Harabasz, and Davies-Bouldin indices.

4. **Predict** - Trains five forecasting models per district with a temporal train/test split (last 52 weeks held out). Evaluates with RMSE, MAE, MAPE, and R².

5. **Visualize** - Generates static time series plots, choropleth maps, interactive Folium maps, cluster comparison charts, and forecast diagnostics.

### Development Mode

Set `DEV_MODE=1` (or use `make dev`) to process a 10% random sample of the raw data for faster iteration.

## Notebooks

The Jupyter notebooks are designed for interactive exploration after running the pipeline:

| Notebook | Prerequisites | Purpose |
|----------|--------------|---------|
| `01_eda_raw_data` | Raw CSV | Column profiling, data quality, temporal coverage |
| `02_eda_processed` | `make process` | Time series patterns, seasonality, correlations |
| `03_clustering_exploration` | `make cluster` | Silhouette analysis, cluster centers, dendrograms |
| `04_prediction_exploration` | `make predict` | Model comparison, error analysis, feature importance |
| `05_spatial_exploration` | `make spatial` + `make cluster` | Interactive maps, spatial autocorrelation (Moran's I) |

```bash
jupyter lab notebooks/
```

## Key Dependencies

- **Data**: pandas, pyarrow, numpy
- **Clustering**: scikit-learn, tslearn (DTW), scipy
- **Forecasting**: statsmodels, pmdarima, prophet, xgboost, lightgbm, torch
- **Spatial**: geopandas, folium, cenpy
- **Visualization**: matplotlib, seaborn

See `requirements.txt` for the full pinned dependency list.

## Reproducibility

- Random seed set globally (`RANDOM_SEED = 42`) across Python, NumPy, and PyTorch
- PyTorch code is device-agnostic (auto-detects MPS / CUDA / CPU)
- All intermediate outputs saved as parquet for reproducible downstream stages
- Visualization is fully separated from analysis (reads saved results)
