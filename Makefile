.PHONY: all dev process spatial cluster predict visualize clean help

PYTHON = .venv/bin/python

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
all: process spatial cluster predict visualize

# ---------------------------------------------------------------------------
# Development mode (10% sample for fast iteration)
# ---------------------------------------------------------------------------
dev:
	DEV_MODE=1 $(MAKE) all

# ---------------------------------------------------------------------------
# Stage 1: Data processing
# ---------------------------------------------------------------------------
process: data/processed/waste_clean.parquet data/processed/weekly_district.parquet

data/processed/waste_clean.parquet: resources/DSNY/workcomp_extract.csv src/processing/clean.py src/config.py
	$(PYTHON) -m src.processing.clean

data/processed/weekly_district.parquet: data/processed/waste_clean.parquet src/processing/aggregate.py
	$(PYTHON) -m src.processing.aggregate

# ---------------------------------------------------------------------------
# Stage 2: Spatial data processing
# ---------------------------------------------------------------------------
spatial: data/spatial/dsny_sections.parquet

data/spatial/dsny_sections.parquet: resources/DSNY/DSNY_Sections/DSNY_sections.shp src/processing/spatial.py
	$(PYTHON) -m src.processing.spatial

# ---------------------------------------------------------------------------
# Stage 3: Clustering analysis
# ---------------------------------------------------------------------------
cluster: outputs/results/cluster_metrics.csv

outputs/results/cluster_metrics.csv: data/processed/weekly_district.parquet data/processed/weekly_section.parquet src/analysis/clustering.py
	$(PYTHON) -m src.analysis.clustering

# ---------------------------------------------------------------------------
# Stage 4: Prediction analysis
# ---------------------------------------------------------------------------
predict: outputs/results/prediction_results.parquet

outputs/results/prediction_results.parquet: data/processed/weekly_district.parquet src/analysis/prediction.py
	$(PYTHON) -m src.analysis.prediction

# ---------------------------------------------------------------------------
# Stage 5: Visualizations
# ---------------------------------------------------------------------------
visualize: outputs/results/cluster_metrics.csv data/spatial/dsny_sections.parquet
	$(PYTHON) -m src.visualization.timeseries
	$(PYTHON) -m src.visualization.spatial
	$(PYTHON) -m src.visualization.comparison

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
clean:
	rm -rf data/processed/* data/spatial/* outputs/figures/* outputs/interactive/* outputs/models/* outputs/results/* logs/*

help:
	@echo "Available targets:"
	@echo "  make all        — Run full pipeline (process → spatial → cluster → predict → visualize)"
	@echo "  make dev        — Run full pipeline on 10% sample"
	@echo "  make process    — Clean raw CSV and aggregate weekly"
	@echo "  make spatial    — Process shapefiles and fetch Census population"
	@echo "  make cluster    — Run all clustering methods"
	@echo "  make predict    — Run ensemble forecasting models"
	@echo "  make visualize  — Generate all figures and interactive maps"
	@echo "  make clean      — Remove all generated files"
