"""
Central configuration for NYC Waste Time Series Analysis.

Paths, material code mappings, constants, and shared utilities.
"""

import logging
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source data (resources directory — read-only reference)
RESOURCES_DIR = PROJECT_ROOT / "resources" / "DSNY"
RAW_CSV = RESOURCES_DIR / "workcomp_extract.csv"
MATERIAL_CODES_XLSX = RESOURCES_DIR / "Material_codes.xlsx"
ROUTE_SUBSECTION_CSV = RESOURCES_DIR / "Route_subSection.csv"
DISPOSAL_NETWORKS_CSV = RESOURCES_DIR / "DSNY_s_Refuse_and_Recycling_Disposal_Networks.csv"
SECTIONS_SHP = RESOURCES_DIR / "DSNY_Sections" / "DSNY_sections.shp"
FREQUENCIES_SHP = RESOURCES_DIR / "DSNY_Citywide_collection_frequencies" / "Citywide_frequencies.shp"

# Pipeline outputs
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPATIAL = PROJECT_ROOT / "data" / "spatial"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
INTERACTIVE_DIR = OUTPUTS_DIR / "interactive"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Dev mode — set DEV_MODE=1 env var to process a 10% sample
# ---------------------------------------------------------------------------
DEV_MODE = os.environ.get("DEV_MODE", "0") == "1"
DEV_SAMPLE_FRAC = 0.10

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

def set_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.manual_seed(seed)
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Material type codes
# ---------------------------------------------------------------------------
# Full mapping from Material_codes.xlsx
MATERIAL_CODES = {
    0: "Unspecified",
    1: "H/H Collection (Curbside)",
    2: "Bulk (Residential)",
    3: "Self-help Bulk",
    4: "Leaves (Not Recycling)",
    5: "Trees (Not Recycling)",
    6: "Street Dirt (MLP/MBD/WEP)",
    7: "Self Help (Community Cleanup)",
    8: "Fly Ash",
    9: "Residue",
    10: "Lot Cleaning",
    11: "Pest Control",
    12: "Lot Cleaning - Federal",
    14: "EZ Pack Collection",
    15: "HFC Collection",
    21: "Abandoned Cars / Junk",
    22: "Miscellaneous Recycling",
    24: "RO/RO (AFF) Collection",
    25: "RO/RO (AFF) Bulk",
    26: "RO/RO (AFF) Miscellaneous",
    27: "RO/RO (AFF) News/Mags/Corrug",
    28: "RO/RO (AFF) Metal Bulk",
    29: "RO/RO (AFF) Metal/Glass/Plastic",
    31: "Newspaper / Magazines / Corrugated",
    32: "News / Mags / Corrug (Intensive)",
    33: "Metal / Glass / Plastic",
    34: "Metal / Glass / Plastic (Intensive)",
    35: "Comingled Recyclables",
    36: "Mixed Waste",
    37: "Metal Bulk",
    38: "News / Mags / Corrug (EZ Pack)",
    39: "Metal / Glass / Plastic (EZ Pack)",
    40: "Pratt Paper",
    41: "Pratt Paper (EZ Pack)",
    45: "Residential Food Waste",
    46: "School Food Waste",
    47: "Residential Mixed Metals",
    48: "School Truck Paper",
    49: "School Truck Metal",
    50: "Brush (Compost)",
    51: "Green Market Food Waste (Recycling)",
    52: "Leaves (Recycling)",
    53: "Yard Waste (Recycling)",
    54: "Christmas Trees (Recycling)",
    55: "Wood Chips (Recycling)",
    56: "Tires (Recycling)",
    57: "Dirt",
    58: "Construction Debris",
    59: "Grass (Compost)",
    60: "ER Cleaning",
    61: "ERD Collection",
    62: "ER Hauling",
    63: "Emergency Collection Service",
    64: "ERD Paper",
    65: "ERD M/G/P",
    66: "Storm Debris",
    67: "ERD Down Trees",
    77: "Metal / Plastic",
    78: "School Truck Metal / Glass / Plastic",
    79: "School Truck Metal & Plastic",
    80: "Housing Authority Trucks",
    81: "School Trucks",
    82: "Goodwill Industries",
    83: "Public Buildings",
    84: "Special Coll (CRS/RI/GWI/PB/CC)",
    85: "Catholic Charities",
    86: "District Shop Load",
    87: "Baskets",
    88: "Tires (Non-Recycling)",
    90: "Millings",
    96: "Passover Refuse",
    97: "Passover Paper",
    98: "Passover Metal/Glass/Plastic",
}

# Primary curbside waste streams for analysis
PRIMARY_STREAMS = {1: "Refuse", 31: "Paper", 33: "MGP"}
PRIMARY_STREAM_CODES = list(PRIMARY_STREAMS.keys())

# Short names for columns
STREAM_COL_NAMES = {1: "tons_refuse", 31: "tons_paper", 33: "tons_mgp"}

# ---------------------------------------------------------------------------
# Columns to keep from raw CSV
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "Truck_ID",
    "Collection_Date",
    "Shift_ID",
    "District_Code",
    "Section_Code",
    "Route_Code",
    "Material_Type_Code",
    "Tons_Collected",
    "Net_Weight",
    "Dump_Time_Stamp",
]

RAW_DTYPES = {
    "Truck_ID": str,
    "Shift_ID": str,
    "District_Code": str,
    "Section_Code": str,
    "Route_Code": str,
    "Material_Type_Code": int,
    "Tons_Collected": float,
    "Net_Weight": float,
}

# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------
# Borough extraction from district/section codes
BOROUGH_PREFIXES = {
    "BK": "Brooklyn",
    "QE": "Queens",
    "QW": "Queens",
    "QN": "Queens",
    "QS": "Queens",
    "MN": "Manhattan",
    "BX": "Bronx",
    "SI": "Staten Island",
}

# NYC county FIPS codes for Census API queries
NYC_COUNTY_FIPS = {
    "36005": "Bronx",
    "36047": "Brooklyn",
    "36061": "Manhattan",
    "36081": "Queens",
    "36085": "Staten Island",
}

# Sections to drop — not represented in the DSNY shapefile
DROP_SECTIONS = [
    "AFFB02", "AFFBK", "AFFBX", "AFFM03", "AFFM10", "AFFMN", "AFFQN",
    "AFFS02", "AFFSI", "BKLC1", "BKLC2", "BKLC3", "BKLC4", "BXLC1",
    "FKA1", "MN012", "MN102", "MNLC1", "MNLC2", "OTHCLN", "OTHSCH",
    "QEB1", "QELC1", "QELC2", "SILC1", "SILC2",
]


def clean_district_code(code: str) -> str:
    """
    Normalize district/section codes to match DSNY shapefile naming.

    Ported from resources/ml-waste-master/waste.py:clean_district().
    Handles Queens (QS/QN→QE), Manhattan (M→MN), Bronx, Brooklyn
    (BKW→BKS, some BKE→BKN) remappings.
    """
    code = str(code).strip()
    if not code:
        return code

    prefix = code[:2]
    if prefix in ("QS", "QN"):
        return "QE" + code[2:]
    elif code[0] == "M" and prefix != "MN":
        return "MN" + code[2:]
    elif prefix == "BX":
        return code[:2] + code[-3:]
    elif prefix == "BK":
        sub = code[:3]
        if sub == "BKW":
            return "BKS" + code[-3:]
        elif sub == "BKE":
            bkn_sections = {
                "091", "092", "093", "161", "162",
                "171", "172", "173", "174", "175",
            }
            if code[-3:] in bkn_sections:
                return "BKN" + code[-3:]
            else:
                return "BKS" + code[-3:]
        else:
            return code
    else:
        return code


def extract_borough(district_code: str) -> str:
    """Extract borough name from a district code."""
    code = str(district_code).strip()
    for prefix, borough in BOROUGH_PREFIXES.items():
        if code.startswith(prefix):
            return borough
    return "Unknown"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with file + console handlers."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(LOGS_DIR / f"{name}.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger
