"""
AeroNet Lite - Central Configuration
# All project-wide constants, paths, and tunable parameters.
# Updated for v2.0 dashboard integration.
"""

from pathlib import Path
import random
import numpy as np

# Project root 
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Grid
GRID_ROWS = 10
GRID_COLS = 10

# Reproducibility 
RANDOM_SEED = 42

# Fleet economics
DEFAULT_BUDGET = 8000
LIGHT_DRONE_COST = 1000
HEAVY_DRONE_COST = 1800
LIGHT_DRONE_PAYLOAD = 2.0
HEAVY_DRONE_PAYLOAD = 5.0
LIGHT_DRONE_RANGE = 12
HEAVY_DRONE_RANGE = 20

# A* movement costs
NORMAL_MOVE_COST = 1.0
COMMERCIAL_MOVE_COST = 0.8

# Simulation
SIMULATION_STEPS = 20
NUM_DELIVERIES = 8          # default deliveries to generate
DEMO_NUM_DELIVERIES = 5     # fewer deliveries for dashboard demo
DRONE_STEPS_PER_TICK = 1    # one logical cell per tick; controller interpolates motion frames

# Zone names
RESIDENTIAL = "Residential"
COMMERCIAL  = "Commercial"
HOSPITAL    = "Hospital"
SCHOOL      = "School"
INDUSTRIAL  = "Industrial"
OPEN        = "Open"

# Zone colours (used by Dashboard and Matplotlib)
ZONE_COLORS = {
    RESIDENTIAL: "#AED6F1",   # light blue
    COMMERCIAL:  "#F5B041",   # orange
    HOSPITAL:    "#F1948A",   # pink-red
    SCHOOL:      "#82E0AA",   # light green
    INDUSTRIAL:  "#B0A090",   # brownish grey
    OPEN:        "#D5DBDB",   # light grey
}

# Data paths 
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"

TRAIN_CSV             = RAW_DIR / "bike_sharing_demand" / "train.csv"
AMAZON_CSV            = RAW_DIR / "amazon_delivery" / "amazon_delivery.csv"
DENSITY_CSV           = RAW_DIR / "us_city_pop_density" / "uscitypopdensity.csv"
TELEMETRY_CSV         = RAW_DIR / "supplemental_drone_telemetry_data" / "Supplemental Drone Telemetry Data - Drone Operations Log.csv"

PROCESSED_DEMAND_CSV  = PROCESSED_DIR / "processed_demand.csv"
PROCESSED_DENSITY_CSV = PROCESSED_DIR / "processed_density.csv"
PROCESSED_ANOMALY_CSV = PROCESSED_DIR / "processed_anomaly.csv"

# Output paths 
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
LOG_DIR     = OUTPUT_DIR / "logs"
MODEL_DIR   = OUTPUT_DIR / "models"
TABLE_DIR   = OUTPUT_DIR / "tables"
FIGURE_DIR  = PROJECT_ROOT / "report" / "figures"

# Ensure directories exist
for _d in (PROCESSED_DIR, LOG_DIR, MODEL_DIR, TABLE_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def SeedEverything(seed: int = RANDOM_SEED) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
