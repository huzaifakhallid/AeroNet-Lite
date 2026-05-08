# AeroNet Lite

**Autonomous Drone Delivery Simulation with CSP, Fleet Planning, A\* Routing, Real-Time Replanning, Demand Forecasting, and Anomaly Detection.**

---

## Overview

AeroNet Lite is a Python-based simulation system that models autonomous drone delivery operations on a simplified **10×10 city grid**. It integrates multiple AI/ML techniques into a single, modular dashboard built with **Tkinter** and **Matplotlib**:

| Technique | Purpose |
|-----------|---------|
| **CSP-style constraint validation** | Verify city-grid layout rules |
| **Brute-force optimisation** | Select drone fleet under budget |
| **A\* search** | Plan optimal delivery routes avoiding no-fly zones |
| **Real-time replanning** | Reroute drones when new no-fly zones appear mid-flight |
| **Linear / Random Forest Regression** | Forecast delivery demand from bike-sharing data |
| **Decision Tree / Random Forest Classification** | Detect drone anomalies from telemetry logs |

The system features a **unified dynamic dashboard** that handles model training, grid visualisation, drone movement, and real-time performance tracking in a single responsive window.

---

## Features

- **Unified Dashboard**: One command starts the entire system including model training and live simulation.
- **Dynamic 10x10 Grid**: Interactive visualisation of zones, facility markers (Hubs, Charging, Medical), and no-fly zones.
- **Animated Drones**: Drones visibly move cell-by-cell along planned A* routes.
- **Embedded ML Analytics**: 
  - Live **Demand Forecast** scatter plot (Actual vs Predicted).
  - **Demand Heatmap** updating live on the city grid.
  - **Anomaly Confusion Matrix** showing model performance.
- **Event-Driven Architecture**: Non-blocking simulation loop ensures a responsive GUI.
- **Real-Time Logs**: Color-coded event log tracking every simulation step, reroute, and alert.
- **CSP Layout Validator**: 4 constraint rules with suggested fixes.
- **Budget-Constrained Fleet Selector**: Brute-force search for optimal (light vs heavy) drone mix.

---

## Datasets

| File | Source | Usage |
|------|--------|-------|
| `data/raw/bike_sharing_demand/train.csv` | Kaggle Bike Sharing Demand | Demand forecasting target: `count` |
| `data/raw/us_city_pop_density/uscitypopdensity.csv` | US City Population Density | Grid cell density initialisation |
| `data/raw/supplemental_drone_telemetry_data/…csv` | Supplemental Drone Telemetry | Anomaly detection with rule-based labels |
| `data/raw/amazon_delivery/amazon_delivery.csv` | Amazon Delivery Dataset | Optional delivery category sampling |

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ required. All dependencies are standard PyPI packages.

---

## How to Run

### Unified Dashboard (Recommended)
This opens the complete dynamic dashboard with all visuals and simulation steps.
```bash
python app_dashboard.py
```

### Run tests
```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
AeroNet-Lite/
├── data/raw/              # Raw CSV datasets
├── data/processed/        # Cleaned/engineered CSV files
├── outputs/               # Logs, models, tables
├── src/
│   ├── config.py          # All constants & paths
│   ├── grid_model.py      # Cell, Drone, Delivery, SimulationState
│   ├── data_loader.py     # CSV loading utilities
│   ├── layout_validator.py# CSP constraint checker
│   ├── fleet_selector.py  # Brute-force fleet optimisation
│   ├── delivery_generator.py
│   ├── delivery_assigner.py
│   ├── astar_planner.py   # A* pathfinding
│   ├── disruption_handler.py
│   ├── ml_demand.py       # Demand regression pipeline
│   ├── ml_anomaly.py      # Anomaly classification pipeline
│   ├── visualization_dashboard.py # Unified Tkinter GUI
│   ├── dashboard_controller.py    # Event-driven simulation driver
│   └── utils.py           # Shared helpers (logging, simulation init)
├── tests/                 # pytest test suite
├── app_dashboard.py       # Main entry point
├── requirements.txt
└── README.md
```

---

## AI Techniques Explained (Viva Notes)

### Why Tkinter & Matplotlib Integration?
The dashboard uses an event-driven loop (`root.after`) to remain responsive while rendering live movement. Matplotlib is embedded directly using `FigureCanvasTkAgg` to provide professional ML analytics alongside the simulation.

### Why A\* search?
A\* is optimal given an admissible heuristic. We use `0.8 × Manhattan distance` because the minimum step cost is 0.8 (commercial cells), ensuring admissibility and guaranteeing shortest paths.

### Why brute-force fleet selection?
With only ~40 valid fleet combinations under the budget, brute-force is exhaustive, explainable, and guaranteed optimal.

### Why Bike Sharing Demand as a delivery proxy?
Real drone delivery demand data is scarce. Bike-sharing demand captures temporal patterns (hour, season, weather) that are analogous to delivery demand patterns.

### Why rule-based anomaly labels?
The telemetry dataset lacks ground-truth labels. We apply domain-knowledge rules (overweight payloads, low battery, GPS drift) to create labels for training classification models.

---

## License

Academic project – BSDS Semester Project.
