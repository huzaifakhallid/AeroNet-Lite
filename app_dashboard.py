#!/usr/bin/env python
"""
AeroNet Lite - app_dashboard.py
Opens the unified dashboard and runs the live 20-step simulation.
Usage:  python app_dashboard.py
"""

import matplotlib
matplotlib.use("TkAgg")

import sys
import os
import tkinter as tk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import SeedEverything
from utils import SetupLogging, SafeRun, InitializeSimulation
from ml_demand import LoadDemandData, PreprocessDemandData, TrainDemandModels
from ml_anomaly import LoadDroneTelemetryData, TrainAnomalyModels
from visualization_dashboard import AeroNetDashboard
from dashboard_controller import SimulationController


def main() -> None:
    SeedEverything()
    SetupLogging()

    print("AeroNet Lite - Dashboard")
    print("Training models (please wait) ...")

    # Train demand model
    demand_info = None
    raw = SafeRun(LoadDemandData, label="demand")
    if raw is not None:
        processed = PreprocessDemandData(raw)
        demand_info = SafeRun(TrainDemandModels, processed, label="demand train")

    # Train anomaly model
    anomaly_info = None
    telem = SafeRun(LoadDroneTelemetryData, label="telemetry")
    if telem is not None:
        anomaly_info = SafeRun(TrainAnomalyModels, telem, label="anomaly train")

    print("Launching dashboard ...")

    # Initialize simulation
    state = InitializeSimulation()

    # Build GUI
    root = tk.Tk()
    dashboard = AeroNetDashboard(root, state, demand_info, anomaly_info)

    # Start simulation controller
    controller = SimulationController(
        root, state, dashboard,
        demand_model_info=demand_info,
        anomaly_model_info=anomaly_info,
    )
    
    # Connect dashboard Run button to the controller
    dashboard.on_run_callback = controller.ResetAndRun
    
    # Initial dashboard update
    dashboard.UpdateDashboard(state)
    if dashboard._event_log:
        dashboard._event_log.insert("end", "Welcome to AeroNet Lite Dashboard.\n", "normal")
        dashboard._event_log.insert("end", "Click 'Run Simulation' in the header to begin.\n", "normal")

    # Enter main loop
    root.mainloop()
    print("Dashboard closed.")


if __name__ == "__main__":
    main()
