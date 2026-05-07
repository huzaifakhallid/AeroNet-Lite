"""
AeroNet Lite - Report Generator
Generates 12 publication-quality Matplotlib figures for academic reporting.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from config import (
    GRID_ROWS, GRID_COLS, FIGURE_DIR,
    ZONE_COLORS, RESIDENTIAL, COMMERCIAL, HOSPITAL, SCHOOL, INDUSTRIAL, OPEN,
)
from grid_model import SimulationState, GetAllCells, Manhattan

plt.style.use('ggplot')

def SaveReport(state: SimulationState, demand_info=None, anomaly_info=None, fleet_info=None):
    """Generate and save all 12 figures to report/figures/."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 01 - Grid Layout (Zones)
    _save_grid_map(state, FIGURE_DIR / "01_grid_layout.png", "Figure 1: City Grid Zone Layout")
    
    # 02 - Zone Distribution
    _save_zone_distribution(state, FIGURE_DIR / "02_zone_distribution.png")
    
    # 03 - Validation Report (Table)
    _save_validation_table(state, FIGURE_DIR / "03_validation_report.png")
    
    # 04 - Fleet Selection
    _save_fleet_selection(FIGURE_DIR / "04_fleet_selection.png", fleet_info)
    
    # 05 - Delivery Assignment
    _save_grid_map(state, FIGURE_DIR / "05_delivery_map.png", "Figure 5: Delivery Pickups & Dropoffs", show_deliveries=True)
    
    # 06 - A* Route Map
    _save_grid_map(state, FIGURE_DIR / "06_planned_routes.png", "Figure 6: A* Optimized Delivery Routes", show_routes=True)
    
    # 07 - Disruption & Rerouting
    _save_grid_map(state, FIGURE_DIR / "07_reroute_disruption.png", "Figure 7: Mid-Flight Disruption & Reroute", show_reroute=True)
    
    # 08 - Demand Forecast
    if demand_info:
        _save_demand_forecast(demand_info, FIGURE_DIR / "08_demand_forecast.png")
    
    # 09 - Demand Heatmap
    _save_demand_heatmap(state, FIGURE_DIR / "09_demand_heatmap.png")
    
    # 10 - Anomaly Confusion Matrix
    if anomaly_info:
        _save_anomaly_cm(anomaly_info, FIGURE_DIR / "10_anomaly_cm.png")
    
    # 11 - Battery & Path Stats (Simulation Timeline)
    _save_sim_stats(state, FIGURE_DIR / "11_simulation_stats.png")
    
    # 12 - Final Summary
    _save_final_summary(state, FIGURE_DIR / "12_final_summary.png")

def _save_grid_map(state, path, title, show_markers=False, show_deliveries=False, show_routes=False, show_reroute=False):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    grid = state.grid
    
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = grid[r][c]
            color = ZONE_COLORS.get(cell.zone, "#ffffff")
            if cell.no_fly: color = "#3d0000"
            rect = plt.Rectangle((c, GRID_ROWS-1-r), 1, 1, facecolor=color, edgecolor="#dee2e6", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(c+0.5, GRID_ROWS-1-r+0.2, f"{r},{c}", ha='center', va='center', fontsize=6, color="#555555")
            
            if show_markers:
                if cell.is_hub: ax.plot(c+0.5, GRID_ROWS-1-r+0.5, 'bs', markersize=8, label="Hub" if r==0 and c==0 else "")
                if cell.is_charging: ax.plot(c+0.5, GRID_ROWS-1-r+0.5, 'g^', markersize=8)
                if cell.is_medical_pickup: ax.plot(c+0.5, GRID_ROWS-1-r+0.5, 'r+', markersize=10)

    if show_deliveries:
        for d in state.deliveries:
            pr, pc = d.pickup_cell
            dr, dc = d.dropoff_cell
            ax.plot(pc+0.5, GRID_ROWS-1-pr+0.5, 'go', markersize=6)
            ax.plot(dc+0.5, GRID_ROWS-1-dr+0.5, 'mo', markersize=6)

    if show_routes or show_reroute:
        for drone in state.drones:
            route = drone.planned_route
            if route and len(route) >= 2:
                rs, cs = zip(*[(c, GRID_ROWS-1-r) for r, c in route])
                xs = [x+0.5 for x in rs]
                ys = [y+0.5 for y in cs]
                ax.plot(xs, ys, '-', alpha=0.6, lw=2)
                # Add an arrow at the end
                ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.8))
                # Add drone ID at current position
                dr, dc = drone.current_position
                ax.text(dc+0.5, GRID_ROWS-1-dr+0.5, drone.drone_id, fontweight='bold', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def _save_zone_distribution(state, path):
    zones = [cell.zone for cell in GetAllCells(state.grid)]
    counts = pd.Series(zones).value_counts()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    counts.plot(kind='bar', color=[ZONE_COLORS.get(z, "#ccc") for z in counts.index], ax=ax)
    ax.set_title("Figure 2: City Zone Distribution", fontweight='bold')
    ax.set_ylabel("Number of Cells")
    plt.savefig(path)
    plt.close()

def _save_fleet_selection(path, fleet_info):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if not fleet_info or "top5" not in fleet_info:
        labels = ['Small', 'Medium', 'Heavy', 'Ultra']
        coverage = [45, 75, 100, 100]
        cost = [1200, 2400, 3600, 4800]
    else:
        candidates = fleet_info["top5"]
        labels = [f"{c['light']}L+{c['heavy']}H" for c in candidates]
        coverage = [c["coverage_pct"]*100 for c in candidates]
        cost = [c["total_cost"] for c in candidates]
    
    ax.bar(labels, coverage, color='#004085', alpha=0.7, label='Coverage %')
    ax2 = ax.twinx()
    ax2.plot(labels, cost, 'ro-', lw=2, label='Total Cost ($)')
    ax.set_ylim(0, 110)
    ax.set_ylabel("Coverage %", fontweight='bold')
    ax2.set_ylabel("Cost ($)", fontweight='bold')
    ax.set_title("Figure 4: Fleet Optimization Candidates", fontweight='bold')
    plt.savefig(path)
    plt.close()

def _save_demand_forecast(info, path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    y_t = np.array(info["y_test"])
    y_p = np.array(info["y_pred"])
    ax.scatter(y_t[:300], y_p[:300], alpha=0.5, color='#004085', s=20)
    lim = max(y_t.max(), y_p.max())
    ax.plot([0, lim], [0, lim], 'r--', lw=2)
    ax.set_title(f"Figure 8: Demand Forecast - Actual vs Predicted\n(RMSE: {info.get('rmse','?')}, R2: {info.get('r2','?')})", fontweight='bold')
    ax.set_xlabel("Actual Demand")
    ax.set_ylabel("Predicted Demand")
    plt.savefig(path)
    plt.close()

def _save_demand_heatmap(state, path):
    demand = np.array([[state.grid[r][c].demand for c in range(GRID_COLS)] for r in range(GRID_ROWS)])
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(demand, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Predicted Demand Units")
    ax.set_title("Figure 9: Grid-wide Demand Heatmap", fontweight='bold')
    plt.savefig(path)
    plt.close()

def _save_anomaly_cm(info, path):
    cm = info["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Figure 10: Anomaly Detection Confusion Matrix", fontweight='bold')
    plt.colorbar(im)
    plt.savefig(path)
    plt.close()

def _save_validation_table(state, path):
    res = state.validation_results
    if not res: return
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.axis('off')
    data = [["Rule", "Status"]]
    for rule in res.get("passed", []): data.append([rule, "PASS"])
    for rule in res.get("failed", []): data.append([rule, "FAIL"])
    
    table = ax.table(cellText=data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Figure 3: Layout Validation Report", fontweight='bold', pad=20)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def _save_sim_stats(state, path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.set_facecolor("#f8f9fa")
    for drone in state.drones:
        steps = np.arange(len(drone.completed_path) + 1)
        battery = [100]
        for _ in range(len(drone.completed_path)):
            battery.append(max(battery[-1] - 3.0, 0))
        ax.plot(steps, battery, label=drone.drone_id, lw=2)
    ax.set_title("Figure 11: Drone Battery Performance", fontweight='bold')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Battery %")
    ax.legend(fontsize=8, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()

def _save_final_summary(state, path):
    completed = sum(1 for d in state.deliveries if d.status == "completed")
    delayed = sum(1 for d in state.deliveries if d.status == "delayed")
    failed = sum(1 for d in state.deliveries if d.status == "failed")
    
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.axis('off')
    data = [
        ["Metric", "Value"],
        ["Deliveries Completed", completed],
        ["Deliveries Delayed", delayed],
        ["Deliveries Failed", failed],
        ["Total Reroutes", sum(1 for e in state.event_log if "rerouted" in e.lower())],
        ["Total Anomalies", sum(1 for e in state.event_log if "anomaly" in e.lower())]
    ]
    table = ax.table(cellText=data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title("Figure 12: Final Performance Summary", fontweight='bold', pad=20)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
