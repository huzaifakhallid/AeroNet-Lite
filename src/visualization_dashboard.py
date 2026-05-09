"""
AeroNet Lite - Tkinter Dashboard
Single-window GUI with embedded matplotlib charts, live grid, and event log.
"""

from __future__ import annotations

import tkinter as tk
import numpy as np
from typing import Optional

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import (
    GRID_ROWS, GRID_COLS,
    ZONE_COLORS, RESIDENTIAL, COMMERCIAL, HOSPITAL, SCHOOL, INDUSTRIAL, OPEN,
)
from grid_model import Cell, Drone, Delivery, SimulationState, GetAllCells

CS = 52
GRID_PAD = 30
CANVAS_W = CS * GRID_COLS + GRID_PAD * 2
CANVAS_H = CS * GRID_ROWS + GRID_PAD * 2

BG = "#f8f9fa"
BG2 = "#ffffff"
ACCENT = "#e9ecef"
CYAN = "#004085"
TEXT = "#000000"
WHITE = "#ffffff"
GREEN = "#155724"
RED = "#721c24"
ORANGE = "#856404"


class AeroNetDashboard:

    def __init__(self, root: tk.Tk, state: SimulationState,
                 demand_model_info: dict | None = None,
                 anomaly_model_info: dict | None = None,
                 on_run_callback: Optional[callable] = None):
        self.root = root
        self.state = state
        self.demand_model_info = demand_model_info
        self.anomaly_model_info = anomaly_model_info
        self.on_run_callback = on_run_callback

        self.root.title("AeroNet Lite - Dashboard")
        self.root.geometry("1280x820")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._grid_canvas: Optional[tk.Canvas] = None
        self._event_log: Optional[tk.Text] = None
        self._fleet_text: Optional[tk.Text] = None
        self._step_label: Optional[tk.Label] = None
        self._alert_frame: Optional[tk.Frame] = None
        self._alert_label: Optional[tk.Label] = None
        self._demand_stats_frame: Optional[tk.Frame] = None
        self._anomaly_stats_frame: Optional[tk.Frame] = None

        self._demand_fig: Optional[Figure] = None
        self._demand_ax = None
        self._demand_canvas = None
        self._heatmap_fig: Optional[Figure] = None
        self._heatmap_ax = None
        self._heatmap_canvas = None
        self._cm_fig: Optional[Figure] = None
        self._cm_ax = None
        self._cm_canvas = None

        self.BuildLayout()

    def _handle_run_click(self):
        if self.on_run_callback:
            self._run_btn.config(state="disabled", bg=ACCENT, text="Running...")
            self.on_run_callback()

    def EnableRunButton(self):
        if self._run_btn:
            self._run_btn.config(state="normal", bg=GREEN, text="▶ Run Simulation")

    # ------------------------------------------------------------------ helpers
    def _cell_bbox(self, row, col):
        x1 = GRID_PAD + col * CS
        y1 = GRID_PAD + row * CS
        return x1, y1, x1 + CS, y1 + CS

    def _cell_center(self, row, col):
        x1, y1, x2, y2 = self._cell_bbox(row, col)
        return (x1 + x2) / 2, (y1 + y2) / 2

    # ----------------------------------------------------------- icon drawing
    def _draw_hub(self, cx, cy, s=10):
        c = self._grid_canvas
        c.create_rectangle(cx-s, cy-s, cx+s, cy+s,
                           fill="#1565c0", outline="#42a5f5", width=2, tags="markers")
        c.create_text(cx, cy, text="H", fill="white",
                      font=("Arial", 8, "bold"), tags="markers")

    def _draw_charging(self, cx, cy, s=8):
        pts = [cx, cy-s, cx-s, cy+s, cx+s, cy+s]
        c = self._grid_canvas
        c.create_polygon(pts, fill="#00897b", outline="#4db6ac", width=2, tags="markers")
        c.create_text(cx, cy+2, text="C", fill="white",
                      font=("Arial", 6, "bold"), tags="markers")

    def _draw_medical(self, cx, cy, arm=8):
        c = self._grid_canvas
        c.create_line(cx-arm, cy, cx+arm, cy, fill="#e53935", width=3, tags="markers")
        c.create_line(cx, cy-arm, cx, cy+arm, fill="#e53935", width=3, tags="markers")

    def _draw_nofly(self, cx, cy, arm=10):
        c = self._grid_canvas
        c.create_line(cx-arm, cy-arm, cx+arm, cy+arm, fill=RED, width=3, tags="markers")
        c.create_line(cx-arm, cy+arm, cx+arm, cy-arm, fill=RED, width=3, tags="markers")

    def _draw_drone(self, cx, cy, dtype, color):
        c = self._grid_canvas
        s = 10
        if dtype == "light":
            pts = [cx, cy-s, cx-s, cy+s, cx+s, cy+s]
            c.create_polygon(pts, fill=color, outline="white", width=1, tags="drones")
        else:
            c.create_rectangle(cx-s, cy-s, cx+s, cy+s,
                               fill=color, outline="white", width=1, tags="drones")

    def _draw_pickup(self, cx, cy):
        c = self._grid_canvas
        c.create_oval(cx-6, cy-6, cx+6, cy+6, fill=GREEN, outline="#33691e",
                      width=2, tags="deliveries")
        c.create_text(cx, cy, text="P", fill="black",
                      font=("Arial", 7, "bold"), tags="deliveries")

    def _draw_dropoff(self, cx, cy):
        c = self._grid_canvas
        s = 8
        pts = [cx, cy-s, cx+s, cy, cx, cy+s, cx-s, cy]
        c.create_polygon(pts, fill="#e040fb", outline="#6a1b9a", width=2, tags="deliveries")
        c.create_text(cx, cy, text="D", fill="white",
                      font=("Arial", 6, "bold"), tags="deliveries")

    # --------------------------------------------------------- layout building
    def BuildLayout(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG, height=40)
        hdr.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(hdr, text="AeroNet Lite", font=("Segoe UI", 16, "bold"),
                 fg=CYAN, bg=BG).pack(side="left")

        self._run_btn = tk.Button(hdr, text="▶ Run Simulation", font=("Segoe UI", 10, "bold"),
                                  bg=GREEN, fg="white", activebackground="#0b3d1b",
                                  activeforeground="white", relief="flat", padx=15,
                                  command=self._handle_run_click)
        self._run_btn.pack(side="left", padx=30)

        self._step_label = tk.Label(hdr, text="Step: 0 / 20",
                                    font=("Segoe UI", 13), fg=TEXT, bg=BG)
        self._step_label.pack(side="right")

        # Main area
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=8, pady=4)

        # Left: grid
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="y", padx=(0, 6))
        self.BuildGridPanel(left)

        # Right: panels
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self.BuildRightPanels(right)

        # Alert banner (hidden initially)
        self._alert_frame = tk.Frame(self.root, bg="#f8d7da", height=0)
        self._alert_label = tk.Label(self._alert_frame, text="",
                                     font=("Segoe UI", 11, "bold"),
                                     fg="#721c24", bg="#f8d7da")

        # Event log
        self.BuildEventLogPanel(self.root)

        # Initial model stats
        if self.demand_model_info or self.anomaly_model_info:
            self.UpdateModelStats(self.demand_model_info, self.anomaly_model_info)

    def BuildGridPanel(self, parent):
        self._grid_canvas = tk.Canvas(parent, width=CANVAS_W, height=CANVAS_H,
                                      bg=BG2, highlightthickness=0)
        self._grid_canvas.pack()

        # Legend
        y = CANVAS_H - 18
        x = GRID_PAD
        items = [("#1565c0", "Hub"), ("#00897b", "Charging"),
                 ("#e53935", "Medical"), (RED, "No-Fly"),
                 (GREEN, "Pickup"), ("#e040fb", "Dropoff")]
        self._grid_canvas.create_text(x, y, text="Legend:", fill=TEXT,
                                      anchor="w", font=("Segoe UI", 8, "bold"))
        x += 55
        for color, label in items:
            self._grid_canvas.create_rectangle(x, y-5, x+10, y+5, fill=color, outline="")
            self._grid_canvas.create_text(x+15, y, text=label, fill=TEXT,
                                          anchor="w", font=("Segoe UI", 7))
            x += 72

    def BuildRightPanels(self, parent):
        # Row 0: model stats side by side
        stats_row = tk.Frame(parent, bg=BG)
        stats_row.pack(fill="x", pady=(0, 4))

        self._demand_stats_frame = self._make_stats_card(
            stats_row, "Demand Forecasting (Regression)", "#004085")
        self._demand_stats_frame.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self._anomaly_stats_frame = self._make_stats_card(
            stats_row, "Anomaly Detection (Classification)", "#721c24")
        self._anomaly_stats_frame.pack(side="left", fill="x", expand=True, padx=(3, 0))

        # Row 1: demand forecast + confusion matrix
        chart_row1 = tk.Frame(parent, bg=BG)
        chart_row1.pack(fill="both", expand=True, pady=3)

        self._demand_fig = Figure(figsize=(4.5, 1.4), dpi=150, facecolor=BG2)
        self._demand_ax = self._demand_fig.add_subplot(111)
        self._demand_canvas = self._embed_fig(chart_row1, self._demand_fig, side="left")

        self._cm_fig = Figure(figsize=(2.5, 1.4), dpi=150, facecolor=BG2)
        self._cm_ax = self._cm_fig.add_subplot(111)
        self._cm_canvas = self._embed_fig(chart_row1, self._cm_fig, side="left")

        # Row 2: heatmap + fleet/delivery
        chart_row2 = tk.Frame(parent, bg=BG)
        chart_row2.pack(fill="both", expand=True, pady=3)

        self._heatmap_fig = Figure(figsize=(3.4, 3.4), dpi=150, facecolor=BG2)
        self._heatmap_ax = self._heatmap_fig.add_subplot(111)
        self._heatmap_canvas = self._embed_fig(chart_row2, self._heatmap_fig, side="left")

        fleet_frame = tk.Frame(chart_row2, bg=BG2, highlightbackground=ACCENT,
                               highlightthickness=1)
        fleet_frame.pack(side="left", fill="both", expand=True, padx=3)
        tk.Label(fleet_frame, text="Fleet & Deliveries", font=("Segoe UI", 10, "bold"),
                 fg=CYAN, bg=BG2).pack(anchor="w", padx=8, pady=(4, 0))
        self._fleet_text = tk.Text(fleet_frame, bg=BG2, fg=TEXT,
                                   font=("Consolas", 8), wrap="word",
                                   relief="flat", bd=0, padx=6, pady=4, height=10)
        self._fleet_text.pack(fill="both", expand=True)

    def _make_stats_card(self, parent, title, accent):
        frame = tk.Frame(parent, bg=BG2, highlightbackground=accent,
                         highlightthickness=2, bd=0)
        tk.Label(frame, text=title, font=("Segoe UI", 9, "bold"),
                 fg=accent, bg=BG2).pack(anchor="w", padx=8, pady=(6, 2))
        inner = tk.Frame(frame, bg=BG2)
        inner.pack(fill="x", padx=8, pady=(0, 6))
        inner._metric_frame = inner
        return frame

    def _embed_fig(self, parent, fig, side="left"):
        frame = tk.Frame(parent, bg=BG2, highlightbackground=ACCENT,
                         highlightthickness=1)
        frame.pack(side=side, fill="both", expand=True, padx=3)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    def BuildEventLogPanel(self, parent):
        log_frame = tk.Frame(parent, bg=BG)
        log_frame.pack(fill="x", padx=8, pady=(2, 6))
        tk.Label(log_frame, text="Event Log", font=("Segoe UI", 9, "bold"),
                 fg=CYAN, bg=BG).pack(anchor="w")
        self._event_log = tk.Text(log_frame, bg=BG2, fg=TEXT,
                                  font=("Consolas", 8, "bold"), wrap="word",
                                  relief="flat", bd=0, padx=10, pady=6, height=7)
        self._event_log.pack(fill="x")
        self._event_log.tag_configure("error", foreground=RED)
        self._event_log.tag_configure("warn", foreground=ORANGE)

    # ------------------------------------------------------------ grid drawing
    def DrawGrid(self):
        if self._grid_canvas is None:
            return
        self._grid_canvas.delete("cells")
        for cell in GetAllCells(self.state.grid):
            x1, y1, x2, y2 = self._cell_bbox(cell.row, cell.col)
            color = ZONE_COLORS.get(cell.zone, "#555555")
            if cell.no_fly:
                color = "#3d0000"
            self._grid_canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                                               outline="#dee2e6", width=1, tags="cells")
            self._grid_canvas.create_text((x1+x2)/2, (y1+y2)/2 + 16,
                                          text=f"{cell.row},{cell.col}",
                                          fill=TEXT, font=("Arial", 6, "bold"), tags="cells")

    def DrawFacilityMarkers(self):
        if self._grid_canvas is None:
            return
        self._grid_canvas.delete("markers")
        for cell in GetAllCells(self.state.grid):
            cx, cy = self._cell_center(cell.row, cell.col)
            if cell.is_hub:
                self._draw_hub(cx, cy)
            if cell.is_charging:
                self._draw_charging(cx, cy - 2)
            if cell.is_medical_pickup:
                self._draw_medical(cx, cy)
            if cell.no_fly:
                self._draw_nofly(cx, cy)

    def DrawDeliveries(self):
        if self._grid_canvas is None:
            return
        self._grid_canvas.delete("deliveries")
        for d in self.state.deliveries:
            cx, cy = self._cell_center(*d.pickup_cell)
            self._draw_pickup(cx - 12, cy - 12)
            cx, cy = self._cell_center(*d.dropoff_cell)
            self._draw_dropoff(cx + 12, cy - 12)

    def DrawDrones(self):
        if self._grid_canvas is None:
            return
        self._grid_canvas.delete("drones")
        for drone in self.state.drones:
            cx, cy = self._cell_center(*drone.current_position)
            color = "#00e5ff" if drone.drone_type == "light" else "#ff9100"
            if drone.anomaly_status != "normal":
                color = RED
            self._draw_drone(cx, cy, drone.drone_type, color)
            self._grid_canvas.create_text(cx, cy - 14, text=drone.drone_id,
                                          fill=TEXT, font=("Consolas", 7), tags="drones")

    def DrawRoutes(self, routes=None, color="#00ff00"):
        if self._grid_canvas is None:
            return
        if routes is None:
            routes = []
            for drone in self.state.drones:
                if drone.status in ("en-route", "rerouted"):
                    remaining = drone.planned_route[drone.route_step_index:]
                    if len(remaining) >= 2:
                        routes.append(remaining)
        for route in routes:
            if len(route) < 2:
                continue
            coords = []
            for r, c in route:
                cx, cy = self._cell_center(r, c)
                coords.extend([cx, cy])
            self._grid_canvas.create_line(*coords, fill=color, width=2,
                                          arrow="last", arrowshape=(8, 10, 4),
                                          smooth=True, tags="routes")

    def ClearRoutes(self):
        if self._grid_canvas:
            self._grid_canvas.delete("routes")

    def DrawNoFlyCells(self):
        self.DrawGrid()
        self.DrawFacilityMarkers()

    # --------------------------------------------------- high-level update
    def UpdateDashboard(self, state: SimulationState):
        self.state = state
        if self._step_label:
            self._step_label.config(text=f"Step: {state.current_step} / 20")
        self.DrawGrid()
        self.DrawFacilityMarkers()
        self.DrawDeliveries()
        self.DrawDrones()
        self.ClearRoutes()
        self.DrawRoutes()
        self.UpdateEventLog(state)
        self.UpdateFleetSummary(state)
        self.root.update_idletasks()

    def UpdateModelStats(self, demand_info, anomaly_info):
        self.demand_model_info = demand_info
        self.anomaly_model_info = anomaly_info
        self._fill_stats_card(self._demand_stats_frame, demand_info,
                              ["mae", "rmse", "r2"], "#004085")
        self._fill_stats_card(self._anomaly_stats_frame, anomaly_info,
                              ["accuracy", "precision", "recall", "f1"], "#721c24")

        if demand_info and "y_test" in demand_info and "y_pred" in demand_info:
            self.UpdateDemandForecastPlot(
                demand_info["y_test"], demand_info["y_pred"], demand_info)
        if demand_info:
            self.UpdateDemandHeatmap(self.state.grid)
        if anomaly_info and "confusion_matrix" in anomaly_info:
            self.UpdateAnomalyConfusionMatrix(
                anomaly_info["confusion_matrix"],
                anomaly_info.get("class_names", []))

    def _fill_stats_card(self, card_frame, info, keys, accent):
        if card_frame is None or info is None:
            return
        # Find the inner metric frame
        children = card_frame.winfo_children()
        inner = children[1] if len(children) > 1 else None
        if inner is None:
            return
        for w in inner.winfo_children():
            w.destroy()

        if "best_name" in info:
            tk.Label(inner, text=f"Best: {info['best_name']}",
                     font=("Consolas", 8, "bold"), fg=TEXT, bg=BG2).pack(anchor="w")
        row = tk.Frame(inner, bg=BG2)
        row.pack(fill="x")
        for k in keys:
            if k in info:
                val = info[k]
                display = f"{val:.4f}" if isinstance(val, float) and val < 10 else str(val)
                cell = tk.Frame(row, bg=ACCENT, bd=0)
                cell.pack(side="left", padx=2, pady=2)
                tk.Label(cell, text=display, font=("Segoe UI", 12, "bold"),
                         fg=TEXT, bg=ACCENT).pack(padx=8, pady=(4, 0))
                tk.Label(cell, text=k.upper(), font=("Segoe UI", 7, "bold"),
                         fg=TEXT, bg=ACCENT).pack(padx=8, pady=(0, 4))

    # ----------------------------------------------------- matplotlib plots
    def UpdateDemandForecastPlot(self, y_true, y_pred, metrics):
        ax = self._demand_ax
        if ax is None:
            return
        ax.clear()
        ax.set_facecolor(BG)
        y_t = np.array(y_true)
        y_p = np.array(y_pred)
        ax.scatter(y_t[:200], y_p[:200], alpha=0.5, s=10, color=CYAN)
        lim = max(np.max(y_t), np.max(y_p)) * 1.05
        ax.plot([0, lim], [0, lim], "--", color=RED, lw=1.5)
        ax.set_xlabel("Actual", color=TEXT, fontsize=8, weight="bold")
        ax.set_ylabel("Predicted", color=TEXT, fontsize=8, weight="bold")
        rmse = metrics.get("rmse", "?")
        r2 = metrics.get("r2", "?")
        ax.set_title(f"Actual vs Predicted (RMSE={rmse}, R2={r2})",
                     color=TEXT, fontsize=9, weight="bold")
        ax.tick_params(colors=TEXT, labelsize=7)
        self._demand_fig.tight_layout(pad=1.0)
        self._demand_canvas.draw_idle()

    def UpdateDemandHeatmap(self, grid):
        ax = self._heatmap_ax
        if ax is None:
            return
        ax.clear()
        ax.set_facecolor(BG)
        demand = np.array([[grid[r][c].demand for c in range(GRID_COLS)]
                           for r in range(GRID_ROWS)])
        im = ax.imshow(demand, cmap="YlOrRd", origin="upper", interpolation="nearest")
        
        # Add proper 10x10 grid
        ax.set_xticks(np.arange(-0.5, GRID_COLS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, GRID_ROWS, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
        ax.grid(which="major", visible=False)  # Remove the default grid lines passing through the middle
        
        # Set major ticks for labels
        ax.set_xticks(np.arange(0, GRID_COLS, 1))
        ax.set_yticks(np.arange(0, GRID_ROWS, 1))
        ax.set_xticklabels(np.arange(0, GRID_COLS))
        ax.set_yticklabels(np.arange(0, GRID_ROWS))
        
        # Add demand values
        demand_max = demand.max()
        demand_min = demand.min()
        threshold = demand_min + (demand_max - demand_min) / 2.0
        
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                text_color = "white" if demand[r, c] > threshold else "black"
                ax.text(c, r, f"{demand[r,c]:.1f}", ha="center", va="center",
                        fontsize=6, color=text_color, weight="bold")
        ax.set_title("Demand Heatmap", color=TEXT, fontsize=9, weight="bold")
        ax.tick_params(colors=TEXT, labelsize=7)
        self._heatmap_fig.tight_layout(pad=0.5)
        self._heatmap_canvas.draw_idle()

    def UpdateAnomalyConfusionMatrix(self, cm, class_names):
        ax = self._cm_ax
        if ax is None:
            return
        ax.clear()
        ax.set_facecolor(BG)
        ax.imshow(cm, cmap="Blues")
        n = len(class_names)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_names = [name[:12] for name in class_names]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6, color=TEXT)
        ax.set_yticklabels(short_names, fontsize=6, color=TEXT)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else TEXT, fontsize=8, weight="bold")
        ax.grid(False)
        ax.set_title("Confusion Matrix", color=TEXT, fontsize=9, weight="bold")
        self._cm_fig.tight_layout(pad=0.2)
        self._cm_canvas.draw_idle()

    # --------------------------------------------------- fleet & delivery
    def UpdateFleetSummary(self, state):
        if self._fleet_text is None:
            return
        self._fleet_text.delete("1.0", "end")
        t = self._fleet_text
        if state.drones:
            for d in state.drones:
                color_tag = "normal"
                if d.anomaly_status != "normal":
                    color_tag = "error"
                elif d.status in ("rerouted", "returning"):
                    color_tag = "warn"
                t.insert("end", f"  {d.drone_id} [{d.drone_type}] {d.status} "
                         f"Bat:{d.battery:.0f}%\n", color_tag)
            t.insert("end", "\n")
        if state.deliveries:
            t.insert("end", "  ID       Pickup  Drop   Wt   Status\n", "header")
            t.insert("end", "  " + "-"*42 + "\n")
            for d in state.deliveries:
                sc = {"completed": "good", "in-transit": "normal",
                      "delayed": "warn", "failed": "error"}.get(d.status, "normal")
                t.insert("end", f"  {d.delivery_id}  {str(d.pickup_cell):>7} "
                         f"{str(d.dropoff_cell):>7}  {d.weight:.1f}  {d.status}\n", sc)
        t.tag_configure("header", foreground=CYAN, font=("Consolas", 8, "bold"))
        t.tag_configure("good", foreground=GREEN)
        t.tag_configure("normal", foreground=TEXT)
        t.tag_configure("warn", foreground=ORANGE)
        t.tag_configure("error", foreground=RED)

    def UpdateDeliveryTable(self, state):
        self.UpdateFleetSummary(state)

    # --------------------------------------------------- event log
    def UpdateEventLog(self, state):
        if self._event_log is None:
            return
        self._event_log.delete("1.0", "end")
        for line in state.event_log:
            if "FAIL" in line or "anomaly" in line.lower():
                self._event_log.insert("end", line + "\n", "error")
            elif "reroute" in line.lower() or "disruption" in line.lower() or "no-fly" in line.lower():
                self._event_log.insert("end", line + "\n", "warn")
            else:
                self._event_log.insert("end", line + "\n", "normal")
        self._event_log.tag_configure("normal", foreground=TEXT)
        self._event_log.see("end")

    # --------------------------------------------------- anomaly alert
    def ShowAnomalyAlert(self, drone_id, anomaly_type, severity="HIGH"):
        if self._alert_frame and self._alert_label:
            self._alert_label.config(
                text=f"  ALERT [{severity}]  {anomaly_type} detected on {drone_id}")
            self._alert_label.pack(fill="x", padx=10, pady=4)
            self._alert_frame.pack(fill="x", padx=8, before=self._event_log.master)
            self.root.update_idletasks()

    # --------------------------------------------------- final summary
    def UpdateFinalSummary(self, summary):
        if self._fleet_text is None:
            return
        self._fleet_text.delete("1.0", "end")
        t = self._fleet_text
        t.insert("end", "  FINAL SUMMARY\n", "header")
        t.insert("end", "  " + "="*30 + "\n")
        for k, v in summary.items():
            t.insert("end", f"  {k:<20}: {v}\n")
        t.tag_configure("header", foreground=CYAN, font=("Consolas", 9, "bold"))

    def Close(self):
        if self.root:
            self.root.destroy()
