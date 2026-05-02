"""
AeroNet Lite - Dashboard Controller
Event-driven simulation using root.after() instead of time.sleep().
"""

from __future__ import annotations

import logging
import random

from config import (
    RANDOM_SEED, DEMO_NUM_DELIVERIES, DEFAULT_BUDGET,
    DRONE_STEPS_PER_TICK,
)
from grid_model import (
    SimulationState, GetHubLocations, GetAllCells,
)
from layout_validator import ValidateLayout, PrintValidationReport, SaveValidationReport
from fleet_selector import SelectFleetBruteForce, BuildDroneFleet, SaveFleetResults
from delivery_generator import GenerateDeliveries, SaveDeliveries
from delivery_assigner import AssignDeliveriesToDrones, SaveDeliveryAssignments
from astar_planner import PlanDeliveryRoute, RunAStar, CalculateRouteCost
from disruption_handler import HandleDisruption
from ml_demand import PredictGridDemand, ApplyDemandToGrid
from ml_anomaly import PredictDroneAnomaly
from data_loader import LoadAmazonDeliveryData
from utils import LogEvent, SaveEventLog, GenerateSimulationSummary, InitializeSimulation
from report_generator import SaveReport

logger = logging.getLogger(__name__)

TICK_MS = 700


class SimulationController:
    """Drives the 20-step simulation via root.after() callbacks."""

    def __init__(self, root, state, dashboard,
                 demand_model_info=None, anomaly_model_info=None):
        self.root = root
        self.state = state
        self.dashboard = dashboard
        self.demand_model_info = demand_model_info
        self.anomaly_model_info = anomaly_model_info
        self.fleet_results = None
        self._anomaly_drone = None

    def Start(self):
        self.root.after(300, self.RunStep1)

    def ResetAndRun(self):
        """Reset state and UI, then start a new simulation."""
        new_state = InitializeSimulation()
        self.state = new_state
        self.dashboard.state = new_state
        
        # Clear UI
        self.dashboard.DrawGrid()
        self.dashboard.DrawFacilityMarkers()
        self.dashboard.DrawDeliveries()
        self.dashboard.DrawDrones()
        self.dashboard.ClearRoutes()
        if self.dashboard._event_log:
            self.dashboard._event_log.delete("1.0", "end")
        if self.dashboard._fleet_text:
            self.dashboard._fleet_text.delete("1.0", "end")
        if self.dashboard._alert_frame:
            self.dashboard._alert_frame.pack_forget()
            
        self.Start()

    def _refresh(self):
        self.dashboard.UpdateDashboard(self.state)

    def _schedule(self, callback, delay=TICK_MS):
        self.root.after(delay, callback)

    # ---------------------------------------------------------------- steps
    def RunStep1(self):
        self.state.current_step = 1
        LogEvent(self.state, "Grid initialised (10x10).")
        self._refresh()
        self._schedule(self.RunStep2)

    def RunStep2(self):
        self.state.current_step = 2
        val_result = ValidateLayout(self.state.grid)
        self.state.validation_results = {
            "is_valid": val_result.is_valid,
            "passed": val_result.passed_rules,
            "failed": val_result.failed_rules,
            "violations": val_result.violations,
        }
        PrintValidationReport(val_result)
        SaveValidationReport(val_result)
        status_str = "PASS" if val_result.is_valid else f"FAIL ({len(val_result.violations)} issues)"
        LogEvent(self.state, f"Layout validation: {status_str}.")
        self._refresh()
        self._schedule(self.RunStep3)

    def RunStep3(self):
        self.state.current_step = 3
        dmi = self.demand_model_info
        if dmi and "model" in dmi:
            try:
                predictions = PredictGridDemand(
                    dmi["model"], self.state.grid, features=dmi.get("features"))
                ApplyDemandToGrid(self.state.grid, predictions)
                LogEvent(self.state, "Demand model predictions applied to grid.")
            except Exception as exc:
                LogEvent(self.state, f"Demand prediction failed ({exc}).")
        else:
            LogEvent(self.state, "No demand model; using density-based demand.")

        self.dashboard.UpdateModelStats(self.demand_model_info, self.anomaly_model_info)
        self.dashboard.UpdateDemandHeatmap(self.state.grid)

        total_demand = DEMO_NUM_DELIVERIES * 1.5
        self.fleet_results = SelectFleetBruteForce(total_demand, DEFAULT_BUDGET)
        SaveFleetResults(self.fleet_results)
        best = self.fleet_results["best"]
        LogEvent(self.state, f"Fleet selected: {best['light']}L + {best['heavy']}H "
                 f"(cost=${best['total_cost']}, coverage={best['coverage_pct']:.0%}).")

        hub_locations = GetHubLocations(self.state.grid)
        self.state.drones = BuildDroneFleet(best, hub_locations)
        LogEvent(self.state, f"Built {len(self.state.drones)} drones across {len(hub_locations)} hubs.")
        self._refresh()
        self._schedule(self.RunStep4)

    def RunStep4(self):
        self.state.current_step = 4
        amazon_df = LoadAmazonDeliveryData()
        self.state.deliveries = GenerateDeliveries(
            self.state.grid, DEMO_NUM_DELIVERIES, amazon_df)
        LogEvent(self.state, f"Generated {len(self.state.deliveries)} deliveries.")
        self._refresh()
        self._schedule(self.RunStep5)

    def RunStep5(self):
        self.state.current_step = 5
        AssignDeliveriesToDrones(self.state.deliveries, self.state.drones, self.state.grid)
        assigned = sum(1 for d in self.state.deliveries if d.status == "assigned")
        LogEvent(self.state, f"Assigned {assigned}/{len(self.state.deliveries)} deliveries to drones.")
        self._refresh()
        self._schedule(self.RunStep6)

    def RunStep6(self):
        self.state.current_step = 6
        for drone in self.state.drones:
            if drone.assigned_delivery_id is None:
                continue
            delivery = next((d for d in self.state.deliveries
                             if d.delivery_id == drone.assigned_delivery_id), None)
            if delivery is None:
                continue
            result = PlanDeliveryRoute(drone, delivery, self.state.grid)
            if result.success:
                drone.planned_route = result.path
                drone.route_step_index = 0
                drone.status = "en-route"
                drone.current_target = delivery.pickup_cell
                delivery.status = "in-transit"
                delivery.route_cost = result.total_cost
                LogEvent(self.state, f"{drone.drone_id} route planned "
                         f"({len(result.path)} steps, cost={result.total_cost:.1f}).")
            else:
                delivery.status = "failed"
                drone.status = "idle"
                LogEvent(self.state, f"{drone.drone_id} route planning FAILED.")
        SaveDeliveryAssignments(self.state.deliveries)
        self._refresh()
        self._schedule(self._make_move_step(7))

    # ------------------------------------------------- movement steps 7-10
    def _make_move_step(self, step_num):
        def _run():
            self.state.current_step = step_num
            self._move_drones()
            self._check_completions()
            LogEvent(self.state, f"Drones advanced (step {step_num}).")
            self._refresh()
            if step_num < 10:
                self._schedule(self._make_move_step(step_num + 1))
            else:
                self._schedule(self.RunStep11)
        return _run

    def _move_drones(self, steps=DRONE_STEPS_PER_TICK):
        for drone in self.state.drones:
            if drone.status not in ("en-route", "rerouted", "returning"):
                continue
            for _ in range(steps):
                if drone.route_step_index < len(drone.planned_route) - 1:
                    drone.route_step_index += 1
                    new_pos = drone.planned_route[drone.route_step_index]
                    drone.current_position = new_pos
                    drone.completed_path.append(new_pos)
                    drone.battery = max(drone.battery - 3.0, 0)
                else:
                    if drone.status == "returning":
                        drone.status = "idle"
                    else:
                        drone.status = "idle"
                    break

    def _check_completions(self):
        for drone in self.state.drones:
            if drone.assigned_delivery_id is None:
                continue
            delivery = next((d for d in self.state.deliveries
                             if d.delivery_id == drone.assigned_delivery_id), None)
            if delivery is None:
                continue
            if (drone.current_position == drone.home_hub
                    and drone.status == "idle"
                    and delivery.status == "in-transit"):
                delivery.status = "completed"
                delivery.route_cost = CalculateRouteCost(drone.completed_path, self.state.grid)

    # ------------------------------------------------- disruption step 11
    def RunStep11(self):
        self.state.current_step = 11
        nf = self._pick_disruption_cell()
        if nf:
            msgs = HandleDisruption(self.state, nf[0], nf[1])
            for m in msgs:
                LogEvent(self.state, m)
            # Show old route in red briefly
            if self.state.old_route_for_reroute:
                self.dashboard.ClearRoutes()
                self.dashboard.DrawRoutes(
                    [self.state.old_route_for_reroute], color="#ff4444")
        else:
            LogEvent(self.state, "No suitable cell for disruption.")
        self._refresh()
        self._schedule(self._show_new_route_after_disruption, 800)

    def _show_new_route_after_disruption(self):
        self.dashboard.ClearRoutes()
        if self.state.new_route_for_reroute:
            self.dashboard.DrawRoutes(
                [self.state.new_route_for_reroute], color="#00ff00")
        self.dashboard.DrawRoutes()
        self.root.update_idletasks()
        self._schedule(self._make_post_disruption_step(12))

    def _pick_disruption_cell(self):
        route_cells = set()
        for drone in self.state.drones:
            if drone.status in ("en-route", "assigned", "rerouted"):
                remaining = drone.planned_route[drone.route_step_index:]
                route_cells.update(remaining)
        current_positions = {d.current_position for d in self.state.drones}
        candidates = [
            c for c in route_cells
            if not self.state.grid[c[0]][c[1]].no_fly
            and not self.state.grid[c[0]][c[1]].is_hub
            and c not in current_positions
        ]
        if candidates:
            return random.choice(candidates)
        fallback = [
            (cell.row, cell.col) for cell in GetAllCells(self.state.grid)
            if not cell.no_fly and not cell.is_hub and cell.zone == "Open"
        ]
        return random.choice(fallback) if fallback else None

    # ------------------------------------------- post-disruption steps 12-14
    def _make_post_disruption_step(self, step_num):
        def _run():
            self.state.current_step = step_num
            self._move_drones()
            self._check_completions()
            LogEvent(self.state, f"Drones advanced (step {step_num}).")
            self._refresh()
            if step_num < 14:
                self._schedule(self._make_post_disruption_step(step_num + 1))
            else:
                self._schedule(self._make_demand_step(15))
        return _run

    # ------------------------------------------- demand update steps 15-17
    def _make_demand_step(self, step_num):
        def _run():
            self.state.current_step = step_num
            if step_num == 15 and self.demand_model_info and "model" in self.demand_model_info:
                try:
                    predictions = PredictGridDemand(
                        self.demand_model_info["model"], self.state.grid,
                        features=self.demand_model_info.get("features"))
                    ApplyDemandToGrid(self.state.grid, predictions)
                    LogEvent(self.state, "Demand heatmap updated from model.")
                    self.dashboard.UpdateDemandHeatmap(self.state.grid)
                except Exception:
                    LogEvent(self.state, "Demand heatmap update skipped.")
            self._move_drones()
            self._check_completions()
            LogEvent(self.state, f"Drones advanced (step {step_num}).")
            self._refresh()
            if step_num < 17:
                self._schedule(self._make_demand_step(step_num + 1))
            else:
                self._schedule(self.RunStep18)
        return _run

    # ------------------------------------------------- anomaly step 18
    def RunStep18(self):
        self.state.current_step = 18
        anomaly_type = "Battery anomaly"
        active_drones = [d for d in self.state.drones
                         if d.status in ("en-route", "rerouted")]
        if not active_drones:
            active_drones = self.state.drones[:1]

        if active_drones:
            self._anomaly_drone = active_drones[0]
            ami = self.anomaly_model_info
            if ami and "model" in ami and "label_encoder" in ami:
                fake_features = {
                    "propeller_count": 4,
                    "max_carry_weight": self._anomaly_drone.payload_capacity,
                    "actual_carry_weight": self._anomaly_drone.payload_capacity * 1.2,
                    "payload_ratio": 1.2,
                    "altitude": 120, "flight_duration": 25,
                    "distance_flown": 8, "battery_remaining": 10,
                    "gps_accuracy": 2.5, "wind_speed": 5,
                    "obstacles_encountered": 1,
                }
                try:
                    anomaly_type = PredictDroneAnomaly(
                        ami["model"], ami["label_encoder"], fake_features)
                except Exception:
                    anomaly_type = "Battery anomaly"
            self._anomaly_drone.anomaly_status = anomaly_type
            LogEvent(self.state, f"Anomaly detected on {self._anomaly_drone.drone_id}: {anomaly_type}.")
            self.dashboard.ShowAnomalyAlert(self._anomaly_drone.drone_id, anomaly_type)
        else:
            LogEvent(self.state, "No drone available for anomaly check.")
        self._refresh()
        self._schedule(self.RunStep19, 1000)

    # ------------------------------------------------- anomaly response step 19
    def RunStep19(self):
        self.state.current_step = 19
        if self._anomaly_drone and self._anomaly_drone.anomaly_status != "Normal":
            drone = self._anomaly_drone
            result = RunAStar(drone.current_position, drone.home_hub, self.state.grid)
            if result.success:
                drone.planned_route = drone.completed_path[:] + result.path
                drone.route_step_index = len(drone.completed_path)
                drone.status = "returning"
                drone.current_target = drone.home_hub
                LogEvent(self.state, f"{drone.drone_id} returning to hub due to anomaly.")
                for d in self.state.deliveries:
                    if d.delivery_id == drone.assigned_delivery_id and d.status == "in-transit":
                        d.status = "delayed"
                        LogEvent(self.state, f"Delivery {d.delivery_id} delayed (anomaly).")
            else:
                drone.status = "failed"
                LogEvent(self.state, f"{drone.drone_id} cannot return to hub - FAILED.")
        else:
            LogEvent(self.state, "No anomaly response needed.")
        self._move_drones()
        self._check_completions()
        self._refresh()
        self._schedule(self.RunStep20, 1000)

    # ------------------------------------------------- final step 20
    def RunStep20(self):
        self.state.current_step = 20
        self._check_completions()
        summary = GenerateSimulationSummary(self.state)
        LogEvent(self.state, "Simulation complete.")
        LogEvent(self.state, f"Completed: {summary['Completed']}  |  "
                 f"Delayed: {summary['Delayed']}  |  "
                 f"Failed: {summary['Failed']}")
        SaveEventLog(self.state.event_log)
        self.dashboard.UpdateDashboard(self.state)
        self.dashboard.UpdateFinalSummary(summary)
        
        LogEvent(self.state, "Generating publication-quality report figures...")
        try:
            SaveReport(self.state, self.demand_model_info, self.anomaly_model_info, self.fleet_results)
            LogEvent(self.state, "Report saved to report/figures/ (12 figures).")
        except Exception as exc:
            LogEvent(self.state, f"Report generation failed: {exc}")
            
        self.dashboard.EnableRunButton()
