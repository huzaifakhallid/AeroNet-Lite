"""
AeroNet Lite - Fleet Selector
Brute-force fleet selection under budget using a weighted fitness function.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from config import (
    DEFAULT_BUDGET,
    LIGHT_DRONE_COST, HEAVY_DRONE_COST,
    LIGHT_DRONE_PAYLOAD, HEAVY_DRONE_PAYLOAD,
    LIGHT_DRONE_RANGE, HEAVY_DRONE_RANGE,
    TABLE_DIR,
)
from grid_model import Drone, GetHubLocations, Cell


def CalculateFleetCost(light_count: int, heavy_count: int) -> int:
    return light_count * LIGHT_DRONE_COST + heavy_count * HEAVY_DRONE_COST


def EstimateFleetCapacity(light_count: int, heavy_count: int) -> float:
    """Total kg capacity of the fleet."""
    return light_count * LIGHT_DRONE_PAYLOAD + heavy_count * HEAVY_DRONE_PAYLOAD


def EstimateCoveragePercentage(light_count: int, heavy_count: int, total_demand: float) -> float:
    """Fraction of total demand (kg) covered, capped at 1.0."""
    if total_demand <= 0:
        return 1.0
    cap = EstimateFleetCapacity(light_count, heavy_count)
    return min(cap / total_demand, 1.0)


def CalculateFleetScore(coverage_pct: float, budget_used_pct: float) -> float:
    """score = 0.75 * coverage - 0.25 * budget_used"""
    return 0.75 * coverage_pct - 0.25 * budget_used_pct


def SelectFleetBruteForce(total_demand: float, budget: int = DEFAULT_BUDGET) -> dict:
    """
    Enumerate all (light, heavy) combos under *budget*.
    Returns dict with best combo and top-5 candidates.
    """
    max_light = budget // LIGHT_DRONE_COST
    max_heavy = budget // HEAVY_DRONE_COST

    candidates: list[dict] = []
    for l in range(0, max_light + 1):
        for h in range(0, max_heavy + 1):
            cost = CalculateFleetCost(l, h)
            if cost > budget or (l + h) == 0:
                continue
            cov = EstimateCoveragePercentage(l, h, total_demand)
            bup = cost / budget
            score = CalculateFleetScore(cov, bup)
            candidates.append({
                "light": l,
                "heavy": h,
                "total_cost": cost,
                "coverage_pct": round(cov, 4),
                "budget_used_pct": round(bup, 4),
                "score": round(score, 4),
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0] if candidates else {"light": 1, "heavy": 0, "total_cost": LIGHT_DRONE_COST,
                                              "coverage_pct": 0, "budget_used_pct": 0, "score": 0}
    return {
        "best": best,
        "top5": candidates[:5],
        "all_candidates": candidates,
    }


def BuildDroneFleet(selected: dict, hub_locations: list[tuple[int, int]]) -> list[Drone]:
    """Create Drone dataclass instances and distribute across hubs."""
    drones: list[Drone] = []
    idx = 1
    hubs = hub_locations if hub_locations else [(0, 0)]

    for _ in range(selected.get("light", 0)):
        hub = hubs[idx % len(hubs)]
        drones.append(Drone(
            drone_id=f"D{idx:02d}",
            drone_type="light",
            payload_capacity=LIGHT_DRONE_PAYLOAD,
            range_limit=LIGHT_DRONE_RANGE,
            cost=LIGHT_DRONE_COST,
            current_position=hub,
            home_hub=hub,
        ))
        idx += 1

    for _ in range(selected.get("heavy", 0)):
        hub = hubs[idx % len(hubs)]
        drones.append(Drone(
            drone_id=f"D{idx:02d}",
            drone_type="heavy",
            payload_capacity=HEAVY_DRONE_PAYLOAD,
            range_limit=HEAVY_DRONE_RANGE,
            cost=HEAVY_DRONE_COST,
            current_position=hub,
            home_hub=hub,
        ))
        idx += 1

    return drones


def SaveFleetResults(results: dict, output_path: Path | None = None) -> Path:
    path = output_path or (TABLE_DIR / "fleet_selection_results.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["light", "heavy", "total_cost", "coverage_pct", "budget_used_pct", "score"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results.get("top5", []):
            writer.writerow(row)
    return path
