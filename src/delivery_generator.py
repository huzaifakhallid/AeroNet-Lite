"""
AeroNet Lite - Delivery Generator
Creates simulation deliveries using grid data and optionally Amazon CSV.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from config import (
    RANDOM_SEED, NUM_DELIVERIES,
    RESIDENTIAL, COMMERCIAL, HOSPITAL, OPEN,
    TABLE_DIR,
)
from grid_model import (
    Cell, Delivery,
    GetAllCells, GetCellsByCondition,
    GetHubLocations, GetMedicalPickupLocations,
)

import pandas as pd

# -- Category pool --------------------------------------------------------
DEFAULT_CATEGORIES = ["Electronics", "Food", "Clothing", "Medical", "Documents", "Sports"]
WEIGHT_RANGES = {
    "Electronics": (0.5, 3.0),
    "Food":        (0.3, 2.5),
    "Clothing":    (0.2, 1.5),
    "Medical":     (0.5, 4.5),
    "Documents":   (0.1, 0.5),
    "Sports":      (0.5, 2.0),
}


def _LoadAmazonCategories(amazon_df: pd.DataFrame | None) -> list[str]:
    """Extract category list from Amazon data if available."""
    if amazon_df is None:
        return DEFAULT_CATEGORIES
    try:
        cats = amazon_df["Category"].dropna().str.strip().unique().tolist()
        return cats if cats else DEFAULT_CATEGORIES
    except Exception:
        return DEFAULT_CATEGORIES


def ChoosePickupCell(grid: list[list[Cell]]) -> tuple[int, int]:
    """Prefer hubs, commercial cells, or medical pickup points."""
    candidates = GetCellsByCondition(grid, lambda c: c.is_hub or c.is_medical_pickup or c.zone == COMMERCIAL)
    if not candidates:
        candidates = GetCellsByCondition(grid, lambda c: not c.no_fly)
    cell = random.choice(candidates)
    return (cell.row, cell.col)


def ChooseDropoffCell(grid: list[list[Cell]], avoid: tuple[int, int] | None = None) -> tuple[int, int]:
    """Prefer high-demand residential / commercial cells or hospitals."""
    weights: list[float] = []
    candidates: list[Cell] = []
    for c in GetAllCells(grid):
        if c.no_fly:
            continue
        if avoid and (c.row, c.col) == avoid:
            continue
        w = max(c.demand, 0.1)
        if c.zone in (RESIDENTIAL, COMMERCIAL):
            w *= 2.0
        if c.zone == HOSPITAL:
            w *= 1.5
        weights.append(w)
        candidates.append(c)
    if not candidates:
        candidates = GetCellsByCondition(grid, lambda c: not c.no_fly)
        weights = [1.0] * len(candidates)
    cell = random.choices(candidates, weights=weights, k=1)[0]
    return (cell.row, cell.col)


def GeneratePackageWeight(category: str) -> float:
    lo, hi = WEIGHT_RANGES.get(category, (0.3, 2.0))
    return round(random.uniform(lo, hi), 2)


def GeneratePriority(zone: str, category: str) -> str:
    if category == "Medical" or zone == HOSPITAL:
        return "medical"
    if random.random() < 0.2:
        return "urgent"
    return "normal"


def GenerateDeliveries(
    grid: list[list[Cell]],
    num_deliveries: int = NUM_DELIVERIES,
    amazon_df: pd.DataFrame | None = None,
) -> list[Delivery]:
    """Generate *num_deliveries* delivery tasks."""
    categories = _LoadAmazonCategories(amazon_df)
    deliveries: list[Delivery] = []

    for i in range(1, num_deliveries + 1):
        pickup = ChoosePickupCell(grid)
        dropoff = ChooseDropoffCell(grid, avoid=pickup)
        cat = random.choice(categories)
        weight = GeneratePackageWeight(cat)
        zone = grid[dropoff[0]][dropoff[1]].zone
        priority = GeneratePriority(zone, cat)

        deliveries.append(Delivery(
            delivery_id=f"DEL-{i:03d}",
            pickup_cell=pickup,
            dropoff_cell=dropoff,
            weight=weight,
            priority=priority,
            category=cat,
        ))
    return deliveries


def SaveDeliveries(deliveries: list[Delivery], output_path: Path | None = None) -> Path:
    path = output_path or (TABLE_DIR / "delivery_assignments.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["delivery_id", "pickup_cell", "dropoff_cell", "weight",
                  "priority", "category", "status", "assigned_drone_id", "route_cost"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for d in deliveries:
            writer.writerow({
                "delivery_id": d.delivery_id,
                "pickup_cell": d.pickup_cell,
                "dropoff_cell": d.dropoff_cell,
                "weight": d.weight,
                "priority": d.priority,
                "category": d.category,
                "status": d.status,
                "assigned_drone_id": d.assigned_drone_id or "",
                "route_cost": d.route_cost,
            })
    return path
