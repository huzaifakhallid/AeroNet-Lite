"""
AeroNet Lite - Grid Model
Core data models (Cell, Drone, Delivery, SimulationState) and the
hand-crafted 10x10 city grid that satisfies most CSP constraints.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config import (
    GRID_ROWS, GRID_COLS,
    RESIDENTIAL, COMMERCIAL, HOSPITAL, SCHOOL, INDUSTRIAL, OPEN,
    DENSITY_CSV,
    LIGHT_DRONE_COST, HEAVY_DRONE_COST,
    LIGHT_DRONE_PAYLOAD, HEAVY_DRONE_PAYLOAD,
    LIGHT_DRONE_RANGE, HEAVY_DRONE_RANGE,
)


@dataclass
class Cell:
    """A single cell in the 10x10 city grid."""
    row: int
    col: int
    zone: str
    density: float
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0


@dataclass
class Drone:
    """Represents a single drone in the fleet."""
    drone_id: str
    drone_type: str                      # "light" | "heavy"
    payload_capacity: float
    range_limit: int
    cost: int
    current_position: tuple[int, int]
    home_hub: tuple[int, int]
    battery: float = 100.0
    status: str = "idle"                 # idle | en-route | rerouted | returning | failed
    assigned_delivery_id: Optional[str] = None
    planned_route: list[tuple[int, int]] = field(default_factory=list)
    completed_path: list[tuple[int, int]] = field(default_factory=list)
    current_target: Optional[tuple[int, int]] = None
    anomaly_status: str = "normal"
    route_step_index: int = 0            # where we are along planned_route


@dataclass
class Delivery:
    """A delivery task from pickup to drop-off."""
    delivery_id: str
    pickup_cell: tuple[int, int]
    dropoff_cell: tuple[int, int]
    weight: float
    priority: str                        # "normal" | "urgent" | "medical"
    category: str
    status: str = "pending"              # pending | assigned | in-transit | completed | delayed | failed
    assigned_drone_id: Optional[str] = None
    route_cost: float = 0.0


@dataclass
class SimulationState:
    """Complete simulation state at any point in time."""
    grid: list[list[Cell]]
    drones: list[Drone] = field(default_factory=list)
    deliveries: list[Delivery] = field(default_factory=list)
    event_log: list[str] = field(default_factory=list)
    current_step: int = 0
    validation_results: dict = field(default_factory=dict)
    demand_metrics: dict = field(default_factory=dict)
    anomaly_metrics: dict = field(default_factory=dict)
    old_route_for_reroute: list[tuple[int, int]] = field(default_factory=list)
    new_route_for_reroute: list[tuple[int, int]] = field(default_factory=list)
    reroute_no_fly_cell: Optional[tuple[int, int]] = None


def LoadDensityValues(path: Path = DENSITY_CSV) -> list[int]:
    """Read city density values from CSV; fall back to defaults if missing."""
    if not path.exists():
        return [1200, 3500, 7500]
    values: list[int] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            col = next((n for n in (reader.fieldnames or []) if "Density" in n), None)
            if col is None:
                return [1200, 3500, 7500]
            for row in reader:
                raw = row.get(col, "").replace(",", "").strip()
                if raw.isdigit():
                    values.append(int(raw))
    except Exception:
        pass
    return values or [1200, 3500, 7500]


def _DensityTier(tier: str, values: list[int]) -> int:
    low    = [v for v in values if v < 2000]
    medium = [v for v in values if 2000 <= v <= 6000]
    high   = [v for v in values if v > 6000]
    buckets = {"low": low, "medium": medium, "high": high}
    bucket = buckets.get(tier.lower(), medium) or values
    return int(sum(bucket) / len(bucket))


def CreateSampleGrid() -> list[list[Cell]]:
    """
    Build a hand-designed 10x10 grid with:
    • 2+ hubs, 2+ charging pads, 2+ hospitals, 2+ schools
    • commercial corridors, residential zones, industrial area
    • 1 initial no-fly cell, 2 medical pickup points
    """
    densities = LoadDensityValues()
    dz = {
        RESIDENTIAL: _DensityTier("medium", densities),
        COMMERCIAL:  _DensityTier("high",   densities),
        HOSPITAL:    _DensityTier("medium", densities),
        SCHOOL:      _DensityTier("medium", densities),
        INDUSTRIAL:  _DensityTier("low",    densities),
        OPEN:        _DensityTier("low",    densities),
    }

    # Row-by-row zone layout (10 rows x 10 cols)
    zone_rows = [
        # Row 0
        [RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL, OPEN, OPEN, RESIDENTIAL, RESIDENTIAL, SCHOOL, RESIDENTIAL],
        # Row 1
        [RESIDENTIAL, HOSPITAL, COMMERCIAL, COMMERCIAL, OPEN, OPEN, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL],
        # Row 2
        [RESIDENTIAL, RESIDENTIAL, COMMERCIAL, OPEN, OPEN, RESIDENTIAL, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL],
        # Row 3
        [COMMERCIAL, COMMERCIAL, OPEN, OPEN, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL, OPEN, OPEN],
        # Row 4
        [COMMERCIAL, COMMERCIAL, OPEN, OPEN, OPEN, OPEN, COMMERCIAL, COMMERCIAL, RESIDENTIAL, RESIDENTIAL],
        # Row 5
        [OPEN, OPEN, OPEN, INDUSTRIAL, INDUSTRIAL, OPEN, OPEN, COMMERCIAL, RESIDENTIAL, RESIDENTIAL],
        # Row 6
        [RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL, OPEN, OPEN, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL],
        # Row 7
        [RESIDENTIAL, SCHOOL, COMMERCIAL, COMMERCIAL, OPEN, OPEN, RESIDENTIAL, HOSPITAL, COMMERCIAL, COMMERCIAL],
        # Row 8
        [RESIDENTIAL, RESIDENTIAL, COMMERCIAL, OPEN, OPEN, RESIDENTIAL, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL],
        # Row 9
        [OPEN, OPEN, COMMERCIAL, COMMERCIAL, OPEN, OPEN, COMMERCIAL, COMMERCIAL, OPEN, OPEN],
    ]

    # Facility overlays - hubs placed to cover ALL residential within distance 3
    hubs            = {(0, 0), (0, 4), (2, 8), (3, 6), (4, 9), (6, 0), (6, 5), (8, 8), (9, 2)}
    charging_pads   = {(0, 1), (0, 5), (2, 9), (3, 7), (5, 0), (6, 1), (6, 6), (8, 9), (9, 3)}
    medical_pickups = {(1, 2), (7, 8)}
    no_fly_initial  = {(4, 3)}           # one initial no-fly

    grid: list[list[Cell]] = []
    for r, row_zones in enumerate(zone_rows):
        row_cells: list[Cell] = []
        for c, zone in enumerate(row_zones):
            density = dz[zone]
            row_cells.append(Cell(
                row=r, col=c,
                zone=zone,
                density=density,
                is_hub=(r, c) in hubs,
                is_charging=(r, c) in charging_pads,
                is_medical_pickup=(r, c) in medical_pickups,
                no_fly=(r, c) in no_fly_initial,
                demand=round(density / 1000, 2),
            ))
        grid.append(row_cells)
    return grid


def GetAllCells(grid: list[list[Cell]]) -> list[Cell]:
    return [cell for row in grid for cell in row]

def GetCellsByZone(grid: list[list[Cell]], zone: str) -> list[Cell]:
    return [c for c in GetAllCells(grid) if c.zone == zone]

def GetHubLocations(grid: list[list[Cell]]) -> list[tuple[int, int]]:
    return [(c.row, c.col) for c in GetAllCells(grid) if c.is_hub]

def GetChargingLocations(grid: list[list[Cell]]) -> list[tuple[int, int]]:
    return [(c.row, c.col) for c in GetAllCells(grid) if c.is_charging]

def GetMedicalPickupLocations(grid: list[list[Cell]]) -> list[tuple[int, int]]:
    return [(c.row, c.col) for c in GetAllCells(grid) if c.is_medical_pickup]

def GetNoFlyLocations(grid: list[list[Cell]]) -> list[tuple[int, int]]:
    return [(c.row, c.col) for c in GetAllCells(grid) if c.no_fly]

def GetCell(grid: list[list[Cell]], row: int, col: int) -> Cell:
    return grid[row][col]

def CloneGrid(grid: list[list[Cell]]) -> list[list[Cell]]:
    return copy.deepcopy(grid)

def GetCellsByCondition(grid: list[list[Cell]], condition: Callable[[Cell], bool]) -> list[Cell]:
    return [c for c in GetAllCells(grid) if condition(c)]

def Manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def GetNeighbors(row: int, col: int, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> list[tuple[int, int]]:
    """Return valid 4-directional neighbours."""
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [(r, c) for r, c in candidates if 0 <= r < rows and 0 <= c < cols]
