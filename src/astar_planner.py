"""
AeroNet Lite - A* Route Planner
A* search on the 10x10 grid with admissible heuristic.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

from config import (
    GRID_ROWS, GRID_COLS,
    NORMAL_MOVE_COST, COMMERCIAL_MOVE_COST,
    COMMERCIAL,
)
from grid_model import Cell, Drone, Delivery, Manhattan, GetNeighbors


@dataclass
class AStarResult:
    """Result of a single A* search."""
    path: list[tuple[int, int]]
    total_cost: float
    success: bool
    message: str


def GetMovementCost(cell: Cell) -> float:
    """Movement cost to enter *cell*."""
    if cell.zone == COMMERCIAL:
        return COMMERCIAL_MOVE_COST
    return NORMAL_MOVE_COST


def Heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Admissible heuristic: 0.8 x Manhattan (min possible step cost is 0.8)."""
    return 0.8 * Manhattan(a, b)


def GetValidRouteNeighbors(
    position: tuple[int, int],
    grid: list[list[Cell]],
) -> list[tuple[int, int]]:
    """4-directional neighbours that are not no-fly."""
    nbrs = GetNeighbors(position[0], position[1], GRID_ROWS, GRID_COLS)
    return [(r, c) for r, c in nbrs if not grid[r][c].no_fly]


def ReconstructPath(
    parent: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Trace back from goal to start using the parent map."""
    path: list[tuple[int, int]] = [goal]
    current = goal
    while current != start:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path


def RunAStar(
    start: tuple[int, int],
    goal: tuple[int, int],
    grid: list[list[Cell]],
) -> AStarResult:
    """
    Standard A* from *start* to *goal* on the grid.
    Returns AStarResult with the path, cost, and success flag.
    """
    if start == goal:
        return AStarResult(path=[start], total_cost=0.0, success=True, message="Start equals goal.")

    open_set: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_set, (Heuristic(start, goal), counter, start))
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    while open_set:
        _f, _cnt, current = heapq.heappop(open_set)

        if current == goal:
            path = ReconstructPath(parent, start, goal)
            return AStarResult(path=path, total_cost=g_score[goal], success=True, message="Path found.")

        if current in closed:
            continue
        closed.add(current)

        for nr, nc in GetValidRouteNeighbors(current, grid):
            nb = (nr, nc)
            tentative = g_score[current] + GetMovementCost(grid[nr][nc])
            if nb in closed:
                continue
            if tentative < g_score.get(nb, float("inf")):
                g_score[nb] = tentative
                parent[nb] = current
                f = tentative + Heuristic(nb, goal)
                counter += 1
                heapq.heappush(open_set, (f, counter, nb))

    return AStarResult(path=[], total_cost=0.0, success=False,
                       message=f"No path from {start} to {goal}.")


def PlanDeliveryRoute(
    drone: Drone,
    delivery: Delivery,
    grid: list[list[Cell]],
) -> AStarResult:
    """
    Plan full delivery route: hub → pickup → dropoff → hub.
    Concatenates three A* segments.
    """
    hub = drone.home_hub
    pickup = delivery.pickup_cell
    dropoff = delivery.dropoff_cell

    segments = [
        (hub, pickup, "hub→pickup"),
        (pickup, dropoff, "pickup→dropoff"),
        (dropoff, hub, "dropoff→hub"),
    ]

    full_path: list[tuple[int, int]] = []
    total_cost = 0.0

    for start, goal, label in segments:
        result = RunAStar(start, goal, grid)
        if not result.success:
            return AStarResult(
                path=[], total_cost=0.0, success=False,
                message=f"Segment {label} failed: {result.message}",
            )
        # avoid duplicating junction nodes
        if full_path and result.path and result.path[0] == full_path[-1]:
            full_path.extend(result.path[1:])
        else:
            full_path.extend(result.path)
        total_cost += result.total_cost

    return AStarResult(path=full_path, total_cost=total_cost, success=True, message="Full route planned.")


def ValidateRoute(route: list[tuple[int, int]], grid: list[list[Cell]]) -> bool:
    """Check that no cell on the route is currently no-fly."""
    return all(not grid[r][c].no_fly for r, c in route)


def CalculateRouteCost(route: list[tuple[int, int]], grid: list[list[Cell]]) -> float:
    """Sum movement costs along a route (cost counted for entering each cell after the first)."""
    if len(route) < 2:
        return 0.0
    return sum(GetMovementCost(grid[r][c]) for r, c in route[1:])
