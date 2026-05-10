"""
AeroNet Lite - Disruption Handler
Handles mid-simulation no-fly cell activations and drone rerouting.
"""

from __future__ import annotations

from grid_model import Cell, Drone, Delivery, SimulationState
from astar_planner import RunAStar, AStarResult


def ActivateNoFlyCell(grid: list[list[Cell]], row: int, col: int) -> None:
    """Mark a cell as no-fly zone at runtime."""
    grid[row][col].no_fly = True


def RouteContainsCell(route: list[tuple[int, int]], cell: tuple[int, int]) -> bool:
    """Check if any remaining step in *route* passes through *cell*."""
    return cell in route


def FindAffectedDrones(
    drones: list[Drone],
    no_fly_cell: tuple[int, int],
) -> list[Drone]:
    """Return drones whose remaining route crosses *no_fly_cell*."""
    affected: list[Drone] = []
    for drone in drones:
        if drone.status in ("idle", "failed"):
            continue
        remaining = drone.planned_route[drone.route_step_index:]
        if RouteContainsCell(remaining, no_fly_cell):
            affected.append(drone)
    return affected


def DetermineRemainingTargets(
    drone: Drone,
    delivery: Delivery | None,
) -> list[tuple[int, int]]:
    """Return the remaining milestones the drone still needs to visit."""
    if delivery is None:
        return [] if drone.current_position == drone.home_hub else [drone.home_hub]

    if drone.current_target == delivery.pickup_cell:
        goals = [delivery.pickup_cell, delivery.dropoff_cell, drone.home_hub]
    elif drone.current_target == delivery.dropoff_cell:
        goals = [delivery.dropoff_cell, drone.home_hub]
    else:
        goals = [drone.home_hub]

    while goals and goals[0] == drone.current_position:
        goals.pop(0)
    return goals


def BuildRemainingRoute(
    start: tuple[int, int],
    goals: list[tuple[int, int]],
    grid: list[list[Cell]],
) -> AStarResult:
    """Plan a stitched route from *start* through each remaining goal in order."""
    if not goals:
        return AStarResult(path=[start], total_cost=0.0, success=True, message="Already at final target.")

    full_path: list[tuple[int, int]] = []
    total_cost = 0.0
    current = start

    for goal in goals:
        result = RunAStar(current, goal, grid)
        if not result.success:
            return result
        if full_path and result.path and result.path[0] == full_path[-1]:
            full_path.extend(result.path[1:])
        else:
            full_path.extend(result.path)
        total_cost += result.total_cost
        current = goal

    return AStarResult(path=full_path, total_cost=total_cost, success=True, message="Remaining route planned.")


def RerouteDrone(
    drone: Drone,
    delivery: Delivery | None,
    grid: list[list[Cell]],
) -> AStarResult:
    """
    Re-plan from drone's current position to its current_target.
    Falls back to hub if target is also blocked.
    """
    start = drone.current_position
    goals = DetermineRemainingTargets(drone, delivery)
    result = BuildRemainingRoute(start, goals, grid)
    if result.success:
        prior_path = drone.completed_path[:-1] if drone.completed_path else []
        drone.planned_route = prior_path + result.path
        drone.route_step_index = len(prior_path)
        drone.status = "rerouted"
    return result


def ForceReturnToHub(drone: Drone, grid: list[list[Cell]]) -> AStarResult:
    """Try to route the drone back to its home hub."""
    result = RunAStar(drone.current_position, drone.home_hub, grid)
    if result.success:
        prior_path = drone.completed_path[:-1] if drone.completed_path else []
        drone.planned_route = prior_path + result.path
        drone.route_step_index = len(prior_path)
        drone.status = "returning"
        drone.current_target = drone.home_hub
    else:
        drone.status = "failed"
    return result


def HandleDisruption(
    state: SimulationState,
    no_fly_row: int,
    no_fly_col: int,
) -> list[str]:
    """
    Activate a no-fly cell and reroute all affected drones.
    Returns list of event-log messages.
    """
    messages: list[str] = []
    no_fly_cell = (no_fly_row, no_fly_col)

    # store for visualization
    state.reroute_no_fly_cell = no_fly_cell

    # activate
    ActivateNoFlyCell(state.grid, no_fly_row, no_fly_col)
    messages.append(f"No-fly zone activated at ({no_fly_row},{no_fly_col}).")

    # find affected
    affected = FindAffectedDrones(state.drones, no_fly_cell)
    if not affected:
        messages.append("No active drones affected by the new no-fly zone.")
        return messages

    # attempt reroute for each
    for drone in affected:
        # find the delivery
        delivery = None
        for d in state.deliveries:
            if d.delivery_id == drone.assigned_delivery_id:
                delivery = d
                break

        old_route = drone.planned_route[drone.route_step_index:]
        state.old_route_for_reroute = old_route[:]

        result = RerouteDrone(drone, delivery, state.grid)
        if result.success:
            state.new_route_for_reroute = result.path[:]
            messages.append(
                f"Drone {drone.drone_id} rerouted successfully "
                f"({len(old_route)} -> {len(result.path)} steps)."
            )
        else:
            # try returning to hub
            hub_result = ForceReturnToHub(drone, state.grid)
            if hub_result.success:
                messages.append(
                    f"Drone {drone.drone_id} cannot reach target; returning to hub."
                )
            else:
                messages.append(
                    f"Drone {drone.drone_id} stranded - marked FAILED."
                )
            if delivery:
                delivery.status = "delayed"
                messages.append(f"Delivery {delivery.delivery_id} marked delayed.")

    return messages
