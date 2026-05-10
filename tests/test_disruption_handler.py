import pytest
from src.disruption_handler import (
    ActivateNoFlyCell,
    RouteContainsCell,
    FindAffectedDrones,
    HandleDisruption,
    DetermineRemainingTargets,
    RerouteDrone,
)
from src.grid_model import CreateSampleGrid, Drone, SimulationState, Delivery

def test_activate_no_fly():
    grid = CreateSampleGrid()
    ActivateNoFlyCell(grid, 5, 5)
    assert grid[5][5].no_fly

def test_route_contains_cell():
    route = [(0,0), (0,1), (0,2)]
    assert RouteContainsCell(route, (0,1))
    assert not RouteContainsCell(route, (1,1))

def test_find_affected_drones():
    drone = Drone("D01", "light", 2.0, 10, 1000, (0,0), (0,0))
    drone.status = "en-route"
    drone.planned_route = [(0,1), (0,2)]
    affected = FindAffectedDrones([drone], (0,1))
    assert drone in affected

def test_handle_disruption():
    grid = CreateSampleGrid()
    state = SimulationState(grid=grid)
    drone = Drone("D01", "light", 2.0, 10, 1000, (0,0), (0,0))
    drone.status = "en-route"
    drone.planned_route = [(0,1), (0,2)]
    state.drones = [drone]
    
    # Block (0,1)
    ActivateNoFlyCell(state.grid, 0, 1)
    results = HandleDisruption(state, 0, 1)
    assert any("D01" in m for m in results)

def test_remaining_targets_advance_after_pickup():
    drone = Drone("D01", "light", 2.0, 10, 1000, (1, 1), (0, 0))
    delivery = Delivery("DEL-001", (1, 1), (2, 2), 1.0, "normal", "Docs")
    drone.current_target = delivery.dropoff_cell

    goals = DetermineRemainingTargets(drone, delivery)
    assert goals == [delivery.dropoff_cell, drone.home_hub]

def test_reroute_keeps_remaining_itinerary():
    grid = CreateSampleGrid()
    drone = Drone("D01", "light", 2.0, 10, 1000, (1, 1), (0, 0))
    delivery = Delivery("DEL-001", (1, 1), (2, 2), 1.0, "normal", "Docs")
    drone.current_target = delivery.dropoff_cell
    drone.completed_path = [(1, 1)]

    result = RerouteDrone(drone, delivery, grid)

    assert result.success
    assert delivery.dropoff_cell in result.path
    assert result.path[-1] == drone.home_hub
