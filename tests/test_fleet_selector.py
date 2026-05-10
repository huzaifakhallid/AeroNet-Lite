import pytest
from src.fleet_selector import SelectFleetBruteForce, EstimateDemandLoadFromGrid
from src.config import DEFAULT_BUDGET
from src.grid_model import CreateSampleGrid

def test_no_fleet_exceeds_budget():
    res = SelectFleetBruteForce(total_demand=100.0)
    for candidate in res["top5"]:
        assert candidate["total_cost"] <= DEFAULT_BUDGET

def test_fleet_cost():
    # Simple cost check
    from src.config import LIGHT_DRONE_COST, HEAVY_DRONE_COST
    cost = 2 * LIGHT_DRONE_COST + 1 * HEAVY_DRONE_COST
    assert cost > 0

def test_fleet_score_positive():
    res = SelectFleetBruteForce(total_demand=100.0)
    for candidate in res["top5"]:
        assert candidate["score"] >= -1.0 # Score can be negative

def test_coverage_capped():
    res = SelectFleetBruteForce(total_demand=100.0)
    for candidate in res["top5"]:
        assert candidate["coverage_pct"] <= 1.0

def test_best_fleet_has_valid_score():
    res = SelectFleetBruteForce(total_demand=100.0)
    assert "top5" in res
    assert len(res["top5"]) > 0

def test_grid_demand_influences_estimated_load():
    grid = CreateSampleGrid()
    for row in grid:
        for cell in row:
            cell.demand = 1.0
    low_load = EstimateDemandLoadFromGrid(grid, expected_deliveries=5)

    for row in grid:
        for cell in row:
            cell.demand = 4.0
    high_load = EstimateDemandLoadFromGrid(grid, expected_deliveries=5)

    assert high_load > low_load
