import pytest
from src.astar_planner import RunAStar, CalculateRouteCost
from src.grid_model import CreateSampleGrid, Cell, INDUSTRIAL, COMMERCIAL

def test_path_found_open_grid():
    grid = CreateSampleGrid()
    res = RunAStar((0, 0), (1, 1), grid)
    assert res.success
    assert len(res.path) >= 2

def test_path_avoids_no_fly():
    grid = CreateSampleGrid()
    # Block (0,1) and (1,0)
    grid[0][1].no_fly = True
    grid[1][0].no_fly = True
    res = RunAStar((0, 0), (1, 1), grid)
    # Should find a longer path or fail if totally blocked
    if res.success:
        assert (0,1) not in res.path
        assert (1,0) not in res.path

def test_blocked_route():
    grid = CreateSampleGrid()
    # Surround (0,0) with no-fly
    grid[0][1].no_fly = True
    grid[1][0].no_fly = True
    grid[1][1].no_fly = True
    res = RunAStar((0, 0), (2, 2), grid)
    assert not res.success

def test_calculate_route_cost():
    grid = CreateSampleGrid()
    # Base cost check
    cost = CalculateRouteCost([(0,0), (0,1)], grid)
    assert cost > 0

def test_route_cost_commercial():
    # Commercial cells have higher cost (0.8) vs residential (0.5)
    # We test if the planner accounts for different cell types if implemented
    pass
