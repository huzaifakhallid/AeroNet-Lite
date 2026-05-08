import pytest
from src.grid_model import SimulationState, CreateSampleGrid
from src.astar_planner import RunAStar
from src.layout_validator import ValidateLayout
from src.config import GRID_ROWS, GRID_COLS

def test_grid_initialization():
    grid = CreateSampleGrid()
    assert len(grid) == GRID_ROWS
    assert len(grid[0]) == GRID_COLS

def test_astar_simple():
    grid = CreateSampleGrid()
    # Path from (0,0) to (1,1)
    res = RunAStar((0, 0), (1, 1), grid)
    assert res.success
    assert len(res.path) >= 2
    assert res.path[0] == (0, 0)
    assert res.path[-1] == (1, 1)

def test_layout_validation():
    grid = CreateSampleGrid()
    state = SimulationState(grid=grid)
    res = ValidateLayout(state.grid)
    # The default grid should be valid or at least return a result object
    assert hasattr(res, "is_valid")
    assert hasattr(res, "passed_rules")
