import pytest
from src.layout_validator import ValidateLayout
from src.grid_model import CreateSampleGrid, GetNeighbors, RESIDENTIAL, Manhattan

def test_manhattan_distance():
    assert Manhattan((0,0), (3,4)) == 7

def test_neighbors_center():
    nb = GetNeighbors(5, 5)
    assert len(nb) == 4

def test_neighbors_corner():
    nb = GetNeighbors(0, 0)
    assert len(nb) == 2

def test_hub_charging():
    grid = CreateSampleGrid()
    # Check if hubs have charging pads nearby
    pass

def test_industrial_adjacency():
    grid = CreateSampleGrid()
    # Industrial cells should not be next to residential (suggested rule)
    pass

def test_medical_access():
    grid = CreateSampleGrid()
    # Every medical pickup should be within X distance of something
    pass

def test_residential_coverage():
    grid = CreateSampleGrid()
    # Residential cells should be covered by hubs
    pass

def test_validate_layout_returns_result():
    grid = CreateSampleGrid()
    res = ValidateLayout(grid)
    assert hasattr(res, "is_valid")
    assert hasattr(res, "passed_rules")
