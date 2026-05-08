from dataclasses import dataclass
from pathlib import Path
import csv


GRID_SIZE = 10

RESIDENTIAL = "Residential"
COMMERCIAL = "Commercial"
HOSPITAL = "Hospital"
SCHOOL = "School"
INDUSTRIAL = "Industrial"
OPEN_FIELD = "Open Field"

DENSITY_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "us_city_pop_density" / "uscitypopdensity.csv"


@dataclass
class Cell:
    row: int
    col: int
    zone: str
    density: int
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(row: int, col: int) -> list[tuple[int, int]]:
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [(r, c) for r, c in candidates if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE]


def load_density_values(path: Path = DENSITY_FILE) -> list[int]:
    """Read city density values; fall back to simple defaults if the CSV is missing."""
    if not path.exists():
        return [1200, 3500, 7500]

    values: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        density_column = next((name for name in reader.fieldnames or [] if "Density" in name), None)
        if density_column is None:
            return [1200, 3500, 7500]

        for row in reader:
            raw_value = row.get(density_column, "").replace(",", "").strip()
            if raw_value.isdigit():
                values.append(int(raw_value))

    return values or [1200, 3500, 7500]


def density_tier_value(tier: str, values: list[int]) -> int:
    low = [value for value in values if value < 2000]
    medium = [value for value in values if 2000 <= value <= 6000]
    high = [value for value in values if value > 6000]
    buckets = {"low": low, "medium": medium, "high": high}
    bucket = buckets.get(tier.lower(), medium) or values
    return int(sum(bucket) / len(bucket))


def create_sample_grid() -> list[list[Cell]]:
    densities = load_density_values()
    density_by_zone = {
        RESIDENTIAL: density_tier_value("medium", densities),
        COMMERCIAL: density_tier_value("high", densities),
        HOSPITAL: density_tier_value("medium", densities),
        SCHOOL: density_tier_value("medium", densities),
        INDUSTRIAL: density_tier_value("low", densities),
        OPEN_FIELD: density_tier_value("low", densities),
    }

    zone_rows = [
        [RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL],
        [RESIDENTIAL, HOSPITAL, COMMERCIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD, RESIDENTIAL, SCHOOL, COMMERCIAL, COMMERCIAL],
        [RESIDENTIAL, RESIDENTIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD, RESIDENTIAL, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL],
        [COMMERCIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD],
        [COMMERCIAL, COMMERCIAL, OPEN_FIELD, INDUSTRIAL, INDUSTRIAL, OPEN_FIELD, COMMERCIAL, COMMERCIAL, RESIDENTIAL, RESIDENTIAL],
        [OPEN_FIELD, OPEN_FIELD, OPEN_FIELD, INDUSTRIAL, INDUSTRIAL, OPEN_FIELD, OPEN_FIELD, COMMERCIAL, RESIDENTIAL, RESIDENTIAL],
        [RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL],
        [RESIDENTIAL, SCHOOL, COMMERCIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD, RESIDENTIAL, HOSPITAL, COMMERCIAL, COMMERCIAL],
        [RESIDENTIAL, RESIDENTIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD, RESIDENTIAL, RESIDENTIAL, RESIDENTIAL, COMMERCIAL, COMMERCIAL],
        [OPEN_FIELD, OPEN_FIELD, COMMERCIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD, COMMERCIAL, COMMERCIAL, OPEN_FIELD, OPEN_FIELD],
    ]

    hubs = {(0, 3), (1, 7), (3, 0), (3, 5), (5, 9), (6, 6), (7, 1), (9, 7)}
    charging_pads = {(0, 4), (1, 8), (3, 1), (3, 6), (5, 8), (6, 5), (7, 2), (9, 6)}
    medical_pickups = {(1, 2), (7, 6)}

    grid: list[list[Cell]] = []
    for row_index, row_zones in enumerate(zone_rows):
        grid_row = []
        for col_index, zone in enumerate(row_zones):
            density = density_by_zone[zone]
            grid_row.append(
                Cell(
                    row=row_index,
                    col=col_index,
                    zone=zone,
                    density=density,
                    is_hub=(row_index, col_index) in hubs,
                    is_charging=(row_index, col_index) in charging_pads,
                    is_medical_pickup=(row_index, col_index) in medical_pickups,
                    demand=round(density / 1000, 2),
                )
            )
        grid.append(grid_row)

    return grid


def get_cells_by_condition(grid: list[list[Cell]], condition) -> list[Cell]:
    return [cell for row in grid for cell in row if condition(cell)]
