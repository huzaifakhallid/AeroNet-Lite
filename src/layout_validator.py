from dataclasses import dataclass

from grid_model import Cell, HOSPITAL, INDUSTRIAL, RESIDENTIAL, SCHOOL, get_cells_by_condition, get_neighbors, manhattan


@dataclass
class ValidationResult:
    is_valid: bool
    passed_rules: list[str]
    violations: list[str]


def check_industrial_safety(grid: list[list[Cell]]) -> list[str]:
    violations = []
    for cell in get_cells_by_condition(grid, lambda item: item.zone == INDUSTRIAL):
        for row, col in get_neighbors(cell.row, cell.col):
            neighbor = grid[row][col]
            if neighbor.zone in {SCHOOL, HOSPITAL}:
                violations.append(
                    f"R1: Industrial cell ({cell.row}, {cell.col}) is adjacent to {neighbor.zone} ({row}, {col}). "
                    "Suggested fix: add an Open Field buffer or move the facility."
                )
    return violations


def check_residential_coverage(grid: list[list[Cell]]) -> list[str]:
    residential_cells = get_cells_by_condition(grid, lambda item: item.zone == RESIDENTIAL)
    hubs = get_cells_by_condition(grid, lambda item: item.is_hub)
    violations = []

    if not hubs:
        return ["R2: No drone hubs exist. Suggested fix: add at least one hub near residential cells."]

    for cell in residential_cells:
        closest_distance = min(manhattan((cell.row, cell.col), (hub.row, hub.col)) for hub in hubs)
        if closest_distance > 3:
            violations.append(
                f"R2: Residential cell ({cell.row}, {cell.col}) is {closest_distance} cells away from the nearest hub. "
                "Suggested fix: add a hub within 3 cells or convert the cell to Open Field."
            )
    return violations


def check_hub_charging(grid: list[list[Cell]]) -> list[str]:
    hubs = get_cells_by_condition(grid, lambda item: item.is_hub)
    charging_pads = get_cells_by_condition(grid, lambda item: item.is_charging)
    violations = []

    if not charging_pads:
        return ["R3: No charging pads exist. Suggested fix: add charging pads near each hub."]

    for hub in hubs:
        closest_distance = min(manhattan((hub.row, hub.col), (pad.row, pad.col)) for pad in charging_pads)
        if closest_distance > 2:
            violations.append(
                f"R3: Hub ({hub.row}, {hub.col}) is {closest_distance} cells away from the nearest charging pad. "
                "Suggested fix: add charging within 2 cells of the hub."
            )
    return violations


def check_medical_access(grid: list[list[Cell]]) -> list[str]:
    hospitals = get_cells_by_condition(grid, lambda item: item.zone == HOSPITAL)
    medical_pickups = get_cells_by_condition(grid, lambda item: item.is_medical_pickup)

    for hospital in hospitals:
        for pickup in medical_pickups:
            if manhattan((hospital.row, hospital.col), (pickup.row, pickup.col)) <= 1:
                return []

    return [
        "R4: No hospital has a medical pickup point within 1 cell. "
        "Suggested fix: place a medical pickup beside at least one hospital."
    ]


def validate_layout(grid: list[list[Cell]]) -> ValidationResult:
    checks = [
        ("R1 Industrial safety", check_industrial_safety),
        ("R2 Residential coverage", check_residential_coverage),
        ("R3 Hub charging access", check_hub_charging),
        ("R4 Medical access", check_medical_access),
    ]

    passed_rules = []
    all_violations = []
    for rule_name, check in checks:
        violations = check(grid)
        if violations:
            all_violations.extend(violations)
        else:
            passed_rules.append(rule_name)

    return ValidationResult(is_valid=not all_violations, passed_rules=passed_rules, violations=all_violations)


def print_validation_report(result: ValidationResult) -> None:
    print("AeroNet Lite Layout Validation Report")
    print("=" * 42)
    print(f"Layout validity = {result.is_valid}")

    print("\nPassed rules:")
    for rule in result.passed_rules:
        print(f"- {rule}")

    if result.violations:
        print("\nFailed rules:")
        for violation in result.violations:
            print(f"- {violation}")
    else:
        print("\nFailed rules: None")
