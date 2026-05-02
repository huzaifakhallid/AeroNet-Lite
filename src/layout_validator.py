"""
AeroNet Lite - CSP Layout Validator
Validates that the 10x10 grid satisfies four constraint-satisfaction-style rules.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from config import GRID_ROWS, GRID_COLS, HOSPITAL, INDUSTRIAL, RESIDENTIAL, SCHOOL, TABLE_DIR
from grid_model import Cell, GetAllCells, GetCellsByCondition, GetNeighbors, Manhattan


@dataclass
class ValidationResult:
    """Structured result of layout validation."""
    is_valid: bool = True
    passed_rules: list[str] = field(default_factory=list)
    failed_rules: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)   # {rule, cell, message, fix}


def CheckIndustrialSafety(grid: list[list[Cell]]) -> list[dict]:
    """R1 - Industrial cells must NOT be adjacent to School or Hospital."""
    violations: list[dict] = []
    for cell in GetCellsByCondition(grid, lambda c: c.zone == INDUSTRIAL):
        for r, c in GetNeighbors(cell.row, cell.col, GRID_ROWS, GRID_COLS):
            nb = grid[r][c]
            if nb.zone in {SCHOOL, HOSPITAL}:
                violations.append({
                    "rule": "R1",
                    "cell": f"({cell.row},{cell.col})",
                    "message": f"Industrial ({cell.row},{cell.col}) adjacent to {nb.zone} ({r},{c})",
                    "fix": "Add Open buffer or relocate facility.",
                })
    return violations


def CheckResidentialCoverage(grid: list[list[Cell]], max_distance: int = 3) -> list[dict]:
    """R2 - Every Residential cell within *max_distance* of a hub."""
    violations: list[dict] = []
    res_cells = GetCellsByCondition(grid, lambda c: c.zone == RESIDENTIAL)
    hubs = GetCellsByCondition(grid, lambda c: c.is_hub)
    if not hubs:
        return [{"rule": "R2", "cell": "N/A", "message": "No hubs exist.", "fix": "Add at least one hub."}]
    for cell in res_cells:
        closest = min(Manhattan((cell.row, cell.col), (h.row, h.col)) for h in hubs)
        if closest > max_distance:
            violations.append({
                "rule": "R2",
                "cell": f"({cell.row},{cell.col})",
                "message": f"Residential ({cell.row},{cell.col}) is {closest} from nearest hub (limit {max_distance}).",
                "fix": f"Add hub near ({cell.row},{cell.col}) or rezone to Open.",
            })
    return violations


def CheckHubCharging(grid: list[list[Cell]], max_distance: int = 2) -> list[dict]:
    """R3 - Every hub has a charging pad within *max_distance*."""
    violations: list[dict] = []
    hubs = GetCellsByCondition(grid, lambda c: c.is_hub)
    pads = GetCellsByCondition(grid, lambda c: c.is_charging)
    if not pads:
        return [{"rule": "R3", "cell": "N/A", "message": "No charging pads.", "fix": "Add pads near hubs."}]
    for hub in hubs:
        closest = min(Manhattan((hub.row, hub.col), (p.row, p.col)) for p in pads)
        if closest > max_distance:
            violations.append({
                "rule": "R3",
                "cell": f"({hub.row},{hub.col})",
                "message": f"Hub ({hub.row},{hub.col}) has nearest pad at distance {closest}.",
                "fix": f"Add charging pad within {max_distance} of ({hub.row},{hub.col}).",
            })
    return violations


def CheckMedicalAccess(grid: list[list[Cell]], max_distance: int = 1) -> list[dict]:
    """R4 - At least one hospital has a medical pickup within *max_distance*."""
    hospitals = GetCellsByCondition(grid, lambda c: c.zone == HOSPITAL)
    pickups   = GetCellsByCondition(grid, lambda c: c.is_medical_pickup)
    for hosp in hospitals:
        for pk in pickups:
            if Manhattan((hosp.row, hosp.col), (pk.row, pk.col)) <= max_distance:
                return []   # satisfied
    return [{
        "rule": "R4",
        "cell": "N/A",
        "message": "No hospital has a medical pickup within 1 cell.",
        "fix": "Place a medical pickup beside a hospital.",
    }]


def ValidateLayout(grid: list[list[Cell]]) -> ValidationResult:
    """Run all CSP rules and return a structured result."""
    checks = [
        ("R1 Industrial safety",  CheckIndustrialSafety),
        ("R2 Residential coverage", CheckResidentialCoverage),
        ("R3 Hub charging access",  CheckHubCharging),
        ("R4 Medical access",       CheckMedicalAccess),
    ]
    result = ValidationResult()
    for name, fn in checks:
        issues = fn(grid)
        if issues:
            result.failed_rules.append(name)
            result.violations.extend(issues)
        else:
            result.passed_rules.append(name)
    result.is_valid = len(result.violations) == 0
    return result


def PrintValidationReport(result: ValidationResult) -> None:
    """Pretty-print the validation report to stdout."""
    print("\n" + "=" * 44)
    print("   AeroNet Lite - Layout Validation")
    print("=" * 44)
    print(f"  Layout valid: {result.is_valid}")
    print(f"  Passed rules: {', '.join(result.passed_rules) or 'None'}")
    print(f"  Failed rules: {', '.join(result.failed_rules) or 'None'}")
    if result.violations:
        print("\n  Violations:")
        for v in result.violations:
            print(f"    [{v['rule']}] {v['message']}")
            print(f"           Fix: {v['fix']}")
    print()


def SaveValidationReport(result: ValidationResult, output_path: Path | None = None) -> Path:
    """Save violation details to CSV."""
    path = output_path or (TABLE_DIR / "validation_report.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rule", "cell", "message", "fix"])
        writer.writeheader()
        for v in result.violations:
            writer.writerow(v)
    # Also write summary row if no violations
    if not result.violations:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["rule", "cell", "message", "fix"])
            writer.writeheader()
            writer.writerow({"rule": "ALL", "cell": "N/A", "message": "All rules passed.", "fix": "N/A"})
    return path
