"""
AeroNet Lite - Utility Helpers
Miscellaneous helpers functions
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import random
from config import LOG_DIR, RANDOM_SEED
from grid_model import SimulationState, CreateSampleGrid
from astar_planner import CalculateRouteCost

logger = logging.getLogger(__name__)


def SetupLogging(level: int = logging.INFO) -> None:
    """Configure root logger for console + file output."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Use utf-8 for file, and wrap stdout to handle encoding errors
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    file_handler = logging.FileHandler(LOG_DIR / "aeronet.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handlers = [stream_handler, file_handler]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def SafeRun(fn, *args, fallback=None, label="operation", **kwargs):
    """Run *fn* and return its result; on failure return *fallback*."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logging.getLogger(__name__).warning("%s failed: %s", label, exc)
        return fallback


def LogEvent(state: SimulationState, message: str) -> None:
    entry = f"[Step {state.current_step:>2}] {message}"
    state.event_log.append(entry)
    logger.info(entry)


def SaveEventLog(event_log: list[str], path: Path | None = None) -> Path:
    p = path or (LOG_DIR / "simulation_event_log.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(event_log))
    return p


def InitializeSimulation() -> SimulationState:
    random.seed(RANDOM_SEED)
    grid = CreateSampleGrid()
    return SimulationState(grid=grid)


def GenerateSimulationSummary(state: SimulationState) -> dict:
    completed = sum(1 for d in state.deliveries if d.status == "completed")
    delayed   = sum(1 for d in state.deliveries if d.status == "delayed")
    failed    = sum(1 for d in state.deliveries if d.status == "failed")
    in_transit = sum(1 for d in state.deliveries if d.status == "in-transit")
    reroutes  = sum(1 for e in state.event_log if "rerouted" in e.lower())
    anomalies = sum(1 for e in state.event_log if "anomaly detected" in e.lower())
    total_cost = sum(d.route_cost for d in state.deliveries)

    summary = {
        "Completed": completed,
        "In-Transit": in_transit,
        "Delayed": delayed,
        "Failed": failed,
        "Reroutes": reroutes,
        "Anomalies": anomalies,
        "Total Route Cost": round(total_cost, 2),
    }

    print("\n" + "=" * 44)
    print("   AeroNet Lite - Final Summary")
    print("=" * 44)
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print()

    return summary
