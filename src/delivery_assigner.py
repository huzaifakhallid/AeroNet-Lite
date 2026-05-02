"""
AeroNet Lite - Delivery Assigner
Assigns deliveries to compatible drones based on proximity and payload.
"""

from __future__ import annotations

import csv
from pathlib import Path

from config import TABLE_DIR
from grid_model import Cell, Delivery, Drone, Manhattan


def IsDroneCompatible(delivery: Delivery, drone: Drone) -> bool:
    """Check payload compatibility."""
    return drone.payload_capacity >= delivery.weight


def FindNearestAvailableDrone(
    delivery: Delivery,
    drones: list[Drone],
) -> Drone | None:
    """
    Choose the nearest idle, compatible drone.
    Prefer light drones for light packages, heavy for medical/heavy.
    """
    available = [d for d in drones if d.status == "idle" and IsDroneCompatible(delivery, d)]
    if not available:
        return None

    # for medical / heavy, prefer heavy drones
    if delivery.priority == "medical" or delivery.weight > 2.0:
        heavy = [d for d in available if d.drone_type == "heavy"]
        if heavy:
            available = heavy

    # for light packages prefer light drones
    if delivery.weight <= 2.0 and delivery.priority != "medical":
        light = [d for d in available if d.drone_type == "light"]
        if light:
            available = light

    # sort by Manhattan distance to pickup
    available.sort(key=lambda d: Manhattan(d.current_position, delivery.pickup_cell))
    return available[0]


def AssignDeliveriesToDrones(
    deliveries: list[Delivery],
    drones: list[Drone],
    grid: list[list[Cell]],
) -> list[Delivery]:
    """Assign each delivery to the best available drone."""
    for delivery in deliveries:
        if delivery.status != "pending":
            continue
        drone = FindNearestAvailableDrone(delivery, drones)
        if drone is None:
            delivery.status = "delayed"
            continue
        delivery.assigned_drone_id = drone.drone_id
        delivery.status = "assigned"
        drone.status = "assigned"
        drone.assigned_delivery_id = delivery.delivery_id
    return deliveries


def SaveDeliveryAssignments(deliveries: list[Delivery], output_path: Path | None = None) -> Path:
    path = output_path or (TABLE_DIR / "delivery_assignments.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["delivery_id", "pickup_cell", "dropoff_cell", "weight",
                  "priority", "category", "status", "assigned_drone_id", "route_cost"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for d in deliveries:
            writer.writerow({
                "delivery_id": d.delivery_id,
                "pickup_cell": d.pickup_cell,
                "dropoff_cell": d.dropoff_cell,
                "weight": d.weight,
                "priority": d.priority,
                "category": d.category,
                "status": d.status,
                "assigned_drone_id": d.assigned_drone_id or "",
                "route_cost": d.route_cost,
            })
    return path
