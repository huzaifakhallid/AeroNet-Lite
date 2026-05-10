"""
AeroNet Lite - Data Loader
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import TRAIN_CSV, AMAZON_CSV, DENSITY_CSV, TELEMETRY_CSV

logger = logging.getLogger(__name__)


def LoadBikeSharingData(path: Path = TRAIN_CSV) -> pd.DataFrame | None:
    """Load bike-sharing demand data; returns None on failure."""
    if not path.exists():
        logger.warning("Bike-sharing CSV not found at %s", path)
        return None
    try:
        df = pd.read_csv(path)
        logger.info("Loaded bike-sharing data: %d rows", len(df))
        return df
    except Exception as exc:
        logger.error("Failed to load bike-sharing data: %s", exc)
        return None


def LoadAmazonDeliveryData(path: Path = AMAZON_CSV) -> pd.DataFrame | None:
    """Load Amazon delivery data; returns None on failure."""
    if not path.exists():
        logger.warning("Amazon delivery CSV not found at %s", path)
        return None
    try:
        df = pd.read_csv(path)
        # strip whitespace from string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
        logger.info("Loaded Amazon delivery data: %d rows", len(df))
        return df
    except Exception as exc:
        logger.error("Failed to load Amazon delivery data: %s", exc)
        return None


def LoadDensityData(path: Path = DENSITY_CSV) -> pd.DataFrame | None:
    """Load US city population density data."""
    if not path.exists():
        logger.warning("Density CSV not found at %s", path)
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        logger.info("Loaded density data: %d rows", len(df))
        return df
    except Exception as exc:
        logger.error("Failed to load density data: %s", exc)
        return None


def LoadTelemetryData(path: Path = TELEMETRY_CSV) -> pd.DataFrame | None:
    """
    Robustly load the drone telemetry CSV.
    Handles: bad lines, whitespace in headers, duplicates, etc.
    """
    if not path.exists():
        logger.warning("Telemetry CSV not found at %s", path)
        return None
    try:
        df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8-sig")
        # strip header whitespace
        df.columns = df.columns.str.strip()
        # drop fully-empty rows and duplicates
        df.dropna(how="all", inplace=True)
        df.drop_duplicates(inplace=True)
        logger.info("Loaded telemetry data: %d rows (after clean)", len(df))
        return df
    except Exception as exc:
        logger.error("Failed to load telemetry data: %s", exc)
        return None
