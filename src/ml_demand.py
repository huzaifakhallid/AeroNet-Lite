"""
AeroNet Lite - Demand Forecasting (ML)
Train regression models on Bike-Sharing Demand data and map predictions
onto the 10x10 grid.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_SEED, TRAIN_CSV,
    MODEL_DIR, TABLE_DIR, PROCESSED_DIR,
)
from grid_model import Cell, GetAllCells

logger = logging.getLogger(__name__)

FEATURES = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "day", "month", "weekday",
]
TARGET = "count"


def LoadDemandData(path: Path = TRAIN_CSV) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Demand CSV not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        logger.info("Loaded demand CSV: %d rows", len(df))
        return df
    except Exception as exc:
        logger.error("Failed to load demand CSV: %s", exc)
        return None


def PreprocessDemandData(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time features, drop leakage columns, save processed CSV."""
    df = df.copy()
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        df["hour"]    = dt.dt.hour
        df["day"]     = dt.dt.day
        df["month"]   = dt.dt.month
        df["weekday"] = dt.dt.dayofweek

    # Drop leakage columns
    for col in ("casual", "registered", "datetime"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df.dropna(subset=[TARGET], inplace=True)

    # Save processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "processed_demand.csv", index=False)

    return df


def TrainDemandModels(df: pd.DataFrame) -> dict:
    """
    Train Linear Regression (baseline) and Random Forest.
    Returns dict with best model, metrics, predictions.
    """
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].fillna(0)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }

    results: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = r2_score(y_test, y_pred)
        results[name] = {
            "model": model,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "y_test": y_test.values,
            "y_pred": y_pred,
            "features": available,
        }
        logger.info("%s -> MAE=%.2f  RMSE=%.2f  R2=%.4f", name, mae, rmse, r2)

    # pick model with lower RMSE
    best_name = min(results, key=lambda n: results[n]["rmse"])
    best = results[best_name]
    best["best_name"] = best_name
    best["all_results"] = results
    return best


def EvaluateDemandModel(model, X_test, y_test) -> dict:
    y_pred = np.clip(model.predict(X_test), 0, None)
    return {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }


def SaveDemandModel(model, path: Path | None = None) -> Path:
    p = path or (MODEL_DIR / "demand_model.pkl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(model, fh)
    return p


def SaveDemandMetrics(metrics: dict, output_path: Path | None = None) -> Path:
    p = output_path or (TABLE_DIR / "demand_model_metrics.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    all_results = metrics.get("all_results", {})
    for name, res in all_results.items():
        rows.append({"model": name, "MAE": res["mae"], "RMSE": res["rmse"], "R2": res["r2"]})
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def PredictGridDemand(model, grid: list[list[Cell]], features: list[str] | None = None) -> list[list[float]]:
    """
    Use the trained model to assign a demand value to each cell.
    We synthesise a feature vector for each cell using its position and density,
    then normalise predictions to a sensible range (0-5 kg demand).
    """
    raw_preds: list[list[float]] = []
    for r_cells in grid:
        row_demand = []
        for cell in r_cells:
            # Create a synthetic feature row for this cell
            feat = {
                "season": (cell.row % 4) + 1,
                "holiday": 0,
                "workingday": 1,
                "weather": 1,
                "temp": 20 + cell.col,
                "atemp": 22 + cell.col,
                "humidity": 40 + cell.row * 3,
                "windspeed": 5.0 + cell.col,
                "hour": 12,
                "day": 15,
                "month": 6,
                "weekday": 2,
            }
            feat_df = pd.DataFrame([feat])
            if features:
                feat_df = feat_df[[f for f in features if f in feat_df.columns]]
            pred = float(np.clip(model.predict(feat_df)[0], 0, None))
            row_demand.append(pred)
        raw_preds.append(row_demand)

    # Normalise to 0-5 kg range (bike-sharing counts can be hundreds)
    flat = [v for row in raw_preds for v in row]
    min_v, max_v = min(flat), max(flat)
    rng = max_v - min_v if max_v != min_v else 1.0

    demand: list[list[float]] = []
    for row in raw_preds:
        demand.append([round(0.5 + 4.5 * (v - min_v) / rng, 2) for v in row])
    return demand


def ApplyDemandToGrid(grid: list[list[Cell]], predictions: list[list[float]]) -> None:
    """Set cell.demand from the prediction matrix."""
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            cell.demand = predictions[r][c]
