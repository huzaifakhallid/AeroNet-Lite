"""
AeroNet Lite - Anomaly Detection (ML)
Train classification models on drone telemetry to detect anomalies.
Rule-based labelling -> Decision Tree + Random Forest -> integration hooks.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_SEED, TELEMETRY_CSV, MODEL_DIR, TABLE_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)


def RepairDroneTelemetryCsvIfNeeded(path: Path) -> Path:
    """Placeholder - the pandas on_bad_lines='skip' handles most issues."""
    return path


def LoadDroneTelemetryData(path: Path = TELEMETRY_CSV) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Telemetry CSV not found: %s", path)
        return None
    try:
        RepairDroneTelemetryCsvIfNeeded(path)
        df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df.dropna(how="all", inplace=True)
        df.drop_duplicates(inplace=True)
        logger.info("Loaded telemetry data: %d rows", len(df))
        return df
    except Exception as exc:
        logger.error("Failed to load telemetry: %s", exc)
        return None


# Column name mapping (handle variations in the CSV)
_COL_MAP = {
    "Max Carry Weight (kg)":       "max_carry_weight",
    "Actual Carry Weight (kg)":    "actual_carry_weight",
    "Battery Remaining (%)":       "battery_remaining",
    "Altitude (meters)":           "altitude",
    "Flight Duration (minutes)":   "flight_duration",
    "Distance Flown (km)":         "distance_flown",
    "GPS Accuracy (meters)":       "gps_accuracy",
    "Wind Speed (m/s)":            "wind_speed",
    "Obstacles Encountered":       "obstacles_encountered",
    "Propeller Count":             "propeller_count",
    "Drone Size":                  "drone_size",
    "Payload Type":                "payload_type",
    "Flight Status":               "flight_status",
    "Notes":                       "notes",
}


def PreprocessDroneTelemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and coerce numerics."""
    df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})

    num_cols = [
        "max_carry_weight", "actual_carry_weight", "battery_remaining",
        "altitude", "flight_duration", "distance_flown",
        "gps_accuracy", "wind_speed", "obstacles_encountered", "propeller_count",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # payload ratio
    if "actual_carry_weight" in df.columns and "max_carry_weight" in df.columns:
        df["payload_ratio"] = (
            df["actual_carry_weight"] / df["max_carry_weight"].replace(0, np.nan)
        ).fillna(0)

    return df


def CreateAnomalyLabels(df: pd.DataFrame) -> pd.Series:
    """Rule-based multi-class anomaly labels, incorporating ground-truth status."""
    labels = pd.Series("Normal", index=df.index)

    # 1. Ground-truth from Flight Status or Notes
    if "flight_status" in df.columns:
        aborted = df["flight_status"].str.contains("Aborted|Unexpected", case=False, na=False)
        labels[aborted] = "Flight failure"
    
    if "notes" in df.columns:
        notes_fail = df["notes"].str.contains("failure|error|invalid", case=False, na=False)
        labels[notes_fail] = "System error"

    # 2. Rule-based augmentation
    # Payload anomaly
    if "actual_carry_weight" in df.columns and "max_carry_weight" in df.columns:
        mask = (df["actual_carry_weight"] > df["max_carry_weight"]) | (df["actual_carry_weight"] < 0)
        labels[mask] = "Payload anomaly"

    # Battery anomaly
    if "battery_remaining" in df.columns:
        labels[df["battery_remaining"] < 15] = "Battery anomaly"

    # Sensor anomaly
    if "gps_accuracy" in df.columns:
        labels[df["gps_accuracy"] > 50] = "Sensor anomaly"
    if "altitude" in df.columns:
        labels[(df["altitude"] < 0) | (df["altitude"] > 500)] = "Sensor anomaly"

    # Route / environment anomaly
    if "wind_speed" in df.columns:
        labels[df["wind_speed"] > 25] = "Route/environment anomaly"
    if "obstacles_encountered" in df.columns:
        # Check for explicit "Landed Unexpectedly" which might have shifted here in some rows
        obs_fail = df["obstacles_encountered"].astype(str).str.contains("Unexpected", case=False, na=False)
        labels[obs_fail] = "Route/environment anomaly"
        
    return labels


_FEATURE_COLS = [
    "propeller_count", "max_carry_weight", "actual_carry_weight",
    "payload_ratio", "altitude", "flight_duration", "distance_flown",
    "battery_remaining", "gps_accuracy", "wind_speed", "obstacles_encountered",
]


def TrainAnomalyModels(df: pd.DataFrame) -> dict:
    """Train DT baseline + RF on anomaly labels. Returns best model info."""
    df = PreprocessDroneTelemetry(df)
    df["anomaly_label"] = CreateAnomalyLabels(df)

    # Encode drone size if present
    if "drone_size" in df.columns:
        le_size = LabelEncoder()
        df["drone_size_enc"] = le_size.fit_transform(df["drone_size"].astype(str))
        feat_cols = _FEATURE_COLS + ["drone_size_enc"]
    else:
        feat_cols = _FEATURE_COLS[:]

    # Encode payload type if present
    if "payload_type" in df.columns:
        le_pt = LabelEncoder()
        df["payload_type_enc"] = le_pt.fit_transform(df["payload_type"].astype(str))
        feat_cols = feat_cols + ["payload_type_enc"]

    available = [c for c in feat_cols if c in df.columns]
    X = df[available].fillna(0)
    y = df["anomaly_label"]

    # Save processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "processed_anomaly.csv", index=False)

    # Drop classes with fewer than 2 samples (cannot stratify)
    label_counts = y.value_counts()
    rare_labels = label_counts[label_counts < 2].index.tolist()
    if rare_labels:
        mask = ~y.isin(rare_labels)
        X = X[mask]
        y = y[mask]
        logger.info("Dropped %d rare anomaly classes: %s", len(rare_labels), rare_labels)

    le_label = LabelEncoder()
    y_enc = le_label.fit_transform(y)
    class_names = list(le_label.classes_)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=RANDOM_SEED, stratify=y_enc
        )
    except ValueError:
        # Fallback: non-stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=RANDOM_SEED
        )

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }

    results: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

        results[name] = {
            "model": model,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "class_names": class_names,
            "y_test": y_test,
            "y_pred": y_pred,
            "features": available,
        }
        logger.info("%s -> Acc=%.4f  F1=%.4f", name, acc, f1)

    # pick best F1
    best_name = max(results, key=lambda n: results[n]["f1"])
    best = results[best_name]
    best["best_name"] = best_name
    best["all_results"] = results
    best["label_encoder"] = le_label
    return best


def PredictDroneAnomaly(model, label_encoder, drone_features: dict) -> str:
    """Predict anomaly class for a single drone's feature dict."""
    feat_cols = _FEATURE_COLS
    row = {c: drone_features.get(c, 0) for c in feat_cols}
    X = pd.DataFrame([row]).fillna(0)
    pred_enc = model.predict(X)[0]
    return label_encoder.inverse_transform([pred_enc])[0]


def SaveAnomalyModel(model, path: Path | None = None) -> Path:
    p = path or (MODEL_DIR / "anomaly_model.pkl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(model, fh)
    return p


def SaveAnomalyMetrics(metrics: dict, output_path: Path | None = None) -> Path:
    p = output_path or (TABLE_DIR / "anomaly_model_metrics.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, res in metrics.get("all_results", {}).items():
        rows.append({
            "model": name,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
        })
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def EvaluateAnomalyModel(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
    }
