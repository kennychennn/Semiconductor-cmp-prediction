"""Experiment input and artifact checks for the CMP pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


SENSOR_REQUIRED_COLUMNS = {"WAFER_ID", "STAGE", "CHAMBER", "TIMESTAMP"}
LABEL_REQUIRED_COLUMNS = {"WAFER_ID", "STAGE", "AVG_REMOVAL_RATE"}
OBSOLETE_DURATION_MARKERS = (
    "polishing_duration",
    "soaking_duration",
    "spinning_duration",
    "idle_duration",
)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少必要欄位: {missing}")


def _check_label_keys(labels: pd.DataFrame, name: str) -> None:
    _require_columns(labels, LABEL_REQUIRED_COLUMNS, name)
    key = ["WAFER_ID", "STAGE"]
    duplicate = labels[labels.duplicated(key, keep=False)]
    if not duplicate.empty:
        conflicts = duplicate.groupby(key, dropna=False)["AVG_REMOVAL_RATE"].nunique()
        conflicts = conflicts[conflicts > 1]
        if not conflicts.empty:
            raise ValueError(f"{name} 存在相同 WAFER_ID/STAGE 但目標值衝突的標籤")


def validate_inputs(
    train_dir: str | Path,
    test_dir: str | Path,
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
) -> dict[str, int]:
    """Validate raw folders and labels before feature engineering."""
    train_dir, test_dir = Path(train_dir), Path(test_dir)
    train_files = sorted(train_dir.glob("*.csv"))
    test_files = sorted(test_dir.glob("*.csv"))
    if not train_files:
        raise FileNotFoundError(f"找不到 training CSV: {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"找不到 test CSV: {test_dir}")
    for file in train_files:
        header = pd.read_csv(file, nrows=0)
        _require_columns(header, SENSOR_REQUIRED_COLUMNS, f"training sensor data ({file.name})")
    for file in test_files:
        header = pd.read_csv(file, nrows=0)
        _require_columns(header, SENSOR_REQUIRED_COLUMNS, f"test sensor data ({file.name})")
    _check_label_keys(train_labels, "training labels")
    _check_label_keys(test_labels, "test labels")
    return {
        "training_files": len(train_files),
        "test_files": len(test_files),
        "training_labels": len(train_labels),
        "test_labels": len(test_labels),
    }


def validate_feature_table(
    frame: pd.DataFrame,
    name: str,
    require_target: bool = True,
) -> dict[str, object]:
    """Validate the schema produced by ``build_features``."""
    _require_columns(frame, {"WAFER_ID", "STAGE", "START_TIMESTAMP"}, name)
    if require_target:
        _require_columns(frame, {"AVG_REMOVAL_RATE"}, name)
    key = ["WAFER_ID", "STAGE"]
    if frame.duplicated(key).any():
        raise ValueError(f"{name} 存在重複的 WAFER_ID/STAGE 模型樣本")
    obsolete = [
        column for column in frame.columns
        if any(marker in column.lower() for marker in OBSOLETE_DURATION_MARKERS)
    ]
    if obsolete:
        raise ValueError(f"{name} 仍含已淘汰的製程狀態 duration 欄位: {obsolete}")
    pressure_duration = [c for c in frame.columns if "pressure_duration" in c]
    if not pressure_duration:
        raise ValueError(f"{name} 找不到 pressure_duration 特徵")
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "pressure_duration_columns": pressure_duration,
        "duplicate_keys": int(frame.duplicated(key).sum()),
    }


def validate_config(config: Mapping[str, object]) -> None:
    """Validate the minimum configuration contract for a run."""
    for section in ("paths", "training", "experiment"):
        if section not in config:
            raise ValueError(f"config.yml 缺少 {section} 區段")
    models = config["experiment"].get("models", [])
    if not models:
        raise ValueError("experiment.models 不可為空")
    if config["training"].get("outer_splits", 0) < 2:
        raise ValueError("training.outer_splits 必須至少為 2")
    if config["training"].get("inner_splits", 0) < 2:
        raise ValueError("training.inner_splits 必須至少為 2")
    if config["training"].get("time_series_diagnostic", True) and config["training"].get("time_series_splits", 0) < 2:
        raise ValueError("training.time_series_splits 必須至少為 2")
