"""將 PHM 2016 CMP 時間序列轉換為晶圓／Stage 層級的模型特徵。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data.make_dataset import load_sensor_data


GROUP_KEYS = ["WAFER_ID", "STAGE", "CHAMBER"]
CONSUMABLE_COLUMNS = ["USAGE_OF_DRESSER", "USAGE_OF_POLISHING_TABLE", "USAGE_OF_DRESSER_TABLE", "USAGE_OF_MEMBRANE"]
COLLINEAR_PRESSURE_COLUMNS = ["CENTER_AIR_BAG_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE"]


def extract_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """從壓力、旋轉及時間差推導各製程狀態的持續時間。"""
    required = set(GROUP_KEYS + ["TIMESTAMP"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"特徵工程缺少必要欄位: {sorted(missing)}")

    result = df.sort_values(GROUP_KEYS + ["TIMESTAMP"]).copy()
    result["TIMESTAMP"] = pd.to_numeric(result["TIMESTAMP"], errors="coerce")
    time_diffs = result.groupby(GROUP_KEYS, sort=False)["TIMESTAMP"].diff().fillna(0).clip(lower=0)
    pressure_active = result.get("PRESSURIZED_CHAMBER_PRESSURE", pd.Series(0, index=result.index)).gt(0)
    head_rotating = result.get("HEAD_ROTATION", pd.Series(0, index=result.index)).gt(0.1)
    wafer_rotating = result.get("WAFER_ROTATION", pd.Series(0, index=result.index)).gt(0.1)
    stage_rotating = result.get("STAGE_ROTATION", pd.Series(0, index=result.index)).gt(0.1)
    polishing_rotation = wafer_rotating & stage_rotating
    any_rotation = head_rotating | wafer_rotating | stage_rotating
    polishing = pressure_active & polishing_rotation

    result["polishing_duration"] = time_diffs.where(polishing, 0)
    result["soaking_duration"] = time_diffs.where(pressure_active & ~polishing_rotation, 0)
    result["idle_duration"] = time_diffs.where(~pressure_active & ~any_rotation, 0)
    result["spinning_duration"] = time_diffs.where(~pressure_active & any_rotation, 0)
    for column in [c for c in result if "PRESSURE" in c and not c.startswith("polishing_")]:
        result[f"polishing_{column}"] = result[column].where(polishing, np.nan)
    return result


def nonzero_mean(values: pd.Series) -> float:
    selected = values[values > 0.1]
    return selected.mean() if not selected.empty else np.nan


def nonzero_std(values: pd.Series) -> float:
    selected = values[values > 0.1]
    return selected.std() if not selected.empty else np.nan


def nonzero_median(values: pd.Series) -> float:
    selected = values[values > 0.1]
    return selected.median() if not selected.empty else np.nan


def _aggregation_rules(df: pd.DataFrame) -> dict[str, list[object]]:
    rules: dict[str, list[object]] = {
        "USAGE_OF_DRESSER": ["max"], "USAGE_OF_POLISHING_TABLE": ["max"],
        "USAGE_OF_DRESSER_TABLE": ["max"], "USAGE_OF_MEMBRANE": ["max"],
        "polishing_duration": ["sum"], "soaking_duration": ["sum"],
        "idle_duration": ["sum"], "spinning_duration": ["sum"],
        "PRESSURIZED_CHAMBER_PRESSURE": [nonzero_mean, nonzero_std],
        "MAIN_OUTER_AIR_BAG_PRESSURE": [nonzero_mean, nonzero_std],
        "RETAINER_RING_PRESSURE": [nonzero_mean, nonzero_std],
        "SLURRY_FLOW_LINE_A": ["sum"], "SLURRY_FLOW_LINE_B": ["sum"], "SLURRY_FLOW_LINE_C": ["sum"],
        "WAFER_ROTATION": [nonzero_mean, nonzero_std],
        "STAGE_ROTATION": [nonzero_median, nonzero_std],
        "HEAD_ROTATION": [nonzero_mean, nonzero_std], "DRESSING_WATER_STATUS": ["mean"],
    }
    for column in ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "RETAINER_RING_PRESSURE"]:
        rules[f"polishing_{column}"] = ["mean", "std"]
    return {column: funcs for column, funcs in rules.items() if column in df.columns}


def _flatten_columns(columns: pd.MultiIndex) -> list[str]:
    flattened = []
    for variable, statistic, chamber in columns:
        statistic_name = statistic if isinstance(statistic, str) else statistic.__name__
        flattened.append(f"{variable}_{statistic_name}_Ch{chamber}")
    return flattened


def build_features(
    sensor_df: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    fit_scalers: bool = True,
    scalers: Mapping[str, MinMaxScaler] | None = None,
    lower_limit: float = 10.0,
    upper_limit: float = 300.0,
) -> tuple[pd.DataFrame, dict[str, MinMaxScaler]]:
    """建立模型特徵；訓練時回傳 fitted scalers，測試時應重用它們。"""
    df = sensor_df.drop(columns=COLLINEAR_PRESSURE_COLUMNS, errors="ignore")
    df = extract_physics_features(df)
    fitted_scalers = {} if fit_scalers else dict(scalers or {})
    for column in CONSUMABLE_COLUMNS:
        if column not in df:
            continue
        if fit_scalers:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[[column]]).ravel()
            fitted_scalers[column] = scaler
        elif column in fitted_scalers:
            df[column] = fitted_scalers[column].transform(df[[column]]).ravel()

    aggregated = df.groupby(GROUP_KEYS).agg(_aggregation_rules(df))
    features = aggregated.unstack("CHAMBER")
    features.columns = _flatten_columns(features.columns)

    if labels is not None:
        required = {"WAFER_ID", "STAGE", "AVG_REMOVAL_RATE"}
        missing = required.difference(labels.columns)
        if missing:
            raise ValueError(f"標籤缺少必要欄位: {sorted(missing)}")
        target = labels.drop_duplicates(["WAFER_ID", "STAGE"]).set_index(["WAFER_ID", "STAGE"])
        features = features.join(target[["AVG_REMOVAL_RATE"]], how="inner")

    start_time = df.groupby(["WAFER_ID", "STAGE"])["TIMESTAMP"].min().rename("START_TIMESTAMP")
    features = features.join(start_time).sort_values("START_TIMESTAMP")
    if fit_scalers and "AVG_REMOVAL_RATE" in features:
        features = features[features["AVG_REMOVAL_RATE"].between(lower_limit, upper_limit, inclusive="neither")]
    return features.reset_index(), fitted_scalers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="時間序列 CSV 資料夾")
    parser.add_argument("--labels", help="可選的 removal-rate 標籤 CSV")
    parser.add_argument("--output", default="outputs/cmp_features.csv")
    args = parser.parse_args()

    sensor_df = load_sensor_data(args.data_dir)
    labels = pd.read_csv(args.labels) if args.labels else None
    features, _ = build_features(sensor_df, labels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(f"特徵工程完成，維度: {features.shape}")
    print(f"已輸出: {output}")


if __name__ == "__main__":
    main()
