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
ROTATION_TIMING_COLUMNS = [
    "wafer_start_time", "head_start_time", "stage_start_time",
    "head_after_wafer_delay", "stage_after_head_delay", "stage_spinup_duration",
    "head_peak_rotation", "stage_peak_rotation",
]


def extract_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """從壓力、旋轉及時間差推導各製程狀態的持續時間。"""
    required = set(GROUP_KEYS + ["TIMESTAMP"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"特徵工程缺少必要欄位: {sorted(missing)}")

    result = df.sort_values(["WAFER_ID", "CHAMBER", "STAGE", "TIMESTAMP"]).copy()
    time_diffs = result["TIMESTAMP"].diff().fillna(0)
    boundary = (
        result["WAFER_ID"].ne(result["WAFER_ID"].shift())
        | result["CHAMBER"].ne(result["CHAMBER"].shift())
        | result["STAGE"].ne(result["STAGE"].shift())
    )
    time_diffs[boundary] = 0
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


def rotation_timing_features(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Extract wafer-stage level rotation startup and spin-up timing features.

    Times are elapsed from the first timestamp of each wafer-stage, so absolute
    dataset chronology is not accidentally used as a process feature.  A start
    time is the first observed sample above ``threshold``.  Stage spin-up is the
    elapsed time from stage activation to the first sample reaching 90% of its
    observed positive peak; it is NaN when stage rotation is never observed.
    """
    required = {"WAFER_ID", "STAGE", "TIMESTAMP"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"時序特徵缺少必要欄位: {sorted(missing)}")
    rows = []
    for (wafer_id, stage), group in df.groupby(["WAFER_ID", "STAGE"], sort=False):
        group = group.sort_values("TIMESTAMP")
        timestamps = pd.to_numeric(group["TIMESTAMP"], errors="coerce")
        t0 = timestamps.min()
        values = {}
        starts = {}
        peaks = {}
        for column in ["WAFER_ROTATION", "HEAD_ROTATION", "STAGE_ROTATION"]:
            if column not in group:
                starts[column] = np.nan
                peaks[column] = np.nan
                continue
            signal = pd.to_numeric(group[column], errors="coerce")
            active = signal > threshold
            if not active.any():
                starts[column] = np.nan
                peaks[column] = np.nan
                continue
            starts[column] = float(timestamps.loc[active].iloc[0] - t0)
            peaks[column] = float(signal.loc[active].max())
        stage_spinup = np.nan
        stage_start = starts.get("STAGE_ROTATION", np.nan)
        stage_peak = peaks.get("STAGE_ROTATION", np.nan)
        if pd.notna(stage_start) and pd.notna(stage_peak):
            signal = pd.to_numeric(group["STAGE_ROTATION"], errors="coerce")
            reached_peak = signal >= stage_peak * 0.9
            if reached_peak.any():
                reached_time = float(timestamps.loc[reached_peak].iloc[0] - t0)
                stage_spinup = max(0.0, reached_time - stage_start)
        wafer_start = starts.get("WAFER_ROTATION", np.nan)
        head_start = starts.get("HEAD_ROTATION", np.nan)
        rows.append({
            "WAFER_ID": wafer_id,
            "STAGE": stage,
            "wafer_start_time": wafer_start,
            "head_start_time": head_start,
            "stage_start_time": stage_start,
            "head_after_wafer_delay": head_start - wafer_start if pd.notna(head_start) and pd.notna(wafer_start) else np.nan,
            "stage_after_head_delay": stage_start - head_start if pd.notna(stage_start) and pd.notna(head_start) else np.nan,
            "stage_spinup_duration": stage_spinup,
            "head_peak_rotation": peaks.get("HEAD_ROTATION", np.nan),
            "stage_peak_rotation": stage_peak,
        })
    return pd.DataFrame(rows).set_index(["WAFER_ID", "STAGE"])[ROTATION_TIMING_COLUMNS]


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
        rules[f"polishing_{column}"] = [np.nanmean, np.nanstd]
    return {column: funcs for column, funcs in rules.items() if column in df.columns}


def _flatten_columns(columns: pd.MultiIndex) -> list[str]:
    flattened = []
    for variable, statistic, chamber in columns:
        statistic_name = statistic if isinstance(statistic, str) else statistic.__name__
        flattened.append(f"{variable}_{statistic_name}_Ch{int(chamber)}")
    return flattened


def build_features(
    df_raw: pd.DataFrame,
    label_path: str | Path | pd.DataFrame | None = None,
    is_train: bool = True,
    scalers: Mapping[str, MinMaxScaler] | None = None,
) -> tuple[pd.DataFrame, dict[str, MinMaxScaler]]:
    """使用與 notebook cell 14 相同的介面與運算建立模型特徵。"""
    df = df_raw.drop(columns=COLLINEAR_PRESSURE_COLUMNS, errors="ignore")
    df = extract_physics_features(df)
    fitted_scalers = {} if is_train else dict(scalers or {})
    for column in CONSUMABLE_COLUMNS:
        if column not in df:
            continue
        if is_train:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[[column]]).ravel()
            fitted_scalers[column] = scaler
        elif column in fitted_scalers:
            df[column] = fitted_scalers[column].transform(df[[column]]).ravel()

    aggregated = df.groupby(GROUP_KEYS).agg(_aggregation_rules(df))
    features = aggregated.unstack("CHAMBER")
    features.columns = _flatten_columns(features.columns)
    timing = rotation_timing_features(df)
    features = features.join(timing, how="left", validate="one_to_one")

    if isinstance(label_path, pd.DataFrame):
        labels = label_path
    elif label_path is not None and Path(label_path).exists():
        labels = pd.read_csv(label_path)
    else:
        labels = None
    if labels is not None:
        required = {"WAFER_ID", "STAGE", "AVG_REMOVAL_RATE"}
        missing = required.difference(labels.columns)
        if missing:
            raise ValueError(f"標籤缺少必要欄位: {sorted(missing)}")
        target = labels.drop_duplicates(["WAFER_ID", "STAGE"]).set_index(["WAFER_ID", "STAGE"])
        features = features.join(target[["AVG_REMOVAL_RATE"]], how="inner")

    start_time = df.groupby(["WAFER_ID", "STAGE"])["TIMESTAMP"].min().rename("START_TIMESTAMP")
    features = features.join(start_time).sort_values("START_TIMESTAMP")
    if is_train and "AVG_REMOVAL_RATE" in features:
        features = features[
            (features["AVG_REMOVAL_RATE"] < 300)
            & (features["AVG_REMOVAL_RATE"] > 10)
        ]
    return features.reset_index(), fitted_scalers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="時間序列 CSV 資料夾")
    parser.add_argument("--labels", help="可選的 removal-rate 標籤 CSV")
    parser.add_argument("--output", default="outputs/cmp_features.csv")
    parser.add_argument("--test-data-dir")
    parser.add_argument("--test-labels")
    parser.add_argument("--test-output")
    args = parser.parse_args()

    sensor_df = load_sensor_data(args.data_dir)
    features, fitted_scalers = build_features(sensor_df, args.labels, is_train=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(f"特徵工程完成，維度: {features.shape}")
    print(f"已輸出: {output}")
    if args.test_data_dir:
        test_sensor_df = load_sensor_data(args.test_data_dir)
        test_features, _ = build_features(
            test_sensor_df,
            args.test_labels,
            is_train=False,
            scalers=fitted_scalers,
        )
        test_output = Path(args.test_output or "outputs/cmp_test_features.csv")
        test_output.parent.mkdir(parents=True, exist_ok=True)
        test_features.to_csv(test_output, index=False)
        print(f"Test set 特徵工程完成，維度: {test_features.shape}")
        print(f"已輸出: {test_output}")


if __name__ == "__main__":
    main()
