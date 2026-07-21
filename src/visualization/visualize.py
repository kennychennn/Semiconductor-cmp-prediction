"""執行 PHM 2016 CMP 探索性分析並輸出圖表與相關係數。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.make_dataset import load_labels, load_sensor_data
from src.preprocess.clean_data import clean_labels


def group_numeric_features(sensor_df: pd.DataFrame) -> dict[str, list[str]]:
    """依 notebook 規則將數值欄位分為壓力、流量、耗材、轉速與其他。"""
    excluded = {"WAFER_ID", "TIMESTAMP", "CHAMBER", "STAGE", "DRESSING_WATER_STATUS"}
    numeric = [c for c in sensor_df.columns if c not in excluded and pd.api.types.is_numeric_dtype(sensor_df[c])]
    groups = {
        "air_bag": [c for c in numeric if "AIR_BAG" in c],
        "slurry": [c for c in numeric if "SLURRY" in c],
        "usage": [c for c in numeric if "USAGE" in c],
        "rotation": [c for c in numeric if "ROTATION" in c],
    }
    categorized = set().union(*groups.values())
    groups["other"] = [c for c in numeric if c not in categorized]
    return groups


def plot_feature_groups(sensor_df: pd.DataFrame, output_dir: str | Path, show: bool = False) -> None:
    """將各類特徵對 TIMESTAMP 的散點圖分別存檔。"""
    if "TIMESTAMP" not in sensor_df:
        raise ValueError("感測資料缺少 TIMESTAMP")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = sensor_df.copy()
    data["TIMESTAMP"] = pd.to_numeric(data["TIMESTAMP"], errors="coerce")
    data = data.sort_values("TIMESTAMP")

    for name, columns in group_numeric_features(data).items():
        if not columns:
            continue
        ncols = 2 if name == "air_bag" else 1 if name == "slurry" else 3
        nrows = int(np.ceil(len(columns) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
        flat_axes = axes.ravel()
        for ax, column in zip(flat_axes, columns):
            ax.scatter(data["TIMESTAMP"], data[column], s=1, color="teal", alpha=0.3)
            ax.set(title=column, xlabel="TIMESTAMP", ylabel="Value")
            ax.grid(linestyle="--", alpha=0.6)
        for ax in flat_axes[len(columns):]:
            ax.remove()
        fig.suptitle(f"{name.replace('_', ' ').title()} Variables vs TIMESTAMP", fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_vs_timestamp.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def feature_target_correlations(sensor_df: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """以晶圓平均值計算數值特徵和 AVG_REMOVAL_RATE 的 Pearson 相關係數。"""
    wafer_features = sensor_df.groupby("WAFER_ID").mean(numeric_only=True)
    wafer_target = labels.groupby("WAFER_ID")["AVG_REMOVAL_RATE"].mean()
    joined = wafer_features.join(wafer_target, how="inner")
    return joined.corr(numeric_only=True)["AVG_REMOVAL_RATE"].drop("AVG_REMOVAL_RATE").sort_values()


def plot_correlation_matrix(df: pd.DataFrame, columns: list[str], title: str, output: Path) -> None:
    if len(columns) < 2:
        return
    matrix = df[columns].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(columns)), max(6, len(columns) * 0.8)))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(sensor_df: pd.DataFrame, labels: pd.DataFrame, output_dir: str | Path, show: bool = False) -> pd.Series:
    """執行所有 EDA，回傳特徵與目標值相關係數。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_feature_groups(sensor_df, output_dir, show)

    pressure_columns = [c for c in ["MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE"] if c in sensor_df]
    usage_columns = [c for c in sensor_df.columns if "USAGE" in c]
    plot_correlation_matrix(sensor_df, pressure_columns, "Correlation Matrix of Air Bag Pressures", output_dir / "air_bag_correlation.png")
    plot_correlation_matrix(sensor_df, usage_columns, "Correlation Matrix of Usage Variables", output_dir / "usage_correlation.png")

    correlations = feature_target_correlations(sensor_df, labels)
    correlations.rename("pearson_correlation").to_csv(output_dir / "feature_target_correlations.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["indianred" if value < 0 else "steelblue" for value in correlations]
    correlations.plot.barh(color=colors, width=0.8, ax=ax)
    ax.set(title="Feature Correlation with AVG_REMOVAL_RATE", xlabel="Pearson Correlation", ylabel="Features")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_target_correlations.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return correlations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", default="outputs/eda")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    sensor_df = load_sensor_data(args.data_dir)
    labels = clean_labels(load_labels(args.labels))
    correlations = run_eda(sensor_df, labels, args.output_dir, args.show)
    print("各變數與 AVG_REMOVAL_RATE 的相關係數（由高至低）:")
    print(correlations.sort_values(ascending=False).to_string())
    print(f"圖表與表格已輸出至: {args.output_dir}")


if __name__ == "__main__":
    main()
