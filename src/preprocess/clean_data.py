"""清理 PHM 2016 CMP 標籤，並診斷感測資料的冗餘欄位。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.make_dataset import load_labels, load_sensor_data


def clean_labels(
    labels: pd.DataFrame, lower_limit: float = 10.0, upper_limit: float = 300.0
) -> pd.DataFrame:
    """依 WAFER_ID/STAGE 去重，並排除開區間外的去除率異常值。"""
    if lower_limit >= upper_limit:
        raise ValueError("lower_limit 必須小於 upper_limit")
    cleaned = labels.drop_duplicates(subset=["WAFER_ID", "STAGE"]).copy()
    mask = cleaned["AVG_REMOVAL_RATE"].between(lower_limit, upper_limit, inclusive="neither")
    return cleaned.loc[mask].reset_index(drop=True)


def diagnose_redundant_columns(sensor_df: pd.DataFrame) -> dict[str, object]:
    """檢查識別欄位映射及耗材欄位共線性。"""
    machine_id_unique = sensor_df["MACHINE_ID"].nunique(dropna=False) if "MACHINE_ID" in sensor_df else None
    if {"MACHINE_DATA", "CHAMBER"}.issubset(sensor_df.columns):
        cross_tab = pd.crosstab(sensor_df["MACHINE_DATA"], sensor_df["CHAMBER"])
        active = cross_tab.gt(0)
        is_one_to_one = bool((active.sum(axis=1) <= 1).all() and (active.sum(axis=0) <= 1).all())
    else:
        cross_tab = pd.DataFrame()
        is_one_to_one = None
    usage_correlation = None
    usage_is_collinear = None
    usage_columns = ["USAGE_OF_BACKING_FILM", "USAGE_OF_PRESSURIZED_SHEET"]
    if set(usage_columns).issubset(sensor_df.columns):
        usage_correlation = float(sensor_df[usage_columns].corr().iloc[0, 1])
        usage_is_collinear = bool(abs(usage_correlation) >= 0.999999)
    return {
        "machine_id_unique": machine_id_unique,
        "machine_data_chamber_crosstab": cross_tab,
        "machine_data_chamber_one_to_one": is_one_to_one,
        "usage_correlation": usage_correlation,
        "usage_is_collinear": usage_is_collinear,
    }


def plot_cleaned_distribution(
    labels: pd.DataFrame, output_path: str | Path | None = None, show: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(labels["AVG_REMOVAL_RATE"], bins=50, kde=True, color="green", ax=ax)
    ax.set(title="Distribution of AVG_REMOVAL_RATE (Cleaned)", xlabel="Average Removal Rate", ylabel="Frequency")
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="時間序列 CSV 所在資料夾")
    parser.add_argument("--labels", required=True, help="原始標籤 CSV")
    parser.add_argument("--output", default="outputs/cleaned_labels.csv", help="清理後標籤 CSV")
    parser.add_argument("--plot", default="outputs/removal_rate_cleaned.png", help="分布圖輸出路徑")
    parser.add_argument("--lower-limit", type=float, default=10.0)
    parser.add_argument("--upper-limit", type=float, default=300.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    sensor_df = load_sensor_data(args.data_dir)
    labels = load_labels(args.labels)
    deduplicated_count = len(labels.drop_duplicates(subset=["WAFER_ID", "STAGE"]))
    cleaned = clean_labels(labels, args.lower_limit, args.upper_limit)
    diagnostics = diagnose_redundant_columns(sensor_df)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    plot_cleaned_distribution(cleaned, args.plot, args.show)

    print(f"去重後標籤: {deduplicated_count} 筆")
    print(f"清理後標籤: {len(cleaned)} 筆；排除 {deduplicated_count - len(cleaned)} 筆異常值")
    print(f"MACHINE_ID 類別數: {diagnostics['machine_id_unique']}")
    print(f"MACHINE_DATA 與 CHAMBER 是否一對一: {diagnostics['machine_data_chamber_one_to_one']}")
    print(f"BACKING_FILM / PRESSURIZED_SHEET 相關係數: {diagnostics['usage_correlation']}")
    print(f"BACKING_FILM / PRESSURIZED_SHEET 是否共線: {diagnostics['usage_is_collinear']}")
    print(f"已輸出: {output}")


if __name__ == "__main__":
    main()
