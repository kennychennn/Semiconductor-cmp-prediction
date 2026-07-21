"""載入 PHM 2016 CMP 時間序列資料並產生基本摘要。

可作為模組引用，也可直接由命令列執行。此版本將 notebook 中寫死的
Windows 路徑改為參數，方便重現與測試。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REQUIRED_SENSOR_COLUMNS = {"WAFER_ID", "STAGE", "CHAMBER"}
REQUIRED_LABEL_COLUMNS = {"WAFER_ID", "STAGE", "AVG_REMOVAL_RATE"}


def _validate_columns(df: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source} 缺少必要欄位: {sorted(missing)}")


def load_sensor_data(data_dir: str | Path) -> pd.DataFrame:
    """合併資料夾內所有製程 CSV，回傳單一 DataFrame。"""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 找不到 CSV 檔案")

    frames = [pd.read_csv(file) for file in files]
    nonempty_frames = [frame for frame in frames if not frame.empty]
    if not nonempty_frames:
        raise ValueError(f"{data_dir} 中的 CSV 均不含資料列")
    frame = pd.concat(nonempty_frames, ignore_index=True)
    _validate_columns(frame, REQUIRED_SENSOR_COLUMNS, str(data_dir))
    return frame


def load_labels(label_path: str | Path) -> pd.DataFrame:
    """讀取材料去除率標籤。"""
    labels = pd.read_csv(label_path)
    _validate_columns(labels, REQUIRED_LABEL_COLUMNS, str(label_path))
    return labels


def summarize_data(sensor_df: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    """計算 notebook 2.1 節使用的摘要統計。"""
    return {
        "sensor_rows": len(sensor_df),
        "sensor_columns": sensor_df.shape[1],
        "label_rows": len(labels),
        "unique_wafers": sensor_df["WAFER_ID"].nunique(),
        "stage_chamber_distribution": (
            sensor_df.groupby(["STAGE", "CHAMBER"], dropna=False)
            .size()
            .reset_index(name="data_points")
        ),
        "removal_rate_statistics": labels["AVG_REMOVAL_RATE"].describe(),
    }


def plot_removal_rate_distribution(
    labels: pd.DataFrame, output_path: str | Path | None = None, show: bool = False
) -> None:
    """繪製原始 AVG_REMOVAL_RATE 分布。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(labels["AVG_REMOVAL_RATE"], bins=50, kde=True, color="teal", ax=ax)
    ax.set(title="Distribution of AVG_REMOVAL_RATE (Raw)", xlabel="Average Removal Rate", ylabel="Frequency")
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
    parser.add_argument("--labels", required=True, help="removal-rate 標籤 CSV")
    parser.add_argument("--plot", default="outputs/removal_rate_raw.png", help="分布圖輸出路徑")
    parser.add_argument("--show", action="store_true", help="顯示圖表視窗")
    args = parser.parse_args()

    sensor_df = load_sensor_data(args.data_dir)
    labels = load_labels(args.labels)
    summary = summarize_data(sensor_df, labels)

    print(f"原始時間序列資料: {summary['sensor_rows']} 列、{summary['sensor_columns']} 欄")
    print(f"目標值資料: {summary['label_rows']} 列")
    print(f"不重複晶圓數量: {summary['unique_wafers']}")
    print("\nSTAGE / CHAMBER 分布:")
    print(summary["stage_chamber_distribution"].to_string(index=False))
    print("\nAVG_REMOVAL_RATE 摘要:")
    print(summary["removal_rate_statistics"].to_string())
    plot_removal_rate_distribution(labels, args.plot, args.show)


if __name__ == "__main__":
    main()
