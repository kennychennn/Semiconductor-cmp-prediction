"""對現有資料與最近一次實驗輸出執行 schema／結果檢查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.make_dataset import load_labels
from src.experiment.checks import validate_config, validate_feature_table, validate_inputs


ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_checks(config_path: str | Path = ROOT / "config.yml", run_dir: str | Path | None = None) -> dict[str, object]:
    config_path = _path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    validate_config(config)
    paths = config["paths"]
    report: dict[str, object] = {
        "inputs": validate_inputs(
            _path(paths["train_data"]), _path(paths["test_data"]),
            load_labels(_path(paths["train_labels"])), load_labels(_path(paths["test_labels"])),
        )
    }
    processed = _path(paths["processed_data"])
    train_file = processed / "cmp_final_dataset.csv"
    test_file = processed / "cmp_final_test_dataset.csv"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("找不到 processed feature table，請先執行 make experiment")
    report["features"] = {
        "train": validate_feature_table(pd.read_csv(train_file), "processed training features"),
        "test": validate_feature_table(pd.read_csv(test_file), "processed test features"),
    }
    if run_dir is None:
        pointer = _path(config["experiment"].get("run_root", "results/runs")) / "latest_run.json"
        if pointer.exists():
            run_dir = json.loads(pointer.read_text(encoding="utf-8"))["result_dir"]
    if run_dir is not None:
        run_path = _path(run_dir)
        metrics_path = run_path / "metrics.csv"
        manifest_path = run_path / "manifest.json"
        if not metrics_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"實驗結果缺少 metrics.csv 或 manifest.json: {run_path}")
        metrics = pd.read_csv(metrics_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {"model", "mse", "split", "n_samples"}
        missing = required.difference(metrics.columns)
        if missing:
            raise ValueError(f"metrics.csv 缺少欄位: {sorted(missing)}")
        if metrics.empty or metrics["mse"].isna().any() or (metrics["mse"] < 0).any():
            raise ValueError("metrics.csv 包含空值或負的 MSE")
        if config["training"].get("time_series_diagnostic", True):
            for model_name in metrics["model"].astype(str):
                columns = (
                    ("ts_all_mean_mse",)
                    if model_name == "neural_network_all"
                    else ("ts_high_mean_mse", "ts_low_mean_mse")
                )
                model_rows = metrics[metrics["model"].astype(str) == model_name]
                for column in columns:
                    if column not in metrics or model_rows[column].isna().any() or (model_rows[column] < 0).any():
                        raise ValueError(f"metrics.csv 缺少有效的時間診斷欄位: {column}")
        model_path = _path(manifest.get("model_dir", ""))
        if not (run_path / "config_snapshot.yml").exists():
            raise FileNotFoundError(f"結果缺少 config_snapshot.yml: {run_path}")
        for row in metrics.to_dict("records"):
            model_name = str(row["model"])
            prediction_path = run_path / f"{model_name}_predictions.csv"
            model_file = model_path / f"{model_name}.joblib"
            if not prediction_path.exists() or not model_file.exists():
                raise FileNotFoundError(f"{model_name} 缺少 prediction 或 model artifact")
            predictions = pd.read_csv(prediction_path)
            if len(predictions) != int(row["n_samples"]):
                raise ValueError(f"{model_name} prediction 筆數與 metrics 不一致")
            if "y_pred" not in predictions or predictions["y_pred"].isna().any():
                raise ValueError(f"{model_name} prediction 含有空值")
        report["run"] = {
            "result_dir": str(run_path),
            "models": metrics["model"].tolist(),
            "best_model": str(metrics.sort_values("mse").iloc[0]["model"]),
            "best_mse": float(metrics["mse"].min()),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yml"))
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    report = run_checks(args.config, args.run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("所有資料與實驗結果檢查通過。")


if __name__ == "__main__":
    main()
