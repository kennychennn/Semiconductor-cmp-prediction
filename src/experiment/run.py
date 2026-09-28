"""一鍵重跑 CMP 特徵工程、模型訓練、評估與結果保存。"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skopt.space import Categorical, Integer, Real
from xgboost import XGBRegressor

from src.data.make_dataset import load_labels, load_sensor_data
from src.model.common import predict_grouped, prepare_data, train_model
from src.preprocess.build_features import build_features
from src.experiment.checks import validate_config, validate_feature_table, validate_inputs


ROOT = Path(__file__).resolve().parents[2]
XGB_SEARCH_SPACE = {
    "n_estimators": Integer(100, 1000),
    "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
    "max_depth": Integer(3, 6),
    "min_child_weight": Categorical([1, 3, 5, 7, 9]),
    "subsample": Real(0.6, 1.0),
    "colsample_bytree": Categorical([0.5, 0.6, 0.7, 0.8, 0.9]),
    "reg_alpha": Categorical([0.1]),
    "reg_lambda": Categorical([1.0]),
}
SUPPORTED_MODELS = {
    "decision_tree",
    "random_forest",
    "xgboost_refined",
    "svr",
    "neural_network_all",
}


def _load_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config)
    return config


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _estimator_xgb(random_state: int, n_jobs: int):
    return XGBRegressor(
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=n_jobs,
        tree_method="hist",
    )


def _pipeline_mlp(hidden_layers: tuple[int, ...], random_state: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            solver="adam",
            activation="relu",
            max_iter=1000,
            early_stopping=True,
            random_state=random_state,
        )),
    ])


def _training_kwargs(config: dict[str, Any], n_iter: int | None = None) -> dict[str, Any]:
    training = config["training"]
    return {
        "n_iter": int(n_iter or training["bayes_search_iterations"]),
        "cv_strategy": training.get("cv_strategy", "competition"),
        "outer_splits": int(training["outer_splits"]),
        "inner_splits": int(training["inner_splits"]),
        "random_state": int(config["project"].get("random_state", 42)),
        "n_jobs": int(training.get("n_jobs", 1)),
        "return_details": True,
    }


def _train_grouped(
    estimator_factory,
    search_space,
    data,
    group_name: str,
    train_kwargs: dict[str, Any],
):
    high = train_model(
        estimator_factory(), search_space, data.X_high_filled, data.y_high,
        f"{group_name}_High", **train_kwargs,
    )
    low = train_model(
        estimator_factory(), search_space, data.X_low_filled, data.y_low,
        f"{group_name}_Low", **train_kwargs,
    )
    return high, low


def _select_features(model, frame: pd.DataFrame, threshold: float) -> list[str]:
    importance = pd.DataFrame({
        "Feature": frame.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance["Cumulative_Importance"] = importance["Importance"].cumsum()
    crossing = importance.index[importance["Cumulative_Importance"] >= threshold]
    end = int(crossing[0]) if len(crossing) else len(importance) - 1
    return importance.loc[:end, "Feature"].tolist()


def _train_refined_xgb(data, config, train_kwargs):
    base_high, base_low = _train_grouped(
        lambda: _estimator_xgb(train_kwargs["random_state"], train_kwargs["n_jobs"]),
        XGB_SEARCH_SPACE,
        data,
        "XGB_Base",
        train_kwargs,
    )
    threshold = float(config["training"].get("feature_importance_threshold", 0.80))
    high_cols = _select_features(base_high["estimator"], data.X_high, threshold)
    low_cols = _select_features(base_low["estimator"], data.X_low, threshold)
    high_imputer = SimpleImputer(strategy="constant", fill_value=0)
    low_imputer = SimpleImputer(strategy="constant", fill_value=0)
    high_values = high_imputer.fit_transform(data.X_high[high_cols])
    low_values = low_imputer.fit_transform(data.X_low[low_cols])
    high = train_model(
        _estimator_xgb(train_kwargs["random_state"], train_kwargs["n_jobs"]),
        XGB_SEARCH_SPACE, high_values, data.y_high, "XGB_Refined_High", **train_kwargs,
    )
    low = train_model(
        _estimator_xgb(train_kwargs["random_state"], train_kwargs["n_jobs"]),
        XGB_SEARCH_SPACE, low_values, data.y_low, "XGB_Refined_Low", **train_kwargs,
    )
    return {
        "high": high,
        "low": low,
        "high_cols": high_cols,
        "low_cols": low_cols,
        "high_imputer": high_imputer,
        "low_imputer": low_imputer,
        "base_high": base_high,
        "base_low": base_low,
    }


def _aligned_full_test(test_df: pd.DataFrame, data):
    frame = test_df.drop(columns=["AVG_REMOVAL_RATE", "START_TIMESTAMP"], errors="ignore").copy()
    if "STAGE" in frame:
        frame["STAGE"] = frame["STAGE"].map({"A": 0, "B": 1}).fillna(-1)
    for column in set(data.X.columns) - set(frame.columns):
        frame[column] = 0
    return frame[data.X.columns]


def _metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    actual = y_true.to_numpy(dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "mse": float(mean_squared_error(actual, prediction)),
        "mape_percent": float(np.mean(np.abs((actual - prediction) / actual)) * 100),
        "r2": float(r2_score(actual, prediction)),
    }


def _time_series_diagnostic(model, X, y, splits: int, n_jobs: int) -> dict[str, Any]:
    """Evaluate a fitted model's settings on expanding temporal folds."""
    splits = min(int(splits), len(y) - 1)
    if splits < 2:
        return {"mean_mse": None, "std_mse": None, "mse": []}
    scores = -cross_val_score(
        model, X, y, cv=TimeSeriesSplit(n_splits=splits),
        scoring="neg_mean_squared_error", n_jobs=n_jobs,
    )
    return {
        "mean_mse": float(np.mean(scores)),
        "std_mse": float(np.std(scores)),
        "mse": [float(score) for score in scores],
    }


def _grouped_time_diagnostic(high_model, low_model, high_X, low_X, data, config):
    training = config["training"]
    if not bool(training.get("time_series_diagnostic", True)):
        return None
    splits = int(training.get("time_series_splits", 5))
    n_jobs = int(training.get("n_jobs", 1))
    return {
        "high": _time_series_diagnostic(high_model, high_X, data.y_high, splits, n_jobs),
        "low": _time_series_diagnostic(low_model, low_X, data.y_low, splits, n_jobs),
    }


def _save_grouped_result(
    name: str,
    models: dict[str, Any],
    test_df: pd.DataFrame,
    data,
    result_dir: Path,
    model_dir: Path,
    cv_details: dict[str, Any] | None = None,
    time_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction, groups = predict_grouped(
        models["high"], models["low"], test_df, data,
        refined_high_cols=models.get("high_cols"),
        refined_low_cols=models.get("low_cols"),
        imputer_high=models.get("high_imputer"),
        imputer_low=models.get("low_imputer"),
    )
    metric = _metrics(test_df["AVG_REMOVAL_RATE"], prediction)
    predictions = test_df[["WAFER_ID", "STAGE"]].copy()
    predictions["group"] = groups
    predictions["y_true"] = test_df["AVG_REMOVAL_RATE"].to_numpy()
    predictions["y_pred"] = prediction
    predictions.to_csv(result_dir / f"{name}_predictions.csv", index=False)
    artifact = dict(models)
    artifact.setdefault("imputer_high", models.get("high_imputer", data.imputer_high))
    artifact.setdefault("imputer_low", models.get("low_imputer", data.imputer_low))
    artifact.update({
        "feature_columns": list(data.X.columns),
        "high_group_cols": list(data.high_group_cols),
        "drop_from_high_cols": list(data.drop_from_high_cols),
        "drop_from_low_cols": list(data.drop_from_low_cols),
    })
    joblib.dump(artifact, model_dir / f"{name}.joblib")
    row = {"model": name, "split": "competition_test", "n_samples": len(test_df), **metric}
    if cv_details:
        row.update({
            "cv_high_mean_mse": cv_details["high"]["cv_mean_mse"],
            "cv_high_std_mse": cv_details["high"]["cv_std_mse"],
            "cv_low_mean_mse": cv_details["low"]["cv_mean_mse"],
            "cv_low_std_mse": cv_details["low"]["cv_std_mse"],
            "best_params_high": json.dumps(_json_safe(cv_details["high"]["best_params"]), sort_keys=True),
            "best_params_low": json.dumps(_json_safe(cv_details["low"]["best_params"]), sort_keys=True),
        })
    if time_diagnostic:
        row.update({
            "ts_high_mean_mse": time_diagnostic["high"]["mean_mse"],
            "ts_high_std_mse": time_diagnostic["high"]["std_mse"],
            "ts_low_mean_mse": time_diagnostic["low"]["mean_mse"],
            "ts_low_std_mse": time_diagnostic["low"]["std_mse"],
        })
    return row


def _save_full_result(name, model_details, model, imputer, test_df, data, result_dir, model_dir, time_diagnostic=None):
    prediction = model.predict(imputer.transform(_aligned_full_test(test_df, data)))
    metric = _metrics(test_df["AVG_REMOVAL_RATE"], prediction)
    predictions = test_df[["WAFER_ID", "STAGE"]].copy()
    predictions["group"] = "all"
    predictions["y_true"] = test_df["AVG_REMOVAL_RATE"].to_numpy()
    predictions["y_pred"] = prediction
    predictions.to_csv(result_dir / f"{name}_predictions.csv", index=False)
    joblib.dump({"model": model, "imputer": imputer, "feature_columns": list(data.X.columns)}, model_dir / f"{name}.joblib")
    row = {
        "model": name,
        "split": "competition_test",
        "n_samples": len(test_df),
        **metric,
        "cv_all_mean_mse": model_details["cv_mean_mse"],
        "cv_all_std_mse": model_details["cv_std_mse"],
        "best_params": json.dumps(_json_safe(model_details["best_params"]), sort_keys=True),
    }
    if time_diagnostic:
        row.update({
            "ts_all_mean_mse": time_diagnostic["mean_mse"],
            "ts_all_std_mse": time_diagnostic["std_mse"],
        })
    return row


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "scikit-optimize", "xgboost", "PyYAML"]
    return {name: metadata.version(name) for name in names if _has_package(name)}


def _json_safe(value: Any) -> Any:
    """Convert numpy/skopt values to values accepted by ``json.dumps``."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _has_package(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def run_experiment(config_path: str | Path = ROOT / "config.yml", models: list[str] | None = None, run_id: str | None = None, n_iter: int | None = None):
    config_path = _path(config_path)
    config = _load_config(config_path)
    paths = config["paths"]
    project_random_state = int(config["project"].get("random_state", 42))
    selected_models = models or list(config["experiment"]["models"])
    unsupported = sorted(set(selected_models).difference(SUPPORTED_MODELS))
    if unsupported:
        raise ValueError(f"不支援的模型名稱: {unsupported}")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = _path(config["experiment"].get("run_root", "results/runs")) / run_id
    model_dir = _path(config["experiment"].get("model_root", "models/runs")) / run_id

    train_labels = load_labels(_path(paths["train_labels"]))
    test_labels = load_labels(_path(paths["test_labels"]))
    input_check = validate_inputs(
        _path(paths["train_data"]), _path(paths["test_data"]), train_labels, test_labels,
    )
    train_raw = load_sensor_data(_path(paths["train_data"]))
    test_raw = load_sensor_data(_path(paths["test_data"]))
    feature_kwargs = {
        "include_timing_features": bool(config["experiment"].get("include_timing_features", False)),
    }
    train_features, scalers = build_features(train_raw, train_labels, is_train=True, **feature_kwargs)
    test_features, _ = build_features(test_raw, test_labels, is_train=False, scalers=scalers, **feature_kwargs)
    train_check = validate_feature_table(train_features, "training features")
    test_check = validate_feature_table(test_features, "test features")
    result_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "config_snapshot.yml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8",
    )
    if config["experiment"].get("save_processed_features", True):
        processed = _path(paths["processed_data"])
        interim = _path(paths["interim_data"])
        processed.mkdir(parents=True, exist_ok=True)
        interim.mkdir(parents=True, exist_ok=True)
        train_features.to_csv(processed / "cmp_final_dataset.csv", index=False)
        test_features.to_csv(processed / "cmp_final_test_dataset.csv", index=False)
        train_features.to_csv(interim / "cmp_processed_features.csv", index=False)

    train_csv = result_dir / "features_train.csv"
    test_csv = result_dir / "features_test.csv"
    train_features.to_csv(train_csv, index=False)
    test_features.to_csv(test_csv, index=False)
    data = prepare_data(str(train_csv))
    train_kwargs = _training_kwargs(config, n_iter=n_iter)
    metrics = []
    for name in selected_models:
        if name == "decision_tree":
            details = _train_grouped(
                lambda: DecisionTreeRegressor(random_state=project_random_state),
                {
                    "max_depth": Integer(4, 10),
                    "min_samples_split": Integer(10, 60),
                    "min_samples_leaf": Integer(2, 20),
                }, data, "DT", train_kwargs,
            )
            time_diagnostic = _grouped_time_diagnostic(
                details[0]["estimator"], details[1]["estimator"],
                data.X_high_filled, data.X_low_filled, data, config,
            )
            metrics.append(_save_grouped_result(
                name, {"high": details[0]["estimator"], "low": details[1]["estimator"]},
                test_features, data, result_dir, model_dir,
                {"high": details[0], "low": details[1]}, time_diagnostic,
            ))
        elif name == "random_forest":
            details = _train_grouped(
                lambda: RandomForestRegressor(random_state=project_random_state, n_jobs=train_kwargs["n_jobs"]),
                {"n_estimators": Integer(10, 500), "max_depth": Integer(5, 50), "min_samples_leaf": Integer(1, 15)},
                data, "RF", train_kwargs,
            )
            time_diagnostic = _grouped_time_diagnostic(
                details[0]["estimator"], details[1]["estimator"],
                data.X_high_filled, data.X_low_filled, data, config,
            )
            metrics.append(_save_grouped_result(
                name, {"high": details[0]["estimator"], "low": details[1]["estimator"]},
                test_features, data, result_dir, model_dir,
                {"high": details[0], "low": details[1]}, time_diagnostic,
            ))
        elif name == "xgboost_refined":
            refined = _train_refined_xgb(data, config, train_kwargs)
            refined_high_X = refined["high_imputer"].transform(data.X_high[refined["high_cols"]])
            refined_low_X = refined["low_imputer"].transform(data.X_low[refined["low_cols"]])
            time_diagnostic = _grouped_time_diagnostic(
                refined["high"]["estimator"], refined["low"]["estimator"],
                refined_high_X, refined_low_X, data, config,
            )
            metrics.append(_save_grouped_result(name, {
                "high": refined["high"]["estimator"],
                "low": refined["low"]["estimator"],
                "high_cols": refined["high_cols"],
                "low_cols": refined["low_cols"],
                "high_imputer": refined["high_imputer"],
                "low_imputer": refined["low_imputer"],
                "base_high": refined["base_high"],
                "base_low": refined["base_low"],
            }, test_features, data, result_dir, model_dir,
                {"high": refined["high"], "low": refined["low"]}, time_diagnostic,
            ))
        elif name == "svr":
            details = _train_grouped(
                lambda: Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))]),
                {"svr__C": Real(1, 100), "svr__epsilon": Real(0.05, 3), "svr__gamma": ["scale", "auto"]},
                data, "SVR", train_kwargs,
            )
            time_diagnostic = _grouped_time_diagnostic(
                details[0]["estimator"], details[1]["estimator"],
                data.X_high_filled, data.X_low_filled, data, config,
            )
            metrics.append(_save_grouped_result(
                name, {"high": details[0]["estimator"], "low": details[1]["estimator"]},
                test_features, data, result_dir, model_dir,
                {"high": details[0], "low": details[1]}, time_diagnostic,
            ))
        elif name == "neural_network_all":
            imputer = SimpleImputer(strategy="constant", fill_value=0)
            X_filled = imputer.fit_transform(data.X)
            details = train_model(
                _pipeline_mlp((128, 64), project_random_state),
                {"mlp__batch_size": Categorical([16, 32, 64]), "mlp__alpha": Real(1e-4, 1e-1, prior="log-uniform"), "mlp__learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform")},
                X_filled, data.y, "MLP_All_Data", **train_kwargs,
            )
            time_diagnostic = None
            if bool(config["training"].get("time_series_diagnostic", True)):
                time_diagnostic = _time_series_diagnostic(
                    details["estimator"], X_filled, data.y,
                    int(config["training"].get("time_series_splits", 5)),
                    int(config["training"].get("n_jobs", 1)),
                )
            metrics.append(_save_full_result(
                name, details, details["estimator"], imputer, test_features, data,
                result_dir, model_dir, time_diagnostic,
            ))
        else:
            raise ValueError(f"不支援的模型名稱: {name}")

    metrics_df = pd.DataFrame(metrics).sort_values("mse")
    metrics_df.to_csv(result_dir / "metrics.csv", index=False)
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "config_path": str(config_path),
        "models": selected_models,
        "input_check": input_check,
        "feature_check": {"train": train_check, "test": test_check},
        "training_config": _json_safe(config["training"]),
        "experiment_config": _json_safe(config["experiment"]),
        "package_versions": _package_versions(),
        "result_dir": str(result_dir),
        "model_dir": str(model_dir),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = result_dir.parent / "latest_run.json"
    latest.write_text(json.dumps({"run_id": run_id, "result_dir": str(result_dir), "model_dir": str(model_dir)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics_df[["model", "mse", "r2"]].to_string(index=False))
    print(f"實驗完成；結果已保存至: {result_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yml"))
    parser.add_argument("--models", help="逗號分隔模型名稱；未指定時使用 config.yml")
    parser.add_argument("--run-id")
    parser.add_argument("--n-iter", type=int, help="覆蓋 Bayesian Search iterations，適合 smoke run")
    parser.add_argument("--check-only", action="store_true", help="只檢查輸入資料，不執行特徵工程或模型")
    args = parser.parse_args()
    if args.check_only:
        config = _load_config(_path(args.config))
        paths = config["paths"]
        checks = validate_inputs(
            _path(paths["train_data"]), _path(paths["test_data"]),
            load_labels(_path(paths["train_labels"])), load_labels(_path(paths["test_labels"])),
        )
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return
    models = [item.strip() for item in args.models.split(",")] if args.models else None
    run_experiment(args.config, models=models, run_id=args.run_id, n_iter=args.n_iter)


if __name__ == "__main__":
    main()
