"""Shared modeling utilities kept in sync with CMP.ipynb."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    X_high: pd.DataFrame
    y_high: pd.Series
    X_low: pd.DataFrame
    y_low: pd.Series
    X_high_filled: np.ndarray
    X_low_filled: np.ndarray
    imputer_high: SimpleImputer
    imputer_low: SimpleImputer
    high_group_cols: list[str]
    drop_from_high_cols: list[str]
    drop_from_low_cols: list[str]


def prepare_data(data_file: str = "data/processed/cmp_final_dataset.csv") -> PreparedData:
    """Reproduce notebook cell 16's ordering, filtering, grouping and imputation."""
    final_df = pd.read_csv(data_file)
    if "WAFER_ID" in final_df.columns:
        final_df = final_df.set_index("WAFER_ID")
    if "START_TIMESTAMP" in final_df.columns:
        final_df = final_df.sort_values("START_TIMESTAMP").drop(columns=["START_TIMESTAMP"])
    else:
        final_df = final_df.sort_index()

    X = final_df.drop(
        columns=["AVG_REMOVAL_RATE", "neighbor_feature", "USAGE_OF_BACKING_FILM"],
        errors="ignore",
    )
    if "STAGE" in X.columns:
        X["STAGE"] = X["STAGE"].map({"A": 0, "B": 1}).fillna(-1)
    y = final_df["AVG_REMOVAL_RATE"]

    high_group_cols = [c for c in X if any(k in c for k in ("_Ch1", "_Ch2", "_Ch3"))]
    if not high_group_cols:
        raise ValueError("找不到 Ch1–Ch3 特徵，請先用 notebook 對齊的特徵工程產生資料。")
    is_high_group = X[high_group_cols].notna().any(axis=1)
    drop_from_high_cols = [c for c in X if any(k in c for k in ("_Ch4", "_Ch5", "_Ch6"))]
    drop_from_low_cols = [c for c in X if any(k in c for k in ("_Ch1", "_Ch2", "_Ch3"))]

    X_high = X[is_high_group].copy().drop(columns=X.columns.intersection(drop_from_high_cols))
    X_low = X[~is_high_group].copy().drop(columns=X.columns.intersection(drop_from_low_cols))
    y_high, y_low = y[is_high_group].copy(), y[~is_high_group].copy()
    imputer_high = SimpleImputer(strategy="constant", fill_value=0)
    imputer_low = SimpleImputer(strategy="constant", fill_value=0)
    return PreparedData(
        X, y, X_high, y_high, X_low, y_low,
        imputer_high.fit_transform(X_high), imputer_low.fit_transform(X_low),
        imputer_high, imputer_low, high_group_cols,
        drop_from_high_cols, drop_from_low_cols,
    )


def train_model(estimator, search_spaces, X_data, y_data, group_name, n_iter=50):
    """Reproduce the notebook's nested time-series Bayesian search."""
    print(f"\n正在為 {group_name} 執行巢狀時間序列交叉驗證...")
    fold_size = len(X_data) // 6
    outer_cv = TimeSeriesSplit(n_splits=5, max_train_size=fold_size * 2)
    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_squared_error",
        n_jobs=1,
        verbose=0,
        random_state=42,
    )
    stopper = DeltaYStopper(delta=0.01, n_best=15)
    score_kwargs = {
        "estimator": search,
        "X": X_data,
        "y": y_data,
        "cv": outer_cv,
        "scoring": "neg_mean_squared_error",
        "n_jobs": 1,
    }
    # sklearn renamed fit_params to params in 1.4; both carry the same callback.
    parameter_name = "params" if "params" in inspect.signature(cross_val_score).parameters else "fit_params"
    score_kwargs[parameter_name] = {"callback": stopper}
    nested_scores = cross_val_score(**score_kwargs)
    print(f"[完成] [{group_name}] 各折 MSE 結果:")
    for i, score in enumerate(nested_scores):
        print(f"   Fold {i + 1}: {-score:.4f}")
    print(f"   平均 MSE: {-np.mean(nested_scores):.4f}(標準差: {np.std(nested_scores):.4f})")
    search.fit(X_data, y_data, callback=stopper)
    print(f"   最佳參數: {search.best_params_}")
    return search.best_estimator_


def show_importance(model, feature_names, title):
    importances = model.feature_importances_ if hasattr(model, "feature_importances_") else np.zeros(len(feature_names))
    result = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    result = result.sort_values("Importance", ascending=False)
    print(f"\n【{title} 特徵重要度 Top 10】\n", result.head(10))
    return result


def evaluate_model(
    model_name,
    model_high,
    model_low,
    test_df,
    prepared,
    refined_high_cols=None,
    refined_low_cols=None,
    imputer_high=None,
    imputer_low=None,
):
    """Reproduce notebook cell 16's grouped validation prediction."""
    if test_df is None or len(test_df) == 0:
        print(f"\n[警告] 無測試集，跳過評估 {model_name}")
        return None
    print(f"\n【{model_name} 測試集預測與評估】")
    test_df = test_df.set_index("WAFER_ID") if "WAFER_ID" in test_df.columns else test_df.copy()
    y_test = test_df["AVG_REMOVAL_RATE"] if "AVG_REMOVAL_RATE" in test_df.columns else None
    X_test = test_df.drop(columns=["AVG_REMOVAL_RATE", "START_TIMESTAMP"], errors="ignore")
    if "STAGE" in X_test.columns:
        X_test["STAGE"] = X_test["STAGE"].map({"A": 0, "B": 1}).fillna(-1)
    for column in set(prepared.X.columns) - set(X_test.columns):
        X_test[column] = 0
    X_test = X_test[prepared.X.columns]
    is_test_high = X_test[prepared.high_group_cols].notna().any(axis=1)
    predictions = pd.Series(index=X_test.index, dtype=float)
    high_imputer = imputer_high or prepared.imputer_high
    low_imputer = imputer_low or prepared.imputer_low
    if is_test_high.any():
        values = X_test[is_test_high].drop(columns=X_test.columns.intersection(prepared.drop_from_high_cols))
        if refined_high_cols is not None:
            values = values[refined_high_cols]
        predictions[is_test_high] = model_high.predict(high_imputer.transform(values))
    if (~is_test_high).any():
        values = X_test[~is_test_high].drop(columns=X_test.columns.intersection(prepared.drop_from_low_cols))
        if refined_low_cols is not None:
            values = values[refined_low_cols]
        predictions[~is_test_high] = model_low.predict(low_imputer.transform(values))
    if y_test is not None:
        actual, predicted = y_test.values, predictions.values
        print(f"   Validation MSE: {mean_squared_error(actual, predicted):.2f}")
        print(f"   Validation MAPE: {np.mean(np.abs((actual - predicted) / actual)) * 100:.2f}%")
        print(f"   Validation R2: {r2_score(y_test, predictions):.4f}")
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test[is_test_high], predictions[is_test_high], alpha=.7, color="red", label="High Group")
        plt.scatter(y_test[~is_test_high], predictions[~is_test_high], alpha=.7, color="blue", label="Low Group")
        low, high = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
        plt.plot([low, high], [low, high], "k--", lw=2)
        plt.xlabel("Actual Rate")
        plt.ylabel("Predicted Rate")
        plt.title(f"{model_name} Validation")
        plt.legend()
        plt.show()
    return predictions


def load_validation(path: str = "data/processed/cmp_final_test_dataset.csv"):
    return pd.read_csv(path)
