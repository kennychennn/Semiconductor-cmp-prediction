"""XGBoost base/refined experiments aligned with notebook cells 22 and 31."""

import pandas as pd
from sklearn.impute import SimpleImputer
from skopt.space import Categorical, Integer, Real
from xgboost import XGBRegressor

from src.model.common import evaluate_model, load_validation, prepare_data, show_importance, train_model

SEARCH_SPACE = {
    "n_estimators": Integer(100, 1000),
    "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
    "max_depth": Integer(3, 6),
    "min_child_weight": Categorical([1, 3, 5, 7, 9]),
    "subsample": Real(0.6, 1.0),
    "colsample_bytree": Categorical([0.5, 0.6, 0.7, 0.8, 0.9]),
    "reg_alpha": Categorical([0.1]),
    "reg_lambda": Categorical([1.0]),
}


def estimator():
    # BayesSearchCV 本身以單一工作執行；這裡也明確限制 XGBoost 的
    # OpenMP 執行緒，避免 ``n_jobs=None`` 在 macOS 上佔滿所有核心。
    return XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
        tree_method="hist",
    )


def refine_model(base_model, X_df, y, group_name, cumulative_threshold=0.80):
    importance = pd.DataFrame({
        "Feature": X_df.columns,
        "Importance": base_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance["Cumulative_Importance"] = importance["Importance"].cumsum()
    threshold_idx = importance[importance["Cumulative_Importance"] >= cumulative_threshold].index.min()
    if pd.isna(threshold_idx):
        threshold_idx = len(importance) - 1
    selected = importance.loc[:threshold_idx, "Feature"].values
    print(f"\n[{group_name}] 特徵縮減: {len(X_df.columns)} -> {len(selected)} 個關鍵特徵")
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    values = imputer.fit_transform(X_df[selected])
    model = train_model(estimator(), SEARCH_SPACE, values, y, f"{group_name}_Refined", 50)
    return model, selected, imputer


def main():
    data = prepare_data()
    high = train_model(estimator(), SEARCH_SPACE, data.X_high_filled, data.y_high, "XGB_High_Group", 50)
    low = train_model(estimator(), SEARCH_SPACE, data.X_low_filled, data.y_low, "XGB_Low_Group", 50)
    show_importance(high, data.X_high.columns, "XGB High Group")
    validation = load_validation()
    evaluate_model("XGBoost", high, low, validation, data)
    high_opt, high_cols, high_imputer = refine_model(high, data.X_high, data.y_high, "XGB_High_Group")
    low_opt, low_cols, low_imputer = refine_model(low, data.X_low, data.y_low, "XGB_Low_Group")
    evaluate_model(
        "XGBoost_Refined", high_opt, low_opt, validation, data,
        refined_high_cols=high_cols, refined_low_cols=low_cols,
        imputer_high=high_imputer, imputer_low=low_imputer,
    )


if __name__ == "__main__":
    main()
