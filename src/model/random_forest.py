"""Random Forest experiment aligned with notebook cell 20."""

from sklearn.ensemble import RandomForestRegressor
from skopt.space import Integer

from src.model.common import evaluate_model, load_validation, prepare_data, show_importance, train_model


def main():
    data = prepare_data()
    search_space = {
        "n_estimators": Integer(10, 500),
        "max_depth": Integer(5, 50),
        "min_samples_leaf": Integer(1, 15),
    }
    estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
    high = train_model(estimator, search_space, data.X_high_filled, data.y_high, "RF_High_Group")
    low = train_model(estimator, search_space, data.X_low_filled, data.y_low, "RF_Low_Group")
    show_importance(high, data.X_high.columns, "RF High Group")
    evaluate_model("Random Forest", high, low, load_validation(), data)


if __name__ == "__main__":
    main()
