"""Decision Tree experiment aligned with notebook cell 18."""

from sklearn.tree import DecisionTreeRegressor
from skopt.space import Integer

from src.model.common import evaluate_model, load_validation, prepare_data, show_importance, train_model


def main():
    data = prepare_data()
    search_space = {
        "max_depth": Integer(4, 10),
        "min_samples_split": Integer(10, 60),
        "min_samples_leaf": Integer(2, 20),
    }
    high = train_model(DecisionTreeRegressor(random_state=42), search_space, data.X_high_filled, data.y_high, "DT_High_Group", 50)
    low = train_model(DecisionTreeRegressor(random_state=42), search_space, data.X_low_filled, data.y_low, "DT_Low_Group", 50)
    show_importance(high, data.X_high.columns, "DT High Group")
    evaluate_model("Decision Tree", high, low, load_validation(), data)


if __name__ == "__main__":
    main()
