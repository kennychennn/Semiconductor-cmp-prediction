"""Grouped and all-data MLP experiments aligned with notebook cells 26 and 28."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt.space import Categorical, Real

from src.model.common import evaluate_model, load_validation, prepare_data, train_model


def _pipeline(hidden_layers):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layers, solver="adam", activation="relu",
            max_iter=1000, early_stopping=True, random_state=42,
        )),
    ])


def main():
    data = prepare_data()
    search_space = {
        "mlp__batch_size": Categorical([16, 32, 64]),
        "mlp__alpha": Real(1e-4, 1e-1, prior="log-uniform"),
        "mlp__learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
    }
    high = train_model(_pipeline((64, 32)), search_space, data.X_high_filled, data.y_high, "MLP_High_Group")
    low = train_model(_pipeline((64, 32)), search_space, data.X_low_filled, data.y_low, "MLP_Low_Group")
    evaluate_model("Neural Network", high, low, load_validation(), data)

    imputer_all = SimpleImputer(strategy="constant", fill_value=0)
    X_filled = imputer_all.fit_transform(data.X)
    model_all = train_model(_pipeline((128, 64)), search_space, X_filled, data.y, "MLP_All_Data")
    test = load_validation()
    test = test.set_index("WAFER_ID") if "WAFER_ID" in test else test
    y_test = test["AVG_REMOVAL_RATE"]
    X_test = test.drop(columns=["AVG_REMOVAL_RATE", "START_TIMESTAMP"], errors="ignore")
    if "STAGE" in X_test:
        X_test["STAGE"] = X_test["STAGE"].map({"A": 0, "B": 1}).fillna(-1)
    for column in set(data.X.columns) - set(X_test.columns):
        X_test[column] = 0
    prediction = model_all.predict(imputer_all.transform(X_test[data.X.columns]))
    print(f"   Validation MSE: {mean_squared_error(y_test, prediction):.2f}")
    print(f"   Validation MAPE: {np.mean(np.abs((y_test.values - prediction) / y_test.values)) * 100:.2f}%")
    print(f"   Validation R2: {r2_score(y_test, prediction):.4f}")
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, prediction, alpha=.7, color="purple", label="All Group (MLP)")
    low_value, high_value = min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())
    plt.plot([low_value, high_value], [low_value, high_value], "k--", lw=2)
    plt.xlabel("Actual Rate")
    plt.ylabel("Predicted Rate")
    plt.title("Neural Network (All Data) Validation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
