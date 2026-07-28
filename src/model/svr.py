"""SVR experiment aligned with notebook cell 24."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skopt.space import Real

from src.model.common import evaluate_model, load_validation, prepare_data, train_model


def main():
    data = prepare_data()
    search_space = {
        "svr__C": Real(1, 100),
        "svr__epsilon": Real(0.05, 3),
        "svr__gamma": ["scale", "auto"],
    }
    estimator = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    high = train_model(estimator, search_space, data.X_high_filled, data.y_high, "SVR_High_Group", 50)
    low = train_model(estimator, search_space, data.X_low_filled, data.y_low, "SVR_Low_Group", 50)
    evaluate_model("SVR", high, low, load_validation(), data)


if __name__ == "__main__":
    main()
