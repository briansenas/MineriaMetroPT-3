import os
import sys
from datetime import datetime

import joblib
import polars as pl
from utils import (
    build_structure,
    evaluate,
    get_xgboost_objective_func,
    perform_cross_validation,
    get_xgboost_space,
    bayesian_optimization,
)
from xgboost import XGBClassifier

def main():
    random_state = 42
    metric = "precision"

    train_df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    # Prepare training data
    X_train = train_df.drop("is_anomaly")
    y_train = train_df["is_anomaly"]

    # Prepare model
    model = XGBClassifier(objective="binary:logistic", random_state=random_state)

    # Perform cross validation
    perform_cross_validation(model, X_train, y_train, metric)

    # Perform bayesian optimization
    space = get_xgboost_space()
    folds = X_train["fold"].unique()
    objective = get_xgboost_objective_func(space, X_train, y_train, folds, "fold")

    # Run Bayesian optimization
    best_params = bayesian_optimization(space, objective)
    print("Best hyperparameters found:", best_params)

    # Train evaluation
    train_metrics = evaluate(train_df.drop("fold"), model)

    # Test data evaluation
    test_metrics = evaluate(test_df, model, prefix="test_")

    # Prepare metadata
    model_filename = f"xgboost_{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    metadata = {
        "best_score": [],
        "best_params": [],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_filename": model_filename,
    }

    model_with_metadata = {"model": model, "metadata": metadata}
    # Save the model
    joblib.dump(model_with_metadata, os.path.join("models", model_filename))
    print(f"Model and metadata saved as {model_filename}")


if __name__ == "__main__":
    paths = [
        "./MineriaMetroPT-3/DecisionTree&XGBoost/src",
        "./DecisionTree&XGBoost/src",
        "./src",
    ]

    file_path = ""
    for path in paths:
        if os.path.exists(path):
            file_path = os.path.dirname(path)
            break
    if file_path:
        sys.path.append(file_path)
        os.chdir(file_path)
    build_structure()
    main()
