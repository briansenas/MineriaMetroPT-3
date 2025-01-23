import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from utils import build_structure, evaluate, get_cv_iterable


def perform_hyperparameter_search(
    model: BaseEstimator, X: pl.DataFrame, y: pl.Series, metric: str
) -> Tuple[float, Dict[str, object]]:
    """
    Perform hyperparameter search using RandomizedSearchCV.

    Args:
        model (BaseEstimator): The model to optimize.
        X (pl.DataFrame): Features for training.
        y (pl.Series): Target variable.
        metric (str): Score metric to use.

    Returns:
        Tuple[float, Dict[str, object]]: Best score and best parameters.
    """
    param_grid = {
        "max_depth": [2, 3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
        "criterion": ["gini", "entropy"],
    }
    folds = X["fold"].unique()

    search = RandomizedSearchCV(
        model,
        param_grid,
        cv=get_cv_iterable(folds, "fold", X),
        scoring=metric,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X.drop("fold"), y)
    return search.best_score_, search.best_params_


def calculate_feature_importance(model: BaseEstimator, X: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate feature importance from a trained Decision Tree.

    Args:
        model (BaseEstimator): Trained pipeline.
        X (pl.DataFrame): Features used in training.

    Returns:
        pl.DataFrame: Feature importance sorted in descending order.
    """
    feature_names = [col for col in X.columns if col != "fold"]
    importances = model.feature_importances_
    importance_df = pl.DataFrame({"feature": feature_names, "importance": importances})

    return importance_df.sort("importance", descending=True)


def save_feature_importance_plot(
    feature_importance: pl.DataFrame, filename: str
) -> None:
    """
    Save a bar plot of the top 10 most important features.

    Args:
        feature_importance (pl.DataFrame): DataFrame containing feature importance.
        filename (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x="importance", y="feature")
    plt.title("Top 10 Most Important Features")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_tree_visualization(
    model: BaseEstimator, feature_names: List[str], filename: str
) -> None:
    """
    Save a visualization of the trained Decision Tree.

    Args:
        pipeline (Pipeline): Trained pipeline.
        feature_names (pl.Index): Names of the features used in training.
        filename (str): Path to save the plot.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Normal", "Anomaly"],
        filled=True,
        rounded=True,
    )
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    random_state = 42
    metric = "precision"

    train_df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    # Prepare training data
    X_train = train_df.drop("is_anomaly")
    y_train = train_df["is_anomaly"]

    # Build and cross-validate the model
    model = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)

    # Perform hyperparameter tuning
    best_score, best_params = perform_hyperparameter_search(
        model, X_train, y_train, metric
    )
    print(f"Best {metric} score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    # Fit the pipeline with the best parameters
    model.set_params(**best_params)
    model.fit(X_train.drop("fold"), y_train)

    # Feature importance
    feature_importance = calculate_feature_importance(model, X_train)
    save_feature_importance_plot(feature_importance, "img/feature_importance.png")

    # Tree visualization
    save_tree_visualization(model, X_train.columns, "img/tree_visualization.svg")

    # Train evaluation
    train_metrics = evaluate(train_df.drop("fold"), model)

    # Test data evaluation
    test_metrics = evaluate(test_df, model, prefix="test_")

    # Prepare metadata
    model_filename = f"decision_tree_{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    metadata = {
        "best_score": best_score,
        "best_params": best_params,
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
