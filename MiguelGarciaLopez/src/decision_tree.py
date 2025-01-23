import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from utils import (
    build_structure,
    evaluate_model,
    get_cv_iterable,
    plot_confusion_matrix,
    plot_roc_curve,
)


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
        "max_depth": [2, 3, 5, 7, 10],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
        "criterion": ["gini", "entropy"],
    }
    folds = X["fold"].unique()

    grid_search = RandomizedSearchCV(
        model,
        param_grid,
        cv=get_cv_iterable(folds, "fold", X),
        scoring=metric,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X.drop("fold"), y)
    return grid_search.best_score_, grid_search.best_params_


def perform_cross_validation(
    model: BaseEstimator, X: pl.DataFrame, y: pl.Series, metric: str
) -> None:
    """
    Perform cross-validation and compute metrics for each fold.

    Args:
        model (BaseEstimator): The model to evaluate.
        X (pl.DataFrame): Features for training, including a "fold" column.
        y (pl.Series): Target variable.
        metric (str): Score metric to use.
    """
    folds = X["fold"].unique()
    fold_metrics: List[Dict[str, Union[int, float]]] = []

    # Iterate over each fold
    for fold in folds:
        train_data = X[X["fold"] != fold].drop(columns=["fold"])
        val_data = X[X["fold"] == fold].drop(columns=["fold"])
        train_labels = y[X["fold"] != fold]
        val_labels = y[X["fold"] == fold]

        # Train the model on the current training fold
        model.fit(train_data, train_labels)

        # Predict and evaluate on the validation fold
        predictions = model.predict(val_data)

        if metric == "accuracy":
            score = accuracy_score(val_labels, predictions)
        elif metric == "precision":
            score = precision_score(val_labels, predictions)
        elif metric == "recall":
            score = recall_score(val_labels, predictions)
        else:
            score = f1_score(val_labels, predictions)

        fold_metrics.append({"fold": fold, metric: score})

    # Calculate average metrics
    avg_score = sum([aux_metric[metric] for aux_metric in fold_metrics]) / len(
        fold_metrics
    )
    fold_metrics.append({"fold": "average", metric: avg_score})

    # Save metrics to CSV
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv("data/cross_validation_metrics.csv", index=False)

    print("Cross-validation metrics saved to cross_validation_cv_metrics.csv")
    print(f"Average {metric} score: {avg_score:.4f}")


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


def eval(df: pl.DataFrame, model: BaseEstimator, prefix: str = ""):
    X = df.drop("is_anomaly")
    y = df["is_anomaly"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    metrics = evaluate_model(y, y_pred)
    print(f"{prefix.replace('_', '').upper()} - Evaluation Metrics:", metrics)

    # Save ROC plot
    plot_roc_curve(y, y_prob, f"img/{prefix}roc_curve_decision_tree.png")

    # Save confusion matrix plot
    plot_confusion_matrix(y, y_pred, f"img/{prefix}confusion_matrix_decision_tree.png")

    return metrics


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
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=metric)
    print(f"Cross-validation {metric} scores: {cv_scores.mean():.4f}")
    print(f"Cross-validation {metric} std_dv: {cv_scores.std():.4f}")

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
    save_tree_visualization(model, X_train.columns, "img/tree_visualization.png")

    # Train evaluation
    train_metrics = eval(train_df.drop("fold"), model)

    # Test data evaluation
    test_metrics = eval(test_df, model, prefix="test_")

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
        "./MineriaMetroPT-3/MiguelGarciaLopez/src",
        "./MiguelGarciaLopez/src",
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
