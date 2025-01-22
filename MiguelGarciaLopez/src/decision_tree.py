import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree


def perform_hyperparameter_search(
    model: DecisionTreeClassifier, X: pl.DataFrame, y: pl.Series
) -> Tuple[float, Dict[str, object]]:
    """
    Perform hyperparameter search using RandomizedSearchCV.

    Args:
        model (DecisionTreeClassifier): The model to optimize.
        X (pl.DataFrame): Features for training.
        y (pl.Series): Target variable.

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

    grid_search = RandomizedSearchCV(
        model,
        param_grid,
        cv=get_cv_iterable(folds, "fold", X),
        scoring="f1_micro",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X.drop("fold"), y)
    return grid_search.best_score_, grid_search.best_params_


def perform_cross_validation(
    model: DecisionTreeClassifier, X: pl.DataFrame, y: pl.Series
) -> None:
    """
    Perform cross-validation and compute metrics for each fold.

    Args:
        model (BaseEstimator): The model to evaluate.
        X (pl.DataFrame): Features for training, including a "fold" column.
        y (pl.Series): Target variable.
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
        f1 = f1_score(val_labels, predictions, average="micro")

        fold_metrics.append({"fold": fold, "f1_score": f1})

    # Calculate average metrics
    avg_f1 = sum([metric["f1_score"] for metric in fold_metrics]) / len(fold_metrics)
    fold_metrics.append({"fold": "average", "f1_score": avg_f1})

    # Save metrics to CSV
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv("data/cross_validation_metrics.csv", index=False)

    print("Cross-validation metrics saved to cross_validation_cv_metrics.csv")
    print(f"Average F1 score: {avg_f1:.4f}")


def calculate_feature_importance(
    model: DecisionTreeClassifier, X: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculate feature importance from a trained Decision Tree.

    Args:
        model (DecisionTreeClassifier): Trained pipeline.
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
    model: DecisionTreeClassifier, feature_names: List[str], filename: str
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


def evaluate_model(
    y_true: pl.Series, y_pred: pl.Series, y_prob: pl.DataFrame
) -> Dict[str, float]:
    """
    Evaluate the model performance using various metrics and return the classification report.

    Args:
        y_true (pl.Series): True labels.
        y_pred (pl.Series): Predicted labels.
        y_prob (pl.DataFrame): Predicted probabilities.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics and classification report.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": recall_score(y_true, y_pred, zero_division=0),
    }

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv("data/classification_report.csv")

    return metrics


def plot_roc_curve(y_true: pl.Series, y_prob: pl.DataFrame, filename: str) -> None:
    """
    Plot the ROC curve and save it to a file.

    Args:
        y_true (pl.Series): True labels.
        y_prob (pl.DataFrame): Predicted probabilities.
        filename (str): Path to save the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(y_true: pl.Series, y_pred: pl.Series, filename: str) -> None:
    """
    Plot the confusion matrix using Seaborn's heatmap and save it to a file.

    Args:
        y_true (pl.Series): True labels.
        y_pred (pl.Series): Predicted labels.
        filename (str): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def build_structure():
    """
    Creates the necessary project folder structure ('data', 'img', 'doc', 'src') if they do not exist.
    Moves files from the current directory into the appropriate folders based on their extensions:
    - .csv files to 'data'
    - .png and .jpg files to 'img'
    - .pdf files to 'doc'
    - .ipynb and .py files to 'src'
    """
    folders = ["data", "img", "doc", "src", "models"]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")

    current_dir = os.getcwd()

    for filename in os.listdir(current_dir):
        if os.path.isdir(filename):
            continue

        file_path = os.path.join(current_dir, filename)

        if filename.endswith(".csv"):
            shutil.move(file_path, os.path.join(current_dir, "data", filename))
            print(f"Moved {filename} to data/")

        elif filename.endswith(".png") or filename.endswith(".jpg"):
            shutil.move(file_path, os.path.join(current_dir, "img", filename))
            print(f"Moved {filename} to img/")

        elif filename.endswith(".pdf"):
            shutil.move(file_path, os.path.join(current_dir, "doc", filename))
            print(f"Moved {filename} to doc/")

        elif filename.endswith(".ipynb") or filename.endswith(".py"):
            shutil.move(file_path, os.path.join(current_dir, "src", filename))
            print(f"Moved {filename} to src/")


def eval(df: pl.DataFrame, model: DecisionTreeClassifier, prefix: str = ""):
    X = df.drop("is_anomaly")
    y = df["is_anomaly"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    metrics = evaluate_model(y, y_pred, y_prob)
    print(f"{prefix.replace('_', '').upper()} - Evaluation Metrics:", metrics)

    # Save ROC plot
    plot_roc_curve(y, y_prob, f"img/{prefix}roc_curve.png")

    # Save confusion matrix plot
    plot_confusion_matrix(y, y_pred, f"img/{prefix}confusion_matrix.png")

    return metrics


def get_cv_iterable(
    folds: list,
    fold_column: str,
    train: pl.DataFrame,
):
    """
    Generates cross-validation (CV) train-test splits for each fold based on a specified fold column in the dataframe.

    Args:
        folds (list): A list of fold identifiers. Each fold corresponds to a subset of the dataset to be used as the test set in each iteration.
        fold_column (str): The name of the column in the `train` DataFrame that contains the fold identifiers.
        train (pl.DataFrame): The training dataset containing the data and fold identifiers.

    Yields:
        tuple: A tuple of two elements:
            - train_indexes (Index): The indices of the training set for the current fold.
            - test_indexes (Index): The indices of the test set for the current fold.
    """
    train_pd = train.to_pandas()
    for fold in folds:
        test_indexes = train_pd[train_pd[fold_column] == fold].index
        train_indexes = train_pd[train_pd[fold_column] != fold].index
        yield (train_indexes, test_indexes)


def main():
    random_state = 42

    train_df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    # Prepare training data
    X_train = train_df.drop("is_anomaly")
    y_train = train_df["is_anomaly"]

    # Build and cross-validate the model
    model = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_micro")
    print(f"Cross-validation F1_micro scores: {cv_scores.mean():.4f}")
    print(f"Cross-validation F1_micro std_dv: {cv_scores.std():.4f}")

    # Perform hyperparameter tuning
    best_score, best_params = perform_hyperparameter_search(model, X_train, y_train)
    print(f"Best F1 score: {best_score:.4f}")
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
    #test_metrics = eval(test_df, model, prefix="test_")

    # Prepare metadata
    model_filename = f"decision_tree_{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    metadata = {
        "best_score": best_score,
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": [],
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
