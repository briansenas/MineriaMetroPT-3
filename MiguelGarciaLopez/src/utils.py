import os
import shutil
from typing import Dict

import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
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


def evaluate_model(y_true: pl.Series, y_pred: pl.Series) -> Dict[str, float]:
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
    }

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv("data/classification_report.csv")

    return metrics
