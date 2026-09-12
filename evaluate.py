"""
Evaluation module for thyroid ultrasound classification.

Metrics:
    - Accuracy
    - Precision
    - Recall (Sensitivity)
    - F1-score
    - Specificity

A confusion matrix is also generated for the three classes:
    Benign
    Malignant
    Normal
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import (
    CLASS_NAMES,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    RESULTS_DIR,
)

from preprocessing import prepare_datasets

from capsule_network import (
    CapsuleLayer,
    CapsuleLength,
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_trained_model(model_path=BEST_MODEL_PATH):
    """
    Load the trained hybrid model.

    Parameters
    ----------
    model_path : str
        Path to the saved Keras model.

    Returns
    -------
    tf.keras.Model
        Loaded model.
    """

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "CapsuleLayer": CapsuleLayer,
            "CapsuleLength": CapsuleLength,
        }
    )

    return model


# ============================================================
# GET PREDICTIONS
# ============================================================

def get_predictions(
    model,
    test_dataset
):
    """
    Generate predictions for the test dataset.

    Returns
    -------
    true_labels : np.ndarray
        True class indices.

    predicted_labels : np.ndarray
        Predicted class indices.

    probabilities : np.ndarray
        Predicted class probabilities.
    """

    probabilities = model.predict(
        test_dataset,
        verbose=1
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    true_labels = []

    for _, labels in test_dataset:

        labels = labels.numpy()

        true_labels.extend(
            np.argmax(
                labels,
                axis=1
            )
        )

    true_labels = np.asarray(
        true_labels,
        dtype=np.int64
    )

    return (
        true_labels,
        predicted_labels,
        probabilities
    )


# ============================================================
# SPECIFICITY
# ============================================================

def calculate_specificity(
    true_labels,
    predicted_labels,
    num_classes
):
    """
    Calculate class-wise and macro-average specificity.

    For each class:

        Specificity = TN / (TN + FP)

    Returns
    -------
    class_specificity : np.ndarray
        Specificity for each class.

    macro_specificity : float
        Macro-average specificity.
    """

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(num_classes)
    )

    class_specificity = []

    for class_index in range(
        num_classes
    ):

        true_positive = matrix[
            class_index,
            class_index
        ]

        false_positive = (
            np.sum(
                matrix[:, class_index]
            )
            - true_positive
        )

        false_negative = (
            np.sum(
                matrix[class_index, :]
            )
            - true_positive
        )

        true_negative = (
            np.sum(matrix)
            - true_positive
            - false_positive
            - false_negative
        )

        denominator = (
            true_negative
            + false_positive
        )

        if denominator == 0:
            specificity = 0.0
        else:
            specificity = (
                true_negative
                / denominator
            )

        class_specificity.append(
            specificity
        )

    class_specificity = np.asarray(
        class_specificity,
        dtype=np.float64
    )

    macro_specificity = float(
        np.mean(class_specificity)
    )

    return (
        class_specificity,
        macro_specificity
    )


# ============================================================
# CALCULATE ALL METRICS
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels
):
    """
    Calculate the classification metrics.

    Returns
    -------
    metrics : dict
        Overall classification metrics.

    class_specificity : np.ndarray
        Specificity for each class.

    confusion : np.ndarray
        Confusion matrix.
    """

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    (
        class_specificity,
        macro_specificity
    ) = calculate_specificity(
        true_labels,
        predicted_labels,
        len(CLASS_NAMES)
    )

    confusion = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(
            len(CLASS_NAMES)
        )
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": macro_specificity,
    }

    return (
        metrics,
        class_specificity,
        confusion
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_metrics(
    metrics,
    class_specificity,
    confusion
):
    """
    Display evaluation results.
    """

    print(
        "\nClassification Results"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy    : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision   : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall      : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-score    : "
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    print(
        f"Specificity : "
        f"{metrics['specificity'] * 100:.2f}%"
    )

    print(
        "\nClass-wise Specificity"
    )

    print(
        "-" * 60
    )

    for class_name, value in zip(
        CLASS_NAMES,
        class_specificity
    ):

        print(
            f"{class_name:<15}: "
            f"{value * 100:.2f}%"
        )

    print(
        "\nConfusion Matrix"
    )

    print(
        "-" * 60
    )

    print(
        confusion
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_metrics(
    metrics,
    class_specificity,
    confusion,
    output_directory=RESULTS_DIR
):
    """
    Save evaluation results to CSV files.
    """

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    metrics_dataframe = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "Specificity",
            ],
            "Value": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["specificity"],
            ],
        }
    )

    metrics_path = os.path.join(
        output_directory,
        "classification_metrics.csv"
    )

    metrics_dataframe.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Class-wise specificity
    # --------------------------------------------------------

    specificity_dataframe = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Specificity": class_specificity,
        }
    )

    specificity_path = os.path.join(
        output_directory,
        "class_specificity.csv"
    )

    specificity_dataframe.to_csv(
        specificity_path,
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion_dataframe = pd.DataFrame(
        confusion,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    confusion_path = os.path.join(
        output_directory,
        "confusion_matrix.csv"
    )

    confusion_dataframe.to_csv(
        confusion_path
    )

    print(
        f"\nResults saved to: "
        f"{output_directory}"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    true_labels,
    predicted_labels,
    output_directory=RESULTS_DIR
):
    """
    Save the detailed class-wise classification report.
    """

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    report_path = os.path.join(
        output_directory,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    print(
        f"Classification report saved to: "
        f"{report_path}"
    )


# ============================================================
# COMPLETE EVALUATION PIPELINE
# ============================================================

def evaluate_model(
    model_path=BEST_MODEL_PATH
):
    """
    Complete evaluation pipeline.

    Parameters
    ----------
    model_path : str
        Saved model path.

    Returns
    -------
    metrics : dict
        Overall metrics.
    """

    print(
        "\nThyroid Ultrasound Model Evaluation"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading trained model..."
    )

    model = load_trained_model(
        model_path
    )

    # --------------------------------------------------------
    # Prepare test dataset
    # --------------------------------------------------------

    print(
        "\nPreparing test dataset..."
    )

    (
        _,
        _,
        test_dataset,
        split_data
    ) = prepare_datasets(
        BATCH_SIZE
    )

    print(
        f"Test samples: "
        f"{len(split_data['test_paths'])}"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    (
        true_labels,
        predicted_labels,
        probabilities
    ) = get_predictions(
        model,
        test_dataset
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    (
        metrics,
        class_specificity,
        confusion
    ) = calculate_metrics(
        true_labels,
        predicted_labels
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print_metrics(
        metrics,
        class_specificity,
        confusion
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_metrics(
        metrics,
        class_specificity,
        confusion
    )

    save_classification_report(
        true_labels,
        predicted_labels
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate_model()
