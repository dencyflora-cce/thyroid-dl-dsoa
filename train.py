"""
Training module for the hybrid thyroid ultrasound classifier.

Training configuration:
    Optimizer      : Adam
    Learning rate  : 0.0001
    Batch size     : 16
    Maximum epochs : 50

The training data are augmented in preprocessing.py.
Validation and test data are not augmented.
"""

import os

import tensorflow as tf

from config import (
    BATCH_SIZE,
    MAX_EPOCHS,
    LEARNING_RATE,
    CAPSULE_VECTOR_LENGTH,
    ROUTING_ITERATIONS,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    RANDOM_SEED,
)

from model import build_hybrid_model
from preprocessing import prepare_datasets


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# TRAINING
# ============================================================

def train_model(
    train_dataset,
    validation_dataset,
    learning_rate=LEARNING_RATE,
    capsule_dimension=CAPSULE_VECTOR_LENGTH,
    routing_iterations=ROUTING_ITERATIONS,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
):
    """
    Train the hybrid EfficientNet-B7 + Capsule Network model.

    Parameters
    ----------
    train_dataset : tf.data.Dataset
        Training dataset.

    validation_dataset : tf.data.Dataset
        Validation dataset.

    learning_rate : float
        Adam learning rate.

    capsule_dimension : int
        Capsule vector dimension.

    routing_iterations : int
        Dynamic-routing iterations.

    epochs : int
        Maximum number of training epochs.

    batch_size : int
        Batch size.

    Returns
    -------
    model : tf.keras.Model
        Trained model.

    history : tf.keras.callbacks.History
        Training history.
    """

    print("\nBuilding hybrid model...")
    print("=" * 60)

    model = build_hybrid_model(
        learning_rate=learning_rate,
        capsule_dimension=capsule_dimension,
        routing_iterations=routing_iterations,
        trainable_backbone=True,
    )

    print("\nTraining configuration")
    print("-" * 60)

    print(
        f"Learning rate       : {learning_rate}"
    )

    print(
        f"Batch size          : {batch_size}"
    )

    print(
        f"Maximum epochs      : {epochs}"
    )

    print(
        f"Capsule dimension   : {capsule_dimension}"
    )

    print(
        f"Routing iterations  : {routing_iterations}"
    )

    # --------------------------------------------------------
    # Save the best validation model.
    #
    # No EarlyStopping is used here because the manuscript
    # specifies a maximum of 50 epochs rather than reporting
    # an early-stopping criterion.
    # --------------------------------------------------------

    checkpoint_directory = os.path.dirname(
        BEST_MODEL_PATH
    )

    if checkpoint_directory:
        os.makedirs(
            checkpoint_directory,
            exist_ok=True
        )

    checkpoint_callback = (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
    )

    # --------------------------------------------------------
    # Train the model.
    # --------------------------------------------------------

    print(
        "\nStarting training..."
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            checkpoint_callback
        ],
        verbose=1,
    )

    # --------------------------------------------------------
    # Save final model.
    # --------------------------------------------------------

    final_directory = os.path.dirname(
        FINAL_MODEL_PATH
    )

    if final_directory:
        os.makedirs(
            final_directory,
            exist_ok=True
        )

    model.save(
        FINAL_MODEL_PATH
    )

    print(
        "\nTraining completed."
    )

    print(
        f"Best model  : {BEST_MODEL_PATH}"
    )

    print(
        f"Final model : {FINAL_MODEL_PATH}"
    )

    return model, history


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_training_history(
    history,
    output_path="outputs/results/training_history.csv"
):
    """
    Save training and validation metrics to CSV.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history.

    output_path : str
        Destination CSV file.
    """

    import pandas as pd

    directory = os.path.dirname(
        output_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    history_dataframe = pd.DataFrame(
        history.history
    )

    history_dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Training history saved to: "
        f"{output_path}"
    )


# ============================================================
# DISPLAY TRAINING SUMMARY
# ============================================================

def print_training_summary(
    history
):
    """
    Display the final training metrics.
    """

    history_data = history.history

    print(
        "\nTraining Summary"
    )

    print(
        "=" * 60
    )

    if "accuracy" in history_data:

        print(
            f"Final training accuracy   : "
            f"{history_data['accuracy'][-1]:.4f}"
        )

    if "val_accuracy" in history_data:

        print(
            f"Final validation accuracy : "
            f"{history_data['val_accuracy'][-1]:.4f}"
        )

    if "loss" in history_data:

        print(
            f"Final training loss       : "
            f"{history_data['loss'][-1]:.4f}"
        )

    if "val_loss" in history_data:

        print(
            f"Final validation loss     : "
            f"{history_data['val_loss'][-1]:.4f}"
        )


# ============================================================
# COMPLETE TRAINING PIPELINE
# ============================================================

def run_training():
    """
    Prepare datasets and train the hybrid model.
    """

    print(
        "\nThyroid Ultrasound Classification"
    )

    print(
        "Training Pipeline"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Prepare datasets.
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        split_data
    ) = prepare_datasets(
        BATCH_SIZE
    )

    print(
        f"\nTraining samples   : "
        f"{len(split_data['train_paths'])}"
    )

    print(
        f"Validation samples : "
        f"{len(split_data['validation_paths'])}"
    )

    print(
        f"Test samples       : "
        f"{len(split_data['test_paths'])}"
    )

    # --------------------------------------------------------
    # Train model.
    # --------------------------------------------------------

    model, history = train_model(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        learning_rate=LEARNING_RATE,
        capsule_dimension=CAPSULE_VECTOR_LENGTH,
        routing_iterations=ROUTING_ITERATIONS,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Save history.
    # --------------------------------------------------------

    save_training_history(
        history
    )

    # --------------------------------------------------------
    # Print summary.
    # --------------------------------------------------------

    print_training_summary(
        history
    )

    return (
        model,
        history,
        test_dataset,
        split_data
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_training()
