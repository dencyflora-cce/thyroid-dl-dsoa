"""
Main execution script for the thyroid ultrasound classification
framework.

Pipeline
--------
1. Prepare the dataset.
2. Define the DSOA fitness function.
3. Run DSOA for hyperparameter optimization.
4. Build and train the hybrid EfficientNet-B7 + Capsule model.
5. Evaluate the trained model.

The DSOA fitness function used here is a configurable reference
function. The manuscript does not provide sufficient detail to
reconstruct the exact historical training objective and therefore
this implementation does not claim to reproduce the reported
98.17% result exactly.
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    CAPSULE_VECTOR_LENGTH,
    ROUTING_ITERATIONS,
    MAX_EPOCHS,
    RANDOM_SEED,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
)

from preprocessing import prepare_datasets
from dsoa import optimize, decode_solution
from model import build_hybrid_model
from evaluate import evaluate_model


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data():
    """
    Prepare training, validation, and test datasets.

    Returns
    -------
    tuple
        Training, validation, test datasets and split data.
    """

    print("\nPreparing dataset...")
    print("=" * 60)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        split_data
    ) = prepare_datasets(
        BATCH_SIZE
    )

    print(
        f"Training samples   : "
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

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        split_data
    )


# ============================================================
# DSOA FITNESS FUNCTION
# ============================================================

def create_fitness_function(
    train_dataset,
    validation_dataset,
    epochs=1
):
    """
    Create the validation-based fitness function used by DSOA.

    Lower validation loss corresponds to better fitness.

    Parameters
    ----------
    train_dataset : tf.data.Dataset
        Training data.

    validation_dataset : tf.data.Dataset
        Validation data.

    epochs : int
        Number of epochs used for each optimization trial.

    Returns
    -------
    callable
        DSOA fitness function.
    """

    def fitness_function(parameters):
        """
        Evaluate one candidate hyperparameter configuration.
        """

        learning_rate = (
            parameters["learning_rate"]
        )

        capsule_dimension = (
            parameters["capsule_dimension"]
        )

        routing_iterations = (
            parameters["routing_iterations"]
        )

        print(
            "\nEvaluating DSOA candidate:"
        )

        print(
            f"  Learning rate      : "
            f"{learning_rate:.6f}"
        )

        print(
            f"  Capsule dimension  : "
            f"{capsule_dimension}"
        )

        print(
            f"  Routing iterations : "
            f"{routing_iterations}"
        )

        # ----------------------------------------------------
        # Build a fresh model for the candidate.
        # ----------------------------------------------------

        model = build_hybrid_model(
            learning_rate=learning_rate,
            capsule_dimension=capsule_dimension,
            routing_iterations=routing_iterations,
            trainable_backbone=True,
        )

        # ----------------------------------------------------
        # Train candidate.
        # ----------------------------------------------------

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            verbose=0,
        )

        # ----------------------------------------------------
        # Validation loss is used as the minimization fitness.
        # ----------------------------------------------------

        validation_losses = (
            history.history.get(
                "val_loss",
                []
            )
        )

        if not validation_losses:
            raise RuntimeError(
                "Validation loss was not recorded."
            )

        fitness = float(
            validation_losses[-1]
        )

        print(
            f"  Validation loss   : "
            f"{fitness:.6f}"
        )

        # Free resources associated with the candidate model.
        tf.keras.backend.clear_session()

        return fitness

    return fitness_function


# ============================================================
# RUN DSOA
# ============================================================

def run_dsoa(
    train_dataset,
    validation_dataset,
    population_size=30,
    max_iterations=100,
    optimization_epochs=1,
):
    """
    Run DSOA hyperparameter optimization.

    Parameters
    ----------
    train_dataset : tf.data.Dataset
        Training dataset.

    validation_dataset : tf.data.Dataset
        Validation dataset.

    population_size : int
        Number of DSOA candidate solutions.

    max_iterations : int
        Number of optimization iterations.

    optimization_epochs : int
        Training epochs used during each candidate evaluation.

    Returns
    -------
    dict
        DSOA optimization result.
    """

    print(
        "\nStarting Domestic Sheep Optimization Algorithm..."
    )

    print(
        "=" * 60
    )

    fitness_function = create_fitness_function(
        train_dataset,
        validation_dataset,
        epochs=optimization_epochs
    )

    result = optimize(
        fitness_function=fitness_function,
        population_size=population_size,
        max_iterations=max_iterations,
        seed=RANDOM_SEED
    )

    print(
        "\nDSOA optimization completed."
    )

    print(
        "\nBest hyperparameters:"
    )

    for key, value in (
        result["best_parameters"].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print(
        f"\nBest validation fitness: "
        f"{result['best_fitness']:.6f}"
    )

    return result


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(
    train_dataset,
    validation_dataset,
    parameters
):
    """
    Train the final hybrid model using selected parameters.

    Parameters
    ----------
    train_dataset : tf.data.Dataset
        Training dataset.

    validation_dataset : tf.data.Dataset
        Validation dataset.

    parameters : dict
        Selected hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Trained model.

    history : tf.keras.callbacks.History
        Training history.
    """

    print(
        "\nTraining final hybrid model..."
    )

    print(
        "=" * 60
    )

    model = build_hybrid_model(
        learning_rate=parameters[
            "learning_rate"
        ],
        capsule_dimension=parameters[
            "capsule_dimension"
        ],
        routing_iterations=parameters[
            "routing_iterations"
        ],
        trainable_backbone=True,
    )

    checkpoint_callback = (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=[
            checkpoint_callback
        ],
        verbose=1
    )

    os.makedirs(
        os.path.dirname(
            FINAL_MODEL_PATH
        ),
        exist_ok=True
    )

    model.save(
        FINAL_MODEL_PATH
    )

    print(
        f"\nFinal model saved to:"
        f" {FINAL_MODEL_PATH}"
    )

    return (
        model,
        history
    )


# ============================================================
# DEFAULT HYPERPARAMETERS
# ============================================================

def get_default_parameters():
    """
    Return the default hyperparameters reported in the manuscript.

    These values are used when DSOA optimization is skipped.
    """

    return {
        "learning_rate": LEARNING_RATE,
        "capsule_dimension": CAPSULE_VECTOR_LENGTH,
        "routing_iterations": ROUTING_ITERATIONS,
    }


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_pipeline(
    use_dsoa=True,
    dsoa_population=30,
    dsoa_iterations=100,
    optimization_epochs=1
):
    """
    Execute the complete classification pipeline.

    Parameters
    ----------
    use_dsoa : bool
        Whether to perform DSOA optimization.

    dsoa_population : int
        DSOA population size.

    dsoa_iterations : int
        Maximum DSOA iterations.

    optimization_epochs : int
        Epochs per DSOA candidate evaluation.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "THYROID ULTRASOUND CLASSIFICATION"
    )

    print(
        "EfficientNet-B7 + Capsule Network + DSOA"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Step 1: Prepare data
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        split_data
    ) = prepare_data()

    # --------------------------------------------------------
    # Step 2: Optimize or use reported default parameters
    # --------------------------------------------------------

    if use_dsoa:

        dsoa_result = run_dsoa(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            population_size=dsoa_population,
            max_iterations=dsoa_iterations,
            optimization_epochs=optimization_epochs
        )

        selected_parameters = (
            dsoa_result["best_parameters"]
        )

    else:

        print(
            "\nDSOA optimization skipped."
        )

        print(
            "Using manuscript-reported default "
            "hyperparameters."
        )

        selected_parameters = (
            get_default_parameters()
        )

    # --------------------------------------------------------
    # Step 3: Train final model
    # --------------------------------------------------------

    (
        model,
        history
    ) = train_final_model(
        train_dataset,
        validation_dataset,
        selected_parameters
    )

    # --------------------------------------------------------
    # Step 4: Evaluate best saved model
    # --------------------------------------------------------

    print(
        "\nEvaluating final model..."
    )

    metrics = evaluate_model(
        BEST_MODEL_PATH
    )

    # --------------------------------------------------------
    # Step 5: Display final results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL EVALUATION"
    )

    print(
        "=" * 70
    )

    for metric_name, value in (
        metrics.items()
    ):

        print(
            f"{metric_name:<15}: "
            f"{value * 100:.2f}%"
        )

    print(
        "\nPipeline completed successfully."
    )

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "parameters": selected_parameters,
        "test_dataset": test_dataset,
        "split_data": split_data,
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Thyroid ultrasound classification using "
            "EfficientNet-B7, Capsule Network and DSOA."
        )
    )

    parser.add_argument(
        "--skip-dsoa",
        action="store_true",
        help=(
            "Skip DSOA optimization and use the "
            "manuscript-reported default hyperparameters."
        )
    )

    parser.add_argument(
        "--dsoa-population",
        type=int,
        default=30,
        help="DSOA population size."
    )

    parser.add_argument(
        "--dsoa-iterations",
        type=int,
        default=100,
        help="Maximum DSOA iterations."
    )

    parser.add_argument(
        "--optimization-epochs",
        type=int,
        default=1,
        help=(
            "Number of epochs used for each DSOA "
            "candidate evaluation."
        )
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    run_pipeline(
        use_dsoa=not args.skip_dsoa,
        dsoa_population=args.dsoa_population,
        dsoa_iterations=args.dsoa_iterations,
        optimization_epochs=args.optimization_epochs
    )
