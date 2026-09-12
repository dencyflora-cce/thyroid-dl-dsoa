"""
Domestic Sheep Optimization Algorithm (DSOA).

This module provides a reference implementation of the optimization
procedure described in the manuscript.

DSOA is used to search for suitable hyperparameter configurations
based on validation-set fitness.

The manuscript specifies:
    - Population size: 30
    - Maximum iterations: 100
    - Learning-rate range: 0.0001 to 0.01
    - Capsule dimension: 8 to 32
    - Routing iterations: 2 to 5
    - Alpha: 0.5
    - Beta: 1.5

Important:
The manuscript describes the DSOA update procedure conceptually,
but the complete original mathematical update equation is not
available. Therefore, this file implements the described
exploration/exploitation procedure as a reference implementation.
"""


import numpy as np

from config import (
    DSOA_POPULATION_SIZE,
    DSOA_MAX_ITERATIONS,
    DSOA_LEARNING_RATE_MIN,
    DSOA_LEARNING_RATE_MAX,
    DSOA_CAPSULE_DIM_MIN,
    DSOA_CAPSULE_DIM_MAX,
    DSOA_ROUTING_MIN,
    DSOA_ROUTING_MAX,
    DSOA_ALPHA,
    DSOA_BETA,
    RANDOM_SEED,
)


# ============================================================
# SEARCH SPACE
# ============================================================

SEARCH_SPACE = {
    "learning_rate": (
        DSOA_LEARNING_RATE_MIN,
        DSOA_LEARNING_RATE_MAX
    ),

    "capsule_dimension": (
        DSOA_CAPSULE_DIM_MIN,
        DSOA_CAPSULE_DIM_MAX
    ),

    "routing_iterations": (
        DSOA_ROUTING_MIN,
        DSOA_ROUTING_MAX
    ),
}


# ============================================================
# PARAMETER BOUNDS
# ============================================================

def get_bounds():
    """
    Return the lower and upper bounds of the search space.

    Returns
    -------
    lower_bounds : np.ndarray
    upper_bounds : np.ndarray
    """

    lower_bounds = np.array(
        [
            SEARCH_SPACE["learning_rate"][0],
            SEARCH_SPACE["capsule_dimension"][0],
            SEARCH_SPACE["routing_iterations"][0],
        ],
        dtype=np.float64
    )

    upper_bounds = np.array(
        [
            SEARCH_SPACE["learning_rate"][1],
            SEARCH_SPACE["capsule_dimension"][1],
            SEARCH_SPACE["routing_iterations"][1],
        ],
        dtype=np.float64
    )

    return lower_bounds, upper_bounds


# ============================================================
# INITIALIZE POPULATION
# ============================================================

def initialize_population(
    population_size=DSOA_POPULATION_SIZE,
    seed=RANDOM_SEED
):
    """
    Initialize the sheep population randomly within the
    specified search space.

    Parameters
    ----------
    population_size : int
        Number of candidate solutions.

    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Population matrix.
    """

    rng = np.random.default_rng(seed)

    lower_bounds, upper_bounds = get_bounds()

    population = rng.uniform(
        lower_bounds,
        upper_bounds,
        size=(
            population_size,
            len(lower_bounds)
        )
    )

    return population


# ============================================================
# DECODE SOLUTION
# ============================================================

def decode_solution(solution):
    """
    Convert a continuous DSOA solution into usable
    hyperparameter values.

    Parameters
    ----------
    solution : np.ndarray
        Candidate solution vector.

    Returns
    -------
    dict
        Decoded hyperparameters.
    """

    learning_rate = float(
        solution[0]
    )

    capsule_dimension = int(
        round(solution[1])
    )

    routing_iterations = int(
        round(solution[2])
    )

    # Keep integer parameters within their specified bounds.
    capsule_dimension = int(
        np.clip(
            capsule_dimension,
            DSOA_CAPSULE_DIM_MIN,
            DSOA_CAPSULE_DIM_MAX
        )
    )

    routing_iterations = int(
        np.clip(
            routing_iterations,
            DSOA_ROUTING_MIN,
            DSOA_ROUTING_MAX
        )
    )

    learning_rate = float(
        np.clip(
            learning_rate,
            DSOA_LEARNING_RATE_MIN,
            DSOA_LEARNING_RATE_MAX
        )
    )

    return {
        "learning_rate": learning_rate,
        "capsule_dimension": capsule_dimension,
        "routing_iterations": routing_iterations,
    }


# ============================================================
# CLIPPING
# ============================================================

def clip_population(population):
    """
    Clip candidate solutions to the valid search space.

    Parameters
    ----------
    population : np.ndarray
        Candidate population.

    Returns
    -------
    np.ndarray
        Clipped population.
    """

    lower_bounds, upper_bounds = get_bounds()

    return np.clip(
        population,
        lower_bounds,
        upper_bounds
    )


# ============================================================
# FITNESS EVALUATION
# ============================================================

def evaluate_population(
    population,
    fitness_function
):
    """
    Evaluate all candidate solutions.

    Parameters
    ----------
    population : np.ndarray
        Candidate solutions.

    fitness_function : callable
        Function receiving a decoded parameter dictionary
        and returning a scalar fitness value.

    Returns
    -------
    np.ndarray
        Fitness values.
    """

    fitness_values = []

    for solution in population:

        parameters = decode_solution(
            solution
        )

        fitness = fitness_function(
            parameters
        )

        fitness_values.append(
            float(fitness)
        )

    return np.asarray(
        fitness_values,
        dtype=np.float64
    )


# ============================================================
# BEST SOLUTION
# ============================================================

def get_best_solution(
    population,
    fitness_values
):
    """
    Identify the best sheep according to minimum fitness.

    Parameters
    ----------
    population : np.ndarray
        Candidate solutions.

    fitness_values : np.ndarray
        Corresponding fitness values.

    Returns
    -------
    best_solution : np.ndarray
    best_fitness : float
    """

    best_index = int(
        np.argmin(fitness_values)
    )

    return (
        population[best_index].copy(),
        float(fitness_values[best_index])
    )


# ============================================================
# DSOA UPDATE
# ============================================================

def update_population(
    population,
    best_solution,
    iteration,
    max_iterations=DSOA_MAX_ITERATIONS,
    seed=None
):
    """
    Perform one DSOA population update.

    The update combines:

        1. Exploratory random perturbation.
        2. Attraction toward the best sheep.

    The alpha and beta coefficients control the relative
    contribution of the exploration and exploitation terms.

    Parameters
    ----------
    population : np.ndarray
        Current population.

    best_solution : np.ndarray
        Current best solution.

    iteration : int
        Current iteration number.

    max_iterations : int
        Maximum number of iterations.

    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        Updated population.
    """

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(
            seed + iteration
        )

    lower_bounds, upper_bounds = get_bounds()

    updated_population = (
        population.copy()
    )

    # Progress increases from approximately 0 to 1.
    progress = (
        iteration
        / max(1, max_iterations)
    )

    # Exploration decreases as optimization progresses.
    exploration_strength = (
        1.0 - progress
    )

    for i in range(
        len(population)
    ):

        # ----------------------------------------------------
        # Exploration component
        # ----------------------------------------------------

        random_direction = rng.uniform(
            -1.0,
            1.0,
            size=population.shape[1]
        )

        exploration = (
            DSOA_ALPHA
            * exploration_strength
            * random_direction
            * (upper_bounds - lower_bounds)
        )

        # ----------------------------------------------------
        # Exploitation component
        # ----------------------------------------------------

        attraction = (
            DSOA_BETA
            * rng.uniform(
                0.0,
                1.0,
                size=population.shape[1]
            )
            * (
                best_solution
                - population[i]
            )
        )

        # ----------------------------------------------------
        # Combined sheep movement
        # ----------------------------------------------------

        updated_population[i] = (
            population[i]
            + exploration
            + attraction
        )

    # --------------------------------------------------------
    # Boundary handling
    # --------------------------------------------------------

    updated_population = clip_population(
        updated_population
    )

    return updated_population


# ============================================================
# MAIN DSOA OPTIMIZER
# ============================================================

def optimize(
    fitness_function,
    population_size=DSOA_POPULATION_SIZE,
    max_iterations=DSOA_MAX_ITERATIONS,
    seed=RANDOM_SEED
):
    """
    Run the Domestic Sheep Optimization Algorithm.

    Parameters
    ----------
    fitness_function : callable
        Objective function to minimize.

    population_size : int
        Number of sheep/candidate solutions.

    max_iterations : int
        Maximum number of optimization iterations.

    seed : int
        Random seed.

    Returns
    -------
    dict
        Optimization result containing:

            best_parameters
            best_fitness
            best_solution
            history
    """

    # --------------------------------------------------------
    # Step 1: Initialize population
    # --------------------------------------------------------

    population = initialize_population(
        population_size=population_size,
        seed=seed
    )

    # --------------------------------------------------------
    # Step 2: Evaluate initial population
    # --------------------------------------------------------

    fitness_values = evaluate_population(
        population,
        fitness_function
    )

    # --------------------------------------------------------
    # Step 3: Identify initial best sheep
    # --------------------------------------------------------

    best_solution, best_fitness = (
        get_best_solution(
            population,
            fitness_values
        )
    )

    history = [
        best_fitness
    ]

    # --------------------------------------------------------
    # Step 4: Optimization iterations
    # --------------------------------------------------------

    for iteration in range(
        max_iterations
    ):

        population = update_population(
            population,
            best_solution,
            iteration,
            max_iterations=max_iterations,
            seed=seed
        )

        fitness_values = evaluate_population(
            population,
            fitness_function
        )

        current_best_solution, current_best_fitness = (
            get_best_solution(
                population,
                fitness_values
            )
        )

        # ----------------------------------------------------
        # Since this is a minimization problem, retain the
        # solution having the lowest fitness.
        # ----------------------------------------------------

        if current_best_fitness < best_fitness:

            best_solution = (
                current_best_solution.copy()
            )

            best_fitness = (
                current_best_fitness
            )

        history.append(
            best_fitness
        )

    # --------------------------------------------------------
    # Step 5: Decode final best solution
    # --------------------------------------------------------

    best_parameters = decode_solution(
        best_solution
    )

    return {
        "best_parameters": best_parameters,
        "best_fitness": best_fitness,
        "best_solution": best_solution,
        "history": history,
    }


# ============================================================
# TOY FITNESS FUNCTION
# ============================================================

def toy_fitness(parameters):
    """
    Simple deterministic objective used only for testing.

    This does NOT represent the neural-network training
    objective reported in the manuscript.
    """

    learning_rate_error = (
        parameters["learning_rate"]
        - 0.001
    ) ** 2

    capsule_error = (
        parameters["capsule_dimension"]
        - 16
    ) ** 2

    routing_error = (
        parameters["routing_iterations"]
        - 3
    ) ** 2

    return (
        learning_rate_error * 1000000
        + capsule_error
        + routing_error
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Domestic Sheep Optimization Algorithm"
    )

    print("=" * 60)

    print(
        "\nSearch space:"
    )

    for name, bounds in SEARCH_SPACE.items():

        print(
            f"{name}: {bounds}"
        )

    print(
        "\nRunning DSOA smoke test..."
    )

    result = optimize(
        fitness_function=toy_fitness,
        population_size=DSOA_POPULATION_SIZE,
        max_iterations=DSOA_MAX_ITERATIONS,
        seed=RANDOM_SEED
    )

    print(
        "\nBest parameters:"
    )

    for key, value in (
        result["best_parameters"].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print(
        f"\nBest fitness: "
        f"{result['best_fitness']:.6f}"
    )

    print(
        "\nDSOA smoke test completed successfully."
    )
