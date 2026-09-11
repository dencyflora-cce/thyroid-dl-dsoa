import sys

from config import (
    ALPHA,
    BETA,
    CAPSULE_DIM_BOUNDS,
    DSOA_ITERATIONS,
    DSOA_POPULATION,
    LR_BOUNDS,
    ROUTING_BOUNDS,
)
from src.dsoa import DSOA
from src.evaluate import evaluate
from src.train import train


def dsoa_smoke_test():
    """Exercise the DSOA software independently of GPU/model training."""

    def objective(params):
        # Deterministic toy objective used only to verify the optimizer API.
        return (
            (params["learning_rate"] - 0.001) ** 2 * 1e6
            + (params["capsule_dim"] - 16) ** 2
            + (params["routing_iters"] - 3) ** 2
        )

    optimizer = DSOA(
        objective=objective,
        population_size=DSOA_POPULATION,
        iterations=DSOA_ITERATIONS,
        lr_bounds=LR_BOUNDS,
        capsule_bounds=CAPSULE_DIM_BOUNDS,
        routing_bounds=ROUTING_BOUNDS,
        alpha=ALPHA,
        beta=BETA,
        seed=42,
    )
    best, fitness, history = optimizer.optimize()
    print("DSOA smoke test")
    print("best:", best)
    print("fitness:", fitness)
    print("iterations recorded:", len(history) - 1)


COMMANDS = {
    "train": train,
    "evaluate": evaluate,
    "dsoa-smoke": dsoa_smoke_test,
}


if __name__ == "__main__":
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "train"
    if command not in COMMANDS:
        raise SystemExit("Use: train, evaluate, or dsoa-smoke")
    COMMANDS[command]()
