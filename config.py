from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = ROOT / "checkpoints"

SEED = 42
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
CLASS_NAMES = ["benign", "malignant", "normal"]

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
CAPSULE_DIM = 16
ROUTING_ITERS = 3
EPOCHS = 50
DROPOUT = 0.5

DSOA_POPULATION = 30
DSOA_ITERATIONS = 100
LR_BOUNDS = (1e-4, 1e-2)
CAPSULE_DIM_BOUNDS = (8, 32)
ROUTING_BOUNDS = (2, 5)
ALPHA = 0.5
BETA = 1.5

FINE_TUNE_FRACTION = 0.20
