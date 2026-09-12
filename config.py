"""
Configuration file for the thyroid ultrasound classification framework.

Study components:
    - Min-Max normalization
    - EfficientNet-B7
    - Capsule Network
    - Domestic Sheep Optimization Algorithm (DSOA)

The values below follow the hyperparameters reported in the manuscript.
"""

import os


# ============================================================
# DATASET
# ============================================================

DATASET_DIR = "data/thyroid_dataset"

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3

CLASS_NAMES = [
    "benign",
    "malignant",
    "normal"
]

# Dataset reported in the manuscript
TOTAL_IMAGES = 1607
BENIGN_IMAGES = 632
MALIGNANT_IMAGES = 804
NORMAL_IMAGES = 171


# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# Patient-level identifiers were not available in the
# publicly released dataset. Therefore, the study used
# an image-level 80:10:10 split.


# ============================================================
# PREPROCESSING
# ============================================================

NORMALIZATION_METHOD = "min-max"

MIN_VALUE = 0.0
MAX_VALUE = 1.0


# ============================================================
# DATA AUGMENTATION
# ============================================================

USE_AUGMENTATION = True

# Augmentation is applied only to the training data,
# after the dataset split.

AUGMENTATION_ROTATION = 15
AUGMENTATION_WIDTH_SHIFT = 0.10
AUGMENTATION_HEIGHT_SHIFT = 0.10
AUGMENTATION_HORIZONTAL_FLIP = True


# ============================================================
# OPTIMIZER / TRAINING
# ============================================================

OPTIMIZER = "Adam"

LEARNING_RATE = 0.0001

BATCH_SIZE = 16

MAX_EPOCHS = 50

DROPOUT_RATE = 0.50


# ============================================================
# EFFICIENTNET-B7
# ============================================================

BACKBONE = "EfficientNetB7"

# ImageNet initialization is used for the EfficientNet-B7
# backbone in the reference implementation.

USE_IMAGENET_WEIGHTS = True

# The exact fine-tuning fraction was not specified in the
# manuscript. It is therefore configurable in the model
# implementation rather than presented as a reported
# experimental value.


# ============================================================
# CAPSULE NETWORK
# ============================================================

CAPSULE_VECTOR_LENGTH = 16

ROUTING_ITERATIONS = 3


# ============================================================
# DSOA
# ============================================================

DSOA_POPULATION_SIZE = 30

DSOA_MAX_ITERATIONS = 100

DSOA_LEARNING_RATE_MIN = 0.0001
DSOA_LEARNING_RATE_MAX = 0.01

DSOA_CAPSULE_DIM_MIN = 8
DSOA_CAPSULE_DIM_MAX = 32

DSOA_ROUTING_MIN = 2
DSOA_ROUTING_MAX = 5

DSOA_ALPHA = 0.5
DSOA_BETA = 1.5


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = "outputs"

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")


# ============================================================
# MODEL FILES
# ============================================================

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "thyroid_hybrid_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "thyroid_hybrid_final.keras"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
