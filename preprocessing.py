"""
Preprocessing module for thyroid ultrasound image classification.

Operations:
    1. Load images from class-wise directories.
    2. Resize images to 224 x 224.
    3. Apply Min-Max normalization.
    4. Split data into training, validation, and test sets.
    5. Apply augmentation only to the training set.

Expected directory structure:

data/
└── thyroid_dataset/
    ├── benign/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── malignant/
    │   ├── image1.jpg
    │   └── ...
    └── normal/
        ├── image1.jpg
        └── ...
"""

import os
import random

import cv2
import numpy as np
import tensorflow as tf

from config import (
    DATASET_DIR,
    IMAGE_SIZE,
    CLASS_NAMES,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    AUGMENTATION_ROTATION,
    AUGMENTATION_WIDTH_SHIFT,
    AUGMENTATION_HEIGHT_SHIFT,
    AUGMENTATION_HORIZONTAL_FLIP,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Load and preprocess one ultrasound image.

    Parameters
    ----------
    image_path : str
        Path to the image.

    Returns
    -------
    np.ndarray
        Preprocessed RGB image with values in [0, 1].
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # OpenCV loads images in BGR format.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to the input size used by EfficientNet-B7.
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Convert to floating-point representation.
    image = image.astype(np.float32)

    # --------------------------------------------------------
    # Min-Max normalization
    #
    # X_normalized = (X - X_min) / (X_max - X_min)
    # --------------------------------------------------------

    image_min = np.min(image)
    image_max = np.max(image)

    if image_max > image_min:
        image = (
            (image - image_min)
            / (image_max - image_min)
        )
    else:
        image = np.zeros_like(image)

    return image


# ============================================================
# DATASET DISCOVERY
# ============================================================

def collect_image_paths():
    """
    Collect image paths and corresponding class labels.

    Returns
    -------
    image_paths : list
        List containing image file paths.

    labels : list
        Integer class labels.
    """

    image_paths = []
    labels = []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_directory = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.isdir(class_directory):
            raise FileNotFoundError(
                f"Class directory not found: "
                f"{class_directory}"
            )

        for filename in sorted(
            os.listdir(class_directory)
        ):

            if filename.lower().endswith(
                valid_extensions
            ):

                image_path = os.path.join(
                    class_directory,
                    filename
                )

                image_paths.append(image_path)
                labels.append(class_index)

    if len(image_paths) == 0:
        raise ValueError(
            "No images were found in the dataset directory."
        )

    return image_paths, labels


# ============================================================
# DATA SPLITTING
# ============================================================

def split_dataset(image_paths, labels):
    """
    Perform an image-level 80:10:10 train/validation/test split.

    Parameters
    ----------
    image_paths : list
        Image file paths.

    labels : list
        Corresponding integer labels.

    Returns
    -------
    train_paths, train_labels,
    validation_paths, validation_labels,
    test_paths, test_labels
    """

    if not np.isclose(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO,
        1.0
    ):
        raise ValueError(
            "Train, validation, and test ratios must sum to 1."
        )

    combined = list(
        zip(image_paths, labels)
    )

    random.Random(RANDOM_SEED).shuffle(combined)

    total_samples = len(combined)

    train_end = int(
        TRAIN_RATIO * total_samples
    )

    validation_end = train_end + int(
        VALIDATION_RATIO * total_samples
    )

    train_data = combined[:train_end]

    validation_data = combined[
        train_end:validation_end
    ]

    test_data = combined[
        validation_end:
    ]

    train_paths, train_labels = zip(
        *train_data
    )

    validation_paths, validation_labels = zip(
        *validation_data
    )

    test_paths, test_labels = zip(
        *test_data
    )

    return (
        list(train_paths),
        list(train_labels),
        list(validation_paths),
        list(validation_labels),
        list(test_paths),
        list(test_labels)
    )


# ============================================================
# LOAD DATA ARRAYS
# ============================================================

def load_dataset(image_paths, labels):
    """
    Load a list of images into NumPy arrays.

    Returns
    -------
    images : np.ndarray
        Image array.

    labels : np.ndarray
        One-hot encoded labels.
    """

    images = []

    for image_path in image_paths:
        images.append(
            load_image(image_path)
        )

    images = np.asarray(
        images,
        dtype=np.float32
    )

    labels = tf.keras.utils.to_categorical(
        labels,
        num_classes=len(CLASS_NAMES)
    )

    labels = np.asarray(
        labels,
        dtype=np.float32
    )

    return images, labels


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_training_augmentation():
    """
    Create the training-data augmentation pipeline.

    Augmentation is applied only to training data.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=AUGMENTATION_ROTATION / 360.0
            ),

            tf.keras.layers.RandomTranslation(
                height=AUGMENTATION_HEIGHT_SHIFT,
                width=AUGMENTATION_WIDTH_SHIFT
            ),

            tf.keras.layers.RandomFlip(
                mode="horizontal"
                if AUGMENTATION_HORIZONTAL_FLIP
                else "vertical"
            ),
        ],
        name="training_augmentation"
    )


# ============================================================
# CREATE TENSORFLOW DATASETS
# ============================================================

def create_tf_dataset(
    images,
    labels,
    batch_size,
    training=False
):
    """
    Convert NumPy arrays into a TensorFlow dataset.

    Parameters
    ----------
    images : np.ndarray
        Input images.

    labels : np.ndarray
        One-hot encoded labels.

    batch_size : int
        Batch size.

    training : bool
        Whether this is the training dataset.

    Returns
    -------
    tf.data.Dataset
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(images),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

        augmentation = (
            create_training_augmentation()
        )

        dataset = dataset.map(
            lambda x, y: (
                augmentation(x, training=True),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def prepare_datasets(batch_size):
    """
    Prepare training, validation, and test datasets.

    Returns
    -------
    train_dataset : tf.data.Dataset
    validation_dataset : tf.data.Dataset
    test_dataset : tf.data.Dataset

    split_data : dict
        File paths and labels for each split.
    """

    image_paths, labels = (
        collect_image_paths()
    )

    (
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths,
        test_labels
    ) = split_dataset(
        image_paths,
        labels
    )

    # --------------------------------------------------------
    # Load each split independently.
    # --------------------------------------------------------

    train_images, train_labels = (
        load_dataset(
            train_paths,
            train_labels
        )
    )

    validation_images, validation_labels = (
        load_dataset(
            validation_paths,
            validation_labels
        )
    )

    test_images, test_labels = (
        load_dataset(
            test_paths,
            test_labels
        )
    )

    # --------------------------------------------------------
    # Augmentation is applied ONLY to training data.
    # --------------------------------------------------------

    train_dataset = create_tf_dataset(
        train_images,
        train_labels,
        batch_size,
        training=True
    )

    validation_dataset = create_tf_dataset(
        validation_images,
        validation_labels,
        batch_size,
        training=False
    )

    test_dataset = create_tf_dataset(
        test_images,
        test_labels,
        batch_size,
        training=False
    )

    split_data = {
        "train_paths": train_paths,
        "train_labels": train_labels,
        "validation_paths": validation_paths,
        "validation_labels": validation_labels,
        "test_paths": test_paths,
        "test_labels": test_labels
    }

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        split_data
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(split_data):
    """
    Print the number of samples in each split.
    """

    print("\nDataset Summary")
    print("-" * 40)

    print(
        f"Training images    : "
        f"{len(split_data['train_paths'])}"
    )

    print(
        f"Validation images  : "
        f"{len(split_data['validation_paths'])}"
    )

    print(
        f"Test images        : "
        f"{len(split_data['test_paths'])}"
    )

    print(
        f"Total images       : "
        f"{len(split_data['train_paths']) + "
        f"len(split_data['validation_paths']) + "
        f"len(split_data['test_paths'])}"
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    from config import BATCH_SIZE

    print("Thyroid ultrasound preprocessing module")
    print("=" * 50)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        split_data
    ) = prepare_datasets(
        BATCH_SIZE
    )

    print_dataset_summary(
        split_data
    )

    print("\nPreprocessing pipeline initialized successfully.")
