"""
EfficientNet-B7 backbone for thyroid ultrasound classification.

This module provides the EfficientNet-B7 feature extractor used
as the convolutional backbone of the hybrid model.
"""

import tensorflow as tf

from config import IMAGE_SIZE, USE_IMAGENET_WEIGHTS


# ============================================================
# EFFICIENTNET-B7 BACKBONE
# ============================================================

def build_efficientnet_b7(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    trainable=True
):
    """
    Build the EfficientNet-B7 feature-extraction backbone.

    Parameters
    ----------
    input_shape : tuple
        Input image dimensions.

    trainable : bool
        Whether the EfficientNet-B7 layers are trainable.

    Returns
    -------
    tf.keras.Model
        EfficientNet-B7 backbone.
    """

    if USE_IMAGENET_WEIGHTS:
        weights = "imagenet"
    else:
        weights = None

    backbone = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )

    backbone.trainable = trainable

    return backbone


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

def extract_features(
    images,
    trainable=False
):
    """
    Extract convolutional feature maps from images.

    Parameters
    ----------
    images : tf.Tensor
        Input images.

    trainable : bool
        Whether the backbone should be trainable.

    Returns
    -------
    tf.Tensor
        EfficientNet-B7 feature maps.
    """

    backbone = build_efficientnet_b7(
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        trainable=trainable
    )

    features = backbone(
        images,
        training=trainable
    )

    return features


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_backbone_summary():
    """
    Display the EfficientNet-B7 architecture summary.
    """

    backbone = build_efficientnet_b7(
        trainable=True
    )

    print("\nEfficientNet-B7 Backbone")
    print("=" * 60)

    backbone.summary()


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Initializing EfficientNet-B7 backbone..."
    )

    model = build_efficientnet_b7(
        trainable=True
    )

    print(
        f"\nInput shape  : {model.input_shape}"
    )

    print(
        f"Output shape : {model.output_shape}"
    )

    print(
        "\nEfficientNet-B7 backbone initialized successfully."
    )
