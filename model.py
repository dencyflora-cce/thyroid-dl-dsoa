"""
Hybrid EfficientNet-B7 + Capsule Network model.

Architecture:
    Input Image
        ↓
    EfficientNet-B7
        ↓
    Feature Maps
        ↓
    Capsule Network
        ↓
    Capsule Length
        ↓
    Softmax
        ↓
    3-class prediction

Classes:
    0 - Benign
    1 - Malignant
    2 - Normal
"""

import tensorflow as tf

from config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    CAPSULE_VECTOR_LENGTH,
    ROUTING_ITERATIONS,
    DROPOUT_RATE,
)

from efficientnet_b7 import build_efficientnet_b7

from capsule_network import (
    CapsuleLayer,
    CapsuleLength,
)


# ============================================================
# HYBRID MODEL
# ============================================================

def build_hybrid_model(
    learning_rate=0.0001,
    capsule_dimension=CAPSULE_VECTOR_LENGTH,
    routing_iterations=ROUTING_ITERATIONS,
    trainable_backbone=True,
):
    """
    Build the EfficientNet-B7 + Capsule Network hybrid model.

    Parameters
    ----------
    learning_rate : float
        Adam optimizer learning rate.

    capsule_dimension : int
        Dimension of each output capsule.

    routing_iterations : int
        Number of dynamic-routing iterations.

    trainable_backbone : bool
        Whether EfficientNet-B7 weights are trainable.

    Returns
    -------
    tf.keras.Model
        Compiled hybrid classification model.
    """

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        name="thyroid_ultrasound_input"
    )

    # --------------------------------------------------------
    # EfficientNet-B7
    # --------------------------------------------------------

    backbone = build_efficientnet_b7(
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        trainable=trainable_backbone
    )

    features = backbone(
        inputs,
        training=trainable_backbone
    )

    # --------------------------------------------------------
    # Convert convolutional feature map into a sequence
    # of capsule inputs.
    # --------------------------------------------------------

    feature_dimension = (
        features.shape[-1]
    )

    if feature_dimension is None:
        raise ValueError(
            "Unable to determine EfficientNet-B7 "
            "feature dimension."
        )

    capsule_inputs = tf.keras.layers.Reshape(
        (
            -1,
            int(feature_dimension)
        ),
        name="feature_map_to_capsules"
    )(features)

    # --------------------------------------------------------
    # Capsule Network
    # --------------------------------------------------------

    capsules = CapsuleLayer(
        num_capsules=NUM_CLASSES,
        dim_capsule=capsule_dimension,
        routing_iterations=routing_iterations,
        name="thyroid_capsule_layer"
    )(capsule_inputs)

    # --------------------------------------------------------
    # Capsule vector lengths
    #
    # The length of each capsule represents its activation
    # magnitude for the corresponding class.
    # --------------------------------------------------------

    capsule_lengths = CapsuleLength(
        name="capsule_lengths"
    )(capsules)

    # --------------------------------------------------------
    # Dropout
    # --------------------------------------------------------

    dropout = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="classification_dropout"
    )(capsule_lengths)

    # --------------------------------------------------------
    # Softmax classification
    # --------------------------------------------------------

    outputs = tf.keras.layers.Softmax(
        name="class_probabilities"
    )(dropout)

    # --------------------------------------------------------
    # Construct model
    # --------------------------------------------------------

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetB7_Capsule_DSOA"
    )

    # --------------------------------------------------------
    # Adam optimizer
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# MODEL SUMMARY
# ============================================================

def print_model_summary(
    learning_rate=0.0001,
    capsule_dimension=CAPSULE_VECTOR_LENGTH,
    routing_iterations=ROUTING_ITERATIONS,
):
    """
    Build and display the hybrid model architecture.
    """

    model = build_hybrid_model(
        learning_rate=learning_rate,
        capsule_dimension=capsule_dimension,
        routing_iterations=routing_iterations,
    )

    print(
        "\nHybrid EfficientNet-B7 + Capsule Network"
    )

    print("=" * 70)

    model.summary()

    return model


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    filepath
):
    """
    Save the trained Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        Model to save.

    filepath : str
        Output model path.
    """

    model.save(
        filepath
    )

    print(
        f"Model saved to: {filepath}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_hybrid_model(
    filepath
):
    """
    Load a previously saved hybrid model.

    Parameters
    ----------
    filepath : str
        Path to saved model.

    Returns
    -------
    tf.keras.Model
        Loaded model.
    """

    model = tf.keras.models.load_model(
        filepath,
        custom_objects={
            "CapsuleLayer": CapsuleLayer,
            "CapsuleLength": CapsuleLength,
        }
    )

    return model


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Building hybrid model..."
    )

    model = build_hybrid_model(
        learning_rate=0.0001,
        capsule_dimension=CAPSULE_VECTOR_LENGTH,
        routing_iterations=ROUTING_ITERATIONS,
        trainable_backbone=True,
    )

    print(
        "\nModel created successfully."
    )

    print(
        f"Input shape  : {model.input_shape}"
    )

    print(
        f"Output shape : {model.output_shape}"
    )

    print(
        "\nModel architecture:"
    )

    model.summary()
