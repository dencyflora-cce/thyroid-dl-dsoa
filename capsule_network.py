"""
Capsule Network module for thyroid ultrasound classification.

The module implements:
    - Squash activation
    - Capsule transformation
    - Dynamic routing between capsules

The routing iteration count is configurable through config.py.
"""

import tensorflow as tf

from config import (
    CAPSULE_VECTOR_LENGTH,
    ROUTING_ITERATIONS,
)


# ============================================================
# SQUASH ACTIVATION
# ============================================================

def squash(vectors, axis=-1):
    """
    Apply the squash non-linearity used in capsule networks.

    Parameters
    ----------
    vectors : tf.Tensor
        Input capsule vectors.

    axis : int
        Axis corresponding to the capsule vector dimension.

    Returns
    -------
    tf.Tensor
        Squashed capsule vectors.
    """

    squared_norm = tf.reduce_sum(
        tf.square(vectors),
        axis=axis,
        keepdims=True
    )

    scale = (
        squared_norm
        / (1.0 + squared_norm)
        / tf.sqrt(
            squared_norm + tf.keras.backend.epsilon()
        )
    )

    return scale * vectors


# ============================================================
# CAPSULE LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="ThyroidDLDSOA"
)
class CapsuleLayer(tf.keras.layers.Layer):
    """
    Capsule layer implementing dynamic routing.

    Parameters
    ----------
    num_capsules : int
        Number of output capsules.

    dim_capsule : int
        Dimension of each output capsule.

    routing_iterations : int
        Number of dynamic-routing iterations.
    """

    def __init__(
        self,
        num_capsules,
        dim_capsule=CAPSULE_VECTOR_LENGTH,
        routing_iterations=ROUTING_ITERATIONS,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routing_iterations = routing_iterations

    # --------------------------------------------------------
    # Build transformation weights
    # --------------------------------------------------------

    def build(self, input_shape):
        """
        Create trainable transformation matrices.
        """

        if len(input_shape) != 3:
            raise ValueError(
                "CapsuleLayer expects input shape "
                "(batch, input_capsules, input_dim)."
            )

        self.input_capsules = int(
            input_shape[1]
        )

        self.input_dim = int(
            input_shape[2]
        )

        self.W = self.add_weight(
            name="transformation_matrix",
            shape=(
                1,
                self.input_capsules,
                self.num_capsules,
                self.dim_capsule,
                self.input_dim
            ),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def call(self, inputs):
        """
        Perform capsule transformation and dynamic routing.
        """

        # ----------------------------------------------------
        # Expand input so that every input capsule can be
        # transformed toward every output capsule.
        # ----------------------------------------------------

        expanded_inputs = tf.expand_dims(
            inputs,
            axis=2
        )

        # Shape:
        # (batch, input_capsules, 1, input_dim)

        expanded_inputs = tf.expand_dims(
            expanded_inputs,
            axis=-1
        )

        # Shape:
        # (batch, input_capsules, 1, input_dim, 1)

        # ----------------------------------------------------
        # Transform input capsules.
        # ----------------------------------------------------

        u_hat = tf.matmul(
            self.W,
            expanded_inputs
        )

        # Shape:
        # (batch, input_capsules,
        #  num_capsules, dim_capsule, 1)

        u_hat = tf.squeeze(
            u_hat,
            axis=-1
        )

        # ----------------------------------------------------
        # Dynamic routing coefficients.
        # ----------------------------------------------------

        batch_size = tf.shape(inputs)[0]

        routing_logits = tf.zeros(
            (
                batch_size,
                self.input_capsules,
                self.num_capsules
            ),
            dtype=inputs.dtype
        )

        # ----------------------------------------------------
        # Dynamic routing iterations.
        # ----------------------------------------------------

        for iteration in range(
            self.routing_iterations
        ):

            coupling_coefficients = tf.nn.softmax(
                routing_logits,
                axis=-1
            )

            weighted_predictions = (
                coupling_coefficients[..., tf.newaxis]
                * u_hat
            )

            output_capsules = tf.reduce_sum(
                weighted_predictions,
                axis=1
            )

            output_capsules = squash(
                output_capsules
            )

            if iteration < (
                self.routing_iterations - 1
            ):

                agreement = tf.reduce_sum(
                    u_hat
                    * output_capsules[
                        :, tf.newaxis, :, :
                    ],
                    axis=-1
                )

                routing_logits += agreement

        return output_capsules

    # --------------------------------------------------------
    # Configuration for model serialization
    # --------------------------------------------------------

    def get_config(self):
        """
        Return layer configuration for saving/loading models.
        """

        config = super().get_config()

        config.update(
            {
                "num_capsules": self.num_capsules,
                "dim_capsule": self.dim_capsule,
                "routing_iterations": self.routing_iterations,
            }
        )

        return config


# ============================================================
# CAPSULE LENGTH LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="ThyroidDLDSOA"
)
class CapsuleLength(tf.keras.layers.Layer):
    """
    Calculate the length of each capsule vector.

    Capsule length is used as the class activation magnitude.
    """

    def call(self, inputs):
        return tf.sqrt(
            tf.reduce_sum(
                tf.square(inputs),
                axis=-1
            )
            + tf.keras.backend.epsilon()
        )


# ============================================================
# BUILD CAPSULE BLOCK
# ============================================================

def build_capsule_block(
    feature_map,
    num_capsules,
    dim_capsule=CAPSULE_VECTOR_LENGTH,
    routing_iterations=ROUTING_ITERATIONS
):
    """
    Convert convolutional feature maps into capsules.

    Parameters
    ----------
    feature_map : tf.Tensor
        Feature map produced by EfficientNet-B7.

    num_capsules : int
        Number of output capsules.

    dim_capsule : int
        Capsule vector dimension.

    routing_iterations : int
        Number of routing iterations.

    Returns
    -------
    tf.Tensor
        Output capsule representations.
    """

    # --------------------------------------------------------
    # Flatten spatial dimensions.
    # --------------------------------------------------------

    flattened = tf.keras.layers.Reshape(
        (
            -1,
            feature_map.shape[-1]
        ),
        name="flatten_feature_map"
    )(feature_map)

    # --------------------------------------------------------
    # Dynamic routing capsule layer.
    # --------------------------------------------------------

    capsules = CapsuleLayer(
        num_capsules=num_capsules,
        dim_capsule=dim_capsule,
        routing_iterations=routing_iterations,
        name="dynamic_routing_capsules"
    )(flattened)

    return capsules


# ============================================================
# TEST CAPSULE LAYER
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Capsule Network module..."
    )

    # Example feature representation:
    # batch = 2
    # input capsules = 16
    # input dimension = 32

    sample_input = tf.random.normal(
        shape=(2, 16, 32),
        seed=42
    )

    capsule_layer = CapsuleLayer(
        num_capsules=3,
        dim_capsule=CAPSULE_VECTOR_LENGTH,
        routing_iterations=ROUTING_ITERATIONS
    )

    output = capsule_layer(
        sample_input
    )

    lengths = CapsuleLength()(
        output
    )

    print(
        f"Input shape  : {sample_input.shape}"
    )

    print(
        f"Capsule shape: {output.shape}"
    )

    print(
        f"Length shape : {lengths.shape}"
    )

    print(
        "\nCapsule Network module initialized successfully."
    )
