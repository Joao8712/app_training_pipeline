from __future__ import annotations

from typing import Callable

import tensorflow as tf


def get_preprocess_fn(architecture: str) -> Callable:
    arch = architecture.lower()
    if arch == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input
    if arch == "efficientnetb0":
        return tf.keras.applications.efficientnet.preprocess_input
    if arch == "convnext_tiny":
        # ConvNeXt applications include preprocessing/normalization in recent TensorFlow/Keras versions.
        # Returning identity keeps inputs in the [0, 255] range expected by those implementations.
        return lambda x: x
    raise ValueError(f"Unknown architecture: {architecture}")


def get_backbone(architecture: str, input_shape: tuple[int, int, int], weights: str | None):
    arch = architecture.lower()

    if arch == "resnet50":
        return tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )

    if arch == "efficientnetb0":
        return tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )

    if arch == "convnext_tiny":
        if not hasattr(tf.keras.applications, "ConvNeXtTiny"):
            raise RuntimeError(
                "tf.keras.applications.ConvNeXtTiny is not available in this TensorFlow/Keras installation. "
                "Use a TensorFlow/Keras version that includes ConvNeXt, or replace this config with resnet50/efficientnetb0."
            )
        return tf.keras.applications.ConvNeXtTiny(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )

    raise ValueError(f"Unknown architecture: {architecture}")


def build_regression_model(
    architecture: str,
    image_size: int,
    n_outputs: int,
    weights: str | None = "imagenet",
    dropout: float = 0.30,
    dense_units: int = 256,
    output_activation: str = "sigmoid",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    input_shape = (image_size, image_size, 3)
    backbone = get_backbone(architecture, input_shape, weights)
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = backbone(inputs, training=False)
    if dense_units and dense_units > 0:
        x = tf.keras.layers.Dense(dense_units, activation="relu", name="head_dense")(x)
        x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        n_outputs,
        activation=output_activation,
        dtype="float32",
        name="trait_outputs",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{architecture}_app_regressor")
    return model, backbone


def set_backbone_trainable(backbone: tf.keras.Model, last_n_layers: int) -> None:
    """Unfreeze the last N non-BatchNorm layers of a backbone."""
    for layer in backbone.layers:
        layer.trainable = False

    if last_n_layers <= 0:
        return

    candidate_layers = [layer for layer in backbone.layers if not isinstance(layer, tf.keras.layers.BatchNormalization)]
    for layer in candidate_layers[-last_n_layers:]:
        layer.trainable = True


def make_optimizer(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def make_loss(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    if loss_name == "mse":
        return tf.keras.losses.MeanSquaredError()
    if loss_name == "huber":
        return tf.keras.losses.Huber(delta=0.05)
    raise ValueError(f"Unsupported loss: {loss_name}")


def compile_model(model: tf.keras.Model, learning_rate: float, loss_name: str) -> None:
    model.compile(
        optimizer=make_optimizer(learning_rate),
        loss=make_loss(loss_name),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
