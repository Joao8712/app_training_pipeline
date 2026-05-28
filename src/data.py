from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import tensorflow as tf

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def normalize_split_value(value: str) -> str:
    return str(value).strip().lower()


def load_manifest(manifest_path: str | Path, image_root: str | Path, traits: list[str] | None = None) -> pd.DataFrame:
    traits = traits or TRAITS
    df = pd.read_csv(manifest_path)

    required = ["image_path", "video_name", "final_split", *traits]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    df = df.copy()
    df["final_split_norm"] = df["final_split"].map(normalize_split_value)

    image_root = Path(image_root)
    df["resolved_image_path"] = df["image_path"].map(lambda p: str(image_root / str(p)))

    if "extraction_status" in df.columns:
        df = df[df["extraction_status"].astype(str).str.lower().eq("ok")].copy()

    for trait in traits:
        df[trait] = pd.to_numeric(df[trait], errors="raise")

    return df.reset_index(drop=True)


def split_manifest(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_name = normalize_split_value(split_name)
    out = df[df["final_split_norm"].eq(split_name)].copy()
    if out.empty:
        available = sorted(df["final_split_norm"].dropna().unique().tolist())
        raise ValueError(f"No rows found for split '{split_name}'. Available splits: {available}")
    return out.reset_index(drop=True)


def make_augmentation_layer(config: dict) -> tf.keras.Sequential:
    layers = []
    if config.get("random_flip"):
        layers.append(tf.keras.layers.RandomFlip(config["random_flip"]))
    if float(config.get("random_rotation", 0.0)) > 0:
        layers.append(tf.keras.layers.RandomRotation(float(config["random_rotation"])))
    if float(config.get("random_zoom", 0.0)) > 0:
        layers.append(tf.keras.layers.RandomZoom(float(config["random_zoom"])))
    if float(config.get("random_contrast", 0.0)) > 0:
        layers.append(tf.keras.layers.RandomContrast(float(config["random_contrast"])))
    return tf.keras.Sequential(layers, name="augmentation")


def decode_and_resize(path: tf.Tensor, image_size: int) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


def build_dataset(
    df: pd.DataFrame,
    traits: list[str],
    image_size: int,
    batch_size: int,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
    shuffle: bool,
    augment: bool,
    augmentation_config: dict | None = None,
    shuffle_buffer: int = 4096,
    cache: bool = False,
) -> tf.data.Dataset:
    paths = df["resolved_image_path"].astype(str).to_numpy()
    labels = df[traits].astype("float32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(df)), reshuffle_each_iteration=True)

    aug_layer = make_augmentation_layer(augmentation_config or {}) if augment else None

    def _load(path, label):
        image = decode_and_resize(path, image_size)
        if aug_layer is not None:
            image = aug_layer(image, training=True)
        image = preprocess_fn(image)
        return image, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_prediction_dataset(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
) -> tf.data.Dataset:
    paths = df["resolved_image_path"].astype(str).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        image = decode_and_resize(path, image_size)
        image = preprocess_fn(image)
        return image

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
