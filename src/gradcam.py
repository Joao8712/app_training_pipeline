from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from data import load_manifest, split_manifest
from models import get_preprocess_fn


TRAITS_DEFAULT = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_backbone(model: tf.keras.Model, architecture: str) -> tf.keras.Model:
    arch = architecture.lower()
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and arch in layer.name.lower():
            return layer

    # Fallback: first nested model with many layers
    nested = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if not nested:
        raise ValueError("Could not find nested backbone model.")
    return nested[0]


def find_last_4d_layer(backbone: tf.keras.Model) -> tf.keras.layers.Layer:
    candidates = []
    for layer in backbone.layers:
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                candidates.append(layer)
        except Exception:
            pass

    if not candidates:
        raise ValueError("Could not find a 4D convolutional feature layer in the backbone.")

    return candidates[-1]


def apply_head_layers(
    full_model: tf.keras.Model,
    backbone: tf.keras.Model,
    backbone_output: tf.Tensor,
) -> tf.Tensor:
    """Apply the top-level layers after the backbone."""
    x = backbone_output
    found = False

    for layer in full_model.layers:
        if layer is backbone:
            found = True
            continue
        if not found:
            continue

        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)

    return x


def read_image_for_model(path: Path, image_size: int, preprocess_fn):
    raw = tf.io.read_file(str(path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32)
    model_img = preprocess_fn(img)
    return tf.expand_dims(model_img, axis=0)


def read_image_for_overlay(path: Path, image_size: int) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img_bgr


def make_gradcam_heatmap(
    full_model: tf.keras.Model,
    backbone: tf.keras.Model,
    target_layer: tf.keras.layers.Layer,
    image_tensor: tf.Tensor,
    trait_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, backbone_output = feature_model(image_tensor, training=False)
        tape.watch(conv_outputs)

        preds = apply_head_layers(full_model, backbone, backbone_output)
        target = preds[:, trait_index]

    grads = tape.gradient(target, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + 1e-8)

    return heatmap.numpy(), preds.numpy()[0]


def overlay_heatmap(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_uint8 = cv2.resize(heatmap_uint8, (w, h))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay


def select_examples(df: pd.DataFrame, traits: list[str], n_low: int, n_high: int) -> pd.DataFrame:
    pred_cols = [f"pred_{t}" for t in traits]

    if all(c in df.columns for c in pred_cols):
        y_true = df[traits].to_numpy(dtype=float)
        y_pred = df[pred_cols].to_numpy(dtype=float)
        df = df.copy()
        df["mean_abs_error"] = np.mean(np.abs(y_true - y_pred), axis=1)

        low = df.sort_values("mean_abs_error", ascending=True).head(n_low)
        high = df.sort_values("mean_abs_error", ascending=False).head(n_high)
        out = pd.concat([low.assign(example_group="low_error"),
                         high.assign(example_group="high_error")])
        return out.reset_index(drop=True)

    # Fallback: deterministic sample if prediction columns are absent
    out = df.sample(n=min(n_low + n_high, len(df)), random_state=42).copy()
    out["example_group"] = "sampled"
    out["mean_abs_error"] = np.nan
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for APP models.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["training", "validation", "test"])
    parser.add_argument("--prediction-csv", type=Path, default=None)
    parser.add_argument("--n-low-error", type=int, default=3)
    parser.add_argument("--n-high-error", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.40)
    parser.add_argument("--target-layer", type=str, default=None)
    parser.add_argument("--traits", nargs="+", default=TRAITS_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    architecture = cfg["model"]["architecture"]
    image_size = int(cfg["data"]["image_size"])
    traits = args.traits

    preprocess_fn = get_preprocess_fn(architecture)

    manifest_df = load_manifest(args.manifest, args.image_root, traits=traits)
    split_df = split_manifest(manifest_df, args.split)

    if args.prediction_csv is not None and args.prediction_csv.exists():
        pred_df = pd.read_csv(args.prediction_csv)
        pred_df["resolved_image_path"] = pred_df["image_path"].map(
            lambda p: str(args.image_root / str(p))
        )
        work_df = pred_df
    else:
        work_df = split_df

    examples = select_examples(
        work_df,
        traits=traits,
        n_low=args.n_low_error,
        n_high=args.n_high_error,
    )

    model = tf.keras.models.load_model(args.checkpoint)
    backbone = find_backbone(model, architecture)

    if args.target_layer:
        target_layer = backbone.get_layer(args.target_layer)
    else:
        target_layer = find_last_4d_layer(backbone)

    print(f"Architecture: {architecture}")
    print(f"Backbone: {backbone.name}")
    print(f"Target Grad-CAM layer: {target_layer.name}")
    print(f"Number of examples: {len(examples)}")

    records = []

    for row_idx, row in examples.iterrows():
        image_path = Path(row["resolved_image_path"])
        image_tensor = read_image_for_model(image_path, image_size, preprocess_fn)
        original_bgr = read_image_for_overlay(image_path, image_size)

        for trait_index, trait in enumerate(traits):
            heatmap, preds = make_gradcam_heatmap(
                full_model=model,
                backbone=backbone,
                target_layer=target_layer,
                image_tensor=image_tensor,
                trait_index=trait_index,
            )

            overlay = overlay_heatmap(original_bgr, heatmap, alpha=args.alpha)

            stem = Path(str(row["image_path"])).stem
            group = row.get("example_group", "example")
            out_name = f"{row_idx:03d}_{group}_{stem}_{trait}_gradcam.jpg"
            out_path = args.out_dir / out_name

            cv2.imwrite(str(out_path), overlay)

            records.append({
                "example_index": row_idx,
                "example_group": group,
                "image_path": row["image_path"],
                "output_file": out_name,
                "trait": trait,
                "true_value": row.get(trait, np.nan),
                "pred_value": float(preds[trait_index]),
                "mean_abs_error": row.get("mean_abs_error", np.nan),
                "architecture": architecture,
                "checkpoint": str(args.checkpoint),
                "target_layer": target_layer.name,
            })

    index_df = pd.DataFrame(records)
    index_df.to_csv(args.out_dir / "gradcam_index.csv", index=False)

    print(f"Wrote Grad-CAM outputs to: {args.out_dir}")
    print(f"Wrote index to: {args.out_dir / 'gradcam_index.csv'}")


if __name__ == "__main__":
    main()