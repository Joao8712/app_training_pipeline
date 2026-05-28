from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import build_prediction_dataset, load_manifest, split_manifest
from metrics import aggregate_to_segment_level, regression_metrics
from models import get_preprocess_fn
from utils import ensure_dir, load_config, save_config, set_reproducibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate APP regression model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["training", "validation", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.manifest is not None:
        cfg["data"]["manifest"] = str(args.manifest)
    if args.image_root is not None:
        cfg["data"]["image_root"] = str(args.image_root)

    out_dir = ensure_dir(args.out_dir)
    save_config(cfg, out_dir / "config_resolved.yaml")
    set_reproducibility(int(cfg.get("seed", 42)))

    traits = cfg.get("traits", ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"])
    df = load_manifest(cfg["data"]["manifest"], cfg["data"]["image_root"], traits=traits)
    split_df = split_manifest(df, args.split)

    preprocess_fn = get_preprocess_fn(cfg["model"]["architecture"])
    pred_ds = build_prediction_dataset(
        split_df,
        image_size=int(cfg["data"]["image_size"]),
        batch_size=int(cfg["data"]["batch_size"]),
        preprocess_fn=preprocess_fn,
    )

    model = tf.keras.models.load_model(args.checkpoint)
    preds = model.predict(pred_ds, verbose=1)
    preds = np.asarray(preds, dtype=float)

    image_pred_df = split_df.copy()
    for i, trait in enumerate(traits):
        image_pred_df[f"pred_{trait}"] = preds[:, i]

    image_pred_df.to_csv(out_dir / f"predictions_image_level_{args.split}.csv", index=False)

    y_true_img = image_pred_df[traits].to_numpy(dtype=float)
    y_pred_img = image_pred_df[[f"pred_{t}" for t in traits]].to_numpy(dtype=float)
    image_by_trait, image_overall = regression_metrics(y_true_img, y_pred_img, traits)
    image_by_trait.to_csv(out_dir / f"metrics_image_level_by_trait_{args.split}.csv", index=False)
    image_overall.to_csv(out_dir / f"metrics_image_level_overall_{args.split}.csv", index=False)

    segment_df = aggregate_to_segment_level(image_pred_df, traits=traits)
    segment_df.to_csv(out_dir / f"predictions_segment_level_{args.split}.csv", index=False)

    y_true_seg = segment_df[traits].to_numpy(dtype=float)
    y_pred_seg = segment_df[[f"pred_{t}" for t in traits]].to_numpy(dtype=float)
    seg_by_trait, seg_overall = regression_metrics(y_true_seg, y_pred_seg, traits)
    seg_by_trait.to_csv(out_dir / f"metrics_segment_level_by_trait_{args.split}.csv", index=False)
    seg_overall.to_csv(out_dir / f"metrics_segment_level_overall_{args.split}.csv", index=False)

    print("Image-level overall metrics:")
    print(image_overall.to_string(index=False))
    print("\nSegment-level overall metrics:")
    print(seg_overall.to_string(index=False))
    print(f"\nWrote evaluation artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
