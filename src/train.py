from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from data import build_dataset, load_manifest, split_manifest
from models import build_regression_model, compile_model, get_preprocess_fn, set_backbone_trainable
from utils import configure_gpu, ensure_dir, load_config, save_config, set_reproducibility, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train APP regression model from image manifest.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None, help="Override manifest path in config.")
    parser.add_argument("--image-root", type=Path, default=None, help="Override image root path in config.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--deterministic-ops", action="store_true")
    return parser.parse_args()


def make_callbacks(run_dir: Path, monitor: str, mode: str, patience: int, reduce_lr_patience: int):
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(logs_dir / "history.csv"), append=True),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.manifest is not None:
        cfg["data"]["manifest"] = str(args.manifest)
    if args.image_root is not None:
        cfg["data"]["image_root"] = str(args.image_root)

    run_dir = ensure_dir(args.run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "artifacts")

    save_config(cfg, run_dir / "config_resolved.yaml")

    set_reproducibility(int(cfg.get("seed", 42)), deterministic_ops=args.deterministic_ops)
    configure_gpu(bool(cfg["training"].get("mixed_precision", True)))

    traits = cfg.get("traits", ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"])
    df = load_manifest(cfg["data"]["manifest"], cfg["data"]["image_root"], traits=traits)

    train_df = split_manifest(df, cfg["data"].get("train_split_name", "training"))
    val_df = split_manifest(df, cfg["data"].get("val_split_name", "validation"))

    train_df.to_csv(run_dir / "artifacts" / "train_rows.csv", index=False)
    val_df.to_csv(run_dir / "artifacts" / "validation_rows.csv", index=False)

    preprocess_fn = get_preprocess_fn(cfg["model"]["architecture"])

    train_ds = build_dataset(
        train_df,
        traits=traits,
        image_size=int(cfg["data"]["image_size"]),
        batch_size=int(cfg["data"]["batch_size"]),
        preprocess_fn=preprocess_fn,
        shuffle=True,
        augment=bool(cfg["augmentation"].get("enabled", True)),
        augmentation_config=cfg.get("augmentation", {}),
        shuffle_buffer=int(cfg["data"].get("shuffle_buffer", 4096)),
        cache=bool(cfg["data"].get("cache", False)),
    )

    val_ds = build_dataset(
        val_df,
        traits=traits,
        image_size=int(cfg["data"]["image_size"]),
        batch_size=int(cfg["data"]["batch_size"]),
        preprocess_fn=preprocess_fn,
        shuffle=False,
        augment=False,
        augmentation_config=None,
        cache=False,
    )

    model, backbone = build_regression_model(
        architecture=cfg["model"]["architecture"],
        image_size=int(cfg["data"]["image_size"]),
        n_outputs=len(traits),
        weights=cfg["model"].get("weights", "imagenet"),
        dropout=float(cfg["model"].get("dropout", 0.30)),
        dense_units=int(cfg["model"].get("dense_units", 256)),
        output_activation=cfg["model"].get("output_activation", "sigmoid"),
    )

    with open(run_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

    training_cfg = cfg["training"]
    monitor = training_cfg.get("monitor", "val_mae")
    mode = training_cfg.get("monitor_mode", "min")

    callbacks = make_callbacks(
        run_dir,
        monitor=monitor,
        mode=mode,
        patience=int(training_cfg.get("early_stopping_patience", 4)),
        reduce_lr_patience=int(training_cfg.get("reduce_lr_patience", 2)),
    )

    compile_model(
        model,
        learning_rate=float(training_cfg.get("learning_rate_frozen", 1e-3)),
        loss_name=training_cfg.get("loss", "mae"),
    )

    frozen_epochs = int(training_cfg.get("frozen_epochs", 0))
    if frozen_epochs > 0:
        history1 = model.fit(train_ds, validation_data=val_ds, epochs=frozen_epochs, callbacks=callbacks)
        pd.DataFrame(history1.history).to_csv(run_dir / "logs" / "history_frozen.csv", index=False)

    fine_tune_epochs = int(training_cfg.get("fine_tune_epochs", 0))
    if fine_tune_epochs > 0:
        set_backbone_trainable(backbone, int(training_cfg.get("fine_tune_last_n_layers", 30)))
        compile_model(
            model,
            learning_rate=float(training_cfg.get("learning_rate_finetune", 1e-5)),
            loss_name=training_cfg.get("loss", "mae"),
        )

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=frozen_epochs,
            epochs=frozen_epochs + fine_tune_epochs,
            callbacks=callbacks,
        )
        pd.DataFrame(history2.history).to_csv(run_dir / "logs" / "history_finetune.csv", index=False)

    model.save(run_dir / "checkpoints" / "final.keras")

    write_json(run_dir / "run_summary.json", {
        "architecture": cfg["model"]["architecture"],
        "n_train_images": int(len(train_df)),
        "n_validation_images": int(len(val_df)),
        "traits": traits,
        "best_checkpoint": str(run_dir / "checkpoints" / "best.keras"),
        "final_checkpoint": str(run_dir / "checkpoints" / "final.keras"),
    })

    print(f"Training complete. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
