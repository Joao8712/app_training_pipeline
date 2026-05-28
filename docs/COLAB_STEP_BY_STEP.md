# Google Colab step-by-step

## 1. Prepare Google Drive

Create this directory in Google Drive:

```text
MyDrive/thesis_app/
  image_output/
    image_manifest.csv
    images/
  app_runs/
```

Upload the complete `image_output/` directory generated in Chapter 3. The manifest and the `images/` folder must stay together.

For faster training, copy the data from Drive to the Colab local disk at the beginning of a session:

```bash
mkdir -p /content/data
rsync -a --info=progress2 "/content/drive/MyDrive/thesis_app/image_output/" /content/data/image_output/
```

## 2. Open Colab and select GPU

Runtime -> Change runtime type -> GPU.

Then run:

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

## 3. Mount Drive and clone repository

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content
git clone <GITHUB-REPOSITORY-URL> thesis-app-training
cd thesis-app-training
bash scripts/colab_setup.sh
```

## 4. Copy data to local runtime

```bash
mkdir -p /content/data
rsync -a --info=progress2 "/content/drive/MyDrive/thesis_app/image_output/" /content/data/image_output/
```

Use these paths:

```bash
export MANIFEST=/content/data/image_output/image_manifest.csv
export IMAGE_ROOT=/content/data/image_output
export RUN_ROOT=/content/drive/MyDrive/thesis_app/app_runs
```

## 5. Run a smoke test

```bash
bash scripts/run_smoke_test.sh "$MANIFEST" "$IMAGE_ROOT" "$RUN_ROOT"
```

Only continue if this completes and writes validation metrics.

## 6. Run the planned trainings

Run one model at a time if the runtime is unstable:

```bash
python src/train.py --config configs/resnet50.yaml \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --run-dir "$RUN_ROOT/resnet50"
```

Then evaluate:

```bash
python src/evaluate.py --config configs/resnet50.yaml \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --checkpoint "$RUN_ROOT/resnet50/checkpoints/best.keras" \
  --out-dir "$RUN_ROOT/resnet50/eval_validation" \
  --split validation

python src/evaluate.py --config configs/resnet50.yaml \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --checkpoint "$RUN_ROOT/resnet50/checkpoints/best.keras" \
  --out-dir "$RUN_ROOT/resnet50/eval_test" \
  --split test
```

Repeat for `efficientnetb0.yaml` and `convnext_tiny.yaml`.

To run all configured models sequentially:

```bash
bash scripts/run_planned_trainings.sh "$MANIFEST" "$IMAGE_ROOT" "$RUN_ROOT"
```

## 7. Compare runs

```bash
python src/compare_runs.py \
  --eval-root "$RUN_ROOT" \
  --split validation \
  --out "$RUN_ROOT/comparison_validation.csv"

python src/compare_runs.py \
  --eval-root "$RUN_ROOT" \
  --split test \
  --out "$RUN_ROOT/comparison_test.csv"
```

## 8. Preserve outputs

The run outputs are written to Google Drive:

```text
MyDrive/thesis_app/app_runs/
  resnet50/
  efficientnetb0/
  convnext_tiny/
  comparison_validation.csv
  comparison_test.csv
```

Each run contains:

```text
checkpoints/best.keras
checkpoints/final.keras
logs/history.csv
logs/history_frozen.csv
logs/history_finetune.csv
eval_validation/
eval_test/
config_resolved.yaml
run_summary.json
```
