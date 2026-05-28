#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:?Usage: scripts/run_planned_trainings.sh <manifest_csv> <image_root> <run_root>}
IMAGE_ROOT=${2:?Usage: scripts/run_planned_trainings.sh <manifest_csv> <image_root> <run_root>}
RUN_ROOT=${3:?Usage: scripts/run_planned_trainings.sh <manifest_csv> <image_root> <run_root>}

for CFG in resnet50 efficientnetb0 convnext_tiny; do
  echo "Training $CFG"
  python src/train.py \
    --config "configs/${CFG}.yaml" \
    --manifest "$MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --run-dir "$RUN_ROOT/${CFG}"

  echo "Evaluating $CFG on validation"
  python src/evaluate.py \
    --config "configs/${CFG}.yaml" \
    --manifest "$MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --checkpoint "$RUN_ROOT/${CFG}/checkpoints/best.keras" \
    --out-dir "$RUN_ROOT/${CFG}/eval_validation" \
    --split validation

  echo "Evaluating $CFG on test"
  python src/evaluate.py \
    --config "configs/${CFG}.yaml" \
    --manifest "$MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --checkpoint "$RUN_ROOT/${CFG}/checkpoints/best.keras" \
    --out-dir "$RUN_ROOT/${CFG}/eval_test" \
    --split test
done

python src/compare_runs.py \
  --eval-root "$RUN_ROOT" \
  --split validation \
  --out "$RUN_ROOT/comparison_validation.csv"

python src/compare_runs.py \
  --eval-root "$RUN_ROOT" \
  --split test \
  --out "$RUN_ROOT/comparison_test.csv"
