#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:?Usage: scripts/run_smoke_test.sh <manifest_csv> <image_root> <run_root>}
IMAGE_ROOT=${2:?Usage: scripts/run_smoke_test.sh <manifest_csv> <image_root> <run_root>}
RUN_ROOT=${3:?Usage: scripts/run_smoke_test.sh <manifest_csv> <image_root> <run_root>}

python src/train.py \
  --config configs/quick_smoke_test.yaml \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --run-dir "$RUN_ROOT/quick_smoke_test"

python src/evaluate.py \
  --config configs/quick_smoke_test.yaml \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --checkpoint "$RUN_ROOT/quick_smoke_test/checkpoints/best.keras" \
  --out-dir "$RUN_ROOT/quick_smoke_test/eval_validation" \
  --split validation
