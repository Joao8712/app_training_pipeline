# Apparent Personality Prediction Training Pipeline

This repository trains static-image models for the thesis APP experiments.

The pipeline expects the prepared image dataset generated in Chapter 3:

```text
image_output/
  image_manifest.csv
  images/
    <extracted images>
```

The code does not redistribute ChaLearn videos, extracted images, or access credentials.
It trains from the prepared manifest and local image directory.

## Recommended execution order

1. Put `image_output/` in Google Drive or on a cloud VM disk.
2. Clone this repository.
3. Install dependencies.
4. Run a short smoke test.
5. Train the planned models.
6. Evaluate the best checkpoint of each model on validation and test splits.
7. Compare segment-level metrics.

## Main commands

```bash
python src/train.py --config configs/resnet50.yaml \
  --manifest /content/data/image_output/image_manifest.csv \
  --image-root /content/data/image_output \
  --run-dir /content/drive/MyDrive/app_runs/resnet50

python src/evaluate.py --config configs/resnet50.yaml \
  --manifest /content/data/image_output/image_manifest.csv \
  --image-root /content/data/image_output \
  --checkpoint /content/drive/MyDrive/app_runs/resnet50/checkpoints/best.keras \
  --out-dir /content/drive/MyDrive/app_runs/resnet50/eval_test \
  --split test
```

## Data conventions

The manifest must contain:

- `image_path`
- `video_name`
- `final_split`
- `openness`
- `conscientiousness`
- `extraversion`
- `agreeableness`
- `neuroticism`

`image_path` is interpreted relative to `--image-root`.

## Evaluation unit

Training uses image rows. Evaluation is reported both at image level and at segment level.
The segment-level prediction is the mean of the predictions for all images sampled from the same `video_name`.
The segment-level metrics are the primary comparison values because the official labels are segment-level targets.
