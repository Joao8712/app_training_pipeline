# Chapter 3 Pipeline Scripts

This folder contains the dataset construction pipeline used to prepare the image-based APP dataset described in Chapter 3.

The scripts are intended to:

- convert official ChaLearn segment annotations into a single unified label manifest,
- optionally construct a leakage-aware split at the source-video level,
- extract deterministic static frames from video segments,
- optionally extract face-crop frames,
- and summarize the final prepared dataset.

## Pipeline overview

The recommended execution order is:

1. `01_analyze_labels.py`
2. `02_make_source_group_split.py`
3. `03_extract_frames.py` or `03b_extract_face_crop_frames.py`
4. `04_analyze_prepared_dataset.py`

Each step preserves the official segment-level labels and writes manifest files that record provenance and extraction metadata.

## Scripts

### `01_analyze_labels.py`

Purpose:
- Read the official ChaLearn annotation CSVs for training, validation, and test.
- Combine them into a unified `labels_manifest.csv`.
- Add source-video identifiers and original partition labels.
- Produce label diagnostics and overlap reports.

Required inputs:
- `--train_csv`: official training annotation CSV.
- `--val_csv`: official validation annotation CSV.
- `--test_csv`: official test annotation CSV.
- `--out_dir`: output directory for manifest and diagnostics.

Outputs:
- `labels_manifest.csv`
- `source_counts.csv`
- `partition_counts.csv`
- `source_split_overlap.csv`
- `source_overlap_summary.csv`
- `label_summary_by_original_split.csv`
- `label_summary_overall.csv`
- `label_correlations_pearson.csv`
- `label_correlations_spearman.csv`
- Optional label histograms under `label_histograms/` when `--make_plots` is used.

Example:
```bash
python chapter3_pipeline_scripts/01_analyze_labels.py \
  --train_csv data/annotation_training.csv \
  --val_csv data/annotation_validation.csv \
  --test_csv data/annotation_test.csv \
  --out_dir chapter3_pipeline_scripts/outputs/labels
```

### `02_make_source_group_split.py`

Purpose:
- Create a leakage-aware final split by grouping segments with the same `source_video_id`.
- Preserve official segment labels while enforcing source-video exclusivity across splits.
- Report split sizes, label balance, and leakage checks.

Required inputs:
- `--labels_manifest`: `labels_manifest.csv` from `01_analyze_labels.py`.
- `--out_dir`: output directory for split manifest and summaries.

Optional inputs:
- `--train_ratio`, `--val_ratio`, `--test_ratio`: target ratios for final splits.
- `--n_seeds`: number of random seeds to try for balanced grouping.

Outputs:
- `final_split_manifest.csv`
- `final_split_counts.csv`
- `final_split_label_summary.csv`
- `final_split_source_overlap_check.csv`

Example:
```bash
python chapter3_pipeline_scripts/02_make_source_group_split.py \
  --labels_manifest chapter3_pipeline_scripts/outputs/labels/labels_manifest.csv \
  --out_dir chapter3_pipeline_scripts/outputs/splits \
  --train_ratio 0.6 \
  --val_ratio 0.2 \
  --test_ratio 0.2
```

### `03_extract_frames.py`

Purpose:
- Extract deterministic image frames from video segments.
- Associate each extracted image with the official segment label.
- Write an image manifest with extraction metadata.

Modes:
- `--video_root`: recursively search a directory for `.mp4` files.
- `--archives_root`: recursively extract zip archives and process contained videos.

Required inputs:
- `--label_manifest`: label manifest with `video_name` and segment trait labels.
- `--out_dir`: output folder for images and manifest.

Key options:
- `--n_frames_per_segment`: number of frames to extract per video segment (default `5`).
- `--sampling_strategy`: `uniform` or `center`.
- `--image_size`: output image size, e.g. `224`.
- `--image_ext`: `jpg` or `png`.
- `--jpeg_quality`: JPEG quality if using `jpg`.
- `--zip_password`: optional password for encrypted archives.
- `--max_zip_depth`: nested zip extraction depth.
- `--cleanup_extracted`: remove temporary extracted archive contents.

Outputs:
- `images/` directory with extracted image files.
- `image_manifest.csv`
- `image_extraction_summary.csv`

Example (raw video directory):
```bash
python chapter3_pipeline_scripts/03_extract_frames.py \
  --video_root data/videos \
  --label_manifest chapter3_pipeline_scripts/outputs/splits/final_split_manifest.csv \
  --out_dir chapter3_pipeline_scripts/outputs/images \
  --n_frames_per_segment 5 \
  --sampling_strategy uniform \
  --image_size 224 \
  --image_ext jpg
```

Example (archive-based video source):
```bash
python chapter3_pipeline_scripts/03_extract_frames.py \
  --archives_root data/archives \
  --label_manifest chapter3_pipeline_scripts/outputs/splits/final_split_manifest.csv \
  --out_dir chapter3_pipeline_scripts/outputs/images \
  --cleanup_extracted
```

### `03b_extract_face_crop_frames.py`

Purpose:
- Extract deterministic face-crop frames as an alternative dataset variant.
- Use the same segment-level labels and frame sampling strategy.
- Detect the largest face, crop around it, then resize to the requested image size.
- Fall back to full-frame output if face detection fails (optional).

Required inputs:
- `--label_manifest`: label manifest with `video_name` and labels.
- `--out_dir`: output folder for face-crop images and manifest.

Key options:
- `--detector`: `opencv_haar` (default) or `mediapipe`.
- `--crop_margin`: margin around the detected face box.
- `--square_crop`: whether to enforce square cropping.
- `--fallback_full_frame`: keep full frame if no face is detected, instead of failing.

Outputs:
- `images/` directory with cropped image files.
- `image_manifest.csv`
- `image_extraction_summary.csv`
- `face_crop_summary.csv`

Example:
```bash
python chapter3_pipeline_scripts/03b_extract_face_crop_frames.py \
  --video_root data/videos \
  --label_manifest chapter3_pipeline_scripts/outputs/splits/final_split_manifest.csv \
  --out_dir chapter3_pipeline_scripts/outputs/face_crops \
  --detector opencv_haar \
  --image_size 224 \
  --fallback_full_frame
```

### `04_analyze_prepared_dataset.py`

Purpose:
- Summarize the extracted image dataset after frame or face-crop extraction.
- Compute split-level counts, label summaries, and images-per-segment statistics.

Required inputs:
- `--image_manifest`: `image_manifest.csv` produced by `03_extract_frames.py` or `03b_extract_face_crop_frames.py`.
- `--out_dir`: output directory for summaries.

Outputs:
- `prepared_dataset_summary_by_split.csv`
- `label_summary_by_final_split.csv`
- `images_per_segment.csv`
- `images_per_segment_summary.csv`

Example:
```bash
python chapter3_pipeline_scripts/04_analyze_prepared_dataset.py \
  --image_manifest chapter3_pipeline_scripts/outputs/images/image_manifest.csv \
  --out_dir chapter3_pipeline_scripts/outputs/image_summary
```

## Input and manifest conventions

- All scripts expect the label manifest to include `video_name` and the five trait columns:
  `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`.
- The extraction scripts preserve segment-level labels for every sampled image.
- `video_name` should match the video filename used in the official ChaLearn annotations.
- Extracted image paths in `image_manifest.csv` are relative to the script `--out_dir`.
- `final_split` is used by `03_extract_frames.py` when a leakage-aware split is desired.

## Recommended pipeline for Chapter 3 dataset preparation

1. Run `01_analyze_labels.py` to build the unified label manifest and inspect label distributions.
2. Run `02_make_source_group_split.py` to create a source-video leakage-aware split.
3. Run `03_extract_frames.py` to create the full-frame image dataset.
4. Optionally run `03b_extract_face_crop_frames.py` to build a face-crop variant.
5. Run `04_analyze_prepared_dataset.py` to summarize the extracted dataset and verify split statistics.

## Dependencies

The scripts require Python packages including:
- `numpy`
- `pandas`
- `opencv-python`
- `Pillow`
- `tqdm`
- `matplotlib` (for optional plotting in `01_analyze_labels.py`)
- `mediapipe` (optional for `03b_extract_face_crop_frames.py`)

## Notes

- The pipeline does not modify official ChaLearn labels; it only aggregates and reuses them.
- `03_extract_frames.py` and `03b_extract_face_crop_frames.py` use deterministic frame sampling to ensure repeatability.
- The leakage-aware split is optional and should be documented if used.
- If face-crop extraction is used, the resulting dataset should be treated as a variant distinct from the full-frame dataset.
