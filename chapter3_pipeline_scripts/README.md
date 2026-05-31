# Chapter 3 dataset construction scripts

These scripts implement the reproducibility pipeline described in Chapter 3.

Suggested order:

1. `01_analyze_labels.py` — create the segment-level label manifest and summarize official labels.
2. `02_make_source_group_split.py` — optionally create a leakage-aware source-video-level split.
3. `03_extract_frames.py` — extract deterministic frames from video segments and create an image manifest.
4. `04_analyze_prepared_dataset.py` — summarize the extracted image dataset.

The main thesis experiments should use the official segment-level ChaLearn labels unless a derived label set is explicitly introduced.
