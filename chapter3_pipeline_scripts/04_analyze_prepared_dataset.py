#!/usr/bin/env python3
"""
Summarize extracted image dataset.

Input:
  image_manifest.csv from 03_extract_frames.py

Outputs:
  prepared_dataset_summary_by_split.csv
  label_summary_by_final_split.csv
  images_per_segment_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_manifest", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.image_manifest)
    split_col = "final_split" if "final_split" in df.columns else "original_partition"
    ok = df[df["extraction_status"] == "ok"].copy()

    summary = (
        df.groupby(split_col)
        .agg(
            sources=("source_video_id", "nunique"),
            segments=("video_name", "nunique"),
            frames_sampled=("frame_index", "count"),
            valid_images=("extraction_status", lambda x: int((x == "ok").sum())),
            failed=("extraction_status", lambda x: int((x != "ok").sum())),
        )
        .reset_index()
        .rename(columns={split_col: "split"})
    )
    summary.to_csv(args.out_dir / "prepared_dataset_summary_by_split.csv", index=False)

    label_rows = []
    for split, g in ok.groupby(split_col):
        # label stats should be computed at segment level to avoid overweighting segments
        # that have more images; use one row per video_name.
        sg = g.drop_duplicates("video_name")
        for trait in TRAITS:
            s = sg[trait]
            label_rows.append({
                "split": split, "trait": trait, "n_segments": int(s.notna().sum()),
                "mean": float(s.mean()), "std": float(s.std(ddof=1)),
                "min": float(s.min()), "median": float(s.median()), "max": float(s.max())
            })
    pd.DataFrame(label_rows).to_csv(args.out_dir / "label_summary_by_final_split.csv", index=False)

    ips = ok.groupby("video_name").size().reset_index(name="n_images")
    ips["source_video_id"] = ok.drop_duplicates("video_name").set_index("video_name").loc[ips["video_name"], "source_video_id"].values
    ips.to_csv(args.out_dir / "images_per_segment.csv", index=False)
    ips["n_images"].describe().to_frame("n_images").to_csv(args.out_dir / "images_per_segment_summary.csv")

    print(f"Wrote summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
