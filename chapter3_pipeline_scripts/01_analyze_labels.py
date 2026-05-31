#!/usr/bin/env python3
"""
Create a segment-level label manifest from official ChaLearn annotation files
and compute split/source/label diagnostics.

Inputs:
  annotation_training.csv
  annotation_validation.csv
  annotation_test.csv

Outputs:
  labels_manifest.csv
  source_counts.csv
  source_split_overlap.csv
  source_overlap_summary.csv
  label_summary_by_original_split.csv
  label_summary_overall.csv
  label_correlations.csv
  label_histograms/*.png (optional)

This script does not recalculate labels. It uses official segment-level
continuous labels as the benchmark targets.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
ALL_LABELS = TRAITS + ["interview"]


def parse_video_name(name: str) -> tuple[str, str]:
    """Return (source_video_id, segment_id) from video_id.segment.mp4."""
    match = re.match(r"^(.+)\.(\d+)\.mp4$", str(name))
    if match is None:
        raise ValueError(f"Unexpected video filename pattern: {name}")
    return match.group(1), match.group(2)


def read_annotation(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"video_name", *ALL_LABELS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["original_partition"] = split
    parsed = df["video_name"].apply(parse_video_name)
    df["source_video_id"] = [p[0] for p in parsed]
    df["segment_id"] = [p[1] for p in parsed]
    df["label_source"] = "official_chalearn_segment"
    return df


def summarize_labels(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    rows = []
    if group_col is None:
        iterable = [("all", df)]
        split_col = "scope"
    else:
        iterable = df.groupby(group_col)
        split_col = group_col
    for group, g in iterable:
        for trait in TRAITS:
            s = g[trait]
            rows.append({
                split_col: group,
                "trait": trait,
                "n": int(s.notna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "q05": float(s.quantile(0.05)),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "q95": float(s.quantile(0.95)),
                "max": float(s.max()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, type=Path)
    parser.add_argument("--val_csv", required=True, type=Path)
    parser.add_argument("--test_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--make_plots", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        read_annotation(args.train_csv, "training"),
        read_annotation(args.val_csv, "validation"),
        read_annotation(args.test_csv, "test"),
    ]
    manifest = pd.concat(frames, ignore_index=True)

    # Basic consistency checks
    if manifest["video_name"].duplicated().any():
        dups = manifest.loc[manifest["video_name"].duplicated(), "video_name"].tolist()
        raise ValueError(f"Duplicate video filenames found: {dups[:10]}")

    manifest.to_csv(args.out_dir / "labels_manifest.csv", index=False)

    source_counts = (
        manifest.groupby("source_video_id")
        .agg(
            n_segments=("video_name", "nunique"),
            original_partitions=("original_partition", lambda x: ",".join(sorted(set(x)))),
            n_original_partitions=("original_partition", lambda x: len(set(x))),
        )
        .reset_index()
    )
    source_counts.to_csv(args.out_dir / "source_counts.csv", index=False)

    # Summaries by original partition
    by_split = (
        manifest.groupby("original_partition")
        .agg(segments=("video_name", "nunique"), sources=("source_video_id", "nunique"))
        .reset_index()
    )
    by_split.to_csv(args.out_dir / "partition_counts.csv", index=False)

    split_sets = manifest.groupby("original_partition")["source_video_id"].apply(set).to_dict()
    overlap_rows = []
    for a, b in combinations(sorted(split_sets), 2):
        inter = split_sets[a] & split_sets[b]
        overlap_rows.append({
            "partition_a": a,
            "partition_b": b,
            "n_shared_sources": len(inter),
            "n_segments_in_shared_sources": int(manifest[manifest["source_video_id"].isin(inter)].shape[0]),
        })
    pd.DataFrame(overlap_rows).to_csv(args.out_dir / "source_split_overlap.csv", index=False)

    overlap_summary = pd.DataFrame([{
        "n_segments": manifest["video_name"].nunique(),
        "n_sources": manifest["source_video_id"].nunique(),
        "n_sources_in_multiple_original_partitions": int((source_counts["n_original_partitions"] > 1).sum()),
        "n_segments_in_multi_partition_sources": int(manifest[manifest["source_video_id"].isin(source_counts.loc[source_counts["n_original_partitions"] > 1, "source_video_id"])].shape[0]),
    }])
    overlap_summary.to_csv(args.out_dir / "source_overlap_summary.csv", index=False)

    summarize_labels(manifest, "original_partition").to_csv(args.out_dir / "label_summary_by_original_split.csv", index=False)
    summarize_labels(manifest, None).to_csv(args.out_dir / "label_summary_overall.csv", index=False)

    manifest[TRAITS].corr(method="pearson").to_csv(args.out_dir / "label_correlations_pearson.csv")
    manifest[TRAITS].corr(method="spearman").to_csv(args.out_dir / "label_correlations_spearman.csv")

    if args.make_plots:
        import matplotlib.pyplot as plt
        plot_dir = args.out_dir / "label_histograms"
        plot_dir.mkdir(exist_ok=True)
        for trait in TRAITS:
            plt.figure()
            manifest[trait].hist(bins=30)
            plt.title(f"Official {trait} labels")
            plt.xlabel("Label value")
            plt.ylabel("Number of segments")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{trait}_histogram.png", dpi=200)
            plt.close()

    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
