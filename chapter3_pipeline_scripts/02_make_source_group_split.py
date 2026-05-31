#!/usr/bin/env python3
"""
Create an optional leakage-aware split using source-video identifiers as groups.

This script preserves official segment-level labels, but changes the split so
that no source_video_id appears in more than one final split. If this split is
used, results should be described as leakage-aware image-based evaluation, not
as a direct reproduction of the official ChaLearn leaderboard protocol.

Input:
  labels_manifest.csv from 01_analyze_labels.py

Outputs:
  final_split_manifest.csv
  final_split_counts.csv
  final_split_label_summary.csv
  final_split_source_overlap_check.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("final_split"):
        for trait in TRAITS:
            s = g[trait]
            rows.append({
                "final_split": split,
                "trait": trait,
                "n": int(s.notna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "median": float(s.median()),
                "max": float(s.max()),
            })
    return pd.DataFrame(rows)


def score_split(df: pd.DataFrame, ratios: dict[str, float]) -> float:
    """Lower is better: split size error + label mean imbalance."""
    n = len(df)
    size_penalty = 0.0
    label_penalty = 0.0
    overall_means = df[TRAITS].mean()
    for split, ratio in ratios.items():
        g = df[df["final_split"] == split]
        size_penalty += abs(len(g) / n - ratio)
        label_penalty += (g[TRAITS].mean() - overall_means).abs().mean()
    return float(size_penalty + 2.0 * label_penalty)


def assign_groups_randomly(df: pd.DataFrame, seed: int, ratios: dict[str, float]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_counts = df.groupby("source_video_id").size().reset_index(name="n_segments")
    group_counts = group_counts.sample(frac=1, random_state=seed).reset_index(drop=True)
    total_segments = int(group_counts["n_segments"].sum())
    target_counts = {k: v * total_segments for k, v in ratios.items()}
    current_counts = {k: 0 for k in ratios}
    group_to_split = {}

    # Greedy assignment: assign larger groups first to split with largest relative deficit.
    group_counts = group_counts.sort_values("n_segments", ascending=False)
    for _, row in group_counts.iterrows():
        deficits = {k: target_counts[k] - current_counts[k] for k in ratios}
        # Prefer the split with largest positive deficit; if all negative, least overfilled.
        split = max(deficits, key=deficits.get)
        group_to_split[row["source_video_id"]] = split
        current_counts[split] += int(row["n_segments"])

    out = df.copy()
    out["final_split"] = out["source_video_id"].map(group_to_split)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_manifest", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--n_seeds", type=int, default=200)
    parser.add_argument("--seed_start", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.labels_manifest)
    ratios = {"training": args.train_ratio, "validation": args.val_ratio, "test": args.test_ratio}
    s = sum(ratios.values())
    ratios = {k: v / s for k, v in ratios.items()}

    best = None
    best_score = float("inf")
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        candidate = assign_groups_randomly(df, seed, ratios)
        sc = score_split(candidate, ratios)
        if sc < best_score:
            best = candidate
            best_score = sc
            best_seed = seed

    best.to_csv(args.out_dir / "final_split_manifest.csv", index=False)
    counts = (
        best.groupby("final_split")
        .agg(segments=("video_name", "nunique"), sources=("source_video_id", "nunique"))
        .reset_index()
    )
    counts["segment_share"] = counts["segments"] / counts["segments"].sum()
    counts.to_csv(args.out_dir / "final_split_counts.csv", index=False)
    label_summary(best).to_csv(args.out_dir / "final_split_label_summary.csv", index=False)

    # Check source leakage in final split
    source_splits = best.groupby("source_video_id")["final_split"].nunique().reset_index(name="n_final_splits")
    overlap_check = pd.DataFrame([{
        "best_seed": best_seed,
        "best_score": best_score,
        "n_sources": best["source_video_id"].nunique(),
        "n_sources_in_multiple_final_splits": int((source_splits["n_final_splits"] > 1).sum()),
    }])
    overlap_check.to_csv(args.out_dir / "final_split_source_overlap_check.csv", index=False)
    print(f"Wrote outputs to {args.out_dir}; best seed={best_seed}, score={best_score:.6f}")


if __name__ == "__main__":
    main()
