from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Compare segment-level metrics across runs.")
    p.add_argument("--eval-root", required=True, type=Path, help="Directory containing run subdirectories.")
    p.add_argument("--split", default="test", choices=["validation", "test", "training"])
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    pattern = f"**/metrics_segment_level_overall_{args.split}.csv"
    for metrics_file in args.eval_root.glob(pattern):
        df = pd.read_csv(metrics_file)
        # Expected path: <run>/eval_<split>/metrics...
        run_name = metrics_file.parent.parent.name
        row = df.iloc[0].to_dict()
        row["run"] = run_name
        row["metrics_file"] = str(metrics_file)
        rows.append(row)

    if not rows:
        raise SystemExit(f"No metrics files found under {args.eval_root} with pattern {pattern}")

    out_df = pd.DataFrame(rows)
    cols = ["run"] + [c for c in out_df.columns if c not in {"run", "metrics_file"}] + ["metrics_file"]
    out_df = out_df[cols].sort_values("mae", ascending=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(out_df.to_string(index=False))
    print(f"Wrote comparison to {args.out}")


if __name__ == "__main__":
    main()
