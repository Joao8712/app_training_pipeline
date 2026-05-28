from __future__ import annotations

import numpy as np
import pandas as pd

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if len(a) < 2:
        return float("nan")
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    return float(s1.corr(s2, method=method))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, traits: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    traits = traits or TRAITS
    rows = []

    for i, trait in enumerate(traits):
        yt = y_true[:, i].astype(float)
        yp = y_pred[:, i].astype(float)
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        rows.append({
            "trait": trait,
            "mae": mae,
            "chalearn_accuracy": 1.0 - mae,
            "rmse": rmse,
            "pearson": _safe_corr(yt, yp, "pearson"),
            "spearman": _safe_corr(yt, yp, "spearman"),
            "n": len(yt),
        })

    by_trait = pd.DataFrame(rows)
    overall = pd.DataFrame([{
        "mae": float(by_trait["mae"].mean()),
        "chalearn_accuracy": float(by_trait["chalearn_accuracy"].mean()),
        "rmse": float(by_trait["rmse"].mean()),
        "pearson": float(by_trait["pearson"].mean(skipna=True)),
        "spearman": float(by_trait["spearman"].mean(skipna=True)),
        "n": int(by_trait["n"].iloc[0]) if len(by_trait) else 0,
    }])
    return by_trait, overall


def aggregate_to_segment_level(image_df: pd.DataFrame, traits: list[str] | None = None) -> pd.DataFrame:
    traits = traits or TRAITS
    pred_cols = [f"pred_{t}" for t in traits]

    agg_dict = {c: "mean" for c in pred_cols}
    for t in traits:
        agg_dict[t] = "first"
    extra_cols = ["final_split", "original_partition", "source_video_id"]
    for c in extra_cols:
        if c in image_df.columns:
            agg_dict[c] = "first"

    return image_df.groupby("video_name", as_index=False).agg(agg_dict)
