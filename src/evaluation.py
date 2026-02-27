"""Evaluation metrics: delta score, rank shift, top-K inclusion, threshold pass/fail.

All comparisons are paired: each variant row is compared against its
control (same Resume_ID, variant_type='control').
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _merge_control(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Join each variant row with its control row on Resume_ID."""
    controls = df.loc[df["variant_type"] == "control"].copy()
    variants = df.loc[df["variant_type"] != "control"].copy()

    control_cols = ["Resume_ID", "Job Role"]
    for m in methods:
        control_cols.extend([f"score_{m}", f"rank_{m}", f"percentile_{m}"])

    rename_map = {c: f"{c}_control" for c in control_cols if c not in ("Resume_ID", "Job Role")}
    controls_slim = controls[control_cols].rename(columns=rename_map)

    merged = variants.merge(controls_slim, on=["Resume_ID", "Job Role"], how="left")
    return merged


def compute_deltas(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Add delta_score and rank_shift columns for each scoring method."""
    merged = _merge_control(df, methods)

    for m in methods:
        merged[f"delta_score_{m}"] = merged[f"score_{m}"] - merged[f"score_{m}_control"]
        merged[f"rank_shift_{m}"] = merged[f"rank_{m}_control"] - merged[f"rank_{m}"]

    return merged


def compute_topk(df: pd.DataFrame, methods: list[str], k_values: list[int]) -> pd.DataFrame:
    """Add boolean top-K inclusion columns for variant and control."""
    for m in methods:
        for k in k_values:
            df[f"topk_{m}_k{k}"] = df[f"rank_{m}"] <= k
            if f"rank_{m}_control" in df.columns:
                df[f"topk_{m}_k{k}_control"] = df[f"rank_{m}_control"] <= k
                df[f"topk_{m}_k{k}_flip"] = (
                    df[f"topk_{m}_k{k}"] != df[f"topk_{m}_k{k}_control"]
                )
    return df


def compute_threshold(
    df: pd.DataFrame,
    methods: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Add threshold pass/fail columns for variant and control."""
    for m in methods:
        df[f"threshold_pass_{m}"] = df[f"score_{m}"] >= threshold
        if f"score_{m}_control" in df.columns:
            df[f"threshold_pass_{m}_control"] = df[f"score_{m}_control"] >= threshold
            df[f"threshold_flip_{m}"] = (
                df[f"threshold_pass_{m}"] != df[f"threshold_pass_{m}_control"]
            )
    return df


def summarize_by_variant_type(
    df: pd.DataFrame,
    methods: list[str],
    k_values: list[int],
) -> pd.DataFrame:
    """Aggregate evaluation metrics grouped by variant_type and Job Role."""
    agg_dict: dict[str, list[str]] = {}

    for m in methods:
        agg_dict[f"delta_score_{m}"] = ["mean", "std", "min", "max"]
        agg_dict[f"rank_shift_{m}"] = ["mean", "std", "min", "max"]
        for k in k_values:
            flip_col = f"topk_{m}_k{k}_flip"
            if flip_col in df.columns:
                agg_dict[flip_col] = ["sum", "mean"]
        thresh_flip = f"threshold_flip_{m}"
        if thresh_flip in df.columns:
            agg_dict[thresh_flip] = ["sum", "mean"]

    available = {k: v for k, v in agg_dict.items() if k in df.columns}

    summary = df.groupby(["variant_type", "Job Role"], observed=True).agg(available)
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()

    return summary


def evaluate(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full evaluation pipeline. Returns (detail_df, summary_df)."""
    methods = config["scoring"]["methods"]
    k_values = config["scoring"]["top_k"]
    threshold = config["scoring"]["threshold"]

    detail = compute_deltas(df, methods)
    detail = compute_topk(detail, methods, k_values)
    detail = compute_threshold(detail, methods, threshold)

    summary = summarize_by_variant_type(detail, methods, k_values)

    eval_dir = Path(config["data"]["eval_summary_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(eval_dir / "eval_summary.csv", index=False)
    detail.to_csv(eval_dir / "eval_detail.csv", index=False)

    logger.info(
        "Evaluation complete — %d detail rows, %d summary rows → %s",
        len(detail), len(summary), eval_dir,
    )

    return detail, summary
