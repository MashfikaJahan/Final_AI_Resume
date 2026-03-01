"""Evaluation metrics: delta score, rank shift, top-K inclusion, threshold pass/fail.

All comparisons are paired: each variant row is compared against its
control (same Resume_ID, Job Role, variant_type='control').
Pipeline-agnostic: consumes variants_scored.csv regardless of variant generation method.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _validate_input(df: pd.DataFrame, methods: list[str]) -> None:
    """Validate input schema and invariants. Raises on critical failures."""
    required_base = ["Resume_ID", "Job Role", "variant_type", "skills_variant"]
    for col in required_base:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    for m in methods:
        for suffix in ("score", "rank", "percentile"):
            col = f"{suffix}_{m}"
            if col not in df.columns:
                raise ValueError(f"Required scoring column missing: {col}")

    controls = df.loc[df["variant_type"] == "control"]
    if len(controls) == 0:
        raise ValueError("No control rows found (variant_type='control')")
    control_keys = controls.groupby(["Resume_ID", "Job Role"], observed=True).size()
    dupes = control_keys[control_keys > 1]
    if len(dupes) > 0:
        raise ValueError(
            f"Multiple control rows per (Resume_ID, Job Role): {dupes.index.tolist()[:5]}..."
        )


def _merge_control(
    df: pd.DataFrame, methods: list[str], how: str = "inner"
) -> tuple[pd.DataFrame, int]:
    """Join each variant row with its control row on Resume_ID + Job Role.

    Returns (merged_df, dropped_rows). Uses inner merge by default so variants
    without matching control are dropped and counted.
    """
    controls = df.loc[df["variant_type"] == "control"].copy()
    variants = df.loc[df["variant_type"] != "control"].copy()
    n_variants_before = len(variants)

    control_cols = ["Resume_ID", "Job Role"]
    for m in methods:
        control_cols.extend([f"score_{m}", f"rank_{m}", f"percentile_{m}"])

    missing = [c for c in control_cols if c not in controls.columns]
    if missing:
        raise ValueError(f"Control rows missing columns: {missing}")

    rename_map = {c: f"{c}_control" for c in control_cols if c not in ("Resume_ID", "Job Role")}
    controls_slim = controls[control_cols].rename(columns=rename_map)

    merged = variants.merge(controls_slim, on=["Resume_ID", "Job Role"], how=how)
    dropped = n_variants_before - len(merged)
    if dropped > 0:
        logger.warning("Dropped %d variant rows without matching control", dropped)
    return merged, dropped


def _sanity_check_row_counts(df: pd.DataFrame, merged: pd.DataFrame, dropped: int) -> None:
    """Log sanity check on row counts."""
    n_controls = (df["variant_type"] == "control").sum()
    n_roles = df["Job Role"].nunique()
    expected_variants = len(df) - n_controls
    actual_detail = len(merged)
    if actual_detail + dropped != expected_variants:
        logger.warning(
            "Row count mismatch: expected %d variant rows, got %d detail + %d dropped",
            expected_variants, actual_detail, dropped,
        )
    logger.info(
        "Eval sanity: %d controls, %d variant rows -> %d detail rows (%d dropped)",
        n_controls, expected_variants, actual_detail, dropped,
    )


def compute_deltas(merged: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Add delta_score and rank_shift columns for each scoring method."""
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
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Run full evaluation pipeline. Returns (detail_df, summary_df, dropped_rows).

    Pipeline-agnostic: works with any variants_scored.csv matching the input contract.
    Optional columns (e.g. sentiment_*) are passed through to eval_detail.csv.
    """
    methods = config["scoring"]["methods"]
    k_values = config["scoring"]["top_k"]
    threshold = config["scoring"]["threshold"]

    _validate_input(df, methods)
    merged, dropped = _merge_control(df, methods)
    _sanity_check_row_counts(df, merged, dropped)

    detail = compute_deltas(merged, methods)
    detail = compute_topk(detail, methods, k_values)
    detail = compute_threshold(detail, methods, threshold)

    summary = summarize_by_variant_type(detail, methods, k_values)

    eval_dir = Path(config["data"]["eval_summary_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(eval_dir / "eval_summary.csv", index=False)
    detail.to_csv(eval_dir / "eval_detail.csv", index=False)

    logger.info(
        "Evaluation complete — %d detail rows, %d summary rows, %d dropped → %s",
        len(detail), len(summary), dropped, eval_dir,
    )

    return detail, summary, dropped
