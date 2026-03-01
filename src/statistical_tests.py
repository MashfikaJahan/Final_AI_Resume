"""Statistical significance tests for paired variant–control comparisons.

Produces per (variant_type × Job Role × method):
  - Wilcoxon signed-rank test (p-value, test statistic)
  - Cohen's d effect size
  - Sample size (n)

Output: data/processed/eval_summary/statistical_tests.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _cohens_d(x: np.ndarray) -> float:
    """Cohen's d for a paired (one-sample) design: mean(delta) / std(delta)."""
    if len(x) < 2 or np.std(x, ddof=1) == 0:
        return 0.0
    return float(np.mean(x) / np.std(x, ddof=1))


def _effect_size_label(d: float) -> str:
    """Conventional interpretation (Cohen, 1988)."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def run_statistical_tests(
    detail: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run Wilcoxon signed-rank + Cohen's d on every group slice.

    Returns a tidy DataFrame with one row per
    (variant_type, Job Role, method) combination.
    """
    methods: list[str] = config["scoring"]["methods"]
    eval_dir = Path(config["data"]["eval_summary_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for (vtype, role), grp in detail.groupby(
        ["variant_type", "Job Role"], observed=True,
    ):
        for m in methods:
            delta_col = f"delta_score_{m}"
            rank_col = f"rank_shift_{m}"
            if delta_col not in grp.columns:
                continue

            deltas = grp[delta_col].dropna().values
            rank_shifts = grp[rank_col].dropna().values if rank_col in grp.columns else np.array([])
            n = len(deltas)

            if n < 10:
                logger.warning(
                    "Too few observations (%d) for %s / %s / %s — skipping",
                    n, vtype, role, m,
                )
                continue

            d_score = _cohens_d(deltas)

            if np.all(deltas == 0):
                w_stat, w_p = np.nan, 1.0
            else:
                w_stat, w_p = stats.wilcoxon(deltas, alternative="two-sided")

            d_rank = _cohens_d(rank_shifts) if len(rank_shifts) >= 2 else np.nan

            records.append({
                "variant_type": vtype,
                "Job Role": role,
                "method": m.upper(),
                "n": n,
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas, ddof=1)),
                "delta_median": float(np.median(deltas)),
                "cohens_d_score": round(d_score, 4),
                "effect_size": _effect_size_label(d_score),
                "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
                "wilcoxon_p": float(w_p),
                "significant_005": w_p < 0.05,
                "significant_001": w_p < 0.01,
                "rank_shift_mean": float(np.mean(rank_shifts)) if len(rank_shifts) else np.nan,
                "cohens_d_rank": round(d_rank, 4) if not np.isnan(d_rank) else np.nan,
            })

    results = pd.DataFrame(records)

    out_path = eval_dir / "statistical_tests.csv"
    results.to_csv(out_path, index=False)
    logger.info(
        "Statistical tests complete — %d tests, %d significant (p<0.05) → %s",
        len(results),
        int(results["significant_005"].sum()),
        out_path,
    )
    return results
