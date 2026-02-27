"""Visualization: publication-ready plots for the quasi-experimental analysis.

Plots produced:
  1. Score distribution by variant type (box/violin) — per method
  2. Rank shift distribution by variant type — per method
  3. Top-K flip rates by variant type — grouped bar
  4. Threshold flip rates by variant type
  5. Delta score heatmap (variant_type × Job Role)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _out_dir(config: dict) -> Path:
    d = Path(config.get("visualization", {}).get("output_dir", "outputs")) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fig_params(config: dict) -> tuple[tuple[int, int], int]:
    vis = config.get("visualization", {})
    figsize = tuple(vis.get("figsize", [10, 6]))
    dpi = vis.get("dpi", 300)
    return figsize, dpi


def plot_score_distributions(
    detail: pd.DataFrame,
    config: dict,
) -> list[Path]:
    """Box plots of score by variant_type for each scoring method."""
    methods = config["scoring"]["methods"]
    figsize, dpi = _fig_params(config)
    out = _out_dir(config)
    paths: list[Path] = []

    for m in methods:
        col = f"score_{m}"
        if col not in detail.columns:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        plot_df = detail[["variant_type", "Job Role", col]].copy()
        plot_df = plot_df.rename(columns={col: "Score"})
        sns.boxplot(data=plot_df, x="variant_type", y="Score", hue="Job Role", ax=ax)
        ax.set_title(f"Score Distribution — {m.upper()}")
        ax.set_xlabel("Variant Type")
        ax.set_ylabel(f"{m.upper()} Score")
        ax.legend(title="Job Role", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        p = out / f"score_distribution_{m}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    logger.info("Score distribution plots → %s", [str(p) for p in paths])
    return paths


def plot_rank_shifts(
    detail: pd.DataFrame,
    config: dict,
) -> list[Path]:
    """Box plots of rank shift by variant_type for each method."""
    methods = config["scoring"]["methods"]
    figsize, dpi = _fig_params(config)
    out = _out_dir(config)
    paths: list[Path] = []

    for m in methods:
        col = f"rank_shift_{m}"
        if col not in detail.columns:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=detail, x="variant_type", y=col, hue="Job Role", ax=ax)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(f"Rank Shift — {m.upper()} (positive = improved)")
        ax.set_xlabel("Variant Type")
        ax.set_ylabel("Rank Shift")
        ax.legend(title="Job Role", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        p = out / f"rank_shift_{m}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    logger.info("Rank shift plots → %s", [str(p) for p in paths])
    return paths


def plot_topk_flip_rates(
    summary: pd.DataFrame,
    config: dict,
) -> list[Path]:
    """Grouped bar chart of top-K flip rates by variant_type."""
    methods = config["scoring"]["methods"]
    k_values = config["scoring"]["top_k"]
    figsize, dpi = _fig_params(config)
    out = _out_dir(config)
    paths: list[Path] = []

    for m in methods:
        flip_cols = [f"topk_{m}_k{k}_flip_mean" for k in k_values]
        available = [c for c in flip_cols if c in summary.columns]
        if not available:
            continue

        grouped = summary.groupby("variant_type", observed=True)[available].mean()

        fig, ax = plt.subplots(figsize=figsize)
        grouped.plot(kind="bar", ax=ax)
        ax.set_title(f"Top-K Flip Rate — {m.upper()}")
        ax.set_xlabel("Variant Type")
        ax.set_ylabel("Flip Rate (fraction of resumes)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(title="K value", labels=[f"K={k}" for k in k_values if f"topk_{m}_k{k}_flip_mean" in available])
        fig.tight_layout()
        p = out / f"topk_flip_{m}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    logger.info("Top-K flip plots → %s", [str(p) for p in paths])
    return paths


def plot_delta_heatmap(
    summary: pd.DataFrame,
    config: dict,
) -> list[Path]:
    """Heatmap of mean delta score (variant_type × Job Role) per method."""
    methods = config["scoring"]["methods"]
    figsize, dpi = _fig_params(config)
    out = _out_dir(config)
    paths: list[Path] = []

    for m in methods:
        col = f"delta_score_{m}_mean"
        if col not in summary.columns:
            continue

        pivot = summary.pivot_table(
            index="variant_type", columns="Job Role", values=col, aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot, annot=True, fmt=".4f", cmap="RdYlGn", center=0,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(f"Mean Delta Score — {m.upper()} (variant − control)")
        fig.tight_layout()
        p = out / f"delta_heatmap_{m}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    logger.info("Delta heatmap plots → %s", [str(p) for p in paths])
    return paths


def plot_threshold_flips(
    summary: pd.DataFrame,
    config: dict,
) -> list[Path]:
    """Bar chart of threshold flip rates by variant type."""
    methods = config["scoring"]["methods"]
    figsize, dpi = _fig_params(config)
    out = _out_dir(config)
    paths: list[Path] = []

    for m in methods:
        col = f"threshold_flip_{m}_mean"
        if col not in summary.columns:
            continue

        grouped = summary.groupby("variant_type", observed=True)[col].mean()

        fig, ax = plt.subplots(figsize=figsize)
        grouped.plot(kind="bar", ax=ax, color="#3498db")
        ax.set_title(f"Threshold Flip Rate — {m.upper()} (threshold={config['scoring']['threshold']})")
        ax.set_xlabel("Variant Type")
        ax.set_ylabel("Flip Rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        p = out / f"threshold_flip_{m}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    logger.info("Threshold flip plots → %s", [str(p) for p in paths])
    return paths


def generate_all_plots(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict,
) -> list[Path]:
    """Run all visualization functions. Returns list of output paths."""
    all_paths: list[Path] = []
    all_paths.extend(plot_score_distributions(detail, config))
    all_paths.extend(plot_rank_shifts(detail, config))
    all_paths.extend(plot_topk_flip_rates(summary, config))
    all_paths.extend(plot_delta_heatmap(summary, config))
    all_paths.extend(plot_threshold_flips(summary, config))
    logger.info("All plots generated — %d files", len(all_paths))
    return all_paths
