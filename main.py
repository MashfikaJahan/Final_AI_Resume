"""CLI entry point for the Same Skills, Different Words pipeline.

Usage:
    python main.py --config configs/default.yaml                    # all stages
    python main.py --config configs/default.yaml --stage ingest     # just ingestion
    python main.py --config configs/default.yaml --stage sentiment  # baseline sentiment
    python main.py --config configs/default.yaml --stage variants   # NLP variants + sentiment QA
    python main.py --config configs/default.yaml --stage score      # score variants
    python main.py --config configs/default.yaml --stage evaluate   # evaluation metrics
    python main.py --config configs/default.yaml --stage stats      # statistical significance tests
    python main.py --config configs/default.yaml --stage logreg     # logistic regression (full + screening-only)
    python main.py --config configs/default.yaml --stage visualize  # plots
    python main.py --config configs/default.yaml --stage report     # Markdown results report (viewable on GitHub)
    python main.py --config configs/default.yaml --stage export     # Excel workbook (requires export.enabled: true in config)
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.data_loader import compute_file_hash, load_raw
from src.preprocessing import preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

STAGES = ["ingest", "sentiment", "variants", "score", "evaluate", "stats", "logreg", "visualize", "report", "export"]


def _git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def write_manifest(
    *,
    stage: str,
    config_path: str,
    config: dict,
    input_path: str,
    input_hash: str | None = None,
    input_rows: int,
    output_path: str,
    output_rows: int,
    dropped_rows: int = 0,
    extra: dict | None = None,
) -> None:
    """Append a JSON-lines entry to outputs/run_manifest.jsonl."""
    manifest_path = Path("outputs/run_manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "git_commit": _git_commit_hash(),
        "config_path": config_path,
        "input_file": input_path,
        "input_sha256": input_hash,
        "input_rows": input_rows,
        "output_file": output_path,
        "output_rows": output_rows,
        "dropped_rows": dropped_rows,
        "seed": config.get("seed"),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
    }
    if extra:
        entry.update(extra)

    with open(manifest_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Run manifest written → %s", manifest_path)


# ── Stage: ingest ─────────────────────────────────────────────────────

def run_ingest(config: dict, config_path: str) -> None:
    """Execute the data ingestion stage."""
    raw_path = config["data"]["raw_path"]
    clean_parquet = config["data"]["clean_parquet"]

    logger.info("=== Stage: data_ingestion ===")

    raw_df, file_hash, _ = load_raw(
        raw_path,
        expected_columns=config["validation"]["required_columns"],
    )
    input_rows = len(raw_df)

    clean_df = preprocess(raw_df, config)

    out_path = Path(clean_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    logger.info("Parquet written → %s (%d rows, %d cols)", out_path, len(clean_df), len(clean_df.columns))

    write_manifest(
        stage="data_ingestion",
        config_path=config_path,
        config=config,
        input_path=raw_path,
        input_hash=file_hash,
        input_rows=input_rows,
        output_path=clean_parquet,
        output_rows=len(clean_df),
        dropped_rows=input_rows - len(clean_df),
    )

    logger.info("=== data_ingestion complete ===")


# ── Stage: sentiment (Pipeline_plan §4.1 Stage 2) ────────────────────

def run_sentiment(config: dict, config_path: str) -> None:
    """Compute sentiment baseline on original skills text."""
    from src.sentiment import score_dataframe

    clean_parquet = config["data"]["clean_parquet"]

    logger.info("=== Stage: sentiment_baseline ===")

    clean_df = pd.read_parquet(clean_parquet)
    clean_df = score_dataframe(clean_df, text_column="Skills", prefix="", config=config)

    dist = clean_df["sentiment_label"].value_counts().to_dict()
    logger.info("Sentiment distribution: %s", dist)

    out_path = Path(clean_parquet)
    clean_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    logger.info("Parquet updated with sentiment columns → %s", out_path)

    write_manifest(
        stage="sentiment_baseline",
        config_path=config_path,
        config=config,
        input_path=clean_parquet,
        input_rows=len(clean_df),
        output_path=clean_parquet,
        output_rows=len(clean_df),
        extra={
            "neutral_count": int(dist.get("neutral", 0)),
            "positive_count": int(dist.get("positive", 0)),
            "negative_count": int(dist.get("negative", 0)),
        },
    )

    logger.info("=== sentiment_baseline complete ===")


# ── Stage: variants (Pipeline_plan §4.2 Stage 3 + Stage 4 QA) ────────

def run_variants(config: dict, config_path: str) -> None:
    """NLP variant generation + sentiment QA (Pipeline_plan §3, §4)."""
    from src.sentiment import check_sentiment_drift, score_dataframe
    from src.variant_generator import generate_variants

    clean_parquet = config["data"]["clean_parquet"]
    variants_parquet = config["data"]["variants_parquet"]
    sent_cfg = config.get("sentiment", {})
    tolerance = sent_cfg.get("tolerance", 0.15)
    drift_action = sent_cfg.get("action_on_drift", "flag")

    logger.info("=== Stage: variant_generation ===")

    clean_df = pd.read_parquet(clean_parquet)

    # Ensure sentiment baseline exists (Pipeline_plan §4.2 Stage 2 output)
    if "sentiment_score" not in clean_df.columns:
        logger.info("Sentiment baseline not found — computing inline")
        clean_df = score_dataframe(clean_df, text_column="Skills", prefix="", config=config)

    variants_df = generate_variants(clean_df, config)

    # --- Sentiment QA (Pipeline_plan §4 Stage 4) ---
    logger.info("Running sentiment QA on variants …")
    variants_df = score_dataframe(
        variants_df, text_column="skills_variant", prefix="variant_", config=config,
    )
    variants_df.rename(
        columns={
            "variant_sentiment_score": "sentiment_score_variant",
            "variant_sentiment_label": "sentiment_label_variant",
            "variant_vader_compound": "vader_compound_variant",
            "variant_textblob_polarity": "textblob_polarity_variant",
            "variant_textblob_subjectivity": "textblob_subjectivity_variant",
        },
        inplace=True,
    )

    # Control rows: variant sentiment == original sentiment, zero delta
    control_mask = variants_df["variant_type"] == "control"
    variants_df.loc[control_mask, "sentiment_score_variant"] = variants_df.loc[control_mask, "sentiment_score"]
    variants_df.loc[control_mask, "sentiment_label_variant"] = variants_df.loc[control_mask, "sentiment_label"]

    variants_df = check_sentiment_drift(variants_df, tolerance=tolerance)
    variants_df.loc[control_mask, "sentiment_delta"] = 0.0
    variants_df.loc[control_mask, "sentiment_drift_flag"] = False

    n_flagged = int(variants_df["sentiment_drift_flag"].sum())
    drift_by_type = (
        variants_df.loc[~control_mask]
        .groupby("variant_type", observed=True)["sentiment_drift_flag"]
        .agg(["sum", "mean"])
    )
    logger.info("Sentiment drift flagged: %d variants", n_flagged)
    logger.info("Drift by variant type:\n%s", drift_by_type.to_string())

    n_rejected = 0
    if drift_action == "reject" and n_flagged > 0:
        n_before = len(variants_df)
        variants_df = variants_df.loc[~variants_df["sentiment_drift_flag"] | control_mask].copy()
        n_rejected = n_before - len(variants_df)
        logger.info("Rejected %d variants exceeding sentiment tolerance", n_rejected)

    out_path = Path(variants_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variants_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    logger.info("Variants written → %s (%d rows)", out_path, len(variants_df))

    write_manifest(
        stage="variant_generation",
        config_path=config_path,
        config=config,
        input_path=clean_parquet,
        input_rows=len(clean_df),
        output_path=variants_parquet,
        output_rows=len(variants_df),
        extra={
            "variant_types": config["variants"]["types"],
            "nlp_model": config.get("variants", {}).get("nlp_model", "en_core_web_md"),
            "sentiment_drift_flagged": n_flagged,
            "sentiment_drift_rejected": n_rejected,
        },
    )

    logger.info("=== variant_generation complete ===")


# ── Stage: score ──────────────────────────────────────────────────────

def run_score(config: dict, config_path: str) -> None:
    """Score variants against job descriptions."""
    from src.scoring import score_variants

    variants_parquet = config["data"]["variants_parquet"]
    scored_csv = config["data"]["variants_scored"]

    logger.info("=== Stage: scoring ===")

    variants_df = pd.read_parquet(variants_parquet)
    scored_df = score_variants(variants_df, config)

    out_path = Path(scored_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(out_path, index=False)
    logger.info("Scored variants written → %s (%d rows)", out_path, len(scored_df))

    write_manifest(
        stage="scoring",
        config_path=config_path,
        config=config,
        input_path=variants_parquet,
        input_rows=len(variants_df),
        output_path=scored_csv,
        output_rows=len(scored_df),
        extra={"methods": config["scoring"]["methods"]},
    )

    logger.info("=== scoring complete ===")


# ── Stage: evaluate ───────────────────────────────────────────────────

def run_evaluate(config: dict, config_path: str) -> None:
    """Compute evaluation metrics (deltas, rank shifts, top-K, threshold)."""
    from src.evaluation import evaluate

    scored_csv = config["data"]["variants_scored"]
    eval_dir = config["data"]["eval_summary_dir"]

    logger.info("=== Stage: evaluation ===")

    scored_df = pd.read_csv(scored_csv)
    detail, summary, dropped = evaluate(scored_df, config)

    write_manifest(
        stage="evaluation",
        config_path=config_path,
        config=config,
        input_path=scored_csv,
        input_rows=len(scored_df),
        output_path=eval_dir,
        output_rows=len(summary),
        dropped_rows=dropped,
        extra={
            "detail_rows": len(detail),
            "summary_rows": len(summary),
            "output_path": eval_dir,  # Eval_plan §6: output_path key
        },
    )

    logger.info("=== evaluation complete ===")


# ── Stage: stats ──────────────────────────────────────────────────────

def run_stats(config: dict, config_path: str) -> None:
    """Run statistical significance tests (Wilcoxon + Cohen's d)."""
    from src.statistical_tests import run_statistical_tests

    eval_dir = Path(config["data"]["eval_summary_dir"])
    detail_path = eval_dir / "eval_detail.csv"

    logger.info("=== Stage: statistical_tests ===")

    detail = pd.read_csv(detail_path)
    results = run_statistical_tests(detail, config)

    write_manifest(
        stage="statistical_tests",
        config_path=config_path,
        config=config,
        input_path=str(detail_path),
        input_rows=len(detail),
        output_path=str(eval_dir / "statistical_tests.csv"),
        output_rows=len(results),
        extra={
            "significant_005": int(results["significant_005"].sum()),
            "total_tests": len(results),
        },
    )

    logger.info("=== statistical_tests complete ===")


# ── Stage: logreg ─────────────────────────────────────────────────────

def run_logreg(config: dict, config_path: str) -> None:
    """Fit logistic regression on scored variants."""
    from src.logistic_regression import run_logistic_regression

    scored_csv = config["data"]["variants_scored"]
    reports_dir = config.get("logistic_regression", {}).get("reports_dir", "reports")

    logger.info("=== Stage: logistic_regression ===")

    scored_df = pd.read_csv(scored_csv)
    metrics = run_logistic_regression(scored_df, config)

    write_manifest(
        stage="logistic_regression",
        config_path=config_path,
        config=config,
        input_path=scored_csv,
        input_rows=len(scored_df),
        output_path=reports_dir,
        output_rows=0,
        extra={
            "accuracy": metrics.get("accuracy"),
            "roc_auc": metrics.get("roc_auc"),
        },
    )

    logger.info("=== logistic_regression complete ===")


# ── Stage: visualize ──────────────────────────────────────────────────

def run_visualize(config: dict, config_path: str) -> None:
    """Generate all publication-ready plots."""
    from src.visualization import generate_all_plots

    eval_dir = Path(config["data"]["eval_summary_dir"])
    detail_path = eval_dir / "eval_detail.csv"
    summary_path = eval_dir / "eval_summary.csv"

    logger.info("=== Stage: visualization ===")

    detail = pd.read_csv(detail_path)
    summary = pd.read_csv(summary_path)
    plot_paths = generate_all_plots(detail, summary, config)

    write_manifest(
        stage="visualization",
        config_path=config_path,
        config=config,
        input_path=str(eval_dir),
        input_rows=len(detail),
        output_path=str(Path(config["visualization"]["output_dir"]) / "figures"),
        output_rows=len(plot_paths),
        extra={"plots": [str(p) for p in plot_paths]},
    )

    logger.info("=== visualization complete (%d plots) ===", len(plot_paths))


# ── Stage: report ─────────────────────────────────────────────────────

def run_report(config: dict, config_path: str) -> None:
    """Generate Markdown results report (RESULTS.md) viewable on GitHub."""
    from src.report import generate_report

    logger.info("=== Stage: report ===")

    report_path = generate_report(config)

    write_manifest(
        stage="report",
        config_path=config_path,
        config=config,
        input_path="(multiple)",
        input_rows=0,
        output_path=str(report_path),
        output_rows=0,
    )

    logger.info("=== report complete → %s ===", report_path)


# ── Stage: export ─────────────────────────────────────────────────────

def run_export(config: dict, config_path: str) -> None:
    """Export all artifacts to a single multi-sheet Excel workbook."""
    if not config.get("export", {}).get("enabled", True):
        logger.info("=== Stage: export — SKIPPED (export.enabled=false) ===")
        return

    from src.export_excel import export_excel

    logger.info("=== Stage: export ===")

    out_path = export_excel(config)

    write_manifest(
        stage="export",
        config_path=config_path,
        config=config,
        input_path="(multiple)",
        input_rows=0,
        output_path=str(out_path),
        output_rows=0,
    )

    logger.info("=== export complete ===")


# ── Dispatch ──────────────────────────────────────────────────────────

STAGE_FNS = {
    "ingest": run_ingest,
    "sentiment": run_sentiment,
    "variants": run_variants,
    "score": run_score,
    "evaluate": run_evaluate,
    "stats": run_stats,
    "logreg": run_logreg,
    "visualize": run_visualize,
    "report": run_report,
    "export": run_export,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Same Skills, Different Words — pipeline runner",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--stage", default="all",
        choices=STAGES + ["all"],
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stage == "all":
        for stage in STAGES:
            STAGE_FNS[stage](config, args.config)
    else:
        STAGE_FNS[args.stage](config, args.config)


if __name__ == "__main__":
    main()
