"""Logistic regression: predict Recruiter Decision from screening outputs + resume features.

Produces:
  - reports/logistic_regression_metrics.json
  - reports/logistic_regression_coefficients.csv
  - reports/logistic_regression_predictions.csv
  - reports/logistic_regression_summary.md
  - reports/logistic_regression_coefficients.png
  - reports/logistic_regression_roc.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CANDIDATE_FEATURES = [
    "AI Score (0-100)",
    "Experience (Years)",
    "Projects Count",
    "Salary Expectation ($)",
    "score_tfidf",
    "score_bm25",
    "score_embedding",
    "rank_tfidf",
    "percentile_tfidf",
    "rank_bm25",
    "percentile_bm25",
]


def _select_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of CANDIDATE_FEATURES that actually exist in *df*."""
    return [c for c in CANDIDATE_FEATURES if c in df.columns]


def _fit_and_report(
    df: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    suffix: str = "",
) -> dict[str, Any]:
    """Fit one logistic regression model and write all artifacts.

    *suffix* is appended to filenames to distinguish model variants
    (e.g. ``_no_ai_score``).
    """
    seed = config.get("seed", 42)
    lr_cfg = config.get("logistic_regression", {})
    test_size = lr_cfg.get("test_size", 0.2)
    class_weight = lr_cfg.get("class_weight", "balanced")
    max_iter = lr_cfg.get("max_iter", 1000)
    reports_dir = Path(lr_cfg.get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_model = df[features + ["target"]].dropna()
    tag = f" ({suffix.strip('_')})" if suffix else ""
    logger.info(
        "Logistic regression%s — %d rows, %d features: %s",
        tag, len(df_model), len(features), features,
    )

    X = df_model[features].values
    y = df_model["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "classification_report": report,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "features": features,
        "class_distribution_train": {
            "Hire": int(y_train.sum()),
            "Reject": int(len(y_train) - y_train.sum()),
        },
    }

    with open(reports_dir / f"logistic_regression_metrics{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", ascending=False)
    coef_df.to_csv(reports_dir / f"logistic_regression_coefficients{suffix}.csv", index=False)

    pred_df = df.copy()
    X_all = scaler.transform(pred_df[features].fillna(0).values)
    pred_df["predicted_hire"] = model.predict(X_all)
    pred_df["hire_probability"] = model.predict_proba(X_all)[:, 1]
    pred_df.to_csv(reports_dir / f"logistic_regression_predictions{suffix}.csv", index=False)

    _write_summary_md(metrics, coef_df, reports_dir, suffix=suffix)
    _plot_coefficients(coef_df, reports_dir, config, suffix=suffix)
    _plot_roc(y_test, y_prob, auc, reports_dir, config, suffix=suffix)

    logger.info(
        "Logistic regression%s complete — accuracy=%.4f, AUC=%.4f → %s",
        tag, acc, auc, reports_dir,
    )
    return metrics


def run_logistic_regression(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit two logistic regression models: full and screening-only (no AI Score)."""
    df = df.copy()
    df["target"] = (df["Recruiter Decision"].astype(str).str.strip() == "Hire").astype(int)

    features_full = _select_features(df)
    if not features_full:
        logger.error("No usable features found — aborting logistic regression")
        return {}

    metrics_full = _fit_and_report(df, features_full, config, suffix="")

    features_no_ai = [f for f in features_full if f != "AI Score (0-100)"]
    if features_no_ai and "AI Score (0-100)" in features_full:
        _fit_and_report(df, features_no_ai, config, suffix="_no_ai_score")

    return metrics_full


def _write_summary_md(
    metrics: dict,
    coef_df: pd.DataFrame,
    reports_dir: Path,
    *,
    suffix: str = "",
) -> None:
    tag = f" ({suffix.strip('_').replace('_', ' ').title()})" if suffix else ""
    lines = [
        f"# Logistic Regression Summary{tag}",
        "",
        "## Performance",
        "",
        f"- **Accuracy:** {metrics['accuracy']}",
        f"- **ROC AUC:** {metrics['roc_auc']}",
        f"- **Train size:** {metrics['n_train']}",
        f"- **Test size:** {metrics['n_test']}",
        "",
        "## Class Distribution (Train)",
        "",
        f"- Hire: {metrics['class_distribution_train']['Hire']}",
        f"- Reject: {metrics['class_distribution_train']['Reject']}",
        "",
        "## Feature Coefficients",
        "",
        "| Feature | Coefficient |",
        "|---------|------------|",
    ]
    for _, row in coef_df.iterrows():
        lines.append(f"| {row['feature']} | {row['coefficient']:.4f} |")
    lines.append("")

    (reports_dir / f"logistic_regression_summary{suffix}.md").write_text("\n".join(lines))


def _plot_coefficients(
    coef_df: pd.DataFrame,
    reports_dir: Path,
    config: dict,
    *,
    suffix: str = "",
) -> None:
    tag = f" ({suffix.strip('_').replace('_', ' ').title()})" if suffix else ""
    dpi = config.get("visualization", {}).get("dpi", 300)
    fig, ax = plt.subplots(figsize=(8, max(4, len(coef_df) * 0.5)))
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["coefficient"]]
    ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    ax.set_xlabel("Coefficient (standardized)")
    ax.set_title(f"Logistic Regression — Feature Coefficients{tag}")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(reports_dir / f"logistic_regression_coefficients{suffix}.png", dpi=dpi)
    plt.close(fig)


def _plot_roc(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    auc_score: float,
    reports_dir: Path,
    config: dict,
    *,
    suffix: str = "",
) -> None:
    tag = f" ({suffix.strip('_').replace('_', ' ').title()})" if suffix else ""
    dpi = config.get("visualization", {}).get("dpi", 300)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — Recruiter Decision{tag}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(reports_dir / f"logistic_regression_roc{suffix}.png", dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    config_path = "configs/default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    scored_path = cfg["data"]["variants_scored"]
    df = pd.read_csv(scored_path)
    run_logistic_regression(df, cfg)
