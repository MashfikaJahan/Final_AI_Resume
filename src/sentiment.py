"""Sentiment scoring: VADER + TextBlob ensemble for resume skill text.

Roles (Pipeline_plan §2.1):
  1. Baseline measurement — quantify tone of original skills before variant generation.
  2. Quality gate — re-score variants, flag/reject those drifting beyond tolerance.

Ensemble (Pipeline_plan §2.2):
  sentiment_score = vader_weight * vader_compound + textblob_weight * textblob_polarity
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_vader = SentimentIntensityAnalyzer()


def compute_sentiment(text: str) -> dict[str, float]:
    """Return sentiment scores for a single text string.

    Returns:
        {
            "vader_compound": float,   # [-1, +1]
            "vader_pos": float,
            "vader_neg": float,
            "vader_neu": float,
            "textblob_polarity": float,  # [-1, +1]
            "textblob_subjectivity": float,  # [0, 1]
            "sentiment_score": float,   # ensemble score
        }
    """
    vs = _vader.polarity_scores(text)
    blob = TextBlob(text)

    return {
        "vader_compound": vs["compound"],
        "vader_pos": vs["pos"],
        "vader_neg": vs["neg"],
        "vader_neu": vs["neu"],
        "textblob_polarity": blob.sentiment.polarity,
        "textblob_subjectivity": blob.sentiment.subjectivity,
        "sentiment_score": 0.6 * vs["compound"] + 0.4 * blob.sentiment.polarity,
    }


def label_sentiment(
    score: float,
    pos_thresh: float = 0.05,
    neg_thresh: float = -0.05,
) -> str:
    """Classify ensemble score into 'positive', 'negative', or 'neutral'."""
    if score > pos_thresh:
        return "positive"
    if score < neg_thresh:
        return "negative"
    return "neutral"


def score_dataframe(
    df: pd.DataFrame,
    text_column: str,
    prefix: str = "",
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add sentiment columns to *df* for the given text column.

    Adds columns (each prefixed with *prefix*):
        sentiment_score, sentiment_label, vader_compound,
        textblob_polarity, textblob_subjectivity
    """
    sent_cfg = (config or {}).get("sentiment", {})
    vader_w = sent_cfg.get("vader_weight", 0.6)
    tb_w = sent_cfg.get("textblob_weight", 0.4)
    pos_thresh = sent_cfg.get("positive_threshold", 0.05)
    neg_thresh = sent_cfg.get("negative_threshold", -0.05)

    scores = df[text_column].astype(str).apply(compute_sentiment).apply(pd.Series)
    scores["sentiment_score"] = vader_w * scores["vader_compound"] + tb_w * scores["textblob_polarity"]
    scores["sentiment_label"] = scores["sentiment_score"].apply(
        lambda s: label_sentiment(s, pos_thresh, neg_thresh),
    )

    for col in ["sentiment_score", "sentiment_label", "vader_compound",
                "textblob_polarity", "textblob_subjectivity"]:
        df[f"{prefix}{col}"] = scores[col]

    return df


def check_sentiment_drift(
    df: pd.DataFrame,
    tolerance: float = 0.15,
) -> pd.DataFrame:
    """Compare variant sentiment to original sentiment.

    Expects columns: sentiment_score (original), sentiment_score_variant.
    Adds:
        sentiment_delta  = sentiment_score_variant - sentiment_score_original
        sentiment_drift_flag = abs(delta) > tolerance
    """
    df["sentiment_delta"] = df["sentiment_score_variant"] - df["sentiment_score"]
    df["sentiment_drift_flag"] = df["sentiment_delta"].abs() > tolerance
    return df
