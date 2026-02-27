"""Score resume variants against job descriptions using TF-IDF and BM25.

Each variant's `skills_variant` text is compared to the job description
for the resume's `Job Role`.  Scores and within-role ranks are appended
to the DataFrame.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _tfidf_scores(
    skills: list[str],
    job_description: str,
) -> np.ndarray:
    """Compute TF-IDF cosine similarity between each skill string and the JD."""
    corpus = [job_description] + skills
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vec = tfidf_matrix[0:1]
    skill_vecs = tfidf_matrix[1:]
    sims = cosine_similarity(skill_vecs, jd_vec).flatten()
    return sims


def _bm25_scores(
    skills: list[str],
    job_description: str,
) -> np.ndarray:
    """Compute BM25 relevance scores for each skill string against the JD."""
    tokenized_corpus = [s.lower().split() for s in skills]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = job_description.lower().split()
    scores = bm25.get_scores(query_tokens)
    return scores


def score_variants(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Score every variant row and add score/rank/percentile columns.

    Scoring is done per Job Role group.  Within each group the variants
    are ranked (ascending rank = better match).
    """
    methods: list[str] = config["scoring"]["methods"]
    jd_map: dict[str, str] = config["scoring"]["job_descriptions"]

    score_cols: list[str] = []

    for method in methods:
        col = f"score_{method}"
        score_cols.append(col)
        df[col] = np.nan

    for role, group in df.groupby("Job Role", observed=True):
        jd = jd_map.get(str(role))
        if jd is None:
            logger.warning("No job description configured for role %r — skipping scoring", role)
            continue

        idx = group.index
        skills_texts = group["skills_variant"].tolist()

        if "tfidf" in methods:
            tfidf = _tfidf_scores(skills_texts, jd)
            df.loc[idx, "score_tfidf"] = tfidf

        if "bm25" in methods:
            bm25 = _bm25_scores(skills_texts, jd)
            df.loc[idx, "score_bm25"] = bm25

    for method in methods:
        score_col = f"score_{method}"
        rank_col = f"rank_{method}"
        pct_col = f"percentile_{method}"

        df[rank_col] = df.groupby("Job Role", observed=True)[score_col].rank(
            ascending=False, method="min",
        ).astype("Int64")

        df[pct_col] = df.groupby("Job Role", observed=True)[score_col].rank(
            pct=True, method="average",
        )

    logger.info(
        "Scoring complete — %d rows, methods=%s, roles=%s",
        len(df),
        methods,
        list(df["Job Role"].unique()),
    )

    return df
