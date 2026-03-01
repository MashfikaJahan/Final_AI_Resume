"""Clean, normalize, cast, derive, and validate the raw resume DataFrame."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_skills(series: pd.Series) -> pd.Series:
    """Strip, normalize whitespace, normalize comma separators."""
    s = series.astype(str).str.strip()
    s = s.apply(lambda x: re.sub(r"\s+", " ", x))
    s = s.apply(lambda x: re.sub(r"\s*,\s*", ", ", x))
    return s


def normalize_certifications(series: pd.Series) -> pd.Series:
    """Replace literal 'None' / NaN with '', strip."""
    s = series.fillna("").astype(str).str.strip()
    s = s.replace({"None": "", "nan": ""})
    return s


def _hash_id(resume_id: int | str) -> str:
    """Deterministic 12-char hex hash from Resume_ID for anonymized sharing."""
    return hashlib.sha256(str(resume_id).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Categorical validation
# ---------------------------------------------------------------------------

def validate_categorical(
    df: pd.DataFrame,
    col_name: str,
    allowlist: set[str],
    dropped_log: Path | None = None,
) -> pd.DataFrame:
    """Drop rows whose *col_name* value is not in *allowlist*. Log dropped rows."""
    df[col_name] = df[col_name].astype(str).str.strip()
    invalid_mask = ~df[col_name].isin(allowlist)
    n_invalid = invalid_mask.sum()

    if n_invalid:
        logger.warning(
            "%d rows with invalid %s values — dropping", n_invalid, col_name,
        )
        if dropped_log:
            _log_dropped(df.loc[invalid_mask], col_name, dropped_log)
        df = df.loc[~invalid_mask].copy()

    return df


# ---------------------------------------------------------------------------
# Numeric casting & range validation
# ---------------------------------------------------------------------------

def cast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns to proper dtypes with coercion + range checks."""
    num_specs: list[tuple[str, str, float | None, float | None]] = [
        ("Resume_ID", "int64", None, None),
        ("Experience (Years)", "int64", 0, 50),
        ("Salary Expectation ($)", "float64", 0.01, None),
        ("Projects Count", "int64", 0, None),
        ("AI Score (0-100)", "float64", 0, 100),
    ]

    for col, dtype, lo, hi in num_specs:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        coerced = df[col].isna().sum()
        if coerced:
            logger.warning("%s: %d values coerced to NaN — dropping those rows", col, coerced)
            df = df.dropna(subset=[col])

        if lo is not None:
            out_lo = (df[col] < lo).sum()
            if out_lo:
                logger.warning("%s: %d values below %.2f (kept)", col, out_lo, lo)
        if hi is not None:
            out_hi = (df[col] > hi).sum()
            if out_hi:
                logger.warning("%s: %d values above %.2f (kept)", col, out_hi, hi)

        if dtype.startswith("int"):
            df[col] = df[col].astype("Int64")
        else:
            df[col] = df[col].astype(dtype)

    return df


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add skills_list, skills_count, skills_lower."""
    df["skills_list"] = df["Skills"].str.split(", ")
    df["skills_count"] = df["skills_list"].apply(len)
    df["skills_lower"] = df["Skills"].str.lower()
    return df


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def run_quality_gate(df: pd.DataFrame, config: dict) -> None:
    """Final validation checks. Raises on failure."""
    checks: list[tuple[str, bool]] = [
        ("Resume_ID unique", df["Resume_ID"].is_unique),
        ("Resume_ID no nulls", df["Resume_ID"].notna().all()),
        ("Skills no empty", (df["Skills"].str.len() > 0).all()),
        ("Skills no nulls", df["Skills"].notna().all()),
        (
            "Job Role in allowlist",
            df["Job Role"].isin(config["validation"]["job_role_allowlist"]).all(),
        ),
        (
            "Recruiter Decision valid",
            df["Recruiter Decision"].isin(config["validation"]["recruiter_decision_values"]).all(),
        ),
        ("Experience no nulls", df["Experience (Years)"].notna().all()),
        (
            "Experience range",
            ((df["Experience (Years)"] >= 0) & (df["Experience (Years)"] <= 50)).all(),
        ),
        ("AI Score no nulls", df["AI Score (0-100)"].notna().all()),
        (
            "AI Score range",
            ((df["AI Score (0-100)"] >= 0) & (df["AI Score (0-100)"] <= 100)).all(),
        ),
        ("Projects Count no nulls", df["Projects Count"].notna().all()),
        ("Projects Count >= 0", (df["Projects Count"] >= 0).all()),
        ("Salary no nulls", df["Salary Expectation ($)"].notna().all()),
        ("Salary > 0", (df["Salary Expectation ($)"] > 0).all()),
        (
            "Row count >= min",
            len(df) >= config["validation"]["min_rows_after_clean"],
        ),
    ]

    failures = [name for name, ok in checks if not ok]
    if failures:
        raise ValueError(f"Quality gate FAILED on: {failures}")

    logger.info("Quality gate PASSED — all %d checks ok", len(checks))


# ---------------------------------------------------------------------------
# Dropped-row logger
# ---------------------------------------------------------------------------

def _log_dropped(rows: pd.DataFrame, reason: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a") as f:
        for _, row in rows.iterrows():
            f.write(f"{ts} | {row.get('Resume_ID', '?')} | {reason} | {row.to_dict()}\n")


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Full pipeline: clean -> cast -> derive -> validate -> return."""
    dropped_log = Path(config["data"]["dropped_log"])

    # --- Drop PII & add anonymized identifier ---
    df = df.drop(columns=["Name"], errors="ignore")
    df["Resume_Hash_ID"] = df["Resume_ID"].apply(_hash_id)

    # --- Text cleaning ---
    df["Skills"] = clean_skills(df["Skills"])
    df["Certifications"] = normalize_certifications(df["Certifications"])

    # Validate & strip Education
    edu_levels = set(config["validation"]["education_levels"])
    df["Education"] = df["Education"].astype(str).str.strip()
    unexpected_edu = set(df["Education"].unique()) - edu_levels
    if unexpected_edu:
        logger.warning("Unexpected Education values: %s", unexpected_edu)

    # Validate Job Role (drop invalid)
    df = validate_categorical(
        df, "Job Role", set(config["validation"]["job_role_allowlist"]), dropped_log,
    )

    # Validate Recruiter Decision (drop invalid)
    df = validate_categorical(
        df, "Recruiter Decision", set(config["validation"]["recruiter_decision_values"]), dropped_log,
    )

    # Flag empty Skills
    empty_skills = df["Skills"].str.strip().eq("").sum()
    if empty_skills:
        logger.warning("Dropping %d rows with empty Skills", empty_skills)
        df = df.loc[df["Skills"].str.strip().ne("")].copy()

    # Flag duplicate skill tokens within a row
    for idx, row in df.iterrows():
        tokens = [t.strip() for t in row["Skills"].split(",")]
        if len(tokens) != len(set(tokens)):
            logger.warning("Resume %s has duplicate skill tokens: %s", row["Resume_ID"], tokens)

    # --- Numeric casting ---
    df = cast_numerics(df)

    # --- Category dtypes ---
    df["Job Role"] = df["Job Role"].astype("category")
    df["Recruiter Decision"] = df["Recruiter Decision"].astype("category")
    df["Education"] = df["Education"].astype("category")
    df["Skills"] = df["Skills"].astype("string")

    # --- Derived columns ---
    df = add_derived_columns(df)

    # --- Quality gate ---
    run_quality_gate(df, config)

    dropped_count = 1000 - len(df)
    logger.info(
        "Preprocessing complete — %d rows retained, %d dropped", len(df), dropped_count,
    )
    return df
