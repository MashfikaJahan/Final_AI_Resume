"""Load and validate the raw Kaggle CSV."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_file_hash(path: str | Path) -> str:
    """SHA-256 hex digest of the file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_schema(df: pd.DataFrame, expected_columns: list[str]) -> None:
    """Raise ValueError if required columns are missing. Warn on extras."""
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    extra = set(df.columns) - set(expected_columns)
    if extra:
        logger.warning("Extra columns found (kept): %s", sorted(extra))


def load_raw(csv_path: str | Path, expected_columns: list[str] | None = None) -> pd.DataFrame:
    """Load CSV, validate schema, return raw DataFrame.

    Tries utf-8 first, falls back to utf-8-sig for BOM-prefixed files.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    file_hash = compute_file_hash(csv_path)
    file_size = csv_path.stat().st_size

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("utf-8 failed, retrying with utf-8-sig")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    logger.info(
        "Loaded %s — rows=%d, cols=%d, size=%d bytes, sha256=%s",
        csv_path.name, len(df), len(df.columns), file_size, file_hash,
    )

    if expected_columns:
        validate_schema(df, expected_columns)

    if df["Resume_ID"].duplicated().any():
        dupes = df.loc[df["Resume_ID"].duplicated(keep=False), "Resume_ID"].unique()
        raise ValueError(f"Duplicate Resume_IDs: {dupes.tolist()}")

    return df, file_hash, file_size
