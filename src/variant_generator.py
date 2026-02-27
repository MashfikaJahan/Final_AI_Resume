"""Generate controlled lexical variants of resume skill strings.

Variation types (per the study design):
  - phrasing:      Reword the skill ("Python" -> "Python programming")
  - abbreviation:  Expand or contract abbreviations ("NLP" -> "Natural Language Processing")
  - word_order:    Shuffle the order of skill tokens
  - placement:     Wrap skills in a contextual sentence / heading
"""

from __future__ import annotations

import logging
import random
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PHRASING_MAP: dict[str, str] = {
    "Python": "Python programming",
    "TensorFlow": "TensorFlow framework",
    "Pytorch": "PyTorch deep learning",
    "NLP": "NLP techniques",
    "Machine Learning": "machine learning methods",
    "Deep Learning": "deep learning techniques",
    "SQL": "SQL querying",
    "Java": "Java development",
    "C++": "C++ programming",
    "React": "React.js",
    "Ethical Hacking": "ethical hacking skills",
    "Cybersecurity": "cybersecurity expertise",
    "Linux": "Linux administration",
    "Networking": "network engineering",
}

ABBREVIATION_MAP: dict[str, str] = {
    "Python": "Python",
    "TensorFlow": "TF",
    "Pytorch": "PyTorch",
    "NLP": "Natural Language Processing",
    "Machine Learning": "ML",
    "Deep Learning": "DL",
    "SQL": "Structured Query Language",
    "Java": "Java",
    "C++": "C++",
    "React": "ReactJS",
    "Ethical Hacking": "Ethical Hacking",
    "Cybersecurity": "Cyber Security",
    "Linux": "GNU/Linux",
    "Networking": "Computer Networking",
}

PLACEMENT_TEMPLATE = "Key Skills: {skills}. Proficient in all listed areas."


def _apply_phrasing(skills_list: list[str], rng: random.Random) -> list[str]:
    """Replace each skill token with its phrasing variant (if mapped)."""
    return [PHRASING_MAP.get(s, s) for s in skills_list]


def _apply_abbreviation(skills_list: list[str], rng: random.Random) -> list[str]:
    """Expand or contract abbreviations for each skill token."""
    return [ABBREVIATION_MAP.get(s, s) for s in skills_list]


def _apply_word_order(skills_list: list[str], rng: random.Random) -> list[str]:
    """Shuffle skill token order (seeded)."""
    shuffled = skills_list.copy()
    rng.shuffle(shuffled)
    return shuffled


def _apply_placement(skills_list: list[str], rng: random.Random) -> list[str]:
    joined = ", ".join(skills_list)
    return [PLACEMENT_TEMPLATE.format(skills=joined)]


VARIANT_FNS = {
    "phrasing": _apply_phrasing,
    "abbreviation": _apply_abbreviation,
    "word_order": _apply_word_order,
    "placement": _apply_placement,
}


def generate_variants(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Generate controlled skill variants for every resume.

    Returns a long-format DataFrame with columns:
        Resume_ID, variant_type, skills_original, skills_variant,
        plus all original columns carried forward.
    """
    seed = config.get("seed", 42)
    variant_types: list[str] = config["variants"]["types"]
    rng = random.Random(seed)

    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        original_skills: str = row["Skills"]
        skills_tokens: list[str] = row["skills_list"]

        control_record = row.to_dict()
        control_record["variant_type"] = "control"
        control_record["skills_original"] = original_skills
        control_record["skills_variant"] = original_skills
        records.append(control_record)

        for vtype in variant_types:
            fn = VARIANT_FNS.get(vtype)
            if fn is None:
                logger.warning("Unknown variant type %r — skipping", vtype)
                continue

            transformed = fn(skills_tokens, rng)
            variant_str = ", ".join(transformed)

            rec = row.to_dict()
            rec["variant_type"] = vtype
            rec["skills_original"] = original_skills
            rec["skills_variant"] = variant_str
            records.append(rec)

    variants_df = pd.DataFrame(records)

    expected_rows = len(df) * (1 + len(variant_types))
    logger.info(
        "Generated %d variant rows (%d resumes × %d types + control). Expected %d.",
        len(variants_df),
        len(df),
        len(variant_types),
        expected_rows,
    )

    return variants_df
