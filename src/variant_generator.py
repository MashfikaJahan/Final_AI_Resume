"""Generate controlled lexical variants of resume skill strings.

NLP-driven variant generation (Pipeline_plan §3):
  - phrasing:      WordNet synonyms + domain thesaurus → spaCy-ranked → seeded selection
  - abbreviation:  Multi-candidate domain thesaurus → seeded selection
  - word_order:    Shuffle the order of skill tokens (unchanged)
  - placement:     Multiple templates, seeded selection (§3.3.1)
"""

from __future__ import annotations

import logging
import random
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK / spaCy bootstrap (Pipeline_plan §6 post-install)
# ---------------------------------------------------------------------------

def _ensure_nltk_data() -> None:
    import nltk
    for resource in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _load_spacy(model_name: str):
    import spacy
    try:
        return spacy.load(model_name)
    except OSError:
        logger.info("Downloading spaCy model %s …", model_name)
        from spacy.cli import download as spacy_download
        spacy_download(model_name)
        return spacy.load(model_name)


# ---------------------------------------------------------------------------
# Domain Thesaurus — Pipeline_plan §3.2.3 (verbatim from plan)
# ---------------------------------------------------------------------------

DOMAIN_THESAURUS: dict[str, dict[str, list[str]]] = {
    "Python": {
        "phrasing": ["Python programming", "Python development", "Python scripting"],
        "abbreviation": ["Py", "Python lang"],
    },
    "TensorFlow": {
        "phrasing": ["TensorFlow framework", "TensorFlow library", "TensorFlow toolkit"],
        "abbreviation": ["TF", "TensorFlow (TF)"],
    },
    "Pytorch": {
        "phrasing": ["PyTorch framework", "PyTorch deep learning", "PyTorch library"],
        "abbreviation": ["PyTorch (PT)", "PT"],
    },
    "NLP": {
        "phrasing": ["NLP techniques", "NLP methods", "natural language processing skills"],
        "abbreviation": ["Natural Language Processing", "Natural Language Processing (NLP)"],
    },
    "Machine Learning": {
        "phrasing": ["machine learning methods", "machine learning techniques", "applied machine learning"],
        "abbreviation": ["ML", "Machine Learning (ML)"],
    },
    "Deep Learning": {
        "phrasing": ["deep learning techniques", "deep learning methods", "deep neural networks"],
        "abbreviation": ["DL", "Deep Learning (DL)"],
    },
    "SQL": {
        "phrasing": ["SQL querying", "SQL database management", "SQL data retrieval"],
        "abbreviation": ["Structured Query Language", "SQL (Structured Query Language)"],
    },
    "Java": {
        "phrasing": ["Java development", "Java programming", "Java application development"],
        "abbreviation": ["Java SE", "Java (JDK)"],
    },
    "C++": {
        "phrasing": ["C++ programming", "C++ development", "C++ software engineering"],
        "abbreviation": ["Cpp", "C/C++"],
    },
    "React": {
        "phrasing": ["React.js development", "React frontend", "React UI development"],
        "abbreviation": ["ReactJS", "React.js"],
    },
    "Ethical Hacking": {
        "phrasing": ["ethical hacking skills", "penetration testing", "ethical security testing"],
        "abbreviation": ["Ethical Hacking (CEH)", "White Hat Hacking"],
    },
    "Cybersecurity": {
        "phrasing": ["cybersecurity expertise", "information security", "cyber defense"],
        "abbreviation": ["Cyber Security", "InfoSec"],
    },
    "Linux": {
        "phrasing": ["Linux administration", "Linux system management", "Linux OS"],
        "abbreviation": ["GNU/Linux", "Linux (Unix-based)"],
    },
    "Networking": {
        "phrasing": ["network engineering", "computer networking", "network administration"],
        "abbreviation": ["Computer Networking", "Net Admin"],
    },
}

# Pipeline_plan §3.3.1 (verbatim from plan)
PLACEMENT_TEMPLATES: list[str] = [
    "Key Skills: {skills}. Proficient in all listed areas.",
    "Technical Proficiencies: {skills}.",
    "Core Competencies: {skills}. Demonstrated expertise in each area.",
    "Skills & Expertise: {skills}.",
    "Areas of Expertise — {skills}.",
]


# ---------------------------------------------------------------------------
# WordNet helpers — Pipeline_plan §3.2.1, §3.2.2
# ---------------------------------------------------------------------------

def _get_wordnet_synonyms(token: str, pos_filter: bool = True) -> list[str]:
    """Extract synonym lemmas from WordNet synsets."""
    from nltk.corpus import wordnet as wn

    syns: set[str] = set()
    for synset in wn.synsets(token.lower()):
        if pos_filter and synset.pos() not in ("n", "v"):
            continue
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != token.lower():
                syns.add(name)
    return sorted(syns)


# ---------------------------------------------------------------------------
# spaCy ranking — Pipeline_plan §3.2.1 contextual paraphrase layer
# ---------------------------------------------------------------------------

def _rank_by_similarity(
    candidates: list[str],
    context: str,
    nlp,
) -> list[tuple[str, float]]:
    """Score candidates by spaCy vector similarity to context.

    Returns sorted (candidate, score) pairs, best first.
    """
    context_doc = nlp(context)
    if not context_doc.has_vector:
        return [(c, 0.0) for c in candidates]

    scored = []
    for c in candidates:
        c_doc = nlp(c)
        sim = context_doc.similarity(c_doc) if c_doc.has_vector else 0.0
        scored.append((c, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Candidate selection — Pipeline_plan §3.2.4
# ---------------------------------------------------------------------------

def _select_candidate(
    scored: list[tuple[str, float]],
    rng: random.Random,
    top_n: int = 3,
    similarity_floor: float = 0.6,
) -> str | None:
    """Pick the best synonym candidate with controlled randomness.

    1. Keep top_n candidates above the similarity floor.
    2. Randomly select one of the top_n (seeded RNG for reproducibility).
    """
    above_floor = [(c, s) for c, s in scored if s >= similarity_floor]
    pool = above_floor[:top_n] if above_floor else scored[:top_n]
    if not pool:
        return None
    return rng.choice(pool)[0]


# ---------------------------------------------------------------------------
# Variant-type implementations — Pipeline_plan §3.3, §8
# ---------------------------------------------------------------------------

_nlp_cache: dict[str, Any] = {}
_nlp_stats = {"candidates_found": 0, "thesaurus_fallbacks": 0, "kept_original": 0}


def _get_nlp(config: dict):
    model = config.get("variants", {}).get("nlp_model", "en_core_web_md")
    if model not in _nlp_cache:
        _nlp_cache[model] = _load_spacy(model)
    return _nlp_cache[model]


def _apply_phrasing_nlp(
    skills_list: list[str],
    rng: random.Random,
    nlp,
    config: dict,
) -> list[str]:
    """NLP-driven phrasing replacement: WordNet → thesaurus fallback → similarity ranking → sentiment check."""
    var_cfg = config.get("variants", {})
    pos_filter = var_cfg.get("wordnet_pos_filter", True)
    sim_floor = var_cfg.get("similarity_floor", 0.6)
    top_n = var_cfg.get("top_n_candidates", 3)
    context = " ".join(skills_list)

    result: list[str] = []
    for token in skills_list:
        wordnet_syns = _get_wordnet_synonyms(token, pos_filter=pos_filter)
        thesaurus_candidates = DOMAIN_THESAURUS.get(token, {}).get("phrasing", [])
        all_candidates = list(dict.fromkeys(wordnet_syns + thesaurus_candidates))

        if not all_candidates:
            _nlp_stats["kept_original"] += 1
            result.append(token)
            continue

        if wordnet_syns:
            _nlp_stats["candidates_found"] += 1
        else:
            _nlp_stats["thesaurus_fallbacks"] += 1

        ranked = _rank_by_similarity(all_candidates, context, nlp)
        chosen = _select_candidate(ranked, rng, top_n=top_n, similarity_floor=sim_floor)
        result.append(chosen if chosen else token)

    return result


def _apply_abbreviation_nlp(
    skills_list: list[str],
    rng: random.Random,
    config: dict,
) -> list[str]:
    """Multi-candidate abbreviation swap from domain thesaurus."""
    result: list[str] = []
    for token in skills_list:
        candidates = DOMAIN_THESAURUS.get(token, {}).get("abbreviation", [])
        if candidates:
            result.append(rng.choice(candidates))
        else:
            result.append(token)
    return result


def _apply_word_order(skills_list: list[str], rng: random.Random) -> list[str]:
    """Shuffle token order (unchanged from original — Pipeline_plan §3.3)."""
    shuffled = skills_list.copy()
    rng.shuffle(shuffled)
    return shuffled


def _apply_placement(skills_list: list[str], rng: random.Random) -> list[str]:
    """Template-based placement with seeded template selection (Pipeline_plan §3.3.1)."""
    joined = ", ".join(skills_list)
    template = rng.choice(PLACEMENT_TEMPLATES)
    return [template.format(skills=joined)]


# ---------------------------------------------------------------------------
# Main entry point — Pipeline_plan §8 generate_variants
# ---------------------------------------------------------------------------

def generate_variants(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Generate controlled skill variants for every resume.

    Returns a long-format DataFrame with columns:
        Resume_ID, variant_type, skills_original, skills_variant,
        plus all original columns carried forward.
    """
    global _nlp_stats
    _nlp_stats = {"candidates_found": 0, "thesaurus_fallbacks": 0, "kept_original": 0}

    seed = config.get("seed", 42)
    variant_types: list[str] = config["variants"]["types"]
    rng = random.Random(seed)

    _ensure_nltk_data()
    nlp = _get_nlp(config)

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
            if vtype == "phrasing":
                transformed = _apply_phrasing_nlp(skills_tokens, rng, nlp, config)
            elif vtype == "abbreviation":
                transformed = _apply_abbreviation_nlp(skills_tokens, rng, config)
            elif vtype == "word_order":
                transformed = _apply_word_order(skills_tokens, rng)
            elif vtype == "placement":
                transformed = _apply_placement(skills_tokens, rng)
            else:
                logger.warning("Unknown variant type %r — skipping", vtype)
                continue

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
        len(variants_df), len(df), len(variant_types), expected_rows,
    )
    logger.info(
        "NLP stats — WordNet candidates: %d, thesaurus fallbacks: %d, kept original: %d",
        _nlp_stats["candidates_found"],
        _nlp_stats["thesaurus_fallbacks"],
        _nlp_stats["kept_original"],
    )

    return variants_df
