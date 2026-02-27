# Pipeline Plan — Sentiment-Aware NLP Variant Generation & Scoring

**Project:** Same Skills, Different Words  
**Scope:** Full pipeline from raw data to scored variants (excludes final evaluation)  
**Predecessor:** `DATA_INGESTION_PLAN.md` (data loading & preprocessing)

---

## 0. Design Rationale

The current `variant_generator.py` uses **static lookup dictionaries** (`PHRASING_MAP`, `ABBREVIATION_MAP`) to swap keywords. This works but produces a single deterministic variant per type — no lexical diversity, no contextual awareness.

The upgraded pipeline replaces static maps with **NLP-driven synonym/paraphrase generation** so that:

1. Variants are **semantically equivalent but lexically distinct** — the same skill described with slightly different words.
2. The replacement engine understands context (e.g., "Python" in a data-science resume vs. "Python" in a web-dev resume can yield different phrasing).
3. Sentiment/tone of the original skill description is **measured and preserved** — a professional-sounding resume should not get a casual variant.

This directly supports RQ1 ("Do wording variations change screening scores?") with richer, more realistic variation than dictionary lookups alone.

---

## 1. Pipeline Architecture (End-to-End)

```
AI_Resume_Screening.csv
        │
        ▼
┌───────────────┐
│  1. Ingest    │  data_loader.py + preprocessing.py  (existing, unchanged)
│               │  → data/processed/resumes_clean.parquet
└───────┬───────┘
        │
        ▼
┌───────────────────────────────┐
│  2. Sentiment Baseline        │  NEW: src/sentiment.py
│     Compute per-resume tone   │  → adds sentiment_score, sentiment_label columns
│     on original skills text   │  → data/processed/resumes_clean.parquet (updated in-memory)
└───────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  3. NLP Variant Generation    │  UPGRADED: src/variant_generator.py
│     WordNet / contextual      │  Uses NLP to produce slight keyword variations
│     synonym replacement       │  Preserves sentiment envelope (±tolerance)
│                               │  → data/processed/variants.parquet
└───────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  4. Variant Sentiment QA      │  src/sentiment.py (reuse)
│     Re-score each variant     │  Attach sentiment_score_variant, sentiment_delta
│     Flag variants that drift  │  Reject/log variants outside tolerance
│     too far from original     │
└───────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  5. Scoring                   │  EXISTING: src/scoring.py (unchanged)
│     TF-IDF + BM25 vs JDs     │  → data/processed/variants_scored.csv
└───────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  6. Run Manifest              │  main.py (append per-stage entries)
└───────────────────────────────┘

NOTE: Evaluation (delta scores, rank shifts, top-K, threshold)
      is handled SEPARATELY — not part of this pipeline.
```

---

## 2. New Module: `src/sentiment.py`

### 2.1 Purpose

Compute a **sentiment / tone score** for resume skill text. This serves two roles:

1. **Baseline measurement** — quantify the tone of the original skills string before variant generation.
2. **Quality gate** — after variants are generated, re-score them and reject any whose sentiment drifts beyond a configurable tolerance (prevents generating variants that sound inappropriately casual, aggressive, etc.).

### 2.2 Approach: VADER + TextBlob Ensemble

| Component | Role | Why |
|-----------|------|-----|
| VADER (`vaderSentiment`) | Rule-based sentiment intensity | Fast, no model download, handles short technical text well, returns compound score [-1, +1] |
| TextBlob | Pattern-based polarity + subjectivity | Captures subjectivity dimension — useful for detecting when a variant sounds more "salesy" vs. neutral |
| Ensemble | Weighted average | `sentiment_score = 0.6 * vader_compound + 0.4 * textblob_polarity` (configurable) |

For this study, most resume skill text is **neutral** (compound ≈ 0). The sentiment module's main job is catching *deviations* — variants that accidentally introduce positive/negative tone shifts.

### 2.3 Function Signatures

```python
# src/sentiment.py

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

def label_sentiment(score: float) -> str:
    """Classify ensemble score into 'positive', 'negative', 'neutral'.

    Thresholds (configurable via config):
        score > +0.05  → 'positive'
        score < -0.05  → 'negative'
        else           → 'neutral'
    """

def score_dataframe(
    df: pd.DataFrame,
    text_column: str,
    prefix: str = "",
    config: dict | None = None,
) -> pd.DataFrame:
    """Add sentiment columns to df for the given text column.

    Adds columns:
        {prefix}sentiment_score
        {prefix}sentiment_label
        {prefix}vader_compound
        {prefix}textblob_polarity
        {prefix}textblob_subjectivity
    """

def check_sentiment_drift(
    df: pd.DataFrame,
    tolerance: float = 0.15,
) -> pd.DataFrame:
    """Compare variant sentiment to original sentiment.

    Adds:
        sentiment_delta = sentiment_score_variant - sentiment_score_original
        sentiment_drift_flag = abs(sentiment_delta) > tolerance

    Returns the DataFrame with new columns.
    """
```

### 2.4 Sentiment Tolerance

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sentiment.tolerance` | 0.15 | Max absolute sentiment drift allowed between original and variant |
| `sentiment.action_on_drift` | `"flag"` | `"flag"` = keep variant but mark it; `"reject"` = drop the variant row |
| `sentiment.vader_weight` | 0.6 | Weight of VADER compound in ensemble |
| `sentiment.textblob_weight` | 0.4 | Weight of TextBlob polarity in ensemble |
| `sentiment.positive_threshold` | 0.05 | Score above this → "positive" label |
| `sentiment.negative_threshold` | -0.05 | Score below this → "negative" label |

---

## 3. Upgraded Variant Generation: NLP-Driven Synonym Replacement

### 3.1 Current State (Static Maps)

The existing `variant_generator.py` uses two hardcoded dicts:

- `PHRASING_MAP`: `"Python" → "Python programming"`
- `ABBREVIATION_MAP`: `"NLP" → "Natural Language Processing"`

These produce exactly **one** variant per type per resume — no diversity, no context sensitivity.

### 3.2 New Strategy: Multi-Source NLP Replacement

The upgraded generator keeps the four variant types (`phrasing`, `abbreviation`, `word_order`, `placement`) but replaces static lookups with NLP-backed methods for `phrasing` and `abbreviation`.

#### 3.2.1 Synonym/Paraphrase Sources

| Source | Method | Use Case | Fallback Priority |
|--------|--------|----------|-------------------|
| **WordNet** (NLTK) | `wordnet.synsets()` → lemma names | Core synonym generation for general terms ("programming" → "coding", "development") | Primary |
| **Domain thesaurus** | Curated dict of tech-specific synonyms not in WordNet | Handles tech terms WordNet doesn't cover well ("TensorFlow" → "TF framework", "React" → "React.js") | Secondary (fallback when WordNet has no useful synsets) |
| **Contextual paraphrase** | spaCy token similarity + word vectors | Rank candidate synonyms by contextual fit within the skill list | Ranking layer on top of WordNet/thesaurus candidates |

**Why not a generative LLM?** Reproducibility. WordNet + curated thesaurus is deterministic given a seed. LLM outputs vary across runs and API versions, breaking the study's reproducibility requirement.

#### 3.2.2 NLP Replacement Algorithm

```
For each resume:
    For each skill token in skills_list:
        1. Look up WordNet synsets for the token
           - Filter to same POS (noun/verb as appropriate)
           - Extract lemma names, exclude the original token
           - Rank candidates by edit distance (prefer slight variations)

        2. If WordNet yields ≥1 candidate:
           - Use spaCy word vectors to score each candidate
             for contextual similarity to the full skills_list
           - Select top candidate above similarity threshold (default 0.6)

        3. If WordNet yields 0 candidates (common for tech terms):
           - Fall back to domain thesaurus
           - If thesaurus has entry → use it
           - If no entry → keep original token (no forced replacement)

        4. Sentiment check:
           - Compute sentiment of candidate replacement in context
           - If |drift| > tolerance → discard candidate, try next-best
           - If all candidates fail sentiment check → keep original

    Assemble variant string from replaced tokens
```

#### 3.2.3 Domain Thesaurus (Expanded)

The thesaurus replaces the current flat `PHRASING_MAP` and `ABBREVIATION_MAP` with a richer structure that provides **multiple candidates** per term:

```python
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
```

#### 3.2.4 Candidate Selection Logic (Seeded)

```python
def _select_candidate(
    candidates: list[str],
    skills_context: list[str],
    rng: random.Random,
    nlp: spacy.Language,
    top_n: int = 3,
) -> str:
    """Pick the best synonym candidate with controlled randomness.

    1. Score each candidate using spaCy vector similarity to the
       full skills context string.
    2. Keep top_n candidates above the similarity floor.
    3. Randomly select one of the top_n (seeded RNG for reproducibility).
    """
```

This gives us **slight variation across runs with different seeds** while keeping results reproducible for any given seed.

### 3.3 Updated Variant Type Implementations

| Type | Current Behavior | New Behavior |
|------|-----------------|-------------|
| `phrasing` | Static 1:1 map lookup | WordNet synonyms + domain thesaurus → spaCy-ranked → seeded selection from top-N candidates |
| `abbreviation` | Static 1:1 map lookup | Domain thesaurus multi-candidate → seeded selection; includes dual-phrasing patterns like `"ML (Machine Learning)"` |
| `word_order` | `rng.shuffle()` on token list | **Unchanged** — shuffling is already NLP-free and correct |
| `placement` | Wrap in `"Key Skills: {skills}. Proficient in all listed areas."` | **Minor upgrade**: multiple placement templates, seeded selection. Templates vary sentence structure while preserving meaning |

#### 3.3.1 Placement Templates (Expanded)

```python
PLACEMENT_TEMPLATES = [
    "Key Skills: {skills}. Proficient in all listed areas.",
    "Technical Proficiencies: {skills}.",
    "Core Competencies: {skills}. Demonstrated expertise in each area.",
    "Skills & Expertise: {skills}.",
    "Areas of Expertise — {skills}.",
]
```

### 3.4 Variant Generation Parameters

| Config Key | Default | Description |
|------------|---------|-------------|
| `variants.types` | `["phrasing", "abbreviation", "word_order", "placement"]` | Which variant types to generate |
| `variants.per_resume` | 4 | Number of variants per resume (one per type currently) |
| `variants.nlp_model` | `"en_core_web_md"` | spaCy model for word vectors (md has 300d vectors, sufficient for similarity) |
| `variants.wordnet_pos_filter` | `true` | Filter WordNet synsets by POS tag |
| `variants.similarity_floor` | 0.6 | Minimum spaCy vector similarity for a synonym candidate to be considered |
| `variants.top_n_candidates` | 3 | How many top candidates to keep before seeded random selection |
| `variants.max_replacements_per_skill` | 1 | Max tokens replaced per individual skill token (prevents "Python programming" → "Py coding dev") |

---

## 4. Pipeline Stage Specifications

### Stage 1: Ingest (Existing — No Changes)

| Property | Value |
|----------|-------|
| Module | `src/data_loader.py`, `src/preprocessing.py` |
| Input | `Dataset/AI_Resume_Screening.csv` |
| Output | `data/processed/resumes_clean.parquet` |
| CLI | `python main.py --config configs/default.yaml --stage ingest` |

Covered fully by `DATA_INGESTION_PLAN.md`. No modifications needed.

---

### Stage 2: Sentiment Baseline (NEW)

| Property | Value |
|----------|-------|
| Module | `src/sentiment.py` |
| Input | `data/processed/resumes_clean.parquet` (in-memory after ingest or loaded from disk) |
| Output | DataFrame with added columns; not persisted separately (carried into variant generation) |
| CLI | Runs as part of `--stage variants` or standalone `--stage sentiment` |

#### 2.1 Processing Steps

```
1. Load resumes_clean.parquet
2. For each row:
   a. Run compute_sentiment(row["Skills"]) → VADER + TextBlob scores
   b. Compute ensemble sentiment_score
   c. Assign sentiment_label (positive / negative / neutral)
3. Attach columns to DataFrame:
   - sentiment_score (float)
   - sentiment_label (str)
   - vader_compound (float)
   - textblob_polarity (float)
   - textblob_subjectivity (float)
4. Log distribution: count of positive / negative / neutral
5. Return augmented DataFrame
```

#### 2.2 Expected Baseline Distribution

Given the dataset contains clinical skill lists like `"Python, TensorFlow, NLP, Pytorch"`, the expected sentiment profile is:

| Label | Expected % | Rationale |
|-------|-----------|-----------|
| Neutral | ~95% | Bare skill lists carry no sentiment |
| Positive | ~5% | Some phrasing-variant skills like "expertise" nudge slightly positive |
| Negative | ~0% | No negative terms in standard skill tokens |

This baseline is important because it establishes the **control sentiment envelope** that variants must stay within.

---

### Stage 3: NLP Variant Generation (UPGRADED)

| Property | Value |
|----------|-------|
| Module | `src/variant_generator.py` (rewrite) |
| Input | Augmented DataFrame from Stage 2 (resumes + sentiment baseline) |
| Output | `data/processed/variants.parquet` |
| CLI | `python main.py --config configs/default.yaml --stage variants` |

#### 3.1 Processing Steps

```
1. Load spaCy model (en_core_web_md) for word vectors
2. Load WordNet corpus (nltk.download('wordnet', 'omw-1.4'))
3. For each resume row:
   a. Extract skills_list (already computed in preprocessing)
   b. Retrieve sentiment_score as the baseline envelope

   c. For variant_type in config["variants"]["types"]:
      i.   If phrasing:
           - For each skill token:
             • Query WordNet for synonyms (filtered by POS)
             • Merge with domain thesaurus phrasing candidates
             • Rank by spaCy vector similarity to skill context
             • Select top candidate (seeded RNG)
             • Sentiment-check the candidate in context
             • If drift > tolerance → try next candidate or keep original
           - Assemble variant string

      ii.  If abbreviation:
           - For each skill token:
             • Look up domain thesaurus abbreviation candidates
             • Select one (seeded RNG)
           - Assemble variant string

      iii. If word_order:
           - Shuffle skills_list (seeded RNG)
           - Assemble variant string

      iv.  If placement:
           - Select a placement template (seeded RNG)
           - Format with original skills
           - Assemble variant string

   d. Create record: {Resume_ID, variant_type, skills_original,
                       skills_variant, ...all original columns}

4. Include control row (variant_type="control", skills_variant=skills_original)
5. Build variants DataFrame (long format)
6. Write to data/processed/variants.parquet
```

#### 3.2 Output Schema

| Column | Type | Description |
|--------|------|-------------|
| Resume_ID | int | From original |
| variant_type | str | `"control"`, `"phrasing"`, `"abbreviation"`, `"word_order"`, `"placement"` |
| skills_original | str | Original comma-separated skills |
| skills_variant | str | Transformed skills text |
| sentiment_score | float | Baseline sentiment of original skills |
| sentiment_label | str | Baseline label |
| *...all other original columns* | mixed | Carried forward unchanged |

#### 3.3 Expected Row Counts

| Resumes | Variant Types | Total Rows |
|---------|--------------|------------|
| 1,000 | 4 types + 1 control = 5 | 5,000 |

---

### Stage 4: Variant Sentiment QA (NEW)

| Property | Value |
|----------|-------|
| Module | `src/sentiment.py` (reused) |
| Input | `data/processed/variants.parquet` (in-memory, before scoring) |
| Output | Augmented variants DataFrame with sentiment QA columns |
| CLI | Integrated into `--stage variants` (runs automatically after generation) |

#### 4.1 Processing Steps

```
1. For each variant row (excluding control):
   a. Compute sentiment of skills_variant text
   b. Calculate sentiment_delta = variant_score - original_score
   c. Flag if abs(sentiment_delta) > config.sentiment.tolerance

2. Add columns:
   - sentiment_score_variant (float)
   - sentiment_label_variant (str)
   - sentiment_delta (float)
   - sentiment_drift_flag (bool)

3. Log summary:
   - Total variants flagged for drift
   - Breakdown by variant_type
   - Mean/max drift per type

4. If config.sentiment.action_on_drift == "reject":
   - Drop flagged rows
   - Log dropped count to dropped_rows.log

5. Re-write variants.parquet with QA columns included
```

#### 4.2 Sentiment QA Output Columns

| Column | Type | Added To |
|--------|------|----------|
| `sentiment_score_variant` | float | All variant rows |
| `sentiment_label_variant` | str | All variant rows |
| `sentiment_delta` | float | All variant rows (0.0 for control) |
| `sentiment_drift_flag` | bool | All variant rows (False for control) |

---

### Stage 5: Scoring (Existing — No Changes)

| Property | Value |
|----------|-------|
| Module | `src/scoring.py` |
| Input | `data/processed/variants.parquet` |
| Output | `data/processed/variants_scored.csv` |
| CLI | `python main.py --config configs/default.yaml --stage score` |

Existing TF-IDF + BM25 scoring against job descriptions. No modifications needed — scoring consumes `skills_variant` column regardless of how it was generated.

Adds columns: `score_tfidf`, `score_bm25`, `rank_tfidf`, `rank_bm25`, `percentile_tfidf`, `percentile_bm25`.

---

### Stage 6: Run Manifest (Existing — Minor Extension)

Each stage already writes to `outputs/run_manifest.jsonl`. The new stages will add:

```json
{
  "stage": "sentiment_baseline",
  "neutral_count": 950,
  "positive_count": 45,
  "negative_count": 5
}
```

```json
{
  "stage": "variant_generation",
  "nlp_model": "en_core_web_md",
  "wordnet_candidates_found": 3200,
  "thesaurus_fallbacks": 800,
  "kept_original_count": 200,
  "sentiment_drift_flagged": 12,
  "sentiment_drift_rejected": 0
}
```

---

## 5. Config Additions (`configs/default.yaml`)

```yaml
sentiment:
  vader_weight: 0.6
  textblob_weight: 0.4
  positive_threshold: 0.05
  negative_threshold: -0.05
  tolerance: 0.15
  action_on_drift: "flag"   # "flag" or "reject"

variants:
  types:
    - phrasing
    - abbreviation
    - word_order
    - placement
  per_resume: 4
  nlp_model: "en_core_web_md"
  wordnet_pos_filter: true
  similarity_floor: 0.6
  top_n_candidates: 3
  max_replacements_per_skill: 1
```

---

## 6. New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `nltk` | ≥ 3.8 | WordNet synsets for synonym generation |
| `spacy` | ≥ 3.6 | Word vectors for contextual similarity ranking |
| `en_core_web_md` | ≥ 3.6 | spaCy medium English model (300d vectors) |
| `vaderSentiment` | ≥ 3.3.2 | Rule-based sentiment scoring |
| `textblob` | ≥ 0.18 | Pattern-based polarity + subjectivity |

Additions to `requirements.txt`:

```
nltk>=3.8
spacy>=3.6
vaderSentiment>=3.3.2
textblob>=0.18
```

Post-install steps:

```bash
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
python -m textblob.download_corpora
```

---

## 7. Updated CLI Stages

| Stage | Command | Description |
|-------|---------|-------------|
| `ingest` | `python main.py --stage ingest` | Load & clean raw CSV → parquet |
| `sentiment` | `python main.py --stage sentiment` | Compute baseline sentiment (optional standalone) |
| `variants` | `python main.py --stage variants` | NLP variant generation + sentiment QA |
| `score` | `python main.py --stage score` | TF-IDF + BM25 scoring |
| `all` | `python main.py` | Run ingest → sentiment → variants → score (no evaluation) |

The `--stage all` default will run stages in order: `ingest → sentiment → variants → score`.

**Evaluation is NOT included** — it runs separately via `--stage evaluate`.

---

## 8. Module-Level Implementation Spec

### `src/sentiment.py` (NEW)

```python
def compute_sentiment(text: str) -> dict[str, float]:
    """VADER + TextBlob ensemble. Returns dict of score components."""

def label_sentiment(score: float, pos_thresh: float = 0.05, neg_thresh: float = -0.05) -> str:
    """Classify score → 'positive' / 'negative' / 'neutral'."""

def score_dataframe(df: pd.DataFrame, text_column: str, prefix: str = "", config: dict | None = None) -> pd.DataFrame:
    """Batch-score a DataFrame column, add sentiment columns."""

def check_sentiment_drift(df: pd.DataFrame, tolerance: float = 0.15) -> pd.DataFrame:
    """Compare variant vs original sentiment, flag drift."""
```

### `src/variant_generator.py` (UPGRADED)

```python
# Retained (modified)
DOMAIN_THESAURUS: dict[str, dict[str, list[str]]] = { ... }
PLACEMENT_TEMPLATES: list[str] = [ ... ]

# New helpers
def _get_wordnet_synonyms(token: str, pos_filter: bool = True) -> list[str]:
    """Extract synonym lemmas from WordNet synsets."""

def _rank_by_similarity(candidates: list[str], context: str, nlp: spacy.Language) -> list[tuple[str, float]]:
    """Score candidates by spaCy vector similarity to context. Returns sorted (candidate, score) pairs."""

def _select_candidate(candidates: list[tuple[str, float]], rng: random.Random, top_n: int = 3) -> str:
    """Pick from top-N candidates using seeded RNG."""

def _apply_phrasing_nlp(skills_list: list[str], rng: random.Random, nlp: spacy.Language, config: dict) -> list[str]:
    """NLP-driven phrasing replacement: WordNet → thesaurus fallback → similarity ranking → sentiment check."""

def _apply_abbreviation_nlp(skills_list: list[str], rng: random.Random, config: dict) -> list[str]:
    """Multi-candidate abbreviation swap from domain thesaurus."""

def _apply_word_order(skills_list: list[str], rng: random.Random) -> list[str]:
    """Shuffle token order (unchanged)."""

def _apply_placement(skills_list: list[str], rng: random.Random) -> list[str]:
    """Template-based placement with seeded template selection."""

def generate_variants(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Main entry point — generate all variant types for every resume."""
```

---

## 9. Data Flow Summary

```
resumes_clean.parquet (1,000 rows × 14 cols)
        │
        │ + sentiment columns (5 new cols)
        ▼
resumes_with_sentiment (1,000 rows × 19 cols)  [in-memory only]
        │
        │ × 5 (control + 4 variant types)
        ▼
variants.parquet (5,000 rows × 24 cols)
  includes: skills_original, skills_variant,
            sentiment_score, sentiment_label,
            sentiment_score_variant, sentiment_label_variant,
            sentiment_delta, sentiment_drift_flag
        │
        │ + scoring columns (6 new cols: score/rank/percentile × 2 methods)
        ▼
variants_scored.csv (5,000 rows × 30 cols)
```

---

## 10. Reproducibility Controls

| Control | Implementation |
|---------|----------------|
| **Seeded RNG** | `random.Random(seed)` passed to all variant functions; seed from config |
| **WordNet determinism** | Synset ordering is stable given NLTK version; candidates sorted alphabetically before similarity ranking |
| **spaCy determinism** | Word vectors are static in a given model version; similarity scores are deterministic |
| **Sentiment determinism** | VADER and TextBlob are both rule-based / pattern-based — no stochastic components |
| **Config snapshot** | Full resolved config written to run manifest |
| **Git commit hash** | Captured in manifest |
| **Dependency versions** | Pinned in requirements.txt; NLTK/spaCy model versions logged |

---

## 11. Testing Checklist

| Test | Type | What to Verify |
|------|------|----------------|
| WordNet returns synonyms for "programming" | Unit | ≥ 1 synonym returned, original excluded |
| WordNet returns empty for "TensorFlow" | Unit | Falls back to domain thesaurus |
| Domain thesaurus returns multiple candidates | Unit | e.g., "Python" → 3 phrasing candidates |
| spaCy similarity ranking is deterministic | Unit | Same input → same ranking across calls |
| Seeded RNG produces same variant across runs | Unit | Two runs with seed=42 → identical output |
| Sentiment of "Python, SQL, Java" ≈ 0.0 | Unit | Neutral skill list scores near zero |
| Sentiment drift flagging works | Unit | Inject a drifted variant, assert flag=True |
| Drift rejection drops rows when configured | Unit | Set action="reject", assert row count reduced |
| Control row has sentiment_delta = 0.0 | Unit | Control variant should have zero drift |
| Full pipeline: clean → sentiment → variants → score | Integration | 5,000 rows out, all columns present |
| Variants parquet is reproducible | Integration | Same config+seed → identical parquet hash |
| Run manifest captures NLP metadata | Integration | Check manifest for nlp_model, candidate counts |

---

## 12. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| WordNet coverage gap for tech terms | Many skills fall back to thesaurus | Domain thesaurus covers all 14 known skill tokens; log fallback rate |
| spaCy model download (93 MB) adds setup friction | First-time setup takes longer | Document in README; add `postinstall.sh` script |
| VADER/TextBlob disagree on sentiment | Ensemble score could be misleading | Log both individual scores; weight is configurable; validate on known-neutral skill lists |
| New NLP deps increase pipeline runtime | Variant generation slower | spaCy model loaded once; WordNet lookups are fast; sentiment scoring is <1ms/text |
| Sentiment tolerance too strict | Many valid variants rejected | Default tolerance (0.15) is generous for neutral skill text; tune based on baseline distribution |
| NLTK data download fails in CI | Pipeline breaks without manual download | Add `nltk.download()` calls in module init with `quiet=True`; check before use |
| Variant diversity still limited (14 unique skills) | WordNet can't produce infinitely many variations | Acceptable — study uses controlled dataset; diversity comes from multiple variant types, not infinite synonyms |

---

## 13. Implementation Order

| Step | Task | Depends On |
|------|------|-----------|
| 1 | Add new dependencies to `requirements.txt` | — |
| 2 | Create `src/sentiment.py` with VADER + TextBlob ensemble | Step 1 |
| 3 | Write unit tests for sentiment scoring | Step 2 |
| 4 | Expand domain thesaurus in `variant_generator.py` | — |
| 5 | Implement WordNet synonym lookup + spaCy ranking | Step 1 |
| 6 | Integrate NLP replacement into `_apply_phrasing` | Steps 4, 5 |
| 7 | Upgrade `_apply_abbreviation` for multi-candidate selection | Step 4 |
| 8 | Add placement template expansion | — |
| 9 | Wire sentiment QA into variant generation pipeline | Steps 2, 6 |
| 10 | Update `main.py` to include sentiment stage | Steps 2, 9 |
| 11 | Update `configs/default.yaml` with new keys | — |
| 12 | Integration test: full pipeline ingest → score | Steps 1–11 |
| 13 | Update `README.md` with new dependencies and setup steps | Step 12 |
