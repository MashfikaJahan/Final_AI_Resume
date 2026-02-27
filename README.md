# Same Skills, Different Words

**A Comparative Quasi-Experimental Analysis of Lexical Variation Effects in Automated Resume Screening**

---

## Overview

This project investigates whether **small wording differences in resumes** (e.g., "Python" vs. "Python programming," "ML" vs. "Machine Learning") can change **automated resume screening outcomes**—specifically matching scores, rankings, and shortlist inclusion. The study uses a controlled quasi-experimental design: hold the resume constant, change only the wording ("treatment"), and measure how screening outputs change ("outcomes").

The pipeline implements ATS-like screening methods (TF-IDF, BM25, and optional embedding similarity) against a public Kaggle dataset to produce reproducible, quantifiable evidence of wording effects.

## Research Questions

| ID  | Question |
|-----|----------|
| RQ1 | Do wording variations for the same skills significantly change automated screening scores and rankings? |
| RQ2 | Which variation types (phrasing, abbreviation expansion, word order, placement) produce the largest rank shifts? |
| RQ3 | Are wording effects consistent across job roles and screening methods (lexical vs. embedding)? |

## Two-Phase Design

### Phase 1 — Diagnosis

Demonstrate and quantify that lexical variation alone changes screening outputs.

- Generate controlled resume variants (same resume, different wording)
- Score and rank variants across job roles and screening methods
- Report effect sizes: delta scores, rank shifts, top-K inclusion, threshold pass/fail
- Identify which variation types matter most

### Phase 2 — Mitigation

Develop evidence-based guidance (and optionally a lightweight tool) to reduce "wording penalties" without misrepresentation.

- Abbreviation expansion: "Machine Learning (ML)"
- Dual phrasing: "Python (Python programming)"
- Placement strategies: dedicated Skills section with key terms
- Before/after evaluation of rank improvements

## Project Structure

```
Final_AI_Resume/
├── configs/                # Run settings (roles, variants, methods, seeds)
├── Dataset/                # Input CSV (AI_Resume_Screening.csv)
├── data/                   # Processed outputs (parquet, stats, dropped log)
│   └── processed/          # resumes_clean.parquet, variants.parquet, variants_scored.csv, eval_summary
├── docs/                   # Documentation (DATA_INGESTION_PLAN.md)
├── outputs/                # Results tables + figures (timestamped)
├── reports/                # Summary outputs (Markdown / HTML / PDF, logistic regression reports)
├── src/                    # Python modules
│   ├── __init__.py
│   ├── data_loader.py      # Load and validate the Kaggle dataset
│   ├── preprocessing.py   # Clean text, handle missing values
│   ├── variant_generator.py # Controlled lexical swaps
│   ├── scoring.py         # TF-IDF, BM25, embedding similarity
│   ├── evaluation.py      # Δ score, rank shift, top-K, threshold
│   ├── logistic_regression.py # Logistic regression on Recruiter Decision
│   └── visualization.py  # Plots and tables
├── main.py                 # CLI entry point for full pipeline runs
├── requirements.txt       # Pinned dependencies
└── README.md
```

## Dataset

**Source:** [AI Resume Screening Dataset](https://www.kaggle.com/) (Kaggle) — `AI_Resume_Screening.csv`

| Column | Role in Study |
|--------|---------------|
| Resume_ID | Unique identifier |
| Skills | Primary text field for lexical-variation experiments |
| Education | Supplemental resume context |
| Certifications | Fill missing with empty string |
| Experience (Years) | Numeric context |
| Job Role | Segment experiments by role |
| Recruiter Decision | Downstream outcome label |
| AI Score (0–100) | Synthetic screening label (used cautiously) |

> **Note:** Place the CSV in `Dataset/` before running. The file is not committed to the repo due to size/licensing.

## Data Ingestion Plan

**Canonical input:** `Dataset/AI_Resume_Screening.csv`  
**Canonical output:** `data/processed/resumes_clean.parquet`

### 1. Source & Acquisition

| Item | Detail |
|------|--------|
| **Source** | Kaggle: AI Resume Screening dataset |
| **Path** | `Dataset/AI_Resume_Screening.csv` |
| **License** | Kaggle dataset terms apply; cite source in publications |
| **Immutability** | Raw data is immutable; all transformations produce new artifacts |

### 2. Schema (Expected)

| Column | Type | Role |
|--------|------|------|
| Resume_ID | int64 | Primary key; must be unique |
| Name | object | Metadata (traceability) |
| Skills | object | **Treatment** — comma-separated; target for variant generation |
| Experience (Years) | int64 | Supplemental |
| Education | object | Supplemental (B.Sc, MBA, B.Tech, M.Tech, PhD) |
| Certifications | object | Supplemental (27% null → fill with `""`) |
| Job Role | object | **Stratification** — AI Researcher, Data Scientist, Cybersecurity Analyst, Software Engineer |
| Recruiter Decision | object | **Outcome** — Hire / Reject |
| Salary Expectation ($) | int64 | Supplemental |
| Projects Count | int64 | Supplemental |
| AI Score (0-100) | int64 | Supplemental |

**Current stats:** 1000 rows, 11 columns.

### 3. Ingestion Pipeline

| Step | Actions |
|------|---------|
| **Load** | Read CSV (utf-8); validate required columns exist; log row count, file size, checksum |
| **Validate** | Resume_ID uniqueness; Skills non-empty; Job Role in allowlist; Recruiter Decision ∈ {Hire, Reject}; drop all-null rows |
| **Preprocess** | Strip Skills whitespace; fill Certifications nulls with `""`; strip Name; cast numeric columns to int |
| **Derived (optional)** | `skills_list`, `skills_count`, `text_concat` (Skills + Education + Certifications) |
| **Output** | Write `resumes_clean.parquet`; run manifest (config, seed, input hash/size, output paths, timestamp) |

### 4. Validation Rules

| Check | Action on failure |
|-------|-------------------|
| Resume_ID unique | Log duplicates; drop keeping first |
| Skills non-empty | Log; exclude row (or raise if > threshold) |
| Job Role in allowlist | Log; map or exclude per config |
| Recruiter Decision valid | Exclude invalid rows |
| Encoding | Fail with clear error; no silent replace |

### 5. Edge Cases

| Case | Decision |
|------|----------|
| Skills = "None" literal | Treat as empty; exclude from variant generation |
| Special characters in Skills | Preserve; no aggressive normalization beyond strip |
| Typos (e.g. "Pytorch") | Keep in raw; variant generator handles expansions |
| Encoding issues | Fail fast |

### 6. Canonical Artifact Chain

`resumes_clean.parquet` → `variants.parquet` → `variants_scored.csv` → `eval_summary/`

Full detail: [docs/DATA_INGESTION_PLAN.md](docs/DATA_INGESTION_PLAN.md)

## Screening Methods

| Method | Type | Purpose |
|--------|------|---------|
| TF-IDF cosine similarity | Lexical | Closest proxy to ATS keyword matching |
| BM25 | Lexical | Probabilistic relevance ranking |
| Sentence embeddings (optional) | Semantic | Tests whether semantic methods reduce wording penalties |

## Evaluation Metrics

| Metric | What It Answers |
|--------|-----------------|
| Δ Score | Did wording change the match score? |
| Rank Shift | Did the resume move up or down in ranking? |
| Top-K Inclusion | Did it enter the top 10 / 25 / 50? |
| Threshold Pass/Fail | Did it meet a minimum match cutoff? |

All comparisons are **paired** (control vs. variant) under identical job-role / JD conditions.

## Logistic Regression Analysis

In addition to stability metrics, the project includes a **logistic regression** model that predicts `Recruiter Decision` (Hire vs other) from screening outputs and basic resume features. This analysis helps quantify which features (screening scores, ranks, experience, etc.) are most predictive of hiring decisions and provides interpretable coefficients for understanding the relationship between automated screening metrics and recruiter outcomes.

- **Input**: `data/processed/variants_scored.csv` (produced by `python main.py --config configs/default.yaml`)
- **Model**: scikit-learn `LogisticRegression` with class weighting
- **Features (if present)**:
  - `AI Score (0-100)`
  - `Experience (Years)`
  - `Projects Count`
  - `Salary Expectation ($)`
  - `score_tfidf`, `score_bm25`, `score_embedding`
  - `rank_tfidf`, `percentile_tfidf`, `rank_bm25`, `percentile_bm25`
- **Target**: `Recruiter Decision` (mapped to 1 for `Hire`, 0 for any other value)

Run the analysis:

```bash
python -m src.logistic_regression
```

Artifacts are written under `reports/`:

- `logistic_regression_metrics.json` – metrics and dataset summary
- `logistic_regression_coefficients.csv` – feature coefficients (interpretability)
- `logistic_regression_predictions.csv` – per-variant predictions and probabilities
- `logistic_regression_summary.md` – human-readable summary for the research report
- `logistic_regression_coefficients.png` – horizontal bar chart of feature coefficients (publication-ready, 300 DPI)
- `logistic_regression_roc.png` – ROC curve with AUC score (publication-ready, 300 DPI)

## Quickstart

```bash
# Clone
git clone <your-repo-url>
cd Final_AI_Resume

# Environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# Place dataset
# Download AI_Resume_Screening.csv from Kaggle and put it in Dataset/

# Run pipeline
python main.py --config configs/default.yaml
# Results: data/processed/*.parquet and data/processed/*.csv (resumes_clean, variants, variants_scored, eval_summary)
```

## Reproducibility

- All random operations seeded via config
- Outputs timestamped and saved to `outputs/`
- Config files tracked in `configs/` for exact replication

## License

This project is for academic research purposes. Dataset usage subject to Kaggle dataset license terms.

## Author

**Mashfika Jahan**
Minnesota State University, Mankato
