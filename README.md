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
│   ├── statistical_tests.py # Wilcoxon signed-rank + Cohen's d
│   ├── logistic_regression.py # Logistic regression (full + screening-only)
│   ├── visualization.py  # Publication-ready plots
│   └── report.py         # Paper-style Markdown report generator
├── main.py                 # CLI entry point for full pipeline runs
├── requirements.txt       # Pinned dependencies
├── RESULTS.md              # ← Auto-generated results (tables + figures, viewable on GitHub)
└── README.md
```

## Dataset

**Source:** [AI Resume Screening Dataset](https://www.kaggle.com/) (Kaggle) — `AI_Resume_Screening.csv`

| Column | Role in Study |
|--------|---------------|
| Resume_ID | Unique numeric identifier |
| Resume_Hash_ID | Deterministic anonymized identifier (added during preprocessing) |
| Skills | Primary text field for lexical-variation experiments |
| Education | Supplemental resume context |
| Certifications | Fill missing with empty string |
| Experience (Years) | Numeric context |
| Job Role | Segment experiments by role |
| Recruiter Decision | Downstream outcome label |
| AI Score (0–100) | Synthetic screening label (used cautiously) |

> **Note:** Place the CSV in `Dataset/` before running. The file is not committed to the repo due to size/licensing.

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

## Statistical Significance Tests

Paired Wilcoxon signed-rank tests with Cohen's d effect sizes are computed for every (variant type × Job Role × screening method) combination. This provides:

- **p-values** confirming whether observed score differences are statistically significant
- **Effect sizes** (Cohen's d) quantifying practical importance (negligible / small / medium / large)
- Output: `data/processed/eval_summary/statistical_tests.csv`

## Logistic Regression Analysis

In addition to stability metrics, the project includes **logistic regression** models that predict `Recruiter Decision` (Hire vs other) from screening outputs and basic resume features. Two models are fit:

1. **Full model** — all available features including `AI Score (0-100)`
2. **Screening-only model** — excludes `AI Score` to isolate the predictive power of ATS-like scoring features

This dual-model approach addresses the near-perfect AUC of the full model (driven by the synthetic `AI Score` variable) by providing a comparison model using only screening-derived features.

- **Input**: `data/processed/variants_scored.csv`
- **Model**: scikit-learn `LogisticRegression` with class weighting
- **Target**: `Recruiter Decision` (Hire = 1, other = 0)

Artifacts are written under `reports/` (with `_no_ai_score` suffix for the screening-only model):

- `logistic_regression_metrics[_no_ai_score].json` – metrics and dataset summary
- `logistic_regression_coefficients[_no_ai_score].csv` – feature coefficients
- `logistic_regression_predictions[_no_ai_score].csv` – per-variant predictions
- `logistic_regression_summary[_no_ai_score].md` – human-readable summary
- `logistic_regression_coefficients[_no_ai_score].png` – coefficient bar chart (300 DPI)
- `logistic_regression_roc[_no_ai_score].png` – ROC curve with AUC (300 DPI)

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

# Run full pipeline (Excel export disabled by default)
python main.py --config configs/default.yaml

# — or run individual stages —
python main.py --config configs/default.yaml --stage evaluate
python main.py --config configs/default.yaml --stage visualize

# Generate Markdown report (tables + figures viewable on GitHub)
python main.py --config configs/default.yaml --stage report
# → see RESULTS.md at the repo root

# View outputs directly (no Excel download needed)
#   RESULTS.md              ← tables + figures, renders on GitHub
#   data/processed/variants_scored.csv
#   data/processed/eval_summary/eval_summary.csv
#   outputs/figures/*.png
open outputs/figures   # macOS: opens Finder with all plots

# Optional: generate Excel workbook (set export.enabled: true in config first)
# python main.py --config configs/default.yaml --stage export
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
