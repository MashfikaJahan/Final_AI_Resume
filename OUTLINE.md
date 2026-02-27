# Project Outline

Aligned with [README.md](README.md). Canonical artifact chain: `resumes_clean.parquet` → `variants.parquet` → `variants_scored.csv` → `eval_summary/`

---

## Phase 1 — Diagnosis

### 1. Setup
- configs/ (YAML: seeds, methods, variant types, job roles)
- requirements.txt
- main.py (CLI entry point)

### 2. Data Ingestion
- data_loader.py: load CSV (utf-8), validate required columns, log row count / file size / checksum
- preprocessing.py: clean text, fill Certifications nulls with `""`, strip Skills, cast numerics
- Validation: Resume_ID unique; Skills non-empty; Job Role in allowlist; Recruiter Decision valid
- Output: `data/processed/resumes_clean.parquet` + run manifest

### 3. Variants
- variant_generator.py: control + phrasing / abbreviation expansion / word order / placement
- Output: `data/processed/variants.parquet`

### 4. Scoring
- scoring.py: TF-IDF cosine similarity, BM25, (optional) sentence embeddings
- Per-job-role ranking; scores + ranks + percentiles
- Output: `data/processed/variants_scored.csv`

### 5. Evaluation
- evaluation.py: paired comparison (control vs variant)
- Δ score, rank shift, top-K inclusion, threshold pass/fail
- Output: `data/processed/eval_summary/`

### 6. Analysis
- visualization.py: plots, distributions
- logistic_regression.py: predict Recruiter Decision from screening scores + resume features; scikit-learn LogisticRegression
- Output: `reports/` (metrics JSON, coefficients CSV, predictions CSV, summary MD, ROC/coefficients PNG)

---

## Phase 2 — Mitigation

- Evidence-based wording guidance (abbreviation expansion, dual phrasing, placement)
- Before/after rank improvements
- Optional: Streamlit app or report generator

---

## Project Structure (Reference)

```
Final_AI_Resume/
├── configs/
├── Dataset/          AI_Resume_Screening.csv
├── data/processed/  resumes_clean, variants, variants_scored, eval_summary
├── docs/             DATA_INGESTION_PLAN.md
├── outputs/          Timestamped results + figures
├── reports/          Logistic regression, mitigation guidance
└── src/              data_loader, preprocessing, variant_generator, scoring, evaluation, logistic_regression, visualization
```
