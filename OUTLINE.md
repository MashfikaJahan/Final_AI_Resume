# Project Outline

## 1. Setup
- configs/ (YAML: seeds, methods, variant types, job roles)
- requirements.txt
- main.py entry point

## 2. Data
- data_loader → load and validate CSV
- preprocessing → clean text, handle missing values
- Output: `resumes_clean.parquet`

## 3. Variants
- variant_generator → control + phrasing / abbreviation / word order / placement
- Output: `variants.parquet`

## 4. Scoring
- TF-IDF cosine similarity
- BM25
- (optional) embeddings
- Output: `variants_scored.csv` with ranks, percentiles

## 5. Evaluation
- Paired comparison: control vs variant
- Δ score, rank shift, top-K inclusion, threshold pass/fail
- Output: `eval_summary`

## 6. Analysis
- visualization (plots, distributions)
- logistic regression on Recruiter Decision

## 7. Mitigation
- Evidence-based wording guidance
- Before/after rank improvements
- Optional: Streamlit app or report generator
