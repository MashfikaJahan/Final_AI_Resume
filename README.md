# Same Skills, Different Words

**A Comparative Quasi-Experimental Analysis of Lexical Variation Effects in Automated Resume Screening**

---

## Summary

Do small wording differences (e.g., "Python" vs "Python programming", "ML" vs "Machine Learning") change automated resume screening scores, rankings, and shortlist proxies? Quasi-experimental design: same resume, change wording only → measure outcomes via ATS-like pipelines (TF-IDF, BM25; optional embeddings). Phase 1 quantifies the effect; Phase 2 provides evidence-based mitigation guidance.

## Research Questions

| ID | Question |
|----|----------|
| RQ1 | Do wording variations for the same skills significantly change automated screening scores and rankings? |
| RQ2 | Which variation types (phrasing, abbreviation expansion, word order, placement) produce the largest rank shifts? |
| RQ3 | Are wording effects consistent across job roles and screening methods (lexical vs embedding)? |

## Two Phases

| Phase | Goal | Outputs |
|-------|-----|---------|
| **1. Diagnosis** | Demonstrate and quantify wording effects | Variant dataset; scores + ranks + Δ; summary stats + plots; which variation types matter most |
| **2. Mitigation** | Reduce wording penalties without misrepresentation | Evidence-based writing guidance; before/after rank improvements; optional Streamlit/report tool |

---

## Dataset

**Source:** Kaggle — AI Resume Screening  
**Path:** `Dataset/AI_Resume_Screening.csv` (1000 rows, 11 columns)

| Column | Type | Role |
|--------|------|------|
| `Resume_ID` | int | Identifier (primary key) |
| `Name` | str | Metadata |
| `Skills` | str | **Treatment** — comma-separated; target for variant generation |
| `Experience (Years)` | int | Supplemental |
| `Education` | str | Supplemental (B.Sc, MBA, B.Tech, M.Tech, PhD) |
| `Certifications` | str | Supplemental (27% null) |
| `Job Role` | str | **Stratification** — AI Researcher, Data Scientist, Cybersecurity Analyst, Software Engineer |
| `Recruiter Decision` | str | **Outcome** — Hire / Reject |
| `Salary Expectation ($)` | int | Supplemental |
| `Projects Count` | int | Supplemental |
| `AI Score (0-100)` | int | Supplemental |

**Note:** Raw data is immutable. All derived artifacts live in `data/processed/`.

---

## Data Ingestion Plan

Ingestion flow: **load → validate → preprocess → output**. See [docs/DATA_INGESTION_PLAN.md](docs/DATA_INGESTION_PLAN.md) for full detail.

| Step | Actions |
|------|---------|
| **Load** | Read CSV (utf-8); validate required columns; log row count, file size |
| **Validate** | Resume_ID uniqueness; Skills non-empty; Job Role in allowlist; Recruiter Decision ∈ {Hire, Reject} |
| **Preprocess** | Strip Skills whitespace; fill Certifications nulls with `""`; optional `skills_list`, `skills_count`, `text_concat` |
| **Output** | `data/processed/resumes_clean.parquet`; run manifest with config, seed, input hash/size, output paths |

**Canonical artifacts:** `resumes_clean.parquet` → `variants.parquet` → `variants_scored.csv` → `eval_summary/`

---

## Evaluation (Paired: control vs variant)

| Metric | What it answers |
|--------|-----------------|
| Δ score | Did wording change match score? |
| Rank shift | Did the resume move up/down? |
| Top-K inclusion | Did it enter top 10/25/50? |
| Threshold pass/fail | Did it meet minimum match? |

---

## Project Outline

| Phase | Tasks |
|-------|-------|
| Setup | configs/, requirements.txt, main.py |
| Data | data_loader → preprocessing → `resumes_clean.parquet` |
| Variants | variant_generator → control + phrasing/abbrev/order/placement → `variants.parquet` |
| Scoring | TF-IDF, BM25, (optional) embeddings → ranks → `variants_scored.csv` |
| Evaluation | Δ score, rank shift, top-K, threshold → `eval_summary` |
| Analysis | visualization, logistic regression |
| Mitigation | Guidance doc, before/after, optional tool |

---

## Structure

```
Final_AI_Resume/
├── Dataset/              AI_Resume_Screening.csv (raw; immutable)
├── configs/              YAML (roles, variants, methods, seeds)
├── data/processed/       resumes_clean, variants, variants_scored, eval_summary
├── docs/                 DATA_INGESTION_PLAN.md
├── outputs/              Figures, timestamped results, run manifests
├── reports/              Summary, logistic regression, mitigation guidance
└── src/                  data_loader, preprocessing, variant_generator, scoring, evaluation, logistic_regression, visualization
```

---

## Prerequisites

- Python 3.9+
- pip (or uv/poetry)

## Run

```bash
pip install -r requirements.txt
python main.py --config configs/default.yaml
python -m src.logistic_regression
```

Seeded, config-driven, timestamped outputs for reproducibility.

---

## Getting the Dataset

Download from [Kaggle: AI Resume Screening](https://www.kaggle.com/datasets/murthyjm/ai-resume-screening) and place `AI_Resume_Screening.csv` in `Dataset/`. Ensure you comply with Kaggle's dataset license and terms.

---

**Author:** Mashfika Jahan · Minnesota State University, Mankato  
**License:** Academic research. Dataset use subject to Kaggle license.
