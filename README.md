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

## Dataset

`Dataset/AI_Resume_Screening.csv` (Kaggle). Primary: `Skills` (treatment), `Job Role` (stratification), `Recruiter Decision` (outcome). Supplemental: Experience, Education, Certifications, AI Score (0–100), Salary, Projects Count.

## Evaluation (Paired: control vs variant)

| Metric | What it answers |
|--------|-----------------|
| Δ score | Did wording change match score? |
| Rank shift | Did the resume move up/down? |
| Top-K inclusion | Did it enter top 10/25/50? |
| Threshold pass/fail | Did it meet minimum match? |

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

## Structure

```
Final_AI_Resume/
├── Dataset/          AI_Resume_Screening.csv
├── configs/          YAML (roles, variants, methods, seeds)
├── data/processed/   resumes_clean, variants, variants_scored, eval_summary
├── outputs/          Figures, timestamped results
├── reports/          Summary, logistic regression, mitigation guidance
└── src/              data_loader, preprocessing, variant_generator, scoring, evaluation, logistic_regression, visualization
```

## Run

```bash
pip install -r requirements.txt
python main.py --config configs/default.yaml
python -m src.logistic_regression
```

Seeded, config-driven, timestamped outputs for reproducibility.

---

**Author:** Mashfika Jahan · Minnesota State University, Mankato  
**License:** Academic research. Dataset use subject to Kaggle license.
