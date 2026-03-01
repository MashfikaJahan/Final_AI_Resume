# Evaluation Plan — Common Eval for All Pipelines

**Project:** Same Skills, Different Words  
**Scope:** Single evaluation stage that consumes `variants_scored.csv` regardless of how variants were generated  
**Predecessors:** [Pipeline_plan.md](Pipeline_plan.md), [DATA_INGESTION_PLAN.md](DATA_INGESTION_PLAN.md)

---

## 1. Design Rationale

Evaluation is **pipeline-agnostic**: it does not care whether variants came from static maps, NLP-driven synonym replacement, or future methods. As long as the scoring stage outputs `variants_scored.csv` with the expected schema, the evaluation stage works unchanged.

This enables:

- Comparing outcomes across different variant-generation pipelines (e.g., static vs NLP)
- Adding new scoring methods (e.g., embeddings) without rewriting eval logic
- One eval implementation serving RQ1–RQ3 for all experiments

---

## 2. Input Contract

### 2.1 Required Schema (`variants_scored.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `Resume_ID` | int | Unique resume identifier |
| `Job Role` | str | Role used for scoring (determines JD) |
| `variant_type` | str | `"control"` or one of: `phrasing`, `abbreviation`, `word_order`, `placement` |
| `skills_variant` | str | Text that was scored (control = original skills) |
| `score_{method}` | float | Match score per method (e.g. `score_tfidf`, `score_bm25`) |
| `rank_{method}` | int | Within-role rank (1 = best) |
| `percentile_{method}` | float | Within-role percentile |

**Methods** come from `config.scoring.methods` (currently `tfidf`, `bm25`; embedding optional).

**Invariant:** Exactly one `variant_type="control"` row per (Resume_ID, Job Role). All other variant types have one row each per (Resume_ID, Job Role).

### 2.2 Optional Columns (Pass-Through)

Sentiment columns (`sentiment_score`, `sentiment_delta`, etc.) from the NLP pipeline are passed through to `eval_detail.csv` but not used in metric computation.

---

## 3. Evaluation Logic

### 3.1 Paired Comparison

Every variant row is compared to its **control** (same `Resume_ID`, same `Job Role`, `variant_type="control"`). Control rows are excluded from the detail output (only used for merging).

### 3.2 Core Metrics

| Metric | Formula | RQ |
|--------|---------|----|
| **Delta Score** | `score_{method} - score_{method}_control` | RQ1: Did wording change the match score? |
| **Rank Shift** | `rank_{method}_control - rank_{method}` (positive = moved up) | RQ1: Did ranking change? |
| **Top-K Flip** | `(rank <= k)_variant != (rank <= k)_control` | RQ1: Did top-K shortlist inclusion change? |
| **Threshold Flip** | `(score >= θ)_variant != (score >= θ)_control` | RQ1: Did pass/fail at cutoff change? |

### 3.3 Aggregation (Summary)

Group by `(variant_type, Job Role)`. For each metric:

| Aggregation | Delta Score | Rank Shift | Top-K Flip | Threshold Flip |
|-------------|-------------|------------|------------|----------------|
| Mean | ✓ | ✓ | ✓ (as proportion) | ✓ |
| Std | ✓ | ✓ | — | — |
| Min | ✓ | ✓ | — | — |
| Max | ✓ | ✓ | — | — |
| Sum | — | — | ✓ (count) | ✓ (count) |

---

## 4. Config Contract

Evaluation reads from `config`:

```yaml
scoring:
  methods: [tfidf, bm25]
  top_k: [10, 25, 50]
  threshold: 0.5
data:
  eval_summary_dir: "data/processed/eval_summary"
```

No eval-specific config keys required beyond what already exists.

---

## 5. Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| `eval_detail.csv` | `data/processed/eval_summary/` | Per-variant metrics (excludes control rows) |
| `eval_summary.csv` | `data/processed/eval_summary/` | Aggregated by variant_type + Job Role |

### 5.1 Detail Columns (Added)

For each method `m` in `config.scoring.methods`:

- `delta_score_{m}`, `rank_shift_{m}`
- `topk_{m}_k{k}`, `topk_{m}_k{k}_control`, `topk_{m}_k{k}_flip` for each `k` in `top_k`
- `threshold_pass_{m}`, `threshold_pass_{m}_control`, `threshold_flip_{m}`

### 5.2 Summary Rows

One row per `(variant_type, Job Role)`. Columns: flattened aggregations (e.g. `delta_score_tfidf_mean`, `topk_tfidf_k10_flip_sum`).

---

## 6. Run Manifest

Append to `outputs/run_manifest.jsonl`:

```json
{
  "stage": "evaluation",
  "input_file": "data/processed/variants_scored.csv",
  "input_rows": 5000,
  "detail_rows": 4000,
  "summary_rows": 16,
  "output_path": "data/processed/eval_summary",
  "dropped_rows": 0
}
```

---

## 7. Pipeline Integration

| Pipeline | Input | Eval Stage |
|----------|-------|------------|
| Static variant (current) | `variants_scored.csv` | Same `--stage evaluate` |
| NLP variant (Pipeline_plan) | `variants_scored.csv` | Same `--stage evaluate` |
| Future (embedding, other) | Any `variants_scored.csv` matching schema | Same eval, no code change |

**CLI:** `python main.py --config configs/default.yaml --stage evaluate`

---

## 8. RQ Mapping

| RQ | Metric(s) | Aggregation |
|----|-----------|-------------|
| RQ1: Do wording variations change scores/rankings? | Delta score, rank shift, top-K flip, threshold flip | Mean delta ≠ 0; flip rate > 0 |
| RQ2: Which variation types matter most? | All metrics by `variant_type` | Compare mean abs(delta), flip rates across types |
| RQ3: Consistent across roles and methods? | All metrics by `Job Role`; separate columns per method | Cross-tabulate variant_type × Job Role × method |

---

## 9. Extensibility

- **New scoring method:** Add to `config.scoring.methods`; evaluation loops over methods, no code change.
- **New variant type:** Dropped into `variant_type`; summarization groups by it automatically.
- **New K or threshold:** Add to config; eval reads and computes.

---

## 10. Validation

| Check | Action on Failure |
|-------|-------------------|
| Control row exists per (Resume_ID, Job Role) | Raise or log warning |
| All variant rows have matching control | Inner merge; log dropped |
| Required score/rank columns present | Raise if missing |
| Row count: `(N_resumes × N_roles × N_variant_types)` ≈ `N_rows - controls` | Log sanity check |

---

## 11. Reproducibility

- **Deterministic:** No randomness in evaluation; same input → same output.
- **Traceability:** Run manifest records input path, row counts, output path.
- **Config:** Full resolved config in manifest (per repo standards).
