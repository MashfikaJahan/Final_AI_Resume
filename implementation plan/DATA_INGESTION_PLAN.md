# Data Ingestion & Preprocessing Plan

**Project:** Same Skills, Different Words
**Dataset:** `Dataset/AI_Resume_Screening.csv` (Kaggle)
**Output:** `data/processed/resumes_clean.parquet`

---

## 1. Dataset Profile (As-Is)

| Property | Value |
|----------|-------|
| Rows | 1,000 |
| Columns | 11 |
| File format | CSV (UTF-8) |
| Resume_ID range | 1–1000 (all unique, no duplicates) |
| Job Roles (4) | AI Researcher (257), Data Scientist (255), Cybersecurity Analyst (255), Software Engineer (233) |
| Recruiter Decision | Hire (812, 81.2%), Reject (188, 18.8%) |
| Education levels (5) | B.Sc (205), MBA (202), B.Tech (200), M.Tech (198), PhD (195) |
| Certifications | 274 rows = `"None"` (string, not null) |
| Skills per resume | min 2, max 4, mean 3.0 |
| Unique skill tokens | 14 distinct skills across all resumes |
| Experience (Years) | 0–10, mean 4.9 |
| AI Score (0–100) | 15–100, mean 84.0 |
| Projects Count | 0–10, mean 5.1 |
| Salary Expectation ($) | 40,085–119,901, mean ~80,000 |

### Skills by Job Role

| Role | Top Skills (descending frequency) |
|------|----------------------------------|
| AI Researcher | TensorFlow (205), NLP (195), Python (192), Pytorch (189) |
| Cybersecurity Analyst | Ethical Hacking (206), Linux (191), Cybersecurity (184), Networking (184) |
| Data Scientist | Machine Learning (200), Python (196), SQL (192), Deep Learning (184) |
| Software Engineer | Java (188), SQL (178), C++ (168), React (165) |

### Known Quality Observations

- No missing `Resume_ID`, `Skills`, `Job Role`, or `Recruiter Decision`.
- `Certifications` uses the literal string `"None"` (274 rows) instead of actual null/empty — needs normalization.
- No leading/trailing whitespace in `Skills`.
- No casing inconsistencies in skill tokens (e.g., no mixed `"python"` / `"Python"`).
- Class imbalance in `Recruiter Decision`: 81% Hire vs 19% Reject.
- Skill vocabulary is small (14 tokens) — this is by design for a controlled synthetic dataset; the real variation comes from lexical rewording in the variant generation step.

---

## 2. Ingestion Pipeline Overview

```
AI_Resume_Screening.csv
        │
        ▼
  ┌─────────────┐
  │ data_loader  │  Load CSV, validate schema, compute file checksum
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  preprocessing   │  Clean text, normalize columns, type-cast, validate
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │  Write parquet   │  data/processed/resumes_clean.parquet
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │  Run manifest    │  Append to outputs/ log
  └──────────────────┘
```

---

## 3. Step-by-Step Implementation

### 3.1 File Loading (`src/data_loader.py`)

**Inputs:** Raw CSV path (default `Dataset/AI_Resume_Screening.csv`)

| Task | Detail |
|------|--------|
| Read CSV | `pd.read_csv(path, encoding="utf-8")` |
| Compute file hash | SHA-256 of the raw file bytes — store in run manifest for reproducibility |
| Log file metadata | Row count, column names, file size (bytes), hash |
| Schema validation | Assert the 11 expected columns exist (see §3.1.1). Fail fast if any are missing. |
| Duplicate ID check | Assert `Resume_ID` is unique. Log and abort if duplicates found. |

#### 3.1.1 Expected Schema

```
Required columns (exact names):
  Resume_ID              int
  Skills                 str
  Experience (Years)     int/float
  Education              str
  Certifications         str
  Job Role               str
  Recruiter Decision     str
  Salary Expectation ($) int/float
  Projects Count         int
  AI Score (0-100)       int/float
```

**Error handling:** If any required column is missing, raise `ValueError` with the column name. If extra columns are present, log a warning but continue.

---

### 3.2 Text Cleaning & Normalization (`src/preprocessing.py`)

#### 3.2.1 `Skills` Column

This is the primary experimental field. Cleaning must be conservative to avoid destroying signal that the variant generator will later manipulate.

| Step | Operation | Rationale |
|------|-----------|-----------|
| 1 | Strip leading/trailing whitespace | Standard hygiene (already clean in this dataset, but guard for future data) |
| 2 | Normalize internal whitespace | Collapse multiple spaces: `re.sub(r'\s+', ' ', s)` |
| 3 | Normalize comma separators | Ensure `", "` (comma-space) between tokens: `re.sub(r'\s*,\s*', ', ', s)` |
| 4 | Validate non-empty | Drop or flag rows where Skills is empty after cleaning |

**Do NOT:**
- Lowercase skills (casing matters for acronyms like "NLP", "SQL")
- Deduplicate skill tokens within a row (if a resume lists "Python, Python" that's a data quality flag to log, not silently fix)
- Stem or lemmatize (variant generator handles phrasing changes)

#### 3.2.2 `Certifications` Column

| Step | Operation |
|------|-----------|
| 1 | Replace literal `"None"` with empty string `""` |
| 2 | Strip whitespace |

This normalizes the 274 "None" entries so downstream code can use simple `if cert:` checks.

#### 3.2.3 PII Removal & Anonymized Identifier

| Step | Operation |
|------|-----------|
| 1 | Drop `Name` column (`errors="ignore"` for safety) |
| 2 | Add `Resume_Hash_ID` — deterministic 12-char SHA-256 hex of `Resume_ID` |

`Name` is PII and is not used in scoring, variant generation, or evaluation. It is dropped
at the start of preprocessing to prevent leakage into any downstream artifact. `Resume_Hash_ID`
provides a stable anonymized identifier for sharing/reporting alongside the numeric `Resume_ID`.

#### 3.2.4 `Education` Column

| Step | Operation |
|------|-----------|
| 1 | Strip whitespace |
| 2 | Validate against known levels | `{"B.Sc", "MBA", "B.Tech", "M.Tech", "PhD"}` |
| 3 | Log any unexpected values |

#### 3.2.5 `Job Role` Column

| Step | Operation |
|------|-----------|
| 1 | Strip whitespace |
| 2 | Validate against allowlist | `{"AI Researcher", "Data Scientist", "Cybersecurity Analyst", "Software Engineer"}` |
| 3 | Reject rows with unknown roles (log to `data/processed/dropped_rows.log`) |

#### 3.2.6 `Recruiter Decision` Column

| Step | Operation |
|------|-----------|
| 1 | Strip whitespace |
| 2 | Validate values ∈ `{"Hire", "Reject"}` |
| 3 | Reject rows with invalid values (log to `data/processed/dropped_rows.log`) |

---

### 3.3 Type Casting

| Column | Target dtype | Operation |
|--------|-------------|-----------|
| `Resume_ID` | `int64` | `pd.to_numeric(errors='coerce')` → drop rows where coercion fails |
| `Experience (Years)` | `int64` | Cast after validating range [0, 50]. Values outside range → log + keep |
| `Salary Expectation ($)` | `float64` | Cast, validate > 0 |
| `Projects Count` | `int64` | Cast, validate ≥ 0 |
| `AI Score (0-100)` | `float64` | Cast, validate range [0, 100] |
| `Skills` | `string` | Ensure pandas `StringDtype` |
| `Job Role` | `category` | Convert for memory efficiency + enforced allowlist |
| `Recruiter Decision` | `category` | Convert |
| `Education` | `category` | Convert |

---

### 3.4 Derived Columns (Added During Preprocessing)

These columns are computed once and stored in the clean parquet to avoid recomputation downstream.

| New Column | Definition | Purpose |
|------------|-----------|---------|
| `skills_list` | `Skills.str.split(', ')` → Python list | Fast iteration in variant generator |
| `skills_count` | `len(skills_list)` | Feature for logistic regression, sanity check |
| `skills_lower` | `Skills.str.lower()` | Lowercased version for case-insensitive matching in scoring |

---

### 3.5 Validation Summary & Quality Gate

After all cleaning, run a final validation pass before writing output:

```
VALIDATION CHECKS (all must pass):
  ✓ Resume_ID: unique, no nulls, int
  ✓ Skills: no empty strings, no nulls
  ✓ Job Role: all values in allowlist
  ✓ Recruiter Decision: all values in {"Hire", "Reject"}
  ✓ Experience (Years): no nulls, range [0, 50]
  ✓ AI Score (0-100): no nulls, range [0, 100]
  ✓ Projects Count: no nulls, range [0, ∞)
  ✓ Salary Expectation ($): no nulls, > 0
  ✓ Row count ≥ 950 (guard against accidental mass-drop)
  ✓ No duplicate Resume_IDs after cleaning
```

If any check fails, log the specific failure and abort before writing parquet. This prevents corrupted data from propagating to variant generation and scoring.

---

### 3.6 Dropped Row Logging

Any row removed during preprocessing is logged to `data/processed/dropped_rows.log` with:

```
timestamp | Resume_ID | reason | raw_row_snapshot
```

Expected: 0 dropped rows for this dataset. The log exists as a safety net for future data additions.

---

### 3.7 Output Artifact

**File:** `data/processed/resumes_clean.parquet`

| Property | Value |
|----------|-------|
| Format | Apache Parquet (Snappy compression) |
| Expected rows | 1,000 (no drops expected) |
| Columns | Original 11 + 3 derived (`skills_list`, `skills_count`, `skills_lower`) = 14 |
| Index | None (default RangeIndex); `Resume_ID` is a column, not the index |

**Why Parquet over CSV:**
- Preserves dtypes (no re-parsing on load)
- Columnar compression (smaller on disk)
- Faster read for downstream pandas/pyarrow operations
- Schema enforcement on re-read

---

## 4. Run Manifest

Each ingestion run writes a manifest entry to `outputs/run_manifest.jsonl` (one JSON line per run):

```json
{
  "timestamp": "2026-02-27T14:30:00Z",
  "stage": "data_ingestion",
  "git_commit": "abc1234",
  "config_path": "configs/default.yaml",
  "input_file": "Dataset/AI_Resume_Screening.csv",
  "input_sha256": "e3b0c44298fc1c14...",
  "input_rows": 1000,
  "output_file": "data/processed/resumes_clean.parquet",
  "output_rows": 1000,
  "dropped_rows": 0,
  "seed": 42,
  "python_version": "3.11.x",
  "pandas_version": "2.x.x"
}
```

---

## 5. Configuration (`configs/default.yaml`)

Ingestion-related config keys:

```yaml
data:
  raw_path: "Dataset/AI_Resume_Screening.csv"
  processed_dir: "data/processed"
  clean_parquet: "data/processed/resumes_clean.parquet"
  dropped_log: "data/processed/dropped_rows.log"

validation:
  required_columns:
    - Resume_ID
    - Skills
    - "Experience (Years)"
    - Education
    - Certifications
    - "Job Role"
    - "Recruiter Decision"
    - "Salary Expectation ($)"
    - "Projects Count"
    - "AI Score (0-100)"
  job_role_allowlist:
    - AI Researcher
    - Data Scientist
    - Cybersecurity Analyst
    - Software Engineer
  recruiter_decision_values:
    - Hire
    - Reject
  education_levels:
    - B.Sc
    - MBA
    - B.Tech
    - M.Tech
    - PhD
  min_rows_after_clean: 950

seed: 42
```

---

## 6. Module-Level Implementation Spec

### `src/data_loader.py`

```python
def load_raw(csv_path: str) -> pd.DataFrame:
    """Load CSV, validate schema, return raw DataFrame."""

def compute_file_hash(path: str) -> str:
    """SHA-256 hex digest of the file."""

def validate_schema(df: pd.DataFrame, expected_columns: list[str]) -> None:
    """Raise ValueError if required columns are missing."""
```

### `src/preprocessing.py`

```python
def clean_skills(series: pd.Series) -> pd.Series:
    """Strip, normalize whitespace, normalize comma separators."""

def normalize_certifications(series: pd.Series) -> pd.Series:
    """Replace 'None' with '', strip."""

def validate_categorical(series: pd.Series, allowlist: set, col_name: str) -> pd.Series:
    """Flag/drop rows with values outside allowlist."""

def cast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns to proper dtypes with coercion + range validation."""

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add skills_list, skills_count, skills_lower."""

def run_quality_gate(df: pd.DataFrame, config: dict) -> None:
    """Final validation checks. Raises on failure."""

def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Full pipeline: clean → cast → derive → validate → return."""
```

---

## 7. Execution Entry Point

From `main.py` (or standalone):

```python
# Ingestion
raw_df = load_raw(config["data"]["raw_path"])
clean_df = preprocess(raw_df, config)
clean_df.to_parquet(config["data"]["clean_parquet"], engine="pyarrow", compression="snappy")
write_manifest(...)
```

CLI usage:

```bash
python main.py --config configs/default.yaml --stage ingest
```

---

## 8. Testing Checklist

| Test | Type | What to Verify |
|------|------|---------------|
| Schema validation rejects missing column | Unit | Remove a column, assert `ValueError` |
| Duplicate Resume_ID detection | Unit | Inject duplicate, assert caught |
| `"None"` → `""` in Certifications | Unit | Check 274 rows transformed |
| Skills comma normalization | Unit | Input `"Python,NLP , SQL"` → `"Python, NLP, SQL"` |
| Invalid Job Role rejected | Unit | Inject `"Janitor"`, assert dropped + logged |
| Type casting coercion | Unit | Inject `"abc"` in numeric column, assert coerced to NaN + logged |
| Quality gate min-rows | Unit | Pass 900-row DF with threshold 950, assert raises |
| Full pipeline integration | Integration | Load real CSV → parquet → re-read → assert schema + row count |
| Run manifest written | Integration | Check `outputs/run_manifest.jsonl` contains expected keys |
| Idempotency | Integration | Run twice, assert identical parquet output (same hash) |

---

## 9. Dependencies

All already covered in `requirements.txt`:

```
pandas >= 2.0
pyarrow >= 14.0
pyyaml >= 6.0
```

---

## 10. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CSV encoding issues (BOM, mixed encoding) | Loader fails or corrupts text | Explicit `encoding="utf-8-sig"` fallback; log detected encoding |
| Future dataset has new Job Roles | Rows silently dropped | Allowlist in config (easy to update), warning log on unknown roles |
| Future dataset has null Skills | Variants break downstream | Non-empty validation gate; dropped_rows.log for audit |
| Large dataset (>100K rows) | Memory pressure | Parquet chunked write; consider dask if needed |
| Skill tokens with special characters or unicode | Regex normalization breaks | Keep cleaning conservative; only strip/normalize whitespace and commas |
| Class imbalance (81/19 Hire/Reject) | Biased logistic regression | Not an ingestion concern, but flag in manifest for downstream awareness |
