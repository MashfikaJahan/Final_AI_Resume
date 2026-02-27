# Literature Review: Lexical Variation Effects in Automated Resume Screening

**Project:** Same Skills, Different Words — A Comparative Quasi-Experimental Analysis  
**Scope:** Peer-reviewed academic journals and conference proceedings only. No blogs, industry reports, or gray literature. Predatory outlets excluded.

---

## 1. Introduction & Motivation

Automated resume screening is pervasive: approximately 75% of recruiters use applicant tracking systems (ATS), with companies receiving hundreds of resumes per position (Deshpande et al., 2020). Lexical variations—how candidates phrase the same skills (e.g., "Python" vs "Python programming," "ML" vs "Machine Learning")—may materially affect screening scores and rankings in ATS pipelines that rely on keyword or TF-IDF matching. This review synthesizes credible literature on (a) wording effects in hiring, (b) automated screening methods and their susceptibility to lexical variation, and (c) bias and mitigation in algorithmic resume filtering.

---

## 2. Wording, Phrasing, and Language Effects in Hiring

### 2.1 Human Judgment: Resume Language and Hireability

**Madera et al. (2024)** — *Journal of Business and Psychology* (Springer)  
"When Words Matter: Communal and Agentic Language on Men and Women's Resumes"

Examined how men and women describe themselves on resumes and how language affects career-related outcomes. Women used more communal language (collaborative, supportive terms) than men; communal language negatively impacted perceived leadership ability and hireability for women applying to masculine-typed jobs. This demonstrates that **word choice—not just skill presence—affects human evaluations**, which motivates studying analogous effects in automated systems.

- **Relevance to RQ1/RQ2:** Evidence that phrasing type (communal vs agentic) affects outcomes; supports investigating variation types (phrasing, abbreviation, etc.) in ATS.

### 2.2 Job Ad Wording and Applicant Behavior

**Abraham, Hallermeier & Stein (2024)** — *Journal of Economic Behavior & Organization* (Elsevier)  
"Words matter: Experimental evidence from job applications"

Randomized experiment with ~60,000 viewers across 600+ Uber job postings. Removing optional qualifications increased applications by 7% but affected men and women differently: less-skilled women applied more, while some highly skilled women self-selected out. **Wording changes have measurable, sometimes heterogeneous effects** on who applies and who gets considered.

- **Relevance to RQ1:** Wording matters even before screening; ATS inputs are shaped by ad phrasing and applicant self-presentation.

### 2.3 Algorithmic Writing Assistance and Hiring Outcomes

**van Inwegen et al. (2024/2025)** — Management Science (INFORMS) / NBER  
"Algorithmic Writing Assistance on Jobseekers' Resumes Increases Hires"

Large-scale field experiment (~500K jobseekers). Treated applicants receiving algorithmic writing assistance were hired ~8% more often; improved readability and fewer errors, with no evidence of lower employer satisfaction. Supports a "clarity view"—better wording helps employers assess ability rather than merely signaling it.

- **Relevance to RQ1/RQ3:** Direct evidence that **wording quality affects hiring outcomes**; suggests ATS and downstream human review respond to lexical variation.

---

## 3. Automated Screening: Methods and Lexical Sensitivity

### 3.1 TF-IDF, BM25, and Resume–Job Matching

**Robertson & Zaragoza (2009)** — *Foundations and Trends in Information Retrieval*  
"The Probabilistic Relevance Framework: BM25 and Beyond"

Canonical reference for BM25 and probabilistic relevance. TF-IDF and BM25 are lexical methods: they match on term overlap and frequency. **By design, they are sensitive to exact phrasing and abbreviation**—"Python" vs "Python programming" may produce different scores. No direct resume application, but foundational for interpreting project methodology.

- **Relevance to RQ1/RQ3:** Explains why lexical methods are expected to vary with wording; motivates paired control-variant experiments.

**Bhat et al. (2007)** — ACM SIGIR  
"Matching resumes and jobs based on relevance models"

Applied relevance modeling to resume–job matching at SIGIR. Uses classical IR techniques; lexical representation assumptions imply susceptibility to synonymy and phrasing variation.

**Bhattacharya et al. (2010)** — ACM CIKM  
"PROSPECT: A system for screening candidates for recruitment"

IBM system for resume screening using structured extraction (skills, experience) and IR ranking. Extracted facets improved ranking accuracy by ~30%, but underlying matching still relies on term-based representations that can be affected by phrasing and abbreviation.

- **Relevance to RQ2:** Systems combining extraction and IR are widely used; understanding which variation types (abbreviation, word order) matter most is practically important.

---

## 4. Bias, Fairness, and Mitigation in Resume Screening

### 4.1 Socio-Linguistic Bias and Fair-TF-IDF

**Deshpande, Pan & Foulds (2020)** — ACM UMAP  
"Mitigating Demographic Bias in AI-based Resume Filtering"

Demonstrates that **socio-linguistic characteristics in resume writing** correlate with protected attributes; standard TF-IDF matching can perpetuate demographic bias. Proposes **fair-TF-IDF** to mitigate socio-linguistic bias while preserving matching quality.

- **Relevance to RQ1/RQ3:** Directly addresses lexical/TF-IDF methods; shows that wording correlates with demographics and affects outcomes. Fair-TF-IDF is a mitigation approach for lexical methods.

### 4.2 Embedding and LLM-Based Screening: Bias and Variation Sensitivity

**Wilson & Caliskan (2024)** — AAAI/ACM AIES  
"Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval"

Audit of Massive Text Embedding (MTE) models in resume screening. Found significant bias: White-associated names favored in 85.1% of cases; Black males disadvantaged in up to 100% of cases. Document length and corpus frequency of names also affected selection.

- **Relevance to RQ3:** Embedding-based screening introduces different bias patterns but remains sensitive to document characteristics. Compares lexical vs embedding behavior.

**Rozado (2024)** — PeerJ Computer Science  
"Gender and positional biases in LLM-based hiring decisions: evidence from comparative CV/résumé evaluations"

Tested 22 LLMs on resume evaluation across 70 professions. LLMs showed positional bias (preference for first-listed candidate) and gender-associated name effects; gender-neutral labels reduced some biases.

- **Relevance to RQ3:** LLMs exhibit structural biases separate from lexical matching but relevant for comparing screening methods.

### 4.3 Procedural Justice and Candidate Experience

**IEEE (2024)** — "Procedural Justice and Fairness in Automated Resume Parsers for Tech Hiring"

Examines candidate perceptions of fairness in automated resume parsing. Supports the importance of transparency and consistency in automated screening—directly relevant when lexical variation causes arbitrary rank shifts.

---

## 5. Synthesis: Gaps and Alignment with Project

| Theme | Key Finding | Project Alignment |
|-------|-------------|-------------------|
| Human wording effects | Phrasing (communal/agentic), writing quality affect hireability (Madera et al.; van Inwegen et al.) | RQ1: Do wording variations change ATS scores? |
| Lexical methods | TF-IDF/BM25 are term-dependent; sensitive to phrasing, abbreviation (Robertson & Zaragoza; Deshpande et al.) | RQ1, RQ2: Quantify effect; identify which variation types matter |
| Fairness | Socio-linguistic bias in TF-IDF; fair-TF-IDF as mitigation (Deshpande et al.) | Phase 2 mitigation; evidence-based guidance |
| Method comparison | Lexical vs embedding/LLM show different bias patterns (Wilson & Caliskan; Rozado) | RQ3: Consistency across methods |
| IR systems | PROSPECT, relevance models use extraction + IR (Bhattacharya et al.; Bhat et al.) | Methodological context for scoring pipelines |

**Identified gap:** Prior work addresses bias mitigation and wording effects in human or LLM settings, but **systematic quasi-experimental quantification of lexical variation effects** (phrasing, abbreviation, word order, placement) within ATS-like pipelines (TF-IDF, BM25) remains sparse. This project fills that gap by holding resume content constant and varying only wording, then measuring Δ score, rank shift, top-K inclusion, and threshold pass/fail.

---

## 6. Reference List (Credible Sources Only)

1. **Deshpande, K. V., Pan, S., & Foulds, J. R. (2020).** Mitigating demographic bias in AI-based resume filtering. *Adjunct Proceedings of the 28th ACM Conference on User Modeling, Adaptation and Personalization (UMAP '20)*. ACM. https://dl.acm.org/doi/10.1145/3386392.3399569

2. **Madera, J. M., et al. (2024/2025).** When words matter: Communal and agentic language on men and women's resumes. *Journal of Business and Psychology*, 40, 479–496. Springer. https://doi.org/10.1007/s10869-024-09969-0

3. **Robertson, S., & Zaragoza, H. (2009).** The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389. Now Publishers.

4. **van Inwegen, E., Munyikwa, Z., & Horton, J. J. (2024/2025).** Algorithmic writing assistance on jobseekers' resumes increases hires. *Management Science*, 71(12), 10144–10164. INFORMS. NBER Working Paper 30886.

5. **Wilson, K., & Caliskan, A. (2024).** Gender, race, and intersectional bias in resume screening via language model retrieval. *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES '24)*, 7(1), 1578–1590. https://doi.org/10.1609/aies.v7i1.31748

6. **Rozado, D. (2024).** Gender and positional biases in LLM-based hiring decisions: evidence from comparative CV/résumé evaluations. *PeerJ Computer Science*. https://peerj.com/articles/cs-3628/

7. **Bhattacharya, P., et al. (2010).** PROSPECT: A system for screening candidates for recruitment. *Proceedings of the 19th ACM Conference on Information and Knowledge Management (CIKM '10)*. ACM. https://dl.acm.org/doi/10.1145/1871437.1871523

8. **Bhat, F., et al. (2007).** Matching resumes and jobs based on relevance models. *Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval*. ACM. https://dl.acm.org/doi/10.1145/1277741.1277920

9. **Abraham, L., Hallermeier, J., & Stein, A. (2024).** Words matter: Experimental evidence from job applications. *Journal of Economic Behavior & Organization*, 225, 348–391. Elsevier. https://doi.org/10.1016/j.jebo.2024.03.017

---

## 7. Exclusion Note

**Excluded:** (1) IJARESM—predatory indicators. (2) Blogs, industry reports, practitioner white papers, and gray literature. (3) ArXiv preprints without confirmed peer-reviewed publication. All cited sources are peer-reviewed academic journals or ACM/IEEE/AAAI conference proceedings.
