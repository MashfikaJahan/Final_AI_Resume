# Literature Review: Lexical Variation Effects in Automated Resume Screening

**Project:** Same Skills, Different Words — A Comparative Quasi-Experimental Analysis  

---

## Literature Summaries

### "An Automated Resume Screening System Using Natural Language Processing and Similarity" (2020)

"An Automated Resume Screening System Using Natural Language Processing and Similarity" (2020) explains that an automated resume screening method is meant to reduce the time and cost of manually reviewing many applications while making shortlisting more consistent. The system extracts important resume information such as skills, education, and experience from unstructured text using NLP, then turns resumes and job descriptions into vectors so they can be compared. Candidates are ranked using similarity scoring (e.g., cosine similarity) between resume content and the job description, which produces a prioritized list for recruiters. This paper shows how basic NLP and similarity scoring can support faster candidate ranking.

**Finding:** The system demonstrates that NLP-based extraction plus similarity scoring can effectively rank candidates and speed up the screening process.

### "Developing an Intelligent Resume Screening Tool With AI-Driven Analysis and Recommendation Features" (2025)

"Developing an Intelligent Resume Screening Tool With AI-Driven Analysis and Recommendation Features" (2025) describes an AI-based resume analyzer that automates resume parsing and analysis for recruiters while also giving improvement feedback to applicants. The tool focuses on extracting structured information from resumes (like contact details, education, experience, and skills) using NLP techniques and then storing and analyzing that information to support hiring decisions. In addition to recruiter-facing analytics (such as skill and role trends), the system includes applicant-facing recommendations to help strengthen resume quality and alignment with job requirements. This paper is useful for showing resume screening tools as full systems—not just ranking—combining automation, usability, and analytics.

**Findings:** The tool shows that combining resume parsing with analytics and recommendations can support recruiter decisions and improve applicant resume quality.

### "Mitigating demographic bias in AI-based resume filtering" (2020)

"Mitigating demographic bias in AI-based resume filtering" (2020) researches how AI resume filters can accidentally become unfair because the way people write, and their word choices, can correlate with demographic groups. The paper treats resume screening as a text-matching problem and proposes a fairness-focused adjustment to common keyword methods, which is described as fair-TF-IDF, so the system is less likely to give higher scores to one group just because of language patterns rather than qualifications. This paper focuses on making automated resume filtering more fair while still ranking resumes by relevance to a job.

**Findings:** The proposed fair-TF-IDF approach reduces demographic-related bias in ranking while still keeping useful relevance-based matching.

### "When words matter: Communal and agentic language on men and women's resumes" (2024)

"When words matter: Communal and agentic language on men and women's resumes" (2024) studies how different types of wording on resumes affect how candidates are judged. They used communal language, such as helpful, supportive, collaborative vs. agentic language, such as leader, achiever, and independent. The study shows that wording can influence perceptions of fit and competence, and that these effects can relate to gender norms in hiring contexts. This paper supports the idea that resumes are not judged only on skills, but also on how those skills are described.

**Findings:** Communal vs. agentic wording changes how applicants are perceived, and the impact is connected to gender expectations in hiring.

### "The probabilistic relevance framework: BM25 and beyond" (2009)

"The probabilistic relevance framework: BM25 and beyond" (2009) explains the theory behind BM25, one of the most common ranking algorithms used to match a query to documents. The paper describes how BM25 scores relevance using term frequency and document-length normalization and discusses extensions that improve ranking in different settings. For resume screening research, this is important because BM25 is a strong baseline for job description and resume matching.

**Findings:** BM25 is shown to be a strong, reliable baseline ranking method, and extensions can improve performance in specialized retrieval settings.

### "Algorithmic writing assistance on jobseekers' resumes increases hires" (2024/2025)

"Algorithmic writing assistance on jobseekers' resumes increases hires" (2024/2025) tests whether helping jobseekers improve resume writing changes real outcomes and reports evidence that writing assistance increases hiring. The paper suggests that how clearly and effectively a resume is written can affect employer decisions, even when a person's underlying experience is the same. This shows that wording tools can have real downstream effects, not just score changes.

**Findings:** Resume writing assistance leads to higher hiring rates, showing wording improvements can directly affect real job outcomes.

### "Gender, race, and intersectional bias in resume screening via language model retrieval" (2024)

"Gender, race, and intersectional bias in resume screening via language model retrieval" (2024) studies bias when large language models are used to retrieve/rank candidates. The paper finds that demographic cues, such as names associated with gender or race, can influence which candidates are surfaced and ranked higher; this also includes intersectional patterns. This work shows how bias can happen early in hiring pipelines through retrieval and ranking, not only through final decisions.

**Findings:** LLM-based retrieval/ranking can systematically favor or disadvantage groups based on demographic cues, including intersectional effects.

### "Gender and positional biases in LLM-based hiring decisions: Evidence from comparative CV/résumé evaluations" (2025)

"Gender and positional biases in LLM-based hiring decisions" (2025) evaluates whether LLMs show unfair preferences when comparing two similar resumes. It reports patterns where decisions can shift based on gender cues and also based on position/order effects. They mention that the model may favor the first candidate shown. This shows that even when qualifications are held constant, LLM-based screening can still be influenced by non-skill cues and prompt structure.

**Findings:** The study finds both gender bias and order bias, where models often prefer the candidate shown first even with similar qualifications.

### "PROSPECT: A system for screening candidates for recruitment" (2010)

"PROSPECT: A system for screening candidates for recruitment" (2010) presents a practical system that helps recruiters screen large applicant pools by extracting structured information from resumes such as skills, experience, education, and supporting filtering through facets. It uses information retrieval ranking to prioritize candidates for a job. This research is about improving recruiter efficiency through extraction, search, and ranking rather than testing wording sensitivity.

**Findings:** Using structured extraction and faceted filtering improves recruiter efficiency and supports better candidate search and ranking.

### "Matching resumes and jobs based on relevance models" (2007)

"Matching resumes and jobs based on relevance models" (2007) explores resume job matching as an information retrieval problem and compares different relevance modeling approaches, including methods that use the structure of resumes. For example, they used sections/fields instead of treating the document as plain text. The researchers mentioned that representation choices, such as structured vs. unstructured can affect matching quality. This paper supports the idea that screening performance depends heavily on how text is modeled.

**Findings:** The study finds that using resume/job structure (fields/sections) can improve matching performance compared to treating documents as one block of text.

---

## References

1. Abhishek, K. L., Niranjanamurthy, M., Aric, S., Ansarullah, S. I., Sinha, A., Tejani, G., & Shah, M. A. (2025). Developing an intelligent resume screening tool with AI-driven analysis and recommendation features. *Applied AI Letters*, 6, e116. https://doi.org/10.1002/ail2.116

2. Daryani, C., Chhabra, G. S., Patel, H., Chhabra, I. K., & Patel, R. (2020). An automated resume screening system using natural language processing and similarity. *Topics in Intelligent Computing and Industry Design*, 2(2), 99–103. https://doi.org/10.26480/etit.02.2020.99.103

3. Deshpande, K. V., Pan, S., & Foulds, J. R. (2020). Mitigating demographic bias in AI-based resume filtering. In *Adjunct Proceedings of the 28th ACM Conference on User Modeling, Adaptation and Personalization (UMAP '20)* (pp. 1–6). ACM. https://dl.acm.org/doi/10.1145/3386392.3399569

4. Madera, J. M., Ng, L., Zajac, S., & Hebl, M. R. (2024). When words matter: Communal and agentic language on men and women's resumes. *Journal of Business and Psychology*, 40(2), 479–496. https://doi.org/10.1007/s10869-024-09969-0

5. Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389. https://doi.org/10.1561/1500000019

6. Rozado, D. (2025). Gender and positional biases in LLM-based hiring decisions: Evidence from comparative CV/résumé evaluations. *PeerJ Computer Science*, 11, e3628. https://doi.org/10.7717/peerj-cs.3628

7. Singh, A., Catherine, R., Visweswariah, K., Chenthamarakshan, V., & Kambhatla, N. (2010). PROSPECT: A system for screening candidates for recruitment. In *Proceedings of the 19th ACM International Conference on Information and Knowledge Management (CIKM '10)* (pp. 659–668). ACM. https://dl.acm.org/doi/10.1145/1871437.1871523

8. van Inwegen, E., Munyikwa, Z., & Horton, J. J. (2025). Algorithmic writing assistance on jobseekers' resumes increases hires. *Management Science*, 71(12), 10144–10164. https://doi.org/10.1287/mnsc.2024.04528

9. Wilson, K., & Caliskan, A. (2024). Gender, race, and intersectional bias in resume screening via language model retrieval. *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES '24)*, 7(1), 1578–1590. https://doi.org/10.1609/aies.v7i1.31748

10. Yi, X., Allan, J., & Croft, W. B. (2007). Matching resumes and jobs based on relevance models. In *Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval* (pp. 809–810). ACM. https://dl.acm.org/doi/10.1145/1277741.1277920

---
