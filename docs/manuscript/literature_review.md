# Literature on the Ahmad et al. (2017) heart-failure dataset

Compiled November 2026 from web search results (publisher sites, PubMed Central and
arXiv were not reachable from the analysis environment, so details marked
**[verify]** come from search-engine snippets and must be checked in the full text).

## A. Source and reference papers

| # | Study | Venue | Relevance | Key facts | Status |
|---|---|---|---|---|---|
| A1 | Ahmad T, Munir A, Bhatti SH, Aftab M, Raza MA. Survival analysis of heart failure patients: a case study | *PLoS ONE* 2017;12(7):e0181001 | Data source | 299 patients with LV systolic dysfunction (NYHA III–IV), Institute of Cardiology and Allied Hospital, Faisalabad, April–December 2015; 96 deaths, 203 censored; Cox model: age, renal dysfunction, blood pressure, EF and anaemia significant (anaemia HR ≈ 1.77) | confirmed by search; anaemia cut-off **[verify]** |
| A2 | Zahid FM, Ramzan S, Faisal S, Hussain I. Gender based survival prediction models for heart failure patients: a case study in Pakistan | *PLoS ONE* 2019;14(2):e0210602 | Same cohort, sex-specific Cox (lasso) | n = 298 (one EF > 75 excluded); C-index 0.72 overall, 0.73 men, 0.77 women; states anaemia = haematocrit < 36 | confirmed by search. **The legacy notebook cited this as "J Med Syst 43:298" with invented AUC/sensitivity values.** |
| A3 | Chicco D, Jurman G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone | *BMC Med Inform Decis Mak* 2020;20:16 | Benchmark this study re-evaluates | Table 4 (all clinical features, **follow-up time excluded**, 100 executions): random forest MCC +0.384, F1 0.547, accuracy 0.740, TPR 0.491, TNR 0.864, PR-AUC 0.657, ROC-AUC 0.800. Table 9 (EF + SC): random forest MCC +0.418. A later analysis adds follow-up time (stratified logistic regression with EF, SC and follow-up month) | Table 4 row confirmed via snippet; Table 9 column mapping **[verify]** |

**Implication for our replication:** Chicco & Jurman's main table *excluded* follow-up time, so the correct comparison is with our scenario S2 (random forest MCC 0.358, ROC-AUC 0.786), not S1. Our S3 (EF + SC) random forest MCC is 0.407 vs 0.418 published.

## B. Studies addressing leakage or survival framing (closest prior work)

| # | Study | Venue | What it does | Overlap with our work | Status |
|---|---|---|---|---|---|
| B1 | Rane A, Choudhari A, Kant S, Deep A. Discrete-time survival analysis for heart failure mortality prediction | arXiv:2608.04140 (stat.AP), 4 Aug 2026 | Person-period (discrete-time) cloglog GLM/GAM, PP random forest, XGBoost, RSF, DeepSurv; 180-day horizon; argues follow-up time as an input causes severe target leakage; reports MCC rising from ≈0.35–0.40 to ≈0.94–0.98 when follow-up is added **[verify]** | **High** — same leakage argument and survival re-framing on the same data | preprint; full text **[verify]** |
| B2 | Özgür & Bekiroğlu. Why follow-up matters in survival analysis: comparing Cox proportional hazard regression and random survival forest for predicting heart failure outcomes | *BMC Cardiovasc Disord* 2025 (doi:10.1186/s12872-025-05165-x) | Person-Time Follow-up Rate (PTFR 45.6% observed, 67.2% simulated); Cox vs RSF on the 299-patient data; RSF better at higher PTFR | Medium — survival framing, no leakage quantification, no external validation | **[verify]** details |
| B3 | Fairness-aware machine learning for heart failure prediction: performance, bias, and clinical deployment insights | medRxiv 10.1101/2025.10.17.25338263 | UCI data + MIMIC-IV HF cohort; all models declined on MIMIC; fairness, decision curves | Medium-high — external MIMIC evaluation and fairness | preprint; authors, endpoint, use of `time` **[verify]** |

**Positioning.** B1 makes the leakage argument and B3 tests transfer to MIMIC. To our
knowledge, no study (i) isolates the mechanism with a time-only model, a permutation
control *and* a null-effect simulation, (ii) quantifies how often public code and published
studies use `time` as a predictor, (iii) externally validates leakage-free *survival*
models with censoring-aware calibration, recalibration and decision curves, and (iv)
releases fully tested, reproducible code. These are the novelty claims to make — and to
check again against B1–B3 before submission.

## C. Studies reporting classification performance (examples of the problem)

| # | Study | Venue | Reported result | Use of `time` | Status |
|---|---|---|---|---|---|
| C1 | Moreno-Sánchez PA. Development of an explainable prediction model of heart failure survival by using ensemble trees | IEEE BigData 2020, pp. 4902–4910 | XGBoost 83% accuracy (unseen data) | follow-up time ranked **most relevant** feature | snippet |
| C2 | Moreno-Sánchez PA. Improvement of a prediction model for heart failure survival through explainable AI | *Front Cardiovasc Med* 2023 (arXiv:2108.10717) | Extra Trees, balanced accuracy 85.1% (CV) / 79.5% (unseen) with 5 of 12 features | follow-up time **most influential**, then SC, EF | snippet |
| C3 | Ishaq A, Sadiq S, Umer M, et al. Improving the prediction of heart failure patients' survival using SMOTE and effective data mining techniques | *IEEE Access* 2021;9:39707–39716 | Extra Trees + SMOTE accuracy 92.62% | conflicting secondary reports **[verify]** | snippet |
| C4 | Newaz A, Ahmed N, Haq FS. Survival prediction of heart failure patients using machine learning techniques | *Inform Med Unlocked* 2021;26:100772 | chi-square / RFE feature selection; several classifiers | **[verify]** | snippet |
| C5 | Karakuş MÖ, Er O. A comparative study on prediction of survival event of heart failure patients using machine learning algorithms | *Neural Comput Appl* 2022;34(16):13895–13908 | claims up to 100% accuracy for some algorithms (70/30 split) | **[verify]** | snippet |
| C6 | Oyewunmi OE, Aladeniyi OB, Bodunwa OK. Comparative study on prediction of survival event of heart failure patients using machine learning and statistical algorithms | *SciMedicine J* 2023;5(2):44–53 | RF accuracy > 90% (SMOTE) | **[verify]** | snippet |
| C7 | Talukder MSH, Sulaiman RB, Angon MBP. Unleashing the power of extra-tree feature selection and random forest classifier for improved survival prediction in heart failure patients | arXiv:2308.05765, 2023 | 98.33% accuracy | **time ranked first** by Extra-Tree importance | snippet |
| C8 | Qadri AM, Hashmi MSA, Raza A, Zaidi SAJ, Rehman A. Heart failure survival prediction using novel transfer learning based probabilistic features | *PeerJ Comput Sci* 2024;10:e1894 | RF accuracy 0.975 (10-fold CV, SMOTE) | **[verify]** | snippet |
| C9 | Shamrat FMJM, Khalid M, Qadah TM, Farrash M, Alshanbari H. An explainable multi-objective hybrid machine learning model for reducing heart failure mortality | *PeerJ Comput Sci* 2025;e2682 | stacked hybrid + NSGA-II + LIME | **[verify]** | snippet |

**Pattern.** Reported accuracies rise from ≈0.74 (Chicco & Jurman, without time) to
0.92–1.00 in later studies that often combine follow-up time, oversampling (SMOTE, which
also leaks when applied before splitting) and single hold-out splits. Extract, for each
study, whether `time` was an input, whether SMOTE was applied before splitting, and the
validation scheme; this becomes Supplementary Table S-Lit.

## D. Methodological references (verified bibliographic details recommended)

Kaufman et al. 2012 (*ACM TKDD*, leakage); Kapoor & Narayanan 2023 (*Patterns*, leakage
and reproducibility); Collins et al. 2024 (*BMJ*, TRIPOD+AI); Riley et al. 2019 (*Stat Med*)
and 2020 (*BMJ*, sample size); Nadeau & Bengio 2003 (*Mach Learn*); Harrell et al. 1996
(*Stat Med*); Uno et al. 2011 (*Stat Med*); Graf et al. 1999 (*Stat Med*); Van Calster et al.
2019 (*BMC Med*); Vickers et al. 2008 (*BMC Med Inform Decis Mak*); Johnson et al. 2016
(*Sci Data*, MIMIC-III); Sun et al. 2022 (*Front Cardiovasc Med*, ML vs statistical models
for HF events — meta-analysis finding no significant ML advantage).

## Sources consulted (search results)

- PLoS ONE (Ahmad 2017): https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181001
- PLoS ONE (Zahid 2019): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6380566/
- Chicco & Jurman 2020: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5 ; ResearchGate copy https://www.researchgate.net/publication/339008240
- Rane et al. 2026: https://arxiv.org/abs/2608.04140
- Özgür & Bekiroğlu 2025: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12465387/
- medRxiv fairness preprint: https://www.medrxiv.org/content/10.1101/2025.10.17.25338263v1
- Moreno-Sánchez: https://ieeexplore.ieee.org/document/9378460/ ; https://arxiv.org/abs/2108.10717
- Ishaq 2021: https://www.semanticscholar.org/paper/cbf287b7ed914e1cbd89bcc681ca866d5b0d19aa
- Newaz 2021: https://www.semanticscholar.org/paper/307e751bc291144daed4c29ef989495bd85ff089
- Karakuş & Er 2022: https://dblp1.uni-trier.de/rec/journals/nca/KarakusE22.html
- Oyewunmi 2023: https://scimedjournal.org/index.php/SMJ/article/view/489
- Talukder 2023: https://arxiv.org/pdf/2308.05765
- Qadri 2024: https://peerj.com/articles/cs-1894/
- Shamrat 2025: https://peerj.com/articles/cs-2682
- Sun 2022 meta-analysis: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9020815/
- UCI dataset page: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
- Example public repository excluding `time`: https://github.com/codefused2/Heart-Failure-Prediction-Project
