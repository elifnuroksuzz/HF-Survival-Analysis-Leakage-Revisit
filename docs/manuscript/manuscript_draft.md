---
title: "Follow-up time is not a predictor: target leakage, survival re-analysis and external validation of machine-learning mortality models in heart failure"
author: "[Authors, affiliations, ORCID — to be completed]"
date: "Draft v0.1 — generated from results/RESULTS_SUMMARY.md"
---

<!--
DRAFTING NOTES (delete before submission)
* Every number below is copied from results/RESULTS_SUMMARY.md (development cohort).
  If the analysis is re-run, update the numbers from that file, never from memory.
* [MIMIC: ...] marks results that must be filled from results/tables/mimic_*.csv after
  running scripts/run_external_validation.py with PhysioNet access.
* [VERIFY] marks statements that must be checked against the cited source.
* Verify every reference (volume, pages, DOI) before submission.
* Convert to Word:  pandoc manuscript_draft.md -o manuscript_draft.docx
-->

# Abstract

**Background.** A public cohort of 299 patients with heart failure is among the most widely used datasets for machine-learning mortality prediction. Many studies, including the influential analysis by Chicco and Jurman (2020), use the follow-up period (`time`) as an input feature. Follow-up ends at death or censoring, so this variable is determined by the outcome and is unknown when a prognosis is made.

**Methods.** We quantified target leakage with five experiments on 100 paired stratified train–test splits: follow-up by vital status, a `time`-only model, models with versus without `time`, a permutation control, and a fixed-horizon reformulation. A simulation study (100 synthetic cohorts per scenario) tested whether the leakage persists when no covariate affects survival. We then re-framed the task as time-to-event prediction from 11 baseline variables. We compared Cox proportional-hazards, random survival forest (RSF) and gradient-boosted Cox models using 10 × 5-fold repeated cross-validation and bootstrap optimism correction, with discrimination, calibration, decision-curve and subgroup analyses. Models were externally validated on [MIMIC: n] adults with a first heart-failure admission in MIMIC-III.

**Results.** With a random forest, the Matthews correlation coefficient (MCC) was 0.36 using baseline variables and 0.54 using `time` alone. Combining the two gave 0.64 (paired difference 0.28, 95% CI 0.15–0.41, p < 0.001), whereas a shuffled `time` gave no gain (0.36). In simulated cohorts with no covariate effect, baseline variables gave an MCC of 0.00 and `time` alone 0.48. Without leakage, Harrell's C-index was 0.70 for Cox, 0.72 for RSF and 0.73 for gradient-boosted Cox. Differences from Cox were not significant (all p > 0.19). A Cox model with ejection fraction and serum creatinine only reached 0.69. Out-of-fold 90-day risks were well calibrated (observed/expected 0.97–1.03) and gave a positive net benefit for thresholds of about 10–45%. In MIMIC-III, [MIMIC: C-index, calibration slope, effect of recalibration].

**Conclusions.** The high performance reported for this dataset largely reflects leakage from follow-up time. Leakage-free survival models achieve moderate discrimination (C ≈ 0.70), and a simple Cox model is not outperformed by flexible learners. Follow-up time should never be used as a predictor of an outcome that terminates follow-up.

**Keywords:** heart failure; survival analysis; target leakage; external validation; machine learning; MIMIC-III; TRIPOD+AI

# 1. Introduction

Heart failure affects tens of millions of people worldwide and carries high short-term mortality. Prognostic models help with risk communication, follow-up intensity and trial enrichment. In 2017 Ahmad et al. released data from 299 patients with left-ventricular systolic dysfunction treated in Faisalabad, Pakistan [1] [VERIFY centre and period]. Chicco and Jurman (2020) used these data to compare machine-learning classifiers and concluded that serum creatinine and ejection fraction alone can predict survival [2]. The dataset has since become a standard benchmark in public repositories and in the literature.

The dataset includes `time`, the number of days of follow-up. Follow-up ends either at death or at censoring. Patients who die early necessarily have short follow-up, while survivors are followed until the end of the study. Using `time` as a predictor of the death indicator is therefore a form of *target leakage*: information that is a consequence of the outcome, and is not available at prediction time, enters the model [3,4]. Leakage inflates apparent performance and is a recognised cause of irreproducible machine-learning findings in science [4].

The death indicator is also right-censored. About two thirds of patients were alive at the end of a follow-up of variable length, so "survived" is not a defined outcome. Treating the indicator as a binary label ignores censoring. The methodologically appropriate formulation is time-to-event analysis, with `time` used as the outcome rather than a feature.

This study had three aims. (i) To quantify the leakage introduced by `time` and to show by simulation that it is structural rather than dataset-specific. (ii) To estimate leakage-free prognostic performance with survival models, following the TRIPOD+AI reporting guideline [5]. (iii) To assess transportability by external validation in MIMIC-III, a large, independent, US hospital cohort [6].

# 2. Methods

## 2.1 Development cohort

The development cohort comprised all 299 patients (96 deaths, 32.1%) in the public dataset (UCI Machine Learning Repository; Kaggle `andrewmvd/heart-failure-clinical-data`). File integrity was verified with SHA-256. The data were de-identified and openly licensed, so no ethics approval was required. Eleven baseline variables were available: age, sex, anaemia, diabetes, hypertension, smoking, creatinine phosphokinase (CPK), ejection fraction (EF), platelets, serum creatinine and serum sodium. Follow-up ranged from 4 to 285 days. There were no missing values. The distributed file is sorted by follow-up time, so it was shuffled with a fixed seed before any analysis to prevent order-dependent artefacts.

## 2.2 Leakage experiments

All experiments used the same 100 stratified 70/30 train–test partitions. Random forest, logistic regression and gradient-boosting classifiers were fitted with pre-processing inside each training partition.

- **A — Distribution.** Follow-up was compared between patients who died and those censored.
- **B — `time` only.** Classifiers were trained on follow-up time alone.
- **C — With versus without `time`.** The 11 baseline variables, with and without `time`.
- **D — Permutation control.** `time` was permuted independently within the training and test partitions and added to the baseline variables. If adding `time` helped only because of an extra column, this control would also gain. If the gain comes from outcome-coupled values, it would not.
- **E — Fixed-horizon reformulation.** Death within 90 or 180 days was predicted from baseline variables only. Patients censored before the horizon were excluded, because their label is undefined.

The primary metric was the MCC; ROC AUC was secondary. Paired differences between feature sets were tested with the corrected resampled t-test of Nadeau and Bengio [7], which accounts for overlap between training sets.

For comparability with Chicco and Jurman [2], we also re-ran ten classifiers (random forest, decision tree, gradient boosting, logistic regression, one-rule stump, naive Bayes, linear and RBF support-vector machines, k-nearest neighbours, multilayer perceptron) on three feature sets: all variables including `time` (S1), baseline variables (S2), and EF plus serum creatinine (S3).

## 2.3 Simulation study

To test whether the leakage is structural, we generated synthetic cohorts of 299 patients. Covariate vectors were resampled from the real cohort. Event times were drawn from an exponential proportional-hazards model under two scenarios: no covariate effect (all log-hazard ratios 0), and effects equal to the standardised Cox estimates from the real data. Censoring times were drawn from the empirical censoring distribution of the real cohort. The baseline hazard was calibrated so that the event proportion matched the observed 32%. For each of 100 cohorts per scenario we computed 5-fold cross-validated MCC with baseline variables, `time` only, and both.

## 2.4 Survival models

We fitted four models:
1. a ridge-penalised Cox model (penalty 0.1) on the 11 baseline variables;
2. the same model on EF and serum creatinine only;
3. a random survival forest [8] with 500 trees, minimum leaf size 15 and √p candidate variables per split;
4. a gradient-boosted Cox model (200 trees of depth 2, learning rate 0.05, subsampling 0.8).

CPK, serum creatinine and platelets were log-transformed. All variables were standardised within training folds. Hyperparameters were fixed a priori to limit optimism given the small sample. DeepSurv [9] is implemented and can be run as a sensitivity analysis.

## 2.5 Internal validation and performance measures

Performance was estimated by 10 × repeated stratified 5-fold cross-validation, with stratification on the event indicator. Measures computed on each test fold:
- Harrell's C-index [10] and Uno's IPCW C-index truncated at 240 days [11];
- cumulative/dynamic time-dependent AUC at 30, 60, 90 and 180 days;
- the Brier score at the same horizons;
- the integrated Brier score (IBS) over 15–240 days [12], compared with a Kaplan–Meier null model.

Models were compared with Cox on identical folds using the corrected resampled t-test. Optimism-corrected C-indices and calibration slopes were also estimated by bootstrap with 200 resamples [10].

Calibration of out-of-fold 90- and 180-day risks was assessed by comparing the mean predicted risk with the Kaplan–Meier observed risk, overall (observed/expected ratio) and within risk quintiles [13]. Clinical utility was assessed by decision-curve analysis adapted for censored data, with event probabilities among high-risk patients estimated by Kaplan–Meier [14].

Hazard ratios were obtained from an unpenalised Cox model on the full cohort, with clinically interpretable scaling (per 10 years, per 5 EF points, per doubling of laboratory values). The proportional-hazards assumption was tested with scaled Schoenfeld residuals [15].

Variable importance was estimated as the out-of-fold decrease in C-index after permutation. SHAP values [16] were computed on held-out folds only. Subgroup performance by sex, age (< 65 / ≥ 65 years) and their intersections was estimated from out-of-fold risks with stratified bootstrap 95% CIs. Subgroups with fewer than 20 events were labelled exploratory. Sample-size adequacy was assessed with the criteria of Riley et al. [17,18].

## 2.6 External validation cohort (MIMIC-III)

We used MIMIC-III v1.4 [6] (Beth Israel Deaconess Medical Center, Boston, 2001–2012) under the PhysioNet credentialed data use agreement.

**Cohort definition.**
- Adults (≥ 18 years) whose first admission, ordered by admission time, carried an ICD-9 428.x diagnosis.
- Laboratory predictors: the first measurement between 6 hours before and 24 hours after admission.
- Comorbidities: ICD-9 codes recorded during the index admission (diabetes 250.x; hypertension 401–405; smoking 305.1, V15.82).
- Anaemia: haematocrit < 36%, the development-cohort definition [1] [VERIFY].
- EF: extracted with a rule-based parser from echocardiography reports of the index stay or, if none was available, from the preceding year. The parser's accuracy was evaluated against manual annotation of [MIMIC: n] reports by two annotators.

**Harmonisation.** Units were harmonised to the development cohort; in particular, platelets were converted from K/µL to /µL. Missing CPK was imputed with the development-cohort median, and a complete-case analysis was run as a sensitivity analysis.

**Endpoints.** The primary endpoint was all-cause death within 90 days of admission, taken from the date of death in the PATIENTS table, which combines hospital and Social Security records [VERIFY coverage]. In-hospital death was a secondary endpoint.

**Validation analysis.**
- Models fitted on the full development cohort were applied without re-estimation.
- We report Harrell's C, the 90-day AUC, calibration-in-the-large, calibration slope and the expected calibration error.
- We then report cross-fitted recalibration (intercept update, logistic recalibration, isotonic regression).
- Sensitivity analyses: numeric EF only, complete-case CPK, and age ≥ 40 years.

## 2.7 Software and reproducibility

Analyses used Python 3.13 with scikit-learn 1.9, scikit-survival 0.28 [19], lifelines 0.30 [20], statsmodels 0.15 and SHAP 0.54. The seed was 42 throughout. All code, unit tests, the result tables and an auto-generated results summary are publicly available at https://github.com/elifnuroksuzz/HF-Survival-Analysis-Leakage-Revisit [Zenodo DOI]. MIMIC-III data cannot be redistributed; the extraction SQL is provided.

# 3. Results

## 3.1 Cohort

Table 1 compares patients who died with those censored. Patients who died were older (median 65 vs 60 years), had lower EF (30% vs 38%), higher serum creatinine (1.3 vs 1.0 mg/dL) and lower serum sodium (135.5 vs 137 mEq/L; all p < 0.001). Kaplan–Meier curves separated by tertiles of serum creatinine (log-rank p < 0.001) and of EF (p = 0.019) (Figure 2a).

## 3.2 Follow-up time leaks the outcome

Median follow-up was 44 days (IQR 26–102) in patients who died and 172 days (IQR 95–213) in censored patients. 36.5% of deaths, but only 2.5% of censorings, occurred within 30 days (Figure 1a).

`time` alone achieved an MCC of 0.54–0.62 and a ROC AUC of 0.83–0.84 across classifiers, compared with 0.33–0.36 and 0.75–0.79 for the 11 baseline variables (Figure 1b). Adding `time` to the baseline variables raised the MCC by 0.28 (95% CI 0.15–0.41) for random forest, 0.28 (0.14–0.41) for gradient boosting and 0.23 (0.09–0.37) for logistic regression (all p ≤ 0.001). A permuted `time` produced no gain (ΔMCC −0.001 to 0.011, all p ≥ 0.81).

In the replication of Chicco and Jurman, all ten classifiers lost performance when `time` was removed. For random forest, MCC fell from 0.64 to 0.36 and ROC AUC from 0.91 to 0.79. The EF-plus-creatinine model reached an MCC of 0.41 (Supplementary Table S1). [VERIFY: compare with the values published in [2], transcribed into data/reference/.]

In the leakage-free fixed-horizon reformulation, ROC AUC was 0.66–0.71 for 90-day death (n = 254 eligible, 27.2% events) and 0.70–0.74 for 180-day death (n = 192, 46.4% events). Excluding censored patients discarded 15% and 36% of the cohort, respectively.

## 3.3 Simulation

When no covariate affected survival, baseline variables gave an MCC of −0.002 (2.5th–97.5th percentile −0.15 to 0.16), as expected. `time` alone, however, gave 0.48 (0.36–0.61), and baseline variables plus `time` 0.35 (0.21–0.50) (random forest; Supplementary Figure S1). Under the observed effects, `time` again outperformed the baseline variables (0.53 vs 0.22). Leakage from follow-up time is therefore a structural consequence of right-censored follow-up, independent of any prognostic signal.

## 3.4 Leakage-free survival models

Cross-validated Harrell's C was 0.70 for Cox (2.5th–97.5th percentile over folds 0.57–0.79), 0.69 for Cox with EF and creatinine (0.57–0.80), 0.72 for RSF (0.62–0.83) and 0.73 for gradient-boosted Cox (0.61–0.84) (Figure 3). Uno's C was 0.71–0.74, and the 180-day AUC 0.71–0.76.

The IBS was 0.16–0.17 for all models, compared with 0.19 for the Kaplan–Meier null model. No model differed significantly from Cox in Harrell's C (RSF +0.02, p = 0.34; gradient-boosted Cox +0.03, p = 0.20; EF-and-creatinine Cox −0.01, p = 0.59), Uno's C or IBS.

Bootstrap optimism was 0.03 for Cox, 0.06 for RSF and 0.07 for gradient-boosted Cox. Calibration slopes were 0.83 (95% interval 0.64–1.09), 0.85 (0.73–0.97) and 0.68 (0.50–0.83), respectively, which indicates overfitting of the boosted model.

Adjusted hazard ratios (Table 2):
- age: 1.54 per 10 years (95% CI 1.28–1.86);
- EF: 0.81 per 5 points (0.73–0.89);
- serum creatinine: 1.94 per doubling (1.47–2.58);
- anaemia: 1.60 (1.05–2.42);
- hypertension: 1.66 (1.09–2.53).

Schoenfeld tests suggested non-proportional hazards for EF and CPK (p < 0.05). In out-of-fold permutation importance, serum creatinine, EF and age were the only variables whose permutation reduced RSF discrimination (Figure 4).

## 3.5 Calibration, clinical utility and subgroups

Out-of-fold 90-day risks were well calibrated:
- mean predicted risk 23.1–24.4%, against an observed Kaplan–Meier risk of 23.7%;
- observed/expected ratio 0.97–1.03 at 90 days and 0.96–1.04 at 180 days.

All models gave a higher net benefit than treating all or no patients for 90-day threshold probabilities between approximately 0.10 and 0.43–0.60 (Figure 6).

Cox C-index was similar in women (0.69, 95% CI 0.59–0.78) and men (0.71, 0.64–0.77), and in patients aged < 65 years (0.67, 0.60–0.75) and ≥ 65 years (0.69, 0.61–0.76). Intersectional subgroups had overlapping intervals (Table 3).

The apparent Cox–Snell R² was 0.24. The minimum sample size by Riley's criteria was 351, against 299 available, with an expected shrinkage of 0.87.

## 3.6 External validation in MIMIC-III

[MIMIC: attrition (flow diagram from mimic_primary_attrition.csv); final n, 90-day deaths; Table 1 comparison.]

[MIMIC: parser accuracy — detection sensitivity/PPV, mean absolute error, Bland–Altman limits, category agreement.]

[MIMIC: Harrell's C and 90-day AUC per model with 95% CI; AUC for in-hospital death.]

[MIMIC: calibration before and after recalibration (calibration-in-the-large, slope, ECE); Figure 5.]

[MIMIC: sensitivity analyses; subgroup table.]

# 4. Discussion

We show that the follow-up variable in a widely used heart-failure dataset leaks the outcome. The leakage accounts for most of the high classification performance reported for this dataset. Without leakage, survival models built from baseline variables achieve moderate discrimination (C ≈ 0.70) and good internal calibration. Simple Cox models perform as well as flexible machine-learning survival models.

Four lines of evidence support the leakage interpretation:
1. `time` alone is as predictive as all clinical variables combined, or more so.
2. Its contribution vanishes when its values are decoupled from the outcome by permutation.
3. A model using `time` reaches an MCC near 0.5 in simulated cohorts in which no variable carries prognostic information.
4. The mechanism is self-evident from the study design: follow-up stops at death.

Chicco and Jurman also reported a follow-up-time-adjusted analysis [2] [VERIFY wording and table]. Our results show that any model taking `time` as input is unsuitable for prospective use, whatever the classifier. This conclusion extends to the many subsequent studies that used the full feature set [VERIFY: a systematic count of such studies would substantially strengthen this point].

Our leakage-free estimates agree with established heart-failure risk scores, whose C-statistics typically lie between 0.65 and 0.75 [VERIFY with references, e.g. MAGGIC, Seattle HF Model]. Serum creatinine, EF and age carried almost all of the discriminative signal. This supports the parsimony argument of Chicco and Jurman, now on a sound methodological footing: the two-variable Cox model was not significantly worse than the 11-variable model.

Flexible learners did not improve discrimination significantly. The boosted model was clearly overfitted (calibration slope 0.68), consistent with the sample-size calculation showing that 299 patients with 96 events fall below the recommended minimum for 11 candidate parameters.

[MIMIC: discuss transportability — case-mix differences (older, US, hospitalised patients), EF from text, ICD-coded comorbidities, endpoint definition; the role of recalibration before local use.]

**Strengths.**
- Complete, tested and public code, with all pre-processing nested within resampling.
- Multiple complementary leakage tests, including simulation.
- Censoring-aware evaluation (time-dependent AUC, IBS, Kaplan–Meier calibration and decision curves).
- Bootstrap optimism correction.
- Reporting according to TRIPOD+AI [5].

**Limitations.**
- The development cohort is small, single-centre and below the recommended sample size; the CIs are correspondingly wide.
- Hyperparameters were fixed rather than tuned, which favours stability over peak performance.
- The proportional-hazards assumption was questionable for EF and CPK.
- In MIMIC-III, EF was derived from text, comorbidities from billing codes recorded during the index admission, and out-of-hospital deaths from administrative records.
- The populations differ substantially in age, setting and era.

# 5. Conclusions

Follow-up time must not be used as a predictor of an outcome that ends follow-up. For the 299-patient heart-failure cohort, correctly specified survival models reach a C-index of about 0.70, and a two-variable Cox model is a reasonable, interpretable choice. Researchers using this or similar datasets should report time-to-event analyses with censoring-aware metrics and external validation.

# Declarations

**Ethics.** Development data: public and de-identified. MIMIC-III: PhysioNet credentialed access [MIMIC: certificate details].
**Data and code availability.** https://github.com/elifnuroksuzz/HF-Survival-Analysis-Leakage-Revisit [Zenodo DOI].
**Competing interests.** [To be completed.] **Funding.** [To be completed.] **Author contributions.** [CRediT roles.]

# References

<!-- [VERIFY] every reference before submission -->

1. Ahmad T, Munir A, Bhatti SH, Aftab M, Raza MA. Survival analysis of heart failure patients: a case study. *PLoS ONE*. 2017;12(7):e0181001.
2. Chicco D, Jurman G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Med Inform Decis Mak*. 2020;20:16. doi:10.1186/s12911-020-1023-5
3. Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation, detection, and avoidance. *ACM Trans Knowl Discov Data*. 2012;6(4):15.
4. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*. 2023;4(9):100804.
5. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378.
6. Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. *Sci Data*. 2016;3:160035.
7. Nadeau C, Bengio Y. Inference for the generalization error. *Mach Learn*. 2003;52:239–281.
8. Ishwaran H, Kogalur UB, Blackstone EH, Lauer MS. Random survival forests. *Ann Appl Stat*. 2008;2(3):841–860.
9. Katzman JL, Shaham U, Cloninger A, et al. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med Res Methodol*. 2018;18:24.
10. Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. *Stat Med*. 1996;15(4):361–387.
11. Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. *Stat Med*. 2011;30(10):1105–1117.
12. Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and comparison of prognostic classification schemes for survival data. *Stat Med*. 1999;18(17–18):2529–2545.
13. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med*. 2019;17:230.
14. Vickers AJ, Cronin AM, Elkin EB, Gonen M. Extensions to decision curve analysis, a novel method for evaluating diagnostic tests, prediction models and molecular markers. *BMC Med Inform Decis Mak*. 2008;8:53.
15. Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika*. 1994;81(3):515–526.
16. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst*. 2017;30.
17. Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable prediction model: part II — binary and time-to-event outcomes. *Stat Med*. 2019;38(7):1276–1296.
18. Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. *BMJ*. 2020;368:m441.
19. Pölsterl S. scikit-survival: a library for time-to-event analysis built on top of scikit-learn. *J Mach Learn Res*. 2020;21(212):1–6.
20. Davidson-Pilon C. lifelines: survival analysis in Python. *J Open Source Softw*. 2019;4(40):1317.

# Figure and table plan

| Item | File | Content |
|---|---|---|
| Table 1 | `results/tables/table1_cohort_by_outcome.csv` | Baseline characteristics by outcome |
| Table 2 | `results/tables/table2_cox_hazard_ratios.csv` | Adjusted hazard ratios |
| Table 3 | `results/tables/table3_subgroup_cindex.csv` | Subgroup C-index |
| Table 4 | `results/tables/mimic_primary_performance.csv` | External validation [MIMIC] |
| Figure 1 | `results/figures/fig1_time_leakage.pdf` | (a) follow-up by outcome; (b) MCC by feature set |
| Figure 2 | `fig2a_km_tertiles.pdf`, `fig2b_cox_forest.pdf` | KM by tertiles; HR forest plot |
| Figure 3 | `fig3_survival_cindex.pdf` | C-index by model |
| Figure 4 | `fig4_rsf_importance.pdf` (+ `fig4b_rsf_shap_beeswarm.pdf`) | Out-of-fold importance |
| Figure 5 | `fig5_mimic_calibration.pdf` | External calibration before/after recalibration [MIMIC] |
| Figure 6 | `fig6_decision_curve_90d.pdf` | Decision curves |
| Suppl. | `fig_s_simulation_leakage.pdf`, `fig_s_calibration_oof_90d.pdf`, replication and PH tables | — |
