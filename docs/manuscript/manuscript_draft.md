---
title: "Follow-up time is not a predictor: target leakage in a widely used heart-failure benchmark, its prevalence in public code, and leakage-free survival models with external validation"
author: "[Authors, affiliations, ORCID — to be completed]"
date: "Draft v0.3 — numbers from results/RESULTS_SUMMARY.md"
---

<!--
DRAFTING NOTES (delete before submission)
* Every number is copied from results/RESULTS_SUMMARY.md. Re-copy after any re-run.
* [MIMIC: ...] = fill from results/tables/mimic_*.csv after scripts/run_external_validation.py.
* [VERIFY] = check against the full text of the cited source. Details of several
  references come from search-engine snippets (publisher sites were not reachable
  when drafting); see docs/manuscript/literature_review.md for the status of each.
* Word: pandoc manuscript_draft.md -o manuscript_draft.docx
* Target length ≈ 4500 words + 6 figures/tables; check the target journal's limits.
-->

# Abstract

**Background.** A public cohort of 299 patients with heart failure (Ahmad et al., 2017) has become a standard benchmark for machine-learning mortality prediction. It was popularised by Chicco and Jurman (2020). The data include the follow-up period (`time`), which ends at death or censoring. This variable is therefore produced by the outcome and is unknown when a prognosis is made.

**Objectives.** (i) To quantify the target leakage caused by using `time` as a predictor and to establish its mechanism. (ii) To measure how often public code uses `time` as a predictor. (iii) To estimate leakage-free prognostic performance with survival models and test it in an independent cohort.

**Methods.**
- *Leakage.* Five experiments on 100 paired stratified splits: follow-up by vital status, a `time`-only model, models with versus without `time`, a permutation control, and a fixed-horizon reformulation. A simulation in which no covariate affects survival (100 cohorts per scenario).
- *Code audit.* Code search of GitHub with rule-based static classification, verified manually on two random samples.
- *Replication.* Ten classifiers re-run and compared with the published values.
- *Literature review.* A PRISMA-guided systematic review of the published studies.
- *Survival models.* Cox, random survival forest (RSF), gradient-boosted Cox and DeepSurv, evaluated with 10 × 5-fold cross-validation, nested tuning and bootstrap optimism correction. Calibration, decision-curve and subgroup analyses.
- *External validation.* Applied in MIMIC-III ([MIMIC: n]).

**Results.**
- *Replication.* Without `time`, we reproduced the published performance: random forest MCC 0.36 vs 0.38 published; ROC AUC 0.79 vs 0.80.
- *Leakage.* With a random forest, MCC was 0.36 with baseline variables, 0.54 with `time` alone and 0.64 with both (Δ 0.28, 95% CI 0.15–0.41). Shuffling `time` removed the gain (0.36). In simulated cohorts with no covariate effect, `time` alone still gave an MCC of 0.48.
- *Code audit.* Of 729 public repositories that fit a model to these data, 68.2% (95% CI 64.7–71.5) used `time` as a predictor. 8.8% excluded it, and 1.1% used survival methods. Rule-based classification agreed with manual review in 93.3% of an independent sample.
- *Literature review.* [SR: proportion of published studies using `time`.]
- *Survival models.* Without leakage, Harrell's C was 0.70 (Cox), 0.72 (RSF), 0.73 (gradient-boosted Cox) and 0.69 (DeepSurv). Differences were not significant and persisted after nested tuning. A two-variable Cox model reached 0.69. Out-of-fold 90-day risks were well calibrated (observed/expected 0.97–1.03).
- *External validation.* In MIMIC-III, [MIMIC: C-index, calibration slope, effect of recalibration].

**Conclusions.** Most of the high performance reported for this benchmark reflects leakage from follow-up time, and the error is common in public code. Correctly specified survival models give moderate discrimination, and a simple Cox model is not outperformed by flexible learners.

**Keywords:** heart failure; survival analysis; target leakage; reproducibility; external validation; machine learning; MIMIC-III; TRIPOD+AI

# 1. Introduction

Heart failure carries high short-term mortality, and prognostic models are used for risk communication, care planning and trial enrichment. Machine-learning (ML) models are often presented as superior to regression for this task. However, a meta-analysis of heart-failure event prediction found no significant advantage of ML over statistical models [1].

In 2017 Ahmad et al. published records of 299 patients with left-ventricular systolic dysfunction (NYHA class III–IV). The patients were treated at the Institute of Cardiology and Allied Hospital, Faisalabad, Pakistan, between April and December 2015, and the study analysed survival with Cox regression [2]. The same group later developed sex-specific Cox models on these data (C-index 0.72–0.77) [3]. After Chicco and Jurman (2020) showed that serum creatinine and ejection fraction suffice to predict survival with ML classifiers [4], the dataset became a popular benchmark on the UCI Machine Learning Repository and Kaggle.

The dataset contains `time`, the number of days of follow-up until death or censoring. Patients who die early necessarily have short follow-up, whereas survivors are followed until the end of the study. Using `time` to predict the death indicator is therefore *target leakage*: information that is a consequence of the outcome, and unavailable at prediction time, enters the model [5,6]. Leakage is a leading cause of irreproducible ML findings across scientific fields [6].

Chicco and Jurman's main analysis excluded follow-up time; they examined it only in a secondary analysis [4] [VERIFY wording]. Many later studies, however, report accuracies of 0.83–1.00 and rank follow-up time as the most important feature [7–13]. The death indicator is also right-censored: two thirds of patients were alive at the end of a follow-up of variable length. Treating "survived" as a class label therefore ignores censoring.

Three recent works point in the same direction:
- A 2026 preprint argues that using follow-up as an input causes severe target leakage and proposes discrete-time survival models [14].
- A 2025 study compares Cox regression and random survival forests on these data in relation to follow-up adequacy [15].
- A 2025 preprint evaluates models trained on these data in MIMIC-IV and reports lower performance [16].

This study therefore does not claim that the leakage is new; Rane et al. have described it [14]. Our aim is to establish *why* it occurs, *how widespread* it is, and *what remains* once it is removed. Relative to existing work, we contribute:

1. **Mechanism.** We isolate the leakage with a `time`-only model, a permutation control and a simulation in which no covariate affects survival. The simulation shows that the inflation follows from right-censored follow-up itself and is not specific to this cohort.
2. **Prevalence.** We give the first quantitative estimate of how often public code uses `time` as a predictor, verified manually on an independent sample, together with a PRISMA-guided systematic review of the published studies.
3. **Replication.** We reproduce the original benchmark against its published values, showing that the original main analysis was sound and that the leakage arose in later reuse of the dataset.
4. **Leakage-free benchmark.** We provide continuous-time survival models (Cox, RSF, gradient-boosted Cox, DeepSurv) with nested tuning, bootstrap optimism correction, censoring-aware calibration and decision curves. Unlike the discrete-time framework of [14], these give survival curves directly.
5. **Transportability.** We externally validate these models in MIMIC-III with cross-fitted recalibration.

All code is released with unit tests. The study is reported following TRIPOD+AI [17].

# 2. Methods

## 2.1 Development cohort

We used all 299 patients (96 deaths, 32.1%) in the public dataset (UCI Machine Learning Repository record 519; Kaggle `andrewmvd/heart-failure-clinical-data`). File integrity was verified by SHA-256.

The eleven baseline variables were:
- age, sex;
- anaemia, diabetes, hypertension, smoking;
- creatinine phosphokinase (CPK), ejection fraction (EF), platelets, serum creatinine and serum sodium.

Follow-up ranged from 4 to 285 days, and there were no missing values. The distributed file is sorted by follow-up time, so it was shuffled with a fixed seed before analysis. The data are de-identified and openly licensed; no ethics approval was required.

## 2.2 Replication and leakage experiments

All classification experiments used the same 100 stratified 70/30 partitions, with pre-processing fitted on training data only.

**Replication.** Ten classifiers mirroring the original study were re-run on three feature sets:
- **S1:** all variables including `time`;
- **S2:** the 11 baseline variables;
- **S3:** EF and serum creatinine.

The classifiers were random forest, decision tree, gradient boosting, logistic regression, a one-rule stump, naive Bayes, linear and RBF support-vector machines, k-nearest neighbours and a multilayer perceptron. Results were compared with the published tables [4].

**Leakage experiments.** Random forest, logistic regression and gradient boosting were used.
- **(A) Distribution:** follow-up time in patients who died vs those censored.
- **(B) `time` only:** classifiers trained on follow-up time alone.
- **(C) With vs without `time`:** baseline variables with and without `time`.
- **(D) Permutation control:** `time` permuted independently within the training and test partitions.
- **(E) Fixed horizon:** death within 90 or 180 days predicted from baseline variables, excluding patients censored before the horizon.

The primary metric was the Matthews correlation coefficient (MCC). Paired differences were tested with the corrected resampled t-test [18].

## 2.3 Simulation study

Synthetic cohorts of 299 patients were generated by resampling covariates from the real data. Event times were exponential under two scenarios: no covariate effect, and standardised effects equal to the Cox estimates from the real data. Censoring times were drawn from the empirical censoring distribution, and the baseline hazard was tuned to give a 32% event proportion. For each of 100 cohorts per scenario we computed 5-fold cross-validated MCC with baseline variables, with `time` only, and with both.

## 2.4 Audit of public code

We searched GitHub on 4 November 2026 for Jupyter notebooks and Python files containing the dataset file name, using the REST code-search API, which returns up to 1,000 results per query. Files were downloaded but never executed. Notebook code cells were extracted and comments removed.

Transparent rules (released with the code) assigned each file to one of five categories:
- `time` used as a predictor (e.g. all columns except the outcome, `iloc[:, :-1]`, explicit lists or formulas containing `time`);
- `time` explicitly excluded;
- both;
- survival analysis;
- unresolved.

Repositories were classified by their most leakage-prone file; forks were excluded. Proportions are reported with Wilson 95% CIs among repositories that fit a model. Two sensitivity analyses were run: de-duplication of identical files and restriction to resolved repositories.

The rules were checked against manual review on two random samples: a development sample of 40 repositories, on which the rules were refined, and an independent sample of 30 repositories, with the rules frozen.

The code audit complements, but does not replace, the review of the published literature (Section 2.5). Public code is a much larger and largely unreviewed body of reuse. It shows how the dataset is used in practice, including in teaching material.

## 2.5 Systematic review of published studies

The review followed PRISMA 2020 [32]; the protocol was registered at [OSF: ID] before screening. We searched:
- OpenAlex for all works citing Ahmad et al. [2] or Chicco and Jurman [4], and for the phrase "heart failure clinical records";
- Europe PMC with an equivalent query.

The search date was [SR: date]. Studies were eligible if they fitted at least one mortality-prediction model to this dataset and reported a performance measure.

Two reviewers independently screened records and extracted, for each study:
- whether `time` was an input feature, with a supporting quotation;
- the handling of censoring;
- oversampling and whether it preceded the train–test split;
- feature selection on the full data;
- the validation scheme;
- the best reported accuracy, MCC and ROC AUC.

Candidate values were pre-filled by text mining of open-access full texts and confirmed or corrected by the reviewers. Risk of bias was assessed with PROBAST [33]. We report the proportion of studies using `time` (Wilson 95% CI) and compare reported performance between studies that did and did not use it.

## 2.6 Survival models and internal validation

We fitted five models:
1. a ridge-penalised Cox model (penalty 0.1) on the 11 baseline variables;
2. the same model on EF and serum creatinine;
3. a random survival forest [19] with 500 trees, minimum leaf size 15 and √p candidate variables per split;
4. a gradient-boosted Cox model (200 trees of depth 2, learning rate 0.05, subsampling 0.8);
5. DeepSurv [34]: two hidden layers (32, 16), dropout 0.3, weight decay 10⁻⁴, and early stopping on an event-stratified random 20% of each training fold, with a Breslow baseline hazard.

CPK, creatinine and platelets were log-transformed and all variables standardised within training folds.

Hyperparameters of the main models were fixed a priori. As a sensitivity analysis, the Cox penalty, RSF leaf size and candidate fraction, and the gradient-boosting learning rate and depth were tuned by nested cross-validation: 3-fold inner loops, 3 × 5-fold outer loops, selection by Harrell's C.

Because Schoenfeld tests suggested non-proportional hazards, we also fitted a Cox model with follow-up split at 60 days. In this model the effects of EF and CPK were estimated separately for each period.

Performance was estimated by 10 × repeated stratified 5-fold cross-validation. Measures:
- Harrell's C [20] and Uno's C truncated at 240 days [21];
- cumulative/dynamic time-dependent AUC at 30, 60, 90 and 180 days;
- Brier scores, and the integrated Brier score over 15–240 days [22] against a Kaplan–Meier null model.

Models were compared with Cox on identical folds using the corrected resampled t-test. Bootstrap optimism correction and calibration slopes used 200 resamples [20].

Out-of-fold 90- and 180-day risks were compared with Kaplan–Meier observed risks, overall and by quintile [23]. Clinical utility was assessed with decision curves adapted for censoring [24]. Hazard ratios were estimated from an unpenalised Cox model, and proportional hazards were tested with Schoenfeld residuals [25].

Importance was estimated by out-of-fold permutation (drop in C-index); SHAP values [26] were computed on held-out folds. Subgroup C-indices (sex, age < 65 / ≥ 65 years, and their intersections) were obtained from out-of-fold risks with bootstrap CIs. Sample-size adequacy followed Riley et al. [27,28].

## 2.7 External validation (MIMIC-III)

**Rationale.** This is a test of *transportability*. The development cohort (ambulatory patients with severe systolic heart failure in Pakistan, 2015) and MIMIC-III (patients hospitalised in a US tertiary centre, 2001–2012) differ in setting, case mix and era. We therefore expected discrimination to be partly preserved and calibration to be lost, and pre-specified recalibration as part of the analysis.

**Data.** MIMIC-III v1.4 [29] under the PhysioNet credentialed data use agreement.

**Cohort.**
- Adults whose first admission (ordered by admission time) carried an ICD-9 428.x code.
- Laboratory values: the first measurement from 6 h before to 24 h after admission.
- Comorbidities: ICD-9 codes recorded during the index admission.
- Anaemia: haematocrit < 36%, the definition used for the development cohort [3] [VERIFY in [2]].
- EF: extracted from echocardiography reports of the index stay or the preceding year with a rule-based parser. The parser was validated against two annotators on [MIMIC: n] reports.

**Harmonisation.** Units were harmonised (platelets ×1000). Missing CPK was imputed with the development-cohort median, with complete-case analysis as a sensitivity analysis.

**Endpoints.** Primary: all-cause death within 90 days of admission (PATIENTS.DOD, which combines hospital and Social Security records [VERIFY coverage]). Secondary: in-hospital death.

**Analysis.**
- Models fitted on the full development cohort were applied without re-estimation.
- We report Harrell's C, the 90-day AUC, calibration-in-the-large, calibration slope and the expected calibration error.
- Recalibration was cross-fitted: intercept update, logistic recalibration and isotonic regression.
- Sensitivity analyses: numeric EF only, complete-case CPK, and age ≥ 40 years.

## 2.8 Software and availability

Python 3.13, scikit-learn 1.9, scikit-survival 0.28 [30], lifelines 0.30 [31], statsmodels 0.15 and SHAP 0.54; seed 42. Code, 49 unit tests, all result tables, the audit classifications and an auto-generated results summary are available at https://github.com/elifnuroksuzz/HF-Survival-Analysis-Leakage-Revisit [Zenodo DOI]. MIMIC-III data cannot be shared; the extraction SQL is provided.

# 3. Results

## 3.1 Cohort

Patients who died were older (median 65 vs 60 years), had lower EF (30% vs 38%), higher serum creatinine (1.3 vs 1.0 mg/dL) and lower serum sodium (135.5 vs 137 mEq/L) than censored patients (all p < 0.001; Table 1). Kaplan–Meier curves separated by tertiles of serum creatinine (log-rank p < 0.001) and EF (p = 0.019; Figure 2a).

## 3.2 Replication of the benchmark

Without follow-up time (S2), our random forest reproduced the published values [4] [VERIFY]:

| Measure | This study | Published |
|---|---|---|
| MCC | 0.358 | 0.384 |
| F1 | 0.520 | 0.547 |
| Accuracy | 0.735 | 0.740 |
| ROC AUC | 0.786 | 0.800 |

With EF and creatinine only (S3), MCC was 0.407 vs 0.418 published.

All ten classifiers performed better when `time` was added (S1). For random forest, MCC rose to 0.637 and ROC AUC to 0.909 (Supplementary Table S1). Across classifiers, the S1–S2 difference in MCC ranged from 0.16 to 0.33 (all p ≤ 0.016).

## 3.3 Follow-up time leaks the outcome

Median follow-up was 44 days (IQR 26–102) in patients who died and 172 days (95–213) in censored patients. 36.5% of deaths, but only 2.5% of censorings, occurred within 30 days (Figure 1a).

With `time` alone, MCC was 0.54–0.62 and ROC AUC 0.83–0.84. With the 11 baseline variables, MCC was 0.33–0.36 and ROC AUC 0.75–0.79 (Figure 1b). Adding `time` to the baseline variables increased MCC by 0.28 (95% CI 0.15–0.41) for random forest, 0.28 (0.14–0.41) for gradient boosting and 0.23 (0.09–0.37) for logistic regression (all p ≤ 0.001). A permuted `time` produced no gain (ΔMCC −0.001 to 0.011; all p ≥ 0.81).

In the leakage-free fixed-horizon formulation, ROC AUC was 0.66–0.71 at 90 days and 0.70–0.74 at 180 days. Excluding censored patients removed 15% and 36% of the cohort, respectively.

## 3.4 Simulation

When no covariate affected survival, baseline variables gave an MCC of −0.002 (2.5th–97.5th percentile −0.15 to 0.16). `time` alone gave 0.48 (0.36–0.61), and baseline variables plus `time` gave 0.35 (0.21–0.50) (random forest; Supplementary Figure S1). With the observed effects, `time` alone again exceeded the baseline variables (0.53 vs 0.22). The apparent signal of `time` is therefore generated by censored follow-up itself.

## 3.5 Prevalence of the error in public code

The search returned 1,579 files from 979 non-fork repositories; 729 repositories fitted a model (Figure 7).

| Category | Repositories | Share (95% CI) |
|---|---|---|
| Used `time` as a predictor in at least one analysis | 497 | 68.2% (64.7–71.5) |
| — of which with and without `time` | 38 | 5.2% |
| Excluded `time` | 64 | 8.8% (6.9–11.1) |
| Used survival methods | 8 | 1.1% (0.6–2.2) |
| Feature set unresolved | 160 | 21.9% |

After de-duplicating identical files, the share using `time` was 68.9% (65.3–72.2). Among repositories with a resolved feature set it was 87.3% (84.4–89.8).

On the independent verification sample (n = 30), the rules agreed with manual review on whether `time` was used in 93.3% of repositories (95% CI 78.7–98.2), with a positive predictive value of 95.8% and a sensitivity of 95.8%. One repository was misclassified in each direction. One of the repositories using `time` as a predictor was itself titled as a "leakage-prevention" pipeline.

In the published literature, several studies on these data ranked follow-up time as the most important predictor [7,8,12]. Reported accuracies ranged from 0.83 to 1.00 [7–13] (Supplementary Table S-Lit) [VERIFY each study's use of `time`, oversampling and validation].

## 3.6 Systematic review of published studies

[SR: PRISMA flow (records identified / screened / full texts / included); number of studies using `time` (Wilson 95% CI); reported accuracy and MCC by use of `time` (median, IQR); prevalence of oversampling before splitting and of single hold-out validation; PROBAST summary.]

## 3.7 Leakage-free survival models

| Model | Harrell's C (2.5th–97.5th percentile over folds) |
|---|---|
| Cox | 0.70 (0.57–0.79) |
| Cox, EF + creatinine | 0.69 (0.57–0.80) |
| RSF | 0.72 (0.62–0.83) |
| Gradient-boosted Cox | 0.73 (0.61–0.84) |

Uno's C was 0.71–0.74 and the 180-day AUC 0.71–0.76 (Figure 3). The IBS was 0.16–0.17, against 0.19 for the Kaplan–Meier null model. No model differed significantly from Cox (Harrell's C: RSF +0.02, p = 0.34; gradient-boosted Cox +0.03, p = 0.20; EF + creatinine −0.01, p = 0.59).

Bootstrap optimism was 0.03 (Cox), 0.06 (RSF) and 0.07 (gradient-boosted Cox). Calibration slopes were 0.83 (0.64–1.09), 0.85 (0.73–0.97) and 0.68 (0.50–0.83), respectively.

Adjusted hazard ratios (Table 2):
- age: 1.54 per 10 years (1.28–1.86);
- EF: 0.81 per 5 points (0.73–0.89);
- creatinine: 1.94 per doubling (1.47–2.58);
- anaemia: 1.60 (1.05–2.42);
- hypertension: 1.66 (1.09–2.53).

DeepSurv reached a Harrell's C of 0.69 (0.58–0.80) on the same folds, which did not differ from Cox (−0.006, 95% CI −0.039 to 0.026, p = 0.71). With hyperparameters tuned by nested cross-validation, results were unchanged: Cox 0.70, RSF 0.72 and gradient-boosted Cox 0.72.

Non-proportional hazards were suggested for EF and CPK. In the period-specific model, the protective association of EF strengthened after day 60: HR per 5 points 0.89 (0.79–1.00) up to day 60 and 0.67 (0.55–0.80) thereafter (interaction p = 0.007). CPK showed no time-varying effect (p = 0.38). All other hazard ratios were essentially unchanged. Serum creatinine, EF and age were the only variables whose permutation reduced RSF discrimination (Figure 4).

## 3.8 Calibration, clinical utility and subgroups

Mean out-of-fold 90-day risk was 23.1–24.4%, against an observed 23.7% (observed/expected 0.97–1.03; 0.96–1.04 at 180 days). All models gave a higher net benefit than treat-all or treat-none for thresholds of about 0.10 to 0.43–0.60 (Figure 6).

Cox C-index was similar in women (0.69, 95% CI 0.59–0.78) and men (0.71, 0.64–0.77), and in patients aged < 65 years (0.67, 0.60–0.75) and ≥ 65 years (0.69, 0.61–0.76) (Table 3).

Riley's criteria required 351 patients, against 299 available; the expected shrinkage was 0.87.

## 3.9 External validation in MIMIC-III

[MIMIC: flow diagram; cohort characteristics vs development cohort; EF parser accuracy; C-index and 90-day AUC with 95% CI; calibration before and after recalibration (Figure 5); sensitivity analyses; subgroups.]

# 4. Discussion

We report three findings. First, the follow-up variable in this widely used heart-failure benchmark leaks the outcome. The leakage explains most of the performance gap between reported results and what is achievable at the point of prognosis. Second, the error is pervasive: about two thirds of public code fitting models to these data uses follow-up time as a predictor, and only about 1% uses survival methods. Third, correctly specified survival models reach a C-index of about 0.70 with good internal calibration, and a simple Cox model is not improved upon by flexible learners.

**Mechanism.** The evidence for leakage is convergent:
1. Follow-up alone predicts death as well as, or better than, all clinical variables.
2. Its contribution vanishes when its values are decoupled from the outcome.
3. It produces an MCC near 0.5 in simulated cohorts in which nothing is prognostic.

The simulation is important. It shows that the inflated performance is not a property of this cohort but a structural consequence of right-censored follow-up. Any dataset whose follow-up ends at the event is exposed to the same problem.

**Relation to prior work.** Chicco and Jurman excluded follow-up time from their main analysis, and our replication matched their published results closely. The leakage therefore arises mainly in later reuse of the dataset, where the full feature set is the default.

The discrete-time analysis of Rane et al. [14] first made the conceptual point and reports a much larger gain from follow-up (MCC ≈ 0.35–0.40 → ≈ 0.94–0.98 [VERIFY]). Our smaller gain (≈ 0.28) may reflect differences in models and validation [VERIFY]. Özgür and Bekiroğlu [15] show that follow-up adequacy affects survival-model performance on these data. The medRxiv study [16] finds that performance declines in MIMIC-IV [VERIFY endpoint and whether `time` was used].

Our study is complementary. It explains *why* the inflation occurs, using the permutation control and a null-effect simulation. It measures *how often* the error occurs, in public code and in the published literature. And it establishes *what remains* once the error is removed: replication against published values, continuous-time survival models with nested tuning and optimism correction, and external validation with recalibration. Our leakage-free estimates (C ≈ 0.70) agree with the sex-specific Cox models of Zahid et al. (0.72–0.77) [3]. They also agree with the meta-analytic finding that ML offers no consistent advantage for heart-failure events [1].

**Flexible models.** Neither tuning, nor tree ensembles, nor a neural network improved on a penalised Cox model. This echoes meta-analytic evidence for heart-failure prediction [1]. It is also expected with 96 events, where the variance of flexible learners dominates any gain from modelling non-linearity. The time-varying EF effect is clinically plausible: the contribution of systolic function to mortality appears to grow beyond the early post-enrolment period. It should, however, be confirmed in larger cohorts.

**Implications.** For researchers using this or similar datasets:
- Treat follow-up time as part of the outcome, never as a predictor.
- Report time-to-event metrics (C-index, time-dependent AUC, IBS) and calibration.
- Include a permutation or `time`-only control whenever a duration-like variable is present.
- Release code, so that this class of error can be detected.

For reviewers and editors, a request to state the role of every time-related variable would prevent most instances. Dataset curators could flag the variable in the dataset description on UCI and Kaggle.

**Limitations.**
- *Development cohort.* Small, single-centre, and below the recommended sample size; the boosted model was overfitted (calibration slope 0.68).
- *Proportional hazards.* The assumption was questionable for EF; a period-specific model addressed this, without changing discrimination-based conclusions.
- *Audit.* Limited to code returned by GitHub search; Kaggle notebooks and non-indexed code were not covered. 22% of repositories could not be resolved by static rules, and duplication across course materials inflates counts, although de-duplication did not change the estimate. Code may also not correspond to reported results.
- *Literature.* The systematic review is limited to indexed and preprint literature in English, and extraction depends on how completely studies report their feature sets.
- *MIMIC-III.* EF from text, comorbidities from billing codes, out-of-hospital deaths from administrative records, and substantial case-mix differences.

# 5. Conclusions

Follow-up time must not be used to predict an outcome that terminates follow-up. In the 299-patient heart-failure benchmark this error inflates reported performance, is present in most public code, and disappears when the task is formulated as survival analysis. Correctly specified models reach a C-index of about 0.70, and a simple Cox model is a reasonable, interpretable choice.

# Declarations

**Ethics.** Public de-identified development data; MIMIC-III under PhysioNet credentialed access [MIMIC: certificate]. The audit analysed publicly available code only; repositories are cited by name in the supplementary data, and no code was executed.
**Data and code availability.** https://github.com/elifnuroksuzz/HF-Survival-Analysis-Leakage-Revisit [Zenodo DOI].
**Competing interests / Funding / Author contributions.** [To be completed.]

# References

<!-- Bibliographic details marked [VERIFY] were taken from search-engine snippets. -->

1. Sun Z, Dong W, Shi H, Ma H, Cheng L, Huang Z. Comparing machine learning models and statistical models for predicting heart failure events: a systematic review and meta-analysis. *Front Cardiovasc Med*. 2022;9:812276.
2. Ahmad T, Munir A, Bhatti SH, Aftab M, Raza MA. Survival analysis of heart failure patients: a case study. *PLoS ONE*. 2017;12(7):e0181001.
3. Zahid FM, Ramzan S, Faisal S, Hussain I. Gender based survival prediction models for heart failure patients: a case study in Pakistan. *PLoS ONE*. 2019;14(2):e0210602.
4. Chicco D, Jurman G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Med Inform Decis Mak*. 2020;20:16. doi:10.1186/s12911-020-1023-5
5. Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation, detection, and avoidance. *ACM Trans Knowl Discov Data*. 2012;6(4):15.
6. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*. 2023;4(9):100804.
7. Moreno-Sánchez PA. Development of an explainable prediction model of heart failure survival by using ensemble trees. In: *2020 IEEE International Conference on Big Data*. 2020:4902–4910.
8. Moreno-Sánchez PA. Improvement of a prediction model for heart failure survival through explainable artificial intelligence. *Front Cardiovasc Med*. 2023 [VERIFY volume/article]. arXiv:2108.10717.
9. Ishaq A, Sadiq S, Umer M, et al. Improving the prediction of heart failure patients' survival using SMOTE and effective data mining techniques. *IEEE Access*. 2021;9:39707–39716.
10. Karakuş MÖ, Er O. A comparative study on prediction of survival event of heart failure patients using machine learning algorithms. *Neural Comput Appl*. 2022;34(16):13895–13908.
11. Oyewunmi OE, Aladeniyi OB, Bodunwa OK. Comparative study on prediction of survival event of heart failure patients using machine learning and statistical algorithms. *SciMedicine J*. 2023;5(2):44–53.
12. Talukder MSH, Sulaiman RB, Angon MBP. Unleashing the power of extra-tree feature selection and random forest classifier for improved survival prediction in heart failure patients. arXiv:2308.05765. 2023.
13. Qadri AM, Hashmi MSA, Raza A, Zaidi SAJ, Rehman A. Heart failure survival prediction using novel transfer learning based probabilistic features. *PeerJ Comput Sci*. 2024;10:e1894.
14. Rane A, Choudhari A, Kant S, Deep A. Discrete-time survival analysis for heart failure mortality prediction. arXiv:2608.04140. 2026.
15. Özgür [VERIFY initials], Bekiroğlu [VERIFY]. Why follow-up matters in survival analysis: comparing Cox proportional hazard regression and random survival forest for predicting heart failure outcomes. *BMC Cardiovasc Disord*. 2025. doi:10.1186/s12872-025-05165-x
16. [VERIFY authors]. Fairness-aware machine learning for heart failure prediction: performance, bias, and clinical deployment insights. medRxiv. 2025. doi:10.1101/2025.10.17.25338263
17. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378.
18. Nadeau C, Bengio Y. Inference for the generalization error. *Mach Learn*. 2003;52:239–281.
19. Ishwaran H, Kogalur UB, Blackstone EH, Lauer MS. Random survival forests. *Ann Appl Stat*. 2008;2(3):841–860.
20. Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. *Stat Med*. 1996;15(4):361–387.
21. Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. *Stat Med*. 2011;30(10):1105–1117.
22. Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and comparison of prognostic classification schemes for survival data. *Stat Med*. 1999;18(17–18):2529–2545.
23. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med*. 2019;17:230.
24. Vickers AJ, Cronin AM, Elkin EB, Gonen M. Extensions to decision curve analysis, a novel method for evaluating diagnostic tests, prediction models and molecular markers. *BMC Med Inform Decis Mak*. 2008;8:53.
25. Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika*. 1994;81(3):515–526.
26. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst*. 2017;30.
27. Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable prediction model: part II — binary and time-to-event outcomes. *Stat Med*. 2019;38(7):1276–1296.
28. Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. *BMJ*. 2020;368:m441.
29. Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. *Sci Data*. 2016;3:160035.
30. Pölsterl S. scikit-survival: a library for time-to-event analysis built on top of scikit-learn. *J Mach Learn Res*. 2020;21(212):1–6.
31. Davidson-Pilon C. lifelines: survival analysis in Python. *J Open Source Softw*. 2019;4(40):1317.
32. Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*. 2021;372:n71.
33. Wolff RF, Moons KGM, Riley RD, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. *Ann Intern Med*. 2019;170(1):51–58.
34. Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med Res Methodol*. 2018;18:24.

# Figures and tables

| Item | File | Content |
|---|---|---|
| Table 1 | `results/tables/table1_cohort_by_outcome.csv` | Baseline characteristics by outcome |
| Table 2 | `results/tables/table2_cox_hazard_ratios.csv` | Adjusted hazard ratios |
| Table 3 | `results/tables/table3_subgroup_cindex.csv` | Subgroup C-index |
| Table 4 | `results/tables/mimic_primary_performance.csv` | External validation [MIMIC] |
| Figure 1 | `fig1_time_leakage` | (a) Follow-up by outcome; (b) MCC by feature set |
| Figure 2 | `fig2a_km_tertiles`, `fig2b_cox_forest` | KM by tertiles; hazard-ratio forest plot |
| Figure 3 | `fig3_survival_cindex` | C-index by model |
| Figure 4 | `fig4_rsf_importance` (+ `fig4b_rsf_shap_beeswarm`) | Out-of-fold importance |
| Figure 5 | `fig5_mimic_calibration` | External calibration [MIMIC] |
| Figure 6 | `fig6_decision_curve_90d` | Decision curves |
| Figure 7 | `fig_github_audit` | Use of `time` in public code |
| Suppl. S1 | `replication_summary.csv`, `replication_vs_published.csv` | Replication |
| Suppl. S2 | `fig_s_simulation_leakage`, `simulation_summary.csv` | Simulation |
| Suppl. S3 | `github_audit_repos.csv`, `github_audit_verification*.csv` | Audit data and verification |
| Suppl. S-Lit | `docs/manuscript/literature_review.md` §C | Literature extraction table |
| Suppl. S4 | `table_s_ph_schoenfeld.csv`, `table_s_piecewise_cox.csv`, `survival_bootstrap_optimism.csv`, `fig_s_calibration_oof_90d` | PH tests and period-specific model, optimism, calibration |
| Suppl. S5 | `survival_cv_deepsurv_summary.csv`, `survival_nested_cv_*.csv` | DeepSurv; nested-CV tuning |
| Suppl. S6 | `docs/manuscript/systematic_review_protocol.md`, extraction sheet | Systematic review protocol and data [SR] |
