# TRIPOD+AI reporting map

Where each TRIPOD+AI reporting topic (Collins et al., BMJ 2024;385:e078378) is
addressed in this code base. Use the official checklist for item numbers when
submitting; this file lists the evidence each item needs.

| Section / topic | What to report | Source in this repository |
|---|---|---|
| Title / abstract | Development + external validation; survival outcome; horizon | `docs/MANUSCRIPT_GUIDE.md` |
| Background, objectives | Leakage in prior work; RQ1-RQ6 | `docs/MANUSCRIPT_GUIDE.md` §1b |
| Data sources | Ahmad et al. 2017 cohort (public, CC BY 4.0); MIMIC-III v1.4 (PhysioNet DUA) | `config.py`, `mimic/queries.py` |
| Participants, eligibility | Development: all 299; validation: first HF admission, age ≥18, EF and labs available | `mimic_primary_attrition.csv` (flow diagram) |
| Data preparation | log-transform + standardisation inside folds; MIMIC unit harmonisation | `survival.preprocessor`, `mimic/cohort.py` |
| Outcome | Development: death during follow-up (right-censored, max 285 d). Validation: all-cause death ≤90 d from admission (PATIENTS.DOD), administrative censoring at 90 d; in-hospital death secondary | `mimic/cohort.build_cohort` |
| Predictors | 11 baseline variables; **follow-up time excluded** (outcome-dependent) | `config.BASELINE_FEATURES`; leakage experiments |
| Sample size | Riley et al. criteria, events per parameter | `sample_size_riley.csv` |
| Missing data | None in development; MIMIC CPK: development-median imputation + complete-case sensitivity | `mimic/cohort.impute_cpk`, `mimic_sens_complete_cpk_*` |
| Analytical methods | Cox (ridge), Cox EF+SC, RSF, GB-Cox, DeepSurv (optional); fixed a-priori hyperparameters | `survival.make_survival_models`, `deepsurv.py` |
| Internal validation | 10×5-fold repeated CV; bootstrap optimism correction (B=500) | `survival_cv_summary.csv`, `survival_bootstrap_optimism.csv` |
| Performance measures | Harrell/Uno C, time-dependent AUC, Brier, IBS vs KM null; calibration O/E, grouped KM calibration, calibration slope | `survival.evaluate_fold`, `clinical_utility.py`, `calibration.py` |
| Model comparison | Corrected resampled t-test on identical folds | `survival_paired_comparison.csv` |
| Clinical utility | Decision curve analysis (censoring-aware) | `dca_*.csv`, `fig6_decision_curve_90d` |
| Fairness | Sex, age and intersectional subgroups; OOF; bootstrap CI; flag <20 events | `table3_subgroup_cindex.csv`, `mimic_primary_subgroups.csv` |
| Model updating | Cross-fitted intercept / logistic / isotonic recalibration in MIMIC | `mimic_primary_recalibration.csv` |
| Explainability | OOF permutation importance, OOF SHAP | `importance_*`, `shap_rsf_oof_values.csv` |
| Model specification | Full-cohort fitted models (pickle) + Cox HRs | `models/development_survival_models.pkl`, `table2_cox_hazard_ratios.csv` |
| PH assumption | Schoenfeld residual tests | `table_s_ph_schoenfeld.csv` |
| Text-derived predictor | EF parser rules + manual-annotation validation | `mimic/ef_parser.py`, `scripts/ef_annotation.py` |
| Limitations | n=299 (below Riley minimum), single-centre development cohort, EF from text, ICD-coded comorbidities, MIMIC case mix (ICU-era US) | `docs/MANUSCRIPT_GUIDE.md` §7 |
| Code / data availability | This repository (archive with Zenodo DOI); MIMIC data not redistributable | `CITATION.cff`, `.zenodo.json` |
| Ethics | Public de-identified data; PhysioNet credentialed access + CITI training | — |
