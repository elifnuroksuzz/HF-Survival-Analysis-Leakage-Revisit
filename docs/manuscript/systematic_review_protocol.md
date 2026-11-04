# Systematic review protocol — use of follow-up time in studies of the Ahmad et al. (2017) heart-failure dataset

Reported following PRISMA 2020 (Page et al., *BMJ* 2021;372:n71). Register the protocol
(e.g. OSF Registries) **before** screening, and report any deviations.

## Question

Among published studies that develop or evaluate mortality-prediction models on the
299-patient heart-failure dataset (Ahmad et al. 2017; distributed by UCI/Kaggle and
popularised by Chicco & Jurman 2020):

1. What proportion use follow-up time (`time`) as a predictor of the death indicator?
2. Is the use of `time` associated with higher reported performance?
3. How often are other leakage-prone practices used (oversampling before splitting,
   feature selection on the full data, single hold-out split, no censoring handling)?

## Eligibility

- **Include:** peer-reviewed articles, conference papers and preprints (2017 to the search
  date) in English that fit at least one prediction model to this dataset and report at
  least one performance measure.
- **Exclude:** reviews without original analysis; studies using the data only for
  benchmarking generic algorithms without reporting mortality-prediction results; studies
  of other heart-failure datasets; duplicate reports (keep the most complete).

## Information sources and search (executed by `scripts/systematic_review.py`)

| Source | Strategy |
|---|---|
| OpenAlex | (a) all works citing Ahmad et al. 2017 (doi:10.1371/journal.pone.0181001); (b) all works citing Chicco & Jurman 2020 (doi:10.1186/s12911-020-1023-5); (c) full-text/abstract search: `"heart failure clinical records"` |
| Europe PMC | `("heart failure clinical records" OR "heart_failure_clinical_records" OR (Chicco AND Jurman AND "heart failure")) AND (machine learning OR classifier OR survival)` |
| Manual | reference lists of included studies and of the reviews retrieved |

Search date and query strings are recorded automatically in the output.

## Screening

Two reviewers independently screen titles/abstracts and then full texts against the
eligibility criteria (Rayyan or the CSV produced by the script). Disagreements are
resolved by discussion. Report Cohen's κ and a PRISMA flow diagram.

## Data extraction (form: `data/reference/systematic_review_extraction_form.csv`)

Per study: bibliographic data; venue type and indexing; model families; **whether `time`
was an input feature** (yes / no / unclear, with quotation and page); outcome definition;
handling of censoring; oversampling (SMOTE etc.) and whether it preceded the split; feature
selection on full data; validation scheme (hold-out / k-fold / repeated / nested /
external); best reported accuracy, MCC, F1, ROC AUC; code availability.

The script pre-fills candidate values by text mining of open-access full texts (Europe
PMC); **every pre-filled value must be confirmed by a reviewer**.

## Risk of bias

PROBAST (Wolff et al., *Ann Intern Med* 2019), analysis domain in particular; or
PROBAST+AI when available. Report per domain.

## Synthesis

Proportion of studies using `time` (Wilson 95% CI). Compare reported performance by use of
`time` (median, IQR; Mann–Whitney or a mixed model with study as unit). No meta-analysis of
performance is planned, because the studies share one dataset.
