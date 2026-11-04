"""Experiments demonstrating that follow-up ``time`` is an outcome-dependent
variable whose use as a predictor constitutes target leakage.

Rationale. In this cohort ``time`` is the number of days until death *or*
censoring. It is therefore a component of the outcome, and it is not known at
the moment a prognosis is made (baseline). A model that uses it answers the
question "given how long this patient was followed, did they die?", which has
no clinical use.

Experiments
-----------
A  Distributional evidence: event times vs. censoring times.
B  ``time`` alone is (almost) as predictive as all baseline variables together.
C  Paired Monte-Carlo comparison of models with vs. without ``time``
   (corrected resampled t-test).
D  Permutation control: shuffling ``time`` removes the gain (the gain is
   specific to the outcome-coupled values, not to adding a column).
E  Correct binary reframing: death within a fixed horizon, censored-before-
   horizon patients excluded, baseline predictors only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit

from . import config as C
from .classification import classification_metrics, make_classifiers, scores
from .data import horizon_label
from .stats import corrected_resampled_ttest


def experiment_a(df: pd.DataFrame) -> pd.DataFrame:
    """Follow-up time by vital status and its univariable association with death."""
    dead = df.loc[df[C.EVENT] == 1, C.TIME]
    cens = df.loc[df[C.EVENT] == 0, C.TIME]
    rho, p_rho = sps.spearmanr(df[C.TIME], df[C.EVENT])
    u = sps.mannwhitneyu(cens, dead, alternative="two-sided")
    auc_time = 1 - u.statistic / (len(cens) * len(dead))  # AUC of (-time) for death
    rows = []
    for name, s in [("Died (event time)", dead), ("Censored (censoring time)", cens)]:
        q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
        rows.append({"group": name, "n": len(s), "median_days": q2, "q1": q1, "q3": q3,
                     "share_<=30d": (s <= 30).mean(), "share_>150d": (s > 150).mean()})
    out = pd.DataFrame(rows)
    out.attrs.update({"spearman_rho": rho, "spearman_p": p_rho, "mannwhitney_p": u.pvalue,
                      "auc_of_time_alone": auc_time})
    return out


def leakage_monte_carlo(df: pd.DataFrame, n_splits: int = 100, test_size: float = 0.30,
                        seed: int = C.RANDOM_SEED, models: dict | None = None) -> pd.DataFrame:
    """Experiments B-D on identical stratified splits.

    Feature sets
    ------------
    baseline        : 11 admission-time variables (legitimate)
    baseline+time   : Chicco & Jurman's full set (leaky)
    time only       : follow-up duration alone
    baseline+shuffled time : ``time`` permuted independently in train and test
    """
    rng = np.random.default_rng(seed)
    models = models or {k: v for k, v in make_classifiers(seed).items()
                        if k in ("Random forest", "Logistic regression", "Gradient boosting")}
    y = df[C.EVENT].values
    base = df[C.BASELINE_FEATURES].values.astype(float)
    t = df[[C.TIME]].values.astype(float)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    rows = []
    for split, (tr, te) in enumerate(sss.split(base, y)):
        t_shuf = t.copy()
        t_shuf[tr] = t[rng.permutation(tr)]
        t_shuf[te] = t[rng.permutation(te)]
        sets = {
            "baseline": base,
            "baseline+time": np.hstack([base, t]),
            "time only": t,
            "baseline+shuffled time": np.hstack([base, t_shuf]),
        }
        for fs, X in sets.items():
            for name, template in models.items():
                m = clone(template).fit(X[tr], y[tr])
                met = classification_metrics(y[te], m.predict(X[te]), scores(m, X[te]))
                rows.append({"split": split, "feature_set": fs, "model": name,
                             "n_train": len(tr), "n_test": len(te), **met})
    return pd.DataFrame(rows)


def paired_contrasts(res: pd.DataFrame, metric: str = "MCC",
                     reference: str = "baseline") -> pd.DataFrame:
    """Corrected resampled t-test of each feature set vs. ``reference``."""
    out = []
    n_train, n_test = int(res["n_train"].iloc[0]), int(res["n_test"].iloc[0])
    for model, g in res.groupby("model"):
        wide = g.pivot(index="split", columns="feature_set", values=metric)
        for fs in wide.columns:
            if fs == reference:
                continue
            r = corrected_resampled_ttest(wide[fs] - wide[reference], n_train, n_test)
            out.append({"model": model, "contrast": f"{fs} - {reference}", "metric": metric, **r})
    return pd.DataFrame(out)


def horizon_reframing(df: pd.DataFrame, tau: float = C.DEFAULT_HORIZON, n_splits: int = 100,
                      test_size: float = 0.30, seed: int = C.RANDOM_SEED,
                      models: dict | None = None) -> pd.DataFrame:
    """Experiment E: *death within tau days* from baseline predictors only.

    Patients censored before ``tau`` have an undefined label and are excluded
    (complete-case; the survival models in :mod:`hfrevisit.survival` use all
    patients via proper censoring handling).
    """
    y_all = horizon_label(df[C.TIME].values, df[C.EVENT].values, tau)
    keep = ~np.isnan(y_all)
    d = df.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int)
    models = models or {k: v for k, v in make_classifiers(seed).items()
                        if k in ("Random forest", "Logistic regression", "Gradient boosting")}
    X = d[C.BASELINE_FEATURES].values.astype(float)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    rows = []
    for split, (tr, te) in enumerate(sss.split(X, y)):
        for name, template in models.items():
            m = clone(template).fit(X[tr], y[tr])
            rows.append({"split": split, "model": name, "tau": tau, "n_eligible": int(keep.sum()),
                         "n_excluded_censored": int((~keep).sum()), "event_rate": y.mean(),
                         **classification_metrics(y[te], m.predict(X[te]), scores(m, X[te]))})
    return pd.DataFrame(rows)
