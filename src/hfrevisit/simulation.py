"""Simulation study: target leakage from follow-up time is *structural*.

We generate synthetic cohorts that reuse the real covariate distribution
(rows resampled from the development cohort) but draw event times from a
proportional-hazards model whose effect sizes we control:

* ``null``      — no covariate affects survival (beta = 0);
* ``observed``  — log-hazard ratios equal to the Cox estimates in the real data.

Censoring times are drawn from the empirical censoring distribution of the
real cohort (follow-up 4-285 days). The binary label is the observed death
indicator and ``time`` is min(T, C), exactly as in the public dataset.

If including ``time`` yields a high MCC even under the null scenario, the
"performance" it adds cannot be prognostic signal: it is produced by the
data-generating mechanism of right-censored follow-up alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .classification import classification_metrics, make_classifiers, scores


def _empirical_censoring(df: pd.DataFrame) -> np.ndarray:
    return df.loc[df[C.EVENT] == 0, C.TIME].values.astype(float)


def simulate_cohort(df: pd.DataFrame, beta: np.ndarray, rng: np.random.Generator,
                    target_event_rate: float, n: int | None = None) -> pd.DataFrame:
    """One synthetic cohort with exponential event times, hazard lambda*exp(x'beta)."""
    n = n or len(df)
    X = df[C.BASELINE_FEATURES].sample(n, replace=True, random_state=int(rng.integers(1 << 31)))
    X = X.reset_index(drop=True)
    Z = ((X - df[C.BASELINE_FEATURES].mean()) / df[C.BASELINE_FEATURES].std()).values
    lp = Z @ beta
    cens_pool = _empirical_censoring(df)
    C_t = rng.choice(cens_pool, n) + rng.uniform(-0.5, 0.5, n)
    U = rng.uniform(size=n)

    def event_rate(log_lam):
        T = -np.log(U) / (np.exp(log_lam) * np.exp(lp))
        return (T <= C_t).mean() - target_event_rate

    log_lam = brentq(event_rate, -20, 5)
    T = -np.log(U) / (np.exp(log_lam) * np.exp(lp))
    out = X.copy()
    out[C.TIME] = np.minimum(T, C_t).clip(min=1.0)
    out[C.EVENT] = (T <= C_t).astype(int)
    return out


def standardized_cox_beta(df: pd.DataFrame) -> np.ndarray:
    """Cox log-HRs per SD of each baseline covariate in the real data."""
    from lifelines import CoxPHFitter

    d = df[C.BASELINE_FEATURES]
    d = (d - d.mean()) / d.std()
    d = d.assign(**{C.TIME: df[C.TIME].values, C.EVENT: df[C.EVENT].values})
    cph = CoxPHFitter().fit(d, C.TIME, C.EVENT)
    return cph.params_.reindex(C.BASELINE_FEATURES).values


def run_simulation(df: pd.DataFrame, n_sim: int = 200, n_folds: int = 5,
                   seed: int = C.RANDOM_SEED, models: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = models or {k: v for k, v in make_classifiers(seed).items()
                        if k in ("Random forest", "Logistic regression")}
    scenarios = {"null (no covariate effect)": np.zeros(len(C.BASELINE_FEATURES)),
                 "observed effects": standardized_cox_beta(df)}
    rate = df[C.EVENT].mean()
    rows = []
    for scen, beta in scenarios.items():
        for s in range(n_sim):
            sim = simulate_cohort(df, beta, rng, rate)
            y = sim[C.EVENT].values
            skf = StratifiedKFold(n_folds, shuffle=True, random_state=s)
            for fs, feats in {"baseline": C.BASELINE_FEATURES,
                              "baseline+time": C.BASELINE_FEATURES + [C.TIME],
                              "time only": [C.TIME]}.items():
                X = sim[feats].values.astype(float)
                for name, tmpl in models.items():
                    pred = np.zeros(len(y), int)
                    sc = np.zeros(len(y))
                    for tr, te in skf.split(X, y):
                        m = clone(tmpl).fit(X[tr], y[tr])
                        pred[te], sc[te] = m.predict(X[te]), scores(m, X[te])
                    rows.append({"scenario": scen, "sim": s, "feature_set": fs, "model": name,
                                 **classification_metrics(y, pred, sc)})
    return pd.DataFrame(rows)


def summarize_simulation(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby(["scenario", "model", "feature_set"])
    out = g[["MCC", "ROC_AUC"]].agg(["mean", lambda v: np.percentile(v, 2.5),
                                     lambda v: np.percentile(v, 97.5)])
    out.columns = [f"{m}_{s}" for m, s in
                   [(a, {"mean": "mean", "<lambda_0>": "p2.5", "<lambda_1>": "p97.5"}[b])
                    for a, b in out.columns]]
    return out.reset_index()
