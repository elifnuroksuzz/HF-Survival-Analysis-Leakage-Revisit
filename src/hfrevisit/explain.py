"""Model explanation computed on held-out data only.

* Permutation importance: drop in Harrell's C on the test fold when one
  variable is permuted (repeated K-fold, so every estimate is out-of-sample).
* SHAP values for the survival model's risk score, computed fold-by-fold
  for the patients in each test fold (model-agnostic permutation explainer
  with a training-fold background sample).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sksurv.metrics import concordance_index_censored

from . import config as C
from .data import survival_target


def oof_permutation_importance(df: pd.DataFrame, features: list[str], template,
                               n_splits: int = 5, n_repeats: int = 5, n_perm: int = 10,
                               seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    y = survival_target(df)
    rng = np.random.default_rng(seed)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(df)), df[C.EVENT].values)):
        X = df[features]
        m = clone(template).fit(X.iloc[tr], y[tr])
        Xte = X.iloc[te].copy()
        base = concordance_index_censored(y[te]["event"], y[te]["time"], m.predict(Xte))[0]
        for f in features:
            drops = []
            for _ in range(n_perm):
                Xp = Xte.copy()
                Xp[f] = rng.permutation(Xp[f].values)
                drops.append(base - concordance_index_censored(
                    y[te]["event"], y[te]["time"], m.predict(Xp))[0])
            rows.append({"split": k, "feature": f, "c_index_drop": float(np.mean(drops))})
    res = pd.DataFrame(rows)
    summ = res.groupby("feature")["c_index_drop"].agg(
        mean="mean", sd="std",
        p2_5=lambda v: np.percentile(v, 2.5), p97_5=lambda v: np.percentile(v, 97.5))
    return summ.sort_values("mean", ascending=False).reset_index()


def oof_shap(df: pd.DataFrame, features: list[str], template, n_splits: int = 5,
             n_background: int = 50, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Out-of-fold SHAP values of the risk score (one value per patient x feature)."""
    import shap

    y = survival_target(df)
    X = df[features]
    sv = np.zeros((len(df), len(features)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(df)), df[C.EVENT].values):
        m = clone(template).fit(X.iloc[tr], y[tr])
        bg = X.iloc[tr].sample(min(n_background, len(tr)), random_state=seed)
        f = lambda a: m.predict(pd.DataFrame(a, columns=features))  # noqa: E731
        expl = shap.PermutationExplainer(f, bg.values, seed=seed)
        sv[te] = expl(X.iloc[te].values, max_evals=2 * len(features) + 1, silent=True).values
    return pd.DataFrame(sv, columns=features)
