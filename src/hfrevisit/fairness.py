"""Subgroup performance (algorithmic fairness) analysis.

All estimates are computed on **out-of-sample** predictions (out-of-fold in the
development cohort, or external predictions in MIMIC-III), never on training
data. With n = 299, subgroups are small; we therefore report bootstrap CIs and
flag subgroups with fewer than 20 events, which should be interpreted as
exploratory only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored

from . import config as C
from .stats import bootstrap_metric

MIN_EVENTS = 20


def define_subgroups(df: pd.DataFrame, age_cutoff: int = 65) -> dict[str, np.ndarray]:
    age = df["age"].values
    sex = df["sex"].values
    return {
        "Overall": np.ones(len(df), dtype=bool),
        "Female": sex == 0,
        "Male": sex == 1,
        f"Age <{age_cutoff}": age < age_cutoff,
        f"Age >={age_cutoff}": age >= age_cutoff,
        f"Female, age <{age_cutoff}": (sex == 0) & (age < age_cutoff),
        f"Female, age >={age_cutoff}": (sex == 0) & (age >= age_cutoff),
        f"Male, age <{age_cutoff}": (sex == 1) & (age < age_cutoff),
        f"Male, age >={age_cutoff}": (sex == 1) & (age >= age_cutoff),
    }


def _cindex(t, e, r):
    return concordance_index_censored(e.astype(bool), t, r)[0]


def survival_subgroups(df: pd.DataFrame, risk: np.ndarray, n_boot: int = 1000,
                       age_cutoff: int = 65) -> pd.DataFrame:
    """Harrell's C per subgroup with stratified bootstrap 95% CI."""
    t, e = df[C.TIME].values, df[C.EVENT].values
    rows = []
    for name, m in define_subgroups(df, age_cutoff).items():
        n_ev = int(e[m].sum())
        row = {"subgroup": name, "n": int(m.sum()), "events": n_ev,
               "exploratory": n_ev < MIN_EVENTS}
        if n_ev >= 2 and (e[m] == 0).sum() >= 2:
            row.update(bootstrap_metric(_cindex, t[m], e[m], risk[m], n_boot=n_boot, stratify=e[m]))
        rows.append(row)
    return pd.DataFrame(rows)


def binary_subgroups(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float,
                     n_boot: int = 1000, age_cutoff: int = 65) -> pd.DataFrame:
    """AUC, sensitivity, specificity, calibration-in-the-large per subgroup,
    plus between-group gaps used by equalised-odds style criteria."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    rows = []
    for name, m in define_subgroups(df, age_cutoff).items():
        yy, pp, dd = y[m], p[m], pred[m]
        row = {"subgroup": name, "n": int(m.sum()), "events": int(yy.sum()),
               "exploratory": yy.sum() < MIN_EVENTS,
               "observed_rate": yy.mean(), "mean_predicted": pp.mean(),
               "selection_rate": dd.mean(),
               "sensitivity": dd[yy == 1].mean() if (yy == 1).any() else np.nan,
               "specificity": 1 - dd[yy == 0].mean() if (yy == 0).any() else np.nan}
        if yy.sum() >= 2 and (yy == 0).sum() >= 2:
            b = bootstrap_metric(roc_auc_score, yy, pp, n_boot=n_boot, stratify=yy)
            row.update({"auc": b["estimate"], "auc_ci_low": b["ci_low"], "auc_ci_high": b["ci_high"]})
        rows.append(row)
    return pd.DataFrame(rows)


def gap_summary(tab: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Absolute between-group differences (demographic parity, TPR, FPR, AUC)."""
    t = tab.set_index("subgroup")
    rows = []
    for a, b in pairs:
        rows.append({
            "comparison": f"{a} vs {b}",
            "selection_rate_diff": abs(t.loc[a, "selection_rate"] - t.loc[b, "selection_rate"]),
            "tpr_diff": abs(t.loc[a, "sensitivity"] - t.loc[b, "sensitivity"]),
            "fpr_diff": abs((1 - t.loc[a, "specificity"]) - (1 - t.loc[b, "specificity"])),
            "auc_diff": abs(t.loc[a].get("auc", np.nan) - t.loc[b].get("auc", np.nan)),
        })
    out = pd.DataFrame(rows)
    out["equalized_odds_diff"] = out[["tpr_diff", "fpr_diff"]].max(axis=1)
    return out
