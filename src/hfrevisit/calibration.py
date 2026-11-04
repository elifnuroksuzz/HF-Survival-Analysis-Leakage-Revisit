"""Calibration assessment and recalibration for binary risk predictions.

Follows Van Calster et al. (2019, BMC Medicine) "Calibration: the Achilles
heel of predictive analytics": we report calibration-in-the-large (CITL),
calibration slope, ECE and smoothed calibration curves. The Hosmer-Lemeshow
test is reported for completeness only — with thousands of patients it
rejects even clinically negligible miscalibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as C

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, p) -> dict[str, float]:
    """CITL (intercept with slope fixed at 1, via offset) and calibration slope."""
    import statsmodels.api as sm

    y = np.asarray(y, dtype=float)
    lp = logit(p)
    slope_fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    citl_fit = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
    return {"citl": float(citl_fit.params[0]), "slope": float(slope_fit.params[1]),
            "slope_ci_low": float(slope_fit.conf_int()[1, 0]),
            "slope_ci_high": float(slope_fit.conf_int()[1, 1])}


def expected_calibration_error(y, p, n_bins: int = 10, strategy: str = "quantile") -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def hosmer_lemeshow(y, p, g: int = 10) -> dict[str, float]:
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"], q=g, labels=False, duplicates="drop")
    t = d.groupby("bin").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    stat = (((t["obs"] - t["exp"]) ** 2) / (t["exp"] * (1 - t["exp"] / t["n"])).clip(lower=EPS)).sum()
    dof = max(len(t) - 2, 1)
    return {"hl_chi2": float(stat), "hl_df": dof, "hl_p": float(chi2.sf(stat, dof))}


def calibration_report(y, p) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    return {"n": len(y), "observed_rate": float(y.mean()), "mean_predicted": float(p.mean()),
            "auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "ece": expected_calibration_error(y, p), **calibration_intercept_slope(y, p),
            **hosmer_lemeshow(y, p)}


def recalibrate_cv(y, p, n_splits: int = 5,
                   seed: int = C.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fitted recalibration in the external cohort.

    Each fold's recalibrator (intercept update, logistic recalibration,
    isotonic regression) is learned on the other folds and applied to the
    held-out fold, so every patient receives an out-of-sample recalibrated
    risk. Returns (metrics table, per-patient recalibrated risks).
    """
    import statsmodels.api as sm

    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    lp = logit(p).reshape(-1, 1)
    out = {k: np.zeros_like(p) for k in ("Intercept update", "Logistic recalibration", "Isotonic")}
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(lp, y):
        # intercept update: logit(p') = a + logit(p)  (slope fixed at 1)
        a = sm.GLM(y[tr], np.ones((len(tr), 1)), family=sm.families.Binomial(),
                   offset=lp[tr, 0]).fit().params[0]
        out["Intercept update"][te] = 1 / (1 + np.exp(-(a + lp[te, 0])))
        lr = LogisticRegression(C=1e6).fit(lp[tr], y[tr])
        out["Logistic recalibration"][te] = lr.predict_proba(lp[te])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p[tr], y[tr])
        out["Isotonic"][te] = iso.predict(p[te])
    out = {"Original": p, **out}
    rows = [{"method": k, **calibration_report(y, v)} for k, v in out.items()]
    return pd.DataFrame(rows), pd.DataFrame(out)
