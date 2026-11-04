"""Clinical utility and censoring-aware calibration at a fixed horizon.

* Decision curve analysis for right-censored outcomes (Vickers et al.,
  BMC Med Inform Decis Mak 2008): the event probability among patients
  classified as high risk is estimated with Kaplan-Meier, so censored
  patients contribute without being labelled survivors.
* Calibration at horizon ``tau``: predicted risk vs Kaplan-Meier observed
  risk within risk groups, plus the observed/expected ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator


def km_risk(time: np.ndarray, event: np.ndarray, tau: float) -> float:
    """Kaplan-Meier cumulative incidence 1 - S(tau)."""
    if len(time) == 0:
        return np.nan
    t, s = kaplan_meier_estimator(np.asarray(event, bool), np.asarray(time, float))
    idx = np.searchsorted(t, tau, side="right") - 1
    return float(1.0 - (s[idx] if idx >= 0 else 1.0))


def decision_curve(time, event, risk, tau: float,
                   thresholds=np.arange(0.05, 0.61, 0.01)) -> pd.DataFrame:
    """Net benefit of the model, treat-all and treat-none at horizon ``tau``."""
    time, event, risk = map(np.asarray, (time, event, risk))
    n = len(time)
    p_all = km_risk(time, event, tau)
    rows = []
    for pt in thresholds:
        w = pt / (1 - pt)
        pos = risk >= pt
        frac = pos.mean()
        if frac > 0:
            r = km_risk(time[pos], event[pos], tau)
            nb = r * frac - (1 - r) * frac * w
        else:
            nb = 0.0
        rows.append({"threshold": pt, "net_benefit_model": nb,
                     "net_benefit_treat_all": p_all - (1 - p_all) * w,
                     "net_benefit_treat_none": 0.0, "fraction_high_risk": frac, "n": n})
    return pd.DataFrame(rows)


def calibration_at_horizon(time, event, risk, tau: float, n_groups: int = 5) -> tuple[pd.DataFrame, dict]:
    """Grouped calibration with Kaplan-Meier observed risk per risk quantile."""
    time, event, risk = map(np.asarray, (time, event, risk))
    grp = pd.qcut(risk, n_groups, labels=False, duplicates="drop")
    rows = []
    for g in np.unique(grp):
        m = grp == g
        rows.append({"group": int(g) + 1, "n": int(m.sum()), "mean_predicted": risk[m].mean(),
                     "observed_km": km_risk(time[m], event[m], tau)})
    tab = pd.DataFrame(rows)
    observed = km_risk(time, event, tau)
    summary = {"tau": tau, "observed_km_risk": observed, "mean_predicted_risk": float(risk.mean()),
               "observed_expected_ratio": observed / float(risk.mean()),
               "mean_abs_group_error": float((tab["observed_km"] - tab["mean_predicted"]).abs().mean())}
    return tab, summary
