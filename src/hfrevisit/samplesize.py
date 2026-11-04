"""Minimum sample size for developing a time-to-event prediction model
(Riley et al., Stat Med 2019; BMJ 2020;368:m441), as implemented in the
``pmsampsize`` package.

Criteria
  (i)   expected uniform shrinkage >= S (default 0.9)
  (ii)  optimism in apparent Nagelkerke R^2 <= delta (default 0.05)
  (iii) overall risk at time ``tau`` estimated within +/- ``margin``
"""

from __future__ import annotations

import numpy as np


def max_r2_cs_survival(rate: float, mean_followup: float) -> float:
    """Maximum Cox-Snell R^2 for an exponential null model.

    ln L_null / n = e * ln(rate) - e, with e = expected events per patient.
    """
    e = rate * mean_followup
    return float(1.0 - np.exp(2.0 * (e * np.log(rate) - e)))


def riley_survival(n_params: int, r2_cs: float, rate: float, mean_followup: float, tau: float,
                   shrinkage: float = 0.9, delta: float = 0.05, margin: float = 0.05) -> dict:
    """Minimum n per criterion.

    Parameters
    ----------
    n_params      : candidate predictor parameters
    r2_cs         : anticipated Cox-Snell R^2 (the apparent value of the fitted
                    Cox model is optimistic, so the result is a lower bound)
    rate          : overall event rate per unit time (events / person-time)
    mean_followup : mean follow-up per patient (same time unit)
    tau           : time point for criterion (iii)
    """
    p = n_params
    n1 = p / ((shrinkage - 1.0) * np.log(1.0 - r2_cs / shrinkage))
    r2_max = max_r2_cs_survival(rate, mean_followup)
    s2 = r2_cs / (r2_cs + delta * r2_max)
    n2 = p / ((s2 - 1.0) * np.log(1.0 - r2_cs / s2))
    risk_tau = 1.0 - np.exp(-rate * tau)
    n3 = (1.96 / margin) ** 2 * risk_tau * (1.0 - risk_tau)
    n_min = int(np.ceil(max(n1, n2, n3)))
    return {"n_params": p, "r2_cs": r2_cs, "r2_cs_max": r2_max, "shrinkage_target": shrinkage,
            "criterion_i_shrinkage": int(np.ceil(n1)), "criterion_ii_optimism": int(np.ceil(n2)),
            "criterion_iii_precision": int(np.ceil(n3)), "minimum_n": n_min,
            "events_required": int(np.ceil(n_min * rate * mean_followup))}


def apparent_cox_snell_r2(lr_statistic: float, n: int) -> float:
    """R^2_CS = 1 - exp(-LR / n) from the likelihood-ratio chi-square."""
    return float(1.0 - np.exp(-lr_statistic / n))


def expected_shrinkage(n: int, n_params: int, r2_cs: float) -> float:
    """Van Houwelingen heuristic shrinkage S = 1 + p / (n ln(1 - R^2_CS))."""
    return float(1.0 + n_params / (n * np.log(1.0 - r2_cs)))
