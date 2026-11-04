"""Time-to-event modelling with ``time`` used correctly as the outcome.

Models (all fitted inside cross-validation folds, pre-processing included):

* Cox PH (ridge-penalised) on the 11 baseline variables
* Cox PH on ejection fraction + serum creatinine (Chicco & Jurman's "top-2")
* Random survival forest
* Gradient-boosted Cox model
* DeepSurv (optional, see :mod:`hfrevisit.deepsurv`)

Metrics per test fold: Harrell's C, Uno's C (truncated at ``tau``), cumulative/
dynamic time-dependent AUC at clinical horizons, Brier score at horizons, and
the integrated Brier score (IBS). A Kaplan-Meier "null model" provides the
IBS reference.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator

from . import config as C
from .data import survival_target


def preprocessor(features: list[str]) -> ColumnTransformer:
    """log1p for right-skewed labs, then standardisation of all columns."""
    log_cols = [f for f in features if f in C.LOG_FEATURES]
    other = [f for f in features if f not in C.LOG_FEATURES]
    steps = []
    if log_cols:
        steps.append(("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                       ("z", StandardScaler())]), log_cols))
    if other:
        steps.append(("z", StandardScaler(), other))
    return ColumnTransformer(steps, verbose_feature_names_out=False)


def make_survival_models(seed: int = C.RANDOM_SEED, features: list[str] | None = None) -> dict[str, tuple[list[str], Pipeline]]:
    feats = features or C.BASELINE_FEATURES
    top2 = C.TOP2_FEATURES

    def pipe(est, f):
        return (f, Pipeline([("prep", preprocessor(f)), ("model", est)]))

    return {
        "Cox PH": pipe(CoxPHSurvivalAnalysis(alpha=0.1, ties="efron"), feats),
        "Cox PH (EF + SC)": pipe(CoxPHSurvivalAnalysis(alpha=0.1, ties="efron"), top2),
        "Random survival forest": pipe(RandomSurvivalForest(
            n_estimators=500, min_samples_leaf=15, max_features="sqrt",
            random_state=seed, n_jobs=-1), feats),
        "Gradient-boosted Cox": pipe(GradientBoostingSurvivalAnalysis(
            n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8,
            random_state=seed), feats),
    }


def _surv_matrix(model, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    fns = model.predict_survival_function(X)
    return np.vstack([fn(times) for fn in fns])


def evaluate_fold(model, X_te: pd.DataFrame, y_tr: np.ndarray, y_te: np.ndarray,
                  horizons=C.EVAL_HORIZONS, tau: float = 240.0,
                  ibs_grid: np.ndarray | None = None) -> dict[str, float]:
    """Discrimination and calibration metrics for one test fold."""
    risk = np.asarray(model.predict(X_te), dtype=float)  # higher = higher risk
    out = {
        "harrell_c": concordance_index_censored(y_te["event"], y_te["time"], risk)[0],
    }
    t_lo, t_hi = y_te["time"].min(), min(y_te["time"].max(), y_tr["time"].max())
    try:
        out["uno_c"] = concordance_index_ipcw(y_tr, y_te, risk, tau=min(tau, t_hi - 1e-6))[0]
    except ValueError:
        out["uno_c"] = np.nan
    valid_h = np.array([h for h in horizons if t_lo <= h < t_hi], dtype=float)
    if valid_h.size:
        aucs, _ = cumulative_dynamic_auc(y_tr, y_te, risk, valid_h)
        for h, a in zip(valid_h, aucs):
            out[f"auc_{int(h)}d"] = a
        if hasattr(model, "predict_survival_function"):
            S = _surv_matrix(model, X_te, valid_h)
            _, bs = brier_score(y_tr, y_te, S, valid_h)
            for h, b in zip(valid_h, bs):
                out[f"brier_{int(h)}d"] = b
    if hasattr(model, "predict_survival_function"):
        grid = ibs_grid if ibs_grid is not None else np.arange(15, 241, 5, dtype=float)
        grid = grid[(grid >= t_lo) & (grid < t_hi)]
        if grid.size > 2:
            out["ibs"] = integrated_brier_score(y_tr, y_te, _surv_matrix(model, X_te, grid), grid)
            # Kaplan-Meier null model (same survival curve for everyone).
            km_t, km_s = kaplan_meier_estimator(y_tr["event"], y_tr["time"])
            s_km = np.interp(grid, km_t, km_s, left=1.0, right=km_s[-1])
            out["ibs_km_null"] = integrated_brier_score(
                y_tr, y_te, np.tile(s_km, (len(y_te), 1)), grid)
    return out


def cross_validate(df: pd.DataFrame, models: dict | None = None, n_splits: int = 5,
                   n_repeats: int = 10, seed: int = C.RANDOM_SEED,
                   extra_models: dict[str, tuple[list[str], Callable]] | None = None,
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated stratified K-fold CV (stratified on the event indicator).

    Returns
    -------
    metrics : one row per (repeat, fold, model)
    oof     : out-of-fold risk scores for every patient and repeat, used for
              subgroup (fairness) analysis and explanation without in-sample bias.
    """
    models = dict(models or make_survival_models(seed))
    if extra_models:
        models.update(extra_models)
    y = survival_target(df)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows, oof = [], []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(df)), df[C.EVENT].values)):
        rep, fold = divmod(k, n_splits)
        for name, (feats, template) in models.items():
            X = df[feats]
            m = clone(template) if hasattr(template, "get_params") else template()
            m.fit(X.iloc[tr], y[tr])
            met = evaluate_fold(m, X.iloc[te], y[tr], y[te])
            rows.append({"repeat": rep, "fold": fold, "model": name, **met})
            risk = np.asarray(m.predict(X.iloc[te]), dtype=float)
            oof.append(pd.DataFrame({"repeat": rep, "model": name, "row": te, "risk": risk}))
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def summarize_cv(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and 2.5-97.5 percentile range of each metric across folds."""
    value_cols = [c for c in metrics.columns if c not in ("repeat", "fold", "model")]
    rows = []
    for model, g in metrics.groupby("model", sort=False):
        row = {"model": model}
        for c in value_cols:
            v = g[c].dropna()
            if v.empty:
                continue
            row[c] = v.mean()
            row[f"{c}_sd"] = v.std(ddof=1)
            row[f"{c}_p2.5"], row[f"{c}_p97.5"] = np.percentile(v, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Interpretable full-cohort Cox model (lifelines) — hazard ratios + PH checks
# --------------------------------------------------------------------------
#: Clinically interpretable scaling for hazard ratios.
HR_SCALING: dict[str, tuple[str, Callable[[pd.Series], pd.Series]]] = {
    "age": ("per 10 years", lambda s: s / 10.0),
    "ejection_fraction": ("per 5 percentage points", lambda s: s / 5.0),
    "serum_sodium": ("per 5 mEq/L", lambda s: s / 5.0),
    "serum_creatinine": ("per doubling", np.log2),
    "creatinine_phosphokinase": ("per doubling", np.log2),
    "platelets": ("per doubling", np.log2),
}


def cox_hazard_ratios(df: pd.DataFrame, features: list[str] | None = None,
                      penalizer: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit an unpenalised Cox model on clinically scaled covariates.

    Returns the HR table, the Schoenfeld-residual PH test table, and the fitted
    lifelines model.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    feats = features or C.BASELINE_FEATURES
    d = pd.DataFrame(index=df.index)
    units = {}
    for f in feats:
        if f in HR_SCALING:
            units[f], fn = HR_SCALING[f]
            d[f] = fn(df[f].astype(float))
        else:
            units[f] = "yes vs no" if f != "sex" else "male vs female"
            d[f] = df[f].astype(float)
    d[C.TIME], d[C.EVENT] = df[C.TIME], df[C.EVENT]
    cph = CoxPHFitter(penalizer=penalizer).fit(d, duration_col=C.TIME, event_col=C.EVENT)
    s = cph.summary
    hr = pd.DataFrame({
        "variable": [C.FEATURE_LABELS.get(f, f) for f in s.index],
        "unit": [units[f] for f in s.index],
        "HR": s["exp(coef)"].values,
        "CI_low": s["exp(coef) lower 95%"].values,
        "CI_high": s["exp(coef) upper 95%"].values,
        "p": s["p"].values,
    }, index=s.index)
    ph = proportional_hazard_test(cph, d, time_transform="rank").summary
    ph = ph.rename(columns={"test_statistic": "chi2"})[["chi2", "p"]]
    return hr, ph, cph
