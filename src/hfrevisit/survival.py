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
                   horizon_risks: tuple[float, ...] = (90.0, 180.0),
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
            frame = {"repeat": rep, "model": name, "row": te, "risk": risk}
            if hasattr(m, "predict_survival_function"):
                for h in horizon_risks:
                    frame[f"risk_{int(h)}d"] = 1.0 - _surv_matrix(m, X.iloc[te], np.array([h]))[:, 0]
            oof.append(pd.DataFrame(frame))
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def paired_model_comparison(metrics: pd.DataFrame, reference: str = "Cox PH",
                            metric: str = "harrell_c", n: int = 299, n_splits: int = 5) -> pd.DataFrame:
    """Corrected resampled t-test (Nadeau-Bengio) of each model vs ``reference``
    using the per-fold metric on identical folds."""
    from .stats import corrected_resampled_ttest

    wide = metrics.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
    n_test = n // n_splits
    rows = []
    for m in wide.columns:
        if m == reference:
            continue
        r = corrected_resampled_ttest(wide[m] - wide[reference], n - n_test, n_test)
        rows.append({"model": m, "reference": reference, "metric": metric, **r})
    return pd.DataFrame(rows)


def oof_average(oof: pd.DataFrame, model: str, column: str, n: int) -> np.ndarray:
    """Average a per-patient out-of-fold quantity over CV repeats."""
    r = oof[oof["model"] == model]
    return r.groupby("row")[column].mean().reindex(range(n)).values


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


# --------------------------------------------------------------------------
# Bootstrap internal validation (Harrell, Lee & Mark 1996; Steyerberg 2019)
# --------------------------------------------------------------------------
def bootstrap_optimism(df: pd.DataFrame, models: dict | None = None, n_boot: int = 200,
                       seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Optimism-corrected Harrell's C and bootstrap calibration slope.

    For each bootstrap sample b: fit the model on b, compute C on b
    (apparent_b) and on the original data (test_b). Optimism is the mean of
    apparent_b - test_b. The calibration slope is the Cox coefficient of the
    standardised bootstrap-model risk score in the original data; values < 1
    indicate overfitting and estimate the uniform shrinkage required.
    """
    from lifelines import CoxPHFitter

    models = models or make_survival_models(seed)
    y = survival_target(df)
    rng = np.random.default_rng(seed)
    idx_all = [rng.integers(0, len(df), len(df)) for _ in range(n_boot)]
    rows = []
    for name, (feats, tmpl) in models.items():
        X = df[feats]
        full = clone(tmpl).fit(X, y)
        apparent = concordance_index_censored(y["event"], y["time"], full.predict(X))[0]
        opt, slopes = [], []
        for b in idx_all:
            m = clone(tmpl).fit(X.iloc[b], y[b])
            c_b = concordance_index_censored(y[b]["event"], y[b]["time"], m.predict(X.iloc[b]))[0]
            lp = np.asarray(m.predict(X), dtype=float)
            c_o = concordance_index_censored(y["event"], y["time"], lp)[0]
            opt.append(c_b - c_o)
            # slope of the bootstrap model's linear predictor (on its own scale,
            # standardised by its SD in the bootstrap sample) in the original data
            lp_b = np.asarray(m.predict(X.iloc[b]), dtype=float)
            z = (lp - lp_b.mean()) / lp_b.std()
            d = pd.DataFrame({"z": z, "T": y["time"], "E": y["event"].astype(int)})
            z_b = (lp_b - lp_b.mean()) / lp_b.std()
            db = pd.DataFrame({"z": z_b, "T": y[b]["time"], "E": y[b]["event"].astype(int)})
            beta_app = CoxPHFitter().fit(db, "T", "E").params_["z"]
            beta_test = CoxPHFitter().fit(d, "T", "E").params_["z"]
            slopes.append(beta_test / beta_app)
        rows.append({"model": name, "apparent_c": apparent, "optimism": float(np.mean(opt)),
                     "corrected_c": apparent - float(np.mean(opt)),
                     "calibration_slope": float(np.mean(slopes)),
                     "calibration_slope_p2.5": float(np.percentile(slopes, 2.5)),
                     "calibration_slope_p97.5": float(np.percentile(slopes, 97.5)),
                     "n_boot": n_boot})
    return pd.DataFrame(rows)
