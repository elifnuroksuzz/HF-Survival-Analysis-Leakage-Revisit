"""Statistical utilities: cohort tables, confidence intervals, paired tests."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table_one(df: pd.DataFrame, group: str = C.EVENT,
              continuous: Sequence[str] = tuple(C.CONTINUOUS_FEATURES) + (C.TIME,),
              binary: Sequence[str] = tuple(C.BINARY_FEATURES),
              labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Baseline characteristics stratified by ``group`` (Table 1).

    Continuous: median [IQR], Mann-Whitney U. Binary: n (%), chi-square
    (Fisher's exact test when any expected cell count < 5). P-values are
    reported unadjusted and descriptive (no inferential claim).
    """
    labels = labels or C.FEATURE_LABELS
    levels = sorted(df[group].unique())
    rows = []

    def med_iqr(s: pd.Series) -> str:
        q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
        return f"{q2:.1f} [{q1:.1f}-{q3:.1f}]"

    for col in continuous:
        groups = [df.loc[df[group] == g, col] for g in levels]
        p = stats.mannwhitneyu(*groups, alternative="two-sided").pvalue if len(groups) == 2 else np.nan
        rows.append({"Variable": labels.get(col, col), "Statistic": "median [IQR]",
                     "Overall": med_iqr(df[col]),
                     **{f"{group}={g} (n={len(s)})": med_iqr(s) for g, s in zip(levels, groups)},
                     "p": format_p(p), "Test": "Mann-Whitney U"})
    for col in binary:
        ct = pd.crosstab(df[col], df[group])
        _, p_chi, _, expected = stats.chi2_contingency(ct)
        if (expected < 5).any() and ct.shape == (2, 2):
            p, test = stats.fisher_exact(ct.values)[1], "Fisher exact"
        else:
            p, test = p_chi, "Chi-square"

        def n_pct(s: pd.Series) -> str:
            return f"{int(s.sum())} ({100 * s.mean():.1f}%)"

        rows.append({"Variable": labels.get(col, col), "Statistic": "n (%)",
                     "Overall": n_pct(df[col]),
                     **{f"{group}={g} (n={int((df[group] == g).sum())})": n_pct(df.loc[df[group] == g, col])
                        for g in levels},
                     "p": format_p(p), "Test": test})
    return pd.DataFrame(rows)


def percentile_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)]))  # type: ignore[return-value]


def bootstrap_metric(metric: Callable[..., float], *arrays: np.ndarray, n_boot: int = 2000,
                     seed: int = C.RANDOM_SEED, stratify: np.ndarray | None = None,
                     alpha: float = 0.05) -> dict[str, float]:
    """Point estimate + percentile bootstrap CI of ``metric(*arrays)``.

    Resamples patients (rows) with replacement, optionally within strata
    (e.g. the event indicator) so every replicate contains both classes.
    """
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(metric(*arrays))
    reps = []
    for _ in range(n_boot):
        if stratify is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([rng.choice(np.flatnonzero(stratify == s), (stratify == s).sum())
                                  for s in np.unique(stratify)])
        try:
            reps.append(metric(*(a[idx] for a in arrays)))
        except ValueError:  # e.g. single class in replicate
            continue
    lo, hi = percentile_ci(reps, alpha)
    return {"estimate": point, "ci_low": lo, "ci_high": hi, "n_boot_valid": len(reps)}


def corrected_resampled_ttest(diffs: Sequence[float], n_train: int, n_test: int,
                              alpha: float = 0.05) -> dict[str, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV differences.

    Repeated random splits share training data, so the naive t-test grossly
    underestimates variance. The correction inflates the variance by
    ``(1/J + n_test/n_train)``.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    j = d.size
    mean = d.mean()
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / j + n_test / n_train) * var)
    t = mean / se if se > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    crit = stats.t.ppf(1 - alpha / 2, df=j - 1)
    return {"mean_diff": mean, "ci_low": mean - crit * se, "ci_high": mean + crit * se,
            "t": t, "p": p, "n_resamples": j}


def summarize_resamples(values: Sequence[float]) -> dict[str, float]:
    """Mean, SD and 2.5/97.5 percentiles across resamples (an empirical range,
    *not* a confidence interval for the mean)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = percentile_ci(v)
    return {"mean": v.mean(), "sd": v.std(ddof=1), "p2.5": lo, "p97.5": hi, "n": v.size}
