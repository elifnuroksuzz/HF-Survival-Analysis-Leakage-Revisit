import numpy as np
import pandas as pd
import pytest
from sksurv.metrics import concordance_index_censored

from hfrevisit import config as C
from hfrevisit.calibration import calibration_intercept_slope, expected_calibration_error, recalibrate_cv
from hfrevisit.data import horizon_label, to_structured
from hfrevisit.stats import bootstrap_metric, corrected_resampled_ttest


def test_horizon_label_excludes_early_censoring():
    time = np.array([10, 50, 120, 30, 200])
    event = np.array([1, 0, 1, 1, 0])
    y = horizon_label(time, event, tau=90)
    # died at 10 -> 1; censored at 50 -> unknown; died at 120 -> alive at 90
    assert y[0] == 1 and np.isnan(y[1]) and y[2] == 0 and y[3] == 1 and y[4] == 0


def test_structured_array_field_names():
    y = to_structured([5.0, 7.0], [1, 0])
    assert y.dtype.names == ("event", "time")
    assert y["time"].tolist() == [5.0, 7.0]


def test_cindex_orientation():
    """sksurv expects *risk* scores (higher = earlier event). Passing the
    negated partial hazard — as the original notebook did — inverts C."""
    t = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    e = np.ones(6, dtype=bool)
    risk = -t  # perfectly ordered: short survival = high risk
    assert concordance_index_censored(e, t, risk)[0] == 1.0
    assert concordance_index_censored(e, t, -risk)[0] == 0.0


def test_corrected_ttest_is_wider_than_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.05, 0.05, 100)
    r = corrected_resampled_ttest(d, n_train=209, n_test=90)
    naive_half = 1.984 * d.std(ddof=1) / np.sqrt(100)
    assert (r["ci_high"] - r["ci_low"]) / 2 > 3 * naive_half


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    b = bootstrap_metric(np.mean, x, n_boot=300)
    assert b["ci_low"] < b["estimate"] < b["ci_high"]


def test_calibration_perfect_model():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.6, 20000)
    y = rng.binomial(1, p)
    cal = calibration_intercept_slope(y, p)
    assert abs(cal["citl"]) < 0.05 and abs(cal["slope"] - 1) < 0.08
    assert expected_calibration_error(y, p) < 0.02


def test_recalibration_fixes_overprediction():
    rng = np.random.default_rng(3)
    true_p = rng.uniform(0.02, 0.4, 5000)
    y = rng.binomial(1, true_p)
    p_bad = np.clip(true_p * 2.2, 0, 0.99)  # systematic over-prediction
    tab, _ = recalibrate_cv(y, p_bad)
    t = tab.set_index("method")
    assert t.loc["Logistic recalibration", "ece"] < t.loc["Original", "ece"] / 3


@pytest.mark.skipif(not C.get_paths().data_file.exists(), reason="dataset not downloaded")
def test_cohort_integrity_and_shuffle():
    from hfrevisit.data import load_cohort

    df = load_cohort()
    assert df.shape == (299, 13)
    assert df[C.EVENT].sum() == 96
    assert not df[C.TIME].is_monotonic_increasing  # loader de-sorts the file
