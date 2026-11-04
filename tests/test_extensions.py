import numpy as np
import pandas as pd
import pytest

from hfrevisit import config as C
from hfrevisit.clinical_utility import calibration_at_horizon, decision_curve, km_risk
from hfrevisit.samplesize import apparent_cox_snell_r2, riley_survival


def test_km_risk_without_censoring_equals_proportion():
    t = np.array([5, 10, 20, 40, 80, 100], float)
    e = np.ones(6, bool)
    assert km_risk(t, e, 30) == pytest.approx(3 / 6)


def test_decision_curve_model_equals_treat_all_when_everyone_positive():
    rng = np.random.default_rng(0)
    t = rng.uniform(1, 200, 300)
    e = rng.random(300) < 0.4
    d = decision_curve(t, e, np.ones(300), tau=90)
    assert np.allclose(d["net_benefit_model"], d["net_benefit_treat_all"])
    d0 = decision_curve(t, e, np.zeros(300), tau=90)
    assert (d0["net_benefit_model"] == 0).all()


def test_informative_risk_has_positive_net_benefit():
    rng = np.random.default_rng(1)
    risk = rng.uniform(0, 0.6, 2000)
    t = np.where(rng.random(2000) < risk, rng.uniform(1, 89, 2000), 200.0)
    e = t < 200
    d = decision_curve(t, e, risk, tau=90, thresholds=np.array([0.3]))
    assert d["net_benefit_model"].iloc[0] > max(d["net_benefit_treat_all"].iloc[0], 0)


def test_calibration_at_horizon_perfect():
    rng = np.random.default_rng(2)
    risk = rng.uniform(0.05, 0.5, 5000)
    died = rng.random(5000) < risk
    t = np.where(died, rng.uniform(1, 89, 5000), 120.0)
    _, s = calibration_at_horizon(t, died, risk, tau=90)
    assert s["observed_expected_ratio"] == pytest.approx(1.0, abs=0.06)


def test_riley_sample_size_monotone_in_parameters():
    base = dict(r2_cs=0.15, rate=0.0025, mean_followup=130, tau=90)
    a = riley_survival(n_params=5, **base)
    b = riley_survival(n_params=20, **base)
    assert b["minimum_n"] > a["minimum_n"]
    assert 0 < apparent_cox_snell_r2(50, 299) < 1


@pytest.mark.skipif(not C.get_paths().data_file.exists(), reason="dataset not downloaded")
def test_simulated_cohort_matches_event_rate():
    from hfrevisit.data import load_cohort
    from hfrevisit.simulation import simulate_cohort

    df = load_cohort()
    sim = simulate_cohort(df, np.zeros(len(C.BASELINE_FEATURES)), np.random.default_rng(0),
                          target_event_rate=0.32)
    assert abs(sim[C.EVENT].mean() - 0.32) < 0.02
    assert sim[C.TIME].max() <= df[C.TIME].max() + 1
