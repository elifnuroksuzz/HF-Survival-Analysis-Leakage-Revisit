#!/usr/bin/env python
"""External validation of development-cohort survival models on MIMIC-III.

Prerequisites
  1. PhysioNet credentialed access to MIMIC-III v1.4 linked to Google BigQuery.
  2. `python scripts/run_development.py --stages export` (creates
     models/development_survival_models.pkl).

The models are applied *without refitting*. Primary endpoint: all-cause
mortality within ``--horizon`` days of admission (default 90), using
PATIENTS.DOD (in-hospital + Social Security death records). Secondary:
in-hospital mortality (discrimination only — a different endpoint).

Usage:
  python scripts/run_external_validation.py --gcp-project my-gcp-project
  python scripts/run_external_validation.py --gcp-project my-gcp-project --horizon 180

Extracted MIMIC data are cached under <root>/data/mimic_cache (git-ignored).
Never commit or share them (PhysioNet Data Use Agreement).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402

from hfrevisit import config as C  # noqa: E402
from hfrevisit import calibration, fairness, plotting  # noqa: E402
from hfrevisit.data import set_seed, survival_target  # noqa: E402
from hfrevisit.mimic.cohort import build_cohort, impute_cpk, run_queries  # noqa: E402
from hfrevisit.reporting import save_table, write_run_log  # noqa: E402
from hfrevisit.stats import bootstrap_metric, table_one  # noqa: E402


def risk_at(model, X: pd.DataFrame, tau: float) -> np.ndarray:
    """Predicted probability of death by ``tau`` days: 1 - S(tau | x)."""
    return 1.0 - np.array([fn(tau) for fn in model.predict_survival_function(X)])


def evaluate(df: pd.DataFrame, models: dict, tau: int, n_boot: int) -> tuple[pd.DataFrame, dict]:
    y = survival_target(df)
    e, t = y["event"], y["time"]
    rows, preds = [], {}
    for name, (feats, m) in models.items():
        X = df[feats]
        lp = np.asarray(m.predict(X), dtype=float)
        p = risk_at(m, X, tau)
        preds[name] = p
        c = bootstrap_metric(lambda ee, tt, rr: concordance_index_censored(ee.astype(bool), tt, rr)[0],
                             e, t, lp, n_boot=n_boot, stratify=e.astype(int))
        # Follow-up is complete up to tau (administrative censoring at tau),
        # so the cumulative/dynamic AUC at tau equals the ordinary AUC of the
        # binary "death by tau" label; IPCW is unnecessary (and undefined at tau).
        auc_h = roc_auc_score(df[f"death_{tau}d"].values, lp)
        b = bootstrap_metric(roc_auc_score, df["in_hospital_death"].values, p, n_boot=n_boot,
                             stratify=df["in_hospital_death"].values)
        cal = calibration.calibration_report(df[f"death_{tau}d"].values, p)
        rows.append({"model": name, "harrell_c": c["estimate"], "harrell_c_ci_low": c["ci_low"],
                     "harrell_c_ci_high": c["ci_high"], f"td_auc_{tau}d": auc_h,
                     "auc_in_hospital_death": b["estimate"], "auc_ihd_ci_low": b["ci_low"],
                     "auc_ihd_ci_high": b["ci_high"],
                     **{f"cal_{k}": v for k, v in cal.items()}})
    return pd.DataFrame(rows), preds


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--gcp-project", required=True, help="your billing GCP project for BigQuery")
    ap.add_argument("--physionet-project", default="physionet-data")
    ap.add_argument("--horizon", type=int, default=C.DEFAULT_HORIZON)
    ap.add_argument("--refresh", action="store_true", help="re-run BigQuery queries")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    P = C.get_paths(args.root)
    set_seed()
    plotting.set_style()
    T, F = P.tables, P.figures
    tau = args.horizon

    bundle = pickle.loads((P.models / "development_survival_models.pkl").read_bytes())
    models, cpk_med = bundle["models"], bundle["cpk_median"]

    raw = run_queries(args.gcp_project, P.data / "mimic_cache", args.physionet_project, args.refresh)
    log = {"horizon_days": tau}
    analyses = {
        "primary": dict(numeric_ef_only=False, require_cpk=False),
        "sens_numeric_ef": dict(numeric_ef_only=True, require_cpk=False),
        "sens_complete_cpk": dict(numeric_ef_only=False, require_cpk=True),
        # case-mix closer to the development cohort (all patients aged >= 40)
        "sens_age40": dict(numeric_ef_only=False, require_cpk=False, min_age=40.0),
    }
    for label, kw in analyses.items():
        cohort, attrition = build_cohort(raw, horizon_days=tau, **kw)
        cohort = impute_cpk(cohort, cpk_med)
        save_table(attrition, f"mimic_{label}_attrition", T)
        res, preds = evaluate(cohort, models, tau, args.n_boot)
        save_table(res, f"mimic_{label}_performance", T)
        log[label] = {"n": len(cohort), "events": int(cohort[C.EVENT].sum()),
                      "cpk_imputed": int(cohort["cpk_imputed"].sum())}
        if label == "primary":
            save_table(table_one(cohort), "mimic_table1_by_outcome", T)
            best = res.sort_values("harrell_c", ascending=False)["model"].iloc[0]
            yb = cohort[f"death_{tau}d"].values
            cal_tab, recal = calibration.recalibrate_cv(yb, preds[best])
            save_table(cal_tab.assign(model=best), "mimic_primary_recalibration", T)
            plotting.save(plotting.calibration_plot({k: (yb, recal[k].values) for k in recal.columns}),
                          "fig5_mimic_calibration", F)
            fair = fairness.binary_subgroups(cohort, yb, recal["Logistic recalibration"].values,
                                             threshold=float(yb.mean()), n_boot=args.n_boot)
            save_table(fair.assign(model=best), "mimic_primary_subgroups", T)
            save_table(fairness.gap_summary(fair, [("Female", "Male"), ("Age <65", "Age >=65")]),
                       "mimic_primary_subgroup_gaps", T)
            log["best_model"] = best
    write_run_log("external_validation", P.logs, **log)
    print("External validation finished:", log)


if __name__ == "__main__":
    main()
