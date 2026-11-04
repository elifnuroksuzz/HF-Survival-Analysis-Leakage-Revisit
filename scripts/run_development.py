#!/usr/bin/env python
"""Development-cohort analyses (Ahmad et al. 2017 / Chicco & Jurman 2020 data).

Stages (run all by default):
  table1       Cohort characteristics (Table 1)
  replication  10 classifiers x {all features incl. time, baseline only, EF+SC}
  leakage      Experiments A-E proving target leakage of `time`
  survival     Repeated 5x10 CV of survival models, Cox HRs, PH tests, KM curves
  explain      Out-of-fold permutation importance and SHAP
  fairness     Subgroup C-index on out-of-fold risk
  export       Fit final models on the full cohort for external validation

Usage:
  python scripts/run_development.py                 # full run (~15-30 min CPU)
  python scripts/run_development.py --quick         # smoke test (~2 min)
  python scripts/run_development.py --stages survival fairness
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfrevisit import config as C  # noqa: E402
from hfrevisit import classification, explain, fairness, leakage, plotting, survival  # noqa: E402
from hfrevisit.data import load_cohort, set_seed, survival_target  # noqa: E402
from hfrevisit.reporting import save_table, write_run_log  # noqa: E402
from hfrevisit.stats import corrected_resampled_ttest, table_one  # noqa: E402

STAGES = ["table1", "replication", "leakage", "survival", "explain", "fairness", "export"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="project root (default $HF_PROJECT_ROOT or repo)")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--quick", action="store_true", help="few resamples, for smoke testing")
    ap.add_argument("--download", action="store_true", help="download the CSV from UCI if absent")
    ap.add_argument("--with-deepsurv", action="store_true", help="include DeepSurv (needs torch)")
    args = ap.parse_args(argv)

    P = C.get_paths(args.root)
    set_seed()
    plotting.set_style()
    df = load_cohort(P.data_file, download=args.download)
    n_mc = 10 if args.quick else 100
    n_rep = 2 if args.quick else 10
    n_boot = 200 if args.quick else 2000
    log: dict = {"quick": args.quick, "n_monte_carlo": n_mc, "n_cv_repeats": n_rep}
    T, F = P.tables, P.figures
    t0 = time.time()

    if "table1" in args.stages:
        save_table(table_one(df), "table1_cohort_by_outcome", T)
        print("[table1] done")

    if "replication" in args.stages:
        scen = {"S1: all features incl. time": C.BASELINE_FEATURES + [C.TIME],
                "S2: baseline features": C.BASELINE_FEATURES,
                "S3: EF + serum creatinine": C.TOP2_FEATURES}
        res = classification.monte_carlo_cv(df, scen, n_splits=n_mc)
        save_table(res, "replication_per_split", T)
        summ = classification.summarize(res)
        save_table(summ, "replication_summary", T)
        wide = res.pivot_table(index=["model", "split"], columns="scenario", values="MCC")
        rows = []
        for model, g in wide.groupby(level=0):
            r = corrected_resampled_ttest(g[list(scen)[0]] - g[list(scen)[1]],
                                          int(res.n_train.iloc[0]), int(res.n_test.iloc[0]))
            rows.append({"model": model, **r})
        save_table(pd.DataFrame(rows).sort_values("mean_diff", ascending=False),
                   "replication_delta_mcc_S1_minus_S2", T)
        ref = P.data / "reference" / "chicco2020_reported.csv"
        if ref.exists():  # user-supplied values transcribed from the paper
            rep = pd.read_csv(ref)
            save_table(rep.merge(summ, on=["scenario", "model"], how="left"), "replication_vs_published", T)
        print("[replication] done")

    if "leakage" in args.stages:
        a = leakage.experiment_a(df)
        save_table(a, "leakage_A_followup_by_status", T)
        log["leakage_A"] = dict(a.attrs)
        lk = leakage.leakage_monte_carlo(df, n_splits=n_mc)
        save_table(lk, "leakage_BCD_per_split", T)
        save_table(lk.groupby(["model", "feature_set"])[classification.METRICS].agg(["mean", "std"])
                   .reset_index().pipe(lambda d: d.set_axis([" ".join(c).strip() for c in d.columns], axis=1)),
                   "leakage_BCD_summary", T)
        save_table(pd.concat([leakage.paired_contrasts(lk, m) for m in ("MCC", "ROC_AUC")]),
                   "leakage_BCD_paired_tests", T)
        hz = pd.concat([leakage.horizon_reframing(df, tau=t, n_splits=n_mc) for t in (90, 180)])
        save_table(hz, "leakage_E_horizon_per_split", T)
        save_table(hz.groupby(["tau", "model"])[classification.METRICS + ["n_eligible", "event_rate"]]
                   .mean().reset_index(), "leakage_E_horizon_summary", T)
        plotting.save(plotting.leakage_figure(df, lk), "fig1_time_leakage", F)
        print("[leakage] done")

    models = survival.make_survival_models()
    if args.with_deepsurv:
        from hfrevisit.deepsurv import DeepSurv
        from sklearn.pipeline import Pipeline
        models["DeepSurv"] = (C.BASELINE_FEATURES, Pipeline([
            ("prep", survival.preprocessor(C.BASELINE_FEATURES)), ("model", DeepSurv())]))

    if "survival" in args.stages or "fairness" in args.stages:
        met, oof = survival.cross_validate(df, models, n_repeats=n_rep)
        save_table(met, "survival_cv_per_fold", T)
        summ = survival.summarize_cv(met)
        save_table(summ, "survival_cv_summary", T)
        save_table(oof, "survival_oof_risk", T)
        plotting.save(plotting.cindex_plot(summ), "fig3_survival_cindex", F)
        hr, ph, _ = survival.cox_hazard_ratios(df)
        save_table(hr.reset_index(names="covariate"), "table2_cox_hazard_ratios", T)
        save_table(ph.reset_index(names="covariate"), "table_s_ph_schoenfeld", T)
        plotting.save(plotting.forest_plot(hr), "fig2b_cox_forest", F)
        fig, p = plotting.km_by_tertile(df)
        plotting.save(fig, "fig2a_km_tertiles", F)
        log["km_logrank_p"] = p
        print("[survival] done")

    if "explain" in args.stages:
        feats, rsf = models["Random survival forest"]
        imp = explain.oof_permutation_importance(df, feats, rsf, n_repeats=1 if args.quick else 5,
                                                 n_perm=3 if args.quick else 10)
        save_table(imp, "importance_rsf_oof_permutation", T)
        plotting.save(plotting.importance_plot(imp), "fig4_rsf_importance", F)
        feats_c, cox = models["Cox PH"]
        save_table(explain.oof_permutation_importance(df, feats_c, cox, n_repeats=1 if args.quick else 5,
                                                      n_perm=3 if args.quick else 10),
                   "importance_cox_oof_permutation", T)
        try:
            sv = explain.oof_shap(df, feats, rsf)
            save_table(sv, "shap_rsf_oof_values", T)
            import shap
            import matplotlib.pyplot as plt
            shap.summary_plot(sv.values, df[feats].rename(columns=plotting.label), show=False,
                              plot_size=(5.5, 4))
            plotting.save(plt.gcf(), "fig4b_rsf_shap_beeswarm", F)
        except Exception as exc:  # noqa: BLE001 - SHAP is optional
            print(f"[explain] SHAP skipped: {exc}")
        print("[explain] done")

    if "fairness" in args.stages:
        tabs = []
        for model in ("Cox PH", "Random survival forest"):
            r = oof[oof["model"] == model]
            # average each patient's out-of-fold risk across repeats (ranks are
            # comparable within a repeat; we z-score per repeat first)
            r = r.assign(z=r.groupby("repeat")["risk"].transform(lambda v: (v - v.mean()) / v.std()))
            risk = r.groupby("row")["z"].mean().reindex(range(len(df))).values
            tabs.append(fairness.survival_subgroups(df, risk, n_boot=n_boot).assign(model=model))
        save_table(pd.concat(tabs), "table3_subgroup_cindex", T)
        print("[fairness] done")

    if "export" in args.stages:
        y = survival_target(df)
        fitted = {name: (feats, clone(tmpl).fit(df[feats], y)) for name, (feats, tmpl) in models.items()}
        with open(P.models / "development_survival_models.pkl", "wb") as fh:
            pickle.dump({"models": fitted, "cpk_median": float(df["creatinine_phosphokinase"].median()),
                         "seed": C.RANDOM_SEED}, fh)
        print("[export] done")

    log["elapsed_min"] = round((time.time() - t0) / 60, 2)
    write_run_log("development", P.logs, **log)
    print(f"Finished in {log['elapsed_min']} min. Tables: {T}  Figures: {F}")


if __name__ == "__main__":
    main()
