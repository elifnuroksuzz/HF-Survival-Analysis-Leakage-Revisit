"""Publication-quality figure helpers (vector PDF + 600-dpi PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

# Okabe-Ito colour-blind-safe palette
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
COLOR_ALIVE, COLOR_DEAD = OKABE_ITO[0], OKABE_ITO[1]


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 600, "font.family": "DejaVu Sans",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType (journal requirement)
    })


def save(fig, name: str, outdir: Path) -> list[Path]:
    """Save ``fig`` as PDF and PNG. Titles are intentionally omitted from
    figures — captions belong in the manuscript."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths


def label(f: str) -> str:
    return C.FEATURE_LABELS.get(f, f)


def leakage_figure(df: pd.DataFrame, lk: pd.DataFrame, model: str = "Random forest"):
    """Fig. 1: (a) follow-up by outcome, (b) MCC by feature set."""
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.4]})
    bins = np.arange(0, 300, 15)
    ax[0].hist(df.loc[df[C.EVENT] == 0, C.TIME], bins=bins, alpha=0.7, color=COLOR_ALIVE, label="Censored")
    ax[0].hist(df.loc[df[C.EVENT] == 1, C.TIME], bins=bins, alpha=0.7, color=COLOR_DEAD, label="Died")
    ax[0].set_xlabel("Follow-up time (days)")
    ax[0].set_ylabel("Patients")
    ax[0].legend(frameon=False)
    ax[0].text(-0.18, 1.02, "a", transform=ax[0].transAxes, fontweight="bold", fontsize=11)

    order = ["baseline", "baseline+shuffled time", "time only", "baseline+time"]
    names = {"baseline": "Baseline (11)", "baseline+shuffled time": "Baseline +\nshuffled time",
             "time only": "Time only", "baseline+time": "Baseline + time\n(Chicco & Jurman)"}
    d = lk[lk["model"] == model]
    data = [d.loc[d["feature_set"] == fs, "MCC"].values for fs in order]
    bp = ax[1].boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], [OKABE_ITO[0], OKABE_ITO[5], OKABE_ITO[4], OKABE_ITO[1]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    ax[1].set_xticks(range(1, 5), [names[o] for o in order])
    ax[1].set_ylabel(f"MCC ({model}, {d['split'].nunique()} splits)")
    ax[1].axhline(0, color="grey", lw=0.6, ls=":")
    ax[1].text(-0.12, 1.02, "b", transform=ax[1].transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def km_by_tertile(df: pd.DataFrame, features=("ejection_fraction", "serum_creatinine")):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    fig, axes = plt.subplots(1, len(features), figsize=(3.6 * len(features), 2.9), sharey=True)
    pvals = {}
    for ax, f, tag in zip(np.atleast_1d(axes), features, "abcdef"):
        grp = pd.qcut(df[f], 3, labels=["T1 (low)", "T2", "T3 (high)"], duplicates="drop")
        for g, c in zip(grp.cat.categories, OKABE_ITO):
            m = grp == g
            lo, hi = df.loc[m, f].min(), df.loc[m, f].max()
            KaplanMeierFitter().fit(df.loc[m, C.TIME], df.loc[m, C.EVENT],
                                    label=f"{g}: {lo:g}-{hi:g}").plot_survival_function(
                ax=ax, ci_alpha=0.12, color=c)
        p = multivariate_logrank_test(df[C.TIME], grp, df[C.EVENT]).p_value
        pvals[f] = p
        ax.set_xlabel("Days since enrolment")
        ax.set_title(f"{label(f)} — log-rank p {'<0.001' if p < 0.001 else f'= {p:.3f}'}", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        ax.text(-0.12, 1.04, tag, transform=ax.transAxes, fontweight="bold", fontsize=11)
    np.atleast_1d(axes)[0].set_ylabel("Survival probability")
    fig.tight_layout()
    return fig, pvals


def forest_plot(hr: pd.DataFrame):
    hr = hr.sort_values("HR")
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(hr) + 0.8))
    y = np.arange(len(hr))
    sig = hr["p"] < 0.05
    ax.errorbar(hr["HR"], y, xerr=[hr["HR"] - hr["CI_low"], hr["CI_high"] - hr["HR"]],
                fmt="none", ecolor="black", elinewidth=0.9, capsize=2)
    ax.scatter(hr["HR"], y, c=np.where(sig, OKABE_ITO[1], "grey"), zorder=3, s=22)
    ax.axvline(1, color="black", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{v} ({u})" for v, u in zip(hr["variable"], hr["unit"])])
    ax.set_xlabel("Hazard ratio (95% CI, log scale)")
    fig.tight_layout()
    return fig


def cindex_plot(summary: pd.DataFrame, metric: str = "harrell_c", xlabel: str = "Harrell's C-index"):
    s = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(4.8, 0.45 * len(s) + 0.8))
    y = np.arange(len(s))
    ax.errorbar(s[metric], y, xerr=[s[metric] - s[f"{metric}_p2.5"], s[f"{metric}_p97.5"] - s[metric]],
                fmt="o", color=OKABE_ITO[0], ecolor="grey", capsize=2)
    ax.axvline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_yticks(y, s["model"])
    ax.set_xlabel(f"{xlabel} (mean, 2.5-97.5 percentile over folds)")
    fig.tight_layout()
    return fig


def importance_plot(imp: pd.DataFrame):
    s = imp.sort_values("mean")
    fig, ax = plt.subplots(figsize=(4.8, 0.32 * len(s) + 0.8))
    y = np.arange(len(s))
    ax.barh(y, s["mean"], color=OKABE_ITO[0], alpha=0.8)
    ax.errorbar(s["mean"], y, xerr=[s["mean"] - s["p2_5"], s["p97_5"] - s["mean"]],
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, [label(f) for f in s["feature"]])
    ax.set_xlabel("Decrease in held-out C-index when permuted")
    fig.tight_layout()
    return fig


def calibration_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], n_bins: int = 10):
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Ideal")
    for (name, (y, p)), c in zip(curves.items(), OKABE_ITO):
        fr, mp = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mp, fr, "o-", ms=3, color=c, label=name)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed proportion")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig
