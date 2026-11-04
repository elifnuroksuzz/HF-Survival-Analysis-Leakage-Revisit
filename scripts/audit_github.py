#!/usr/bin/env python
"""Audit of public GitHub code that analyses the heart-failure dataset.

Question: how often is follow-up ``time`` used as a predictor of DEATH_EVENT?

Method (static analysis only — downloaded code is NEVER executed)
  1. GitHub code search for files (.ipynb, .py) that reference the dataset
     file name ``heart_failure_clinical_records_dataset``.
  2. Download each file's blob via the GitHub API (cached under
     data/github_audit/raw/, git-ignored).
  3. Extract code (notebook code cells only) and classify it with transparent
     rules (see ``classify``):
       survival      time used as the duration of a survival model
       time_used     a classifier is fitted on a feature matrix that includes time
       time_excluded time explicitly removed from the features
       both          analyses with and without time
       no_model      no model fitting found (EDA only)
       unclear       a model is fitted but the feature set cannot be resolved
  4. Aggregate per repository (forks excluded), report proportions with Wilson
     95% CIs, and draw a random sample for manual verification of the rules.

Requires an authenticated ``gh`` CLI. Search results change over time; the
snapshot date is recorded in the output.

  python scripts/audit_github.py collect      # search + download (rate-limited)
  python scripts/audit_github.py classify     # rules -> results/tables/github_audit_*.csv
  python scripts/audit_github.py verify       # agreement with manual labels
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import urllib.parse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "github_audit"
RAW = CACHE / "raw"
TABLES = ROOT / "results" / "tables"
QUERIES = [
    "heart_failure_clinical_records_dataset extension:ipynb",
    "heart_failure_clinical_records_dataset extension:py",
]


def gh(path: str, **params) -> dict:
    cmd = ["gh", "api", "-X", "GET", path]
    for k, v in params.items():
        cmd += ["-f", f"{k}={v}"]
    for attempt in range(5):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return json.loads(r.stdout)
        if "rate limit" in r.stderr.lower() or "secondary" in r.stderr.lower():
            time.sleep(30 * (attempt + 1))
            continue
        raise RuntimeError(r.stderr.strip()[:300])
    raise RuntimeError("rate limit not recovering")


def collect() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if (CACHE / "index.csv").exists():  # reuse an existing search snapshot
        return download(pd.read_csv(CACHE / "index.csv"))
    rows = []
    for q in QUERIES:
        for page in range(1, 11):  # API returns at most 1000 results per query
            res = gh("search/code", q=q, per_page=100, page=page)
            items = res.get("items", [])
            for it in items:
                rows.append({"query": q, "repo": it["repository"]["full_name"],
                             "fork": it["repository"].get("fork", False), "path": it["path"],
                             "sha": it["sha"], "html_url": it["html_url"]})
            print(f"{q!r} page {page}: {len(items)} (total {res.get('total_count')})", flush=True)
            if len(items) < 100:
                break
            time.sleep(7)  # code search: 10 requests/minute
    idx = pd.DataFrame(rows).drop_duplicates(["repo", "path"])
    idx["snapshot_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    idx.to_csv(CACHE / "index.csv", index=False)
    download(idx)


def download(idx: pd.DataFrame) -> None:
    for i, r in enumerate(idx.itertuples()):
        f = RAW / f"{r.sha}.blob"
        if f.exists():
            continue
        # public raw file at the commit referenced by the search hit
        raw_url = r.html_url.replace("https://github.com/", "https://raw.githubusercontent.com/", 1)
        raw_url = raw_url.replace("/blob/", "/", 1)
        head, _, rest = raw_url.partition("raw.githubusercontent.com/")
        raw_url = head + "raw.githubusercontent.com/" + urllib.parse.quote(urllib.parse.unquote(rest))
        res = subprocess.run(["curl", "-fsSL", "--max-time", "60", "-o", str(f), raw_url],
                             capture_output=True, text=True)
        if res.returncode != 0:
            f.unlink(missing_ok=True)
            print("skip", r.repo, r.path, res.stderr.strip()[:120])
        if i % 50 == 0:
            print(f"downloaded {i}/{len(idx)}", flush=True)
    print("done:", len(idx), "files")


# --------------------------------------------------------------------------
# Classification rules
# --------------------------------------------------------------------------
def extract_code(raw: bytes, path: str) -> str:
    text = raw.decode("utf-8", errors="replace")
    if path.endswith(".ipynb"):
        try:
            nb = json.loads(text)
            cells = nb.get("cells") or nb.get("worksheets", [{}])[0].get("cells", [])
            text = "\n".join("".join(c.get("source", c.get("input", ""))) for c in cells
                             if c.get("cell_type") == "code")
        except (json.JSONDecodeError, IndexError, AttributeError, TypeError):
            return ""
    return strip_comments(text)


def strip_comments(text: str) -> str:
    """Remove ``#`` comments so commented-out code does not count."""
    return "\n".join(re.sub(r"(^|\s)#.*$", "", line) for line in text.splitlines())


Q = r"""['"]"""
TIME = Q + r"time" + Q
FIT = re.compile(r"\.fit\(|cross_val_score|cross_validate|GridSearchCV|train_test_split|\.fit_predict\(|compare_models|setup\(")
SURV = re.compile(r"CoxPHFitter|KaplanMeierFitter|WeibullAFTFitter|sksurv|RandomSurvivalForest|"
                  r"duration_col\s*=\s*" + TIME + r"|Surv\.from|CoxPHSurvivalAnalysis", re.I)
# time removed: drop(...'time'...), del df['time'], != 'time', feature list comprehension excluding time
EXCL = re.compile(r"drop\([^)]*" + TIME + r"|del\s+\w+\[" + TIME + r"\]|!=\s*" + TIME +
                  r"|not\s+in\s*\[[^\]]*" + TIME + r"|exclude[^\n]*" + TIME, re.I)
# all columns except the target (time silently included)
ALLX = re.compile(r"drop\(\s*(?:columns\s*=\s*)?\[?\s*" + Q + r"DEATH_EVENT" + Q + r"\s*\]?\s*(?:,\s*axis\s*=\s*1\s*)?\)"
                  r"|drop\(\s*\[?\s*" + Q + r"DEATH_EVENT" + Q + r"\s*\]?\s*,\s*axis\s*=\s*(?:1|'columns')\s*\)"
                  r"|iloc\[\s*:\s*,\s*(?:0\s*)?:\s*(?:-1|12)\s*\]|iloc\[\s*:\s*,\s*0\s*:\s*12\s*\]"
                  r"|columns\[\s*:\s*-1\s*\]|\.values\[\s*:\s*,\s*:\s*-1\s*\]", re.I)
# explicit feature list that contains time
LIST_WITH_TIME = re.compile(r"\[[^\]\n]*" + Q + r"serum_creatinine" + Q + r"[^\]\n]*" + TIME + r"[^\]\n]*\]"
                            r"|\[[^\]\n]*" + TIME + r"[^\]\n]*" + Q + r"serum_creatinine" + Q + r"[^\]\n]*\]", re.I)


CLIN = ("age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
        "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking")
DROP_CALL = re.compile(r"drop\(([^)]*)\)", re.S)
TARGET_VAR = re.compile(r"(\w+)\s*=\s*" + Q + r"DEATH_EVENT" + Q)
# first 11 columns = the 11 baseline variables (time is column 12)
ILOC_11 = re.compile(r"iloc\[\s*:\s*,\s*(?:0\s*)?:\s*11\s*\]")
# X = df[[...]] with an explicit list of clinical variables
X_LIST = re.compile(r"^\s*(?:X|x|features?|feature_cols|cols|predictors|X_\w+)\s*=\s*(?:\w+\s*\[\s*)?\[([^\]]+)\]", re.M)


# statsmodels / patsy formula for the death indicator
FORMULA = re.compile(r"DEATH_EVENT\s*~\s*([^'\"]+)")


def _formula_rules(code: str) -> tuple[bool, bool]:
    """(formula with time on the right-hand side, formula without time)."""
    with_t = without_t = False
    for rhs in FORMULA.findall(code.replace("\\\n", " ")):
        if re.search(r"\btime\b", rhs):
            with_t = True
        else:
            without_t = True
    return with_t, without_t


def _drop_rules(code: str) -> tuple[bool, bool]:
    """(drops target but keeps time, drops time) from every drop(...) call."""
    targets = set(TARGET_VAR.findall(code))
    keeps_time = drops_time = False
    for args in DROP_CALL.findall(code):
        has_target = "DEATH_EVENT" in args or any(re.search(rf"\b{t}\b", args) for t in targets)
        has_time = re.search(TIME, args) is not None
        if has_target and not has_time:
            keeps_time = True
        if has_time:
            drops_time = True
    return keeps_time, drops_time


def _explicit_lists(code: str) -> tuple[bool, bool]:
    """(explicit feature list with time, explicit clinical list without time)."""
    with_t = without_t = False
    for body in X_LIST.findall(code):
        n_clin = sum(f"'{c}'" in body or f'"{c}"' in body for c in CLIN)
        if n_clin >= 2:
            if re.search(TIME, body):
                with_t = True
            else:
                without_t = True
    return with_t, without_t


def classify(code: str) -> dict:
    code = strip_comments(code)
    has_fit = bool(FIT.search(code))
    surv = bool(SURV.search(code))
    keeps_time, drops_time = _drop_rules(code)
    list_t, list_no_t = _explicit_lists(code)
    form_t, form_no_t = _formula_rules(code)
    excl = (bool(EXCL.search(code)) or drops_time or bool(ILOC_11.search(code)) or list_no_t
            or form_no_t)
    allx = bool(ALLX.search(code)) or keeps_time
    lst = bool(LIST_WITH_TIME.search(code)) or list_t or form_t
    used = allx or lst
    if not code.strip():
        label = "unreadable"
    elif not has_fit and not surv:
        label = "no_model"
    elif used and excl:
        label = "both"
    elif used:
        label = "time_used"
    elif excl:
        label = "time_excluded"
    elif surv:
        label = "survival"
    else:
        label = "unclear"
    return {"label": label, "has_fit": has_fit, "survival_code": surv, "excl_rule": excl,
            "all_features_rule": allx, "list_with_time_rule": lst}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (c - h, c + h)


REPO_PRIORITY = ["time_used", "both", "time_excluded", "survival", "unclear", "no_model", "unreadable"]


def run_classification(sample_n: int = 40, seed: int = 42) -> None:
    idx = pd.read_csv(CACHE / "index.csv")
    rows = []
    for r in idx.itertuples():
        f = RAW / f"{r.sha}.blob"
        code = extract_code(f.read_bytes(), r.path) if f.exists() else ""
        rows.append({"repo": r.repo, "fork": r.fork, "path": r.path, "html_url": r.html_url,
                     "n_code_chars": len(code), **classify(code)})
    files = pd.DataFrame(rows)
    files.to_csv(TABLES / "github_audit_files.csv", index=False)

    nf = files[~files["fork"].astype(bool)]
    rank = {k: i for i, k in enumerate(REPO_PRIORITY)}
    repo = (nf.assign(r=nf["label"].map(rank)).sort_values("r")
              .groupby("repo", as_index=False).first()[["repo", "label", "html_url"]])
    repo.to_csv(TABLES / "github_audit_repos.csv", index=False)

    modelling = repo[~repo["label"].isin(["no_model", "unreadable"])]
    n = len(modelling)
    summ = []
    for lab in ["time_used", "both", "time_excluded", "survival", "unclear"]:
        k = int((modelling["label"] == lab).sum())
        lo, hi = wilson(k, n)
        summ.append({"category": lab, "n_repos": k, "share_of_modelling_repos": k / n if n else np.nan,
                     "ci_low": lo, "ci_high": hi})
    k_any = int(modelling["label"].isin(["time_used", "both"]).sum())
    lo, hi = wilson(k_any, n)
    summ.append({"category": "time used as predictor in >=1 analysis", "n_repos": k_any,
                 "share_of_modelling_repos": k_any / n if n else np.nan, "ci_low": lo, "ci_high": hi})
    # sensitivity: de-duplicate repositories whose deciding file has identical content
    sha = idx.set_index(["repo", "path"])["sha"]
    dec = (nf.assign(r=nf["label"].map(rank)).sort_values("r").groupby("repo", as_index=False).first())
    dec["sha"] = [sha.get((r, p_), np.nan) if not isinstance(sha.get((r, p_)), pd.Series)
                  else sha.get((r, p_)).iloc[0] for r, p_ in zip(dec["repo"], dec["path"])]
    dedup = dec[~dec["label"].isin(["no_model", "unreadable"])].drop_duplicates("sha")
    k_d = int(dedup["label"].isin(["time_used", "both"]).sum())
    lo_d, hi_d = wilson(k_d, len(dedup))
    summ.append({"category": "time used as predictor (content-deduplicated)", "n_repos": k_d,
                 "share_of_modelling_repos": k_d / len(dedup), "ci_low": lo_d, "ci_high": hi_d})
    resolved = modelling[modelling["label"] != "unclear"]
    k_r = int(resolved["label"].isin(["time_used", "both"]).sum())
    lo_r, hi_r = wilson(k_r, len(resolved))
    summ.append({"category": "time used as predictor (among resolved repos)", "n_repos": k_r,
                 "share_of_modelling_repos": k_r / len(resolved), "ci_low": lo_r, "ci_high": hi_r})
    out = pd.DataFrame(summ)
    out["n_modelling_repos"] = n
    out["n_repos_total_nonfork"] = repo["repo"].nunique()
    out["n_files"] = len(files)
    out["snapshot_utc"] = idx["snapshot_utc"].iloc[0]
    out.to_csv(TABLES / "github_audit_summary.csv", index=False)
    print(out.to_string(index=False))

    # sample for manual verification of the rules
    ver = modelling.sample(min(sample_n, n), random_state=seed)[["repo", "label", "html_url"]]
    ver["manual_label"] = ""
    ver.to_csv(CACHE / "manual_verification_sheet.csv", index=False)
    print(f"Manual verification sheet: {CACHE / 'manual_verification_sheet.csv'}")


def verify() -> None:
    """Agreement between rules and manual labels (data/reference/github_audit_manual_labels.csv)."""
    man = pd.read_csv(ROOT / "data" / "reference" / "github_audit_manual_labels.csv")
    rep = pd.read_csv(TABLES / "github_audit_repos.csv")[["repo", "label"]]
    m = man.merge(rep, on="repo", how="left")
    used = {"time_used", "both"}
    m["rule_used"] = m["label"].isin(used)
    m["manual_used"] = m["manual_label"].isin(used)
    out = []
    for sample, g in m.groupby("sample", sort=False):
        tp = int((g.rule_used & g.manual_used).sum()); fp = int((g.rule_used & ~g.manual_used).sum())
        fn = int((~g.rule_used & g.manual_used).sum()); tn = int((~g.rule_used & ~g.manual_used).sum())
        lo, hi = wilson(tp + tn, len(g))
        out.append({"sample": sample, "n": len(g), "exact_label_agreement": float((g.label == g.manual_label).mean()),
                    "binary_agreement": (tp + tn) / len(g), "binary_agreement_ci_low": lo,
                    "binary_agreement_ci_high": hi, "ppv_time_used": tp / (tp + fp) if tp + fp else np.nan,
                    "sensitivity_time_used": tp / (tp + fn) if tp + fn else np.nan,
                    "npv": tn / (tn + fn) if tn + fn else np.nan, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    res = pd.DataFrame(out)
    res.to_csv(TABLES / "github_audit_verification.csv", index=False)
    m.to_csv(TABLES / "github_audit_verification_detail.csv", index=False)
    print(res.round(3).to_string(index=False))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["collect", "classify", "verify"])
    ap.add_argument("--sample", type=int, default=40)
    a = ap.parse_args(argv)
    TABLES.mkdir(parents=True, exist_ok=True)
    {"collect": collect, "classify": lambda: run_classification(a.sample), "verify": verify}[a.action]()


if __name__ == "__main__":
    sys.exit(main())
