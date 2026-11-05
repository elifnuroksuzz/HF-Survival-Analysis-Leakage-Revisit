"""Validation of the rule-based EF parser against manual annotation.

Workflow
  1. ``sample_for_annotation`` draws a stratified random sample of echo
     reports (by parser outcome) and writes a sheet with an empty
     ``ef_manual`` column. Two annotators fill it independently
     (``ef_manual_a``, ``ef_manual_b``); "NA" when no EF is stated.
  2. ``score_annotation`` reports inter-rater agreement and parser accuracy:
     detection sensitivity/PPV, mean absolute error, Bland-Altman limits and
     agreement on the HFrEF/HFmrEF/HFpEF category (ESC 2021: <40, 40-49, >=50).

The annotation sheet contains free text from MIMIC-III and must stay on the
credentialed machine (never commit it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ef_parser import parse_ef


def ef_category(v: float) -> str:
    if not np.isfinite(v):
        return "missing"
    return "HFrEF" if v < 40 else ("HFmrEF" if v < 50 else "HFpEF")


def sample_for_annotation(echo: pd.DataFrame, n: int = 200, seed: int = 42) -> pd.DataFrame:
    parsed = echo["note_text"].apply(parse_ef)
    e = echo.assign(ef_parser=[p[0] for p in parsed], ef_source=[p[1] for p in parsed])
    # oversample failures and qualitative values so that their error is estimable
    weights = e["ef_source"].map({"numeric": 1.0, "range": 2.0, "bound": 2.0,
                                  "qualitative": 3.0, "none": 3.0}).fillna(1.0)
    s = e.sample(n=min(n, len(e)), weights=weights, random_state=seed)
    out = s[["note_id", "HADM_ID", "note_text", "ef_parser", "ef_source"]].copy()
    out["sampling_weight"] = weights.loc[s.index].values
    out["ef_manual_a"] = ""
    out["ef_manual_b"] = ""
    return out.reset_index(drop=True)


def _num(col: pd.Series) -> np.ndarray:
    """Numbers typed by annotators: accepts "NA"/blank and decimal commas ("55,5")."""
    txt = col.astype("string").str.strip().str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(txt.mask(txt.str.upper().isin(["NA", "N/A", "", "-", "NAN"])),
                         errors="coerce").to_numpy(dtype=float, na_value=np.nan)


REQUIRED_COLUMNS = ("ef_parser", "sampling_weight", "ef_manual_a", "ef_manual_b")


def read_annotation_sheet(path) -> pd.DataFrame:
    """Read the sheet whether saved with ',' or ';' (Excel in Turkish/European locales)."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    seps = sorted((",", ";", "\t"), key=header.count, reverse=True)  # header has no free text
    for sep in seps:
        try:
            d = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except pd.errors.ParserError:
            continue
        if set(REQUIRED_COLUMNS) <= set(d.columns):
            return d
    raise ValueError(f"{path}: columns {REQUIRED_COLUMNS} not found; keep the header row unchanged.")


def score_annotation(sheet: pd.DataFrame) -> dict:
    """Parser accuracy vs. the consensus of two annotators (mean if both numeric).

    Estimates are weighted by inverse sampling weight to represent all reports.
    """
    a, b = _num(sheet["ef_manual_a"]), _num(sheet["ef_manual_b"])
    p = _num(sheet["ef_parser"])
    w = 1.0 / _num(sheet["sampling_weight"])
    both = np.isfinite(a) & np.isfinite(b)
    ref = np.where(both, (a + b) / 2, np.where(np.isfinite(a), a, b))
    has_ref, has_p = np.isfinite(ref), np.isfinite(p)

    def wmean(x, m):
        return float(np.sum(w[m] * x[m]) / np.sum(w[m])) if m.any() else np.nan

    agree = (np.isfinite(a) == np.isfinite(b))
    cat_a = np.array([ef_category(v) for v in a])
    cat_b = np.array([ef_category(v) for v in b])
    m = has_ref & has_p
    diff = p - ref
    out = {
        "n_annotated": int(len(sheet)),
        "interrater_detection_agreement": float(agree.mean()),
        "interrater_mae": float(np.nanmean(np.abs(a - b)[both])) if both.any() else np.nan,
        "interrater_category_agreement": float((cat_a == cat_b).mean()),
        "parser_detection_sensitivity": wmean(has_p.astype(float), has_ref),
        "parser_detection_ppv": wmean(has_ref.astype(float), has_p),
        "parser_mae": wmean(np.abs(diff), m),
        "bland_altman_bias": wmean(diff, m),
        "bland_altman_loa_low": np.nan, "bland_altman_loa_high": np.nan,
        "parser_category_agreement": wmean(
            (np.array([ef_category(v) for v in p]) == np.array([ef_category(v) for v in ref])).astype(float), m),
    }
    if m.sum() > 2:
        sd = float(np.std(diff[m], ddof=1))
        out["bland_altman_loa_low"] = out["bland_altman_bias"] - 1.96 * sd
        out["bland_altman_loa_high"] = out["bland_altman_bias"] + 1.96 * sd
    return out
