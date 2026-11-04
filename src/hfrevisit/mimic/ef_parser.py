"""Rule-based extraction of left-ventricular ejection fraction from echo reports.

Priority:
1. explicit numeric value with a percent sign ("LVEF = 55%", "EF 30-35%",
   "LVEF >55%"); ranges are mapped to their midpoint, open bounds to
   bound +/- 5 points;
2. (optional) qualitative description of LV systolic function.

Qualitative values are flagged so that the primary analysis can be restricted
to numeric measurements. Parser accuracy should be reported against a manually
annotated random sample of reports (see ``docs/MANUSCRIPT_GUIDE.md``).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

EF_MIN, EF_MAX = 5.0, 85.0
_NUM = r"(\d{1,2}(?:\.\d+)?)"
_KEY = r"(?:lvef|l\.?v\.?\s*ejection\s+fraction|ejection\s+fraction|\bef\b)"
_FILL = r"(?:\s|:|=|~|\(|is|was|of|at|about|approximately|approx\.?|estimated|calculated|visually|biplane)*"
_BOUND = r"(>=|<=|>|<|greater\s+than|less\s+than|at\s+least)?\s*"
NUMERIC_RE = re.compile(
    _KEY + _FILL + _BOUND + _NUM + r"\s*%?\s*(?:(?:-|–|to)\s*" + _NUM + r")?\s*%",
    flags=re.IGNORECASE,
)

#: Ordered most-specific first.
QUALITATIVE = [
    (r"severely\s+(?:depressed|reduced|impaired)", 20.0),
    (r"moderate(?:ly)?\s+to\s+severe(?:ly)?\s+(?:depressed|reduced|impaired)", 27.5),
    (r"moderately\s+(?:depressed|reduced|impaired)", 35.0),
    (r"mild(?:ly)?\s+to\s+moderate(?:ly)?\s+(?:depressed|reduced|impaired)", 40.0),
    (r"mildly\s+(?:depressed|reduced|impaired)", 47.5),
    (r"hyperdynamic", 75.0),
    (r"(?:normal|preserved)\s+(?:global\s+)?(?:lv\s+|left\s+ventricular\s+)?systolic\s+function", 60.0),
]
QUALITATIVE_RE = [(re.compile(p, re.IGNORECASE), v) for p, v in QUALITATIVE]


def parse_ef(text: str, allow_qualitative: bool = True) -> tuple[float, str]:
    """Return ``(ef_percent, source)``; ``(nan, 'none')`` if nothing is found.

    ``source`` is one of ``numeric``, ``range``, ``bound``, ``qualitative``.
    """
    if not isinstance(text, str) or not text:
        return np.nan, "none"
    for m in NUMERIC_RE.finditer(text):
        bound, lo, hi = m.group(1), float(m.group(2)), m.group(3)
        if hi is not None:
            val, src = (lo + float(hi)) / 2.0, "range"
        elif bound:
            b = bound.lower()
            val = lo + 5.0 if (">" in b or "greater" in b or "least" in b) else lo - 5.0
            src = "bound"
        else:
            val, src = lo, "numeric"
        if EF_MIN <= val <= EF_MAX:
            return val, src
    if allow_qualitative:
        for rx, v in QUALITATIVE_RE:
            if rx.search(text):
                return v, "qualitative"
    return np.nan, "none"


def extract_index_ef(echo: pd.DataFrame, allow_qualitative: bool = True,
                     allow_prior: bool = True) -> pd.DataFrame:
    """One EF per index admission.

    Preference: earliest report during the index stay; otherwise (if
    ``allow_prior``) the most recent report in the look-back window before
    admission.
    """
    parsed = echo["note_text"].apply(lambda t: parse_ef(t, allow_qualitative))
    e = echo.assign(ejection_fraction=[p[0] for p in parsed], ef_source=[p[1] for p in parsed])
    e = e.dropna(subset=["ejection_fraction"])
    during = e[e["during_index_stay"].astype(bool)].sort_values("days_from_admission")
    chosen = during.groupby("HADM_ID", as_index=False).first()
    chosen["ef_timing"] = "index stay"
    if allow_prior:
        prior = e[~e["during_index_stay"].astype(bool) & (e["days_from_admission"] < 0)]
        prior = prior[~prior["HADM_ID"].isin(chosen["HADM_ID"])]
        prior = prior.sort_values("days_from_admission").groupby("HADM_ID", as_index=False).last()
        prior["ef_timing"] = "prior year"
        chosen = pd.concat([chosen, prior], ignore_index=True)
    return chosen[["HADM_ID", "ejection_fraction", "ef_source", "ef_timing", "days_from_admission"]]
