#!/usr/bin/env python
"""Systematic literature search and text-mining pre-extraction.

Implements docs/manuscript/systematic_review_protocol.md.

Requires network access to api.openalex.org and www.ebi.ac.uk (Europe PMC).

  python scripts/systematic_review.py search    # records -> data/systematic_review/records.csv
  python scripts/systematic_review.py fulltext  # open-access full texts (Europe PMC)
  python scripts/systematic_review.py extract   # pre-filled extraction sheet for reviewers

Pre-filled values are *candidates* found by regular expressions; reviewers must
confirm or correct every value in the extraction sheet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "systematic_review"
UA = {"User-Agent": "hfrevisit-systematic-review/1.0"}

SEED_DOIS = {"ahmad2017": "10.1371/journal.pone.0181001", "chicco2020": "10.1186/s12911-020-1023-5"}
OPENALEX_SEARCH = '"heart failure clinical records"'
EUROPEPMC_QUERY = ('("heart failure clinical records" OR "heart_failure_clinical_records" OR '
                   '(Chicco AND Jurman AND "heart failure")) AND '
                   '("machine learning" OR classifier OR "survival")')


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed APIs
        return json.loads(r.read().decode())


def openalex_works(filter_: str | None = None, search: str | None = None) -> list[dict]:
    out, cursor = [], "*"
    while cursor:
        params = {"per-page": 200, "cursor": cursor}
        if filter_:
            params["filter"] = filter_
        if search:
            params["search"] = search
        data = get_json("https://api.openalex.org/works?" + urllib.parse.urlencode(params))
        out += data["results"]
        cursor = data["meta"].get("next_cursor")
        time.sleep(0.2)
    return out


def abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = [(i, w) for w, idx in inv.items() for i in idx]
    return " ".join(w for _, w in sorted(pos))


def search() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for key, doi in SEED_DOIS.items():
        seed = get_json(f"https://api.openalex.org/works/doi:{doi}")
        for w in openalex_works(filter_=f"cites:{seed['id'].split('/')[-1]}"):
            recs.append({"source": f"openalex_cites_{key}", **_rec(w)})
    for w in openalex_works(search=OPENALEX_SEARCH):
        recs.append({"source": "openalex_search", **_rec(w)})
    page, cursor = 0, "*"
    while cursor:
        q = urllib.parse.urlencode({"query": EUROPEPMC_QUERY, "format": "json", "pageSize": 1000,
                                    "cursorMark": cursor, "resultType": "core"})
        data = get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + q)
        for r in data.get("resultList", {}).get("result", []):
            recs.append({"source": "europepmc", "doi": (r.get("doi") or "").lower(), "title": r.get("title"),
                         "year": r.get("pubYear"), "venue": r.get("journalTitle"),
                         "abstract": r.get("abstractText", ""), "pmcid": r.get("pmcid"),
                         "is_oa": r.get("isOpenAccess") == "Y", "openalex_id": None})
        nxt = data.get("nextCursorMark")
        cursor = None if not nxt or nxt == cursor else nxt
        page += 1
    df = pd.DataFrame(recs)
    df["doi"] = df["doi"].fillna("").str.lower().str.replace("https://doi.org/", "", regex=False)
    df["dedup_key"] = df["doi"].where(df["doi"] != "", df["title"].str.lower().str[:120])
    agg = df.groupby("dedup_key", as_index=False).agg(
        {"doi": "first", "title": "first", "year": "first", "venue": "first", "abstract": "max",
         "pmcid": "first", "is_oa": "max", "openalex_id": "first",
         "source": lambda s: ";".join(sorted(set(s)))})
    agg["search_date_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    agg["include_title_abstract"] = ""
    agg["include_full_text"] = ""
    agg["exclusion_reason"] = ""
    agg.to_csv(OUT / "records.csv", index=False)
    print(f"{len(df)} hits -> {len(agg)} unique records")


def _rec(w: dict) -> dict:
    loc = (w.get("primary_location") or {}).get("source") or {}
    return {"doi": (w.get("doi") or "").lower(), "title": w.get("title"), "year": w.get("publication_year"),
            "venue": loc.get("display_name"), "abstract": abstract(w.get("abstract_inverted_index")),
            "pmcid": (w.get("ids") or {}).get("pmcid"), "is_oa": (w.get("open_access") or {}).get("is_oa"),
            "openalex_id": w.get("id")}


def fulltext() -> None:
    recs = pd.read_csv(OUT / "records.csv")
    ft = OUT / "fulltext"
    ft.mkdir(exist_ok=True)
    for r in recs.itertuples():
        if not isinstance(r.pmcid, str) or not r.pmcid:
            continue
        pmc = r.pmcid.split("/")[-1]
        f = ft / f"{pmc}.xml"
        if f.exists():
            continue
        try:
            req = urllib.request.Request(
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML", headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                f.write_bytes(resp.read())
        except Exception as exc:  # noqa: BLE001
            print("no full text:", pmc, exc)
        time.sleep(0.2)


# Candidate-extraction patterns (to be confirmed by reviewers)
PAT = {
    "mentions_time_feature": r"\b(follow[- ]?up (time|period|duration)|\btime\b feature|feature[s]? .{0,40}\btime\b)",
    "time_top_feature": r"(follow[- ]?up (time|period)|\btime\b).{0,80}(most (important|relevant|influential|significant)|ranked first|highest importance)",
    "time_excluded": r"(exclud|remov|drop|discard)\w*.{0,60}(follow[- ]?up|\btime\b)",
    "smote": r"\bSMOTE\b|oversampl",
    "survival_methods": r"Cox|Kaplan|survival forest|C-index|concordance index",
    "cross_validation": r"cross[- ]validation|k-fold|\b10-fold\b|\b5-fold\b",
}
ACC = re.compile(r"accuracy (?:of |=|:)?\s*(\d{2}(?:\.\d+)?)\s*%|accuracy (?:of |=|:)?\s*(0\.\d+)", re.I)


def extract() -> None:
    recs = pd.read_csv(OUT / "records.csv")
    rows = []
    for r in recs.itertuples():
        text = str(r.abstract or "")
        pmc = r.pmcid.split("/")[-1] if isinstance(r.pmcid, str) else None
        if pmc and (OUT / "fulltext" / f"{pmc}.xml").exists():
            text += " " + re.sub(r"<[^>]+>", " ", (OUT / "fulltext" / f"{pmc}.xml").read_text(errors="ignore"))
        row = {"doi": r.doi, "title": r.title, "year": r.year, "venue": r.venue,
               "full_text_available": bool(pmc and (OUT / 'fulltext' / f'{pmc}.xml').exists())}
        for k, p in PAT.items():
            m = re.search(p, text, re.I)
            row[f"cand_{k}"] = bool(m)
            row[f"cand_{k}_quote"] = text[max(0, m.start() - 80): m.end() + 80].replace("\n", " ") if m else ""
        accs = [float(a or b) * (100 if b else 1) for a, b in ACC.findall(text)]
        row["cand_max_accuracy_pct"] = max(accs) if accs else None
        for k in ("time_as_predictor", "censoring_handled", "oversampling_before_split",
                  "validation_scheme", "best_accuracy", "best_mcc", "best_auc", "code_available",
                  "reviewer_1", "reviewer_2", "notes"):
            row[k] = ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "extraction_sheet_prefilled.csv", index=False)
    print("Pre-filled extraction sheet:", OUT / "extraction_sheet_prefilled.csv")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["search", "fulltext", "extract"])
    a = ap.parse_args(argv)
    {"search": search, "fulltext": fulltext, "extract": extract}[a.action]()


if __name__ == "__main__":
    sys.exit(main())
