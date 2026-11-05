#!/usr/bin/env python
"""Validate the EF parser against manual annotation (MIMIC-III echo reports).

  # 1) draw 200 reports for two independent annotators
  python scripts/ef_annotation.py sample --n 200

  # 2) after filling ef_manual_a / ef_manual_b (number or NA)
  python scripts/ef_annotation.py score

The sheet (data/mimic_cache/ef_annotation_sheet.csv) contains MIMIC free text:
keep it on the credentialed machine. Only the aggregate scores are publishable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfrevisit import config as C  # noqa: E402
from hfrevisit.mimic.cohort import read_cache  # noqa: E402
from hfrevisit.mimic.ef_validation import (read_annotation_sheet, sample_for_annotation,  # noqa: E402
                                            score_annotation)  # noqa: E402
from hfrevisit.reporting import save_table  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["sample", "score"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--force", action="store_true", help="overwrite an existing annotation sheet")
    args = ap.parse_args(argv)
    P = C.get_paths(args.root)
    cache = P.data / "mimic_cache"
    sheet = cache / "ef_annotation_sheet.csv"
    if args.action == "sample":
        f = cache / "mimic_echo.parquet"
        if not f.exists():
            sys.exit(f"{f} not found: run scripts/run_external_validation.py first.")
        if sheet.exists() and not args.force:
            print(f"Annotation sheet already exists, left unchanged (use --force to redraw): {sheet}")
            return
        echo = read_cache(f)
        sample_for_annotation(echo, n=args.n).to_csv(sheet, index=False)
        print(f"Annotation sheet written: {sheet}")
    else:
        if not sheet.exists():
            sys.exit(f"{sheet} not found: run 'python scripts/ef_annotation.py sample' first.")
        d = read_annotation_sheet(sheet)
        filled = (d["ef_manual_a"].str.strip() != "") | (d["ef_manual_b"].str.strip() != "")
        if not filled.any():
            sys.exit(f"No annotations yet: fill ef_manual_a / ef_manual_b in {sheet} first.")
        res = score_annotation(d)
        save_table(pd.DataFrame([res]), "mimic_ef_parser_validation", P.tables)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
