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
from hfrevisit.mimic.ef_validation import sample_for_annotation, score_annotation  # noqa: E402
from hfrevisit.reporting import save_table  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["sample", "score"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args(argv)
    P = C.get_paths(args.root)
    cache = P.data / "mimic_cache"
    sheet = cache / "ef_annotation_sheet.csv"
    if args.action == "sample":
        echo = pd.read_parquet(cache / "mimic_echo.parquet")
        sample_for_annotation(echo, n=args.n).to_csv(sheet, index=False)
        print(f"Annotation sheet written: {sheet}")
    else:
        res = score_annotation(pd.read_csv(sheet, keep_default_na=False))
        save_table(pd.DataFrame([res]), "mimic_ef_parser_validation", P.tables)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
