import importlib.util
import re
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "sr", Path(__file__).resolve().parents[1] / "scripts" / "systematic_review.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def test_abstract_reconstruction():
    assert sr.abstract({"follow-up": [1], "The": [0], "time": [2]}) == "The follow-up time"


def test_patterns_and_accuracy(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "OUT", tmp_path)
    pd.DataFrame([{"doi": "10.1/x", "title": "t", "year": 2022, "venue": "v", "pmcid": None,
                   "abstract": "Follow-up time was the most important feature. SMOTE was applied. "
                               "The random forest reached an accuracy of 97.5% with 10-fold cross-validation."}]
                 ).to_csv(tmp_path / "records.csv", index=False)
    sr.extract()
    row = pd.read_csv(tmp_path / "extraction_sheet_prefilled.csv").iloc[0]
    assert row.cand_time_top_feature and row.cand_smote and row.cand_cross_validation
    assert row.cand_max_accuracy_pct == 97.5
    assert not row.cand_survival_methods
