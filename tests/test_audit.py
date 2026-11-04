import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parents[1] / "scripts" / "audit_github.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("code,label", [
    ("X = df.drop('DEATH_EVENT', axis=1)\ny = df['DEATH_EVENT']\nmodel.fit(X, y)", "time_used"),
    ("X = df.drop(columns=['DEATH_EVENT'])\nrf.fit(X, y)", "time_used"),
    ("X = df.iloc[:, :-1]\nclf.fit(X, y)", "time_used"),
    ("features = ['age','ejection_fraction','serum_creatinine','time']\nclf.fit(df[features], y)", "time_used"),
    ("X = df.drop(['DEATH_EVENT', 'time'], axis=1)\nclf.fit(X, y)", "time_excluded"),
    ("X = df.drop(columns=['time','DEATH_EVENT'])\nclf.fit(X, y)", "time_excluded"),
    ("cph = CoxPHFitter()\ncph.fit(df, duration_col='time', event_col='DEATH_EVENT')", "survival"),
    ("X = df.drop('DEATH_EVENT', axis=1)\nclf.fit(X, y)\nX2 = df.drop(['DEATH_EVENT','time'], axis=1)\nclf.fit(X2, y)", "both"),
    ("df.describe()\nsns.heatmap(df.corr())", "no_model"),
    ("mod = smf.logit(formula='DEATH_EVENT ~ age + serum_creatinine', data=df)\nres = mod.fit()", "time_excluded"),
    ("mod = smf.logit(formula='DEATH_EVENT ~ age + time', data=df)\nres = mod.fit()", "time_used"),
    ("X = df.drop(['DEATH_EVENT', 'platelets'], axis=1)\nclf.fit(X, y)", "time_used"),
    ("target = 'DEATH_EVENT'\nX = data.drop(columns=[target])\nclf.fit(X, y)", "time_used"),
    ("X = df.iloc[:,0:11]\nclf.fit(X, y)", "time_excluded"),
    ("X = df[['age','ejection_fraction','serum_sodium','serum_creatinine']]\nclf.fit(X, y)", "time_excluded"),
    ("# X = df.drop('DEATH_EVENT', axis=1)\nX = df[cols]\nclf.fit(X, y)", "unclear"),
])
def test_classify_rules(code, label):
    assert audit.classify(code)["label"] == label


def test_extract_code_ignores_markdown_and_comments():
    nb = {"cells": [{"cell_type": "markdown", "source": ["X = df.drop('DEATH_EVENT', axis=1)"]},
                    {"cell_type": "code", "source": ["x = 1  # .fit(\n", "y = 2"]}]}
    code = audit.extract_code(json.dumps(nb).encode(), "a.ipynb")
    assert "DEATH_EVENT" not in code and ".fit(" not in code


def test_wilson_interval():
    lo, hi = audit.wilson(50, 100)
    assert lo < 0.5 < hi and 0.39 < lo < 0.41
