import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hfrevisit import config as C
from hfrevisit.mimic import queries
from hfrevisit.mimic.cohort import build_cohort, impute_cpk
from hfrevisit.mimic.ef_parser import extract_index_ef, parse_ef


@pytest.mark.parametrize("text,expected,source", [
    ("Conclusions: LVEF = 55%.", 55.0, "numeric"),
    ("The left ventricular ejection fraction is estimated at 30-35%.", 32.5, "range"),
    ("Overall left ventricular systolic function is normal (LVEF>55%).", 60.0, "bound"),
    ("EF 20 %", 20.0, "numeric"),
    ("Severely depressed LV systolic function.", 20.0, "qualitative"),
    ("Moderately depressed left ventricular systolic function.", 35.0, "qualitative"),
    ("Height: 70 in, Weight: 180 lb, BP 120/80, HR 72", np.nan, "none"),
    ("EF 2 views were obtained", np.nan, "none"),   # no percent sign -> not an EF
])
def test_parse_ef(text, expected, source):
    val, src = parse_ef(text)
    assert src == source
    if np.isnan(expected):
        assert np.isnan(val)
    else:
        assert val == pytest.approx(expected)


def test_parse_ef_numeric_only_flag():
    assert np.isnan(parse_ef("Severely depressed LV function", allow_qualitative=False)[0])


def test_sql_design_choices():
    cohort = queries.cohort_sql()
    assert "ORDER BY a.ADMITTIME" in cohort and "MIN(HADM_ID)" not in cohort
    labs = queries.labs_sql()
    assert "50911" not in labs  # CK-MB is not CPK
    assert "INTERVAL 6 HOUR" in labs
    echo = queries.echo_sql()
    assert "during_index_stay" in echo and "CAST(idx.DISCHTIME AS DATE)" in echo


def _synthetic_raw(n=400, seed=0):
    rng = np.random.default_rng(seed)
    hadm = np.arange(1000, 1000 + n)
    died_days = np.where(rng.random(n) < 0.3, rng.integers(0, 400, n), np.nan)
    cohort = pd.DataFrame({
        "SUBJECT_ID": hadm + 50000, "HADM_ID": hadm, "age": rng.uniform(30, 90, n),
        "sex": rng.integers(0, 2, n), "in_hospital_death": (died_days < 7).astype(int),
        "days_to_death": died_days, "ETHNICITY": "WHITE", "ADMISSION_TYPE": "EMERGENCY"})
    labs = pd.DataFrame({
        "HADM_ID": hadm, "creatinine": rng.lognormal(0.2, 0.5, n), "sodium": rng.normal(138, 4, n),
        "platelets": rng.normal(230, 70, n).clip(20), "cpk": np.where(rng.random(n) < 0.4, np.nan,
                                                                       rng.lognormal(5, 1, n)),
        "hematocrit": rng.normal(35, 5, n), "hemoglobin": rng.normal(11.5, 2, n)})
    com = pd.DataFrame({"HADM_ID": hadm, "diabetes": rng.integers(0, 2, n),
                        "high_blood_pressure": rng.integers(0, 2, n), "smoking": rng.integers(0, 2, n)})
    ef = rng.integers(15, 70, n)
    echo = pd.DataFrame({"HADM_ID": hadm, "note_id": hadm, "days_from_admission": 1,
                         "during_index_stay": True,
                         "note_text": [f"Conclusions: LVEF = {v}%." for v in ef]})
    return {"cohort": cohort, "labs": labs, "comorbidities": com, "echo": echo}


def test_build_cohort_harmonisation():
    raw = _synthetic_raw()
    df, attr = build_cohort(raw, horizon_days=90)
    assert df["platelets"].median() > 50_000          # converted to Kaggle units
    assert set(df["anaemia"].unique()) <= {0, 1}
    assert (df[C.TIME] <= 90).all()
    assert df.loc[df[C.EVENT] == 0, C.TIME].eq(90).all()  # administrative censoring
    assert attr["n"].is_monotonic_decreasing
    d2 = impute_cpk(df, 250.0)
    assert d2["creatinine_phosphokinase"].notna().all() and d2["cpk_imputed"].sum() > 0


def test_ef_prefers_index_stay():
    echo = pd.DataFrame({"HADM_ID": [1, 1, 2], "note_id": [1, 2, 3],
                         "days_from_admission": [-100, 2, -30],
                         "during_index_stay": [False, True, False],
                         "note_text": ["LVEF = 60%", "LVEF = 25%", "LVEF = 40%"]})
    out = extract_index_ef(echo).set_index("HADM_ID")
    assert out.loc[1, "ejection_fraction"] == 25 and out.loc[1, "ef_timing"] == "index stay"
    assert out.loc[2, "ef_timing"] == "prior year"


def test_external_evaluation_end_to_end():
    """Fit development models on synthetic data and run the validation code path."""
    from sklearn.base import clone

    from hfrevisit.data import survival_target
    from hfrevisit.survival import make_survival_models

    spec = importlib.util.spec_from_file_location(
        "ev", Path(__file__).resolve().parents[1] / "scripts" / "run_external_validation.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    df, _ = build_cohort(_synthetic_raw(600, seed=1), horizon_days=90)
    df = impute_cpk(df, 250.0)
    dev = df.iloc[:300].assign(**{C.TIME: lambda d: d[C.TIME] * 3})  # any survival data
    y = survival_target(dev)
    models = {k: (f, clone(m).fit(dev[f], y)) for k, (f, m) in list(make_survival_models().items())[:2]}
    res, preds = ev.evaluate(df, models, tau=90, n_boot=20)
    assert {"harrell_c", "td_auc_90d", "cal_slope"} <= set(res.columns)
    assert all(((p >= 0) & (p <= 1)).all() for p in preds.values())


def test_ef_annotation_scoring():
    from hfrevisit.mimic.ef_validation import sample_for_annotation, score_annotation

    raw = _synthetic_raw(50)["echo"]
    raw.loc[:4, "note_text"] = "No measurement reported."
    sheet = sample_for_annotation(raw, n=30)
    truth = sheet["note_text"].str.extract(r"LVEF = (\d+)%")[0].fillna("NA")
    sheet["ef_manual_a"] = truth
    sheet["ef_manual_b"] = truth
    r = score_annotation(sheet.astype({"ef_manual_a": str, "ef_manual_b": str}))
    assert r["parser_mae"] == 0 and r["parser_detection_sensitivity"] == 1
    assert r["interrater_category_agreement"] == 1


def test_bigquery_output_normalisation_and_cache_roundtrip(tmp_path):
    """Lower-case column names and db-dtypes dates (as pandas-gbq returns them)."""
    db_dtypes = pytest.importorskip("db_dtypes")  # noqa: F841
    from hfrevisit.mimic.cohort import normalise_columns, read_cache

    raw = pd.DataFrame({"hadm_id": pd.array([1, 2], dtype="Int64"),
                        "chartdate": pd.array(["2150-01-01", None], dtype="dbdate"),
                        "Days_From_Admission": [0, -3], "NOTE_TEXT": ["LVEF 50%", "x"]})
    d = normalise_columns(raw)
    assert list(d.columns) == ["HADM_ID", "CHARTDATE", "days_from_admission", "note_text"]
    assert str(d["CHARTDATE"].dtype).startswith("datetime64")
    d.to_parquet(tmp_path / "c.parquet", index=False)
    back = read_cache(tmp_path / "c.parquet")
    assert back["CHARTDATE"].isna().tolist() == [False, True]


def test_annotation_sheet_roundtrip_excel_locales(tmp_path):
    from hfrevisit.mimic.ef_validation import (read_annotation_sheet, sample_for_annotation,
                                               score_annotation)

    raw = _synthetic_raw(60)["echo"]
    raw.loc[:9, "note_text"] = "No measurement reported."
    sheet = sample_for_annotation(raw, n=40)
    truth = sheet["note_text"].str.extract(r"LVEF = (\d+)%")[0]
    sheet["ef_manual_a"] = truth.fillna("NA")
    sheet["ef_manual_b"] = truth.fillna("")            # blank instead of NA
    f1 = tmp_path / "comma.csv"
    sheet.to_csv(f1, index=False)
    f2 = tmp_path / "semicolon.csv"                    # Excel (Turkish locale) style
    sheet.assign(sampling_weight=sheet["sampling_weight"].astype(str).str.replace(".", ",", regex=False),
                 ef_manual_a=sheet["ef_manual_a"].str.replace("NA", "na")).to_csv(f2, sep=";", index=False)
    for f in (f1, f2):
        r = score_annotation(read_annotation_sheet(f))
        assert r["parser_mae"] == 0 and r["parser_detection_sensitivity"] == 1


def test_sql_is_type_portable():
    for name, build in queries.ALL_QUERIES.items():
        sql = build()
        assert "DATETIME_DIFF" not in sql and "USING (" not in sql, name
        assert "DATE(idx." not in sql.replace("AS DATE)", ""), name
    try:
        import sqlglot
    except ImportError:
        return
    for build in queries.ALL_QUERIES.values():
        sqlglot.parse_one(build(), read="bigquery")
