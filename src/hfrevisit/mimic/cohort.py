"""Assemble the MIMIC-III cohort and harmonise it to the development schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import queries
from .ef_parser import extract_index_ef

#: Physiological plausibility ranges in *development-cohort* units.
RANGES = {
    "serum_creatinine": (0.1, 20.0),
    "serum_sodium": (100.0, 180.0),
    "platelets": (5_000.0, 1_500_000.0),  # kiloplatelets/mL == platelets/uL
    "ejection_fraction": (5.0, 85.0),
}
#: Anaemia definition of the development cohort (Ahmad et al. 2017):
#: haematocrit below 36 %.
ANAEMIA_HCT_THRESHOLD = 36.0


#: Columns referenced in upper case by the Python code; all others are lower case.
_UPPER_COLUMNS = {"SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "ETHNICITY",
                  "ADMISSION_TYPE", "DOD", "CHARTDATE"}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names independent of the case BigQuery returned them in, and
    turn BigQuery extension dtypes (``dbdate``/``dbtime``) into plain pandas
    dtypes so the parquet cache can be read without ``db-dtypes``."""
    df = df.rename(columns=lambda c: c.upper() if c.upper() in _UPPER_COLUMNS else c.lower())
    for col in df.columns:
        if str(df[col].dtype) in ("dbdate", "dbtime"):
            df[col] = pd.to_datetime(df[col].astype("string"), errors="coerce")
    return df


def read_cache(path: Path) -> pd.DataFrame:
    """Read one cached extraction (also caches written by older versions)."""
    try:
        import db_dtypes  # noqa: F401 - registers dbdate for old caches
    except ImportError:
        pass
    return normalise_columns(pd.read_parquet(path))


def run_queries(gcp_project: str, cache_dir: Path, physionet_project: str = "physionet-data",
                refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Execute the four extraction queries (cached as parquet in ``cache_dir``).

    Do **not** commit the cache: MIMIC-III data must not be redistributed
    (PhysioNet Credentialed Health Data Use Agreement).
    """
    import pandas_gbq

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, builder in queries.ALL_QUERIES.items():
        f = cache_dir / f"mimic_{name}.parquet"
        if f.exists() and not refresh:
            out[name] = read_cache(f)
            continue
        print(f"[mimic] running BigQuery query '{name}' ...", flush=True)
        try:
            df = pandas_gbq.read_gbq(builder(physionet_project), project_id=gcp_project,
                                     progress_bar_type=None, location="US")
        except Exception as exc:
            raise RuntimeError(
                f"BigQuery query '{name}' failed: {exc}\n\nCheck that (1) you signed in with the "
                f"Google account that PhysioNet granted MIMIC-III BigQuery access to, and (2) "
                f"'{gcp_project}' is a Google Cloud project ID that this account can run "
                f"BigQuery jobs in (console.cloud.google.com, project selector).") from exc
        df = normalise_columns(df)
        print(f"[mimic]   {len(df):,} rows", flush=True)
        if df.empty:
            raise RuntimeError(f"BigQuery query '{name}' returned no rows; check MIMIC-III access.")
        df.to_parquet(f, index=False)
        out[name] = df
    return out


def build_cohort(raw: dict[str, pd.DataFrame], horizon_days: int = C.DEFAULT_HORIZON,
                 numeric_ef_only: bool = False, require_cpk: bool = False,
                 min_age: float = 18.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (harmonised cohort, attrition table)."""
    attrition = []

    def log(step, d):
        attrition.append({"step": step, "n": len(d),
                          "deaths_in_hospital": int(d["in_hospital_death"].sum())})

    df = raw["cohort"].copy()
    log("First admission with ICD-9 428.x (heart failure)", df)
    df = df[df["age"] >= min_age]
    log(f"Age >= {min_age:g} years", df)

    ef = extract_index_ef(raw["echo"], allow_qualitative=not numeric_ef_only)
    df = df.merge(ef, on="HADM_ID", how="inner")
    log("Ejection fraction available (index stay or prior year)"
        + (" — numeric only" if numeric_ef_only else ""), df)

    labs = raw["labs"].rename(columns={"creatinine": "serum_creatinine", "sodium": "serum_sodium",
                                       "cpk": "creatinine_phosphokinase"})
    df = df.merge(labs, on="HADM_ID", how="left")
    # Unit harmonisation: MIMIC platelets are K/uL; development cohort uses
    # kiloplatelets/mL (numerically = platelets/uL), i.e. x 1000.
    df["platelets"] = df["platelets"] * 1000.0
    df = df.dropna(subset=["serum_creatinine", "serum_sodium", "platelets", "hematocrit"])
    log("Creatinine, sodium, platelets, haematocrit within first 24 h", df)
    if require_cpk:
        df = df.dropna(subset=["creatinine_phosphokinase"])
        log("CPK available (complete-case sensitivity analysis)", df)

    for col, (lo, hi) in RANGES.items():
        df = df[df[col].between(lo, hi)]
    log("Values within physiological ranges", df)

    df = df.merge(raw["comorbidities"], on="HADM_ID", how="left")
    for col in ("diabetes", "high_blood_pressure", "smoking"):
        df[col] = df[col].fillna(0).astype(int)
    df["anaemia"] = (df["hematocrit"] < ANAEMIA_HCT_THRESHOLD).astype(int)

    # Outcomes ---------------------------------------------------------------
    # Time from admission to death (days); deaths on the admission day -> 0.5.
    dtd = df["days_to_death"].astype(float).clip(lower=0.5)
    died = dtd.notna() & (dtd <= horizon_days)
    df[C.TIME] = np.where(died, dtd, float(horizon_days))  # administrative censoring
    df[C.EVENT] = died.astype(int)
    df[f"death_{horizon_days}d"] = df[C.EVENT]
    df["in_hospital_death"] = df["in_hospital_death"].astype(int)

    cols = (["SUBJECT_ID", "HADM_ID"] + C.BASELINE_FEATURES +
            [C.TIME, C.EVENT, f"death_{horizon_days}d", "in_hospital_death",
             "ef_source", "ef_timing", "ETHNICITY", "ADMISSION_TYPE"])
    return df[cols].reset_index(drop=True), pd.DataFrame(attrition)


def impute_cpk(df: pd.DataFrame, value: float) -> pd.DataFrame:
    """Single imputation of missing CPK with a value from the *development*
    cohort (keeps the external cohort untouched by its own statistics)."""
    d = df.copy()
    d["cpk_imputed"] = d["creatinine_phosphokinase"].isna().astype(int)
    d["creatinine_phosphokinase"] = d["creatinine_phosphokinase"].fillna(value)
    return d
