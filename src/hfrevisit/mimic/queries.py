"""BigQuery SQL for the MIMIC-III v1.4 heart-failure cohort.

Design decisions (each one fixes a problem in the original notebooks):

1. **Index admission = earliest admission by ``ADMITTIME``** with an ICD-9
   428.x code. ``MIN(HADM_ID)`` is *not* chronological in MIMIC-III.
2. **Age** is computed in days / 365.2425. Patients older than 89 have
   shifted dates of birth (apparent age ~300); they are set to 90.
3. **Laboratory values** are the first measurement in
   [ADMITTIME - 6 h, ADMITTIME + 24 h] (emergency-department draws often
   precede ADMITTIME). For sodium the earliest of the chemistry (50983) and
   blood-gas (50824) item is used — not the maximum of the two.
4. **CPK** uses ITEMID 50910 (creatine kinase, IU/L) only. ITEMID 50911 is
   CK-MB (ng/mL), a different analyte and unit.
5. **Ejection fraction** comes from echo reports linked to the *index*
   admission (HADM_ID match or chart date within the stay), or from the year
   before it. The original query took each subject's first echo ever, which
   could come from a later admission (future information).
6. **Outcomes**: in-hospital death and all-cause death from admission using
   ``PATIENTS.DOD`` (hospital + Social Security death records), so that a
   time-to-event endpoint comparable to the development cohort is available.

Portability: every date/time column is CAST explicitly (``AS DATE`` /
``AS DATETIME``) so the SQL is valid whether a BigQuery release stores it as
DATE, DATETIME or TIMESTAMP, and every output column has an explicit alias so
its name does not depend on the case used in the table schema.
"""

from __future__ import annotations

ITEMIDS = {
    "creatinine": (50912,),
    "sodium": (50983, 50824),
    "platelets": (51265,),
    "cpk": (50910,),
    "hematocrit": (51221,),
    "hemoglobin": (51222,),
}

DM_PREFIX = ("250",)
HTN_PREFIX = ("401", "402", "403", "404", "405")
SMOKING_CODES = ("3051", "V1582")  # tobacco use disorder; history of tobacco use


def _ds(project: str = "physionet-data") -> tuple[str, str]:
    return f"{project}.mimiciii_clinical", f"{project}.mimiciii_notes"


def index_cte(clinical: str) -> str:
    return f"""
hf_adm AS (
  SELECT a.SUBJECT_ID AS SUBJECT_ID, a.HADM_ID AS HADM_ID,
         CAST(a.ADMITTIME AS DATETIME) AS ADMITTIME, CAST(a.DISCHTIME AS DATETIME) AS DISCHTIME,
         a.HOSPITAL_EXPIRE_FLAG AS HOSPITAL_EXPIRE_FLAG,
         a.ETHNICITY AS ETHNICITY, a.ADMISSION_TYPE AS ADMISSION_TYPE,
         ROW_NUMBER() OVER (PARTITION BY a.SUBJECT_ID ORDER BY a.ADMITTIME, a.HADM_ID) AS rn
  FROM `{clinical}.admissions` a
  WHERE a.HADM_ID IN (SELECT HADM_ID FROM `{clinical}.diagnoses_icd`
                      WHERE ICD9_CODE LIKE '428%')
),
idx AS (SELECT * EXCEPT(rn) FROM hf_adm WHERE rn = 1)"""


def cohort_sql(project: str = "physionet-data") -> str:
    clinical, _ = _ds(project)
    return f"""
WITH {index_cte(clinical)}
SELECT
  idx.SUBJECT_ID AS SUBJECT_ID, idx.HADM_ID AS HADM_ID,
  idx.ADMITTIME AS ADMITTIME, idx.DISCHTIME AS DISCHTIME,
  idx.ETHNICITY AS ETHNICITY, idx.ADMISSION_TYPE AS ADMISSION_TYPE,
  CASE WHEN DATE_DIFF(CAST(idx.ADMITTIME AS DATE), CAST(p.DOB AS DATE), DAY) / 365.2425 > 89 THEN 90.0
       ELSE DATE_DIFF(CAST(idx.ADMITTIME AS DATE), CAST(p.DOB AS DATE), DAY) / 365.2425 END AS age,
  IF(p.GENDER = 'M', 1, 0)                         AS sex,
  CAST(idx.HOSPITAL_EXPIRE_FLAG AS INT64)          AS in_hospital_death,
  CAST(p.DOD AS DATE)                              AS DOD,
  DATE_DIFF(CAST(p.DOD AS DATE), CAST(idx.ADMITTIME AS DATE), DAY) AS days_to_death
FROM idx
JOIN `{clinical}.patients` p ON p.SUBJECT_ID = idx.SUBJECT_ID
"""


def labs_sql(project: str = "physionet-data") -> str:
    clinical, _ = _ds(project)
    cases = "\n".join(
        f"    WHEN l.ITEMID IN ({', '.join(map(str, ids))}) THEN '{name}'"
        for name, ids in ITEMIDS.items())
    all_ids = ", ".join(str(i) for ids in ITEMIDS.values() for i in ids)
    pivots = ",\n  ".join(f"MAX(IF(label = '{n}', VALUENUM, NULL)) AS {n}" for n in ITEMIDS)
    return f"""
WITH {index_cte(clinical)},
labs AS (
  -- labs are matched on SUBJECT_ID + time window because pre-admission
  -- (emergency department) draws carry a NULL HADM_ID
  SELECT idx.HADM_ID AS HADM_ID, CAST(l.CHARTTIME AS DATETIME) AS CHARTTIME,
    CAST(l.VALUENUM AS FLOAT64) AS VALUENUM,
    CASE
{cases}
    END AS label
  FROM `{clinical}.labevents` l
  JOIN idx ON l.SUBJECT_ID = idx.SUBJECT_ID
  WHERE l.ITEMID IN ({all_ids})
    AND l.VALUENUM IS NOT NULL AND l.VALUENUM > 0
    AND CAST(l.CHARTTIME AS DATETIME) BETWEEN DATETIME_SUB(idx.ADMITTIME, INTERVAL 6 HOUR)
                                          AND DATETIME_ADD(idx.ADMITTIME, INTERVAL 24 HOUR)
),
first_lab AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT labs.*, ROW_NUMBER() OVER (PARTITION BY labs.HADM_ID, labs.label
                                      ORDER BY labs.CHARTTIME) AS rn
    FROM labs) WHERE rn = 1
)
SELECT idx.HADM_ID AS HADM_ID,
  {pivots}
FROM idx LEFT JOIN first_lab ON first_lab.HADM_ID = idx.HADM_ID
GROUP BY idx.HADM_ID
"""


def comorbidity_sql(project: str = "physionet-data") -> str:
    clinical, _ = _ds(project)

    def prefix_any(codes, n):
        return " OR ".join(f"SUBSTR(d.ICD9_CODE, 1, {n}) = '{c}'" for c in codes)

    smk = ", ".join(f"'{c}'" for c in SMOKING_CODES)
    return f"""
WITH {index_cte(clinical)}
SELECT idx.HADM_ID AS HADM_ID,
  MAX(IF({prefix_any(DM_PREFIX, 3)}, 1, 0))  AS diabetes,
  MAX(IF({prefix_any(HTN_PREFIX, 3)}, 1, 0)) AS high_blood_pressure,
  MAX(IF(SUBSTR(d.ICD9_CODE, 1, 4) IN ({smk}) OR d.ICD9_CODE IN ({smk}), 1, 0)) AS smoking
FROM idx LEFT JOIN `{clinical}.diagnoses_icd` d ON d.HADM_ID = idx.HADM_ID
GROUP BY idx.HADM_ID
"""


def echo_sql(project: str = "physionet-data", lookback_days: int = 365) -> str:
    """All echo reports from ``lookback_days`` before admission to discharge."""
    clinical, notes = _ds(project)
    return f"""
WITH {index_cte(clinical)}
SELECT idx.HADM_ID AS HADM_ID, n.ROW_ID AS note_id, CAST(n.CHARTDATE AS DATE) AS CHARTDATE,
  n.HADM_ID AS note_hadm_id,
  DATE_DIFF(CAST(n.CHARTDATE AS DATE), CAST(idx.ADMITTIME AS DATE), DAY) AS days_from_admission,
  COALESCE(n.HADM_ID = idx.HADM_ID, FALSE)
   OR CAST(n.CHARTDATE AS DATE) BETWEEN CAST(idx.ADMITTIME AS DATE)
                                    AND CAST(idx.DISCHTIME AS DATE) AS during_index_stay,
  n.TEXT AS note_text
FROM idx
JOIN `{notes}.noteevents` n ON n.SUBJECT_ID = idx.SUBJECT_ID
WHERE n.CATEGORY = 'Echo'
  -- ISERROR is '1' (STRING) or 1 (INT64) depending on the release
  AND TRIM(COALESCE(CAST(n.ISERROR AS STRING), '')) NOT IN ('1', '1.0')
  AND CAST(n.CHARTDATE AS DATE) BETWEEN DATE_SUB(CAST(idx.ADMITTIME AS DATE), INTERVAL {lookback_days} DAY)
                                    AND CAST(idx.DISCHTIME AS DATE)
"""


ALL_QUERIES = {"cohort": cohort_sql, "labs": labs_sql, "comorbidities": comorbidity_sql, "echo": echo_sql}
