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
  SELECT a.SUBJECT_ID, a.HADM_ID, a.ADMITTIME, a.DISCHTIME, a.HOSPITAL_EXPIRE_FLAG,
         a.ETHNICITY, a.ADMISSION_TYPE,
         ROW_NUMBER() OVER (PARTITION BY a.SUBJECT_ID ORDER BY a.ADMITTIME) AS rn
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
  idx.SUBJECT_ID, idx.HADM_ID, idx.ADMITTIME, idx.DISCHTIME, idx.ETHNICITY, idx.ADMISSION_TYPE,
  CASE WHEN DATETIME_DIFF(idx.ADMITTIME, p.DOB, DAY) / 365.2425 > 89 THEN 90.0
       ELSE DATETIME_DIFF(idx.ADMITTIME, p.DOB, DAY) / 365.2425 END AS age,
  IF(p.GENDER = 'M', 1, 0)                         AS sex,
  idx.HOSPITAL_EXPIRE_FLAG                         AS in_hospital_death,
  p.DOD,
  DATE_DIFF(DATE(p.DOD), DATE(idx.ADMITTIME), DAY) AS days_to_death
FROM idx
JOIN `{clinical}.patients` p USING (SUBJECT_ID)
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
  SELECT idx.HADM_ID, l.CHARTTIME, l.VALUENUM,
    CASE
{cases}
    END AS label
  FROM `{clinical}.labevents` l
  JOIN idx ON l.SUBJECT_ID = idx.SUBJECT_ID
  WHERE l.ITEMID IN ({all_ids})
    AND l.VALUENUM IS NOT NULL AND l.VALUENUM > 0
    AND l.CHARTTIME BETWEEN DATETIME_SUB(idx.ADMITTIME, INTERVAL 6 HOUR)
                        AND DATETIME_ADD(idx.ADMITTIME, INTERVAL 24 HOUR)
),
first_lab AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT labs.*, ROW_NUMBER() OVER (PARTITION BY labs.HADM_ID, labs.label
                                      ORDER BY labs.CHARTTIME) AS rn
    FROM labs) WHERE rn = 1
)
SELECT idx.HADM_ID,
  {pivots}
FROM idx LEFT JOIN first_lab USING (HADM_ID)
GROUP BY idx.HADM_ID
"""


def comorbidity_sql(project: str = "physionet-data") -> str:
    clinical, _ = _ds(project)

    def prefix_any(codes, n):
        return " OR ".join(f"SUBSTR(d.ICD9_CODE, 1, {n}) = '{c}'" for c in codes)

    smk = ", ".join(f"'{c}'" for c in SMOKING_CODES)
    return f"""
WITH {index_cte(clinical)}
SELECT idx.HADM_ID,
  MAX(IF({prefix_any(DM_PREFIX, 3)}, 1, 0))  AS diabetes,
  MAX(IF({prefix_any(HTN_PREFIX, 3)}, 1, 0)) AS high_blood_pressure,
  MAX(IF(SUBSTR(d.ICD9_CODE, 1, 4) IN ({smk}) OR d.ICD9_CODE IN ({smk}), 1, 0)) AS smoking
FROM idx LEFT JOIN `{clinical}.diagnoses_icd` d USING (HADM_ID)
GROUP BY idx.HADM_ID
"""


def echo_sql(project: str = "physionet-data", lookback_days: int = 365) -> str:
    """All echo reports from ``lookback_days`` before admission to discharge."""
    clinical, notes = _ds(project)
    return f"""
WITH {index_cte(clinical)}
SELECT idx.HADM_ID, n.ROW_ID AS note_id, n.CHARTDATE, n.HADM_ID AS note_hadm_id,
  DATE_DIFF(n.CHARTDATE, DATE(idx.ADMITTIME), DAY) AS days_from_admission,
  (n.HADM_ID = idx.HADM_ID
   OR n.CHARTDATE BETWEEN DATE(idx.ADMITTIME) AND DATE(idx.DISCHTIME)) AS during_index_stay,
  n.TEXT AS note_text
FROM idx
JOIN `{notes}.noteevents` n ON n.SUBJECT_ID = idx.SUBJECT_ID
WHERE n.CATEGORY = 'Echo'
  AND (n.ISERROR IS NULL OR n.ISERROR != 1)
  AND n.CHARTDATE BETWEEN DATE_SUB(DATE(idx.ADMITTIME), INTERVAL {lookback_days} DAY)
                      AND DATE(idx.DISCHTIME)
"""


ALL_QUERIES = {"cohort": cohort_sql, "labs": labs_sql, "comorbidities": comorbidity_sql, "echo": echo_sql}
