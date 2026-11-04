"""Project-wide constants and paths.

Paths are resolved from the ``HF_PROJECT_ROOT`` environment variable so the
same code runs locally, on a server, or on Google Colab with Drive mounted::

    export HF_PROJECT_ROOT=/content/drive/MyDrive/HF_Revisit_Project
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------
# Dataset (Ahmad et al. 2017; distributed via UCI ML Repository / Kaggle)
# --------------------------------------------------------------------------
DATA_FILE = "heart_failure_clinical_records_dataset.csv"
KAGGLE_DATASET = "andrewmvd/heart-failure-clinical-data"
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/"
    "heart_failure_clinical_records_dataset.csv"
)
# SHA-256 of the canonical 299-row file (verified against the UCI/Kaggle copy).
DATA_SHA256 = "9c73cea7468ff5d517801ec050fe9993da5912fce4b56f296f8df3b38dd75912"

TIME = "time"          # follow-up duration in days (outcome-dependent!)
EVENT = "DEATH_EVENT"  # 1 = died during follow-up, 0 = censored

CONTINUOUS_FEATURES: list[str] = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
]
BINARY_FEATURES: list[str] = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
]
#: Baseline (admission-time) predictors — the only legitimate model inputs.
BASELINE_FEATURES: list[str] = CONTINUOUS_FEATURES + BINARY_FEATURES
#: Chicco & Jurman's "top-2" feature set.
TOP2_FEATURES: list[str] = ["ejection_fraction", "serum_creatinine"]
#: Right-skewed laboratory variables, log-transformed for (semi-)parametric models.
LOG_FEATURES: list[str] = ["creatinine_phosphokinase", "serum_creatinine", "platelets"]

FEATURE_LABELS: dict[str, str] = {
    "age": "Age (years)",
    "anaemia": "Anaemia",
    "creatinine_phosphokinase": "CPK (mcg/L)",
    "diabetes": "Diabetes",
    "ejection_fraction": "Ejection fraction (%)",
    "high_blood_pressure": "Hypertension",
    "platelets": "Platelets (kiloplatelets/mL)",
    "serum_creatinine": "Serum creatinine (mg/dL)",
    "serum_sodium": "Serum sodium (mEq/L)",
    "sex": "Male sex",
    "smoking": "Smoking",
    "time": "Follow-up time (days)",
    "DEATH_EVENT": "Death during follow-up",
}

#: Evaluation horizons (days) for time-dependent metrics. The cohort's maximum
#: follow-up is 285 days, so horizons beyond ~240 days are unstable.
EVAL_HORIZONS: tuple[int, ...] = (30, 60, 90, 180)
#: Fixed horizon used for binary "death within tau days" labels and for
#: external validation (MIMIC-III all-cause mortality from admission).
DEFAULT_HORIZON: int = 90


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path = field(init=False)
    results: Path = field(init=False)
    tables: Path = field(init=False)
    figures: Path = field(init=False)
    models: Path = field(init=False)
    logs: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", self.root / "data")
        object.__setattr__(self, "results", self.root / "results")
        object.__setattr__(self, "tables", self.root / "results" / "tables")
        object.__setattr__(self, "figures", self.root / "results" / "figures")
        object.__setattr__(self, "models", self.root / "models")
        object.__setattr__(self, "logs", self.root / "logs")

    def make(self) -> "Paths":
        for p in (self.data, self.tables, self.figures, self.models, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def data_file(self) -> Path:
        return self.data / DATA_FILE


def get_paths(root: str | os.PathLike | None = None) -> Paths:
    """Return project paths (``root`` > ``$HF_PROJECT_ROOT`` > repository root)."""
    if root is None:
        root = os.environ.get("HF_PROJECT_ROOT")
    if root is None:
        root = Path(__file__).resolve().parents[2]
    return Paths(Path(root)).make()
