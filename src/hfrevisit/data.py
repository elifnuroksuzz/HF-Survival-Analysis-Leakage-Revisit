"""Data acquisition, integrity checking and schema validation."""

from __future__ import annotations

import hashlib
import os
import random
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def set_seed(seed: int = C.RANDOM_SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # pragma: no cover - torch optional
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(dest: Path, urls: tuple[str, ...] = C.DATA_URLS) -> Path:
    """Download the public CSV, trying each mirror until one matches the hash.

    The file is written to a temporary path first, so a failed or wrong
    download never replaces an existing copy. ZIP archives (UCI's current
    format) are unpacked.
    """
    import io
    import zipfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hfrevisit/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed public URLs
                payload = r.read()
            if payload[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(payload)) as z:
                    name = next(n for n in z.namelist() if n.endswith(C.DATA_FILE))
                    payload = z.read(name)
            if hashlib.sha256(payload).hexdigest() != C.DATA_SHA256:
                errors.append(f"{url}: SHA-256 mismatch")
                continue
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download the dataset from any mirror:\n  " + "\n  ".join(errors)
                       + f"\nDownload it manually from Kaggle ({C.KAGGLE_DATASET}) to {dest}.")


def validate_schema(df: pd.DataFrame) -> None:
    """Raise if the cohort does not match the published dataset description."""
    expected = set(C.BASELINE_FEATURES + [C.TIME, C.EVENT])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[list(expected)].isna().any().any():
        raise ValueError("Dataset contains missing values; expected none.")
    for col in C.BINARY_FEATURES + [C.EVENT]:
        if not set(np.unique(df[col])) <= {0, 1}:
            raise ValueError(f"Column {col!r} is not binary.")
    if (df[C.TIME] <= 0).any():
        raise ValueError("Follow-up time must be strictly positive.")


def load_cohort(path: Path | None = None, verify: bool = True,
                download: bool = False) -> pd.DataFrame:
    """Load the 299-patient cohort with integrity and schema checks.

    Parameters
    ----------
    path : CSV location (defaults to ``<root>/data/<DATA_FILE>``).
    verify : compare the file's SHA-256 with :data:`config.DATA_SHA256`.
    download : fetch from UCI if the file is absent.
    """
    path = Path(path) if path is not None else C.get_paths().data_file
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} not found. Download it from Kaggle ({C.KAGGLE_DATASET}) "
                "or call load_cohort(download=True)."
            )
        download_dataset(path)
    if verify:
        digest = sha256(path)
        if digest != C.DATA_SHA256:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: got {digest[:16]}..., "
                f"expected {C.DATA_SHA256[:16]}..."
            )
    df = pd.read_csv(path)
    validate_schema(df)
    for col in C.BINARY_FEATURES + [C.EVENT]:
        df[col] = df[col].astype(int)
    df[C.TIME] = df[C.TIME].astype(float)
    # The public file is sorted by follow-up time. Shuffle once with a fixed
    # seed so that any order-dependent code (e.g. naive validation slicing)
    # cannot silently select short-follow-up patients.
    df = df.sample(frac=1.0, random_state=C.RANDOM_SEED).reset_index(drop=True)
    return df


def to_structured(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Return a scikit-survival structured array with fields ``event``/``time``."""
    return np.array(
        list(zip(np.asarray(event, dtype=bool), np.asarray(time, dtype=float))),
        dtype=[("event", bool), ("time", float)],
    )


def survival_target(df: pd.DataFrame) -> np.ndarray:
    return to_structured(df[C.TIME].values, df[C.EVENT].values)


def horizon_label(time: np.ndarray, event: np.ndarray, tau: float) -> np.ndarray:
    """Binary label *death within ``tau`` days*; ``NaN`` if censored before ``tau``.

    This is the correct way to cast right-censored data as a classification
    problem: patients censored before ``tau`` have an unknown label and must be
    excluded (or handled with IPCW), never labelled as survivors.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    y = np.full(time.shape, np.nan)
    y[(event == 1) & (time <= tau)] = 1.0
    y[time > tau] = 0.0
    return y
