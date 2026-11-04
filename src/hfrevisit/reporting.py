"""Result persistence and provenance logging."""

from __future__ import annotations

import datetime as dt
import importlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from . import config as C

TRACKED_PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "sksurv", "lifelines",
                    "statsmodels", "shap", "matplotlib", "torch"]


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "hfrevisit": __version__, "seed": C.RANDOM_SEED}
    for pkg in TRACKED_PACKAGES:
        try:
            env[pkg] = importlib.import_module(pkg).__version__
        except Exception:  # noqa: BLE001
            env[pkg] = None
    return env


def save_table(df: pd.DataFrame, name: str, outdir: Path, float_fmt: str = "%.4f") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{name}.csv"
    df.to_csv(p, index=False, float_format=float_fmt)
    return p


def write_run_log(stage: str, outdir: Path, **extra) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    log = {"stage": stage, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
           "environment": environment(), **extra}
    p = outdir / f"{stage}_run_log.json"
    p.write_text(json.dumps(log, indent=2, default=str))
    with open(outdir / "run_history.jsonl", "a") as fh:  # append-only provenance
        fh.write(json.dumps(log, default=str) + "\n")
    return p
