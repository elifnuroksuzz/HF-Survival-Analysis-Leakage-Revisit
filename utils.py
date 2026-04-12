"""
utils.py — HF Revisit Project Ortak Yardımcı Fonksiyonlar
==========================================================
Tüm notebook'lar bu dosyayı import eder.
Hiçbir notebook'ta tekrar eden kod bulunmaz.

Kullanım:
    import sys
    sys.path.append('/content/drive/MyDrive/HF_Revisit_Project')
    from utils import set_seed, save_results, load_and_verify_data, COLUMN_MAP, RESULTS_DIR
"""

import os
import sys
import json
import random
import hashlib
import logging
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. PROJE SABİTLERİ
# ──────────────────────────────────────────────

# Bu iki satırı kendi Drive yapınıza göre değiştirin:
PROJECT_ROOT = Path("/content/drive/MyDrive/HF_Revisit_Project")
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR  / "figures"
MODELS_DIR   = PROJECT_ROOT / "models"
LOGS_DIR     = PROJECT_ROOT / "logs"

RANDOM_SEED  = 42
DATA_FILE    = "heart_failure_clinical_records_dataset.csv"
MANIFEST_FILE = DATA_DIR / "data_manifest.json"

# Feature kolonlarının standart isimleri
# Kaggle'daki orijinal isimler zaten temiz ama
# downstream uyumsuzlukları önlemek için map tutuyoruz.
COLUMN_MAP = {
    "age":                       "age",
    "anaemia":                   "anaemia",
    "creatinine_phosphokinase":  "creatinine_phosphokinase",
    "diabetes":                  "diabetes",
    "ejection_fraction":         "ejection_fraction",
    "high_blood_pressure":       "high_blood_pressure",
    "platelets":                 "platelets",
    "serum_creatinine":          "serum_creatinine",
    "serum_sodium":              "serum_sodium",
    "sex":                       "sex",
    "smoking":                   "smoking",
    "time":                      "time",
    "DEATH_EVENT":               "DEATH_EVENT",
}

# Continuous feature'lar (EDA'da kullanılacak)
CONTINUOUS_FEATURES = [
    "age", "creatinine_phosphokinase", "ejection_fraction",
    "platelets", "serum_creatinine", "serum_sodium", "time",
]

# Binary feature'lar
BINARY_FEATURES = [
    "anaemia", "diabetes", "high_blood_pressure", "sex", "smoking",
]

TARGET = "DEATH_EVENT"


# ──────────────────────────────────────────────
# 2. REPRODUCIBILITY
# ──────────────────────────────────────────────

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Tüm random kaynaklarını sabitler.
    Her notebook'un ilk hücresinde çağrılmalıdır.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch varsa onu da sabitle (NB03 için)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[utils] Random seed sabitlendi → {seed}")


# ──────────────────────────────────────────────
# 3. KLASÖR YAPISI
# ──────────────────────────────────────────────

def create_project_structure() -> None:
    """
    Proje klasör yapısını oluşturur.
    Varsa dokunmaz (exist_ok=True), yoksa açar.
    """
    dirs = [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("[utils] Klasör yapısı hazır:")
    for d in dirs:
        print(f"         {d}")


# ──────────────────────────────────────────────
# 4. VERİ DOĞRULAMA (SHA256 HASH)
# ──────────────────────────────────────────────

def compute_sha256(filepath: Path) -> str:
    """Dosyanın SHA256 hash'ini döndürür."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_data_manifest(filepath: Path, df: pd.DataFrame) -> dict:
    """
    Veri dosyasının manifest kaydını oluşturur ve kaydeder.
    İçerik: hash, tarih, boyut, kolon isimleri.
    """
    manifest = {
        "filename":    filepath.name,
        "sha256":      compute_sha256(filepath),
        "downloaded":  datetime.datetime.now().isoformat(),
        "rows":        int(df.shape[0]),
        "columns":     int(df.shape[1]),
        "column_names": list(df.columns),
        "positive_rate": float(df[TARGET].mean()),
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[utils] Manifest kaydedildi → {MANIFEST_FILE}")
    print(f"        SHA256 : {manifest['sha256'][:16]}...")
    print(f"        Boyut  : {manifest['rows']} satır × {manifest['columns']} sütun")
    return manifest


def verify_data_integrity(filepath: Path) -> bool:
    """
    Mevcut dosyanın hash'ini manifest ile karşılaştırır.
    Eşleşmezse ValueError fırlatır — veri değişmiş demektir.
    """
    if not MANIFEST_FILE.exists():
        print("[utils] UYARI: Manifest bulunamadı, doğrulama atlanıyor.")
        return False

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    current_hash = compute_sha256(filepath)
    if current_hash != manifest["sha256"]:
        raise ValueError(
            f"VERİ BÜTÜNLÜĞÜ HATASI!\n"
            f"  Beklenen hash : {manifest['sha256'][:16]}...\n"
            f"  Mevcut hash   : {current_hash[:16]}...\n"
            f"  Veri dosyası değişmiş olabilir."
        )
    print(f"[utils] Veri bütünlüğü doğrulandı ✓  ({current_hash[:16]}...)")
    return True


def load_and_verify_data(verify: bool = True) -> pd.DataFrame:
    """
    Veriyi yükler, kolonları standardize eder,
    isteğe bağlı olarak hash doğrulaması yapar.
    """
    filepath = DATA_DIR / DATA_FILE
    if not filepath.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {filepath}\n"
            f"Lütfen önce NB00'ı çalıştırın."
        )

    if verify:
        verify_data_integrity(filepath)

    df = pd.read_csv(filepath)

    # Kolon isimlerini standardize et
    df = df.rename(columns=COLUMN_MAP)

    # Dtype kontrolü — binary kolonlar int olmalı
    for col in BINARY_FEATURES + [TARGET]:
        df[col] = df[col].astype(int)

    return df


# ──────────────────────────────────────────────
# 5. LOGGING
# ──────────────────────────────────────────────

def get_logger(notebook_name: str) -> logging.Logger:
    """
    Her notebook için ayrı log dosyası oluşturur.
    Hem konsola hem dosyaya yazar.
    """
    logger = logging.getLogger(notebook_name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    # Dosya handler
    log_path = LOGS_DIR / f"{notebook_name}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # Konsol handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────
# 6. KAYDETME
# ──────────────────────────────────────────────

def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
    """
    Figürü /results/figures/ altına kaydeder.
    Filename örneği: 'NB00_class_balance'  (uzantı otomatik eklenir)
    """
    path = FIGURES_DIR / f"{filename}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[utils] Figür kaydedildi → {path}")
    return path


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """
    DataFrame'i /results/ altına kaydeder.
    Filename örneği: 'NB00_cohort_characteristics'
    """
    path = RESULTS_DIR / f"{filename}.csv"
    df.to_csv(path, index=True)
    print(f"[utils] CSV kaydedildi  → {path}")
    return path


def save_run_log(notebook_name: str, extra: dict = None) -> Path:
    """
    Notebook çalışma bilgilerini JSON olarak kaydeder.
    extra: notebook'a özel eklemek istediğin key-value çiftleri.
    """
    import platform
    log = {
        "notebook":       notebook_name,
        "timestamp":      datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform":       platform.platform(),
        "numpy_version":  np.__version__,
        "pandas_version": pd.__version__,
        "random_seed":    RANDOM_SEED,
    }

    # Opsiyonel kütüphaneler
    for lib in ["sklearn", "scipy", "matplotlib", "seaborn"]:
        try:
            mod = __import__(lib)
            log[f"{lib}_version"] = mod.__version__
        except ImportError:
            pass

    if extra:
        log.update(extra)

    path = LOGS_DIR / f"{notebook_name}_run_log.json"
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[utils] Run log kaydedildi → {path}")
    return path


# ──────────────────────────────────────────────
# 7. PLOT STİLİ (global)
# ──────────────────────────────────────────────

def set_plot_style() -> None:
    """
    Tüm figürler için tutarlı dergi kalitesi stil ayarları.
    Her notebook'un başında bir kez çağrılır.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "figure.titlesize":  14,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

# Renk paleti — tüm notebook'larda tutarlı
COLOR_ALIVE   = "#2196F3"   # Mavi → yaşayan
COLOR_DEAD    = "#F44336"   # Kırmızı → ölen
COLOR_NEUTRAL = "#9E9E9E"   # Gri → nötr
PALETTE       = [COLOR_ALIVE, COLOR_DEAD]
