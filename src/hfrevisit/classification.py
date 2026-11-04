"""Binary-classification replication of Chicco & Jurman (2020).

The ten classifiers mirror the families used in the original study. Every
model is wrapped in a pipeline so that pre-processing is fitted on the training
fold only.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C

METRICS = ["MCC", "F1", "Accuracy", "TPR", "TNR", "ROC_AUC", "PR_AUC"]


def make_classifiers(seed: int = C.RANDOM_SEED) -> dict[str, Pipeline]:
    """Ten classifiers following Chicco & Jurman (2020), Methods section."""
    def scaled(est):
        return Pipeline([("scale", StandardScaler()), ("clf", est)])

    return {
        "Random forest": scaled(RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)),
        "Decision tree": scaled(DecisionTreeClassifier(random_state=seed)),
        "Gradient boosting": scaled(GradientBoostingClassifier(random_state=seed)),
        "Logistic regression": scaled(LogisticRegression(max_iter=5000)),
        "One rule (stump)": scaled(DecisionTreeClassifier(max_depth=1, random_state=seed)),
        "Naive Bayes": scaled(GaussianNB()),
        "SVM (linear)": scaled(SVC(kernel="linear", random_state=seed)),
        "SVM (RBF)": scaled(SVC(kernel="rbf", gamma="scale", random_state=seed)),
        "k-nearest neighbours": scaled(KNeighborsClassifier(n_neighbors=5)),
        "Neural network (MLP)": scaled(MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000,
                                                     early_stopping=True, random_state=seed)),
    }


def scores(model, X) -> np.ndarray:
    """Continuous score for ranking metrics (probability or decision function)."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "TPR": tp / (tp + fn) if tp + fn else np.nan,
        "TNR": tn / (tn + fp) if tn + fp else np.nan,
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "PR_AUC": average_precision_score(y_true, y_score),
    }


def monte_carlo_cv(df: pd.DataFrame, scenarios: dict[str, list[str]], target: str = C.EVENT,
                   models: dict | None = None, n_splits: int = 100, test_size: float = 0.30,
                   seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Repeated stratified hold-out, **paired across scenarios**.

    The same ``n_splits`` partitions are used for every feature set, so the
    per-split difference between scenarios is a paired quantity suitable for
    :func:`stats.corrected_resampled_ttest`.
    """
    models = models or make_classifiers(seed)
    y = df[target].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    rows = []
    for split, (tr, te) in enumerate(sss.split(np.zeros(len(y)), y)):
        for scen, feats in scenarios.items():
            X = df[feats].values.astype(float)
            for name, template in models.items():
                m = clone(template).fit(X[tr], y[tr])
                met = classification_metrics(y[te], m.predict(X[te]), scores(m, X[te]))
                rows.append({"split": split, "scenario": scen, "model": name,
                             "n_features": len(feats), "n_train": len(tr), "n_test": len(te), **met})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, metrics: Iterable[str] = METRICS) -> pd.DataFrame:
    """Mean, SD and empirical 95% range (2.5th-97.5th percentile) per model x scenario."""
    out = []
    for (scen, model), g in results.groupby(["scenario", "model"], sort=False):
        row = {"scenario": scen, "model": model}
        for m in metrics:
            v = g[m].dropna()
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_sd"] = v.std(ddof=1)
            row[f"{m}_p2.5"], row[f"{m}_p97.5"] = np.percentile(v, [2.5, 97.5])
        out.append(row)
    return pd.DataFrame(out)
