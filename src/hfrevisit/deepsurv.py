"""DeepSurv (Katzman et al., 2018) with a scikit-learn compatible interface.

Requires PyTorch (``pip install torch``). Differences from the original
project notebook, which are required for a valid evaluation:

* the early-stopping validation set is a *random, event-stratified* subset of
  the training fold (the public CSV is sorted by follow-up time, so slicing the
  first 20 % of rows selected the shortest-follow-up patients);
* a Breslow baseline hazard is estimated so that survival curves, Brier score
  and IBS can be computed exactly as for the other models.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.functions import StepFunction

from . import config as C


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("DeepSurv requires PyTorch: pip install torch") from exc
    import torch
    return torch


def cox_partial_likelihood_loss(log_risk, time, event):
    """Negative Breslow partial log-likelihood (mean over events).

    Sorting by descending time makes the cumulative log-sum-exp equal to the
    log of the sum over the risk set ``{j : t_j >= t_i}``.
    """
    torch = _require_torch()
    order = torch.argsort(time, descending=True)
    lr, ev = log_risk.view(-1)[order], event.view(-1)[order]
    log_cum = torch.logcumsumexp(lr, dim=0)
    n_events = ev.sum().clamp(min=1.0)
    return -((lr - log_cum) * ev).sum() / n_events


class DeepSurv(BaseEstimator):
    def __init__(self, hidden=(32, 16), dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 max_epochs=500, patience=30, val_size=0.2, random_state=C.RANDOM_SEED,
                 device="cpu"):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_size = val_size
        self.random_state = random_state
        self.device = device

    def _net(self, n_in):
        torch = _require_torch()
        nn = torch.nn
        layers, prev = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1, bias=False))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch = _require_torch()
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        t, e = y["time"].astype(np.float32), y["event"].astype(np.float32)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)
        tr, va = train_test_split(np.arange(len(X)), test_size=self.val_size,
                                  stratify=e, random_state=self.random_state)
        T = lambda a: torch.tensor(a, device=self.device)  # noqa: E731
        Xtr, ttr, etr = T(Xs[tr]), T(t[tr]), T(e[tr])
        Xva, tva, eva = T(Xs[va]), T(t[va]), T(e[va])
        self.net_ = self._net(X.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net_.train()
            opt.zero_grad()
            loss = cox_partial_likelihood_loss(self.net_(Xtr), ttr, etr)
            loss.backward()
            opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = cox_partial_likelihood_loss(self.net_(Xva), tva, eva).item()
            if v < best - 1e-6:
                best, best_state, wait = v, copy.deepcopy(self.net_.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self._fit_baseline(Xs, t, e)
        return self

    def _log_risk(self, Xs):
        torch = _require_torch()
        self.net_.eval()
        with torch.no_grad():
            return self.net_(torch.tensor(Xs, dtype=torch.float32, device=self.device)).cpu().numpy().ravel()

    def _fit_baseline(self, Xs, t, e):
        """Breslow estimator of the cumulative baseline hazard."""
        r = np.exp(self._log_risk(Xs))
        times = np.unique(t[e == 1])
        h0 = np.array([e[t == s].sum() / r[t >= s].sum() for s in times])
        self.baseline_times_ = times
        self.cum_baseline_hazard_ = np.cumsum(h0)

    def predict(self, X):
        return self._log_risk(self.scaler_.transform(np.asarray(X, dtype=np.float32)))

    def predict_survival_function(self, X):
        r = np.exp(self.predict(X))
        x = np.concatenate([[0.0], self.baseline_times_])
        H = np.concatenate([[0.0], self.cum_baseline_hazard_])
        return np.array([StepFunction(x, np.exp(-H * ri)) for ri in r])
