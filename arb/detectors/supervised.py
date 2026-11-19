"""Supervised failure-state classifier (multinomial logistic regression, NumPy).

Why this model and not a transformer:

* the paper's claim is about *signals*, not capacity -- a linear model over
  monitor features is the sharpest possible test of "is this detectable at all
  from these signals";
* it trains in seconds, so every number in the paper is reproducible by a
  reviewer on a laptop;
* the released artifact is a JSON weight matrix, not an opaque pickle.

The model has 8 classes: the 7 failure types plus ``none``. Detection score is
``1 - P(none)``; the type posterior is the renormalised mass over the 7 types.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from arb.detectors.features import FEATURE_NAMES, vector
from arb.detectors.rules import Detection
from arb.taxonomy import FailureType

CLASSES: tuple[str, ...] = ("none",) + tuple(t.value for t in FailureType)


class MultinomialLogistic:
    def __init__(self, n_features: int, classes: Sequence[str], l2: float = 2e-3,
                 lr: float = 0.15, epochs: int = 900, seed: int = 0) -> None:
        self.classes = list(classes)
        self.n_features = n_features
        self.l2, self.lr, self.epochs, self.seed = l2, lr, epochs, seed
        self.W = np.zeros((len(self.classes), n_features))
        self.b = np.zeros(len(self.classes))
        self.mu = np.zeros(n_features)
        self.sd = np.ones(n_features)
        self.class_weight: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: Sequence[str]) -> "MultinomialLogistic":
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        Xs = (X - self.mu) / self.sd
        idx = np.array([self.classes.index(c) for c in y])
        Y = np.zeros((len(idx), len(self.classes)))
        Y[np.arange(len(idx)), idx] = 1.0

        # Inverse-frequency weighting: failure steps are rarer than clean steps,
        # and without this the model collapses to "none".
        counts = Y.sum(axis=0)
        w = np.where(counts > 0, len(idx) / np.maximum(counts, 1), 0.0)
        w = w / w[w > 0].mean()
        self.class_weight = w.astype(np.float64)
        sw = self.class_weight[idx][:, None]

        mW = np.zeros_like(self.W)
        vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b)
        vb = np.zeros_like(self.b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        n = len(idx)
        for step in range(1, self.epochs + 1):
            Z = Xs @ self.W.T + self.b
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            G = (P - Y) * sw / n
            gW = G.T @ Xs + self.l2 * self.W
            gb = G.sum(axis=0)
            for p, g, m, v in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                m *= beta1
                m += (1 - beta1) * g
                v *= beta2
                v += (1 - beta2) * (g * g)
                mhat = m / (1 - beta1 ** step)
                vhat = v / (1 - beta2 ** step)
                p -= self.lr * mhat / (np.sqrt(vhat) + eps)
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xs = (X - self.mu) / self.sd
        Z = Xs @ self.W.T + self.b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    def predict_proba_one(self, x: Sequence[float]) -> np.ndarray:
        return self.predict_proba(np.asarray([x]))[0]

    # ---- artifact ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "arb.multinomial_logistic.v1",
            "classes": self.classes,
            "features": list(FEATURE_NAMES),
            "W": self.W.round(6).tolist(),
            "b": self.b.round(6).tolist(),
            "mu": self.mu.round(6).tolist(),
            "sd": self.sd.round(6).tolist(),
            "hyper": {"l2": self.l2, "lr": self.lr, "epochs": self.epochs},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MultinomialLogistic":
        m = cls(len(d["features"]), d["classes"], **d.get("hyper", {}))
        m.W = np.array(d["W"], dtype=np.float64)
        m.b = np.array(d["b"], dtype=np.float64)
        m.mu = np.array(d["mu"], dtype=np.float64)
        m.sd = np.array(d["sd"], dtype=np.float64)
        return m

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "MultinomialLogistic":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class SupervisedDetector:
    model: MultinomialLogistic
    threshold: float = 0.5
    name: str = "supervised"

    def reset(self) -> None:
        self.n_obs = 0

    def observe(self, f: dict[str, float]) -> Detection:
        self.n_obs = getattr(self, "n_obs", 0) + 1
        p = self.model.predict_proba_one(vector(f))
        p_none = float(p[self.model.classes.index("none")])
        types = [c for c in self.model.classes if c != "none"]
        probs = {t: float(p[self.model.classes.index(t)]) for t in types}
        total = sum(probs.values()) or 1.0
        posterior = {t: v / total for t, v in probs.items()}
        best = max(posterior, key=posterior.get) if posterior else None
        score = 1.0 - p_none
        fired = score >= self.threshold and best is not None
        return Detection(fired=fired, type=best if fired else None, score=score,
                         posterior=posterior, evidence={})

    @classmethod
    def load(cls, path: str, threshold: float = 0.5) -> "SupervisedDetector":
        return cls(model=MultinomialLogistic.load(path), threshold=threshold)
