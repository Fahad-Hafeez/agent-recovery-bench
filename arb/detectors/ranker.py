"""Recovery action ranker.

Formulation: one **conditional value model per operator**,
``P(success | a, x) = sigmoid(w_a · x + b_a)``. This is deliberately *not* a
single shared scoring function over action one-hots: a shared linear model with
one-hot inputs has no action-feature interaction, so its argmax over actions is
constant across the whole input space (it would answer "which action is best on
average", not "which action is best here"). Per-operator heads are the smallest
model that can express the thing we actually care about.

Operators with too few logged executions fall back to their empirical success
rate (shrinkage toward the prior), which is the honest thing to do with
off-policy data.

Known limitation, stated in the paper: training data are logged actions from the
three baseline policies, so the ranker never observes the counterfactual outcome
of an action the logging policy did not take. We report it as a limited baseline
and keep the rule-based policy as the unbiased reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from arb.taxonomy import RECOVERY_ACTIONS, FailureType, RecoveryAction

#: Operator cost in agent steps (used as a tie-breaker, not a hard constraint).
ACTION_STEP_COST: dict[str, float] = {
    RecoveryAction.CONTINUE.value: 0.0,
    RecoveryAction.RETRY.value: 1.0,
    RecoveryAction.REVALIDATE_ARGS.value: 1.0,
    RecoveryAction.REPLAN.value: 1.0,
    RecoveryAction.REFRESH_STATE.value: 2.0,
    RecoveryAction.RE_READ_AUTHORITATIVE.value: 2.0,
    RecoveryAction.RESYNC_MESSAGE.value: 2.0,
    RecoveryAction.REANCHOR_OBJECTIVE.value: 2.0,
    RecoveryAction.ESCALATE_ABORT.value: 0.0,
}

RANK_FEATURES: tuple[str, ...] = (
    *[f"p_{t.value}" for t in FailureType],
    "score",
    "step_pressure",
    "err_any",
    "retry_streak",
    "version_lag",
    "policy_violation",
    "no_progress_run3",
    "retries_used",
)

MIN_PER_ACTION = 25


def rank_vector(posterior: dict[str, float], features: dict[str, float],
                retries_used: int) -> list[float]:
    return [
        *[float(posterior.get(t.value, 0.0)) for t in FailureType],
        float(features.get("_score", 0.0)),
        float(features.get("step_pressure", 0.0)),
        float(features.get("err_any", 0.0)),
        float(features.get("retry_streak", 0.0)),
        float(features.get("version_lag", 0.0)),
        float(features.get("policy_violation", 0.0)),
        float(features.get("no_progress_run3", 0.0)),
        float(retries_used),
    ]


@dataclass
class RecoveryRanker:
    n_features: int
    actions: tuple[str, ...] = tuple(a.value for a in RECOVERY_ACTIONS)
    W: np.ndarray | None = None      # (n_actions, n_features)
    b: np.ndarray | None = None      # (n_actions,)
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    counts: dict[str, int] | None = None
    cost_penalty: float = 0.03

    # ---- training ------------------------------------------------------ #
    def fit(self, X: np.ndarray, actions: Sequence[str], y: Sequence[int],
            l2: float = 2e-2, lr: float = 0.2, epochs: int = 500) -> "RecoveryRanker":
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        Xs = (X - self.mu) / self.sd
        A = len(self.actions)
        self.W = np.zeros((A, self.n_features))
        self.b = np.zeros(A)
        yy = np.asarray(y, dtype=np.float64)
        self.counts = {}
        for ai, a in enumerate(self.actions):
            idx = np.array([i for i, act in enumerate(actions) if act == a])
            self.counts[a] = int(len(idx))
            if len(idx) < MIN_PER_ACTION:
                # Shrinkage: log-odds of the empirical success rate, flat weights.
                p = float(yy[idx].mean()) if len(idx) else float(yy.mean())
                p = min(max(p, 1e-3), 1 - 1e-3)
                self.b[ai] = float(np.log(p / (1 - p)))
                continue
            Xi, yi = Xs[idx], yy[idx]
            w = np.zeros(self.n_features)
            bias = 0.0
            m = np.zeros_like(w); v = np.zeros_like(w)
            mb = 0.0; vb = 0.0
            balance = len(yi) / max(1.0, 2.0 * yi.sum()), len(yi) / max(1.0, 2.0 * (len(yi) - yi.sum()))
            sw = np.where(yi > 0, balance[0], balance[1])
            for step in range(1, epochs + 1):
                z = Xi @ w + bias
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                g = (p - yi) * sw / len(yi)
                gw = Xi.T @ g + l2 * w
                gb = float(g.sum())
                for par, gr, mm, vv in ((w, gw, m, v),):
                    mm *= 0.9; mm += 0.1 * gr
                    vv *= 0.999; vv += 0.001 * gr * gr
                    par -= lr * (mm / (1 - 0.9 ** step)) / (np.sqrt(vv / (1 - 0.999 ** step)) + 1e-8)
                m, v = mm, vv
                mb = 0.9 * mb + 0.1 * gb
                vb = 0.999 * vb + 0.001 * gb * gb
                bias -= lr * (mb / (1 - 0.9 ** step)) / (np.sqrt(vb / (1 - 0.999 ** step)) + 1e-8)
            self.W[ai] = w
            self.b[ai] = bias
        return self

    # ---- inference ----------------------------------------------------- #
    def probabilities(self, x: Sequence[float], retries_used: int = 0) -> dict[str, float]:
        if self.W is None:
            return {a: 0.5 for a in self.actions}
        xs = (np.asarray(x, dtype=np.float64) - self.mu) / self.sd
        z = self.W @ xs + self.b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        out = {a: float(p[i]) for i, a in enumerate(self.actions)}
        if retries_used >= 2:
            out[RecoveryAction.RETRY.value] = -1.0
        return out

    def has_evidence(self, action: str) -> bool:
        """An operator with no logged executions cannot be ranked, only ignored."""
        return (self.counts or {}).get(action, 0) >= MIN_PER_ACTION

    def rank(self, *, posterior: dict[str, float], features: dict[str, float],
             retries_used: int, max_retries: int = 2,
             candidates: Sequence[str] | None = None
             ) -> tuple[RecoveryAction, float, dict[str, float]]:
        x = rank_vector(posterior, features, retries_used)
        probs = self.probabilities(x, retries_used)
        pool = list(candidates) if candidates else [
            a for a, n in (self.counts or {}).items() if n >= MIN_PER_ACTION]
        if not pool:
            pool = [RecoveryAction.ESCALATE_ABORT.value]
        best_action, best_score = RecoveryAction(pool[0]), -1e9
        for a in pool:
            score = probs.get(a, 0.0) - self.cost_penalty * ACTION_STEP_COST.get(a, 1.0)
            if score > best_score:
                best_action, best_score = RecoveryAction(a), score
        return best_action, max(0.0, min(1.0, float(best_score))), probs

    # ---- artifact ------------------------------------------------------ #
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "type": "arb.recovery_ranker.v2",
                "features": list(RANK_FEATURES),
                "actions": list(self.actions),
                "W": np.asarray(self.W).round(6).tolist(),
                "b": np.asarray(self.b).round(6).tolist(),
                "mu": np.asarray(self.mu).round(6).tolist(),
                "sd": np.asarray(self.sd).round(6).tolist(),
                "counts": self.counts,
                "cost_penalty": self.cost_penalty,
            }, fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "RecoveryRanker":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        r = cls(n_features=len(d["features"]), actions=tuple(d["actions"]))
        r.W = np.array(d["W"], dtype=np.float64)
        r.b = np.array(d["b"], dtype=np.float64)
        r.mu = np.array(d["mu"], dtype=np.float64)
        r.sd = np.array(d["sd"], dtype=np.float64)
        r.counts = d.get("counts")
        r.cost_penalty = float(d["cost_penalty"])
        return r
