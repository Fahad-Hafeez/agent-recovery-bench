"""Rule-based detector: the interpretable reference monitor.

Every rule is a named, auditable predicate over monitor-visible features. The
rule set is written *before* seeing results (it is the domain-expert encoding of
the taxonomy), which is what makes it a fair baseline for the learned detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arb.taxonomy import FailureType


@dataclass
class Detection:
    fired: bool
    type: str | None
    score: float
    posterior: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"fired": self.fired, "type": self.type, "score": round(self.score, 4)}


#: (failure type, base strength, activation predicate name)
RULES: tuple[tuple[str, float, str], ...] = (
    (FailureType.INVALID_ARGUMENTS.value, 0.90, "err_schema"),
    (FailureType.INVALID_ARGUMENTS.value, 0.97, "err_schema_run"),
    (FailureType.UNRECOVERABLE_EXTERNAL_ERROR.value, 0.88, "err_unavailable"),
    (FailureType.UNRECOVERABLE_EXTERNAL_ERROR.value, 0.96, "err_unavailable_run"),
    (FailureType.STALE_STATE.value, 0.80, "version_lag"),
    (FailureType.STALE_STATE.value, 0.96, "stale_flag"),
    (FailureType.CONTRADICTORY_EVIDENCE.value, 0.93, "conflict_flag"),
    (FailureType.CONTRADICTORY_EVIDENCE.value, 0.78, "cross_source_disagree"),
    (FailureType.PLANNING_ERROR.value, 0.86, "err_precondition"),
    (FailureType.PLANNING_ERROR.value, 0.56, "plan_silent"),
    (FailureType.OBJECTIVE_DRIFT.value, 0.82, "policy_violation"),
    (FailureType.PLANNING_ERROR.value, 0.68, "audit_mismatch"),
    (FailureType.COMMUNICATION_FAILURE.value, 0.95, "message_lost"),
)


class RuleDetector:
    name = "rules"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.n_obs = 0

    @staticmethod
    def _activation(name: str, f: dict[str, float]) -> float:
        if name == "err_schema":
            return 1.0 if f.get("err_schema", 0) > 0 else 0.0
        if name == "err_schema_run":
            return 1.0 if f.get("err_schema_run3", 0) >= 1.0 else 0.0
        if name == "err_unavailable":
            return 1.0 if f.get("err_unavailable", 0) > 0 else 0.0
        if name == "err_unavailable_run":
            return 1.0 if f.get("err_unavailable_run3", 0) >= 1.0 else 0.0
        if name == "version_lag":
            return 1.0 if f.get("version_lag", 0) > 0 else 0.0
        if name == "stale_flag":
            return 1.0 if f.get("stale_flag", 0) > 0 else 0.0
        if name == "conflict_flag":
            return 1.0 if f.get("conflict_flag", 0) > 0 else 0.0
        if name == "cross_source_disagree":
            return 1.0 if f.get("cross_source_disagree", 0) > 0 else 0.0
        if name == "err_precondition":
            return 1.0 if f.get("err_precondition", 0) > 0 else 0.0
        if name == "plan_silent":
            return 1.0 if (f.get("no_progress_run3", 0) > 0 and f.get("mutate_without_read", 0) > 0) else 0.0
        if name == "policy_violation":
            return 1.0 if f.get("policy_violation", 0) >= 0.5 else 0.0
        if name == "audit_mismatch":
            return 1.0 if f.get("audit_mismatch", 0) > 0 else 0.0
        if name == "message_lost":
            return 1.0 if f.get("message_lost", 0) > 0 else 0.0
        return 0.0

    def score(self, f: dict[str, float]) -> tuple[float, str | None, dict[str, float]]:
        best, best_type, evidence = 0.0, None, {}
        for ftype, strength, name in RULES:
            a = self._activation(name, f)
            if a <= 0:
                continue
            evidence[name] = strength
            if strength > best:
                best, best_type = strength, ftype
        # Graded background suspicion, so the score is not degenerate on clean steps.
        background = 0.10 * f.get("repeat_call", 0.0) + 0.05 * f.get("step_pressure", 0.0) \
            + 0.05 * f.get("err_any", 0.0) + 0.08 * f.get("no_progress_run3", 0.0)
        return max(best, min(background, 0.3)), best_type, evidence

    def observe(self, f: dict[str, float]) -> Detection:
        self.n_obs += 1
        s, t, ev = self.score(f)
        fired = s >= self.threshold and t is not None
        posterior = {t: 1.0} if (fired and t) else {}
        return Detection(fired=fired, type=t if fired else None, score=float(s),
                         posterior=posterior, evidence=ev)


def make_detector(name: str, **kw: Any) -> Any:
    if name in ("none", None):
        return None
    if name == "rules":
        return RuleDetector(**kw)
    if name == "supervised":
        from arb.detectors.supervised import SupervisedDetector

        return SupervisedDetector.load("models/detector.json", **kw)
    raise KeyError(f"unknown detector {name!r}")
