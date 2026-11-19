"""Recovery policies: the four baselines compared in the paper.

All four share one action space and one operator table (``perturb.PROGRAMS``).
Only the *selection function* differs, which is what makes the comparison a
controlled experiment rather than a comparison of implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from arb.taxonomy import FAILURE_SPECS, FailureType, RecoveryAction
from arb.perturb import CANONICAL_PROGRAM, program_for


@dataclass
class Decision:
    action: RecoveryAction
    program: tuple[RecoveryAction, ...]
    confidence: float = 1.0
    reason: str = ""


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def decide(
        self,
        *,
        detected_type: str | None,
        posterior: dict[str, float],
        step_index: int,
        injections_handled: int,
        retries_used: int,
    ) -> Decision: ...


# --------------------------------------------------------------------------- #
class NoRecovery:
    """Ignore detections; keep executing the plan. The null hypothesis."""

    name = "no_recovery"
    on_error = False

    def reset(self) -> None:
        return None

    def decide(self, **kw: Any) -> Decision:
        return Decision(RecoveryAction.CONTINUE, (RecoveryAction.CONTINUE,), 1.0,
                        "policy ignores detections")


# --------------------------------------------------------------------------- #
class RetryOnly:
    """The most common production reflex: retry the same call a bounded number of times.

    This baseline does **not** use a detector: it retries on any tool error, which
    is what production systems actually do. Giving it a detector would make it a
    straw man rather than the behaviour we want to beat.
    """

    name = "retry_only"
    on_error = True

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    def reset(self) -> None:
        return None

    def decide(self, *, retries_used: int = 0, **kw: Any) -> Decision:
        if retries_used >= self.max_retries:
            return Decision(RecoveryAction.ESCALATE_ABORT, (RecoveryAction.ESCALATE_ABORT,), 0.5,
                            "retry budget exhausted")
        return Decision(RecoveryAction.RETRY, (RecoveryAction.RETRY,), 0.6,
                        "blind retry of the last call")


# --------------------------------------------------------------------------- #
class RuleBasedRecovery:
    """Domain-expert mapping from detected class to canonical remediation.

    This is the strong, interpretable baseline: it is what a careful engineer
    writes after reading the taxonomy, and it uses no learned parameters.
    """

    name = "rule_based"
    on_error = False

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    def reset(self) -> None:
        return None

    def decide(self, *, detected_type: str | None = None, retries_used: int = 0, **kw: Any) -> Decision:
        if detected_type is None:
            return Decision(RecoveryAction.CONTINUE, (RecoveryAction.CONTINUE,), 0.3,
                            "no detection")
        primary, secondary = CANONICAL_PROGRAM[detected_type]
        if primary is RecoveryAction.RETRY and retries_used >= self.max_retries:
            primary = secondary
        return Decision(primary, program_for(primary), 0.9,
                        f"canonical remediation for {detected_type}")


# --------------------------------------------------------------------------- #
class ModelBasedRecovery:
    """Rank the action space with a supervised ranker over monitor features.

    Trained offline on the training split (see ``arb/models/train.py``). At
    inference it consumes only the detector's posterior and the same
    monitor-visible quantities the rules see -- never the ground-truth label.
    """

    name = "model_based"
    on_error = False

    def __init__(self, ranker: Any | None = None, max_retries: int = 2) -> None:
        self.ranker = ranker
        self.max_retries = max_retries

    def reset(self) -> None:
        return None

    def decide(
        self,
        *,
        detected_type: str | None = None,
        posterior: dict[str, float] | None = None,
        step_index: int = 0,
        injections_handled: int = 0,
        retries_used: int = 0,
        features: dict[str, float] | None = None,
        **kw: Any,
    ) -> Decision:
        if detected_type is None:
            return Decision(RecoveryAction.CONTINUE, (RecoveryAction.CONTINUE,), 0.3,
                            "no detection")
        if self.ranker is None:
            # Degenerate fallback: behave like the rules, but flagged as such.
            primary, _ = CANONICAL_PROGRAM[detected_type]
            return Decision(primary, program_for(primary), 0.5, "ranker unavailable")
        post = posterior or {detected_type: 1.0}
        # Candidate set: the canonical repair for the detected class, or a clean
        # containment. The ranker decides whether repair is worth attempting.
        canon = CANONICAL_PROGRAM[detected_type][0]
        candidates = [canon.value]
        if self.ranker.has_evidence(RecoveryAction.ESCALATE_ABORT.value):
            candidates.append(RecoveryAction.ESCALATE_ABORT.value)
        action, score, probs = self.ranker.rank(
            posterior=post, features=features or {}, retries_used=retries_used,
            max_retries=self.max_retries, candidates=candidates,
        )
        return Decision(action, program_for(action), float(score),
                        f"ranker choice={action.value} (p={score:.2f})")


POLICIES = {
    "no_recovery": NoRecovery,
    "retry_only": RetryOnly,
    "rule_based": RuleBasedRecovery,
    "model_based": ModelBasedRecovery,
}


def make_policy(name: str, **kw: Any) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}")
    return POLICIES[name](**kw)  # type: ignore[call-arg]
