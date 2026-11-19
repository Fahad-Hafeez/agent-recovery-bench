"""Compact, operational failure taxonomy for ARB.

Design constraints that shaped this taxonomy:

1. **Operational, not descriptive.** Every class must be *injectable* by the
   harness and *observable* by a monitor that only sees the trajectory. That
   rules out classes such as "the model reasoned poorly", which cannot be
   injected deterministically or verified mechanically.
2. **Disjoint at the injection site.** One perturbation event instantiates
   exactly one class, so ground-truth labels are unambiguous. Multi-label
   co-occurrence is possible only across *separate* injection events.
3. **Recovery-addressable.** Each class has a canonical remediation, which is
   what makes the rule-based recovery baseline a well-defined reference point
   rather than a hand-tuned heuristic.
4. **Severity-separated.** ``UNRECOVERABLE_EXTERNAL_ERROR`` is deliberately kept
   as its own class so that policies cannot be rewarded for "recovering" from
   faults that admit no in-episode repair. For that class the reported quantity
   is *containment*, not recovery.

The taxonomy is intentionally small (7 classes). A larger ontology (MAST has 14
modes across 3 categories) buys descriptive resolution at the cost of
inter-annotator and label-injection reliability; ARB trades resolution for
deterministic labelling, and documents the mapping in ``docs/taxonomy.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureType(str, Enum):
    """The seven ARB failure classes."""

    INVALID_ARGUMENTS = "invalid_arguments"
    STALE_STATE = "stale_state"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    PLANNING_ERROR = "planning_error"
    COMMUNICATION_FAILURE = "communication_failure"
    OBJECTIVE_DRIFT = "objective_drift"
    UNRECOVERABLE_EXTERNAL_ERROR = "unrecoverable_external_error"

    def __str__(self) -> str:  # stable, JSON-friendly
        return self.value

    @classmethod
    def all(cls) -> tuple["FailureType", ...]:
        return tuple(cls)


class RecoveryAction(str, Enum):
    """Recovery operators available to every policy (a shared action space).

    Keeping the action space shared across policies is what makes the
    no-recovery / retry-only / rule-based / model-based comparison a controlled
    experiment: only the *selection function* varies, never the capabilities.
    """

    CONTINUE = "continue"
    RETRY = "retry"
    REVALIDATE_ARGS = "revalidate_args"
    REFRESH_STATE = "refresh_state"
    RE_READ_AUTHORITATIVE = "re_read_authoritative"
    REPLAN = "replan"
    RESYNC_MESSAGE = "resync_message"
    REANCHOR_OBJECTIVE = "reanchor_objective"
    ESCALATE_ABORT = "escalate_abort"

    def __str__(self) -> str:
        return self.value


RECOVERY_ACTIONS: tuple[RecoveryAction, ...] = tuple(RecoveryAction)

# Actions that repair by re-observing the world before acting again.
REOBserve_ACTIONS = frozenset(
    {
        RecoveryAction.REVALIDATE_ARGS,
        RecoveryAction.REFRESH_STATE,
        RecoveryAction.RE_READ_AUTHORITATIVE,
        RecoveryAction.RESYNC_MESSAGE,
    }
)

# Actions that change what the agent intends to do.
REPLAN_ACTIONS = frozenset({RecoveryAction.REPLAN, RecoveryAction.REANCHOR_OBJECTIVE})


@dataclass(frozen=True)
class FailureSpec:
    """Machine-readable definition of one failure class."""

    type: FailureType
    short: str
    definition: str
    observable_signature: str
    # "surface": detectable from error codes / version counters alone.
    # "latent": requires cross-referencing evidence or the objective spec.
    detectability: str
    canonical_remediation: RecoveryAction
    recoverable_by_design: bool
    tracks: tuple[str, ...]
    injection: str
    example: str

    @property
    def recoverable(self) -> bool:
        return self.recoverable_by_design


_T = FailureType

FAILURE_SPECS: dict[FailureType, FailureSpec] = {
    _T.INVALID_ARGUMENTS: FailureSpec(
        type=_T.INVALID_ARGUMENTS,
        short="misuse",
        definition=(
            "A tool is invoked with arguments that violate its declared schema or "
            "its documented preconditions (missing required field, wrong type, "
            "out-of-domain enumeration, violated guard such as a negative quantity)."
        ),
        observable_signature="Tool returns a schema/precondition error code; state unchanged.",
        detectability="surface",
        canonical_remediation=RecoveryAction.REVALIDATE_ARGS,
        recoverable_by_design=True,
        tracks=("single", "multi"),
        injection="env_side_schema_reject",
        example="issue_refund(amount_cents='50.00') where the schema requires integer cents.",
    ),
    _T.STALE_STATE: FailureSpec(
        type=_T.STALE_STATE,
        short="staleness",
        definition=(
            "The agent acts on an observation that no longer reflects the world "
            "version it is about to mutate, producing a lost update or a "
            "write based on a superseded value."
        ),
        observable_signature=(
            "Returned state_version lags the maximum version the trajectory has "
            "already observed; the subsequent write silently overwrites a newer value."
        ),
        detectability="surface",
        canonical_remediation=RecoveryAction.REFRESH_STATE,
        recoverable_by_design=True,
        tracks=("single", "multi"),
        injection="read_lag",
        example="read_record returns v3 after the agent already saw v4; the write reverts a concurrent fix.",
    ),
    _T.CONTRADICTORY_EVIDENCE: FailureSpec(
        type=_T.CONTRADICTORY_EVIDENCE,
        short="contradiction",
        definition=(
            "Two observations of the same quantity disagree and the agent proceeds "
            "on one of them without resolving the conflict against the "
            "authoritative source."
        ),
        observable_signature=(
            "The same field is observed with two different values; the "
            "authoritative audit view agrees with neither the cached nor the "
            "most recent read."
        ),
        detectability="latent",
        canonical_remediation=RecoveryAction.RE_READ_AUTHORITATIVE,
        recoverable_by_design=True,
        tracks=("single", "multi"),
        injection="planted_disagreement",
        example="get_stock reports 41 units while the ledger implies 38; the agent posts an adjustment for 41.",
    ),
    _T.PLANNING_ERROR: FailureSpec(
        type=_T.PLANNING_ERROR,
        short="planning",
        definition=(
            "The step sequence violates a precondition or dependency: a mutation "
            "is attempted before its prerequisite read/validation, a required "
            "step is omitted, or steps are ordered so a guard cannot be satisfied."
        ),
        observable_signature=(
            "A precondition error, or a mutation whose intent-relative expected "
            "state differs from the observed state with no prior fault active."
        ),
        detectability="latent",
        canonical_remediation=RecoveryAction.REPLAN,
        recoverable_by_design=True,
        tracks=("single", "multi"),
        injection="step_skip_or_reorder",
        example="close_period runs before the last post_adjustment, so the audit totals never reconcile.",
    ),
    _T.COMMUNICATION_FAILURE: FailureSpec(
        type=_T.COMMUNICATION_FAILURE,
        short="communication",
        definition=(
            "In a multi-agent system, a message that a downstream agent depends on "
            "is dropped, truncated, or acknowledged without delivery, so the "
            "recipient acts on the wrong or an empty plan."
        ),
        observable_signature=(
            "Send returns without delivery receipt; the recipient's next action is "
            "inconsistent with the sender's plan."
        ),
        detectability="surface",
        canonical_remediation=RecoveryAction.RESYNC_MESSAGE,
        recoverable_by_design=True,
        tracks=("multi",),
        injection="message_drop",
        example="planner's supplier selection is dropped; buyer defaults to the most expensive quote.",
    ),
    _T.OBJECTIVE_DRIFT: FailureSpec(
        type=_T.OBJECTIVE_DRIFT,
        short="drift",
        definition=(
            "The agent optimises a proxy KPI that is locally correlated with, but "
            "not equal to, the stated objective, so its actions are individually "
            "well-formed yet collectively wrong."
        ),
        observable_signature=(
            "Declared objective unchanged while the optimised KPI diverges from the "
            "objective-consistent KPI; no error is ever raised."
        ),
        detectability="latent",
        canonical_remediation=RecoveryAction.REANCHOR_OBJECTIVE,
        recoverable_by_design=True,
        tracks=("single", "multi"),
        injection="proxy_objective_swap",
        example="maximising tickets_closed rather than policy_correct_resolutions, so refunds are over-issued.",
    ),
    _T.UNRECOVERABLE_EXTERNAL_ERROR: FailureSpec(
        type=_T.UNRECOVERABLE_EXTERNAL_ERROR,
        short="external",
        definition=(
            "A dependency required for task completion is persistently unavailable. "
            "No in-episode policy can complete the task; the only correct behaviour "
            "is a clean, side-effect-free halt that escalates."
        ),
        observable_signature="Repeated hard failure code on the same dependency with no state mutation.",
        detectability="surface",
        canonical_remediation=RecoveryAction.ESCALATE_ABORT,
        recoverable_by_design=False,
        tracks=("single", "multi"),
        injection="persistent_dependency_outage",
        example="The payments endpoint returns 503 for the whole episode; retrying burns budget without progress.",
    ),
}


def canonical_action(failure_type: FailureType | str) -> RecoveryAction:
    """The remediation a domain expert would prescribe for this class."""
    ft = FailureType(str(failure_type))
    return FAILURE_SPECS[ft].canonical_remediation


def is_recoverable(failure_type: FailureType | str) -> bool:
    ft = FailureType(str(failure_type))
    return FAILURE_SPECS[ft].recoverable


def taxonomy_table() -> list[dict[str, str]]:
    """Flat table used by docs generation and the paper."""
    rows = []
    for ft in FailureType.all():
        spec = FAILURE_SPECS[ft]
        rows.append(
            {
                "type": ft.value,
                "detectability": spec.detectability,
                "tracks": "/".join(spec.tracks),
                "canonical_remediation": spec.canonical_remediation.value,
                "recoverable": "yes" if spec.recoverable else "no (containment)",
                "injection": spec.injection,
                "observable_signature": spec.observable_signature,
            }
        )
    return rows


DETECTABILITY_CLASSES = ("surface", "latent")
