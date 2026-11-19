"""Controlled perturbation scheduling.

A perturbation is *not* a random glitch: it is a scheduled, reproducible event
with a locus (environment vs agent), a target, and -- for the recoverable
classes -- a well-defined observation that a correct monitor can register.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from arb.envs.base import Environment, World, stable_seed
from arb.schema import FaultPlan, TaskSpec
from arb.taxonomy import FAILURE_SPECS, FailureType, RecoveryAction


@dataclass(frozen=True)
class Program:
    """Expansion of a single recovery action into a concrete operator sequence."""

    action: RecoveryAction
    steps: tuple[RecoveryAction, ...]
    rationale: str


#: Every action expands to an operator program. All policies share this table,
#: so the comparison isolates *selection*, not capability.
PROGRAMS: dict[RecoveryAction, Program] = {
    RecoveryAction.CONTINUE: Program(RecoveryAction.CONTINUE, (RecoveryAction.CONTINUE,),
                                     "do nothing; keep executing the plan"),
    RecoveryAction.RETRY: Program(RecoveryAction.RETRY, (RecoveryAction.RETRY,),
                                  "re-execute the same call unchanged"),
    RecoveryAction.REVALIDATE_ARGS: Program(
        RecoveryAction.REVALIDATE_ARGS, (RecoveryAction.REVALIDATE_ARGS,),
        "re-derive the arguments from current state, then re-issue"),
    RecoveryAction.REFRESH_STATE: Program(
        RecoveryAction.REFRESH_STATE, (RecoveryAction.REFRESH_STATE, RecoveryAction.REPLAN),
        "re-read the touched resource, then replan from fresh state"),
    RecoveryAction.RE_READ_AUTHORITATIVE: Program(
        RecoveryAction.RE_READ_AUTHORITATIVE,
        (RecoveryAction.RE_READ_AUTHORITATIVE, RecoveryAction.REPLAN),
        "consult the authoritative view, then replan"),
    RecoveryAction.REPLAN: Program(RecoveryAction.REPLAN, (RecoveryAction.REPLAN,),
                                   "recompute the remaining plan from current state"),
    RecoveryAction.RESYNC_MESSAGE: Program(
        RecoveryAction.RESYNC_MESSAGE, (RecoveryAction.RESYNC_MESSAGE, RecoveryAction.REPLAN),
        "re-send / re-request the message, then replan"),
    RecoveryAction.REANCHOR_OBJECTIVE: Program(
        RecoveryAction.REANCHOR_OBJECTIVE,
        (RecoveryAction.REANCHOR_OBJECTIVE, RecoveryAction.REPLAN),
        "restore the stated objective, then replan"),
    RecoveryAction.ESCALATE_ABORT: Program(RecoveryAction.ESCALATE_ABORT,
                                           (RecoveryAction.ESCALATE_ABORT,),
                                           "halt without side effects and hand off"),
}

#: Canonical programs per failure class (used by the rule-based policy).
CANONICAL_PROGRAM: dict[str, tuple[RecoveryAction, RecoveryAction]] = {
    FailureType.INVALID_ARGUMENTS.value: (RecoveryAction.REVALIDATE_ARGS, RecoveryAction.RETRY),
    FailureType.STALE_STATE.value: (RecoveryAction.REFRESH_STATE, RecoveryAction.REPLAN),
    FailureType.CONTRADICTORY_EVIDENCE.value: (RecoveryAction.RE_READ_AUTHORITATIVE, RecoveryAction.REPLAN),
    FailureType.PLANNING_ERROR.value: (RecoveryAction.REPLAN, RecoveryAction.RETRY),
    FailureType.COMMUNICATION_FAILURE.value: (RecoveryAction.RESYNC_MESSAGE, RecoveryAction.REPLAN),
    FailureType.OBJECTIVE_DRIFT.value: (RecoveryAction.REANCHOR_OBJECTIVE, RecoveryAction.REPLAN),
    FailureType.UNRECOVERABLE_EXTERNAL_ERROR.value: (RecoveryAction.ESCALATE_ABORT, RecoveryAction.ESCALATE_ABORT),
}


def program_for(action: RecoveryAction) -> tuple[RecoveryAction, ...]:
    return PROGRAMS[RecoveryAction(action)].steps


def locus(env: Environment, failure_type: str) -> str:
    return "agent" if failure_type in env.AGENT_LATERAL else "env"


def schedule_fault(
    env: Environment,
    task: TaskSpec,
    world: World,
    failure_type: str,
    *,
    seed: int,
    horizon: int | None = None,
) -> FaultPlan | None:
    """Instantiate a fault and align its injection step with the SOP.

    Injection is *not* uniform over steps: it is placed at the step that calls
    the targeted tool, so the perturbation is guaranteed to be exercised. A
    fault scheduled uniformly at random would silently never fire in short
    episodes, inflating apparent detection rates by construction.
    """
    rng = random.Random(stable_seed("fault", task.task_id, failure_type, seed))
    probe = env.make_fault(task, world, rng, failure_type, 0)
    if probe is None:
        return None
    plan = env.solve(world, task, "true")
    idx = [i for i, s in enumerate(plan) if s.tool == probe.target_tool] if probe.target_tool else []
    if idx:
        # The fault becomes active exactly at the step that calls its target, so
        # ``latency_steps == 0`` means "detected before the damage propagated to
        # a write" -- which is the quantity the paper reports, not a scheduling
        # artefact. Read faults attach to the *last* such read, i.e. the one the
        # agent relies on immediately before writing.
        inject_at = idx[-1] if probe.target_tool in env.READ_TOOLS else idx[0]
    else:
        hi = max(2, (horizon or int(task.horizon)) - 2)
        inject_at = rng.randint(1, hi)
    probe.inject_at = int(inject_at)
    probe.payload = dict(probe.payload)
    probe.payload.setdefault("locus", locus(env, failure_type))
    return probe


def applicable_faults(env: Environment, task: TaskSpec, world: World) -> list[str]:
    """Which classes are physically injectable in this family (for reporting)."""
    out: list[str] = []
    for ft in FailureType.all():
        spec = FAILURE_SPECS[ft]
        if env.track not in spec.tracks:
            continue
        probe = env.make_fault(task, world, random.Random(0), ft.value, 1)
        if probe is not None:
            out.append(ft.value)
    return out
