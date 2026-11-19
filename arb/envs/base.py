"""Deterministic environment contract shared by the single- and multi-agent tracks.

Design invariants (these are what make ARB a *benchmark* rather than a demo):

I1  **Inject, don't observe-and-hope.** Failures are created by explicit
    perturbation schedules, so ground truth is exact.
I2  **Verify, don't ask.** Success is a property of final environment state
    compared against a goal computed from the task, never of agent text.
I3  **One monitor view.** Detectors and policies receive ``Step`` records whose
    signals are computed by :class:`MonitorView`, which never exposes
    ``fault``/``inject_at``. Detection numbers are therefore label-clean.
I4  **Recovery changes observations, not the world.** A recovery action only
    alters what the agent subsequently sees or intends. A write that already
    corrupted state stays corrupted until a *later* correct write repairs it.
    This is why detecting a stale read does not imply recovering from it.
I5  **Paired counterfactuals.** Every faulty episode has a fault-free twin with
    the same task, seed and background-update schedule, so downstream impact is
    a difference of two runs rather than a proxy score.
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from arb.schema import FaultPlan, GradeResult, PlanStep, TaskSpec
from arb.taxonomy import FailureType


# --------------------------------------------------------------------------- #
# Errors and tool contracts
# --------------------------------------------------------------------------- #
class ToolError(Exception):
    """Semantic (not schema) tool failure. Always leaves state unchanged."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ToolSpec:
    name: str
    mutates: bool
    required: tuple[str, ...] = ()
    schema: dict[str, str] = field(default_factory=dict)
    enum: dict[str, tuple[str, ...]] = field(default_factory=dict)
    base_latency_ms: float = 80.0
    description: str = ""


@dataclass
class ToolResult:
    ok: bool
    data: Any
    error_code: str | None = None
    detail: str = ""
    state_version: int = 0
    stale: bool = False
    conflict: bool = False
    message_lost: bool = False
    authoritative: bool = False
    delivered: bool | None = None


@dataclass
class ActiveFault:
    """A fault plan plus the current step, so `active` is a pure function."""

    plan: FaultPlan
    step: int

    @property
    def type(self) -> str:
        return self.plan.type

    @property
    def active(self) -> bool:
        return self.step >= self.plan.inject_at


@dataclass
class CallContext:
    fault: ActiveFault | None = None
    step_index: int = 0
    agent: str = "solo"
    attempt: int = 0
    suppressed: frozenset[str] = frozenset()
    monitor_mode: str = "instrumented"
    episode_id: str = ""


# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #
class World:
    """Mutable environment state with a version counter and a short history.

    ``version`` is a logical clock, not a timestamp. Staleness is defined
    against it: an observation is stale iff its ``state_version`` is behind the
    highest version the trajectory has already seen.
    """

    __slots__ = ("data", "version", "_hist", "max_version_seen")

    HISTORY = 6

    def __init__(self, data: dict[str, Any], version: int = 0) -> None:
        self.data = data
        self.version = version
        self._hist: list[tuple[int, dict[str, Any]]] = []
        self.max_version_seen = version

    def clone(self) -> "World":
        w = World(copy.deepcopy(self.data), self.version)
        w._hist = list(self._hist)
        w.max_version_seen = self.max_version_seen
        return w

    def record_and_bump(self, pre_version: int, pre_data: dict[str, Any]) -> None:
        self._hist.append((pre_version, copy.deepcopy(pre_data)))
        if len(self._hist) > self.HISTORY:
            self._hist.pop(0)
        self.version = pre_version + 1

    def view_lagged(self, lag: int) -> tuple[dict[str, Any], int]:
        """Return the state as of ``version - lag`` (best effort)."""
        if lag <= 0 or not self._hist:
            return copy.deepcopy(self.data), self.version
        idx = max(0, len(self._hist) - lag)
        ver, data = self._hist[idx]
        return copy.deepcopy(data), ver

    # ---- projections ------------------------------------------------------ #
    def project(self, keys: tuple[str, ...] | list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in keys:
            node: Any = self.data
            ok = True
            for part in k.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                    node = node[int(part)]
                else:
                    ok = False
                    break
            out[k] = copy.deepcopy(node) if ok else None
        return out


def diff_fields(a: dict[str, Any], b: dict[str, Any], prefix: str = "") -> list[str]:
    """Recursively list dotted paths where two state projections differ."""
    out: list[str] = []
    for key in sorted(set(a) | set(b)):
        pa, pb = a.get(key, "__missing__"), b.get(key, "__missing__")
        path = f"{prefix}{key}"
        if isinstance(pa, dict) and isinstance(pb, dict):
            out.extend(diff_fields(pa, pb, prefix=path + "."))
        elif pa != pb:
            out.append(path)
    return out


def stable_seed(*parts: Any) -> int:
    """Deterministic cross-run seed (independent of PYTHONHASHSEED)."""
    h = hashlib.blake2b(digest_size=8)
    h.update("|".join(str(p) for p in parts).encode("utf-8"))
    return int.from_bytes(h.digest(), "big") % (2**63 - 1)


# --------------------------------------------------------------------------- #
# Environment base
# --------------------------------------------------------------------------- #
class Environment:
    """Base class. Subclasses define tools, state transitions and grading."""

    family: str = "base"
    track: str = "single"
    TEMPLATES: tuple[str, ...] = ("A",)
    CONFIGS: tuple[str, ...] = ("cfg1",)
    TOOLS: dict[str, ToolSpec] = {}
    READ_TOOLS: frozenset[str] = frozenset()
    AUTHORITATIVE_TOOLS: frozenset[str] = frozenset()
    MESSAGE_TOOL: str = "send_message"
    #: Sentinel returned by ``derive_args`` meaning "this step is unnecessary".
    #: It must be distinct from ``{}``, which legitimately means "no arguments".
    SKIP: str = "__arb_skip__"

    # ---- task construction ------------------------------------------------ #
    def make_task(
        self, template_id: str, config_id: str, seed: int, split: str, horizon_class: str
    ) -> TaskSpec:
        raise NotImplementedError

    def initial_world(self, task: TaskSpec) -> World:
        raise NotImplementedError

    # ---- planning --------------------------------------------------------- #
    def solve(self, world: World, task: TaskSpec, objective: str = "true") -> list[PlanStep]:
        """Minimal correct step list from the *current* world state.

        Being state-dependent is what makes ``REPLAN`` a real repair operator
        rather than a relabelling: it re-derives the plan from what is true now.
        """
        raise NotImplementedError

    def drifted_plan(self, world: World, task: TaskSpec) -> list[PlanStep]:
        """Plan produced by an agent optimising the proxy KPI instead."""
        return self.solve(world, task, objective="proxy")

    def objective_text(self, task: TaskSpec) -> str:
        raise NotImplementedError

    def kpis(self, world: World, task: TaskSpec) -> dict[str, float]:
        return {}

    # ---- execution -------------------------------------------------------- #
    def _dispatch(
        self, world: World, tool: str, args: dict[str, Any], task: TaskSpec
    ) -> tuple[Any, bool]:
        """Run a handler. Returns (data, mutated). Raises ToolError."""
        raise NotImplementedError

    def _validate(self, spec: ToolSpec, args: dict[str, Any]) -> str | None:
        for req in spec.required:
            if req not in args:
                return f"missing required field '{req}'"
        for key, val in args.items():
            want = spec.schema.get(key)
            if want is None:
                continue  # extra fields tolerated; documented in the protocol
            if want == "int":
                if isinstance(val, bool) or not isinstance(val, int):
                    return f"field '{key}' must be int, got {type(val).__name__}"
            elif want == "str":
                if not isinstance(val, str):
                    return f"field '{key}' must be str, got {type(val).__name__}"
            elif want == "list":
                if not isinstance(val, list):
                    return f"field '{key}' must be list"
            elif want == "dict":
                if not isinstance(val, dict):
                    return f"field '{key}' must be dict"
            if key in spec.enum and val not in spec.enum[key]:
                return f"field '{key}' must be one of {spec.enum[key]}"
        return None

    def invoke(
        self,
        world: World,
        tool: str,
        args: dict[str, Any],
        task: TaskSpec,
        ctx: CallContext,
    ) -> tuple[World, ToolResult]:
        spec = self.TOOLS.get(tool)
        if spec is None:
            return world, ToolResult(
                ok=False, data=None, error_code="E_UNKNOWN_TOOL",
                detail=f"unknown tool '{tool}'", state_version=world.version,
            )
        args = dict(args or {})
        verr = self._validate(spec, args)
        if verr is not None:
            return world, ToolResult(
                ok=False, data=None, error_code="E_SCHEMA", detail=verr,
                state_version=world.version,
            )

        plan = ctx.fault.plan if (ctx.fault is not None and ctx.fault.active) else None
        ftype = plan.type if plan else None
        suppressed = bool(ftype) and ftype in ctx.suppressed
        is_auth = tool in self.AUTHORITATIVE_TOOLS

        # ---- fault pre-conditions (before any state change) --------------- #
        if plan is not None and not suppressed and not is_auth:
            if ftype == FailureType.UNRECOVERABLE_EXTERNAL_ERROR.value and tool == plan.target_tool:
                return world, ToolResult(
                    ok=False, data=None, error_code="E_UNAVAILABLE",
                    detail="upstream dependency unavailable", state_version=world.version,
                )
            if ftype == FailureType.INVALID_ARGUMENTS.value and tool == plan.target_tool:
                # Persistent while the call is byte-identical; a *repaired* call
                # (different arguments) gets through. This is the difference
                # between retry-only and revalidation, made mechanical.
                sig = _args_signature(args)
                if "_sig" not in plan.payload:
                    plan.payload["_sig"] = sig
                if sig == plan.payload["_sig"]:
                    return world, ToolResult(
                        ok=False, data=None, error_code="E_SCHEMA",
                        detail=plan.payload.get("detail", "argument rejected by tool contract"),
                        state_version=world.version,
                    )
            # COMMUNICATION_FAILURE is handled after dispatch: the sender must
            # believe it delivered, so the handler still runs.

        # ---- stale reads: serve from a lagged snapshot -------------------- #
        if (
            plan is not None
            and not suppressed
            and ftype == FailureType.STALE_STATE.value
            and tool == plan.target_tool
            and tool in self.READ_TOOLS
            and not is_auth
        ):
            lag = int(plan.payload.get("lag", 1))
            lag_data, lag_ver = world.view_lagged(lag)
            tmp = World(lag_data, lag_ver)
            try:
                data, _ = self._dispatch(tmp, tool, args, task)
            except ToolError as exc:
                return world, ToolResult(
                    ok=False, data=None, error_code=exc.code, detail=exc.detail,
                    state_version=world.version,
                )
            return world, ToolResult(
                ok=True, data=data, state_version=lag_ver, stale=(lag_ver < world.version),
                detail=f"served from replica at version {lag_ver}",
            )

        # ---- communication failure: sender believes delivery, recipient is blind -- #
        if (
            plan is not None
            and not suppressed
            and ftype == FailureType.COMMUNICATION_FAILURE.value
            and tool == self.MESSAGE_TOOL
        ):
            w = world.clone()
            pre_version, pre_data = world.version, copy.deepcopy(world.data)
            try:
                data, mutated = self._dispatch(w, tool, args, task)
            except ToolError as exc:
                return world, ToolResult(
                    ok=False, data=None, error_code=exc.code, detail=exc.detail,
                    state_version=world.version,
                )
            if mutated:
                w.record_and_bump(pre_version, pre_data)
                inbox = w.data.get("inbox")
                if isinstance(inbox, dict):
                    queue = inbox.get(args.get("to"))
                    if queue:
                        queue.pop()  # vanished in transit; outbox still records it
            return w, ToolResult(
                ok=True, data={"delivered": False, "to": args.get("to")},
                detail="accepted by the bus but never delivered",
                state_version=w.version, message_lost=True, delivered=False,
            )

        # ---- normal execution --------------------------------------------- #
        w = world.clone()
        pre_version, pre_data = world.version, copy.deepcopy(world.data)
        try:
            data, mutated = self._dispatch(w, tool, args, task)
        except ToolError as exc:
            return world, ToolResult(
                ok=False, data=None, error_code=exc.code, detail=exc.detail,
                state_version=world.version,
            )

        conflict = False
        if (
            plan is not None
            and not suppressed
            and ftype == FailureType.CONTRADICTORY_EVIDENCE.value
            and tool == plan.target_tool
            and isinstance(data, dict)
        ):
            fieldname = plan.payload.get("field")
            if fieldname in data:
                data = dict(data)
                data[fieldname] = plan.payload["planted"]
                conflict = True

        if mutated:
            w.record_and_bump(pre_version, pre_data)
        else:
            # Telemetry-only updates (agent-progress bookkeeping) are kept but do
            # not advance the logical clock: a read must not make the world newer.
            w.version = pre_version

        return w, ToolResult(
            ok=True, data=data, state_version=w.version if mutated else pre_version,
            stale=False, conflict=conflict, authoritative=is_auth,
        )

    # ---- background process ---------------------------------------------- #
    def background_step(self, world: World, task: TaskSpec, step_index: int) -> World | None:
        """A concurrent writer. This is what makes staleness physically real.

        Deterministic: driven by a per-task seeded schedule, identical in the
        fault-free counterfactual.
        """
        return None

    # ---- grading ---------------------------------------------------------- #
    def grade(self, world: World, task: TaskSpec) -> GradeResult:
        raise NotImplementedError

    def goal_projection(self, task: TaskSpec) -> dict[str, Any]:
        """Ground-truth goal state, computed from the task, not from any run."""
        raise NotImplementedError

    def actionable_tools(self) -> list[str]:
        return sorted(self.TOOLS)

    # ---- recovery operators --------------------------------------------- #
    #: Tool used by ``REFRESH_STATE`` to re-observe the resource last touched.
    REFRESH_TOOL: str = ""

    def _step_for(self, tool: str, last_step: PlanStep | None, note: str) -> PlanStep | None:
        spec = self.TOOLS.get(tool)
        if spec is None:
            return None
        args: dict[str, Any] = {}
        if last_step is not None:
            for k in spec.required:
                if k in last_step.args:
                    args[k] = last_step.args[k]
        if any(k not in args for k in spec.required):
            return None
        agent = last_step.agent if last_step is not None else "solo"
        return PlanStep(tool, args, (), agent=agent, note=note)

    def repair_args(
        self, tool: str, args: dict[str, Any], belief: Any
    ) -> dict[str, Any] | None:
        """Schema-directed repair of a rejected call (the REVALIDATE_ARGS operator).

        Coerces values to the declared types and drops/replaces fields that the
        contract cannot accept. Returns ``None`` when the call cannot be made
        valid, in which case revalidation is honestly reported as a failed repair.
        """
        spec = self.TOOLS.get(tool)
        if spec is None or not isinstance(args, dict):
            return None
        out: dict[str, Any] = {}
        for key, want in spec.schema.items():
            if key not in args:
                continue
            val = args[key]
            if want == "int":
                if isinstance(val, bool) or not isinstance(val, int):
                    try:
                        val = int(float(str(val).strip()))
                    except (TypeError, ValueError):
                        return None
            elif want == "str":
                if not isinstance(val, str):
                    val = str(val)
            elif want == "dict":
                if not isinstance(val, dict):
                    return None
            elif want == "list":
                if not isinstance(val, list):
                    return None
            out[key] = val
        for req in spec.required:
            if req not in out:
                return None
        for key, allowed in spec.enum.items():
            if key in out and out[key] not in allowed:
                return None
        return out

    def refresh_step(self, world: World, task: TaskSpec, last_step: PlanStep | None) -> PlanStep | None:
        """Re-observe the resource the last action acted upon, at current version."""
        tool = self.REFRESH_TOOL or (sorted(self.READ_TOOLS)[0] if self.READ_TOOLS else "")
        step = self._step_for(tool, last_step, "recovery:refresh")
        if step is None and self.AUTHORITATIVE_TOOLS:
            for cand in sorted(self.AUTHORITATIVE_TOOLS):
                step = self._step_for(cand, last_step, "recovery:refresh")
                if step is not None:
                    break
        return step

    def authoritative_step(self, world: World, task: TaskSpec, last_step: PlanStep | None) -> PlanStep | None:
        """Consult the authoritative view to resolve a disagreement."""
        for cand in sorted(self.AUTHORITATIVE_TOOLS):
            step = self._step_for(cand, last_step, "recovery:authoritative")
            if step is not None:
                return step
        return None

    def resync_step(self, world: World, task: TaskSpec, last_step: PlanStep | None) -> PlanStep | None:
        """Re-send the last outbound message (or re-read shared state)."""
        if self.MESSAGE_TOOL and last_step is not None and last_step.tool == self.MESSAGE_TOOL:
            spec = self.TOOLS[self.MESSAGE_TOOL]
            args = {k: last_step.args[k] for k in spec.required if k in last_step.args}
            if len(args) == len(spec.required):
                # Replay must keep the original sender: the message body is derived
                # from that role's belief, and a different role has no quote.
                return PlanStep(self.MESSAGE_TOOL, args, ("inbox",),
                                agent=last_step.agent, note="recovery:resync")
        return self.authoritative_step(world, task, last_step) or self.refresh_step(world, task, last_step)

    # ---- fault construction ---------------------------------------------- #
    #: Which side owns each failure class for this family ("env" or "agent").
    AGENT_LATERAL: frozenset[str] = frozenset(
        {FailureType.PLANNING_ERROR.value, FailureType.OBJECTIVE_DRIFT.value}
    )

    def make_fault(
        self, task: TaskSpec, world: World, rng: random.Random, failure_type: str, inject_at: int
    ) -> FaultPlan | None:
        raise NotImplementedError

    def default_expected(self, world: World, step: PlanStep) -> dict[str, Any]:
        """What the observable state should look like after the intended step."""
        w = world.clone()
        try:
            _, mutated = self._dispatch(w, step.tool, dict(step.args), self._task_hint)
        except (ToolError, Exception):
            return world.project(step.touched)
        return w.project(step.touched)

    _task_hint: TaskSpec  # set per-episode by the harness (see apply_intended)


def _args_signature(args: dict[str, Any]) -> str:
    import json

    return json.dumps(args, sort_keys=True, default=str)


def apply_intended(env: Environment, world: World, step: PlanStep, task: TaskSpec) -> World:
    """Apply a plan step with no faults, for expected-state projection."""
    w = world.clone()
    try:
        _, mutated = env._dispatch(w, step.tool, dict(step.args), task)
    except ToolError:
        return world
    if mutated:
        w.record_and_bump(world.version, copy.deepcopy(world.data))
    return w


def project_after(
    env: Environment, world: World, step: PlanStep, task: TaskSpec
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Return (expected_projection, pre_projection, mutated)."""
    pre = world.project(step.touched)
    after = apply_intended(env, world, step, task)
    mut = after.version != world.version
    return (after.project(step.touched) if mut else pre), pre, mut
