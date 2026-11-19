"""The reference agent.

Two deliberate abstractions, both documented in the paper:

1. **Known SOP, unknown data.** The agent knows *which* steps the task requires
   (``env.solve`` / ``env.next_step`` give the structure) but derives every
   argument value from its own belief state (``env.derive_args``). This is the
   standard "workflow agent with skills" assumption; it avoids simulating an LLM
   planner while keeping the failure mechanics honest, because no argument is
   ever taken from oracle state.
2. **Errors are processes, not noise.** A deviation is a named failure class with
   a locus, chosen either by the controlled schedule (``fault_source="controlled"``)
   or by the agent's competence model (``fault_source="natural"``). The natural
   mode exists so that detectors can be tested on faults *not* produced by the
   injection machinery.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from arb.belief import Belief
from arb.envs.base import Environment, World
from arb.schema import FaultPlan, PlanStep, TaskSpec
from arb.taxonomy import FailureType, RecoveryAction


class StopEpisode(Exception):
    """Raised when the agent halts (escalation, completion, or budget exhaustion)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class Action:
    tool: str
    args: dict[str, Any]
    agent: str = "solo"
    note: str = ""
    intended: PlanStep | None = None
    recovery_action: str | None = None
    is_recovery: bool = False
    deviation: dict[str, Any] | None = None
    natural_fault: FaultPlan | None = None


@dataclass
class AgentConfig:
    """Competence model for the natural-fault condition."""

    competence: float = 1.0
    #: Conditional distribution over agent-side failure classes.
    natural_mix: dict[str, float] = field(
        default_factory=lambda: {
            FailureType.INVALID_ARGUMENTS.value: 0.40,
            FailureType.PLANNING_ERROR.value: 0.40,
            FailureType.OBJECTIVE_DRIFT.value: 0.20,
        }
    )
    max_retries: int = 2


class ReferenceAgent:
    def __init__(self, config: AgentConfig | None = None, seed: int = 0, role: str = "solo") -> None:
        self.config = config or AgentConfig()
        self.rng = random.Random(seed)
        self.role = role
        #: Per-role belief stores. Roles in the multi-agent track do NOT share
        #: observations, which is what makes a dropped message consequential.
        self.beliefs: dict[str, Belief] = {role: Belief()}
        self.plan: list[PlanStep] = []
        self.cursor = 0
        self.objective = "true"
        self.last_action: Action | None = None
        self.last_step: PlanStep | None = None
        self.retries_used = 0
        self.natural_emitted = False
        self._suppressed: set[str] = set()

    # ------------------------------------------------------------------ #
    def belief_for(self, role: str) -> Belief:
        if role not in self.beliefs:
            self.beliefs[role] = Belief()
        return self.beliefs[role]

    @property
    def belief(self) -> Belief:
        return self.belief_for(self.current_role)

    def reset(self, env: Environment, world: World, task: TaskSpec) -> None:
        self.env = env
        self.task = task
        self.current_role = self.role
        self.beliefs = {self.role: Belief()}
        for b in self.beliefs.values():
            b.reset()
        self.objective = "true"
        self.plan = env.solve(world, task, "true")
        self.cursor = 0
        self.last_action = None
        self.last_step = None
        self.retries_used = 0
        self.natural_emitted = False
        self.natural_fault_plan: FaultPlan | None = None
        self.skip_next_mutation = False
        self.pending_repeat = False
        self._suppressed.clear()

    # ---- structure ------------------------------------------------------ #
    def _structural_step(self, env: Environment, world: World, task: TaskSpec) -> PlanStep | None:
        if getattr(env, "INCREMENTAL", False):
            for role in getattr(env, "AGENTS", ("solo",)):
                belief = self.belief_for(role)
                best = env.belief_best(belief, self.objective) if hasattr(env, "belief_best") else None
                step = env.next_step(role, world, task, self.objective, best=best)
                if step is not None:
                    return step
            return None
        # Non-incremental: walk the SOP, skipping reads whose belief is current.
        guard = 0
        while self.cursor < len(self.plan) and guard < len(self.plan) + 4:
            step = self.plan[self.cursor]
            if step.tool in env.READ_TOOLS and self._belief_current(step, env):
                self.cursor += 1
                guard += 1
                continue
            return step
        return None

    def _belief_current(self, step: PlanStep, env: Environment) -> bool:
        """Reads are never skipped.

        A global version counter bumps on *any* mutation, so "is this read still
        fresh?" would become true and false repeatedly and the SOP would never
        terminate. The SOP lists each read exactly once per object, which is
        sufficient; a replan may re-read, and that cost is legitimate.
        """
        return False

    def _key(self, step: PlanStep) -> str:
        from arb.belief import obs_key

        return obs_key(step.tool, step.args)

    # ---- main entry point ----------------------------------------------- #
    def act(
        self,
        env: Environment,
        world: World,
        task: TaskSpec,
        *,
        step_index: int,
        hint: RecoveryAction | None = None,
        deviation: dict[str, Any] | None = None,
    ) -> Action:
        self.env = env
        self.task = task
        self._last_world_version = world.version
        self.belief.inbox = list(world.data.get("inbox", {}).get(self.role, [])) if self.role != "solo" else []

        self.belief_for(self.current_role).objective = self.objective

        # ---- recovery operators ------------------------------------------ #
        if hint is not None:
            act = self._recovery(env, world, task, hint, step_index)
            if act is not None:
                return act

        # ---- a scheduled duplicate fires on the next mutating step -------- #
        if self.pending_repeat and self.last_action is not None and self.last_action.tool not in env.READ_TOOLS:
            self.pending_repeat = False
            prev = self.last_action
            act = Action(tool=prev.tool, args=dict(prev.args), agent=prev.agent,
                         note="deviation:duplicate", intended=prev.intended,
                         deviation={"type": FailureType.PLANNING_ERROR.value, "variant": "duplicate"})
            self.last_action = act
            return act

        # ---- scheduled agent-side deviations ----------------------------- #
        if deviation is not None:
            act = self._apply_deviation(env, world, task, deviation, step_index)
            if act is not None:
                return act
            return self._execute(env, world, task, step_index)

        # ---- natural competence failures --------------------------------- #
        if not self.natural_emitted and self.rng.random() >= self.config.competence:
            dev = self._sample_natural(env, world, task)
            if dev is not None:
                self.natural_emitted = True
                probe = self._structural_step(env, world, task)
                self.natural_fault_plan = FaultPlan(
                    type=dev["type"], inject_at=step_index,
                    target_tool=(probe.tool if probe else None), target_agent=None,
                    payload={"variant": dev.get("variant", ""), "locus": "agent"},
                    propagation_window=(99 if dev["type"] == FailureType.OBJECTIVE_DRIFT.value else 3),
                    recoverable=True, severity=0.6,
                )
                act = self._apply_deviation(env, world, task, dev, step_index)
                if act is not None:
                    return act
                return self._execute(env, world, task, step_index)

        return self._execute(env, world, task, step_index)

    # ---- normal execution ------------------------------------------------ #
    def _execute(
        self, env: Environment, world: World, task: TaskSpec, step_index: int,
        *, override_step: PlanStep | None = None, is_recovery: bool = False,
        recovery_action: str | None = None,
    ) -> Action:
        step = override_step or self._structural_step(env, world, task)
        if step is None:
            raise StopEpisode("plan exhausted")
        if self.skip_next_mutation and step.tool not in env.READ_TOOLS:
            # Omit a required mutation: the classic silent planning error.
            self.skip_next_mutation = False
            if not getattr(env, "INCREMENTAL", False):
                self.cursor += 1
            return self._execute(env, world, task, step_index, is_recovery=is_recovery,
                                 recovery_action=recovery_action)
        self.current_role = step.agent
        belief = self.belief_for(step.agent)
        belief.objective = self.objective

        intended = step
        tool, args = step.tool, dict(step.args)

        # Mutations must be derived from belief, never from the oracle step.
        if tool not in env.READ_TOOLS:
            derived = env.derive_args(tool, task, belief, step)
            if derived is None:
                # Belief insufficient: pull the prerequisite read forward.
                pull = env.refresh_step(world, task, step) or env.refresh_step(
                    world, task, self.last_step)
                if pull is None:
                    raise StopEpisode("cannot ground mutation in observation")
                step = pull
                tool, args = pull.tool, dict(pull.args)
                intended = pull
            elif derived == env.SKIP:
                # The step is unnecessary given current belief: skip it.
                if not getattr(env, "INCREMENTAL", False):
                    self.cursor += 1
                return self._execute(env, world, task, step_index, is_recovery=is_recovery,
                                     recovery_action=recovery_action)
            else:
                args = derived

        action = Action(tool=tool, args=args, agent=step.agent, note=step.note,
                        intended=intended, is_recovery=is_recovery,
                        recovery_action=recovery_action)
        if not getattr(env, "INCREMENTAL", False) and not is_recovery and tool == intended.tool:
            self.cursor += 1
        self.last_action = action
        self.last_step = step
        return action

    # ---- recovery -------------------------------------------------------- #
    def _recovery(
        self, env: Environment, world: World, task: TaskSpec,
        hint: RecoveryAction, step_index: int,
    ) -> Action | None:
        if hint is RecoveryAction.CONTINUE:
            return None
        if hint is RecoveryAction.ESCALATE_ABORT:
            raise StopEpisode("escalated_abort")
        if hint is RecoveryAction.RETRY:
            if self.last_action is None or self.retries_used >= self.config.max_retries:
                raise StopEpisode("retry budget exhausted")
            self.retries_used += 1
            act = Action(tool=self.last_action.tool, args=dict(self.last_action.args),
                         agent=self.last_action.agent, note="recovery:retry",
                         intended=self.last_action.intended, is_recovery=True,
                         recovery_action=RecoveryAction.RETRY.value)
            self.last_action = act
            return act
        if hint is RecoveryAction.REVALIDATE_ARGS:
            self._suppressed.add(FailureType.INVALID_ARGUMENTS.value)
            step = self.last_step or self._structural_step(env, world, task)
            if step is None:
                raise StopEpisode("nothing to revalidate")
            belief = self.belief_for(step.agent)
            base_args = dict(self.last_action.args) if self.last_action else dict(step.args)
            repaired = env.repair_args(step.tool, base_args, belief)
            if repaired is None:
                # Repair not possible: fall through to replanning.
                return self._recovery(env, world, task, RecoveryAction.REPLAN, step_index)
            act = Action(tool=step.tool, args=repaired, agent=step.agent,
                         note="recovery:revalidate", intended=step, is_recovery=True,
                         recovery_action=hint.value)
            self.last_action = act
            return act
        if hint is RecoveryAction.REFRESH_STATE:
            self._suppressed.add(FailureType.STALE_STATE.value)
            step = env.refresh_step(world, task, self.last_step or self.last_action and self.last_action.intended)
            if step is None:
                return None
            return self._execute(env, world, task, step_index, override_step=step,
                                 is_recovery=True, recovery_action=hint.value)
        if hint is RecoveryAction.RE_READ_AUTHORITATIVE:
            self._suppressed.add(FailureType.CONTRADICTORY_EVIDENCE.value)
            step = env.authoritative_step(world, task, self.last_step or (self.last_action.intended if self.last_action else None))
            if step is None:
                return None
            return self._execute(env, world, task, step_index, override_step=step,
                                 is_recovery=True, recovery_action=hint.value)
        if hint is RecoveryAction.REPLAN:
            self.plan = env.solve(world, task, self.objective)
            self.cursor = 0
            if getattr(env, "INCREMENTAL", False):
                return self._execute(env, world, task, step_index, is_recovery=True,
                                     recovery_action=hint.value)
            step = self._structural_step(env, world, task)
            if step is None:
                raise StopEpisode("replan found nothing to do")
            return self._execute(env, world, task, step_index, override_step=step,
                                 is_recovery=True, recovery_action=hint.value)
        if hint is RecoveryAction.RESYNC_MESSAGE:
            self._suppressed.add(FailureType.COMMUNICATION_FAILURE.value)
            step = env.resync_step(world, task, self.last_step)
            if step is None:
                return None
            return self._execute(env, world, task, step_index, override_step=step,
                                 is_recovery=True, recovery_action=hint.value)
        if hint is RecoveryAction.REANCHOR_OBJECTIVE:
            self._suppressed.add(FailureType.OBJECTIVE_DRIFT.value)
            self.objective = "true"
            self.belief.objective = "true"
            return self._recovery(env, world, task, RecoveryAction.REPLAN, step_index)
        return None

    @property
    def suppressed(self) -> frozenset[str]:
        return frozenset(self._suppressed)

    # ---- deviations ------------------------------------------------------ #
    def _sample_natural(
        self, env: Environment, world: World, task: TaskSpec
    ) -> dict[str, Any] | None:
        step = self._structural_step(env, world, task)
        if step is None or step.tool in env.READ_TOOLS:
            return None
        mix = self.config.natural_mix
        keys = list(mix)
        weights = [mix[k] for k in keys]
        kind = self.rng.choices(keys, weights=weights, k=1)[0]
        variant = ""
        if kind == FailureType.PLANNING_ERROR.value:
            variant = "duplicate" if getattr(self.env, "INCREMENTAL", False) \
                else self.rng.choice(["skip", "duplicate"])
        return {"type": kind, "variant": variant, "locus": "agent", "inject_at": -1}

    def _apply_deviation(
        self, env: Environment, world: World, task: TaskSpec,
        dev: dict[str, Any], step_index: int,
    ) -> Action | None:
        kind = dev["type"]
        if kind == FailureType.OBJECTIVE_DRIFT.value:
            self.objective = "proxy"
            self.belief.objective = "proxy"
            self.plan = env.solve(world, task, "proxy")
            self.cursor = 0
            return None  # execute normally under the drifted objective
        if kind == FailureType.PLANNING_ERROR.value:
            variant = dev.get("variant") or "skip"
            if variant == "skip" and not getattr(env, "INCREMENTAL", False):
                self.skip_next_mutation = True
                return None
            # duplicate: re-issue the next mutating action once it has run
            self.pending_repeat = True
            return None
        if kind == FailureType.INVALID_ARGUMENTS.value:
            act = self._execute(env, world, task, step_index)
            args = dict(act.args)
            # Type-confuse the first *scalar* argument: the call stays valid
            # JSON but violates the tool schema.
            scalar = [k for k in sorted(args) if isinstance(args[k], (int, float))]
            if scalar:
                k = scalar[0]
                args[k] = f"{args[k]}"
            act.args = args
            act.note = "deviation:invalid_arguments"
            act.deviation = {"type": kind, "variant": "type_confusion"}
            return act
        return None

    # ---- convenience ----------------------------------------------------- #
    def observe(self, step_index: int, role: str, tool: str, args: dict[str, Any], ok: bool,
                data: Any, version: int) -> None:
        b = self.belief_for(role)
        if ok and data is not None:
            b.put(tool, args, data, version, step_index)
        if ok:
            b.put_applied(step_index, tool, args)
