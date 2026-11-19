"""Episode runner and task grid.

One episode = task instance + optional scheduled fault + policy + detector.
Everything is deterministic given ``EpisodeConfig``: same config, same bytes.

The runner is the only place that knows both the ground-truth fault and the
monitor's view. It keeps them apart on purpose: ``signals`` come from
``SignalExtractor``, while ``fault``/``labels`` are written to the trajectory but
never handed to a detector or a policy.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

from arb.agents.scripted import AgentConfig, ReferenceAgent, StopEpisode
from arb.costs import PRICE, step_cost
from arb.detectors.features import SignalExtractor
from arb.detectors.rules import make_detector
from arb.envs import make_env
from arb.envs.base import ActiveFault, CallContext, ToolSpec, diff_fields, project_after, stable_seed
from arb.perturb import schedule_fault
from arb.schema import FaultPlan, PlanStep, Step, Trajectory
from arb.taxonomy import FAILURE_SPECS, FailureType, RecoveryAction

# --------------------------------------------------------------------------- #
# Split definition: held-out templates AND held-out configurations.
# --------------------------------------------------------------------------- #
SEEN_TEMPLATES: tuple[str, ...] = ("A", "B")
HELD_OUT_TEMPLATE: str = "C"
TRAIN_CONFIGS: tuple[str, ...] = ("cfg1", "cfg2", "cfg3")
DEV_CONFIG: str = "cfg5"
TEST_CONFIG: str = "cfg6"


def split_for(template: str, config: str) -> str:
    if template == HELD_OUT_TEMPLATE:
        return "test_ood"          # unseen task template (and unseen configs)
    if config == TEST_CONFIG:
        return "test_id"           # seen template, unseen configuration
    if config == DEV_CONFIG:
        return "dev"
    return "train"


@dataclass(frozen=True)
class EpisodeConfig:
    family: str
    template: str
    config: str
    seed: int
    failure_type: str | None
    policy: str
    detector: str
    horizon_class: str = "short"
    monitor_mode: str = "instrumented"
    fault_source: str = "controlled"
    competence: float = 1.0
    split: str = ""

    @property
    def resolved_split(self) -> str:
        return self.split or split_for(self.template, self.config)

    @property
    def episode_id(self) -> str:
        ft = self.failure_type or "none"
        src = "nat" if self.fault_source == "natural" else "ctl"
        return (f"{self.family}|{self.template}|{self.config}|s{self.seed}|{ft}|"
                f"{self.policy}|{self.detector}|{self.monitor_mode}|{src}|{self.horizon_class}")

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["episode_id"] = self.episode_id
        d["resolved_split"] = self.resolved_split
        return d


def _label_window(fault: FaultPlan, horizon: int) -> int:
    """Steps after injection during which the fault counts as present."""
    w = fault.propagation_window
    if w >= 90:
        return horizon + 5
    return max(3, min(w, horizon))


def run_episode(
    cfg: EpisodeConfig,
    *,
    detector: Any | None = None,
    policy: Any | None = None,
    counterfactual: dict[str, Any] | None = None,
    max_steps_cap: int = 90,
) -> dict[str, Any]:
    env = make_env(cfg.family)
    split = cfg.resolved_split
    task = env.make_task(cfg.template, cfg.config, cfg.seed, split, cfg.horizon_class)
    world = env.initial_world(task)

    role = "solo" if env.track == "single" else "planner"
    agent = ReferenceAgent(AgentConfig(competence=cfg.competence),
                           seed=stable_seed("agent", cfg.episode_id), role=role)
    agent.reset(env, world, task)

    extractor = SignalExtractor(env.family, cfg.monitor_mode, task.horizon)
    det = detector if detector is not None else (make_detector(cfg.detector) if cfg.detector != "none" else None)
    if det is not None:
        det.reset()
    if policy is not None:
        policy.reset()

    fault: FaultPlan | None = None
    if cfg.failure_type:
        fault = schedule_fault(env, task, world, cfg.failure_type,
                               seed=cfg.seed, horizon=task.horizon)
        if fault is None:
            raise ValueError(f"{cfg.family} cannot inject {cfg.failure_type}")

    queue: deque[RecoveryAction] = deque()
    handled = False
    steps: list[Step] = []
    attempts: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    score_log: list[dict[str, Any]] = []
    retries = 0
    early_alarms = 0
    fail_counts: dict[str, int] = {}
    handled_at: int | None = None
    version_at_inject: int | None = None
    first_success: int | None = None
    stop_reason = ""
    max_steps = min(max_steps_cap, task.horizon * 3 + 12)

    for t in range(max_steps):
        hint = queue.popleft() if queue else None
        deviation = None
        if fault is not None and fault.payload.get("locus") == "agent" and t == fault.inject_at:
            deviation = {"type": fault.type,
                         "variant": fault.payload.get("variant", "skip"),
                         "locus": "agent"}
        try:
            action = agent.act(env, world, task, step_index=t, hint=hint, deviation=deviation)
        except StopEpisode as exc:
            stop_reason = exc.reason
            break

        if fault is None and getattr(agent, "natural_fault_plan", None) is not None:
            fault = agent.natural_fault_plan
            agent.natural_fault_plan = None

        if fault is not None and version_at_inject is None and t >= fault.inject_at:
            version_at_inject = world.version

        ctx = CallContext(
            fault=ActiveFault(fault, t) if fault is not None else None,
            step_index=t, agent=action.agent, suppressed=agent.suppressed,
            monitor_mode=cfg.monitor_mode, episode_id=cfg.episode_id,
        )
        intended = action.intended or PlanStep(action.tool, action.args, ())
        world_before = world
        expected_proj, _, _ = project_after(env, world_before, intended, task)
        world, res = env.invoke(world_before, action.tool, action.args, task, ctx)

        spec = env.TOOLS.get(action.tool, ToolSpec(action.tool, mutates=True))
        context_tokens = 180 if (cfg.family == "procurement" and action.agent != "planner") else 0
        cost = step_cost(tool=action.tool, args=action.args, result_payload=res.data,
                         step_index=t, base_latency_ms=spec.base_latency_ms,
                         jitter_key=f"{cfg.episode_id}:{t}", price=PRICE,
                         context_tokens=context_tokens)
        observed_proj = world.project(intended.touched) if intended.touched else {}

        step = Step(
            index=t, agent=action.agent, tool=action.tool, args=action.args,
            ok=res.ok, error_code=res.error_code, data=res.data,
            state_version=res.state_version,
            expected_state=expected_proj, observed_state=observed_proj,
            expected_matches=(expected_proj == observed_proj),
            retry_index=retries,
            signals={
                "_stale_flag": 1.0 if res.stale else 0.0,
                "_conflict_flag": 1.0 if res.conflict else 0.0,
                "_message_lost": 1.0 if res.message_lost else 0.0,
            },
            tokens_in=cost.tokens_in, tokens_out=cost.tokens_out,
            tool_payload_tokens=cost.tool_payload_tokens,
            latency_ms=cost.latency_ms, usd=cost.usd,
        )
        if cfg.monitor_mode != "instrumented":
            for k in ("_stale_flag", "_conflict_flag", "_message_lost"):
                step.signals[k] = 0.0
        extractor.observe(step)
        steps.append(step)
        agent.observe(t, action.agent, action.tool, action.args, res.ok, res.data, res.state_version)

        if hint is not None:
            attempts.append({"step": t, "action": hint.value, "ok": res.ok})
            if hint is RecoveryAction.RETRY:
                retries += 1

        # ---- repeated-failure guard: bound pathological loops ------------- #
        if not res.ok:
            sig = f"{action.tool}|{json.dumps(action.args, sort_keys=True, default=str)}"
            fail_counts[sig] = fail_counts.get(sig, 0) + 1
        else:
            fail_counts.clear()
        looped = bool(fail_counts) and max(fail_counts.values()) >= 3

        # ---- retry-on-error baseline (no detector involved) --------------- #
        if (
            policy is not None
            and getattr(policy, "on_error", False)
            and not res.ok
            and retries < getattr(policy, "max_retries", 2)
            and not queue
        ):
            queue.append(RecoveryAction.RETRY)

        # ---- detection (monitor view only) ---------------------------- #
        if det is not None:
            d = det.observe(step.signals)
            score_log.append({"t": t, "score": float(d.score), "type": d.type})
            if d.fired and fault is not None and t < fault.inject_at:
                early_alarms += 1
            if d.fired and fault is not None and t >= fault.inject_at and not handled:
                handled = True
                handled_at = t
                detections.append({"step": t, "type": d.type, "score": float(d.score),
                                   "posterior": dict(d.posterior)})
                if policy is not None and not getattr(policy, "on_error", False):
                    dec = policy.decide(
                        detected_type=d.type, posterior=d.posterior, step_index=t,
                        injections_handled=0, retries_used=retries,
                        features={**step.signals, "_score": float(d.score)},
                    )
                    for a in dec.program:
                        queue.append(a)

        if fault is not None and t >= fault.inject_at and first_success is None:
            if env.grade(world, task).success:
                first_success = t

        if looped:
            stop_reason = "repeated_failure"
            break

        bg = env.background_step(world, task, t)
        if bg is not None:
            world = bg

    grade = env.grade(world, task)

    # ---- outcome bookkeeping ------------------------------------------- #
    spec = FAILURE_SPECS[FailureType(fault.type)] if fault is not None else None
    # Containment = a clean halt: after a persistent outage begins, the system
    # makes no further successful mutations. Retrying reads is fine; writing is
    # not. This is stricter and more meaningful than "version unchanged", which
    # would be satisfied by an agent that simply kept hammering other tools.
    consequential = [
        s for s in steps
        if fault is not None and s.index > fault.inject_at and s.ok and s.tool not in env.READ_TOOLS
    ]
    contained = bool(spec is not None and not spec.recoverable and not consequential)
    if fault is None:
        recovery_outcome = "no_fault"
    elif spec is not None and not spec.recoverable:
        recovery_outcome = "contained" if contained else "not_contained"
    elif not handled:
        recovery_outcome = "undetected"
    elif grade.success:
        recovery_outcome = "recovered"
    else:
        recovery_outcome = "attempted_failed"

    totals = {
        "tokens_in": sum(s.tokens_in for s in steps),
        "tokens_out": sum(s.tokens_out for s in steps),
        "tool_payload_tokens": sum(s.tool_payload_tokens for s in steps),
        "latency_ms": round(sum(s.latency_ms for s in steps), 3),
        "usd": round(sum(s.usd for s in steps), 8),
        "tool_calls": len(steps),
    }
    totals["tokens_total"] = totals["tokens_in"] + totals["tokens_out"] + totals["tool_payload_tokens"]

    detection = None
    if detections:
        first = detections[0]
        detection = {
            "step": first["step"],
            "type": first["type"],
            "score": first["score"],
            "detected": True,
            "latency_steps": (first["step"] - fault.inject_at) if fault is not None else None,
            "latency_ms": round(sum(s.latency_ms for s in steps[: first["step"] + 1]), 3),
            "posterior": first["posterior"],
            "n_events": len(detections),
        }
    elif fault is not None:
        detection = {"step": None, "type": None, "score": 0.0, "detected": False,
                     "latency_steps": None, "latency_ms": None, "posterior": {}, "n_events": 0}

    # ---- per-step labels (never shown to detectors) -------------------- #
    # Built from the fault schedule and the executed steps, independent of
    # whether a detector ran. Scores/predictions are merged in when available.
    labels = []
    if fault is not None:
        win = _label_window(fault, task.horizon)
        by_index = {sc["t"]: sc for sc in score_log}
        for s in steps:
            active = fault.inject_at <= s.index <= fault.inject_at + win
            sc = by_index.get(s.index, {})
            labels.append({"t": s.index, "label": fault.type if active else "none",
                           "score": float(sc.get("score", 0.0)),
                           "pred": sc.get("type")})

    cf = counterfactual or grade.observed
    traj = Trajectory(
        episode_id=cfg.episode_id, track=env.track, family=env.family,
        template_id=cfg.template, config_id=cfg.config, split=split,
        task_id=task.task_id, seed=cfg.seed, objective=env.objective_text(task),
        monitor_mode=cfg.monitor_mode, policy=cfg.policy, detector=cfg.detector,
        steps=steps,
        fault=(fault.to_dict() if fault is not None else None),
        detection=detection,
        recovery={
            "attempts": attempts,
            "outcome": recovery_outcome,
            "n_attempts": len(attempts),
            "actions": [a["action"] for a in attempts],
            "ttr_steps": (first_success - fault.inject_at) if (first_success is not None and fault) else None,
            "contained": contained,
            "stop_reason": stop_reason or "plan_or_budget_exhausted",
        },
        outcome={
            "success": grade.success,
            "violations": grade.violations[:8],
            "n_violations": len(grade.violations),
            "claimed_success": _self_report(steps, grade),
            "retries": retries,
            "tool_calls": len(steps),
            "n_corrupted_fields": len(diff_fields(grade.observed, grade.expected)),
            "n_downstream_fields": len(diff_fields(grade.observed, cf)),
            "detected": bool(detection and detection["detected"]),
            "early_alarms": early_alarms,
            "handled_at": handled_at,
        },
        cost=totals,
        expected_final_state=grade.expected,
        observed_final_state=grade.observed,
        counterfactual_final_state=cf,
        meta={
            "labels": labels,
            "fault_source": cfg.fault_source,
            "competence": cfg.competence,
            "horizon": task.horizon,
            "split": split,
            "policy_config": {"name": cfg.policy},
            "env_task": task.to_dict(),
            "price_sheet": PRICE.as_dict(),
        },
    )
    return traj.to_dict()


def _self_report(steps: list[Step], grade: Any) -> bool:
    """What the agent's own artifacts claim, independent of verified state.

    Deliberately naive: an agent "claims success" if it wrote a terminal status or
    sent a confirmation. The gap between this and ``grade.success`` is the
    self-report failure rate reported in the paper.
    """
    for s in steps:
        if s.tool in ("update_status", "approve", "close_period", "send_notification") and s.ok:
            if isinstance(s.args, dict):
                if s.args.get("status") in ("resolved", "closed"):
                    return True
                if s.args.get("template") == "confirmed":
                    return True
                if s.tool in ("approve", "close_period"):
                    return True
    return bool(grade.success)


# --------------------------------------------------------------------------- #
def build_tasks(
    families: tuple[str, ...] = ("tickets", "migration", "inventory", "procurement"),
    templates: tuple[str, ...] = ("A", "B", "C"),
    configs: tuple[str, ...] = ("cfg1", "cfg2", "cfg3", "cfg5", "cfg6"),
    seeds: tuple[int, ...] = (0, 1, 2),
    horizon_class: str = "short",
) -> Iterator[dict[str, Any]]:
    for fam in families:
        for tpl in templates:
            for cfg in configs:
                for seed in seeds:
                    yield {"family": fam, "template": tpl, "config": cfg,
                           "seed": seed, "split": split_for(tpl, cfg),
                           "horizon_class": horizon_class}


def counterfactual_key(cfg: EpisodeConfig) -> str:
    return f"{cfg.family}|{cfg.template}|{cfg.config}|s{cfg.seed}|{cfg.horizon_class}"
