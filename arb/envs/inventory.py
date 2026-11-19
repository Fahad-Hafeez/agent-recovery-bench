"""Family 3 (single-agent): inventory reconciliation with an accounting invariant.

The goal is an *invariant* rather than a constant: for every SKU,
``on_hand + sum(adjustments) == ledger``. That makes grading robust to
concurrent warehouse movements while still making stale or contradictory reads
fatal, because the delta computed from a wrong ``on_hand`` breaks the invariant.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from arb.envs.base import Environment, ToolError, ToolSpec, World, stable_seed
from arb.schema import FaultPlan, GradeResult, PlanStep, TaskSpec
from arb.taxonomy import FailureType

TEMPLATE_PARAMS: dict[str, dict[str, Any]] = {
    "A": {"n_skus": 3, "drift_tolerance": 0.02, "movement_rate": 0.0},
    "B": {"n_skus": 4, "drift_tolerance": 0.05, "movement_rate": 0.5},
    # Held out: more SKUs, a looser "materiality" threshold for the proxy policy.
    "C": {"n_skus": 6, "drift_tolerance": 0.10, "movement_rate": 1.0},
}

CONFIG_PARAMS: dict[str, dict[str, Any]] = {
    "cfg1": {"concurrency": 0.0, "noise": 0.05},
    "cfg2": {"concurrency": 0.2, "noise": 0.10},
    "cfg3": {"concurrency": 0.35, "noise": 0.15},
    "cfg5": {"concurrency": 0.5, "noise": 0.20},
    "cfg6": {"concurrency": 0.7, "noise": 0.25},
}


class InventoryEnv(Environment):
    family = "inventory"
    track = "single"
    TEMPLATES = ("A", "B", "C")
    CONFIGS = ("cfg1", "cfg2", "cfg3", "cfg5", "cfg6")

    TOOLS = {
        "list_skus": ToolSpec("list_skus", mutates=False, base_latency_ms=60.0,
                              description="List SKUs in the period."),
        "get_stock": ToolSpec("get_stock", mutates=False, required=("sku",),
                              schema={"sku": "str"}, base_latency_ms=65.0,
                              description="Read on-hand stock for a SKU."),
        "get_ledger": ToolSpec("get_ledger", mutates=False, required=("sku",),
                               schema={"sku": "str"}, base_latency_ms=70.0,
                               description="Read the ledger balance for a SKU."),
        "post_adjustment": ToolSpec("post_adjustment", mutates=True, required=("sku", "delta"),
                                    schema={"sku": "str", "delta": "int"}, base_latency_ms=140.0,
                                    description="Post a stock adjustment."),
        "close_period": ToolSpec("close_period", mutates=True, base_latency_ms=220.0,
                                 description="Close the accounting period."),
        "audit_totals": ToolSpec("audit_totals", mutates=False, base_latency_ms=190.0,
                                 description="Authoritative reconciliation report."),
    }
    READ_TOOLS = frozenset({"list_skus", "get_stock", "get_ledger"})
    AUTHORITATIVE_TOOLS = frozenset({"audit_totals"})
    REFRESH_TOOL = "get_stock"

    # ---- task ------------------------------------------------------------- #
    def make_task(
        self, template_id: str, config_id: str, seed: int, split: str, horizon_class: str = "short"
    ) -> TaskSpec:
        tp = TEMPLATE_PARAMS[template_id]
        cp = CONFIG_PARAMS[config_id]
        rng = random.Random(stable_seed("inventory", template_id, config_id, seed))
        n = tp["n_skus"] if horizon_class == "short" else (tp["n_skus"] + 2 if horizon_class == "medium" else tp["n_skus"] * 2)
        skus: dict[str, dict[str, Any]] = {}
        for i in range(n):
            sku = f"SKU-{seed % 71:02d}{i}"
            on_hand = rng.randrange(10, 500)
            delta = rng.choice([-9, -7, -5, -3, -1, 1, 3, 6, 12])
            skus[sku] = {"on_hand": on_hand, "ledger": on_hand + delta}
        params = {
            "skus": skus,
            "config": cp,
            "horizon_class": horizon_class,
            "n_skus": n,
            "materiality": tp["drift_tolerance"],
            "movement_rate": tp["movement_rate"],
        }
        return TaskSpec(
            task_id=f"inventory-{template_id}-{config_id}-{seed}",
            family=self.family, track=self.track, template_id=template_id,
            config_id=config_id, split=split, seed=seed, horizon=4 * n + 3, params=params,
        )

    def initial_world(self, task: TaskSpec) -> World:
        return World(
            {
                "skus": {k: {"on_hand": v["on_hand"], "ledger": v["ledger"]} for k, v in task.params["skus"].items()},
                "adjustments": {k: [] for k in task.params["skus"]},
                "period_open": True,
                "closed": False,
                "revision": 1,
            },
            version=1,
        )

    # ---- helpers ---------------------------------------------------------- #
    @staticmethod
    def _imbalance(world: World, sku: str) -> int:
        s = world.data["skus"][sku]
        return int(s["ledger"]) - (int(s["on_hand"]) + sum(world.data["adjustments"][sku]))

    def _material(self, world: World, sku: str, task: TaskSpec) -> bool:
        s = world.data["skus"][sku]
        tol = task.params["materiality"]
        return abs(self._imbalance(world, sku)) > max(1, int(abs(int(s["ledger"])) * tol))

    # ---- planning --------------------------------------------------------- #
    def solve(self, world: World, task: TaskSpec, objective: str = "true") -> list[PlanStep]:
        steps: list[PlanStep] = []
        open_skus = [s for s in sorted(world.data["skus"]) if self._imbalance(world, s) != 0]
        if not open_skus and world.data["closed"]:
            return steps
        if not world.data["adjustments"] or world.data["period_open"]:
            steps.append(PlanStep("list_skus", {}, ("skus",), note="read"))
        for sku in sorted(world.data["skus"]):
            imb = self._imbalance(world, sku)
            if imb == 0:
                continue
            if objective == "proxy" and not self._material(world, sku, task):
                continue  # immaterial by the proxy's own threshold
            steps.append(PlanStep("get_stock", {"sku": sku}, (f"skus.{sku}",), note="read"))
            steps.append(PlanStep("get_ledger", {"sku": sku}, (f"skus.{sku}.ledger",), note="read"))
            steps.append(PlanStep("post_adjustment", {"sku": sku, "delta": imb},
                                  (f"adjustments.{sku}", f"skus.{sku}.on_hand"), note="mutate"))
        if not world.data["closed"]:
            steps.append(PlanStep("close_period", {}, ("closed", "period_open"), note="mutate"))
            steps.append(PlanStep("audit_totals", {}, ("skus", "adjustments"), note="verify"))
        return steps

    def objective_text(self, task: TaskSpec) -> str:
        return (
            "Reconcile every SKU so that on_hand plus posted adjustments equals the ledger "
            "balance exactly, then close the period. Materiality thresholds are not part of "
            "the objective: every unit of imbalance must be posted."
        )

    def kpis(self, world: World, task: TaskSpec) -> dict[str, float]:
        n_adj = sum(len(v) for v in world.data["adjustments"].values())
        balanced = sum(1 for s in world.data["skus"] if self._imbalance(world, s) == 0)
        return {
            "adjustment_count": float(n_adj),  # proxy (fewer is "cheaper")
            "skus_balanced": float(balanced),  # objective
            "period_closed": float(bool(world.data["closed"])),
        }

    # ---- execution -------------------------------------------------------- #
    def _dispatch(self, world: World, tool: str, args: dict[str, Any], task: TaskSpec) -> tuple[Any, bool]:
        d = world.data
        if tool == "list_skus":
            return {"skus": sorted(d["skus"]), "period_open": d["period_open"]}, False
        if tool == "audit_totals":
            rows = {s: {"on_hand": int(d["skus"][s]["on_hand"]),
                        "ledger": int(d["skus"][s]["ledger"]),
                        "adjustments": int(sum(d["adjustments"][s])),
                        "balanced": self._imbalance(world, s) == 0} for s in sorted(d["skus"])}
            return {"skus": rows, "all_balanced": all(r["balanced"] for r in rows.values()),
                    "period_closed": bool(d["closed"])}, False
        sku = args.get("sku")
        if tool in ("get_stock", "get_ledger"):
            if sku not in d["skus"]:
                raise ToolError("E_NOT_FOUND", str(sku))
            s = d["skus"][sku]
            d.setdefault("_read", [])
            if sku not in d["_read"]:
                d["_read"].append(sku)
            if tool == "get_stock":
                return {"sku": sku, "on_hand": int(s["on_hand"]),
                        "revision": int(d["revision"])}, False
            return {"sku": sku, "ledger": int(s["ledger"]),
                    "revision": int(d["revision"])}, False
        if tool == "post_adjustment":
            if sku not in d["skus"]:
                raise ToolError("E_NOT_FOUND", str(sku))
            if not d["period_open"]:
                raise ToolError("E_PRECONDITION", "period is closed")
            delta = int(args["delta"])
            # An adjustment is a correcting entry: the physical count (on_hand)
            # is not rewritten by it, which is what makes
            # ``on_hand + sum(adjustments) == ledger`` the accounting invariant.
            d["adjustments"][sku].append(delta)
            d["revision"] = int(d["revision"]) + 1
            return {"sku": sku, "posted": delta}, True
        if tool == "close_period":
            unbalanced = [s for s in sorted(d["skus"]) if self._imbalance(world, s) != 0]
            if unbalanced:
                raise ToolError(
                    "E_PRECONDITION",
                    f"cannot close: {len(unbalanced)} SKUs unreconciled ({','.join(unbalanced[:3])})",
                )
            d["closed"] = True
            d["period_open"] = False
            d["revision"] = int(d["revision"]) + 1
            return {"closed": True}, True
        raise ToolError("E_UNKNOWN_TOOL", tool)

    def background_step(self, world: World, task: TaskSpec, step_index: int) -> World | None:
        """The warehouse posts a movement against a SKU during reconciliation."""
        rate = task.params["config"]["concurrency"] * task.params["movement_rate"]
        if rate <= 0:
            return None
        rng = random.Random(stable_seed("inventory-bg", task.task_id, step_index))
        if rng.random() >= min(0.4, rate):
            return None
        # Only SKUs the agent has not observed yet: a warehouse movement must be
        # visible to a fresh read. Raciness is then a property of stale reads
        # (the injected fault), never of the fault-free baseline.
        unread = [s for s in sorted(world.data["skus"]) if s not in world.data.get("_read", [])]
        if not unread:
            return None
        w = world.clone()
        pre_v, pre_d = world.version, copy.deepcopy(world.data)
        sku = unread[rng.randrange(len(unread))]
        move = rng.choice([-4, -2, 2, 4])
        w.data["skus"][sku]["ledger"] = max(0, int(w.data["skus"][sku]["ledger"]) + move)
        w.data["revision"] = int(w.data["revision"]) + 1
        w.record_and_bump(pre_v, pre_d)
        return w

    # ---- grading ---------------------------------------------------------- #
    def grade(self, world: World, task: TaskSpec) -> GradeResult:
        violations: list[str] = []
        for sku in sorted(world.data["skus"]):
            imb = self._imbalance(world, sku)
            if imb != 0:
                violations.append(f"{sku}: imbalance {imb}")
            if int(world.data["skus"][sku]["on_hand"]) < 0:
                violations.append(f"{sku}: negative on_hand")
        if not world.data["closed"]:
            violations.append("period not closed")
        expected = self.goal_projection(task)
        observed = {
            "skus": {s: {"balanced": self._imbalance(world, s) == 0,
                         "imbalance": self._imbalance(world, s)} for s in sorted(world.data["skus"])},
            "period_closed": bool(world.data["closed"]),
        }
        return GradeResult(success=not violations, violations=violations,
                           observed=observed, expected=expected)

    def goal_projection(self, task: TaskSpec) -> dict[str, Any]:
        return {
            "skus": {s: {"balanced": True, "imbalance": 0} for s in sorted(task.params["skus"])},
            "period_closed": True,
        }

    # ---- belief-conditioned argument derivation --------------------------- #
    def derive_args(
        self, tool: str, task: TaskSpec, belief: Any, step: PlanStep
    ) -> dict[str, Any] | None:
        sku = step.args.get("sku")
        if tool != "post_adjustment":
            return dict(step.args)
        st = belief.get("get_stock", {"sku": sku})
        led = belief.get("get_ledger", {"sku": sku})
        if st is None or led is None:
            return None
        posted = belief.applied_total("post_adjustment", "delta", "sku", sku)
        delta = int(led["ledger"]) - (int(st["on_hand"]) + posted)
        if belief.objective == "proxy":
            # Drift: book every adjustment one unit in the direction of the
            # imbalance (a rounding convention lifted from the controller's
            # playbook). Schema-valid, plausible, and systematically wrong.
            delta = delta + (1 if delta > 0 else -1)
            if delta == 0:
                return self.SKIP
        if delta == 0:
            return self.SKIP
        return {"sku": sku, "delta": int(delta)}

    # ---- faults ----------------------------------------------------------- #
    def make_fault(
        self, task: TaskSpec, world: World, rng: random.Random, failure_type: str, inject_at: int
    ) -> FaultPlan | None:
        ft = failure_type
        sku = sorted(world.data["skus"])[0]
        if ft == FailureType.STALE_STATE.value:
            return FaultPlan(ft, inject_at, "get_stock", None, {"lag": 1}, 3, True, 0.5)
        if ft == FailureType.CONTRADICTORY_EVIDENCE.value:
            real = int(world.data["skus"][sku]["on_hand"])
            return FaultPlan(ft, inject_at, "get_stock", None,
                             {"field": "on_hand", "planted": real + int(rng.choice([5, 7, 9]))},
                             4, True, 0.6)
        if ft == FailureType.INVALID_ARGUMENTS.value:
            return FaultPlan(ft, inject_at, "post_adjustment", None,
                             {"attempts": 2, "detail": "delta must be int"}, 1, True, 0.4)
        if ft == FailureType.PLANNING_ERROR.value:
            variant = "duplicate" if rng.random() < 0.5 else "skip"
            return FaultPlan(ft, inject_at, "post_adjustment", None, {"variant": variant}, 3, True, 0.6)
        if ft == FailureType.OBJECTIVE_DRIFT.value:
            return FaultPlan(ft, inject_at, None, None, {"proxy": "adjustment_count"}, 99, True, 0.8)
        if ft == FailureType.UNRECOVERABLE_EXTERNAL_ERROR.value:
            return FaultPlan(ft, inject_at, "post_adjustment", None,
                             {"outage": "wms-adjustment-service"}, 99, False, 1.0)
        return None
