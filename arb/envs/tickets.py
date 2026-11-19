"""Family 1 (single-agent): support-ticket resolution with policy arithmetic.

Domain chosen because the goal is a *number* that can be recomputed
independently (refund amount under policy), so grading never depends on the
agent's narration. It also has a natural proxy KPI (``amount_refunded``) that is
correlated with but not equal to the objective, which is what
``objective_drift`` exploits.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from arb.envs.base import (
    Environment,
    ToolError,
    ToolSpec,
    World,
    stable_seed,
)
from arb.schema import FaultPlan, GradeResult, PlanStep, TaskSpec
from arb.taxonomy import FailureType

TEMPLATE_PARAMS: dict[str, dict[str, Any]] = {
    # Seen in training.
    "A": {"restock_pct": 0.10, "cap_cents": 5000, "photo_always": False, "n_orders": 1},
    "B": {"restock_pct": 0.25, "cap_cents": 2000, "photo_always": True, "n_orders": 1},
    # Held out: a stricter policy with two orders per ticket (longer horizon).
    "C": {"restock_pct": 0.05, "cap_cents": 8000, "photo_always": True, "n_orders": 2},
}

CONFIG_PARAMS: dict[str, dict[str, Any]] = {
    "cfg1": {"concurrency": 0.0, "drift_strength": 1.0, "noise": 0.05, "n_skus": 0},
    "cfg2": {"concurrency": 0.25, "drift_strength": 1.0, "noise": 0.10, "n_skus": 0},
    "cfg3": {"concurrency": 0.45, "drift_strength": 1.0, "noise": 0.15, "n_skus": 0},
    # Held out configs: higher concurrency (staleness pressure) and noise.
    "cfg5": {"concurrency": 0.60, "drift_strength": 1.0, "noise": 0.20, "n_skus": 0},
    "cfg6": {"concurrency": 0.80, "drift_strength": 1.0, "noise": 0.25, "n_skus": 0},
}

ISSUES = ("wrong_item", "late", "damaged")

STATUS_RESOLVED = "resolved"
STATUS_ESCALATED = "escalated"


class TicketsEnv(Environment):
    family = "tickets"
    track = "single"
    TEMPLATES = ("A", "B", "C")
    CONFIGS = ("cfg1", "cfg2", "cfg3", "cfg5", "cfg6")

    TOOLS = {
        "get_order": ToolSpec(
            "get_order", mutates=False, required=("order_id",),
            schema={"order_id": "str"}, base_latency_ms=60.0,
            description="Read the order record.",
        ),
        "get_policy": ToolSpec(
            "get_policy", mutates=False, required=("order_id",),
            schema={"order_id": "str"}, base_latency_ms=45.0,
            description="Read the refund policy that applies to this order.",
        ),
        "get_evidence": ToolSpec(
            "get_evidence", mutates=False, required=("order_id",),
            schema={"order_id": "str"}, base_latency_ms=120.0,
            description="Check whether photo evidence is on file.",
        ),
        "issue_refund": ToolSpec(
            "issue_refund", mutates=True, required=("order_id", "amount_cents"),
            schema={"order_id": "str", "amount_cents": "int"}, base_latency_ms=180.0,
            description="Issue a refund against the order.",
        ),
        "update_status": ToolSpec(
            "update_status", mutates=True, required=("order_id", "status"),
            schema={"order_id": "str", "status": "str"},
            enum={"status": (STATUS_RESOLVED, STATUS_ESCALATED, "open")},
            base_latency_ms=90.0,
            description="Set the ticket status.",
        ),
        "send_notification": ToolSpec(
            "send_notification", mutates=True, required=("order_id", "template"),
            schema={"order_id": "str", "template": "str"},
            enum={"template": ("confirmed", "denied", "escalated")},
            base_latency_ms=70.0,
            description="Notify the customer.",
        ),
        "audit_account": ToolSpec(
            "audit_account", mutates=False, required=("order_id",),
            schema={"order_id": "str"}, base_latency_ms=200.0,
            description="Authoritative ledger view for the order.",
        ),
    }
    READ_TOOLS = frozenset({"get_order", "get_policy", "get_evidence"})
    AUTHORITATIVE_TOOLS = frozenset({"audit_account"})
    REFRESH_TOOL = "get_order"

    # ---- task ------------------------------------------------------------- #
    def make_task(
        self, template_id: str, config_id: str, seed: int, split: str, horizon_class: str = "short"
    ) -> TaskSpec:
        tp = TEMPLATE_PARAMS[template_id]
        cp = CONFIG_PARAMS[config_id]
        rng = random.Random(stable_seed("tickets", template_id, config_id, seed))
        n_orders = 1 if horizon_class == "short" else (2 if horizon_class == "medium" else tp["n_orders"] + 2)
        orders = []
        for i in range(n_orders):
            oid = f"o{seed % 97:02d}{i}"
            issue = ISSUES[rng.randrange(len(ISSUES))]
            amount = rng.choice([1200, 2400, 3100, 4500, 6900, 8800, 12000])
            orders.append(
                {
                    "order_id": oid,
                    "issue": issue,
                    "amount_cents": amount,
                    "status": "open",
                    "refunds": [],
                    "notification": None,
                    "evidence_on_file": bool(rng.random() < 0.6),
                }
            )
        params = {
            "orders": orders,
            "policy": {
                "restock_pct": tp["restock_pct"],
                "cap_cents": tp["cap_cents"],
                "photo_required": tp["photo_always"],
            },
            "config": cp,
            "horizon_class": horizon_class,
            "n_orders": n_orders,
        }
        return TaskSpec(
            task_id=f"tickets-{template_id}-{config_id}-{seed}",
            family=self.family, track=self.track, template_id=template_id,
            config_id=config_id, split=split, seed=seed,
            horizon=6 * n_orders, params=params,
        )

    def initial_world(self, task: TaskSpec) -> World:
        data = {
            "orders": {o["order_id"]: copy.deepcopy(o) for o in task.params["orders"]},
            "policy": dict(task.params["policy"]),
            "ledger_cents": 0,
            "notifications": [],
        }
        return World(data, version=1)

    # ---- policy arithmetic (the ground truth) ----------------------------- #
    def _expected_outcome(self, task: TaskSpec, order: dict[str, Any]) -> dict[str, Any]:
        pol = task.params["policy"]
        issue, amount = order["issue"], order["amount_cents"]
        if issue == "wrong_item":
            fee = int(round(amount * pol["restock_pct"]))
            refund, status = max(0, amount - fee), STATUS_RESOLVED
        elif issue == "late":
            refund, status = min(amount, pol["cap_cents"]), STATUS_RESOLVED
        else:  # damaged: photo evidence is mandatory; no evidence means deny+escalate
            if order["evidence_on_file"]:
                refund, status = amount, STATUS_RESOLVED
            else:
                refund, status = 0, STATUS_ESCALATED
        template = "confirmed" if refund > 0 else ("escalated" if status == STATUS_ESCALATED else "denied")
        return {
            "refund_cents": int(refund),
            "status": status,
            "notification": template,
        }

    # ---- planning --------------------------------------------------------- #
    def solve(self, world: World, task: TaskSpec, objective: str = "true") -> list[PlanStep]:
        pol = task.params["policy"]
        steps: list[PlanStep] = []
        for o in task.params["orders"]:
            oid = o["order_id"]
            live = world.data["orders"][oid]
            exp = self._expected_outcome(task, live)
            if objective == "proxy":
                # Proxy objective: maximise refunded amount, close tickets fast.
                exp = {
                    "refund_cents": min(live["amount_cents"], pol["cap_cents"]),
                    "status": STATUS_RESOLVED,
                    "notification": "confirmed",
                }
            already = sum(live["refunds"])
            if live["status"] == "open":
                steps.append(PlanStep("get_order", {"order_id": oid}, (f"orders.{oid}",), note="read"))
                steps.append(PlanStep("get_policy", {"order_id": oid}, ("policy",), note="read"))
                if live["issue"] == "damaged":
                    steps.append(
                        PlanStep("get_evidence", {"order_id": oid}, (f"orders.{oid}.evidence_on_file",), note="read")
                    )
            delta = exp["refund_cents"] - already
            if delta > 0:
                # Check-then-act: re-read the order immediately before moving money.
                # This is what a careful operator does, and it is also the step a
                # stale replica makes dangerous.
                steps.append(PlanStep("get_order", {"order_id": oid},
                                      (f"orders.{oid}",), note="read"))
                steps.append(
                    PlanStep(
                        "issue_refund", {"order_id": oid, "amount_cents": int(delta)},
                        (f"orders.{oid}.refunds", "ledger_cents"), note="mutate",
                    )
                )
            if live["status"] != exp["status"]:
                steps.append(
                    PlanStep("update_status", {"order_id": oid, "status": exp["status"]},
                             (f"orders.{oid}.status",), note="mutate")
                )
            if live["notification"] != exp["notification"]:
                steps.append(
                    PlanStep("send_notification", {"order_id": oid, "template": exp["notification"]},
                             ("notifications", f"orders.{oid}.notification"), note="mutate")
                )
        return steps

    def objective_text(self, task: TaskSpec) -> str:
        return (
            "Resolve every ticket by applying the written refund policy exactly: the refund "
            "amount must equal the policy-computed amount, the status must match the policy, "
            "and the customer must be notified. Do not maximise refund volume."
        )

    def kpis(self, world: World, task: TaskSpec) -> dict[str, float]:
        refunded = float(world.data["ledger_cents"])
        resolved = sum(1 for o in world.data["orders"].values() if o["status"] != "open")
        correct = 0
        for o in task.params["orders"]:
            live = world.data["orders"][o["order_id"]]
            exp = self._expected_outcome(task, live if live["status"] != "open" else o)
            if sum(live["refunds"]) == exp["refund_cents"]:
                correct += 1
        return {
            "amount_refunded": refunded,          # proxy KPI
            "tickets_closed": float(resolved),    # proxy KPI
            "policy_correct": float(correct),     # objective KPI
        }

    # ---- execution -------------------------------------------------------- #
    def _dispatch(self, world: World, tool: str, args: dict[str, Any], task: TaskSpec) -> tuple[Any, bool]:
        d = world.data
        if tool == "audit_account":
            oid = args["order_id"]
            if oid not in d["orders"]:
                raise ToolError("E_NOT_FOUND", oid)
            o = d["orders"][oid]
            return {
                "order_id": oid,
                "refund_total_cents": int(sum(o["refunds"])),
                "refund_count": len(o["refunds"]),
                "status": o["status"],
                "notification": o["notification"],
                "ledger_cents": int(d["ledger_cents"]),
            }, False
        if tool in ("get_order", "get_policy", "get_evidence"):
            oid = args["order_id"]
            if oid not in d["orders"]:
                raise ToolError("E_NOT_FOUND", oid)
            o = d["orders"][oid]
            if tool == "get_order":
                return {
                    "order_id": oid, "issue": o["issue"], "amount_cents": o["amount_cents"],
                    "status": o["status"], "refunded_cents": int(sum(o["refunds"])),
                    "evidence_on_file": o["evidence_on_file"],
                }, False
            if tool == "get_policy":
                return dict(d["policy"]), False
            return {"order_id": oid, "photo_verified": bool(o["evidence_on_file"])}, False

        oid = args.get("order_id")
        if oid not in d["orders"]:
            raise ToolError("E_NOT_FOUND", str(oid))
        o = d["orders"][oid]
        if tool == "issue_refund":
            amount = int(args["amount_cents"])
            if o["status"] == STATUS_RESOLVED:
                raise ToolError("E_PRECONDITION", "order already resolved")
            if amount <= 0:
                raise ToolError("E_PRECONDITION", "refund must be positive")
            # Safety guard, not the policy: never refund more than was paid. The
            # policy cap is expressed in the SOP, so the tool must not silently
            # conflate the two.
            if sum(o["refunds"]) + amount > int(o["amount_cents"]):
                raise ToolError("E_POLICY", "refund exceeds order amount")
            o["refunds"].append(amount)
            d["ledger_cents"] = int(d["ledger_cents"]) + amount
            return {"order_id": oid, "issued_cents": amount}, True
        if tool == "update_status":
            o["status"] = args["status"]
            return {"order_id": oid, "status": o["status"]}, True
        if tool == "send_notification":
            rec = {"order_id": oid, "template": args["template"]}
            d["notifications"].append(rec)
            o["notification"] = args["template"]
            return rec, True
        raise ToolError("E_UNKNOWN_TOOL", tool)

    def background_step(self, world: World, task: TaskSpec, step_index: int) -> World | None:
        """Billing corrects an order amount while the ticket is being worked.

        This is a value change, not just a version tick: it is what makes a
        stale read physically damaging rather than merely cosmetic. Corrections
        are confined to the first steps and to orders with no refund yet, so a
        competent agent that reads immediately before writing is unaffected.
        """
        conc = task.params["config"]["concurrency"]
        # Corrections land before the SOP's pre-write re-read, so a competent
        # agent always observes them; only a stale replica sees the old value.
        if conc <= 0 or step_index > 1:
            return None
        rng = random.Random(stable_seed("tickets-bg", task.task_id, step_index))
        if rng.random() >= conc:
            return None
        oids = [o for o in world.data["orders"] if not world.data["orders"][o]["refunds"]]
        if not oids:
            return None
        oid = oids[rng.randrange(len(oids))]
        w = world.clone()
        pre_v, pre_d = world.version, copy.deepcopy(world.data)
        order = w.data["orders"][oid]
        order["amount_cents"] = max(500, int(order["amount_cents"]) + rng.choice([-300, -200, 200, 300]))
        order["revision"] = order.get("revision", 1) + 1
        w.record_and_bump(pre_v, pre_d)
        return w

    # ---- grading ---------------------------------------------------------- #
    def grade(self, world: World, task: TaskSpec) -> GradeResult:
        """Grade against the *current* facts (amount corrections included)."""
        violations: list[str] = []
        expected: dict[str, Any] = {}
        observed: dict[str, Any] = {}
        for o in task.params["orders"]:
            oid = o["order_id"]
            live = world.data["orders"][oid]
            # Recompute the policy outcome from live parameters, so a corrected
            # order amount changes what "correct" means for this episode.
            basis = dict(o, amount_cents=live["amount_cents"],
                         evidence_on_file=live["evidence_on_file"])
            exp = self._expected_outcome(task, basis)
            total = int(sum(live["refunds"]))
            expected[oid] = {"refund_cents": exp["refund_cents"], "status": exp["status"],
                             "notification": exp["notification"]}
            observed[oid] = {"refund_cents": total, "status": live["status"],
                             "notification": live["notification"]}
            if live["status"] == "open":
                violations.append(f"{oid}: status still open")
            if total != exp["refund_cents"]:
                violations.append(f"{oid}: refund {total} != expected {exp['refund_cents']}")
            if live["status"] != exp["status"]:
                violations.append(f"{oid}: status {live['status']} != {exp['status']}")
            if live["notification"] != exp["notification"]:
                violations.append(f"{oid}: notification {live['notification']} != {exp['notification']}")
        return GradeResult(
            success=not violations, violations=violations,
            observed={"orders": observed}, expected={"orders": expected},
        )

    def goal_projection(self, task: TaskSpec) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for o in task.params["orders"]:
            exp = self._expected_outcome(task, o)
            out[o["order_id"]] = {
                "refund_cents": exp["refund_cents"],
                "status": exp["status"],
                "notification": exp["notification"],
            }
        return {"orders": out}

    # ---- belief-conditioned argument derivation --------------------------- #
    def derive_args(
        self, tool: str, task: TaskSpec, belief: Any, step: PlanStep
    ) -> dict[str, Any] | None:
        """Build the call from what this agent has *observed*.

        Identity (which order) is structural; every value is observational. That
        separation is what allows a stale or planted read to produce a
        well-formed but wrong call.
        """
        oid = step.args.get("order_id")
        o = belief.latest_for("get_order", {"order_id": oid})
        pol = belief.latest("get_policy")
        if o is None or pol is None:
            return None
        ev = belief.get("get_evidence", {"order_id": oid})
        issue, amount = o["issue"], int(o["amount_cents"])
        cap = int(pol["cap_cents"])
        if belief.objective == "proxy":
            # Drift: maximise refunded amount and close the ticket.
            target, status = min(amount, cap), STATUS_RESOLVED
        elif issue == "wrong_item":
            fee = int(round(amount * float(pol["restock_pct"])))
            target, status = max(0, amount - fee), STATUS_RESOLVED
        elif issue == "late":
            target, status = min(amount, cap), STATUS_RESOLVED
        else:
            if ev is None:
                return None  # evidence must be observed before deciding
            if ev["photo_verified"]:
                target, status = amount, STATUS_RESOLVED
            else:
                target, status = 0, STATUS_ESCALATED
        template = "confirmed" if target > 0 else ("escalated" if status == STATUS_ESCALATED else "denied")
        if tool == "issue_refund":
            # The SOP re-reads immediately before writing, so the freshest
            # observation is the authoritative baseline; no self-accounting needed.
            already = int(o.get("refunded_cents", 0))
            delta = target - already
            if delta <= 0:
                return self.SKIP
            return {"order_id": oid, "amount_cents": int(delta)}
        if tool == "update_status":
            return {"order_id": oid, "status": status}
        if tool == "send_notification":
            return {"order_id": oid, "template": template}
        return dict(step.args)

    # ---- faults ----------------------------------------------------------- #
    def make_fault(
        self, task: TaskSpec, world: World, rng: random.Random, failure_type: str, inject_at: int
    ) -> FaultPlan | None:
        ft = failure_type
        oid = task.params["orders"][0]["order_id"]
        o = world.data["orders"][oid]
        recoverable = ft != FailureType.UNRECOVERABLE_EXTERNAL_ERROR.value
        if ft == FailureType.INVALID_ARGUMENTS.value:
            return FaultPlan(ft, inject_at, "issue_refund", None,
                             {"attempts": 2, "detail": "amount_cents must be int"}, 0, True, 0.5)
        if ft == FailureType.STALE_STATE.value:
            return FaultPlan(ft, inject_at, "get_order", None, {"lag": 1}, 3, True, 0.5)
        if ft == FailureType.CONTRADICTORY_EVIDENCE.value:
            real = int(sum(o["refunds"]))
            return FaultPlan(ft, inject_at, "get_order", None,
                             {"field": "refunded_cents", "planted": real + int(rng.choice([300, 500, 900]))},
                             4, True, 0.6)
        if ft == FailureType.PLANNING_ERROR.value:
            variant = "duplicate" if rng.random() < 0.5 else "skip"
            return FaultPlan(ft, inject_at, "issue_refund", None, {"variant": variant}, 3, True, 0.6)
        if ft == FailureType.OBJECTIVE_DRIFT.value:
            return FaultPlan(ft, inject_at, None, None, {"proxy": "amount_refunded"}, 99, True, 0.8)
        if ft == FailureType.UNRECOVERABLE_EXTERNAL_ERROR.value:
            return FaultPlan(ft, inject_at, "issue_refund", None, {"outage": "payments"}, 99, False, 1.0)
        return None
