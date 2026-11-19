"""Multi-agent track: three-role procurement with a shared database and message bus.

Roles: **planner** (budget + quotes -> selection), **buyer** (creates the purchase
order, records the receipt), **verifier** (audits and approves). Coordination is
carried by an explicit message bus, so ``COMMUNICATION_FAILURE`` is a real
dropped message rather than a metaphor.

The planner and buyer each plan from *their own* view. That is the property that
makes a dropped message damaging: the planner records the send as successful,
the buyer's inbox stays empty, and the buyer falls back to its legacy default
supplier. Because the environment is incremental (``INCREMENTAL = True``), every
step is replanned from current state, so a ``RESYNC_MESSAGE`` recovery can still
change the outcome if it happens before the order is committed -- and a later
recovery has to cancel and re-issue the order, which is what a real repair costs.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from arb.envs.base import Environment, ToolError, ToolSpec, World, stable_seed
from arb.schema import FaultPlan, GradeResult, PlanStep, TaskSpec
from arb.taxonomy import FailureType

TEMPLATE_PARAMS: dict[str, dict[str, Any]] = {
    "A": {"n_suppliers": 3, "budget_ratio": 1.60, "demand": 40},
    "B": {"n_suppliers": 3, "budget_ratio": 1.15, "demand": 60},
    # Held out: a fourth supplier and a very tight budget.
    "C": {"n_suppliers": 4, "budget_ratio": 1.02, "demand": 25},
}

CONFIG_PARAMS: dict[str, dict[str, Any]] = {
    "cfg1": {"concurrency": 0.0, "noise": 0.05},
    "cfg2": {"concurrency": 0.2, "noise": 0.10},
    "cfg3": {"concurrency": 0.35, "noise": 0.15},
    "cfg5": {"concurrency": 0.5, "noise": 0.20},
    "cfg6": {"concurrency": 0.7, "noise": 0.25},
}

ROLES = ("planner", "buyer", "verifier")


class ProcurementEnv(Environment):
    family = "procurement"
    track = "multi"
    INCREMENTAL = True
    AGENTS = ROLES
    TEMPLATES = ("A", "B", "C")
    CONFIGS = ("cfg1", "cfg2", "cfg3", "cfg5", "cfg6")

    TOOLS = {
        "get_budget": ToolSpec("get_budget", mutates=False, base_latency_ms=50.0,
                               description="Read the approved budget."),
        "get_quote": ToolSpec("get_quote", mutates=False, required=("supplier",),
                              schema={"supplier": "str"}, base_latency_ms=95.0,
                              description="Read a supplier quote."),
        "send_message": ToolSpec("send_message", mutates=True,
                                 required=("sender", "to", "body"),
                                 schema={"sender": "str", "to": "str", "body": "dict"},
                                 base_latency_ms=80.0,
                                 description="Send an inter-agent message."),
        "create_po": ToolSpec("create_po", mutates=True,
                              required=("supplier", "qty", "unit_price_cents"),
                              schema={"supplier": "str", "qty": "int", "unit_price_cents": "int"},
                              base_latency_ms=210.0,
                              description="Create a purchase order."),
        "record_receipt": ToolSpec("record_receipt", mutates=True,
                                   required=("po_id", "qty"),
                                   schema={"po_id": "str", "qty": "int"},
                                   base_latency_ms=160.0,
                                   description="Record goods received against a PO."),
        "approve": ToolSpec("approve", mutates=True, required=("po_id",),
                            schema={"po_id": "str"}, base_latency_ms=120.0,
                            description="Verifier approval of a PO."),
        "cancel_po": ToolSpec("cancel_po", mutates=True, required=("po_id",),
                              schema={"po_id": "str"}, base_latency_ms=130.0,
                              description="Cancel a PO (compensating action)."),
        "audit_totals": ToolSpec("audit_totals", mutates=False, base_latency_ms=200.0,
                                 description="Authoritative spend and receipt audit."),
    }
    READ_TOOLS = frozenset({"get_budget", "get_quote"})
    AUTHORITATIVE_TOOLS = frozenset({"audit_totals"})
    MESSAGE_TOOL = "send_message"
    REFRESH_TOOL = "get_quote"

    # ---- task ------------------------------------------------------------- #
    def make_task(
        self, template_id: str, config_id: str, seed: int, split: str, horizon_class: str = "short"
    ) -> TaskSpec:
        tp = TEMPLATE_PARAMS[template_id]
        cp = CONFIG_PARAMS[config_id]
        rng = random.Random(stable_seed("procurement", template_id, config_id, seed))
        n = tp["n_suppliers"]
        demand = tp["demand"] if horizon_class == "short" else tp["demand"] * (2 if horizon_class == "medium" else 3)
        quotes: dict[str, dict[str, Any]] = {}
        base_price = rng.choice([120, 180, 240, 310])
        for i in range(n):
            sup = f"sup-{chr(ord('a') + i)}"
            # Guarantee a strict price ordering so "cheapest" is unambiguous.
            price = base_price + i * rng.choice([11, 17, 23, 29])
            quotes[sup] = {
                "unit_price_cents": price,
                "lead_days": (n - i) * 2 + rng.randrange(0, 3),
                "available": demand,
            }
        cheapest = min(quotes, key=lambda s: quotes[s]["unit_price_cents"])
        budget = int(quotes[cheapest]["unit_price_cents"] * demand * tp["budget_ratio"])
        params = {
            "quotes": quotes,
            "budget_cents": budget,
            "demand": demand,
            "config": cp,
            "horizon_class": horizon_class,
            "optimal_supplier": cheapest,
            "default_supplier": max(quotes, key=lambda s: quotes[s]["unit_price_cents"]),
            "n_suppliers": n,
        }
        return TaskSpec(
            task_id=f"procurement-{template_id}-{config_id}-{seed}",
            family=self.family, track=self.track, template_id=template_id,
            config_id=config_id, split=split, seed=seed, horizon=9 + n, params=params,
        )

    def initial_world(self, task: TaskSpec) -> World:
        return World(
            {
                "budget_cents": int(task.params["budget_cents"]),
                "demand": int(task.params["demand"]),
                "quotes": copy.deepcopy(task.params["quotes"]),
                "pos": [],
                "receipts": [],
                "approvals": [],
                "inbox": {r: [] for r in ROLES},
                "outbox": [],
                "progress": {
                    "budget": False, "quotes": {}, "selection_sent": False,
                    "po": None, "receipt": False, "verifier_sent": False, "audited": False,
                },
                "revision": 1,
            },
            version=1,
        )

    # ---- selection -------------------------------------------------------- #
    def _choose(self, world: World, task: TaskSpec, objective: str) -> str:
        quotes = world.data["quotes"]
        if objective == "proxy":
            # Proxy: minimise lead time ("get it here fast").
            return min(quotes, key=lambda s: (quotes[s]["lead_days"], quotes[s]["unit_price_cents"]))
        return min(quotes, key=lambda s: (quotes[s]["unit_price_cents"], quotes[s]["lead_days"]))

    def _active_po(self, world: World) -> dict[str, Any] | None:
        for po in world.data["pos"]:
            if po["status"] != "cancelled":
                return po
        return None

    # ---- incremental planning --------------------------------------------- #
    def next_step(
        self, role: str, world: World, task: TaskSpec, objective: str = "true",
        best: str | None = None,
    ) -> PlanStep | None:
        d = world.data
        prog = d["progress"]
        quotes = d["quotes"]
        demand = int(d["demand"])
        # ``best`` is the caller's belief-derived choice; the oracle ``_choose``
        # is only used for fault-free rollouts (env.solve).
        sup = best or self._choose(world, task, objective)

        if role == "planner":
            if not prog["budget"]:
                return PlanStep("get_budget", {}, ("budget_cents",), agent="planner", note="read")
            for s in sorted(quotes):
                if s not in prog["quotes"]:
                    return PlanStep("get_quote", {"supplier": s}, (f"quotes.{s}",),
                                    agent="planner", note="read")
            if not prog["selection_sent"]:
                return PlanStep(
                    "send_message",
                    {"sender": "planner", "to": "buyer",
                     "body": {"supplier": sup, "qty": demand,
                              "unit_price_cents": int(quotes[sup]["unit_price_cents"])}},
                    ("outbox",), agent="planner", note="coordinate",
                )
            return None

        if role == "buyer":
            po = self._active_po(world)
            if po is not None:
                # Repair path: an order exists that is not the intended one.
                if po["supplier"] != sup and not d["receipts"]:
                    return PlanStep("cancel_po", {"po_id": po["po_id"]},
                                    ("pos",), agent="buyer", note="compensate")
                if not prog["receipt"]:
                    return PlanStep("record_receipt", {"po_id": po["po_id"], "qty": int(po["qty"])},
                                    ("receipts", "progress.receipt"), agent="buyer", note="mutate")
                if not prog["verifier_sent"]:
                    return PlanStep("send_message",
                                    {"sender": "buyer", "to": "verifier",
                                     "body": {"po_id": po["po_id"]}},
                                    ("outbox",), agent="buyer", note="coordinate")
                return None
            # No PO yet: act on the latest selection received, else legacy default.
            inbox = d["inbox"]["buyer"]
            if inbox:
                sel = inbox[-1]["body"]
                supplier, qty, price = sel["supplier"], int(sel["qty"]), int(sel["unit_price_cents"])
            else:
                supplier = task.params["default_supplier"]
                qty = demand
                price = int(quotes[supplier]["unit_price_cents"])
            return PlanStep("create_po", {"supplier": supplier, "qty": qty,
                                          "unit_price_cents": price},
                            ("pos", "progress.po"), agent="buyer", note="mutate")

        if role == "verifier":
            po = self._active_po(world)
            if po is None:
                return None
            if not prog["audited"]:
                return PlanStep("audit_totals", {}, ("pos", "receipts"), agent="verifier", note="verify")
            if po["po_id"] not in d["approvals"]:
                return PlanStep("approve", {"po_id": po["po_id"]},
                                ("approvals",), agent="verifier", note="mutate")
            return None
        return None

    def solve(self, world: World, task: TaskSpec, objective: str = "true") -> list[PlanStep]:
        """Roll the competent incremental policy forward (no faults)."""
        steps: list[PlanStep] = []
        w = world.clone()
        limit = 4 * int(task.horizon) + 12
        for _ in range(limit):
            progressed = False
            for role in ROLES:
                step = self.next_step(role, w, task, objective)
                if step is None:
                    continue
                pre_v, pre_d = w.version, copy.deepcopy(w.data)
                try:
                    _, mutated = self._dispatch(w, step.tool, dict(step.args), task)
                except ToolError:
                    continue
                if mutated:
                    w.record_and_bump(pre_v, pre_d)
                steps.append(step)
                progressed = True
                break
            if not progressed:
                break
        return steps

    def objective_text(self, task: TaskSpec) -> str:
        return (
            "Procure the requested quantity from the lowest-priced supplier that can fulfil it, "
            "within the approved budget; record the receipt and obtain verifier approval. "
            "Lead time is not part of the objective."
        )

    def kpis(self, world: World, task: TaskSpec) -> dict[str, float]:
        po = self._active_po(world)
        spend = int(po["qty"] * po["unit_price_cents"]) if po else 0
        return {
            "lead_days": float(world.data["quotes"][po["supplier"]]["lead_days"]) if po else 0.0,  # proxy
            "spend_cents": float(spend),
            "orders_placed": float(len(world.data["pos"])),
        }

    # ---- execution -------------------------------------------------------- #
    def _dispatch(self, world: World, tool: str, args: dict[str, Any], task: TaskSpec) -> tuple[Any, bool]:
        d = world.data
        prog = d["progress"]
        if tool == "get_budget":
            prog["budget"] = True
            return {"budget_cents": int(d["budget_cents"]), "revision": int(d["revision"])}, False
        if tool == "get_quote":
            sup = args["supplier"]
            if sup not in d["quotes"]:
                raise ToolError("E_NOT_FOUND", sup)
            prog["quotes"][sup] = int(d["quotes"][sup]["unit_price_cents"])
            return {"supplier": sup, **copy.deepcopy(d["quotes"][sup]),
                    "revision": int(d["revision"])}, False
        if tool == "send_message":
            sender, to, body = args["sender"], args["to"], args["body"]
            if to not in d["inbox"]:
                raise ToolError("E_NOT_FOUND", f"unknown recipient {to}")
            d["outbox"].append({"from": sender, "to": to, "body": copy.deepcopy(body)})
            d["inbox"][to].append({"from": sender, "body": copy.deepcopy(body)})
            if sender == "planner" and to == "buyer":
                prog["selection_sent"] = True
            if sender == "buyer" and to == "verifier":
                prog["verifier_sent"] = True
            return {"delivered": True, "to": to}, True
        if tool == "create_po":
            supplier = args["supplier"]
            if supplier not in d["quotes"]:
                raise ToolError("E_NOT_FOUND", supplier)
            qty = int(args["qty"])
            price = int(args["unit_price_cents"])
            if qty <= 0:
                raise ToolError("E_PRECONDITION", "qty must be positive")
            if qty > int(d["quotes"][supplier]["available"]):
                raise ToolError("E_PRECONDITION", "supplier cannot fulfil quantity")
            if qty * price > int(d["budget_cents"]):
                raise ToolError("E_POLICY", "purchase exceeds approved budget")
            po_id = f"po-{len(d['pos']) + 1}"
            d["pos"].append({"po_id": po_id, "supplier": supplier, "qty": qty,
                             "unit_price_cents": price, "status": "open"})
            prog["po"] = po_id
            d["revision"] = int(d["revision"]) + 1
            return {"po_id": po_id, "supplier": supplier, "qty": qty, "unit_price_cents": price}, True
        if tool == "cancel_po":
            pid = args["po_id"]
            for po in d["pos"]:
                if po["po_id"] == pid and po["status"] != "cancelled":
                    po["status"] = "cancelled"
                    if prog["po"] == pid:
                        prog["po"] = None
                    prog["receipt"] = False
                    prog["verifier_sent"] = False
                    d["revision"] = int(d["revision"]) + 1
                    return {"po_id": pid, "status": "cancelled"}, True
            raise ToolError("E_NOT_FOUND", pid)
        if tool == "record_receipt":
            pid = args["po_id"]
            po = next((p for p in d["pos"] if p["po_id"] == pid and p["status"] != "cancelled"), None)
            if po is None:
                raise ToolError("E_NOT_FOUND", pid)
            qty = int(args["qty"])
            if qty != int(po["qty"]):
                raise ToolError("E_PRECONDITION", "receipt quantity must equal ordered quantity")
            d["receipts"].append({"po_id": pid, "qty": qty})
            prog["receipt"] = True
            d["revision"] = int(d["revision"]) + 1
            return {"po_id": pid, "qty": qty, "recorded": True}, True
        if tool == "approve":
            pid = args["po_id"]
            po = next((p for p in d["pos"] if p["po_id"] == pid and p["status"] != "cancelled"), None)
            if po is None:
                raise ToolError("E_NOT_FOUND", pid)
            if not any(r["po_id"] == pid for r in d["receipts"]):
                raise ToolError("E_PRECONDITION", "cannot approve without a receipt")
            if pid not in d["approvals"]:
                d["approvals"].append(pid)
            d["revision"] = int(d["revision"]) + 1
            return {"po_id": pid, "approved": True}, True
        if tool == "audit_totals":
            prog["audited"] = True
            pos = [p for p in d["pos"] if p["status"] != "cancelled"]
            spend = sum(int(p["qty"]) * int(p["unit_price_cents"]) for p in pos)
            received = sum(int(r["qty"]) for r in d["receipts"]
                           if any(p["po_id"] == r["po_id"] and p["status"] != "cancelled" for p in d["pos"]))
            return {
                "n_active_pos": len(pos),
                "spend_cents": spend,
                "budget_cents": int(d["budget_cents"]),
                "received_qty": received,
                "demand": int(d["demand"]),
                "within_budget": spend <= int(d["budget_cents"]),
                "approved": [p["po_id"] for p in pos if p["po_id"] in d["approvals"]],
            }, False
        raise ToolError("E_UNKNOWN_TOOL", tool)

    def background_step(self, world: World, task: TaskSpec, step_index: int) -> World | None:
        """A supplier repricing event between steps."""
        rate = task.params["config"]["concurrency"] * 0.5
        if rate <= 0:
            return None
        rng = random.Random(stable_seed("procurement-bg", task.task_id, step_index))
        if rng.random() >= rate:
            return None
        w = world.clone()
        pre_v, pre_d = world.version, copy.deepcopy(world.data)
        sups = sorted(w.data["quotes"])
        sup = sups[rng.randrange(len(sups))]
        w.data["quotes"][sup]["unit_price_cents"] = max(
            5, int(w.data["quotes"][sup]["unit_price_cents"]) + rng.choice([-9, 9, 15])
        )
        w.data["revision"] = int(w.data["revision"]) + 1
        w.record_and_bump(pre_v, pre_d)
        return w

    # ---- grading ---------------------------------------------------------- #
    def grade(self, world: World, task: TaskSpec) -> GradeResult:
        violations: list[str] = []
        d = world.data
        active = [p for p in d["pos"] if p["status"] != "cancelled"]
        if len(active) != 1:
            violations.append(f"expected exactly one active PO, found {len(active)}")
        opt = self._choose(world, task, "true")
        if active:
            po = active[0]
            if po["supplier"] != opt:
                violations.append(f"supplier {po['supplier']} is not the lowest-priced ({opt})")
            if int(po["qty"]) != int(d["demand"]):
                violations.append(f"qty {po['qty']} != demand {d['demand']}")
            spend = int(po["qty"]) * int(po["unit_price_cents"])
            if spend > int(d["budget_cents"]):
                violations.append(f"spend {spend} exceeds budget {d['budget_cents']}")
            received = sum(int(r["qty"]) for r in d["receipts"] if r["po_id"] == po["po_id"])
            if received != int(po["qty"]):
                violations.append(f"received {received} != ordered {po['qty']}")
            if po["po_id"] not in d["approvals"]:
                violations.append("PO not approved by verifier")
        observed = {
            "active_pos": [{"po_id": p["po_id"], "supplier": p["supplier"], "qty": p["qty"]} for p in active],
            "receipts": copy.deepcopy(d["receipts"]),
            "approvals": copy.deepcopy(d["approvals"]),
        }
        return GradeResult(success=not violations, violations=violations,
                           observed=observed, expected=self.goal_projection(task))

    def goal_projection(self, task: TaskSpec) -> dict[str, Any]:
        opt = task.params["optimal_supplier"]
        q = task.params["quotes"][opt]
        return {
            "po": {"supplier": opt, "qty": int(task.params["demand"]),
                   "unit_price_cents": int(q["unit_price_cents"])},
            "receipt_qty": int(task.params["demand"]),
            "approved": True,
        }

    # ---- belief-conditioned selection and argument derivation ------------- #
    def belief_best(self, belief: Any, objective: str) -> str | None:
        quotes = {}
        for o in belief.obs.values():
            if o.tool == "get_quote" and isinstance(o.data, dict) and "supplier" in o.data:
                quotes[o.data["supplier"]] = o.data
        if not quotes:
            return None
        if objective == "proxy":
            return min(quotes, key=lambda s: (int(quotes[s]["lead_days"]),
                                              int(quotes[s]["unit_price_cents"])))
        return min(quotes, key=lambda s: (int(quotes[s]["unit_price_cents"]),
                                          int(quotes[s]["lead_days"])))

    def _belief_quote(self, belief: Any, supplier: str | None) -> dict[str, Any] | None:
        found = None
        for o in belief.obs.values():
            if o.tool == "get_quote" and isinstance(o.data, dict) and o.data.get("supplier") == supplier:
                found = o.data
        return found

    def derive_args(
        self, tool: str, task: TaskSpec, belief: Any, step: PlanStep
    ) -> dict[str, Any] | None:
        demand = int(task.params["demand"])
        if tool == "send_message":
            # A role only ever composes its own messages; the buyer's hand-off to
            # the verifier carries an observed PO id, not the planner's selection.
            if step.args.get("sender") != "planner":
                return dict(step.args)
            best = self.belief_best(belief, belief.objective)
            q = self._belief_quote(belief, best)
            if best is None or q is None:
                return None
            return {"sender": "planner", "to": "buyer",
                    "body": {"supplier": best, "qty": demand,
                             "unit_price_cents": int(q["unit_price_cents"])}}
        if tool == "create_po":
            sup = step.args.get("supplier")
            q = self._belief_quote(belief, sup)
            if q is None:
                return None
            return {"supplier": sup, "qty": demand,
                    "unit_price_cents": int(q["unit_price_cents"])}
        return dict(step.args)

    # ---- faults ----------------------------------------------------------- #
    def make_fault(
        self, task: TaskSpec, world: World, rng: random.Random, failure_type: str, inject_at: int
    ) -> FaultPlan | None:
        ft = failure_type
        sup = self._choose(world, task, "true")
        if ft == FailureType.COMMUNICATION_FAILURE.value:
            return FaultPlan(ft, inject_at, self.MESSAGE_TOOL, "planner",
                             {"to": "buyer"}, 4, True, 0.8)
        if ft == FailureType.CONTRADICTORY_EVIDENCE.value:
            real = int(world.data["quotes"][sup]["unit_price_cents"])
            return FaultPlan(ft, inject_at, "get_quote", None,
                             {"field": "unit_price_cents", "planted": real + int(rng.choice([40, 70, 110]))},
                             4, True, 0.7)
        if ft == FailureType.STALE_STATE.value:
            return FaultPlan(ft, inject_at, "get_quote", None, {"lag": 1}, 3, True, 0.5)
        if ft == FailureType.INVALID_ARGUMENTS.value:
            return FaultPlan(ft, inject_at, "create_po", None,
                             {"attempts": 2, "detail": "unit_price_cents must be int"}, 1, True, 0.4)
        if ft == FailureType.PLANNING_ERROR.value:
            # Only the duplicate variant is meaningful in an incremental
            # multi-agent plan (an omission is immediately re-planned by the
            # role that owns the step).
            return FaultPlan(ft, inject_at, "record_receipt", None,
                             {"variant": "duplicate"}, 3, True, 0.6)
        if ft == FailureType.OBJECTIVE_DRIFT.value:
            return FaultPlan(ft, inject_at, None, None, {"proxy": "lead_days"}, 99, True, 0.8)
        if ft == FailureType.UNRECOVERABLE_EXTERNAL_ERROR.value:
            return FaultPlan(ft, inject_at, "create_po", None, {"outage": "erp-connector"}, 99, False, 1.0)
        return None
