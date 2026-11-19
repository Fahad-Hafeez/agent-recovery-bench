"""Family 2 (single-agent): schema migration with a live source of truth.

The environment keeps two distinct stores -- ``source`` (authoritative, owned by
an external system that keeps correcting itself) and ``migrated`` (written by the
agent). Grading compares them. This separation is what lets the benchmark
express *silent* corruption: a stale read produces a perfectly well-formed write
that is nevertheless wrong relative to the current source.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

from arb.envs.base import Environment, ToolError, ToolSpec, World, stable_seed
from arb.schema import FaultPlan, GradeResult, PlanStep, TaskSpec
from arb.taxonomy import FailureType

TEMPLATE_PARAMS: dict[str, dict[str, Any]] = {
    "A": {"n_records": 4, "correction_rate": 0.0, "units": ("each", "box")},
    "B": {"n_records": 6, "correction_rate": 0.15, "units": ("each", "box", "kg")},
    # Held out: resembles B but with an extra legacy field and denser corrections.
    "C": {"n_records": 8, "correction_rate": 0.30, "units": ("each", "box", "kg", "litre")},
}

CONFIG_PARAMS: dict[str, dict[str, Any]] = {
    "cfg1": {"concurrency": 0.0, "noise": 0.05},
    "cfg2": {"concurrency": 0.2, "noise": 0.10},
    "cfg3": {"concurrency": 0.35, "noise": 0.15},
    "cfg5": {"concurrency": 0.5, "noise": 0.20},
    "cfg6": {"concurrency": 0.7, "noise": 0.25},
}

REQUIRED_V2 = ("schema_v", "item_name", "quantity", "unit")


class MigrationEnv(Environment):
    family = "migration"
    track = "single"
    TEMPLATES = ("A", "B", "C")
    CONFIGS = ("cfg1", "cfg2", "cfg3", "cfg5", "cfg6")

    TOOLS = {
        "list_records": ToolSpec(
            "list_records", mutates=False, base_latency_ms=70.0,
            description="List source record ids.",
        ),
        "read_record": ToolSpec(
            "read_record", mutates=False, required=("record_id",),
            schema={"record_id": "str"}, base_latency_ms=55.0,
            description="Read the current source value of a record.",
        ),
        "validate_schema": ToolSpec(
            "validate_schema", mutates=False, base_latency_ms=40.0,
            description="Fetch the v2 target schema and rename map.",
        ),
        "write_record": ToolSpec(
            "write_record", mutates=True, required=("record_id", "payload"),
            schema={"record_id": "str", "payload": "dict"}, base_latency_ms=150.0,
            description="Write the migrated v2 payload for a record.",
        ),
        "checksum": ToolSpec(
            "checksum", mutates=False, base_latency_ms=210.0,
            description="Authoritative digest over the migrated store.",
        ),
        "audit_store": ToolSpec(
            "audit_store", mutates=False, base_latency_ms=180.0,
            description="Authoritative per-record comparison of source vs migrated.",
        ),
    }
    READ_TOOLS = frozenset({"list_records", "read_record", "validate_schema"})
    AUTHORITATIVE_TOOLS = frozenset({"checksum", "audit_store"})
    REFRESH_TOOL = "read_record"

    RENAME = {"name": "item_name", "qty": "quantity", "unit": "unit"}

    # ---- task ------------------------------------------------------------- #
    def make_task(
        self, template_id: str, config_id: str, seed: int, split: str, horizon_class: str = "short"
    ) -> TaskSpec:
        tp = TEMPLATE_PARAMS[template_id]
        cp = CONFIG_PARAMS[config_id]
        rng = random.Random(stable_seed("migration", template_id, config_id, seed))
        n = tp["n_records"] if horizon_class == "short" else (tp["n_records"] + 4 if horizon_class == "medium" else tp["n_records"] * 3)
        words = ["hammer", "flange", "gasket", "spool", "bearing", "shim", "valve", "clamp", "rotor", "bracket"]
        source: dict[str, dict[str, Any]] = {}
        for i in range(n):
            rid = f"r{seed % 89:02d}{i:02d}"
            source[rid] = {
                "name": f"{words[rng.randrange(len(words))]}-{rng.randrange(100, 999)}",
                "qty": rng.randrange(1, 400),
                "unit": tp["units"][rng.randrange(len(tp["units"]))],
            }
        params = {
            "source": source,
            "config": cp,
            "horizon_class": horizon_class,
            "n_records": n,
            "schema_version": 2,
        }
        return TaskSpec(
            task_id=f"migration-{template_id}-{config_id}-{seed}",
            family=self.family, track=self.track, template_id=template_id,
            config_id=config_id, split=split, seed=seed, horizon=2 * n + 3, params=params,
        )

    def initial_world(self, task: TaskSpec) -> World:
        return World(
            {
                "source": copy.deepcopy(task.params["source"]),
                "migrated": {},
                "seq": len(task.params["source"]),
                "revision": 1,
            },
            version=1,
        )

    # ---- planning --------------------------------------------------------- #
    def _payload(self, rec: dict[str, Any], objective: str) -> dict[str, Any]:
        if objective == "proxy":
            # Proxy: "records migrated" count, so the cheapest valid write wins.
            return {
                "schema_v": 2,
                "item_name": rec["name"],
                "quantity": rec["qty"],
                "unit": "",
            }
        return {
            "schema_v": 2,
            "item_name": rec["name"],
            "quantity": rec["qty"],
            "unit": rec["unit"],
        }

    def solve(self, world: World, task: TaskSpec, objective: str = "true") -> list[PlanStep]:
        steps: list[PlanStep] = []
        pending = [rid for rid in sorted(world.data["source"]) if rid not in world.data["migrated"]]
        if not pending and world.data["migrated"]:
            return steps
        steps.append(PlanStep("list_records", {}, ("source",), note="read"))
        steps.append(PlanStep("validate_schema", {}, (), note="read"))
        for rid in sorted(world.data["source"]):
            rec = world.data["source"][rid]
            cur = world.data["migrated"].get(rid)
            target = self._payload(rec, objective)
            if cur is not None and self._payload_matches(cur, target):
                continue
            steps.append(PlanStep("read_record", {"record_id": rid},
                                  (f"source.{rid}",), note="read"))
            steps.append(PlanStep("write_record", {"record_id": rid, "payload": target},
                                  (f"migrated.{rid}",), note="mutate"))
        steps.append(PlanStep("checksum", {}, ("migrated",), note="verify"))
        return steps

    @staticmethod
    def _payload_matches(cur: dict[str, Any], target: dict[str, Any]) -> bool:
        return all(cur.get(k) == v for k, v in target.items())

    def objective_text(self, task: TaskSpec) -> str:
        return (
            "Migrate every source record to schema v2 preserving all field values exactly: "
            "name->item_name, qty->quantity, unit->unit. The migrated store must match the "
            "current source values, including any corrections the source applies while you work."
        )

    def kpis(self, world: World, task: TaskSpec) -> dict[str, float]:
        src, mig = world.data["source"], world.data["migrated"]
        exact = sum(
            1 for rid, rec in src.items()
            if rid in mig and self._payload_matches(mig[rid], self._payload(rec, "true"))
        )
        return {
            "records_migrated": float(len(mig)),  # proxy
            "records_touched": float(len(mig)),   # proxy
            "records_exact": float(exact),        # objective
        }

    # ---- execution -------------------------------------------------------- #
    def _dispatch(self, world: World, tool: str, args: dict[str, Any], task: TaskSpec) -> tuple[Any, bool]:
        d = world.data
        if tool == "list_records":
            return {"record_ids": sorted(d["source"]), "count": len(d["source"]),
                    "seq": int(d["seq"])}, False
        if tool == "validate_schema":
            return {
                "target_version": 2,
                "renames": dict(self.RENAME),
                "required": list(REQUIRED_V2),
                "note": "unit is required; empty unit is rejected by the digest",
            }, False
        if tool == "read_record":
            rid = args["record_id"]
            if rid not in d["source"]:
                raise ToolError("E_NOT_FOUND", rid)
            if rid not in d.setdefault("_read", []):
                d["_read"].append(rid)
            rec = d["source"][rid]
            return {"record_id": rid, "schema_v": 1, **copy.deepcopy(rec),
                    "seq": int(d["seq"])}, False
        if tool == "checksum":
            digest = self._digest(d["migrated"])
            return {"digest": digest, "migrated": len(d["migrated"]),
                    "source_count": len(d["source"])}, False
        if tool == "audit_store":
            rows = {}
            for rid, rec in d["source"].items():
                cur = d["migrated"].get(rid)
                rows[rid] = {
                    "present": cur is not None,
                    "matches": bool(cur is not None and self._payload_matches(cur, self._payload(rec, "true"))),
                }
            return {"records": rows, "all_match": all(r["matches"] for r in rows.values())}, False
        if tool == "write_record":
            rid = args["record_id"]
            payload = args["payload"]
            if rid not in d["source"]:
                raise ToolError("E_NOT_FOUND", rid)
            if payload.get("schema_v") != 2:
                raise ToolError("E_PRECONDITION", "payload must declare schema_v=2")
            missing = [f for f in REQUIRED_V2 if f not in payload]
            if missing:
                raise ToolError("E_PRECONDITION", f"payload missing {missing}")
            if not isinstance(payload.get("quantity"), int) or isinstance(payload.get("quantity"), bool):
                raise ToolError("E_SCHEMA", "quantity must be int")
            if not payload.get("unit"):
                raise ToolError("E_PRECONDITION", "unit must be non-empty")
            d["migrated"][rid] = {k: payload[k] for k in REQUIRED_V2}
            d["seq"] = int(d["seq"]) + 1
            return {"record_id": rid, "written": True}, True
        raise ToolError("E_UNKNOWN_TOOL", tool)

    @staticmethod
    def _digest(migrated: dict[str, Any]) -> str:
        blob = json.dumps(migrated, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    def background_step(self, world: World, task: TaskSpec, step_index: int) -> World | None:
        """The source system corrects a quantity while the migration runs."""
        rate = task.params["config"]["concurrency"] * TEMPLATE_PARAMS[task.template_id]["correction_rate"] * 4
        if rate <= 0:
            return None
        rng = random.Random(stable_seed("migration-bg", task.task_id, step_index))
        if rng.random() >= min(0.5, rate):
            return None
        # Corrections target records the agent has not read yet, so a competent
        # agent always picks them up. Only a stale replica serves the old value.
        unread = [r for r in sorted(world.data["source"]) if r not in world.data.get("_read", [])]
        if not unread:
            return None
        w = world.clone()
        pre_v, pre_d = world.version, copy.deepcopy(world.data)
        rid = unread[rng.randrange(len(unread))]
        rec = w.data["source"][rid]
        rec["qty"] = max(1, int(rec["qty"]) + rng.choice([-7, -3, 3, 7]))
        w.data["seq"] = int(w.data["seq"]) + 1
        w.data["revision"] = int(w.data["revision"]) + 1
        w.record_and_bump(pre_v, pre_d)
        return w

    # ---- grading ---------------------------------------------------------- #
    def grade(self, world: World, task: TaskSpec) -> GradeResult:
        violations: list[str] = []
        src, mig = world.data["source"], world.data["migrated"]
        for rid in sorted(src):
            cur = mig.get(rid)
            if cur is None:
                violations.append(f"{rid}: not migrated")
                continue
            target = self._payload(src[rid], "true")
            for k, v in target.items():
                if cur.get(k) != v:
                    violations.append(f"{rid}: {k}={cur.get(k)!r} != {v!r}")
        extra = [rid for rid in mig if rid not in src]
        if extra:
            violations.append(f"unknown records written: {extra}")
        expected_rows = {rid: self._payload(rec, "true") for rid, rec in sorted(src.items())}
        expected = {"records": expected_rows, "digest": self._digest(expected_rows)}
        observed = {
            "digest": self._digest(mig),
            "records": {rid: copy.deepcopy(mig.get(rid)) for rid in sorted(src)},
        }
        return GradeResult(success=not violations, violations=violations,
                           observed=observed, expected=expected)

    def goal_projection(self, task: TaskSpec) -> dict[str, Any]:
        """The digest is over *task-declared* values; the predicate form is
        recomputed live in ``grade`` so source corrections are respected."""
        rows = {
            rid: self._payload(rec, "true") for rid, rec in sorted(task.params["source"].items())
        }
        return {"records": rows, "digest": self._digest(rows)}

    # ---- belief-conditioned argument derivation --------------------------- #
    def derive_args(
        self, tool: str, task: TaskSpec, belief: Any, step: PlanStep
    ) -> dict[str, Any] | None:
        rid = step.args.get("record_id")
        if tool != "write_record":
            return dict(step.args)
        rec = belief.get("read_record", {"record_id": rid})
        if rec is None:
            return None
        qty = int(rec["qty"])
        if belief.objective == "proxy":
            # Drift: "normalise" quantities to reduce variance. Schema-valid,
            # silently wrong -- exactly the failure a schema check cannot catch.
            qty = int(round(qty / 10.0)) * 10
        return {
            "record_id": rid,
            "payload": {"schema_v": 2, "item_name": rec["name"],
                        "quantity": qty, "unit": rec["unit"]},
        }

    # ---- faults ----------------------------------------------------------- #
    def make_fault(
        self, task: TaskSpec, world: World, rng: random.Random, failure_type: str, inject_at: int
    ) -> FaultPlan | None:
        ft = failure_type
        rid = sorted(world.data["source"])[0]
        if ft == FailureType.STALE_STATE.value:
            return FaultPlan(ft, inject_at, "read_record", None, {"lag": 1}, 3, True, 0.5)
        if ft == FailureType.INVALID_ARGUMENTS.value:
            return FaultPlan(ft, inject_at, "write_record", None,
                             {"attempts": 2, "detail": "payload.quantity must be int"}, 1, True, 0.4)
        if ft == FailureType.CONTRADICTORY_EVIDENCE.value:
            real = int(world.data["source"][rid]["qty"])
            return FaultPlan(ft, inject_at, "read_record", None,
                             {"field": "qty", "planted": real + int(rng.choice([4, 9, 11]))},
                             4, True, 0.6)
        if ft == FailureType.PLANNING_ERROR.value:
            return FaultPlan(ft, inject_at, "write_record", None, {"variant": "skip_read"}, 3, True, 0.5)
        if ft == FailureType.OBJECTIVE_DRIFT.value:
            return FaultPlan(ft, inject_at, None, None, {"proxy": "records_migrated"}, 99, True, 0.8)
        if ft == FailureType.UNRECOVERABLE_EXTERNAL_ERROR.value:
            return FaultPlan(ft, inject_at, "write_record", None, {"outage": "warehouse-api"}, 99, False, 1.0)
        return None
