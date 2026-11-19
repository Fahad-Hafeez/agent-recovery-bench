"""Monitor-visible feature extraction.

The extractor is a *streaming* monitor: it sees one ``Step`` at a time and may
accumulate its own memory, exactly as a production observability layer would.
It never receives ``fault``, ``inject_at``, the grader's verdict, or the
oracle-intent divergence flag (``expected_matches``).

``policy_violation`` deserves a note. It is a *domain conformance check* -- the
kind a business-rules gateway performs in production -- implemented against the
written policy using only payloads the monitor has already observed. It is not a
label: it fires on any action that contradicts the declared objective's
constraints, whether or not a fault was injected, and it produces false alarms
when the world moves underneath a legitimate action (those are counted in the
precision numbers, not hidden).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FEATURE_NAMES: tuple[str, ...] = (
    "err_schema",
    "err_schema_run3",
    "err_unavailable",
    "err_unavailable_run3",
    "err_precondition",
    "err_policy",
    "err_any",
    "retry_streak",
    "version_lag",
    "stale_flag",
    "conflict_flag",
    "message_lost",
    "no_progress_run3",
    "cross_source_disagree",
    "policy_violation",
    "audit_mismatch",
    "step_pressure",
    "mutate_without_read",
    "repeat_call",
)

READ_TOOL_NAMES = frozenset({
    "get_order", "get_policy", "get_evidence", "read_record", "list_records",
    "get_stock", "get_ledger", "get_quote", "get_budget", "list_skus",
})


@dataclass
class _Mem:
    policy: dict[str, Any] | None = None
    objs: dict[str, dict[str, Any]] = field(default_factory=dict)
    quotes: dict[str, dict[str, Any]] = field(default_factory=dict)
    seen: dict[tuple[str, str], tuple[Any, int]] = field(default_factory=dict)
    reads: set[str] = field(default_factory=set)


class SignalExtractor:
    def __init__(self, family: str, monitor_mode: str = "instrumented", horizon: int = 12) -> None:
        self.family = family
        self.monitor_mode = monitor_mode
        self.horizon = max(1, horizon)
        self.reset()

    def reset(self) -> None:
        self.mem = _Mem()
        self.max_version = 0
        self.prev_version = 0
        self.recent: list[dict[str, float]] = []
        self._noprog: list[float] = []
        self.last_tool: str | None = None
        self.last_args: dict[str, Any] | None = None
        self.step = 0

    # ------------------------------------------------------------------ #
    def observe(self, step: Any) -> dict[str, float]:
        self.step = int(step.index) + 1
        f = {k: 0.0 for k in FEATURE_NAMES}
        code = step.error_code or ""
        data = step.data if isinstance(step.data, dict) else {}

        f["err_schema"] = 1.0 if code == "E_SCHEMA" else 0.0
        f["err_unavailable"] = 1.0 if code == "E_UNAVAILABLE" else 0.0
        f["err_precondition"] = 1.0 if code == "E_PRECONDITION" else 0.0
        f["err_policy"] = 1.0 if code == "E_POLICY" else 0.0
        f["err_any"] = 1.0 if code else 0.0

        if self.recent:
            prev = self.recent[-1]
            prev2 = self.recent[-2] if len(self.recent) > 1 else {}
            f["err_schema_run3"] = min(1.0, f["err_schema"] + prev.get("err_schema", 0.0)
                                       + prev2.get("err_schema", 0.0))
            f["err_unavailable_run3"] = min(1.0, f["err_unavailable"]
                                            + prev.get("err_unavailable", 0.0))

        same = (self.last_tool == step.tool and self.last_args == step.args)
        f["repeat_call"] = 1.0 if same else 0.0
        f["retry_streak"] = 1.0 if same else 0.0

        # ---- version / staleness ---------------------------------------- #
        if step.ok and step.tool in READ_TOOL_NAMES:
            lag = max(0, self.max_version - int(step.state_version))
            f["version_lag"] = 1.0 if lag > 0 else 0.0
        self.max_version = max(self.max_version, int(step.state_version))

        if self.monitor_mode == "instrumented":
            f["stale_flag"] = 1.0 if step.signals.get("_stale_flag", 0.0) > 0 else 0.0
            f["conflict_flag"] = 1.0 if step.signals.get("_conflict_flag", 0.0) > 0 else 0.0
            f["message_lost"] = 1.0 if step.signals.get("_message_lost", 0.0) > 0 else 0.0

        # ---- progress ---------------------------------------------------- #
        progressed = int(step.state_version) > self.prev_version
        no_prog = 0.0 if (step.ok and progressed) else 1.0
        f["no_progress_run3"] = 1.0 if sum(1 for v in self._noprog[-2:] if v > 0) >= 2 else 0.0
        self._noprog.append(no_prog)
        if len(self._noprog) > 6:
            self._noprog.pop(0)
        self.prev_version = max(self.prev_version, int(step.state_version))

        # ---- domain memory + conformance check --------------------------- #
        self._update_memory(step, data)
        f["policy_violation"] = self._policy_violation(step, data)
        f["audit_mismatch"] = self._audit_mismatch(step, data)
        f["cross_source_disagree"] = self._disagreement(step)

        f["step_pressure"] = min(1.0, self.step / float(self.horizon))
        f["mutate_without_read"] = 1.0 if self._mutation_without_read(step) else 0.0

        self.recent.append(dict(f))
        if len(self.recent) > 6:
            self.recent.pop(0)
        self.last_tool = step.tool
        self.last_args = dict(step.args) if isinstance(step.args, dict) else None
        step.signals.update({k: v for k, v in f.items()})
        return f

    # ---- memory --------------------------------------------------------- #
    def _update_memory(self, step: Any, data: dict[str, Any]) -> None:
        tool = step.tool
        args = step.args if isinstance(step.args, dict) else {}
        if self.family == "tickets":
            oid = args.get("order_id")
            if tool == "get_order" and isinstance(data, dict) and oid:
                rec = self.mem.objs.setdefault(oid, {})
                rec.update({
                    "amount": data.get("amount_cents"),
                    "issue": data.get("issue"),
                    "refunded": data.get("refunded_cents", 0),
                    "evidence": data.get("evidence_on_file"),
                    "applied": 0,
                })
                self.mem.reads.add(f"order:{oid}")
            if tool == "get_policy" and isinstance(data, dict):
                self.mem.policy = dict(data)
            if tool == "get_evidence" and isinstance(data, dict) and oid:
                self.mem.objs.setdefault(oid, {})["evidence"] = data.get("photo_verified")
            if tool == "issue_refund" and oid in self.mem.objs:
                rec = self.mem.objs[oid]
                rec["applied"] = rec.get("applied", 0) + int(args.get("amount_cents", 0) or 0)
        elif self.family == "inventory":
            sku = args.get("sku")
            if not sku:
                return
            rec = self.mem.objs.setdefault(sku, {"posted": 0})
            if tool == "get_stock":
                rec["on_hand"] = data.get("on_hand")
                self.mem.reads.add(f"sku:{sku}")
            if tool == "get_ledger":
                rec["ledger"] = data.get("ledger")
                self.mem.reads.add(f"sku:{sku}")
            if tool == "post_adjustment":
                rec["posted"] = rec.get("posted", 0) + int(args.get("delta", 0) or 0)
        elif self.family == "migration":
            rid = args.get("record_id")
            if tool == "read_record" and isinstance(data, dict) and rid:
                self.mem.objs[rid] = {"qty": data.get("qty"), "unit": data.get("unit"),
                                      "name": data.get("name")}
                self.mem.reads.add(f"record:{rid}")
            if tool == "write_record" and rid in self.mem.objs:
                payload = args.get("payload")
                if isinstance(payload, dict):
                    self.mem.objs[rid]["written_qty"] = payload.get("quantity")
        elif self.family == "procurement":
            if tool == "get_quote" and isinstance(data, dict) and data.get("supplier"):
                self.mem.quotes[data["supplier"]] = dict(data)
            if tool == "create_po" and isinstance(data, dict):
                self.mem.objs[data.get("po_id", "po")] = {
                    "supplier": data.get("supplier"),
                    "price": data.get("unit_price_cents"),
                }
                self.mem.reads.add("po")

    # ---- conformance ---------------------------------------------------- #
    @staticmethod
    def _expected_refund(amount: Any, issue: Any, evidence: Any,
                         policy: dict[str, Any] | None) -> int | None:
        if not isinstance(amount, int) or policy is None:
            return None
        cap = int(policy.get("cap_cents", amount))
        if issue == "wrong_item":
            fee = int(round(amount * float(policy.get("restock_pct", 0.0))))
            return max(0, amount - fee)
        if issue == "late":
            return min(amount, cap)
        if issue == "damaged":
            return amount if evidence else 0
        return None

    def _policy_violation(self, step: Any, data: dict[str, Any]) -> float:
        tool = step.tool
        args = step.args if isinstance(step.args, dict) else {}
        if self.family == "tickets":
            if tool != "issue_refund" or self.mem.policy is None:
                return 0.0
            rec = self.mem.objs.get(args.get("order_id")) or {}
            expected = self._expected_refund(rec.get("amount"), rec.get("issue"),
                                             rec.get("evidence"), self.mem.policy)
            if expected is None:
                return 0.0
            cumulative = int(rec.get("refunded", 0) or 0) + int(rec.get("applied", 0) or 0)
            return 1.0 if cumulative != expected else 0.0
        if self.family == "inventory":
            if tool != "post_adjustment":
                return 0.0
            rec = self.mem.objs.get(args.get("sku")) or {}
            on_hand, ledger = rec.get("on_hand"), rec.get("ledger")
            if not isinstance(on_hand, int) or not isinstance(ledger, int):
                return 0.0
            return 1.0 if int(rec.get("posted", 0)) != ledger - on_hand else 0.0
        if self.family == "migration":
            if tool != "write_record":
                return 0.0
            payload = args.get("payload") or {}
            rec = self.mem.objs.get(args.get("record_id")) or {}
            if not isinstance(rec.get("qty"), int):
                return 0.0
            if int(payload.get("quantity", -1)) != int(rec["qty"]):
                return 1.0
            if str(payload.get("unit", "")) != str(rec.get("unit")):
                return 1.0
            return 0.0
        if self.family == "procurement":
            if tool != "create_po" or not self.mem.quotes:
                return 0.0
            best = min(int(q["unit_price_cents"]) for q in self.mem.quotes.values())
            return 1.0 if int(args.get("unit_price_cents", -1)) > best else 0.0
        return 0.0

    def _audit_mismatch(self, step: Any, data: dict[str, Any]) -> float:
        """The application's own consistency check came back negative."""
        if step.tool not in ("audit_totals", "audit_store", "audit_account") or not step.ok:
            return 0.0
        if data.get("all_balanced") is False or data.get("all_match") is False:
            return 1.0
        if data.get("within_budget") is False:
            return 1.0
        rec, dem = data.get("received_qty"), data.get("demand")
        if isinstance(rec, int) and isinstance(dem, int) and rec != dem:
            return 1.0
        return 0.0

    def _disagreement(self, step: Any) -> float:
        """Same field, same version, different value -> a genuine contradiction."""
        data = step.data if isinstance(step.data, dict) else {}
        args = step.args if isinstance(step.args, dict) else {}
        ident = str(args.get("order_id") or args.get("sku") or args.get("record_id")
                    or args.get("supplier") or "")
        if not ident or not step.ok:
            return 0.0
        hit = 0.0
        for key in ("amount_cents", "refunded_cents", "on_hand", "ledger", "qty",
                    "unit_price_cents", "available"):
            v = data.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            prev = self.mem.seen.get((ident, key))
            if prev is not None and prev[1] == int(step.state_version) and prev[0] != v:
                hit = 1.0
            self.mem.seen[(ident, key)] = (v, int(step.state_version))
        return hit

    def _mutation_without_read(self, step: Any) -> bool:
        tool = step.tool
        args = step.args if isinstance(step.args, dict) else {}
        if self.family == "tickets" and tool in ("issue_refund", "update_status", "send_notification"):
            return f"order:{args.get('order_id')}" not in self.mem.reads
        if self.family == "inventory" and tool == "post_adjustment":
            return f"sku:{args.get('sku')}" not in self.mem.reads
        if self.family == "migration" and tool == "write_record":
            return f"record:{args.get('record_id')}" not in self.mem.reads
        return False


def vector(features: dict[str, float], names: tuple[str, ...] = FEATURE_NAMES) -> list[float]:
    return [float(features.get(n, 0.0)) for n in names]
