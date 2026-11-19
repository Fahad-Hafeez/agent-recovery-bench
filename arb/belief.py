"""The reference agent's belief state.

This is the module that makes failures *physical*. A reference agent executes a
known procedure (an SOP) but never sees the oracle state: every argument it
passes is derived from something it has actually observed. Consequently:

* a stale read makes the next write wrong without raising any error;
* a planted contradiction is indistinguishable from truth until audited;
* a dropped message leaves the buyer with an empty inbox and a legacy default.

``applied`` records the agent's own successful actions, which is knowledge it
legitimately has (it did them) and which it needs to avoid double-counting its
own effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def obs_key(tool: str, args: dict[str, Any] | None = None) -> str:
    ids: list[str] = []
    if args:
        for k in ("order_id", "record_id", "sku", "supplier", "po_id"):
            if k in args and args[k] is not None:
                ids.append(f"{k}={args[k]}")
    return f"{tool}::{'|'.join(ids)}"


@dataclass
class Observation:
    tool: str
    args: dict[str, Any]
    data: Any
    version: int
    step: int


@dataclass
class Belief:
    obs: dict[str, Observation] = field(default_factory=dict)
    applied: list[tuple[int, str, dict[str, Any]]] = field(default_factory=list)
    inbox: list[dict[str, Any]] = field(default_factory=list)
    objective: str = "true"

    # ---- writes ----------------------------------------------------------- #
    def put(self, tool: str, args: dict[str, Any], data: Any, version: int, step: int) -> None:
        self.obs[obs_key(tool, args)] = Observation(tool, dict(args), data, version, step)

    def put_applied(self, step: int, tool: str, args: dict[str, Any]) -> None:
        self.applied.append((step, tool, dict(args)))

    # ---- reads ------------------------------------------------------------ #
    def get(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any] | None:
        o = self.obs.get(obs_key(tool, args))
        return None if o is None else o.data

    def latest_for(self, tool: str, args: dict[str, Any] | None) -> dict[str, Any] | None:
        """Most recent observation of a specific object (same id arguments)."""
        key = obs_key(tool, args)
        o = self.obs.get(key)
        return None if o is None else o.data

    def latest(self, tool: str) -> dict[str, Any] | None:
        best: Observation | None = None
        for o in self.obs.values():
            if o.tool == tool and (best is None or o.step >= best.step):
                best = o
        return None if best is None else best.data

    def latest_version(self) -> int:
        return max((o.version for o in self.obs.values()), default=0)

    def applied_total(self, tool: str, field_name: str, key: str, value: Any) -> int:
        total = 0
        for _, t, a in self.applied:
            if t == tool and a.get(key) == value:
                v = a.get(field_name, 0)
                if isinstance(v, int) and not isinstance(v, bool):
                    total += v
        return total

    def applied_count(self, tool: str, key: str, value: Any) -> int:
        return sum(1 for _, t, a in self.applied if t == tool and a.get(key) == value)

    # ---- accounting ------------------------------------------------------- #
    def cost_tokens(self) -> int:
        return sum(len(str(o.data)) // 4 for o in self.obs.values())

    def reset(self) -> None:
        self.obs.clear()
        self.applied.clear()
        self.inbox.clear()
        self.objective = "true"
