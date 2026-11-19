"""The ARB trajectory record: the dataset's unit of exchange.

Every episode serialises to exactly this shape, for both the single-agent and
multi-agent tracks. The evaluator consumes only this object, which is what makes
"both subsets evaluated consistently" a structural property rather than a claim.

Field map to the release specification:

======================  ==================================================
required field          attribute
======================  ==================================================
trajectories            ``Trajectory`` (one JSON object per line)
tool call sequence      ``Trajectory.steps[*].tool`` / ``.args``
expected state          ``Trajectory.expected_final_state`` (+ per step)
observed state          ``Trajectory.observed_final_state`` (+ per step)
failure type            ``Trajectory.fault["type"]``
detection point         ``Trajectory.detection["step"]``
recovery action         ``Trajectory.recovery["attempts"][*]["action"]``
recovery outcome        ``Trajectory.recovery["outcome"]``
number of retries       ``Trajectory.outcome["retries"]``
latency                 ``Trajectory.cost["latency_ms"]``
token / tool cost       ``Trajectory.cost["tokens_total"]``, ``.usd``
======================  ==================================================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass
class PlanStep:
    """A step the environment considers correct from the current world state."""

    tool: str
    args: dict[str, Any]
    touched: tuple[str, ...] = ()
    agent: str = "solo"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["touched"] = list(self.touched)
        return d


@dataclass
class Step:
    """One agent action plus its observable effect.

    ``signals`` holds the *monitor-visible* feature vector for this step. It is
    computed by a redacting monitor view, never from ground-truth fault fields,
    so detectors cannot leak labels.
    """

    index: int
    agent: str
    tool: str
    args: dict[str, Any]
    ok: bool
    error_code: str | None
    data: Any
    state_version: int
    expected_state: dict[str, Any]
    observed_state: dict[str, Any]
    expected_matches: bool
    retry_index: int
    signals: dict[str, float] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    tool_payload_tokens: int = 0
    latency_ms: float = 0.0
    usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    episode_id: str
    track: str
    family: str
    template_id: str
    config_id: str
    split: str
    task_id: str
    seed: int
    objective: str
    monitor_mode: str
    policy: str
    detector: str
    steps: list[Step]
    fault: dict[str, Any] | None
    detection: dict[str, Any] | None
    recovery: dict[str, Any] | None
    outcome: dict[str, Any]
    cost: dict[str, Any]
    expected_final_state: dict[str, Any]
    observed_final_state: dict[str, Any]
    counterfactual_final_state: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- derived, single source of truth -------------------------------------
    @property
    def tool_calls(self) -> list[str]:
        return [s.tool for s in self.steps]

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def failure_type(self) -> str | None:
        return None if self.fault is None else self.fault["type"]

    @property
    def detected(self) -> bool:
        return bool(self.detection and self.detection.get("detected"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "track": self.track,
            "family": self.family,
            "template_id": self.template_id,
            "config_id": self.config_id,
            "split": self.split,
            "task_id": self.task_id,
            "seed": self.seed,
            "objective": self.objective,
            "monitor_mode": self.monitor_mode,
            "policy": self.policy,
            "detector": self.detector,
            "steps": [s.to_dict() for s in self.steps],
            "fault": self.fault,
            "detection": self.detection,
            "recovery": self.recovery,
            "outcome": self.outcome,
            "cost": self.cost,
            "expected_final_state": self.expected_final_state,
            "observed_final_state": self.observed_final_state,
            "counterfactual_final_state": self.counterfactual_final_state,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        steps = [Step(**s) for s in d["steps"]]
        return cls(
            episode_id=d["episode_id"],
            track=d["track"],
            family=d["family"],
            template_id=d["template_id"],
            config_id=d["config_id"],
            split=d["split"],
            task_id=d["task_id"],
            seed=d["seed"],
            objective=d["objective"],
            monitor_mode=d["monitor_mode"],
            policy=d["policy"],
            detector=d["detector"],
            steps=steps,
            fault=d["fault"],
            detection=d["detection"],
            recovery=d["recovery"],
            outcome=d["outcome"],
            cost=d["cost"],
            expected_final_state=d["expected_final_state"],
            observed_final_state=d["observed_final_state"],
            counterfactual_final_state=d["counterfactual_final_state"],
            meta=d.get("meta", {}),
        )


@dataclass
class TaskSpec:
    task_id: str
    family: str
    track: str
    template_id: str
    config_id: str
    split: str
    seed: int
    horizon: int
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FaultPlan:
    """Ground-truth perturbation schedule. Never visible to monitor or policy."""

    type: str
    inject_at: int
    target_tool: str | None
    target_agent: str | None
    payload: dict[str, Any]
    propagation_window: int
    recoverable: bool
    severity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GradeResult:
    success: bool
    violations: list[str]
    observed: dict[str, Any]
    expected: dict[str, Any]
    corrupted_fields: list[str] = field(default_factory=list)
    partial_credit: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: str, rows: Iterator[dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=False, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
