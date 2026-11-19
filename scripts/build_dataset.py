#!/usr/bin/env python
"""Build the released ARB dataset from the harness.

    python scripts/build_dataset.py --scope release     # default, ~1.3k episodes
    python scripts/build_dataset.py --scope mini        # ~120 episodes, fast
    python scripts/build_dataset.py --scope full        # every split, larger

Outputs (data/):
  arb_dataset.jsonl   one row per (task, fault, policy) episode: full trajectory,
                      tool-call sequence, expected/observed state per step, failure
                      type, detection point, recovery action, recovery outcome,
                      retries, latency, tokens, USD, and both grader-verified and
                      self-reported success.
  arb_steps.jsonl     the same data flattened to one row per step.
  arb_mini.jsonl      small subset for tutorials and CI.
  *.parquet           columnar copies (only if pyarrow is installed).

The "harmful" flag on each row comes from the screening rule in the paper: a
(task, fault) pair is harmful iff the no-recovery episode fails. Rows are kept
even when the pair is inert, so downstream users can reproduce the screen.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arb.agents.policies import make_policy  # noqa: E402
from arb.detectors.ranker import RecoveryRanker  # noqa: E402
from arb.envs import make_env  # noqa: E402
from arb.harness import EpisodeConfig, run_episode, split_for  # noqa: E402
from arb.perturb import applicable_faults  # noqa: E402

POLICIES = ("no_recovery", "retry_only", "rule_based", "model_based")
DETECTOR_FOR = {"no_recovery": "none", "retry_only": "rules",
                "rule_based": "rules", "model_based": "supervised"}

SCOPES = {
    # families, templates, configs, seeds, limit
    "mini": (("tickets", "procurement"), ("A",), ("cfg1",), (0,), 120),
    "release": (("tickets", "migration", "inventory", "procurement"),
                ("A", "B", "C"), ("cfg1", "cfg6"), (0, 1), None),
    "full": (("tickets", "migration", "inventory", "procurement"),
             ("A", "B", "C"), ("cfg1", "cfg2", "cfg3", "cfg5", "cfg6"), (0, 1, 2), None),
}

# --------------------------------------------------------------------------- #
# One encoding rule, applied everywhere. A published JSONL dataset must be
# self-describing: Arrow infers one schema from the first rows and then rejects
# the file if any later row disagrees. Free-form subtrees here genuinely differ
# in shape -- a tool payload is a list for one tool and an object for another, a
# fault payload only carries the keys its class needs, `attempts` is empty for a
# policy that never recovers, and `fault`/`detection` are absent for fault-free
# episodes. So each such subtree travels as a JSON string under ``<key>_json``,
# and absent objects are normalised to an explicit empty one. Everything the
# protocol scores -- success, detection step, recovery outcome, retries, cost --
# stays a native scalar.
# --------------------------------------------------------------------------- #
STEP_JSON = ("args", "data", "expected_state", "observed_state")
EPISODE_JSON = ("expected_final_state", "observed_final_state", "counterfactual_final_state")
NO_FAULT = {"type": "none", "inject_at": None, "propagation_window": None, "payload": {}}
NO_DETECTION = {"step": None, "type": None, "score": 0.0, "detected": False,
                "latency_steps": None, "latency_ms": None, "posterior": {}, "n_events": 0}


def _j(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _encode_json(node: dict, *keys: str) -> dict:
    """Move free-form subtrees of ``node`` to ``<key>_json`` strings, in place.

    Replacement happens at the original key position, so the Arrow struct field
    order of a rebuilt artifact does not depend on whether the subtree came from
    the harness or from a normalisation template.
    """
    if not any(k in node for k in keys):
        return node
    encoded = [(k + "_json" if k in keys else k, _j(v) if k in keys else v)
               for k, v in node.items()]
    node.clear()
    node.update(encoded)
    return node


def _encode_step(step: dict) -> dict:
    return _encode_json(dict(step), *STEP_JSON)


def _step_rows(episode: dict, fault: str, policy: str) -> list[dict]:
    """Flatten one episode's steps, attaching the episode's own fault label."""
    labels = episode["meta"]["labels"]
    return [{
        "episode_id": episode["episode_id"],
        "family": episode["family"], "template": episode["template_id"],
        "seed": episode["seed"], "split": episode["split"],
        "fault": fault, "policy": policy,
        "t": s["index"], "agent": s["agent"], "tool": s["tool"], "args": s["args"],
        "ok": s["ok"], "error_code": s["error_code"],
        "expected_state": s["expected_state"], "observed_state": s["observed_state"],
        "expected_matches": s["expected_matches"], "signals": s["signals"],
        "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"],
        "tool_payload_tokens": s["tool_payload_tokens"],
        "latency_ms": s["latency_ms"], "usd": s["usd"],
        "label": (labels[s["index"]]["label"] if s["index"] < len(labels) else None),
    } for s in episode["steps"]]


def _flat_row(e: dict) -> dict:
    """Scalar summary column set for the `arb_episodes` table."""
    return {
        "episode_id": e["episode_id"], "family": e["family"],
        "track": e["track"], "template": e["template_id"],
        "config": e["config_id"], "split": e["split"], "seed": e["seed"],
        "fault": e["fault"]["type"], "policy": e["policy"], "detector": e["detector"],
        "monitor_mode": e["monitor_mode"],
        "success": e["outcome"]["success"],
        "claimed_success": e["outcome"]["claimed_success"],
        "detected": e["outcome"]["detected"],
        "detection_step": e["detection"]["step"],
        "detection_type": e["detection"]["type"],
        "recovery_outcome": e["recovery"]["outcome"],
        "recovery_actions": ",".join(e["recovery"]["actions"]),
        "n_attempts": e["recovery"]["n_attempts"],
        "ttr_steps": e["recovery"]["ttr_steps"],
        "retries": e["outcome"]["retries"],
        "tool_calls": e["cost"]["tool_calls"],
        "tokens": e["cost"]["tokens_total"],
        "latency_ms": e["cost"]["latency_ms"],
        "usd": e["cost"]["usd"],
        "n_corrupted": e["outcome"]["n_corrupted_fields"],
        "n_downstream": e["outcome"]["n_downstream_fields"],
        "inject_at": e["fault"]["inject_at"],
        "harmful": e["harmful"],
    }


def build(scope: str = "release", out_dir: str = "data") -> dict:
    families, templates, configs, seeds, limit = SCOPES[scope]
    ranker = None
    if os.path.exists("models/recovery_ranker.json"):
        ranker = RecoveryRanker.load("models/recovery_ranker.json")

    os.makedirs(out_dir, exist_ok=True)
    episodes: list[dict] = []
    steps_out: list[dict] = []
    envs: dict[str, Any] = {}

    for family, template, config, seed in itertools.product(families, templates, configs, seeds):
        if limit and len(episodes) >= limit:
            break
        if family not in envs:
            envs[family] = make_env(family)
        env = envs[family]
        # Which faults are injectable depends on the concrete task (its objects,
        # its step that calls the targeted tool).
        task = env.make_task(template, config, seed, split_for(template, config), "short")
        world = env.initial_world(task)
        faults = ["none", *applicable_faults(env, task, world)]
        for fault in faults:
            base = run_episode(
                EpisodeConfig(family=family, template=template, config=config, seed=seed,
                              failure_type=None if fault == "none" else fault,
                              policy="no_recovery", detector="none"),
                policy=make_policy("no_recovery"),
            )
            harmful = not base["outcome"]["success"]
            for pol in POLICIES:
                if pol == "no_recovery":
                    episode = base
                else:
                    episode = run_episode(
                        EpisodeConfig(family=family, template=template, config=config,
                                      seed=seed,
                                      failure_type=None if fault == "none" else fault,
                                      policy=pol, detector=DETECTOR_FOR[pol]),
                        policy=make_policy(pol, **({"ranker": ranker}
                                                   if pol == "model_based" else {})),
                    )
                episode["harmful"] = fault != "none" and harmful
                episode["trace_provenance"] = "controlled"
                episodes.append(episode)
                steps_out.extend(_step_rows(episode, fault, pol))
        print(f"  {family}/{template}/{config}/s{seed}: "
              f"{len(faults)} faults x {len(POLICIES)} policies", flush=True)

    for e in episodes:
        e["fault"] = e.get("fault") or dict(NO_FAULT)
        e["detection"] = e.get("detection") or dict(NO_DETECTION)
        _encode_json(e, *EPISODE_JSON)
        _encode_json(e["fault"], "payload")
        _encode_json(e["detection"], "posterior")
        _encode_json(e["recovery"], "attempts")
        _encode_json(e["outcome"], "violations")
        _encode_json(e["meta"], "env_task", "labels", "price_sheet", "policy_config")
        e["steps"] = [_encode_step(s) for s in e["steps"]]
    steps_out = [_encode_step(s) for s in steps_out]

    def dump(name: str, rows: list[dict]) -> str:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        return path

    mini = episodes[: min(120, len(episodes))]
    paths = [dump("arb_dataset.jsonl", episodes),
             dump("arb_steps.jsonl", steps_out),
             dump("arb_mini.jsonl", mini)]

    # Typed parquet is the canonical loadable artifact: Arrow unifies the schema
    # across all rows in one pass, whereas `datasets`' streaming JSON reader
    # infers per block and can disagree with itself on nullable columns.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        def write_parquet(rows: list[dict], name: str) -> str:
            path = os.path.join(out_dir, name)
            pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
            return path

        paths += [write_parquet(episodes, "arb_dataset.parquet"),
                  write_parquet(steps_out, "arb_steps.parquet"),
                  write_parquet(mini, "arb_mini.parquet"),
                  write_parquet([_flat_row(e) for e in episodes], "arb_episodes.parquet")]
    except Exception as exc:  # pragma: no cover - parquet is optional
        print(f"  (parquet skipped: {type(exc).__name__}: {exc})")

    stats = {
        "scope": scope,
        "episodes": len(episodes),
        "steps": len(steps_out),
        "families": list(families),
        "policies": list(POLICIES),
        "harmful_pairs": sum(1 for e in episodes
                             if e["harmful"] and e["policy"] == "no_recovery"),
        "files": paths,
    }
    with open(os.path.join(out_dir, "build_stats.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(stats, fh, indent=1)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", default="release", choices=sorted(SCOPES))
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    stats = build(args.scope, args.out)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
