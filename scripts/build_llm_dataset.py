#!/usr/bin/env python
"""Export the LLM-controller arm as the ``llm`` dataset config.

    python scripts/build_llm_dataset.py            # replay from the cache, no model calls

``scripts/eval_llm.py`` scores 240 harmful pairs with ``policy="llm_based"`` but writes
only aggregates. This script replays the **same** sampled pairs through the harness, so the
released rows are the episodes behind ``results/llm_agent.json``:

* every reply comes from ``results/llm_cache.jsonl``, keyed by the SHA-256 of the exact
  prompt, so the model is never contacted and the trajectories replay bit for bit;
* the script refuses to write if a prompt is missing from the cache, because contacting
  the model would produce a run the paper never scored;
* it recomputes the recovery rate overall and per class from the exported rows and aborts
  unless they match ``results/llm_agent.json`` exactly.

Encoding is imported from ``build_dataset`` so the ``llm`` config has exactly the schema
of ``episodes``: same fields, same ``*_json`` strings, same parquet types.

Outputs (data/): ``arb_llm.jsonl``, ``arb_llm.parquet`` (if pyarrow), ``llm_stats.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arb.agents.llm import LLMClient, LLMConfig, LLMRecoveryPolicy  # noqa: E402
from arb.detectors.rules import RuleDetector  # noqa: E402
from arb.evaluator import harmful_keys, load_episodes  # noqa: E402
from arb.harness import EpisodeConfig, run_episode  # noqa: E402
from scripts.build_dataset import (EPISODE_JSON, NO_DETECTION, NO_FAULT,  # noqa: E402
                                   _encode_json, _encode_step)
from scripts.eval_llm import sample_pairs  # noqa: E402


def export(*, n: int = 240, out_dir: str = "data",
           baseline: str = "results/episodes.jsonl",
           scored: str = "results/llm_agent.json",
           cache: str = "results/llm_cache.jsonl") -> dict:
    baseline_rows = [r for r in load_episodes(baseline)
                     if r.get("policy") == "no_recovery" and r.get("tag") == "screen"]
    if not baseline_rows:
        raise SystemExit(f"{baseline}: no no_recovery screen rows")
    pairs = sample_pairs(harmful_keys(baseline_rows), n)

    client = LLMClient(LLMConfig(cache_path=cache))
    policy = LLMRecoveryPolicy(client=client)
    episodes: list[dict] = []
    for family, template, config, seed, fault in pairs:
        traj = run_episode(
            EpisodeConfig(family=family, template=template, config=config, seed=seed,
                          failure_type=fault, policy="llm_based", detector="rules"),
            detector=RuleDetector(), policy=policy,
        )
        traj["harmful"] = True
        traj["trace_provenance"] = "controlled"
        traj["meta"]["policy_config"] = {
            "name": "llm_based", "model": client.config.model,
            "temperature": client.config.temperature, "reply_source": "cache",
        }
        episodes.append(traj)

    if client.usage["calls"]:
        raise SystemExit(
            f"{cache} does not cover this sample: {client.usage['calls']} prompts would "
            "need a model call. Re-run scripts/eval_llm.py first; refusing to export a "
            "run that differs from the scored one."
        )

    # Same encoding, applied in the same order as build_dataset.build().
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

    # ---- equivalence with the scored run -------------------------------- #
    with open(scored, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    rate = sum(1 for e in episodes if e["outcome"]["success"]) / len(episodes)
    want = report["llm"]["overall"]["recovery_rate"]
    if abs(rate - want) > 1e-9:
        raise SystemExit(f"replay recovery {rate:.6f} != scored {want:.6f}")
    by_class = Counter(e["fault"]["type"] for e in episodes)
    for ft, n_class in by_class.items():
        hits = sum(1 for e in episodes
                   if e["fault"]["type"] == ft and e["outcome"]["success"])
        if abs(hits / n_class - report["by_class"][ft]["llm"]) > 1e-9:
            raise SystemExit(f"replay disagrees with {scored} on {ft}")

    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "arb_llm.jsonl")
    with open(jsonl, "w", encoding="utf-8", newline="\n") as fh:
        for r in episodes:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    paths = [jsonl]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = os.path.join(out_dir, "arb_llm.parquet")
        pq.write_table(pa.Table.from_pylist(episodes), path, compression="zstd")
        paths.append(path)
    except Exception as exc:  # pragma: no cover - parquet is optional
        print(f"  (parquet skipped: {type(exc).__name__}: {exc})")

    stats = {
        "config": "llm",
        "policy": "llm_based",
        "model": client.config.model,
        "temperature": client.config.temperature,
        "episodes": len(episodes),
        "recovery_rate": round(rate, 4),
        "replay_cache_hits": client.usage["cache_hits"],
        "replay_model_calls": client.usage["calls"],
        "fallbacks": policy.fallbacks,
        "actions": dict(sorted(policy.actions.items())),
        "by_fault": dict(sorted(by_class.items())),
        "by_family": dict(sorted(Counter(e["family"] for e in episodes).items())),
        "harmful_pairs": True,
        "files": paths,
    }
    with open(os.path.join(out_dir, "llm_stats.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(stats, fh, indent=1)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", default="data")
    ap.add_argument("--baseline", default="results/episodes.jsonl")
    ap.add_argument("--scored", default="results/llm_agent.json")
    ap.add_argument("--cache", default="results/llm_cache.jsonl")
    args = ap.parse_args()
    stats = export(n=args.n, out_dir=args.out, baseline=args.baseline,
                   scored=args.scored, cache=args.cache)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
