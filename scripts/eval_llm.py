#!/usr/bin/env python
"""Evaluate an LLM as the ARB recovery controller.

    python scripts/eval_llm.py --n 240                     # local Ollama model
    python scripts/eval_llm.py --n 240 --model qwen2.5-coder:7b

Every sampled harmful (task, fault) pair is run through the deterministic harness
with ``policy="llm_based"``: the same rules detector raises the alarm and the LLM
picks one operator from the shared action space (``arb/agents/llm.py``). The
rule-based arm is read from ``results/episodes.jsonl`` -- identical tasks, seeds
and detector -- and re-run live for any pair the file lacks, so the comparison is
paired on exactly the same pairs.

The frozen harmful set always comes from the baseline's ``no_recovery`` episodes,
and success is the grader's state comparison, never the model's own claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arb.agents.llm import LLMClient, LLMConfig, LLMRecoveryPolicy  # noqa: E402
from arb.agents.policies import make_policy  # noqa: E402
from arb.detectors.rules import RuleDetector  # noqa: E402
from arb.evaluator import harmful_keys, load_episodes, paired, score  # noqa: E402
from arb.experiments import summary_row  # noqa: E402
from arb.harness import EpisodeConfig, run_episode  # noqa: E402


def sample_pairs(harmful: set[tuple], n: int) -> list[tuple]:
    """Round-robin over (failure class, family) buckets.

    Sampling by class alone silently biases the run toward whichever family sorts
    first -- in this grid, inventory -- which changes the rule baseline the LLM is
    being compared with. Bucketing by class *and* family keeps both margins honest.
    """
    buckets: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for key in sorted(harmful):
        buckets[(key[-1], key[0])].append(key)
    names = sorted(buckets)
    picked: list[tuple] = []
    while len(picked) < n and any(buckets.values()):
        for name in names:
            if buckets[name] and len(picked) < n:
                picked.append(buckets[name].pop(0))
    return picked


def _row(traj: dict[str, Any], tag: str) -> dict[str, Any]:
    """Summary row plus the fields the protocol evaluator reads."""
    row = summary_row(traj, tag=tag)
    row["claimed_success"] = row.pop("claimed")
    row["cost"] = {"usd": row["usd"], "tool_calls": row["tool_calls"],
                   "tokens_total": row["tokens"]}
    row["detection"] = traj["detection"] or {}
    row["meta"] = {"labels": traj["meta"]["labels"]}
    return row


def _normalise_loaded(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["claimed_success"] = out.pop("claimed", False)
    out["cost"] = {"usd": out.get("usd"), "tool_calls": out.get("tool_calls"),
                   "tokens_total": out.get("tokens")}
    return out


def _table(rows_by_class: dict[str, dict[str, Any]]) -> str:
    lines = ["| failure class | n | rule-based | LLM | delta |", "|---|---|---|---|---|"]
    for ft in sorted(rows_by_class):
        r = rows_by_class[ft]
        lines.append(f"| {ft} | {r['n']} | {100 * r['rule']:.1f}% | "
                     f"{100 * r['llm']:.1f}% | {100 * (r['llm'] - r['rule']):+.1f} pp |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=240, help="harmful pairs to sample")
    ap.add_argument("--baseline", default="results/episodes.jsonl")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--cache", default="results/llm_cache.jsonl")
    ap.add_argument("--workers", type=int, default=6,
                    help="concurrent episodes (the local server batches them)")
    ap.add_argument("--out", default="results/llm_agent.json")
    args = ap.parse_args()

    all_rows = load_episodes(args.baseline)
    baseline = [r for r in all_rows if r.get("policy") == "no_recovery"
                and r.get("tag") == "screen"]
    if not baseline:
        raise SystemExit(f"{args.baseline}: no no_recovery screen rows")
    harmful = harmful_keys(baseline)
    pairs = sample_pairs(harmful, args.n)

    rule_index = {
        (r["family"], r["template"], r["config"], r["seed"], r["fault"]): _normalise_loaded(r)
        for r in all_rows if r.get("policy") == "rule_based" and r.get("tag") == "policy_pass"
    }

    client = LLMClient(LLMConfig(model=args.model, base_url=args.base_url,
                                 cache_path=args.cache))
    policy = LLMRecoveryPolicy(client=client)

    def run_llm(key: tuple) -> dict[str, Any]:
        family, template, config, seed, fault = key
        traj = run_episode(
            EpisodeConfig(family=family, template=template, config=config, seed=seed,
                          failure_type=fault, policy="llm_based", detector="rules"),
            detector=RuleDetector(), policy=policy,
        )
        return _row(traj, "llm")

    def run_rule(key: tuple) -> dict[str, Any]:
        family, template, config, seed, fault = key
        traj = run_episode(
            EpisodeConfig(family=family, template=template, config=config, seed=seed,
                          failure_type=fault, policy="rule_based", detector="rules"),
            detector=RuleDetector(), policy=make_policy("rule_based"),
        )
        return _row(traj, "rule_live")

    started = time.time()
    llm_rows: list[dict[str, Any] | None] = [None] * len(pairs)
    rule_rows: list[dict[str, Any] | None] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_llm, key): i for i, key in enumerate(pairs)}
        for done, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            llm_rows[i] = fut.result()
            rule_rows[i] = rule_index.get(pairs[i]) or run_rule(pairs[i])
            if done % 20 == 0:
                print(f"  {done}/{len(pairs)} pairs, {client.usage['calls']} calls, "
                      f"{client.usage['seconds']:.0f}s in model", flush=True)
    wall = time.time() - started

    llm_report = score([r for r in llm_rows if r], baseline)
    rule_report = score([r for r in rule_rows if r], baseline)
    comparison = paired(rule_rows, llm_rows, baseline,
                        label_a="rule_based", label_b="llm_based")

    by_class: dict[str, dict[str, Any]] = {}
    for ft in sorted({r["fault"] for r in llm_rows}):
        g_llm = [r for r in llm_rows if r["fault"] == ft]
        g_rule = [r for r in rule_rows if r["fault"] == ft]
        by_class[ft] = {
            "n": len(g_llm),
            "rule": sum(1 for r in g_rule if r["success"]) / max(1, len(g_rule)),
            "llm": sum(1 for r in g_llm if r["success"]) / max(1, len(g_llm)),
            "llm_contained": sum(1 for r in g_llm if r.get("contained")) / max(1, len(g_llm)),
        }

    out = {
        "config": {"model": args.model, "base_url": args.base_url,
                   "temperature": 0.0, "cache": args.cache},
        "sample": {"n": len(pairs), "harmful_pairs": len(harmful),
                   "by_class": dict(Counter(k[-1] for k in pairs)),
                   "by_family": dict(Counter(k[0] for k in pairs)),
                   "consulted": sum(1 for r in llm_rows if r and r.get("detected"))},
        "usage": {**client.usage,
                  "wall_seconds": round(wall, 1),
                  "fallbacks": policy.fallbacks,
                  "actions": dict(sorted(policy.actions.items()))},
        "llm": llm_report,
        "rule_based": rule_report,
        "by_class": by_class,
        "paired": comparison,
        "baseline": args.baseline,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)

    o, r = llm_report["overall"], rule_report["overall"]
    print(f"\npairs            : {len(pairs)} sampled from {len(harmful)} harmful")
    print(f"model            : {args.model}  ({client.usage['calls']} calls, "
          f"{client.usage['cache_hits']} cache hits, {client.usage['seconds']:.0f}s)")
    print(f"recovery (LLM)   : {100 * o['recovery_rate']:.1f}% "
          f"[{100 * o['recovery_ci'][0]:.1f}, {100 * o['recovery_ci'][1]:.1f}]")
    print(f"recovery (rules) : {100 * r['recovery_rate']:.1f}% "
          f"[{100 * r['recovery_ci'][0]:.1f}, {100 * r['recovery_ci'][1]:.1f}]")
    if comparison.get("status") == "ok":
        print(f"paired delta     : {100 * comparison['diff']:+.1f} pp on "
              f"{comparison['n_pairs']} pairs "
              f"[{100 * comparison['ci'][0]:+.1f}, {100 * comparison['ci'][1]:+.1f}]")
    print(f"containment      : {o['containment_rate']} "
          f"(n={o['n_unrecoverable']})")
    print(f"self-report gap  : {100 * o['selfreport_gap']:+.1f} pp")
    print(f"fallbacks        : {policy.fallbacks}   actions: {dict(sorted(policy.actions.items()))}")
    print("\n" + _table(by_class))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
