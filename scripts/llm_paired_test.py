#!/usr/bin/env python
"""Paired significance test for the LLM-controller arm.

    python scripts/llm_paired_test.py

``scripts/eval_llm.py`` writes aggregates only, so the arm's per-pair outcomes are
read back from the released export ``data/arb_llm.jsonl``. That export is produced
by ``scripts/build_llm_dataset.py`` from the prompt cache alone: it never contacts
the model, and it refuses to write unless it replays the scored recovery rate, so
its rows are the episodes behind ``results/llm_agent.json``.

Nothing here is allowed to drift from the scored run. The script joins the LLM
episodes to the rule-based ones on the same (family, template, config, seed, fault)
key, then aborts unless the joined rates and the paired cluster bootstrap reproduce
``results/llm_agent.json`` exactly. Only then does it compute the exact McNemar
test, which answers the question the interval cannot: is a difference present at
all?

The arm is a single pre-specified test on its own 240-pair sample, so no
multiplicity correction applies to it; it is reported separately from the three
Holm-corrected families over the 861-pair grid.

Outputs: ``results/llm_paired.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arb.evaluator import load_episodes  # noqa: E402
from arb.metrics import mcnemar, paired_bootstrap_diff  # noqa: E402

TOL = 1e-9


def _key(row: dict) -> tuple:
    """The pair a row was scored on, in the form both files share."""
    fault = row["fault"]["type"] if isinstance(row["fault"], dict) else row["fault"]
    template = row.get("template_id") or row.get("template")
    config = row.get("config_id") or row.get("config")
    return (row["family"], template, config, int(row["seed"]), fault)


def _load_llm(path: str) -> dict[tuple, bool]:
    outcomes: dict[tuple, bool] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            outcomes[_key(row)] = bool(row["outcome"]["success"])
    return outcomes


def _load_rule(path: str) -> dict[tuple, bool]:
    outcomes: dict[tuple, bool] = {}
    for row in load_episodes(path):
        if row.get("policy") == "rule_based" and row.get("tag") == "policy_pass":
            outcomes[_key(row)] = bool(row["success"])
    return outcomes


def _close(got: float, want: float, what: str) -> None:
    if abs(got - want) > TOL:
        raise SystemExit(f"{what}: recomputed {got:.9f} != scored {want:.9f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", default="data/arb_llm.jsonl",
                    help="LLM-arm export (scripts/build_llm_dataset.py)")
    ap.add_argument("--baseline", default="results/episodes.jsonl",
                    help="main grid episodes, for the rule-based arm")
    ap.add_argument("--report", default="results/llm_agent.json",
                    help="scored aggregates the test must reproduce")
    ap.add_argument("--out", default="results/llm_paired.json")
    args = ap.parse_args()

    with open(args.report, encoding="utf-8") as fh:
        report = json.load(fh)

    llm = _load_llm(args.episodes)
    rule = _load_rule(args.baseline)
    keys = sorted(set(llm) & set(rule))
    if not keys:
        raise SystemExit(f"{args.episodes} and {args.baseline} share no pairs")
    if len(keys) != len(llm):
        raise SystemExit(f"{len(llm) - len(keys)} LLM pairs have no rule-based "
                         "counterpart; the comparison would not be paired")

    a = [rule[k] for k in keys]          # canonical mapping
    b = [llm[k] for k in keys]           # LLM controller
    clusters = [k[1] for k in keys]

    # ---- the join must be the scored run, not merely a similar one ---------- #
    n_scored = report["sample"]["n"]
    if len(keys) != n_scored:
        raise SystemExit(f"{len(keys)} pairs joined but the report scored {n_scored}")
    _close(sum(a) / len(a), report["rule_based"]["overall"]["recovery_rate"],
           "rule-based recovery")
    _close(sum(b) / len(b), report["llm"]["overall"]["recovery_rate"],
           "LLM recovery")
    boot = paired_bootstrap_diff([float(x) for x in a], [float(x) for x in b],
                                 clusters, seed=0)
    scored = report["paired"]
    _close(boot["diff"], scored["diff"], "paired difference")
    _close(boot["lo"], scored["ci"][0], "paired interval (low)")
    _close(boot["hi"], scored["ci"][1], "paired interval (high)")

    test = mcnemar(a, b)

    out = {
        "policy_a": "rule_based",
        "policy_b": "llm_based",
        "n_pairs": test["n"],
        "n_clusters": len(set(clusters)),
        "clusters": sorted(set(clusters)),
        "rule_recovery_rate": sum(a) / len(a),
        "llm_recovery_rate": sum(b) / len(b),
        "paired_diff": boot["diff"],
        "paired_ci": [boot["lo"], boot["hi"]],
        "a_only": test["a_only"],
        "b_only": test["b_only"],
        "n_discordant": test["n_discordant"],
        "p_value": test["p_value"],
        "test": "exact two-sided McNemar (binomial on discordant pairs)",
        "family_size": 1,
        "sources": {"episodes": args.episodes, "baseline": args.baseline,
                    "report": args.report},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")

    print(f"pairs            : {out['n_pairs']} ({out['n_clusters']} templates)")
    print(f"rules            : {100 * out['rule_recovery_rate']:.2f}%")
    print(f"LLM              : {100 * out['llm_recovery_rate']:.2f}%  "
          f"({100 * out['paired_diff']:+.2f} pp "
          f"[{100 * out['paired_ci'][0]:+.2f}, {100 * out['paired_ci'][1]:+.2f}])")
    print(f"discordant       : b={out['a_only']} (rules only), "
          f"c={out['b_only']} (LLM only)")
    print(f"exact McNemar p  : {out['p_value']:.3e}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
