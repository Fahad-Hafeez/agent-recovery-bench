"""Smoke test: every family, every applicable fault class, every policy."""

from __future__ import annotations

import sys
import traceback

from arb.agents.policies import make_policy
from arb.detectors.rules import RuleDetector
from arb.envs import make_env
from arb.envs.base import stable_seed
from arb.harness import EpisodeConfig, run_episode
from arb.taxonomy import FAILURE_SPECS, FailureType

FAMILIES = ["tickets", "migration", "inventory", "procurement"]


def applicable(fam: str) -> list[str]:
    env = make_env(fam)
    task = env.make_task("A", "cfg1", 0, "train", "short")
    world = env.initial_world(task)
    out = []
    for ft in FailureType:
        if env.track not in FAILURE_SPECS[ft].tracks:
            continue
        import random

        if env.make_fault(task, world, random.Random(0), ft.value, 1) is not None:
            out.append(ft.value)
    return out


def main() -> int:
    fails = 0
    print("== fault-free baseline (must all succeed) ==")
    for fam in FAMILIES:
        for tpl in ("A", "B", "C"):
            for cfg in ("cfg1", "cfg3"):
                c = EpisodeConfig(fam, tpl, cfg, 0, None, "no_recovery", "rules")
                try:
                    t = run_episode(c)
                except Exception:
                    traceback.print_exc()
                    fails += 1
                    continue
                ok = t["outcome"]["success"]
                print(f"  {fam:11s} {tpl} {cfg} steps={len(t['steps']):2d} success={ok} "
                      f"stop={t['recovery']['stop_reason']} "
                      f"viol={t['outcome']['violations'][:2]}")
                if not ok:
                    fails += 1

    print("== injected faults ==")
    for fam in FAMILIES:
        for ft in applicable(fam):
            row = []
            for pol in ("no_recovery", "retry_only", "rule_based"):
                c = EpisodeConfig(fam, "A", "cfg2", 1, ft, pol, "rules")
                try:
                    t = run_episode(c, policy=make_policy(pol), detector=RuleDetector())
                except Exception:
                    traceback.print_exc()
                    fails += 1
                    continue
                d = t["detection"]
                row.append(f"{pol[:6]}:{t['recovery']['outcome'][:9]:9s}/{d['type'] or '-'}")
            print(f"  {fam:11s} {ft:28s} " + "  ".join(row))

    print("== monitor mode ==")
    for mode in ("instrumented", "blackbox"):
        c = EpisodeConfig("inventory", "B", "cfg2", 2, "stale_state", "rule_based", "rules",
                          monitor_mode=mode)
        t = run_episode(c, policy=make_policy("rule_based"), detector=RuleDetector())
        print(f"  {mode:12s} det={t['detection']['detected']} outcome={t['recovery']['outcome']}")

    print("== natural fault provenance ==")
    for fam in FAMILIES:
        c = EpisodeConfig(fam, "A", "cfg1", 3, None, "rule_based", "rules",
                          fault_source="natural", competence=0.55)
        t = run_episode(c, policy=make_policy("rule_based"), detector=RuleDetector())
        print(f"  {fam:11s} fault={(t['fault'] or {}).get('type')} det={t['detection']['detected']} "
              f"outcome={t['recovery']['outcome']}")

    print("== determinism ==")
    c = EpisodeConfig("procurement", "B", "cfg3", 4, "communication_failure", "rule_based", "rules")
    a = run_episode(c, policy=make_policy("rule_based"), detector=RuleDetector())
    b = run_episode(c, policy=make_policy("rule_based"), detector=RuleDetector())
    same = a["steps"] == b["steps"] and a["cost"] == b["cost"]
    print(f"  identical={same}")
    if not same:
        fails += 1

    print(f"\nFAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
