#!/usr/bin/env python
"""ARB regression checks. No pytest: plain asserts, so it runs anywhere.

    python tests/run_tests.py          # all checks, ~20s
    python tests/run_tests.py -v       # per-check output

These are the invariants the paper depends on. If one fails, a number in the
paper is not trustworthy.
"""

from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arb.agents.policies import make_policy  # noqa: E402
from arb.detectors.rules import make_detector  # noqa: E402
from arb.envs import make_env  # noqa: E402
from arb.harness import EpisodeConfig, run_episode, split_for  # noqa: E402
from arb.metrics import prf, roc_auc, wilson_ci  # noqa: E402
from arb.perturb import (CANONICAL_PROGRAM, applicable_faults, program_for,  # noqa: E402
                         schedule_fault)
from arb.taxonomy import FAILURE_SPECS, RECOVERY_ACTIONS, FailureType, RecoveryAction  # noqa: E402

FAMILIES = ("tickets", "migration", "inventory", "procurement")
RESULTS: list[tuple[str, bool, str]] = []
VERBOSE = "-v" in sys.argv


def check(name: str):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
            if VERBOSE:
                print(f"  PASS {name}")
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc) or "assertion failed"))
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # a crash is a failure, not a skip
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        return fn
    return deco


def ep(family="tickets", template="A", config="cfg1", seed=0, fault=None,
       policy="rule_based", detector="rules", horizon="short") -> dict:
    cfg = EpisodeConfig(family=family, template=template, config=config, seed=seed,
                        failure_type=fault, policy=policy, detector=detector,
                        horizon_class=horizon)
    return run_episode(cfg, policy=make_policy(policy))


# --------------------------------------------------------------------------- #
@check("taxonomy: exactly seven classes, all with a canonical operator")
def _():
    assert len(FailureType) == 7, len(FailureType)
    for ft in FailureType:
        assert ft in FAILURE_SPECS
        primary, _ = CANONICAL_PROGRAM[ft.value]
        assert isinstance(primary, RecoveryAction)
        assert ft.value in str(FAILURE_SPECS[ft])


@check("taxonomy: only unrecoverable external error is contain-only")
def _():
    bad = [ft.value for ft in FailureType if not FAILURE_SPECS[ft].recoverable]
    assert bad == ["unrecoverable_external_error"], bad
    assert len(RECOVERY_ACTIONS) == 9


def _faults(family: str, template: str = "A", config: str = "cfg1", seed: int = 0) -> list[str]:
    env = make_env(family)
    task = env.make_task(template, config, seed, "train", "short")
    world = env.initial_world(task)
    return list(applicable_faults(env, task, world))


@check("taxonomy: communication_failure is multi-agent only")
def _():
    assert "communication_failure" not in _faults("tickets")
    assert "communication_failure" in _faults("procurement")


@check("metrics: AUC is tie-corrected and degenerate cases are honest")
def _():
    assert abs(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-9
    assert abs(roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])) < 1e-9
    assert abs(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9


@check("metrics: precision/recall/F1 from counts")
def _():
    d = prf(1, 0, 2)
    assert abs(d["precision"] - 1.0) < 1e-9, d
    assert abs(d["recall"] - 1 / 3) < 1e-9, d
    assert abs(d["f1"] - 0.5) < 1e-9, d


@check("metrics: Wilson interval brackets the estimate")
def _():
    lo, hi = wilson_ci(3, 10)
    assert 0.0 <= lo < 0.3 < hi <= 1.0, (lo, hi)
    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == 0.0 and hi0 > 0.0


@check("harness: episodes are deterministic byte for byte")
def _():
    a = json.dumps(ep(fault="stale_state"), sort_keys=True)
    b = json.dumps(ep(fault="stale_state"), sort_keys=True)
    assert a == b, "two identical configs produced different trajectories"


@check("harness: fault-free baselines succeed in all four families")
def _():
    for fam in FAMILIES:
        r = ep(family=fam, policy="no_recovery", detector="none")
        assert r["outcome"]["success"], f"{fam} failed without any fault"


@check("harness: every scheduled fault is actually exercised")
def _():
    for fam in FAMILIES:
        for ft in _faults(fam):
            r = ep(family=fam, fault=ft, policy="no_recovery", detector="none")
            at = r["fault"]["inject_at"]
            assert at is not None and at < len(r["steps"]), f"{fam}/{ft} never reached"


@check("harness: detection is never counted before injection")
def _():
    for fam in ("tickets", "migration", "inventory"):
        r = ep(family=fam, fault="stale_state")
        d = r["detection"]
        if d and d["detected"]:
            assert d["step"] >= r["fault"]["inject_at"], (fam, d["step"], r["fault"]["inject_at"])


@check("harness: success is state-verified, never self-reported")
def _():
    seen_gap = False
    for fam in FAMILIES:
        for ft in _faults(fam)[:3]:
            for pol in ("no_recovery", "rule_based"):
                r = ep(family=fam, fault=ft, policy=pol,
                       detector="none" if pol == "no_recovery" else "rules")
                claim, ok = r["outcome"]["claimed_success"], r["outcome"]["success"]
                assert isinstance(ok, bool)
                # The verifier must be the state comparison, not the claim.
                assert ok == (r["expected_final_state"] == r["observed_final_state"])
                seen_gap = seen_gap or (claim and not ok)
            if seen_gap:
                break
        if seen_gap:
            break
    assert seen_gap, "no self-report gap observed; claims may be leaking into the grader"


@check("harness: no recovery scores zero recovery on harmful pairs")
def _():
    for fam in FAMILIES:
        r = ep(family=fam, fault="stale_state", policy="no_recovery", detector="none")
        assert r["recovery"]["outcome"] in ("undetected", "contained"), r["recovery"]["outcome"]


@check("harness: rule-based policy recovers a local fault")
def _():
    r = ep(family="tickets", fault="invalid_arguments")
    assert r["recovery"]["outcome"] in ("recovered", "attempted_failed")
    assert r["outcome"]["detected"]


@check("detector: the rule monitor fires on a stale read")
def _():
    det = make_detector("rules")
    det.reset()
    fired = False
    for _ in range(3):
        d = det.observe({"_stale_flag": 1.0, "version_lag": 2.0, "err_any": 0.0})
        fired = fired or d.fired
    assert fired, "rules never fired on a stale signal"


@check("env: inventory success means the accounting invariant holds and is closed")
def _():
    env = make_env("inventory")
    task = env.make_task("A", "cfg1", 0, "train", "short")
    start = env.grade(env.initial_world(task), task)
    assert not start.success, "the inventory task must start unbalanced"
    r = ep(family="inventory", policy="no_recovery", detector="none")
    assert r["outcome"]["success"], "fault-free inventory run failed"
    final = r["observed_final_state"]
    assert final["period_closed"] is True, final
    for sku, st in final["skus"].items():
        assert st["balanced"] and st["imbalance"] == 0, (sku, st)


@check("dataset: released rows carry every field the paper specifies")
def _():
    r = ep(family="procurement", fault="communication_failure")
    for k in ("steps", "fault", "detection", "recovery", "outcome", "cost"):
        assert k in r, k
    s = r["steps"][0]
    for k in ("tool", "args", "expected_state", "observed_state", "expected_matches",
              "tokens_in", "tokens_out", "latency_ms", "usd"):
        assert k in s, k
    assert r["fault"]["type"] == "communication_failure"
    assert "outcome" in r["recovery"] and "actions" in r["recovery"]
    assert isinstance(r["outcome"]["retries"], int)


@check("splits: held-out template and configuration are disjoint from train")
def _():
    assert split_for("C", "cfg1") == "test_ood"
    assert split_for("A", "cfg6") == "test_id"
    assert split_for("A", "cfg1") == "train"
    assert split_for("A", "cfg5") == "dev"


@check("perturb: every class has an injection program")
def _():
    for ft in FailureType:
        assert ft.value in CANONICAL_PROGRAM


@check("perturb: schedule_fault is deterministic for a fixed seed")
def _():
    env = make_env("tickets")
    task = env.make_task("A", "cfg1", 0, "train", "short")
    world = env.initial_world(task)
    a = schedule_fault(env, task, world, "stale_state", seed=7, horizon=task.horizon)
    b = schedule_fault(env, task, world, "stale_state", seed=7, horizon=task.horizon)
    assert a.to_dict() == b.to_dict()


BASE_ROWS = [
    {"family": "tickets", "template": "A", "config": "cfg1", "seed": 0,
     "policy": "no_recovery", "fault": "stale_state", "success": False},
    {"family": "tickets", "template": "A", "config": "cfg1", "seed": 1,
     "policy": "no_recovery", "fault": "stale_state", "success": False},
    {"family": "tickets", "template": "B", "config": "cfg1", "seed": 0,
     "policy": "no_recovery", "fault": "stale_state", "success": False},
    {"family": "tickets", "template": "A", "config": "cfg1", "seed": 0,
     "policy": "no_recovery", "fault": "invalid_arguments", "success": True},
]


@check("evaluator: the harmful set comes from the baseline, not the submission")
def _():
    from arb.evaluator import harmful_keys
    keys = harmful_keys(BASE_ROWS)
    assert len(keys) == 3, keys          # the successful pair is inert and excluded
    assert all(k[-1] == "stale_state" for k in keys)
    try:
        harmful_keys([{**BASE_ROWS[0], "policy": "rule_based"}])
    except ValueError:
        pass
    else:
        raise AssertionError("a mixed-policy baseline must be rejected")


@check("evaluator: rejects submissions that report claims instead of state")
def _():
    from arb.evaluator import score
    sub = [{"family": "tickets", "template": "A", "config": "cfg1", "seed": 0,
            "fault": "stale_state", "claimed_success": True}]
    rep = score(sub, BASE_ROWS)
    assert rep["status"] == "rejected", rep
    assert rep["n_errors"] == 1


@check("evaluator: perfect and null submissions score 100% and 0%")
def _():
    from arb.evaluator import harmful_keys, score
    keys = harmful_keys(BASE_ROWS)

    def sub(success: bool) -> list[dict]:
        return [{"family": k[0], "template": k[1], "config": k[2], "seed": k[3],
                 "fault": k[4], "policy": "test", "success": success} for k in keys]

    hi = score(sub(True), BASE_ROWS)
    lo = score(sub(False), BASE_ROWS)
    assert hi["status"] == "scored", hi
    assert abs(hi["overall"]["recovery_rate"] - 1.0) < 1e-9, hi["overall"]
    assert abs(lo["overall"]["recovery_rate"]) < 1e-9
    assert lo["overall"]["usd_per_recovered"] is None


@check("evaluator: paired comparison recovers a known difference")
def _():
    from arb.evaluator import harmful_keys, paired
    keys = harmful_keys(BASE_ROWS)

    def sub(pattern: list[bool]) -> list[dict]:
        return [{"family": k[0], "template": k[1], "config": k[2], "seed": k[3],
                 "fault": k[4], "policy": "test", "success": v}
                for k, v in zip(keys, pattern)]

    a = sub([True, True, True])
    b = sub([True, False, False])
    d = paired(a, b, BASE_ROWS)
    assert d["status"] == "ok", d
    assert abs(d["diff"] - 2 / 3) < 1e-9, d
    assert d["n_pairs"] == 3


@check("metrics: exact McNemar counts discordant pairs and matches the binomial")
def _():
    from arb.metrics import mcnemar

    # Six discordant pairs, all in one direction: p = 2 * 2^-6 = 0.03125.
    a = [True] * 6 + [False] * 4
    b = [False] * 6 + [False] * 4
    r = mcnemar(a, b)
    assert (r["a_only"], r["b_only"], r["n_discordant"]) == (6, 0, 6), r
    assert abs(r["p_value"] - 2 / 2 ** 6) < 1e-12, r

    # Symmetric disagreement is no evidence at all, and agreement alone is p = 1.
    assert abs(mcnemar([1, 1, 0, 0], [0, 0, 1, 1])["p_value"] - 1.0) < 1e-12
    assert mcnemar([1, 1, 0, 0], [1, 1, 0, 0])["p_value"] == 1.0
    try:
        mcnemar([True], [True, False])
    except ValueError:
        pass
    else:
        raise AssertionError("unaligned outcomes must be rejected")


@check("metrics: Holm is step-down, monotone, and capped at one")
def _():
    from arb.metrics import holm

    adj = holm({"a": 0.01, "c": 0.03, "b": 0.04})
    assert abs(adj["a"] - 0.03) < 1e-12, adj      # 3 * 0.01
    assert abs(adj["c"] - 0.06) < 1e-12, adj      # 2 * 0.03
    assert abs(adj["b"] - 0.06) < 1e-12, adj      # monotone: never below c
    assert holm({"x": 0.9, "y": 0.95}) == {"x": 1.0, "y": 1.0}
    assert all(v <= 1.0 for v in adj.values())


class _StubLLM:
    """Stands in for ``LLMClient``: records prompts, replays one fixed reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.lock = threading.Lock()

    def complete(self, user: str) -> str:
        self.prompts.append(user)
        return self.reply


@check("llm: an off-menu or unparseable reply falls back to the canonical repair")
def _():
    from arb.agents.llm import LLMRecoveryPolicy

    want = CANONICAL_PROGRAM["stale_state"][0]
    for reply in ('{"action": "restart_everything", "confidence": 0.95}',
                  '{"action": ',            # malformed JSON
                  "I would retry the call."):
        stub = _StubLLM(reply)
        pol = LLMRecoveryPolicy(client=stub)
        d = pol.decide(detected_type="stale_state", posterior={"stale_state": 0.9},
                       step_index=4, retries_used=0, features={"stale_flag": 1.0})
        assert d.action is want, (reply, d.action)
        assert d.program == program_for(want), (reply, d.program)
        assert d.confidence == 0.0 and "fallback" in d.reason, (reply, d.reason)
        assert pol.actions == {want.value: 1}, (reply, pol.actions)
    assert pol.fallbacks == 1, "fallbacks counts per decision, not per policy"


@check("llm: a retry is substituted once the retry budget is spent")
def _():
    from arb.agents.llm import LLMRecoveryPolicy

    stub = _StubLLM('{"action": "retry", "confidence": 0.8, "reason": "transient"}')
    pol = LLMRecoveryPolicy(client=stub, max_retries=2)
    for used in (0, 1):
        d = pol.decide(detected_type="stale_state", posterior={"stale_state": 0.9},
                       step_index=2, retries_used=used)
        assert d.action is RecoveryAction.RETRY, (used, d.action)

    want = CANONICAL_PROGRAM["stale_state"][1]
    d = pol.decide(detected_type="stale_state", posterior={"stale_state": 0.9},
                   step_index=2, retries_used=2)
    assert want is not RecoveryAction.RETRY, "this check needs a non-retry secondary"
    assert d.action is want, d.action
    assert "exhausted" in d.reason, d.reason
    assert pol.fallbacks == 0
    assert pol.actions == {RecoveryAction.RETRY.value: 2, want.value: 1}, pol.actions


@check("llm: the prompt carries the monitor view, never the fault or expected state")
def _():
    from arb.agents.llm import LLMRecoveryPolicy

    banned = ("fault", "expected", "counterfactual", "inject_at", "payload",
              "grader", "label")
    for family, fault, reply in (
        ("tickets", "invalid_arguments", '{"action": "revalidate_args"}'),
        ("procurement", "communication_failure", '{"action": "resync_message"}'),
    ):
        stub = _StubLLM(reply)
        cfg = EpisodeConfig(family=family, template="A", config="cfg1", seed=0,
                            failure_type=fault, policy="llm_based", detector="rules")
        run_episode(cfg, policy=LLMRecoveryPolicy(client=stub))
        assert stub.prompts, f"{family}/{fault}: the controller was never consulted"
        for p in stub.prompts:
            low = p.lower()
            leaked = [b for b in banned if b in low]
            assert not leaked, (family, leaked)
            assert "failure posterior" in low and "monitor evidence" in low
            for a in RecoveryAction:
                assert a.value in p, (family, a.value)


def main() -> int:
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("\nFAILURES:")
        for name, _, msg in failed:
            print(f"  - {name}: {msg}")
    print(f"FAILURES: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
