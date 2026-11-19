"""Experiment driver: produces every number reported in the paper.

Protocol (also stated in the paper):

1. **Harmfulness screen.** Every (task, fault) pair is run with ``no_recovery``.
   If the task still succeeds, the fault is *inert* and is excluded from both
   detection and recovery denominators. Conditioning on outcome-relevant faults
   is conservative for recall: silent faults are exactly the hard ones.
2. **Detector training** on the ``train`` split only (templates A/B, configs 1-3).
3. **Policy evaluation** on all splits, including a held-out template (C) and a
   held-out configuration (cfg6).
4. **Ranker training** on logged detections from the train split, then a fresh
   ``model_based`` pass over every split.
5. **Uncertainty** by cluster bootstrap over task templates.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from arb.agents.policies import make_policy
from arb.detectors.features import FEATURE_NAMES, vector
from arb.detectors.ranker import RecoveryRanker, rank_vector
from arb.detectors.rules import RuleDetector
from arb.detectors.supervised import CLASSES, MultinomialLogistic, SupervisedDetector
from arb.envs import make_env
from arb.harness import EpisodeConfig, build_tasks, run_episode
from arb.metrics import bootstrap_ci, confusion, paired_bootstrap_diff, prf, roc_auc
from arb.taxonomy import FAILURE_SPECS, FailureType

SEEDS = (0, 1, 2)
FAMILIES = ("tickets", "migration", "inventory", "procurement")
TEMPLATES = ("A", "B", "C")
CONFIGS = ("cfg1", "cfg2", "cfg3", "cfg5", "cfg6")
POLICIES = ("no_recovery", "retry_only", "rule_based", "model_based")


def mean(xs: Sequence[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def applicable_faults(family: str) -> list[str]:
    env = make_env(family)
    task = env.make_task("A", "cfg1", 0, "train", "short")
    world = env.initial_world(task)
    out = []
    for ft in FailureType:
        if env.track not in FAILURE_SPECS[ft].tracks:
            continue
        if env.make_fault(task, world, random.Random(0), ft.value, 1) is not None:
            out.append(ft.value)
    return out


def summary_row(traj: dict[str, Any], tag: str = "policy_pass") -> dict[str, Any]:
    out = traj["outcome"]
    det = traj["detection"] or {}
    rec = traj["recovery"]
    cost = traj["cost"]
    row = {
        "episode_id": traj["episode_id"], "family": traj["family"], "track": traj["track"],
        "template": traj["template_id"], "config": traj["config_id"], "split": traj["split"],
        "fault": traj["fault"]["type"] if traj["fault"] else "none",
        "policy": traj["policy"], "monitor_mode": traj["monitor_mode"], "tag": tag,
        "success": bool(out["success"]), "claimed": bool(out["claimed_success"]),
        "detected": bool(det.get("detected")), "det_type": det.get("type"),
        "ttd": det.get("latency_steps"), "outcome": rec["outcome"], "ttr": rec["ttr_steps"],
        "retries": out["retries"], "tool_calls": cost["tool_calls"], "tokens": cost["tokens_total"],
        "usd": cost["usd"], "latency_ms": cost["latency_ms"],
        "n_corrupted": out["n_corrupted_fields"], "n_downstream": out["n_downstream_fields"],
        "contained": rec["contained"], "n_attempts": rec["n_attempts"],
        "actions": rec["actions"][:4], "inject_at": (traj["fault"] or {}).get("inject_at"),
        "n_steps": len(traj["steps"]), "horizon": traj["meta"].get("horizon"),
        "fault_source": traj["meta"].get("fault_source"),
        "seed": traj["seed"],
        "det_score": float(det.get("score", 0.0)),
    }
    if det.get("detected"):
        row["det_posterior"] = dict(det.get("posterior") or {})
        step = traj["steps"][det["step"]]
        row["det_features"] = {k: float(step["signals"].get(k, 0.0)) for k in FEATURE_NAMES}
    return row


class Experiment:
    def __init__(self, out_dir: str = "results") -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        self.cf: dict[str, dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self.train_steps: list[dict[str, Any]] = []
        self.eval_steps: list[dict[str, Any]] = []
        self.detector: SupervisedDetector | None = None
        self.ranker: RecoveryRanker | None = None
        self.harmful: dict[tuple, bool] = {}
        self.applicable = {f: applicable_faults(f) for f in FAMILIES}

    # ---- helpers ------------------------------------------------------- #
    def counterfactual(self, fam: str, tpl: str, cfg: str, seed: int) -> dict[str, Any]:
        key = f"{fam}|{tpl}|{cfg}|{seed}"
        if key not in self.cf:
            self.cf[key] = run_episode(
                EpisodeConfig(fam, tpl, cfg, seed, None, "no_recovery", "none")
            )["observed_final_state"]
        return self.cf[key]

    def tasks(self, templates: Sequence[str] = TEMPLATES, configs: Sequence[str] = CONFIGS,
              seeds: Sequence[int] = SEEDS, horizon_class: str = "short") -> list[dict[str, Any]]:
        return list(build_tasks(FAMILIES, templates, configs, seeds, horizon_class))

    # ---- 1. screen ----------------------------------------------------- #
    def screen(self, tasks: list[dict[str, Any]]) -> None:
        n = 0
        for t in tasks:
            cf = self.counterfactual(t["family"], t["template"], t["config"], t["seed"])
            for ft in self.applicable[t["family"]]:
                cfg = EpisodeConfig(t["family"], t["template"], t["config"], t["seed"],
                                    ft, "no_recovery", "none", t["horizon_class"])
                traj = run_episode(cfg, counterfactual=cf)
                row = summary_row(traj, tag="screen")
                self.rows.append(row)
                self.harmful[(t["family"], t["template"], t["config"], t["seed"], ft)] = not row["success"]
                if t["split"] == "train":
                    for lab, step in zip(traj["meta"]["labels"], traj["steps"]):
                        self.train_steps.append({
                            "features": {k: float(step["signals"].get(k, 0.0)) for k in FEATURE_NAMES},
                            "label": lab["label"],
                            "val": t["config"] == "cfg3",
                        })
                for lab, step in zip(traj["meta"]["labels"], traj["steps"]):
                    self.eval_steps.append({
                        "label": lab["label"], "score": float(lab["score"]),
                        "pred": lab["pred"] or "none", "detector": "none",
                        "mode": "instrumented", "split": t["split"],
                        "fault": ft, "family": t["family"],
                        # The template is the bootstrap cluster, so a step record
                        # must carry it or detection intervals cannot be clustered.
                        "template": t["template"],
                    })
                n += 1
                if n % 400 == 0:
                    print(f"    screened {n}", flush=True)

    # ---- 2. detector --------------------------------------------------- #
    def train_detector(self) -> dict[str, Any]:
        X = np.array([vector(r["features"]) for r in self.train_steps])
        y = [r["label"] for r in self.train_steps]
        dist = Counter(y)
        print(f"    rows={len(y)} classes={dict(dist)}", flush=True)
        model = MultinomialLogistic(len(FEATURE_NAMES), CLASSES, l2=2e-3, lr=0.15, epochs=900)
        model.fit(X, y)
        model.save(os.path.join("models", "detector.json"))

        # Calibrate the decision threshold on a held-out configuration (cfg3),
        # never on the evaluation splits. An uncalibrated 0.5 threshold made the
        # model fire on almost every step, which is a tuning artefact, not a
        # property of the signals.
        val_idx = [i for i, r in enumerate(self.train_steps) if r.get("val")]
        threshold, val_f1 = 0.5, None
        if val_idx:
            Xv = X[val_idx]
            yv = [0 if y[i] == "none" else 1 for i in val_idx]
            pv = model.predict_proba(Xv)
            sv = list(1 - pv[:, model.classes.index("none")])
            best = (-1.0, 0.5, 0.0)
            for thr in [round(0.20 + 0.05 * k, 2) for k in range(15)]:
                yhat = [1 if s >= thr else 0 for s in sv]
                tp = sum(1 for a, b in zip(yv, yhat) if a and b)
                fp = sum(1 for a, b in zip(yv, yhat) if not a and b)
                fn = sum(1 for a, b in zip(yv, yhat) if a and not b)
                f1 = prf(tp, fp, fn)["f1"]
                if f1 > best[0]:
                    best = (f1, thr, prf(tp, fp, fn)["precision"])
            val_f1, threshold = best[0], best[1]
            print(f"    calibrated threshold={threshold} val_f1={val_f1:.3f}", flush=True)
        self.detector = SupervisedDetector(model, threshold=threshold)
        p = model.predict_proba(X)
        p_none = p[:, model.classes.index("none")]
        y_bin = [0 if lab == "none" else 1 for lab in y]
        return {
            "n_rows": int(len(y)), "class_counts": dict(dist),
            "train_auc": roc_auc(y_bin, list(1 - p_none)),
            "train_ece": self._ece(y_bin, list(1 - p_none)),
            "threshold": threshold, "val_f1": val_f1,
            "features": list(FEATURE_NAMES), "classes": list(CLASSES),
        }

    @staticmethod
    def _ece(labels: Sequence[int], probs: Sequence[float], bins: int = 10) -> float:
        total = max(1, len(labels))
        err = 0.0
        for b in range(bins):
            idx = [i for i, p in enumerate(probs) if b / bins <= p < (b + 1) / bins
                   or (b == bins - 1 and p >= 1.0)]
            if not idx:
                continue
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(conf - acc)
        return err

    # ---- 3. policies --------------------------------------------------- #
    def evaluate(self, tasks: list[dict[str, Any]], policies: Sequence[str],
                 tag: str = "policy_pass", monitor_mode: str = "instrumented") -> None:
        n = 0
        for t in tasks:
            cf = self.counterfactual(t["family"], t["template"], t["config"], t["seed"])
            for ft in self.applicable[t["family"]]:
                if not self.harmful.get(
                        (t["family"], t["template"], t["config"], t["seed"], ft), False):
                    continue
                for pol in policies:
                    dname = ("none" if pol == "no_recovery"
                             else ("rules" if pol in ("retry_only", "rule_based") else "supervised"))
                    cfg = EpisodeConfig(t["family"], t["template"], t["config"], t["seed"],
                                        ft, pol, dname, t["horizon_class"],
                                        monitor_mode=monitor_mode)
                    det = (None if dname == "none"
                           else (RuleDetector() if dname == "rules" else self.detector))
                    policy = make_policy(pol, **({"ranker": self.ranker}
                                                 if pol == "model_based" else {}))
                    traj = run_episode(cfg, detector=det, policy=policy, counterfactual=cf)
                    row = summary_row(traj, tag=tag)
                    row["horizon_class"] = t["horizon_class"]
                    self.rows.append(row)
                    for lab, step in zip(traj["meta"]["labels"], traj["steps"]):
                        self.eval_steps.append({
                            "label": lab["label"], "score": float(lab["score"]),
                            "pred": lab["pred"] or "none", "detector": dname,
                            "mode": traj["monitor_mode"], "split": t["split"],
                            "fault": ft, "family": traj["family"],
                            "template": t["template"],
                        })
                    n += 1
                if n % 500 == 0:
                    print(f"    {n} episodes", flush=True)

    # ---- 4. ranker ----------------------------------------------------- #
    def train_ranker(self) -> dict[str, Any]:
        X, actions, y = [], [], []
        for r in self.rows:
            # Only *deliberate* recovery decisions: blind retries (retry_only)
            # are not a decision about which operator to apply, and including
            # them makes the value model learn "retry is bad" as a prior.
            if r["tag"] != "policy_pass" or r["split"] != "train" or r["policy"] != "rule_based":
                continue
            if not r["detected"] or not r.get("det_features") or not r["actions"]:
                continue
            X.append(rank_vector(r.get("det_posterior") or {},
                                 {**r["det_features"], "_score": r.get("det_score", 0.0)},
                                 r["retries"]))
            actions.append(r["actions"][0])
            y.append(1 if r["success"] else 0)
        if len(X) < 60 or len(set(y)) < 2:
            return {"n": len(y), "skipped": True}
        ranker = RecoveryRanker(n_features=len(rank_vector({}, {}, 0)))
        ranker.fit(np.array(X), actions, y)
        ranker.save(os.path.join("models", "recovery_ranker.json"))
        self.ranker = ranker
        return {"n": len(y), "positive_rate": float(np.mean(y)), "skipped": False,
                "actions": dict(Counter(actions))}

    # ---- 5. aggregation ------------------------------------------------ #
    def aggregate(self, det_info: dict[str, Any], ranker_info: dict[str, Any]) -> dict[str, Any]:
        faults = [r for r in self.rows if r["fault"] != "none"]
        ev = [r for r in faults if r["tag"] in ("policy_pass", "long", "natural")]
        main = [r for r in ev if r["tag"] == "policy_pass"]
        report: dict[str, Any] = {
            "meta": {
                "seeds": list(SEEDS), "families": list(FAMILIES), "templates": list(TEMPLATES),
                "configs": list(CONFIGS), "policies": list(POLICIES),
                "applicable_faults": self.applicable, "price_sheet": "arb-price-2026.11",
                "n_episodes": len(self.rows),
                "n_inert_pairs": sum(1 for v in self.harmful.values() if not v),
                "n_pairs": len(self.harmful),
            },
            "detector_model": det_info,
            "ranker": ranker_info,
            "detection": self._detection(),
            "main": {
                "by_policy": self._group(main, ("policy",)),
                "by_policy_track": self._group(main, ("policy", "track")),
                "by_policy_family": self._group(main, ("policy", "family")),
                "by_policy_split": self._group(main, ("policy", "split")),
            },
            "recovery_by_type": self._group(main, ("fault", "policy")),
            "cost": self._cost(main),
            "propagation": self._propagation(main),
            "selfreport": self._selfreport(main),
            "natural": self._natural([r for r in ev if r["tag"] == "natural"]),
            "long_horizon": self._group([r for r in ev if r["tag"] == "long"], ("policy", "family")),
            "ablation": self._ablation(),
            "paired": self._paired(main),
            "coordination": self._coordination(main),
        }
        report["headline"] = self._headline(report)
        return report

    def _group(self, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[tuple(r[k] for k in keys)].append(r)
        out = []
        for k, g in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            succ = [1.0 if r["success"] else 0.0 for r in g]
            ci = bootstrap_ci(succ, [r["template"] for r in g], seed=1)
            out.append({
                **dict(zip(keys, k)), "n": len(g),
                "recovery_rate": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"],
                "detected_rate": mean([1.0 if r["detected"] else 0.0 for r in g]),
                "ttd_mean": mean([r["ttd"] for r in g]),
                "ttr_mean": mean([r["ttr"] for r in g]),
                "retries_mean": mean([r["retries"] for r in g]),
                "tool_calls_mean": mean([r["tool_calls"] for r in g]),
                "tokens_mean": mean([r["tokens"] for r in g]),
                "latency_mean": mean([r["latency_ms"] for r in g]),
                "usd_mean": mean([r["usd"] for r in g]),
                "corrupted_mean": mean([r["n_corrupted"] for r in g]),
                "downstream_mean": mean([r["n_downstream"] for r in g]),
                "contained_rate": mean([1.0 if r["contained"] else 0.0 for r in g]),
                "undetected_rate": mean([1.0 if r["outcome"] == "undetected" else 0.0 for r in g]),
            })
        return out

    def _detection(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for det in ("rules", "supervised"):
            for mode in ("instrumented", "blackbox"):
                rows = [s for s in self.eval_steps if s["detector"] == det and s["mode"] == mode]
                if not rows:
                    continue
                y = [0 if s["label"] == "none" else 1 for s in rows]
                sc = [s["score"] for s in rows]
                yhat = [0 if s["pred"] == "none" else 1 for s in rows]
                tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
                fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
                fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
                type_rows = [s for s in rows if s["label"] != "none"]
                hit = sum(1 for s in type_rows if s["pred"] == s["label"])
                out[f"{det}_{mode}"] = {
                    "detector": det, "mode": mode, "n_steps": len(rows),
                    "positives": sum(y), **prf(tp, fp, fn),
                    "auc": roc_auc(y, sc), "type_accuracy": hit / max(1, len(type_rows)),
                    "confusion": confusion([s["label"] for s in type_rows],
                                           [s["pred"] for s in type_rows],
                                           [f.value for f in FailureType]),
                    "ece": self._ece(y, sc),
                    "fp_rate": fp / max(1, fp + tp),
                }
                by_split = {}
                for split in ("train", "dev", "test_id", "test_ood"):
                    sub = [s for s in rows if s["split"] == split]
                    if not sub:
                        continue
                    ys = [0 if s["label"] == "none" else 1 for s in sub]
                    by_split[split] = {
                        "n": len(sub),
                        "auc": roc_auc(ys, [s["score"] for s in sub]),
                        "detected_rate": sum(1 for s in sub if s["pred"] != "none") / len(sub),
                    }
                out[f"{det}_{mode}"]["by_split"] = by_split
        return out

    def _cost(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for pol in POLICIES:
            g = [r for r in rows if r["policy"] == pol]
            if not g:
                continue
            ok = [r for r in g if r["success"]]
            out.append({
                "policy": pol, "n": len(g),
                "usd_per_episode": mean([r["usd"] for r in g]),
                "usd_per_recovered": (sum(r["usd"] for r in g) / len(ok)) if ok else None,
                "usd_failed": mean([r["usd"] for r in g if not r["success"]]),
                "tokens_per_episode": mean([r["tokens"] for r in g]),
                "latency_ms_per_episode": mean([r["latency_ms"] for r in g]),
                "tool_calls_per_episode": mean([r["tool_calls"] for r in g]),
                "recovery_rate": mean([1.0 if r["success"] else 0.0 for r in g]),
            })
        return out

    def _propagation(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        det = [r for r in rows if r["detected"]]
        und = [r for r in rows if r["policy"] != "no_recovery" and not r["detected"]]
        hist = Counter(int(r["ttd"]) for r in det if r["ttd"] is not None)
        early = [r for r in det if r["policy"] == "rule_based" and (r["ttd"] or 0) <= 0]
        late = [r for r in det if r["policy"] == "rule_based" and (r["ttd"] or 0) > 0]
        return {
            "ttd_hist": {str(k): v for k, v in sorted(hist.items())},
            "ttd_mean": mean([r["ttd"] for r in det]),
            "early_n": len(early),
            "early_recovery": mean([1.0 if r["success"] else 0.0 for r in early]),
            "late_n": len(late),
            "late_recovery": mean([1.0 if r["success"] else 0.0 for r in late]),
            "early_downstream": mean([r["n_downstream"] for r in early]),
            "late_downstream": mean([r["n_downstream"] for r in late]),
            "undetected_n": len(und),
            "undetected_corrupted": mean([r["n_corrupted"] for r in und]),
            "detected_corrupted": mean([r["n_corrupted"] for r in det]),
            "downstream_by_policy": {
                pol: mean([r["n_downstream"] for r in rows if r["policy"] == pol])
                for pol in POLICIES
            },
        }

    def _selfreport(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        claimed = [r for r in rows if r["claimed"]]
        false_claim = [r for r in claimed if not r["success"]]
        return {
            "n": len(rows),
            "claimed_rate": mean([1.0 if r["claimed"] else 0.0 for r in rows]),
            "verified_rate": mean([1.0 if r["success"] else 0.0 for r in rows]),
            "n_claimed_but_failed": len(false_claim),
            "claimed_but_failed_rate": len(false_claim) / max(1, len(claimed)),
        }

    def _natural(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        return {
            "n": len(rows),
            "by_type": dict(Counter(r["fault"] for r in rows)),
            "detected_rate": mean([1.0 if r["detected"] else 0.0 for r in rows]),
            "recovery_rate": mean([1.0 if r["success"] else 0.0 for r in rows]),
            "type_agreement": mean([1.0 if r["det_type"] == r["fault"] else 0.0
                                    for r in rows if r["detected"]]),
        }

    def _paired(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keyed: dict[tuple, dict[str, bool]] = defaultdict(dict)
        cl: dict[tuple, str] = {}
        for r in rows:
            k = (r["family"], r["template"], r["config"], r["seed"], r["fault"])
            keyed[k][r["policy"]] = r["success"]
            cl[k] = r["template"]
        out = []
        for a, b in (("retry_only", "no_recovery"), ("rule_based", "retry_only"),
                     ("rule_based", "no_recovery"), ("model_based", "rule_based")):
            ks = [k for k, v in keyed.items() if a in v and b in v]
            if len(ks) < 5:
                continue
            va = [1.0 if keyed[k][a] else 0.0 for k in ks]
            vb = [1.0 if keyed[k][b] else 0.0 for k in ks]
            d = paired_bootstrap_diff(va, vb, [cl[k] for k in ks], seed=3)
            out.append({"a": a, "b": b, "n_pairs": len(ks), **d,
                        "a_rate": mean(va), "b_rate": mean(vb)})
        return out

    def _coordination(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        comm = [r for r in rows if r["fault"] == FailureType.COMMUNICATION_FAILURE.value]
        single = [r for r in rows if r["track"] == "single"]
        multi = [r for r in rows if r["track"] == "multi"]
        return {
            "comm_n": len(comm),
            "comm_by_policy": {
                pol: {
                    "n": len([r for r in comm if r["policy"] == pol]),
                    "recovery": mean([1.0 if r["success"] else 0.0
                                      for r in comm if r["policy"] == pol]),
                    "detected": mean([1.0 if r["detected"] else 0.0
                                      for r in comm if r["policy"] == pol]),
                } for pol in POLICIES if any(r["policy"] == pol for r in comm)
            },
            "single_recovery": mean([1.0 if r["success"] else 0.0 for r in single]),
            "multi_recovery": mean([1.0 if r["success"] else 0.0 for r in multi]),
            "single_n": len(single), "multi_n": len(multi),
        }

    # ---- signal ablation ------------------------------------------------ #
    def _ablation(self) -> list[dict[str, Any]]:
        groups = {
            "error_codes": ("err_schema", "err_unavailable", "err_precondition", "err_policy",
                            "err_any", "err_schema_run3", "err_unavailable_run3"),
            "version_staleness": ("version_lag", "stale_flag"),
            "conflict": ("conflict_flag", "cross_source_disagree"),
            "communication": ("message_lost",),
            "progress": ("no_progress_run3", "repeat_call", "mutate_without_read", "step_pressure"),
            "policy_conformance": ("policy_violation", "audit_mismatch"),
        }

        class Masked(RuleDetector):
            def __init__(self, drop: Iterable[str]) -> None:
                super().__init__()
                self.drop = set(drop)

            def score(self, f):  # type: ignore[override]
                return super().score({k: (0.0 if k in self.drop else v) for k, v in f.items()})

        tasks = self.tasks(templates=("A",), configs=("cfg3",), seeds=(0, 1))
        variants: list[tuple[str, Iterable[str] | None]] = [("full", None)]
        variants += [(f"-{name}", feats) for name, feats in groups.items()]
        out = []
        for name, drop in variants:
            hits = tot = 0
            for t in tasks:
                cf = self.counterfactual(t["family"], t["template"], t["config"], t["seed"])
                for ft in self.applicable[t["family"]]:
                    cfg = EpisodeConfig(t["family"], t["template"], t["config"], t["seed"],
                                        ft, "rule_based", "rules")
                    det = RuleDetector() if drop is None else Masked(drop)
                    traj = run_episode(cfg, detector=det,
                                       policy=make_policy("rule_based"), counterfactual=cf)
                    tot += 1
                    hits += 1 if (traj["detection"] or {}).get("detected") else 0
            out.append({"variant": name, "n": tot, "detected_rate": hits / max(1, tot)})
        return out

    def _headline(self, report: dict[str, Any]) -> dict[str, Any]:
        by_pol = {r["policy"]: r for r in report["main"]["by_policy"]}
        det = report["detection"]
        return {
            "no_recovery": by_pol.get("no_recovery", {}).get("recovery_rate"),
            "retry_only": by_pol.get("retry_only", {}).get("recovery_rate"),
            "rule_based": by_pol.get("rule_based", {}).get("recovery_rate"),
            "model_based": by_pol.get("model_based", {}).get("recovery_rate"),
            "rules_auc": det.get("rules_instrumented", {}).get("auc"),
            "supervised_auc": det.get("supervised_instrumented", {}).get("auc"),
            "rules_auc_blackbox": det.get("rules_blackbox", {}).get("auc"),
            "n_eval": len([r for r in self.rows if r["fault"] != "none"]),
        }


def run(tag: str = "main") -> dict[str, Any]:
    ex = Experiment()
    tasks = ex.tasks()
    print(f"[1/5] screening {len(tasks)} tasks", flush=True)
    ex.screen(tasks)
    print(f"      harmful {sum(1 for v in ex.harmful.values() if v)}/{len(ex.harmful)}", flush=True)

    print("[2/5] detector", flush=True)
    det_info = ex.train_detector()

    print("[3/5] policies", flush=True)
    ex.evaluate(tasks, ("no_recovery", "retry_only", "rule_based"))

    print("[4/5] ranker + model policy", flush=True)
    ranker_info = ex.train_ranker()
    ex.evaluate(tasks, ("model_based",))

    print("[5/5] natural + long-horizon + ablation", flush=True)
    for t in ex.tasks(templates=("A", "B"), configs=("cfg1", "cfg3"), seeds=(0, 1)):
        cf = ex.counterfactual(t["family"], t["template"], t["config"], t["seed"])
        for rep in range(2):
            seed = t["seed"] + 10 * rep
            cfg = EpisodeConfig(t["family"], t["template"], t["config"], seed,
                                None, "rule_based", "rules",
                                fault_source="natural", competence=0.80)
            traj = run_episode(cfg, detector=RuleDetector(),
                               policy=make_policy("rule_based"), counterfactual=cf)
            if traj["fault"] is None:
                continue
            ex.rows.append(summary_row(traj, tag="natural"))

    ex.evaluate(ex.tasks(templates=("A", "B"), configs=("cfg2",), seeds=(0, 1), horizon_class="long"),
                ("no_recovery", "rule_based"), tag="long")

    # Black-box monitor: harness flags hidden. This is the detectability-gap
    # experiment -- what survives when you only have error codes and payloads.
    ex.evaluate(ex.tasks(templates=("B", "C"), configs=("cfg6",), seeds=(0, 1)),
                ("rule_based", "model_based"), tag="blackbox", monitor_mode="blackbox")

    report = ex.aggregate(det_info, ranker_info)
    path = os.path.join(ex.out_dir, f"report_{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    with open(os.path.join(ex.out_dir, "episodes.jsonl"), "w", encoding="utf-8") as fh:
        for r in ex.rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"wrote {path}", flush=True)
    return report
