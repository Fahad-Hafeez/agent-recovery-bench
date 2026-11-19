#!/usr/bin/env python
"""Cluster-bootstrap intervals for the step-level detection table.

    python scripts/detection_intervals.py

``results/episodes.jsonl`` stores one summary row per episode, so the step-level
predictions behind Table~\\ref{tab:detection} are not in any artifact. This script
replays the detector arm of the main grid and recomputes that table, then attaches
95% cluster-bootstrap intervals over task templates.

Two guarantees keep the intervals describing the *published* run:

* the replay must reproduce every point estimate in ``results/report_main.json``
  exactly (rules are deterministic; the supervised arm reuses the stored model and
  the report's calibrated threshold), and the script exits non-zero otherwise;
* it writes only ``results/detection_ci.json`` -- never the report, ``results/episodes.jsonl``
  or ``models/``.

With three templates the bootstrap has few clusters, so each interval is coarse; the
artifact records ``n_clusters`` next to every interval rather than hiding it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arb.detectors.ranker import RecoveryRanker  # noqa: E402
from arb.detectors.supervised import SupervisedDetector  # noqa: E402
from arb.experiments import Experiment  # noqa: E402
from arb.metrics import cluster_bootstrap_rows, prf, roc_auc  # noqa: E402

METRICS = ("precision", "recall", "f1", "auc", "type_accuracy")


def metrics(rows: list[dict]) -> dict[str, float]:
    """Step-level detection metrics, identical to ``Experiment._detection``."""
    y = [0 if s["label"] == "none" else 1 for s in rows]
    yhat = [0 if s["pred"] == "none" else 1 for s in rows]
    tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
    positive = [s for s in rows if s["label"] != "none"]
    hit = sum(1 for s in positive if s["pred"] == s["label"])
    return {
        **prf(tp, fp, fn),
        "auc": roc_auc(y, [s["score"] for s in rows]),
        "type_accuracy": hit / max(1, len(positive)),
        "n_steps": float(len(rows)),
    }


def replay(out_dir: str = "results", detector_path: str = "models/detector.json",
           ranker_path: str = "models/recovery_ranker.json") -> dict[str, list[dict]]:
    """Re-run the detection arm of the main grid and return steps per arm."""
    with open(os.path.join(out_dir, "report_main.json"), encoding="utf-8") as fh:
        report = json.load(fh)

    ex = Experiment(out_dir=out_dir)
    tasks = ex.tasks()
    print(f"  screening {len(tasks)} tasks", flush=True)
    ex.screen(tasks)

    # Never refit: reuse the artefacts the report was produced with.
    threshold = report["detector_model"]["threshold"]
    ex.detector = SupervisedDetector.load(detector_path, threshold=threshold)
    ex.ranker = RecoveryRanker.load(ranker_path)

    print("  policy pass", flush=True)
    ex.evaluate(tasks, ("no_recovery", "retry_only", "rule_based"))
    ex.evaluate(tasks, ("model_based",))
    print("  long-horizon pass", flush=True)
    ex.evaluate(ex.tasks(templates=("A", "B"), configs=("cfg2",), seeds=(0, 1),
                         horizon_class="long"),
                ("no_recovery", "rule_based"), tag="long")
    print("  blackbox pass", flush=True)
    ex.evaluate(ex.tasks(templates=("B", "C"), configs=("cfg6",), seeds=(0, 1)),
                ("rule_based", "model_based"), tag="blackbox", monitor_mode="blackbox")

    arms: dict[str, list[dict]] = {}
    for det in ("rules", "supervised"):
        for mode in ("instrumented", "blackbox"):
            rows = [s for s in ex.eval_steps
                    if s["detector"] == det and s["mode"] == mode]
            if rows:
                arms[f"{det}_{mode}"] = rows
    return {"report": report, "arms": arms}


def check(arm: str, rows: list[dict], expected: dict) -> None:
    """Refuse to publish intervals for a replay that is not the reported run."""
    got = metrics(rows)
    keys = ("precision", "recall", "f1", "auc", "type_accuracy")
    bad = [(k, got[k], expected[k]) for k in keys
           if abs(got[k] - expected[k]) > 1e-9]
    if abs(got["n_steps"] - expected["n_steps"]) > 1e-9:
        bad.append(("n_steps", got["n_steps"], expected["n_steps"]))
    if bad:
        detail = "\n".join(f"    {k}: replay={g!r} report={e!r}" for k, g, e in bad)
        raise SystemExit(
            f"{arm}: replay does not reproduce results/report_main.json\n{detail}\n"
            "Refusing to attach intervals to a different run."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    replayed = replay(args.out)
    report = replayed["report"]
    arms = replayed["arms"]

    payload: dict = {
        "source": os.path.join(args.out, "report_main.json"),
        "cluster_unit": "task template",
        "n_boot": args.boot,
        "seed": args.seed,
        "arms": {},
    }
    for arm, rows in arms.items():
        expected = report["detection"].get(arm)
        if expected is None:
            continue
        check(arm, rows, expected)
        stats = {name: (lambda rs, n=name: metrics(rs)[n] if rs else float("nan"))
                 for name in METRICS}
        res = cluster_bootstrap_rows(rows, [s["template"] for s in rows], stats,
                                     n_boot=args.boot, seed=args.seed)
        payload["arms"][arm] = {
            "point": {k: res["point"][k] for k in METRICS},
            "ci": res["ci"],
            "n_steps": int(res["n"]),
            "n_clusters": res["n_clusters"],
            "clusters": sorted(Counter(s["template"] for s in rows)),
        }
        ci = res["ci"]
        print(f"  {arm:24s} n={len(rows):6d} clusters={res['n_clusters']}  "
              f"AUC {res['point']['auc']:.3f} [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}]  "
              f"F1 {100 * res['point']['f1']:.1f} "
              f"[{100 * ci['f1'][0]:.1f}, {100 * ci['f1'][1]:.1f}]", flush=True)

    path = os.path.join(args.out, "detection_ci.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
