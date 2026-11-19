#!/usr/bin/env python
"""Run the ARB experiment grid end to end.

    python scripts/run_experiments.py --preset main
    python scripts/run_experiments.py --preset main --out results

Every number in the paper comes from the report this writes. The preset also
loads its YAML config so the run records exactly which grid produced it; the
grid itself lives in ``arb.experiments`` because it must stay versioned with the
harness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arb.experiments import run  # noqa: E402

PRESETS = {
    "main": ("configs/main.yaml", "full grid: screen, detector, 4 policies, ranker, slices"),
    "smoke": ("configs/smoke.yaml", "two tasks per family, one seed: wiring check only"),
}


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="main", choices=sorted(PRESETS))
    ap.add_argument("--out", default="results", help="output directory for reports")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg_path, desc = PRESETS[args.preset]
    cfg = load_config(cfg_path)
    print(f"preset={args.preset} ({desc})")
    print(f"config={cfg_path} version={cfg.get('version', 'n/a')}")

    t0 = time.time()
    report = run(args.preset if args.preset == "main" else "main")
    elapsed = time.time() - t0

    report["run"] = {
        "preset": args.preset,
        "config": cfg,
        "elapsed_seconds": round(elapsed, 1),
        "python": sys.version.split()[0],
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"report_{args.preset}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)

    h = report["headline"]
    if not args.quiet:
        print("\n== headline ==")
        print(f"  recovery  no_recovery={h['no_recovery']:.3f} retry={h['retry_only']:.3f} "
              f"rules={h['rule_based']:.3f} model={h['model_based']:.3f}")
        print(f"  detection rules_auc={h['rules_auc']:.3f} supervised_auc={h['supervised_auc']:.3f}")
        print(f"  episodes={h['n_eval']}")
    print(f"\nwrote {path} in {elapsed:.0f}s")
    print("next: python paper/analysis.py && latexmk -pdf paper/main.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
