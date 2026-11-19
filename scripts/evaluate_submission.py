#!/usr/bin/env python
"""Score a submission against the ARB protocol.

    python scripts/evaluate_submission.py \
        --submission data/arb_dataset.jsonl \
        --baseline results/episodes.jsonl \
        --out results/eval_report.json

The baseline file supplies the frozen harmful set (its no-recovery episodes). The
submission supplies the episodes being scored. Neither file's claimed outcomes are
used for scoring: only the grader-verified ``success`` flag is.

With ``--compare`` a second submission is scored and the recovery difference is
reported with a paired cluster bootstrap on the shared pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arb.evaluator import load_episodes, paired, score, to_markdown  # noqa: E402


def _baseline_rows(path: str) -> list[dict]:
    rows = load_episodes(path)
    keep = [r for r in rows if r.get("policy") in (None, "no_recovery")]
    if not keep:
        raise SystemExit(f"{path}: no no_recovery episodes; cannot derive the harmful set")
    return keep


def _submission_rows(path: str, policy: str | None) -> list[dict]:
    rows = load_episodes(path)
    if policy:
        rows = [r for r in rows if r.get("policy") == policy]
        if not rows:
            raise SystemExit(f"{path}: no episodes for policy={policy!r}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--baseline", default="results/episodes.jsonl")
    ap.add_argument("--policy", default=None,
                    help="score only this policy from the submission file")
    ap.add_argument("--compare", default=None,
                    help="second submission file to compare against")
    ap.add_argument("--out", default=None)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    baseline = _baseline_rows(args.baseline)
    rows = _submission_rows(args.submission, args.policy)

    report = score(rows, baseline)
    report["submission"] = args.submission
    report["baseline"] = args.baseline
    report["policy_filter"] = args.policy

    if args.compare:
        other = _submission_rows(args.compare, args.policy)
        report["comparison"] = paired(other, rows, baseline,
                                      label_a=os.path.basename(args.compare),
                                      label_b=os.path.basename(args.submission))

    if report.get("status") != "scored":
        print(json.dumps(report, indent=1))
        return 2

    o = report["overall"]
    print(f"pairs covered : {report['n_pairs_covered']}/{report['n_harmful_pairs']} "
          f"({100 * report['coverage']:.0f}%)")
    print(f"policy        : {report['policy']}")
    print(f"recovery      : {100 * o['recovery_rate']:.1f}% "
          f"[{100 * o['recovery_ci'][0]:.1f}, {100 * o['recovery_ci'][1]:.1f}]")
    print(f"detected      : {100 * (o['detected_rate'] or 0):.1f}%")
    if o["containment_rate"] is None:
        print("containment   : n/a (no unrecoverable faults in the covered set)")
    else:
        print(f"containment   : {100 * o['containment_rate']:.1f}% "
              f"(n={o['n_unrecoverable']})")
    print(f"self-report   : claimed {100 * o['claimed_rate']:.1f}% vs verified "
          f"{100 * o['verified_rate']:.1f}% (gap {100 * o['selfreport_gap']:+.1f}pp)")
    upr = ("--" if o["usd_per_recovered"] is None
           else f"{o['usd_per_recovered']:.4f}")
    print(f"USD/episode   : {o['usd_per_episode']:.4f}   USD/recovered: {upr}")
    if "detection" in o:
        d = o["detection"]
        print(f"step-level    : AUC {d['auc']:.3f}  P {100 * d['precision']:.1f}%  "
              f"R {100 * d['recall']:.1f}%  F1 {100 * d['f1']:.1f}%  (n={d['n_steps']} steps)")
    for w in report.get("warnings", []):
        print(f"warning       : {w}")
    if "comparison" in report and report["comparison"].get("status") == "ok":
        c = report["comparison"]
        print(f"\npaired vs {c['label_a']}: {100 * c['diff']:+.1f}pp on {c['n_pairs']} pairs "
              f"(95% CI [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}])")

    if args.markdown:
        print("\n" + to_markdown(report))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
