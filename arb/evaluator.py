"""Reference evaluator for the ARB leaderboard protocol.

Scores a submission against a fixed baseline grid and reports exactly the five
numbers the protocol requires:

  1. recovery rate on harmful pairs (cluster bootstrap over task templates)
  2. detection: step-level AUC, precision/recall/F1, class accuracy among
     positive steps
  3. USD per recovered episode under the pinned price sheet
  4. containment rate on unrecoverable external errors
  5. the self-report gap (claimed minus verified success)

Design rules that the rest of the benchmark depends on:

* the harmful set comes from the **baseline**, never from the submission, so an
  entrant cannot shrink its own denominator by claiming a fault was harmless;
* ``success`` must be the grader's state comparison. A submission that only
  reports ``claimed_success`` is rejected, not scored;
* episodes are the unit of resampling only within a task template; templates are
  the cluster, because episodes sharing a template share the procedure.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

from arb.metrics import bootstrap_ci, ece, mean, paired_bootstrap_diff, prf, roc_auc


def load_episodes(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _key(row: dict[str, Any]) -> tuple:
    """Task identity, tolerant of both the trajectory and the summary schema.

    Released episodes carry ``template_id``/``config_id``; ``results/episodes.jsonl``
    carries ``template``/``config``. Getting this wrong would silently merge
    different tasks into one cluster.
    """
    return (
        row.get("family"),
        row.get("template_id", row.get("template")),
        row.get("config_id", row.get("config")),
        row.get("seed"),
    )


def _first(row: dict[str, Any], name: str, *blocks: str) -> Any:
    """First non-null ``name`` on the row itself or inside the named sub-blocks.

    A submission may report ``success`` at the top level or nested under
    ``outcome``; both are accepted, and a JSON-string field (``labels_json``,
    ``posterior_json``) is read the same way as its native counterpart.
    """
    for src in (row, *(row.get(b) for b in blocks)):
        if isinstance(src, dict) and src.get(name) is not None:
            return src[name]
    return None


def _fault(row: dict[str, Any]) -> str:
    f = row.get("fault")
    if isinstance(f, dict):
        return f.get("type") or "none"
    return f or "none"


def _success(row: dict[str, Any]) -> bool | None:
    """Grader-verified success. ``None`` means the submission did not supply it."""
    v = _first(row, "success", "outcome")
    return v if isinstance(v, bool) else None


def _claimed(row: dict[str, Any]) -> bool:
    return bool(_first(row, "claimed_success", "outcome"))


def _detected(row: dict[str, Any]) -> bool:
    return bool(_first(row, "detected", "detection", "outcome"))


def _contained(row: dict[str, Any]) -> bool:
    return bool(_first(row, "contained", "recovery"))


def _cost(row: dict[str, Any], name: str, flat: str | None = None) -> float:
    """A cost field from the nested ``cost`` block, else from a flat column."""
    v = (row.get("cost") or {}).get(name)
    if v is None and flat:
        v = row.get(flat)
    return float(v or 0.0)


def _rate(rows: Sequence[dict[str, Any]], pred: Callable[[dict[str, Any]], bool]) -> float:
    return mean([1.0 if pred(r) else 0.0 for r in rows])


def harmful_keys(baseline: Iterable[dict[str, Any]]) -> set[tuple]:
    """A (task, fault) pair is harmful iff the no-recovery run fails.

    This is the frozen screen from the paper. Inert pairs are excluded so that no
    submission is rewarded for ``detecting`` a fault that changed nothing.
    """
    out = set()
    for r in baseline:
        if r.get("policy") not in (None, "no_recovery"):
            raise ValueError("baseline file must contain only no_recovery episodes, "
                             f"found policy={r.get('policy')!r}")
        if _fault(r) == "none" or r.get("success", False):
            continue
        out.add((*_key(r), _fault(r)))
    return out


def _detection(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Step-level detection metrics from ``meta.labels`` when a submission has them."""
    labels: list[dict[str, Any]] = []
    for r in rows:
        lab = _first(r, "labels", "meta")
        if lab is None:
            raw = _first(r, "labels_json", "meta")
            lab = json.loads(raw) if raw else []
        labels.extend(lab)
    if not labels:
        return {}

    ys = [0 if lab.get("label") == "none" else 1 for lab in labels]
    scores = [float(lab.get("score") or 0.0) for lab in labels]
    preds = [lab.get("pred") if isinstance(lab.get("pred"), str) else "none" for lab in labels]
    tp = sum(1 for y, p in zip(ys, preds) if y and p != "none")
    fp = sum(1 for y, p in zip(ys, preds) if not y and p != "none")
    fn = sum(1 for y, p in zip(ys, preds) if y and p == "none")
    positive = [(y, p) for y, p in zip(ys, preds) if y]
    return {
        "n_steps": len(ys),
        "n_positive": sum(ys),
        "auc": roc_auc(ys, scores),
        **prf(tp, fp, fn),
        "type_acc_any_positive": (
            sum(1 for _, p in positive if p != "none") / len(positive) if positive else 0.0
        ),
        "ece": ece(ys, scores),
    }


def score(
    submission: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    *,
    require_state_verified: bool = True,
) -> dict[str, Any]:
    harmful = harmful_keys(baseline)

    if require_state_verified:
        errors = [
            f"{r.get('episode_id', '?')}: missing grader-verified 'success' "
            "(claims are not scored)"
            for r in submission if _success(r) is None
        ]
        if errors:
            return {"status": "rejected", "errors": errors[:20], "n_errors": len(errors)}

    warnings = [f"unknown monitor_mode={r.get('monitor_mode')!r}" for r in submission
                if r.get("monitor_mode") not in (None, "instrumented", "blackbox")]

    rows = [r for r in submission if (*_key(r), _fault(r)) in harmful]
    covered = {(*_key(r), _fault(r)) for r in rows}
    missing = harmful - covered
    if not rows:
        return {"status": "rejected",
                "errors": ["no submission episode matches the frozen harmful set"],
                "missing_pairs": len(missing)}

    def block(g: Sequence[dict[str, Any]]) -> dict[str, Any]:
        ci = bootstrap_ci([1.0 if _success(r) else 0.0 for r in g],
                          [str(_key(r)[1]) for r in g], seed=0)
        recovered = [r for r in g if _success(r)]
        unrecoverable = [r for r in g if _fault(r) == "unrecoverable_external_error"]
        total_usd = sum(_cost(r, "usd") for r in g)
        out = {
            "n": len(g),
            "recovery_rate": ci["estimate"],
            "recovery_ci": [ci["lo"], ci["hi"]],
            "detected_rate": _rate(g, _detected),
            "containment_rate": _rate(unrecoverable, _contained) if unrecoverable else None,
            "n_unrecoverable": len(unrecoverable),
            "claimed_rate": _rate(g, _claimed),
            "verified_rate": ci["estimate"],
            "selfreport_gap": (_rate(g, _claimed) or 0.0) - ci["estimate"],
            "usd_total": round(total_usd, 6),
            "usd_per_episode": round(total_usd / len(g), 6),
            "usd_per_recovered": round(total_usd / len(recovered), 6) if recovered else None,
            "tool_calls_mean": mean([_cost(r, "tool_calls", "tool_calls") for r in g]),
            "tokens_mean": mean([_cost(r, "tokens_total", "tokens") for r in g]),
        }
        det = _detection(g)
        if det:
            out["detection"] = det
        return out

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_mode[r.get("monitor_mode") or "instrumented"].append(r)

    report = {
        "status": "scored",
        "policy": sorted({r.get("policy") for r in rows}),
        "n_harmful_pairs": len(harmful),
        "n_pairs_covered": len(covered),
        "n_missing_pairs": len(missing),
        "coverage": len(covered) / len(harmful) if harmful else 0.0,
        "overall": block(rows),
        "by_monitor_mode": {m: block(g) for m, g in by_mode.items()},
        "by_class": {
            ft: {"n": len(g := [r for r in rows if _fault(r) == ft]),
                 "recovery_rate": _rate(g, _success),
                 "detected_rate": _rate(g, _detected)}
            for ft in sorted({_fault(r) for r in rows})
        },
        "by_family": {
            fam: {"n": len(g := [r for r in rows if r.get("family") == fam]),
                  "recovery_rate": _rate(g, _success)}
            for fam in sorted({r.get("family") for r in rows})
        },
        "protocol_checks": {
            "state_verified_outcomes": require_state_verified,
            "harmful_set_from_baseline": True,
            "cluster_unit": "task template",
            "containment_not_recovery": True,
            "claims_not_scored": True,
        },
    }
    if missing:
        warnings.append(f"{len(missing)} harmful pairs have no submission episode; "
                        "recovery is reported over covered pairs only")
    if len(covered) < 0.95 * len(harmful):
        warnings.append("coverage below 95%: entry is incomplete for the frozen set")
    report["warnings"] = warnings
    return report


def paired(
    submission_a: Sequence[dict[str, Any]],
    submission_b: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, Any]:
    """Paired bootstrap of recovery difference on the shared harmful set."""
    harmful = harmful_keys(baseline)

    def index(rows: Sequence[dict[str, Any]]) -> dict[tuple, float]:
        return {k: 1.0 if _success(r) else 0.0
                for r in rows if (k := (*_key(r), _fault(r))) in harmful}

    ia, ib = index(submission_a), index(submission_b)
    shared = sorted(set(ia) & set(ib))
    if not shared:
        return {"status": "no_overlap"}
    a = [ia[k] for k in shared]
    b = [ib[k] for k in shared]
    d = paired_bootstrap_diff(a, b, [str(k[1]) for k in shared], seed=0)
    return {
        "status": "ok",
        "label_a": label_a,
        "label_b": label_b,
        "n_pairs": len(shared),
        "mean_a": sum(a) / len(a),
        "mean_b": sum(b) / len(b),
        "diff": d["diff"],
        "ci": [d["lo"], d["hi"]],
    }


def _pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.1f}%"


def to_markdown(report: dict[str, Any]) -> str:
    if report.get("status") != "scored":
        return f"**Rejected**: {report.get('errors')}"
    o = report["overall"]
    lines = [
        "| metric | value |",
        "|---|---|",
        f"| recovery rate | {_pct(o['recovery_rate'])} "
        f"[{100 * o['recovery_ci'][0]:.1f}, {100 * o['recovery_ci'][1]:.1f}] |",
        f"| detected | {_pct(o['detected_rate'])} |",
        f"| containment (unrecoverable) | {_pct(o['containment_rate'])} |",
        f"| self-report gap | {100 * o['selfreport_gap']:.1f} pp |",
        f"| USD / episode | {o['usd_per_episode']:.4f} |",
        f"| USD / recovered | "
        f"{'--' if o['usd_per_recovered'] is None else round(o['usd_per_recovered'], 4)} |",
        f"| pairs covered | {report['n_pairs_covered']}/{report['n_harmful_pairs']} "
        f"({100 * report['coverage']:.0f}%) |",
    ]
    det = o.get("detection")
    if det:
        lines += [
            "",
            f"Detection at step level (n={det['n_steps']}, positives={det['n_positive']}): "
            f"AUC {det['auc']:.3f}, precision {100 * det['precision']:.1f}%, "
            f"recall {100 * det['recall']:.1f}%, F1 {100 * det['f1']:.1f}%.",
        ]
    return "\n".join(lines)
