"""Detection metrics: discrimination, calibration, and clustered uncertainty.

Uncertainty is reported with a **cluster bootstrap over task templates** (with
seeds nested inside templates), not over episodes. Episodes generated from the
same template are not independent draws -- they share the scenario generator --
so an episode-level interval would be too narrow. This is the interval reported
everywhere in the paper.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUC with correct handling of ties (Mann-Whitney U)."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def confusion(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> dict[str, dict[str, int]]:
    m = {a: {b: 0 for b in classes} for a in classes}
    for t, p in zip(y_true, y_pred):
        if t in m and p in m[t]:
            m[t][p] += 1
    return m


def brier(labels: Sequence[int], probs: Sequence[float]) -> float:
    if not labels:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(labels)


def ece(labels: Sequence[int], probs: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error over equal-width bins."""
    if not labels:
        return float("nan")
    total = len(labels)
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        err += (len(idx) / total) * abs(conf - acc)
    return err


def bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    stat: Callable[[Sequence[float]], float] = mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster bootstrap: resample whole clusters (task templates) with replacement."""
    if not values:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_clusters": 0}
    by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    keys = sorted(by_cluster)
    point = stat(list(values))
    if len(keys) < 2:
        return {"estimate": point, "lo": float("nan"), "hi": float("nan"),
                "n": len(values), "n_clusters": len(keys)}
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in keys:
            k = keys[rng.randrange(len(keys))]
            sample.extend(values[i] for i in by_cluster[k])
        if sample:
            draws.append(stat(sample))
    draws.sort()
    lo = draws[int((alpha / 2) * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return {"estimate": point, "lo": lo, "hi": hi, "n": len(values), "n_clusters": len(keys)}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# Paired significance tests on aligned episode-level outcomes
# --------------------------------------------------------------------------- #
def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict[str, Any]:
    """Exact (binomial) McNemar test for two systems scored on the same episodes.

    Only the discordant pairs carry information: ``a_only`` counts episodes the first
    system recovered and the second did not, ``b_only`` the reverse. The two-sided
    p-value is an exact binomial test on those pairs at p = 0.5, which stays valid at
    the sample sizes here and needs no chi-square approximation.

    This tests *equality of the paired outcomes*; it is not an interval for the
    difference. Effect sizes and intervals come from ``paired_bootstrap_diff``.
    """
    if len(a) != len(b):
        raise ValueError(f"aligned outcomes required, got {len(a)} and {len(b)}")
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
        p = min(1.0, 2.0 * tail)
    return {"n": len(a), "a_only": a_only, "b_only": b_only,
            "n_discordant": n, "p_value": p}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm--Bonferroni step-down adjustment over one family of tests.

    Holm controls the family-wise error rate without assuming independence and is
    uniformly at least as powerful as Bonferroni. The adjusted values are forced
    non-decreasing so a smaller raw p never gets a larger adjusted p.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out


def cluster_bootstrap_rows(
    rows: Sequence[Any],
    clusters: Sequence[str],
    stats: dict[str, Callable[[Sequence[Any]], float]],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Cluster bootstrap for several statistics computed on the same rows.

    Whole clusters (task templates) are resampled with replacement. With few
    clusters the resampled datasets repeat, so each distinct composition is
    evaluated once and reused -- the interval is identical to a naive loop, without
    recomputing a sort-heavy statistic a thousand times.

    ``stats`` maps a name to a function of a list of rows; a statistic may return
    ``nan`` for a degenerate resample (e.g. AUC needs both classes), and those draws
    are dropped rather than coerced.
    """
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        by_cluster[c].append(i)
    keys = sorted(by_cluster)
    point = {name: fn(list(rows)) for name, fn in stats.items()}
    result: dict[str, Any] = {"point": point, "n": len(rows),
                              "n_clusters": len(keys), "ci": {}}
    if len(keys) < 2:
        result["ci"] = {name: [float("nan")] * 2 for name in stats}
        return result

    rng = random.Random(seed)
    cache: dict[tuple, dict[str, float]] = {}
    draws: dict[str, list[float]] = {name: [] for name in stats}
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        sig = tuple(sorted((k, pick.count(k)) for k in set(pick)))
        if sig not in cache:
            idx = [i for k, c in sig for i in by_cluster[k]]
            cache[sig] = {name: fn([rows[i] for i in idx]) for name, fn in stats.items()}
        for name, value in cache[sig].items():
            if value == value:  # drop nan resamples
                draws[name].append(value)

    for name, values in draws.items():
        if len(values) < 2:
            result["ci"][name] = [float("nan")] * 2
            continue
        values.sort()
        result["ci"][name] = [values[int((alpha / 2) * len(values))],
                              values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]]
    result["n_valid_draws"] = len(next(iter(draws.values()))) if draws else 0
    return result


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], clusters: Sequence[str], *, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Paired cluster bootstrap of mean(a) - mean(b) on the same episodes."""
    assert len(a) == len(b) == len(clusters)
    by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    keys = sorted(by_cluster)
    point = mean(a) - mean(b)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        idx: list[int] = []
        for _ in keys:
            idx.extend(by_cluster[keys[rng.randrange(len(keys))]])
        draws.append(mean([a[i] for i in idx]) - mean([b[i] for i in idx]))
    draws.sort()
    return {
        "diff": point,
        "lo": draws[int(0.025 * len(draws))],
        "hi": draws[min(len(draws) - 1, int(0.975 * len(draws)))],
        "p_two_sided": 2 * min(
            sum(1 for d in draws if d <= 0) / len(draws),
            sum(1 for d in draws if d >= 0) / len(draws),
        ),
    }
