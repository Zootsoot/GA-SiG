"""Search of the smallest reduced mechanism within an error tolerance.

The rank fixes the removal order, so the candidate mechanisms form one chain
indexed by the size ``n``. The error ``err(n)`` is not monotonic in ``n``:
solver noise moves it by a few tenths of a percent, and error cancellation
can lower it after a removal that raised it. This module holds the search
logic without Cantera, so that it runs on synthetic curves in the tests:

* :func:`coarse_sizes`: the sizes of the coarse pass, denser at small sizes
* :func:`gap_probes`: the sizes on both sides of the largest importance gaps
* :func:`adaptive_search`: coarse pass, probes, bisection of every crossing,
  near-miss refinement, noise band, robust window, evaluation budget
* :func:`select_optima`: the strict and the robust optimum of an evaluated
  set of sizes

Two optima
----------
An isolated size that passes while its neighbours fail passes because of
error cancellation at the evaluation conditions, so it is not reliable at
other conditions. The strict optimum is the smallest evaluated size within
the tolerance. The robust optimum is the smallest size such that every size
in ``[n, n + window]`` passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SearchParams:
    coarse_points: int = 16
    importance_gap_probes: int = 5
    near_miss_factor: float = 2.0
    robust_window: int = 3
    noise: float = 0.003
    max_evaluations: int | None = None


@dataclass
class SearchResult:
    errors: dict[int, float]            # size -> maximum relative error (inf: no ignition)
    strict_optimum: int | None
    robust_optimum: int | None
    unresolved: list[dict] = field(default_factory=list)   # intervals that may hide a smaller passing size
    budget_exhausted: bool = False

    @property
    def evaluations(self) -> int:
        return len(self.errors)


class BudgetExhausted(Exception):
    """Raised inside the search when ``max_evaluations`` is reached."""


def coarse_sizes(n_max: int, n_min: int, points: int) -> list[int]:
    """Sizes of the coarse pass from ``n_max`` down to ``n_min``, denser near ``n_min``.

    The spacing is quadratic: the k-th of ``points`` sizes is
    ``n_min + span (1 - k / (points - 1))^2``. Both ends are included. When the
    chain has ``points`` sizes or fewer, every size is returned.
    """
    if n_min > n_max:
        raise ValueError(f"n_min {n_min} is above n_max {n_max}.")
    if points < 2:
        raise ValueError("points must be at least 2.")
    span = n_max - n_min
    if span + 1 <= points:
        return list(range(n_max, n_min - 1, -1))
    sizes = set()
    for k in range(points):
        f = k / (points - 1)
        sizes.add(n_min + int(round(span * (1.0 - f) ** 2)))
    return sorted(sizes, reverse=True)


def gap_probes(importance: list[float], n_max: int, n_min: int, count: int, floor: float = 1e-300) -> list[int]:
    """Sizes on both sides of the ``count`` largest importance gaps of the removal order.

    ``importance[i]`` belongs to the i-th species of the removal order (least
    important first). The removal of the species 0 to i leaves the size
    ``n_max - (i + 1)``; the gap between the species i and i + 1 is probed at
    that size and at ``n_max - (i + 2)``. Gaps that are not positive are
    skipped.
    """
    if count <= 0 or len(importance) < 2:
        return []
    logs = [math.log(max(float(v), floor)) for v in importance]
    gaps = sorted(((logs[i + 1] - logs[i], i) for i in range(len(logs) - 1)), reverse=True)
    sizes = set()
    for gap, i in gaps[:count]:
        if gap <= 0.0:
            break
        for n in (n_max - (i + 1), n_max - (i + 2)):
            if n_min <= n <= n_max:
                sizes.add(n)
    return sorted(sizes, reverse=True)


def select_optima(errors: dict[int, float], tol: float, window: int,
                  require_full_window: bool = False) -> tuple[int | None, int | None]:
    """The strict and the robust optimum of the evaluated sizes.

    Strict: the smallest size with ``err <= tol``. Robust: the smallest size
    ``n`` such that every evaluated size in ``[n, n + window]`` passes. With
    ``require_full_window`` every size of the window must be evaluated. The
    largest size (the detailed mechanism) is not a candidate.
    """
    n_max = max(errors) if errors else 0

    def passes(n: int) -> bool:
        return errors[n] <= tol

    passing = sorted(n for n in errors if passes(n) and n != n_max)
    strict = passing[0] if passing else None
    robust = None
    for n in passing:
        top = min(n + window, n_max)
        if require_full_window and any(m not in errors for m in range(n, top + 1)):
            continue
        if all(passes(m) for m in errors if n <= m <= top):
            robust = n
            break
    return strict, robust


def adaptive_search(evaluate: Callable[[int], float], n_max: int, n_min: int, tol: float,
                    probes: list[int] | None = None, params: SearchParams | None = None) -> SearchResult:
    """Find the smallest size within ``tol`` along the chain with few evaluations.

    ``evaluate(n)`` returns the maximum relative error of the mechanism with
    ``n`` species; ``inf`` means that a condition did not ignite. The search
    evaluates every size at most once.
    """
    params = params or SearchParams()
    if n_min > n_max:
        raise ValueError(f"n_min {n_min} is above n_max {n_max}.")
    errors: dict[int, float] = {}
    limit = params.max_evaluations if params.max_evaluations is not None else math.inf

    def ev(n: int) -> float:
        if n in errors:
            return errors[n]
        if len(errors) >= limit:
            raise BudgetExhausted()
        errors[n] = float(evaluate(n))
        return errors[n]

    def passes(n: int) -> bool:
        return errors[n] <= tol

    exhausted = False
    robust: int | None = None
    try:
        for n in coarse_sizes(n_max, n_min, params.coarse_points):
            ev(n)
        for n in probes or []:
            if n_min <= n <= n_max:
                ev(n)
        while True:
            new = False
            sizes = sorted(errors)
            best = min((n for n in sizes if passes(n)), default=None)
            for lo, hi in zip(sizes, sizes[1:]):
                if hi - lo <= 1:
                    continue
                crossing = passes(lo) != passes(hi)
                near_miss = (not passes(lo) and not passes(hi) and (best is None or hi <= best)
                             and min(errors[lo], errors[hi]) <= params.near_miss_factor * tol)
                if crossing or near_miss:
                    ev((lo + hi) // 2)
                    new = True
            for n in sizes:
                if abs(errors[n] - tol) <= params.noise:
                    for m in (n - 1, n + 1):
                        if n_min <= m <= n_max and m not in errors:
                            ev(m)
                            new = True
            if not new:
                break
        # Robust window: the search confirms the window itself, so a robust optimum is
        # reported only when every size of its window was evaluated and passed.
        checked: set[int] = set()
        while True:
            candidates = [n for n in sorted(errors) if passes(n) and n != n_max and n not in checked]
            if not candidates:
                break
            n = candidates[0]
            checked.add(n)
            top = min(n + params.robust_window, n_max)
            for m in range(n + 1, top + 1):
                ev(m)
            if all(passes(m) for m in range(n, top + 1)):
                robust = n
                break
    except BudgetExhausted:
        exhausted = True

    strict, _ = select_optima(errors, tol, params.robust_window, require_full_window=True)
    unresolved = []
    sizes = sorted(errors)
    for lo, hi in zip(sizes, sizes[1:]):
        if hi - lo <= 1:
            continue
        if passes(lo) != passes(hi):
            unresolved.append({"from": lo, "to": hi, "kind": "crossing",
                               "smallest_error": min(errors[lo], errors[hi])})
        elif not passes(lo) and (strict is None or hi <= strict):
            unresolved.append({"from": lo, "to": hi, "kind": "gap",
                               "smallest_error": min(errors[lo], errors[hi])})
    return SearchResult(errors=errors, strict_optimum=strict, robust_optimum=robust,
                        unresolved=unresolved, budget_exhausted=exhausted)
