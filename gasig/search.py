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
* :func:`greedy_search`: one species per test in rank order
* :func:`lk_greedy_search`: pieces (chain units and single species) of the
  LK order, tier-0 pieces in blocks that are halved on failure, and an
  optional second pass at a second tolerance

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


@dataclass
class GreedyResult:
    accepted: list[str]                 # species removed, in the order of acceptance
    kept_back: list[str]                # species kept because their removal broke the tolerance
    tests: list[dict] = field(default_factory=list)   # one entry per test: species, size, error, accepted
    budget_exhausted: bool = False

    @property
    def evaluations(self) -> int:
        return len(self.tests)


def greedy_search(evaluate: Callable[[list[str]], float], candidates: list[str], n_max: int, n_min: int,
                  tol: float, params: SearchParams | None = None) -> GreedyResult:
    """Remove the candidates one at a time and keep every species whose removal breaks ``tol``.

    ``evaluate(removed)`` returns the maximum relative error of the mechanism
    without the species in ``removed``; ``inf`` means that a condition did not
    ignite. Each test removes the next candidate on top of the removals
    accepted so far. The pass accepts a removal when the error is at or below
    ``tol``. It ends at ``n_min`` species, at the end of the candidates or
    when the tests reach ``max_evaluations``.
    """
    params = params or SearchParams()
    if n_min > n_max:
        raise ValueError(f"n_min {n_min} is above n_max {n_max}.")
    limit = params.max_evaluations if params.max_evaluations is not None else math.inf
    accepted: list[str] = []
    kept_back: list[str] = []
    tests: list[dict] = []
    exhausted = False
    for species in candidates:
        if n_max - len(accepted) <= n_min:
            break
        if len(tests) >= limit:
            exhausted = True
            break
        error = float(evaluate(accepted + [species]))
        ok = error <= tol
        tests.append({"species": species, "n_species": n_max - len(accepted) - 1, "error": error, "accepted": ok})
        if ok:
            accepted.append(species)
        else:
            kept_back.append(species)
    return GreedyResult(accepted=accepted, kept_back=kept_back, tests=tests, budget_exhausted=exhausted)


# ----------------------------------------------------------------------------- greedy pass in pieces and blocks

def greedy_pieces(order: list[str], units: list[list[str]]) -> list[tuple[str, ...]]:
    """The pieces of a removal order: a chain unit, or a single species.

    A unit sits at the position of its first member in ``order`` and lists
    its members in their order. A species of a unit is never a piece of its
    own. Every member of a unit must be in ``order``.
    """
    pos = {s: i for i, s in enumerate(order)}
    unit_of: dict[str, tuple[str, ...]] = {}
    for unit in units:
        missing = [s for s in unit if s not in pos]
        if missing:
            raise ValueError(f"the unit {unit} has members that are not in the order: {missing}")
        members = tuple(sorted(unit, key=pos.__getitem__))
        for s in members:
            if s in unit_of:
                raise ValueError(f"{s} is in more than one unit.")
            unit_of[s] = members
    out, seen = [], set()
    for s in order:
        if s not in seen:
            piece = unit_of.get(s, (s,))
            seen.update(piece)
            out.append(piece)
    return out


def greedy_items(pieces: list[tuple[str, ...]], tier0: set[str], block: int) -> list[list[tuple[str, ...]]]:
    """The test items of the first pass: lists of pieces.

    Consecutive pieces whose members are all in ``tier0`` form blocks of up
    to ``block`` species; a tier-0 piece larger than ``block`` is a block of
    its own. Every other piece is an item alone.
    """
    if block < 1:
        raise ValueError("block must be at least 1.")
    out: list[list[tuple[str, ...]]] = []
    cur: list[tuple[str, ...]] = []
    for p in pieces:
        if all(s in tier0 for s in p):
            if cur and sum(map(len, cur)) + len(p) > block:
                out.append(cur)
                cur = []
            cur.append(p)
        else:
            if cur:
                out.append(cur)
                cur = []
            out.append([p])
    if cur:
        out.append(cur)
    return out


@dataclass
class GreedyPass:
    tolerance: float
    accepted: list[str]                                 # species removed in this pass, in the order of acceptance
    kept: list[tuple[str, ...]]                         # pieces whose test failed alone
    untested: list[tuple[str, ...]] = field(default_factory=list)   # pieces not tested (size limit or budget)
    tests: list[dict] = field(default_factory=list)     # one entry per test, cached repeats included

    @property
    def evaluations(self) -> int:
        return sum(1 for t in self.tests if not t["cached"])


@dataclass
class LKGreedyResult:
    passes: list[GreedyPass]
    budget_exhausted: bool = False

    @property
    def accepted(self) -> list[str]:
        """Every removal of every pass, in the order of acceptance."""
        return [s for p in self.passes for s in p.accepted]

    @property
    def evaluations(self) -> int:
        return sum(p.evaluations for p in self.passes)


def _piece_kind(item: list[tuple[str, ...]]) -> str:
    if len(item) > 1:
        return "block"
    return "unit" if len(item[0]) > 1 else "single"


def lk_greedy_search(evaluate: Callable[[list[str], float, dict], float], order: list[str],
                     units: list[list[str]], tier0: set[str], block: int, tolerances: tuple,
                     n_max: int, n_min: int, params: SearchParams | None = None) -> LKGreedyResult:
    """Greedy passes over the pieces of ``order``, with tier-0 blocks.

    ``evaluate(removed, tol, test)`` returns the maximum relative error of the
    mechanism without the species in ``removed`` (``inf``: no ignition or a
    failed solve); ``test`` describes the test (pass, kind, members). A test
    passes when the error is at or below its tolerance.

    Pass 1 walks the items of :func:`greedy_items` at ``tolerances[0]``. A
    passed item is removed. A failed block is halved by pieces and its halves
    go first in the queue, down to single pieces. A failed piece stays in the
    mechanism. With a second tolerance, pass 2 starts from the removals of
    pass 1 and tests every piece that pass 1 kept, alone and in order.

    A test of a removed set already tested at the same tolerance repeats the
    stored result without an evaluation; this happens when the first half of
    a failed block passes, because the second half then repeats the block. An
    item that would leave fewer than ``n_min`` species is not tested: a block
    is halved, a piece goes to ``untested``. When ``max_evaluations`` is
    reached, the remaining pieces go to ``untested`` and no second pass runs.

    If the error never falls when a species leaves, the result equals the
    piece-by-piece pass with fewer evaluations.
    """
    params = params or SearchParams()
    if n_min > n_max:
        raise ValueError(f"n_min {n_min} is above n_max {n_max}.")
    if not 1 <= len(tolerances) <= 2:
        raise ValueError("tolerances must hold one or two values.")
    limit = params.max_evaluations if params.max_evaluations is not None else math.inf
    cache: dict[tuple[float, frozenset], float] = {}
    state = {"evaluations": 0, "exhausted": False}

    def run_pass(number: int, items: list[list[tuple[str, ...]]], tol: float, start: list[str]) -> GreedyPass:
        result = GreedyPass(tolerance=tol, accepted=[], kept=[])
        removed = list(start)
        queue = [list(it) for it in items]
        while queue:
            item = queue.pop(0)
            members = [s for p in item for s in p]
            if n_max - len(removed) - len(members) < n_min:
                if len(item) > 1:
                    h = (len(item) + 1) // 2
                    queue[0:0] = [item[:h], item[h:]]
                else:
                    result.untested.append(item[0])
                continue
            candidate = removed + members
            key = (tol, frozenset(candidate))
            cached = key in cache
            if not cached and state["evaluations"] >= limit:
                state["exhausted"] = True
                result.untested.extend(p for it in [item] + queue for p in it)
                break
            test = {"pass": number, "kind": _piece_kind(item), "pieces": [list(p) for p in item], "members": members,
                    "n_species": n_max - len(candidate), "tolerance": tol}
            if cached:
                error = cache[key]
            else:
                error = float(evaluate(candidate, tol, test))
                cache[key] = error
                state["evaluations"] += 1
            ok = error <= tol
            result.tests.append({**test, "error": error, "accepted": ok, "cached": cached})
            if ok:
                removed = candidate
                result.accepted.extend(members)
            elif len(item) > 1:
                h = (len(item) + 1) // 2
                queue[0:0] = [item[:h], item[h:]]
            else:
                result.kept.append(item[0])
        return result

    pieces = greedy_pieces(order, units)
    first = run_pass(1, greedy_items(pieces, tier0, block), float(tolerances[0]), [])
    passes = [first]
    if len(tolerances) == 2 and first.kept and state["evaluations"] >= limit:
        state["exhausted"] = True                       # the budget ends with pass 1: no second pass
    if len(tolerances) == 2 and not state["exhausted"]:
        passes.append(run_pass(2, [[p] for p in first.kept], float(tolerances[1]), first.accepted))
    return LKGreedyResult(passes=passes, budget_exhausted=state["exhausted"])
