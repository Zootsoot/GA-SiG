import pytest

from gasig.search import SearchParams, adaptive_search, coarse_sizes, gap_probes, select_optima


def step_curve(jump: int):
    """Small error above ``jump``, large error at ``jump`` and below."""
    def err(n: int) -> float:
        if n > jump:
            return 0.002 + 1e-5 * (200 - n)
        return 0.3 + 0.01 * (jump - n)
    return err


def test_coarse_sizes_cover_both_ends_and_are_denser_at_small_sizes():
    sizes = coarse_sizes(111, 15, 12)
    assert sizes[0] == 111 and sizes[-1] == 15
    assert len(sizes) == len(set(sizes)) == 12
    gaps = [a - b for a, b in zip(sizes, sizes[1:])]
    assert gaps[0] > gaps[-1]
    assert coarse_sizes(20, 15, 16) == [20, 19, 18, 17, 16, 15]
    with pytest.raises(ValueError):
        coarse_sizes(10, 11, 4)


def test_gap_probes_mark_both_sides_of_the_largest_gaps():
    assert gap_probes([0.0, 0.0, 1e-9, 1e-8, 1e-3, 1e-2, 0.5, 0.9], 20, 10, 2) == [18, 17, 16, 15]
    assert gap_probes([1.0, 1.0, 1.0], 20, 10, 3) == []
    assert gap_probes([], 20, 10, 3) == []
    assert gap_probes([1e-3, 1.0], 20, 10, 0) == []


def test_select_optima_strict_and_robust():
    errors = {111: 0.0, 30: 0.01, 29: 0.06, 28: 0.01, 27: 0.01, 26: 0.3}
    assert select_optima(errors, 0.05, 3) == (27, 30)
    assert select_optima(errors, 0.05, 0) == (27, 27)
    assert select_optima({111: 0.0, 26: 0.3, 25: 0.4}, 0.05, 3) == (None, None)   # the detailed size is no candidate
    sparse = {111: 0.0, 40: 0.01, 37: 0.01, 36: 0.01}
    assert select_optima(sparse, 0.05, 3) == (36, 36)
    assert select_optima(sparse, 0.05, 3, require_full_window=True) == (36, None)


def test_adaptive_search_finds_the_break_with_few_evaluations():
    calls = []

    def evaluate(n):
        calls.append(n)
        return step_curve(27)(n)

    res = adaptive_search(evaluate, 111, 15, 0.05, probes=[28, 27], params=SearchParams(coarse_points=12))
    assert (res.strict_optimum, res.robust_optimum) == (28, 28)
    assert len(calls) == len(set(calls)) == res.evaluations < 25
    assert not res.budget_exhausted
    assert all(u["kind"] == "gap" and u["to"] <= 28 for u in res.unresolved)


def test_adaptive_search_refines_a_crossing_by_bisection():
    res = adaptive_search(step_curve(50), 111, 15, 0.05, probes=[], params=SearchParams(coarse_points=8))
    assert (res.strict_optimum, res.robust_optimum) == (51, 51)
    assert res.evaluations <= 8 + 4 + 2          # coarse sizes, four bisections, two window sizes


def test_adaptive_search_near_miss_island_and_unresolved_gap():
    def err(n):
        if n >= 60:
            return 0.01
        if 44 <= n < 60:
            return 0.08                           # near miss: below 2 x tolerance
        if n in (40, 41, 42, 43):
            return 0.04                           # island under the tolerance, reached through the near miss
        if n in (25, 26):
            return 0.03                           # island between two far points: not found, reported
        return 0.5

    res = adaptive_search(err, 111, 15, 0.05, probes=[], params=SearchParams(coarse_points=8))
    assert (res.strict_optimum, res.robust_optimum) == (40, 40)
    assert any(u["from"] <= 25 and u["to"] >= 26 and u["kind"] == "gap" for u in res.unresolved)
    assert 25 not in res.errors and 26 not in res.errors


def test_adaptive_search_respects_the_budget():
    res = adaptive_search(step_curve(27), 111, 15, 0.05, probes=[],
                          params=SearchParams(coarse_points=12, max_evaluations=10))
    assert res.evaluations == 10 and res.budget_exhausted
    assert res.strict_optimum == 28
    assert res.robust_optimum is None            # no window was confirmed
    assert any(u["kind"] == "crossing" for u in res.unresolved)


def test_adaptive_search_noise_band_resolves_neighbours():
    def err(n):
        return 0.051 if n == 40 else (0.01 if n > 40 else 0.5)

    res = adaptive_search(err, 60, 20, 0.05, probes=[40], params=SearchParams(coarse_points=4, noise=0.003))
    assert {39, 41} <= set(res.errors)
    assert (res.strict_optimum, res.robust_optimum) == (41, 41)
