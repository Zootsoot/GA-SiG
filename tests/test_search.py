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


def test_select_optima_window_counts_evaluated_sizes_of_a_stepped_sweep():
    """Step 5, window 3: the window holds the next 3 evaluated sizes, not the sizes n + 1 to n + 3."""
    isolated = {100: 0.0, 95: 0.2, 90: 0.01, 85: 0.3}
    assert select_optima(isolated, 0.05, 3) == (90, None)          # 90 passes alone; it is not robust
    stepped = {100: 0.0, 95: 0.01, 90: 0.2, 85: 0.01, 80: 0.01, 75: 0.01, 70: 0.01, 65: 0.4}
    assert select_optima(stepped, 0.05, 3) == (70, 70)
    assert select_optima(stepped, 0.05, 4) == (70, 95)             # 90 is in the window of 70 to 85
    assert select_optima(stepped, 0.05, 3, require_full_window=True) == (70, None)


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
    assert res.evaluations == 11 and res.budget_exhausted        # 10 reduced sizes and the detailed size
    assert res.strict_optimum == 28
    assert res.robust_optimum is None            # no window was confirmed
    assert any(u["kind"] == "crossing" for u in res.unresolved)


def test_adaptive_budget_counts_reduced_mechanisms_only():
    """max_evaluations limits the reduced sizes; the detailed size n_max is not one of them."""
    for budget in (1, 5):
        calls = []

        def evaluate(n):
            calls.append(n)
            return 0.0 if n == 100 else 0.01

        res = adaptive_search(evaluate, 100, 20, 0.05, probes=[], params=SearchParams(max_evaluations=budget))
        assert calls[0] == 100 and len(calls) == budget + 1 and len(set(calls)) == len(calls)
        assert res.budget_exhausted and res.strict_optimum == min(calls)


def test_adaptive_search_noise_band_resolves_neighbours():
    def err(n):
        return 0.051 if n == 40 else (0.01 if n > 40 else 0.5)

    res = adaptive_search(err, 60, 20, 0.05, probes=[40], params=SearchParams(coarse_points=4, noise=0.003))
    assert {39, 41} <= set(res.errors)
    assert (res.strict_optimum, res.robust_optimum) == (41, 41)


def test_greedy_search_keeps_the_breaker_and_continues():
    """Removing b breaks the tolerance whatever else is removed; a, c and d go, b stays."""
    from gasig.search import greedy_search

    seen = []

    def err(removed):
        seen.append(list(removed))
        return 0.3 if "b" in removed else 0.01 * len(removed)

    result = greedy_search(err, ["a", "b", "c", "d", "e"], 10, 7, 0.05)
    assert result.accepted == ["a", "c", "d"] and result.kept_back == ["b"]
    assert [t["n_species"] for t in result.tests] == [9, 8, 8, 7]
    assert [t["accepted"] for t in result.tests] == [True, False, True, True]
    assert seen == [["a"], ["a", "b"], ["a", "c"], ["a", "c", "d"]]       # e is never tested: n_min reached
    assert result.evaluations == 4 and result.budget_exhausted is False


def test_greedy_search_budget_and_no_ignition():
    from gasig.search import greedy_search

    result = greedy_search(lambda removed: float("inf"), ["a", "b", "c"], 10, 5, 0.05,
                           SearchParams(max_evaluations=2))
    assert result.accepted == [] and result.kept_back == ["a", "b"]
    assert result.budget_exhausted is True and result.evaluations == 2
    with pytest.raises(ValueError):
        greedy_search(lambda removed: 0.0, ["a"], 5, 6, 0.05)


# ----------------------------------------------------------------------------- lk_greedy: pieces, blocks, two passes

def _synthetic_case():
    """40 species; two chain units; s0-s17 tier 0; s25 fails at 15 % and passes at 20 %."""
    order = [f"s{i}" for i in range(40)]
    units = [["s6", "s5"], ["s20", "s21"]]
    tier0 = {f"s{i}" for i in range(18)}
    eff = {"s3": 0.30, "s12": 0.04, "s25": 0.10, "s33": 0.02, "s38": 0.5}

    def err(removed: set) -> float:
        e = sum(eff.get(s, 0.001) for s in removed)
        e += 0.10 if {"s10", "s30"} <= removed else 0.0            # an interaction with an earlier removal
        e += 1.0 if len({"s20", "s21"} & removed) == 1 else 0.0     # a split chain unit
        return e

    return order, units, tier0, err


def test_greedy_pieces_and_items():
    from gasig.search import greedy_items, greedy_pieces

    order, units, tier0, _ = _synthetic_case()
    pieces = greedy_pieces(order, units)
    assert ("s5", "s6") in pieces and ("s20", "s21") in pieces and len(pieces) == 38
    assert pieces.index(("s5", "s6")) == 5                          # at the position of its first member
    items = greedy_items(pieces, tier0, 8)
    assert [sum(map(len, it)) for it in items[:4]] == [8, 8, 2, 1]
    assert all(len(it) == 1 for it in items[3:])                    # tier-1 pieces are tested alone
    with pytest.raises(ValueError, match="not in the order"):
        greedy_pieces(order, [["s1", "x"]])
    with pytest.raises(ValueError, match="more than one unit"):
        greedy_pieces(order, [["s1", "s2"], ["s2", "s3"]])


def test_lk_greedy_blocks_equal_the_piece_by_piece_pass():
    """With an error that never falls when a species leaves, blocks give the piece-by-piece result, cheaper."""
    from gasig.search import lk_greedy_search

    order, units, tier0, err = _synthetic_case()
    calls = []

    def evaluate(removed, tol, test):
        calls.append((tol, frozenset(removed)))
        return err(set(removed))

    res = lk_greedy_search(evaluate, order, units, tier0, 8, (0.15, 0.20), 50, 0)
    n_block = len(calls)
    calls.clear()
    ref = lk_greedy_search(evaluate, order, units, set(), 8, (0.15, 0.20), 50, 0)    # no tier 0: every piece alone
    n_piece = len(calls)
    p1, p2 = res.passes
    assert set(p1.accepted) == set(ref.passes[0].accepted) and p1.kept == ref.passes[0].kept
    assert set(p2.accepted) == set(ref.passes[1].accepted) and p2.kept == ref.passes[1].kept
    assert n_block < n_piece and res.evaluations == n_block
    assert p1.kept == [("s3",), ("s25",), ("s30",), ("s38",)]
    assert p2.kept == [("s3",), ("s30",), ("s38",)] and "s25" in p2.accepted
    assert err(set(p1.accepted)) <= 0.15 and err(set(res.accepted)) <= 0.20
    tests = p1.tests + p2.tests
    assert all(len({"s20", "s21"} & set(t["members"])) != 1 for t in tests)    # a unit is never split
    assert all(not t["accepted"] or t["error"] <= t["tolerance"] for t in tests)
    assert [t["pieces"] for t in p2.tests] == [[list(p)] for p in p1.kept]
    assert {t["kind"] for t in p1.tests} == {"block", "unit", "single"}
    assert res.accepted == p1.accepted + p2.accepted and not res.budget_exhausted


def test_lk_greedy_repeats_a_failed_block_without_an_evaluation():
    """The first half of a failed block passes, so the second half repeats the block: a stored result."""
    from gasig.search import lk_greedy_search

    calls = []

    def evaluate(removed, tol, test):
        calls.append(frozenset(removed))
        return 0.5 if {"b", "c"} <= set(removed) else 0.0

    res = lk_greedy_search(evaluate, ["a", "b", "c", "d"], [], {"a", "b", "c", "d"}, 4, (0.1,), 10, 0)
    p = res.passes[0]
    assert [t["members"] for t in p.tests] == [["a", "b", "c", "d"], ["a", "b"], ["c", "d"], ["c"], ["d"]]
    assert [t["cached"] for t in p.tests] == [False, False, True, False, False]
    assert p.tests[2]["error"] == 0.5 and len(calls) == len(set(calls)) == 4
    assert p.accepted == ["a", "b", "d"] and p.kept == [("c",)] and p.evaluations == 4


def test_lk_greedy_size_limit_and_budget():
    from gasig.search import lk_greedy_search

    res = lk_greedy_search(lambda r, t, x: 0.0, ["a", "b", "c", "d", "e"], [["c", "d"]], {"a", "b"}, 4, (0.1, 0.2),
                           10, 7)
    p1 = res.passes[0]
    assert p1.accepted == ["a", "b", "e"] and p1.untested == [("c", "d")]    # the unit would leave 6 < 7 species
    assert res.passes[1].tests == []                                       # nothing kept: an empty second pass
    res = lk_greedy_search(lambda r, t, x: float("inf"), ["a", "b", "c"], [], set(), 4, (0.1, 0.2), 10, 0,
                           SearchParams(max_evaluations=2))
    assert res.budget_exhausted and len(res.passes) == 1 and res.evaluations == 2
    assert res.passes[0].kept == [("a",), ("b",)] and res.passes[0].untested == [("c",)]
    # A budget used up exactly by pass 1 runs no second pass.
    res = lk_greedy_search(lambda r, t, x: 0.15, ["a", "b"], [], set(), 4, (0.1, 0.2), 10, 0,
                           SearchParams(max_evaluations=2))
    assert res.budget_exhausted and len(res.passes) == 1 and res.passes[0].kept == [("a",), ("b",)]
    res = lk_greedy_search(lambda r, t, x: 0.15, ["a", "b"], [], set(), 4, (0.1, 0.2), 10, 0,
                           SearchParams(max_evaluations=3))
    assert res.budget_exhausted and res.passes[1].accepted == ["a"] and res.passes[1].untested == [("b",)]
    with pytest.raises(ValueError):
        lk_greedy_search(lambda r, t, x: 0.0, ["a"], [], set(), 4, (0.1,), 5, 6)
    with pytest.raises(ValueError):
        lk_greedy_search(lambda r, t, x: 0.0, ["a"], [], set(), 4, (0.1, 0.2, 0.3), 5, 0)
