# Search for the smallest reduced mechanism within an error tolerance

Date: 2026-10-23. Author: Claude for Yun-Jui Wu. Status: accepted and built (refer to the Decision section).

## Purpose

This note proposes how the package selects the most compact reduced mechanism within a user-given error tolerance. The error does not decrease monotonically with the number of species. It also lists the report that shows the result.

## Facts

- The rank fixes the removal order, so the candidate mechanisms form one chain: M_N, M_N-1, ..., M_min. The search space is one integer, the size `n`.
- `err(n)` is the maximum relative error over the evaluation conditions. It is not monotonic in `n` for two reasons:
  - solver noise: the ignition sample moves by up to one solver step, up to 0.3 % on methane
  - error cancellation: one removal raises the error, a later removal lowers it again. Methane, original law: 2.60 % at 35 species, 1.93 % at 30, 1.33 % at 27.
- The curve has few large jumps. On methane every version breaks at one species (C2H4): 1.3 % to 26 %. Between the jumps the error drifts slowly.
- One evaluation costs one reduced Solution build plus one reactor integration per condition. For USC Mech II the full sweep (96 sizes, 12 conditions) takes 2.3 min. For a 1800-species mechanism one integration takes minutes, so a step-1 sweep is not practical.
- The importance of every species is known before any evaluation. A large gap in importance between two consecutive species of the rank marks a probable jump of the error.

## Two optima

An isolated size that passes while its neighbours fail usually passes because of error cancellation at the evaluation conditions. Such a mechanism is not reliable at other conditions. The package therefore reports two sizes:

- strict optimum: the smallest evaluated `n` with `err(n) <= tol`
- robust optimum: the smallest `n` such that every evaluated size in `[n, n + w]` passes (`w` = 3 by default)

The report recommends the robust optimum and shows both.

## Proposed search

1. Early exit per condition (exact). Evaluate the conditions in the order of their error at the last evaluated size, largest first. Stop a size as soon as one condition exceeds the tolerance: the maximum over conditions cannot pass any more. A failed size then costs one or two integrations instead of twelve.
2. Coarse pass. Evaluate `K` sizes from `N` down to `min_species` (`K` = 16 by default), denser at small `n`, where the break usually is.
3. Structure probes. Evaluate the sizes just above and just below the `J` largest importance gaps of the rank (`J` = 5 by default). These are the sizes where a jump of the error is most probable; a coarse grid is blind to them.
4. Refinement of every crossing. For each interval whose end points disagree (pass and fail), evaluate the midpoint and split, down to width 1. Each crossing costs `log2(h)` evaluations for an interval of width `h`.
5. Near-miss refinement. Take an interval whose end points both fail, below the current best passing size. Refine it only when the smaller end-point error is below `c` times the tolerance (`c` = 2 by default). A dip under the tolerance between two points at ten times the tolerance is not plausible; between two points near the tolerance it is.
6. Noise band. Treat a size with `|err - tol|` below the noise level as undecided (noise = 0.3 % by default, or the measured step at ignition). Refine its neighbours to width 1 and apply the window rule.
7. Budget and coverage. Stop at `max_evaluations`. The result lists every evaluated size, the two optima and the unresolved intervals with their smallest end-point error. The user can then spend more budget, or run `step: 1` inside one interval.

Estimated cost for a 1800-species mechanism: 16 coarse sizes, 10 probes, 3 crossings of 8 evaluations each and about 10 near-miss evaluations. That is approximately 60 evaluations instead of 1800. The early exit per condition lowers the cost of the failed sizes further.

The full sweep with `step: 1` stays available and stays the reference. It is the default for mechanisms with fewer than 200 species. The adaptive search is for the large mechanisms.

## Configuration

```json
"reduction": {
  "error_tolerance": 0.05,
  "search": {
    "mode": "full",
    "coarse_points": 16,
    "importance_gap_probes": 5,
    "near_miss_factor": 2.0,
    "robust_window": 3,
    "noise": 0.003,
    "max_evaluations": null
  },
  "write_mechanism": true
}
```

`mode` is `full` (the sweep of today, with the early exit per condition) or `adaptive` (steps 2 to 7).

## Report

The command `gasig report` reads the sweep or search result and writes a folder `report/`:

- the error against the number of species, with the tolerance line and the two optima
- the ignition delay time against 1000/T, lines for the detailed mechanism and markers for the selected one
- one panel per pressure and equivalence ratio, on a denser temperature grid than the evaluation conditions
- the flame speed against the equivalence ratio, when the fs mode is on
- a table of the errors per condition
- the selected mechanism as a Cantera YAML file (`write_mechanism`), so the user can run it outside the package
- `report.md` in ASD-STE100 and `report.json` with the numbers

The plots need Matplotlib; it becomes an optional dependency (`pip install gasig[report]`) and goes into both images.

## User guide

The README becomes an overview with a quick start. A new `docs/user_guide.md` walks through the methane example from installation to the report:

1. Install (Docker or pip).
2. Prepare the inputs: the mechanism, the configuration from `gasig template`, the conditions, the targets, the tolerance, or the precomputed files.
3. Run `gasig run`, `gasig rank`, `gasig reduce` and `gasig report`, with the expected run times.
4. Read the results: the files, the error curve, the selected mechanism, and how to load the YAML in Cantera.
5. Solve problems.

## Decision

The owner accepted the four points on 2026-10-23, and the package implements them (`gasig/search.py`, `gasig/reduce.py`, `gasig/report.py`, `docs/user_guide.md`). Three details differ from the proposal above:

- The robust optimum of the adaptive search is confirmed by the search itself: it is reported only when every size of its window was evaluated. When the budget ends first, the result has no robust optimum.
- The detailed mechanism is not a candidate for either optimum.
- The sweep writes the selected mechanism (`reduced_<n>sp.yaml`), not the report; the report writes it only when the sweep did not. The sweep JSON stores the species order, the removal order and the selection, so the report rebuilds any size without the rank file.

The default mode stays `full` for every size of mechanism; the user selects `adaptive`. The owner validates the reduction performance on the methane example next.
