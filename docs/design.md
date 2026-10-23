# GA-SiG package design

Date: 2026-10-23. Author: Yun-Jui Wu with Claude. Status: accepted.

## Purpose

The `gasig` package merges the temporal (idt) and the spatial (fs) GA-SiG integrators into one code with one configuration file. It replaces the code of the version 0.1 scripts `GA-SiG.py` and `graph_search_to_importance.py` and of the spatial notebook. The two scripts remain at the root as thin wrappers around the package. The package runs the same way on a workstation, in a Docker container and on a GPU server.

## Overview

The run has three stages. Each stage reads and writes files in the output folder, so a later stage can run alone.

1. Data. The package runs the Cantera solvers, or it loads precomputed files. The result is a list of trajectories.
2. Sensitivity. The package integrates each trajectory, aggregates the sensitivities to `r_AB`, and writes `r_AB_<mode>.npy`.
3. Ranking. The package selects or blends the matrices, runs the graph search, and writes the rank.

Table 1. Modules

| Module | Content |
|---|---|
| `gasig/config.py` | Schema, validation, defaults, template, path resolution |
| `gasig/kinetics.py` | `TorchKinetics`: rate constants from Cantera, differentiable rates in torch, two rate laws, self-check |
| `gasig/jacobian.py` | `J1` by reverse-mode AD, `J2` in closed form |
| `gasig/sensitivity.py` | `Trajectory`, implicit Euler step with the step-size guard, residence time, flame tip |
| `gasig/aggregate.py` | `S_AB`, the maximum over trajectories, two normalizations |
| `gasig/graph.py` | Adjacency, Dijkstra, importance, hybrid matrix, rank |
| `gasig/io.py` | Precomputed format: load and save |
| `gasig/solvers/idt.py` | 0-D reactor trajectories and the ignition sample |
| `gasig/solvers/flame.py` | 1-D `FreeFlame` profiles and the flame tip |
| `gasig/pipeline.py` | The three stages, logging, `summary.json` |
| `gasig/cli.py` | Commands `run`, `rank`, `template`, `info` |

## Inputs and outputs

Table 2. Inputs

| Name | Type / shape | Unit | Source |
|---|---|---|---|
| Mechanism | Cantera `.yaml` | | User |
| Configuration | JSON, schema in `config.py` | | User |
| idt trajectories | dictionaries of arrays, refer to the README | s, K, Pa, 1 | Solver or user |
| fs trajectories (flame profiles) | dictionaries of arrays, refer to the README | m, m/s, K, Pa, 1 | Solver or user |

Table 3. Outputs

| Name | Type / shape | Content |
|---|---|---|
| `r_AB_<mode>.npy` | float64 (Ns, Ns) | Normalized coupling matrix of the mode |
| `S_AB_<mode>.npy` | float64 (Ns, Ns) | Coupling matrix before normalization |
| `S_abs_max_<mode>.npy` | float64 (Ns, Nr) | Optional: element-wise maximum of `abs(S)` |
| `r_AB.npy` | float64 (Ns, Ns) | Input matrix of the graph search |
| `importance.npy` | float64 (Ns,) | Importance of each species |
| `ranked_species.csv`, `ranked_sp.npy` | text, (Ns, 2) strings | Rank from the least to the most important |
| `trajectories_<mode>.json` | JSON | Conditions and integration counters per trajectory |
| `idt_data/`, `fs_data/` | NumPy | Solver results in the precomputed format |
| `summary.json`, `config_resolved.json`, `gasig.log` | JSON, text | Run summary, the resolved configuration, the log |

## Method

### Parameterization and sensitivity equation

The forward rate constants are `k_f,theta = k_f exp(theta)`, so `theta = 0` gives the original mechanism. The reverse rate constants scale with the same factor. The sensitivity `S = dA/dtheta` has the shape (Ns, Nr) and obeys

```
dS/dtau = J1 S + J2,    S(0) = 0,
J1 = d(dY/dt)/dY,       J2 = d(dY/dt)/dtheta.
```

`tau` is the time in idt mode. In fs mode `tau` is the residence time of a gas parcel along the flame, `dtau = dx / u(x)`. The package computes `tau` once with the trapezoidal rule on `1/u` and then uses the same integrator for both modes.

### Jacobians

`J1` comes from `torch.func.jacrev` over the Ns mass fractions. `J2` has the closed form `J2[a, i] = (W_a / rho) nu_net[a, i] R_i`, because every net rate scales with `exp(theta_i)`. The closed form equals the automatic derivative to machine precision and avoids one vector-Jacobian product per reaction.

### Rate laws

With `rate_law: "corrected"`, the package sets the Cantera state to the temperature, pressure and composition of each sample and reads `forward_rate_constants` and `reverse_rate_constants`. Cantera 3 excludes the third-body concentration from the rate constant of a three-body reaction. The package therefore multiplies those reactions by `M_eff = E C`, where `E` holds the third-body efficiencies, inside the differentiated code. Falloff, pressure-dependent and Chebyshev rate constants contain their pressure and composition dependence already; they are constants in the Jacobian. A self-check at start-up compares the forward and the reverse rates of progress with the Cantera values reaction by reaction and stops the run if they differ.

With `rate_law: "original"`, the package reproduces the computation of the published results:

- rate constants at the temperature and pressure of each sample, with the default composition of the mechanism
- no third-body concentration `M_eff`
- the reverse rate `k_f / Kc` for every reaction
- concentrations and `Kc` clamped at 1e-300 (at the smallest normal number in float32).

### Time integration

Each data interval is one backward Euler step with `J1` and `J2` at the end of the interval:

```
(I - dtau J1) S_new = S + dtau J2.
```

The package factors `I - dtau J1` once with an LU decomposition and solves for all Nr right-hand sides. Before each step, the package clips negative mass fractions to zero and renormalizes the sample, because `dY/dt` and `J2` scale with the sum of the mass fractions.

Backward Euler changes the sign of a growing mode when `dtau` times its growth rate exceeds 1. With `substep_tol` set, the package compares one step with two half steps, both with the same frozen Jacobians. The comparison is made species row by species row. A row agrees when its change between the two levels is within the tolerance times the larger of its norm and its increment, with a floor of 1e-12 of the Frobenius norm of the matrix. If a row disagrees, the package doubles the number of sub-steps and compares again, up to `max_substeps`. The package records the number of solves and the largest number of sub-steps for each trajectory.

### Coupling matrix and normalization

```
S_AB(a, b)  = sum over j in R_b of abs(S(a, j)),    R_b = { j : nu_net(b, j) != 0 }
S_AB,global = element-wise maximum of S_AB over the trajectories
r_AB(a, b)  = S_AB,global(a, b) / sum over b' of S_AB,global(a, b')        (S_AB_row_sum)
r_AB(a, b)  = S_AB,global(a, b) / sum over j of max_m abs(S_m(a, j))       (S_row_sum)
```

`S_AB_row_sum` is the definition of the paper; each row of `r_AB` sums to 1. `S_row_sum` is the computation behind the published results. For one trajectory, the two denominators differ by a factor: the sensitivity-weighted mean number of species per reaction in each row. With several trajectories the element-wise maximum makes that factor an upper bound. Rows whose denominator is not positive stay zero. With `abs_S: false` and `S_AB_row_sum`, the signed row sums can cancel; the package warns and reports how many rows become zero.

### Graph search

Every entry `r_AB(a, b) > eps_edge` is an edge with the length `-ln r_AB(a, b)`. Dijkstra's algorithm gives the distance `D` from each target. The importance with respect to that target is `exp(-D)`. The overall importance is the maximum over the targets. The hybrid matrix is `w r_AB,idt + (1 - w) r_AB,fs`; both matrices must have the same shape.

### Flame tip and ignition sample

In fs mode the integration ends at the first grid point where `T` reaches `flame_tip_T_fraction` of the maximum `T`. If the first point already satisfies the rule, the profile is flat or decreasing, and the package stops with an error. A precomputed `index` file overrides the rule. In idt mode the solver sets the end index at the peak of the reference species, or at the maximum of dT/dt. If the peak is the last sample, or the temperature rise is below 100 K, the solver reports no ignition. If the peak is the first sample although the reactor ignited, the reference species is a reactant, and the solver stops with an error.

## Data structures

`Trajectory` has these fields:

- `tau` (N,): time or residence time in s, strictly increasing
- `T` (N,): temperature in K
- `P` (N,): pressure in Pa
- `Y` (N, Ns): mass fractions
- `end_index`: last sample of the integration, inclusive
- `label`: name for log messages
- `meta`: dictionary of solver metadata

The fs solver stores `x` and `u` in `meta` so that `save_fs_data` can write the profile in the precomputed format. `validate` makes sure that:

- `tau`, `T` and `P` have the shape (N,) and `Y` has the shape (N, Ns)
- `end_index` is an integer in [0, N - 1]
- every value before the end index is finite, `tau` increases strictly, and `T` and `P` are positive
- every mass-fraction row before the end index has a positive entry.

`validate` warns when a mass fraction before the end index is below -1e-6. The integrator clips negative mass fractions to zero at every step and renormalizes the sample; the saved `Y_all.npy` keeps the solver values.

## Decisions

| ID | Decision | Reason | Alternative not chosen |
|---|---|---|---|
| D1 | Two rate laws behind one switch; `original` is the default | The published results used the original rate law; the corrected rate law is a second version to evaluate | Corrected rate law only |
| D2 | `abs(S)` inside the `S_AB` sum by default | Definition of the paper and of the benchmark code; the signed sum cancels production and consumption and deletes edges | Signed sum of the version 0.1 script |
| D3 | `S_AB_row_sum` normalization by default, `S_row_sum` as a switch | The paper defines the row-stochastic form; the published computation used the other form and must stay reproducible | One form only |
| D4 | fs mode returns `S` at the flame tip; the 99 percent rule is in the package | The paper describes an integration from the inlet to the flame tip; the residence-time average of the notebook was dead code | Residence-time average |
| D5 | Per-row step-size guard behind `substep_tol`, off by default | Backward Euler flips growing modes when `dtau` times the growth rate exceeds 1; the guard changes the original results, so it is opt-in; a global norm would hide small rows | Always on; global norm |
| D6 | Additive concentration floor in the corrected rate law | The clamp of version 0.1 zeroes the Jacobian column of an absent species | Clamp |
| D7 | `jacrev` for `J1`, closed form for `J2` | Identical values, approximately 4 times faster, less memory | `autograd.functional.jacobian` over (Y, theta) |
| D8 | Ignition at the peak of the reference species, with a no-ignition warning and a reactant check | Same definition as the data collector of the paper; a peak at the last sample means no ignition; a peak at the first sample means a wrong reference species | Maximum of dT/dt only |
| D9 | `.gitignore` for `not-to-track/`; the example folder keeps data and configurations, not script copies | Git must not track the source files of the paper; one package replaces the duplicated scripts | Keep duplicated scripts |
| D10 | Tests run in the container | No local environment has a working PyTorch; the container is the reference environment | Repair of the local environment |

## Tests

- `test_config.py`:
  - defaults and path resolution, including defaulted paths
  - unknown keys, type and choice errors, the new range checks
  - mode requirements and the flame-tip alias
  - UTF-8 with byte-order mark and UTF-16 configuration files
- `test_aggregate_graph.py`:
  - `S_AB` by hand
  - both normalizations and the warning for the signed sum
  - shortest paths on a toy graph
  - the hybrid matrix and its shape checks
  - an exact reproduction of the rank of the shipped ethylene example from its `r_AB.npy`
- `test_kinetics.py`:
  - the corrected rates equal the Cantera rates at three states to 1e-10 of the largest rate
  - the third-body concentrations equal the Cantera values
  - the original rate law equals the version 0.1 code to 1e-11 of the largest rate
  - `J1` and `J2` equal the autograd reference to 1e-12
  - `J1` equals central finite differences to 1e-5, column by column
  - the mass-action part of `J1` equals Cantera's `net_production_rates_ddCi` to 1e-8
  - the floor keeps the self-consumption derivative of an absent species
  - the original rate law stays finite in float32
- `test_sensitivity.py`:
  - the guard recovers a growing mode that plain backward Euler flips, and respects the cap
  - the per-row criterion sees a small growing row behind a large one
  - first-order convergence
  - residence time and flame tip, including the rejected profiles
  - the original mode reproduces the version 0.1 integrator on 40 steps of the ethylene example to 1e-9 of the largest value, and to 1e-8 row by row above the noise floor
  - GPU equals CPU
- `test_pipeline.py` (slow):
  - methane with USC Mech II, both modes through the solvers
  - all output files, row sums of 1, targets at importance 1, radicals in the top 20
  - the ranking stage alone, with the sensitivity record kept
  - the saved solver data as precomputed input
  - the resolved configuration as an input file
  - a reactant as reference species is rejected

## Open items

- Performance test of the corrected rate law on the four mechanisms of the paper (reduction error against the number of species).
- Evaluation of the two normalizations on the same data.
- A utility that removes species in rank order and computes the ignition delay time error.
- Differentiation of the falloff dependence on the composition, if the performance test shows that it matters. Cantera's `forward_rate_constants_ddC` gives the needed derivative.
- Cantera 3.3 will change the default of the reactor `clone` argument. The idt solver passes `clone=False` and reads the reactor phase, so the package needs no change.
