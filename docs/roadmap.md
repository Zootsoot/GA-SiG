# GA-SiG roadmap

Date: 2026-10-23. Owner: Yun-Jui Wu.

## Goal

One package that:

- ranks species for mechanism reduction from ignition-delay-time and flame-speed trajectories
- runs in a container with or without a GPU
- reproduces the published results on request.

## Scope

- In scope:
  - the sensitivity integration
  - the graph search and the species rank
  - the Cantera solvers for both modes
  - the precomputed route
  - the Docker images
  - the tests with USC Mech II (methane in the solvers, ethylene in the regression data)
  - the documentation in ASD-STE100
- Out of scope:
  - the construction of reduced mechanism files
  - the evaluation of reduction errors
  - the chemistry solver for the GPU

## Milestones

| ID | Deliverable | Done when | Target | Status |
|---|---|---|---|---|
| M1 | `gasig` package with both modes, JSON configuration and the four method switches | The unit tests pass in the container | 2026-10 | tests pass; review applied |
| M2a | CPU-only Docker image (`Dockerfile.cpu`) | `gasig info` runs in the container and the methane end-to-end test passes on the CPU | 2026-10 | done |
| M2b | CUDA Docker image (`Dockerfile`) | `gasig info` shows the GPU inside the container and the GPU tests pass | later | deferred |
| M3 | Documentation | README, design, roadmap follow ASD-STE100 and describe every switch | 2026-10 | review applied |
| M4 | Merge of `feat-spatial-GA-SiG` into `main` | Review done, tag `v0.2.0` | after M1 to M3 | not started |
| M5 | Performance test of the corrected rate law | Reduction error against species count for C2H4, n-C7H16, n-C10H22 and Gevo AtJ-SPK with both rate laws | future | not started |
| M6 | Evaluation of the normalization switch | The same curves with `S_AB_row_sum` and `S_row_sum` | future | not started |
| M7 | Reduction utility | A command that removes species in rank order and reports the ignition-delay-time and flame-speed errors | future | not started |

## Risks

| Risk | Effect | Action |
|---|---|---|
| The corrected rate law changes the published ranks | The reduction curves of the paper change | M5 before any claim; keep `original` as the default until then |
| The local GPU has 4 GB | The AtJ mechanism does not fit on the local machine | Run M5 on an L4 or larger GPU |
| Cantera API changes after 3.2 | Deprecation warnings or errors in the solvers | Pin Cantera 3.2.0 in the image; test before an upgrade |
| Docker Desktop start-up failures on Windows | No local container tests | Rename the stale `run` folder in `%LOCALAPPDATA%\Docker`; use WSL 2 |
| Slow network link | The CUDA base image (3.7 GB) takes hours to download | Build M2b on a fast link; use the CPU image meanwhile |

## Progress log

| Date | Entry |
|---|---|
| 2026-10-23 | Analysis of the version 0.1 scripts, the spatial notebook and the manuscript done. Three rate-law defects found under Cantera 3.x (third-body factor, falloff composition, reverse rate of irreversible reactions). The manuscript computation (`benchmark.ipynb`) uses the absolute sum in `S_AB` and the `S_row_sum` normalization. |
| 2026-10-23 | Decisions D1 to D10 agreed. Defaults: `rate_law: original`, `substep_tol: null`, `abs_S: true`, `normalization: S_AB_row_sum`. The corrected rate law and the published normalization stay as switches for the performance tests M5 and M6. |
| 2026-10-23 | Package `gasig` written: config, kinetics with two rate laws, Jacobians, integrator with step-size guard, aggregation, graph, io, idt and fs solvers, pipeline, CLI, compatibility wrappers. Tests and three example configurations written. Documentation in ASD-STE100. |
| 2026-10-23 | Docker Desktop on the development machine failed to start because of stale socket files; fixed by a rename of the socket folders. CPU image built (1.5 GB); `gasig info` in the container reports Python 3.11.17, NumPy 1.26.4, Cantera 3.2.0, PyTorch 2.8.0+cpu. The CUDA base image downloads at approximately 0.5 MB/s on this link, so M2b is deferred. |
| 2026-10-23 | Fast test suite in the CPU container: 32 of 36 pass at the first run. Fixed: default `output_dir` was not resolved against the config folder; the Cantera derivative test used `net_production_rates_ddC` instead of `net_production_rates_ddCi`; two tolerances were per element instead of relative to the largest value. Second run: 34 passed, 2 skipped (GPU only). |
| 2026-10-23 | Slow tests in the CPU container: the methane end-to-end case (both solvers, all output files, row sums of 1, targets at importance 1, radicals in the top 20, ranking stage alone, saved solver data as precomputed input) passed in 259 s; the ethylene precomputed case with the version 0.1 switches passed in 36 s. |
| 2026-10-23 | Regression against version 0.1 with the original rate law: the sensitivities agree to 3e-12 of the largest value; rows above 1e-6 of it agree to 4e-10. Rows below that floor are floating-point noise in both codes because the linear systems have condition numbers of 1e9 to 1e10. |
| 2026-10-23 | `example/config_ch4_solver.json` ran through the container entry point on the CPU in 742 s: 12 ignition cases (28 ms at 1200 K, 1 atm, phi 0.5 to 72 us at 1600 K, 10 atm, phi 0.5), 3 flames (S_L = 0.277, 0.364, 0.316 m/s at phi 0.8, 1.0, 1.2), 11664 graph edges, 3 unreachable species. After the targets the rank starts with H, CH3, H2, HO2, O, CH3CHOCH2, CH2O, HCO, CH3O, C2H6. |
| 2026-10-23 | Review of the package by 5 reviewers and 77 verifiers (82 agents): 75 findings confirmed, 2 refuted. Applied: float32-safe clamp, tolerant third-body loop, per-reaction self-check, per-row step-size guard with the cap checked first, mass-fraction renormalization, flame-tip rule that rejects flat profiles, reactant as reference species rejected, shape checks of the hybrid blend, ranking-only runs keep the summary, loadable `config_resolved.json`, closed log file, CLI exit codes and encodings, transport-model validation, index range checks, 31 STE corrections in the documents. Left open: the falloff composition derivative (design open item). |

## Dependencies

- M2a and M2b need a working Docker engine on the test machine.
- M4 needs the review of the numerics tests and the author's decision on the default switches.
- M5 needs the trajectory data of the four mechanisms from the author's archive.
- M7 needs a definition of the error metric and of the validation conditions per fuel.
