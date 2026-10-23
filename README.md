# GA-SiG

GPU-Aided Sensitivity-in-Graph (GA-SiG) ranks the species of a detailed kinetic mechanism. The rank shows how strongly the reaction sensitivities couple each species to a set of target species. Use the rank to remove species from the least important to the most important until you reach the error limit.

## Purpose

GA-SiG integrates the forward sensitivity equation along thermo-chemical trajectories, converts the sensitivities to a species-to-species matrix `r_AB`, and runs Dijkstra's algorithm from each target species. The sensitivity index is `dA/dtheta`. `A` is the time-integrated source term of a species. `theta` is the vector of logarithmic perturbations of the forward rate constants.

The package has two modes:

- `idt`: 0-D reactor trajectories, integrated in time up to ignition (ignition-delay-time oriented).
- `fs`: 1-D freely propagating flame profiles, each integrated as one trajectory in residence time up to the flame tip (flame-speed oriented).

One JSON file configures a run. For each mode, the package can run its own Cantera solver from the fuel, the oxidizer, the conditions and the mechanism. As an alternative, you can supply precomputed trajectories in the format of the `example` folder.

A reduction has three commands. `gasig run` makes the rank. `gasig reduce` measures the error of each reduced mechanism and selects the smallest one within your error tolerance. `gasig report` compares the selected mechanism with the detailed one in tables and figures. The user guide `docs/user_guide.md` walks through a complete reduction on the methane example.

## Tested environment

| Item | Version |
|---|---|
| Python | 3.11 |
| NumPy | 1.26.4 |
| Cantera | 3.2.0 |
| PyTorch | 2.8.0, CUDA 12.6 build |
| Matplotlib | 3.7 or later, optional: the figures of `gasig report` |
| GPU | optional: an NVIDIA GPU with a current driver |

The Docker image contains all of these. The package accepts Python 3.10 to 3.12, NumPy 1.24 or later, Cantera 3.1 or 3.2 and PyTorch 2.1 or later. Without a GPU the package runs on the CPU. The commands `reduce`, `report` and `template` need Cantera and NumPy only; they run without PyTorch.

## Install

### With Docker

1. Install Docker. For GPU runs, also install the NVIDIA Container Toolkit (Linux) or use Docker Desktop with WSL 2 (Windows).
2. Build the image in the repository folder:

   ```bash
   docker build -t ga-sig .
   ```

3. Do a check of the image:

   ```bash
   docker run --rm ga-sig info
   ```

For a machine without an NVIDIA GPU, build the CPU-only image instead. It is approximately one tenth of the size:

```bash
docker build -f Dockerfile.cpu -t ga-sig:cpu .
```

Use `ga-sig:cpu` in place of `ga-sig` in every command that follows, without `--gpus all`.

### With pip

1. Make a Python 3.10, 3.11 or 3.12 environment (3.11 is the tested version).
2. Install PyTorch for your CUDA version. Refer to pytorch.org for the command.
3. Install the package in the repository folder, with the report extra for the figures:

   ```bash
   pip install ".[report]"
   ```

4. Do a check:

   ```bash
   gasig info
   ```

## Configure

One JSON file holds every input path and every switch. Make a template that contains every default:

```bash
gasig template -o config.json
```

Then change the keys. The package resolves relative paths against the folder of the JSON file. Unknown keys are errors. Table 1 lists the main keys; `gasig template` shows all of them.

Table 1. Main configuration keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mechanism` | string | none | Cantera mechanism file (`.yaml`) |
| `device` | string | `auto` | `auto`, `cpu`, `cuda` or `cuda:N` |
| `output_dir` | string | `results` | Folder for all outputs, relative to the configuration file |
| `sensitivity.rate_law` | string | `original` | `original` or `corrected`. Refer to the section Versions of the method |
| `sensitivity.substep_tol` | number or null | `null` | Tolerance of the step-size guard; `null` disables the guard |
| `sensitivity.max_substeps` | integer | `64` | Upper limit of sub-steps per data interval |
| `sensitivity.concentration_floor` | number | `1e-30` | Additive floor on concentrations in the corrected rate law, kmol/m3 |
| `sensitivity.save_S` | boolean | `false` | Also write `S_abs_max_<mode>.npy` |
| `sensitivity.self_check` | boolean | `true` | Compare the torch rates with the Cantera rates at start-up |
| `aggregation.abs_S` | boolean | `true` | Sum `abs(S)` over the reactions of each species |
| `aggregation.normalization` | string | `S_AB_row_sum` | `S_AB_row_sum` or `S_row_sum` |
| `idt.enabled` | boolean | `false` | Run the idt mode |
| `idt.source` | string | `solver` | `solver` or `precomputed` |
| `idt.solver.fuel`, `idt.solver.oxidizer` | object | none | Mole-based compositions, for example `{"CH4": 1.0}` |
| `idt.solver.T_K`, `idt.solver.P_atm`, `idt.solver.phi` | lists | none | Initial conditions; the solver runs every combination |
| `idt.solver.reactor` | string | `constant_volume` | `constant_volume` or `constant_pressure` |
| `idt.solver.end_time_s` | number | `0.2` | Simulated time of each trajectory |
| `idt.solver.reference_species` | string | `OH` | Species whose mole-fraction peak marks ignition |
| `idt.solver.ignition_criterion` | string | `reference_peak` | Peak of the reference species within the ignition event, or `max_dTdt` |
| `idt.precomputed.*` | strings | none | Files `time`, `T`, `P`, `Y` and `index` |
| `fs.enabled` | boolean | `false` | Run the fs mode |
| `fs.source` | string | `solver` | `solver` or `precomputed` |
| `fs.flame_tip_T_fraction` | number or null | `0.99` | The integration stops where `T` first reaches this fraction of the maximum `T` |
| `fs.solver.fuel`, `fs.solver.oxidizer` | object | none | Mole-based compositions |
| `fs.solver.T_unburnt_K`, `fs.solver.P_atm`, `fs.solver.phi` | number, lists | `300`, none, none | Flame conditions |
| `fs.solver.domain_width_m` | number | `0.03` | Width of the flame domain |
| `fs.solver.transport_model` | string | `mixture-averaged` | Cantera transport model |
| `fs.precomputed.*` | strings | none | Files `position`, `velocity`, `T`, `P`, `Y` and optional `index` |
| `ranking.target_species` | list | `[]` | Target species names |
| `ranking.combine` | string | `auto` | `auto`, `idt`, `fs` or `hybrid` |
| `ranking.hybrid_weight_idt` | number | `0.5` | Weight of the idt matrix in the hybrid matrix |
| `ranking.eps_edge` | number | `1e-16` | Entries of `r_AB` at or below this value are not graph edges |
| `reduction.ranking_file` | string | `ranked_species.csv` | Rank file in `output_dir`, from the rank stage |
| `reduction.step` | integer | `1` | Species removed per sweep step |
| `reduction.min_species` | integer | `20` | Smallest mechanism size of the sweep; the last step is shorter when necessary |
| `reduction.stop_error` | number or null | `null` | The sweep stops after the maximum IDT error exceeds this fraction |
| `reduction.keep_species` | list | `[]` | Species that the sweep keeps, besides the targets; each name must be in the mechanism |
| `reduction.evaluate_fs` | boolean | `false` | Also evaluate the flame speed error |
| `reduction.error_tolerance` | number or null | `null` | Relative IDT error limit for the selection of the reduced mechanism |
| `reduction.write_mechanism` | boolean | `true` | Write the selected mechanism as `reduced_<n>sp.yaml` |
| `reduction.search.mode` | string | `full` | `full` evaluates every size in steps; `adaptive` searches the size within the tolerance |
| `reduction.search.early_exit` | boolean or null | `null` | Stop a size at the first condition above the tolerance; `null` is true in adaptive mode only |
| `reduction.search.robust_window` | integer | `3` | Sizes above the optimum that must also pass for the robust optimum |
| `reduction.search.max_evaluations` | integer or null | `null` | Adaptive: upper limit of reduced mechanisms evaluated |
| `reduction.idt.*`, `reduction.fs.*` | various | `null` | Overrides of the solver conditions for the evaluation |
| `report.T_points` | integer | `12` | Temperatures per pressure and equivalence ratio in the report |
| `report.plots` | boolean | `true` | Write the PNG figures (needs Matplotlib) |

The `example` folder holds the mechanism, the ethylene data and these configurations; the methane cases are in `example/methane`:

- `config_c2h4_precomputed.json`: the ethylene trajectories of the folder, idt mode, precomputed route.
- `methane/config_ch4_solver.json`: methane with USC Mech II, both modes, solver route, original method, with a `reduction` block (tolerance 5 percent) and a `report` block.
- `methane/config_ch4_corrected.json`: the same case with the corrected rate law and the step-size guard.
- `methane/config_ch4_idt.json`: the IDT-oriented reduction of methane with the settings of the manuscript: 1000 to 1600 K every 50 K, 1, 5 and 30 atm, phi 0.5, 1.0 and 2.0, constant-volume reactor, OH peak, tolerance 5 percent.

The folder `example/methane/sweeps` holds four configurations for the comparison of the method versions on methane. The file names give the rate law and the normalization: `ch4_<original|corrected>_<SAB|Srow>.json`. Each file has a `reduction` block for `gasig reduce`. On methane the corrected rate law keeps 25 species within 5 percent IDT error. The original law keeps 26 or 27 species. Every curve breaks at the removal of C2H4.

`example/r_AB.npy` and `example/ranked_sp.npy` are the results of the version 0.1 script for the ethylene trajectories. They correspond to `rate_law: "original"`, `substep_tol: null`, `abs_S: false` and `normalization: "S_row_sum"`. `config_c2h4_precomputed.json` uses the manuscript switches and gives a different rank.

## Run

1. Put the configuration file, the mechanism file and the data files in one folder. This is the run folder.
2. Start the run.

   With Docker and a GPU:

   ```bash
   docker run --rm --gpus all -v "<run folder>:/workspace" ga-sig run config.json
   ```

   With Docker on the CPU:

   ```bash
   docker run --rm -v "<run folder>:/workspace" ga-sig run config.json --device cpu
   ```

   With pip:

   ```bash
   gasig run config.json
   ```

3. Read `summary.json` and `ranked_species.csv` in the output folder.

To rank the same sensitivities again with other target species:

1. Change `ranking` in the configuration.
2. Run the rank command:

   ```bash
   gasig rank config.json
   ```

To measure how the rank performs, remove species in rank order and get the error of each reduced mechanism:

```bash
gasig reduce config.json
```

The command reads the rank file of the run (`reduction.ranking_file`). It removes `reduction.step` species per step, down to `reduction.min_species`. It builds each reduced mechanism in memory and computes the ignition delay time at the `idt.solver` conditions, or at the `reduction.idt` overrides. It writes `reduction_sweep.csv` with the maximum relative error per size. The sweep keeps the target species, the fuel and oxidizer species, the reference species and `reduction.keep_species`. Set `reduction.evaluate_fs` to true to add the flame speed error; this is slow. The command needs Cantera and NumPy only.

With `reduction.error_tolerance`, the command selects the smallest mechanism within the tolerance and writes it as `reduced_<n>sp.yaml`. It reports two sizes. The strict optimum is the smallest evaluated size within the tolerance. The robust optimum is the smallest size whose next `robust_window` larger sizes also pass; the report uses it. Set `reduction.search.mode` to `adaptive` for a large mechanism. The search evaluates a coarse set of sizes and probes the largest importance gaps of the rank. It refines every crossing of the tolerance and lists the intervals it did not refine.

To compare the selected mechanism with the detailed one:

```bash
gasig report config.json
```

The command writes the folder `report/` with `report.md` and `report.json`. The figures show the error curve, the ignition delay time against 1000/T for both mechanisms, and the flame speed when the sweep evaluated flames. A grid point where the Cantera solver fails gets no value and a warning in `report.log`; the report completes.

**NOTE:** The ignition delay time is the time of a solver sample. The solver steps change with the mechanism, so the ignition sample moves by up to one step. The step at ignition sets the noise of the error, which reaches 0.3 percent in the methane example at 1600 K and 1 atm.

**NOTE:** On Windows PowerShell, write the volume as `-v "${PWD}:/workspace"` from inside the run folder.

**NOTE:** The scripts `GA-SiG.py` and `graph_search_to_importance.py` at the root of the source checkout are wrappers around the package. `python GA-SiG.py config.json` runs the sensitivity stage only. `python graph_search_to_importance.py config.json` runs the rank. They are not in the wheel or in the image.

## Outputs

Table 2. Files in the output folder

| File | Content | Format |
|---|---|---|
| `r_AB_idt.npy`, `r_AB_fs.npy` | Normalized coupling matrix of each mode, shape (Ns, Ns) | NumPy |
| `S_AB_idt.npy`, `S_AB_fs.npy` | Coupling matrix before normalization | NumPy |
| `S_abs_max_idt.npy`, `S_abs_max_fs.npy` | Element-wise maximum of `abs(S)` over the trajectories, shape (Ns, Nr); written only with `sensitivity.save_S` | NumPy |
| `r_AB.npy` | Input matrix of the graph search (one mode or the hybrid) | NumPy |
| `importance.npy` | Importance index of every species, in mechanism order | NumPy |
| `ranked_species.csv` | Species from the least to the most important, with the index | CSV |
| `ranked_sp.npy` | The same rank in the format of version 0.1, shape (Ns, 2) | NumPy, strings |
| `trajectories_idt.json`, `trajectories_fs.json` | Conditions, ignition delay times, flame speeds and integration counters | JSON |
| `idt_data/`, `fs_data/` | Solver results in the precomputed format, for reuse | NumPy |
| `summary.json`, `config_resolved.json`, `gasig.log` | Run summary, the configuration with all defaults, the log | JSON, text |
| `reduction_sweep.csv`, `reduction_sweep.json`, `reduce.log` | Error of each reduced mechanism against its species count, the search record, the selection, and the log, from `gasig reduce` | CSV, JSON, text |
| `reduced_<n>sp.yaml` | The selected reduced mechanism, with transport data when the detailed mechanism has it | Cantera YAML |
| `report/` | `report.md`, `report.json`, `error_vs_species.png`, `idt_comparison.png`, `flame_speed_comparison.png`, from `gasig report` | Markdown, JSON, PNG |

A ranking-only run keeps the sensitivity record in `summary.json` and adds its own ranking block. `config_resolved.json` is a valid configuration file.

`gasig reduce` writes a new `reduce.log`, `reduction_sweep.csv` and `reduction_sweep.json` at each sweep. A case with no ignition gives `inf` in the CSV and `null` in the JSON; `n_unignited` counts these cases. A condition where the Cantera integrator fails on a reduced mechanism also gives `inf`; `n_solver_failures` counts these cases, and the sweep continues. The detailed mechanism must integrate at every condition. The JSON file is written after every size with `status: running`, and completed with the selection at the end. `max_fs_error` is `null` without `reduction.evaluate_fs`. A row with `complete: false` stopped at the first failed condition (early exit); its error is a lower bound, and `smallest_mechanism_within` counts complete rows only.

## Precomputed data format

Every file except `index` holds a Python dictionary. The keys are the trajectory numbers `0` to `N-1`. `index` is a plain integer array of shape (N,). Table 3 gives the arrays.

Table 3. Arrays of the precomputed route

| Mode | File | Value for trajectory `m` | Unit |
|---|---|---|---|
| idt | `time` | array (Nt,) | s |
| idt | `T` | array (Nt,) | K |
| idt | `P` | array (Nt,) | Pa |
| idt | `Y` | array (Nt, Ns) mass fractions | 1 |
| idt | `index` | integer array (N,): the ignition sample of each trajectory | 1 |
| fs | `position` | array (Nx,) | m |
| fs | `velocity` | array (Nx,), positive, flame-fixed frame | m/s |
| fs | `T`, `P`, `Y` | as for idt, along the grid | K, Pa, 1 |
| fs | `index` | optional integer array (N,): the flame-tip sample of each trajectory | 1 |

The solvers of the package write these files into `idt_data/` and `fs_data/`. A later run can read them with `source: "precomputed"`. The integrator clips negative mass fractions to zero and renormalizes each sample before it evaluates the rates.

## Versions of the method

The package keeps two versions of the rate law and two versions of the normalization. Select them in the configuration. Table 4 gives the meaning of each switch.

Table 4. Method switches

| Key and value | Meaning |
|---|---|
| `rate_law: "original"` | The rate law of the published results. Rate constants come from Cantera at the temperature and pressure of each sample, with the default composition of the mechanism. Three-body reactions have no third-body concentration `M_eff`. Every reaction has the reverse rate `kf / Kc`. |
| `rate_law: "corrected"` | The rates equal the Cantera rates at the temperature, pressure and composition of each sample. Three-body reactions carry the third-body concentration `M_eff`, which the package differentiates. Irreversible reactions have no reverse rate. |
| `substep_tol: null` | No step-size guard. Each data interval is one implicit Euler step. |
| `substep_tol: 0.01` | Step-size guard. The package compares one step with two half steps, species row by species row. If a row differs by more than the tolerance, the package doubles the number of sub-steps and compares again, up to `max_substeps`. We recommend this value with the corrected rate law. |
| `abs_S: true` | `S_AB(a, b)` is the sum of `abs(S(a, j))` over the reactions `j` of species `b`, as in the paper. |
| `abs_S: false` | The signed sum, as in the version 0.1 script. |
| `normalization: "S_AB_row_sum"` | `r_AB(a, b) = S_AB(a, b) / sum_b' S_AB(a, b')`. Each row of `r_AB` sums to 1. This is the definition in the paper. |
| `normalization: "S_row_sum"` | `r_AB(a, b) = S_AB(a, b) / sum_j max_m abs(S_m(a, j))`. This is the computation behind the published results. |

**CAUTION:** The two rate laws give different species ranks. Compare them on the same data before you use the corrected rate law for a reduction.

## Tests

The tests use USC Mech II. The solver tests burn methane. The regression tests use the shipped ethylene trajectories. Run the tests in the container from the repository folder:

```bash
docker run --rm -v "<repository folder>:/workspace" --entrypoint pytest ga-sig -q
```

Add `--gpus all` to include the GPU tests. Add `-m "not slow"` to skip the end-to-end tests that take minutes.

## Problems and causes

| Symptom | Cause | Action |
|---|---|---|
| `Configuration error: Unknown key` | A key name has a typo | Compare the key with `gasig template` |
| `device 'cuda' was requested but CUDA is not available` | No GPU, no driver, or the container has no `--gpus all` | Set `device` to `auto` or `cpu`, or add `--gpus all` |
| `The torch rate law does not reproduce Cantera` | The mechanism has a reaction type that the corrected rate law does not support | Report the reaction named in the message, or use `rate_law: "original"` |
| `the reference species ... peaks at the first sample` | The reference species is a reactant or a diluent | Use a radical such as `OH`, or `ignition_criterion: "max_dTdt"` |
| `no ignition detected` in the log | The reactor did not ignite before `end_time_s` | Increase `end_time_s`, or set `skip_unignited` to true |
| `u must be positive everywhere` | The profile is not a freely propagating flame in the flame-fixed frame | Supply a `FreeFlame` or `BurnerFlame` profile |
| `The temperature reaches ... at the first point already` | A flat or decreasing precomputed profile, or a low `flame_tip_T_fraction` | Supply the profile from the unburnt inlet, or raise the fraction |
| `reduction.keep_species names species that are not in the mechanism` | A name in the keep list has a typo | Correct the name |
| `The mechanism has species that are not in the rank file` | The rank comes from another mechanism | Run `gasig rank` with this mechanism, then `gasig reduce` |
| `conditions do not ignite with the detailed mechanism` | `end_time_s` or `max_steps` is too small for the evaluation conditions | Raise them in `reduction.idt`, or change the conditions |
| `no evaluated mechanism is within the tolerance` in the log | Every reduced size exceeds `reduction.error_tolerance` | Raise the tolerance, lower `reduction.min_species`, or rank with other targets |
| `Matplotlib is not installed` in the log | The report extra is not installed | `pip install ".[report]"`; the tables are written in every case |
| `Error: ...` with exit code 3 | A run-time error in a solver or in the data | Read the message; set `GASIG_DEBUG=1` for the traceback; `reduce.log` and `report.log` hold the traceback |
| Out of GPU memory | The mechanism is too large for the GPU | Use a larger GPU or `device: "cpu"` |

## Limits

- The Jacobian holds the falloff, pressure-dependent and Chebyshev rate constants fixed. It does not differentiate their composition dependence.
- The flame solver uses the Cantera `FreeFlame` object with mixture-averaged transport by default. Set `fs.solver.transport_model` to `multicomponent` for multicomponent transport.
- The sensitivity integration is serial over trajectories. One trajectory of a 1833-species mechanism took approximately 10 min on an NVIDIA L4 GPU.
- `gasig reduce` evaluates reduced mechanisms in memory and writes the selected mechanism only, not a file for every size.
- The selection uses the ignition delay time. The flame speed is evaluated and reported, but it does not decide the selection.

## Reference

Y.-J. Wu and K. C. Lin, GPU-Aided Sensitivity-in-graph (GA-SiG): An efficient sensitivity-based reduction method and its application to sustainable aviation fuel (submitted).

The user guide is `docs/user_guide.md`. The license is MIT; refer to `LICENSE`.
