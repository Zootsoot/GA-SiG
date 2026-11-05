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
| GPU | optional: an NVIDIA GPU with a full double-precision rate; a consumer card is slower than the CPU |

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

The CPU image also compiles the Zero-RK flame solvers (refer to the section Flame speeds with Zero-RK). To build the image without Zero-RK, add `--target gasig`:

```bash
docker build -f Dockerfile.cpu --target gasig -t ga-sig:cpu .
```

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

Then change the keys. The package resolves relative paths against the folder of the JSON file. Unknown keys are errors. Table 1 lists the keys of the stable features. Table 6 lists the keys of the features under development. `gasig template` shows all keys with their defaults.

Table 1. Configuration keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mechanism` | string | none | Cantera mechanism file (`.yaml`) |
| `phase` | string or null | `null` | Phase name in the mechanism file; `null` takes the first phase |
| `device` | string | `auto` | `auto`, `cpu`, `cuda` or `cuda:N` |
| `dtype` | string | `float64` | Floating-point type of the torch computation: `float64` or `float32`. The start-up self-check always runs in float64 |
| `output_dir` | string | `results` | Folder for all outputs, relative to the configuration file |
| `log_level` | string | `info` | Level of the console log and of the log file: `debug`, `info` or `warning` |
| `sensitivity.rate_law` | string | `original` | `original` or `corrected`. Refer to the section Versions of the method |
| `sensitivity.substep_tol` | number or null | `null` | Tolerance of the step-size guard; `null` disables the guard |
| `sensitivity.max_substeps` | integer | `64` | Upper limit of sub-steps per data interval |
| `sensitivity.concentration_floor` | number | `1e-30` | Additive floor on concentrations in the corrected rate law, kmol/m3 |
| `sensitivity.save_S` | boolean | `false` | Also write `S_abs_max_<mode>.npy` |
| `sensitivity.self_check` | boolean | `true` | Compare the torch rates with the Cantera rates at start-up |
| `sensitivity.progress_every` | integer | `50` | Write a progress line to the log every N steps; `0` writes one line per trajectory only |
| `aggregation.abs_S` | boolean | `true` | Sum `abs(S)` over the reactions of each species |
| `aggregation.normalization` | string | `S_AB_row_sum` | `S_AB_row_sum` or `S_row_sum` |
| `idt.enabled` | boolean | `false` | Run the idt mode |
| `idt.source` | string | `solver` | `solver` or `precomputed` |
| `idt.solver.fuel`, `idt.solver.oxidizer` | object | none | Mole-based compositions, for example `{"CH4": 1.0}` |
| `idt.solver.T_K`, `idt.solver.P_atm`, `idt.solver.phi` | lists | none | Initial conditions; the solver runs every combination |
| `idt.solver.reactor` | string | `constant_volume` | `constant_volume` or `constant_pressure` |
| `idt.solver.end_time_s` | number | `0.2` | Simulated time of each trajectory |
| `idt.solver.preconditioner` | string | `adaptive` | `adaptive` (Cantera preconditioner, GMRES), `adaptive_fast` (third-body and falloff terms left out of the preconditioner), `none` (dense solver) |
| `idt.solver.reference_species` | string | `OH` | Species whose mole-fraction peak marks ignition |
| `idt.solver.ignition_criterion` | string | `reference_peak` | Peak of the reference species within the ignition event, or `max_dTdt` |
| `idt.solver.rtol`, `idt.solver.atol`, `idt.solver.max_steps` | number, number, integer | `1e-6`, `1e-12`, `200000` | Relative and absolute tolerances of the reactor integrator; upper limit of solver steps per trajectory |
| `idt.solver.skip_unignited` | boolean | `false` | Leave out the trajectories with no detected ignition |
| `idt.solver.save_trajectories` | boolean | `true` | Write the trajectories to `idt_data/` in the precomputed format |
| `idt.precomputed.*` | strings | none | Files `time`, `T`, `P`, `Y` and `index` |
| `fs.enabled` | boolean | `false` | Run the fs mode |
| `fs.source` | string | `solver` | `solver` or `precomputed` |
| `fs.flame_tip_T_fraction` | number or null | `0.99` | The integration stops where `T` first reaches this fraction of the maximum `T` |
| `fs.solver.fuel`, `fs.solver.oxidizer` | object | none | Mole-based compositions |
| `fs.solver.T_unburnt_K`, `fs.solver.P_atm`, `fs.solver.phi` | number, lists | `300`, none, none | Flame conditions |
| `fs.solver.domain_width_m` | number | `0.03` | Width of the flame domain |
| `fs.solver.transport_model` | string | `mixture-averaged` | Cantera transport model; the `zerork` backend ignores it |
| `fs.solver.refine.ratio`, `fs.solver.refine.slope`, `fs.solver.refine.curve`, `fs.solver.refine.prune` | numbers | `3.0`, `0.06`, `0.12`, `0.03` | Grid refinement criteria of the Cantera `FreeFlame`: maximum ratio of adjacent cell widths, slope, curvature and prune |
| `fs.solver.loglevel` | integer | `0` | Log level of the Cantera flame solver |
| `fs.solver.save_profiles` | boolean | `true` | Write the profiles to `fs_data/` in the precomputed format |
| `fs.solver.flame_tip_T_fraction` | number or null | not set | Alias of `fs.flame_tip_T_fraction`; the alias wins when both are set |
| `fs.solver.backend` | string | `cantera` | `cantera` solves the `FreeFlame`; `zerork` runs the Zero-RK premixed steady flame solver. Refer to the section Flame speeds with Zero-RK |
| `fs.solver.zerork.home` | string or null | `null` | Zero-RK install folder, relative to the configuration file; `null` uses `ZERORK_HOME`, then `/opt/zero-rk` |
| `fs.solver.zerork.processes` | integer or null | `null` | MPI ranks per flame; `null` uses the available CPUs, at most 8 |
| `fs.solver.zerork.min_points_per_thickness` | number | `12` | Grid points per flame thickness that end the regrid loop |
| `fs.solver.zerork.rel_tol` | number | `1e-2` | Relative tolerance of the steady solver |
| `fs.solver.zerork.work_dir`, `fs.solver.zerork.keep_files` | string or null, boolean | `null`, `false` | Folder for the solver files, relative to the configuration file. Each process writes to its own folder `run-<host>-<pid>-<token>/`: `<flame>/` for each flame, `mechanisms/` for the CHEMKIN files. `keep_files` keeps them and needs `work_dir` |
| `fs.solver.zerork.continuation` | boolean | `true` | Start each flame from the nearest flame that the process solved before, of the same mechanism or of another one. The cold start runs if this start fails |
| `fs.solver.zerork.mpiexec`, `fs.solver.zerork.lewis_temperature_K`, `fs.solver.zerork.max_solves`, `fs.solver.zerork.timeout_s` | various | `mpirun --oversubscribe`, `1500`, `10`, `null` | MPI launcher, temperature of the Lewis numbers, solver calls per flame, time limit per call. A call over the time limit counts as failed, and the next attempt runs |
| `fs.precomputed.*` | strings | none | Files `position`, `velocity`, `T`, `P`, `Y` and optional `index` |
| `ranking.target_species` | list | `[]` | Target species names. The rank stage needs at least one species of the mechanism |
| `ranking.combine` | string | `auto` | `auto`, `idt`, `fs` or `hybrid`. `auto` takes the hybrid when both modes are enabled, else the enabled mode. `hybrid` needs both modes enabled; `idt` and `fs` need that mode enabled |
| `ranking.hybrid_weight_idt` | number | `0.5` | Weight of the idt matrix in the hybrid matrix |
| `ranking.eps_edge` | number | `1e-16` | Entries of `r_AB` at or below this value are not graph edges |
| `ranking.top_n_print` | integer | `15` | Number of species at each end of the rank in the log and in `summary.json` |
| `reduction.ranking_file` | string | `ranked_species.csv` | Rank file in `output_dir`, from the rank stage |
| `reduction.step` | integer | `1` | Species removed per sweep step |
| `reduction.min_species` | integer | `20` | Smallest mechanism size of the sweep; the last step is shorter when necessary |
| `reduction.stop_error` | number or null | `null` | The sweep stops after the maximum IDT error exceeds this fraction |
| `reduction.keep_species` | list | `[]` | Species that the sweep keeps, besides the targets; each name must be in the mechanism |
| `reduction.evaluate_fs` | boolean | `false` | Also evaluate the flame speed error |
| `reduction.error_tolerance` | number or null | `null` | Relative IDT error limit for the selection of the reduced mechanism |
| `reduction.write_mechanism` | boolean | `true` | Write the selected mechanism as `reduced_<n>sp.yaml` |
| `reduction.search.mode` | string | `full` | `full` evaluates every size in steps. `adaptive` searches the size within the tolerance. `greedy` and `lk_greedy` are under development; refer to the section Features under development |
| `reduction.search.early_exit` | boolean or null | `null` | Stop a size at the first condition above the tolerance; `null` is true in `adaptive`, `greedy` and `lk_greedy` mode |
| `reduction.search.robust_window` | integer | `3` | Number of the next larger evaluated sizes that must also pass for the robust optimum |
| `reduction.search.max_evaluations` | integer or null | `null` | `adaptive`, `greedy` and `lk_greedy`: upper limit of reduced mechanisms evaluated |
| `reduction.search.coarse_points` | integer | `16` | `adaptive`: number of sizes in the coarse pass, denser at small sizes |
| `reduction.search.importance_gap_probes` | integer | `5` | `adaptive`: number of the largest importance gaps of the rank that the search probes on both sides |
| `reduction.search.near_miss_factor` | number | `2.0` | `adaptive`: the search refines a failed interval when its smaller error is below this factor times the tolerance |
| `reduction.search.noise` | number | `0.003` | `adaptive`: an error within this band of the tolerance is undecided; the search resolves it to width 1 |
| `lk.*`, `units.*`, `reduction.search.block_size`, `tier0_score`, `pass2_tolerance`, `units` | various | various | Keys of the features under development; refer to Table 6 |
| `reduction.idt.*`, `reduction.fs.*` | various | `null` | Overrides of the solver conditions for the evaluation |
| `report.T_points` | integer | `12` | Temperatures per pressure and equivalence ratio in the report |
| `report.plots` | boolean | `true` | Write the PNG figures (needs Matplotlib) |

The `example` folder holds the mechanism and the ethylene data. The ethylene case is in `example/ethylene`; the methane cases are in `example/methane`:

- `ethylene/config_c2h4_precomputed.json`: the ethylene trajectories of `example`, idt mode, precomputed route, with a `reduction` block (tolerance 5 percent) and a `report` block.
- `ethylene/config_c2h4_solver.json`: the same case through the solver route: the 99 ignition cases of the manuscript, 1000 to 1600 K every 60 K, 1, 5 and 30 atm, phi 0.5, 1.0 and 2.0, constant volume, 0.2 s.
- `methane/config_ch4_solver.json`: methane with USC Mech II, both modes, solver route, original method, with a `reduction` block (tolerance 5 percent) and a `report` block.
- `methane/config_ch4_corrected.json`: the same case with the corrected rate law and the step-size guard.
- `methane/config_ch4_idt.json`: the IDT-oriented reduction of methane with the conditions of the manuscript: 1000 to 1600 K every 50 K, 1, 5 and 30 atm, phi 0.5, 1.0 and 2.0, constant-volume reactor, OH peak, tolerance 5 percent.
- `methane/config_ch4_lk_greedy.json`: under development; the `lk_greedy` search, with the linearised knockout (LK), on methane, with every configuration key. Refer to the section Features under development.

The folder `example/methane/sweeps` holds four configurations for the comparison of the method versions on methane. The file names give the rate law and the normalization: `ch4_<original|corrected>_<SAB|Srow>.json`. Each file has a `reduction` block for `gasig reduce`. On methane the corrected rate law keeps 25 species within 5 percent IDT error. The original law keeps 26 or 27 species. Every curve breaks at the removal of C2H4.

`example/r_AB.npy` and `example/ranked_sp.npy` are the results of the version 0.1 script for the ethylene trajectories. They correspond to `rate_law: "original"`, `substep_tol: null`, `abs_S: false` and `normalization: "S_row_sum"`. `ethylene/config_c2h4_precomputed.json` uses `rate_law: "original"`, `abs_S: true` and `normalization: "S_AB_row_sum"`, the switches of the published results. It gives a different rank.

**NOTE:** The version 0.1 script is an upload of an earlier test version of the notebook. It is not the method of the published results. Every configuration in `example/ethylene` and every configuration directly in `example/methane` uses the switches of the published results: `rate_law: "original"`, `abs_S: true` and `normalization: "S_AB_row_sum"` (refer to Table 4). These are also the defaults.

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

The rank stage reads only the `r_AB_<mode>.npy` files of the enabled modes. To rank the hybrid matrix from the files of earlier runs in the same output folder, enable both modes. The run stops with a configuration error before the sensitivity integration when `ranking.target_species` is empty or holds no species of the mechanism.

To measure how the rank performs, remove species in rank order and get the error of each reduced mechanism:

```bash
gasig reduce config.json
```

The command reads the rank file of the run (`reduction.ranking_file`). It removes `reduction.step` species per step, down to `reduction.min_species`. It builds each reduced mechanism in memory and computes the ignition delay time at the `idt.solver` conditions, or at the `reduction.idt` overrides. It writes `reduction_sweep.csv` with the maximum relative error per size. The sweep keeps the target species, the fuel and oxidizer species, the reference species and `reduction.keep_species`. Set `reduction.evaluate_fs` to true to add the flame speed error; this is slow. The command needs Cantera and NumPy only.

With `reduction.error_tolerance`, the command selects the smallest mechanism within the tolerance and writes it as `reduced_<n>sp.yaml`. It reports two sizes. The strict optimum is the smallest evaluated size within the tolerance. The robust optimum is the smallest size whose next `robust_window` larger sizes also pass; the report uses it. The window counts evaluated sizes. With `reduction.step` above 1, the window covers `robust_window` times `step` species. If no size meets the window, the command selects the strict optimum. Set `reduction.search.mode` to `adaptive` for a large mechanism. The search evaluates a coarse set of sizes and probes the largest importance gaps of the rank. It refines every crossing of the tolerance and lists the intervals it did not refine. Two more search modes, `greedy` and `lk_greedy`, are under development. Refer to the section Features under development.

To compare the selected mechanism with the detailed one:

```bash
gasig report config.json
```

The command writes the folder `report/` with `report.md` and `report.json`. The figures show the error curve, the ignition delay time against 1000/T for both mechanisms, and the flame speed when the sweep evaluated flames. A grid point where the Cantera solver fails gets no value and a warning in `report.log`; the report completes.

When `reduction_sweep.json` has a status other than `complete`, the sweep did not complete. `gasig report` then selects only from the evaluated sizes, and a smaller mechanism within the tolerance can exist. The command writes a warning to `report.log` and to `report.md`. It sets `sweep_complete` to `false` in `report.json`. Run `gasig reduce` again to complete the sweep before you use the selected mechanism.

To compare any mechanism file with the detailed mechanism, without a sweep:

```bash
gasig report config.json --mechanism reduced_112sp.yaml
```

The command computes the ignition delay time of both mechanisms on the evaluation conditions (`reduction.idt` over `idt.solver`, with `report.T_points` temperatures per pressure and equivalence ratio). With `reduction.evaluate_fs`, it also computes the flame speeds at the flame conditions. It writes the folder `report_<name>/` and the log `report_<name>.log` in the output folder.

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
| `report/` | `report.md`, `report.json`, `error_vs_species.png`, `idt_comparison.png`, `flame_speed_comparison.png`, from `gasig report`. In `report.json`, `sweep_status` is the status of `reduction_sweep.json` (`unknown` when the file has no status), and `sweep_complete` is `true` when the sweep completed | Markdown, JSON, PNG |
| `report_<name>/`, `report_<name>.log` | The comparison of the mechanism file `<name>.yaml` with the detailed mechanism, from `gasig report --mechanism`; no sweep is necessary | Markdown, JSON, PNG, text |
| `gasig.failures.jsonl`, `reduce.failures.jsonl`, `report.failures.jsonl`, `report_<name>.failures.jsonl` | The failure file beside each log. It holds one JSON line per recorded failure: `time`, `fatal`, `category`, `type`, `message`, and `stage` and `context` when known. A non-fatal failure, for example a solver failure at one condition, has `fatal: false`. The error that stops a command has `fatal: true`. `gasig run` and `gasig rank` add lines to `gasig.failures.jsonl`; `gasig reduce` and `gasig report` start a new file at each run | JSON lines |
| `lk_order.csv` | Under development: the LK order, in the format of `ranked_species.csv`. `importance` holds the LK score, `gasig_importance` the GA-SiG importance | CSV |
| `lk_scores.npy` | Under development: the signed time-shift estimate `v_b` of every species on each screened trajectory, shape (number of screened trajectories, Ns). A positive value is a longer ignition delay time, a negative value a shorter one. The LK score of a species is the maximum of `abs(v_b)` over the screened trajectories | NumPy |
| `chain_units.json` | Under development: the detected chain units, the links with their production and consumption reactions, and the detection settings | JSON |

A ranking-only run keeps the sensitivity record in `summary.json` and adds its own ranking block. `config_resolved.json` is a valid configuration file.

`gasig reduce` writes a new `reduce.log`, `reduction_sweep.csv` and `reduction_sweep.json` at each sweep. A case with no ignition gives `inf` in the CSV and `null` in the JSON; `n_unignited` counts these cases. A condition where the Cantera integrator fails on a reduced mechanism also gives `inf`; `n_solver_failures` counts these cases, and the sweep continues. A reduced mechanism that ignites but does not form the reference species also gives `inf`. `n_solver_failures` counts these cases too. `reduce.failures.jsonl` records each of these failures. The detailed mechanism must integrate at every condition and form the reference species; else the sweep stops. The JSON file is written after every size with `status: running`, and completed with the selection at the end. If the sweep stops before the end, the status stays `running`. `max_fs_error` is `null` without `reduction.evaluate_fs`. A row with `complete: false` stopped at the first failed condition (early exit); its error is a lower bound, and `smallest_mechanism_within` counts complete rows only. In greedy mode every test is a row. `removed_in_step` holds the species tested. `kept_back` is true when the species stays because its removal breaks the tolerance. The JSON `removal_order` then lists the accepted removals in their order, and `rank_removal_order` the rank.

In `lk_greedy` mode, every test is also a row. `removed_in_step` lists every species of the tested piece or block, and `rank_removal_order` holds the LK order. Each JSON row also holds `pass`, `test` (`single`, `unit` or `block`) and `tolerance`. The JSON `search.passes` gives the size, the kept pieces and the untested pieces after each pass.

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
| `normalization: "S_AB_row_sum"` | `r_AB(a, b) = S_AB(a, b) / sum_b' S_AB(a, b')`. Each row of `r_AB` sums to 1. This is the definition in the paper and the computation of the published results. |
| `normalization: "S_row_sum"` | `r_AB(a, b) = S_AB(a, b) / sum_j max_m abs(S_m(a, j))`. This is the normalization of the version 0.1 script, an earlier test version. |

**CAUTION:** The two rate laws give different species ranks. Compare them on the same data before you use the corrected rate law for a reduction.

## Flame speeds with Zero-RK

Zero-RK (Lawrence Livermore National Laboratory, BSD-3-Clause) solves one-dimensional laminar flames with sparse, preconditioned iterative methods. It is built for large mechanisms. This package has not measured a speed gain over the Cantera `FreeFlame`. On the 111-species methane flame Zero-RK took 224 s and the `FreeFlame` 64 s. On a 112-species jet-fuel mechanism at 473 K it took 118 s against 43 s. Each Zero-RK time includes a failed MPI start and the serial retry. Use this backend for constant-Lewis-number flames, for a cross-check, or for a mechanism that the `FreeFlame` does not solve. Set `fs.solver.backend` to `zerork` to use it for the flames of the fs mode, of `gasig reduce` with `reduction.evaluate_fs`, and of `gasig report`.

The package drives the solver in the way of the Zero-RK `flame_speed` example:

1. It writes the mechanism as CHEMKIN files with `cantera.yaml2ck`, once per mechanism.
2. It computes one Lewis number per species with `lewisGenerator.x`, from the inlet mixture at `lewis_temperature_K`.
3. It solves the first flame on a coarse grid with the pseudo-unsteady start, from an initial front at 30 percent of the domain. Every later flame starts from the nearest flame that is already solved (`continuation`). Then it rebuilds the grid around the flame and solves the steady problem again, until the grid holds `min_points_per_thickness` points per flame thickness.
4. It reads the flame speed from the solver log and the profile from the solver data file.

The steady solver of Zero-RK supports constant Lewis numbers only. Its flame speeds differ from the Cantera mixture-averaged values: by 1 percent for methane at phi 1.0. For a 112-species jet-fuel mechanism at 473 K and 1 atm, the difference is 8 percent at phi 1.0 and 21 percent at phi 0.9. Compare the two backends on the detailed mechanism before you mix them in one study. Each solver call runs on `processes` MPI ranks. The MPI build of Zero-RK stops at the first failed Newton step; the package then repeats the call with the serial solver. A solver failure at one condition gives `inf` for that condition in `gasig reduce` and no value in `gasig report`; the run continues.

`Dockerfile.cpu` compiles Zero-RK from source at a fixed commit (`ZERORK_REF`) and lets Open MPI run as root in the container. The CUDA image does not contain Zero-RK. Outside the image, install Zero-RK from its repository and set `ZERORK_HOME` or `fs.solver.zerork.home`.

If you use this backend, cite Zero-RK:

- M. J. McNenly, R. A. Whitesides, D. L. Flowers, Faster solvers for large kinetic mechanisms using adaptive preconditioners, Proc. Combust. Inst. 35 (2015) 581–587. https://doi.org/10.1016/j.proci.2014.05.113
- S. Lapointe, R. A. Whitesides, M. J. McNenly, Sparse, iterative simulation methods for one-dimensional laminar flames, Combust. Flame 204 (2019) 23–32. https://doi.org/10.1016/j.combustflame.2019.02.030
- Zero-RK, Lawrence Livermore National Laboratory, https://github.com/LLNL/zero-rk (LLNL-CODE-2025129).

## Features under development

The features in this section go beyond the method of the manuscript. They are under development:

- The manuscript does not use them and gives no evidence for them.
- The unit tests cover their logic, but only the methane example has an end-to-end evaluation.
- Their keys, outputs and results can change in a later version.

**CAUTION:** These features can give a reduced mechanism with a large error. You use them at your own risk. Compare each reduced mechanism with the detailed mechanism before you use it, for example with `gasig report --mechanism`.

Table 5. Features under development

| Feature | Key | Function |
|---|---|---|
| Greedy search | `reduction.search.mode: "greedy"` | Tests the species of the GA-SiG rank one at a time. Keeps each species whose removal breaks the tolerance |
| Linearised knockout (LK) | `lk.enabled` | Estimates the effect of the removal of each species along the idt trajectories, without a new reactor solve. Writes the LK order |
| Chain units | `units.detect` | Finds sequences of species that one reaction flux goes through, so that a search can remove each sequence as one piece |
| LK greedy search | `reduction.search.mode: "lk_greedy"` | Tests the LK order in pieces and blocks, with an optional second pass at a second tolerance |

### Greedy search

The `greedy` mode uses the GA-SiG rank (`reduction.ranking_file`):

1. The search removes the next species of the rank, together with the removals that it accepted before.
2. If the maximum IDT error is at or below `reduction.error_tolerance`, the removal stays.
3. If not, the species stays in the mechanism, and the search continues with the next species.

The search stops at `reduction.min_species`, at the end of the rank, or after `reduction.search.max_evaluations` tests. The selection is the smallest accepted mechanism; `robust_window` does not apply. Use this mode if a size fails but most of the later species have no effect. A failed test stops at the first condition above the tolerance. An accepted removal costs a full evaluation.

### Linearised knockout and chain units

The LK stage runs in `gasig run` after the rank, when `lk.enabled` is true. For each species, it removes the reactions of that species. Then it integrates the linearised change of the mass fractions along each screened idt trajectory, up to the sample at `lk.sample_fraction`. It does not solve the reactor again. The LK score of a species is the largest absolute value of the estimated shift of the ignition delay time over the screened trajectories. Each shift is a signed fraction of the ignition delay time of its trajectory; `lk_scores.npy` holds the signed shifts. The LK order sorts the species from the smallest to the largest score; equal scores keep the GA-SiG order. The stage runs on the CPU in float64, for every value of `device`.

A chain unit is a sequence of species in which each member has one significant production reaction and one significant consumption reaction. The consumption reaction of each member is the production reaction of the next member. The removal of one member cuts the sequence, so the `lk_greedy` search removes a unit as one piece. The detection stage runs in `gasig run` when `units.detect` is true. It uses the reaction extents along the idt trajectories.

Both stages need the idt mode. The LK stage and the `tail` scope of the detection also need the GA-SiG rank. `gasig run` needs PyTorch, so run the stages in the container. To add a stage to an earlier run, set `lk.enabled` or `units.detect` to true and use `--stage`:

```bash
gasig run config.json --stage lk
```

```bash
gasig run config.json --stage units
```

The stage then reads the idt trajectories: the precomputed files, or the solver results in `idt_data/`. When it needs the rank, it also reads `ranked_species.csv`.

### LK greedy search

The `lk_greedy` mode uses the LK order (`lk.output_file`), not the GA-SiG rank. It divides the order into pieces. A piece is a chain unit or one species. A unit takes the position of its first member in the LK order, and the search never divides it. The search does not use a unit with a protected member as a unit; it makes each other member a single piece. The search runs as follows:

1. A piece is tier 0 when every member has an LK score at or below `reduction.search.tier0_score`.
2. The search tests consecutive tier-0 pieces as one block of up to `block_size` species.
3. If a block fails, the search divides it into two halves and tests each half. It continues down to single pieces.
4. Each other piece gets its own test. A failed piece stays in the mechanism.
5. With `pass2_tolerance`, a second pass starts from the result of the first pass. It tests each kept piece again, alone and in order, at the second tolerance.

The selection is the mechanism at the end of the first pass, at `reduction.error_tolerance`. When the second pass removes at least one piece, it writes one more mechanism file, `reduced_<n>sp.yaml`, with its own size. `reduction.stop_error` and `robust_window` do not apply. The search does not test the same set of removed species twice at the same tolerance. When the tests reach `max_evaluations`, the other pieces stay untested, and no second pass runs.

To use the mode:

1. Set `lk.enabled` to true. If you want chain units, set `units.detect` to true.
2. Set `reduction.search.mode` to `lk_greedy` and set `reduction.error_tolerance`.
3. Run `gasig run config.json`.
4. Run `gasig reduce config.json`.
5. Run `gasig report config.json`.

Table 6. Keys of the features under development

| Key | Type | Default | Meaning |
|---|---|---|---|
| `reduction.search.block_size` | integer | `16` | `lk_greedy`: upper limit of species in one block of tier-0 pieces; `1` tests every piece alone |
| `reduction.search.tier0_score` | number | `0.003` | `lk_greedy`: a piece is tier 0 when every member has an LK score at or below this value |
| `reduction.search.pass2_tolerance` | number or null | `null` | `lk_greedy`: tolerance of the second pass; at least `reduction.error_tolerance`; `null` makes no second pass |
| `reduction.search.units` | list or null | `null` | `lk_greedy`: chain units as lists of species names; `null` reads `units.output_file`. Without that file, every species is its own piece |
| `lk.enabled` | boolean | `false` | Run the LK stage after the rank; needs the idt mode |
| `lk.trajectories` | list or null | `null` | Indices of the idt trajectories to screen, in the order of `trajectories_idt.json`; `null` screens every trajectory |
| `lk.sample_fraction` | number | `0.5` | The stage takes the LK score at the stored sample nearest to this fraction of the ignition delay time of the trajectory; in (0, 1] |
| `lk.rate_law` | string | `corrected` | Rate law of the LK Jacobians: `original` or `corrected` |
| `lk.output_file` | string | `lk_order.csv` | LK order in `output_dir`; `lk_greedy` reads this file |
| `units.detect` | boolean | `false` | Run the detection of chain units after the rank; needs the idt mode |
| `units.scope` | string | `all` | `all` examines every removable species; `tail` examines only the removable species below the median GA-SiG importance of the removable species |
| `units.share` | number | `0.01` | A reaction is significant for a species when it gives more than this share of the time-integrated production, or consumption, of the species |
| `units.output_file` | string | `chain_units.json` | Detected units in `output_dir`; `lk_greedy` reads this file when `reduction.search.units` is `null` |

`example/methane/config_ch4_lk_greedy.json` holds a complete configuration. It burns methane with USC Mech II at 12 ignition conditions: 1200, 1400 and 1600 K, 1 and 10 atm, phi 0.5 and 1.0. The tolerance is 15 percent, and the second pass is at 20 percent. In the CPU container on one laptop, `gasig run` took 511 s, and the LK stage took 326 s of this time. `gasig reduce` made 33 tests in 14 s. The first pass kept 19 species, and the second pass kept 18. The detection found no chain unit in this case.

## Tests

The tests use USC Mech II. The solver tests burn methane. The regression tests use the shipped ethylene trajectories. Run the tests in the container from the repository folder:

```bash
docker run --rm -v "<repository folder>:/workspace" --entrypoint pytest ga-sig -q
```

Add `--gpus all` to include the GPU tests. Add `-m "not slow"` to skip the end-to-end tests that take minutes, among them the methane flame with Zero-RK (markers `slow` and `zerork`). The `zerork` test runs only where Zero-RK is installed, for example in the CPU image.

## Problems and causes

| Symptom | Cause | Action |
|---|---|---|
| `Configuration error: Unknown key` | A key name has a typo | Compare the key with `gasig template` |
| `Unknown key 'lk'` or `'units'`, or `lk_greedy` is not an allowed mode, in the container | The image holds an older version of the package | Build the image again from the current source |
| `lk_order.csv is missing; the lk_greedy mode reads the LK order` | The run had no LK stage | Set `lk.enabled` to true. Then run `gasig run config.json --stage lk` |
| `no chain units: ... is missing` in `reduce.log` | The run had no detection stage, and `reduction.search.units` is `null` | No action is necessary: every species is its own piece. To use units, set `units.detect` to true. Then run `gasig run config.json --stage units` |
| `device 'cuda' was requested but CUDA is not available` | No GPU, no driver, or the container has no `--gpus all` | Set `device` to `auto` or `cpu`, or add `--gpus all` |
| `The torch rate law does not reproduce Cantera` | The mechanism has a reaction type that the corrected rate law does not support | Report the reaction named in the message, or use `rate_law: "original"` |
| `'ranking.target_species' is empty` or `No name in 'ranking.target_species' is a species of the mechanism` | The rank has no valid target species. The run stops before the sensitivity integration | Name at least one species of the mechanism in `ranking.target_species` |
| `'ranking.combine' ... needs ...` | `ranking.combine` names a mode that is not enabled | Enable the mode, or change `ranking.combine`. `auto` ranks the enabled modes |
| `the reference species ... peaks at the first sample` | The reference species is a reactant or a diluent | Use a radical such as `OH`, or `ignition_criterion: "max_dTdt"` |
| `the reactor ignited ... but the mechanism does not form the reference species` | The mechanism has no formation path of the reference species. For the detailed mechanism, the run stops. For a reduced mechanism, the condition counts as failed in `gasig reduce` and gets no value in `gasig report` | Choose a reference species that the detailed mechanism forms, for example `OH`, or set `ignition_criterion: "max_dTdt"` |
| `the reference species has no peak inside the ignition event` in the log | The reference species still rises after the ignition event. A local maximum later than 2 times the heat-release time is drift, not ignition. The ignition sample is the temperature peak, or the mean of the two steepest rises when the temperature also has no peak in the event | No action is necessary. For a sharper criterion, choose another reference species or set `ignition_criterion: "max_dTdt"` |
| `no ignition detected` in the log | The reactor did not ignite before `end_time_s` | Increase `end_time_s`, or set `skip_unignited` to true |
| `u must be positive everywhere` | The profile is not a freely propagating flame in the flame-fixed frame | Supply a `FreeFlame` or `BurnerFlame` profile |
| `The temperature reaches ... at the first point already` | A flat or decreasing precomputed profile, or a low `flame_tip_T_fraction` | Supply the profile from the unburnt inlet, or raise the fraction |
| `reduction.keep_species names species that are not in the mechanism` | A name in the keep list has a typo | Correct the name |
| `The mechanism has species that are not in the rank file` | The rank comes from another mechanism | Run `gasig rank` with this mechanism, then `gasig reduce` |
| `conditions do not ignite with the detailed mechanism` | `end_time_s` or `max_steps` is too small for the evaluation conditions | Raise them in `reduction.idt`, or change the conditions |
| `reduction_sweep.json has the status ...: the sweep did not complete` in `report.log` | `gasig reduce` stopped before the end of the sweep. The report uses only the evaluated sizes | Run `gasig reduce` again, then `gasig report` |
| `no evaluated mechanism is within the tolerance` in the log | Every reduced size exceeds `reduction.error_tolerance` | Raise the tolerance, lower `reduction.min_species`, or rank with other targets |
| `Matplotlib is not installed` in the log | The report extra is not installed | `pip install ".[report]"`; the tables are written in every case |
| `Zero-RK is not available` | The image was built with `--target gasig`, or `fs.solver.zerork.home` is wrong | Build the full CPU image, or point `home` or `ZERORK_HOME` at a Zero-RK install |
| `the flame solver failed` in `reduce.log` or in the report log | Zero-RK gave no flame at this condition after the serial retry; the condition gets `inf` (`gasig reduce`) or no value (`gasig report`). A `gave no flame` line of an MPI attempt, followed by `the serial solver repeats it` or `the pseudo-unsteady solver continues`, is normal | Set `fs.solver.zerork.work_dir` to a folder in the run folder and `keep_files` to true. Read `log_<n>`, `log_<n>s` and `log_<n>p` in the subfolder of the flame. Change `rel_tol` or `processes` |
| `Error: ...` with exit code 3 | A run-time error in a solver or in the data | Read the message; set `GASIG_DEBUG=1` for the traceback; `reduce.log` and `report.log` hold the traceback. The failure file (`*.failures.jsonl`) in the output folder records the error with its stage and context |
| Out of GPU memory | The mechanism is too large for the GPU | Use a larger GPU or `device: "cpu"` |
| The GPU run is slower than the CPU run | The card computes double precision slowly (consumer cards) | Use `device: "cpu"`; on an RTX 2050 the CPU was faster up to 745 species |

## Limits

- The Jacobian holds the falloff, pressure-dependent and Chebyshev rate constants fixed. It does not differentiate their composition dependence.
- The flame solver uses the Cantera `FreeFlame` object with mixture-averaged transport by default. Set `fs.solver.transport_model` to `multicomponent` for multicomponent transport, or `fs.solver.backend` to `zerork` for the Zero-RK solver with constant Lewis numbers.
- The sensitivity integration is serial over trajectories. One trajectory of a 1833-species mechanism took approximately 10 min on an NVIDIA L4 GPU.
- `gasig reduce` evaluates reduced mechanisms in memory and writes the selected mechanism only, not a file for every size.
- The selection uses the ignition delay time. The flame speed is evaluated and reported, but it does not decide the selection.
- The greedy search, the LK stage, the chain units and the LK greedy search are under development. Only the methane example has an end-to-end evaluation. Refer to the section Features under development.

## Reference

Y.-J. Wu and K. C. Lin, GPU-Aided Sensitivity-in-graph (GA-SiG): An efficient sensitivity-based reduction method and its application to sustainable aviation fuel (submitted).

The user guide is `docs/user_guide.md`. The license is MIT; refer to `LICENSE`.
