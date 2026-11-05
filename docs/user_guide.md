# GA-SiG user guide

Date: 2026-11-05. Version: gasig 0.2.0. Status: current.

## Purpose

This guide shows how to reduce a detailed kinetic mechanism with GA-SiG, from the installation to the report. The example is methane with USC Mech II; the files are in the `example` folder of the repository. The README gives the overview and the full table of keys. The manuscript, cited in the README, gives the method.

## Workflow

A reduction has five steps:

1. Install the package (Docker or pip).
2. Prepare the run folder: the mechanism, the configuration file and, if you have them, precomputed trajectories.
3. Run the commands in order: `gasig run`, `gasig reduce`, `gasig report`.
4. Read the results: the error curve, the selected mechanism and the report.
5. Use the reduced mechanism in your own Cantera scripts.

Each command reads the same configuration file and writes into the same output folder.

## 1 Install

### With Docker

1. Install Docker. For GPU runs, install the NVIDIA Container Toolkit (Linux) or use Docker Desktop with WSL 2 (Windows).
2. Build the image in the repository folder. The CPU image is enough for the reduction steps and, on a laptop, for the sensitivity integration too. A consumer GPU in float64 was slower than the CPU up to 745 species.

   ```bash
   docker build -f Dockerfile.cpu -t ga-sig:cpu .
   ```

   For the CUDA image:

   ```bash
   docker build -t ga-sig .
   ```

3. Check the image:

   ```bash
   docker run --rm ga-sig:cpu info
   ```

   The output lists the versions of Python, NumPy, Cantera, PyTorch and Matplotlib.

**NOTE:** The CPU image also compiles the Zero-RK flame solvers for the flame-speed backend `fs.solver.backend: "zerork"`. Add `--target gasig` to the build command to leave them out. The README, section Flame speeds with Zero-RK, describes the backend and its citation.

### With pip

1. Make a Python 3.10 to 3.12 environment.
2. Install PyTorch for your CUDA version (refer to pytorch.org), or the CPU build.
3. Install the package with the report extra in the repository folder:

   ```bash
   pip install ".[report]"
   ```

4. Check the installation:

   ```bash
   gasig info
   ```

**NOTE:** The commands `reduce`, `report` and `template` need Cantera, NumPy and Matplotlib only. You can run them in a Cantera environment without PyTorch, for example on a machine without a GPU driver.

## 2 Prepare the inputs

### The run folder

Put these files in one folder:

- the mechanism file (Cantera `.yaml`)
- the configuration file (`config.json`)
- the precomputed data files, when you use that route.

The package resolves every relative path in the configuration against the folder of the configuration file. The output folder (`output_dir`) is relative to it too.

### The mechanism

The mechanism must load with `cantera.Solution`. The reduction steps work with any phase that Cantera's ideal-gas reactors accept. Transport data is necessary for the flame-speed mode and for a reduced mechanism file with transport.

### The configuration file

1. Write a template with every default:

   ```bash
   gasig template -o config.json
   ```

2. Set the keys of Table 1. Delete the keys you do not change, or keep them; unknown keys are errors.

Table 1. Keys to set for a reduction

| Step | Key | What to set |
|---|---|---|
| all | `mechanism` | The mechanism file name |
| all | `output_dir` | The output folder, for example `results` |
| run | `idt.enabled`, `idt.source` | `true`; `solver` or `precomputed` |
| run | `idt.solver.fuel`, `idt.solver.oxidizer` | Mole-based compositions, for example `{"CH4": 1.0}` and `{"O2": 0.21, "N2": 0.79}` |
| run | `idt.solver.T_K`, `idt.solver.P_atm`, `idt.solver.phi` | The ignition conditions; the solver runs every combination |
| run | `idt.solver.end_time_s` | Longer than the longest ignition delay time |
| run, reduce | `idt.solver.preconditioner` | `adaptive` by default; `adaptive_fast` halves the solve time of large mechanisms; `none` is the dense solver |
| run | `fs.enabled`, `fs.solver.*` | `true` and the flame conditions, when you want the flame-speed mode |
| run, reduce, report | `fs.solver.backend` | `cantera` (the `FreeFlame`) or `zerork` (the Zero-RK steady flame solver, constant Lewis numbers, in the CPU image). Compare the time and the flame speed of the two backends on your detailed mechanism before a sweep |
| run | `sensitivity.rate_law`, `aggregation.normalization`, `aggregation.abs_S` | `original`, `S_AB_row_sum` and `true` for the method of the paper and its published results. These are the defaults |
| run | `ranking.target_species` | The species that the reduction must predict, for example the fuel, O2, OH, H2O and CO2. At least one name must be a species of the mechanism |
| run | `ranking.combine` | `auto` ranks the hybrid matrix when both modes are enabled, else the enabled mode. `hybrid`, `idt` and `fs` need the modes they name to be enabled |
| reduce | `reduction.error_tolerance` | The allowed relative error of the ignition delay time, for example `0.05` |
| reduce | `reduction.step`, `reduction.min_species` | The sweep resolution and the smallest size |
| reduce | `reduction.search.mode` | `full` for the whole curve; `adaptive` for large mechanisms. `greedy` and `lk_greedy` are under development; refer to section 3.4 |
| reduce | `reduction.keep_species` | Species that must stay, besides the targets and the mixture species |
| reduce | `reduction.idt`, `reduction.fs` | Evaluation conditions that differ from the solver conditions |
| report | `report.T_points` | Temperatures per pressure and equivalence ratio in the comparison |

The example configuration `example/methane/config_ch4_solver.json` has every key of a complete methane reduction.

### Route A: the package runs the solvers

Set `idt.source` to `solver` and fill `idt.solver`. The package integrates a closed homogeneous reactor for every condition and stops at ignition. With `fs.enabled`, it also solves a freely propagating flame for every flame condition. The package writes the trajectories to `idt_data/` and `fs_data/`, so a later run can read them.

### Route B: precomputed trajectories

Set `idt.source` to `precomputed` and name the five files of `idt.precomputed` (`time`, `T`, `P`, `Y`, `index`). The README, Table 3, gives the format. The folder `example` holds ethylene trajectories in this format. The reduction step always runs the Cantera reactor; it needs the solver conditions in `idt.solver` or in `reduction.idt`.

### The targets, the tolerance and the protected species

- The target species anchor the rank. Choose the fuel, the oxidizer and the main products or radicals that your application must predict.
- The error tolerance is the relative error of the ignition delay time, as a fraction. `0.05` is 5 percent. The package compares the maximum error over all evaluation conditions with this value.
- The reduction never removes the target species, the fuel and oxidizer species, the reference species of the ignition criterion, or the species of `reduction.keep_species`.

## 3 Run

The commands below use the CPU image from inside the run folder. With pip, write `gasig` in place of the Docker prefix. On Windows PowerShell, write the volume as `-v "${PWD}:/workspace"`.

### 3.1 Sensitivity and rank

```bash
docker run --rm -v "${PWD}:/workspace" ga-sig:cpu run config.json
```

The command runs the solvers, or loads the precomputed files. It integrates the sensitivity equation along every trajectory, aggregates the coupling matrix and writes the rank `ranked_species.csv`. With a GPU, add `--gpus all` and use the CUDA image.

To rank the same sensitivities again with other targets, change `ranking` and run:

```bash
docker run --rm -v "${PWD}:/workspace" ga-sig:cpu rank config.json
```

The rank stage reads only the `r_AB` files of the enabled modes. To rank a hybrid matrix from two earlier runs in the same output folder, enable both modes and run `gasig run config.json --stage rank`. The run stops with a configuration error before the sensitivity stage when `ranking.target_species` is empty or holds no species of the mechanism.

### 3.2 Reduction sweep

```bash
docker run --rm -v "${PWD}:/workspace" ga-sig:cpu reduce config.json
```

The command removes species in rank order, builds every reduced mechanism in memory and computes the ignition delay time at the evaluation conditions. With `reduction.error_tolerance` it selects the smallest mechanism within the tolerance and writes it as `reduced_<n>sp.yaml`.

Two search modes are stable:

- `full` evaluates every size from the detailed mechanism down to `reduction.min_species`, `reduction.step` species at a time. Use it for mechanisms with up to a few hundred species, and whenever you want the complete error curve.
- `adaptive` evaluates a coarse set of sizes and probes the largest importance gaps of the rank. It refines every crossing of the tolerance by bisection and reports the intervals it did not refine. Use it for large mechanisms. Set `reduction.search.max_evaluations` to bound the cost.

Two more modes, `greedy` and `lk_greedy`, are under development. Section 3.4 describes them.

With the early exit (`reduction.search.early_exit`, on by default in adaptive, greedy and lk_greedy mode), a size stops at the first condition above the tolerance. Its error in the table is then a lower bound, marked `complete: false`.

A condition where the solver fails on a reduced mechanism counts as a failed condition: its error is `inf`, and `n_solver_failures` counts it. A reduced mechanism that ignites but does not form the reference species also counts as a failed condition. For the detailed mechanism, these failures stop the run. `reduce.failures.jsonl` records each failure.

### 3.3 Report

```bash
docker run --rm -v "${PWD}:/workspace" ga-sig:cpu report config.json
```

The command reads the sweep result and selects the mechanism. It computes the ignition delay time of the detailed and the selected mechanism on `report.T_points` temperatures per pressure and equivalence ratio. It writes the folder `report/`.

`gasig reduce` writes `reduction_sweep.json` with the status `running` after every size, and with the status `complete` at the end. When `reduction_sweep.json` has a status other than `complete`, the sweep did not complete. `gasig report` then selects only from the evaluated sizes. It writes a warning to `report.log` and to `report.md`, and sets `sweep_complete` to `false` in `report.json`. A smaller mechanism within the tolerance can exist. Run `gasig reduce` again to complete the sweep before you use the selected mechanism.

To compare a mechanism file from another source, or an earlier result, add `--mechanism`:

```bash
docker run --rm -v "${PWD}:/workspace" ga-sig:cpu report config.json --mechanism reduced_112sp.yaml
```

The command needs no sweep. The package looks for the file in the current folder, then in the output folder, then beside the configuration file. It uses the evaluation conditions of the configuration and, with `reduction.evaluate_fs`, the flame conditions. It writes the folder `report_<name>/` and the log `report_<name>.log`. Set `reduction.idt.T_K` to the two ends of the temperature range. Set `report.T_points` to the number of temperatures; the report fills the range with equal steps.

### 3.4 Search modes under development

The modes `greedy` and `lk_greedy` go beyond the method of the manuscript. The manuscript gives no evidence for them. Only the methane example has an end-to-end evaluation. Their keys, outputs and results can change in a later version.

**CAUTION:** These modes can give a reduced mechanism with a large error. You use them at your own risk. Compare each reduced mechanism with the detailed mechanism before you use it, for example with `gasig report --mechanism`.

The `greedy` mode tests one species at a time in rank order. A species whose removal takes the error above the tolerance stays in the mechanism, and the search continues with the next species. Use it if a size fails but most of the later species have no effect. A failed test stops at the first condition above the tolerance. An accepted removal costs a full evaluation. Set `reduction.search.max_evaluations` to bound the number of tests.

The `lk_greedy` mode uses the order of the linearised knockout (LK), not the GA-SiG rank. The LK stage of `gasig run` estimates the effect of the removal of each species along the stored ignition trajectories. The search tests this order in pieces: one species, or one chain unit that it removes as a whole. It tests the pieces with a very small LK score in blocks. An optional second pass tests the kept pieces again at a second tolerance. To use the mode:

1. Set `lk.enabled` to true. If you want chain units, set `units.detect` to true.
2. Set `reduction.search.mode` to `lk_greedy`, and set `reduction.error_tolerance`.
3. Optional: set `reduction.search.pass2_tolerance` for the second pass.
4. Run the three steps:

   ```bash
   docker run --rm -v "${PWD}:/workspace" ga-sig:cpu run config.json
   ```

   ```bash
   docker run --rm -v "${PWD}:/workspace" ga-sig:cpu reduce config.json
   ```

   ```bash
   docker run --rm -v "${PWD}:/workspace" ga-sig:cpu report config.json
   ```

To add the LK stage to an earlier run, use this command:

```bash
docker run --rm -v "${PWD}:/workspace" ga-sig:cpu run config.json --stage lk
```

The README, section Features under development, gives the method, the keys and the outputs. `example/methane/config_ch4_lk_greedy.json` is a complete configuration: on 12 methane ignition conditions it kept 19 species at 15 percent and 18 species at 20 percent.

**NOTE:** An image built before the `lk_greedy` mode rejects the `lk` and `units` keys. In that case, build the image again from the current source.

### Validate a rank without Docker

The reduction and report steps need Cantera, NumPy and Matplotlib only. When a rank exists (`ranked_species.csv` in the output folder, from a `gasig run` in the container or on another machine), you can run both steps in a Cantera environment on your own machine:

1. Install the package into that environment without its dependencies, so that pip does not pull PyTorch:

   ```bash
   pip install --no-deps -e .
   ```

2. Run the two steps from the folder of the configuration file:

   ```bash
   gasig reduce config.json
   gasig report config.json
   ```

`gasig info` shows `torch not installed` in such an environment; that is expected. To run the test suite there, install pytest (`pip install pytest`); the tests of the sensitivity integration skip without torch, the tests of the reduction, the search, the report and the configuration run.

### Expected run times

Table 2. Run times of the methane example on one CPU core in the container

| Command | Work | Time |
|---|---|---|
| `run` | 12 ignition cases, 3 flames, sensitivity integration, rank | approximately 12 min |
| `reduce`, full, step 1 to 15 species | 96 reduced mechanisms, 12 conditions each | approximately 2.5 min |
| `report`, 12 temperatures | 96 ignition cases | approximately 2 min |

A flame takes approximately 90 s with the default refinement criteria. `reduction.evaluate_fs` multiplies the sweep time by the number of flames; use it on a short list of sizes.

## 4 Read the results

### The output folder

Table 3. Files of a complete reduction

| File | Content |
|---|---|
| `ranked_species.csv` | The rank, from the least to the most important species, with the importance |
| `summary.json`, `gasig.log` | The run summary and its log |
| `lk_order.csv`, `lk_scores.npy`, `chain_units.json` | Under development, from `gasig run` with `lk.enabled` and `units.detect`: the LK order with the LK scores, the signed time-shift estimate of each species per trajectory, and the chain units. The LK score of a species is the maximum of the absolute time shift over the screened trajectories |
| `reduction_sweep.csv` | One row per evaluated size, or per test in `greedy` and `lk_greedy` mode: species, reactions, maximum and mean IDT error, unignited cases, solver failures, flame-speed error, `complete`, the species removed since the size above, `kept_back`, seconds |
| `reduction_sweep.json` | The same rows with every ignition delay time, the removal order (the accepted removals in greedy mode, with the rank in `rank_removal_order`), the search record and the selection; written after every size, `status` is `running` until the end |
| `reduced_<n>sp.yaml` | The selected mechanism |
| `reduce.log` | The log of the sweep; it holds the traceback of a failure |
| `gasig.failures.jsonl`, `reduce.failures.jsonl`, `report.failures.jsonl` | The failure file beside each log: one JSON line per recorded failure, with `time`, `fatal`, `category`, `type`, `message`, `stage` and `context`. A solver failure at one condition has `fatal: false`. The error that stops a command has `fatal: true` |
| `report/report.md`, `report/report.json` | The report and its numbers. `sweep_status` in `report.json` is the status of `reduction_sweep.json` (`complete`, `running`, or `unknown` when the file has no status). `sweep_complete` is `true` when the sweep completed |
| `report/error_vs_species.png` | The error curve with the tolerance and the optima |
| `report/idt_comparison.png` | The ignition delay time of the detailed mechanism (lines) and the selected mechanism (markers) |
| `report/flame_speed_comparison.png` | The flame speed, when the sweep evaluated flames |
| `report_<name>/`, `report_<name>.log`, `report_<name>.failures.jsonl` | The comparison of a mechanism file with the detailed mechanism, from `gasig report --mechanism` |

### The error curve and the selection

The error does not decrease monotonically with the number of species. Solver noise moves it by a few tenths of a percent. The removal of one species can lower an error that an earlier removal raised. The package therefore reports two sizes:

- the strict optimum: the smallest evaluated size within the tolerance
- the robust optimum: the smallest size whose next `reduction.search.robust_window` larger sizes are also within the tolerance.

The window counts evaluated sizes. With `reduction.step` above 1, it covers `robust_window` times `step` species. If no size meets the window, the package selects the strict optimum.

The report selects the robust optimum. An isolated size that passes while its neighbours fail passes because of error cancellation at the evaluation conditions; do not use it at other conditions. The `selection` block of `reduction_sweep.json` holds both sizes. In adaptive mode the `search` block lists the unresolved intervals; a smaller mechanism within the tolerance can exist there. To examine one, set `reduction.min_species` and `reduction.step` to that interval and run a full sweep.

A row with `complete: false` stopped at the first failed condition. Its error is a lower bound. The columns `smallest_mechanism_within` count complete rows only.

### The reduced mechanism in Cantera

The file `reduced_<n>sp.yaml` is a complete Cantera mechanism of the kept species: thermodynamic data, reactions and, when the detailed mechanism has it, transport data. Load it like any mechanism:

```python
import cantera as ct

gas = ct.Solution("results/reduced_25sp.yaml")
gas.TPX = 1400.0, ct.one_atm, "CH4:1, O2:2, N2:7.52"
reactor = ct.IdealGasMoleReactor(gas)
```

The file states `skip-undeclared-third-bodies: true`, so the third-body efficiencies of removed species are ignored, as in the sweep.

## 5 Solve problems

The README, section Problems and causes, lists the error messages and the actions. The most frequent cases of a reduction are:

| Symptom | Action |
|---|---|
| `conditions do not ignite with the detailed mechanism` | Raise `end_time_s` or `max_steps` in `reduction.idt`, or change the conditions |
| `no evaluated mechanism is within the tolerance` in the log | Raise the tolerance, lower `reduction.min_species`, or run `gasig rank` with other targets |
| `reduction_sweep.json has the status ...: the sweep did not complete` in `report.log` | Run `gasig reduce` again to complete the sweep, then `gasig report` |
| `'ranking.target_species' is empty` or `No name in 'ranking.target_species' is a species of the mechanism` | Name at least one species of the mechanism in `ranking.target_species` |
| `'ranking.combine' ... needs ...` | Enable the mode that `ranking.combine` names, or set `ranking.combine` to `auto` |
| `the reactor ignited ... but the mechanism does not form the reference species` | For the detailed mechanism, choose a reference species that the mechanism forms, for example `OH`, or set `ignition_criterion` to `max_dTdt` |
| `The mechanism has species that are not in the rank file` | Run `gasig rank` with this mechanism before `gasig reduce` |
| `Matplotlib is not installed` | Install the report extra: `pip install ".[report]"`; the tables are written in every case |
| The sweep is slow | Use `reduction.search.mode: "adaptive"`, fewer evaluation conditions, or `max_evaluations` |
| The sweep stops at a large size although the later species are harmless | A species early in the removal order breaks the tolerance at one condition. Try `reduction.search.mode: "greedy"` (under development, section 3.4) |
| `lk_order.csv is missing; the lk_greedy mode reads the LK order` | Set `lk.enabled` to true. Then run the `run` command of section 3.4 with `--stage lk` |
| `Unknown key 'lk'` in the container | The image holds an older version of the package. Build the image again from the current source |
| `Zero-RK is not available` | Build the CPU image without `--target gasig`, or set `fs.solver.zerork.home` or `ZERORK_HOME` |
| `the flame solver failed` with the `zerork` backend | The condition gets `inf` (`gasig reduce`) or no value (`gasig report`) and the run continues. A `gave no flame` line of an MPI attempt before the serial retry is normal. Set `fs.solver.zerork.work_dir` to a folder in the run folder and `keep_files` to true; read `log_<n>` in the subfolder of that flame |

## Example: methane with USC Mech II

The configuration `example/methane/config_ch4_solver.json` reduces methane with the original method and a tolerance of 5 percent. The evaluation uses 12 ignition conditions (1200 to 1600 K, 1 and 10 atm, phi 0.5 and 1) and three flames.

1. Open a shell in the `example/methane` folder of the repository.
2. Run the sensitivity integration and the rank (approximately 12 min on the CPU):

   ```bash
   docker run --rm -v "${PWD}:/workspace" ga-sig:cpu run config_ch4_solver.json
   ```

3. Run the sweep (approximately 2.5 min):

   ```bash
   docker run --rm -v "${PWD}:/workspace" ga-sig:cpu reduce config_ch4_solver.json
   ```

4. Write the report (approximately 2 min):

   ```bash
   docker run --rm -v "${PWD}:/workspace" ga-sig:cpu report config_ch4_solver.json
   ```

5. Open `results_ch4/report/report.md` and the figures. The selected mechanism is `results_ch4/reduced_<n>sp.yaml`.

To try the adaptive search on the same rank, set `reduction.search.mode` to `adaptive` in a copy of the configuration. Then run steps 3 and 4 again. The sweep then evaluates approximately 30 sizes instead of 96.

The four configurations in `example/methane/sweeps/` repeat the sweep with the corrected rate law and the second normalization. The corrected rate law keeps 25 species within 5 percent IDT error. The original law keeps 26 or 27 species. Every curve breaks at the removal of C2H4.

## Example: the IDT-oriented reduction of the manuscript

`example/methane/config_ch4_idt.json` applies the conditions of the manuscript's case study to methane: ignition delay times only (`fs.enabled` false, `ranking.combine` `idt`). It uses the switches of the published results: the original rate law, `abs_S` true and `S_AB_row_sum`. The conditions follow the paper's data collector: 1000 to 1600 K every 50 K, 1, 5 and 30 atm, phi 0.5, 1.0 and 2.0, a constant-volume reactor, rtol 1e-6, atol 1e-12 and the OH peak as the ignition sample. That is 117 trajectories. The targets are the fuel, O2, OH, H2O, HO2 and CO2, as in the ethylene example. The sweep runs every size down to 15 species with every condition evaluated. This gives a complete error curve, of the same type as Fig. 5 of the paper. The tolerance is 5 percent.

Table 4. Run times of the manuscript case

| Command | Work | Time |
|---|---|---|
| `run` | 117 ignition cases and their sensitivity integration, one CPU core | 2.5 min (measured) |
| `run`, `device` `cuda` | the same on an RTX 2050 (4 GB), float64 | 3.9 min (measured) |
| `reduce`, full, every condition | 96 sizes, 117 conditions each | estimate: 20 to 30 min |
| `report` | 13 temperatures, 9 pressure and equivalence-ratio pairs | estimate: approximately 2 min |

The GPU run is slower here, and also on a 745-species mechanism: 0.23 s per step on the card against 0.17 s on the CPU. A consumer card computes double precision slowly. Use `device` `cpu` on such a card. The ranks of both devices agree, up to a swap of two species with equal importance. A card with a full double-precision rate, such as an A100, is needed for a gain.

To shorten the sweep, set `reduction.search.mode` to `adaptive` (the early exit is then on); the search needs approximately 25 sizes, and failed sizes stop at the first failed condition. To shorten the run, keep every second temperature.
