# GA-SiG user guide

Date: 2026-10-23. Version: gasig 0.2.0. Status: current.

## Purpose

This guide shows how to reduce a detailed kinetic mechanism with GA-SiG, from the installation to the report. The example is methane with USC Mech II; the files are in the `example` folder of the repository. The README gives the overview and the full table of keys; `docs/design.md` gives the method.

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
2. Build the image in the repository folder. The CPU image is enough for the reduction steps; the sensitivity integration is faster on a GPU.

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
| run | `fs.enabled`, `fs.solver.*` | `true` and the flame conditions, when you want the flame-speed mode |
| run | `sensitivity.rate_law`, `aggregation.normalization` | `original` and `S_AB_row_sum` for the method of the paper |
| run | `ranking.target_species` | The species that the reduction must predict, for example the fuel, O2, OH, H2O and CO2 |
| reduce | `reduction.error_tolerance` | The allowed relative error of the ignition delay time, for example `0.05` |
| reduce | `reduction.step`, `reduction.min_species` | The sweep resolution and the smallest size |
| reduce | `reduction.search.mode` | `full` for the whole curve; `adaptive` for large mechanisms |
| reduce | `reduction.keep_species` | Species that must stay, besides the targets and the mixture species |
| reduce | `reduction.idt`, `reduction.fs` | Evaluation conditions that differ from the solver conditions |
| report | `report.T_points` | Temperatures per pressure and equivalence ratio in the comparison |

The example configuration `example/config_ch4_solver.json` has every key of a complete methane reduction.

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

### 3.2 Reduction sweep

```bash
docker run --rm -v "${PWD}:/workspace" ga-sig:cpu reduce config.json
```

The command removes species in rank order, builds every reduced mechanism in memory and computes the ignition delay time at the evaluation conditions. With `reduction.error_tolerance` it selects the smallest mechanism within the tolerance and writes it as `reduced_<n>sp.yaml`.

Two search modes exist:

- `full` evaluates every size from the detailed mechanism down to `reduction.min_species`, `reduction.step` species at a time. Use it for mechanisms with up to a few hundred species, and whenever you want the complete error curve.
- `adaptive` evaluates a coarse set of sizes and probes the largest importance gaps of the rank. It refines every crossing of the tolerance by bisection and reports the intervals it did not refine. Use it for large mechanisms. Set `reduction.search.max_evaluations` to bound the cost.

With the early exit (`reduction.search.early_exit`, on by default in adaptive mode), a size stops at the first condition above the tolerance. Its error in the table is then a lower bound, marked `complete: false`.

### 3.3 Report

```bash
docker run --rm -v "${PWD}:/workspace" ga-sig:cpu report config.json
```

The command reads the sweep result and selects the mechanism. It computes the ignition delay time of the detailed and the selected mechanism on `report.T_points` temperatures per pressure and equivalence ratio. It writes the folder `report/`.

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
| `reduction_sweep.csv` | One row per evaluated size: species, reactions, maximum and mean IDT error, unignited cases, flame-speed error, `complete`, the species removed since the size above, seconds |
| `reduction_sweep.json` | The same rows with every ignition delay time, the removal order, the search record and the selection |
| `reduced_<n>sp.yaml` | The selected mechanism |
| `reduce.log` | The log of the sweep; it holds the traceback of a failure |
| `report/report.md`, `report/report.json` | The report and its numbers |
| `report/error_vs_species.png` | The error curve with the tolerance and the optima |
| `report/idt_comparison.png` | The ignition delay time of the detailed mechanism (lines) and the selected mechanism (markers) |
| `report/flame_speed_comparison.png` | The flame speed, when the sweep evaluated flames |

### The error curve and the selection

The error does not decrease monotonically with the number of species. Solver noise moves it by a few tenths of a percent. The removal of one species can lower an error that an earlier removal raised. The package therefore reports two sizes:

- the strict optimum: the smallest evaluated size within the tolerance
- the robust optimum: the smallest size whose next `reduction.search.robust_window` larger sizes are also within the tolerance.

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
| `The mechanism has species that are not in the rank file` | Run `gasig rank` with this mechanism before `gasig reduce` |
| `Matplotlib is not installed` | Install the report extra: `pip install ".[report]"`; the tables are written in every case |
| The sweep is slow | Use `reduction.search.mode: "adaptive"`, fewer evaluation conditions, or `max_evaluations` |

## Example: methane with USC Mech II

The configuration `example/config_ch4_solver.json` reduces methane with the original method and a tolerance of 5 percent. The evaluation uses 12 ignition conditions (1200 to 1600 K, 1 and 10 atm, phi 0.5 and 1) and three flames.

1. Open a shell in the `example` folder of the repository.
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

The four configurations in `example/sweeps/` repeat the sweep with the corrected rate law and the second normalization. The note `docs/notes/2026-10-reduction-sweep-ch4.md` holds their results.
