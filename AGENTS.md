# Development guidance

This repository holds the `gasig` package:

- the GA-SiG sensitivity integration (idt and fs modes)
- the Cantera solvers that make the trajectories
- the graph search that writes the species rank

One JSON file configures a run. The `example` folder holds data and
configuration files, not code copies. `GA-SiG.py` and
`graph_search_to_importance.py` at the root are thin compatibility wrappers
around the package; change the package, not the wrappers.

## Layout

- `gasig/config.py` holds the configuration schema. Add every new key there
  with a type, a default and a one-line doc. Add the key to the README table.
- `gasig/kinetics.py` holds the two rate laws (`original`, `corrected`).
  `gasig/aggregate.py` holds the two normalizations and the `abs_S` switch.
  Keep both versions of each switch. The published results depend on
  `original`, `S_row_sum` and `abs_S: true`. The version 0.1 script and the
  shipped example depend on `original`, `S_row_sum` and `abs_S: false`.
- `tests/reference_v01.py` is the frozen version 0.1 code. Do not change it;
  the regression tests compare the package against it.
- `docs/design.md` records the method and the decisions. Update it when a
  decision changes.

## Documentation style

Write every document (README, docs, changelog, release note, docstring
narrative) in ASD-STE100 Simplified Technical English:

- instructions of 20 words or fewer
- descriptive sentences of 25 words or fewer
- active voice
- present tense for descriptions, imperatives for steps
- one term per concept
- `must` / `can` / `do not` for requirements
- no -ing verb forms, except in technical names such as `ranking`
- vertical lists for sequences and for three or more items
- numerals for measured quantities; counts below ten can be words

The user-level skill `ste100-docs` holds the rules and the templates for
notes, roadmaps, user guides and design details.

## Tests

- The tests use USC Mech II (`example/USCMechII.yaml`). The solver tests
  burn methane. The regression tests use the shipped ethylene trajectories.
- No local Python environment has a working PyTorch. Run the tests in the
  container from the repository folder:
  `docker run --rm -v "${PWD}:/workspace" --entrypoint pytest ga-sig -q`
  Add `--gpus all` for the GPU tests and `-m "not slow"` to skip the
  end-to-end solver tests. Use `ga-sig:cpu` for the CPU-only image.
- Validate numerical changes against Cantera (`net_rates_of_progress`,
  `net_production_rates_ddCi`) or against `tests/reference_v01.py`. A
  passing code graph is not proof of numerical correctness.

## Code graph cross-checking

- The local Codex project configuration provides `graphify` and
  `codebase_memory` MCP servers. Use both for structural questions and
  substantial refactors. Then verify the call sites in source. A missing
  graph edge is not proof of an absent dependency.
- Graphify is a snapshot. Refresh both graphs after code changes with:
  `& "$env:USERPROFILE\.local\share\code-intelligence\ga-sig\refresh.ps1"`.
  The command writes its report, indexes, source hashes and logs outside
  the checkout. If the tools are unavailable, use source inspection. Report
  that the graph validation did not run.
- Compare direct calls separately from callbacks, module entry points,
  constructors and external library calls. Follow `jacobians` into
  `TorchKinetics.rhs` in source: the callback through `torch.func.jacrev`
  is not a direct call edge. The `r_AB_<mode>.npy` hand-off between the
  sensitivity stage and the ranking stage needs a direct file-flow check.

## Repository hygiene

- `not-to-track/` holds the source files of the paper and notebooks with
  account metadata. It is in `.gitignore`; never stage it.
- Stage files by explicit path, not with `git add .`.
- Commit the example data only when it changes; the `.npy` files are large.
