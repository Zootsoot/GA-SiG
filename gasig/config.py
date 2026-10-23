"""Configuration file loading, validation and defaults.

One JSON file configures a GA-SiG run. This module checks every key against
the schema below, fills in the defaults, and resolves every file path against
the folder that holds the JSON file. Unknown keys are errors so that a typo in
a key name does not silently fall back to a default.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

NoneType = type(None)
NUMBER = (int, float)

TRANSPORT_MODELS = (
    "mixture-averaged", "multicomponent", "unity-Lewis-number",
    "mixture-averaged-CK", "multicomponent-CK", "ionized-gas",
)


class ConfigError(ValueError):
    """Raised when the configuration file is invalid."""


class Spec:
    """Schema entry for one leaf key."""

    def __init__(self, types, default=None, *, required=False, choices=None, path=False, doc=""):
        self.types = types if isinstance(types, tuple) else (types,)
        self.default = default
        self.required = required
        self.choices = choices
        self.path = path
        self.doc = doc


SCHEMA: dict[str, Any] = {
    "mechanism": Spec(str, required=True, path=True, doc="Cantera mechanism file (.yaml)."),
    "phase": Spec((str, NoneType), None, doc="Phase name inside the mechanism file, or null for the first phase."),
    "device": Spec(str, "auto", doc="'auto' (CUDA when available), 'cpu', 'cuda' or 'cuda:N'."),
    "dtype": Spec(str, "float64", choices=("float64", "float32"), doc="Floating-point type of the torch computation."),
    "output_dir": Spec(str, "results", path=True, doc="Folder for all outputs, relative to the configuration file."),
    "log_level": Spec(str, "info", choices=("debug", "info", "warning"), doc="Level of the console and file log."),
    "sensitivity": {
        "rate_law": Spec(str, "original", choices=("original", "corrected"),
                         doc="'original' is the rate law of the manuscript computation; 'corrected' reproduces "
                             "Cantera's rates (third-body concentration M_eff, falloff at the true composition, "
                             "no reverse rate on irreversible reactions)."),
        "substep_tol": Spec((*NUMBER, NoneType), None,
                            doc="Relative tolerance of the step-size guard per species row; null disables it. "
                                "1e-2 is the recommended value with the corrected rate law."),
        "max_substeps": Spec(int, 64, doc="Upper limit of sub-steps per data interval (at least 2 with the guard)."),
        "concentration_floor": Spec(NUMBER, 1e-30,
                                    doc="Additive floor on concentrations [kmol/m3] in the corrected rate law. "
                                        "Spurious rates are at most the floor times the largest pseudo-first-order "
                                        "coefficient of a reaction with an absent reactant."),
        "save_S": Spec(bool, False,
                       doc="Also save the element-wise maximum of abs(S) over the trajectories, shape (Ns, Nr), "
                           "as S_abs_max_<mode>.npy."),
        "progress_every": Spec(int, 50, doc="Log a progress line every N steps; 0 logs per trajectory only."),
        "self_check": Spec(bool, True, doc="Compare the torch rate law with Cantera reaction by reaction at start-up."),
    },
    "idt": {
        "enabled": Spec(bool, False, doc="Run the idt mode (0-D reactor trajectories up to ignition)."),
        "source": Spec(str, "solver", choices=("solver", "precomputed"),
                       doc="'solver' runs the Cantera reactor; 'precomputed' reads the files of the precomputed block."),
        "solver": {
            "fuel": Spec((dict, NoneType), None, doc="Fuel composition, mole basis, for example {'CH4': 1.0}."),
            "oxidizer": Spec((dict, NoneType), None, doc="Oxidizer composition, mole basis."),
            "T_K": Spec((list, NoneType), None, doc="Initial temperatures [K]; the solver runs every combination."),
            "P_atm": Spec((list, NoneType), None, doc="Initial pressures [atm]."),
            "phi": Spec((list, NoneType), None, doc="Equivalence ratios."),
            "reactor": Spec(str, "constant_volume", choices=("constant_volume", "constant_pressure"),
                            doc="Closed homogeneous reactor type."),
            "end_time_s": Spec(NUMBER, 0.2, doc="Simulated time per trajectory [s]."),
            "reference_species": Spec(str, "OH", doc="Species whose mole-fraction peak marks ignition."),
            "ignition_criterion": Spec(str, "reference_peak", choices=("reference_peak", "max_dTdt"),
                                       doc="Ignition sample: peak of the reference species, or maximum of dT/dt."),
            "rtol": Spec(NUMBER, 1e-6, doc="Relative tolerance of the reactor integrator."),
            "atol": Spec(NUMBER, 1e-12, doc="Absolute tolerance of the reactor integrator."),
            "max_steps": Spec(int, 200000, doc="Upper limit of solver steps per trajectory."),
            "skip_unignited": Spec(bool, False, doc="Drop trajectories with no detected ignition."),
            "save_trajectories": Spec(bool, True, doc="Write the trajectories to idt_data/ in the precomputed format."),
        },
        "precomputed": {
            "time": Spec((str, NoneType), None, path=True, doc="Dictionary file {m: (Nt,)} of times [s]."),
            "T": Spec((str, NoneType), None, path=True, doc="Dictionary file {m: (Nt,)} of temperatures [K]."),
            "P": Spec((str, NoneType), None, path=True, doc="Dictionary file {m: (Nt,)} of pressures [Pa]."),
            "Y": Spec((str, NoneType), None, path=True, doc="Dictionary file {m: (Nt, Ns)} of mass fractions."),
            "index": Spec((str, NoneType), None, path=True, doc="Integer array (N,) with the end index of each trajectory."),
        },
    },
    "fs": {
        "enabled": Spec(bool, False, doc="Run the fs mode (1-D flame profiles up to the flame tip)."),
        "source": Spec(str, "solver", choices=("solver", "precomputed"),
                       doc="'solver' runs the Cantera FreeFlame; 'precomputed' reads the files of the precomputed block."),
        "flame_tip_T_fraction": Spec((*NUMBER, NoneType), 0.99,
                                     doc="Integration stops where T first reaches this fraction of max T; "
                                         "null uses the full profile."),
        "solver": {
            "fuel": Spec((dict, NoneType), None, doc="Fuel composition, mole basis."),
            "oxidizer": Spec((dict, NoneType), None, doc="Oxidizer composition, mole basis."),
            "T_unburnt_K": Spec(NUMBER, 300.0, doc="Unburnt gas temperature [K]."),
            "P_atm": Spec((list, NoneType), None, doc="Pressures [atm]; the solver runs every combination."),
            "phi": Spec((list, NoneType), None, doc="Equivalence ratios."),
            "domain_width_m": Spec(NUMBER, 0.03, doc="Width of the flame domain [m]."),
            "transport_model": Spec(str, "mixture-averaged", choices=TRANSPORT_MODELS,
                                    doc="Cantera transport model of the flame."),
            "refine": {
                "ratio": Spec(NUMBER, 3.0, doc="Grid refinement: maximum ratio of adjacent cell widths."),
                "slope": Spec(NUMBER, 0.06, doc="Grid refinement: slope criterion."),
                "curve": Spec(NUMBER, 0.12, doc="Grid refinement: curvature criterion."),
                "prune": Spec(NUMBER, 0.03, doc="Grid refinement: prune criterion."),
            },
            "loglevel": Spec(int, 0, doc="Cantera flame solver log level."),
            "flame_tip_T_fraction": Spec((*NUMBER, NoneType), "unset",
                                         doc="Alias of fs.flame_tip_T_fraction; the alias wins when both are set."),
            "save_profiles": Spec(bool, True, doc="Write the profiles to fs_data/ in the precomputed format."),
        },
        "precomputed": {
            "position": Spec((str, NoneType), None, path=True, doc="Dictionary file {m: (Nx,)} of positions [m]."),
            "velocity": Spec((str, NoneType), None, path=True, doc="Dictionary file {m: (Nx,)} of velocities [m/s]."),
            "T": Spec((str, NoneType), None, path=True, doc="Dictionary file {m: (Nx,)} of temperatures [K]."),
            "P": Spec((str, NoneType), None, path=True, doc="Dictionary file {m: (Nx,)} of pressures [Pa]."),
            "Y": Spec((str, NoneType), None, path=True, doc="Dictionary file {m: (Nx, Ns)} of mass fractions."),
            "index": Spec((str, NoneType), None, path=True,
                          doc="Optional integer array (N,) with the flame-tip index of each profile; "
                              "overrides flame_tip_T_fraction."),
        },
    },
    "aggregation": {
        "abs_S": Spec(bool, True,
                      doc="Sum abs(S) over the reactions of each species (manuscript). "
                          "false sums the signed S, as the version 0.1 script did."),
        "normalization": Spec(str, "S_AB_row_sum", choices=("S_AB_row_sum", "S_row_sum"),
                              doc="'S_AB_row_sum' divides by the row sum of S_AB (manuscript text, intended); "
                                  "'S_row_sum' divides by the row sum of max abs(S) over reactions "
                                  "(the computation behind the published results)."),
    },
    "reduction": {
        "ranking_file": Spec(str, "ranked_species.csv", doc="Rank file in output_dir, written by the rank stage."),
        "step": Spec(int, 1, doc="Species removed per sweep step."),
        "min_species": Spec(int, 20, doc="The sweep stops before the mechanism has fewer species than this."),
        "stop_error": Spec((*NUMBER, NoneType), None,
                           doc="The sweep stops after the maximum IDT error exceeds this fraction; null runs to min_species."),
        "keep_species": Spec(list, [], doc="Species that are never removed, besides the target species."),
        "evaluate_fs": Spec(bool, False, doc="Also evaluate the flame speed error (slow)."),
        "error_tolerance": Spec((*NUMBER, NoneType), None,
                                doc="Relative IDT error limit for the selection of the reduced mechanism; "
                                    "null selects no mechanism."),
        "write_mechanism": Spec(bool, True,
                                doc="Write the selected reduced mechanism as a Cantera YAML file (needs error_tolerance)."),
        "search": {
            "mode": Spec(str, "full", choices=("full", "adaptive"),
                         doc="'full' evaluates every size in steps; 'adaptive' searches the smallest size "
                             "within error_tolerance with few evaluations."),
            "early_exit": Spec((bool, NoneType), None,
                               doc="Stop a size at the first condition above error_tolerance; null means true "
                                   "in adaptive mode and false in full mode."),
            "coarse_points": Spec(int, 16, doc="Adaptive: number of sizes in the coarse pass, denser at small sizes."),
            "importance_gap_probes": Spec(int, 5,
                                          doc="Adaptive: the largest importance gaps of the rank are probed on both sides."),
            "near_miss_factor": Spec(NUMBER, 2.0,
                                     doc="Adaptive: a failed interval is refined when its smaller error is below "
                                         "this factor times the tolerance."),
            "robust_window": Spec(int, 3,
                                  doc="Sizes above the optimum that must also pass for the robust optimum."),
            "noise": Spec(NUMBER, 0.003,
                          doc="Adaptive: errors within this band of the tolerance are undecided and resolved to width 1."),
            "max_evaluations": Spec((int, NoneType), None,
                                    doc="Adaptive: upper limit of reduced mechanisms evaluated; null is no limit."),
        },
        "idt": {
            "fuel": Spec((dict, NoneType), None, doc="Overrides idt.solver.fuel for the evaluation."),
            "oxidizer": Spec((dict, NoneType), None, doc="Overrides idt.solver.oxidizer."),
            "T_K": Spec((list, NoneType), None, doc="Overrides idt.solver.T_K."),
            "P_atm": Spec((list, NoneType), None, doc="Overrides idt.solver.P_atm."),
            "phi": Spec((list, NoneType), None, doc="Overrides idt.solver.phi."),
            "reactor": Spec((str, NoneType), None, choices=("constant_volume", "constant_pressure", None),
                            doc="Overrides idt.solver.reactor."),
            "end_time_s": Spec((*NUMBER, NoneType), None, doc="Overrides idt.solver.end_time_s."),
            "reference_species": Spec((str, NoneType), None, doc="Overrides idt.solver.reference_species."),
            "ignition_criterion": Spec((str, NoneType), None, choices=("reference_peak", "max_dTdt", None),
                                       doc="Overrides idt.solver.ignition_criterion."),
            "rtol": Spec((*NUMBER, NoneType), None, doc="Overrides idt.solver.rtol."),
            "atol": Spec((*NUMBER, NoneType), None, doc="Overrides idt.solver.atol."),
            "max_steps": Spec((int, NoneType), None, doc="Overrides idt.solver.max_steps."),
        },
        "fs": {
            "fuel": Spec((dict, NoneType), None, doc="Overrides fs.solver.fuel for the evaluation."),
            "oxidizer": Spec((dict, NoneType), None, doc="Overrides fs.solver.oxidizer."),
            "T_unburnt_K": Spec((*NUMBER, NoneType), None, doc="Overrides fs.solver.T_unburnt_K."),
            "P_atm": Spec((list, NoneType), None, doc="Overrides fs.solver.P_atm."),
            "phi": Spec((list, NoneType), None, doc="Overrides fs.solver.phi."),
            "domain_width_m": Spec((*NUMBER, NoneType), None, doc="Overrides fs.solver.domain_width_m."),
            "transport_model": Spec((str, NoneType), None, choices=(*TRANSPORT_MODELS, None),
                                    doc="Overrides fs.solver.transport_model."),
            "loglevel": Spec((int, NoneType), None, doc="Overrides fs.solver.loglevel."),
        },
    },
    "report": {
        "T_points": Spec(int, 12,
                         doc="Temperatures per pressure and equivalence ratio in the ignition-delay comparison."),
        "plots": Spec(bool, True, doc="Write the PNG figures (needs Matplotlib)."),
    },
    "ranking": {
        "target_species": Spec(list, [], doc="Target species names."),
        "combine": Spec(str, "auto", choices=("auto", "idt", "fs", "hybrid"),
                        doc="Matrix to rank: 'auto' takes the hybrid when both modes ran, else the one that ran."),
        "hybrid_weight_idt": Spec(NUMBER, 0.5, doc="Weight of r_AB(idt) in the hybrid matrix."),
        "eps_edge": Spec(NUMBER, 1e-16, doc="Entries of r_AB at or below this value are not graph edges."),
        "top_n_print": Spec(int, 15, doc="Number of species listed at each end of the rank in the log and summary."),
    },
}


def _type_name(types) -> str:
    names = []
    for t in types:
        names.append("null" if t is NoneType else t.__name__)
    return " or ".join(names)


def _validate(node_schema: dict, node: Any, where: str, base_dir: Path) -> dict:
    if not isinstance(node, dict):
        raise ConfigError(f"'{where or '<root>'}' must be a JSON object.")
    out: dict[str, Any] = {}
    for key in node:
        if key not in node_schema:
            raise ConfigError(f"Unknown key '{key}' in '{where or '<root>'}'. Allowed: {sorted(node_schema)}.")
    for key, spec in node_schema.items():
        path = f"{where}.{key}" if where else key
        if isinstance(spec, dict):
            out[key] = _validate(spec, node.get(key, {}), path, base_dir)
            continue
        if key in node:
            value = node[key]
            if isinstance(value, bool) and bool not in spec.types:
                raise ConfigError(f"'{path}' must be {_type_name(spec.types)}, got a boolean.")
            if not isinstance(value, spec.types):
                raise ConfigError(f"'{path}' must be {_type_name(spec.types)}, got {type(value).__name__}.")
            if spec.choices is not None and value not in spec.choices:
                allowed = ", ".join("null" if c is None else repr(c) for c in spec.choices)
                raise ConfigError(f"'{path}' must be one of {allowed}, got {value!r}.")
        elif spec.required:
            raise ConfigError(f"Missing required key '{path}'.")
        else:
            value = copy.deepcopy(spec.default)
        if spec.path and isinstance(value, str) and not Path(value).is_absolute():
            value = str((base_dir / value).resolve())
        out[key] = value
    return out


def _require(cfg: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise ConfigError(f"'{where}' needs the keys {missing}.")


def _is_positive_number(value) -> bool:
    return isinstance(value, NUMBER) and not isinstance(value, bool) and float(value) > 0.0


def _check_names(values: list, where: str) -> None:
    bad = [v for v in values if not isinstance(v, str) or not v]
    if bad:
        raise ConfigError(f"'{where}' must hold species names (strings), got {bad[:3]}.")


def _check_composition(block: dict, key: str, where: str) -> None:
    comp = block.get(key)
    if comp is None:
        return
    if not comp or not all(isinstance(k, str) and _is_positive_number(v) for k, v in comp.items()):
        raise ConfigError(f"'{where}.{key}' must map species names to positive amounts, for example {{\"CH4\": 1.0}}.")


def _check_positive_list(block: dict, key: str, where: str) -> None:
    values = block.get(key)
    if values is None:
        return
    if not values or not all(_is_positive_number(v) for v in values):
        raise ConfigError(f"'{where}.{key}' must be a non-empty list of positive numbers.")


def _check_positive(block: dict, key: str, where: str) -> None:
    value = block.get(key)
    if value is not None and not _is_positive_number(value):
        raise ConfigError(f"'{where}.{key}' must be positive.")


def _check_at_least(block: dict, key: str, where: str, minimum: int) -> None:
    value = block.get(key)
    if value is not None and value < minimum:
        raise ConfigError(f"'{where}.{key}' must be at least {minimum}.")


def _check_solver_values(cfg: dict) -> None:
    """Range checks of the solver blocks and of their overrides in the reduction block."""
    for where, block in (("idt.solver", cfg["idt"]["solver"]), ("reduction.idt", cfg["reduction"]["idt"])):
        _check_composition(block, "fuel", where)
        _check_composition(block, "oxidizer", where)
        for key in ("T_K", "P_atm", "phi"):
            _check_positive_list(block, key, where)
        for key in ("end_time_s", "rtol", "atol"):
            _check_positive(block, key, where)
        _check_at_least(block, "max_steps", where, 1)
    for where, block in (("fs.solver", cfg["fs"]["solver"]), ("reduction.fs", cfg["reduction"]["fs"])):
        _check_composition(block, "fuel", where)
        _check_composition(block, "oxidizer", where)
        for key in ("P_atm", "phi"):
            _check_positive_list(block, key, where)
        for key in ("T_unburnt_K", "domain_width_m"):
            _check_positive(block, key, where)
    _check_names(cfg["reduction"]["keep_species"], "reduction.keep_species")
    _check_names(cfg["ranking"]["target_species"], "ranking.target_species")


def _check_mode_requirements(cfg: dict) -> None:
    idt = cfg["idt"]
    if idt["enabled"]:
        if idt["source"] == "solver":
            _require(idt["solver"], ["fuel", "oxidizer", "T_K", "P_atm", "phi"], "idt.solver")
        else:
            _require(idt["precomputed"], ["time", "T", "P", "Y", "index"], "idt.precomputed")
    fs = cfg["fs"]
    if fs["enabled"]:
        if fs["source"] == "solver":
            _require(fs["solver"], ["fuel", "oxidizer", "P_atm", "phi"], "fs.solver")
        else:
            _require(fs["precomputed"], ["position", "velocity", "T", "P", "Y"], "fs.precomputed")
    alias = fs["solver"].get("flame_tip_T_fraction", "unset")
    if not (isinstance(alias, str) and alias == "unset"):
        fs["flame_tip_T_fraction"] = alias
    fs["solver"]["flame_tip_T_fraction"] = fs["flame_tip_T_fraction"]
    frac = fs["flame_tip_T_fraction"]
    if frac is not None and not (0.0 < float(frac) <= 1.0):
        raise ConfigError("'fs.flame_tip_T_fraction' must be in (0, 1] or null.")
    if not (idt["enabled"] or fs["enabled"]):
        raise ConfigError("Enable at least one mode: set 'idt.enabled' or 'fs.enabled' to true.")
    w = cfg["ranking"]["hybrid_weight_idt"]
    if not (0.0 <= float(w) <= 1.0):
        raise ConfigError("'ranking.hybrid_weight_idt' must be in [0, 1].")
    if cfg["ranking"]["top_n_print"] < 0:
        raise ConfigError("'ranking.top_n_print' must be 0 or more.")
    sens = cfg["sensitivity"]
    if sens["max_substeps"] < 1:
        raise ConfigError("'sensitivity.max_substeps' must be at least 1.")
    if sens["substep_tol"] is not None:
        if float(sens["substep_tol"]) <= 0.0:
            raise ConfigError("'sensitivity.substep_tol' must be positive or null.")
        if sens["max_substeps"] < 2:
            raise ConfigError("'sensitivity.max_substeps' must be at least 2 when the step-size guard is on.")
    if float(sens["concentration_floor"]) <= 0.0:
        raise ConfigError("'sensitivity.concentration_floor' must be positive.")
    red = cfg["reduction"]
    if red["step"] < 1:
        raise ConfigError("'reduction.step' must be at least 1.")
    if red["min_species"] < 1:
        raise ConfigError("'reduction.min_species' must be at least 1.")
    if red["stop_error"] is not None and float(red["stop_error"]) <= 0.0:
        raise ConfigError("'reduction.stop_error' must be positive or null.")
    if red["error_tolerance"] is not None and float(red["error_tolerance"]) <= 0.0:
        raise ConfigError("'reduction.error_tolerance' must be positive or null.")
    search = red["search"]
    if search["mode"] == "adaptive" and red["error_tolerance"] is None:
        raise ConfigError("'reduction.search.mode' adaptive needs 'reduction.error_tolerance'.")
    if search["coarse_points"] < 2:
        raise ConfigError("'reduction.search.coarse_points' must be at least 2.")
    if search["importance_gap_probes"] < 0:
        raise ConfigError("'reduction.search.importance_gap_probes' must be 0 or more.")
    if float(search["near_miss_factor"]) < 1.0:
        raise ConfigError("'reduction.search.near_miss_factor' must be at least 1.")
    if search["robust_window"] < 0:
        raise ConfigError("'reduction.search.robust_window' must be 0 or more.")
    if float(search["noise"]) < 0.0:
        raise ConfigError("'reduction.search.noise' must be 0 or more.")
    if search["max_evaluations"] is not None and search["max_evaluations"] < 1:
        raise ConfigError("'reduction.search.max_evaluations' must be at least 1 or null.")
    if cfg["report"]["T_points"] < 2:
        raise ConfigError("'report.T_points' must be at least 2.")
    _check_solver_values(cfg)


def validate_config(raw: dict, base_dir: str | Path) -> dict:
    """Validate a configuration dictionary and fill in the defaults.

    Relative paths, including defaulted ones, are resolved against
    ``base_dir``. The returned dictionary carries the resolved folder in
    ``cfg["_config_dir"]``; keys that start with an underscore are not part
    of the file format.
    """
    base_dir = Path(base_dir).resolve()
    cfg = _validate(SCHEMA, raw, "", base_dir)
    _check_mode_requirements(cfg)
    cfg["_config_dir"] = str(base_dir)
    return cfg


def load_config(path: str | Path) -> dict:
    """Load and validate the JSON configuration file at ``path``.

    UTF-8 with or without a byte-order mark and UTF-16 are accepted, so a
    template captured through a PowerShell redirection loads as well.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        raw = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path} is not valid JSON text: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must hold a JSON object at the top level.")
    return validate_config(raw, path.parent)


def public_config(cfg: dict) -> dict:
    """The configuration without the internal keys, for ``config_resolved.json``."""
    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}


def _template_node(node_schema: dict) -> dict:
    out = {}
    for key, spec in node_schema.items():
        if isinstance(spec, dict):
            out[key] = _template_node(spec)
        else:
            out[key] = copy.deepcopy(spec.default)
    return out


def template_config() -> dict:
    """Return a complete configuration with every default, as a starting point."""
    cfg = _template_node(SCHEMA)
    cfg["mechanism"] = "mechanism.yaml"
    cfg["idt"]["enabled"] = True
    cfg["idt"]["solver"].update({
        "fuel": {"CH4": 1.0},
        "oxidizer": {"O2": 0.21, "N2": 0.79},
        "T_K": [1200.0, 1400.0, 1600.0],
        "P_atm": [1.0, 10.0],
        "phi": [0.5, 1.0],
    })
    cfg["idt"]["precomputed"].update({
        "time": "time_all.npy", "T": "T_all.npy", "P": "P_all.npy",
        "Y": "Y_all.npy", "index": "index_all.npy",
    })
    cfg["fs"]["solver"].update({
        "fuel": {"CH4": 1.0},
        "oxidizer": {"O2": 0.21, "N2": 0.79},
        "P_atm": [1.0],
        "phi": [0.8, 1.0, 1.2],
    })
    cfg["fs"]["solver"].pop("flame_tip_T_fraction", None)
    cfg["fs"]["precomputed"].update({
        "position": "position_all.npy", "velocity": "velocity_all.npy",
        "T": "T_all.npy", "P": "P_all.npy", "Y": "Y_all.npy", "index": None,
    })
    cfg["ranking"]["target_species"] = ["CH4", "O2", "OH", "H2O", "CO2"]
    return cfg


def schema_rows(node_schema: dict | None = None, prefix: str = "") -> list[tuple[str, str, str, str]]:
    """Flatten the schema into (key, type, default, doc) rows for documentation."""
    rows = []
    for key, spec in (node_schema or SCHEMA).items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(spec, dict):
            rows.extend(schema_rows(spec, path))
        else:
            rows.append((path, _type_name(spec.types), json.dumps(spec.default), spec.doc))
    return rows
