"""Reduction sweep: remove species in rank order and measure the error.

The sweep reads the rank of an earlier run (``ranked_species.csv`` in the
output folder), removes species from the least important upward, builds
each reduced mechanism in memory, and evaluates the ignition delay time
(and optionally the flame speed) against the detailed mechanism. The
result is the curve of the maximum relative error against the number of
species, the quantity the paper uses to select a reduced mechanism. With
``reduction.error_tolerance`` the sweep also selects the smallest mechanism
within the tolerance and writes it as a Cantera YAML file.

Four search modes exist. ``full`` evaluates every size in steps of
``reduction.step``. ``adaptive`` evaluates a coarse set of sizes, probes the
largest importance gaps of the rank, refines every crossing of the
tolerance and reports the intervals it left unresolved
(:mod:`gasig.search`). ``greedy`` tests one species at a time in rank order.
It keeps a species whose removal breaks the tolerance and continues.
``lk_greedy`` walks the order of the linearised knockout (``lk.output_file``)
in pieces: a chain unit is one piece, every other species is its own piece.
Consecutive tier-0 pieces are tested in blocks, a failed block is halved,
and an optional second pass tests the kept pieces again at
``reduction.search.pass2_tolerance``. With the early exit, a size stops at
the first condition above the tolerance; its error is then a lower bound and
the row is marked ``complete: false``.

A solver failure at one condition counts as a failed condition (inf). The
sweep records it in ``reduce.failures.jsonl`` (:mod:`gasig.errors`) and
continues. A reduced mechanism that ignites but does not form the reference
species gives no ignition delay time; that condition also counts as failed.

The module needs Cantera and NumPy only; it does not import torch.

Reduced mechanism
-----------------
A reduced mechanism keeps the species of the ``keep`` list and every
reaction whose reactants and products are all kept. A three-body or
falloff reaction with explicit colliders (default efficiency 0) is kept
only when at least one of its colliders is kept; these are the reactions
that Cantera itself keeps alive when it builds a Solution from a species
subset. The kinetics ignores the third-body efficiencies of removed
species. The reduced Solution shares the Species, Reaction and rate
objects with the detailed Solution, so do not change them in place. The
reduced Solution is an ideal gas with the kinetics of the detailed one.
"""
from __future__ import annotations

import collections
import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import cantera as ct
import numpy as np

from .config import load_config
from .errors import (ConfigError, FlameSolverError, IgnitionSolverError, InputError, MechanismError,
                     MissingInputError, ReductionError, SolverError, failure_counts, record_failure)
from .logs import log_failure, setup_logging, teardown_logging
from .mechanism import drop_blank_text, write_yaml
from .search import (GreedyResult, SearchParams, adaptive_search, gap_probes, greedy_search, lk_greedy_search,
                     select_optima)
from .solvers.flame import flame_conditions, run_flame
from .solvers.idt import idt_conditions, run_idt

log = logging.getLogger("gasig.reduce")

IDT_KEYS = ("fuel", "oxidizer", "T_K", "P_atm", "phi", "reactor", "end_time_s", "reference_species",
            "ignition_criterion", "rtol", "atol", "max_steps", "preconditioner")
FS_KEYS = ("fuel", "oxidizer", "T_unburnt_K", "P_atm", "phi", "domain_width_m", "transport_model", "loglevel",
           "backend")
CSV_FIELDS = ["n_species", "n_reactions", "max_idt_error", "mean_idt_error", "n_unignited", "n_solver_failures",
              "max_fs_error", "complete", "removed_in_step", "kept_back", "seconds"]


def load_mechanism(path: str, phase: str | None = None, stage: str | None = None) -> ct.Solution:
    """Load a Cantera mechanism; a file that Cantera cannot read raises a :class:`MechanismError`."""
    if not Path(path).is_file():
        raise MissingInputError(f"the mechanism file {path} does not exist.", stage=stage, path=str(path))
    try:
        gas = ct.Solution(path) if phase is None else ct.Solution(path, phase)
    except ct.CanteraError as exc:
        raise MechanismError(f"Cantera cannot load the mechanism {path}: {_failure_text(exc)}", stage=stage,
                             path=str(path), phase=phase) from exc
    blank = drop_blank_text(gas, stage=stage)
    if blank:
        log.warning("the mechanism %s has %d text field(s) that hold only blanks and a line break (%s%s); Cantera "
                    "cannot write them, so they are left out.", Path(path).name, len(blank),
                    ", ".join(blank[:3]), ", ..." if len(blank) > 3 else "")
    return gas


def reaction_requirements(gas: ct.Solution) -> list[tuple[frozenset, frozenset]]:
    """Per reaction: the species that must all stay, and the explicit colliders of which one must stay.

    The second set is empty for a reaction without explicit colliders. This
    is the rule of :func:`reduced_solution`; the linearised knockout uses it
    to find the reactions that the removal of a species deletes.
    """
    out = []
    for rx in gas.reactions():
        participants = frozenset(rx.reactants) | frozenset(rx.products)
        colliders: frozenset = frozenset()
        third = rx.third_body
        if third is not None and float(third.default_efficiency) == 0.0:
            colliders = frozenset(name for name, eff in third.efficiencies.items() if float(eff) > 0.0)
        out.append((participants, colliders))
    return out


def reaction_kept(requirement: tuple[frozenset, frozenset], keep_set: set) -> bool:
    """True when the reaction with this requirement stays in the mechanism of ``keep_set``."""
    participants, colliders = requirement
    if not participants <= keep_set:
        return False
    return not colliders or any(name in keep_set for name in colliders)


def reduced_solution(gas_full: ct.Solution, keep: list[str], transport_model: str | None = None) -> ct.Solution:
    """Build the reduced mechanism that keeps the species of ``keep``."""
    keep_set = set(keep)
    unknown = keep_set - set(gas_full.species_names)
    if unknown:
        raise MechanismError(f"Species not in the mechanism: {sorted(unknown)}", species=sorted(unknown)[:10])
    species = [sp for sp in gas_full.species() if sp.name in keep_set]
    reactions = [rx for rx, req in zip(gas_full.reactions(), reaction_requirements(gas_full))
                 if reaction_kept(req, keep_set)]
    kwargs = dict(thermo="ideal-gas", kinetics="gas", species=species, reactions=reactions)
    if transport_model:
        kwargs["transport_model"] = transport_model
    return ct.Solution(**kwargs)


def write_reduced_mechanism(gas_full: ct.Solution, keep: list[str], path: Path, stage: str | None = None) -> Path:
    """Write the reduced mechanism as a Cantera YAML file; with transport data when the species have it.

    Text fields that Cantera cannot write (blank notes, :mod:`gasig.mechanism`)
    are left out.
    """
    try:
        gas = reduced_solution(gas_full, keep, "mixture-averaged")
    except Exception:  # no transport data in the mechanism
        gas = reduced_solution(gas_full, keep)
    gas.name = gas_full.name or "gas"
    write_yaml(gas, path, stage=stage)
    return path


def read_rank(path: Path) -> list[str]:
    """Species names from ``ranked_species.csv``, from the least to the most important."""
    if not path.is_file():
        raise MissingInputError(f"{path} is missing; run 'gasig run' or 'gasig rank' first.", path=str(path))
    names = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "species" not in reader.fieldnames:
            raise InputError(f"{path} has no 'species' column.", path=str(path))
        for row in reader:
            names.append(row["species"])
    if not names:
        raise InputError(f"{path} holds no species.", path=str(path))
    repeated = sorted(name for name, count in collections.Counter(names).items() if count > 1)
    if repeated:
        raise InputError(f"{path} lists species more than once: {repeated[:5]}", path=str(path))
    return names


def read_importance(path: Path) -> dict[str, float]:
    """Importance per species from the rank file; empty when the file has no importance column."""
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "importance" not in reader.fieldnames or "species" not in reader.fieldnames:
            return {}
        for row in reader:
            try:
                values[row["species"]] = float(row["importance"])
            except (TypeError, ValueError):
                values[row["species"]] = 0.0
    return values


def _merge(base: dict, override: dict, keys: tuple[str, ...], where: str) -> dict:
    merged = dict(base)
    for key in keys:
        value = override.get(key)
        if value is not None:
            merged[key] = value
    missing = [k for k in keys if merged.get(k) is None]
    if missing:
        raise ConfigError(f"'{where}' needs the keys {missing} (set them there or in the solver block).")
    return merged


def evaluation_config(cfg: dict) -> tuple[dict, list[tuple[float, float, float]], dict | None, list[tuple[float, float]], bool]:
    """The evaluation conditions of the sweep: ``(idt_cfg, conditions, fs_cfg, fs_conditions, evaluate_fs)``."""
    rcfg = cfg["reduction"]
    idt_cfg = _merge(cfg["idt"]["solver"], rcfg["idt"], IDT_KEYS, "reduction.idt")
    idt_cfg["skip_unignited"] = False
    conditions = idt_conditions(idt_cfg)
    if not conditions:
        raise ConfigError("'reduction.idt' gives no conditions: T_K, P_atm and phi must be non-empty lists "
                          "(there or in idt.solver).")
    evaluate_fs = bool(rcfg["evaluate_fs"])
    fs_cfg = None
    fs_conds: list[tuple[float, float]] = []
    if evaluate_fs:
        fs_cfg = _merge(cfg["fs"]["solver"], rcfg["fs"], FS_KEYS, "reduction.fs")
        fs_cfg["refine"] = cfg["fs"]["solver"]["refine"]
        fs_cfg["zerork"] = cfg["fs"]["solver"]["zerork"]
        fs_cfg["flame_tip_T_fraction"] = cfg["fs"]["flame_tip_T_fraction"]
        fs_conds = flame_conditions(fs_cfg)
        if not fs_conds:
            raise ConfigError("'reduction.fs' gives no conditions: P_atm and phi must be non-empty lists "
                              "(there or in fs.solver).")
    return idt_cfg, conditions, fs_cfg, fs_conds, evaluate_fs


def _relative_errors(values: list, reference: list[float]) -> np.ndarray:
    """Relative errors; a value of ``None`` (condition not evaluated) gives nan."""
    v = np.array([np.nan if x is None else float(x) for x in values], dtype=float)
    r = np.asarray(reference, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(v - r) / np.abs(r)


def _json_safe(obj):
    """Replace non-finite floats by None, so that the JSON file stays standard JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _failure_text(exc: Exception) -> str:
    """The first informative line of a Cantera error message."""
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip() and not ln.strip().startswith("*")]
    return " ".join(lines[:2])[:160] if lines else exc.__class__.__name__


def _idt_values(gas: ct.Solution, conditions: list[tuple[float, float, float]], idt_cfg: dict, label: str,
                tol: float | None = None, reference: list[float] | None = None,
                order: list[int] | None = None, on_failure: str = "inf") -> tuple[list, int, bool, int]:
    """Ignition delay times at the conditions; unignited cases give inf.

    With ``tol`` and ``reference`` the conditions are evaluated in ``order``
    and the loop stops at the first condition whose error exceeds ``tol``.
    Conditions that were not evaluated hold ``None``. A failure of the
    integrator counts as a failed condition (inf); it is logged and recorded
    in the failure file. With ``on_failure="raise"`` it raises an
    :class:`IgnitionSolverError` with the condition. A reduced mechanism
    that ignites but does not form the reference species gives an
    :class:`IgnitionSolverError` of :func:`run_idt`, so the same rule
    applies. A :class:`ConfigError` of :func:`run_idt` (the reference
    species peaks at the first sample) also counts as failed with
    ``on_failure="inf"``; with ``on_failure="raise"`` it goes to the
    caller. The result is ``(taus,
    n_unignited, complete, n_failed)``.
    """
    taus: list = [None] * len(conditions)
    n_unignited = 0
    n_failed = 0
    indices = list(order) if order is not None else list(range(len(conditions)))
    evaluated = 0
    for i in indices:
        T0, P0, phi = conditions[i]
        try:
            traj = run_idt("", None, T0, P0, phi, idt_cfg, label=f"{label}-c{i}", gas=gas)
        except (ct.CanteraError, SolverError) as exc:
            context = {"label": f"{label}-c{i}", "condition": i, "T0_K": T0, "P0_atm": P0 / ct.one_atm,
                       "phi": phi, "n_species": gas.n_species}
            if on_failure == "raise":
                if isinstance(exc, SolverError):
                    raise
                raise IgnitionSolverError(f"the integrator failed at T0={T0:.0f} K, P0={P0 / ct.one_atm:.2f} atm, "
                                          f"phi={phi:.2f}: {_failure_text(exc)}", stage="reduce", **context) from exc
            log.warning("%s-c%d: T0=%.0f K, P0=%.2f atm, phi=%.2f: the integrator failed (%s); "
                        "the condition counts as failed.", label, i, T0, P0 / ct.one_atm, phi, _failure_text(exc))
            record_failure(exc if isinstance(exc, SolverError) else IgnitionSolverError(_failure_text(exc)),
                           stage="reduce", **context)
            tau = float("inf")
            n_failed += 1
        except ConfigError as exc:
            # run_idt raises this error when the reactor ignites but the reference species peaks at the
            # first sample. The detailed mechanism passed the same check at the same conditions, so for a
            # reduced mechanism the cause is the mechanism: no kept reaction forms the reference species.
            if on_failure == "raise":
                raise
            species = exc.context.get("species", idt_cfg.get("reference_species"))
            context = {"label": f"{label}-c{i}", "condition": i, "T0_K": T0, "P0_atm": P0 / ct.one_atm,
                       "phi": phi, "n_species": gas.n_species, "species": species}
            message = (f"the reactor ignited, but the reduced mechanism does not form the reference species "
                       f"'{species}'; no ignition sample is available")
            log.warning("%s-c%d: T0=%.0f K, P0=%.2f atm, phi=%.2f: %s; the condition counts as failed.",
                        label, i, T0, P0 / ct.one_atm, phi, message)
            record_failure(IgnitionSolverError(message), stage="reduce", **context)
            tau = float("inf")
            n_failed += 1
        else:
            if traj.meta["ignited"]:
                tau = float(traj.meta["tau_ign_s"])
            else:
                tau = float("inf")
                n_unignited += 1
        taus[i] = tau
        evaluated += 1
        if tol is not None and reference is not None:
            err = abs(tau - reference[i]) / abs(reference[i])
            if not err <= tol:          # also stops on nan
                break
    return taus, n_unignited, evaluated == len(conditions), n_failed


def _fs_values(gas: ct.Solution, conditions: list[tuple[float, float]], fs_cfg: dict, label: str,
               on_failure: str = "inf") -> list[float]:
    """Flame speeds at the conditions.

    A solver failure gives inf; it is logged and recorded in the failure file.
    With ``on_failure="raise"`` it raises a :class:`FlameSolverError` with the condition.
    """
    speeds = []
    for i, (P, phi) in enumerate(conditions):
        try:
            traj = run_flame("", None, P, phi, fs_cfg, label=f"{label}-f{i}", gas=gas)
        except (ct.CanteraError, SolverError) as exc:
            context = {"label": f"{label}-f{i}", "condition": i, "P_atm": P / ct.one_atm, "phi": phi,
                       "n_species": gas.n_species}
            if on_failure == "raise":
                if isinstance(exc, SolverError):
                    raise
                raise FlameSolverError(f"the flame solver failed at P={P / ct.one_atm:.2f} atm, phi={phi:.2f}: "
                                       f"{_failure_text(exc)}", stage="reduce", **context) from exc
            log.warning("%s-f%d: P=%.2f atm, phi=%.2f: the flame solver failed (%s); the condition counts as failed.",
                        label, i, P / ct.one_atm, phi, _failure_text(exc))
            record_failure(exc if isinstance(exc, SolverError) else FlameSolverError(_failure_text(exc)),
                           stage="reduce", **context)
            speeds.append(float("inf"))
            continue
        speeds.append(float(traj.meta["flame_speed_m_s"]))
    return speeds


def keep_list(names: list[str], removal_order: list[str], n: int) -> list[str]:
    """The species of the mechanism with ``n`` species along the chain of the rank."""
    removed = set(removal_order[:len(names) - n])
    return [s for s in names if s not in removed]


def protected_species(cfg: dict, names: list[str], idt_cfg: dict, fs_cfg: dict | None, warn: bool = False) -> set[str]:
    """The species that the sweep never removes, restricted to ``names``.

    These are the targets, ``reduction.keep_species``, the species of the
    mixtures of the evaluation (``fs_cfg`` only when the sweep evaluates
    flames) and the reference species of the ignition criterion.
    """
    rcfg = cfg["reduction"]
    unknown_keep = sorted(set(rcfg["keep_species"]) - set(names))
    if unknown_keep:
        raise MechanismError(f"reduction.keep_species names species that are not in the mechanism: {unknown_keep}",
                             species=unknown_keep[:10])
    if warn:
        for name in sorted(set(cfg["ranking"]["target_species"]) - set(names)):
            log.warning("Target species '%s' is not in the mechanism and is not protected.", name)
    protected = set(cfg["ranking"]["target_species"]) | set(rcfg["keep_species"])
    protected |= set(idt_cfg["fuel"]) | set(idt_cfg["oxidizer"]) | {idt_cfg["reference_species"]}
    if fs_cfg is not None:
        protected |= set(fs_cfg["fuel"]) | set(fs_cfg["oxidizer"])
    return protected & set(names)


def read_units(path: Path) -> list[list[str]]:
    """The chain units of a detection file (``units.output_file``)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        units = data["units"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InputError(f"{path} holds no readable list 'units' ({exc}).", path=str(path)) from exc
    if not isinstance(units, list) or not all(
            isinstance(u, list) and u and all(isinstance(s, str) and s for s in u) for u in units):
        raise InputError(f"{path}: 'units' must be a list of non-empty lists of species names.", path=str(path))
    members = [s for u in units for s in u]
    repeated = sorted({s for s in members if members.count(s) > 1})
    if repeated:
        raise InputError(f"{path} names species in more than one place: {repeated[:5]}.", path=str(path))
    return units


def lk_greedy_units(cfg: dict, out_dir: Path, names: list[str], removal_order: list[str]) -> tuple[list[list[str]], str]:
    """The chain units of the lk_greedy mode and their source.

    A list in ``reduction.search.units`` comes first; else the detection file
    in the output folder; else no unit. A unit with a protected member is
    dropped, because the sweep cannot remove it whole; its other members
    become single pieces.
    """
    given = cfg["reduction"]["search"]["units"]
    if given is not None:
        units, source = [list(u) for u in given], "reduction.search.units"
    else:
        path = out_dir / cfg["units"]["output_file"]
        if path.is_file():
            units, source = read_units(path), str(path)
        else:
            log.warning("no chain units: %s is missing and reduction.search.units is null; every species is its "
                        "own piece. Set units.detect to true in 'gasig run' to detect them.", path)
            return [], "none"
    unknown = sorted({s for u in units for s in u} - set(names))
    if unknown:
        raise MechanismError(f"the chain units name species that are not in the mechanism: {unknown[:5]}",
                             stage="reduce", species=unknown[:10], source=source)
    removable = set(removal_order)
    kept = [u for u in units if all(s in removable for s in u)]
    for u in units:
        if u not in kept:
            log.warning("the chain unit %s has a protected member and is not used as a unit.", " -> ".join(u))
    return kept, source


def sweep(cfg: dict, out_dir: Path) -> dict:
    """Run the reduction sweep of a configuration; writes the CSV and JSON results."""
    rcfg = cfg["reduction"]
    scfg = rcfg["search"]
    ct.suppress_thermo_warnings()
    failures_at_start = failure_counts()
    gas_full = load_mechanism(cfg["mechanism"], cfg["phase"], stage="reduce")
    names = list(gas_full.species_names)
    N = len(names)

    mode = scfg["mode"]
    # The rank file must be a permutation of the mechanism species. The lk_greedy mode reads the LK order.
    rank_path = out_dir / (cfg["lk"]["output_file"] if mode == "lk_greedy" else rcfg["ranking_file"])
    if mode == "lk_greedy" and not rank_path.is_file():
        raise MissingInputError(f"{rank_path} is missing; the lk_greedy mode reads the LK order. Run 'gasig run' "
                                "with 'lk.enabled' true first.", stage="reduce", path=str(rank_path))
    ranked = read_rank(rank_path)
    unknown = [s for s in ranked if s not in names]
    if unknown:
        raise MechanismError(f"The rank file names species that are not in the mechanism: {unknown[:5]}",
                             stage="reduce", path=str(rank_path), species=unknown[:10])
    missing = sorted(set(names) - set(ranked))
    if missing:
        raise MechanismError(f"The mechanism has species that are not in the rank file: {missing[:5]}; "
                             "run 'gasig rank' with this mechanism first.", stage="reduce", path=str(rank_path),
                             species=missing[:10])
    importance = read_importance(rank_path)

    idt_cfg, conditions, fs_cfg, fs_conds, evaluate_fs = evaluation_config(cfg)
    transport = fs_cfg["transport_model"] if evaluate_fs else None
    protected = protected_species(cfg, names, idt_cfg, fs_cfg if evaluate_fs else None, warn=True)
    removal_order = [s for s in ranked if s not in protected]

    step = int(rcfg["step"])
    min_species = int(rcfg["min_species"])
    n_min = max(min_species, N - len(removal_order))
    stop_error = rcfg["stop_error"]
    tol = None if rcfg["error_tolerance"] is None else float(rcfg["error_tolerance"])
    early_exit = scfg["early_exit"] if scfg["early_exit"] is not None else (mode in ("adaptive", "greedy", "lk_greedy"))
    if early_exit and tol is None:
        log.warning("early_exit needs reduction.error_tolerance; every condition is evaluated.")
        early_exit = False
    window = int(scfg["robust_window"])
    if not removal_order or N <= min_species:
        log.warning("nothing to remove: %d species, %d of them removable, min_species %d; "
                    "the sweep evaluates the detailed mechanism only.", N, len(removal_order), min_species)
    log.info("reduction sweep (%s): %d species, %d protected (%s), %s, down to %d species, %d IDT conditions%s%s",
             mode, N, len(protected), ", ".join(sorted(protected)),
             f"step {step}" if mode == "full" else "one species per test" if mode == "greedy"
             else "pieces of the LK order" if mode == "lk_greedy" else f"{scfg['coarse_points']} coarse sizes", n_min,
             len(conditions), f", {len(fs_conds)} flames" if evaluate_fs else "",
             f", tolerance {100.0 * tol:.3g}%{' with early exit' if early_exit else ''}" if tol is not None else "")
    t0 = time.perf_counter()
    ref_tau, n_un, _, _ = _idt_values(gas_full, conditions, idt_cfg, "full", on_failure="raise")
    if n_un:
        raise ReductionError(f"{n_un} of {len(conditions)} conditions do not ignite with the detailed mechanism; "
                             "change reduction.idt (end_time_s, max_steps or the conditions).", stage="reduce",
                             unignited=[i for i, t in enumerate(ref_tau) if not math.isfinite(t)][:20])
    if any(t <= 0.0 for t in ref_tau):
        raise ReductionError("a condition ignites at the first sample with the detailed mechanism, so the relative "
                             "error is undefined; change reduction.idt (the conditions or the ignition criterion).",
                             stage="reduce", conditions=[i for i, t in enumerate(ref_tau) if t <= 0.0][:20])
    ref_sl = _fs_values(gas_full, fs_conds, fs_cfg, "full", on_failure="raise") if evaluate_fs else []
    log.info("detailed mechanism: IDT %s s%s (%.1f s)", ["%.3e" % t for t in ref_tau],
             f", S_L {['%.4f' % s for s in ref_sl]} m/s" if evaluate_fs else "", time.perf_counter() - t0)

    # The greedy mode removes species out of rank order. The JSON carries the accepted removals,
    # in their order, as removal_order, so keep_list rebuilds every accepted size; the rank order
    # stays beside it as rank_removal_order.
    effective_order: list[str] = removal_order if mode not in ("greedy", "lk_greedy") else []
    kept_back: list[str] = []
    csv_path = out_dir / "reduction_sweep.csv"
    json_path = out_dir / "reduction_sweep.json"
    rows: list[dict] = []
    search_info: dict = {"mode": mode, "early_exit": bool(early_exit)}
    base = {
        "mechanism": cfg["mechanism"], "n_species_full": N, "n_reactions_full": gas_full.n_reactions,
        "protected_species": sorted(protected), "step": step, "min_species": min_species,
        "species_order": names, "removal_order": effective_order, "rank_removal_order": removal_order,
        "kept_back_species": kept_back,
        "idt_conditions": [{"T_K": T, "P_atm": P / ct.one_atm, "phi": phi} for T, P, phi in conditions],
        "fs_conditions": [{"P_atm": P / ct.one_atm, "phi": phi} for P, phi in fs_conds],
        "reference_idt_s": ref_tau, "reference_flame_speed_m_s": ref_sl,
        "error_tolerance": tol, "search": search_info,
        "files": {"csv": str(csv_path), "json": str(json_path)},
    }

    def write_json(status: str, **extra) -> None:
        """Write the JSON record; unignited and failed cases are inf in memory and null in the file."""
        payload = {**base, "status": status, **extra, "seconds": round(time.perf_counter() - t0, 2)}
        try:
            tmp_path = json_path.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(_json_safe(payload), fh, indent=2, allow_nan=False)
            os.replace(tmp_path, json_path)
        except OSError as exc:
            log.warning("could not write %s (%s).", json_path, exc)

    def ordered_rows() -> list[dict]:
        """Rows from the largest to the smallest size, with the species removed between consecutive rows.

        The greedy modes record one row per test, in test order. A kept-back
        test shares its size with the next test, so the rows stay as recorded.
        """
        if mode in ("greedy", "lk_greedy"):
            return list(rows)
        table = sorted(rows, key=lambda r: -r["n_species"])
        previous = N
        for r in table:
            r["removed_in_step"] = ";".join(removal_order[N - previous:N - r["n_species"]])
            previous = r["n_species"]
        return table

    def record(row: dict) -> None:
        """Keep the row and rewrite the CSV through a temporary file; a locked file is not fatal."""
        rows.append(row)
        tmp_path = csv_path.with_suffix(".csv.tmp")
        try:
            with tmp_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for r in ordered_rows():
                    writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS})
            os.replace(tmp_path, csv_path)
        except OSError as exc:
            log.warning("could not write %s (%s); the rows stay in memory.", csv_path, exc)
        search_info["evaluations"] = len(rows) - 1
        write_json("running", rows=ordered_rows())

    record({"n_species": N, "n_reactions": gas_full.n_reactions, "max_idt_error": 0.0,
            "mean_idt_error": 0.0, "n_unignited": 0, "n_solver_failures": 0,
            "max_fs_error": 0.0 if evaluate_fs else None,
            "complete": True, "seconds": round(time.perf_counter() - t0, 2),
            "idt_s": ref_tau, "flame_speed_m_s": ref_sl})

    condition_errors = np.zeros(len(conditions))

    def evaluate_keep(keep: list[str], label: str, removed_in_step: str | None = None, on_row=None,
                      tol_eval: float | None = None) -> float:
        """Evaluate the mechanism with the species ``keep``, record the row; returns the maximum error.

        ``on_row(row, worst)`` runs before the flames and the record of the row, so
        a mode can add its own fields; the greedy modes set ``kept_back``.
        ``tol_eval`` replaces the tolerance of the early exit (the second pass
        of lk_greedy). A reduced mechanism that Cantera cannot build counts as
        a failed mechanism (error inf); the failure file records it.
        """
        t_step = time.perf_counter()
        tol_exit = tol if tol_eval is None else tol_eval
        try:
            gas_red = reduced_solution(gas_full, keep, transport)
        except ct.CanteraError as exc:
            error = MechanismError(f"Cantera cannot build the reduced mechanism: {_failure_text(exc)}",
                                   stage="reduce", label=label, n_species=len(keep), removed=removed_in_step)
            log.warning("%s: %s; the mechanism counts as failed.", label, error)
            record_failure(error)
            row = {"n_species": len(keep), "n_reactions": None, "max_idt_error": float("inf"),
                   "mean_idt_error": float("inf"), "n_unignited": 0, "n_solver_failures": 0, "complete": False,
                   "idt_s": [None] * len(conditions), "build_failed": True, "max_fs_error": None}
            if removed_in_step is not None:
                row["removed_in_step"] = removed_in_step
            if on_row is not None:
                on_row(row, float("inf"))
            row["seconds"] = round(time.perf_counter() - t_step, 2)
            record(row)
            return float("inf")
        order = [int(i) for i in np.argsort(-condition_errors, kind="stable")] if early_exit else None
        taus, n_un, complete, n_failed = _idt_values(gas_red, conditions, idt_cfg, label,
                                                     tol=tol_exit if early_exit else None, reference=ref_tau,
                                                     order=order)
        err = _relative_errors(taus, ref_tau)
        seen = np.isfinite(err) | np.isinf(err)
        condition_errors[seen] = np.where(np.isinf(err[seen]), 1e300, err[seen])
        worst = float(np.nanmax(err))
        finite = err[np.isfinite(err)]
        row = {
            "n_species": gas_red.n_species, "n_reactions": gas_red.n_reactions,
            "max_idt_error": worst,
            "mean_idt_error": float(np.mean(finite)) if finite.size else float("inf"),
            "n_unignited": n_un, "n_solver_failures": n_failed, "complete": bool(complete), "idt_s": taus,
        }
        if removed_in_step is not None:
            row["removed_in_step"] = removed_in_step
        if on_row is not None:
            on_row(row, worst)
        if evaluate_fs and complete and not row.get("kept_back"):
            speeds = _fs_values(gas_red, fs_conds, fs_cfg, label)
            row["max_fs_error"] = float(np.max(_relative_errors(speeds, ref_sl)))
            row["flame_speed_m_s"] = speeds
        else:
            row["max_fs_error"] = None
        row["seconds"] = round(time.perf_counter() - t_step, 2)
        record(row)
        verdict = ""
        if removed_in_step is not None:
            verdict = f"  {removed_in_step} {'kept back' if row.get('kept_back') else 'removed'}"
        log.info("%4d species %5d reactions  max IDT error %8.3f%%%s  unignited %d%s%s  (%.1f s)",
                 row["n_species"], row["n_reactions"], 100.0 * worst,
                 "" if complete else " (lower bound, early exit)", n_un,
                 f"  solver failures {n_failed}" if n_failed else "", verdict, row["seconds"])
        return worst

    def evaluate(n: int) -> float:
        """Build the mechanism with ``n`` species along the rank order and evaluate it."""
        if n == N:
            return 0.0
        return evaluate_keep(keep_list(names, removal_order, n), f"n{n}")

    result = None
    if mode == "full":
        if stop_error is not None:
            stop_error = float(stop_error)
        removed_count = 0
        while True:
            room = N - removed_count - min_species
            take = min(step, max(room, 0), len(removal_order) - removed_count)
            if take <= 0:
                log.info("the sweep ends at %d species: %s.", N - removed_count,
                         "min_species reached" if room <= 0 else "the rank holds no more removable species")
                break
            removed_count += take
            worst = evaluate(N - removed_count)
            if stop_error is not None and (not math.isfinite(worst) or worst > stop_error):
                log.info("maximum IDT error %s above stop_error=%.3g; the sweep stops.",
                         f"{100.0 * worst:.3f}%" if math.isfinite(worst) else "undefined", stop_error)
                break
    elif mode == "greedy":
        if stop_error is not None:
            log.info("stop_error is not used by the greedy search.")
        params = SearchParams(robust_window=window, noise=float(scfg["noise"]),
                              max_evaluations=scfg["max_evaluations"])
        search_info["params"] = {"max_evaluations": params.max_evaluations}
        search_info["robust_window_applies"] = False

        def flag(row: dict, worst: float) -> None:
            ok = math.isfinite(worst) and worst <= tol
            row["kept_back"] = not ok
            (effective_order if ok else kept_back).append(row["removed_in_step"])

        def evaluate_removed(removed: list[str]) -> float:
            gone = set(removed)
            return evaluate_keep([s for s in names if s not in gone], f"g{N - len(removed)}-{removed[-1]}",
                                 removed_in_step=removed[-1], on_row=flag)

        if n_min < N:
            greedy = greedy_search(evaluate_removed, removal_order, N, n_min, tol, params)
        else:
            greedy = GreedyResult(accepted=[], kept_back=[])
        if greedy.accepted != effective_order or greedy.kept_back != kept_back:
            log.warning("the greedy bookkeeping differs from the search result; the JSON follows the search.")
            effective_order[:] = greedy.accepted
            kept_back[:] = greedy.kept_back
        search_info["accepted"] = len(greedy.accepted)
        search_info["kept_back"] = len(greedy.kept_back)
        search_info["unresolved"] = []
        search_info["budget_exhausted"] = greedy.budget_exhausted
        if greedy.budget_exhausted:
            log.warning("the evaluation budget of %d was reached; %d candidates were not tested.",
                        params.max_evaluations, len(removal_order) - greedy.evaluations)
        log.info("greedy pass: %d tests, %d species removed, %d kept back.",
                 greedy.evaluations, len(greedy.accepted), len(greedy.kept_back))
    elif mode == "lk_greedy":
        if stop_error is not None:
            log.info("stop_error is not used by the lk_greedy search.")
        if not importance:
            raise InputError(f"{rank_path} has no 'importance' column with the LK scores.", stage="reduce",
                             path=str(rank_path))
        params = SearchParams(robust_window=window, noise=float(scfg["noise"]),
                              max_evaluations=scfg["max_evaluations"])
        tol2 = None if scfg["pass2_tolerance"] is None else float(scfg["pass2_tolerance"])
        tolerances = (tol,) if tol2 is None else (tol, tol2)
        block = int(scfg["block_size"])
        tier0_score = float(scfg["tier0_score"])
        tier0 = {s for s in removal_order if importance.get(s, math.inf) <= tier0_score}
        units, units_source = lk_greedy_units(cfg, out_dir, names, removal_order)
        search_info["params"] = {"max_evaluations": params.max_evaluations, "block_size": block,
                                 "tier0_score": tier0_score, "tolerances": list(tolerances)}
        search_info["robust_window_applies"] = False
        search_info["tier0_size"] = len(tier0)
        search_info["units"] = units
        search_info["units_source"] = units_source
        log.info("lk_greedy: %d removable species, %d of them tier 0 (LK score <= %.3g), %d chain units (%s), "
                 "blocks of up to %d species, tolerances %s.", len(removal_order), len(tier0), tier0_score,
                 len(units), units_source, block, " / ".join(f"{100.0 * t:.3g}%" for t in tolerances))

        def evaluate_pieces(removed: list[str], tol_test: float, test: dict) -> float:
            def flag(row: dict, worst: float) -> None:
                ok = math.isfinite(worst) and worst <= tol_test
                row["kept_back"] = not ok
                row.update({"pass": test["pass"], "test": test["kind"], "tolerance": tol_test,
                            "n_removed_in_step": len(test["members"])})
                if ok:
                    effective_order.extend(test["members"])

            gone = set(removed)
            members = test["members"]
            return evaluate_keep([s for s in names if s not in gone],
                                 f"p{test['pass']}-{N - len(removed)}-{members[0]}",
                                 removed_in_step=";".join(members), on_row=flag, tol_eval=tol_test)

        if n_min < N:
            lkg = lk_greedy_search(evaluate_pieces, removal_order, units, tier0, block, tolerances, N, n_min, params)
        else:
            lkg = None
        passes = lkg.passes if lkg is not None else []
        if lkg is not None and lkg.accepted != effective_order:
            log.warning("the lk_greedy bookkeeping differs from the search result; the JSON follows the search.")
            effective_order[:] = lkg.accepted
        gone = set(effective_order)                     # pass-1 failures that a later pass did not remove
        kept_back[:] = [s for p in (passes[0].kept if passes else []) for s in p if s not in gone]
        search_info["passes"] = []
        removed_so_far = 0
        for p in passes:
            removed_so_far += len(p.accepted)
            search_info["passes"].append({
                "pass": len(search_info["passes"]) + 1, "tolerance": p.tolerance, "n_species": N - removed_so_far,
                "accepted": len(p.accepted), "kept_pieces": [list(x) for x in p.kept],
                "untested_pieces": [list(x) for x in p.untested], "tests": len(p.tests),
                "tests_cached": sum(1 for t in p.tests if t["cached"]), "evaluations": p.evaluations})
            log.info("lk_greedy pass %d at %.3g%%: %d tests (%d cached), %d species removed, %d pieces kept, "
                     "%d untested; %d species.", len(search_info["passes"]), 100.0 * p.tolerance, len(p.tests),
                     sum(1 for t in p.tests if t["cached"]), len(p.accepted), len(p.kept), len(p.untested),
                     N - removed_so_far)
        search_info["accepted"] = len(effective_order)
        search_info["kept_back"] = len(kept_back)
        search_info["unresolved"] = []
        search_info["budget_exhausted"] = bool(lkg is not None and lkg.budget_exhausted)
        if search_info["budget_exhausted"]:
            log.warning("the evaluation budget of %d was reached; the remaining pieces were not tested and no "
                        "second pass ran.", params.max_evaluations)
    else:
        if stop_error is not None:
            log.info("stop_error is not used by the adaptive search.")
        params = SearchParams(coarse_points=int(scfg["coarse_points"]),
                              importance_gap_probes=int(scfg["importance_gap_probes"]),
                              near_miss_factor=float(scfg["near_miss_factor"]), robust_window=window,
                              noise=float(scfg["noise"]), max_evaluations=scfg["max_evaluations"])
        probes = gap_probes([importance.get(s, 0.0) for s in removal_order], N, n_min,
                            params.importance_gap_probes) if importance else []
        search_info["params"] = params.__dict__
        search_info["probes"] = probes
        if n_min < N:
            result = adaptive_search(evaluate, N, n_min, tol, probes, params)
            search_info["unresolved"] = result.unresolved
            search_info["budget_exhausted"] = result.budget_exhausted
            if result.budget_exhausted:
                log.warning("the evaluation budget of %d was reached; %d intervals are unresolved.",
                            params.max_evaluations, len(result.unresolved))
            elif result.unresolved:
                log.info("%d intervals were not refined (errors above %.3g times the tolerance).",
                         len(result.unresolved), params.near_miss_factor)
        else:
            search_info["unresolved"] = []
            search_info["budget_exhausted"] = False
    search_info["evaluations"] = len(rows) - 1

    table = ordered_rows()
    errors = {r["n_species"]: r["max_idt_error"] for r in table if not r.get("kept_back")}
    strict = robust = selected = None
    mech_file = None
    if tol is not None:
        if result is not None:
            strict, robust = result.strict_optimum, result.robust_optimum
        elif mode == "greedy":
            # Every accepted size passes, so the smallest accepted size is both optima.
            strict, _ = select_optima(errors, tol, 0)
            robust = strict
        elif mode == "lk_greedy":
            # The end of pass 1 is the mechanism at error_tolerance; pass 2 runs at its own tolerance.
            first = search_info["passes"][0] if search_info.get("passes") else None
            strict = first["n_species"] if first and first["accepted"] else None
            robust = strict
        else:
            # The window counts evaluated sizes: with a step above 1 it covers window x step species.
            strict, robust = select_optima(errors, tol, window)
            if step > 1 and window > 0:
                log.info("the robust window holds the next %d evaluated sizes, %d species with step %d.",
                         window, window * step, step)
        selected = robust if robust is not None else strict
        if selected is None:
            log.warning("no evaluated mechanism is within the tolerance of %.3g%%.", 100.0 * tol)
        else:
            log.info("selection within %.3g%%: strict optimum %s species, robust optimum %s species (window %d).",
                     100.0 * tol, strict, robust, window)
            if rcfg["write_mechanism"]:
                mech_file = write_reduced_mechanism(gas_full, keep_list(names, effective_order, selected),
                                                    out_dir / f"reduced_{selected}sp.yaml", stage="reduce")
                log.info("reduced mechanism written to %s", mech_file)
        if mode == "lk_greedy" and len(search_info.get("passes", [])) == 2:
            second = search_info["passes"][1]
            second["mechanism_file"] = None
            if second["accepted"] and rcfg["write_mechanism"]:
                path = write_reduced_mechanism(gas_full, keep_list(names, effective_order, second["n_species"]),
                                               out_dir / f"reduced_{second['n_species']}sp.yaml", stage="reduce")
                second["mechanism_file"] = str(path)
                log.info("pass 2 within %.3g%%: %d species; mechanism written to %s",
                         100.0 * second["tolerance"], second["n_species"], path)
    failures = {k: v - failures_at_start.get(k, 0) for k, v in failure_counts().items()
                               if v - failures_at_start.get(k, 0) > 0}
    thresholds = {}
    for limit in (0.05, 0.10, 0.20):
        ok = [r["n_species"] for r in table if r["complete"] and not r.get("kept_back") and r["max_idt_error"] <= limit]
        thresholds[f"{int(limit * 100)}%"] = min(ok) if ok else None
    selection = {"strict_optimum": strict, "robust_optimum": robust, "robust_window": window,
                 "selected": selected, "mechanism_file": None if mech_file is None else str(mech_file)}
    write_json("complete", selection=selection, smallest_mechanism_within=thresholds, failures=failures, rows=table)
    summary = {**base, "status": "complete", "selection": selection, "smallest_mechanism_within": thresholds,
               "failures": failures, "rows": table, "seconds": round(time.perf_counter() - t0, 2)}
    log.info("smallest complete mechanism within 5%% / 10%% / 20%% IDT error: %s / %s / %s species; "
             "%d evaluations; results in %s (%.1f s)", thresholds["5%"], thresholds["10%"], thresholds["20%"],
             search_info["evaluations"], csv_path, summary["seconds"])
    return summary


def run_reduce(config_path: str | Path) -> dict:
    """Load the configuration and run the sweep in its output folder."""
    config_path = Path(config_path)
    cfg = load_config(config_path)
    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_dir():
        raise MissingInputError(f"{out_dir} does not exist; run 'gasig run' first.", stage="reduce", path=str(out_dir))
    # One sweep, one log: reduce.log is rewritten, like reduction_sweep.csv and .json.
    setup_logging(out_dir, cfg["log_level"], file_name="reduce.log", mode="w")
    try:
        log.info("GA-SiG reduction sweep | config %s", config_path)
        return sweep(cfg, out_dir)
    except Exception:
        log_failure(log, "the reduction sweep failed")
        raise
    finally:
        teardown_logging()
