"""Reduction sweep: remove species in rank order and measure the error.

The sweep reads the rank of an earlier run (``ranked_species.csv`` in the
output folder), removes species from the least important upward, builds
each reduced mechanism in memory, and evaluates the ignition delay time
(and optionally the flame speed) against the detailed mechanism. The
result is the curve of the maximum relative error against the number of
species, the quantity the paper uses to select a reduced mechanism. With
``reduction.error_tolerance`` the sweep also selects the smallest mechanism
within the tolerance and writes it as a Cantera YAML file.

Two search modes exist. ``full`` evaluates every size in steps of
``reduction.step``. ``adaptive`` evaluates a coarse set of sizes, probes the
largest importance gaps of the rank, refines every crossing of the
tolerance and reports the intervals it left unresolved
(:mod:`gasig.search`). With the early exit, a size stops at the first
condition above the tolerance; its error is then a lower bound and the row
is marked ``complete: false``.

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

from .config import ConfigError, load_config
from .logs import log_failure, setup_logging, teardown_logging
from .search import SearchParams, adaptive_search, gap_probes, select_optima
from .solvers.flame import flame_conditions, run_flame
from .solvers.idt import idt_conditions, run_idt

log = logging.getLogger("gasig.reduce")

IDT_KEYS = ("fuel", "oxidizer", "T_K", "P_atm", "phi", "reactor", "end_time_s", "reference_species",
            "ignition_criterion", "rtol", "atol", "max_steps")
FS_KEYS = ("fuel", "oxidizer", "T_unburnt_K", "P_atm", "phi", "domain_width_m", "transport_model", "loglevel")
CSV_FIELDS = ["n_species", "n_reactions", "max_idt_error", "mean_idt_error", "n_unignited", "n_solver_failures",
              "max_fs_error", "complete", "removed_in_step", "seconds"]


def reduced_solution(gas_full: ct.Solution, keep: list[str], transport_model: str | None = None) -> ct.Solution:
    """Build the reduced mechanism that keeps the species of ``keep``."""
    keep_set = set(keep)
    unknown = keep_set - set(gas_full.species_names)
    if unknown:
        raise ValueError(f"Species not in the mechanism: {sorted(unknown)}")
    species = [sp for sp in gas_full.species() if sp.name in keep_set]
    reactions = []
    for rx in gas_full.reactions():
        if not (set(rx.reactants) | set(rx.products)) <= keep_set:
            continue
        third = rx.third_body
        if third is not None and float(third.default_efficiency) == 0.0:
            colliders = [name for name, eff in third.efficiencies.items() if float(eff) > 0.0]
            if colliders and not any(name in keep_set for name in colliders):
                continue
        reactions.append(rx)
    kwargs = dict(thermo="ideal-gas", kinetics="gas", species=species, reactions=reactions)
    if transport_model:
        kwargs["transport_model"] = transport_model
    return ct.Solution(**kwargs)


def write_reduced_mechanism(gas_full: ct.Solution, keep: list[str], path: Path) -> Path:
    """Write the reduced mechanism as a Cantera YAML file; with transport data when the species have it."""
    try:
        gas = reduced_solution(gas_full, keep, "mixture-averaged")
    except Exception:  # no transport data in the mechanism
        gas = reduced_solution(gas_full, keep)
    gas.name = gas_full.name or "gas"
    gas.write_yaml(str(path))
    return path


def read_rank(path: Path) -> list[str]:
    """Species names from ``ranked_species.csv``, from the least to the most important."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run 'gasig run' or 'gasig rank' first.")
    names = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "species" not in reader.fieldnames:
            raise ValueError(f"{path} has no 'species' column.")
        for row in reader:
            names.append(row["species"])
    if not names:
        raise ValueError(f"{path} holds no species.")
    repeated = sorted(name for name, count in collections.Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(f"{path} lists species more than once: {repeated[:5]}")
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
    Conditions that were not evaluated hold ``None``. A Cantera failure of
    the integrator counts as a failed condition (inf) and is logged, unless
    ``on_failure`` is ``"raise"``. The result is ``(taus, n_unignited,
    complete, n_failed)``.
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
        except ct.CanteraError as exc:
            if on_failure == "raise":
                raise
            log.warning("%s-c%d: T0=%.0f K, P0=%.2f atm, phi=%.2f: the integrator failed (%s); "
                        "the condition counts as failed.", label, i, T0, P0 / ct.one_atm, phi, _failure_text(exc))
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
    """Flame speeds at the conditions; a Cantera failure gives inf unless ``on_failure`` is ``"raise"``."""
    speeds = []
    for i, (P, phi) in enumerate(conditions):
        try:
            traj = run_flame("", None, P, phi, fs_cfg, label=f"{label}-f{i}", gas=gas)
        except ct.CanteraError as exc:
            if on_failure == "raise":
                raise
            log.warning("%s-f%d: P=%.2f atm, phi=%.2f: the flame solver failed (%s); the condition counts as failed.",
                        label, i, P / ct.one_atm, phi, _failure_text(exc))
            speeds.append(float("inf"))
            continue
        speeds.append(float(traj.meta["flame_speed_m_s"]))
    return speeds


def keep_list(names: list[str], removal_order: list[str], n: int) -> list[str]:
    """The species of the mechanism with ``n`` species along the chain of the rank."""
    removed = set(removal_order[:len(names) - n])
    return [s for s in names if s not in removed]


def sweep(cfg: dict, out_dir: Path) -> dict:
    """Run the reduction sweep of a configuration; writes the CSV and JSON results."""
    rcfg = cfg["reduction"]
    scfg = rcfg["search"]
    ct.suppress_thermo_warnings()
    gas_full = ct.Solution(cfg["mechanism"]) if cfg["phase"] is None else ct.Solution(cfg["mechanism"], cfg["phase"])
    names = list(gas_full.species_names)
    N = len(names)

    # The rank file must be a permutation of the mechanism species.
    rank_path = out_dir / rcfg["ranking_file"]
    ranked = read_rank(rank_path)
    unknown = [s for s in ranked if s not in names]
    if unknown:
        raise ValueError(f"The rank file names species that are not in the mechanism: {unknown[:5]}")
    missing = sorted(set(names) - set(ranked))
    if missing:
        raise ValueError(f"The mechanism has species that are not in the rank file: {missing[:5]}; "
                         "run 'gasig rank' with this mechanism first.")
    importance = read_importance(rank_path)

    idt_cfg, conditions, fs_cfg, fs_conds, evaluate_fs = evaluation_config(cfg)
    transport = fs_cfg["transport_model"] if evaluate_fs else None

    # Never remove the targets, the user's keep list, the species that define the mixtures,
    # and the reference species of the ignition criterion.
    unknown_keep = sorted(set(rcfg["keep_species"]) - set(names))
    if unknown_keep:
        raise ValueError(f"reduction.keep_species names species that are not in the mechanism: {unknown_keep}")
    for name in sorted(set(cfg["ranking"]["target_species"]) - set(names)):
        log.warning("Target species '%s' is not in the mechanism and is not protected.", name)
    protected = set(cfg["ranking"]["target_species"]) | set(rcfg["keep_species"])
    protected |= set(idt_cfg["fuel"]) | set(idt_cfg["oxidizer"]) | {idt_cfg["reference_species"]}
    if evaluate_fs:
        protected |= set(fs_cfg["fuel"]) | set(fs_cfg["oxidizer"])
    protected &= set(names)
    removal_order = [s for s in ranked if s not in protected]

    mode = scfg["mode"]
    step = int(rcfg["step"])
    min_species = int(rcfg["min_species"])
    n_min = max(min_species, N - len(removal_order))
    stop_error = rcfg["stop_error"]
    tol = None if rcfg["error_tolerance"] is None else float(rcfg["error_tolerance"])
    early_exit = scfg["early_exit"] if scfg["early_exit"] is not None else (mode == "adaptive")
    if early_exit and tol is None:
        log.warning("early_exit needs reduction.error_tolerance; every condition is evaluated.")
        early_exit = False
    window = int(scfg["robust_window"])
    if not removal_order or N <= min_species:
        log.warning("nothing to remove: %d species, %d of them removable, min_species %d; "
                    "the sweep evaluates the detailed mechanism only.", N, len(removal_order), min_species)
    log.info("reduction sweep (%s): %d species, %d protected (%s), %s, down to %d species, %d IDT conditions%s%s",
             mode, N, len(protected), ", ".join(sorted(protected)),
             f"step {step}" if mode == "full" else f"{scfg['coarse_points']} coarse sizes", n_min,
             len(conditions), f", {len(fs_conds)} flames" if evaluate_fs else "",
             f", tolerance {100.0 * tol:.3g}%{' with early exit' if early_exit else ''}" if tol is not None else "")
    t0 = time.perf_counter()
    ref_tau, n_un, _, _ = _idt_values(gas_full, conditions, idt_cfg, "full", on_failure="raise")
    if n_un:
        raise RuntimeError(f"{n_un} of {len(conditions)} conditions do not ignite with the detailed mechanism; "
                           "change reduction.idt (end_time_s, max_steps or the conditions).")
    if any(t <= 0.0 for t in ref_tau):
        raise RuntimeError("a condition ignites at the first sample with the detailed mechanism, so the relative "
                           "error is undefined; change reduction.idt (the conditions or the ignition criterion).")
    ref_sl = _fs_values(gas_full, fs_conds, fs_cfg, "full", on_failure="raise") if evaluate_fs else []
    log.info("detailed mechanism: IDT %s s%s (%.1f s)", ["%.3e" % t for t in ref_tau],
             f", S_L {['%.4f' % s for s in ref_sl]} m/s" if evaluate_fs else "", time.perf_counter() - t0)

    csv_path = out_dir / "reduction_sweep.csv"
    json_path = out_dir / "reduction_sweep.json"
    rows: list[dict] = []
    search_info: dict = {"mode": mode, "early_exit": bool(early_exit)}
    base = {
        "mechanism": cfg["mechanism"], "n_species_full": N, "n_reactions_full": gas_full.n_reactions,
        "protected_species": sorted(protected), "step": step, "min_species": min_species,
        "species_order": names, "removal_order": removal_order,
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
        """Rows from the largest to the smallest size, with the species removed between consecutive rows."""
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

    def evaluate(n: int) -> float:
        """Build the mechanism with ``n`` species, evaluate it, record the row; returns the maximum error."""
        if n == N:
            return 0.0
        keep = keep_list(names, removal_order, n)
        t_step = time.perf_counter()
        gas_red = reduced_solution(gas_full, keep, transport)
        order = [int(i) for i in np.argsort(-condition_errors, kind="stable")] if early_exit else None
        taus, n_un, complete, n_failed = _idt_values(gas_red, conditions, idt_cfg, f"n{n}",
                                                     tol=tol if early_exit else None, reference=ref_tau, order=order)
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
        if evaluate_fs and complete:
            speeds = _fs_values(gas_red, fs_conds, fs_cfg, f"n{n}")
            row["max_fs_error"] = float(np.max(_relative_errors(speeds, ref_sl)))
            row["flame_speed_m_s"] = speeds
        else:
            row["max_fs_error"] = None
        row["seconds"] = round(time.perf_counter() - t_step, 2)
        record(row)
        log.info("%4d species %5d reactions  max IDT error %8.3f%%%s  unignited %d%s  (%.1f s)",
                 row["n_species"], row["n_reactions"], 100.0 * worst,
                 "" if complete else " (lower bound, early exit)", n_un,
                 f"  solver failures {n_failed}" if n_failed else "", row["seconds"])
        return worst

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
    errors = {r["n_species"]: r["max_idt_error"] for r in table}
    strict = robust = selected = None
    mech_file = None
    if tol is not None:
        if result is not None:
            strict, robust = result.strict_optimum, result.robust_optimum
        else:
            strict, robust = select_optima(errors, tol, window)
        selected = robust if robust is not None else strict
        if selected is None:
            log.warning("no evaluated mechanism is within the tolerance of %.3g%%.", 100.0 * tol)
        else:
            log.info("selection within %.3g%%: strict optimum %s species, robust optimum %s species (window %d).",
                     100.0 * tol, strict, robust, window)
            if rcfg["write_mechanism"]:
                mech_file = write_reduced_mechanism(gas_full, keep_list(names, removal_order, selected),
                                                    out_dir / f"reduced_{selected}sp.yaml")
                log.info("reduced mechanism written to %s", mech_file)
    thresholds = {}
    for limit in (0.05, 0.10, 0.20):
        ok = [r["n_species"] for r in table if r["complete"] and r["max_idt_error"] <= limit]
        thresholds[f"{int(limit * 100)}%"] = min(ok) if ok else None
    selection = {"strict_optimum": strict, "robust_optimum": robust, "robust_window": window,
                 "selected": selected, "mechanism_file": None if mech_file is None else str(mech_file)}
    write_json("complete", selection=selection, smallest_mechanism_within=thresholds, rows=table)
    summary = {**base, "status": "complete", "selection": selection, "smallest_mechanism_within": thresholds,
               "rows": table, "seconds": round(time.perf_counter() - t0, 2)}
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
        raise FileNotFoundError(f"{out_dir} does not exist; run 'gasig run' first.")
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
