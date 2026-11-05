"""Report of a reduction: the selected mechanism against the detailed one.

``gasig report`` reads ``reduction_sweep.json`` of a run, selects the
mechanism within ``reduction.error_tolerance`` (the robust optimum, or the
strict optimum when no robust one exists), and writes the folder ``report/``
in the output folder:

* ``error_vs_species.png``: the maximum error against the number of species,
  with the tolerance and the two optima
* ``idt_comparison.png``: the ignition delay time against 1000/T for the
  detailed mechanism (lines) and the selected mechanism (markers), one panel
  per pressure and equivalence ratio, on ``report.T_points`` temperatures
* ``flame_speed_comparison.png``: the flame speed against the equivalence
  ratio, when the sweep evaluated flames
* ``report.md`` in ASD-STE100 and ``report.json`` with every number
* the selected mechanism as a Cantera YAML file, when the sweep did not
  write it

A sweep that did not complete (an interrupted or failed ``gasig reduce``)
has a status other than ``complete``. The report then selects only from
the evaluated sizes. It logs a warning, puts a warning in ``report.md`` and
sets ``sweep_complete`` to false in ``report.json``.

The module needs Cantera and NumPy; the figures need Matplotlib.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import cantera as ct
import numpy as np

from . import __version__
from .config import load_config
from .errors import ConfigError, InputError, MechanismError, MissingInputError, SolverError, record_failure
from .logs import log_failure, setup_logging, teardown_logging
from .reduce import (_failure_text, evaluation_config, keep_list, load_mechanism, reduced_solution,
                     write_reduced_mechanism)
from .search import select_optima
from .solvers.flame import FlameSolverError, run_flame
from .solvers.idt import run_idt

log = logging.getLogger("gasig.report")


def _idt(gas: ct.Solution, T: float, P: float, phi: float, idt_cfg: dict, label: str) -> float:
    """Ignition delay time at one grid point; no ignition or a Cantera failure gives inf (logged)."""
    try:
        traj = run_idt("", None, T, P, phi, idt_cfg, label=label, gas=gas)
    except (ct.CanteraError, SolverError) as exc:
        log.warning("%s: T=%.0f K, P=%.2f atm, phi=%.2f: the integrator failed (%s); the point has no value.",
                    label, T, P / ct.one_atm, phi, _failure_text(exc))
        record_failure(exc, stage="report", label=label, T0_K=float(T), P0_atm=float(P) / ct.one_atm, phi=float(phi),
                       mechanism=gas.name or None, n_species=int(gas.n_species))
        return float("inf")
    return float(traj.meta["tau_ign_s"]) if traj.meta["ignited"] else float("inf")


def _flame_speed(gas: ct.Solution, P: float, phi: float, fs_cfg: dict, label: str) -> float:
    """Flame speed at one condition; a solver failure gives inf (logged)."""
    try:
        traj = run_flame("", None, P, phi, fs_cfg, label=label, gas=gas)
    except (ct.CanteraError, SolverError) as exc:
        log.warning("%s: P=%.2f atm, phi=%.2f: the flame solver failed (%s); the point has no value.",
                    label, P / ct.one_atm, phi, _failure_text(exc))
        record_failure(exc, stage="report", label=label, P0_atm=float(P) / ct.one_atm, phi=float(phi),
                       T_unburnt_K=float(fs_cfg["T_unburnt_K"]), mechanism=gas.name or None,
                       n_species=int(gas.n_species))
        return float("inf")
    return float(traj.meta["flame_speed_m_s"])


def _error(value: float, reference: float) -> float:
    if not (math.isfinite(value) and math.isfinite(reference)) or reference == 0.0:
        return float("inf")
    return abs(value - reference) / abs(reference)


def _temperature_grid(conditions: list[tuple[float, float, float]], points: int) -> list[float]:
    temperatures = sorted({T for T, _, _ in conditions})
    low, high = temperatures[0], temperatures[-1]
    if low == high:
        low, high = 0.9 * low, 1.1 * high
    return [float(x) for x in np.linspace(low, high, points)]


def _finite_or_none(value: float):
    return value if math.isfinite(value) else None


def _fmt_ms(value) -> str:
    return "no ignition" if value is None or not math.isfinite(value) else f"{1e3 * value:.4g}"


def _fmt_pct(value) -> str:
    return "undefined" if value is None or not math.isfinite(value) else f"{100.0 * value:.2f}"


def _fmt_cm(value) -> str:
    return "no flame" if value is None or not math.isfinite(value) else f"{100.0 * value:.2f}"


def _check_flame_backend(fs_cfg: dict | None, evaluate_fs: bool) -> None:
    """Fail before the ignition grid when the flame backend is missing, instead of one `inf` per flame."""
    if evaluate_fs and fs_cfg is not None and fs_cfg.get("backend", "cantera") == "zerork":
        from .solvers.zerork import zerork_available, zerork_home
        if not zerork_available(fs_cfg["zerork"]):
            raise FlameSolverError(f"Zero-RK is not available: {zerork_home(fs_cfg['zerork']) / 'bin'} lacks the "
                                   "flame solver. Build the image from Dockerfile.cpu, or point "
                                   "fs.solver.zerork.home or ZERORK_HOME at a Zero-RK install.", stage="report",
                                   path=str(zerork_home(fs_cfg["zerork"]) / "bin"))


def _flame_solver_text(fs_cfg: dict) -> str:
    if fs_cfg.get("backend", "cantera") == "zerork":
        return "Zero-RK premixed steady flame solver, constant Lewis numbers"
    return f"Cantera FreeFlame, {fs_cfg['transport_model']} transport"


def _idt_rows(gas_full: ct.Solution, gas_red: ct.Solution, conditions: list, idt_cfg: dict, points: int,
              label_red: str, t0: float) -> list[dict]:
    """Ignition delay times of both mechanisms on ``points`` temperatures per pressure and equivalence ratio."""
    grid = _temperature_grid(conditions, points)
    pairs = list(dict.fromkeys((P, phi) for _, P, phi in conditions))
    rows = []
    for P, phi in pairs:
        for T in grid:
            tau_full = _idt(gas_full, T, P, phi, idt_cfg, f"full-T{T:.0f}")
            tau_red = _idt(gas_red, T, P, phi, idt_cfg, f"{label_red}-T{T:.0f}")
            rows.append({"T_K": T, "P_atm": P / ct.one_atm, "phi": phi,
                         "idt_detailed_s": _finite_or_none(tau_full), "idt_reduced_s": _finite_or_none(tau_red),
                         "error": _finite_or_none(_error(tau_red, tau_full))})
    log.info("ignition delay times computed at %d temperatures x %d (P, phi) pairs (%.1f s)",
             len(grid), len(pairs), time.perf_counter() - t0)
    return rows


def _fs_rows(gas_full: ct.Solution, gas_red: ct.Solution, fs_conds: list, fs_cfg: dict, label_red: str,
             t0: float) -> list[dict]:
    """Flame speeds of both mechanisms at every (P, phi) of the flame conditions.

    The reduced mechanism goes first at each condition: with the Zero-RK
    continuation its flame is the initial guess of the detailed flame, so the
    expensive cold start runs on the small mechanism only.
    """
    rows = []
    for i, (P, phi) in enumerate(fs_conds):
        sl_red = _flame_speed(gas_red, P, phi, fs_cfg, f"{label_red}-f{i}")
        sl_full = _flame_speed(gas_full, P, phi, fs_cfg, f"full-f{i}")
        rows.append({"P_atm": P / ct.one_atm, "phi": phi, "detailed_m_s": _finite_or_none(sl_full),
                     "reduced_m_s": _finite_or_none(sl_red), "error": _finite_or_none(_error(sl_red, sl_full))})
        log.info("flame %d of %d: P=%.3g atm, phi=%.3g, detailed %s cm/s, reduced %s cm/s, error %s %% (%.1f s)",
                 i + 1, len(fs_conds), P / ct.one_atm, phi, _fmt_cm(sl_full), _fmt_cm(sl_red),
                 _fmt_pct(rows[-1]["error"]), time.perf_counter() - t0)
    return rows


def build_report(cfg: dict, out_dir: Path, data: dict) -> dict:
    """Build the report of a sweep result ``data`` (the content of ``reduction_sweep.json``)."""
    t0 = time.perf_counter()
    ct.suppress_thermo_warnings()
    rcfg = cfg["reduction"]
    tol = rcfg["error_tolerance"] if rcfg["error_tolerance"] is not None else data.get("error_tolerance")
    if tol is None:
        raise ConfigError("set 'reduction.error_tolerance' to select a mechanism for the report.", stage="report")
    tol = float(tol)
    if "species_order" not in data or "removal_order" not in data:
        raise InputError("reduction_sweep.json comes from an older version; run 'gasig reduce' again.", stage="report")
    window = int(rcfg["search"]["robust_window"])
    rows = sorted(data["rows"], key=lambda r: -r["n_species"])
    for r in rows:                                   # null in the JSON file means inf
        if r["max_idt_error"] is None:
            r["max_idt_error"] = float("inf")
    errors = {r["n_species"]: r["max_idt_error"] for r in rows if not r.get("kept_back")}
    mode = data.get("search", {}).get("mode", "full")
    # An interrupted or failed sweep leaves a status other than 'complete' and no selection.
    status = str(data.get("status") or "unknown")
    complete = status == "complete"
    if not complete:
        log.warning("reduction_sweep.json has the status '%s': the sweep did not complete. The report selects "
                    "only from the %d evaluated sizes. A smaller mechanism within the tolerance can exist. "
                    "Run 'gasig reduce' again to complete the sweep.", status, len(rows))
    stored = data.get("selection") or {}
    if data.get("error_tolerance") == tol and stored.get("robust_window") == window:
        strict, robust = stored.get("strict_optimum"), stored.get("robust_optimum")
    else:
        if complete:
            log.info("the tolerance or the window differs from the sweep; the optima are selected again "
                     "from the evaluated sizes.")
        strict, robust = select_optima(errors, tol, window, require_full_window=(mode == "adaptive"))
    selected = robust if robust is not None else strict
    by_size = {r["n_species"]: r for r in rows}
    names = data["species_order"]
    removal_order = data["removal_order"]
    N = len(names)

    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "mechanism": cfg["mechanism"], "n_species_full": N, "n_reactions_full": data["n_reactions_full"],
        "error_tolerance": tol, "robust_window": window, "search_mode": mode,
        "sweep_status": status, "sweep_complete": complete,
        "sizes_evaluated": len(rows), "unresolved": data.get("search", {}).get("unresolved", []),
        "strict_optimum": None if strict is None else _size_entry(by_size[strict]),
        "robust_optimum": None if robust is None else _size_entry(by_size[robust]),
        "selected": None if selected is None else _size_entry(by_size[selected]),
        "mechanism_file": None, "idt": [], "flame_speed": [], "files": {},
    }
    log.info("report: tolerance %.3g%%, %d sizes evaluated, strict optimum %s, robust optimum %s",
             100.0 * tol, len(rows), strict, robust)

    idt_cfg, conditions, fs_cfg, fs_conds, evaluate_fs = evaluation_config(cfg)
    _check_flame_backend(fs_cfg, evaluate_fs and selected is not None)
    gas_full = load_mechanism(cfg["mechanism"], cfg["phase"], stage="report")
    gas_red = None
    if selected is not None:
        keep = keep_list(names, removal_order, selected)
        mech_file = stored.get("mechanism_file") if stored.get("selected") == selected else None
        if not (mech_file and Path(mech_file).is_file()) and rcfg["write_mechanism"]:
            mech_file = str(write_reduced_mechanism(gas_full, keep, out_dir / f"reduced_{selected}sp.yaml",
                                                    stage="report"))
        result["mechanism_file"] = mech_file if mech_file and Path(mech_file).is_file() else None
        result["species"] = keep
        transport = fs_cfg["transport_model"] if (evaluate_fs and data.get("fs_conditions")) else None
        gas_red = reduced_solution(gas_full, keep, transport)
        result["T_unburnt_K"] = float(fs_cfg["T_unburnt_K"]) if evaluate_fs else None

        result["idt"] = _idt_rows(gas_full, gas_red, conditions, idt_cfg, int(cfg["report"]["T_points"]),
                                  f"n{selected}", t0)
        if data.get("fs_conditions"):
            if evaluate_fs:
                result["flame_solver"] = _flame_solver_text(fs_cfg)
                result["flame_speed"] = _fs_rows(gas_full, gas_red, fs_conds, fs_cfg, f"n{selected}", t0)
            else:
                log.info("the sweep evaluated flames, but reduction.evaluate_fs is false now; no flame comparison.")
    else:
        log.warning("no evaluated mechanism is within %.3g%%; the report holds the error curve only.", 100.0 * tol)

    if cfg["report"]["plots"]:
        try:
            result["files"].update(_plots(rows, tol, strict, robust, result, report_dir))
        except ImportError as exc:
            log.warning("no figures: Matplotlib is not installed (%s). Install it with 'pip install gasig[report]'.", exc)
    json_path = report_dir / "report.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_ready(result), fh, indent=2, allow_nan=False)
    md_path = report_dir / "report.md"
    md_path.write_text(_markdown(result, rows, cfg), encoding="utf-8")
    result["files"].update({"json": str(json_path), "markdown": str(md_path)})
    result["seconds"] = round(time.perf_counter() - t0, 2)
    log.info("report written to %s (%.1f s)", report_dir, result["seconds"])
    return result


def _size_entry(row: dict) -> dict:
    return {"n_species": row["n_species"], "n_reactions": row["n_reactions"],
            "max_idt_error": _finite_or_none(row["max_idt_error"]), "complete": row.get("complete", True)}


def _json_ready(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _plots(rows: list[dict], tol: float, strict, robust, result: dict, report_dir: Path) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = {}
    floor = 1e-3                                                     # percent, for the log axis
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    kept = [r for r in rows if r.get("kept_back")]                 # greedy tests that kept their species
    rows = [r for r in rows if not r.get("kept_back")]
    complete = [r for r in rows if r.get("complete", True) and math.isfinite(r["max_idt_error"])]
    bounds = [r for r in rows if not r.get("complete", True) and math.isfinite(r["max_idt_error"])]
    unignited = [r for r in rows if not math.isfinite(r["max_idt_error"])]
    ax.plot([r["n_species"] for r in complete], [max(100.0 * r["max_idt_error"], floor) for r in complete],
            "o-", ms=4, lw=1, color="tab:blue", label="maximum IDT error")
    if bounds:
        ax.plot([r["n_species"] for r in bounds], [max(100.0 * r["max_idt_error"], floor) for r in bounds],
                "^", ms=6, mfc="none", color="tab:blue", label="lower bound (early exit)")
    top = max([100.0 * r["max_idt_error"] for r in complete + bounds] + [100.0 * tol]) * 3.0
    if unignited:
        ax.plot([r["n_species"] for r in unignited], [top] * len(unignited), "x", color="tab:red", label="no ignition")
    if kept:
        ax.plot([r["n_species"] for r in kept],
                [max(100.0 * r["max_idt_error"], floor) if math.isfinite(r["max_idt_error"]) else top for r in kept],
                "s", ms=5, mfc="none", color="tab:gray", label="kept back (greedy test)")
    ax.axhline(100.0 * tol, ls="--", color="tab:gray", label=f"tolerance {100.0 * tol:.3g} %")
    if strict is not None and strict == robust:
        ax.axvline(strict, ls=":", color="tab:orange", label=f"selected: {strict} species (strict and robust optimum)")
    else:
        if strict is not None:
            ax.axvline(strict, ls=":", color="tab:orange", label=f"strict optimum: {strict} species")
        if robust is not None:
            ax.axvline(robust, ls="-.", color="tab:green", label=f"robust optimum (selected): {robust} species")
    ax.set_yscale("log")
    ax.set_xlabel("Number of species")
    ax.set_ylabel("Maximum relative IDT error, %")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = report_dir / "error_vs_species.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files["error_plot"] = str(path)
    files.update(_plot_idt(result, report_dir))
    files.update(_plot_fs(result, report_dir))
    return files


def _plot_idt(result: dict, report_dir: Path) -> dict[str, str]:
    """The ignition delay time against 1000/T, one panel per pressure and equivalence ratio."""
    import matplotlib.pyplot as plt

    files = {}
    if result["idt"]:
        pairs = list(dict.fromkeys((e["P_atm"], e["phi"]) for e in result["idt"]))
        ncols = min(3, len(pairs))
        nrows = int(math.ceil(len(pairs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
        n_sel = result["selected"]["n_species"]
        for k, (P, phi) in enumerate(pairs):
            ax = axes[k // ncols][k % ncols]
            entries = [e for e in result["idt"] if (e["P_atm"], e["phi"]) == (P, phi)]
            x = [1000.0 / e["T_K"] for e in entries]
            full = [1e3 * e["idt_detailed_s"] if e["idt_detailed_s"] else np.nan for e in entries]
            red = [1e3 * e["idt_reduced_s"] if e["idt_reduced_s"] else np.nan for e in entries]
            ax.semilogy(x, full, "-", color="k", lw=1.2, label=f"detailed, {result['n_species_full']} species")
            ax.semilogy(x, red, "o", mfc="none", color="tab:red", ms=6, label=f"reduced, {n_sel} species")
            ax.set_title(f"P = {P:.3g} atm, phi = {phi:.3g}", fontsize=10)
            ax.set_xlabel("1000 / T, 1/K")
            ax.set_ylabel("Ignition delay time, ms")
            ax.grid(True, which="both", alpha=0.3)
            if k == 0:
                ax.legend(fontsize=8)
        for k in range(len(pairs), nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)
        fig.tight_layout()
        path = report_dir / "idt_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files["idt_plot"] = str(path)
    return files


def _plot_fs(result: dict, report_dir: Path) -> dict[str, str]:
    """The flame speed against the equivalence ratio; one colour per pressure, lines detailed, markers reduced."""
    import matplotlib.pyplot as plt

    files = {}
    if result["flame_speed"]:
        pressures = list(dict.fromkeys(e["P_atm"] for e in result["flame_speed"]))
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        n_sel = result["selected"]["n_species"]
        for k, P in enumerate(pressures):
            entries = [e for e in result["flame_speed"] if e["P_atm"] == P]
            phi = [e["phi"] for e in entries]
            full = [100.0 * e["detailed_m_s"] if e["detailed_m_s"] else np.nan for e in entries]
            red = [100.0 * e["reduced_m_s"] if e["reduced_m_s"] else np.nan for e in entries]
            ax.plot(phi, full, "s-", color=f"C{k}", ms=5, lw=1.2,
                    label=f"detailed ({result['n_species_full']} species), {P:.3g} atm")
            ax.plot(phi, red, "o", mfc="none", color=f"C{k}", ms=8, mew=1.5,
                    label=f"reduced ({n_sel} species), {P:.3g} atm")
        ax.set_xlabel("Equivalence ratio")
        ax.set_ylabel("Laminar flame speed, cm/s")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = report_dir / "flame_speed_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files["flame_speed_plot"] = str(path)
    return files


def _markdown(result: dict, rows: list[dict], cfg: dict) -> str:
    tol = result["error_tolerance"]
    lines = [f"# Reduction report: {Path(result['mechanism']).name}", "",
             f"Date: {time.strftime('%Y-%m-%d')}. Tool: gasig {__version__}. Status: generated.", ""]
    if not result["sweep_complete"]:
        lines += [f"**Warning:** the sweep did not complete (status `{result['sweep_status']}`). This report uses "
                  f"only the {result['sizes_evaluated']} evaluated sizes. A smaller mechanism within the tolerance "
                  "can exist. Run `gasig reduce` again to complete the sweep.", ""]
    lines += ["## Selection", "",
              "| Item | Value |", "|---|---|",
              f"| Detailed mechanism | {result['n_species_full']} species, {result['n_reactions_full']} reactions |",
              f"| Sweep status | {result['sweep_status']}"
              f"{'' if result['sweep_complete'] else ': the sweep did not complete'} |",
              f"| Error tolerance | {100.0 * tol:.3g} % of the ignition delay time |",
              f"| Sizes evaluated | {result['sizes_evaluated']} ({result['sizes_evaluated'] - 1} reduced mechanisms, "
              f"{result['search_mode']} search) |"]
    for key, label in (("strict_optimum", "Strict optimum"), ("robust_optimum", f"Robust optimum (window {result['robust_window']})")):
        entry = result[key]
        value = "none" if entry is None else \
            f"{entry['n_species']} species, {entry['n_reactions']} reactions, maximum error {_fmt_pct(entry['max_idt_error'])} %"
        lines.append(f"| {label} | {value} |")
    sel = result["selected"]
    if sel is None:
        lines.append("| Selected mechanism | none: no evaluated size is within the tolerance |")
    else:
        file_text = f"`{Path(result['mechanism_file']).name}`" if result["mechanism_file"] else "not written"
        lines.append(f"| Selected mechanism | {sel['n_species']} species, {sel['n_reactions']} reactions, file {file_text} |")
    lines.append("")
    if result["unresolved"]:
        lines.append("The search left these intervals unresolved. A smaller mechanism within the tolerance can exist there:")
        lines.append("")
        for u in result["unresolved"]:
            lines.append(f"- sizes {u['from']} to {u['to']} ({u['kind']}), smallest error at the ends {_fmt_pct(u['smallest_error'])} %")
        lines.append("")
    lines += ["## Error against the number of species", ""]
    if result["files"].get("error_plot"):
        lines += [f"Figure 1. `{Path(result['files']['error_plot']).name}`: the maximum relative error of the ignition delay time against the number of species.", ""]
    centre = sel["n_species"] if sel else rows[-1]["n_species"]
    near = [r for r in rows if abs(r["n_species"] - centre) <= 5]
    lines += ["Table 2. Sizes near the selection (`reduction_sweep.csv` holds all sizes)", "",
              "| Species | Reactions | Maximum IDT error, % | Complete | Removed since the size above |", "|---|---|---|---|---|"]
    for r in near:
        lines.append(f"| {r['n_species']} | {r['n_reactions']} | {_fmt_pct(r['max_idt_error'])} | "
                     f"{'yes' if r.get('complete', True) else 'no, lower bound'} | {(r.get('removed_in_step', '') or '-') + (' (kept back)' if r.get('kept_back') else '')} |")
    lines.append("")
    lines += _md_idt(result, figure=2, table=3, name="the selected mechanism")
    lines += _md_fs(result, figure=3, table=4, name="the selected mechanism")
    lines += ["## Files", ""]
    if result["mechanism_file"]:
        lines.append(f"- `{Path(result['mechanism_file']).name}`: the selected mechanism; load it with `cantera.Solution`.")
    lines.append("- `report.json`: every number of this report.")
    for key in ("error_plot", "idt_plot", "flame_speed_plot"):
        if result["files"].get(key):
            lines.append(f"- `{Path(result['files'][key]).name}`")
    lines.append("- `reduction_sweep.csv`, `reduction_sweep.json`: the sweep that this report reads.")
    lines.append("")
    return "\n".join(lines)


def _md_idt(result: dict, figure: int, table: int, name: str) -> list[str]:
    if not result["idt"]:
        return []
    lines = ["## Ignition delay time", ""]
    if result["files"].get("idt_plot"):
        lines += [f"Figure {figure}. `{Path(result['files']['idt_plot']).name}`: the detailed mechanism (lines) and "
                  f"{name} (markers).", ""]
    valued = [e for e in result["idt"] if e["error"] is not None]
    if valued:
        worst = max(valued, key=lambda e: e["error"])
        mean = sum(e["error"] for e in valued) / len(valued)
        lines += [f"The largest error on the grid is {_fmt_pct(worst['error'])} % at {worst['T_K']:.0f} K, "
                  f"{worst['P_atm']:.3g} atm, phi {worst['phi']:.3g}. The mean error over the {len(valued)} points "
                  f"with a value is {_fmt_pct(mean)} %.", ""]
    missing = len(result["idt"]) - len(valued)
    if missing:
        lines += [f"{missing} of {len(result['idt'])} points have no value: a mechanism did not ignite or the "
                  "integrator failed there.", ""]
    lines += [f"Table {table}. Ignition delay time of the detailed mechanism and {name}", "",
              "| T, K | P, atm | phi | Detailed, ms | Reduced, ms | Error, % |", "|---|---|---|---|---|---|"]
    for e in result["idt"]:
        lines.append(f"| {e['T_K']:.0f} | {e['P_atm']:.3g} | {e['phi']:.3g} | {_fmt_ms(e['idt_detailed_s'])} | "
                     f"{_fmt_ms(e['idt_reduced_s'])} | {_fmt_pct(e['error'])} |")
    lines.append("")
    return lines


def _md_fs(result: dict, figure: int, table: int, name: str) -> list[str]:
    if not result["flame_speed"]:
        return []
    lines = ["## Flame speed", ""]
    if result.get("flame_solver"):
        lines += [f"Flame solver: {result['flame_solver']}. Unburnt temperature {result.get('T_unburnt_K', 300):.0f} K.", ""]
    if result["files"].get("flame_speed_plot"):
        lines += [f"Figure {figure}. `{Path(result['files']['flame_speed_plot']).name}`: the laminar flame speed "
                  f"against the equivalence ratio; lines for the detailed mechanism, markers for {name}.", ""]
    valued = [e for e in result["flame_speed"] if e["error"] is not None]
    if valued:
        worst = max(valued, key=lambda e: e["error"])
        mean = sum(e["error"] for e in valued) / len(valued)
        lines += [f"The largest error is {_fmt_pct(worst['error'])} % at {worst['P_atm']:.3g} atm, phi "
                  f"{worst['phi']:.3g}. The mean error over the {len(valued)} flames with a value is {_fmt_pct(mean)} %.",
                  ""]
    missing = len(result["flame_speed"]) - len(valued)
    if missing:
        lines += [f"{missing} of {len(result['flame_speed'])} flames have no value: the flame solver failed there.", ""]
    lines += [f"Table {table}. Flame speed of the detailed mechanism and {name}", "",
              "| P, atm | phi | Detailed, cm/s | Reduced, cm/s | Error, % |", "|---|---|---|---|---|"]
    for e in result["flame_speed"]:
        lines.append(f"| {e['P_atm']:.3g} | {e['phi']:.3g} | {_fmt_cm(e['detailed_m_s'])} | "
                     f"{_fmt_cm(e['reduced_m_s'])} | {_fmt_pct(e['error'])} |")
    lines.append("")
    return lines


def _markdown_mechanism(result: dict) -> str:
    sel = result["selected"]
    lines = [f"# Mechanism comparison: {Path(result['compared_mechanism']).name}", "",
             f"Date: {time.strftime('%Y-%m-%d')}. Tool: gasig {__version__}. Status: generated.", "",
             "## Mechanisms", "",
             "Table 1. The two mechanisms", "",
             "| Mechanism | File | Species | Reactions |", "|---|---|---|---|",
             f"| Detailed | `{Path(result['mechanism']).name}` | {result['n_species_full']} | {result['n_reactions_full']} |",
             f"| Compared | `{Path(result['compared_mechanism']).name}` | {sel['n_species']} | {sel['n_reactions']} |", ""]
    if result["species_not_in_detailed"]:
        lines += [f"The compared mechanism has {len(result['species_not_in_detailed'])} species that are not in the "
                  f"detailed mechanism: {', '.join(result['species_not_in_detailed'][:10])}.", ""]
    lines += _md_idt(result, figure=1, table=2, name="the compared mechanism")
    lines += _md_fs(result, figure=2, table=3, name="the compared mechanism")
    lines += ["## Files", "", "- `report.json`: every number of this report."]
    for key in ("idt_plot", "flame_speed_plot"):
        if result["files"].get(key):
            lines.append(f"- `{Path(result['files'][key]).name}`")
    lines.append("")
    return "\n".join(lines)


def build_mechanism_report(cfg: dict, out_dir: Path, mechanism: str | Path) -> dict:
    """Compare the mechanism file ``mechanism`` with the detailed mechanism of ``cfg``; no sweep is needed.

    The comparison runs on the evaluation conditions of the configuration
    (``reduction.idt`` over ``idt.solver``, with ``report.T_points``
    temperatures per pressure and equivalence ratio) and, with
    ``reduction.evaluate_fs``, on the flame conditions. The folder
    ``report_<stem>/`` in the output folder receives the tables and figures.
    """
    t0 = time.perf_counter()
    ct.suppress_thermo_warnings()
    mech_path = Path(mechanism)
    if not mech_path.is_file():
        raise MissingInputError(f"the mechanism to compare, {mech_path}, does not exist.", stage="report",
                                path=str(mech_path))
    idt_cfg, conditions, fs_cfg, fs_conds, evaluate_fs = evaluation_config(cfg)
    gas_full = load_mechanism(cfg["mechanism"], cfg["phase"], stage="report")
    gas_red = ct.Solution(str(mech_path))
    if evaluate_fs and fs_cfg["backend"] == "cantera" and gas_red.transport_model == "none":
        gas_red = ct.Solution(str(mech_path), transport_model=fs_cfg["transport_model"])
    mixture = set(idt_cfg["fuel"]) | set(idt_cfg["oxidizer"])
    if evaluate_fs:
        mixture |= set(fs_cfg["fuel"]) | set(fs_cfg["oxidizer"])
    missing = sorted(mixture - set(gas_red.species_names))
    if missing:
        raise MechanismError(f"the mechanism {mech_path.name} lacks the mixture species {missing}.", stage="report",
                             mechanism=str(mech_path), species=missing)
    _check_flame_backend(fs_cfg, evaluate_fs)
    report_dir = out_dir / f"report_{mech_path.stem}"
    report_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "mechanism": cfg["mechanism"], "n_species_full": gas_full.n_species, "n_reactions_full": gas_full.n_reactions,
        "compared_mechanism": str(mech_path),
        "selected": {"n_species": gas_red.n_species, "n_reactions": gas_red.n_reactions},
        "species": list(gas_red.species_names),
        "species_not_in_detailed": sorted(set(gas_red.species_names) - set(gas_full.species_names)),
        "idt": [], "flame_speed": [], "files": {},
    }
    log.info("mechanism report: %s (%d species, %d reactions) against %s (%d species, %d reactions)",
             mech_path.name, gas_red.n_species, gas_red.n_reactions, Path(cfg["mechanism"]).name,
             gas_full.n_species, gas_full.n_reactions)
    label = f"n{gas_red.n_species}"
    result["idt"] = _idt_rows(gas_full, gas_red, conditions, idt_cfg, int(cfg["report"]["T_points"]), label, t0)
    if evaluate_fs:
        result["flame_solver"] = _flame_solver_text(fs_cfg)
        result["T_unburnt_K"] = float(fs_cfg["T_unburnt_K"])
        result["flame_speed"] = _fs_rows(gas_full, gas_red, fs_conds, fs_cfg, label, t0)
    if cfg["report"]["plots"]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            result["files"].update(_plot_idt(result, report_dir))
            result["files"].update(_plot_fs(result, report_dir))
        except ImportError as exc:
            log.warning("no figures: Matplotlib is not installed (%s). Install it with 'pip install gasig[report]'.", exc)
    json_path = report_dir / "report.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_ready(result), fh, indent=2, allow_nan=False)
    md_path = report_dir / "report.md"
    md_path.write_text(_markdown_mechanism(result), encoding="utf-8")
    result["files"].update({"json": str(json_path), "markdown": str(md_path)})
    result["seconds"] = round(time.perf_counter() - t0, 2)
    log.info("report written to %s (%.1f s)", report_dir, result["seconds"])
    return result


def _mechanism_path(mechanism: Path, out_dir: Path, config_dir: Path) -> Path:
    """The file of ``--mechanism``: as given (the current folder), else in the output folder, else beside the config."""
    if mechanism.is_absolute():
        return mechanism
    candidates = [mechanism, out_dir / mechanism, config_dir / mechanism]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise MissingInputError(f"the mechanism to compare does not exist; tried "
                           f"{', '.join(str(c.resolve()) for c in candidates)}.", stage="report",
                           path=str(mechanism), tried=[str(c.resolve()) for c in candidates])


def run_report(config_path: str | Path, mechanism: str | Path | None = None) -> dict:
    """Load the configuration and write the report.

    Without ``mechanism`` the report reads the sweep result and compares the
    selected mechanism. With ``mechanism`` (a Cantera YAML file, relative to
    the current folder) the report compares that file with the detailed
    mechanism and needs no sweep.
    """
    config_path = Path(config_path)
    cfg = load_config(config_path)
    out_dir = Path(cfg["output_dir"])
    if mechanism is not None:
        mech_path = _mechanism_path(Path(mechanism), out_dir, config_path.resolve().parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(out_dir, cfg["log_level"], file_name=f"report_{mech_path.stem}.log", mode="w")
        try:
            log.info("GA-SiG mechanism report | config %s | mechanism %s", config_path, mech_path)
            return build_mechanism_report(cfg, out_dir, mech_path)
        except Exception:
            log_failure(log, "the report failed")
            raise
        finally:
            teardown_logging()
    json_path = out_dir / "reduction_sweep.json"
    if not json_path.is_file():
        raise MissingInputError(f"{json_path} is missing; run 'gasig reduce' first.", stage="report",
                                path=str(json_path))
    data = json.loads(json_path.read_text(encoding="utf-8"))
    setup_logging(out_dir, cfg["log_level"], file_name="report.log", mode="w")
    try:
        log.info("GA-SiG reduction report | config %s", config_path)
        return build_report(cfg, out_dir, data)
    except Exception:
        log_failure(log, "the report failed")
        raise
    finally:
        teardown_logging()
