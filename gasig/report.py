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
from .config import ConfigError, load_config
from .logs import log_failure, setup_logging, teardown_logging
from .reduce import evaluation_config, keep_list, reduced_solution, write_reduced_mechanism
from .search import select_optima
from .solvers.flame import run_flame
from .solvers.idt import run_idt

log = logging.getLogger("gasig.report")


def _idt(gas: ct.Solution, T: float, P: float, phi: float, idt_cfg: dict, label: str) -> float:
    traj = run_idt("", None, T, P, phi, idt_cfg, label=label, gas=gas)
    return float(traj.meta["tau_ign_s"]) if traj.meta["ignited"] else float("inf")


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


def build_report(cfg: dict, out_dir: Path, data: dict) -> dict:
    """Build the report of a sweep result ``data`` (the content of ``reduction_sweep.json``)."""
    t0 = time.perf_counter()
    ct.suppress_thermo_warnings()
    rcfg = cfg["reduction"]
    tol = rcfg["error_tolerance"] if rcfg["error_tolerance"] is not None else data.get("error_tolerance")
    if tol is None:
        raise ConfigError("set 'reduction.error_tolerance' to select a mechanism for the report.")
    tol = float(tol)
    if "species_order" not in data or "removal_order" not in data:
        raise ValueError("reduction_sweep.json comes from an older version; run 'gasig reduce' again.")
    window = int(rcfg["search"]["robust_window"])
    rows = sorted(data["rows"], key=lambda r: -r["n_species"])
    for r in rows:                                   # null in the JSON file means inf
        if r["max_idt_error"] is None:
            r["max_idt_error"] = float("inf")
    errors = {r["n_species"]: r["max_idt_error"] for r in rows}
    mode = data.get("search", {}).get("mode", "full")
    stored = data.get("selection") or {}
    if data.get("error_tolerance") == tol and stored.get("robust_window") == window:
        strict, robust = stored.get("strict_optimum"), stored.get("robust_optimum")
    else:
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
        "sizes_evaluated": len(rows), "unresolved": data.get("search", {}).get("unresolved", []),
        "strict_optimum": None if strict is None else _size_entry(by_size[strict]),
        "robust_optimum": None if robust is None else _size_entry(by_size[robust]),
        "selected": None if selected is None else _size_entry(by_size[selected]),
        "mechanism_file": None, "idt": [], "flame_speed": [], "files": {},
    }
    log.info("report: tolerance %.3g%%, %d sizes evaluated, strict optimum %s, robust optimum %s",
             100.0 * tol, len(rows), strict, robust)

    idt_cfg, conditions, fs_cfg, fs_conds, evaluate_fs = evaluation_config(cfg)
    gas_full = ct.Solution(cfg["mechanism"]) if cfg["phase"] is None else ct.Solution(cfg["mechanism"], cfg["phase"])
    gas_red = None
    if selected is not None:
        keep = keep_list(names, removal_order, selected)
        mech_file = data.get("selection", {}).get("mechanism_file")
        if not (mech_file and Path(mech_file).is_file()) and rcfg["write_mechanism"]:
            mech_file = str(write_reduced_mechanism(gas_full, keep, out_dir / f"reduced_{selected}sp.yaml"))
        result["mechanism_file"] = mech_file if mech_file and Path(mech_file).is_file() else None
        result["species"] = keep
        transport = fs_cfg["transport_model"] if (evaluate_fs and data.get("fs_conditions")) else None
        gas_red = reduced_solution(gas_full, keep, transport)

        # Ignition delay time on a denser temperature grid, per pressure and equivalence ratio.
        grid = _temperature_grid(conditions, int(cfg["report"]["T_points"]))
        pairs = list(dict.fromkeys((P, phi) for _, P, phi in conditions))
        for P, phi in pairs:
            for T in grid:
                tau_full = _idt(gas_full, T, P, phi, idt_cfg, f"full-T{T:.0f}")
                tau_red = _idt(gas_red, T, P, phi, idt_cfg, f"n{selected}-T{T:.0f}")
                result["idt"].append({"T_K": T, "P_atm": P / ct.one_atm, "phi": phi,
                                      "idt_detailed_s": _finite_or_none(tau_full),
                                      "idt_reduced_s": _finite_or_none(tau_red),
                                      "error": _finite_or_none(_error(tau_red, tau_full))})
        log.info("ignition delay times computed at %d temperatures x %d (P, phi) pairs (%.1f s)",
                 len(grid), len(pairs), time.perf_counter() - t0)
        if data.get("fs_conditions"):
            if evaluate_fs:
                for P, phi in fs_conds:
                    sl_full = float(run_flame("", None, P, phi, fs_cfg, label="full", gas=gas_full).meta["flame_speed_m_s"])
                    sl_red = float(run_flame("", None, P, phi, fs_cfg, label=f"n{selected}", gas=gas_red).meta["flame_speed_m_s"])
                    result["flame_speed"].append({"P_atm": P / ct.one_atm, "phi": phi, "detailed_m_s": sl_full,
                                                  "reduced_m_s": sl_red, "error": _error(sl_red, sl_full)})
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

    if result["flame_speed"]:
        pressures = list(dict.fromkeys(e["P_atm"] for e in result["flame_speed"]))
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for P in pressures:
            entries = [e for e in result["flame_speed"] if e["P_atm"] == P]
            phi = [e["phi"] for e in entries]
            ax.plot(phi, [100.0 * e["detailed_m_s"] for e in entries], "s-", color="k", ms=5, lw=1.2,
                    label=f"detailed, {P:.3g} atm")
            ax.plot(phi, [100.0 * e["reduced_m_s"] for e in entries], "o", mfc="none", color="tab:red", ms=7,
                    label=f"reduced, {P:.3g} atm")
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
             f"Date: {time.strftime('%Y-%m-%d')}. Tool: gasig {__version__}. Status: generated.", "",
             "## Selection", "",
             "| Item | Value |", "|---|---|",
             f"| Detailed mechanism | {result['n_species_full']} species, {result['n_reactions_full']} reactions |",
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
                     f"{'yes' if r.get('complete', True) else 'no, lower bound'} | {r.get('removed_in_step', '') or '-'} |")
    lines.append("")
    if result["idt"]:
        lines += ["## Ignition delay time", ""]
        if result["files"].get("idt_plot"):
            lines += [f"Figure 2. `{Path(result['files']['idt_plot']).name}`: the detailed mechanism (lines) and the selected mechanism (markers).", ""]
        worst = max(result["idt"], key=lambda e: (e["error"] if e["error"] is not None else math.inf))
        lines += [f"The largest error on the grid is {_fmt_pct(worst['error'])} % at {worst['T_K']:.0f} K, "
                  f"{worst['P_atm']:.3g} atm, phi {worst['phi']:.3g}.", "",
                  "Table 3. Ignition delay time of the detailed and the selected mechanism", "",
                  "| T, K | P, atm | phi | Detailed, ms | Reduced, ms | Error, % |", "|---|---|---|---|---|---|"]
        for e in result["idt"]:
            lines.append(f"| {e['T_K']:.0f} | {e['P_atm']:.3g} | {e['phi']:.3g} | {_fmt_ms(e['idt_detailed_s'])} | "
                         f"{_fmt_ms(e['idt_reduced_s'])} | {_fmt_pct(e['error'])} |")
        lines.append("")
    if result["flame_speed"]:
        lines += ["## Flame speed", ""]
        if result["files"].get("flame_speed_plot"):
            lines += [f"Figure 3. `{Path(result['files']['flame_speed_plot']).name}`: the laminar flame speed against the equivalence ratio.", ""]
        lines += ["Table 4. Flame speed of the detailed and the selected mechanism", "",
                  "| P, atm | phi | Detailed, cm/s | Reduced, cm/s | Error, % |", "|---|---|---|---|---|"]
        for e in result["flame_speed"]:
            lines.append(f"| {e['P_atm']:.3g} | {e['phi']:.3g} | {100.0 * e['detailed_m_s']:.2f} | "
                         f"{100.0 * e['reduced_m_s']:.2f} | {_fmt_pct(e['error'])} |")
        lines.append("")
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


def run_report(config_path: str | Path) -> dict:
    """Load the configuration, read the sweep result and write the report."""
    config_path = Path(config_path)
    cfg = load_config(config_path)
    out_dir = Path(cfg["output_dir"])
    json_path = out_dir / "reduction_sweep.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"{json_path} is missing; run 'gasig reduce' first.")
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
