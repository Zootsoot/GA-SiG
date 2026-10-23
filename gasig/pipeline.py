"""Run a GA-SiG configuration from data to ranked species.

Stages
------
1. Data: run the Cantera solvers or load precomputed files, per enabled mode.
2. Sensitivity: integrate every trajectory, aggregate to ``r_AB`` per mode,
   write ``r_AB_<mode>.npy`` and ``S_AB_<mode>.npy``.
3. Ranking: select or blend the per-mode matrices, run the graph search,
   write ``r_AB.npy`` (input matrix of the graph search), ``importance.npy``,
   ``ranked_sp.npy`` (version 0.1 format) and ``ranked_species.csv``.

Every run also writes ``summary.json`` and ``gasig.log`` into the output
folder; the sensitivity stages also write ``config_resolved.json``. A
ranking-only run keeps the record of the earlier sensitivity run in
``summary.json`` and adds its own ranking block.
"""
from __future__ import annotations

import csv
import json
import logging
import platform
import sys
import time
from pathlib import Path

import cantera as ct
import numpy as np
import torch

from . import __version__
from .aggregate import aggregate
from .config import ConfigError, load_config, public_config
from .graph import combine_matrices, overall_importance, rank_species, resolve_targets
from .io import load_fs_precomputed, load_idt_precomputed, save_fs_data, save_idt_data
from .kinetics import TorchKinetics
from .sensitivity import Trajectory, integrate_trajectory
from .solvers.flame import run_flame_set
from .solvers.idt import run_idt_set

log = logging.getLogger("gasig")
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def resolve_device(name: str) -> torch.device:
    """Turn the config value into a torch device; 'auto' prefers CUDA."""
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError(f"device '{name}' was requested but CUDA is not available; use 'cpu' or 'auto'.")
    return torch.device(name)


def teardown_logging() -> None:
    """Remove and close the handlers that :func:`setup_logging` added."""
    logger = logging.getLogger("gasig")
    for handler in list(logger.handlers):
        if getattr(handler, "_gasig", False):
            logger.removeHandler(handler)
            handler.close()


def setup_logging(output_dir: Path, level: str) -> None:
    """Console and file logging for the 'gasig' logger tree."""
    teardown_logging()
    logger = logging.getLogger("gasig")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
    console._gasig = True  # type: ignore[attr-defined]
    logger.addHandler(console)
    file_handler = logging.FileHandler(output_dir / "gasig.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_handler._gasig = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)
    logger.propagate = False


def _json_ready(obj):
    """Convert numpy scalars so that ``json.dump`` accepts the object; arrays are dropped."""
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                continue
            out[str(k)] = _json_ready(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj if not isinstance(v, np.ndarray)]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def collect_trajectories(mode: str, cfg: dict, out_dir: Path) -> list[Trajectory]:
    """Run the solver or load the precomputed files for one mode."""
    mcfg = cfg[mode]
    if mcfg["source"] == "solver":
        if mode == "idt":
            trajs = run_idt_set(cfg["mechanism"], cfg["phase"], mcfg["solver"])
            if mcfg["solver"]["save_trajectories"]:
                save_idt_data(out_dir / "idt_data", trajs)
                log.info("idt trajectories written to %s", out_dir / "idt_data")
        else:
            trajs = run_flame_set(cfg["mechanism"], cfg["phase"], mcfg["solver"])
            if mcfg["solver"]["save_profiles"]:
                save_fs_data(out_dir / "fs_data", trajs)
                log.info("fs profiles written to %s", out_dir / "fs_data")
        return trajs
    if mode == "idt":
        return load_idt_precomputed(mcfg["precomputed"])
    return load_fs_precomputed(mcfg["precomputed"], mcfg["flame_tip_T_fraction"])


def run_mode(mode: str, cfg: dict, kin: TorchKinetics, out_dir: Path) -> dict:
    """Data, sensitivity integration and aggregation for one mode."""
    t0 = time.perf_counter()
    scfg = cfg["sensitivity"]
    acfg = cfg["aggregation"]
    trajs = collect_trajectories(mode, cfg, out_dir)
    log.info("[%s] integrating %d trajectories on %s (rate_law=%s, substep_tol=%s)",
             mode, len(trajs), kin.device, kin.rate_law, scfg["substep_tol"])

    S_list, records = [], []
    for traj in trajs:
        S, stats = integrate_trajectory(
            kin, traj,
            substep_tol=scfg["substep_tol"],
            max_substeps=scfg["max_substeps"],
            progress_every=scfg["progress_every"],
        )
        S_list.append(S)
        record = {"label": traj.label, "end_index": traj.end_index, "n_points": int(traj.tau.shape[0]),
                  "tau_end_s": float(traj.tau[traj.end_index])}
        record.update(_json_ready(traj.meta))
        record["integration"] = stats.as_dict()
        records.append(record)

    agg = aggregate(S_list, kin.nu_net_np, abs_S=acfg["abs_S"], normalization=acfg["normalization"])
    files = {
        "r_AB": out_dir / f"r_AB_{mode}.npy",
        "S_AB": out_dir / f"S_AB_{mode}.npy",
        "trajectories": out_dir / f"trajectories_{mode}.json",
    }
    np.save(files["r_AB"], agg.r_AB)
    np.save(files["S_AB"], agg.S_AB)
    if scfg["save_S"]:
        files["S_abs_max"] = out_dir / f"S_abs_max_{mode}.npy"
        np.save(files["S_abs_max"], agg.S_abs_max)
    with files["trajectories"].open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    seconds = time.perf_counter() - t0
    log.info("[%s] r_AB written to %s (%d x %d), %.1f s", mode, files["r_AB"], *agg.r_AB.shape, seconds)
    return {
        "source": cfg[mode]["source"],
        "n_trajectories": len(trajs),
        "abs_S": acfg["abs_S"],
        "normalization": acfg["normalization"],
        "rate_law": kin.rate_law,
        "substep_tol": scfg["substep_tol"],
        "files": {k: str(v) for k, v in files.items()},
        "trajectories": records,
        "seconds": round(seconds, 2),
    }


def _modes_for_ranking(cfg: dict) -> list[str]:
    """The modes whose r_AB files the ranking needs, from ``ranking.combine``."""
    combine = cfg["ranking"]["combine"]
    enabled = [m for m in ("idt", "fs") if cfg[m]["enabled"]]
    if combine == "auto":
        return enabled
    if combine == "hybrid":
        return ["idt", "fs"]
    return [combine]


def rank(cfg: dict, species_names: list[str], out_dir: Path) -> dict:
    """Graph search on the per-mode matrices of ``out_dir``; writes the ranking files."""
    rcfg = cfg["ranking"]
    matrices = {}
    for mode in _modes_for_ranking(cfg):
        path = out_dir / f"r_AB_{mode}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run the sensitivity stage for mode '{mode}' first.")
        matrices[mode] = np.load(path)
    R, used = combine_matrices(matrices.get("idt"), matrices.get("fs"), rcfg["combine"], rcfg["hybrid_weight_idt"])
    if R.shape != (len(species_names), len(species_names)):
        raise ValueError(f"r_AB has shape {R.shape} but the mechanism has {len(species_names)} species.")

    targets = resolve_targets(species_names, rcfg["target_species"])
    if not targets:
        log.warning("No valid target species; every importance index is zero.")
    importance = overall_importance(R, targets, rcfg["eps_edge"])
    ranked = rank_species(species_names, importance)

    np.save(out_dir / "r_AB.npy", R)
    np.save(out_dir / "importance.npy", importance)
    np.save(out_dir / "ranked_sp.npy", np.array(ranked))          # version 0.1 format: (Ns, 2) strings
    with (out_dir / "ranked_species.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank_from_least_important", "species", "importance"])
        for i, (name, value) in enumerate(ranked, start=1):
            writer.writerow([i, name, f"{value:.10e}"])

    top_n = int(rcfg["top_n_print"])
    top = ranked[max(len(ranked) - top_n, 0):] if top_n > 0 else []
    bottom = ranked[:top_n] if top_n > 0 else []
    n_edges = int(np.sum(R > rcfg["eps_edge"]))
    log.info("Ranking with the %s matrix; targets %s; %d edges above eps_edge=%.1e",
             used, [species_names[i] for i in targets], n_edges, rcfg["eps_edge"])
    if top:
        log.info("Top %d most important species:", len(top))
        for name, value in reversed(top):
            log.info("  %-16s I = %.4e", name, value)
    return {
        "matrix": used,
        "targets": [species_names[i] for i in targets],
        "n_edges": n_edges,
        "n_unreachable": int(np.sum(importance <= 0.0)),
        "files": {k: str(out_dir / k) for k in ("r_AB.npy", "importance.npy", "ranked_sp.npy", "ranked_species.csv")},
        "most_important": [{"species": n, "importance": v} for n, v in reversed(top)],
        "least_important": [{"species": n, "importance": v} for n, v in bottom],
    }


def _previous_summary(out_dir: Path) -> dict:
    path = out_dir / "summary.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def run(config_path: str | Path, device: str | None = None, stage: str = "all") -> dict:
    """Run the stages of a configuration. ``stage`` is 'all', 'sensitivity' or 'rank'."""
    if stage not in ("all", "sensitivity", "rank"):
        raise ValueError("stage must be 'all', 'sensitivity' or 'rank'.")
    t0 = time.perf_counter()
    ct.suppress_thermo_warnings()
    config_path = Path(config_path)
    cfg = load_config(config_path)
    if device:
        cfg["device"] = device
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, cfg["log_level"])
    try:
        return _run_stages(cfg, config_path, out_dir, stage, t0)
    finally:
        teardown_logging()


def _run_stages(cfg: dict, config_path: Path, out_dir: Path, stage: str, t0: float) -> dict:
    log.info("GA-SiG %s | config %s | stage %s", __version__, config_path, stage)
    previous = _previous_summary(out_dir) if stage == "rank" else {}
    summary: dict = {
        "gasig_version": __version__,
        "config": str(config_path),
        "stage": stage,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cantera": ct.__version__,
        "device": previous.get("device"),
        "mechanism": previous.get("mechanism"),
        "modes": previous.get("modes", {}),
        "ranking": None,
    }

    if stage in ("all", "sensitivity"):
        with (out_dir / "config_resolved.json").open("w", encoding="utf-8") as fh:
            json.dump(_json_ready(public_config(cfg)), fh, indent=2)
        dev = resolve_device(cfg["device"])
        if dev.type == "cuda":
            log.info("device: %s (%s)", dev, torch.cuda.get_device_name(dev))
        else:
            log.info("device: %s", dev)
        dtype = torch.float64 if cfg["dtype"] == "float64" else torch.float32
        kin = TorchKinetics(
            cfg["mechanism"], cfg["phase"], device=dev, dtype=dtype,
            concentration_floor=cfg["sensitivity"]["concentration_floor"],
            rate_law=cfg["sensitivity"]["rate_law"],
            self_check=cfg["sensitivity"]["self_check"],
        )
        summary["device"] = str(dev)
        summary["mechanism"] = kin.describe()
        log.info("mechanism: %d species, %d reactions, %d three-body, %d irreversible",
                 kin.Ns, kin.Nr, kin.n_three_body, summary["mechanism"]["n_irreversible"])
        for mode in ("idt", "fs"):
            if cfg[mode]["enabled"]:
                summary["modes"][mode] = run_mode(mode, cfg, kin, out_dir)
        species_names = kin.species_names
    else:
        gas = ct.Solution(cfg["mechanism"]) if cfg["phase"] is None else ct.Solution(cfg["mechanism"], cfg["phase"])
        species_names = list(gas.species_names)
        if not summary["mechanism"]:
            summary["mechanism"] = {"mechanism": cfg["mechanism"], "n_species": gas.n_species,
                                    "n_reactions": gas.n_reactions}

    if stage in ("all", "rank"):
        summary["ranking"] = rank(cfg, species_names, out_dir)
        summary["ranking"]["config_at_ranking"] = _json_ready(cfg["ranking"])

    summary["seconds"] = round(time.perf_counter() - t0, 2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_json_ready(summary), fh, indent=2)
    log.info("done in %.1f s; summary at %s", summary["seconds"], out_dir / "summary.json")
    return summary
