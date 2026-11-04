"""Detection of chain units from the reaction extents along the idt trajectories.

A chain unit is a run of species A -> B -> ... in which each member has one
significant producing reaction and one significant consuming reaction, and
the consuming reaction of one member is the producing reaction of the next.
The flux of the run passes through every member, so the removal of any
member cuts the run. The ``lk_greedy`` search removes a unit as one piece.

Method, for the species b and the reaction j:

1. The reaction extent along a trajectory up to its end index,
   ``xi_j = integral of q_j / rho dt`` [kmol/kg], with the Cantera net rates
   of progress at every stored state (trapezoidal rule). The extents are
   summed over the trajectories.
2. The species change per reaction ``X_bj = nu_bj xi_j``. The production of
   b is ``P_b = sum_j max(X_bj, 0)``, the consumption ``C_b = sum_j
   max(-X_bj, 0)``.
3. The reaction j is a significant producer of b when ``X_bj > share P_b``,
   a significant consumer when ``-X_bj > share C_b``. A candidate species with
   exactly one significant producer and one significant consumer is a link.
4. Two links join when the consumer of one is the producer of the other. A
   connected group of two or more links is a unit.

The candidates are the removable species (``scope: all``) or the removable
species below the median GA-SiG importance of the removable species
(``scope: tail``, the rule of the prototype on the jet-fuel skeleton). The
module needs Cantera and NumPy only; it does not import torch.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cantera as ct
import numpy as np

from .errors import InputError, MechanismError
from .trajectory import Trajectory

log = logging.getLogger("gasig.units")


def reaction_extents(gas: ct.Solution, traj: Trajectory) -> np.ndarray:
    """Time-integrated net rates of progress per unit mass, (Nr,) [kmol/kg], up to the end index."""
    e = int(traj.end_index)
    rates = np.zeros((e + 1, gas.n_reactions))
    for k in range(e + 1):
        Y = np.clip(np.asarray(traj.Y[k], dtype=float), 0.0, None)
        total = float(Y.sum())
        if total <= 0.0:
            raise InputError(f"{traj.label or 'trajectory'}: the mass fractions at sample {k} have no positive entry.",
                             stage="units", label=traj.label, sample=k)
        gas.TPY = float(traj.T[k]), float(traj.P[k]), Y / total
        rates[k] = gas.net_rates_of_progress / gas.density
    t = np.asarray(traj.tau[:e + 1], dtype=float)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return trapezoid(rates, t, axis=0)


def chain_links(X: np.ndarray, candidates: list[int], share: float) -> dict[int, tuple[int, int]]:
    """The candidates with one significant producer and one significant consumer: {b: (producer, consumer)}."""
    links = {}
    for b in candidates:
        pos = np.maximum(X[b], 0.0)
        neg = np.maximum(-X[b], 0.0)
        P, C = float(pos.sum()), float(neg.sum())
        if P <= 0.0 or C <= 0.0:
            continue
        producers = np.flatnonzero(pos > share * P)
        consumers = np.flatnonzero(neg > share * C)
        if len(producers) == 1 and len(consumers) == 1:
            links[int(b)] = (int(producers[0]), int(consumers[0]))
    return links


def join_links(links: dict[int, tuple[int, int]], names: list[str]) -> list[list[int]]:
    """Connected groups of two or more links, each in chain order (producer side first)."""
    parent = {b: b for b in links}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    by_producer: dict[int, list[int]] = {}
    for b, (prod, _) in links.items():
        by_producer.setdefault(prod, []).append(b)
    for a, (_, cons) in links.items():
        for b in by_producer.get(cons, []):
            if a != b:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for b in links:
        groups.setdefault(find(b), []).append(b)
    units = []
    for members in groups.values():
        if len(members) < 2:
            continue
        member_set = set(members)
        # A member is upstream of b when its consumer is the producer of b.
        upstream = {b: {a for a in members if a != b and links[a][1] == links[b][0]} for b in members}
        ordered, placed = [], set()
        while len(ordered) < len(members):
            ready = sorted((b for b in member_set - placed if upstream[b] <= placed), key=lambda i: names[i])
            if not ready:          # a cycle: place the rest by name
                ready = sorted(member_set - placed, key=lambda i: names[i])
            ordered.append(ready[0])
            placed.add(ready[0])
        units.append(ordered)
    return sorted(units, key=lambda u: names[u[0]])


def detect_units(gas: ct.Solution, trajs: list[Trajectory], removable: set[str], share: float = 0.01,
                 importance: dict[str, float] | None = None) -> dict:
    """Detect the chain units; ``importance`` (GA-SiG) restricts the candidates to the tail.

    Returns a JSON-ready dictionary with the units (species names in chain
    order), the links with their producing and consuming reactions, and the
    numbers of candidates and trajectories.
    """
    t0 = time.perf_counter()
    names = list(gas.species_names)
    nu = gas.product_stoich_coeffs - gas.reactant_stoich_coeffs
    xi = np.zeros(gas.n_reactions)
    for traj in trajs:
        traj.validate(gas.n_species)
        xi += reaction_extents(gas, traj)
    X = nu * xi[None, :]
    cand = [i for i, s in enumerate(names) if s in removable]
    median = None
    if importance is not None:
        values = np.array([float(importance.get(names[i], 0.0)) for i in cand])
        median = float(np.median(values)) if values.size else 0.0
        cand = [i for i, v in zip(cand, values) if v < median]
    links = chain_links(X, cand, share)
    units = join_links(links, names)
    equations = gas.reaction_equations()
    out = {
        "scope": "all" if importance is None else "tail",
        "share": share,
        "n_trajectories": len(trajs),
        "n_candidates": len(cand),
        "tail_median_importance": median,
        "n_links": len(links),
        "units": [[names[i] for i in u] for u in units],
        "links": {names[b]: {"producer": equations[p], "consumer": equations[c]} for b, (p, c) in sorted(links.items())},
        "seconds": round(time.perf_counter() - t0, 2),
    }
    return out


def write_units(path: Path, result: dict) -> Path:
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return Path(path)


def run_units(cfg: dict, out_dir: Path, trajs: list[Trajectory]) -> dict:
    """The detection stage: write ``units.output_file`` in the output folder; returns the summary block."""
    from .errors import ConfigError, MissingInputError
    from .reduce import evaluation_config, load_mechanism, protected_species, read_importance

    ucfg = cfg["units"]
    gas = load_mechanism(cfg["mechanism"], cfg["phase"], stage="units")
    names = list(gas.species_names)
    try:
        idt_cfg, _, fs_cfg, _, evaluate_fs = evaluation_config(cfg)
        protected = protected_species(cfg, names, idt_cfg, fs_cfg if evaluate_fs else None)
    except ConfigError as exc:
        protected = (set(cfg["ranking"]["target_species"]) | set(cfg["reduction"]["keep_species"])) & set(names)
        log.warning("[units] the evaluation conditions are incomplete (%s); only the targets and "
                    "reduction.keep_species count as protected.", exc)
    importance = None
    if ucfg["scope"] == "tail":
        rank_path = out_dir / "ranked_species.csv"
        if not rank_path.is_file():
            raise MissingInputError(f"{rank_path} is missing; units.scope 'tail' needs the GA-SiG rank. Run the "
                                    "rank stage first.", stage="units", path=str(rank_path))
        importance = read_importance(rank_path)
        if not importance:
            raise InputError(f"{rank_path} has no 'importance' column.", stage="units", path=str(rank_path))
        if set(importance) != set(names):
            raise MechanismError(f"{rank_path} does not list the species of the mechanism; run the rank stage "
                                 "with this mechanism.", stage="units", path=str(rank_path))
    result = detect_units(gas, trajs, set(names) - protected, float(ucfg["share"]), importance)
    result["protected_species"] = sorted(protected)
    path = write_units(out_dir / ucfg["output_file"], result)
    log.info("[units] %d chain units with %d species from %d links among %d candidates (%s scope, share %.3g), "
             "%d trajectories, %.1f s; written to %s", len(result["units"]), sum(map(len, result["units"])),
             result["n_links"], result["n_candidates"], result["scope"], result["share"], result["n_trajectories"],
             result["seconds"], path)
    for unit in result["units"]:
        log.info("[units]   %s", " -> ".join(unit))
    return {k: v for k, v in result.items() if k != "links"} | {"file": str(path)}
