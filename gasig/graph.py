"""Graph search: from r_AB to species importance indices.

Every entry ``r_AB(a, b) > eps_edge`` is a directed edge from ``a`` to ``b``
with the length ``w = -ln r_AB(a, b)``. Dijkstra's algorithm gives the
distance ``D`` from each target species to every species, and the
importance with respect to that target is ``exp(-D)``. The overall
importance of a species is the maximum over the targets. Target species
have importance 1; species with no path from any target have importance 0.
"""
from __future__ import annotations

import heapq
import logging

import numpy as np

log = logging.getLogger(__name__)


def build_adjacency(R: np.ndarray, eps_edge: float = 1e-16) -> list[list[tuple[int, float]]]:
    """Adjacency lists ``adj[a] = [(b, w_ab), ...]`` for every edge with ``R[a, b] > eps_edge``."""
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square, got shape {R.shape}.")
    if np.any(R > 1.0 + 1e-12):
        log.warning("r_AB has %d entries above 1; they are clipped to 1 so that all edge lengths stay non-negative.",
                    int(np.sum(R > 1.0 + 1e-12)))
    R = np.clip(R, None, 1.0)
    n = R.shape[0]
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    rows, cols = np.nonzero(R > eps_edge)
    weights = -np.log(R[rows, cols])
    for a, b, w in zip(rows.tolist(), cols.tolist(), weights.tolist()):
        adj[a].append((b, w))
    return adj


def dijkstra(adj: list[list[tuple[int, float]]], source: int) -> np.ndarray:
    """Shortest distances from ``source``; unreachable nodes get ``np.inf``."""
    n = len(adj)
    dist = np.full(n, np.inf, dtype=float)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist[u]:
            continue
        for v, w_uv in adj[u]:
            nd = d_u + w_uv
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def overall_importance(R: np.ndarray, target_indices: list[int], eps_edge: float = 1e-16) -> np.ndarray:
    """Importance index of every species: ``max_T exp(-D(T -> A))``."""
    n = np.asarray(R).shape[0]
    importance = np.zeros(n, dtype=float)
    if not target_indices:
        return importance
    adj = build_adjacency(R, eps_edge)
    for t in target_indices:
        dist = dijkstra(adj, int(t))
        with np.errstate(over="ignore", under="ignore"):
            R_TA = np.where(np.isfinite(dist), np.exp(-dist), 0.0)
        importance = np.maximum(importance, R_TA)
    return importance


def combine_matrices(
    r_idt: np.ndarray | None,
    r_fs: np.ndarray | None,
    combine: str = "auto",
    weight_idt: float = 0.5,
) -> tuple[np.ndarray, str]:
    """Select or blend the IDT and FS matrices. Returns ``(R, mode_used)``."""
    for name, r in (("r_AB_idt", r_idt), ("r_AB_fs", r_fs)):
        if r is not None:
            r = np.asarray(r)
            if r.ndim != 2 or r.shape[0] != r.shape[1]:
                raise ValueError(f"{name} must be a square matrix, got shape {r.shape}.")
    if r_idt is not None and r_fs is not None and np.shape(r_idt) != np.shape(r_fs):
        raise ValueError(f"r_AB_idt has shape {np.shape(r_idt)} but r_AB_fs has shape {np.shape(r_fs)}; "
                         "both modes must come from the same mechanism.")
    have_idt = r_idt is not None
    have_fs = r_fs is not None
    if combine == "auto":
        combine = "hybrid" if (have_idt and have_fs) else ("idt" if have_idt else "fs")
    if combine == "idt":
        if not have_idt:
            raise ValueError("ranking.combine is 'idt' but no IDT matrix is available.")
        return np.asarray(r_idt, dtype=float), "idt"
    if combine == "fs":
        if not have_fs:
            raise ValueError("ranking.combine is 'fs' but no FS matrix is available.")
        return np.asarray(r_fs, dtype=float), "fs"
    if combine == "hybrid":
        if not (have_idt and have_fs):
            raise ValueError("ranking.combine is 'hybrid' but both matrices are needed.")
        w = float(weight_idt)
        return w * np.asarray(r_idt, dtype=float) + (1.0 - w) * np.asarray(r_fs, dtype=float), "hybrid"
    raise ValueError(f"Unknown combine mode {combine!r}.")


def rank_species(species_names: list[str], importance: np.ndarray) -> list[tuple[str, float]]:
    """Species sorted from least to most important (stable for ties)."""
    pairs = [(name, float(importance[i])) for i, name in enumerate(species_names)]
    pairs.sort(key=lambda p: p[1])
    return pairs


def resolve_targets(species_names: list[str], target_species: list[str]) -> list[int]:
    """Map target names to indices; unknown names are logged and skipped."""
    index = {name: i for i, name in enumerate(species_names)}
    out = []
    for name in target_species:
        if name in index:
            out.append(index[name])
        else:
            log.warning("Target species '%s' is not in the mechanism and is ignored.", name)
    return out
