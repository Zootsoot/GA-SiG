"""From sensitivity matrices to the normalized species-to-species matrix r_AB.

Manuscript definitions (defaults)
---------------------------------
``S_AB(a, b) = sum_{j in R_b} abs(S(a, j))`` where ``R_b`` is the set of
reactions with a non-zero net stoichiometric coefficient of species ``b``.
Over several trajectories the element-wise maximum of ``S_AB`` is taken,
and the result is normalized by its row sum: ``r_AB(a, b) = S_AB(a, b) /
sum_b' S_AB(a, b')``, so every row of ``r_AB`` sums to 1 (or is all zero).

Switches
--------
``normalization="S_row_sum"`` divides by the row sum of the element-wise
maximum of ``abs(S)`` over the reactions instead. This is the computation
behind the published results and is kept as a second version to test.
``abs_S=False`` sums the signed sensitivities, as the version 0.1 script did.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


def species_coupling(S: np.ndarray, nu_net: np.ndarray, abs_S: bool = True) -> np.ndarray:
    """Return ``S_AB`` (Ns, Ns) from ``S`` (Ns, Nr) and ``nu_net`` (Ns, Nr)."""
    S = np.asarray(S, dtype=float)
    nu_net = np.asarray(nu_net, dtype=float)
    if S.shape != nu_net.shape:
        raise ValueError(f"S has shape {S.shape} but nu_net has shape {nu_net.shape}.")
    participates = (nu_net != 0.0).astype(float)          # (Ns_b, Nr)
    base = np.abs(S) if abs_S else S
    return base @ participates.T                           # (Ns_a, Ns_b)


def normalize_rows(S_AB: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide each row of ``S_AB`` by ``denominator``; rows whose denominator is not positive become zero."""
    S_AB = np.asarray(S_AB, dtype=float)
    denominator = np.asarray(denominator, dtype=float).reshape(-1)
    r = np.zeros_like(S_AB)
    rows = denominator > 0.0
    r[rows] = S_AB[rows] / denominator[rows, None]
    return r


@dataclass
class Aggregate:
    S_abs_max: np.ndarray   # (Ns, Nr) element-wise max of |S| over trajectories
    S_AB: np.ndarray        # (Ns, Ns) element-wise max of S_AB over trajectories
    r_AB: np.ndarray        # (Ns, Ns) normalized coupling matrix
    n_trajectories: int


def aggregate(
    S_list: list[np.ndarray],
    nu_net: np.ndarray,
    abs_S: bool = True,
    normalization: str = "S_AB_row_sum",
) -> Aggregate:
    """Combine the sensitivity matrices of all trajectories into ``r_AB``."""
    if not S_list:
        raise ValueError("No sensitivity matrices to aggregate.")
    if normalization not in ("S_AB_row_sum", "S_row_sum"):
        raise ValueError("normalization must be 'S_AB_row_sum' or 'S_row_sum'.")
    S_abs_max = np.abs(np.asarray(S_list[0], dtype=float))
    S_AB = species_coupling(S_list[0], nu_net, abs_S)
    for S_m in S_list[1:]:
        S_abs_max = np.maximum(S_abs_max, np.abs(np.asarray(S_m, dtype=float)))
        S_AB = np.maximum(S_AB, species_coupling(S_m, nu_net, abs_S))
    if normalization == "S_AB_row_sum":
        denominator = S_AB.sum(axis=1)
        if not abs_S:
            n_bad = int(np.sum(denominator <= 0.0))
            log.warning("abs_S is false with the S_AB_row_sum normalization: the signed row sums can cancel; "
                        "%d of %d rows have a non-positive sum and become zero.", n_bad, S_AB.shape[0])
    else:
        denominator = S_abs_max.sum(axis=1)
    r_AB = normalize_rows(S_AB, denominator)
    return Aggregate(S_abs_max=S_abs_max, S_AB=S_AB, r_AB=r_AB, n_trajectories=len(S_list))
