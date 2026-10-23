"""Trajectory container and the two profile helpers, free of torch imports.

A :class:`Trajectory` is one time series (idt mode) or one flame profile in
residence time (fs mode): a strictly increasing ``tau`` grid with the
thermo-chemical state at every point and an end index. The Cantera solvers,
the precomputed loaders and the reduction sweep use this module without
PyTorch. The integrator in :mod:`gasig.sensitivity` re-exports its names.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """One time series (idt) or one flame profile in residence time (fs)."""

    tau: np.ndarray            # (N,) time or residence time [s], strictly increasing
    T: np.ndarray              # (N,) temperature [K]
    P: np.ndarray              # (N,) pressure [Pa]
    Y: np.ndarray              # (N, Ns) mass fractions
    end_index: int             # integrate from index 0 to this index (inclusive)
    label: str = ""
    meta: dict = field(default_factory=dict)

    def validate(self, Ns: int) -> None:
        self.tau = np.asarray(self.tau, dtype=float).reshape(-1)
        self.T = np.asarray(self.T, dtype=float).reshape(-1)
        self.P = np.asarray(self.P, dtype=float).reshape(-1)
        self.Y = np.asarray(self.Y, dtype=float)
        N = self.tau.shape[0]
        name = self.label or "trajectory"
        if self.Y.ndim != 2 or self.Y.shape[0] != N:
            raise ValueError(f"{name}: Y must have shape (N, Ns) with N={N}, got {self.Y.shape}.")
        if self.Y.shape[1] != Ns:
            raise ValueError(f"{name}: Y has {self.Y.shape[1]} species, the mechanism has {Ns}.")
        if self.T.shape[0] != N or self.P.shape[0] != N:
            raise ValueError(f"{name}: T and P must have shape ({N},).")
        if not float(self.end_index).is_integer():
            raise ValueError(f"{name}: end_index must be an integer, got {self.end_index!r}.")
        if not (0 <= int(self.end_index) < N):
            raise ValueError(f"{name}: end_index {self.end_index} is outside [0, {N - 1}].")
        self.end_index = int(self.end_index)
        used = slice(0, self.end_index + 1)
        for arr, what in ((self.tau[used], "tau"), (self.T[used], "T"), (self.P[used], "P"), (self.Y[used], "Y")):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name}: {what} contains non-finite values before the end index.")
        if np.any(np.diff(self.tau[used]) <= 0.0):
            raise ValueError(f"{name}: tau must be strictly increasing up to the end index.")
        if np.any(self.T[used] <= 0.0) or np.any(self.P[used] <= 0.0):
            raise ValueError(f"{name}: T and P must be positive.")
        if np.any(np.clip(self.Y[used], 0.0, None).max(axis=1) <= 0.0):
            raise ValueError(f"{name}: a mass-fraction row before the end index has no positive entry.")
        y_min = float(self.Y[used].min())
        if y_min < -1e-6:
            log.warning("%s: mass fractions as low as %.3e are clipped to zero.", name, y_min)


def residence_time(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Residence time ``tau`` along a 1-D profile from ``dtau = dx / u``.

    Trapezoidal rule on ``1/u`` between grid points; ``tau[0] = 0``.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    if x.shape != u.shape:
        raise ValueError("x and u must have the same shape.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(u)):
        raise ValueError("x and u must be finite.")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("x must be strictly increasing.")
    if np.any(u <= 0.0):
        raise ValueError("u must be positive everywhere (flame-fixed frame, inlet to outlet).")
    dtau = 0.5 * np.diff(x) * (1.0 / u[:-1] + 1.0 / u[1:])
    return np.concatenate([[0.0], np.cumsum(dtau)])


def flame_tip_index(T: np.ndarray, fraction: float | None) -> int:
    """First index where ``T`` reaches ``fraction`` of its maximum; the last index if ``fraction`` is None.

    Raises ``ValueError`` when the first point already satisfies the rule
    (flat or decreasing profile, or a fraction at or below ``T[0] / T_max``),
    because such a profile is not a flame from the unburnt inlet to the tip.
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    if T.shape[0] <= 1:
        return 0
    if fraction is None:
        return int(T.shape[0] - 1)
    T_max = float(T.max())
    hits = np.flatnonzero(T >= fraction * T_max)
    idx = int(hits[0])
    if idx == 0:
        raise ValueError(
            f"The temperature reaches {fraction:.3g} of its maximum at the first point already "
            f"(T[0]={T[0]:.1f} K, T_max={T_max:.1f} K): the profile is flat or decreasing, or the "
            "fraction is too low for this flame."
        )
    return idx
