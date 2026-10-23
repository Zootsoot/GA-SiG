"""Implicit-Euler integration of the forward sensitivity equation.

The sensitivity ``S = dA/dtheta`` (Ns, Nr) obeys ``dS/dtau = J1 S + J2`` with
``S(0) = 0``, where ``tau`` is time (idt mode) or residence time along a
flame (fs mode). Both modes are one :class:`Trajectory`: a strictly
increasing ``tau`` grid with the thermo-chemical state at every point and an
end index. The integrator evaluates ``J1`` and ``J2`` at the end of each
interval (backward Euler) and solves ``(I - dtau J1) S_new = S + dtau J2``.

Step-size guard
---------------
Backward Euler is stable for decaying modes, but when ``dtau`` times a
positive eigenvalue of ``J1`` exceeds 1 the growing mode changes sign. The
guard compares one step with two half steps, both with the same frozen
Jacobians, and keeps halving the sub-step until the two agree within
``substep_tol`` for every species row (2-norm over the reactions, relative
to the larger of the row norm and the row increment, with a floor of 1e-12
of the Frobenius norm of the whole matrix) or ``max_substeps`` is reached.
``substep_tol=None`` disables the guard and reproduces version 0.1.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .jacobian import jacobians
from .kinetics import TorchKinetics

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


@dataclass
class StepStats:
    """Counters for one integrated trajectory."""

    n_steps: int = 0
    n_solves: int = 0
    max_substeps: int = 1
    n_guard_limit: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "n_solves": self.n_solves,
            "max_substeps": self.max_substeps,
            "n_guard_limit": self.n_guard_limit,
            "seconds": round(self.seconds, 3),
        }


def rows_agree(S_next: torch.Tensor, S_prev: torch.Tensor, S_old: torch.Tensor, tol: float) -> bool:
    """Per-species-row agreement of two sub-step levels.

    Row ``a`` agrees when ``||S_next[a] - S_prev[a]|| <= tol * scale[a]`` with
    ``scale[a] = max(||S_next[a]||, ||S_next[a] - S_old[a]||, 1e-12 ||S_next||_F)``.
    The increment term keeps the test meaningful for rows that start near
    zero; the floor skips rows that are numerically negligible.
    """
    diff = torch.linalg.norm(S_next - S_prev, dim=1)
    scale = torch.maximum(torch.linalg.norm(S_next, dim=1), torch.linalg.norm(S_next - S_old, dim=1))
    scale = torch.maximum(scale, 1e-12 * torch.linalg.norm(S_next))
    return bool(torch.all(diff <= tol * scale).item())


def implicit_euler_step(
    S: torch.Tensor,
    J1: torch.Tensor,
    J2: torch.Tensor,
    dtau: float,
    I: torch.Tensor,
    substep_tol: float | None,
    max_substeps: int,
) -> tuple[torch.Tensor, int, int, bool]:
    """Advance ``S`` over one interval. Returns ``(S_new, n_substeps, n_solves, converged)``.

    With the guard on, the sub-step count doubles until two levels agree
    (:func:`rows_agree`) or the next level would exceed ``max_substeps``;
    then the finest computed level is returned with ``converged=False``.
    """

    def advance(n: int) -> torch.Tensor:
        h = dtau / n
        LU, piv = torch.linalg.lu_factor(I - h * J1)
        X = S
        for _ in range(n):
            X = torch.linalg.lu_solve(LU, piv, X + h * J2)
        return X

    S_prev = advance(1)
    n_solves = 1
    if substep_tol is None:
        return S_prev, 1, n_solves, True

    n = 1
    while True:
        if 2 * n > max_substeps:
            return S_prev, n, n_solves, False
        S_next = advance(2 * n)
        n_solves += 2 * n
        if rows_agree(S_next, S_prev, S, substep_tol):
            return S_next, 2 * n, n_solves, True
        n *= 2
        S_prev = S_next


def integrate_trajectory(
    kin: TorchKinetics,
    traj: Trajectory,
    substep_tol: float | None = 1e-2,
    max_substeps: int = 64,
    progress_every: int = 50,
) -> tuple[np.ndarray, StepStats]:
    """Integrate the sensitivity equation along ``traj`` and return ``S`` at the end index."""
    traj.validate(kin.Ns)
    device, dtype = kin.device, kin.dtype
    S = torch.zeros((kin.Ns, kin.Nr), dtype=dtype, device=device)
    I = torch.eye(kin.Ns, dtype=dtype, device=device)
    stats = StepStats()
    name = traj.label or "trajectory"
    t0 = time.perf_counter()

    for k in range(traj.end_index):
        dtau = float(traj.tau[k + 1] - traj.tau[k])
        T = float(traj.T[k + 1])
        P = float(traj.P[k + 1])
        # Clip negative mass fractions and renormalize, as Cantera does when it takes the state;
        # dY/dt and J2 scale with sum(Y), so the data must sum to 1.
        Y_np = np.clip(traj.Y[k + 1], 0.0, None)
        y_sum = float(Y_np.sum())
        if y_sum > 0.0:
            Y_np = Y_np / y_sum
        kf, kr = kin.rate_constants(T, P, Y_np)
        Y_t = torch.tensor(Y_np, dtype=dtype, device=device)
        J1, J2 = jacobians(kin, Y_t, T, P, kf, kr)
        S, n_sub, n_solves, converged = implicit_euler_step(S, J1, J2, dtau, I, substep_tol, max_substeps)
        stats.n_steps += 1
        stats.n_solves += n_solves
        stats.max_substeps = max(stats.max_substeps, n_sub)
        if not converged:
            stats.n_guard_limit += 1
            log.warning("%s: step %d reached max_substeps=%d without convergence (dtau=%.3e s).",
                        name, k + 1, max_substeps, dtau)
        if not torch.isfinite(S).all():
            raise RuntimeError(f"{name}: non-finite sensitivities at step {k + 1} (tau={traj.tau[k + 1]:.6e} s).")
        if progress_every and (k + 1) % progress_every == 0:
            log.info("%s: step %d/%d, tau=%.4e s, T=%.1f K, substeps=%d",
                     name, k + 1, traj.end_index, traj.tau[k + 1], T, n_sub)

    stats.seconds = time.perf_counter() - t0
    log.info("%s: done, %d steps, %d solves, max substeps %d, %.1f s",
             name, stats.n_steps, stats.n_solves, stats.max_substeps, stats.seconds)
    return S.detach().cpu().numpy(), stats


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
