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
from dataclasses import dataclass

import numpy as np
import torch

from .jacobian import jacobians
from .kinetics import TorchKinetics
from .trajectory import Trajectory, flame_tip_index, residence_time  # noqa: F401  (re-exported)

log = logging.getLogger(__name__)


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
