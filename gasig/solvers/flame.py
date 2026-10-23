"""1-D freely propagating flames for the flame-speed (fs) mode.

For every combination of pressure and equivalence ratio at the unburnt
temperature, a ``FreeFlame`` is solved with Cantera. The profile of
position, velocity, temperature, pressure and mass fractions is converted
to residence time ``tau`` with ``dtau = dx / u`` and the sensitivity
integration ends at the flame tip, the first grid point where the
temperature reaches ``flame_tip_T_fraction`` of the maximum temperature.
"""
from __future__ import annotations

import itertools
import logging
import time

import cantera as ct
import numpy as np

from ..sensitivity import Trajectory, flame_tip_index, residence_time

log = logging.getLogger(__name__)


def flame_conditions(solver_cfg: dict) -> list[tuple[float, float]]:
    """All (P [Pa], phi) combinations: P outer, phi inner."""
    return [(float(P_atm) * ct.one_atm, float(phi))
            for P_atm, phi in itertools.product(solver_cfg["P_atm"], solver_cfg["phi"])]


def run_flame(
    mech_file: str,
    phase_name: str | None,
    P: float,
    phi: float,
    solver_cfg: dict,
    label: str = "",
) -> Trajectory:
    """Solve one flame and return its profile in residence time."""
    gas = ct.Solution(mech_file) if phase_name is None else ct.Solution(mech_file, phase_name)
    T_u = float(solver_cfg["T_unburnt_K"])
    gas.TP = T_u, P
    gas.set_equivalence_ratio(phi, solver_cfg["fuel"], solver_cfg["oxidizer"])

    flame = ct.FreeFlame(gas, width=float(solver_cfg["domain_width_m"]))
    flame.transport_model = solver_cfg["transport_model"]
    refine = solver_cfg["refine"]
    flame.set_refine_criteria(ratio=float(refine["ratio"]), slope=float(refine["slope"]),
                              curve=float(refine["curve"]), prune=float(refine["prune"]))
    t_start = time.perf_counter()
    flame.solve(loglevel=int(solver_cfg["loglevel"]), auto=True)

    x = np.asarray(flame.grid, dtype=float)
    u = np.asarray(flame.velocity, dtype=float)
    T = np.asarray(flame.T, dtype=float)
    Y = np.asarray(flame.Y, dtype=float).T          # (Nx, Ns)
    P_arr = np.full(x.shape[0], float(flame.P))
    tau = residence_time(x, u)
    tip = flame_tip_index(T, solver_cfg["flame_tip_T_fraction"])
    S_L = float(u[0])
    log.info("%s: T_u=%.0f K, P=%.2f atm, phi=%.2f, S_L=%.4f m/s, %d points, tip at %d (x=%.4e m), %.1f s",
             label, T_u, P / ct.one_atm, phi, S_L, x.shape[0], tip, x[tip], time.perf_counter() - t_start)
    return Trajectory(
        tau=tau, T=T, P=P_arr, Y=Y, end_index=tip, label=label,
        meta={"source": "solver", "T_unburnt_K": T_u, "P_Pa": P, "P_atm": P / ct.one_atm, "phi": phi,
              "flame_speed_m_s": S_L, "n_points": int(x.shape[0]), "tip_index": tip,
              "tip_position_m": float(x[tip]), "T_max_K": float(T.max()), "x": x, "u": u},
    )


def run_flame_set(mech_file: str, phase_name: str | None, solver_cfg: dict) -> list[Trajectory]:
    """Run every condition of the fs solver block."""
    conditions = flame_conditions(solver_cfg)
    log.info("fs solver: %d flames (T_u=%.0f K, width %.3f m, %s transport).",
             len(conditions), solver_cfg["T_unburnt_K"], solver_cfg["domain_width_m"], solver_cfg["transport_model"])
    trajs = []
    for m, (P, phi) in enumerate(conditions):
        trajs.append(run_flame(mech_file, phase_name, P, phi, solver_cfg, label=f"fs-{m}"))
    return trajs
