"""0-D reactor trajectories for the ignition-delay-time (idt) mode.

For every combination of initial temperature, pressure and equivalence ratio
a closed homogeneous reactor is integrated with Cantera. The states at every
solver step are recorded. The ignition sample is the end index of the
sensitivity integration:

* ``reference_peak``: the sample where the mole fraction of the reference
  species (default OH) is largest;
* ``max_dTdt``: the sample where dT/dt is largest.

If the ignition sample is the last sample, or the temperature rise is below
100 K, no ignition is detected. The trajectory is then reported with
``ignited = False`` and, with ``skip_unignited``, left out.
"""
from __future__ import annotations

import itertools
import logging
import time

import cantera as ct
import numpy as np

from ..sensitivity import Trajectory

log = logging.getLogger(__name__)

MIN_TEMPERATURE_RISE_K = 100.0


def idt_conditions(solver_cfg: dict) -> list[tuple[float, float, float]]:
    """All (T [K], P [Pa], phi) combinations: T outer, then P, then phi."""
    return [
        (float(T), float(P_atm) * ct.one_atm, float(phi))
        for T, P_atm, phi in itertools.product(solver_cfg["T_K"], solver_cfg["P_atm"], solver_cfg["phi"])
    ]


def _make_reactor(gas: ct.Solution, kind: str):
    cls = ct.IdealGasMoleReactor if kind == "constant_volume" else ct.IdealGasConstPressureMoleReactor
    try:
        reactor = cls(gas, clone=False)
    except TypeError:
        reactor = cls(gas)
    return reactor


def _phase_of(reactor):
    phase = getattr(reactor, "phase", None)
    return phase if phase is not None else reactor.thermo


def run_idt(
    mech_file: str,
    phase_name: str | None,
    T0: float,
    P0: float,
    phi: float,
    solver_cfg: dict,
    label: str = "",
) -> Trajectory:
    """Integrate one reactor and return its trajectory with the ignition sample as ``end_index``."""
    gas = ct.Solution(mech_file) if phase_name is None else ct.Solution(mech_file, phase_name)
    gas.TP = T0, P0
    gas.set_equivalence_ratio(phi, solver_cfg["fuel"], solver_cfg["oxidizer"])
    reactor = _make_reactor(gas, solver_cfg["reactor"])
    phase = _phase_of(reactor)
    net = ct.ReactorNet([reactor])
    try:
        net.preconditioner = ct.AdaptivePreconditioner()
    except Exception as exc:  # pragma: no cover - depends on the Cantera build
        log.debug("Adaptive preconditioner not used: %s", exc)
    net.rtol = float(solver_cfg["rtol"])
    net.atol = float(solver_cfg["atol"])

    ref = solver_cfg["reference_species"]
    if ref not in gas.species_names:
        raise ValueError(f"Reference species '{ref}' is not in the mechanism.")
    ref_index = gas.species_index(ref)

    end_time = float(solver_cfg["end_time_s"])
    max_steps = int(solver_cfg["max_steps"])
    t_hist = [0.0]
    T_hist = [float(phase.T)]
    P_hist = [float(phase.P)]
    Y_hist = [np.array(phase.Y, dtype=float)]
    ref_hist = [float(phase.X[ref_index])]

    t_start = time.perf_counter()
    n = 0
    while net.time < end_time and n < max_steps:
        t = net.step()
        n += 1
        Y = np.clip(np.array(phase.Y, dtype=float), 0.0, None)
        s = Y.sum()
        if s > 0.0:
            Y /= s
        t_hist.append(float(t))
        T_hist.append(float(phase.T))
        P_hist.append(float(phase.P))
        Y_hist.append(Y)
        ref_hist.append(float(phase.X[ref_index]))
    if n >= max_steps:
        log.warning("%s: reached max_steps=%d at t=%.3e s before end_time=%.3e s.", label, max_steps, net.time, end_time)

    t_arr = np.asarray(t_hist)
    T_arr = np.asarray(T_hist)
    if solver_cfg["ignition_criterion"] == "max_dTdt":
        dTdt = np.gradient(T_arr, t_arr, edge_order=1) if t_arr.shape[0] > 2 else np.zeros_like(T_arr)
        i_ign = int(np.argmax(dTdt))
    else:
        i_ign = int(np.argmax(np.asarray(ref_hist)))
    T_rise = float(T_arr.max() - T_arr[0])
    if i_ign == 0 and T_rise >= MIN_TEMPERATURE_RISE_K and solver_cfg["ignition_criterion"] == "reference_peak":
        raise ValueError(
            f"{label}: the reactor ignited (temperature rise {T_rise:.0f} K) but the reference species "
            f"'{ref}' peaks at the first sample; it is a reactant or a diluent. Choose a radical such as OH "
            "as reference_species, or set ignition_criterion to 'max_dTdt'."
        )
    ignited = (i_ign < t_arr.shape[0] - 1) and (T_rise >= MIN_TEMPERATURE_RISE_K)
    tau_ign = float(t_arr[i_ign])
    if not ignited:
        log.warning("%s: no ignition detected (ignition sample %d of %d, temperature rise %.1f K).",
                    label, i_ign, t_arr.shape[0], T_rise)
    else:
        log.info("%s: T0=%.0f K, P0=%.2f atm, phi=%.2f, IDT=%.4e s, %d samples, %.1f s",
                 label, T0, P0 / ct.one_atm, phi, tau_ign, t_arr.shape[0], time.perf_counter() - t_start)
    return Trajectory(
        tau=t_arr, T=T_arr, P=np.asarray(P_hist), Y=np.vstack(Y_hist), end_index=i_ign, label=label,
        meta={"source": "solver", "T0_K": T0, "P0_Pa": P0, "P0_atm": P0 / ct.one_atm, "phi": phi,
              "reactor": solver_cfg["reactor"], "tau_ign_s": tau_ign, "ignited": bool(ignited),
              "n_samples": int(t_arr.shape[0]), "temperature_rise_K": T_rise},
    )


def run_idt_set(mech_file: str, phase_name: str | None, solver_cfg: dict) -> list[Trajectory]:
    """Run every condition of the idt solver block."""
    conditions = idt_conditions(solver_cfg)
    log.info("idt solver: %d conditions (%s reactor, end time %.3e s).",
             len(conditions), solver_cfg["reactor"], solver_cfg["end_time_s"])
    trajs = []
    for m, (T0, P0, phi) in enumerate(conditions):
        traj = run_idt(mech_file, phase_name, T0, P0, phi, solver_cfg, label=f"idt-{m}")
        if not traj.meta["ignited"] and solver_cfg["skip_unignited"]:
            log.warning("%s is skipped (skip_unignited is true).", traj.label)
            continue
        trajs.append(traj)
    if not trajs:
        raise RuntimeError("The idt solver produced no usable trajectory.")
    return trajs
