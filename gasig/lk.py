"""Linearised knockout (LK): the effect of the removal of each species along the idt trajectories.

The removal of the species b deletes the set G of its reactions (the rule of
:func:`gasig.reduce.reduced_solution`). The source term of the mechanism
without b is

    f_-G(Y) = f(Y) - (W / rho) nu_G R_G(Y).

Linearised in the state around the stored trajectory, but exact in the
removal (the rates of G are set to zero, not scaled), the change of the
mass fractions Delta_G = Y_-G - Y solves

    dDelta_G/dt = J1^(-G) Delta_G - (W / rho) nu_G R_G,   Delta_G(0) = 0,
    J1^(-G) = J1 - U_G V_G^T,
    U_G = [diag(W / rho) nu_G, W (nu_G R_G)],   V_G^T = [dR_G/dY; d(1/rho)/dY],

with J1 = d(dY/dt)/dY = diag(W / rho) nu dR/dY + W (nu R) d(1/rho)/dY^T. The
GA-SiG estimate of the same removal, -sum_{j in G} S_j, solves the equation
with J1 in place of J1^(-G); it is linear in the removal and misses a species
whose effect appears only at zero rate.

Integration: backward Euler on the stored grid, with the Jacobians at the end
of each interval, without the sub-step guard. The step matrix of a group,
A_G = A + h U_G V_G^T with A = I - h J1, differs from A by a matrix of rank
|G| + 1, so the Woodbury identity solves it with the one LU factorisation of
A per step:

    A_G^-1 x = y - (A^-1 U)(I / h + V^T A^-1 U)^-1 V^T y,   y = A^-1 x.

Score: at the stored sample s nearest to ``sample_fraction`` of the end time
t_end, the time-shift estimate of the removal of b is

    v_b = -(dY/dt . Delta_b) / (|dY/dt|^2 t_end),

with dY/dt from Cantera at the stored state and the row of b set to zero in
Delta_b (the removed species itself does not count). The LK score is
L_b = max over the screened trajectories of |v_b|. The LK order sorts the
species by ascending L_b; ties keep the GA-SiG order. The integration runs
on the CPU in float64, whatever the ``device`` key says.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import cantera as ct
import numpy as np
import torch

from .errors import ConfigError, InputError, IntegrationError, MechanismError, MissingInputError
from .jacobian import jacobians
from .kinetics import TorchKinetics
from .reduce import reaction_kept, reaction_requirements, read_importance, read_rank
from .trajectory import Trajectory

log = logging.getLogger("gasig.lk")


def removal_sets(gas: ct.Solution, groups: list[list[str]]) -> list[list[int]]:
    """Indices of the reactions that the removal of each group deletes."""
    names = set(gas.species_names)
    req = reaction_requirements(gas)
    if not all(reaction_kept(r, names) for r in req):
        raise MechanismError("the full mechanism already drops reactions under the removal rule; the rule does "
                             "not match this mechanism.", stage="lk")
    sets = []
    for grp in groups:
        keep = names - set(grp)
        sets.append([j for j, r in enumerate(req) if not reaction_kept(r, keep)])
    return sets


def sample_index(traj: Trajectory, fraction: float) -> int:
    """The stored sample nearest to ``fraction`` of the end time, at most the end index."""
    end = int(traj.end_index)
    t_end = float(traj.tau[end])
    if end < 1 or t_end <= 0.0:
        raise InputError(f"{traj.label}: the trajectory has no interval before its end index; the LK needs one.",
                         stage="lk", label=traj.label, end_index=end)
    idx = int(np.argmin(np.abs(np.asarray(traj.tau[:end + 1]) / t_end - fraction)))
    if idx < 1:
        raise InputError(f"{traj.label}: the sample nearest to t/t_end = {fraction:g} is the initial state; the LK "
                         "needs at least one interval. Increase 'lk.sample_fraction' or store a finer trajectory.",
                         stage="lk", label=traj.label, sample_fraction=fraction)
    return idx


def lk_trajectory(kin: TorchKinetics, traj: Trajectory, sets: list[list[int]], n_steps: int,
                  with_ga: bool = False, check_steps: set[int] | None = None, check_groups: list[int] | None = None,
                  progress_every: int = 0) -> dict:
    """Integrate the LK of every group up to the sample ``n_steps``.

    Returns ``Dlk`` (Ns, groups) at that sample, ``Dga`` (the GA-SiG estimate)
    when ``with_ga``, and ``checks``: at the steps of ``check_steps`` the
    relative deviation of the decomposed J1 from :func:`gasig.jacobian.jacobians`
    and, for the groups of ``check_groups``, of the Woodbury update from a
    direct solve.
    """
    if kin.device.type != "cpu" or kin.dtype != torch.float64:
        raise ConfigError("the LK integration needs TorchKinetics on the CPU in float64.", stage="lk")
    traj.validate(kin.Ns)
    Ns, NG = kin.Ns, len(sets)
    nu = kin.nu_net_np
    Wt = kin.W_np
    R_u = kin.R_u
    Dlk = np.zeros((Ns, NG))
    Dga = np.zeros((Ns, NG)) if with_ga else None
    eye = torch.eye(Ns, dtype=torch.float64)
    check_steps = check_steps or set()
    check_groups = list(check_groups or [])
    checks = {"J1_decomposition_rel": [], "woodbury_vs_direct_rel": []}
    t0 = time.perf_counter()
    for k in range(n_steps):
        h = float(traj.tau[k + 1] - traj.tau[k])
        Tk, Pk = float(traj.T[k + 1]), float(traj.P[k + 1])
        Yk = np.clip(np.asarray(traj.Y[k + 1], dtype=float), 0.0, None)
        Yk = Yk / Yk.sum()
        kf, kr = kin.rate_constants(Tk, Pk, Yk)
        Yt = torch.tensor(Yk, dtype=torch.float64)
        D = torch.func.jacfwd(lambda y: kin.rhs(y, Tk, Pk, kf, kr)[1])(Yt).detach().numpy()   # dR/dY (Nr, Ns)
        with torch.no_grad():
            R = kin.rhs(Yt, Tk, Pk, kf, kr)[1].numpy()
            rho = float(kin.density(Yt, Tk, Pk))
        g = (R_u * Tk / Pk) / Wt                                    # d(1/rho)/dY
        WnuR = (Wt / rho)[:, None] * nu                             # diag(W/rho) nu
        J1 = WnuR @ D + np.outer(Wt * (nu @ R), g)
        if (k + 1) in check_steps:
            J1_ref = jacobians(kin, Yt, Tk, Pk, kf, kr)[0].numpy()
            checks["J1_decomposition_rel"].append(
                {"step": k + 1, "rel": float(np.abs(J1 - J1_ref).max() / max(np.abs(J1_ref).max(), 1e-300))})
        A = eye - h * torch.from_numpy(J1)
        LU, piv = torch.linalg.lu_factor(A)
        Z = torch.linalg.lu_solve(LU, piv, torch.from_numpy(WnuR)).numpy()     # A^-1 diag(W/rho) nu
        Ylk = torch.linalg.lu_solve(LU, piv, torch.from_numpy(Dlk)).numpy()
        Yga = torch.linalg.lu_solve(LU, piv, torch.from_numpy(Dga)).numpy() if with_ga else None
        new_lk = np.empty_like(Dlk)
        new_ga = np.empty_like(Dga) if with_ga else None
        for gi, G in enumerate(sets):
            if not G:
                new_lk[:, gi] = Ylk[:, gi]
                if with_ga:
                    new_ga[:, gi] = Yga[:, gi]
                continue
            ZG = Z[:, G]
            ZR = ZG @ R[G]
            if with_ga:
                new_ga[:, gi] = Yga[:, gi] - h * ZR
            yb = Ylk[:, gi] - h * ZR
            AU = np.empty((Ns, len(G) + 1))
            AU[:, :-1] = ZG
            AU[:, -1] = rho * ZR                                    # A^-1 W (nu_G R_G)
            VT = np.empty((len(G) + 1, Ns))
            VT[:-1] = D[G]
            VT[-1] = g
            K = VT @ AU
            K[np.diag_indices_from(K)] += 1.0 / h
            new_lk[:, gi] = yb - AU @ np.linalg.solve(K, VT @ yb)
            if (k + 1) in check_steps and gi in check_groups:
                U = np.empty((Ns, len(G) + 1))
                U[:, :-1] = WnuR[:, G]
                U[:, -1] = Wt * (nu[:, G] @ R[G])
                direct = np.linalg.solve(A.numpy() + h * (U @ VT), Dlk[:, gi] - h * (WnuR[:, G] @ R[G]))
                scale = max(np.abs(direct).max(), 1e-300)
                checks["woodbury_vs_direct_rel"].append(
                    {"step": k + 1, "group": gi, "rel": float(np.abs(direct - new_lk[:, gi]).max() / scale)})
        Dlk = new_lk
        if with_ga:
            Dga = new_ga
        if not np.all(np.isfinite(Dlk)):
            raise IntegrationError(f"{traj.label}: non-finite linearised knockout at step {k + 1} "
                                   f"(tau={traj.tau[k + 1]:.6e} s).", stage="lk", label=traj.label, step=k + 1,
                                   tau_s=float(traj.tau[k + 1]))
        if progress_every and (k + 1) % progress_every == 0:
            el = time.perf_counter() - t0
            log.info("%s: LK step %d/%d, T=%.1f K, %.0f s (%.2f s/step)", traj.label, k + 1, n_steps, Tk, el,
                     el / (k + 1))
    return {"Dlk": Dlk, "Dga": Dga, "checks": checks, "seconds": time.perf_counter() - t0}


def time_shift(gas: ct.Solution, traj: Trajectory, sample: int, Delta: np.ndarray) -> np.ndarray:
    """The time-shift estimate v_b of every single-species group b; ``Delta`` is (Ns, Ns)."""
    Y = np.clip(np.asarray(traj.Y[sample], dtype=float), 0.0, None)
    gas.TPY = float(traj.T[sample]), float(traj.P[sample]), Y / Y.sum()
    yd = gas.net_production_rates * gas.molecular_weights / gas.density
    D = np.array(Delta, dtype=float, copy=True)
    D[np.arange(D.shape[0]), np.arange(D.shape[0])] = 0.0           # the row of the removed species
    t_end = float(traj.tau[traj.end_index])
    denom = float(yd @ yd)
    if denom <= 0.0:
        raise InputError(f"{traj.label}: dY/dt is zero at sample {sample}; the time-shift estimate is undefined.",
                         stage="lk", label=traj.label, sample=sample)
    return -(yd @ D) / denom / t_end


def lk_order(names: list[str], scores: np.ndarray, gasig_rank: list[str]) -> list[str]:
    """Species by ascending LK score; ties keep the GA-SiG order (least important first)."""
    pos = {s: i for i, s in enumerate(gasig_rank)}
    score = dict(zip(names, scores))
    return sorted(names, key=lambda s: (score[s], pos[s]))


def run_lk(cfg: dict, out_dir: Path, trajs: list[Trajectory]) -> dict:
    """The LK stage: scores on the selected trajectories, ``lk.output_file`` and ``lk_scores.npy``."""
    lcfg = cfg["lk"]
    t0 = time.perf_counter()
    rank_path = out_dir / "ranked_species.csv"
    if not rank_path.is_file():
        raise MissingInputError(f"{rank_path} is missing; the LK order needs the GA-SiG rank for its ties. "
                                "Run the rank stage first.", stage="lk", path=str(rank_path))
    gasig_rank = read_rank(rank_path)
    gasig_importance = read_importance(rank_path)
    selected = lcfg["trajectories"] if lcfg["trajectories"] is not None else list(range(len(trajs)))
    bad = [m for m in selected if m >= len(trajs)]
    if bad:
        raise ConfigError(f"'lk.trajectories' names trajectories {bad} but there are {len(trajs)} "
                          f"(indices 0 to {len(trajs) - 1}).", stage="lk")
    kin = TorchKinetics(cfg["mechanism"], cfg["phase"], device="cpu", dtype=torch.float64,
                        concentration_floor=cfg["sensitivity"]["concentration_floor"],
                        rate_law=lcfg["rate_law"], self_check=cfg["sensitivity"]["self_check"])
    names = kin.species_names
    if sorted(gasig_rank) != sorted(names):
        raise MechanismError(f"{rank_path} does not list the species of the mechanism; run the rank stage with "
                             "this mechanism.", stage="lk", path=str(rank_path))
    gas = ct.Solution(cfg["mechanism"]) if cfg["phase"] is None else ct.Solution(cfg["mechanism"], cfg["phase"])
    sets = removal_sets(gas, [[s] for s in names])
    sizes = np.array([len(G) for G in sets])
    log.info("[lk] %d single-species knockouts (median %d reactions, largest %d for %s) on %d trajectories, "
             "rate law %s, score at t/t_end = %.3g", len(names), int(np.median(sizes)), int(sizes.max()),
             names[int(sizes.argmax())], len(selected), lcfg["rate_law"], lcfg["sample_fraction"])
    per_traj = np.zeros((len(selected), len(names)))
    records = []
    progress = int(cfg["sensitivity"]["progress_every"])
    for row, m in enumerate(selected):
        traj = trajs[m]
        traj.validate(kin.Ns)
        s = sample_index(traj, float(lcfg["sample_fraction"]))
        res = lk_trajectory(kin, traj, sets, s, check_steps={1, s}, progress_every=progress)
        per_traj[row] = time_shift(gas, traj, s, res["Dlk"])
        t_end = float(traj.tau[traj.end_index])
        j1 = max((c["rel"] for c in res["checks"]["J1_decomposition_rel"]), default=None)
        records.append({"trajectory": m, "label": traj.label, "sample": s, "t_over_tend": float(traj.tau[s] / t_end),
                        "T_K": float(traj.T[s]), "n_steps": s, "seconds": round(res["seconds"], 1),
                        "J1_decomposition_rel_max": j1, **{k: v for k, v in traj.meta.items()
                                                         if k in ("T0_K", "P0_atm", "phi")}})
        log.info("[lk] %s: %d steps to t/t_end=%.3f in %.1f s; J1 decomposition check %.2e", traj.label, s,
                 traj.tau[s] / t_end, res["seconds"], j1 if j1 is not None else float("nan"))
    scores = np.max(np.abs(per_traj), axis=0)
    order = lk_order(names, scores, gasig_rank)
    score = dict(zip(names, scores))
    out_path = out_dir / lcfg["output_file"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank_from_least_important", "species", "importance", "gasig_importance"])
        for i, sp in enumerate(order, start=1):
            writer.writerow([i, sp, f"{score[sp]:.17g}", f"{gasig_importance.get(sp, 0.0):.10e}"])
    np.save(out_dir / "lk_scores.npy", per_traj)
    seconds = time.perf_counter() - t0
    log.info("[lk] order written to %s (%d species with score 0), %.1f s", out_path, int(np.sum(scores == 0.0)),
             seconds)
    return {"rate_law": lcfg["rate_law"], "sample_fraction": float(lcfg["sample_fraction"]),
            "trajectories": records, "n_zero_score": int(np.sum(scores == 0.0)),
            "files": {"order": str(out_path), "scores": str(out_dir / "lk_scores.npy")},
            "seconds": round(seconds, 2)}
