"""Read precomputed trajectories and profiles; write run data.

Precomputed format (the format of the ``example`` folder)
--------------------------------------------------------
Every quantity is one ``.npy`` file that holds a Python dictionary keyed by
the trajectory number ``0 .. N-1``:

* idt mode: ``time`` {m: (Nt,)} [s], ``T`` {m: (Nt,)} [K], ``P`` {m: (Nt,)} [Pa],
  ``Y`` {m: (Nt, Ns)}, and ``index`` as a plain integer array (N,) with the
  end index of each trajectory (the ignition sample).
* fs mode: ``position`` {m: (Nx,)} [m], ``velocity`` {m: (Nx,)} [m/s],
  ``T``, ``P``, ``Y`` as above, and an optional ``index`` array with the
  flame-tip sample of each profile.

The solvers of this package write the same files, so a solver run can be
reused later as precomputed input.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .trajectory import Trajectory, flame_tip_index, residence_time

log = logging.getLogger(__name__)


def load_dict_npy(path: str | Path) -> dict[int, np.ndarray]:
    """Load a ``.npy`` file that holds ``{int: array}`` and return it keyed ``0..N-1``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    arr = np.load(path, allow_pickle=True)
    obj = arr.item() if (arr.dtype == object and arr.shape == ()) else arr
    if isinstance(obj, dict):
        items = obj
    elif isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.dtype == object):
        items = {i: v for i, v in enumerate(obj)}
    else:
        raise ValueError(f"{path}: expected a dictionary {{trajectory: array}}, got {type(obj).__name__}.")
    out: dict[int, np.ndarray] = {}
    for key, value in items.items():
        out[int(key)] = np.asarray(value, dtype=float)
    expected = list(range(len(out)))
    if sorted(out) != expected:
        raise ValueError(f"{path}: dictionary keys must be 0..{len(out) - 1}, got {sorted(out)[:10]}...")
    return out


def load_index_npy(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Index file not found: {path}")
    idx = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
    if not np.issubdtype(idx.dtype, np.integer):
        if np.all(np.mod(idx, 1) == 0):
            idx = idx.astype(int)
        else:
            raise ValueError(f"{path}: the index array must hold integers.")
    return idx.astype(int)


def _check_same_keys(named: dict[str, dict[int, np.ndarray]]) -> int:
    lengths = {name: len(d) for name, d in named.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"The data files hold different numbers of trajectories: {lengths}.")
    return next(iter(lengths.values()))


def load_idt_precomputed(pre: dict) -> list[Trajectory]:
    """Build the idt trajectories from the ``idt.precomputed`` configuration block."""
    time_all = load_dict_npy(pre["time"])
    T_all = load_dict_npy(pre["T"])
    P_all = load_dict_npy(pre["P"])
    Y_all = load_dict_npy(pre["Y"])
    n = _check_same_keys({"time": time_all, "T": T_all, "P": P_all, "Y": Y_all})
    idx = load_index_npy(pre["index"])
    if idx.shape[0] != n:
        raise ValueError(f"The index file has {idx.shape[0]} entries but there are {n} trajectories.")
    trajs = []
    for m in range(n):
        end = int(idx[m])
        n_pts = int(time_all[m].shape[0])
        if not (0 <= end < n_pts):
            raise ValueError(f"idt-{m}: end index {end} from {pre['index']} is outside [0, {n_pts - 1}].")
        trajs.append(Trajectory(
            tau=time_all[m], T=T_all[m], P=P_all[m], Y=Y_all[m],
            end_index=end, label=f"idt-{m}",
            meta={"source": "precomputed", "end_time_s": float(time_all[m][end])},
        ))
    log.info("Loaded %d precomputed idt trajectories.", n)
    return trajs


def load_fs_precomputed(pre: dict, flame_tip_T_fraction: float | None) -> list[Trajectory]:
    """Build the fs profiles (in residence time) from the ``fs.precomputed`` block."""
    x_all = load_dict_npy(pre["position"])
    u_all = load_dict_npy(pre["velocity"])
    T_all = load_dict_npy(pre["T"])
    P_all = load_dict_npy(pre["P"])
    Y_all = load_dict_npy(pre["Y"])
    n = _check_same_keys({"position": x_all, "velocity": u_all, "T": T_all, "P": P_all, "Y": Y_all})
    idx = load_index_npy(pre["index"]) if pre.get("index") else None
    if idx is not None and idx.shape[0] != n:
        raise ValueError(f"The index file has {idx.shape[0]} entries but there are {n} profiles.")
    trajs = []
    for m in range(n):
        x, u, T = x_all[m], u_all[m], T_all[m]
        P = P_all[m]
        if P.ndim == 0 or P.shape[0] == 1:
            P = np.full(x.shape[0], float(P.reshape(-1)[0]))
        tau = residence_time(x, u)
        if idx is not None:
            end = int(idx[m])
            if not (0 <= end < x.shape[0]):
                raise ValueError(f"fs-{m}: end index {end} from {pre['index']} is outside [0, {x.shape[0] - 1}].")
        else:
            try:
                end = flame_tip_index(T, flame_tip_T_fraction)
            except ValueError as exc:
                raise ValueError(f"fs-{m}: {exc}") from exc
        trajs.append(Trajectory(
            tau=tau, T=T, P=P, Y=Y_all[m], end_index=end, label=f"fs-{m}",
            meta={"source": "precomputed", "x": x, "u": u, "flame_speed_m_s": float(u[0]),
                  "tip_index": end, "tip_position_m": float(x[end]), "n_points": int(x.shape[0])},
        ))
    log.info("Loaded %d precomputed fs profiles.", n)
    return trajs


def save_dict_npy(path: str | Path, data: dict[int, np.ndarray]) -> None:
    np.save(Path(path), np.array(data, dtype=object), allow_pickle=True)


def save_idt_data(folder: str | Path, trajs: list[Trajectory]) -> None:
    """Write idt trajectories in the precomputed format."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_dict_npy(folder / "time_all.npy", {m: t.tau for m, t in enumerate(trajs)})
    save_dict_npy(folder / "T_all.npy", {m: t.T for m, t in enumerate(trajs)})
    save_dict_npy(folder / "P_all.npy", {m: t.P for m, t in enumerate(trajs)})
    save_dict_npy(folder / "Y_all.npy", {m: t.Y for m, t in enumerate(trajs)})
    np.save(folder / "index_all.npy", np.array([t.end_index for t in trajs], dtype=int))


def save_fs_data(folder: str | Path, trajs: list[Trajectory]) -> None:
    """Write fs profiles in the precomputed format (position and velocity from ``meta``)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_dict_npy(folder / "position_all.npy", {m: t.meta["x"] for m, t in enumerate(trajs)})
    save_dict_npy(folder / "velocity_all.npy", {m: t.meta["u"] for m, t in enumerate(trajs)})
    save_dict_npy(folder / "T_all.npy", {m: t.T for m, t in enumerate(trajs)})
    save_dict_npy(folder / "P_all.npy", {m: t.P for m, t in enumerate(trajs)})
    save_dict_npy(folder / "Y_all.npy", {m: t.Y for m, t in enumerate(trajs)})
    np.save(folder / "index_all.npy", np.array([t.end_index for t in trajs], dtype=int))
