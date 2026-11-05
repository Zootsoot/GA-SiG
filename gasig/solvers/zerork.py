"""Zero-RK backend of the flame-speed mode.

Zero-RK (Lawrence Livermore National Laboratory, BSD-3-Clause,
https://github.com/LLNL/zero-rk) solves one-dimensional laminar flames with
sparse, preconditioned iterative methods [Lapointe, Whitesides and McNenly,
Combust. Flame 204 (2019) 23-32]. With ``fs.solver.backend: "zerork"`` the
package drives its ``premixed_steady_flame_solver`` instead of the Cantera
``FreeFlame``, in the way of the ``flame_speed`` example of Zero-RK
(``flamesolver.py``, function ``compute_pseudounsteady_flame_speeds``):

1. the mechanism goes to CHEMKIN files with ``cantera.yaml2ck``; the species
   notes are removed first, because a note that starts with ``!`` breaks the
   fixed columns of the thermodynamic data;
2. ``lewisGenerator.x`` computes one Lewis number per species from the
   mixture-averaged transport of the inlet mixture at ``lewis_temperature_K``;
3. the flame starts on a coarse grid with the pseudo-unsteady term; then the
   grid is rebuilt around the flame and the steady problem is solved again,
   until the grid holds ``min_points_per_thickness`` points per flame thickness
   and the flame sits inside the fine part of the grid;
4. the flame speed comes from the ``Final`` line of the solver log and the
   profile from the binary data file of the last solve.

The steady solver of Zero-RK supports constant Lewis numbers only; the
mixture-averaged and multicomponent models of Cantera are not available with
this backend, and ``fs.solver.transport_model`` is ignored. Every solver call
is a subprocess (``mpiexec -n processes`` or the serial executable), so this
module imports Cantera and NumPy only.
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import math
import os
import secrets
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cantera as ct
import numpy as np

from ..errors import FlameSolverError, MechanismError, SolverTimeout  # noqa: F401  SolverTimeout re-exported here
from ..mechanism import write_yaml
from ..trajectory import Trajectory, flame_tip_index, residence_time

log = logging.getLogger(__name__)

DEFAULT_HOME = "/opt/zero-rk"
STEADY_SERIAL = "premixed_steady_flame_solver.x"
STEADY_MPI = "premixed_steady_flame_solver_mpi.x"
LEWIS_GENERATOR = "lewisGenerator.x"
REF_TEMPERATURE = 1000.0        # the solver stores T / ref_temperature in its data files
MAX_POINTS = 1024               # upper limit of grid points, as in flamesolver.py
START_EXTENT = 0.3              # the initial front at 30 percent of the domain; see solver_input
INPUT_FILE = "input.yml"
LEWIS_INPUT_FILE = "input_Le.yml"
LOG_FILE = "log"


@dataclass(frozen=True)
class ChemkinFiles:
    mechanism: Path
    thermo: Path
    transport: Path


CHEMKIN_CACHE_SIZE = 2          # the detailed mechanism and the current candidate
_CHEMKIN_CACHE: "OrderedDict[tuple, ChemkinFiles]" = OrderedDict()
_OWNED_FOLDERS: set[Path] = set()   # CHEMKIN folders this process created; removed on eviction and at exit
_RUN_NAMES: dict[int, str] = {}     # PID -> name of the run folder of that process in work_dir
_RUN_ROOTS: set[Path] = set()       # run folders this process created; removed at exit when they are empty


def run_folder_name() -> str:
    """The subfolder of ``work_dir`` for this process, ``run-<host>-<pid>-<token>``.

    The PID alone is not unique: every Docker container starts the package
    as PID 1. The host name (the container ID in Docker) and a random token
    make the name unique, so two runs that share ``work_dir`` do not delete
    each other's files. A forked child gets a new name.
    """
    pid = os.getpid()
    if pid not in _RUN_NAMES:
        host = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in socket.gethostname()) or "host"
        _RUN_NAMES[pid] = f"run-{host[:40]}-{pid}-{secrets.token_hex(4)}"
    return _RUN_NAMES[pid]


def _run_root(zcfg: dict) -> Path:
    """``<work_dir>/<run folder>``; the folder is created and recorded for the clean-up at exit."""
    root = Path(zcfg["work_dir"]) / run_folder_name()
    root.mkdir(parents=True, exist_ok=True)
    _RUN_ROOTS.add(root)
    return root


SEED_LIMIT = 32                 # solved flames kept in memory as initial guesses (continuation)
OTHER_MECHANISM_PENALTY = 0.1   # added to the condition distance of a seed from another mechanism
_SEEDS: list[dict] = []


def condition_distance(T_u: float, P: float, phi: float, seed: dict) -> float:
    """Distance between a flame condition and the condition of a seed: |ln T ratio| + |ln P ratio| + |phi diff|."""
    return (abs(math.log(T_u / seed["T_u"])) + abs(math.log(P / seed["P"])) + abs(phi - seed["phi"]))


def pick_seed(seeds: list[dict], key: tuple, T_u: float, P: float, phi: float) -> dict | None:
    """The solved flame nearest to the condition; a flame of the same mechanism wins a tie with another mechanism.

    The restart reader of Zero-RK reads the fields by name: a species that
    the seed lacks starts at zero, and a field that the mechanism lacks is
    ignored. So a flame of any mechanism is a usable initial guess.
    """
    best, best_d = None, math.inf
    for seed in seeds:
        d = condition_distance(T_u, P, phi, seed) + (0.0 if seed["key"] == key else OTHER_MECHANISM_PENALTY)
        if d < best_d:
            best, best_d = seed, d
    return best


def scale_seed(seed: dict, T_u: float, P: float, phi: float) -> tuple[np.ndarray, np.ndarray, float]:
    """``(x, values, length)`` of a seed moved to the condition ``(T_u, P, phi)`` by flame similarity.

    The positions scale with the estimated flame thickness, the relative
    volume (1/density) with the inverse pressure ratio, and the mass flux
    (density times flame speed) with ``(P / P_seed)**0.7``, the pressure
    exponent of the flame-speed correlation of the Zero-RK driver. Temperature
    and mass fractions keep their profiles. At the condition of the seed the
    function returns the seed unchanged.
    """
    ratio_x = estimate_thickness(P, T_u, phi) / estimate_thickness(seed["P"], seed["T_u"], seed["phi"])
    values = np.array(seed["values"], dtype=float)
    names = seed["names"]
    volume = [j for j, n in enumerate(names) if "vol" in n.lower() and not n.startswith("MassFraction_")]
    for j in volume:
        values[j] *= seed["P"] / P
    if values.shape[0] == len(names) + 1:
        values[len(names)] *= (P / seed["P"]) ** 0.7
    return np.asarray(seed["x"], dtype=float) * ratio_x, values, float(seed["length"]) * ratio_x


def remember_seed(key: tuple, T_u: float, P: float, phi: float, out: dict) -> None:
    """Keep a converged flame as an initial guess for the next conditions; the oldest goes past SEED_LIMIT."""
    _SEEDS.append({"key": key, "T_u": float(T_u), "P": float(P), "phi": float(phi), "names": list(out["names"]),
                   "values": np.array(out["values"], dtype=float), "x": np.array(out["x"], dtype=float),
                   "num_points": int(out["num_points"]), "length": float(out["length"])})
    del _SEEDS[:-SEED_LIMIT]


def clear_seeds() -> None:
    """Forget every solved flame (for a cold start of the next flame)."""
    _SEEDS.clear()


def _remove_owned_folders() -> None:
    for folder in list(_OWNED_FOLDERS):
        shutil.rmtree(folder, ignore_errors=True)
    _OWNED_FOLDERS.clear()
    for root in list(_RUN_ROOTS):           # rmdir removes empty folders only: kept files stay
        for folder in (root / "mechanisms", root):
            try:
                folder.rmdir()
            except OSError:
                pass
    _RUN_ROOTS.clear()


atexit.register(_remove_owned_folders)


# ---------------------------------------------------------------------------
# Installation and mechanism files
# ---------------------------------------------------------------------------

def zerork_home(zcfg: dict | None = None) -> Path:
    """The Zero-RK install folder: ``fs.solver.zerork.home``, else ``ZERORK_HOME``, else ``/opt/zero-rk``."""
    home = (zcfg or {}).get("home") or os.environ.get("ZERORK_HOME") or DEFAULT_HOME
    return Path(home).absolute()        # the solver runs with cwd=work, so a relative ZERORK_HOME must not move


def zerork_available(zcfg: dict | None = None) -> bool:
    """True when the flame solver and the Lewis-number generator exist under the install folder."""
    home = zerork_home(zcfg)
    return all((home / "bin" / name).is_file() for name in (STEADY_SERIAL, LEWIS_GENERATOR))


def _executables(zcfg: dict, processes: int) -> tuple[Path, Path, Path]:
    """``(serial solver, MPI solver, Lewis generator)``; the MPI solver must exist only for more than one rank."""
    home = zerork_home(zcfg)
    missing = [name for name in (STEADY_SERIAL, LEWIS_GENERATOR) if not (home / "bin" / name).is_file()]
    if processes > 1 and not (home / "bin" / STEADY_MPI).is_file():
        missing.append(STEADY_MPI)
    if missing:
        raise FlameSolverError(f"Zero-RK is not available: {home / 'bin'} lacks {missing}. Build the image from "
                               "Dockerfile.cpu, or point fs.solver.zerork.home or ZERORK_HOME at a Zero-RK install.",
                               path=str(home / "bin"), missing=missing)
    return home / "bin" / STEADY_SERIAL, home / "bin" / STEADY_MPI, home / "bin" / LEWIS_GENERATOR


def _mechanism_key(gas: ct.Solution) -> tuple:
    return (getattr(gas, "source", None), tuple(gas.species_names), gas.n_reactions)


def strip_notes(yaml_path: Path) -> int:
    """Remove the ``note`` fields of the species and reactions in place; returns the number removed.

    ``yaml2ck`` writes a species note into the comment field of the first
    NASA-7 line. A note that starts with ``!`` ends the line for every CHEMKIN
    reader (Cantera's own ``ck2yaml`` included), and the species loses its
    thermodynamic data.
    """
    from ruamel import yaml as ruamel_yaml

    loader = ruamel_yaml.YAML(typ="rt")
    data = loader.load(yaml_path.read_text(encoding="utf-8"))
    removed = 0
    for species in data.get("species", []) or []:
        if species.pop("note", None) is not None:
            removed += 1
        thermo = species.get("thermo") or {}
        if thermo.pop("note", None) is not None:
            removed += 1
    for reaction in data.get("reactions", []) or []:
        if reaction.pop("note", None) is not None:
            removed += 1
    if removed:
        with yaml_path.open("w", encoding="utf-8") as fh:
            loader.dump(data, fh)
    return removed


def chemkin_files(gas: ct.Solution, folder: Path | None = None, keep: bool = False) -> ChemkinFiles:
    """CHEMKIN mechanism, thermodynamic and transport files of ``gas``, converted once per process.

    The Solution is written as YAML (``Solution.write_yaml``), the notes are
    removed, the file is reloaded with mixture-averaged transport (Zero-RK
    needs transport data for every species) and ``cantera.yaml2ck`` converts
    it. The result is cached on the species list and the reaction count, so a
    reduced mechanism that the sweep evaluates at several conditions is
    converted once. The cache holds ``CHEMKIN_CACHE_SIZE`` mechanisms; the
    folder of an evicted entry is removed when this function created it and
    ``keep`` was false, and the remaining ones at exit. A conversion failure
    raises :class:`FlameSolverError`.
    """
    key = _mechanism_key(gas)
    hit = _CHEMKIN_CACHE.get(key)
    if hit is not None and hit.mechanism.is_file():
        _CHEMKIN_CACHE.move_to_end(key)
        return hit
    from cantera import yaml2ck

    folder = Path(folder) if folder is not None else Path(tempfile.mkdtemp(prefix="gasig_zerork_mech_"))
    folder.mkdir(parents=True, exist_ok=True)
    if not keep:
        _OWNED_FOLDERS.add(folder)
    yaml_path = folder / "mechanism.yaml"
    files = ChemkinFiles(folder / "mech.inp", folder / "therm.dat", folder / "tran.dat")
    try:
        write_yaml(gas, yaml_path, quiet=True)  # leaves out the blank notes that Cantera cannot write
        removed = strip_notes(yaml_path)
        solution = ct.Solution(str(yaml_path), transport_model="mixture-averaged")
        yaml2ck.convert(solution, mechanism_path=files.mechanism, thermo_path=files.thermo,
                        transport_path=files.transport, overwrite=True)
    except MechanismError:
        raise                               # a mechanism that cannot be written is not a flame failure
    except (ct.CanteraError, ValueError, OSError) as exc:
        raise FlameSolverError(f"Zero-RK needs CHEMKIN files with transport data for every species; the conversion "
                               f"of the {gas.n_species}-species mechanism failed "
                               f"({str(exc).strip().splitlines()[0][:200] if str(exc).strip() else exc!r}).",
                               n_species=int(gas.n_species), path=str(folder)) from exc
    log.debug("CHEMKIN files of %d species written to %s (%d notes removed)", gas.n_species, folder, removed)
    _CHEMKIN_CACHE[key] = files
    while len(_CHEMKIN_CACHE) > CHEMKIN_CACHE_SIZE:
        _, old = _CHEMKIN_CACHE.popitem(last=False)
        old_folder = old.mechanism.parent
        if old_folder in _OWNED_FOLDERS:
            shutil.rmtree(old_folder, ignore_errors=True)
            _OWNED_FOLDERS.discard(old_folder)
    return files


def _mechanism_folder(zcfg: dict, gas: ct.Solution) -> Path | None:
    """``<work_dir>/<run folder>/mechanisms/<n>sp_<hash>`` for the CHEMKIN files when ``work_dir`` is set, else None.

    The folder is in the run folder of this process: no other process
    removes it while the solver of this process reads the files.
    """
    if not zcfg.get("work_dir"):
        return None
    digest = hashlib.sha1(repr(_mechanism_key(gas)).encode("utf-8")).hexdigest()[:10]
    return _run_root(zcfg) / "mechanisms" / f"{gas.n_species}sp_{digest}"


# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------

def yaml_text(values: dict) -> str:
    """The Zero-RK input file: one scalar, flow map or flow list per line; species names quoted."""
    lines = []
    for key, value in values.items():
        if isinstance(value, dict):
            inner = ", ".join(f"'{name}': {float(x)!r}" for name, x in value.items())
            lines.append(f"{key}: {{{inner}}}")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'y' if value else 'n'}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(x) for x in value)}]")
        elif isinstance(value, (int, float)):
            lines.append(f"{key}: {value!r}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def solver_input(files: ChemkinFiles, zcfg: dict, fuel: dict, oxidizer: dict, P: float, T_u: float, phi: float,
                 length: float, num_points: int, *, transport_model: str, pseudo_unsteady: bool,
                 restart_file: str | None) -> dict:
    """The keys of the premixed steady flame solver (and of the Lewis-number generator) for one flame.

    The values that are not conditions follow the ``flame_speed`` example of
    Zero-RK: second-order convection, the preconditioned integrator with the
    stored Jacobian, a relative tolerance of ``rel_tol`` for the steady solve
    and an ignition temperature of 1500 K for the initial burnt side. The
    initial front sits at ``START_EXTENT`` of the domain, not at the 3 percent
    of the example: the solver anchors the flame at the first point of the
    initial profile that is 250 K above the inlet, and a front 3 percent from
    the inlet leaves no room for a preheat zone of a few tenths of a
    millimetre (the jet-fuel flames at 473 K lost their flame and ended with a
    negative mass flux). The regrid rules move the flame to 30 percent of the
    domain anyway. The mass flux is an unknown of the steady problem, so the
    solver needs no inlet velocity; the Lewis-number generator ignores the
    keys it does not know.
    """
    values = {
        "mech_file": str(files.mechanism), "therm_file": str(files.thermo), "trans_file": str(files.transport),
        "transport_model": transport_model, "log_file": "mechanism_parser.log", "Lewis_file": "Lewis_file",
        "inlet_fuel_comp": fuel, "inlet_oxidizer_comp": oxidizer, "inlet_full_comp": {}, "initial_comp": oxidizer,
        "egr": 0.0, "grid_file": "grid_file",
    }
    if restart_file is not None:
        values["restart_file"] = restart_file
    values.update({
        "convective_scheme_type": 2, "step_limiter": 1.0e300, "simulation_type": 1, "ignition_temperature": 1500.0,
        "pressure": float(P), "diameter": 1.0e-3, "length": float(length),
        "nusselt": 0.0, "initial_inlet_extent": START_EXTENT * float(length), "thickness": 1.0e-5, "max_time": 1.0,
        "stats_dt": 1.0e-6, "field_dt_multiplier": 1, "integrator_type": 3, "store_jacobian": True,
        "max_internal_dt": 2.0e-5, "max_steps": 1000000, "max_subiter": 10, "num_points": int(num_points),
        "rel_tol": float(zcfg["rel_tol"]), "abs_tol": 1.0e-20, "ref_temperature": REF_TEMPERATURE,
        "inlet_temperature": float(T_u), "inlet_phi": float(phi),
        "pseudo_unsteady": bool(pseudo_unsteady), "pseudo_unsteady_dt": 1.0e-6,
    })
    return values


# ---------------------------------------------------------------------------
# Grid, data files and the regrid rules of flamesolver.py
# ---------------------------------------------------------------------------

def estimate_thickness(P: float, T_u: float, phi: float) -> float:
    """Flame thickness estimate [m] of flamesolver.py; it sets the first domain length only."""
    return (4.0e-4 + 2.4e-3 * (phi - 1.07) ** 2) * (1.0e5 / P) ** 0.8 * (300.0 / T_u) ** 0.3


def generate_grid(num_points: int, length: float) -> np.ndarray:
    """``num_points + 1`` positions from 0 to ``length``: uniform over the first 75 percent with 90 percent of
    the points, geometrically stretched after that (``generate_grid`` of flamesolver.py)."""
    n = int(num_points) + 1
    n_fine = int(n * 0.9)
    fine_end = 0.75 * length
    dz = fine_end / (n_fine - 1)
    n_end = n - n_fine
    stretch = 1.1
    for _ in range(40):
        stretch = ((length - fine_end) * (stretch - 1.0) / dz + 1.0) ** (1.0 / n_end)
    z = np.zeros(n)
    for j in range(1, n_fine):
        z[j] = z[j - 1] + dz
    for j in range(n_fine, n):
        z[j] = z[j - 1] + dz
        dz *= stretch
    return z


def check_grid(length: float, num_points: int, lF: float, flame_loc: float, min_pts_per_lF: float,
               add_points: int) -> tuple[bool, int, float, float]:
    """Decide whether the grid must change (``check_grid`` of flamesolver.py).

    Returns ``(regrid, num_points, length, shift)``: the domain grows or
    shrinks by 20 percent when it holds fewer than 15 or more than 30 flame
    thicknesses, the flame is moved to 30 percent of the length when it
    comes within 10 percent of the inlet or within three thicknesses of the
    end of the fine region, and points are added while the fine spacing
    exceeds ``lF / min_pts_per_lF``.
    """
    flame_loc_rel = 0.3
    shift = length * flame_loc_rel - flame_loc
    regrid = False
    if length < 15.0 * lF:
        length *= 1.2
        shift = length * flame_loc_rel - flame_loc
        flame_loc = length * flame_loc_rel
        regrid = True
    if length > 30.0 * lF:
        length *= 0.8
        shift = length * flame_loc_rel - flame_loc
        flame_loc = length * flame_loc_rel
        regrid = True
    if flame_loc - lF < 0.1 * length:
        shift = length * flame_loc_rel - flame_loc
        regrid = True
    if flame_loc + 3.0 * lF > 0.75 * length:
        shift = length * flame_loc_rel - flame_loc
        regrid = True
    dx = 0.75 * length / (0.9 * num_points)
    if dx > lF / min_pts_per_lF:
        num_points = min(num_points + add_points, MAX_POINTS)
        shift = 0.0
        regrid = True
    return regrid, int(num_points), float(length), float(shift)


def round_points(num_points: int, processes: int) -> int:
    """The smallest multiple of ``processes`` at or above ``num_points``, at most ``MAX_POINTS``.

    The MPI solver splits the interior points evenly over the ranks and stops
    when the count is not divisible.
    """
    processes = max(1, int(processes))
    n = -(-int(num_points) // processes) * processes
    if n > MAX_POINTS:
        n = (MAX_POINTS // processes) * processes
    return int(n)


def read_data_file(path: Path) -> tuple[list[str], np.ndarray, float]:
    """A Zero-RK data (restart) file: ``(names, values, time)`` with ``values`` of shape (blocks, points).

    Layout: ``int`` points, ``int`` named variables, ``double`` time, 64
    bytes per name, then the blocks of ``points`` doubles. The steady solver
    writes one block more than it names: the mass flux.
    """
    raw = Path(path).read_bytes()
    npts, nvars, t = struct.unpack_from("iid", raw)
    offset = struct.calcsize("iid")
    names = []
    for _ in range(nvars):
        names.append(raw[offset:offset + 64].split(b"\0")[0].decode("ascii", errors="replace"))
        offset += 64
    blocks = (len(raw) - offset) // (8 * npts)
    values = np.frombuffer(raw, dtype="<f8", count=blocks * npts, offset=offset).reshape(blocks, npts)
    return names, values.copy(), float(t)


def write_data_file(path: Path, names: list[str], values: np.ndarray, t: float) -> None:
    values = np.ascontiguousarray(values, dtype="<f8")
    with Path(path).open("wb") as fh:
        fh.write(struct.pack("iid", int(values.shape[1]), len(names), float(t)))
        for name in names:
            fh.write(name.encode("ascii")[:64].ljust(64, b"\0"))
        fh.write(values.tobytes())


def interpolate_data(old_data: Path, old_grid: np.ndarray, new_grid: np.ndarray, shift: float, new_data: Path) -> None:
    """Move a solution to a new grid; the old grid is shifted by ``shift`` first (``interpolate_data`` of flamesolver.py)."""
    names, values, t = read_data_file(old_data)
    x_old = np.asarray(old_grid, dtype=float) + shift
    moved = np.empty((values.shape[0], new_grid.shape[0]))
    for j in range(values.shape[0]):
        moved[j] = np.interp(new_grid, x_old[1:-1], values[j, 1:-1])
    write_data_file(new_data, names, moved, t)


def read_final_line(log_path: Path) -> tuple[float, float, float] | None:
    """``(flame speed [m/s], thickness [m], location [m])`` from the ``Final`` line of the log; None without a flame."""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("Final"):
            tokens = line.split()
            try:
                speed, thickness, location = float(tokens[2]), float(tokens[3]), float(tokens[5])
            except (IndexError, ValueError):
                return None
            return (speed, thickness, location) if speed > 0.0 else None
    return None


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------

def _processes(zcfg: dict) -> int:
    n = zcfg.get("processes")
    if n is None:
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - not Linux
            n = os.cpu_count() or 1
        n = min(8, n)
    return max(1, int(n))


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the solver and its MPI ranks: the launcher runs in its own process group."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - not POSIX
            proc.kill()
    except OSError:
        proc.kill()
    proc.wait()


def seeded_start(attempt, processes: int, step: dict):
    """The continuation ladder: a pseudo-unsteady solve from the seed on all ranks, then serially.

    ``attempt(n, run_id, pseudo=..., tag=..., **step)`` is one solver call; it returns None when the call ends
    without a flame. A steady solve from the seed is not tried: in the jet-fuel comparison it converged in 3 of 29
    tries and once ran 100 Newton iterations (75 min on 745 species) before it failed, whereas the
    pseudo-unsteady solve converged in 25 of 26 tries.
    """
    final = attempt(processes, 1, pseudo=True, tag="cp", **step)
    if final is None and processes > 1:
        final = attempt(1, 1, pseudo=True, tag="cs", **step)
    return final


def _run(command: list[str], cwd: Path, log_path: Path, timeout: float | None) -> int:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with log_path.open("w", encoding="utf-8") as fh:
        try:
            proc = subprocess.Popen(command, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
        except OSError as exc:
            raise FlameSolverError(f"Zero-RK could not start: {command[0]} ({exc}).", command=command[0]) from exc
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_tree(proc)
            raise SolverTimeout(f"Zero-RK did not finish within {timeout:.0f} s ({command[0]}).",
                                command=command[0], timeout_s=float(timeout)) from exc


def _log_tail(log_path: Path, lines: int = 5) -> str:
    try:
        tail = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return ""
    return " | ".join(ln.strip() for ln in tail if ln.strip())[:300]


def _solve(files: ChemkinFiles, zcfg: dict, fuel: dict, oxidizer: dict, P: float, T_u: float, phi: float,
           work: Path, label: str, seed: dict | None = None) -> dict:
    processes = _processes(zcfg)
    mpiexec = shlex.split(zcfg["mpiexec"]) if processes > 1 else []
    if processes > 1 and shutil.which(mpiexec[0]) is None:
        log.warning("%s: the MPI launcher %r is not on PATH; the serial solver runs every solve.", label, mpiexec[0])
        processes, mpiexec = 1, []
    solver_serial, solver_mpi, lewis = _executables(zcfg, processes)
    timeout = zcfg.get("timeout_s")
    min_per_lF = float(zcfg["min_points_per_thickness"])
    max_solves = int(zcfg["max_solves"])
    processes_init = min(processes, 16)
    points_init = round_points(max(2 * processes_init, 128), processes_init)
    points_fine = round_points(max(2 * processes, 128), processes)
    add_points = max(processes, 64)
    length = 20.0 * estimate_thickness(P, T_u, phi)
    grid_file, log_file = work / "grid_file", work / LOG_FILE

    common = dict(files=files, zcfg=zcfg, fuel=fuel, oxidizer=oxidizer, P=P, phi=phi)

    def attempt(n: int, run_id: int, *, pseudo: bool, restart: str | None, length: float, num_points: int,
                tag: str) -> tuple[float, float, float] | None:
        """One solver call on ``n`` ranks; None when it ends without a flame. The log stays as ``log_<run_id><tag>``."""
        (work / INPUT_FILE).write_text(yaml_text(solver_input(
            T_u=T_u, length=length, num_points=num_points, transport_model="ConstantLewis",
            pseudo_unsteady=pseudo, restart_file=restart, **common)), encoding="utf-8")
        command = [*mpiexec, "-n", str(n), str(solver_mpi), INPUT_FILE] if n > 1 else [str(solver_serial), INPUT_FILE]
        try:
            rc = _run(command, work, log_file, timeout)
        except SolverTimeout as exc:
            shutil.copy(log_file, work / f"log_{run_id}{tag}")
            log.info("%s: solve %d%s (%s, %d rank%s) gave no flame: %s", label, run_id, tag,
                     "pseudo-unsteady" if pseudo else "steady", n, "" if n == 1 else "s", exc)
            return None
        shutil.copy(log_file, work / f"log_{run_id}{tag}")
        final = read_final_line(log_file) if rc == 0 else None
        if final is None:
            log.info("%s: solve %d%s (%s, %d rank%s) gave no flame: code %d, %s", label, run_id, tag,
                     "pseudo-unsteady" if pseudo else "steady", n, "" if n == 1 else "s", rc, _log_tail(log_file))
        return final

    # 1. Lewis numbers of the inlet mixture (one point, mixture-averaged transport at lewis_temperature_K).
    (work / LEWIS_INPUT_FILE).write_text(yaml_text(solver_input(
        T_u=float(zcfg["lewis_temperature_K"]), length=length, num_points=1, transport_model="MixAvg",
        pseudo_unsteady=False, restart_file=None, **common)), encoding="utf-8")
    rc = _run([str(lewis), LEWIS_INPUT_FILE], work, work / "log_lewis", timeout)
    if rc != 0 or not (work / "Lewis_file").is_file():
        raise FlameSolverError(f"{label}: the Lewis-number generator failed (code {rc}; "
                               f"{_log_tail(work / 'log_lewis')}).", label=label, return_code=rc,
                               P0_atm=P / ct.one_atm, phi=phi, T_unburnt_K=T_u)

    solves = 1
    final = None
    seeded = False
    # 2a. Continuation: the solved flame of a near condition (seed) on its own grid is the initial guess of the
    #     ladder in seeded_start. If it fails, the cold start runs.
    if seed is not None:
        seed_x, seed_values, seed_length = scale_seed(seed, T_u, P, phi)
        seed_points = round_points(seed["num_points"], processes)
        seed_grid = generate_grid(seed_points, seed_length)
        write_data_file(work / "seed_data", seed["names"], seed_values, 0.0)
        interpolate_data(work / "seed_data", seed_x, seed_grid, 0.0, work / "data_new")
        np.savetxt(grid_file, seed_grid, fmt="%e")
        step = dict(restart="data_new", length=seed_length, num_points=seed_points)
        final = seeded_start(attempt, processes, step)
        if final is None:
            log.warning("%s: the start from the solved flame at P=%.3g atm, phi=%.3g failed; the cold start runs.",
                        label, seed["P"] / ct.one_atm, seed["phi"])
            for stale in work.glob("data_*"):
                stale.unlink()
        else:
            length, num_points, seeded = seed_length, seed_points, True
    # 2b. Cold start: coarse grid with the pseudo-unsteady start. The MPI build aborts on the first failed Newton
    #     solve (its KINSOL error handler calls MPI_Abort); the serial build halves the pseudo time step and goes on.
    #     So the MPI solver gets the first try and the serial solver repeats a failed start.
    if final is None:
        num_points = points_init
        np.savetxt(grid_file, generate_grid(points_init, length), fmt="%e")
        start = dict(pseudo=True, restart=None, length=length, num_points=points_init)
        final = attempt(processes_init, solves, tag="", **start)
        if final is None and processes_init > 1:
            log.warning("%s: the MPI pseudo-unsteady start failed; the serial solver repeats it.", label)
            final = attempt(1, solves, tag="s", **start)
        if final is None:
            raise FlameSolverError(f"{label}: the pseudo-unsteady start on {points_init} points gave no flame "
                                   f"({_log_tail(log_file)}).", label=label, num_points=points_init,
                                   P0_atm=P / ct.one_atm, phi=phi, T_unburnt_K=T_u)
        num_points = max(points_fine, points_init)
    speed, thickness, location = final
    log.debug("%s: start  S_L=%.4f m/s, l_F=%.2e m, x_f=%.2e m on %d points", label, speed, thickness, location,
              num_points)

    # 3. Regrid around the flame and solve the steady problem until the grid is fine enough.
    solved = (int(np.loadtxt(grid_file).shape[0]) - 1, length)      # the grid of the last solve
    regrid, num_points, length, shift = check_grid(length, num_points, thickness, location, min_per_lF, add_points)
    while regrid and solves < max_solves:
        num_points = round_points(num_points, processes)
        if (num_points, length, shift) == (*solved, 0.0):
            log.warning("%s: the limit of %d grid points is reached; the grid stays at %d points.", label,
                        MAX_POINTS, num_points)
            break
        newest = _newest_data(work)
        old_grid = np.loadtxt(grid_file)
        new_grid = generate_grid(num_points, length)
        np.savetxt(grid_file, new_grid, fmt="%e")
        interpolate_data(newest, old_grid, new_grid, shift, work / "data_new")
        solves += 1
        step = dict(restart="data_new", length=length, num_points=num_points)
        final = attempt(processes, solves, pseudo=False, tag="", **step)
        if final is None:
            log.warning("%s: the steady solve %d failed; the pseudo-unsteady solver continues from the restart.",
                        label, solves)
            final = attempt(processes, solves, pseudo=True, tag="p", **step)
        if final is None and processes > 1:
            final = attempt(1, solves, pseudo=True, tag="s", **step)
        if final is None:
            raise FlameSolverError(f"{label}: the steady solve {solves} on {num_points} points gave no flame "
                                   f"({_log_tail(log_file)}).", label=label, solves=solves, num_points=num_points,
                                   P0_atm=P / ct.one_atm, phi=phi, T_unburnt_K=T_u)
        speed, thickness, location = final
        solved = (num_points, length)
        log.debug("%s: solve %d S_L=%.4f m/s, l_F=%.2e m, x_f=%.2e m on %d points over %.2e m", label, solves,
                  speed, thickness, location, num_points, length)
        regrid, num_points, length, shift = check_grid(length, num_points, thickness, location, min_per_lF,
                                                       add_points)
    if regrid:
        # The position rules decide whether the last solution is a flame in the fine region; the spacing rule
        # alone (too few points per thickness) leaves a usable but coarse flame.
        near_inlet = location - thickness < 0.1 * solved[1]
        past_fine = location + 3.0 * thickness > 0.75 * solved[1]
        if near_inlet or past_fine:
            raise FlameSolverError(f"{label}: the flame sits outside the fine region after {solves} solves "
                                   f"(S_L={speed:.4f} m/s, x_f={location:.2e} m of {solved[1]:.2e} m); raise "
                                   "fs.solver.zerork.max_solves.", label=label, solves=solves,
                                   flame_speed_m_s=float(speed), flame_position_m=float(location),
                                   P0_atm=P / ct.one_atm, phi=phi, T_unburnt_K=T_u)
        log.warning("%s: fewer than %.0f points per flame thickness after %d solves; the last solution is used.",
                    label, min_per_lF, solves)
    names, values, _ = read_data_file(_newest_data(work))
    return {"speed": speed, "thickness": thickness, "location": location, "x": np.loadtxt(grid_file),
            "names": names, "values": values, "solves": solves, "grid_converged": not regrid,
            "num_points": solved[0], "length": solved[1], "seeded": seeded}


def _data_time(path: Path) -> float:
    try:
        return float(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1.0


def _newest_data(work: Path) -> Path:
    """The data file of the last solve: the largest time in the name, then the newest modification time."""
    files = [p for p in work.glob("data_*") if p.name != "data_new"]
    if not files:
        raise FlameSolverError(f"Zero-RK wrote no data file in {work}.", path=str(work))
    return max(files, key=lambda p: (_data_time(p), p.stat().st_mtime))


def _composition(value: dict) -> dict:
    return {str(k): float(v) for k, v in value.items()}


def _work_dir(zcfg: dict, label: str) -> tuple[Path, bool]:
    """The folder of one flame, ``<work_dir>/<run folder>/<label>``; the second value says whether it is removed.

    The run folder (:func:`run_folder_name`) is unique per process, also in
    Docker. So a sweep and a report that share ``work_dir`` do not delete
    each other's files.
    """
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label) or "flame"
    if zcfg.get("work_dir"):
        folder = _run_root(zcfg) / safe
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        return folder, not bool(zcfg["keep_files"])
    return Path(tempfile.mkdtemp(prefix=f"gasig_zerork_{safe}_")), not bool(zcfg["keep_files"])


def profile_from_data(names: list[str], values: np.ndarray, species: list[str], label: str
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(T [K], u [m/s], Y)`` on the grid from the blocks of a steady-solver data file.

    The named blocks are the species mass fractions (``MassFraction_<name>``
    or the bare name), the relative volume and the temperature divided by
    ``ref_temperature``; the unnamed last block is the mass flux, and ``u`` is
    the mass flux times the relative volume.
    """
    prefix = "MassFraction_"
    index = {(name[len(prefix):] if name.startswith(prefix) else name): j for j, name in enumerate(names)}
    missing = [s for s in species if s not in index]
    if missing:
        raise FlameSolverError(f"{label}: the Zero-RK data file lacks the species {missing[:5]}.", label=label,
                               species=missing)
    others = [(j, n.lower()) for j, n in enumerate(names) if j not in {index[s] for s in species}]
    temp = [j for j, n in others if "temp" in n]
    volume = [j for j, n in others if "vol" in n]
    if len(temp) != 1 or len(volume) != 1 or values.shape[0] != len(names) + 1:
        raise FlameSolverError(f"{label}: unexpected variables in the Zero-RK data file: "
                               f"{[names[j] for j, _ in others][:6]}, {values.shape[0]} blocks for {len(names)} names.",
                               label=label, n_blocks=int(values.shape[0]), n_names=len(names))
    T = np.array(values[temp[0]], dtype=float)
    if T.max() < 20.0:                      # the state vector holds T / ref_temperature; the data file holds K
        T *= REF_TEMPERATURE
    u = values[len(names)] * values[volume[0]]
    Y = np.stack([values[index[s]] for s in species], axis=1)
    return T, u, Y


def run_flame_zerork(
    mech_file: str,
    phase_name: str | None,
    P: float,
    phi: float,
    solver_cfg: dict,
    label: str = "",
    gas: ct.Solution | None = None,
) -> Trajectory:
    """Solve one freely propagating flame with Zero-RK; same contract as :func:`gasig.solvers.flame.run_flame`."""
    zcfg = solver_cfg["zerork"]
    if gas is None:
        gas = ct.Solution(mech_file) if phase_name is None else ct.Solution(mech_file, phase_name)
    T_u = float(solver_cfg["T_unburnt_K"])
    fuel, oxidizer = _composition(solver_cfg["fuel"]), _composition(solver_cfg["oxidizer"])
    unknown = [s for s in (*fuel, *oxidizer) if s not in gas.species_names]
    if unknown:
        raise FlameSolverError(f"{label}: the mixture species {unknown} are not in the mechanism.", label=label or None,
                               species=unknown, n_species=int(gas.n_species))
    t_start = time.perf_counter()
    files = chemkin_files(gas, folder=_mechanism_folder(zcfg, gas), keep=bool(zcfg["keep_files"]))
    key = _mechanism_key(gas)
    seed = pick_seed(_SEEDS, key, T_u, float(P), float(phi)) if zcfg["continuation"] else None
    if seed is not None:
        log.debug("%s: initial guess from the solved flame at T_u=%.0f K, P=%.3g atm, phi=%.3g (%s mechanism)",
                  label, seed["T_u"], seed["P"] / ct.one_atm, seed["phi"],
                  "same" if seed["key"] == key else f"{len(seed['key'][1])}-species")
    work, remove = _work_dir(zcfg, label)
    try:
        out = _solve(files, zcfg, fuel, oxidizer, float(P), T_u, float(phi), work, label or "flame", seed=seed)
    except FlameSolverError as exc:
        if not remove:
            raise type(exc)(f"{exc} The solver files are in {work}.", stage=exc.stage,
                            **{**exc.context, "work_dir": str(work)}) from exc
        raise
    finally:
        if remove:
            shutil.rmtree(work, ignore_errors=True)
        else:
            log.info("%s: the Zero-RK files stay in %s", label, work)
    x = np.asarray(out["x"], dtype=float)
    T, u, Y = profile_from_data(out["names"], out["values"], gas.species_names, label or "flame")
    if x.shape[0] != T.shape[0]:
        raise FlameSolverError(f"{label}: the grid has {x.shape[0]} points, the data file {T.shape[0]}.",
                               label=label or None, n_grid=int(x.shape[0]), n_data=int(T.shape[0]))
    if abs(T[0] - T_u) > 0.05 * T_u:
        raise FlameSolverError(f"{label}: the inlet temperature of the profile is {T[0]:.0f} K, expected {T_u:.0f} K.",
                               label=label or None, T_inlet_K=float(T[0]), T_unburnt_K=T_u)
    try:
        tau = residence_time(x, u)
        tip = flame_tip_index(T, solver_cfg["flame_tip_T_fraction"])
    except ValueError as exc:
        raise FlameSolverError(f"{label}: the Zero-RK profile is not usable ({exc}).", label=label or None,
                               P0_atm=float(P) / ct.one_atm, phi=float(phi), T_unburnt_K=T_u) from exc
    S_L = float(out["speed"])
    if zcfg["continuation"] and out["grid_converged"]:
        remember_seed(key, T_u, float(P), float(phi), out)
    log.info("%s: T_u=%.0f K, P=%.2f atm, phi=%.2f, S_L=%.4f m/s (Zero-RK, %s start, %d solves, %d points), "
             "tip at %d (x=%.4e m), %.1f s", label, T_u, P / ct.one_atm, phi, S_L,
             "continued" if out["seeded"] else "cold", out["solves"], x.shape[0], tip, x[tip],
             time.perf_counter() - t_start)
    return Trajectory(
        tau=tau, T=T, P=np.full(x.shape[0], float(P)), Y=Y, end_index=tip, label=label,
        meta={"source": "solver", "backend": "zerork", "T_unburnt_K": T_u, "P_Pa": float(P),
              "P_atm": float(P) / ct.one_atm, "phi": float(phi), "flame_speed_m_s": S_L,
              "flame_thickness_m": float(out["thickness"]), "n_points": int(x.shape[0]), "tip_index": tip,
              "tip_position_m": float(x[tip]), "T_max_K": float(T.max()), "x": x, "u": u,
              "zerork_solves": int(out["solves"]), "grid_converged": bool(out["grid_converged"]),
              "continued": bool(out["seeded"]),
              "domain_length_m": float(out["length"]), "seconds": round(time.perf_counter() - t_start, 2)},
    )
