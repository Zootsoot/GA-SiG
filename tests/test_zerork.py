"""Tests of the Zero-RK flame backend.

The pure parts (input text, grid rules, data files, log parsing, CHEMKIN
conversion, dispatch and configuration) run everywhere. The solve itself
runs only where Zero-RK is installed (marker ``zerork``; the CPU image
``ga-sig:cpu`` built from ``Dockerfile.cpu`` has it under ``/opt/zero-rk``).
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

ct = pytest.importorskip("cantera")

from helpers_reduction import make_config  # noqa: E402
from gasig.config import ConfigError, validate_config  # noqa: E402
from gasig.reduce import _fs_values, evaluation_config  # noqa: E402
from gasig.solvers import zerork  # noqa: E402
from gasig.solvers.flame import FlameSolverError, run_flame  # noqa: E402

FS_SOLVER = {"fuel": {"CH4": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79}, "P_atm": [1.0], "phi": [1.0],
             "T_unburnt_K": 300.0, "backend": "zerork"}
needs_zerork = pytest.mark.skipif(not zerork.zerork_available(), reason="Zero-RK is not installed")


def fs_config(tmp_path, mech_path, **zerork_keys) -> dict:
    """The evaluation flame configuration with the zerork backend and the given ``fs.solver.zerork`` keys."""
    fs = {"solver": dict(FS_SOLVER, zerork=zerork_keys)}
    cfg = make_config(tmp_path, mech_path, reduction={"evaluate_fs": True}, fs=fs)
    return evaluation_config(cfg)[2]


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------

def test_input_text_quotes_species_and_writes_booleans_as_flags():
    text = zerork.yaml_text({"inlet_fuel_comp": {"CH4": 1.0}, "pseudo_unsteady": True, "store_jacobian": False,
                             "rel_tol": 1e-2, "num_points": 128, "transport_model": "ConstantLewis",
                             "track_max": ["OH", "CH"]})
    lines = text.splitlines()
    assert lines[0] == "inlet_fuel_comp: {'CH4': 1.0}"
    assert "pseudo_unsteady: y" in lines and "store_jacobian: n" in lines
    assert "rel_tol: 0.01" in lines and "num_points: 128" in lines
    assert "transport_model: ConstantLewis" in lines and "track_max: [OH, CH]" in lines
    assert text.endswith("\n")


def test_solver_input_holds_the_conditions_and_the_files(tmp_path):
    files = zerork.ChemkinFiles(tmp_path / "m.inp", tmp_path / "t.dat", tmp_path / "tr.dat")
    values = zerork.solver_input(files, {"rel_tol": 0.02}, {"CH4": 1.0}, {"O2": 0.21, "N2": 0.79}, 5 * ct.one_atm,
                                 473.0, 0.9, 0.01, 128, transport_model="ConstantLewis", pseudo_unsteady=True,
                                 restart_file=None)
    assert values["mech_file"] == str(files.mechanism) and values["trans_file"] == str(files.transport)
    assert values["pressure"] == pytest.approx(5 * ct.one_atm) and values["inlet_temperature"] == 473.0
    assert values["inlet_phi"] == 0.9 and values["length"] == 0.01 and values["num_points"] == 128
    assert values["initial_inlet_extent"] == pytest.approx(0.003) and values["rel_tol"] == 0.02
    assert values["pseudo_unsteady"] is True and "restart_file" not in values
    assert values["transport_model"] == "ConstantLewis" and values["Lewis_file"] == "Lewis_file"
    again = zerork.solver_input(files, {"rel_tol": 0.02}, {"CH4": 1.0}, {"O2": 1.0}, ct.one_atm, 300.0, 1.0, 0.01,
                                256, transport_model="ConstantLewis", pseudo_unsteady=False,
                                restart_file="data_new")
    assert again["restart_file"] == "data_new" and again["pseudo_unsteady"] is False


def test_generate_grid_is_uniform_then_stretched():
    z = zerork.generate_grid(128, 0.012)
    assert z.shape == (129,) and z[0] == 0.0 and z[-1] == pytest.approx(0.012, rel=1e-6)
    assert np.all(np.diff(z) > 0)
    n_fine = int(129 * 0.9)
    dz = np.diff(z)
    assert np.allclose(dz[:n_fine - 1], dz[0]) and z[n_fine - 1] == pytest.approx(0.75 * 0.012)
    assert np.all(np.diff(dz[n_fine - 1:]) > 0)                          # the tail stretches


def test_check_grid_rules():
    lF, add = 1e-3, 64
    # A domain that holds 20 thicknesses with the flame at 30 percent and enough points: no change.
    regrid, n, L, shift = zerork.check_grid(0.02, 512, lF, 0.006, 12.0, add)
    assert not regrid and n == 512 and L == 0.02 and shift == 0.0
    # Fewer than 15 thicknesses: the domain grows by 20 percent.
    regrid, n, L, shift = zerork.check_grid(0.012, 512, lF, 0.0036, 12.0, add)
    assert regrid and L == pytest.approx(0.0144) and shift == pytest.approx(0.3 * 0.0144 - 0.0036)
    # More than 30 thicknesses: the domain shrinks by 20 percent.
    regrid, n, L, _ = zerork.check_grid(0.04, 1024, lF, 0.012, 12.0, add)
    assert regrid and L == pytest.approx(0.032)
    # The flame within 10 percent of the inlet: moved to 30 percent.
    regrid, n, L, shift = zerork.check_grid(0.02, 512, lF, 0.0025, 12.0, add)
    assert regrid and L == 0.02 and shift == pytest.approx(0.006 - 0.0025)
    # Too few points per thickness: points are added, up to 1024, without a shift.
    regrid, n, L, shift = zerork.check_grid(0.02, 128, lF, 0.006, 12.0, add)
    assert regrid and n == 192 and shift == 0.0
    regrid, n, _, _ = zerork.check_grid(0.02, 1000, 1e-5, 0.006, 12.0, add)
    assert regrid and n == 1024


def test_estimates_follow_the_driver_correlations():
    assert zerork.estimate_thickness(1e5, 300.0, 1.07) == pytest.approx(4e-4)
    assert zerork.estimate_thickness(1e6, 300.0, 1.07) < zerork.estimate_thickness(1e5, 300.0, 1.07)
    assert zerork.estimate_thickness(1e5, 473.0, 1.0) < zerork.estimate_thickness(1e5, 300.0, 1.0)


def test_round_points_is_a_multiple_of_the_ranks_within_the_limit():
    assert zerork.round_points(128, 6) == 132 and zerork.round_points(128, 4) == 128
    assert zerork.round_points(1030, 6) == 1020 and zerork.round_points(1, 1) == 1


def test_pick_seed_prefers_the_nearest_condition_and_the_same_mechanism():
    a, b = ("a.yaml", ("H2", "O2"), 3), ("b.yaml", ("H2", "O2", "N2"), 5)
    seeds = [{"key": a, "T_u": 300.0, "P": 1e5, "phi": 1.0}, {"key": a, "T_u": 300.0, "P": 1e5, "phi": 0.7},
             {"key": b, "T_u": 300.0, "P": 1e5, "phi": 0.9}]
    assert zerork.pick_seed([], a, 300.0, 1e5, 1.0) is None
    assert zerork.pick_seed(seeds, a, 300.0, 1e5, 0.95)["phi"] == 1.0        # same mechanism, 0.05 away
    assert zerork.pick_seed(seeds, b, 300.0, 1e5, 1.0)["key"] == b           # 0.1 away beats 0 + 0.1 penalty
    assert zerork.pick_seed(seeds, ("c", (), 0), 300.0, 1e5, 0.72)["phi"] == 0.7
    assert zerork.condition_distance(300.0, 5e5, 1.0, seeds[0]) == pytest.approx(math.log(5.0))


def test_scale_seed_follows_flame_similarity():
    names = ["MassFraction_O2", "RelativeVolume", "Temperature"]
    values = np.vstack([np.full(4, 0.2), np.full(4, 2.0), np.full(4, 1.5), np.full(4, 0.3)])
    seed = {"names": names, "values": values, "x": np.linspace(0.0, 0.01, 4), "length": 0.01,
            "T_u": 473.0, "P": 1e5, "phi": 1.0}
    x, v, L = zerork.scale_seed(seed, 473.0, 1e5, 1.0)                       # the same condition: unchanged
    assert np.array_equal(x, seed["x"]) and np.array_equal(v, values) and L == 0.01
    x, v, L = zerork.scale_seed(seed, 473.0, 5e5, 1.0)
    assert L == pytest.approx(0.01 * 5.0 ** -0.8) and x[-1] == pytest.approx(L)
    assert v[1] == pytest.approx(np.full(4, 0.4)) and v[3] == pytest.approx(np.full(4, 0.3 * 5.0 ** 0.7))
    assert np.array_equal(v[0], values[0]) and np.array_equal(v[2], values[2])
    assert np.array_equal(seed["values"], values)                            # the seed itself is not changed


def test_remember_seed_keeps_the_newest_flames(monkeypatch):
    monkeypatch.setattr(zerork, "_SEEDS", [])
    out = {"names": ["T"], "values": np.zeros((2, 3)), "x": np.zeros(3), "num_points": 2, "length": 0.01}
    for i in range(zerork.SEED_LIMIT + 3):
        zerork.remember_seed(("m", (), 0), 300.0, 1e5, 0.5 + 0.01 * i, out)
    assert len(zerork._SEEDS) == zerork.SEED_LIMIT and zerork._SEEDS[0]["phi"] == pytest.approx(0.53)
    zerork.clear_seeds()
    assert zerork._SEEDS == []


def _fake_attempt(results: dict):
    """An ``attempt`` stand-in that records each call and returns ``results[tag]`` (None means no flame)."""
    calls = []

    def attempt(n, run_id, *, pseudo, tag, **step):
        calls.append((tag, n, pseudo, step["num_points"]))
        return results.get(tag)
    return attempt, calls


def test_seeded_start_tries_pseudo_unsteady_on_all_ranks_then_serially():
    step = dict(restart="data_new", length=0.01, num_points=192)
    attempt, calls = _fake_attempt({"cs": (0.3, 1e-3, 3e-3)})
    assert zerork.seeded_start(attempt, 4, step) == (0.3, 1e-3, 3e-3)
    assert calls == [("cp", 4, True, 192), ("cs", 1, True, 192)]      # no steady solve from the seed
    attempt, calls = _fake_attempt({"cp": (0.3, 1e-3, 3e-3)})
    assert zerork.seeded_start(attempt, 4, step) == (0.3, 1e-3, 3e-3)
    assert calls == [("cp", 4, True, 192)]


def test_seeded_start_on_one_rank_has_no_serial_retry():
    attempt, calls = _fake_attempt({})
    assert zerork.seeded_start(attempt, 1, dict(restart="data_new", length=0.01, num_points=128)) is None
    assert calls == [("cp", 1, True, 128)]


def test_run_raises_a_solver_timeout_that_is_a_flame_solver_error(tmp_path):
    import sys
    with pytest.raises(zerork.SolverTimeout):
        zerork._run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, tmp_path / "log", 0.5)
    assert issubclass(zerork.SolverTimeout, FlameSolverError)


def test_kept_solver_files_keep_the_class_and_the_stage_of_the_failure(tmp_path, mech_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise zerork.SolverTimeout("over the limit", stage="fs", label="f0")

    monkeypatch.setattr(zerork, "_solve", timeout)
    fs_cfg = fs_config(tmp_path, mech_path, work_dir=str(tmp_path / "zr"), keep_files=True)
    with pytest.raises(zerork.SolverTimeout, match="solver files are in") as info:
        zerork.run_flame_zerork("", None, ct.one_atm, 1.0, fs_cfg, label="f0", gas=ct.Solution(mech_path))
    assert info.value.stage == "fs" and info.value.context["label"] == "f0" and "work_dir" in info.value.context


def test_newest_data_prefers_the_largest_time_in_the_name(tmp_path):
    for name in ("data_1.000000", "data_0.000000", "data_new"):
        (tmp_path / name).write_bytes(b"x")
    assert zerork._newest_data(tmp_path).name == "data_1.000000"
    with pytest.raises(FlameSolverError, match="no data file"):
        zerork._newest_data(tmp_path / "empty")


def test_data_file_round_trip_and_interpolation(tmp_path):
    names = ["H2", "O2", "RelativeVolume", "Temperature"]
    x_old = np.linspace(0.0, 1.0, 11)
    values = np.stack([x_old, 1.0 - x_old, 2.0 + x_old, 0.3 + x_old, 0.1 * np.ones(11)])      # names + mass flux
    path = tmp_path / "data_1.000000"
    zerork.write_data_file(path, names, values, 1.0)
    got_names, got_values, t = zerork.read_data_file(path)
    assert got_names == names and t == 1.0 and got_values.shape == (5, 11)
    assert np.array_equal(got_values, values)
    assert path.stat().st_size == 8 + 8 + 4 * 64 + 5 * 11 * 8
    x_new = np.linspace(0.0, 1.0, 21)
    zerork.interpolate_data(path, x_old, x_new, 0.0, tmp_path / "data_new")
    new_names, new_values, _ = zerork.read_data_file(tmp_path / "data_new")
    assert new_names == names and new_values.shape == (5, 21)
    interior = (x_new >= x_old[1]) & (x_new <= x_old[-2])          # the boundary points of the old data are not used
    assert np.allclose(new_values[0][interior], x_new[interior]) and np.allclose(new_values[4], 0.1)
    assert np.all(new_values[0][x_new < x_old[1]] == x_old[1]) and np.all(new_values[0][x_new > x_old[-2]] == x_old[-2])
    zerork.interpolate_data(path, x_old, x_new, 0.25, tmp_path / "data_shift")       # the old grid moves right
    _, shifted, _ = zerork.read_data_file(tmp_path / "data_shift")
    inside = (x_new > 0.35) & (x_new < 1.15)
    assert np.allclose(shifted[0][inside], x_new[inside] - 0.25)
    assert np.all(shifted[0][x_new <= 0.35] == x_old[1])                              # clamped to the first interior value


def test_read_final_line(tmp_path):
    log = tmp_path / "log"
    log.write_text("# header\n0.5 1 2 3\nFinal  1.000000e+00  3.812e-01  4.10e-04  5.0e-04  3.1e-03  0 0\n", encoding="utf-8")
    assert zerork.read_final_line(log) == pytest.approx((0.3812, 4.1e-4, 3.1e-3))
    log.write_text("Final  0  0  0\n", encoding="utf-8")
    assert zerork.read_final_line(log) is None
    log.write_text("# Error, pseudo-unsteady solver is not converging\n", encoding="utf-8")
    assert zerork.read_final_line(log) is None
    assert zerork.read_final_line(tmp_path / "missing") is None


def test_profile_from_data_orders_the_species_and_scales_the_temperature():
    species = ["O2", "N2", "CH4"]
    names = ["CH4", "O2", "N2", "RelativeVolume", "Temperature"]
    values = np.array([[0.05, 0.0], [0.22, 0.0], [0.73, 1.0], [1.0, 7.0], [0.3, 2.2], [0.4, 0.4]])
    T, u, Y = zerork.profile_from_data(names, values, species, "t")
    assert np.allclose(T, [300.0, 2200.0]) and np.allclose(u, [0.4, 2.8])
    assert np.allclose(Y, [[0.22, 0.73, 0.05], [0.0, 1.0, 0.0]])
    with pytest.raises(FlameSolverError, match="lacks the species"):
        zerork.profile_from_data(names, values, ["H2"], "t")
    with pytest.raises(FlameSolverError, match="unexpected variables"):
        zerork.profile_from_data(names, values[:-1], species, "t")


def test_chemkin_files_round_trip(mech_path, tmp_path):
    from cantera import ck2yaml

    gas = ct.Solution(mech_path)
    files = zerork.chemkin_files(gas, tmp_path / "ck")
    assert files.mechanism.is_file() and files.thermo.is_file() and files.transport.is_file()
    assert zerork.chemkin_files(gas) is files                              # cached on the species and reactions
    thermo = files.thermo.read_text(encoding="utf-8")
    assert "CH4" in thermo and "THERMO" in thermo.upper()
    out = tmp_path / "back.yaml"
    ck2yaml.convert(str(files.mechanism), thermo_file=str(files.thermo), transport_file=str(files.transport),
                    out_name=str(out), quiet=True, permissive=True)
    back = ct.Solution(str(out))
    assert back.n_species == gas.n_species and back.n_reactions == gas.n_reactions
    assert set(back.species_names) == set(gas.species_names)
    gas.TPX = back.TPX = 1200.0, ct.one_atm, "CH4:1, O2:2, N2:7.52"
    assert np.allclose(back.net_production_rates, gas.net_production_rates, rtol=1e-6, atol=1e-12)


def test_strip_notes_removes_species_thermo_and_reaction_notes(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("species:\n- name: A\n  note: '!bad'\n  thermo: {model: constant-cp, note: x}\n"
                    "- name: B\nreactions:\n- equation: A => B\n  note: y\n", encoding="utf-8")
    assert zerork.strip_notes(path) == 3
    text = path.read_text(encoding="utf-8")
    assert "note" not in text and "name: A" in text and "equation: A => B" in text
    assert zerork.strip_notes(path) == 0


# ---------------------------------------------------------------------------
# Dispatch, failures and configuration
# ---------------------------------------------------------------------------

def test_missing_install_is_a_flame_solver_error_and_inf_in_the_sweep(tmp_path, mech_path):
    fs_cfg = fs_config(tmp_path, mech_path, home=str(tmp_path / "nowhere"))
    gas = ct.Solution(mech_path)
    with pytest.raises(FlameSolverError, match="Zero-RK is not available"):
        run_flame("", None, ct.one_atm, 1.0, fs_cfg, label="x", gas=gas)
    assert _fs_values(gas, [(ct.one_atm, 1.0)], fs_cfg, "x") == [math.inf]
    import gasig.report as report

    assert report._flame_speed(gas, ct.one_atm, 1.0, fs_cfg, "x") == math.inf
    assert report._flame_solver_text(fs_cfg).startswith("Zero-RK")
    assert report._flame_solver_text(dict(fs_cfg, backend="cantera")) == "Cantera FreeFlame, mixture-averaged transport"


def test_unknown_mixture_species_fail_before_the_solver(tmp_path, mech_path, monkeypatch):
    fs_cfg = fs_config(tmp_path, mech_path)
    fs_cfg["fuel"] = {"NC12H26": 1.0}
    monkeypatch.setattr(zerork, "_executables", lambda *a, **k: (Path("x"), Path("y"), Path("z")))
    with pytest.raises(FlameSolverError, match="not in the mechanism"):
        run_flame("", None, ct.one_atm, 1.0, fs_cfg, label="x", gas=ct.Solution(mech_path))


def test_dispatch_goes_to_the_zerork_backend(tmp_path, mech_path, monkeypatch):
    calls = []

    def fake(mech_file, phase_name, P, phi, solver_cfg, label="", gas=None):
        calls.append((P, phi, label, solver_cfg["backend"]))
        raise FlameSolverError("stop")

    monkeypatch.setattr(zerork, "run_flame_zerork", fake)
    fs_cfg = fs_config(tmp_path, mech_path)
    with pytest.raises(FlameSolverError, match="stop"):
        run_flame("", None, 2 * ct.one_atm, 0.8, fs_cfg, label="d", gas=ct.Solution(mech_path))
    assert calls == [(2 * ct.one_atm, 0.8, "d", "zerork")]


def test_config_defaults_overrides_and_checks(tmp_path, mech_path):
    cfg = make_config(tmp_path, mech_path)
    z = cfg["fs"]["solver"]["zerork"]
    assert cfg["fs"]["solver"]["backend"] == "cantera" and cfg["reduction"]["fs"]["backend"] is None
    assert z["home"] is None and z["processes"] is None and z["mpiexec"] == "mpirun --oversubscribe"
    assert z["min_points_per_thickness"] == 12 and z["rel_tol"] == 1e-2 and z["lewis_temperature_K"] == 1500.0
    assert z["max_solves"] == 10 and z["timeout_s"] is None and z["work_dir"] is None and z["keep_files"] is False
    assert z["continuation"] is True
    fs_cfg = evaluation_config(make_config(tmp_path, mech_path, reduction={"evaluate_fs": True, "fs": {"backend": "zerork"}},
                                           fs={"solver": dict(FS_SOLVER, backend="cantera",
                                                              zerork={"work_dir": "zr", "processes": 2})}))[2]
    assert fs_cfg["backend"] == "zerork" and fs_cfg["zerork"]["processes"] == 2
    assert fs_cfg["zerork"]["work_dir"] == str(tmp_path / "zr")                    # resolved against the config
    for bad in ({"backend": "cantera2"}, {"zerork": {"processes": 0}}, {"zerork": {"rel_tol": -1}},
                {"zerork": {"mpiexec": " "}}, {"zerork": {"max_solves": 0}}, {"zerork": {"unknown": 1}},
                {"zerork": {"keep_files": True}}):
        with pytest.raises(ConfigError):
            make_config(tmp_path, mech_path, fs={"solver": dict(FS_SOLVER, **bad)})
    with pytest.raises(ConfigError):
        make_config(tmp_path, mech_path, reduction={"fs": {"backend": "x"}})


def test_home_defaults_and_resolves_against_the_config(tmp_path, mech_path, monkeypatch):
    monkeypatch.delenv("ZERORK_HOME", raising=False)
    assert zerork.zerork_home({}) == Path("/opt/zero-rk").absolute()
    monkeypatch.setenv("ZERORK_HOME", "zr-env")
    assert zerork.zerork_home({"home": None}) == Path("zr-env").absolute()
    fs_cfg = evaluation_config(make_config(tmp_path, mech_path, reduction={"evaluate_fs": True},
                                           fs={"solver": dict(FS_SOLVER, zerork={"home": "zr"})}))[2]
    assert zerork.zerork_home(fs_cfg["zerork"]) == (tmp_path / "zr").absolute()


def test_missing_mpi_launcher_falls_back_to_the_serial_solver(tmp_path, mech_path, monkeypatch):
    seen = {}

    def fake_executables(zcfg, processes):
        seen["processes"] = processes
        return Path("serial"), Path("mpi"), Path("lewis")

    monkeypatch.setattr(zerork.shutil, "which", lambda name: None)
    monkeypatch.setattr(zerork, "_executables", fake_executables)
    monkeypatch.setattr(zerork, "_run", lambda *args, **kwargs: 1)
    fs_cfg = fs_config(tmp_path, mech_path, processes=4)
    with pytest.raises(FlameSolverError, match="Lewis-number generator failed"):
        run_flame("", None, ct.one_atm, 1.0, fs_cfg, label="s", gas=ct.Solution(mech_path))
    assert seen["processes"] == 1


def test_run_folders_of_two_pid_1_processes_do_not_collide(tmp_path, monkeypatch):
    """Two Docker containers both run as PID 1; they share work_dir but not the flame and CHEMKIN folders."""
    monkeypatch.setattr(zerork.os, "getpid", lambda: 1)
    monkeypatch.setattr(zerork, "_RUN_NAMES", {})
    monkeypatch.setattr(zerork, "_RUN_ROOTS", set())
    monkeypatch.setattr(zerork, "_OWNED_FOLDERS", set())
    zcfg = {"work_dir": str(tmp_path / "zr"), "keep_files": False}
    gas = ct.Solution("h2o2.yaml")

    name_a = zerork.run_folder_name()
    assert name_a == zerork.run_folder_name() and name_a.startswith("run-") and "-1-" in name_a
    work_a, remove = zerork._work_dir(zcfg, "full-f0")
    mech_a = zerork._mechanism_folder(zcfg, gas)
    (work_a / "log").write_text("A")
    assert remove and work_a.parent.name == name_a and mech_a.parent.parent.name == name_a

    zerork._RUN_NAMES.clear()                                   # the second container: PID 1 again
    name_b = zerork.run_folder_name()
    work_b, _ = zerork._work_dir(zcfg, "full-f0")
    mech_b = zerork._mechanism_folder(zcfg, gas)
    assert name_b != name_a and work_b != work_a and mech_b != mech_a
    assert (work_a / "log").read_text() == "A"                  # the second run does not remove the first one's flame

    # Clean-up at exit: empty run folders go, a folder with kept files stays.
    shutil.rmtree(work_b)
    zerork._remove_owned_folders()
    assert not (tmp_path / "zr" / name_b).exists()
    assert (work_a / "log").is_file()


def test_processes_default_is_at_most_eight(monkeypatch):
    monkeypatch.setattr(zerork.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    assert zerork._processes({"processes": None}) == 8
    assert zerork._processes({"processes": 3}) == 3


# ---------------------------------------------------------------------------
# The solve (needs Zero-RK)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.zerork
@needs_zerork
def test_methane_flame_zerork_against_cantera(tmp_path, mech_path):
    """One methane flame with both backends: the speeds agree within 10 percent and the profile is usable."""
    fs_cfg = fs_config(tmp_path, mech_path, work_dir=str(tmp_path / "zr"), keep_files=True)
    gas = ct.Solution(mech_path)
    zr = run_flame("", None, ct.one_atm, 1.0, fs_cfg, label="zr-f0", gas=gas)
    cf = run_flame("", None, ct.one_atm, 1.0, dict(fs_cfg, backend="cantera"), label="ct-f0", gas=gas)
    sl_zr, sl_ct = zr.meta["flame_speed_m_s"], cf.meta["flame_speed_m_s"]
    assert zr.meta["backend"] == "zerork" and zr.meta["grid_converged"] and zr.meta["zerork_solves"] >= 2
    assert 0.3 < sl_ct < 0.45 and abs(sl_zr - sl_ct) / sl_ct < 0.10
    assert abs(zr.meta["T_max_K"] - cf.meta["T_max_K"]) < 60.0
    assert zr.Y.shape == (zr.T.shape[0], gas.n_species) and zr.P.shape == zr.T.shape
    assert np.all(np.diff(zr.tau) > 0) and zr.tau[0] == 0.0
    assert np.allclose(zr.Y.sum(axis=1), 1.0, atol=1e-2) and np.all(zr.Y > -1e-8)   # the solver does not renormalize
    assert abs(zr.T[0] - 300.0) < 5.0 and 0 < zr.end_index < zr.T.shape[0] - 1
    assert zr.T[zr.end_index] >= 0.99 * zr.T.max()
    # Zero-RK's CHEMKIN parser uses the 2007 atomic weights (C 12.0107, H 1.00794, O 15.9994, N 14.0067);
    # Cantera 3 uses 12.011, 1.008, 15.999, 14.007: the inlet mass fractions differ by up to 8.2e-6 (O2).
    assert np.abs(zr.Y[0] - cf.Y[0]).max() < 2e-5                     # the same inlet mixture, up to the weights
    work = tmp_path / "zr" / zerork.run_folder_name() / "zr-f0"
    assert (work / "log").is_file() and (work / "Lewis_file").is_file() and (work / "input.yml").is_file()
    assert json.loads(json.dumps({"S_L": sl_zr}))["S_L"] == sl_zr
