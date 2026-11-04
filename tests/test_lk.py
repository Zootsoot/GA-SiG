"""Linearised knockout: numerics against the direct forms and the GA-SiG integrator, limits, order and stage.

Reference values (USC Mech II, 1500 K, 1 atm, phi 1, score at t/t_end = 0.5): removing CH3 stops the ignition;
removing C2H6 or H2 shortens the ignition delay by 46 % and 48 %; removing HO2 lengthens it by 49 % (brute-force
knockouts with the Cantera reactor). The GA-SiG estimate of these removals is about ten times smaller than the LK.
"""
import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ct = pytest.importorskip("cantera")

from helpers_reduction import IDT, TARGETS  # noqa: E402
from gasig.config import ConfigError, validate_config  # noqa: E402
from gasig.kinetics import TorchKinetics  # noqa: E402
from gasig.lk import lk_order, lk_trajectory, removal_sets, sample_index, time_shift  # noqa: E402
from gasig.reduce import reduced_solution  # noqa: E402
from gasig.sensitivity import integrate_trajectory  # noqa: E402
from gasig.solvers.idt import run_idt  # noqa: E402

SOLVER = dict(IDT, reactor="constant_volume", reference_species="OH", ignition_criterion="reference_peak",
              rtol=1e-6, atol=1e-12, max_steps=200000)


@pytest.fixture(scope="module")
def case(mech_path):
    gas = ct.Solution(mech_path)
    traj = run_idt("", None, 1500.0, ct.one_atm, 1.0, SOLVER, label="idt-0", gas=gas)
    kin = TorchKinetics(mech_path, None, device="cpu", dtype=torch.float64, rate_law="corrected")
    names = kin.species_names
    sets = removal_sets(gas, [[s] for s in names])
    s = sample_index(traj, 0.5)
    checks = [names.index(x) for x in ("CH3", "HO2", "H2O")]
    res = lk_trajectory(kin, traj, sets, s, with_ga=True, check_steps={1, s // 2, s}, check_groups=checks)
    return {"gas": gas, "traj": traj, "kin": kin, "names": names, "sets": sets, "s": s, "res": res}


def test_removal_sets_follow_the_reduced_mechanism(case):
    gas, names, sets = case["gas"], case["names"], case["sets"]
    for b in ("H2O", "AR", "CH3", "C6H5C2H", "N2"):
        reduced = reduced_solution(gas, [x for x in names if x != b])
        assert len(sets[names.index(b)]) == gas.n_reactions - reduced.n_reactions, b
    assert sets[names.index("C6H5C2H")] == []                         # a species without reactions


def test_decomposed_jacobian_and_woodbury_update_match_the_direct_forms(case):
    checks = case["res"]["checks"]
    assert len(checks["J1_decomposition_rel"]) == 3 and len(checks["woodbury_vs_direct_rel"]) == 9
    assert max(c["rel"] for c in checks["J1_decomposition_rel"]) < 1e-12
    assert max(c["rel"] for c in checks["woodbury_vs_direct_rel"]) < 1e-9


def test_ga_estimate_equals_minus_the_sum_of_the_gasig_sensitivities(case):
    """The GA-SiG integrator (corrected rate law, no guard) gives S; the LK code without the update gives -sum S_j."""
    from dataclasses import replace

    traj = replace(case["traj"], end_index=case["s"])
    S, _ = integrate_trajectory(case["kin"], traj, substep_tol=None, progress_every=0)
    ref = np.stack([-S[:, G].sum(axis=1) for G in case["sets"]], axis=1)
    Dga = case["res"]["Dga"]
    scale = np.maximum(np.abs(ref).max(axis=0), 1e-300)
    rel = np.abs(Dga - ref).max(axis=0) / scale
    assert np.all(rel[np.abs(ref).max(axis=0) > 0] < 1e-8)


def test_lk_limits_small_and_zero_rate_removals(case):
    names = case["names"]
    v = time_shift(case["gas"], case["traj"], case["s"], case["res"]["Dlk"])
    vga = time_shift(case["gas"], case["traj"], case["s"], case["res"]["Dga"])
    lk, ga = dict(zip(names, v)), dict(zip(names, vga))
    for b in ("iC3H7", "C2H", "CH3CHCH"):                             # small removals: the two estimates agree
        assert abs(lk[b] - ga[b]) <= 0.02 * abs(ga[b]), b
    assert lk["CH3"] > 0 and lk["HO2"] > 0 and lk["C2H6"] < 0 and lk["H2"] < 0     # signs of the brute force
    for b in ("CH3", "C2H6", "H2"):                                    # effects at zero rate that GA-SiG misses
        assert abs(lk[b]) > 3.0 * abs(ga[b]), b
    order = [names[i] for i in np.argsort(-np.abs(v)) if names[i] != "CH4"]          # the fuel leads the list
    assert order[0] == "CH3"


def test_time_shift_leaves_out_the_row_of_the_removed_species(case):
    Ns = len(case["names"])
    only_own = np.diag(np.linspace(1.0, 2.0, Ns))
    assert np.all(time_shift(case["gas"], case["traj"], case["s"], only_own) == 0.0)


def test_sample_index_rejects_the_initial_state(case):
    from dataclasses import replace

    from gasig.errors import InputError

    traj = case["traj"]
    assert 0 < sample_index(traj, 0.5) <= traj.end_index and sample_index(traj, 1.0) == traj.end_index
    t_end = float(traj.tau[traj.end_index])
    coarse = replace(traj, tau=np.concatenate([[0.0], np.linspace(0.05, 1.0, len(traj.tau) - 1)]) * t_end)
    with pytest.raises(InputError, match="initial state"):
        sample_index(coarse, 0.02)


def test_lk_order_sorts_by_score_and_breaks_ties_by_the_gasig_rank():
    names = ["A", "B", "C", "D"]
    scores = np.array([0.0, 0.2, 0.0, 0.1])
    assert lk_order(names, scores, ["C", "A", "D", "B"]) == ["C", "A", "D", "B"]
    assert lk_order(names, scores, ["A", "C", "B", "D"]) == ["A", "C", "D", "B"]


def write_config(tmp_path, mech_path, **extra):
    raw = {"mechanism": mech_path, "device": "cpu", "output_dir": "out",
           "sensitivity": {"progress_every": 0, "rate_law": "corrected"},
           "idt": {"enabled": True, "solver": dict(SOLVER)},
           "ranking": {"target_species": TARGETS},
           "lk": {"enabled": True}, "units": {"detect": True}, **extra}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_run_writes_the_lk_order_and_the_units_and_the_stages_rerun(tmp_path, mech_path):
    from gasig.cli import main
    from gasig.pipeline import run

    config = write_config(tmp_path, mech_path)
    summary = run(config)
    out = tmp_path / "out"
    with (out / "lk_order.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    gas = ct.Solution(mech_path)
    assert sorted(r["species"] for r in rows) == sorted(gas.species_names)
    scores = [float(r["importance"]) for r in rows]
    assert scores == sorted(scores) and scores[-1] > 0.0
    assert set(rows[0]) == {"rank_from_least_important", "species", "importance", "gasig_importance"}
    lk = summary["lk"]
    assert lk["rate_law"] == "corrected" and len(lk["trajectories"]) == 1
    assert 0.4 < lk["trajectories"][0]["t_over_tend"] < 0.6 and lk["trajectories"][0]["J1_decomposition_rel_max"] < 1e-12
    assert np.load(out / "lk_scores.npy").shape == (1, gas.n_species)
    units = json.loads((out / "chain_units.json").read_text(encoding="utf-8"))
    assert units["scope"] == "all" and summary["units"]["file"].endswith("chain_units.json")
    assert summary["failures"] == {} and (out / "gasig.failures.jsonl").read_text(encoding="utf-8") == ""
    # The LK stage alone reloads the saved trajectories and gives the same order.
    first = [r["species"] for r in rows]
    summary2 = run(config, stage="lk")
    with (out / "lk_order.csv").open(encoding="utf-8", newline="") as fh:
        assert [r["species"] for r in csv.DictReader(fh)] == first
    assert summary2["ranking"] == summary["ranking"] and summary2["units"] == summary["units"]
    summary3 = run(config, stage="units")
    assert summary3["lk"]["files"] == summary["lk"]["files"]
    # idt_data/ that does not match the record of the last sensitivity stage is refused.
    record_path = out / "trajectories_idt.json"
    records = json.loads(record_path.read_text(encoding="utf-8"))
    record_path.write_text(json.dumps(records + records), encoding="utf-8")
    assert main(["run", str(config), "--stage", "lk"]) == 3
    record_path.write_text(json.dumps(records), encoding="utf-8")
    bad = write_config(tmp_path, mech_path, lk={"enabled": False})
    assert main(["run", str(bad), "--stage", "lk"]) == 2
    bad = write_config(tmp_path, mech_path, lk={"enabled": True, "trajectories": [3]})
    assert main(["run", str(bad), "--stage", "lk"]) == 2


def test_lk_config_checks(tmp_path, mech_path):
    base = {"mechanism": mech_path, "idt": {"enabled": True, "solver": dict(IDT)}}
    cfg = validate_config(base, tmp_path)
    assert cfg["lk"] == {"enabled": False, "trajectories": None, "sample_fraction": 0.5, "rate_law": "corrected",
                         "output_file": "lk_order.csv"}
    for lk, match in (({"sample_fraction": 0.0}, "sample_fraction"), ({"trajectories": []}, "trajectories"),
                      ({"trajectories": [1, 1]}, "trajectories"), ({"trajectories": [-1]}, "trajectories"),
                      ({"rate_law": "exact"}, "rate_law")):
        with pytest.raises(ConfigError, match=match):
            validate_config({**base, "lk": lk}, tmp_path)
