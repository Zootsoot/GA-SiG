import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ct = pytest.importorskip("cantera")

from gasig.cli import main  # noqa: E402
from gasig.pipeline import run  # noqa: E402

TARGETS = ["CH4", "O2", "OH", "H2O", "CO2"]


def write_config(path, mech_path, **overrides):
    cfg = {
        "mechanism": mech_path,
        "device": "cpu",
        "output_dir": "out",
        "sensitivity": {"progress_every": 0},
        "idt": {
            "enabled": True,
            "source": "solver",
            "solver": {
                "fuel": {"CH4": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79},
                "T_K": [1500.0], "P_atm": [1.0], "phi": [1.0],
                "reactor": "constant_volume", "end_time_s": 2.0e-3, "reference_species": "OH",
            },
        },
        "fs": {
            "enabled": True,
            "source": "solver",
            "solver": {
                "fuel": {"CH4": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79},
                "T_unburnt_K": 300.0, "P_atm": [1.0], "phi": [1.0],
                "domain_width_m": 0.02,
                "refine": {"ratio": 4.0, "slope": 0.25, "curve": 0.4, "prune": 0.05},
            },
        },
        "ranking": {"target_species": TARGETS, "top_n_print": 10},
    }
    for key, value in overrides.items():
        cfg[key] = value
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return path


@pytest.mark.slow
def test_end_to_end_methane_both_modes(tmp_path, mech_path):
    config = write_config(tmp_path / "config.json", mech_path)
    summary = run(config)
    out = tmp_path / "out"

    for name in ("r_AB_idt.npy", "r_AB_fs.npy", "S_AB_idt.npy", "S_AB_fs.npy", "r_AB.npy",
                 "importance.npy", "ranked_sp.npy", "ranked_species.csv", "summary.json",
                 "config_resolved.json", "gasig.log", "trajectories_idt.json", "trajectories_fs.json"):
        assert (out / name).is_file(), name
    for name in ("time_all.npy", "T_all.npy", "P_all.npy", "Y_all.npy", "index_all.npy"):
        assert (out / "idt_data" / name).is_file()
    for name in ("position_all.npy", "velocity_all.npy", "index_all.npy"):
        assert (out / "fs_data" / name).is_file()

    gas = ct.Solution(mech_path)
    names = list(gas.species_names)
    for mode in ("idt", "fs"):
        r = np.load(out / f"r_AB_{mode}.npy")
        assert r.shape == (gas.n_species, gas.n_species)
        assert np.all(r >= 0.0) and np.all(r <= 1.0 + 1e-12)
        rows = r.sum(axis=1)
        assert np.all((np.abs(rows - 1.0) < 1e-9) | (rows == 0.0))
    assert summary["ranking"]["matrix"] == "hybrid"

    idt_rec = summary["modes"]["idt"]["trajectories"][0]
    assert idt_rec["ignited"] is True
    assert 1e-5 < idt_rec["tau_ign_s"] < 2e-3
    fs_rec = summary["modes"]["fs"]["trajectories"][0]
    assert 0.25 < fs_rec["flame_speed_m_s"] < 0.50          # CH4/air, 1 atm, 300 K
    assert 0 < fs_rec["tip_index"] < fs_rec["n_points"] - 1

    importance = np.load(out / "importance.npy")
    for t in TARGETS:
        assert importance[names.index(t)] == pytest.approx(1.0)
    ranked = np.load(out / "ranked_sp.npy")
    assert ranked.shape == (gas.n_species, 2)
    top20 = [row[0] for row in ranked[-20:]]
    for radical in ("H", "O", "OH", "HO2", "CH3", "CO"):
        assert radical in top20, radical

    # the resolved configuration is itself a valid configuration file
    from gasig.config import load_config

    resolved = load_config(out / "config_resolved.json")
    assert resolved["output_dir"] == str(out.resolve())

    # ranking stage alone reproduces the same importance and keeps the sensitivity record
    summary2 = run(config, stage="rank")
    assert np.load(out / "importance.npy") == pytest.approx(importance)
    assert summary2["ranking"]["targets"] == TARGETS
    assert set(summary2["modes"]) == {"idt", "fs"}
    assert summary2["mechanism"]["n_three_body"] == 15
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["modes"]["fs"]["n_trajectories"] == 1 and saved["stage"] == "rank"

    # the saved solver data works as precomputed input and gives the same r_AB
    pre_cfg = json.loads(config.read_text(encoding="utf-8"))
    pre_cfg["output_dir"] = "out_pre"
    pre_cfg["fs"]["enabled"] = False
    pre_cfg["idt"] = {"enabled": True, "source": "precomputed", "precomputed": {
        "time": "out/idt_data/time_all.npy", "T": "out/idt_data/T_all.npy", "P": "out/idt_data/P_all.npy",
        "Y": "out/idt_data/Y_all.npy", "index": "out/idt_data/index_all.npy"}}
    config2 = tmp_path / "config_pre.json"
    config2.write_text(json.dumps(pre_cfg), encoding="utf-8")
    run(config2)
    r_pre = np.load(tmp_path / "out_pre" / "r_AB_idt.npy")
    assert r_pre == pytest.approx(np.load(out / "r_AB_idt.npy"), rel=1e-10, abs=1e-14)


def test_reference_species_that_is_a_reactant_is_rejected(mech_path):
    """A reference species that peaks at t = 0 (a reactant) must not pass as an ignition sample."""
    from gasig.solvers.idt import run_idt

    cfg = {"fuel": {"CH4": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79}, "reactor": "constant_volume",
           "end_time_s": 2.0e-3, "reference_species": "CH4", "ignition_criterion": "reference_peak",
           "rtol": 1e-6, "atol": 1e-12, "max_steps": 200000}
    with pytest.raises(ValueError, match="reference species"):
        run_idt(mech_path, None, 1500.0, ct.one_atm, 1.0, cfg, label="idt-test")


@pytest.mark.slow
def test_precomputed_example_runs_with_version_01_switches(tmp_path, example_dir):
    """The shipped ethylene example runs through the package with the switches of the version 0.1
    script that produced example/r_AB.npy (original rate law, S_row_sum, signed sum) on two trajectories."""
    pre = example_dir
    sub = {}
    for key, stem in (("time", "traj"), ("T", "T"), ("P", "P"), ("Y", "Y")):
        d = np.load(pre / f"c2h4_{stem}_all.npy", allow_pickle=True).item()
        sub[key] = {0: d[0], 1: d[1]}
        np.save(tmp_path / f"{key}.npy", np.array(sub[key], dtype=object), allow_pickle=True)
    idx = np.load(pre / "c2h4_index_all.npy")[:2]
    idx = np.minimum(idx, 60)
    np.save(tmp_path / "index.npy", idx)
    cfg = {
        "mechanism": str(pre / "USCMechII.yaml"), "device": "cpu", "output_dir": "out",
        "sensitivity": {"rate_law": "original", "substep_tol": None, "progress_every": 0},
        "aggregation": {"abs_S": False, "normalization": "S_row_sum"},
        "idt": {"enabled": True, "source": "precomputed", "precomputed": {
            "time": "time.npy", "T": "T.npy", "P": "P.npy", "Y": "Y.npy", "index": "index.npy"}},
        "ranking": {"target_species": ["C2H4", "O2", "OH", "H2O", "HO2", "CO2"]},
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    summary = run(config)
    assert summary["modes"]["idt"]["n_trajectories"] == 2
    r = np.load(tmp_path / "out" / "r_AB.npy")
    assert r.shape == (111, 111) and np.all(np.isfinite(r))


def test_cli_template_and_info(capsys, tmp_path):
    assert main(["template"]) == 0
    text = capsys.readouterr().out
    tpl = json.loads(text)
    assert tpl["idt"]["enabled"] is True and "fs" in tpl
    out_file = tmp_path / "t.json"
    assert main(["template", "-o", str(out_file)]) == 0
    assert json.loads(out_file.read_text(encoding="utf-8"))["mechanism"] == "mechanism.yaml"
    assert main(["info"]) == 0
    assert "cantera" in capsys.readouterr().out


def test_cli_reports_config_errors(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mechanism": "m.yaml", "idt": {"enabled": True, "sovler": {}}}), encoding="utf-8")
    assert main(["run", str(bad)]) == 2
    assert "sovler" in capsys.readouterr().err
    assert main(["run", str(tmp_path / "missing.json")]) == 2
