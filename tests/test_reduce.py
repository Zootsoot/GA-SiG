import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ct = pytest.importorskip("cantera")

import gasig  # noqa: E402
from helpers_reduction import IDT, PROTECTED, TARGETS, make_config, rank_order, write_rank  # noqa: E402
from gasig.cli import main  # noqa: E402
from gasig.config import ConfigError  # noqa: E402
from gasig.logs import setup_logging, teardown_logging  # noqa: E402
from gasig.reduce import (_json_safe, _relative_errors, keep_list, read_importance, read_rank,  # noqa: E402
                          reduced_solution, run_reduce, sweep)


@pytest.fixture(scope="module")
def gas(mech_path):
    ct.suppress_thermo_warnings()
    return ct.Solution(mech_path)


@pytest.fixture
def out(tmp_path, gas):
    folder = tmp_path / "out"
    folder.mkdir()
    write_rank(folder, rank_order(gas))
    return folder


# ----------------------------------------------------------------------------- reduced_solution

def test_reduced_solution_identity(gas):
    g = reduced_solution(gas, list(gas.species_names))
    assert (g.n_species, g.n_reactions) == (gas.n_species, gas.n_reactions)


def test_inert_and_spectator_removal(gas):
    """AR takes part in one quenching reaction only; C6H5C2H in none. Rates without AR are unchanged."""
    keep = [s for s in gas.species_names if s not in ("AR", "C6H5C2H")]
    g = reduced_solution(gas, keep, transport_model="mixture-averaged")
    assert g.n_species == gas.n_species - 2
    assert g.n_reactions == gas.n_reactions - 1
    gone = {r.equation for r in gas.reactions()} - {r.equation for r in g.reactions()}
    assert gone == {"AR + CH2* <=> AR + CH2"}
    for x in (gas, g):
        x.TPX = 1500.0, ct.one_atm, "CH4:1, O2:2, N2:7.52"
    idx = [gas.species_index(s) for s in keep]
    w_full = gas.net_production_rates
    assert np.allclose(g.net_production_rates, w_full[idx], rtol=1e-12, atol=1e-12 * np.abs(w_full).max())


def test_removed_species_leaves_no_reaction_behind(gas):
    """H2O is a reactant, a product and an explicit collider in USC Mech II; every reaction with H2O goes."""
    keep = [s for s in gas.species_names if s != "H2O"]
    g = reduced_solution(gas, keep)
    equations = {r.equation for r in g.reactions()}
    assert "2 H + H2O <=> H2 + H2O" not in equations          # H2O as product and collider
    assert "HCO + H2O <=> CO + H + H2O" not in equations       # H2O as explicit collider only
    assert g.n_reactions == 698
    for r in g.reactions():
        assert "H2O" not in (set(r.reactants) | set(r.products))
    with pytest.raises(ValueError, match="not in the mechanism"):
        reduced_solution(gas, ["CH4", "NOPE"])


def test_explicit_collider_rule_keeps_live_reactions(gas):
    """A reaction with several explicit colliders stays while one collider is kept, as in Cantera."""
    species = [gas.species(s) for s in ("H", "O2", "HO2", "H2O", "AR", "N2")]
    base = ct.Solution(thermo="ideal-gas", kinetics="gas", species=species, reactions=[])
    rx = ct.Reaction.from_dict({"equation": "H + O2 + M <=> HO2 + M", "type": "three-body",
                                "rate-constant": {"A": 2.8e18, "b": -0.86, "Ea": 0.0},
                                "default-efficiency": 0.0, "efficiencies": {"H2O": 1.0, "AR": 0.5}}, base)
    small = ct.Solution(thermo="ideal-gas", kinetics="gas", species=species, reactions=[rx])
    assert small.n_reactions == 1
    without_ar = reduced_solution(small, ["H", "O2", "HO2", "H2O", "N2"])
    assert without_ar.n_reactions == 1
    without_ar.TPX = 1000.0, ct.one_atm, "H:0.1, O2:1, H2O:0.5, N2:3"
    assert without_ar.third_body_concentrations[0] > 0.0
    without_both = reduced_solution(small, ["H", "O2", "HO2", "N2"])
    assert without_both.n_reactions == 0


# ----------------------------------------------------------------------------- rank file and helpers

def test_read_rank_and_relative_errors(tmp_path):
    p = tmp_path / "ranked_species.csv"
    p.write_text("rank_from_least_important,species,importance\n1,A,0\n2,B,0.5\n", encoding="utf-8")
    assert read_rank(p) == ["A", "B"]
    assert read_importance(p) == {"A": 0.0, "B": 0.5}
    err = _relative_errors([1.0, float("inf"), 2.0, None], [1.0, 1.0, 1.0, 1.0])
    assert err[0] == 0.0 and np.isinf(err[1]) and err[2] == pytest.approx(1.0) and np.isnan(err[3])
    with pytest.raises(FileNotFoundError):
        read_rank(tmp_path / "missing.csv")
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'species' column"):
        read_rank(tmp_path / "bad.csv")
    assert read_importance(tmp_path / "bad.csv") == {}
    (tmp_path / "empty.csv").write_text("rank_from_least_important,species,importance\n", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no species"):
        read_rank(tmp_path / "empty.csv")
    (tmp_path / "twice.csv").write_text("rank_from_least_important,species,importance\n1,A,0\n2,A,0\n3,B,1\n",
                                        encoding="utf-8")
    with pytest.raises(ValueError, match="more than once"):
        read_rank(tmp_path / "twice.csv")
    assert keep_list(["a", "b", "c", "d"], ["d", "c", "b"], 2) == ["a", "b"]


def test_json_safe_replaces_non_finite_values():
    assert _json_safe({"a": [1.0, float("inf"), float("nan")], "b": 2, "c": "x"}) == \
        {"a": [1.0, None, None], "b": 2, "c": "x"}


def test_logging_state_is_restored(tmp_path):
    logger = logging.getLogger("gasig")
    before = (logger.propagate, logger.level)
    setup_logging(tmp_path, "debug", file_name="x.log", mode="w")
    assert logger.propagate is False and logger.level == logging.DEBUG
    teardown_logging()
    assert (logger.propagate, logger.level) == before
    assert not [h for h in logger.handlers if getattr(h, "_gasig", False)]


# ----------------------------------------------------------------------------- full sweep

def test_sweep_removes_heavy_species_with_small_error(tmp_path, mech_path, gas, out):
    """One step of 50 species in reverse mechanism order barely changes the methane IDT at 1500 K."""
    cfg = make_config(tmp_path, mech_path)
    summary = sweep(cfg, out)
    rows = summary["rows"]
    assert rows[0]["n_species"] == 111
    assert rows[0]["idt_s"][0] == pytest.approx(1.124e-3, rel=0.02)     # methane, 1500 K, 1 atm, phi 1
    assert [r["n_species"] for r in rows] == [111, 61]
    assert rows[1]["n_reactions"] == 439                                  # fixed by USC Mech II and this removal set
    assert 0.0 <= rows[1]["max_idt_error"] < 0.01
    assert rows[1]["complete"] is True
    # the protection filter: nothing protected leaves, and the batch starts at the first unprotected species
    order = rank_order(gas)
    assert set(summary["protected_species"]) == PROTECTED
    assert rows[1]["removed_in_step"].split(";") == [s for s in order if s not in PROTECTED][:50]
    assert all(PROTECTED.isdisjoint(r["removed_in_step"].split(";")) for r in rows[1:])
    assert summary["removal_order"] == [s for s in order if s not in PROTECTED]
    assert summary["species_order"] == list(gas.species_names)
    assert summary["smallest_mechanism_within"] == {"5%": 61, "10%": 61, "20%": 61}
    assert summary["selection"]["selected"] is None and summary["error_tolerance"] is None
    text = (out / "reduction_sweep.json").read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    data = json.loads(text)
    assert data["rows"][0]["max_fs_error"] is None
    assert data["reference_idt_s"] == rows[0]["idt_s"]
    with (out / "reduction_sweep.csv").open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 2 and csv_rows[1]["max_fs_error"] == "" and csv_rows[1]["complete"] == "True"


def test_last_batch_is_clamped_at_min_species(tmp_path, mech_path, out, caplog):
    cfg = make_config(tmp_path, mech_path, reduction={"step": 50, "min_species": 100})
    rows = sweep(cfg, out)["rows"]
    assert [r["n_species"] for r in rows] == [111, 100]
    assert len(rows[1]["removed_in_step"].split(";")) == 11
    cfg = make_config(tmp_path, mech_path, reduction={"step": 1, "min_species": 111})
    with caplog.at_level(logging.WARNING, logger="gasig.reduce"):
        rows = sweep(cfg, out)["rows"]
    assert len(rows) == 1 and "nothing to remove" in caplog.text


def test_stop_error_ends_the_sweep(tmp_path, mech_path, out):
    cfg = make_config(tmp_path, mech_path, reduction={"step": 50, "min_species": 1, "stop_error": 1e-9})
    rows = sweep(cfg, out)["rows"]
    assert [r["n_species"] for r in rows] == [111, 61]
    assert rows[1]["max_idt_error"] > 1e-9


def test_tolerance_selects_and_writes_the_reduced_mechanism(tmp_path, mech_path, out):
    """Sizes 111, 61 and 11: the 11-species mechanism has no carbon chemistry and does not ignite."""
    cfg = make_config(tmp_path, mech_path, reduction={"step": 50, "min_species": 11, "error_tolerance": 0.05})
    summary = sweep(cfg, out)
    rows = summary["rows"]
    assert [r["n_species"] for r in rows] == [111, 61, 11]
    assert rows[2]["n_unignited"] == 1 and rows[2]["max_idt_error"] == float("inf") and rows[2]["complete"] is True
    assert summary["smallest_mechanism_within"] == {"5%": 61, "10%": 61, "20%": 61}
    selection = summary["selection"]
    assert (selection["strict_optimum"], selection["robust_optimum"], selection["selected"]) == (61, 61, 61)
    path = Path(selection["mechanism_file"])
    assert path == out / "reduced_61sp.yaml" and path.is_file()
    reloaded = ct.Solution(str(path))
    assert (reloaded.n_species, reloaded.n_reactions) == (61, 439)
    assert reloaded.transport_model == "mixture-averaged"
    reloaded.TPX = 1500.0, ct.one_atm, "CH4:1, O2:2, N2:7.52"
    in_memory = reduced_solution(ct.Solution(mech_path), reloaded.species_names)
    in_memory.TPX = 1500.0, ct.one_atm, "CH4:1, O2:2, N2:7.52"
    assert np.allclose(reloaded.net_production_rates, in_memory.net_production_rates, rtol=1e-8,
                       atol=1e-10 * np.abs(in_memory.net_production_rates).max())
    data = json.loads((out / "reduction_sweep.json").read_text(encoding="utf-8"))
    assert data["rows"][2]["idt_s"] == [None] and data["selection"]["selected"] == 61


def test_early_exit_stops_at_the_first_failed_condition(tmp_path, mech_path, out):
    cfg = make_config(tmp_path, mech_path, reduction={"step": 100, "min_species": 11, "error_tolerance": 0.05,
                                                      "search": {"early_exit": True}})
    cfg["idt"]["solver"]["T_K"] = [1500.0, 1600.0]           # both ignite within 2 ms
    summary = sweep(cfg, out)
    rows = summary["rows"]
    assert [r["n_species"] for r in rows] == [111, 11]
    assert rows[0]["complete"] is True and len(rows[0]["idt_s"]) == 2
    assert rows[1]["complete"] is False
    assert rows[1]["idt_s"][0] == float("inf") and rows[1]["idt_s"][1] is None    # the second condition was skipped
    assert rows[1]["n_unignited"] == 1
    assert summary["search"] == {"mode": "full", "early_exit": True, "evaluations": 1}
    assert summary["selection"]["selected"] is None                                 # the detailed size is no candidate
    assert not list(out.glob("reduced_*sp.yaml"))
    with (out / "reduction_sweep.csv").open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows[1]["complete"] == "False" and csv_rows[1]["max_idt_error"] == "inf"


def test_adaptive_search_on_methane(tmp_path, mech_path, gas, out):
    """Between 111 and 95 species every size passes: coarse sizes, then the window of the smallest size."""
    cfg = make_config(tmp_path, mech_path, reduction={"min_species": 95, "error_tolerance": 0.05,
                                                      "search": {"mode": "adaptive", "coarse_points": 4,
                                                                 "importance_gap_probes": 2}})
    summary = sweep(cfg, out)
    rows = summary["rows"]
    assert [r["n_species"] for r in rows] == [111, 102, 98, 97, 96, 95]
    assert all(r["complete"] for r in rows) and all(r["max_idt_error"] < 0.01 for r in rows)
    search = summary["search"]
    assert search["mode"] == "adaptive" and search["early_exit"] is True and search["evaluations"] == 5
    assert search["probes"] == [] and search["unresolved"] == [] and search["budget_exhausted"] is False
    assert (summary["selection"]["strict_optimum"], summary["selection"]["robust_optimum"]) == (95, 95)
    assert (out / "reduced_95sp.yaml").is_file()
    order = rank_order(gas)
    removable = [s for s in order if s not in PROTECTED]
    assert rows[1]["removed_in_step"].split(";") == removable[:9]
    assert rows[2]["removed_in_step"].split(";") == removable[9:13]


def test_adaptive_search_probes_the_importance_gaps(tmp_path, mech_path, gas):
    out = tmp_path / "out"
    out.mkdir()
    order = rank_order(gas)
    removable = [s for s in order if s not in PROTECTED]
    importance = {s: 1.0 for s in order}
    for i, s in enumerate(removable):
        importance[s] = 1e-6 if i < 10 else 1e-2                          # one large gap after 10 removals
    write_rank(out, order, importance)
    cfg = make_config(tmp_path, mech_path, reduction={"min_species": 95, "error_tolerance": 0.05,
                                                      "search": {"mode": "adaptive", "coarse_points": 2,
                                                                 "importance_gap_probes": 1}})
    summary = sweep(cfg, out)
    assert summary["search"]["probes"] == [101, 100]
    assert {101, 100} <= {r["n_species"] for r in summary["rows"]}


def test_rank_file_must_match_the_mechanism(tmp_path, mech_path, gas, out):
    cfg = make_config(tmp_path, mech_path)
    write_rank(out, rank_order(gas) + ["ZZZ"])
    with pytest.raises(ValueError, match="not in the mechanism"):
        sweep(cfg, out)
    write_rank(out, [s for s in rank_order(gas) if s != "C6H5C2H"])
    with pytest.raises(ValueError, match="not in the rank file"):
        sweep(cfg, out)


def test_keep_species_and_targets_are_checked(tmp_path, mech_path, out, caplog):
    cfg = make_config(tmp_path, mech_path, reduction={"keep_species": ["NOPE"]})
    with pytest.raises(ValueError, match="NOPE"):
        sweep(cfg, out)
    cfg = make_config(tmp_path, mech_path, reduction={"step": 1, "min_species": 111, "keep_species": ["H2O2"]},
                      targets=TARGETS + ["NOPE"])
    with caplog.at_level(logging.WARNING, logger="gasig.reduce"):
        summary = sweep(cfg, out)
    assert "Target species 'NOPE'" in caplog.text
    assert set(summary["protected_species"]) == PROTECTED | {"H2O2"}


def test_evaluate_fs_needs_flame_conditions(tmp_path, mech_path, out):
    cfg = make_config(tmp_path, mech_path, reduction={"evaluate_fs": True})
    with pytest.raises(ConfigError, match="reduction.fs"):
        sweep(cfg, out)


def test_fs_mixture_is_protected(tmp_path, mech_path, out, monkeypatch):
    """The flame species join the protected set; the flame itself is stubbed (a real one takes minutes)."""
    import gasig.reduce as module

    calls = []

    def fake_fs(gas, conditions, fs_cfg, label):
        calls.append((label, len(conditions)))
        return [0.4] * len(conditions)

    monkeypatch.setattr(module, "_fs_values", fake_fs)
    cfg = make_config(tmp_path, mech_path, reduction={
        "evaluate_fs": True, "step": 1, "min_species": 111,
        "fs": {"fuel": {"H2": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79}, "P_atm": [1.0], "phi": [1.0]}})
    summary = sweep(cfg, out)
    assert "H2" in summary["protected_species"]
    assert calls == [("full", 1)]
    assert summary["rows"][0]["max_fs_error"] == 0.0
    assert summary["fs_conditions"] == [{"P_atm": pytest.approx(1.0), "phi": 1.0}]


def test_unignited_reference_is_rejected(tmp_path, mech_path, out):
    cfg = make_config(tmp_path, mech_path, reduction={"idt": {"end_time_s": 1e-7}})
    with pytest.raises(RuntimeError, match="do not ignite"):
        sweep(cfg, out)


# ----------------------------------------------------------------------------- run_reduce and CLI

def test_run_reduce_log_is_rewritten_and_holds_failures(tmp_path, mech_path, gas, out):
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps({
        "mechanism": mech_path, "output_dir": "out",
        "idt": {"enabled": True, "solver": dict(IDT)},
        "ranking": {"target_species": TARGETS},
        "reduction": {"step": 1, "min_species": 111}}), encoding="utf-8")
    run_reduce(cfg_path)
    run_reduce(cfg_path)
    log_text = (out / "reduce.log").read_text(encoding="utf-8")
    assert log_text.count("GA-SiG reduction sweep | config") == 1
    assert logging.getLogger("gasig").propagate is True
    write_rank(out, rank_order(gas) + ["ZZZ"])
    assert main(["reduce", str(cfg_path)]) == 3
    log_text = (out / "reduce.log").read_text(encoding="utf-8")
    assert "the reduction sweep failed" in log_text and "Traceback" in log_text and "ZZZ" in log_text


def test_reduce_cli_reports_missing_output(tmp_path, mech_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"mechanism": mech_path, "output_dir": "nope",
                               "idt": {"enabled": True, "source": "precomputed", "precomputed": {
                                   "time": "t.npy", "T": "T.npy", "P": "P.npy", "Y": "Y.npy", "index": "i.npy"}}}),
                   encoding="utf-8")
    assert main(["reduce", str(cfg)]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_reduce_imports_without_torch():
    """`gasig reduce`, `gasig report` and the solvers need Cantera and NumPy only."""
    root = Path(gasig.__file__).resolve().parents[1]
    code = ("import sys; sys.modules['torch'] = None; "
            "import gasig.reduce, gasig.report, gasig.search, gasig.cli, gasig.solvers, gasig.io, gasig.trajectory; "
            "print('torch-free ok')")
    env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(root))
    assert result.returncode == 0 and "torch-free ok" in result.stdout, result.stderr
