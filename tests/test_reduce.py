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
from gasig.config import ConfigError, validate_config  # noqa: E402
from gasig.errors import IgnitionSolverError, InputError, MechanismError, MissingInputError  # noqa: E402
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


def test_adaptive_budget_counts_reduced_mechanisms_only(tmp_path, mech_path, out, caplog):
    cfg = make_config(tmp_path, mech_path, reduction={"min_species": 95, "error_tolerance": 0.05,
                                                      "search": {"mode": "adaptive", "coarse_points": 4,
                                                                 "importance_gap_probes": 0, "max_evaluations": 2}})
    with caplog.at_level(logging.WARNING, logger="gasig.reduce"):
        summary = sweep(cfg, out)
    assert [r["n_species"] for r in summary["rows"]] == [111, 102, 97]     # two reduced sizes, as configured
    assert summary["search"]["evaluations"] == 2 and summary["search"]["budget_exhausted"] is True
    assert "evaluation budget of 2" in caplog.text


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

    def fake_fs(gas, conditions, fs_cfg, label, on_failure="inf"):
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


def test_solver_failure_counts_as_a_failed_condition(tmp_path, mech_path, out, monkeypatch):
    """A Cantera failure on a reduced mechanism does not abort the sweep; the detailed mechanism must integrate."""
    import gasig.reduce as module

    real_run_idt = module.run_idt

    def flaky(mech_file, phase_name, T0, P0, phi, solver_cfg, label="", gas=None):
        if label.startswith("n61"):
            raise ct.CanteraError("CVodes error encountered. Error code: -4")
        return real_run_idt(mech_file, phase_name, T0, P0, phi, solver_cfg, label=label, gas=gas)

    monkeypatch.setattr(module, "run_idt", flaky)
    cfg = make_config(tmp_path, mech_path, reduction={"step": 50, "min_species": 11, "error_tolerance": 0.05})
    summary = sweep(cfg, out)
    rows = summary["rows"]
    assert [r["n_species"] for r in rows] == [111, 61, 11]
    assert rows[1]["n_solver_failures"] == 1 and rows[1]["max_idt_error"] == float("inf")
    assert rows[1]["n_unignited"] == 0 and rows[1]["complete"] is True
    assert rows[2]["n_solver_failures"] == 0 and rows[2]["n_unignited"] == 1
    assert summary["status"] == "complete" and summary["selection"]["selected"] is None
    with (out / "reduction_sweep.csv").open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows[1]["n_solver_failures"] == "1" and csv_rows[1]["max_idt_error"] == "inf"
    assert summary["failures"] == {"ignition solver": 1}
    data = json.loads((out / "reduction_sweep.json").read_text(encoding="utf-8"))
    assert data["failures"] == {"ignition solver": 1}

    def broken_reference(*args, **kwargs):
        raise ct.CanteraError("boom")

    # A failure of the detailed mechanism is fatal; the error names the condition and keeps the Cantera cause.
    monkeypatch.setattr(module, "run_idt", broken_reference)
    with pytest.raises(IgnitionSolverError, match="integrator failed at T0=1500 K") as info:
        sweep(cfg, out)
    assert isinstance(info.value.__cause__, ct.CanteraError)
    assert info.value.stage == "reduce" and info.value.context["label"] == "full-c0"


def test_reference_species_that_is_not_formed_counts_as_a_failed_condition(tmp_path, mech_path, out, monkeypatch):
    """run_idt raises a ConfigError when the reactor ignites and the reference species peaks at t = 0.

    For a reduced mechanism this means that no kept reaction forms the reference species: the condition
    counts as failed and the sweep continues. For the detailed mechanism the error stays fatal.
    """
    import gasig.reduce as module

    real_run_idt = module.run_idt

    def no_reference(label):
        return ConfigError(f"{label}: the reactor ignited (temperature rise 1245 K) but the reference species "
                           "'OH' peaks at the first sample; it is a reactant or a diluent.",
                           label=label, species="OH", temperature_rise_K=1245.0)

    def lost_reference(mech_file, phase_name, T0, P0, phi, solver_cfg, label="", gas=None):
        if label.startswith("n61"):
            raise no_reference(label)
        return real_run_idt(mech_file, phase_name, T0, P0, phi, solver_cfg, label=label, gas=gas)

    monkeypatch.setattr(module, "run_idt", lost_reference)
    cfg = make_config(tmp_path, mech_path, reduction={"step": 50, "min_species": 11, "error_tolerance": 0.05})
    summary = sweep(cfg, out)
    rows = summary["rows"]
    assert [r["n_species"] for r in rows] == [111, 61, 11]                 # the sweep continues after 61
    assert rows[1]["n_solver_failures"] == 1 and rows[1]["max_idt_error"] == float("inf")
    assert rows[1]["n_unignited"] == 0 and rows[1]["complete"] is True
    assert summary["status"] == "complete" and summary["failures"] == {"ignition solver": 1}

    def fatal(mech_file, phase_name, T0, P0, phi, solver_cfg, label="", gas=None):
        raise no_reference(label)

    monkeypatch.setattr(module, "run_idt", fatal)
    with pytest.raises(ConfigError, match="peaks at the first sample"):
        sweep(cfg, out)


def test_run_reduce_records_solver_failures_in_the_failure_file(tmp_path, mech_path, gas, out, monkeypatch):
    import gasig.reduce as module

    real_run_idt = module.run_idt

    def flaky(mech_file, phase_name, T0, P0, phi, solver_cfg, label="", gas=None):
        if label.startswith("n61"):
            raise ct.CanteraError("CVodes error encountered. Error code: -4")
        return real_run_idt(mech_file, phase_name, T0, P0, phi, solver_cfg, label=label, gas=gas)

    monkeypatch.setattr(module, "run_idt", flaky)
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps({
        "mechanism": mech_path, "output_dir": "out", "idt": {"enabled": True, "solver": dict(IDT)},
        "ranking": {"target_species": TARGETS}, "reduction": {"step": 50, "min_species": 61}}), encoding="utf-8")
    run_reduce(cfg_path)
    lines = [json.loads(x) for x in (out / "reduce.failures.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["category"] == "ignition solver" and entry["stage"] == "reduce" and entry["fatal"] is False
    assert entry["context"]["label"] == "n61-c0" and entry["context"]["n_species"] == 61
    assert entry["context"]["T0_K"] == 1500.0 and "CVodes" in entry["message"]


def test_a_reduced_mechanism_that_cannot_be_built_counts_as_failed(tmp_path, mech_path, gas, out, monkeypatch):
    import gasig.reduce as module

    real = module.reduced_solution

    def fragile(gas_full, keep, transport_model=None):
        if len(keep) == 61:
            raise ct.CanteraError("a broken reaction")
        return real(gas_full, keep, transport_model)

    monkeypatch.setattr(module, "reduced_solution", fragile)
    cfg = make_config(tmp_path, mech_path, reduction={"step": 50, "min_species": 11, "error_tolerance": 0.05})
    summary = sweep(cfg, out)
    row = summary["rows"][1]
    assert row["n_species"] == 61 and row["build_failed"] is True and row["max_idt_error"] == float("inf")
    assert row["complete"] is False and summary["failures"] == {"mechanism": 1}


def test_json_is_written_after_every_size(tmp_path, mech_path, out, monkeypatch):
    """A crash late in the sweep leaves the rows so far in reduction_sweep.json with status running."""
    import gasig.reduce as module

    real_run_idt = module.run_idt

    def dies_on_reduced(mech_file, phase_name, T0, P0, phi, solver_cfg, label="", gas=None):
        if label.startswith("n"):
            raise RuntimeError("the machine went away")
        return real_run_idt(mech_file, phase_name, T0, P0, phi, solver_cfg, label=label, gas=gas)

    monkeypatch.setattr(module, "run_idt", dies_on_reduced)
    cfg = make_config(tmp_path, mech_path)
    with pytest.raises(RuntimeError, match="went away"):
        sweep(cfg, out)
    data = json.loads((out / "reduction_sweep.json").read_text(encoding="utf-8"))
    assert data["status"] == "running" and [r["n_species"] for r in data["rows"]] == [111]
    assert data["species_order"] and data["removal_order"] and "selection" not in data
    assert data["search"]["evaluations"] == 0


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


def test_greedy_mode_keeps_the_breaker_and_continues(tmp_path, mech_path, gas, out):
    """CH3 cannot leave a methane mechanism: the greedy pass keeps it and removes the harmless species around it."""
    spectators = [s for s in reversed(gas.species_names) if s not in PROTECTED and s != "CH3"][:2]
    rest = [s for s in gas.species_names if s not in PROTECTED and s not in spectators and s != "CH3"]
    write_rank(out, sorted(PROTECTED) + [spectators[0], "CH3", spectators[1]] + rest)
    cfg = make_config(tmp_path, mech_path, reduction={"min_species": 109, "error_tolerance": 0.05,
                                                      "search": {"mode": "greedy"}})
    summary = sweep(cfg, out)
    rows = summary["rows"]
    assert [r["n_species"] for r in rows] == [111, 110, 109, 109]
    assert [r.get("kept_back") for r in rows] == [None, False, True, False]
    assert [r["removed_in_step"] for r in rows[1:]] == [spectators[0], "CH3", spectators[1]]
    assert rows[2]["max_idt_error"] > 0.05
    assert rows[1]["max_idt_error"] < 0.01 and rows[3]["max_idt_error"] < 0.01
    assert summary["removal_order"] == spectators and summary["kept_back_species"] == ["CH3"]
    assert summary["rank_removal_order"][:3] == [spectators[0], "CH3", spectators[1]]
    search = summary["search"]
    assert search["mode"] == "greedy" and search["early_exit"] is True and search["evaluations"] == 3
    assert search["accepted"] == 2 and search["kept_back"] == 1 and search["budget_exhausted"] is False
    assert search["robust_window_applies"] is False and search["unresolved"] == []
    sel = summary["selection"]
    assert (sel["strict_optimum"], sel["robust_optimum"], sel["selected"]) == (109, 109, 109)
    assert summary["smallest_mechanism_within"] == {"5%": 109, "10%": 109, "20%": 109}
    reloaded = ct.Solution(sel["mechanism_file"])
    assert reloaded.n_species == 109 and "CH3" in reloaded.species_names
    assert not set(spectators) & set(reloaded.species_names)
    with (out / "reduction_sweep.csv").open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert [r["kept_back"] for r in csv_rows] == ["", "False", "True", "False"]
    assert csv_rows[2]["removed_in_step"] == "CH3"
    data = json.loads((out / "reduction_sweep.json").read_text(encoding="utf-8"))
    assert data["removal_order"] == spectators and data["kept_back_species"] == ["CH3"]


def test_greedy_mode_needs_a_tolerance(tmp_path, mech_path):
    with pytest.raises(ConfigError, match="greedy needs"):
        make_config(tmp_path, mech_path, reduction={"search": {"mode": "greedy"}})


def _breaker_rank(gas, out):
    """A rank with two harmless heavy species around CH3; returns the two harmless species."""
    spectators = [s for s in reversed(gas.species_names) if s not in PROTECTED and s != "CH3"][:2]
    rest = [s for s in gas.species_names if s not in PROTECTED and s not in spectators and s != "CH3"]
    write_rank(out, sorted(PROTECTED) + [spectators[0], "CH3", spectators[1]] + rest)
    return spectators


def test_greedy_budget_stops_the_pass(tmp_path, mech_path, gas, out, caplog):
    spectators = _breaker_rank(gas, out)
    cfg = make_config(tmp_path, mech_path, reduction={"min_species": 100, "error_tolerance": 0.05,
                                                      "search": {"mode": "greedy", "max_evaluations": 1}})
    with caplog.at_level(logging.WARNING, logger="gasig.reduce"):
        summary = sweep(cfg, out)
    assert [r["n_species"] for r in summary["rows"]] == [111, 110]
    search = summary["search"]
    assert search["budget_exhausted"] is True and search["evaluations"] == 1
    assert search["params"] == {"max_evaluations": 1}
    assert "evaluation budget of 1" in caplog.text
    assert summary["selection"]["selected"] == 110 and summary["removal_order"] == spectators[:1]


def test_greedy_skips_the_flames_of_a_kept_back_test(tmp_path, mech_path, gas, out, monkeypatch):
    """The verdict comes before the flames, so a kept-back test costs no flame solve."""
    import gasig.reduce as module

    calls = []

    def fake_fs(gas_red, conditions, fs_cfg, label, on_failure="inf"):
        calls.append(label)
        return [0.4] * len(conditions)

    monkeypatch.setattr(module, "_fs_values", fake_fs)
    spectators = _breaker_rank(gas, out)
    cfg = make_config(tmp_path, mech_path, reduction={
        "min_species": 109, "error_tolerance": 0.05, "evaluate_fs": True, "search": {"mode": "greedy"},
        "fs": {"fuel": {"H2": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79}, "P_atm": [1.0], "phi": [1.0]}})
    summary = sweep(cfg, out)
    rows = summary["rows"]
    assert calls == ["full", f"g110-{spectators[0]}", f"g109-{spectators[1]}"]
    assert rows[2]["kept_back"] is True and rows[2]["max_fs_error"] is None
    assert rows[1]["max_fs_error"] == 0.0 and rows[3]["max_fs_error"] == 0.0


# ----------------------------------------------------------------------------- lk_greedy mode

# Single-removal IDT errors at 1500 K, 1 atm, phi 1 (USC Mech II): the four aromatics are at the solver noise
# (below 2e-5), C2H3 gives 1.85 %, CH3 stops the ignition.
LKG_SPECIES = ["C6H5CHO", "C6H5CH2", "C2H3", "CH3", "HOC6H4CH3", "C6H5O"]
LKG_SCORES = {"C6H5CHO": 0.0, "C6H5CH2": 0.001, "C2H3": 0.01, "CH3": 1.0, "HOC6H4CH3": 0.02, "C6H5O": 0.5}


def write_lk_order(folder, gas, scores=LKG_SCORES):
    """``lk_order.csv``: the scored species in their order, then every other species with score 1."""
    order = list(scores) + [s for s in gas.species_names if s not in scores]
    with (folder / "lk_order.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank_from_least_important", "species", "importance", "gasig_importance"])
        for i, s in enumerate(order, start=1):
            writer.writerow([i, s, scores.get(s, 1.0), 0.0])


def lkg_config(tmp_path, mech_path, gas, units=(("C6H5O", "HOC6H4CH3"),), **search):
    keep = [s for s in gas.species_names if s not in LKG_SPECIES]
    return make_config(tmp_path, mech_path, reduction={
        "error_tolerance": 0.01, "keep_species": keep,
        "search": {"mode": "lk_greedy", "pass2_tolerance": 0.05,
                   "units": None if units is None else [list(u) for u in units], **search}})


def test_lk_greedy_mode_blocks_units_and_second_pass(tmp_path, mech_path, gas, out):
    write_lk_order(out, gas)
    summary = sweep(lkg_config(tmp_path, mech_path, gas), out)
    rows = summary["rows"]
    assert [r["n_species"] for r in rows] == [111, 109, 108, 108, 107, 106, 105]
    assert [r.get("kept_back") for r in rows] == [None, False, True, True, False, False, True]
    assert [r.get("pass") for r in rows[1:]] == [1, 1, 1, 1, 2, 2]
    assert [r.get("test") for r in rows[1:]] == ["block", "single", "single", "unit", "single", "single"]
    assert [r["removed_in_step"] for r in rows[1:]] == ["C6H5CHO;C6H5CH2", "C2H3", "CH3", "HOC6H4CH3;C6H5O",
                                                       "C2H3", "CH3"]
    assert 0.01 < rows[2]["max_idt_error"] < 0.05 and rows[2]["tolerance"] == 0.01     # fails at 1 %
    assert rows[5]["max_idt_error"] < 0.05 and rows[5]["tolerance"] == 0.05 and rows[5]["complete"] is True
    assert summary["removal_order"] == ["C6H5CHO", "C6H5CH2", "HOC6H4CH3", "C6H5O", "C2H3"]
    assert summary["kept_back_species"] == ["CH3"]
    search = summary["search"]
    assert search["mode"] == "lk_greedy" and search["early_exit"] is True and search["evaluations"] == 6
    assert search["tier0_size"] == 2 and search["units"] == [["C6H5O", "HOC6H4CH3"]]
    assert search["units_source"] == "reduction.search.units" and search["budget_exhausted"] is False
    p1, p2 = search["passes"]
    assert (p1["n_species"], p1["accepted"], p1["kept_pieces"]) == (107, 4, [["C2H3"], ["CH3"]])
    assert (p2["n_species"], p2["accepted"], p2["kept_pieces"], p2["tolerance"]) == (106, 1, [["CH3"]], 0.05)
    sel = summary["selection"]
    assert (sel["strict_optimum"], sel["robust_optimum"], sel["selected"]) == (107, 107, 107)
    first = ct.Solution(sel["mechanism_file"])
    second = ct.Solution(p2["mechanism_file"])
    assert first.n_species == 107 and "C2H3" in first.species_names and "C6H5O" not in first.species_names
    assert second.n_species == 106 and "C2H3" not in second.species_names and "CH3" in second.species_names
    with (out / "reduction_sweep.csv").open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows[1]["removed_in_step"] == "C6H5CHO;C6H5CH2" and csv_rows[3]["kept_back"] == "True"
    data = json.loads((out / "reduction_sweep.json").read_text(encoding="utf-8"))
    assert data["search"]["passes"][1]["mechanism_file"] == p2["mechanism_file"]
    assert data["rank_removal_order"] == LKG_SPECIES
    # A budget that ends with pass 1 runs no second pass and keeps the pass-1 failures in kept_back_species.
    summary = sweep(lkg_config(tmp_path, mech_path, gas, max_evaluations=4), out)
    assert summary["search"]["budget_exhausted"] is True and len(summary["search"]["passes"]) == 1
    assert summary["kept_back_species"] == ["C2H3", "CH3"] and summary["search"]["evaluations"] == 4


def test_lk_greedy_reads_the_detected_units_and_checks_its_inputs(tmp_path, mech_path, gas, out, caplog):
    cfg = lkg_config(tmp_path, mech_path, gas, units=None, max_evaluations=1)
    with pytest.raises(MissingInputError, match="lk_order.csv is missing"):
        sweep(cfg, out)
    write_lk_order(out, gas)
    with caplog.at_level(logging.WARNING, logger="gasig.reduce"):
        summary = sweep(cfg, out)
    assert summary["search"]["units_source"] == "none" and "no chain units" in caplog.text
    (out / "chain_units.json").write_text(json.dumps({"units": [["C6H5O", "HOC6H4CH3"], ["CH4", "C2H3"]]}),
                                          encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="gasig.reduce"):
        summary = sweep(cfg, out)
    assert summary["search"]["units"] == [["C6H5O", "HOC6H4CH3"]]          # the unit with protected CH4 is dropped
    assert summary["search"]["units_source"].endswith("chain_units.json") and "protected member" in caplog.text
    assert summary["search"]["budget_exhausted"] is True and len(summary["search"]["passes"]) == 1
    (out / "chain_units.json").write_text(json.dumps({"units": [["NOPE", "C2H3"]]}), encoding="utf-8")
    with pytest.raises(MechanismError, match="NOPE"):
        sweep(cfg, out)
    for bad, match in (({"units": ["C2H3"]}, "non-empty lists"), ({"units": [["C2H3", "C2H3"]]}, "more than one"),
                       ({"units": [["C2H3", "CH3"], ["CH3", "C6H5O"]]}, "more than one")):
        (out / "chain_units.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(InputError, match=match):
            sweep(cfg, out)
    with (out / "lk_order.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["species"]] + [[s] for s in gas.species_names])
    with pytest.raises(InputError, match="LK scores"):
        sweep(lkg_config(tmp_path, mech_path, gas), out)


def test_lk_greedy_config_checks(tmp_path, mech_path):
    with pytest.raises(ConfigError, match="lk_greedy needs"):
        make_config(tmp_path, mech_path, reduction={"search": {"mode": "lk_greedy"}})
    with pytest.raises(ConfigError, match="pass2_tolerance"):
        make_config(tmp_path, mech_path, reduction={"error_tolerance": 0.1, "search": {"pass2_tolerance": 0.05}})
    with pytest.raises(ConfigError, match="more than one place"):
        make_config(tmp_path, mech_path, reduction={"search": {"units": [["A", "B"], ["B", "C"]]}})
    with pytest.raises(ConfigError, match="non-empty lists"):
        make_config(tmp_path, mech_path, reduction={"search": {"units": [[]]}})
    with pytest.raises(ConfigError, match="block_size"):
        make_config(tmp_path, mech_path, reduction={"search": {"block_size": 0}})
    for extra in ({"lk": {"output_file": "ranked_species.csv"}}, {"units": {"output_file": "sub/units.json"}},
                  {"lk": {"output_file": "x.csv"}, "units": {"output_file": "x.csv"}}, {"lk": {"output_file": " "}}):
        with pytest.raises(ConfigError, match="output_file"):
            validate_config({"mechanism": mech_path, "idt": {"enabled": True, "solver": dict(IDT)}, **extra}, tmp_path)
    # The plain greedy mode may walk the LK order.
    make_config(tmp_path, mech_path, reduction={"ranking_file": "lk_order.csv", "error_tolerance": 0.1,
                                                "search": {"mode": "greedy"}})
