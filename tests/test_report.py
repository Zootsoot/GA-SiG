import importlib.util
import json

import pytest

ct = pytest.importorskip("cantera")

from helpers_reduction import rank_order, write_rank, make_config  # noqa: E402
from gasig.cli import main  # noqa: E402
from gasig.reduce import sweep  # noqa: E402
from gasig.report import build_report  # noqa: E402

HAS_MPL = importlib.util.find_spec("matplotlib") is not None


@pytest.fixture(scope="module")
def gas(mech_path):
    ct.suppress_thermo_warnings()
    return ct.Solution(mech_path)


@pytest.fixture
def swept(tmp_path, mech_path, gas):
    """A full sweep with three sizes (111, 61, 11) and a tolerance of 5 percent."""
    out = tmp_path / "out"
    out.mkdir()
    write_rank(out, rank_order(gas))
    cfg = make_config(tmp_path, mech_path, reduction={"step": 50, "min_species": 11, "error_tolerance": 0.05},
                      report={"T_points": 3}, end_time_s=0.02)
    sweep(cfg, out)
    data = json.loads((out / "reduction_sweep.json").read_text(encoding="utf-8"))
    return cfg, out, data


def test_report_compares_the_selected_mechanism(swept):
    cfg, out, data = swept
    result = build_report(cfg, out, data)
    assert result["selected"]["n_species"] == 61
    assert result["strict_optimum"]["n_species"] == 61 and result["robust_optimum"]["n_species"] == 61
    assert result["mechanism_file"].endswith("reduced_61sp.yaml")
    assert [round(e["T_K"]) for e in result["idt"]] == [1350, 1500, 1650]        # one temperature spans 0.9 T to 1.1 T
    assert all(e["error"] is not None and e["error"] < 0.05 for e in result["idt"])
    assert result["flame_speed"] == []
    report_dir = out / "report"
    text = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "| Selected mechanism | 61 species" in text and "Table 3" in text
    saved = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert saved["selected"]["n_species"] == 61 and len(saved["idt"]) == 3
    if HAS_MPL:
        assert (report_dir / "error_vs_species.png").is_file()
        assert (report_dir / "idt_comparison.png").is_file()
        assert not (report_dir / "flame_speed_comparison.png").exists()


def test_report_survives_a_solver_failure_on_a_grid_point(swept, monkeypatch, caplog):
    import logging

    import gasig.report as module

    real_run_idt = module.run_idt

    def flaky(mech_file, phase_name, T0, P0, phi, solver_cfg, label="", gas=None):
        if label.startswith("n61-T1350"):
            raise ct.CanteraError("CVodes error encountered. Error code: -4")
        return real_run_idt(mech_file, phase_name, T0, P0, phi, solver_cfg, label=label, gas=gas)

    monkeypatch.setattr(module, "run_idt", flaky)
    cfg, out, data = swept
    with caplog.at_level(logging.WARNING, logger="gasig.report"):
        result = build_report(cfg, out, data)
    failed = [e for e in result["idt"] if round(e["T_K"]) == 1350]
    assert failed and failed[0]["idt_reduced_s"] is None and failed[0]["error"] is None
    assert failed[0]["idt_detailed_s"] is not None
    assert "the integrator failed" in caplog.text
    assert (out / "report" / "report.md").is_file()


def test_report_without_a_mechanism_within_tolerance(swept, tmp_path, mech_path):
    cfg, out, data = swept
    tight = make_config(tmp_path, mech_path, reduction={"step": 50, "min_species": 11, "error_tolerance": 1e-9},
                        report={"T_points": 3}, end_time_s=0.02)
    result = build_report(tight, out, data)
    assert result["selected"] is None and result["idt"] == []
    text = (out / "report" / "report.md").read_text(encoding="utf-8")
    assert "none: no evaluated size is within the tolerance" in text


def test_report_cli_errors(tmp_path, mech_path, gas):
    out = tmp_path / "out"
    out.mkdir()
    cfg_path = tmp_path / "c.json"
    base = {"mechanism": mech_path, "output_dir": "out",
            "idt": {"enabled": True, "solver": {"fuel": {"CH4": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79},
                                                "T_K": [1500.0], "P_atm": [1.0], "phi": [1.0]}}}
    cfg_path.write_text(json.dumps(base), encoding="utf-8")
    assert main(["report", str(cfg_path)]) == 2                      # no sweep yet
    write_rank(out, rank_order(gas))
    (out / "reduction_sweep.json").write_text(json.dumps({"rows": [], "n_reactions_full": 784}), encoding="utf-8")
    assert main(["report", str(cfg_path)]) == 2                      # no tolerance set
    base["reduction"] = {"error_tolerance": 0.05}
    cfg_path.write_text(json.dumps(base), encoding="utf-8")
    assert main(["report", str(cfg_path)]) == 3                      # old sweep file without the species order
    assert "Traceback" in (out / "report.log").read_text(encoding="utf-8")


def test_report_of_a_mechanism_file_without_a_sweep(tmp_path, mech_path, gas):
    """``gasig report --mechanism FILE`` compares the file with the detailed mechanism on the evaluation grid."""
    from helpers_reduction import IDT, PROTECTED, TARGETS
    from gasig.reduce import write_reduced_mechanism

    out = tmp_path / "out"
    out.mkdir()
    removed = set([s for s in reversed(gas.species_names) if s not in PROTECTED][:50])
    keep = [s for s in gas.species_names if s not in removed]
    path = write_reduced_mechanism(gas, keep, tmp_path / "reduced_61sp.yaml")
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps({"mechanism": mech_path, "output_dir": "out",
                                    "idt": {"enabled": True, "solver": dict(IDT, end_time_s=0.02)},
                                    "ranking": {"target_species": TARGETS}, "report": {"T_points": 3}}),
                        encoding="utf-8")
    assert main(["report", str(cfg_path), "--mechanism", str(path)]) == 0
    report_dir = out / "report_reduced_61sp"
    saved = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert saved["selected"] == {"n_species": 61, "n_reactions": ct.Solution(str(path)).n_reactions}
    assert saved["n_species_full"] == 111 and saved["species_not_in_detailed"] == []
    assert [round(e["T_K"]) for e in saved["idt"]] == [1350, 1500, 1650] and saved["flame_speed"] == []
    assert all(e["error"] is not None and e["error"] < 0.05 for e in saved["idt"])
    text = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "# Mechanism comparison: reduced_61sp.yaml" in text and "| Compared | `reduced_61sp.yaml` | 61 |" in text
    assert "Table 2. Ignition delay time" in text and "The mean error over the 3 points" in text
    assert (out / "report_reduced_61sp.log").is_file()
    if HAS_MPL:
        assert (report_dir / "idt_comparison.png").is_file()
    assert main(["report", str(cfg_path), "--mechanism", str(tmp_path / "missing.yaml")]) == 2


def test_report_after_a_greedy_sweep(tmp_path, mech_path, gas):
    """The report rebuilds the selected size from the accepted removals, not from the rank order."""
    from helpers_reduction import PROTECTED

    out = tmp_path / "out"
    out.mkdir()
    spectators = [s for s in reversed(gas.species_names) if s not in PROTECTED and s != "CH3"][:2]
    rest = [s for s in gas.species_names if s not in PROTECTED and s not in spectators and s != "CH3"]
    write_rank(out, sorted(PROTECTED) + [spectators[0], "CH3", spectators[1]] + rest)
    cfg = make_config(tmp_path, mech_path, reduction={"min_species": 109, "error_tolerance": 0.05,
                                                      "search": {"mode": "greedy"}},
                      report={"T_points": 3}, end_time_s=0.02)
    sweep(cfg, out)
    data = json.loads((out / "reduction_sweep.json").read_text(encoding="utf-8"))
    result = build_report(cfg, out, data)
    assert result["selected"]["n_species"] == 109 and result["search_mode"] == "greedy"
    assert result["selected"]["max_idt_error"] is not None and result["selected"]["max_idt_error"] < 0.01
    assert "CH3" in result["species"] and not set(spectators) & set(result["species"])
    assert result["mechanism_file"] == data["selection"]["mechanism_file"]
    assert all(point["error"] < 0.01 for point in result["idt"])
    text = (out / "report" / "report.md").read_text(encoding="utf-8")
    assert "kept back" in text
