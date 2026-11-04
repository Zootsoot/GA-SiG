"""The error classes, the failure file and the error output of the command line."""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import gasig
from gasig import errors
from gasig.errors import (ConfigError, FlameSolverError, GasigError, IgnitionSolverError, InputError,
                          IntegrationError, MechanismError, MissingInputError, ReductionError, SolverError,
                          SolverTimeout)
from gasig.logs import log_failure, setup_logging, teardown_logging


def test_every_class_keeps_its_built_in_base_and_exit_code():
    expected = {
        ConfigError: (ValueError, 2), InputError: (ValueError, 3), MissingInputError: (FileNotFoundError, 2),
        MechanismError: (ValueError, 3), errors.MechanismCheckError: (RuntimeError, 3), SolverError: (RuntimeError, 3), IgnitionSolverError: (RuntimeError, 3),
        FlameSolverError: (RuntimeError, 3), SolverTimeout: (RuntimeError, 3), IntegrationError: (RuntimeError, 3),
        ReductionError: (RuntimeError, 3),
    }
    for cls, (base, code) in expected.items():
        exc = cls("text")
        assert isinstance(exc, GasigError) and isinstance(exc, base), cls
        assert errors.exit_code(exc) == code and str(exc) == "text"
    assert isinstance(MissingInputError("x"), InputError) and isinstance(MissingInputError("x"), ValueError)
    assert issubclass(SolverTimeout, FlameSolverError) and issubclass(IgnitionSolverError, SolverError)
    assert errors.exit_code(FileNotFoundError("x")) == 2 and errors.exit_code(KeyError("x")) == 3


def test_the_old_import_paths_give_the_same_classes():
    from gasig.config import ConfigError as config_error
    from gasig.solvers.flame import FlameSolverError as flame_error
    from gasig.solvers.zerork import SolverTimeout as timeout_error

    assert config_error is ConfigError and flame_error is FlameSolverError and timeout_error is SolverTimeout


def test_describe_keeps_stage_and_json_ready_context():
    exc = IgnitionSolverError("boom", stage="reduce", T0_K=800.0, phi=float("inf"), skip=None, names=("A", "B"))
    info = exc.describe()
    assert info == {"category": "ignition solver", "type": "IgnitionSolverError", "message": "boom",
                    "stage": "reduce", "context": {"T0_K": 800.0, "phi": None, "names": ["A", "B"]}}
    json.dumps(info, allow_nan=False)
    assert errors.describe_exception(KeyError("k"))["category"] == "unexpected"


def test_failure_log_writes_one_json_line_per_failure(tmp_path):
    path = tmp_path / "x.failures.jsonl"
    log = errors.open_failure_log(path, "w")
    try:
        errors.record_failure(FlameSolverError("no flame", label="f0"), stage="reduce", P_atm=1.0)
        errors.record_failure(RuntimeError("other"), fatal=True)
        assert errors.failure_counts() == {"flame solver": 1, "unexpected": 1}
    finally:
        errors.close_failure_log()
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [e["category"] for e in lines] == ["flame solver", "unexpected"]
    assert lines[0]["context"] == {"label": "f0", "P_atm": 1.0} and lines[0]["stage"] == "reduce"
    assert lines[0]["fatal"] is False and lines[1]["fatal"] is True
    assert log.counts["flame solver"] == 1
    errors.open_failure_log(None)                        # back to counting without a file
    errors.record_failure(InputError("x"))
    assert errors.failure_counts() == {"input": 1}


def test_setup_logging_opens_the_failure_file_and_log_failure_records_fatal_errors(tmp_path):
    setup_logging(tmp_path, "info", file_name="reduce.log", mode="w")
    try:
        try:
            raise MechanismError("species X is not in the mechanism", stage="reduce", species=["X"])
        except MechanismError:
            log_failure(logging.getLogger("gasig.test"), "the sweep failed")
    finally:
        teardown_logging()
    entry = json.loads((tmp_path / "reduce.failures.jsonl").read_text(encoding="utf-8"))
    assert entry["fatal"] is True and entry["category"] == "mechanism" and entry["context"]["species"] == ["X"]
    assert entry["context"]["during"] == "the sweep failed"
    text = (tmp_path / "reduce.log").read_text(encoding="utf-8")
    assert "the sweep failed (mechanism error): species X" in text and "Traceback" in text
    setup_logging(tmp_path, "info", file_name="reduce.log", mode="w")      # a new sweep starts a new file
    teardown_logging()
    assert (tmp_path / "reduce.failures.jsonl").read_text(encoding="utf-8") == ""


def test_cli_prints_the_category_stage_and_context(monkeypatch, capsys):
    from gasig import cli

    def fail(args):
        raise MechanismError("bad species", stage="reduce", species=["ZZZ"])

    monkeypatch.setattr(cli, "_cmd_reduce", fail)
    assert cli.main(["reduce", "unused.json"]) == 3
    err = capsys.readouterr().err
    assert "Mechanism error in the reduce stage: bad species" in err and "species=['ZZZ']" in err
    assert "failures.jsonl" in err

    def missing(args):
        raise MissingInputError("x.csv is missing", path="x.csv")

    monkeypatch.setattr(cli, "_cmd_reduce", missing)
    assert cli.main(["reduce", "unused.json"]) == 2
    err = capsys.readouterr().err
    assert "Missing input error: x.csv is missing" in err and "GASIG_DEBUG" not in err


def test_errors_module_is_free_of_cantera_and_torch():
    root = Path(gasig.__file__).resolve().parents[1]
    code = ("import sys; sys.modules['torch'] = None; sys.modules['cantera'] = None; "
            "import gasig.errors, gasig.search; print('pure ok')")
    env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(root))
    assert result.returncode == 0 and "pure ok" in result.stdout, result.stderr


@pytest.fixture(autouse=True)
def _fresh_failure_log():
    errors.open_failure_log(None)
    yield
    errors.open_failure_log(None)
