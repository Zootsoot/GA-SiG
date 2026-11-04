"""The linear-solver choice of the reactor network changes the run time, not the ignition delay time."""
import cantera as ct
import pytest

from gasig.config import ConfigError, validate_config
from gasig.reduce import evaluation_config
from gasig.solvers.idt import run_idt
from helpers_reduction import IDT, TARGETS, make_config

SOLVER = {"fuel": {"CH4": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79}, "reactor": "constant_volume",
          "end_time_s": 2.0e-3, "reference_species": "OH", "ignition_criterion": "reference_peak",
          "rtol": 1e-6, "atol": 1e-12, "max_steps": 200000}
MODES = ("adaptive", "adaptive_fast", "none")


@pytest.fixture(scope="module")
def taus(mech_path):
    out = {}
    for mode in MODES:
        traj = run_idt(mech_path, None, 1500.0, ct.one_atm, 1.0, dict(SOLVER, preconditioner=mode), label=mode)
        assert traj.meta["ignited"], mode
        out[mode] = traj.meta["tau_ign_s"]
    return out


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_gives_the_same_ignition_delay(taus, mode):
    """GMRES with or without the derivative approximations and the dense solver agree within 1 percent."""
    assert abs(taus[mode] - taus["adaptive"]) <= 1e-2 * taus["adaptive"]


def test_missing_key_means_adaptive(mech_path):
    """A solver block without the key (an older test or caller) keeps the earlier behaviour."""
    traj = run_idt(mech_path, None, 1500.0, ct.one_atm, 1.0, dict(SOLVER), label="no-key")
    assert traj.meta["ignited"]


def test_default_and_override(tmp_path, mech_path):
    """The solver default is adaptive; reduction.idt.preconditioner overrides it for the evaluation."""
    assert evaluation_config(make_config(tmp_path, mech_path))[0]["preconditioner"] == "adaptive"
    cfg = make_config(tmp_path, mech_path, reduction={"idt": {"preconditioner": "adaptive_fast"}})
    assert cfg["idt"]["solver"]["preconditioner"] == "adaptive"
    assert evaluation_config(cfg)[0]["preconditioner"] == "adaptive_fast"


def test_unknown_choice_is_rejected(tmp_path, mech_path):
    with pytest.raises(ConfigError):
        make_config(tmp_path, mech_path, reduction={"idt": {"preconditioner": "fast"}})
    with pytest.raises(ConfigError):
        validate_config({"mechanism": mech_path, "idt": {"enabled": True, "solver": dict(IDT, preconditioner="dense")},
                         "ranking": {"target_species": list(TARGETS)}}, tmp_path)
