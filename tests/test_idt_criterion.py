import numpy as np
import pytest

pytest.importorskip("cantera")

from gasig.solvers.idt import ignition_sample  # noqa: E402

T_IGN = 2.0e-3          # heat-release peak
T_OH = 2.01e-3          # OH peak, slightly after the heat-release peak


def grid():
    """Dense around ignition, sparse afterwards, up to 2 s."""
    return np.concatenate([np.linspace(0.0, 1.9e-3, 200), np.linspace(1.9e-3, 2.1e-3, 401)[1:],
                           np.geomspace(2.1e-3, 2.0, 300)[1:]])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def profiles(drift=0.0, oh_peak=1.5e-2, oh_plateau=1.4e-2, T_rise=1900.0, T_overshoot=0.0, T_drift=0.0):
    """Sigmoid temperature rise at T_IGN; OH as a peak at T_OH on a plateau, plus a slow linear drift.

    The drift starts at the heat-release peak and is strictly increasing, so a profile without
    the peak term has no local maximum after ignition. ``T_overshoot`` adds a temperature peak
    at 2.05 ms that decays afterwards; ``T_drift`` makes the temperature rise until the end.
    """
    t = grid()
    late = np.clip((t - T_IGN) / (2.0 - T_IGN), 0.0, 1.0)
    T = (1200.0 + T_rise * sigmoid((t - T_IGN) / 1e-5) + T_overshoot * np.exp(-((t - 2.05e-3) / 2e-5) ** 2)
         + T_drift * late)
    ref = oh_peak * np.exp(-((t - T_OH) / 2e-5) ** 2) + oh_plateau * sigmoid((t - T_OH) / 1e-5) + drift * late
    return t, T, ref


def test_rule_1_normal_case_is_the_global_peak():
    t, T, ref = profiles()
    i, tau, note = ignition_sample(t, T, ref, "reference_peak")
    assert i == int(np.argmax(ref)) and note is None and tau == t[i]
    assert tau == pytest.approx(T_OH, abs=1e-5)


def test_rule_1_post_ignition_drift_takes_the_peak_inside_the_event():
    t, T, ref = profiles(drift=1.0e-2)        # the end value 2.4e-2 is above the ignition peak of 2.2e-2
    assert int(np.argmax(ref)) == t.shape[0] - 1
    i, tau, note = ignition_sample(t, T, ref, "reference_peak")
    assert tau == pytest.approx(T_OH, abs=1e-5) and tau == t[i]
    assert note is not None and "after the ignition event" in note


def test_rule_2_reference_grows_to_the_end_takes_the_temperature_peak():
    t, T, ref = profiles(drift=1.0e-2, oh_peak=0.0, T_overshoot=150.0)
    assert np.all(np.diff(ref[np.argmax(np.gradient(T, t)):]) > 0.0)
    assert int(np.argmax(T)) < t.shape[0] - 1
    i, tau, note = ignition_sample(t, T, ref, "reference_peak")
    assert tau == pytest.approx(2.05e-3, abs=1e-5) and i == int(np.argmax(T))
    assert note is not None and "temperature peak" in note


def test_rule_3_both_grow_to_the_end_takes_the_mean_of_the_steepest_rises():
    t, T, ref = profiles(drift=1.0e-2, oh_peak=0.0, T_drift=50.0)
    assert int(np.argmax(T)) == t.shape[0] - 1
    i, tau, note = ignition_sample(t, T, ref, "reference_peak")
    assert tau == pytest.approx(0.5 * (T_IGN + T_OH), abs=1e-5)
    assert abs(t[i] - tau) <= 1e-6                      # the nearest sample
    assert note is not None and "mean of their steepest rises" in note


def test_other_rules_are_unchanged():
    t, T, ref = profiles(drift=1.0e-2)
    i, tau, note = ignition_sample(t, T, ref, "max_dTdt")
    assert tau == pytest.approx(T_IGN, abs=1e-5) and note is None
    # no ignition: the temperature rise is below 100 K, so the global maximum stays, wherever it is
    t, T, ref = profiles(drift=1.0e-2, T_rise=10.0)
    i, tau, note = ignition_sample(t, T, ref, "reference_peak")
    assert i == t.shape[0] - 1 and note is None
    # a reference species that peaks at the first sample (a reactant) is reported as such
    t, T, ref = profiles()
    ref = ref.max() * 2.0 - ref
    i, tau, note = ignition_sample(t, T, ref, "reference_peak")
    assert i == 0 and note is None


def with_late_dip(t, ref, t_dip, rel=1e-9):
    """One non-increase of relative size ``rel`` at ``t_dip``; the profile rises again after it."""
    ref = ref.copy()
    k = int(np.argmin(np.abs(t - t_dip)))
    ref[k + 1] = ref[k] * (1.0 - rel)
    assert ref[k + 2] > ref[k + 1]
    return ref


def test_rule_1_ignores_a_local_maximum_after_the_event():
    # a 1e-9 wiggle on the drift at 0.5 s is no ignition; rule 2 (temperature peak) applies
    t, T, ref = profiles(drift=1.0e-2, oh_peak=0.0, T_overshoot=150.0)
    i, tau, note = ignition_sample(t, T, with_late_dip(t, ref, 0.5), "reference_peak")
    assert tau == pytest.approx(2.05e-3, abs=1e-5) and i == int(np.argmax(T))
    assert note is not None and "temperature peak" in note
    # the same wiggle on the rule-3 profile gives the mean of the steepest rises
    t, T, ref = profiles(drift=1.0e-2, oh_peak=0.0, T_drift=50.0)
    i, tau, note = ignition_sample(t, T, with_late_dip(t, ref, 0.5), "reference_peak")
    assert tau == pytest.approx(0.5 * (T_IGN + T_OH), abs=1e-5)
    assert note is not None and "mean of their steepest rises" in note


def test_rule_1_accepts_a_local_maximum_inside_the_event():
    # 3 ms is below POST_IGNITION_FACTOR times the heat-release time (about 4 ms)
    t, T, ref = profiles(drift=1.0e-2, oh_peak=0.0, T_overshoot=150.0)
    i, tau, note = ignition_sample(t, T, with_late_dip(t, ref, 3.0e-3), "reference_peak")
    assert tau == pytest.approx(3.0e-3, rel=0.05)
    assert note is not None and "first peak after the heat-release peak" in note


METHANE = {"fuel": {"CH4": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79}, "reactor": "constant_volume",
           "end_time_s": 2.0e-3, "ignition_criterion": "reference_peak", "rtol": 1e-6, "atol": 1e-12,
           "max_steps": 200000}


def test_reference_species_that_never_forms_is_a_solver_failure(mech_path):
    """An ignited reactor whose reference species is absent and never forms gives a solver error, not a
    ConfigError, so a reduction sweep counts the condition as failed and continues."""
    import cantera as ct

    from gasig.errors import ConfigError, IgnitionSolverError
    from gasig.reduce import _idt_values
    from gasig.solvers.idt import run_idt

    cfg = dict(METHANE, reference_species="AR")        # inert and not in the mixture: X_AR stays 0
    with pytest.raises(IgnitionSolverError, match="does not form the reference species 'AR'"):
        run_idt(mech_path, None, 1500.0, ct.one_atm, 1.0, cfg, label="idt-ar")
    taus, n_unignited, complete, n_failed = _idt_values(ct.Solution(mech_path), [(1500.0, ct.one_atm, 1.0)],
                                                        cfg, "red")
    assert taus == [float("inf")] and n_failed == 1 and complete
    # a reference species in the initial mixture (a reactant) stays a configuration error
    with pytest.raises(ConfigError, match="reactant or a diluent"):
        run_idt(mech_path, None, 1500.0, ct.one_atm, 1.0, dict(METHANE, reference_species="CH4"), label="idt-ch4")
