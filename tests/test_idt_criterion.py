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
