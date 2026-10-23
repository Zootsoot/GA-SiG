import numpy as np
import pytest

torch = pytest.importorskip("torch")
ct = pytest.importorskip("cantera")

from gasig.jacobian import jacobians, jacobians_autograd  # noqa: E402
from gasig.kinetics import TorchKinetics  # noqa: E402


def reacted_states(mech_path):
    """Three CH4/air states: fresh, partially reacted, near the end of the reaction."""
    gas = ct.Solution(mech_path)
    states = []
    for T0, P0, advance in ((1500.0, ct.one_atm, 0.0), (1500.0, ct.one_atm, 2.0e-4), (1200.0, 10 * ct.one_atm, 5.0e-3)):
        gas.TPX = T0, P0, "CH4:1, O2:2, N2:7.52"
        if advance > 0.0:
            reactor = ct.IdealGasConstPressureReactor(gas)
            net = ct.ReactorNet([reactor])
            net.advance(advance)
            phase = getattr(reactor, "phase", None) or reactor.thermo
            states.append((float(phase.T), float(phase.P), np.array(phase.Y, dtype=float)))
        else:
            states.append((float(gas.T), float(gas.P), np.array(gas.Y, dtype=float)))
    return states


@pytest.fixture(scope="module")
def kin(mech_path):
    return TorchKinetics(mech_path, device="cpu", dtype=torch.float64, rate_law="corrected")


@pytest.fixture(scope="module")
def kin_original(mech_path):
    return TorchKinetics(mech_path, device="cpu", dtype=torch.float64, rate_law="original")


@pytest.fixture(scope="module")
def states(mech_path):
    return reacted_states(mech_path)


def test_describe_and_self_check(kin):
    info = kin.describe()
    assert info["n_species"] == 111 and info["n_reactions"] == 784
    assert info["n_three_body"] == 15 and info["n_irreversible"] == 2
    assert kin.self_check() < 1e-10


def test_corrected_rates_match_cantera(kin, states, mech_path):
    gas = ct.Solution(mech_path)
    for T, P, Y in states:
        kf, kr = kin.rate_constants(T, P, Y)
        Y_t = torch.tensor(Y, dtype=torch.float64)
        with torch.no_grad():
            dYdt, R = kin.rhs(Y_t, T, P, kf, kr)
        gas.TPY = T, P, Y
        R_ct = gas.net_rates_of_progress
        dYdt_ct = gas.net_production_rates * gas.molecular_weights / gas.density
        assert np.max(np.abs(R.numpy() - R_ct)) <= 1e-10 * np.max(np.abs(R_ct))
        assert np.max(np.abs(dYdt.numpy() - dYdt_ct)) <= 1e-10 * np.max(np.abs(dYdt_ct))
        # forward and reverse parts separately
        q_f_ct = gas.forward_rates_of_progress
        q_r_ct = gas.reverse_rates_of_progress
        assert np.all(kr.numpy()[~kin.reversible_np] == 0.0)
        assert np.max(np.abs(q_r_ct[~kin.reversible_np])) == 0.0
        assert np.max(np.abs(q_f_ct - q_r_ct - R_ct)) <= 1e-12 * np.max(np.abs(R_ct))


def test_third_body_concentrations_match_cantera(kin, states, mech_path):
    gas = ct.Solution(mech_path)
    if not hasattr(gas, "third_body_concentrations"):
        pytest.skip("Cantera without third_body_concentrations")
    T, P, Y = states[1]
    gas.TPY = T, P, Y
    M_ct = np.asarray(gas.third_body_concentrations)
    C = torch.tensor(gas.concentrations, dtype=torch.float64)
    M_torch = (kin.E @ C).numpy()
    mask = kin.tb_mask.numpy()
    assert np.all(np.isfinite(M_ct[mask]))
    assert M_torch[mask] == pytest.approx(M_ct[mask], rel=1e-12)


def test_original_rate_law_reproduces_version_01(kin_original, states, mech_path):
    """The 'original' switch equals the rhs of GA-SiG.py at commit affd2cf."""
    from tests.reference_v01 import TorchMassActionKinetics

    ref = TorchMassActionKinetics(mech_path, device="cpu", dtype=torch.float64)
    theta = torch.zeros(kin_original.Nr, dtype=torch.float64)
    for T, P, Y in states:
        kf, kr = kin_original.rate_constants(T, P, Y)
        Y_t = torch.tensor(Y, dtype=torch.float64)
        with torch.no_grad():
            dYdt, _ = kin_original.rhs(Y_t, T, P, kf, kr)
            dYdt_ref = ref.rhs_dYdt(Y_t, T, P, theta)
        # equal to machine precision relative to the largest rate (the order of the
        # floating-point operations differs, so tiny elements differ in the last digits)
        assert np.max(np.abs(dYdt.numpy() - dYdt_ref.numpy())) <= 1e-11 * np.max(np.abs(dYdt_ref.numpy()))


@pytest.mark.parametrize("which", ["corrected", "original"])
def test_closed_form_J2_and_jacrev_J1_match_autograd(kin, kin_original, states, which):
    k = kin if which == "corrected" else kin_original
    T, P, Y = states[1]
    kf, kr = k.rate_constants(T, P, Y)
    Y_t = torch.tensor(Y, dtype=torch.float64)
    J1, J2 = jacobians(k, Y_t, T, P, kf, kr)
    J1_ref, J2_ref = jacobians_autograd(k, Y_t, T, P, kf, kr)
    assert J1.shape == (k.Ns, k.Ns) and J2.shape == (k.Ns, k.Nr)
    scale1 = J1_ref.abs().max().item()
    scale2 = J2_ref.abs().max().item()
    assert (J1 - J1_ref).abs().max().item() <= 1e-12 * scale1
    assert (J2 - J2_ref).abs().max().item() <= 1e-12 * scale2


def test_J1_matches_central_finite_differences(kin, states):
    T, P, Y = states[1]
    kf, kr = kin.rate_constants(T, P, Y)
    Y_t = torch.tensor(Y, dtype=torch.float64)
    J1, _ = jacobians(kin, Y_t, T, P, kf, kr)
    # columns of the most abundant species and of two radicals, each checked on its own scale
    names = kin.species_names
    cols = [names.index(s) for s in ("CH4", "O2", "OH", "H")]
    for j in cols:
        assert Y[j] > 0.0, names[j]
        h = 1e-4 * Y[j]
        Yp, Ym = Y.copy(), Y.copy()
        Yp[j] += h
        Ym[j] -= h
        with torch.no_grad():
            fp = kin.rhs(torch.tensor(Yp), T, P, kf, kr)[0].numpy()
            fm = kin.rhs(torch.tensor(Ym), T, P, kf, kr)[0].numpy()
        fd = (fp - fm) / (2 * h)
        col = J1[:, j].numpy()
        col_scale = np.abs(col).max()
        big = np.abs(col) > 1e-6 * col_scale
        assert big.sum() >= 5, names[j]
        assert np.allclose(col[big], fd[big], rtol=1e-5, atol=1e-8 * col_scale), names[j]


def test_float32_original_law_stays_finite_with_absent_species(mech_path, states):
    """The clamp of the original rate law must not underflow to zero in float32."""
    k32 = TorchKinetics(mech_path, device="cpu", dtype=torch.float32, rate_law="original")
    T, P, Y = states[0]                      # fresh mixture: most species are exactly zero
    kf, kr = k32.rate_constants(T, P, Y)
    with torch.no_grad():
        dYdt, R = k32.rhs(torch.tensor(Y, dtype=torch.float32), T, P, kf, kr)
    assert torch.isfinite(dYdt).all() and torch.isfinite(R).all()
    with pytest.raises(ValueError, match="concentration_floor"):
        TorchKinetics(mech_path, device="cpu", dtype=torch.float32, rate_law="corrected",
                      concentration_floor=1e-300)


def test_J1_mass_action_part_matches_cantera_derivatives(kin, states, mech_path):
    """d(omega_a)/dC_i from the torch rate law equals Cantera's net_production_rates_ddCi
    (derivative with respect to each species concentration at constant T, P and the other
    concentrations; Cantera skips the falloff dependence by default, as the torch rate law does)."""
    gas = ct.Solution(mech_path)
    if not hasattr(gas, "net_production_rates_ddCi"):
        pytest.skip("Cantera without species-concentration derivatives")
    T, P, Y = states[1]
    gas.TPY = T, P, Y
    kf, kr = kin.rate_constants(T, P, Y)
    C0 = torch.tensor(gas.concentrations, dtype=torch.float64)

    def omega_of_C(C):
        lnC = torch.log(C + kin.floor)
        q_f = torch.exp(kin.nu_react.T @ lnC)
        q_r = torch.exp(kin.nu_prod.T @ lnC)
        R = kf * q_f - kr * q_r
        R = torch.where(kin.tb_mask, R * (kin.E @ C), R)
        return kin.nu_net @ R

    J = torch.func.jacrev(omega_of_C)(C0).numpy()
    J_ct = gas.net_production_rates_ddCi
    if hasattr(J_ct, "toarray"):
        J_ct = J_ct.toarray()
    J_ct = np.asarray(J_ct, dtype=float)
    assert J_ct.shape == (kin.Ns, kin.Ns)
    scale = np.abs(J_ct).max()
    assert np.max(np.abs(J - J_ct)) <= 1e-8 * scale


def test_floor_keeps_consumption_derivative_of_absent_species(kin, kin_original, states):
    T, P, Y = states[1]
    j = kin.species_names.index("CH3")
    Y0 = Y.copy()
    Y0[j] = 0.0
    Y0 /= Y0.sum()
    Y_t = torch.tensor(Y0, dtype=torch.float64)
    kf, kr = kin.rate_constants(T, P, Y0)
    J1, _ = jacobians(kin, Y_t, T, P, kf, kr)
    kf_o, kr_o = kin_original.rate_constants(T, P, Y0)
    J1_o, _ = jacobians(kin_original, Y_t, T, P, kf_o, kr_o)
    # CH3 consumes itself at a reactive state; the clamp of the original law removes the
    # self-consumption (lnC) term, and only the density coupling through M_mix remains
    assert J1[j, j].item() < -1.0
    assert abs(J1_o[j, j].item()) < 1e-3 * abs(J1[j, j].item())


def test_device_round_trip(kin, states, device):
    if device == "cpu":
        pytest.skip("no CUDA device")
    k = TorchKinetics(kin.mech_file, device=device, dtype=torch.float64, rate_law="corrected")
    T, P, Y = states[1]
    kf, kr = k.rate_constants(T, P, Y)
    J1, J2 = jacobians(k, torch.tensor(Y, device=device), T, P, kf, kr)
    kf_c, kr_c = kin.rate_constants(T, P, Y)
    J1_c, J2_c = jacobians(kin, torch.tensor(Y), T, P, kf_c, kr_c)
    assert torch.allclose(J1.cpu(), J1_c, rtol=1e-10, atol=1e-10 * J1_c.abs().max().item())
    assert torch.allclose(J2.cpu(), J2_c, rtol=1e-10, atol=1e-10 * J2_c.abs().max().item())
