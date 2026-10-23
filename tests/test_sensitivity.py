import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gasig.aggregate import aggregate  # noqa: E402
from gasig.io import load_idt_precomputed  # noqa: E402
from gasig.kinetics import TorchKinetics  # noqa: E402
from gasig.sensitivity import (  # noqa: E402
    Trajectory,
    flame_tip_index,
    implicit_euler_step,
    integrate_trajectory,
    residence_time,
)


def exact_linear(J1, J2, t):
    """S(t) = (exp(t J1) - I) J1^-1 J2 for constant J1, J2 and S(0) = 0."""
    n = J1.shape[0]
    expm = torch.linalg.matrix_exp(t * J1)
    return (expm - torch.eye(n, dtype=J1.dtype)) @ torch.linalg.solve(J1, J2)


def test_implicit_euler_guard_recovers_growing_mode():
    J1 = torch.diag(torch.tensor([-1000.0, 0.5], dtype=torch.float64))
    J2 = torch.ones((2, 3), dtype=torch.float64)
    S0 = torch.zeros((2, 3), dtype=torch.float64)
    I = torch.eye(2, dtype=torch.float64)
    dtau = 3.0                                     # dtau * lambda = 1.5 for the growing mode
    exact = exact_linear(J1, J2, dtau)

    S_plain, n_sub, n_solves, ok = implicit_euler_step(S0, J1, J2, dtau, I, None, 64)
    assert (n_sub, n_solves, ok) == (1, 1, True)
    assert S_plain[1, 0].item() < 0.0 < exact[1, 0].item()      # sign flip without the guard

    S_guard, n_sub, n_solves, ok = implicit_euler_step(S0, J1, J2, dtau, I, 1e-3, 1 << 16)
    assert ok and n_sub >= 8 and n_solves == 2 * n_sub - 1
    assert S_guard[1].numpy() == pytest.approx(exact[1].numpy(), rel=2e-3)
    assert S_guard[0].numpy() == pytest.approx(exact[0].numpy(), rel=2e-3)

    # the cap is respected: the finest computed level is returned, flagged as not converged
    S_cap, n_sub, n_solves, ok = implicit_euler_step(S0, J1, J2, dtau, I, 1e-12, 4)
    assert not ok and n_sub == 4 and n_solves == 1 + 2 + 4
    S_cap1, n_sub, n_solves, ok = implicit_euler_step(S0, J1, J2, dtau, I, 1e-12, 1)
    assert not ok and n_sub == 1 and n_solves == 1
    assert S_cap1.numpy() == pytest.approx(S_plain.numpy())


def test_guard_sees_a_small_row_behind_a_large_one():
    """The per-row criterion catches a growing component that the global norm would hide."""
    J1 = torch.diag(torch.tensor([0.0, 0.5], dtype=torch.float64))
    J2 = torch.zeros((2, 1), dtype=torch.float64)
    S0 = torch.tensor([[1e3], [1e-4]], dtype=torch.float64)
    I = torch.eye(2, dtype=torch.float64)
    dtau = 6.0
    exact = torch.linalg.matrix_exp(dtau * J1) @ S0
    S_guard, n_sub, _, ok = implicit_euler_step(S0, J1, J2, dtau, I, 1e-2, 1 << 12)
    assert ok and n_sub >= 8
    assert S_guard[1, 0].item() == pytest.approx(exact[1, 0].item(), rel=2e-2)


def test_implicit_euler_converges_first_order_on_decaying_system():
    J1 = torch.tensor([[-2.0, 1.0], [0.5, -3.0]], dtype=torch.float64)
    J2 = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    I = torch.eye(2, dtype=torch.float64)
    exact = exact_linear(J1, J2, 1.0)
    errors = []
    for n in (10, 20, 40):
        S = torch.zeros((2, 1), dtype=torch.float64)
        for _ in range(n):
            S, *_ = implicit_euler_step(S, J1, J2, 1.0 / n, I, None, 1)
        errors.append((S - exact).abs().max().item())
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.15)
    assert errors[1] / errors[2] == pytest.approx(2.0, rel=0.15)


def test_residence_time_and_flame_tip():
    x = np.linspace(0.0, 1.0, 11)
    tau = residence_time(x, np.full(11, 2.0))
    assert tau == pytest.approx(x / 2.0)
    u = 1.0 + x
    tau = residence_time(x, u)
    assert tau[-1] == pytest.approx(np.log(2.0), rel=2e-3)      # integral of dx/(1+x)
    with pytest.raises(ValueError):
        residence_time(x, np.zeros(11))
    with pytest.raises(ValueError):
        residence_time(x[::-1], u)
    T = np.array([300.0, 310.0, 800.0, 1900.0, 1990.0, 2000.0, 1999.0])
    assert flame_tip_index(T, 0.99) == 4
    assert flame_tip_index(T, 0.5) == 3
    assert flame_tip_index(T, None) == 6
    assert flame_tip_index(np.array([300.0, 2000.0]), 0.99) == 1
    assert flame_tip_index(np.array([300.0]), 0.99) == 0
    with pytest.raises(ValueError, match="first point"):
        flame_tip_index(T[::-1], 0.99)                 # decreasing profile
    with pytest.raises(ValueError, match="first point"):
        flame_tip_index(T, 0.1)                         # fraction below T[0] / T_max


def test_trajectory_validation():
    N, Ns = 5, 3
    good = dict(tau=np.linspace(0, 1, N), T=np.full(N, 1000.0), P=np.full(N, 1e5), Y=np.full((N, Ns), 1 / 3))
    Trajectory(**good, end_index=4).validate(Ns)
    with pytest.raises(ValueError, match="species"):
        Trajectory(**good, end_index=4).validate(Ns + 1)
    with pytest.raises(ValueError, match="end_index"):
        Trajectory(**good, end_index=N).validate(Ns)
    bad = dict(good, tau=np.array([0.0, 0.1, 0.1, 0.3, 0.4]))
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory(**bad, end_index=4).validate(Ns)
    # a defect after the end index is not an error
    Trajectory(**bad, end_index=1).validate(Ns)
    with pytest.raises(ValueError, match="integer"):
        Trajectory(**good, end_index=2.5).validate(Ns)
    Trajectory(**good, end_index=np.int64(3)).validate(Ns)
    empty = dict(good, Y=np.zeros((N, Ns)))
    with pytest.raises(ValueError, match="no positive entry"):
        Trajectory(**empty, end_index=4).validate(Ns)


@pytest.fixture(scope="module")
def example_trajectory(example_dir):
    pre = {k: str(example_dir / f"c2h4_{v}_all.npy") for k, v in
           (("time", "traj"), ("T", "T"), ("P", "P"), ("Y", "Y"))}
    pre["index"] = str(example_dir / "c2h4_index_all.npy")
    trajs = load_idt_precomputed(pre)
    assert len(trajs) == 99
    traj = trajs[0]
    traj.end_index = 40                 # first 40 intervals keep the test fast
    traj.label = "idt-0-truncated"
    return traj


def test_original_mode_reproduces_version_01_sensitivities(mech_path, example_trajectory):
    """rate_law='original' with the guard off equals the version 0.1 integrator step by step."""
    from tests import reference_v01 as ref

    kin = TorchKinetics(mech_path, device="cpu", rate_law="original")
    S_new, stats = integrate_trajectory(kin, example_trajectory, substep_tol=None, progress_every=0)
    assert stats.n_steps == 40 and stats.n_solves == 40 and stats.max_substeps == 1

    kin_ref = ref.TorchMassActionKinetics(mech_path, device="cpu")
    t = example_trajectory
    S_ref = ref.compute_S_at_index_implicit_torch(kin_ref, t.tau, t.T, t.P, t.Y, np.zeros(kin.Nr), 40)
    scale = np.abs(S_ref).max()
    assert np.max(np.abs(S_new - S_ref)) <= 1e-9 * scale

    # Rows whose sensitivities are below 1e-6 of the largest one are floating-point noise in
    # both codes: the linear systems of the original rate law have condition numbers of 1e9
    # to 1e10, so only rows above that floor can be compared after row normalization.
    row_max = np.abs(S_ref).max(axis=1)
    rows = row_max >= 1e-6 * scale
    assert rows.sum() >= 30
    assert np.max(np.abs(S_new - S_ref)[rows] / row_max[rows, None]) <= 1e-8

    # aggregation with the published normalization equals the version 0.1 aggregate (abs variant)
    agg = aggregate([S_new, 0.5 * S_new], kin.nu_net_np, abs_S=True, normalization="S_row_sum")
    S_abs_ref, S_AB_ref, r_ref = ref.aggregate_v01([S_ref, 0.5 * S_ref], kin_ref, use_abs=True)
    assert np.max(np.abs(agg.S_AB - S_AB_ref)) <= 1e-9 * S_AB_ref.max()
    assert agg.r_AB[rows] == pytest.approx(r_ref[rows], rel=1e-7, abs=1e-9)
    agg_signed = aggregate([S_new], kin.nu_net_np, abs_S=False, normalization="S_row_sum")
    _, S_AB_signed_ref, r_signed_ref = ref.aggregate_v01([S_ref], kin_ref, use_abs=False)
    assert agg_signed.r_AB[rows] == pytest.approx(r_signed_ref[rows], rel=1e-7, abs=1e-9)


def test_corrected_mode_runs_and_guard_reports(mech_path, example_trajectory):
    kin = TorchKinetics(mech_path, device="cpu", rate_law="corrected")
    S, stats = integrate_trajectory(kin, example_trajectory, substep_tol=1e-2, max_substeps=16, progress_every=0)
    assert S.shape == (kin.Ns, kin.Nr) and np.all(np.isfinite(S))
    assert stats.n_steps == 40 and stats.n_solves >= 3 * 40
    assert 1 <= stats.max_substeps <= 16


@pytest.mark.gpu
def test_gpu_matches_cpu(mech_path, example_trajectory, device):
    if device == "cpu":
        pytest.skip("no CUDA device")
    S_cpu, _ = integrate_trajectory(TorchKinetics(mech_path, device="cpu", rate_law="corrected"),
                                    example_trajectory, substep_tol=None, progress_every=0)
    S_gpu, _ = integrate_trajectory(TorchKinetics(mech_path, device=device, rate_law="corrected"),
                                    example_trajectory, substep_tol=None, progress_every=0)
    assert np.max(np.abs(S_gpu - S_cpu)) <= 1e-8 * np.abs(S_cpu).max()
