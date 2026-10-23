"""Reference implementation of GA-SiG version 0.1 for regression tests.

This is the sensitivity code of ``GA-SiG.py`` at commit affd2cf (identical to
``not-to-track/benchmark.ipynb`` apart from the absolute value in
``build_S_AB_matrix``, which is given here as an option). Do not change it:
the tests compare the package against it.
"""
import cantera as ct
import numpy as np
import torch


class TorchMassActionKinetics:
    def __init__(self, mech_file, phase_name=None, device="cpu", dtype=torch.float64):
        gas = ct.Solution(mech_file) if phase_name is None else ct.Solution(mech_file, phase_name)
        self.gas = gas
        self.Ns = gas.n_species
        self.Nr = gas.n_reactions
        self.device = torch.device(device)
        self.dtype = dtype
        nu_react_np = gas.reactant_stoich_coeffs
        nu_prod_np = gas.product_stoich_coeffs
        nu_net_np = nu_prod_np - nu_react_np
        self.nu_react = torch.tensor(nu_react_np, dtype=dtype, device=self.device)
        self.nu_prod = torch.tensor(nu_prod_np, dtype=dtype, device=self.device)
        self.nu_net = torch.tensor(nu_net_np, dtype=dtype, device=self.device)
        self.W = torch.tensor(gas.molecular_weights, dtype=dtype, device=self.device)
        self.R_u = ct.gas_constant

    def mixture_mw(self, Y):
        return 1.0 / torch.sum(Y / self.W)

    def density(self, Y, T, P):
        T_t = torch.as_tensor(T, dtype=self.dtype, device=self.device)
        P_t = torch.as_tensor(P, dtype=self.dtype, device=self.device)
        M_mix = self.mixture_mw(Y)
        R_spec = self.R_u / M_mix
        return P_t / (R_spec * T_t)

    def get_Kc(self, T, P):
        gas = self.gas
        gas.TP = float(T), float(P)
        return torch.tensor(gas.equilibrium_constants, dtype=self.dtype, device=self.device)

    def get_kf_phys(self, T, P):
        gas = self.gas
        gas.TP = float(T), float(P)
        return torch.tensor(gas.forward_rate_constants, dtype=self.dtype, device=self.device)

    def rhs_dYdt(self, Y, T, P, theta):
        if not torch.is_tensor(Y):
            Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        Y = Y.to(self.device).type(self.dtype)
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=self.dtype, device=self.device)
        theta = theta.to(self.device).type(self.dtype)
        rho = self.density(Y, T, P)
        C = rho * (Y / self.W)
        C_safe = torch.clamp(C, min=1e-300)
        lnC = torch.log(C_safe)
        q_f = torch.exp(torch.matmul(self.nu_react.T, lnC))
        q_r = torch.exp(torch.matmul(self.nu_prod.T, lnC))
        kf_phys = self.get_kf_phys(T, P)
        Kc = torch.clamp(self.get_Kc(T, P), min=1e-300)
        scale = torch.exp(theta)
        k_f = scale * kf_phys
        k_r = k_f / Kc
        R_net = k_f * q_f - k_r * q_r
        omega = torch.matmul(self.nu_net, R_net)
        return (self.W / rho) * omega


def compute_Jy_and_B_AD_single_state_torch(kin, Y_t, T0, P0, theta_t):
    device, dtype = kin.device, kin.dtype
    Y0 = Y_t.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
    theta0 = theta_t.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
    T_t = torch.as_tensor(T0, dtype=dtype, device=device)
    P_t = torch.as_tensor(P0, dtype=dtype, device=device)

    def f_Y_theta(Y, theta):
        return kin.rhs_dYdt(Y, T_t, P_t, theta)

    J_Y, J_theta = torch.autograd.functional.jacobian(f_Y_theta, (Y0, theta0), vectorize=True)
    return J_Y, J_theta


def compute_S_at_index_implicit_torch(kin, t_arr, T_traj, P_traj, Y_traj, theta0_np, idx_target):
    t_arr = np.asarray(t_arr, dtype=float)
    T_traj = np.asarray(T_traj, dtype=float)
    P_traj = np.asarray(P_traj, dtype=float)
    Y_traj = np.asarray(Y_traj, dtype=float)
    theta0_np = np.asarray(theta0_np, dtype=float)
    Nt, Ns = Y_traj.shape
    Nr = theta0_np.shape[0]
    if idx_target < 0 or idx_target >= Nt:
        raise ValueError("idx_target must be between 0 and Nt-1")
    device, dtype = kin.device, kin.dtype
    t_t = torch.from_numpy(t_arr).to(dtype=torch.float64)
    T_t = torch.from_numpy(T_traj).to(dtype=dtype, device=device)
    P_t = torch.from_numpy(P_traj).to(dtype=dtype, device=device)
    Y_t = torch.from_numpy(Y_traj).to(dtype=dtype, device=device)
    theta0_t = torch.from_numpy(theta0_np).to(dtype=dtype, device=device)
    S = torch.zeros((Ns, Nr), dtype=dtype, device=device)
    if idx_target == 0:
        return S.detach().cpu().numpy()
    I = torch.eye(Ns, dtype=dtype, device=device)
    for k in range(idx_target):
        dt = (t_t[k + 1] - t_t[k]).item()
        if dt <= 0.0:
            raise ValueError("t_arr must be strictly increasing")
        J_y, B = compute_Jy_and_B_AD_single_state_torch(kin, Y_t[k + 1, :], T_t[k + 1], P_t[k + 1], theta0_t)
        A = I - dt * J_y
        RHS = S + dt * B
        S = torch.linalg.solve(A, RHS)
    return S.detach().cpu().numpy()


def build_S_AB_matrix(S, kin, use_abs):
    S = np.asarray(S, dtype=float)
    Ns, Nr = S.shape
    nu_net_np = kin.nu_net.detach().cpu().numpy()
    nu_abs = np.abs(nu_net_np)
    S_AB = np.zeros((Ns, Ns), dtype=float)
    for B in range(Ns):
        R_B = np.where(nu_abs[B, :] != 0.0)[0]
        if R_B.size == 0:
            continue
        S_AB[:, B] = np.sum(np.abs(S[:, R_B]), axis=1) if use_abs else np.sum(S[:, R_B], axis=1)
    return S_AB


def aggregate_v01(S_list, kin, use_abs):
    S_AB_list = [build_S_AB_matrix(S, kin, use_abs) for S in S_list]
    S_abs_max = np.abs(S_list[0]).copy()
    for S in S_list[1:]:
        S_abs_max = np.maximum(S_abs_max, np.abs(S))
    S_AB_global = S_AB_list[0].copy()
    for S_AB in S_AB_list[1:]:
        S_AB_global = np.maximum(S_AB_global, S_AB)
    row_sum = np.sum(S_abs_max, axis=1)
    row_sum[row_sum == 0.0] = 1e-120
    return S_abs_max, S_AB_global, S_AB_global / row_sum[:, None]
