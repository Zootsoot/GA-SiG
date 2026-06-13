import numpy as np
import torch
import cantera as ct

class TorchMassActionKinetics:
    def __init__(self, mech_file, phase_name=None,
                 device="cpu", dtype=torch.float64):

        if phase_name is None:
            gas = ct.Solution(mech_file)
        else:
            gas = ct.Solution(mech_file, phase_name)

        self.gas = gas
        self.Ns = gas.n_species
        self.Nr = gas.n_reactions
        self.device = torch.device(device)
        self.dtype = dtype

        # Stoichiometry matrices (Ns, Nr)
        nu_react_np = gas.reactant_stoich_coeffs  # nu'_{ij}
        nu_prod_np  = gas.product_stoich_coeffs   # nu''_{ij}
        nu_net_np   = nu_prod_np - nu_react_np      # nu_{ij} = nu'' - nu'

        self.nu_react = torch.tensor(nu_react_np, dtype=dtype, device=self.device)
        self.nu_prod  = torch.tensor(nu_prod_np,  dtype=dtype, device=self.device)
        self.nu_net   = torch.tensor(nu_net_np,   dtype=dtype, device=self.device)

        # Molecular weights [kg/kmol]
        self.W = torch.tensor(gas.molecular_weights, dtype=dtype, device=self.device)

        # Universal gas constant
        self.R_u = ct.gas_constant

    def mixture_mw(self, Y: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sum(Y / self.W)

    def density(self, Y, T, P) -> torch.Tensor:
        T_t = torch.as_tensor(T, dtype=self.dtype, device=self.device)
        P_t = torch.as_tensor(P, dtype=self.dtype, device=self.device)

        M_mix = self.mixture_mw(Y)           # [kg/kmol]
        R_spec = self.R_u / M_mix            # [J/kg/K]
        return P_t / (R_spec * T_t)

    def get_Kc(self, T, P) -> torch.Tensor:
        gas = self.gas
        gas.TP = float(T), float(P)
        Kc_np = gas.equilibrium_constants  # (Nr,)
        return torch.tensor(Kc_np, dtype=self.dtype, device=self.device)

    def get_kf_phys(self, T, P) -> torch.Tensor:
        gas = self.gas
        gas.TP = float(T), float(P)
        kf_np = gas.forward_rate_constants  # (Nr,)
        return torch.tensor(kf_np, dtype=self.dtype, device=self.device)

    def rhs_dYdt(self, Y, T, P, theta) -> torch.Tensor:
        # Ensure tensors
        if not torch.is_tensor(Y):
            Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        Y = Y.to(self.device).type(self.dtype)

        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=self.dtype, device=self.device)
        theta = theta.to(self.device).type(self.dtype)

        # Density and concentrations
        rho = self.density(Y, T, P)          # scalar
        C   = rho * (Y / self.W)            # (Ns,)
        C_safe = torch.clamp(C, min=1e-300)
        lnC = torch.log(C_safe)             # (Ns,)

        # Mass-action terms q_f, q_r 
        ln_q_f = torch.matmul(self.nu_react.T, lnC)  # (Nr,)
        ln_q_r = torch.matmul(self.nu_prod.T,  lnC)  # (Nr,)
        q_f = torch.exp(ln_q_f)                      # (Nr,)
        q_r = torch.exp(ln_q_r)                      # (Nr,)

        # Physical k_f and equilibrium constants at (T,P)
        kf_phys = self.get_kf_phys(T, P)             # (Nr,)
        Kc      = self.get_Kc(T, P)                  # (Nr,)
        Kc = torch.clamp(Kc, min=1e-300)

        # parameterization
        scale = torch.exp(theta)                     # (Nr,)
        k_f   = scale * kf_phys                      # (Nr,)
        k_r   = k_f / Kc                             # (Nr,)

        # Net rates of progress
        R_net = k_f * q_f - k_r * q_r               # (Nr,)

        # Species source and dY/dt
        omega = torch.matmul(self.nu_net, R_net)    # (Ns,)
        dYdt  = (self.W / rho) * omega             # (Ns,)

        return dYdt

def compute_Jy_and_B_AD_single_state_torch(
    kin: TorchMassActionKinetics,
    Y_t: torch.Tensor,
    T0,
    P0,
    theta_t: torch.Tensor,
):
    device = kin.device
    dtype  = kin.dtype

    Y0     = Y_t.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
    theta0 = theta_t.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)

    T_t = torch.as_tensor(T0, dtype=dtype, device=device)
    P_t = torch.as_tensor(P0, dtype=dtype, device=device)

    def f_Y_theta(Y, theta):
        return kin.rhs_dYdt(Y, T_t, P_t, theta)  # (Ns,)

    J_Y, J_theta = torch.autograd.functional.jacobian(
        f_Y_theta,
        (Y0, theta0),
        vectorize=True  
    )
    # J_Y: (Ns, Ns), J_theta: (Ns, Nr)
    return J_Y, J_theta

def compute_S_at_index_implicit_torch(
    kin: TorchMassActionKinetics,
    t_arr: np.ndarray,
    T_traj: np.ndarray,
    P_traj: np.ndarray,
    Y_traj: np.ndarray,
    theta0_np: np.ndarray,
    idx_target: int,
):
    t_arr   = np.asarray(t_arr, dtype=float)
    T_traj  = np.asarray(T_traj, dtype=float)
    P_traj  = np.asarray(P_traj, dtype=float)
    Y_traj  = np.asarray(Y_traj, dtype=float)
    theta0_np = np.asarray(theta0_np, dtype=float)

    Nt, Ns = Y_traj.shape
    Nr = theta0_np.shape[0]

    if idx_target < 0 or idx_target >= Nt:
        raise ValueError("idx_target must be between 0 and Nt-1")

    device = kin.device
    dtype  = kin.dtype

    # Move data to torch
    t_t   = torch.from_numpy(t_arr).to(dtype=torch.float64)  # for dt
    T_t   = torch.from_numpy(T_traj).to(dtype=dtype, device=device)
    P_t   = torch.from_numpy(P_traj).to(dtype=dtype, device=device)
    Y_t   = torch.from_numpy(Y_traj).to(dtype=dtype, device=device)  # (Nt, Ns)
    theta0_t = torch.from_numpy(theta0_np).to(dtype=dtype, device=device)

    # Initial condition S(t0) = 0
    S = torch.zeros((Ns, Nr), dtype=dtype, device=device)

    if idx_target == 0:
        return S.detach().cpu().numpy()

    # Identity matrix for implicit system
    I = torch.eye(Ns, dtype=dtype, device=device)

    for k in range(idx_target):
        dt = (t_t[k+1] - t_t[k]).item()
        if dt <= 0.0:
            raise ValueError("t_arr must be strictly increasing")

        # State at t_{k+1}
        Yk1 = Y_t[k+1, :]    # (Ns,)
        Tk1 = T_t[k+1]
        Pk1 = P_t[k+1]

        # Compute J_y and B at (Yk1, Tk1, Pk1, theta0)
        J_y, B = compute_Jy_and_B_AD_single_state_torch(
            kin,
            Y_t=Yk1,
            T0=Tk1,
            P0=Pk1,
            theta_t=theta0_t,
        )
        # J_y: (Ns, Ns), B: (Ns, Nr)

        # A = I - dt J_y
        A = I - dt * J_y  # (Ns, Ns)

        # RHS = S_k + dt B_{k+1}
        RHS = S + dt * B  # (Ns, Nr)

        # Solve A S_{k+1} = RHS
        S = torch.linalg.solve(A, RHS)

        print(f"step {k+1}, dt={dt:.3e}")

    return S.detach().cpu().numpy()

def build_S_AB_matrix(S: np.ndarray, kin: TorchMassActionKinetics) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    Ns, Nr = S.shape

    if isinstance(kin.nu_net, torch.Tensor):
        nu_net_np = kin.nu_net.detach().cpu().numpy()
    else:
        nu_net_np = np.asarray(kin.nu_net, dtype=float)

    if nu_net_np.shape != (Ns, Nr):
        raise ValueError(f"Shape mismatch: nu_net is {nu_net_np.shape}, "
                         f"but S is {S.shape}.")

    nu_abs = np.abs(nu_net_np)
    S_AB = np.zeros((Ns, Ns), dtype=float)

    for B in range(Ns):
        R_B = np.where(nu_abs[B, :] != 0.0)[0]  # reactions with species B
        if R_B.size == 0:
            continue
        S_AB[:, B] = np.sum(S[:, R_B], axis=1)

    return S_AB


def build_r_AB_matrix(S_AB: np.ndarray, S: np.ndarray) -> np.ndarray:
    S    = np.asarray(S, dtype=float)
    S_AB = np.asarray(S_AB, dtype=float)

    Ns, Nr = S.shape
    if S_AB.shape != (Ns, Ns):
        raise ValueError("S_AB must have shape (Ns, Ns).")

    row_sum = np.sum(np.abs(S), axis=1)  # (Ns,)
    row_sum[row_sum == 0.0] = 1e-30      # avoid divide-by-zero

    r_AB = S_AB / row_sum[:, None]
    return r_AB

def aggregate_S_AB_over_trajectories(
    kin,
    time_all,   # list or array of shape (n_traj, Nt_m)
    T_all,
    P_all,
    Y_all,
    theta0,
    idx_targets
):
    n_traj = len(time_all)

    S_list = []
    S_AB_list = []

    for m in range(n_traj):
        t_arr  = np.asarray(time_all[m], dtype=float)
        T_traj = np.asarray(T_all[m],   dtype=float)
        P_traj = np.asarray(P_all[m],   dtype=float)
        Y_traj = np.asarray(Y_all[m],   dtype=float)

        idx_target = idx_targets[m]
        S_m = compute_S_at_index_implicit_torch(
            kin,
            t_arr=t_arr,
            T_traj=T_traj,
            P_traj=P_traj,
            Y_traj=Y_traj,
            theta0_np=theta0,
            idx_target=idx_target,
        )
        S_list.append(S_m)

        S_AB_m = build_S_AB_matrix(S_m, kin)
        S_AB_list.append(S_AB_m)

        print(f"Finished trajectory {m}.")

    # max S over trajectories in abs value
    S_abs_max = np.abs(S_list[0]).copy()
    for m in range(1, n_traj):
        S_abs_max = np.maximum(S_abs_max, np.abs(S_list[m]))

    # max S_AB over trajectories
    S_AB_global = S_AB_list[0].copy()
    for m in range(1, n_traj):
        S_AB_global = np.maximum(S_AB_global, S_AB_list[m])

    # 3) Build r_AB 
    row_sum = np.sum(S_abs_max, axis=1)  # (Ns,)
    row_sum[row_sum == 0.0] = 1e-120

    r_AB_global = S_AB_global / row_sum[:, None]

    return S_abs_max, S_AB_global, r_AB_global

if __name__ == '__main__': 

    ct.suppress_thermo_warnings()
    kin = TorchMassActionKinetics(
        "YOUR MECHNISM", # .yaml
        device="cuda", 
        dtype=torch.float64,
    )

    theta0 = np.zeros(kin.Nr, dtype=np.float64)

    # Load trajectories
    time_np = np.load("YOUR TIME TRAJECTORIES", allow_pickle=True).item()   # Python dic. (.npy) {N_traj: (Nt,)} 
    T_np    = np.load("YOUR TEMPERATURE TRAJECTORIES", allow_pickle=True).item()   # Python dic. (.npy) {N_traj: (Nt,)} 
    P_np    = np.load("YOUR PRESSURE TRAJECTORIES", allow_pickle=True).item()   # Python dic. (.npy) {N_traj: (Nt,)} 
    Y_np    = np.load("YOUR SPECIES MASS FRACTION TRAJECTORIES", allow_pickle=True).item()   # Python dic. (.npy) {N_traj: (Nt, Ns)} 

    # Load end time index
    idx_target = np.load("YOUR END TIME INDEX") # Numpy arr. (.npy) (N_traj)

    # Sensitivity integration (implicit Euler)
    S_global, S_AB_global, r_AB_global = aggregate_S_AB_over_trajectories(
        kin,
        time_all=time_np,
        T_all=T_np,
        P_all=P_np,
        Y_all=Y_np,
        theta0=theta0,
        idx_targets=idx_target
    )

    print("S_global shape:", S_global.shape)        # (Ns, Nr)
    print("S_AB_global shape:", S_AB_global.shape)  # (Ns, Ns)
    print("r_AB_global shape:", r_AB_global.shape)  # (Ns, Ns)

    np.save("r_AB.npy", r_AB_global)
