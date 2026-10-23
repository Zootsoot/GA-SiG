"""Differentiable mass-action kinetics built on Cantera rate constants.

At every state the rate constants come from Cantera, evaluated at the true
temperature, pressure and composition. The rates of progress, the species
production rates and their derivatives with respect to the mass fractions
are computed in torch so that automatic differentiation can give the
Jacobian ``J1 = d(dY/dt)/dY``.

Rate law (``rate_law="corrected"``)
-----------------------------------
For reaction ``i`` with forward and reverse rate constants ``kf_i``, ``kr_i``
from Cantera (``kr_i = 0`` for an irreversible reaction) and concentrations
``C``::

    q_f,i = kf_i * prod_j C_j^nu'_ij
    q_r,i = kr_i * prod_j C_j^nu''_ij
    R_i   = m_i * exp(theta_i) * (q_f,i - q_r,i)

``m_i`` is the effective third-body concentration ``sum_j e_ij C_j`` for the
three-body reactions whose Cantera rate constant excludes it (Cantera 3.x
semantics), and 1 for every other reaction. Falloff, pressure-dependent and
Chebyshev rate constants already contain their pressure and composition
dependence; they are taken from Cantera and treated as constants in the
Jacobian. Only the mass-action part and ``m_i`` are differentiated.

``rate_law="original"`` is the rate law behind the published GA-SiG results
(rate constants at the mechanism's default composition, no third-body
factor, ``kr = kf / Kc`` for every reaction, concentrations clamped at
1e-300). The two rate laws are two versions of the method; select one with
``sensitivity.rate_law`` in the configuration.
"""
from __future__ import annotations

import numpy as np
import torch
import cantera as ct


class TorchKinetics:
    """Mass-action kinetics of a Cantera mechanism as differentiable torch code."""

    def __init__(
        self,
        mech_file: str,
        phase_name: str | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        concentration_floor: float = 1e-30,
        rate_law: str = "corrected",
        self_check: bool = True,
    ):
        if rate_law not in ("original", "corrected"):
            raise ValueError("rate_law must be 'original' or 'corrected'.")
        gas = ct.Solution(mech_file) if phase_name is None else ct.Solution(mech_file, phase_name)
        self.gas = gas
        self.mech_file = mech_file
        self.phase_name = phase_name
        self.species_names: list[str] = list(gas.species_names)
        self.Ns = gas.n_species
        self.Nr = gas.n_reactions
        self.device = torch.device(device)
        self.dtype = dtype
        self.rate_law = rate_law
        self.floor = float(concentration_floor)
        tiny = float(torch.finfo(dtype).tiny)
        if self.floor < tiny:
            raise ValueError(f"concentration_floor={concentration_floor:g} is below the smallest normal number "
                             f"of {dtype} ({tiny:.3g}).")
        # The clamp of the original rate law: 1e-300 in float64, the smallest normal number otherwise.
        self._orig_clamp = max(1e-300, tiny)

        # Stoichiometry (Ns, Nr)
        nu_react_np = np.asarray(gas.reactant_stoich_coeffs, dtype=float)
        nu_prod_np = np.asarray(gas.product_stoich_coeffs, dtype=float)
        self.nu_react_np = nu_react_np
        self.nu_prod_np = nu_prod_np
        self.nu_net_np = nu_prod_np - nu_react_np
        self.nu_react = torch.tensor(nu_react_np, dtype=dtype, device=self.device)
        self.nu_prod = torch.tensor(nu_prod_np, dtype=dtype, device=self.device)
        self.nu_net = torch.tensor(self.nu_net_np, dtype=dtype, device=self.device)

        # Molecular weights [kg/kmol] and the gas constant [J/kmol/K]
        self.W_np = np.asarray(gas.molecular_weights, dtype=float)
        self.W = torch.tensor(self.W_np, dtype=dtype, device=self.device)
        self.R_u = float(ct.gas_constant)

        # Third-body efficiencies for the reactions whose Cantera k_f excludes [M]
        E = np.zeros((self.Nr, self.Ns), dtype=float)
        tb_mask = np.zeros(self.Nr, dtype=bool)
        reversible = np.zeros(self.Nr, dtype=bool)
        reaction_types: list[str] = []
        name_to_index = {name: j for j, name in enumerate(self.species_names)}
        for i, rx in enumerate(gas.reactions()):
            reaction_types.append(rx.reaction_type)
            reversible[i] = bool(rx.reversible)
            if rx.reaction_type.startswith("three-body"):
                third = rx.third_body
                E[i, :] = float(third.default_efficiency)
                # Efficiencies of species that are not in the phase are ignored, as Cantera does.
                for name, val in third.efficiencies.items():
                    j = name_to_index.get(name)
                    if j is not None:
                        E[i, j] = float(val)
                tb_mask[i] = True
        self.reaction_types = reaction_types
        self.reversible_np = reversible
        self.n_three_body = int(tb_mask.sum())
        self.E = torch.tensor(E, dtype=dtype, device=self.device)
        self.tb_mask = torch.tensor(tb_mask, dtype=torch.bool, device=self.device)

        if rate_law == "original":
            # A second Solution whose composition is never set, as in the original computation.
            self._gas_original = ct.Solution(mech_file) if phase_name is None else ct.Solution(mech_file, phase_name)

        if self_check and rate_law == "corrected":
            self.self_check()

    # ------------------------------------------------------------------ state
    def rate_constants(self, T: float, P: float, Y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward and reverse rate constants from Cantera at the state (T, P, Y)."""
        T = float(T)
        P = float(P)
        if self.rate_law == "original":
            g = self._gas_original
            g.TP = T, P
            kf = np.asarray(g.forward_rate_constants, dtype=float)
            Kc = np.clip(np.asarray(g.equilibrium_constants, dtype=float), 1e-300, None)
            kr = kf / Kc
        else:
            Y = np.clip(np.asarray(Y, dtype=float), 0.0, None)
            self.gas.TPY = T, P, Y
            kf = np.asarray(self.gas.forward_rate_constants, dtype=float)
            kr = np.asarray(self.gas.reverse_rate_constants, dtype=float)
        kf_t = torch.tensor(kf, dtype=self.dtype, device=self.device)
        kr_t = torch.tensor(kr, dtype=self.dtype, device=self.device)
        return kf_t, kr_t

    def mixture_mw(self, Y: torch.Tensor) -> torch.Tensor:
        """Mean molecular weight [kg/kmol] of the mixture with mass fractions Y."""
        return 1.0 / torch.sum(Y / self.W)

    def density(self, Y: torch.Tensor, T, P) -> torch.Tensor:
        """Ideal-gas density [kg/m3]."""
        T_t = torch.as_tensor(T, dtype=self.dtype, device=self.device)
        P_t = torch.as_tensor(P, dtype=self.dtype, device=self.device)
        R_spec = self.R_u / self.mixture_mw(Y)
        return P_t / (R_spec * T_t)

    def concentrations(self, Y: torch.Tensor, T, P) -> torch.Tensor:
        """Molar concentrations [kmol/m3]."""
        rho = self.density(Y, T, P)
        return rho * Y / self.W

    # ------------------------------------------------------------------ rates
    def rates_of_progress(
        self,
        Y: torch.Tensor,
        T,
        P,
        kf: torch.Tensor,
        kr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q_f, q_r, rho)``: forward and reverse rates of progress [kmol/m3/s]
        with the third-body concentration applied, and the density [kg/m3]."""
        rho = self.density(Y, T, P)
        C = rho * Y / self.W
        if self.rate_law == "original":
            C_safe = torch.clamp(C, min=self._orig_clamp)
        else:
            C_safe = C + self.floor
        lnC = torch.log(C_safe)
        q_f = torch.exp(torch.matmul(self.nu_react.T, lnC))
        q_r = torch.exp(torch.matmul(self.nu_prod.T, lnC))
        if self.rate_law == "corrected":
            M_eff = torch.matmul(self.E, C)
            q_f = torch.where(self.tb_mask, q_f * M_eff, q_f)
            q_r = torch.where(self.tb_mask, q_r * M_eff, q_r)
        return kf * q_f, kr * q_r, rho

    def rhs(
        self,
        Y: torch.Tensor,
        T,
        P,
        kf: torch.Tensor,
        kr: torch.Tensor,
        theta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(dY/dt, R_net)`` at fixed T, P with frozen rate constants.

        ``dY/dt`` has the unit 1/s and ``R_net`` the unit kmol/m3/s. The
        function is differentiable with respect to ``Y`` and ``theta``.
        """
        q_f, q_r, rho = self.rates_of_progress(Y, T, P, kf, kr)
        R = q_f - q_r
        if theta is not None:
            R = R * torch.exp(theta)
        omega = torch.matmul(self.nu_net, R)
        dYdt = (self.W / rho) * omega
        return dYdt, R

    # ------------------------------------------------------------------ checks
    def self_check(self, T: float = 1500.0, P: float = ct.one_atm, rtol: float = 1e-8) -> float:
        """Compare the torch rate law with Cantera reaction by reaction; raise on mismatch.

        The forward and the reverse rates of progress are compared separately
        at a state where every species is present, each relative to its own
        magnitude (with a floor of 1e-12 of the largest rate). Returns the
        largest relative deviation. A mismatch means the mechanism uses a
        reaction type that this rate law does not reproduce.
        """
        Y = np.full(self.Ns, 1.0 / self.Ns)
        kf, kr = self.rate_constants(T, P, Y)
        Y_t = torch.tensor(Y, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            q_f, q_r, _ = self.rates_of_progress(Y_t, T, P, kf, kr)
        qf_ct = np.asarray(self.gas.forward_rates_of_progress, dtype=float)
        qr_ct = np.asarray(self.gas.reverse_rates_of_progress, dtype=float)
        floor = 1e-12 * max(float(np.max(np.abs(qf_ct))), float(np.max(np.abs(qr_ct))), 1e-300)
        err_f = np.abs(q_f.cpu().numpy() - qf_ct) / np.maximum(np.abs(qf_ct), floor)
        err_r = np.abs(q_r.cpu().numpy() - qr_ct) / np.maximum(np.abs(qr_ct), floor)
        err = float(max(err_f.max(), err_r.max()))
        if not np.isfinite(err) or err > rtol:
            worst = int(np.argmax(np.maximum(err_f, err_r)))
            raise RuntimeError(
                "The torch rate law does not reproduce Cantera for this mechanism "
                f"(largest relative deviation {err:.3e}, tolerance {rtol:.1e}, worst reaction {worst}: "
                f"'{self.gas.reaction(worst).equation}' of type '{self.reaction_types[worst]}'). "
                "Check the mechanism for reaction types that GA-SiG does not support."
            )
        return err

    def describe(self) -> dict:
        """Short summary of the mechanism for logs and run summaries."""
        counts: dict[str, int] = {}
        for t in self.reaction_types:
            counts[t] = counts.get(t, 0) + 1
        return {
            "mechanism": self.mech_file,
            "phase": self.phase_name,
            "n_species": self.Ns,
            "n_reactions": self.Nr,
            "n_three_body": self.n_three_body,
            "n_irreversible": int((~self.reversible_np).sum()),
            "reaction_types": counts,
            "rate_law": self.rate_law,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
        }
