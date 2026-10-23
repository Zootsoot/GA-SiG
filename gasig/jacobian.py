"""Sensitivity Jacobians of the species source term.

For the state ``(Y, T, P)`` with frozen rate constants, the sensitivity
equation ``dS/dt = J1 S + J2`` needs

* ``J1 = d(dY/dt) / dY``  (Ns, Ns): reverse-mode automatic differentiation
  over the Ns mass fractions with :func:`torch.func.jacrev`.
* ``J2 = d(dY/dt) / dtheta`` (Ns, Nr) at ``theta = 0``: closed form. With
  ``k_f,theta = k_f exp(theta)`` and ``k_r,theta = k_r exp(theta)`` every net
  rate scales as ``R_i exp(theta_i)``, so
  ``J2[a, i] = (W_a / rho) * nu_net[a, i] * R_i``.

The closed form equals the automatic derivative to machine precision and
avoids a vector-Jacobian product per reaction.
"""
from __future__ import annotations

import torch

from .kinetics import TorchKinetics


def jacobians(
    kin: TorchKinetics,
    Y: torch.Tensor,
    T: float,
    P: float,
    kf: torch.Tensor,
    kr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(J1, J2)`` at one state. ``Y`` is a (Ns,) tensor on ``kin.device``."""
    Y = Y.detach().to(device=kin.device, dtype=kin.dtype)
    T = float(T)
    P = float(P)

    def f(y: torch.Tensor) -> torch.Tensor:
        return kin.rhs(y, T, P, kf, kr)[0]

    J1 = torch.func.jacrev(f)(Y)
    with torch.no_grad():
        _, R = kin.rhs(Y, T, P, kf, kr)
        rho = kin.density(Y, T, P)
        J2 = (kin.W / rho)[:, None] * kin.nu_net * R[None, :]
    return J1.detach(), J2


def jacobians_autograd(
    kin: TorchKinetics,
    Y: torch.Tensor,
    T: float,
    P: float,
    kf: torch.Tensor,
    kr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation with ``torch.autograd.functional.jacobian`` over
    ``(Y, theta)``, as in GA-SiG version 0.1. Used by the tests only."""
    Y0 = Y.detach().clone().to(device=kin.device, dtype=kin.dtype).requires_grad_(True)
    theta0 = torch.zeros(kin.Nr, dtype=kin.dtype, device=kin.device, requires_grad=True)
    T = float(T)
    P = float(P)

    def f(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return kin.rhs(y, T, P, kf, kr, theta)[0]

    J1, J2 = torch.autograd.functional.jacobian(f, (Y0, theta0), vectorize=True)
    return J1.detach(), J2.detach()
