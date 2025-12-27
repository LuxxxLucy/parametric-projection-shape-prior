"""
Beam Energy Functions (PyTorch)
================================

Energy functions and optimization objective for beam form-finding.
All operations are differentiable for gradient-based optimization.
"""

import torch
from .parametric_curve import ParametricCurve


def bending_energy(curve: ParametricCurve) -> torch.Tensor:
    """
    Compute bending energy: E = 0.5 * integral(curvature^2).

    Args:
        curve: ParametricCurve instance

    Returns:
        Bending energy value (torch.Tensor)
    """
    # Sample curve densely
    x_eval = torch.linspace(
        0, 1, 100, dtype=curve.coeffs.dtype, device=curve.coeffs.device
    )
    kappa = curve.curvature(x_eval)

    # Integrate curvature^2 using trapezoidal rule
    dx_eval = x_eval[1] - x_eval[0]
    energy = 0.5 * torch.sum(kappa**2) * dx_eval * curve.length

    return energy


def external_work(
    curve: ParametricCurve, force_magnitude: float, force_location: float
) -> torch.Tensor:
    """
    Compute work done by external forces.

    Args:
        curve: ParametricCurve instance
        force_magnitude: Magnitude of applied force
        force_location: Location of force (0 to 1, normalized)

    Returns:
        External work value (torch.Tensor)
    """
    x_force = torch.tensor(
        [force_location], dtype=curve.coeffs.dtype, device=curve.coeffs.device
    )
    y_force = curve.eval(x_force)[0]
    return -force_magnitude * y_force
