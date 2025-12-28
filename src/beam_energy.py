"""Beam Energy Functions (PyTorch)."""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .curves import Curve


def compute_beam_energy(
    curve: "Curve",
    force_magnitude: float,
    force_location: float,
    regularization_weight: float = 0.1,
    n_eval_points: int = 100,
) -> torch.Tensor:
    """Compute E = E_bending + E_external + E_regularization."""
    # Sample the curve uniformly
    t_eval = torch.linspace(
        0, 1, n_eval_points, dtype=curve.dtype, device=curve.device
    )
    y_eval = curve.eval_at(t_eval)
    x_eval = curve.x_at(t_eval)
    
    if len(y_eval) < 3:
        return torch.tensor(0.0, dtype=curve.dtype, device=curve.device)
    
    dx = x_eval[1] - x_eval[0]
    
    # Bending energy: 0.5 * ∫ κ² dx ≈ 0.5 * ∫ (d²y/dx²)² dx
    d2y = (y_eval[2:] - 2*y_eval[1:-1] + y_eval[:-2]) / (dx**2)
    E_bend = 0.5 * torch.sum(d2y**2) * dx
    
    # External work: -F * y(x_force)
    x_force = force_location * curve.length
    idx_force = torch.argmin(torch.abs(x_eval - x_force))
    E_ext = -force_magnitude * y_eval[idx_force]
    
    # Regularization: penalize large deformations
    E_reg = regularization_weight * torch.sum(y_eval**2) * dx
    
    return E_bend + E_ext + E_reg
