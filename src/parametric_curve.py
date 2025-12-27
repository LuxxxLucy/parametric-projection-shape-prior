"""
Parametric Curve Abstraction (PyTorch)
======================================

Polynomial-based parametric curve with boundary conditions.
Fully differentiable for gradient-based optimization.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParametricCurve:
    """
    Parametric curve representation using polynomial.

    Enforces boundary conditions: y(0) = y(L) = 0
    Form: y = x(L-x) * p(x) where p(x) is a polynomial

    Attributes:
        coeffs: Polynomial coefficients [a0, a1, a2, ...]
        degree: Polynomial degree
        length: Physical length of the curve
    """

    coeffs: torch.Tensor
    degree: int
    length: float

    @classmethod
    def from_discrete_points(
        cls,
        y_discrete: torch.Tensor,
        x_normalized: torch.Tensor,
        degree: int,
        length: float,
    ) -> "ParametricCurve":
        """
        Fit parametric curve from discrete points via least squares projection.

        This operation is differentiable - gradients can flow through the fitting
        process back to y_discrete.

        Args:
            y_discrete: Discrete y-coordinates (torch.Tensor)
            x_normalized: Normalized x-coordinates (0 to 1)
            degree: Polynomial degree
            length: Physical length

        Returns:
            ParametricCurve instance
        """
        if degree < 2:
            raise ValueError("Polynomial degree must be >= 2 for boundary conditions")

        # Build design matrix for least squares
        # y = x(1-x) * (a0 + a1*x + a2*x^2 + ...)
        n_points = len(x_normalized)
        n_coeffs = degree - 1

        X = torch.zeros(
            n_points, n_coeffs, dtype=y_discrete.dtype, device=y_discrete.device
        )
        for i in range(n_coeffs):
            X[:, i] = x_normalized**i * x_normalized * (1 - x_normalized)

        # Least squares fit: solve X @ coeffs = y_discrete
        # Using torch.linalg.lstsq for differentiable least squares
        coeffs = torch.linalg.lstsq(X, y_discrete).solution

        return cls(coeffs=coeffs, degree=degree, length=length)

    def eval(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Evaluate curve at normalized coordinates.

        Args:
            x_norm: Normalized x-coordinates (0 to 1)

        Returns:
            y-coordinates
        """
        result = torch.zeros_like(x_norm)
        for i, coeff in enumerate(self.coeffs):
            result = result + coeff * x_norm**i
        result = result * x_norm * (1 - x_norm)
        return result

    def curvature(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute curvature (second derivative) at given points.
        Uses finite differences for numerical differentiation.

        Args:
            x_norm: Normalized x-coordinates (0 to 1)

        Returns:
            Curvature values (length n-2 for interior points)
        """
        dx_norm = x_norm[1] - x_norm[0] if len(x_norm) > 1 else 0.01

        # Evaluate curve
        y = self.eval(x_norm)

        # Compute second derivative using finite differences
        if len(y) < 3:
            return torch.zeros_like(y)

        # d2y/dx2 ≈ (y[i+1] - 2*y[i] + y[i-1]) / dx^2
        d2y = torch.zeros(len(y) - 2, dtype=y.dtype, device=y.device)
        for i in range(1, len(y) - 1):
            d2y[i - 1] = (y[i + 1] - 2 * y[i] + y[i - 1]) / (dx_norm * self.length) ** 2

        return d2y
