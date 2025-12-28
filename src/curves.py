"""Curve Abstractions (PyTorch)."""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CurvatureResult:
    """Curvature computation result with statistics."""
    values: torch.Tensor  # Curvature values at sample points
    t_values: torch.Tensor  # Parameter values where curvature was computed
    max_curvature: float
    min_curvature: float
    avg_curvature: float


class Curve(ABC):
    """Abstract base for curves parameterized by t ∈ [0,1] with endpoints at (0,0) and (L,0)."""
    
    @property
    @abstractmethod
    def length(self) -> float: ...
    
    @property
    @abstractmethod
    def device(self) -> torch.device: ...
    
    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...
    
    @abstractmethod
    def eval_at(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate y-coordinates at t ∈ [0,1]."""
        ...
    
    @abstractmethod
    def curvature_at(self, t: torch.Tensor) -> CurvatureResult:
        """Compute curvature at t ∈ [0,1]."""
        ...
    
    def x_at(self, t: torch.Tensor) -> torch.Tensor:
        """Physical x = t * L."""
        return t * self.length


class BezierCurve(Curve):
    """Cubic Bezier with fixed endpoints P0=(0,0), P3=(L,0). Optimizes P1, P2."""
    
    def __init__(self, control_points: torch.Tensor, curve_length: float):
        self._control_points = control_points
        self._length = curve_length
    
    @property
    def length(self) -> float:
        return self._length
    
    @property
    def device(self) -> torch.device:
        return self._control_points.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self._control_points.dtype
    
    @property
    def control_points(self) -> torch.Tensor:
        """Control points P1, P2 as (2, 2) tensor."""
        return self._control_points
    
    @classmethod
    def from_discrete_points(
        cls,
        y_discrete: torch.Tensor,
        t_values: torch.Tensor,
        curve_length: float,
    ) -> "BezierCurve":
        """Fit Bezier via differentiable least squares."""
        n_points = len(t_values)
        t = t_values
        
        # Design matrix: B_y(t) = 3(1-t)²t y1 + 3(1-t)t² y2
        A = torch.zeros(n_points, 2, dtype=y_discrete.dtype, device=y_discrete.device)
        A[:, 0] = 3 * (1 - t)**2 * t
        A[:, 1] = 3 * (1 - t) * t**2
        
        # Least squares fit
        control_y = torch.linalg.lstsq(A, y_discrete).solution
        
        # Default x-positions at 1/3 and 2/3
        control_x = torch.tensor(
            [curve_length / 3, 2 * curve_length / 3],
            dtype=y_discrete.dtype,
            device=y_discrete.device,
        )
        
        control_points = torch.stack([
            torch.stack([control_x[0], control_y[0]]),
            torch.stack([control_x[1], control_y[1]]),
        ])
        
        return cls(control_points=control_points, curve_length=curve_length)
    
    def eval_at(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate y-coordinates using Bezier formula."""
        y1 = self._control_points[0, 1]
        y2 = self._control_points[1, 1]
        
        # B_y(t) = 3(1-t)²t y1 + 3(1-t)t² y2  (endpoints at y=0)
        return 3 * (1 - t)**2 * t * y1 + 3 * (1 - t) * t**2 * y2
    
    def curvature_at(self, t: torch.Tensor) -> CurvatureResult:
        """Analytical κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)."""
        L = self._length
        ctrl = self._control_points
        
        # Control points: P0=(0,0), P1, P2, P3=(L,0)
        P0 = torch.tensor([0.0, 0.0], dtype=ctrl.dtype, device=ctrl.device)
        P1 = ctrl[0]
        P2 = ctrl[1]
        P3 = torch.tensor([L, 0.0], dtype=ctrl.dtype, device=ctrl.device)
        
        # First derivative: B'(t) = 3(1-t)²(P1-P0) + 6(1-t)t(P2-P1) + 3t²(P3-P2)
        c0 = 3 * (1 - t)**2
        c1 = 6 * (1 - t) * t
        c2 = 3 * t**2
        
        dx = c0 * (P1[0] - P0[0]) + c1 * (P2[0] - P1[0]) + c2 * (P3[0] - P2[0])
        dy = c0 * (P1[1] - P0[1]) + c1 * (P2[1] - P1[1]) + c2 * (P3[1] - P2[1])
        
        # Second derivative: B''(t) = 6(1-t)(P2-2P1+P0) + 6t(P3-2P2+P1)
        d0 = 6 * (1 - t)
        d1 = 6 * t
        
        ddx = d0 * (P2[0] - 2*P1[0] + P0[0]) + d1 * (P3[0] - 2*P2[0] + P1[0])
        ddy = d0 * (P2[1] - 2*P1[1] + P0[1]) + d1 * (P3[1] - 2*P2[1] + P1[1])
        
        # Curvature (use absolute value)
        cross = dx * ddy - dy * ddx
        speed = torch.sqrt(dx**2 + dy**2)
        kappa = torch.abs(cross) / (speed**3 + 1e-10)
        
        return CurvatureResult(
            values=kappa,
            t_values=t,
            max_curvature=kappa.max().item(),
            min_curvature=kappa.min().item(),
            avg_curvature=kappa.mean().item(),
        )


class DiscreteCurve(Curve):
    """Discrete points with curvature via finite differences. y[0] = y[-1] = 0."""
    
    def __init__(self, y_values: torch.Tensor, curve_length: float):
        self._y_values = y_values
        self._length = curve_length
        self._n_points = len(y_values)
        self._t_values = torch.linspace(
            0, 1, self._n_points, dtype=y_values.dtype, device=y_values.device
        )
    
    @property
    def length(self) -> float:
        return self._length
    
    @property
    def device(self) -> torch.device:
        return self._y_values.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self._y_values.dtype
    
    @property
    def y_values(self) -> torch.Tensor:
        """Raw y-coordinate values."""
        return self._y_values
    
    @property
    def n_points(self) -> int:
        """Number of discrete points."""
        return self._n_points
    
    @classmethod
    def create_with_boundary_conditions(cls, y_interior: torch.Tensor, curve_length: float) -> "DiscreteCurve":
        """Create curve with y[0] = y[-1] = 0."""
        n_total = len(y_interior) + 2
        y_full = torch.zeros(n_total, dtype=y_interior.dtype, device=y_interior.device)
        y_full[1:-1] = y_interior
        return cls(y_values=y_full, curve_length=curve_length)
    
    def eval_at(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate via linear interpolation."""
        idx_float = t * (self._n_points - 1)
        idx_low = idx_float.long().clamp(0, self._n_points - 2)
        idx_high = (idx_low + 1).clamp(0, self._n_points - 1)
        w = idx_float - idx_low.float()
        return (1 - w) * self._y_values[idx_low] + w * self._y_values[idx_high]
    
    def curvature_at(self, t: torch.Tensor) -> CurvatureResult:
        """Curvature via finite differences κ ≈ d²y/dx²."""
        y = self._y_values
        dx = self._length / (self._n_points - 1)
        
        if self._n_points < 3:
            empty = torch.zeros(0, dtype=y.dtype, device=y.device)
            return CurvatureResult(empty, empty, 0.0, 0.0, 0.0)
        
        kappa = torch.abs((y[2:] - 2*y[1:-1] + y[:-2]) / (dx**2))
        return CurvatureResult(
            kappa, self._t_values[1:-1],
            kappa.max().item(), kappa.min().item(), kappa.mean().item()
        )

