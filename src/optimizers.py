"""Optimizer Classes for Beam Form-Finding."""

import torch
import torch.nn.utils
from dataclasses import dataclass
from typing import List
from .curves import Curve, BezierCurve, DiscreteCurve
from .beam_energy import compute_beam_energy


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    curve: Curve  # Final optimized curve
    energy_history: List[float]
    iterations: int
    final_energy: float
    discrete_y_values: torch.Tensor = None  # Optional: discrete y values (for BezierOptimizer)


class DirectOptimizer:
    """Optimize beam shape using discrete y-coordinates with pinned boundary."""
    
    def __init__(
        self,
        n_points: int,
        length: float,
        force_magnitude: float,
        force_location: float,
        regularization_weight: float = 0.1,
        n_eval_points: int = 100,
        device: torch.device = None,
    ):
        self.n_points = n_points
        self.length = length
        self.force_magnitude = force_magnitude
        self.force_location = force_location
        self.regularization_weight = regularization_weight
        self.n_eval_points = n_eval_points
        self.device = device or torch.device("cpu")
        self.n_interior = n_points - 2
    
    def _make_curve(self, y_interior: torch.Tensor) -> DiscreteCurve:
        """Build curve with boundary conditions."""
        return DiscreteCurve.create_with_boundary_conditions(
            y_interior=y_interior,
            curve_length=self.length,
        )
    
    def _objective(self, y_interior: torch.Tensor) -> torch.Tensor:
        """Compute energy via Curve interface."""
        curve = self._make_curve(y_interior)
        return compute_beam_energy(
            curve=curve,
            force_magnitude=self.force_magnitude,
            force_location=self.force_location,
            regularization_weight=self.regularization_weight,
            n_eval_points=self.n_eval_points,
        )
    
    def optimize(self, n_iterations: int = 200) -> OptimizationResult:
        """Run optimization on interior points."""
        y_interior = torch.zeros(
            self.n_interior, device=self.device, requires_grad=True
        )
        
        optimizer = torch.optim.LBFGS(
            [y_interior], lr=1.0, max_iter=20,
            history_size=10, line_search_fn="strong_wolfe"
        )
        
        iteration_count = [0]
        energy_history = []
        
        def closure():
            optimizer.zero_grad()
            loss = self._objective(y_interior)
            loss.backward()
            iteration_count[0] += 1
            return loss
        
        for step in range(n_iterations):
            optimizer.step(closure)
            with torch.no_grad():
                energy_history.append(self._objective(y_interior).item())
            if len(energy_history) > 1 and abs(energy_history[-1] - energy_history[-2]) < 1e-6:
                break
        
        final_curve = self._make_curve(y_interior.detach())
        
        return OptimizationResult(
            curve=final_curve,
            energy_history=energy_history,
            iterations=iteration_count[0],
            final_energy=self._objective(y_interior.detach()).item(),
        )


class BezierOptimizer:
    """Optimize beam shape via differentiable Bezier curve fitting."""
    
    def __init__(
        self,
        n_points: int,
        length: float,
        force_magnitude: float,
        force_location: float,
        regularization_weight: float = 0.1,
        n_eval_points: int = 100,
        device: torch.device = None,
    ):
        self.n_points = n_points
        self.length = length
        self.force_magnitude = force_magnitude
        self.force_location = force_location
        self.regularization_weight = regularization_weight
        self.n_eval_points = n_eval_points
        self.device = device or torch.device("cpu")
        
        self.t_discrete = torch.linspace(0, 1, n_points, device=self.device)
    
    def _make_curve(self, y_discrete: torch.Tensor) -> BezierCurve:
        """Fit Bezier curve to discrete points."""
        return BezierCurve.from_discrete_points(
            y_discrete=y_discrete,
            t_values=self.t_discrete,
            curve_length=self.length,
        )
    
    def _objective(self, y_discrete: torch.Tensor) -> torch.Tensor:
        """Compute energy via Curve interface."""
        curve = self._make_curve(y_discrete)
        return compute_beam_energy(
            curve=curve,
            force_magnitude=self.force_magnitude,
            force_location=self.force_location,
            regularization_weight=self.regularization_weight,
            n_eval_points=self.n_eval_points,
        )
    
    def optimize(self, n_iterations: int = 200) -> OptimizationResult:
        """Run optimization."""
        y = torch.zeros(self.n_points, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([y], lr=1.0)
        energy_history = []
        
        for step in range(n_iterations):
            optimizer.zero_grad()
            loss = self._objective(y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([y], max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                energy_history.append(self._objective(y).item())
            if len(energy_history) > 1 and abs(energy_history[-1] - energy_history[-2]) < 1e-6:
                break
        
        final_curve = self._make_curve(y.detach())
        
        return OptimizationResult(
            curve=final_curve,
            energy_history=energy_history,
            iterations=len(energy_history),
            final_energy=self._objective(y.detach()).item(),
            discrete_y_values=y.detach(),
        )
