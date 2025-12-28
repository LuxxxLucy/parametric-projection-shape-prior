"""Comparison Visualization Functions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from .optimizers import OptimizationResult


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array for plotting."""
    return tensor.detach().cpu().numpy()


def plot_curvature_comparison(
    result_direct: OptimizationResult,
    result_bezier: OptimizationResult,
    output_path: str = "./outputs/curvature_comparison.png"
):
    """Side-by-side curvature plots with statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    L = result_bezier.curve.length
    
    # Direct curve curvature (finite differences)
    curv_direct = result_direct.curve.curvature_at(
        torch.linspace(0, 1, 100, dtype=result_direct.curve.dtype,
                       device=result_direct.curve.device)
    )
    x_direct = _to_numpy(curv_direct.t_values) * L
    k_direct = _to_numpy(curv_direct.values)
    
    # Bezier curve curvature (analytical)
    t_eval = torch.linspace(0.01, 0.99, 100, dtype=result_bezier.curve.dtype,
                            device=result_bezier.curve.device)
    curv_bezier = result_bezier.curve.curvature_at(t_eval)
    x_bezier = _to_numpy(curv_bezier.t_values) * L
    k_bezier = _to_numpy(curv_bezier.values)
    
    # Plot direct curvature
    ax1.plot(x_direct, k_direct, 'b-', linewidth=2.5, marker='o', markersize=3)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax1.fill_between(x_direct, 0, k_direct, alpha=0.2, color='blue')
    ax1.set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Curvature κ', fontsize=12, fontweight='bold')
    ax1.set_title(f'Direct Optimization (Finite Diff)\nEnergy: {result_direct.final_energy:.4f}',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Stats box
    stats_direct = (f"max κ: {curv_direct.max_curvature:.4f}\n"
                    f"min κ: {curv_direct.min_curvature:.4f}\n"
                    f"avg κ: {curv_direct.avg_curvature:.4f}")
    ax1.text(0.02, 0.98, stats_direct, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='blue'))
    
    # Plot Bezier curvature
    ax2.plot(x_bezier, k_bezier, 'r-', linewidth=2.5)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax2.fill_between(x_bezier, 0, k_bezier, alpha=0.2, color='red')
    ax2.set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Curvature κ', fontsize=12, fontweight='bold')
    ax2.set_title(f'Parametric Projection (Analytical)\nEnergy: {result_bezier.final_energy:.4f}',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Stats box
    stats_bezier = (f"max κ: {curv_bezier.max_curvature:.4f}\n"
                    f"min κ: {curv_bezier.min_curvature:.4f}\n"
                    f"avg κ: {curv_bezier.avg_curvature:.4f}")
    ax2.text(0.02, 0.98, stats_bezier, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Set common y-axis limits for both plots
    all_curvatures = []
    if len(k_direct) > 0:
        all_curvatures.extend(k_direct)
    if len(k_bezier) > 0:
        all_curvatures.extend(k_bezier)
    
    if len(all_curvatures) > 0:
        y_min = min(all_curvatures)
        y_max = max(all_curvatures)
        y_range = y_max - y_min
        margin = y_range * 0.1 if y_range > 0 else 0.1
        ax1.set_ylim([y_min - margin, y_max + margin])
        ax2.set_ylim([y_min - margin, y_max + margin])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_shape_comparison(
    result_direct: OptimizationResult,
    result_bezier: OptimizationResult,
    force_magnitude: float,
    force_location: float,
    output_path: str = "./outputs/shape_comparison.png"
):
    """Side-by-side shape plots using curve.eval_at() interface."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    L = result_bezier.curve.length
    t_smooth = torch.linspace(0, 1, 200, dtype=result_bezier.curve.dtype,
                              device=result_bezier.curve.device)
    
    # Direct curve
    discrete_curve = result_direct.curve
    x_discrete = _to_numpy(discrete_curve.x_at(
        torch.linspace(0, 1, discrete_curve.n_points, dtype=discrete_curve.dtype,
                       device=discrete_curve.device)
    ))
    y_discrete = _to_numpy(discrete_curve.y_values)
    
    ax1.plot(x_discrete, y_discrete, 'bo-', markersize=6, linewidth=2, alpha=0.7,
             label='points')
    ax1.plot(x_discrete, np.zeros_like(x_discrete), 'k--', alpha=0.3, linewidth=1.5)
    ax1.scatter([0, L], [0, 0], c='red', s=150, marker='^',
                edgecolors='darkred', linewidths=2, zorder=5, label='Load Force')
    
    # Force arrow: from (force_x, y(force_location) + 20) to (force_x, y(force_location) + 1)
    force_idx = int(force_location * (len(x_discrete) - 1))
    force_x_1 = x_discrete[force_idx]
    # Get curve point at force location
    curve_y_at_force_1 = y_discrete[force_idx]
    # Start from y + 1, end at y + 0.5
    arrow_start_y_1 = curve_y_at_force_1 + 1
    arrow_end_y_1 = curve_y_at_force_1 + 0.5
    arrow_dy_1 = arrow_end_y_1 - arrow_start_y_1
    ax1.arrow(force_x_1, arrow_start_y_1, 0, arrow_dy_1,
              head_width=0.3, head_length=0.2, fc='green', ec='darkgreen',
              linewidth=2.5, alpha=0.8)
    
    ax1.set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Deflection (y)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Direct Optimization',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bezier curve
    bezier_curve = result_bezier.curve
    x_smooth = _to_numpy(bezier_curve.x_at(t_smooth))
    y_smooth = _to_numpy(bezier_curve.eval_at(t_smooth))
    
    ax2.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='Bezier curve')
    
    # Add discrete points being optimized in BezierOptimizer
    if result_bezier.discrete_y_values is not None:
        y_bezier_discrete = _to_numpy(result_bezier.discrete_y_values)
        # Get x coordinates for these discrete points
        t_discrete_bezier = torch.linspace(0, 1, len(y_bezier_discrete), 
                                           dtype=bezier_curve.dtype,
                                           device=bezier_curve.device)
        x_bezier_discrete = _to_numpy(bezier_curve.x_at(t_discrete_bezier))
        ax2.plot(x_bezier_discrete, y_bezier_discrete, 'bo-', markersize=6, linewidth=2, alpha=0.7,
                 label='points')
    
    # Control polygon
    ctrl = _to_numpy(bezier_curve.control_points)
    P0, P3 = [0.0, 0.0], [L, 0.0]
    polygon_x = [P0[0], ctrl[0, 0], ctrl[1, 0], P3[0]]
    polygon_y = [P0[1], ctrl[0, 1], ctrl[1, 1], P3[1]]
    ax2.plot(polygon_x, polygon_y, 'm--', alpha=0.4, linewidth=1.5, label='Control polygon')
    ax2.scatter([ctrl[0, 0], ctrl[1, 0]], [ctrl[0, 1], ctrl[1, 1]], c='magenta', s=120,
                marker='s', edgecolors='darkmagenta', linewidths=2, zorder=10,
                label='Control points')
    
    ax2.plot([0, L], [0, 0], 'k--', alpha=0.3, linewidth=1.5)
    ax2.scatter([0, L], [0, 0], c='red', s=150, marker='^',
                edgecolors='darkred', linewidths=2, zorder=5, label='Load Force')
    
    # Force arrow: from (force_x, y(force_location) + 20) to (force_x, y(force_location) + 1)
    force_x = force_location * L
    # Get curve point at force location
    t_force = torch.tensor(force_location, dtype=bezier_curve.dtype, device=bezier_curve.device)
    curve_y_at_force_tensor = bezier_curve.eval_at(t_force)
    curve_y_at_force = _to_numpy(curve_y_at_force_tensor)
    if curve_y_at_force.ndim > 0:
        curve_y_at_force = curve_y_at_force[0]
    # Start from y + 1, end at y + 0.5
    arrow_start_y = curve_y_at_force + 1 
    arrow_end_y = curve_y_at_force + 0.5
    arrow_dy = arrow_end_y - arrow_start_y
    ax2.arrow(force_x, arrow_start_y, 0, arrow_dy,
              head_width=0.3, head_length=0.2, fc='green', ec='darkgreen',
              linewidth=2.5, alpha=0.8)
    
    ax2.set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Deflection (y)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Parametric Projection',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Set common y-axis limits for both plots
    all_y_values = []
    if len(y_discrete) > 0:
        all_y_values.extend(y_discrete)
    if len(y_smooth) > 0:
        all_y_values.extend(y_smooth)
    if result_bezier.discrete_y_values is not None:
        y_bezier_discrete = _to_numpy(result_bezier.discrete_y_values)
        if len(y_bezier_discrete) > 0:
            all_y_values.extend(y_bezier_discrete)
    # Include arrow positions from both plots
    all_y_values.extend([arrow_start_y_1, arrow_end_y_1, arrow_start_y, arrow_end_y])
    
    if len(all_y_values) > 0:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min
        margin = y_range * 0.4 if y_range > 0 else 0.1
        ax1.set_ylim([y_min - margin, y_max + margin])
        ax2.set_ylim([y_min - margin, y_max + margin])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_energy_convergence(
    result_direct: OptimizationResult,
    result_bezier: OptimizationResult,
    output_path: str = "./outputs/energy_convergence.png"
):
    """Side-by-side energy convergence plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Energy history is now recorded per optimization step
    if len(result_direct.energy_history) > 0:
        iters_direct = np.arange(1, len(result_direct.energy_history) + 1)
        ax1.plot(iters_direct, result_direct.energy_history, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Optimization Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy', fontsize=12, fontweight='bold')
    ax1.set_title(f'Direct Optimization\nFinal: {result_direct.final_energy:.4f}',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    if len(result_bezier.energy_history) > 0:
        iters_bezier = np.arange(1, len(result_bezier.energy_history) + 1)
        ax2.plot(iters_bezier, result_bezier.energy_history, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Optimization Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Energy', fontsize=12, fontweight='bold')
    ax2.set_title(f'Parametric Projection\nFinal: {result_bezier.final_energy:.4f}',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Set common y-axis limits: max at 0, min at constant
    all_energies = []
    if len(result_direct.energy_history) > 0:
        all_energies.extend(result_direct.energy_history)
    if len(result_bezier.energy_history) > 0:
        all_energies.extend(result_bezier.energy_history)
    
    if len(all_energies) > 0:
        y_min = min(all_energies) * 1.2
        # Set y-axis limits: max=0, min=minimum energy value
        ax1.set_ylim([y_min, 0])
        ax2.set_ylim([y_min, 0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

