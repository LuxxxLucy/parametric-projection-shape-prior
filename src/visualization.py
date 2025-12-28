"""Visualization Functions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from .curves import BezierCurve


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array for plotting."""
    return tensor.detach().cpu().numpy()


def plot_results(
    curve: BezierCurve,
    output_path: str = "./outputs/parametric_shape_prior.png",
    force_magnitude: float = None,
    force_location: float = None,
):
    """Create visualization of a BezierCurve result."""
    L = curve.length
    
    # Sample the curve
    t_smooth = torch.linspace(0, 1, 200, dtype=curve.dtype, device=curve.device)
    x_smooth = _to_numpy(curve.x_at(t_smooth))
    y_smooth = _to_numpy(curve.eval_at(t_smooth))
    
    # Control points
    ctrl = _to_numpy(curve.control_points)
    P0 = [0.0, 0.0]
    P1 = [ctrl[0, 0], ctrl[0, 1]]
    P2 = [ctrl[1, 0], ctrl[1, 1]]
    P3 = [L, 0.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Curve shape with control points
    ax1 = axes[0, 0]
    ax1.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='Bezier curve')
    
    # Control polygon
    polygon_x = [P0[0], P1[0], P2[0], P3[0]]
    polygon_y = [P0[1], P1[1], P2[1], P3[1]]
    ax1.plot(polygon_x, polygon_y, 'm--', alpha=0.4, linewidth=1.5, label='Control polygon')
    
    # Control points
    ax1.scatter([P1[0], P2[0]], [P1[1], P2[1]], c='magenta', s=120, marker='s',
                edgecolors='darkmagenta', linewidths=2, zorder=10, label='Control points')
    
    # Supports
    ax1.plot([0, L], [0, 0], 'k--', alpha=0.3, linewidth=1.5)
    ax1.scatter([0, L], [0, 0], c='red', s=150, marker='^',
                edgecolors='darkred', linewidths=2, zorder=5, label='Supports')

    # Force arrow
    if force_magnitude is not None and force_location is not None:
        force_x = force_location * L
        ax1.arrow(force_x, 0, 0, force_magnitude * 0.5,
                  head_width=0.3, head_length=0.2, fc='green', ec='darkgreen',
                  linewidth=2.5, alpha=0.8)

    ax1.set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Deflection (y)', fontsize=12, fontweight='bold')
    ax1.set_title('Bezier Curve with Control Points', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Curvature
    ax2 = axes[0, 1]
    t_curv = torch.linspace(0.01, 0.99, 100, dtype=curve.dtype, device=curve.device)
    curv = curve.curvature_at(t_curv)
    x_curv = _to_numpy(curv.t_values) * L
    k_curv = _to_numpy(curv.values)

    ax2.plot(x_curv, k_curv, 'r-', linewidth=2.5)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax2.fill_between(x_curv, 0, k_curv, alpha=0.2, color='red')
    ax2.set_xlabel('Position (x)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Curvature κ', fontsize=11, fontweight='bold')
    ax2.set_title('Analytical Curvature\n(κ = |x\'y\'\' - y\'x\'\'| / (x\'² + y\'²)^{3/2})',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Stats
    stats = (f"max κ: {curv.max_curvature:.4f}\n"
             f"min κ: {curv.min_curvature:.4f}\n"
             f"avg κ: {curv.avg_curvature:.4f}")
    ax2.text(0.02, 0.98, stats, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))

    # Plot 3: First derivative
    ax3 = axes[1, 0]
    y_vals = curve.eval_at(t_curv)
    dt = t_curv[1] - t_curv[0]
    dy = (y_vals[1:] - y_vals[:-1]) / dt
    x_dy = _to_numpy(t_curv[:-1]) * L
    
    ax3.plot(x_dy, _to_numpy(dy), 'g-', linewidth=2)
    ax3.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax3.set_xlabel('Position (x)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('dy/dt', fontsize=11, fontweight='bold')
    ax3.set_title('Tangent Slope', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Info
    ax4 = axes[1, 1]
    ax4.axis('off')

    info_text = f"""
BEZIER CURVE SUMMARY
{'='*35}

Control Points:
  P0 = (0.0000, 0.0000) [fixed]
  P1 = ({P1[0]:.4f}, {P1[1]:.4f})
  P2 = ({P2[0]:.4f}, {P2[1]:.4f})
  P3 = ({L:.4f}, 0.0000) [fixed]

Curvature Statistics:
  • Max κ: {curv.max_curvature:.6f}
  • Min κ: {curv.min_curvature:.6f}
  • Avg κ: {curv.avg_curvature:.6f}

Properties:
  ✓ C² continuous
  ✓ Compact (2 control points)
  ✓ Analytical curvature
  ✓ CAD-ready
"""

    ax4.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3, pad=1.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
