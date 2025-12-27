"""
Visualization Functions (PyTorch compatible)
============================================

Plotting functions for beam form-finding results.
Handles torch tensors by converting to numpy for matplotlib.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from .parametric_curve import ParametricCurve


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array for plotting."""
    return tensor.detach().cpu().numpy()


def plot_results(
    curve: ParametricCurve,
    y_discrete: torch.Tensor,
    x_discrete: torch.Tensor,
    x_normalized: torch.Tensor,
    force_magnitude: float,
    force_location: float,
    N_points: int,
    output_path: str = "./outputs/parametric_shape_prior.png",
):
    """
    Create comprehensive visualization of optimization results.

    Args:
        curve: Final optimized ParametricCurve
        y_discrete: Discrete y-coordinates (optimized, torch.Tensor)
        x_discrete: Physical x-coordinates (torch.Tensor)
        x_normalized: Normalized x-coordinates (0 to 1, torch.Tensor)
        force_magnitude: Magnitude of applied force
        force_location: Location of force (0 to 1, normalized)
        N_points: Number of discrete points
        output_path: Path to save the figure
    """
    L = curve.length

    # Convert torch tensors to numpy for plotting
    x_discrete_np = _to_numpy(x_discrete)
    y_discrete_np = _to_numpy(y_discrete)
    x_normalized_np = _to_numpy(x_normalized)
    y_parametric_np = _to_numpy(curve.eval(x_normalized))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Final shapes comparison
    ax1 = axes[0, 0]
    ax1.plot(
        x_discrete_np,
        y_discrete_np,
        "bo-",
        markersize=6,
        linewidth=2,
        alpha=0.6,
        label="Discrete points (optimized)",
    )

    # Plot smooth parametric curve
    x_smooth = torch.linspace(
        0, 1, 200, dtype=curve.coeffs.dtype, device=curve.coeffs.device
    )
    y_smooth = curve.eval(x_smooth)
    x_smooth_np = _to_numpy(x_smooth)
    y_smooth_np = _to_numpy(y_smooth)

    ax1.plot(
        x_smooth_np * L,
        y_smooth_np,
        "r-",
        linewidth=3,
        label=f"Parametric (polynomial deg {curve.degree})",
    )

    ax1.plot(x_discrete_np, np.zeros(N_points), "k--", alpha=0.3, linewidth=1.5)
    ax1.scatter(
        [0, L],
        [0, 0],
        c="red",
        s=150,
        marker="^",
        edgecolors="darkred",
        linewidths=2,
        zorder=5,
        label="Pinned supports",
    )

    # Force
    force_idx = int(force_location * (N_points - 1))
    ax1.arrow(
        x_discrete_np[force_idx],
        0,
        0,
        force_magnitude * 0.5,
        head_width=0.3,
        head_length=0.2,
        fc="green",
        ec="darkgreen",
        linewidth=2.5,
        alpha=0.8,
    )

    ax1.set_xlabel("Position (x)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Deflection (y)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Discrete Points → Parametric Shape Projection", fontsize=13, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Plot 2: Curvature from parametric representation
    ax2 = axes[0, 1]
    x_curv_eval = torch.linspace(
        0, 1, 100, dtype=curve.coeffs.dtype, device=curve.coeffs.device
    )
    kappa = curve.curvature(x_curv_eval)
    x_curv_np = _to_numpy(x_curv_eval)[1:-1]  # Interior points
    kappa_np = _to_numpy(kappa)

    ax2.plot(x_curv_np * L, kappa_np, "r-", linewidth=2.5, marker="o", markersize=4)
    ax2.axhline(0, color="k", linestyle="--", alpha=0.4)
    ax2.fill_between(x_curv_np * L, 0, kappa_np, alpha=0.2, color="red")
    ax2.set_xlabel("Position (x)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Curvature κ", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Curvature from Parametric Form\n(Bending Energy ∝ κ²)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residual (discrete - parametric)
    ax3 = axes[1, 0]
    residual_np = y_discrete_np - y_parametric_np
    ax3.plot(x_discrete_np, residual_np, "go-", markersize=5, linewidth=2)
    ax3.axhline(0, color="k", linestyle="--", alpha=0.4)
    ax3.fill_between(x_discrete_np, 0, residual_np, alpha=0.2, color="green")
    ax3.set_xlabel("Position (x)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Residual (discrete - parametric)", fontsize=11, fontweight="bold")
    ax3.set_title("Projection Quality", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Key concept diagram
    ax4 = axes[1, 1]
    ax4.axis("off")

    concept_text = """
KEY CONCEPT: Parametric Shape Prior
═════════════════════════════════════

Pipeline (Differentiable with PyTorch):
  1. Optimize discrete y-coordinates
  2. Project → Polynomial fitting
  3. Compute objectives on polynomial
  4. Gradients flow back through pipeline

Why Parametric?
  ✓ Manufacturable (export to CAD)
  ✓ Smooth (inherent in polynomial)
  ✓ Compact (few parameters)
  ✓ Fully differentiable (PyTorch)

This Example:
  • Polynomial degree: {deg}
  • Coefficients: {ncoef} parameters
  • vs {npts} discrete points
  
The polynomial representation can be
directly used for manufacturing!

Max fitting error: {err:.4f}
"""

    coeffs_np = _to_numpy(curve.coeffs)
    concept_text = concept_text.format(
        deg=curve.degree,
        ncoef=len(coeffs_np),
        npts=N_points,
        err=np.max(np.abs(residual_np)),
    )

    ax4.text(
        0.1,
        0.5,
        concept_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.3, pad=1.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")
