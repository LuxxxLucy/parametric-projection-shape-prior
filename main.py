"""
Parametric Shape Prior for Beam Form-Finding (PyTorch)
======================================================

Main script: optimizes beam shape using parametric curve projection.
Uses PyTorch for automatic differentiation through the entire pipeline.
"""

import json
import torch
import numpy as np

from src import ParametricCurve, bending_energy, external_work, plot_results


def main():
    # ============ Configuration ============
    N_points = 30  # Number of discrete points
    L = 5.0  # Beam length
    poly_degree = 4  # Polynomial degree
    force_magnitude = -2.0
    force_location = 0.5  # Relative position (0 to 1)
    n_iterations = 200

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nParametric Shape Prior for Beam Form-Finding (PyTorch)")
    print(f"=" * 70)
    print(f"Device: {device}")
    print(f"Discrete points: {N_points}")
    print(f"Beam length: {L}")
    print(f"Parametric representation: Polynomial degree {poly_degree}")
    print(f"Boundary: Simply supported (both ends pinned)")
    print(f"Load: {force_magnitude} at position {force_location*L}")
    print(f"=" * 70)

    # ============ Setup ============
    x_discrete = torch.linspace(0, L, N_points, device=device)
    x_normalized = x_discrete / L

    # Initialize y-coordinates as optimization variables
    y = torch.zeros(N_points, device=device, requires_grad=True)

    # ============ Optimization ============
    # Define objective function (combines bending energy + external work)
    def objective(y_discrete):
        """
        Objective function: minimize total energy.

        Pipeline (all differentiable):
        1. Project discrete points → parametric curve
        2. Compute bending energy (integral of curvature^2)
        3. Compute external work (force × displacement)
        4. Return total energy
        """
        curve = ParametricCurve.from_discrete_points(
            y_discrete, x_normalized, poly_degree, L
        )

        E_bend = bending_energy(curve)
        E_ext = external_work(curve, force_magnitude, force_location)

        return E_bend + E_ext

    # Use L-BFGS optimizer (same as scipy's L-BFGS-B)
    optimizer = torch.optim.LBFGS(
        [y], lr=1.0, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
    )

    print(f"\nOptimizing with PyTorch automatic differentiation...")
    print(f"Optimizer: L-BFGS\n")

    # Track progress
    iteration_count = [0]
    energy_history = []

    def closure():
        """Closure for L-BFGS optimizer."""
        optimizer.zero_grad()
        loss = objective(y)
        loss.backward()

        iteration_count[0] += 1
        if iteration_count[0] % 20 == 0:
            energy_history.append(loss.item())
            print(f"Iteration {iteration_count[0]}: Energy = {loss.item():.6f}")

        return loss

    # Run optimization
    for i in range(n_iterations // 20):
        optimizer.step(closure)

        # Check convergence
        if len(energy_history) > 1:
            if abs(energy_history[-1] - energy_history[-2]) < 1e-6:
                print(f"Converged at iteration {iteration_count[0]}")
                break

    # ============ Results ============
    y_final = y.detach()

    # Construct final parametric curve
    curve_final = ParametricCurve.from_discrete_points(
        y_final, x_normalized, poly_degree, L
    )
    y_parametric = curve_final.eval(x_normalized)

    # Compute final energy
    final_energy = objective(y_final).item()

    print(f"\nOptimization complete!")
    print(f"Final energy: {final_energy:.6f}")
    print(f"Total iterations: {iteration_count[0]}")

    # ============ Analysis ============
    print(f"\n" + "=" * 70)
    print(f"Comparison: Discrete Points vs Parametric Representation")
    print(f"=" * 70)

    diff = (y_final - y_parametric).detach().cpu()
    print(f"Max difference: {torch.max(torch.abs(diff)):.6f}")
    print(f"RMS difference: {torch.sqrt(torch.mean(diff**2)):.6f}")

    print(f"\nPolynomial coefficients (for manufacturing):")
    coeffs_np = curve_final.coeffs.detach().cpu().numpy()
    for i, c in enumerate(coeffs_np):
        print(f"  a{i} = {c:.6f}")

    # ============ Save Results ============
    plot_results(
        curve_final,
        y_final,
        x_discrete,
        x_normalized,
        force_magnitude,
        force_location,
        N_points,
    )

    output_data = {
        "polynomial_degree": curve_final.degree,
        "coefficients": coeffs_np.tolist(),
        "length": curve_final.length,
        "boundary_condition": "simply_supported",
        "max_deflection": float(torch.max(torch.abs(y_parametric)).item()),
        "formula": "y(x) = x(L-x) * sum(a_i * (x/L)^i)",
        "optimizer": "PyTorch L-BFGS",
        "gradients": "Automatic differentiation through parametric fitting",
    }

    with open("./outputs/beam_parametric.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nParametric representation saved to: beam_parametric.json")
    print(f"\nManufacturing formula:")
    print(f"  y(x) = x(L-x) * [", end="")
    for i, c in enumerate(coeffs_np):
        if i > 0:
            print(f" + {c:.6f}*(x/L)^{i}", end="")
        else:
            print(f"{c:.6f}", end="")
    print(f"]\n  where L = {curve_final.length}")

    print(f"\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"  Optimized {N_points} discrete points using PyTorch")
    print(f"  Projected to polynomial with {len(coeffs_np)} coefficients")
    print(f"  Gradients flow through parametric fitting (differentiable!)")
    print(f"  The polynomial is smooth, manufacturable, and compact!")
    print(f"=" * 70)


if __name__ == "__main__":
    main()
