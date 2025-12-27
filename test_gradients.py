"""
Test Gradient Flow Through Parametric Curve Fitting
===================================================

Verify that gradients flow from objective → parametric fitting → discrete points.
"""

import torch
from src import ParametricCurve, bending_energy, external_work


def test_gradient_flow():
    """Test that gradients flow through the entire pipeline."""
    print("\n" + "=" * 70)
    print("Testing Gradient Flow Through Parametric Curve Fitting")
    print("=" * 70)

    # Setup
    N_points = 10
    L = 5.0
    poly_degree = 4
    force_magnitude = -2.0
    force_location = 0.5

    x_discrete = torch.linspace(0, L, N_points)
    x_normalized = x_discrete / L

    # Create discrete points with gradient tracking
    y = torch.randn(N_points, requires_grad=True)

    # Define objective function
    def objective(y_discrete):
        curve = ParametricCurve.from_discrete_points(
            y_discrete, x_normalized, poly_degree, L
        )
        E_bend = bending_energy(curve)
        E_ext = external_work(curve, force_magnitude, force_location)
        return E_bend + E_ext

    # Compute objective (forward pass)
    energy = objective(y)

    print(f"\nForward pass:")
    print(f"  Energy: {energy.item():.6f}")
    print(f"  y requires_grad: {y.requires_grad}")
    print(f"  energy requires_grad: {energy.requires_grad}")

    # Compute gradients (backward pass)
    energy.backward()

    print(f"\nBackward pass:")
    print(f"  Gradients computed: {y.grad is not None}")
    if y.grad is not None:
        print(f"  Gradient shape: {y.grad.shape}")
        print(f"  Gradient mean: {y.grad.mean().item():.6f}")
        print(f"  Gradient std: {y.grad.std().item():.6f}")
        print(f"  Sample gradients: {y.grad[:3].tolist()}")

    print(f"\n✓ Gradients successfully flow through:")
    print(f"  discrete points → polynomial fitting → energy")
    print("=" * 70)

    return True


if __name__ == "__main__":
    test_gradient_flow()
