"""
Beam Form-Finding: Discrete vs Bezier Curve Optimization
=========================================================

Compares two curve representations using the common Curve interface:
1. DiscreteCurve: Direct y-coordinate optimization with boundary conditions
2. BezierCurve: Parametric curve with analytical curvature

Both use identical compute_beam_energy() for fair comparison.
"""

import torch
from src import (
    DirectOptimizer,
    BezierOptimizer,
    BezierCurve,
    plot_curvature_comparison,
    plot_shape_comparison,
    plot_energy_convergence,
)


def main():
    # Configuration
    n_points = 30
    length = 5.0
    force_magnitude = -2.0
    force_location = 0.5
    n_iterations = 15
    regularization = 0.1
    n_eval_points = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print("BEAM FORM-FINDING: Discrete vs Bezier Optimization")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Discrete points: {n_points}")
    print(f"Beam length: {length}")
    print(f"Boundary: Simply supported (y[0]=y[-1]=0)")
    print(f"Load: {force_magnitude} at x={force_location*length}")
    print(f"{'='*70}")
    print("Both methods use identical energy via Curve.eval_at() interface")
    print(f"{'='*70}\n")
    
    # Method 1: Discrete curve
    print("[1/2] Running Discrete Curve Optimization...")
    print("      (Optimizing interior y-coordinates, boundaries fixed)")
    
    opt_discrete = DirectOptimizer(
        n_points=n_points,
        length=length,
        force_magnitude=force_magnitude,
        force_location=force_location,
        regularization_weight=regularization,
        n_eval_points=n_eval_points,
        device=device,
    )
    result_discrete = opt_discrete.optimize(n_iterations=n_iterations)
    
    print(f"      ✓ Converged in {result_discrete.iterations} iterations")
    print(f"      ✓ Final energy: {result_discrete.final_energy:.6f}")
    
    curv_discrete = result_discrete.curve.curvature_at(
        torch.linspace(0, 1, 100, device=device)
    )
    print(f"      ✓ Curvature: max={curv_discrete.max_curvature:.4f}, "
          f"avg={curv_discrete.avg_curvature:.4f}\n")
    
    # Method 2: Bezier curve
    print("[2/2] Running Bezier Curve Optimization...")
    print("      (Optimizing through Bezier projection)")
    
    opt_bezier = BezierOptimizer(
        n_points=n_points,
        length=length,
        force_magnitude=force_magnitude,
        force_location=force_location,
        regularization_weight=regularization,
        n_eval_points=n_eval_points,
        device=device,
    )
    result_bezier = opt_bezier.optimize(n_iterations=n_iterations)
    
    print(f"      ✓ Converged in {result_bezier.iterations} iterations")
    print(f"      ✓ Final energy: {result_bezier.final_energy:.6f}")
    
    curv_bezier = result_bezier.curve.curvature_at(
        torch.linspace(0.01, 0.99, 100, device=device)
    )
    print(f"      ✓ Curvature: max={curv_bezier.max_curvature:.4f}, "
          f"avg={curv_bezier.avg_curvature:.4f}\n")
    
    # Comparison summary
    print(f"{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nMethod 1: Discrete Curve (DiscreteCurve)")
    print(f"  • Iterations: {result_discrete.iterations}")
    print(f"  • Final Energy: {result_discrete.final_energy:.6f}")
    print(f"  • Curvature: max={curv_discrete.max_curvature:.4f}, "
          f"min={curv_discrete.min_curvature:.4f}, avg={curv_discrete.avg_curvature:.4f}")
    print(f"  • Representation: {n_points} discrete points")
    
    print(f"\nMethod 2: Bezier Curve (BezierCurve)")
    print(f"  • Iterations: {result_bezier.iterations}")
    print(f"  • Final Energy: {result_bezier.final_energy:.6f}")
    print(f"  • Curvature: max={curv_bezier.max_curvature:.4f}, "
          f"min={curv_bezier.min_curvature:.4f}, avg={curv_bezier.avg_curvature:.4f}")
    
    bezier_curve = result_bezier.curve
    ctrl = bezier_curve.control_points.detach().cpu().numpy()
    print(f"\n  Control Points:")
    print(f"    P0 = (0.0000, 0.0000) [fixed]")
    print(f"    P1 = ({ctrl[0, 0]:.4f}, {ctrl[0, 1]:.4f})")
    print(f"    P2 = ({ctrl[1, 0]:.4f}, {ctrl[1, 1]:.4f})")
    print(f"    P3 = ({length:.4f}, 0.0000) [fixed]")
    
    energy_diff = abs(result_discrete.final_energy - result_bezier.final_energy)
    energy_pct = 100 * energy_diff / abs(result_discrete.final_energy)
    print(f"\nEnergy Difference: {energy_diff:.6f} ({energy_pct:.2f}%)")
    
    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*70}\n")
    
    plot_curvature_comparison(result_discrete, result_bezier)
    plot_shape_comparison(result_discrete, result_bezier, force_magnitude, force_location)
    plot_energy_convergence(result_discrete, result_bezier)


if __name__ == "__main__":
    main()
