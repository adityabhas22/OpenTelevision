#!/usr/bin/env python3
"""
Comprehensive IK Solver Benchmark.

This script benchmarks both the original gradient descent solver and
the optimized Gauss-Newton solver, providing detailed latency and
frequency metrics for VR teleoperation.

Usage:
    JAX_ENABLE_X64=True python scripts/benchmark_ik.py

Expected results on NVIDIA GPU:
    Original Solver (GD):   ~50-200ms warm-start, ~2-20 Hz
    Optimized Solver (GN):  ~1-10ms warm-start, ~100-1000 Hz
    
For smooth VR teleop at 60 Hz, you need <16ms solve time.
For 90 Hz (Quest 3), you need <11ms solve time.
"""

import sys
from pathlib import Path
import time
from typing import List, Tuple
from dataclasses import dataclass

# Add the repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from teleop.ik import RobotModel, IKSolver, SE3Pose
from teleop.ik.optimized_solver import OptimizedIKSolver, BatchedIKSolver


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    cold_start_ms: float
    warm_start_mean_ms: float
    warm_start_std_ms: float
    warm_start_min_ms: float
    warm_start_max_ms: float
    warm_start_p95_ms: float
    achievable_hz: float
    avg_position_error_mm: float
    avg_iterations: float
    convergence_rate: float  # % of solves that converged


def print_banner(title: str) -> None:
    """Print a styled section banner."""
    width = 70
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}\n")


def print_result_table(results: List[BenchmarkResult]) -> None:
    """Print results as a comparison table."""
    print(f"\n{'─'*80}")
    print(f"{'Metric':<25} | ", end="")
    for r in results:
        print(f"{r.name:<20} | ", end="")
    print()
    print(f"{'─'*80}")
    
    metrics = [
        ("Cold Start (ms)", lambda r: f"{r.cold_start_ms:.2f}"),
        ("Warm Start Mean (ms)", lambda r: f"{r.warm_start_mean_ms:.2f}"),
        ("Warm Start Std (ms)", lambda r: f"{r.warm_start_std_ms:.2f}"),
        ("Warm Start P95 (ms)", lambda r: f"{r.warm_start_p95_ms:.2f}"),
        ("Achievable Hz", lambda r: f"{r.achievable_hz:.1f}"),
        ("Avg Position Err (mm)", lambda r: f"{r.avg_position_error_mm:.2f}"),
        ("Avg Iterations", lambda r: f"{r.avg_iterations:.1f}"),
        ("Convergence Rate", lambda r: f"{r.convergence_rate:.1%}"),
    ]
    
    for name, getter in metrics:
        print(f"{name:<25} | ", end="")
        for r in results:
            print(f"{getter(r):<20} | ", end="")
        print()
    
    print(f"{'─'*80}")


def generate_trajectory(
    robot: RobotModel,
    num_points: int,
    seed: int = 42
) -> Tuple[List[SE3Pose], jnp.ndarray]:
    """
    Generate a smooth trajectory of reachable targets.
    
    Simulates VR hand tracking where consecutive frames are similar.
    """
    rng = jax.random.PRNGKey(seed)
    
    # Start from a nominal configuration
    lower, upper = robot.joint_limits
    mid_pose = (lower + upper) / 2
    
    # Generate smooth random walk in joint space
    trajectory_angles = [mid_pose]
    current = mid_pose
    
    for i in range(num_points - 1):
        rng, subkey = jax.random.split(rng)
        # Small random step (simulates smooth hand movement)
        delta = jax.random.normal(subkey, shape=(robot.num_joints,)) * 0.02
        current = jnp.clip(current + delta, lower, upper)
        trajectory_angles.append(current)
    
    trajectory_angles = jnp.stack(trajectory_angles)
    
    # Convert to target poses via FK
    targets = []
    for angles in trajectory_angles:
        pose = robot.forward_kinematics(angles)
        targets.append(pose)
    
    return targets, trajectory_angles


def benchmark_original_solver(
    robot: RobotModel,
    targets: List[SE3Pose],
    ground_truth: jnp.ndarray
) -> BenchmarkResult:
    """Benchmark the original gradient descent solver."""
    
    solver = IKSolver(
        robot,
        position_weight=1.0,
        orientation_weight=0.3,
        joint_limit_weight=0.1,
        regularization_weight=0.01,
        learning_rate=0.2,
        max_iterations=100,
        tolerance=0.005
    )
    
    # Cold start (first solve includes JIT)
    start = time.perf_counter()
    result = solver.solve(targets[0])
    cold_start_ms = (time.perf_counter() - start) * 1000
    
    # Warm start benchmark
    warm_times = []
    position_errors = []
    iterations = []
    converged_count = 0
    
    current_solution = result.joint_angles
    
    for i in range(1, len(targets)):
        start = time.perf_counter()
        result = solver.solve_with_warmstart(targets[i], current_solution)
        warm_time = (time.perf_counter() - start) * 1000
        
        warm_times.append(warm_time)
        position_errors.append(result.position_error * 1000)  # Convert to mm
        iterations.append(result.iterations)
        if result.converged:
            converged_count += 1
        
        current_solution = result.joint_angles
    
    warm_times = np.array(warm_times)
    
    return BenchmarkResult(
        name="Original (GD)",
        cold_start_ms=cold_start_ms,
        warm_start_mean_ms=np.mean(warm_times),
        warm_start_std_ms=np.std(warm_times),
        warm_start_min_ms=np.min(warm_times),
        warm_start_max_ms=np.max(warm_times),
        warm_start_p95_ms=np.percentile(warm_times, 95),
        achievable_hz=1000.0 / np.mean(warm_times),
        avg_position_error_mm=np.mean(position_errors),
        avg_iterations=np.mean(iterations),
        convergence_rate=converged_count / (len(targets) - 1)
    )


def benchmark_optimized_solver(
    robot: RobotModel,
    targets: List[SE3Pose],
    ground_truth: jnp.ndarray,
    solver_type: str = "gauss_newton"
) -> BenchmarkResult:
    """Benchmark the optimized Gauss-Newton solver."""
    
    solver = OptimizedIKSolver(
        robot,
        position_weight=1.0,
        orientation_weight=0.1,
        max_iterations=30,
        tolerance=0.001,
        solver_type=solver_type
    )
    
    # Cold start (first solve includes JIT)
    start = time.perf_counter()
    result = solver.solve(targets[0])
    cold_start_ms = (time.perf_counter() - start) * 1000
    
    # Warm start benchmark
    warm_times = []
    position_errors = []
    iterations = []
    converged_count = 0
    
    current_solution = result.joint_angles
    
    for i in range(1, len(targets)):
        start = time.perf_counter()
        result = solver.solve_warmstart(targets[i], current_solution)
        warm_time = (time.perf_counter() - start) * 1000
        
        warm_times.append(warm_time)
        position_errors.append(result.position_error * 1000)  # Convert to mm
        iterations.append(result.iterations)
        if result.converged:
            converged_count += 1
        
        current_solution = result.joint_angles
    
    warm_times = np.array(warm_times)
    
    name = "Optimized (GN)" if solver_type == "gauss_newton" else "Optimized (LM)"
    
    return BenchmarkResult(
        name=name,
        cold_start_ms=cold_start_ms,
        warm_start_mean_ms=np.mean(warm_times),
        warm_start_std_ms=np.std(warm_times),
        warm_start_min_ms=np.min(warm_times),
        warm_start_max_ms=np.max(warm_times),
        warm_start_p95_ms=np.percentile(warm_times, 95),
        achievable_hz=1000.0 / np.mean(warm_times),
        avg_position_error_mm=np.mean(position_errors),
        avg_iterations=np.mean(iterations),
        convergence_rate=converged_count / (len(targets) - 1)
    )


def print_vr_compatibility(results: List[BenchmarkResult]) -> None:
    """Print VR compatibility analysis."""
    print_banner("VR Teleoperation Compatibility")
    
    headsets = [
        ("Quest 2", 72),
        ("Quest 3", 90),
        ("Quest Pro", 90),
        ("Apple Vision Pro", 90),
        ("Valve Index", 144),
    ]
    
    print(f"{'Headset':<25} | {'Required (ms)':<15} | ", end="")
    for r in results:
        print(f"{r.name:<20} | ", end="")
    print()
    print(f"{'─'*100}")
    
    for headset, hz in headsets:
        required_ms = 1000.0 / hz
        print(f"{headset:<25} | {required_ms:<15.2f} | ", end="")
        for r in results:
            if r.warm_start_p95_ms < required_ms:
                status = "✅ Compatible"
            elif r.warm_start_mean_ms < required_ms:
                status = "⚠️ Marginal"
            else:
                status = "❌ Too slow"
            print(f"{status:<20} | ", end="")
        print()


def main():
    """Run the complete benchmark suite."""
    
    print_banner("IK Solver Benchmark Suite")
    
    # Check JAX device
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    
    if any("cuda" in str(d).lower() for d in devices):
        print("✅ GPU acceleration enabled")
    else:
        print("⚠️ Running on CPU - results will be slower")
    
    # Load robot
    urdf_path = REPO_ROOT / "assets" / "h1_inspire" / "urdf" / "h1_inspire.urdf"
    
    if not urdf_path.exists():
        print(f"\n❌ ERROR: URDF not found at {urdf_path}")
        return 1
    
    robot = RobotModel(
        str(urdf_path),
        base_link="torso_link",
        end_effector_link="right_wrist_yaw_link"
    )
    
    print(f"\nRobot: {robot.name}")
    print(f"Joints: {robot.num_joints}")
    print(f"Chain: torso_link → right_wrist_yaw_link")
    
    # Generate test trajectory
    print_banner("Generating Test Trajectory")
    NUM_FRAMES = 200
    print(f"Generating {NUM_FRAMES} frames (simulating ~3 seconds at 60Hz)...")
    
    targets, ground_truth = generate_trajectory(robot, NUM_FRAMES)
    print(f"Generated {len(targets)} reachable target poses")
    
    # Run benchmarks
    print_banner("Running Benchmarks")
    results = []
    
    print("Benchmarking Original Solver (Gradient Descent)...")
    result_original = benchmark_original_solver(robot, targets, ground_truth)
    results.append(result_original)
    print(f"  Done: {result_original.warm_start_mean_ms:.2f} ms mean, {result_original.achievable_hz:.1f} Hz")
    
    print("\nBenchmarking Optimized Solver (Gauss-Newton)...")
    try:
        result_gn = benchmark_optimized_solver(robot, targets, ground_truth, "gauss_newton")
        results.append(result_gn)
        print(f"  Done: {result_gn.warm_start_mean_ms:.2f} ms mean, {result_gn.achievable_hz:.1f} Hz")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\nBenchmarking Optimized Solver (Levenberg-Marquardt)...")
    try:
        result_lm = benchmark_optimized_solver(robot, targets, ground_truth, "levenberg_marquardt")
        results.append(result_lm)
        print(f"  Done: {result_lm.warm_start_mean_ms:.2f} ms mean, {result_lm.achievable_hz:.1f} Hz")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Print comparison table
    print_banner("Benchmark Results")
    print_result_table(results)
    
    # VR compatibility analysis
    print_vr_compatibility(results)
    
    # Performance summary
    print_banner("Summary")
    
    best = min(results, key=lambda r: r.warm_start_mean_ms)
    speedup = results[0].warm_start_mean_ms / best.warm_start_mean_ms if best != results[0] else 1.0
    
    print(f"Best solver: {best.name}")
    print(f"Speedup vs Original: {speedup:.1f}x")
    print(f"Achievable frequency: {best.achievable_hz:.1f} Hz")
    print(f"P95 latency: {best.warm_start_p95_ms:.2f} ms")
    
    if best.warm_start_p95_ms < 11:
        print("\n✅ Ready for 90Hz VR teleoperation!")
    elif best.warm_start_p95_ms < 16:
        print("\n⚠️ Suitable for 60Hz VR, may drop frames at 90Hz")
    else:
        print("\n❌ Further optimization needed for smooth VR")
    
    print("\n" + "="*70)
    print("  Benchmark Complete")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
