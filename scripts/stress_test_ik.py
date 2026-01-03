#!/usr/bin/env python3
"""
Comprehensive Stress Tests for IK Solvers.

Simulates realistic VR teleoperation scenarios:
1. Smooth continuous tracking (normal operation)
2. Fast jerky movements (user quickly moving)
3. Edge of workspace (reaching limits)
4. Random noise (tracking jitter from VR)
5. Long duration (thermal/memory stability)
6. Comparison of all solvers

Usage:
    # Run all tests
    python scripts/stress_test_ik.py
    
    # Run specific test
    python scripts/stress_test_ik.py --test smooth
    
    # Quick mode (fewer iterations)
    python scripts/stress_test_ik.py --quick
"""

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Tuple, Callable
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teleop.ik import (
    RobotModel, 
    OptimizedIKSolver, 
    SmoothIKSolver,
    DualArmIKSolver,
    SE3Pose
)
from teleop.utils.filters import WeightedMovingFilter


class TestResult:
    """Container for test results."""
    def __init__(self, name: str):
        self.name = name
        self.solve_times: List[float] = []
        self.position_errors: List[float] = []
        self.orientation_errors: List[float] = []
        self.converged: List[bool] = []
        self.joint_changes: List[float] = []
    
    def add(self, solve_time_ms: float, pos_err: float, ori_err: float, 
            converged: bool, joint_change: float = 0):
        self.solve_times.append(solve_time_ms)
        self.position_errors.append(pos_err)
        self.orientation_errors.append(ori_err)
        self.converged.append(converged)
        self.joint_changes.append(joint_change)
    
    def summary(self) -> dict:
        times = np.array(self.solve_times)
        pos_errs = np.array(self.position_errors) * 1000  # to mm
        joint_changes = np.array(self.joint_changes)
        
        return {
            "name": self.name,
            "solve_time_mean_ms": float(np.mean(times)),
            "solve_time_p95_ms": float(np.percentile(times, 95)),
            "solve_time_max_ms": float(np.max(times)),
            "frequency_hz": 1000 / float(np.mean(times)),
            "position_error_mean_mm": float(np.mean(pos_errs)),
            "position_error_max_mm": float(np.max(pos_errs)),
            "convergence_rate": float(np.mean(self.converged)),
            "joint_change_mean_rad": float(np.mean(joint_changes)),
            "joint_change_max_rad": float(np.max(joint_changes)),
        }
    
    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"Test: {s['name']}")
        print(f"{'='*60}")
        print(f"  Solve Time:     {s['solve_time_mean_ms']:.2f} ms mean | "
              f"{s['solve_time_p95_ms']:.2f} ms P95 | {s['solve_time_max_ms']:.2f} ms max")
        print(f"  Frequency:      {s['frequency_hz']:.1f} Hz")
        print(f"  Position Error: {s['position_error_mean_mm']:.3f} mm mean | "
              f"{s['position_error_max_mm']:.3f} mm max")
        print(f"  Convergence:    {s['convergence_rate']*100:.1f}%")
        print(f"  Joint Change:   {s['joint_change_mean_rad']:.4f} rad mean | "
              f"{s['joint_change_max_rad']:.4f} rad max")
        
        # VR compatibility
        if s['solve_time_p95_ms'] < 6.9:
            print(f"  VR Compat:      ✅ Valve Index (144 Hz)")
        elif s['solve_time_p95_ms'] < 11.1:
            print(f"  VR Compat:      ✅ Quest 3 / AVP (90 Hz)")
        elif s['solve_time_p95_ms'] < 16.7:
            print(f"  VR Compat:      ✅ Quest 2 (60 Hz)")
        else:
            print(f"  VR Compat:      ⚠️ Data collection only")


def generate_smooth_trajectory(
    robot: RobotModel,
    num_frames: int,
    speed: float = 0.02,
    seed: int = 42
) -> List[SE3Pose]:
    """Generate smooth continuous trajectory (normal VR tracking)."""
    rng = jax.random.PRNGKey(seed)
    lower, upper = robot.joint_limits
    current = (lower + upper) / 2
    
    targets = []
    for i in range(num_frames):
        rng, subkey = jax.random.split(rng)
        delta = jax.random.normal(subkey, shape=(robot.num_joints,)) * speed
        current = jnp.clip(current + delta, lower, upper)
        targets.append(robot.forward_kinematics(current))
    
    return targets


def generate_jerky_trajectory(
    robot: RobotModel,
    num_frames: int,
    jerk_probability: float = 0.1,
    seed: int = 42
) -> List[SE3Pose]:
    """Generate trajectory with occasional fast movements."""
    rng = jax.random.PRNGKey(seed)
    lower, upper = robot.joint_limits
    current = (lower + upper) / 2
    
    targets = []
    for i in range(num_frames):
        rng, k1, k2 = jax.random.split(rng, 3)
        
        # Random jerk check
        if float(jax.random.uniform(k1)) < jerk_probability:
            # Large sudden movement
            delta = jax.random.normal(k2, shape=(robot.num_joints,)) * 0.2
        else:
            # Normal smooth movement
            delta = jax.random.normal(k2, shape=(robot.num_joints,)) * 0.02
        
        current = jnp.clip(current + delta, lower, upper)
        targets.append(robot.forward_kinematics(current))
    
    return targets


def generate_workspace_edge_trajectory(
    robot: RobotModel,
    num_frames: int,
    seed: int = 42
) -> List[SE3Pose]:
    """Generate trajectory that explores workspace edges."""
    rng = jax.random.PRNGKey(seed)
    lower, upper = robot.joint_limits
    
    targets = []
    for i in range(num_frames):
        rng, subkey = jax.random.split(rng)
        
        # Bias towards extremes (U-shaped distribution)
        raw = jax.random.uniform(subkey, shape=(robot.num_joints,))
        biased = jnp.where(raw < 0.5, raw * 0.3, 1 - (1 - raw) * 0.3)
        
        angles = lower + biased * (upper - lower)
        targets.append(robot.forward_kinematics(angles))
    
    return targets


def generate_noisy_trajectory(
    robot: RobotModel,
    num_frames: int,
    noise_level: float = 0.05,
    seed: int = 42
) -> List[SE3Pose]:
    """Smooth trajectory with added VR tracking noise."""
    rng = jax.random.PRNGKey(seed)
    lower, upper = robot.joint_limits
    current = (lower + upper) / 2
    
    targets = []
    for i in range(num_frames):
        rng, k1, k2 = jax.random.split(rng, 3)
        
        # Smooth underlying motion
        delta = jax.random.normal(k1, shape=(robot.num_joints,)) * 0.02
        current = jnp.clip(current + delta, lower, upper)
        
        # Add noise
        noise = jax.random.normal(k2, shape=(robot.num_joints,)) * noise_level
        noisy = jnp.clip(current + noise, lower, upper)
        
        targets.append(robot.forward_kinematics(noisy))
    
    return targets


def run_solver_test(
    solver,
    targets: List[SE3Pose],
    test_name: str,
    warmup: int = 5
) -> TestResult:
    """Run a solver through a trajectory and collect metrics."""
    result = TestResult(test_name)
    
    # Reset solver if possible
    if hasattr(solver, 'reset'):
        solver.reset()
    
    prev_angles = None
    
    for i, target in enumerate(targets):
        start = time.perf_counter()
        
        if hasattr(solver, 'solve'):
            ik_result = solver.solve(target)
            angles = ik_result.joint_angles
            pos_err = ik_result.position_error
            ori_err = getattr(ik_result, 'orientation_error', 0)
            converged = ik_result.converged
        else:
            # For solvers with solve_warmstart
            if prev_angles is None:
                ik_result = solver.solve(target)
            else:
                ik_result = solver.solve_warmstart(target, prev_angles)
            angles = ik_result.joint_angles
            pos_err = ik_result.position_error
            ori_err = ik_result.orientation_error
            converged = ik_result.converged
        
        solve_time = (time.perf_counter() - start) * 1000
        
        # Calculate joint change
        if prev_angles is not None:
            joint_change = float(jnp.linalg.norm(angles - prev_angles))
        else:
            joint_change = 0
        
        prev_angles = angles
        
        # Skip warmup frames
        if i >= warmup:
            result.add(solve_time, pos_err, ori_err, converged, joint_change)
    
    return result


def test_smooth_tracking(robot: RobotModel, solver, num_frames: int = 500) -> TestResult:
    """Test normal smooth VR tracking."""
    print("\n[TEST] Smooth Continuous Tracking...")
    targets = generate_smooth_trajectory(robot, num_frames)
    return run_solver_test(solver, targets, "Smooth Tracking")


def test_jerky_movements(robot: RobotModel, solver, num_frames: int = 500) -> TestResult:
    """Test handling of sudden fast movements."""
    print("\n[TEST] Jerky Movements (10% sudden jumps)...")
    targets = generate_jerky_trajectory(robot, num_frames, jerk_probability=0.1)
    return run_solver_test(solver, targets, "Jerky Movements")


def test_workspace_edges(robot: RobotModel, solver, num_frames: int = 500) -> TestResult:
    """Test reaching to workspace limits."""
    print("\n[TEST] Workspace Edge Exploration...")
    targets = generate_workspace_edge_trajectory(robot, num_frames)
    return run_solver_test(solver, targets, "Workspace Edges")


def test_noisy_tracking(robot: RobotModel, solver, num_frames: int = 500) -> TestResult:
    """Test with simulated VR tracking noise."""
    print("\n[TEST] Noisy VR Tracking...")
    targets = generate_noisy_trajectory(robot, num_frames, noise_level=0.05)
    return run_solver_test(solver, targets, "Noisy Tracking")


def test_long_duration(robot: RobotModel, solver, duration_seconds: float = 60) -> TestResult:
    """Test stability over long duration."""
    # Estimate frames needed for target duration
    # First, measure baseline solve time
    test_targets = generate_smooth_trajectory(robot, 10)
    start = time.perf_counter()
    for t in test_targets:
        if hasattr(solver, 'solve'):
            solver.solve(t)
        else:
            solver.solve(t)
    baseline_time = (time.perf_counter() - start) / 10
    
    num_frames = int(duration_seconds / baseline_time)
    print(f"\n[TEST] Long Duration ({duration_seconds}s, ~{num_frames} frames)...")
    
    targets = generate_smooth_trajectory(robot, num_frames)
    return run_solver_test(solver, targets, f"Long Duration ({duration_seconds}s)")


def compare_solvers(robot: RobotModel, num_frames: int = 200):
    """Compare all available solvers."""
    print("\n" + "="*70)
    print("SOLVER COMPARISON")
    print("="*70)
    
    # Generate shared trajectory
    targets = generate_smooth_trajectory(robot, num_frames)
    
    solvers = [
        ("OptimizedIKSolver (LM)", OptimizedIKSolver(robot, solver_type='levenberg_marquardt')),
        ("OptimizedIKSolver (GN)", OptimizedIKSolver(robot, solver_type='gauss_newton')),
        ("SmoothIKSolver (no filter)", SmoothIKSolver(robot, enable_filter=False)),
        ("SmoothIKSolver (filtered)", SmoothIKSolver(robot, enable_filter=True)),
    ]
    
    results = []
    for name, solver in solvers:
        print(f"\nTesting: {name}")
        result = run_solver_test(solver, targets, name)
        result.print_summary()
        results.append(result)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Solver':<35} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Hz':<10} {'Error (mm)':<12}")
    print("-"*70)
    for r in results:
        s = r.summary()
        print(f"{s['name']:<35} {s['solve_time_mean_ms']:<12.2f} "
              f"{s['solve_time_p95_ms']:<12.2f} {s['frequency_hz']:<10.1f} "
              f"{s['position_error_mean_mm']:<12.3f}")


def test_dual_arm(num_frames: int = 200):
    """Test dual-arm solver with both arms."""
    print("\n" + "="*70)
    print("DUAL-ARM SOLVER TEST")
    print("="*70)
    
    left_robot = RobotModel(
        'assets/h1_inspire/urdf/h1_inspire.urdf',
        base_link='torso_link',
        end_effector_link='left_wrist_yaw_link'
    )
    right_robot = RobotModel(
        'assets/h1_inspire/urdf/h1_inspire.urdf',
        base_link='torso_link',
        end_effector_link='right_wrist_yaw_link'
    )
    
    solver = DualArmIKSolver(left_robot, right_robot)
    
    # Generate targets for both arms
    left_targets = generate_smooth_trajectory(left_robot, num_frames)
    right_targets = generate_smooth_trajectory(right_robot, num_frames, seed=123)
    
    result = TestResult("Dual Arm (both arms)")
    
    print(f"\nTesting dual-arm with {num_frames} frames...")
    solver.reset()
    
    for i in range(num_frames):
        start = time.perf_counter()
        ik_result = solver.solve(left_targets[i], right_targets[i])
        solve_time = (time.perf_counter() - start) * 1000
        
        if i >= 5:  # Skip warmup
            # Handle optimized result without error metrics
            left_err = getattr(ik_result, 'left_position_error', 0.0)
            right_err = getattr(ik_result, 'right_position_error', 0.0)
            
            result.add(
                solve_time,
                (left_err + right_err) / 2,
                0,
                ik_result.left_converged and ik_result.right_converged,
                0
            )
    
    result.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Stress test IK solvers")
    parser.add_argument('--test', choices=['smooth', 'jerky', 'edge', 'noisy', 'long', 'compare', 'dual', 'all'],
                        default='all', help='Which test to run')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer iterations)')
    parser.add_argument('--solver', choices=['optimized', 'smooth', 'smooth_filtered'],
                        default='smooth_filtered', help='Which solver to test')
    args = parser.parse_args()
    
    # Set frame counts
    if args.quick:
        num_frames = 100
        duration = 10
    else:
        num_frames = 500
        duration = 60
    
    print("="*70)
    print("IK SOLVER STRESS TESTS")
    print("="*70)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Frames per test: {num_frames}")
    
    # Load robot
    print("\nLoading robot model...")
    robot = RobotModel(
        'assets/h1_inspire/urdf/h1_inspire.urdf',
        base_link='torso_link',
        end_effector_link='right_wrist_yaw_link'
    )
    print(f"Robot loaded: {robot.num_joints} joints")
    
    # Create solver
    if args.solver == 'optimized':
        solver = OptimizedIKSolver(robot, solver_type='levenberg_marquardt')
    elif args.solver == 'smooth':
        solver = SmoothIKSolver(robot, enable_filter=False)
    else:
        solver = SmoothIKSolver(robot, enable_filter=True)
    
    print(f"Solver: {args.solver}")
    
    # Run tests
    results = []
    
    if args.test in ['smooth', 'all']:
        results.append(test_smooth_tracking(robot, solver, num_frames))
        results[-1].print_summary()
    
    if args.test in ['jerky', 'all']:
        results.append(test_jerky_movements(robot, solver, num_frames))
        results[-1].print_summary()
    
    if args.test in ['edge', 'all']:
        results.append(test_workspace_edges(robot, solver, num_frames))
        results[-1].print_summary()
    
    if args.test in ['noisy', 'all']:
        results.append(test_noisy_tracking(robot, solver, num_frames))
        results[-1].print_summary()
    
    if args.test in ['long', 'all'] and not args.quick:
        results.append(test_long_duration(robot, solver, duration))
        results[-1].print_summary()
    
    if args.test in ['compare', 'all']:
        compare_solvers(robot, num_frames)
    
    if args.test in ['dual', 'all']:
        test_dual_arm(num_frames)
    
    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)
    main()
