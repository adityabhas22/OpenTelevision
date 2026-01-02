#!/usr/bin/env python3
"""
Comprehensive IK Solver Stress Test

This script tests the IK solver across the robot's entire reachable workspace
to evaluate accuracy and identify failure modes.

Tests include:
1. Grid sampling of the workspace
2. Random pose sampling
3. Edge cases (singularities, joint limits)
4. Parameter tuning comparison
"""

import sys
import time
from pathlib import Path
from typing import Tuple, List
import numpy as np

# Add the repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
from jax import random

from teleop.ik import RobotModel, IKSolver, SE3Pose


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_stats(errors: List[float], times: List[float], label: str) -> None:
    """Print statistics for a set of IK solves."""
    errors = np.array(errors)
    times = np.array(times)
    
    success_rate = np.sum(errors < 0.01) / len(errors) * 100  # < 1cm = success
    
    print(f"\n{label}:")
    print(f"  Position Error (cm):")
    print(f"    Mean: {np.mean(errors)*100:.2f}, Median: {np.median(errors)*100:.2f}")
    print(f"    Min: {np.min(errors)*100:.2f}, Max: {np.max(errors)*100:.2f}")
    print(f"    Std: {np.std(errors)*100:.2f}")
    print(f"  Success Rate (<1cm): {success_rate:.1f}%")
    print(f"  Solve Time (ms):")
    print(f"    Mean: {np.mean(times):.2f}, Median: {np.median(times):.2f}")


def test_parameter_comparison(robot: RobotModel) -> dict:
    """
    Test 1: Compare different solver parameter configurations.
    """
    print_header("Test 1: Parameter Tuning Comparison")
    
    # Generate a set of test targets using FK from random joint configs
    key = random.PRNGKey(42)
    num_tests = 20
    
    test_targets = []
    for i in range(num_tests):
        key, subkey = random.split(key)
        lower, upper = robot.joint_limits
        random_angles = random.uniform(subkey, (robot.num_joints,), minval=lower, maxval=upper)
        pose = robot.forward_kinematics(random_angles)
        test_targets.append((pose, random_angles))
    
    # Different parameter configurations to test
    configs = {
        "Conservative (current)": {
            "position_weight": 1.0,
            "orientation_weight": 0.3,
            "joint_limit_weight": 0.1,
            "regularization_weight": 0.01,
            "learning_rate": 0.15,
            "max_iterations": 200,
        },
        "Position-focused": {
            "position_weight": 2.0,
            "orientation_weight": 0.1,
            "joint_limit_weight": 0.05,
            "regularization_weight": 0.0,
            "learning_rate": 0.2,
            "max_iterations": 150,
        },
        "Aggressive LR": {
            "position_weight": 1.0,
            "orientation_weight": 0.2,
            "joint_limit_weight": 0.1,
            "regularization_weight": 0.01,
            "learning_rate": 0.4,
            "max_iterations": 100,
        },
        "High Iterations": {
            "position_weight": 1.0,
            "orientation_weight": 0.3,
            "joint_limit_weight": 0.1,
            "regularization_weight": 0.01,
            "learning_rate": 0.1,
            "max_iterations": 500,
        },
    }
    
    best_config = None
    best_success = 0
    
    for config_name, params in configs.items():
        print(f"\nTesting: {config_name}")
        solver = IKSolver(robot, **params)
        
        errors = []
        times = []
        
        for target_pose, _ in test_targets:
            start = time.time()
            result = solver.solve(target_pose, initial_guess=jnp.zeros(robot.num_joints))
            elapsed = (time.time() - start) * 1000
            
            errors.append(result.position_error)
            times.append(elapsed)
        
        print_stats(errors, times, config_name)
        
        success_rate = np.sum(np.array(errors) < 0.01) / len(errors) * 100
        if success_rate > best_success:
            best_success = success_rate
            best_config = config_name
    
    print(f"\n★ Best Config: {best_config} ({best_success:.1f}% success)")
    return configs[best_config]


def test_workspace_grid(robot: RobotModel, solver: IKSolver) -> None:
    """
    Test 2: Grid sampling across the workspace.
    """
    print_header("Test 2: Workspace Grid Sampling")
    
    # Sample the joint space in a grid
    lower, upper = robot.joint_limits
    num_samples_per_joint = 3  # 3^7 = 2187 samples (for 7-DOF)
    
    # For speed, only sample a subset of joints fully
    grid_samples = []
    for i in range(num_samples_per_joint ** min(robot.num_joints, 4)):
        # Create a sample by varying first 4 joints, keeping rest at 0
        sample = jnp.zeros(robot.num_joints)
        idx = i
        for j in range(min(robot.num_joints, 4)):
            val = idx % num_samples_per_joint
            idx //= num_samples_per_joint
            # Map to joint range
            t = val / (num_samples_per_joint - 1)
            sample = sample.at[j].set(lower[j] + t * (upper[j] - lower[j]))
        grid_samples.append(sample)
    
    print(f"Testing {len(grid_samples)} grid positions...")
    
    errors = []
    times = []
    failures = []
    
    for i, joint_config in enumerate(grid_samples):
        # Get target pose from FK
        target_pose = robot.forward_kinematics(joint_config)
        
        # Solve IK from zeros (cold start)
        start = time.time()
        result = solver.solve(target_pose, initial_guess=jnp.zeros(robot.num_joints))
        elapsed = (time.time() - start) * 1000
        
        errors.append(result.position_error)
        times.append(elapsed)
        
        if result.position_error > 0.05:  # > 5cm is a significant failure
            failures.append({
                "target_joints": joint_config,
                "solved_joints": result.joint_angles,
                "error": result.position_error,
            })
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(grid_samples)}")
    
    print_stats(errors, times, "Grid Sampling Results")
    
    if failures:
        print(f"\n⚠️  Significant failures (>5cm): {len(failures)}")
        print("  Worst failures:")
        failures.sort(key=lambda x: x["error"], reverse=True)
        for fail in failures[:3]:
            print(f"    Error: {fail['error']*100:.1f}cm at joints: {fail['target_joints'][:3]}...")


def test_edge_cases(robot: RobotModel, solver: IKSolver) -> None:
    """
    Test 3: Edge cases and potential singularities.
    """
    print_header("Test 3: Edge Cases & Singularities")
    
    lower, upper = robot.joint_limits
    
    edge_cases = {
        "All joints at lower limits": lower,
        "All joints at upper limits": upper,
        "All joints at zero": jnp.zeros(robot.num_joints),
        "All joints at midpoint": (lower + upper) / 2,
        "Alternating limits": jnp.where(jnp.arange(robot.num_joints) % 2 == 0, lower, upper),
    }
    
    for case_name, joint_config in edge_cases.items():
        target_pose = robot.forward_kinematics(joint_config)
        
        start = time.time()
        result = solver.solve(target_pose, initial_guess=jnp.zeros(robot.num_joints))
        elapsed = (time.time() - start) * 1000
        
        status = "✅" if result.position_error < 0.01 else "⚠️" if result.position_error < 0.05 else "❌"
        print(f"{status} {case_name}: err={result.position_error*100:.2f}cm, time={elapsed:.1f}ms")


def test_continuous_trajectory(robot: RobotModel, solver: IKSolver) -> None:
    """
    Test 4: Simulate a continuous trajectory (teleop-like).
    """
    print_header("Test 4: Continuous Trajectory (Teleop Simulation)")
    
    # Create a circular trajectory in Cartesian space
    num_waypoints = 50
    center = jnp.array([0.15, -0.1, 0.0])  # Approximate center of workspace
    radius = 0.05  # 5cm radius circle
    
    print(f"Simulating {num_waypoints}-point circular trajectory...")
    
    errors = []
    times = []
    current_angles = jnp.zeros(robot.num_joints)
    
    for i in range(num_waypoints):
        angle = 2 * jnp.pi * i / num_waypoints
        target_pos = center + jnp.array([radius * jnp.cos(angle), 0, radius * jnp.sin(angle)])
        target_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # Fixed orientation
        target = SE3Pose(position=target_pos, quaternion=target_quat)
        
        start = time.time()
        result = solver.solve_with_warmstart(target, current_angles)
        elapsed = (time.time() - start) * 1000
        
        current_angles = result.joint_angles
        errors.append(result.position_error)
        times.append(elapsed)
    
    print_stats(errors, times, "Trajectory Tracking Results")
    
    # Calculate "smoothness" (jerk in joint space)
    angles_history = [jnp.zeros(robot.num_joints)]  # placeholder
    print(f"  Average solve frequency: {1000/np.mean(times):.1f} Hz")


def main():
    """Run all optimization tests."""
    print("\n" + "=" * 70)
    print("  IK Solver Comprehensive Stress Test")
    print("=" * 70)
    
    # Load robot
    urdf_path = REPO_ROOT / "assets" / "h1_inspire" / "urdf" / "h1_inspire.urdf"
    robot = RobotModel(
        str(urdf_path),
        base_link="torso_link",
        end_effector_link="right_wrist_yaw_link"
    )
    print(f"\nLoaded: {robot}")
    
    # Test 1: Find best parameters
    best_params = test_parameter_comparison(robot)
    
    # Create optimized solver
    print("\n--- Using optimized parameters for remaining tests ---")
    solver = IKSolver(robot, **best_params)
    
    # Test 2: Grid sampling
    test_workspace_grid(robot, solver)
    
    # Test 3: Edge cases
    test_edge_cases(robot, solver)
    
    # Test 4: Trajectory
    test_continuous_trajectory(robot, solver)
    
    print_header("All Tests Complete")
    print("\nReview the results above to understand the solver's performance")
    print("across different regions of the robot's workspace.\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
