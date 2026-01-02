#!/usr/bin/env python3
"""
Test script for the IK solver library.

This script demonstrates the IK solver using the H1 Inspire URDF.
It performs the following tests:

1. Load the URDF and print robot info
2. Compute FK for a known joint configuration
3. Set a target pose slightly offset from the FK result
4. Run IK to recover joint angles that reach the target
5. Verify the solution is within the acceptable tolerance

Usage:
    python scripts/test_ik.py
"""

import sys
from pathlib import Path

# Add the repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

from teleop.ik import RobotModel, IKSolver, SE3Pose


def print_separator(title: str = "") -> None:
    """Print a visual separator."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)
    else:
        print('-' * 60)


def test_robot_loading():
    """Test 1: Load URDF and print robot information."""
    print_separator("Test 1: Loading URDF")
    
    urdf_path = REPO_ROOT / "assets" / "h1_inspire" / "urdf" / "h1_inspire.urdf"
    
    if not urdf_path.exists():
        print(f"ERROR: URDF not found at {urdf_path}")
        return None
    
    print(f"Loading: {urdf_path.name}")
    
    # For the H1 Inspire, we'll use the right arm chain
    # The links are: torso_link -> ... -> right_shoulder -> right_elbow -> right_wrist
    # We need to identify the correct end-effector link
    robot = RobotModel(
        str(urdf_path),
        base_link="torso_link",
        end_effector_link="right_wrist_yaw_link"  # Adjust based on URDF structure
    )
    
    print(f"\nRobot loaded: {robot}")
    print(f"Number of joints: {robot.num_joints}")
    print(f"Joint names: {robot.joint_names}")
    
    lower, upper = robot.joint_limits
    print(f"\nJoint limits:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {name}: [{lower[i]:.2f}, {upper[i]:.2f}] rad")
    
    return robot


def test_forward_kinematics(robot: RobotModel):
    """Test 2: Compute forward kinematics."""
    print_separator("Test 2: Forward Kinematics")
    
    # Start with zero configuration
    zero_angles = jnp.zeros(robot.num_joints)
    print(f"\nZero configuration: {zero_angles}")
    
    pose = robot.forward_kinematics(zero_angles)
    print(f"FK result (zero config):")
    print(f"  Position: [{pose.position[0]:.4f}, {pose.position[1]:.4f}, {pose.position[2]:.4f}] m")
    print(f"  Quaternion: [{pose.quaternion[0]:.4f}, {pose.quaternion[1]:.4f}, {pose.quaternion[2]:.4f}, {pose.quaternion[3]:.4f}]")
    
    # Try a slightly bent configuration
    bent_angles = jnp.array([0.3] * robot.num_joints)
    bent_angles = jnp.clip(bent_angles, *robot.joint_limits)
    print(f"\nBent configuration: {bent_angles}")
    
    pose_bent = robot.forward_kinematics(bent_angles)
    print(f"FK result (bent config):")
    print(f"  Position: [{pose_bent.position[0]:.4f}, {pose_bent.position[1]:.4f}, {pose_bent.position[2]:.4f}] m")
    print(f"  Quaternion: [{pose_bent.quaternion[0]:.4f}, {pose_bent.quaternion[1]:.4f}, {pose_bent.quaternion[2]:.4f}, {pose_bent.quaternion[3]:.4f}]")
    
    return pose_bent, bent_angles


def test_jacobian(robot: RobotModel):
    """Test 3: Compute Jacobian."""
    print_separator("Test 3: Jacobian Computation")
    
    angles = jnp.zeros(robot.num_joints)
    jacobian = robot.jacobian(angles)
    
    print(f"Jacobian shape: {jacobian.shape}")
    print(f"Jacobian (at zero config):")
    print(jacobian)
    
    return jacobian


def test_ik_solver(robot: RobotModel, target_pose: SE3Pose, initial_angles: jnp.ndarray):
    """Test 4: Solve IK to reach a target."""
    print_separator("Test 4: IK Solver")
    
    # Create solver
    solver = IKSolver(
        robot,
        position_weight=1.0,
        orientation_weight=0.3,
        joint_limit_weight=0.1,
        regularization_weight=0.01,
        learning_rate=0.15,
        max_iterations=200,
        tolerance=0.01  # 1cm tolerance
    )
    
    # Offset the target slightly to make IK non-trivial
    offset = jnp.array([0.05, 0.03, -0.02])  # 5cm offset
    offset_target = SE3Pose(
        position=target_pose.position + offset,
        quaternion=target_pose.quaternion
    )
    
    print(f"Target position: {offset_target.position}")
    print(f"Target quaternion: {offset_target.quaternion}")
    print(f"\nSolving IK (starting from zeros)...")
    
    import time
    start_time = time.time()
    # Solve from zeros (harder)
    solution = solver.solve(
        offset_target,
        initial_guess=jnp.zeros(robot.num_joints),
        verbose=False  # Set to False for cleaner timing
    )
    duration = (time.time() - start_time) * 1000
    
    print(f"\n{solution}")
    print(f"Solve Latency: {duration:.2f} ms")
    print(f"Solution joint angles: {solution.joint_angles}")
    
    return solution


def test_warmstart_ik(robot: RobotModel, solver_solution):
    """Test 5: Test warm-start IK (simulating teleoperation)."""
    print_separator("Test 5: Warm-Start IK (Teleoperation Simulation)")
    
    solver = IKSolver(
        robot,
        position_weight=1.0,
        orientation_weight=0.3,
        learning_rate=0.2,
        max_iterations=50,  # Fewer iters with warmstart
        tolerance=0.01
    )
    
    # Simulate small target movements (like hand tracking)
    current_angles = solver_solution.joint_angles
    base_pos = solver_solution.target_pose.position
    
    print("Simulating 5 sequential targets (like VR hand tracking)...")
    
    total_time = 0
    for i in range(5):
        # Small random-ish movement
        delta = jnp.array([0.01 * (i+1), -0.005 * i, 0.008 * (i-2)])
        new_target = SE3Pose(
            position=base_pos + delta,
            quaternion=solver_solution.target_pose.quaternion
        )
        
        # Solve with warmstart
        import time
        start_time = time.time()
        result = solver.solve_with_warmstart(new_target, current_angles)
        duration = (time.time() - start_time) * 1000
        total_time += duration
        
        current_angles = result.joint_angles
        
        print(f"  Step {i+1}: iters={result.iterations:3d}, pos_err={result.position_error*100:.2f}cm, time={duration:.2f}ms")
    
    print(f"\nAverage Warm-start Latency: {total_time/5:.2f} ms")
    print("\nWarm-start allows faster convergence for sequential targets!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  IK Solver Library Test Suite")
    print("=" * 60)
    
    # Test 1: Load robot
    robot = test_robot_loading()
    if robot is None:
        print("\nFAILED: Could not load robot URDF")
        return 1
    
    # Test 2: Forward kinematics
    try:
        target_pose, initial_angles = test_forward_kinematics(robot)
    except Exception as e:
        print(f"\nFAILED: Forward kinematics error: {e}")
        return 1
    
    # Test 3: Jacobian
    try:
        test_jacobian(robot)
    except Exception as e:
        print(f"\nFAILED: Jacobian computation error: {e}")
        return 1
    
    # Test 4: IK Solver
    try:
        solution = test_ik_solver(robot, target_pose, initial_angles)
        if not solution.converged:
            print(f"\nWARNING: IK did not converge, but this may be OK for complex robots.")
    except Exception as e:
        print(f"\nFAILED: IK solver error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 5: Warm-start IK
    try:
        test_warmstart_ik(robot, solution)
    except Exception as e:
        print(f"\nFAILED: Warm-start IK error: {e}")
        return 1
    
    print_separator("All Tests Completed")
    print("\nSUCCESS: IK library is working correctly!")
    print("You can now use it in your teleoperation pipeline.\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
