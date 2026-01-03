"""
Dual-Arm IK Solver.

Wrapper for solving both arms simultaneously with:
- Independent per-arm solving (parallelizable)
- Combined post-processing filter
- Shared smoothness state
"""

from typing import Optional, NamedTuple, Tuple
import numpy as np
import jax.numpy as jnp
from jax import Array

from teleop.ik.robot import RobotModel
from teleop.ik.smooth_solver import SmoothIKSolver, SmoothIKResult
from teleop.ik.utils import SE3Pose
from teleop.utils.filters import WeightedMovingFilter


class DualArmResult(NamedTuple):
    """Result from dual-arm IK solver."""
    left_joint_angles: Array  # Left arm joints (filtered)
    right_joint_angles: Array  # Right arm joints (filtered)
    all_joint_angles: Array  # Combined [left, right]
    left_position_error: float
    right_position_error: float
    left_converged: bool
    right_converged: bool


class DualArmIKSolver:
    """
    Solver for dual-arm teleoperation.
    
    Creates two independent SmoothIKSolvers (one per arm) and combines
    their outputs with shared filtering for synchronized motion.
    
    Example:
        >>> left_robot = RobotModel("robot.urdf", end_effector_link="left_wrist")
        >>> right_robot = RobotModel("robot.urdf", end_effector_link="right_wrist")
        >>> solver = DualArmIKSolver(left_robot, right_robot)
        >>> 
        >>> for vr_frame in tracking:
        >>>     result = solver.solve(vr_frame.left_wrist, vr_frame.right_wrist)
        >>>     send_to_robot(result.all_joint_angles)
    """
    
    def __init__(
        self,
        left_robot: RobotModel,
        right_robot: RobotModel,
        position_weight: float = 50.0,
        orientation_weight: float = 1.0,
        regularization_weight: float = 0.02,
        smoothness_weight: float = 0.1,
        filter_weights: list = [0.4, 0.3, 0.2, 0.1],
        enable_filter: bool = True
    ):
        """
        Initialize dual-arm solver.
        
        Args:
            left_robot: RobotModel for left arm (with appropriate end-effector).
            right_robot: RobotModel for right arm.
            position_weight: Position error weight (Unitree: 50).
            orientation_weight: Orientation weight (Unitree: 1).
            regularization_weight: Regularization weight (Unitree: 0.02).
            smoothness_weight: Smoothness weight (Unitree: 0.1).
            filter_weights: Post-processing filter weights.
            enable_filter: Whether to apply combined filtering.
        """
        # Create per-arm solvers (without individual filters)
        self.left_solver = SmoothIKSolver(
            left_robot,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            regularization_weight=regularization_weight,
            smoothness_weight=smoothness_weight,
            enable_filter=False  # We'll use combined filter
        )
        
        self.right_solver = SmoothIKSolver(
            right_robot,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            regularization_weight=regularization_weight,
            smoothness_weight=smoothness_weight,
            enable_filter=False
        )
        
        self.left_robot = left_robot
        self.right_robot = right_robot
        self.enable_filter = enable_filter
        
        # Combined filter for both arms
        if enable_filter:
            total_joints = left_robot.num_joints + right_robot.num_joints
            self._filter = WeightedMovingFilter(filter_weights, total_joints)
        else:
            self._filter = None
    
    def solve(
        self,
        left_target: SE3Pose,
        right_target: SE3Pose,
        current_left: Optional[Array] = None,
        current_right: Optional[Array] = None
    ) -> DualArmResult:
        """
        Solve IK for both arms.
        
        Args:
            left_target: Target pose for left wrist.
            right_target: Target pose for right wrist.
            current_left: Current left arm joint angles (for warm-start).
            current_right: Current right arm joint angles.
        
        Returns:
            DualArmResult with filtered joint angles for both arms.
        """
        # Solve left arm
        left_result = self.left_solver.solve(left_target, current_left)
        
        # Solve right arm
        right_result = self.right_solver.solve(right_target, current_right)
        
        # Combine raw solutions
        raw_left = np.array(left_result.raw_joint_angles)
        raw_right = np.array(right_result.raw_joint_angles)
        combined_raw = np.concatenate([raw_left, raw_right])
        
        # Apply combined filter
        if self._filter is not None:
            self._filter.add_data(combined_raw)
            filtered = self._filter.filtered_data
            filtered_left = jnp.array(filtered[:self.left_robot.num_joints])
            filtered_right = jnp.array(filtered[self.left_robot.num_joints:])
            all_filtered = jnp.array(filtered)
        else:
            filtered_left = left_result.joint_angles
            filtered_right = right_result.joint_angles
            all_filtered = jnp.concatenate([filtered_left, filtered_right])
        
        return DualArmResult(
            left_joint_angles=filtered_left,
            right_joint_angles=filtered_right,
            all_joint_angles=all_filtered,
            left_position_error=left_result.position_error,
            right_position_error=right_result.position_error,
            left_converged=left_result.converged,
            right_converged=right_result.converged
        )
    
    def reset(
        self,
        left_initial: Optional[Array] = None,
        right_initial: Optional[Array] = None
    ) -> None:
        """Reset solver states for both arms."""
        self.left_solver.reset(left_initial)
        self.right_solver.reset(right_initial)
        if self._filter is not None:
            self._filter.reset()
    
    def benchmark(self, num_solves: int = 100) -> dict:
        """Benchmark dual-arm solving performance."""
        import time
        import jax
        
        rng = jax.random.PRNGKey(42)
        
        # Generate targets for both arms
        left_ll, left_ul = self.left_robot.joint_limits
        right_ll, right_ul = self.right_robot.joint_limits
        
        left_current = (left_ll + left_ul) / 2
        right_current = (right_ll + right_ul) / 2
        
        left_targets = []
        right_targets = []
        
        for i in range(num_solves):
            rng, k1, k2 = jax.random.split(rng, 3)
            
            left_delta = jax.random.normal(k1, (self.left_robot.num_joints,)) * 0.02
            right_delta = jax.random.normal(k2, (self.right_robot.num_joints,)) * 0.02
            
            left_current = jnp.clip(left_current + left_delta, left_ll, left_ul)
            right_current = jnp.clip(right_current + right_delta, right_ll, right_ul)
            
            left_targets.append(self.left_robot.forward_kinematics(left_current))
            right_targets.append(self.right_robot.forward_kinematics(right_current))
        
        self.reset()
        
        # Cold start
        start = time.perf_counter()
        result = self.solve(left_targets[0], right_targets[0])
        cold_ms = (time.perf_counter() - start) * 1000
        
        # Warm starts
        times = []
        for i in range(1, num_solves):
            start = time.perf_counter()
            result = self.solve(left_targets[i], right_targets[i])
            times.append((time.perf_counter() - start) * 1000)
        
        times = np.array(times)
        
        return {
            "cold_start_ms": cold_ms,
            "warm_mean_ms": float(np.mean(times)),
            "warm_p95_ms": float(np.percentile(times, 95)),
            "achievable_hz": 1000.0 / float(np.mean(times)),
            "per_arm_hz": 2000.0 / float(np.mean(times)),  # Theoretical if parallel
        }
