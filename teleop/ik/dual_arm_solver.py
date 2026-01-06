"""
Optimized Dual-Arm IK Solver.

Solves both arms in a single JIT-compiled kernel for maximum parallelism.
Eliminates Python overhead and sequential execution.
"""

from typing import Optional, NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from teleop.ik.robot import RobotModel
from teleop.ik.smooth_solver import SmoothIKSolver
from teleop.ik.utils import SE3Pose
from teleop.utils.filters import WeightedMovingFilter

class DualArmResult(NamedTuple):
    """Result from dual-arm IK solver."""
    left_joint_angles: Array
    right_joint_angles: Array
    all_joint_angles: Array
    left_converged: bool
    right_converged: bool
    left_position_error: float = 0.0
    right_position_error: float = 0.0
    left_orientation_error: float = 0.0
    right_orientation_error: float = 0.0

class DualArmIKSolver:
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
        # Initialize single-arm solvers to get their residual functions
        self.left_solver = SmoothIKSolver(
            left_robot, position_weight, orientation_weight, 
            regularization_weight, smoothness_weight, enable_filter=False
        )
        self.right_solver = SmoothIKSolver(
            right_robot, position_weight, orientation_weight, 
            regularization_weight, smoothness_weight, enable_filter=False
        )
        
        self.left_robot = left_robot
        self.right_robot = right_robot
        
        # Combined filter
        if enable_filter:
            total_joints = left_robot.num_joints + right_robot.num_joints
            self._filter = WeightedMovingFilter(filter_weights, total_joints)
        else:
            self._filter = None
            
        # Compile the combined solve kernel
        self._solve_dual_jit = jax.jit(self._solve_dual_impl)

    def _solve_dual_impl(
        self, 
        left_tgt_pos, left_tgt_quat, left_prev,
        right_tgt_pos, right_tgt_quat, right_prev
    ):
        # These run in parallel within the XLA graph
        left_sol, left_state = self.left_solver._solve_impl(
            left_prev, left_tgt_pos, left_tgt_quat, left_prev
        )
        right_sol, right_state = self.right_solver._solve_impl(
            right_prev, right_tgt_pos, right_tgt_quat, right_prev
        )
        return left_sol, right_sol, left_state, right_state

    def solve(
        self,
        left_target: SE3Pose,
        right_target: SE3Pose,
        compute_error: bool = True,
        tolerance: float = 0.01
    ) -> DualArmResult:
        """
        Solve IK for both arms simultaneously.
        
        Args:
            left_target: Target pose for left arm.
            right_target: Target pose for right arm.
            compute_error: If True, compute FK to verify solution quality.
            tolerance: Position error threshold for convergence (meters).
        
        Returns:
            DualArmResult with solutions and convergence status.
        """
        # Single JIT call for both arms
        left_sol, right_sol, l_state, r_state = self._solve_dual_jit(
            left_target.position, left_target.quaternion, self.left_solver._prev_solution,
            right_target.position, right_target.quaternion, self.right_solver._prev_solution
        )
        
        # Update state
        self.left_solver._prev_solution = left_sol
        self.right_solver._prev_solution = right_sol
        
        # Combined filtering (JAX)
        if self._filter is not None:
            combined = jnp.concatenate([left_sol, right_sol])
            filtered = self._filter.add_data_jax(combined)
            left_final = filtered[:self.left_robot.num_joints]
            right_final = filtered[self.left_robot.num_joints:]
            all_final = filtered
        else:
            left_final = left_sol
            right_final = right_sol
            all_final = jnp.concatenate([left_sol, right_sol])
        
        # Compute actual errors by running FK on solutions
        if compute_error:
            # Left arm error
            left_fk = self.left_robot.forward_kinematics(left_sol)
            left_pos_err = float(jnp.linalg.norm(left_fk.position - left_target.position))
            left_ori_dot = jnp.sum(left_fk.quaternion * left_target.quaternion)
            left_ori_err = float(1.0 - jnp.abs(left_ori_dot))
            left_converged = left_pos_err < tolerance
            
            # Right arm error
            right_fk = self.right_robot.forward_kinematics(right_sol)
            right_pos_err = float(jnp.linalg.norm(right_fk.position - right_target.position))
            right_ori_dot = jnp.sum(right_fk.quaternion * right_target.quaternion)
            right_ori_err = float(1.0 - jnp.abs(right_ori_dot))
            right_converged = right_pos_err < tolerance
        else:
            # Skip error computation for maximum speed (caller's responsibility)
            left_pos_err = 0.0
            left_ori_err = 0.0
            left_converged = True
            right_pos_err = 0.0
            right_ori_err = 0.0
            right_converged = True
            
        return DualArmResult(
            left_joint_angles=left_final,
            right_joint_angles=right_final,
            all_joint_angles=all_final,
            left_converged=left_converged,
            right_converged=right_converged,
            left_position_error=left_pos_err,
            right_position_error=right_pos_err,
            left_orientation_error=left_ori_err,
            right_orientation_error=right_ori_err
        )

    def reset(self):
        self.left_solver.reset()
        self.right_solver.reset()
        if self._filter:
            self._filter.jax_filter.reset()
