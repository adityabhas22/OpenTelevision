"""
Smooth IK Solver with Regularization.

Enhanced version of OptimizedIKSolver that adds:
1. Regularization cost - keeps joints near a rest pose (prevents elbow drift)
2. Smoothness cost - penalizes large changes from previous solution
3. Post-processing filter - reduces output jitter

Based on Unitree's xr_teleoperate approach for natural arm motion.

Performance:
- Achieves >90 Hz with smoothness costs enabled
- Produces natural, jitter-free arm motion suitable for VR teleoperation
"""

from typing import Optional, Tuple, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
import jaxopt
import numpy as np

from teleop.ik.robot import RobotModel
from teleop.ik.utils import SE3Pose
from teleop.utils.filters import WeightedMovingFilter


class SmoothIKResult(NamedTuple):
    """Result from the smooth IK solver."""
    joint_angles: Array  # (n_joints,) filtered solution
    raw_joint_angles: Array  # (n_joints,) unfiltered solution
    position_error: float  # L2 distance to target (meters)
    orientation_error: float  # Quaternion geodesic distance
    iterations: int  # Number of solver iterations
    converged: bool  # Whether tolerance was met


class SmoothIKSolver:
    """
    IK solver with smoothness and regularization for natural arm motion.
    
    Key features:
    - Regularization: Keeps joints near a rest pose, preventing elbow drift
    - Smoothness: Penalizes large changes between consecutive solves
    - Post-filter: WeightedMovingFilter reduces output jitter
    - Warm-start: Uses previous solution for faster convergence
    
    Loss function (matching Unitree's approach):
        L = 50*position² + 1*orientation² + 0.02*regularization² + 0.1*smoothness²
    
    Example:
        >>> robot = RobotModel("robot.urdf", end_effector_link="wrist")
        >>> solver = SmoothIKSolver(robot)
        >>> 
        >>> # Teleoperation loop
        >>> for vr_frame in tracking:
        >>>     target = SE3Pose(vr_frame.position, vr_frame.quaternion)
        >>>     result = solver.solve(target)
        >>>     send_to_robot(result.joint_angles)  # Smooth, filtered output
    """
    
    def __init__(
        self,
        robot: RobotModel,
        position_weight: float = 50.0,
        orientation_weight: float = 1.0,
        regularization_weight: float = 0.02,
        smoothness_weight: float = 0.1,
        max_iterations: int = 30,
        tolerance: float = 1e-3,
        rest_pose: Optional[Array] = None,
        filter_weights: list = [0.4, 0.3, 0.2, 0.1],
        enable_filter: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the smooth IK solver.
        
        Args:
            robot: The RobotModel to solve IK for.
            position_weight: Weight for position error (default 50.0, matches Unitree).
            orientation_weight: Weight for orientation error (default 1.0).
            regularization_weight: Weight for distance from rest pose (default 0.02).
            smoothness_weight: Weight for change from previous (default 0.1).
            max_iterations: Maximum solver iterations.
            tolerance: Convergence tolerance (meters).
            rest_pose: Preferred joint configuration. If None, uses center of limits.
            filter_weights: Weights for post-processing filter.
            enable_filter: Whether to apply post-processing smoothing.
            verbose: Print debug information.
        """
        self.robot = robot
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.regularization_weight = regularization_weight
        self.smoothness_weight = smoothness_weight
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.enable_filter = enable_filter
        
        # Get joint limits
        self._lower_limits, self._upper_limits = robot.joint_limits
        
        # Rest pose (center of limits if not specified)
        if rest_pose is not None:
            self._rest_pose = jnp.array(rest_pose)
        else:
            self._rest_pose = (self._lower_limits + self._upper_limits) / 2
        
        # Previous solution for smoothness cost
        self._prev_solution = self._rest_pose.copy()
        
        # Post-processing filter
        if enable_filter:
            self._filter = WeightedMovingFilter(filter_weights, robot.num_joints)
        else:
            self._filter = None
        
        # Create the residual function with all costs
        self._residual_fn = self._make_residual_fn()
        
        # Create Levenberg-Marquardt solver (robust for this problem)
        self._solver = jaxopt.LevenbergMarquardt(
            residual_fun=self._residual_fn,
            maxiter=max_iterations,
            tol=tolerance,
            verbose=verbose,
            damping_parameter=1e-4,
        )
        
        # JIT compile
        self._solve_jit = jax.jit(self._solve_impl)
    
    def _make_residual_fn(self):
        """
        Create residual function with all cost terms.
        
        Residual structure:
        [
            sqrt(50) * position_error (3),       # Position
            sqrt(1) * orientation_error (4),     # Orientation
            sqrt(0.02) * (q - rest_pose) (n),    # Regularization  
            sqrt(0.1) * (q - prev_solution) (n)  # Smoothness
        ]
        """
        robot = self.robot
        
        # Pre-compute weight sqrts
        pos_w = jnp.sqrt(self.position_weight)
        ori_w = jnp.sqrt(self.orientation_weight)
        reg_w = jnp.sqrt(self.regularization_weight)
        smooth_w = jnp.sqrt(self.smoothness_weight)
        
        rest_pose = self._rest_pose
        
        def residual_fn(
            joint_angles: Array, 
            target_data: Tuple[Array, Array, Array]
        ) -> Array:
            """
            Compute residual vector with all cost terms.
            
            Args:
                joint_angles: Current joint configuration (n_joints,)
                target_data: (target_pos, target_quat, prev_solution)
            
            Returns:
                Residual vector for least-squares minimization
            """
            target_pos, target_quat, prev_solution = target_data
            
            # Forward kinematics
            current_pose = robot.forward_kinematics(joint_angles)
            
            # Position residual (3 elements)
            pos_residual = pos_w * (current_pose.position - target_pos)
            
            # Orientation residual (4 elements)
            q_curr = current_pose.quaternion
            q_targ = target_quat
            
            # Handle quaternion double-cover
            dot = jnp.sum(q_curr * q_targ)
            q_targ_aligned = jnp.where(dot < 0, -q_targ, q_targ)
            
            # Quaternion difference scaled by 2 (approximates axis-angle)
            ori_residual = ori_w * 2.0 * (q_curr - q_targ_aligned)
            
            # Regularization residual - keep near rest pose (n elements)
            reg_residual = reg_w * (joint_angles - rest_pose)
            
            # Smoothness residual - minimize change from previous (n elements)
            smooth_residual = smooth_w * (joint_angles - prev_solution)
            
            # Combine all residuals
            return jnp.concatenate([
                pos_residual,
                ori_residual,
                reg_residual,
                smooth_residual
            ])
        
        return residual_fn
    
    def _solve_impl(
        self,
        initial_guess: Array,
        target_pos: Array,
        target_quat: Array,
        prev_solution: Array
    ) -> Tuple[Array, jaxopt.OptStep]:
        """Internal JIT-compiled solve implementation."""
        target_data = (target_pos, target_quat, prev_solution)
        
        result = self._solver.run(initial_guess, target_data)
        
        # Clip to joint limits
        solution = jnp.clip(result.params, self._lower_limits, self._upper_limits)
        
        return solution, result.state
    
    def solve(
        self,
        target: SE3Pose,
        initial_guess: Optional[Array] = None
    ) -> SmoothIKResult:
        """
        Solve IK for a target pose with smoothness.
        
        Uses the previous solution for smoothness cost and as initial guess
        (warm-starting). After solving, updates internal state for next solve.
        
        Args:
            target: Desired end-effector pose.
            initial_guess: Starting joint angles. If None, uses previous solution.
        
        Returns:
            SmoothIKResult with filtered and raw solutions.
        """
        if initial_guess is None:
            initial_guess = self._prev_solution
        
        # Ensure within limits
        initial_guess = jnp.clip(initial_guess, self._lower_limits, self._upper_limits)
        
        # Solve
        raw_solution, state = self._solve_jit(
            initial_guess,
            target.position,
            target.quaternion,
            self._prev_solution
        )
        
        # Update previous solution for next frame
        self._prev_solution = raw_solution
        
        # Apply post-processing filter
        if self._filter is not None:
            self._filter.add_data(np.array(raw_solution))
            filtered_solution = jnp.array(self._filter.filtered_data)
        else:
            filtered_solution = raw_solution
        
        # Compute errors on raw solution
        final_pose = self.robot.forward_kinematics(raw_solution)
        pos_error = float(jnp.linalg.norm(final_pose.position - target.position))
        
        dot = jnp.sum(final_pose.quaternion * target.quaternion)
        ori_error = float(1.0 - jnp.abs(dot))
        
        converged = pos_error < self.tolerance
        iterations = int(getattr(state, 'iter_num', self.max_iterations))
        
        return SmoothIKResult(
            joint_angles=filtered_solution,
            raw_joint_angles=raw_solution,
            position_error=pos_error,
            orientation_error=ori_error,
            iterations=iterations,
            converged=converged
        )
    
    def reset(self, initial_pose: Optional[Array] = None) -> None:
        """
        Reset the solver state.
        
        Call this when starting a new teleoperation session.
        
        Args:
            initial_pose: Joint angles to reset to. If None, uses rest pose.
        """
        if initial_pose is not None:
            self._prev_solution = jnp.array(initial_pose)
        else:
            self._prev_solution = self._rest_pose.copy()
        
        if self._filter is not None:
            self._filter.reset()
    
    def set_rest_pose(self, rest_pose: Array) -> None:
        """Update the rest pose for regularization."""
        self._rest_pose = jnp.array(rest_pose)
        # Need to recreate residual function with new rest pose
        self._residual_fn = self._make_residual_fn()
        self._solver = jaxopt.LevenbergMarquardt(
            residual_fun=self._residual_fn,
            maxiter=self.max_iterations,
            tol=self.tolerance,
            verbose=self.verbose,
            damping_parameter=1e-4,
        )
        self._solve_jit = jax.jit(self._solve_impl)
    
    def benchmark(self, num_solves: int = 100) -> dict:
        """
        Run a performance benchmark with smoothness enabled.
        
        Returns timing statistics including filter overhead.
        """
        import time
        
        rng = jax.random.PRNGKey(42)
        
        # Generate smooth trajectory (simulates VR tracking)
        angles_list = []
        current = self._rest_pose
        
        for i in range(num_solves):
            rng, subkey = jax.random.split(rng)
            delta = jax.random.normal(subkey, shape=(self.robot.num_joints,)) * 0.02
            current = jnp.clip(current + delta, self._lower_limits, self._upper_limits)
            angles_list.append(current)
        
        # Compute target poses via FK
        targets = []
        for angles in angles_list:
            pose = self.robot.forward_kinematics(angles)
            targets.append(pose)
        
        # Reset solver state
        self.reset()
        
        # Cold start
        start = time.perf_counter()
        result = self.solve(targets[0])
        cold_start_ms = (time.perf_counter() - start) * 1000
        
        # Warm start benchmark
        warm_times = []
        
        for i in range(1, num_solves):
            start = time.perf_counter()
            result = self.solve(targets[i])
            warm_time = (time.perf_counter() - start) * 1000
            warm_times.append(warm_time)
        
        warm_times = np.array(warm_times)
        
        return {
            "cold_start_ms": cold_start_ms,
            "warm_start_mean_ms": float(np.mean(warm_times)),
            "warm_start_std_ms": float(np.std(warm_times)),
            "warm_start_min_ms": float(np.min(warm_times)),
            "warm_start_max_ms": float(np.max(warm_times)),
            "warm_start_p95_ms": float(np.percentile(warm_times, 95)),
            "achievable_hz": 1000.0 / float(np.mean(warm_times)),
            "filter_enabled": self.enable_filter,
        }
