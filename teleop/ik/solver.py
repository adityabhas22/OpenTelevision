"""
IK Solver using gradient-based optimization.

This module provides the IKSolver class which uses JAX's automatic
differentiation to solve inverse kinematics via gradient descent.

The solver is highly configurable through composable loss functions,
making it easy to add custom objectives like obstacle avoidance.
"""

from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from teleop.ik.robot import RobotModel
from teleop.ik.utils import SE3Pose
from teleop.ik import losses


class IKSolver:
    """
    Gradient-based inverse kinematics solver.
    
    Uses JAX autodiff to minimize a weighted sum of loss functions,
    finding joint angles that place the end-effector at a target pose.
    
    Attributes:
        robot: The RobotModel to solve IK for.
        position_weight: Weight for position matching loss.
        orientation_weight: Weight for orientation matching loss.
        joint_limit_weight: Weight for joint limit avoidance.
        regularization_weight: Weight for staying near rest pose.
    
    Example:
        >>> robot = RobotModel("robot.urdf", end_effector_link="hand")
        >>> solver = IKSolver(robot, position_weight=1.0, orientation_weight=0.5)
        >>> target = SE3Pose(jnp.array([0.5, 0.0, 0.3]), jnp.array([1, 0, 0, 0]))
        >>> solution = solver.solve(target)
        >>> print(solution.joint_angles)
    """
    
    def __init__(
        self,
        robot: RobotModel,
        position_weight: float = 1.0,
        orientation_weight: float = 0.5,
        joint_limit_weight: float = 0.1,
        regularization_weight: float = 0.01,
        rest_pose: Optional[Array] = None,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ):
        """
        Initialize the IK solver.
        
        Args:
            robot: The RobotModel instance.
            position_weight: Weight for position loss (higher = more important).
            orientation_weight: Weight for orientation loss.
            joint_limit_weight: Weight for joint limit penalty.
            regularization_weight: Weight for staying near rest_pose.
            rest_pose: Preferred joint configuration. Defaults to zeros.
            learning_rate: Step size for gradient descent.
            max_iterations: Maximum optimization steps.
            tolerance: Convergence threshold for position error.
        """
        self.robot = robot
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.joint_limit_weight = joint_limit_weight
        self.regularization_weight = regularization_weight
        self.rest_pose = rest_pose if rest_pose is not None else jnp.zeros(robot.num_joints)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Pre-compile the optimization step
        self._step_fn = jax.jit(self._optimization_step)
    
    def _total_loss(
        self,
        joint_angles: Array,
        target_pose: SE3Pose
    ) -> Array:
        """
        Compute the total weighted loss for IK optimization.
        
        This is the function we minimize via gradient descent.
        """
        # Get current end-effector pose
        current_pose = self.robot.forward_kinematics(joint_angles)
        
        total = 0.0
        
        # Position loss
        if self.position_weight > 0:
            pos_loss = losses.position_loss(
                current_pose.position, target_pose.position
            )
            total = total + self.position_weight * pos_loss
        
        # Orientation loss
        if self.orientation_weight > 0:
            ori_loss = losses.orientation_loss(
                current_pose.quaternion, target_pose.quaternion
            )
            total = total + self.orientation_weight * ori_loss
        
        # Joint limit penalty
        if self.joint_limit_weight > 0:
            lower, upper = self.robot.joint_limits
            limit_loss = losses.joint_limit_loss(joint_angles, lower, upper)
            total = total + self.joint_limit_weight * limit_loss
        
        # Regularization (stay near rest pose)
        if self.regularization_weight > 0:
            reg_loss = losses.regularization_loss(joint_angles, self.rest_pose)
            total = total + self.regularization_weight * reg_loss
        
        return total
    
    def _optimization_step(
        self,
        joint_angles: Array,
        target_pos: Array,
        target_quat: Array
    ) -> Tuple[Array, Array]:
        """
        Perform one gradient descent step.
        
        Returns the updated joint angles and the loss value.
        """
        target_pose = SE3Pose(position=target_pos, quaternion=target_quat)
        
        loss, grads = jax.value_and_grad(self._total_loss)(
            joint_angles, target_pose
        )
        
        # Gradient descent update
        new_angles = joint_angles - self.learning_rate * grads
        
        # Clip to joint limits
        lower, upper = self.robot.joint_limits
        new_angles = jnp.clip(new_angles, lower, upper)
        
        return new_angles, loss
    
    def solve(
        self,
        target: SE3Pose,
        initial_guess: Optional[Array] = None,
        max_iterations: Optional[int] = None,
        verbose: bool = False
    ) -> "IKSolution":
        """
        Solve IK for a target end-effector pose.
        
        Args:
            target: Desired end-effector pose (SE3Pose).
            initial_guess: Starting joint angles. Defaults to rest_pose.
            max_iterations: Override default max iterations.
            verbose: If True, print progress during optimization.
            
        Returns:
            IKSolution containing the result and metadata.
        """
        max_iters = max_iterations if max_iterations is not None else self.max_iterations
        
        # Initialize joint angles
        if initial_guess is not None:
            angles = initial_guess
        else:
            angles = self.rest_pose.copy()
        
        # Ensure angles are within limits
        lower, upper = self.robot.joint_limits
        angles = jnp.clip(angles, lower, upper)
        
        # Optimization loop
        final_loss = float('inf')
        for i in range(max_iters):
            angles, loss = self._step_fn(
                angles, target.position, target.quaternion
            )
            final_loss = float(loss)
            
            if verbose and i % 10 == 0:
                print(f"Iter {i:4d}: loss = {final_loss:.6f}")
            
            # Check convergence (position error)
            current_pose = self.robot.forward_kinematics(angles)
            pos_error = jnp.linalg.norm(current_pose.position - target.position)
            if pos_error < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {i} with error {pos_error:.6f}")
                break
        
        # Compute final pose
        final_pose = self.robot.forward_kinematics(angles)
        
        return IKSolution(
            joint_angles=angles,
            final_pose=final_pose,
            target_pose=target,
            iterations=i + 1,
            final_loss=final_loss,
            converged=(pos_error < self.tolerance)
        )
    
    def solve_with_warmstart(
        self,
        target: SE3Pose,
        previous_solution: Array
    ) -> "IKSolution":
        """
        Solve IK using a previous solution as the starting point.
        
        This is useful for teleoperation where consecutive targets
        are close together. Using the previous solution as initial_guess
        speeds up convergence and produces smoother motion.
        
        Args:
            target: Desired end-effector pose.
            previous_solution: Joint angles from the previous timestep.
            
        Returns:
            IKSolution for the new target.
        """
        return self.solve(target, initial_guess=previous_solution)


class IKSolution:
    """
    Result of an IK solve operation.
    
    Attributes:
        joint_angles: The solution joint angles (n_joints,).
        final_pose: The achieved end-effector pose.
        target_pose: The requested target pose.
        iterations: Number of optimization iterations performed.
        final_loss: Final value of the loss function.
        converged: Whether the solver converged within tolerance.
    """
    
    def __init__(
        self,
        joint_angles: Array,
        final_pose: SE3Pose,
        target_pose: SE3Pose,
        iterations: int,
        final_loss: float,
        converged: bool
    ):
        self.joint_angles = joint_angles
        self.final_pose = final_pose
        self.target_pose = target_pose
        self.iterations = iterations
        self.final_loss = final_loss
        self.converged = converged
    
    @property
    def position_error(self) -> float:
        """L2 distance between achieved and target positions."""
        return float(jnp.linalg.norm(
            self.final_pose.position - self.target_pose.position
        ))
    
    @property
    def orientation_error(self) -> float:
        """Quaternion geodesic distance between achieved and target orientations."""
        dot = jnp.sum(self.final_pose.quaternion * self.target_pose.quaternion)
        return float(1.0 - jnp.abs(dot))
    
    def __repr__(self) -> str:
        return (
            f"IKSolution(converged={self.converged}, "
            f"iters={self.iterations}, "
            f"pos_error={self.position_error:.4f}m, "
            f"ori_error={self.orientation_error:.4f})"
        )
