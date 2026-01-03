"""
IK Solver Package

A modular, extensible inverse kinematics library built on JAX.
Designed for VR teleoperation of robot arms.

Solvers:
    IKSolver: Original gradient descent solver (simple, slower)
    OptimizedIKSolver: Gauss-Newton/LM solver (fast, for production)
    SmoothIKSolver: LM solver with smoothness/regularization (natural motion)
    DualArmIKSolver: Wrapper for solving both arms with combined filtering
    BatchedIKSolver: Parallel solver for multiple targets (GPU-optimized)
"""

from teleop.ik.robot import RobotModel
from teleop.ik.solver import IKSolver
from teleop.ik.utils import SE3Pose
from teleop.ik import losses
from teleop.ik.optimized_solver import OptimizedIKSolver, BatchedIKSolver, IKResult
from teleop.ik.smooth_solver import SmoothIKSolver, SmoothIKResult
from teleop.ik.dual_arm_solver import DualArmIKSolver, DualArmResult

__all__ = [
    "RobotModel",
    "IKSolver",
    "OptimizedIKSolver",
    "SmoothIKSolver",
    "DualArmIKSolver",
    "BatchedIKSolver", 
    "IKResult",
    "SmoothIKResult",
    "DualArmResult",
    "SE3Pose",
    "losses",
]
