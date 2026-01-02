"""
IK Solver Package

A modular, extensible inverse kinematics library built on JAX.
Designed for VR teleoperation of robot arms.
"""

from teleop.ik.robot import RobotModel
from teleop.ik.solver import IKSolver
from teleop.ik.utils import SE3Pose
from teleop.ik import losses

__all__ = ["RobotModel", "IKSolver", "SE3Pose", "losses"]
