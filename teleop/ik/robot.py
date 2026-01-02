"""
Robot model abstraction over URDF files.

This module provides the RobotModel class which:
1. Parses a URDF file to extract kinematic structure
2. Provides a differentiable forward kinematics function
3. Computes Jacobians for IK optimization

The implementation uses urdf-parser-py for URDF parsing and
JAX for differentiable computations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from urdf_parser_py.urdf import URDF, Joint, Link

from teleop.ik.utils import SE3Pose, rotation_matrix_from_quaternion


class RobotModel:
    """
    A differentiable robot model built from a URDF file.
    
    This class parses a URDF and creates JAX-compatible functions
    for computing forward kinematics and Jacobians.
    
    Attributes:
        name: The robot's name from the URDF.
        num_joints: Number of actuated joints.
        joint_names: List of joint names in kinematic order.
        joint_limits: Tuple of (lower_limits, upper_limits) arrays.
    
    Example:
        >>> robot = RobotModel("path/to/robot.urdf", end_effector_link="hand")
        >>> pose = robot.forward_kinematics(jnp.zeros(robot.num_joints))
        >>> print(pose.position)
    """
    
    def __init__(
        self,
        urdf_path: str,
        end_effector_link: Optional[str] = None,
        base_link: Optional[str] = None
    ):
        """
        Initialize the robot model from a URDF file.
        
        Args:
            urdf_path: Path to the URDF file.
            end_effector_link: Name of the end-effector link. If None,
                               uses the last link in the kinematic chain.
            base_link: Name of the base link. If None, uses the URDF root.
        """
        self._urdf_path = Path(urdf_path)
        self._urdf = URDF.from_xml_file(str(self._urdf_path))
        self.name = self._urdf.name
        
        # Build the kinematic chain
        self._base_link = base_link or self._find_root_link()
        self._end_effector_link = end_effector_link or self._find_last_link()
        
        # Extract actuated joints (revolute/prismatic only)
        self._chain = self._build_kinematic_chain()
        self._actuated_joints = [
            j for j in self._chain 
            if j.type in ("revolute", "prismatic", "continuous")
        ]
        
        self.joint_names = [j.name for j in self._actuated_joints]
        self.num_joints = len(self._actuated_joints)
        
        # Extract joint limits
        self._lower_limits = []
        self._upper_limits = []
        for joint in self._actuated_joints:
            if joint.limit is not None:
                self._lower_limits.append(joint.limit.lower or -jnp.pi)
                self._upper_limits.append(joint.limit.upper or jnp.pi)
            else:
                # Continuous joints have no limits
                self._lower_limits.append(-jnp.pi)
                self._upper_limits.append(jnp.pi)
        
        self._lower_limits = jnp.array(self._lower_limits)
        self._upper_limits = jnp.array(self._upper_limits)
        
        # Pre-compile the FK function for speed
        self._fk_fn = jax.jit(self._forward_kinematics_impl)
        self._jacobian_fn = jax.jit(jax.jacobian(self._fk_positions_only))
    
    @property
    def joint_limits(self) -> Tuple[Array, Array]:
        """Return (lower_limits, upper_limits) as JAX arrays."""
        return (self._lower_limits, self._upper_limits)
    
    def _find_root_link(self) -> str:
        """Find the root link of the URDF (link with no parent)."""
        child_links = {j.child for j in self._urdf.joints}
        for link in self._urdf.links:
            if link.name not in child_links:
                return link.name
        return self._urdf.links[0].name
    
    def _find_last_link(self) -> str:
        """Find a leaf link (link with no children)."""
        parent_links = {j.parent for j in self._urdf.joints}
        for link in self._urdf.links:
            if link.name not in parent_links:
                return link.name
        return self._urdf.links[-1].name
    
    def _build_kinematic_chain(self) -> List[Joint]:
        """
        Build the kinematic chain from base to end-effector.
        
        Returns a list of joints in order from base to end-effector.
        """
        # Build parent->child and child->parent maps
        joint_by_child: Dict[str, Joint] = {}
        for joint in self._urdf.joints:
            joint_by_child[joint.child] = joint
        
        # Walk from end-effector back to base
        chain = []
        current_link = self._end_effector_link
        
        while current_link != self._base_link:
            if current_link not in joint_by_child:
                raise ValueError(
                    f"Cannot find path from {self._base_link} to "
                    f"{self._end_effector_link}. Broken at {current_link}."
                )
            joint = joint_by_child[current_link]
            chain.append(joint)
            current_link = joint.parent
        
        # Reverse to get base->end-effector order
        chain.reverse()
        return chain
    
    def _joint_transform(self, joint: Joint, angle: float) -> Array:
        """
        Compute the 4x4 transformation matrix for a joint at a given angle.
        
        This includes both the fixed offset (origin) and the variable
        rotation/translation from the joint angle.
        """
        # Fixed transform from joint origin
        origin = joint.origin
        if origin is not None:
            xyz = origin.xyz if origin.xyz is not None else [0, 0, 0]
            rpy = origin.rpy if origin.rpy is not None else [0, 0, 0]
        else:
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
        
        T_origin = self._transform_from_xyz_rpy(xyz, rpy)
        
        # Variable transform from joint motion
        if joint.type == "fixed":
            T_joint = jnp.eye(4)
        elif joint.type in ("revolute", "continuous"):
            axis = joint.axis if joint.axis is not None else [0, 0, 1]
            T_joint = self._rotation_about_axis(axis, angle)
        elif joint.type == "prismatic":
            axis = joint.axis if joint.axis is not None else [0, 0, 1]
            T_joint = self._translation_along_axis(axis, angle)
        else:
            T_joint = jnp.eye(4)
        
        return T_origin @ T_joint
    
    @staticmethod
    def _transform_from_xyz_rpy(xyz: List[float], rpy: List[float]) -> Array:
        """Create a 4x4 transform from position XYZ and rotation RPY."""
        x, y, z = xyz
        roll, pitch, yaw = rpy
        
        # Rotation matrices for roll, pitch, yaw (ZYX convention)
        cr, sr = jnp.cos(roll), jnp.sin(roll)
        cp, sp = jnp.cos(pitch), jnp.sin(pitch)
        cy, sy = jnp.cos(yaw), jnp.sin(yaw)
        
        R = jnp.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        T = jnp.eye(4)
        T = T.at[:3, :3].set(R)
        T = T.at[:3, 3].set(jnp.array([x, y, z]))
        return T
    
    @staticmethod
    def _rotation_about_axis(axis: List[float], angle: float) -> Array:
        """Create a 4x4 rotation matrix about an arbitrary axis."""
        axis = jnp.array(axis, dtype=jnp.float32)
        axis = axis / jnp.linalg.norm(axis)
        
        c, s = jnp.cos(angle), jnp.sin(angle)
        t = 1 - c
        x, y, z = axis
        
        R = jnp.array([
            [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
        ])
        
        T = jnp.eye(4)
        T = T.at[:3, :3].set(R)
        return T
    
    @staticmethod
    def _translation_along_axis(axis: List[float], distance: float) -> Array:
        """Create a 4x4 translation matrix along an axis."""
        axis = jnp.array(axis, dtype=jnp.float32)
        axis = axis / jnp.linalg.norm(axis)
        
        T = jnp.eye(4)
        T = T.at[:3, 3].set(axis * distance)
        return T
    
    def _forward_kinematics_impl(self, joint_angles: Array) -> Array:
        """
        Internal FK implementation that returns a 4x4 matrix.
        
        This is the core FK function that gets JIT-compiled.
        """
        T = jnp.eye(4)
        joint_idx = 0
        
        for joint in self._chain:
            if joint.type in ("revolute", "prismatic", "continuous"):
                angle = joint_angles[joint_idx]
                joint_idx += 1
            else:
                angle = 0.0
            
            T = T @ self._joint_transform(joint, angle)
        
        return T
    
    def _fk_positions_only(self, joint_angles: Array) -> Array:
        """FK that returns only position (for Jacobian computation)."""
        T = self._forward_kinematics_impl(joint_angles)
        return T[:3, 3]
    
    def forward_kinematics(self, joint_angles: Array) -> SE3Pose:
        """
        Compute the end-effector pose for given joint angles.
        
        Args:
            joint_angles: Array of shape (num_joints,) with joint values.
            
        Returns:
            SE3Pose with the end-effector position and orientation.
        """
        T = self._fk_fn(joint_angles)
        return SE3Pose.from_matrix(T)
    
    def forward_kinematics_matrix(self, joint_angles: Array) -> Array:
        """
        Compute the end-effector pose as a 4x4 matrix.
        
        Useful when you need the raw matrix for further calculations.
        """
        return self._fk_fn(joint_angles)
    
    def jacobian(self, joint_angles: Array) -> Array:
        """
        Compute the positional Jacobian (3 x num_joints).
        
        The Jacobian relates joint velocities to end-effector velocity:
            dx/dt = J @ dq/dt
        
        Args:
            joint_angles: Current joint angles (num_joints,).
            
        Returns:
            Jacobian matrix of shape (3, num_joints).
        """
        return self._jacobian_fn(joint_angles)
    
    def __repr__(self) -> str:
        return (
            f"RobotModel(name='{self.name}', "
            f"joints={self.num_joints}, "
            f"chain='{self._base_link}' → '{self._end_effector_link}')"
        )
