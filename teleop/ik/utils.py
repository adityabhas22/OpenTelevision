"""
Utility classes and functions for SE3 poses and rotation math.

This module provides a simple SE3Pose dataclass and helper functions
for converting between quaternions and rotation matrices using JAX.
"""

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
from jax import Array


@dataclass
class SE3Pose:
    """
    Represents a rigid body pose in 3D space (Special Euclidean group SE(3)).
    
    Attributes:
        position: A (3,) array representing [x, y, z] translation.
        quaternion: A (4,) array representing [w, x, y, z] orientation.
                    Uses scalar-first (Hamilton) convention.
    """
    position: Array
    quaternion: Array
    
    @classmethod
    def from_matrix(cls, matrix: Array) -> "SE3Pose":
        """Create an SE3Pose from a 4x4 homogeneous transformation matrix."""
        position = matrix[:3, 3]
        quaternion = quaternion_from_rotation_matrix(matrix[:3, :3])
        return cls(position=position, quaternion=quaternion)
    
    def to_matrix(self) -> Array:
        """Convert to a 4x4 homogeneous transformation matrix."""
        R = rotation_matrix_from_quaternion(self.quaternion)
        T = jnp.eye(4)
        T = T.at[:3, :3].set(R)
        T = T.at[:3, 3].set(self.position)
        return T


def rotation_matrix_from_quaternion(quat: Array) -> Array:
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    
    Args:
        quat: A (4,) array with scalar-first convention [w, x, y, z].
        
    Returns:
        A (3, 3) rotation matrix.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Normalize quaternion for safety
    norm = jnp.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Build rotation matrix
    return jnp.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


def quaternion_from_rotation_matrix(R: Array) -> Array:
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].
    
    Uses a numerically stable method with epsilon protection.
    
    Args:
        R: A (3, 3) rotation matrix.
        
    Returns:
        A (4,) quaternion array with scalar-first convention [w, x, y, z].
    """
    # Add small epsilon to avoid sqrt of negative numbers due to numerical errors
    eps = 1e-8
    
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    # Compute all cases with epsilon protection
    # Case 1: trace > 0
    s_w = jnp.sqrt(jnp.maximum(1.0 + trace, eps)) * 2
    qw_1 = 0.25 * s_w
    qx_1 = (R[2, 1] - R[1, 2]) / s_w
    qy_1 = (R[0, 2] - R[2, 0]) / s_w
    qz_1 = (R[1, 0] - R[0, 1]) / s_w
    
    # Case 2: R[0,0] is largest diagonal
    s_x = jnp.sqrt(jnp.maximum(1.0 + R[0, 0] - R[1, 1] - R[2, 2], eps)) * 2
    qw_2 = (R[2, 1] - R[1, 2]) / s_x
    qx_2 = 0.25 * s_x
    qy_2 = (R[0, 1] + R[1, 0]) / s_x
    qz_2 = (R[0, 2] + R[2, 0]) / s_x
    
    # Case 3: R[1,1] is largest diagonal
    s_y = jnp.sqrt(jnp.maximum(1.0 + R[1, 1] - R[0, 0] - R[2, 2], eps)) * 2
    qw_3 = (R[0, 2] - R[2, 0]) / s_y
    qx_3 = (R[0, 1] + R[1, 0]) / s_y
    qy_3 = 0.25 * s_y
    qz_3 = (R[1, 2] + R[2, 1]) / s_y
    
    # Case 4: R[2,2] is largest diagonal
    s_z = jnp.sqrt(jnp.maximum(1.0 + R[2, 2] - R[0, 0] - R[1, 1], eps)) * 2
    qw_4 = (R[1, 0] - R[0, 1]) / s_z
    qx_4 = (R[0, 2] + R[2, 0]) / s_z
    qy_4 = (R[1, 2] + R[2, 1]) / s_z
    qz_4 = 0.25 * s_z
    
    # Select based on conditions
    # Using nested where for JAX compatibility
    quat = jnp.where(
        trace > 0,
        jnp.array([qw_1, qx_1, qy_1, qz_1]),
        jnp.where(
            (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]),
            jnp.array([qw_2, qx_2, qy_2, qz_2]),
            jnp.where(
                R[1, 1] > R[2, 2],
                jnp.array([qw_3, qx_3, qy_3, qz_3]),
                jnp.array([qw_4, qx_4, qy_4, qz_4])
            )
        )
    )
    
    # Normalize to ensure unit quaternion
    quat = quat / jnp.linalg.norm(quat)
    
    # Ensure positive w (canonical form)
    quat = jnp.where(quat[0] < 0, -quat, quat)
    return quat


def normalize_quaternion(quat: Array) -> Array:
    """Normalize a quaternion to unit length."""
    return quat / jnp.linalg.norm(quat)


def quaternion_multiply(q1: Array, q2: Array) -> Array:
    """
    Multiply two quaternions (Hamilton product).
    
    Args:
        q1, q2: Quaternions in [w, x, y, z] format.
        
    Returns:
        The product quaternion q1 * q2 in [w, x, y, z] format.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
