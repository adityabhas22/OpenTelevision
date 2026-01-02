"""
Composable loss functions for IK optimization.

All loss functions are designed to be:
1. Differentiable (JAX-compatible)
2. Composable (return scalars that can be summed)
3. Extensible (easy to add custom losses)

Usage:
    total_loss = (
        1.0 * position_loss(current, target) +
        0.5 * orientation_loss(current_q, target_q) +
        0.1 * joint_limit_loss(angles, limits)
    )
"""

import jax.numpy as jnp
from jax import Array


def position_loss(current_pos: Array, target_pos: Array) -> Array:
    """
    L2 (squared) distance between current and target positions.
    
    Args:
        current_pos: Current end-effector position (3,).
        target_pos: Target position (3,).
        
    Returns:
        Scalar loss value (squared L2 distance).
    """
    diff = current_pos - target_pos
    return jnp.sum(diff * diff)


def orientation_loss(current_quat: Array, target_quat: Array) -> Array:
    """
    Geodesic distance on SO(3) using quaternion dot product.
    
    The formula 1 - |q1 · q2| gives 0 when orientations match
    and 1 when they are 180° apart. This handles the quaternion
    double-cover (q and -q represent the same rotation).
    
    Args:
        current_quat: Current orientation quaternion [w, x, y, z] (4,).
        target_quat: Target orientation quaternion [w, x, y, z] (4,).
        
    Returns:
        Scalar loss value in [0, 1].
    """
    dot = jnp.sum(current_quat * target_quat)
    return 1.0 - jnp.abs(dot)


def joint_limit_loss(
    joint_angles: Array,
    lower_limits: Array,
    upper_limits: Array,
    margin: float = 0.1
) -> Array:
    """
    Soft penalty for approaching joint limits.
    
    Uses a quadratic penalty that activates within `margin` radians
    of the joint limits. This encourages the solver to stay away
    from mechanical limits.
    
    Args:
        joint_angles: Current joint angles (n_joints,).
        lower_limits: Lower joint limits (n_joints,).
        upper_limits: Upper joint limits (n_joints,).
        margin: Distance (in radians) from limit at which penalty starts.
        
    Returns:
        Scalar loss value.
    """
    # Penalty for being close to lower limit
    lower_violation = jnp.maximum(0, (lower_limits + margin) - joint_angles)
    
    # Penalty for being close to upper limit
    upper_violation = jnp.maximum(0, joint_angles - (upper_limits - margin))
    
    return jnp.sum(lower_violation**2 + upper_violation**2)


def smoothness_loss(current_angles: Array, previous_angles: Array) -> Array:
    """
    Penalizes large changes in joint angles between timesteps.
    
    This helps prevent jerky motion during teleoperation by
    encouraging smooth trajectories.
    
    Args:
        current_angles: Current joint angles (n_joints,).
        previous_angles: Previous timestep's joint angles (n_joints,).
        
    Returns:
        Scalar loss value (squared L2 distance in joint space).
    """
    diff = current_angles - previous_angles
    return jnp.sum(diff * diff)


def regularization_loss(joint_angles: Array, rest_pose: Array) -> Array:
    """
    Encourages the robot to stay near a comfortable rest pose.
    
    Useful when there are multiple IK solutions; this biases
    toward a "home" configuration.
    
    Args:
        joint_angles: Current joint angles (n_joints,).
        rest_pose: Preferred resting joint angles (n_joints,).
        
    Returns:
        Scalar loss value (squared L2 distance from rest pose).
    """
    diff = joint_angles - rest_pose
    return jnp.sum(diff * diff)


def manipulability_loss(jacobian: Array) -> Array:
    """
    Penalizes configurations near kinematic singularities.
    
    Maximizes the manipulability measure (sqrt of det(J @ J.T)),
    which is zero at singularities. We return the negative log
    to penalize low manipulability.
    
    Args:
        jacobian: The robot's Jacobian matrix (6, n_joints) or (3, n_joints).
        
    Returns:
        Scalar loss value (higher = closer to singularity).
    """
    # Yoshikawa's manipulability measure
    JJT = jacobian @ jacobian.T
    det = jnp.linalg.det(JJT)
    
    # Add small epsilon to avoid log(0)
    manipulability = jnp.sqrt(jnp.maximum(det, 1e-10))
    
    # Return negative log (minimize this to maximize manipulability)
    return -jnp.log(manipulability + 1e-10)
