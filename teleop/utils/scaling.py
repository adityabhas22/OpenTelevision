"""
Arm scaling utilities for VR teleoperation.

Maps human arm movements to robot workspace, handling differences in:
- Arm length (human vs robot)
- Workspace size
- Coordinate frames
"""

import numpy as np
from typing import Tuple, Optional


def scale_wrist_pose(
    human_pose: np.ndarray,
    human_arm_length: float = 0.60,
    robot_arm_length: float = 0.50,
    origin_offset: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Scale human wrist pose to robot workspace.
    
    Scales the translation component of the pose matrix by the ratio
    of robot to human arm length. This maps human arm movements to
    the robot's reachable workspace.
    
    Args:
        human_pose: 4x4 transformation matrix from VR tracking.
        human_arm_length: Human arm length in meters (default 0.60m).
        robot_arm_length: Robot arm length in meters (default 0.50m).
        origin_offset: Optional [x, y, z] offset to apply after scaling.
    
    Returns:
        4x4 transformation matrix scaled for robot.
    
    Example:
        >>> human_pose = preprocessor.get_wrist_pose()  # From VR
        >>> robot_pose = scale_wrist_pose(human_pose, 
        ...     human_arm_length=0.60, 
        ...     robot_arm_length=0.50)
        >>> ik_solver.solve(SE3Pose.from_matrix(robot_pose))
    """
    scale = robot_arm_length / human_arm_length
    
    robot_pose = human_pose.copy()
    robot_pose[:3, 3] = robot_pose[:3, 3] * scale
    
    if origin_offset is not None:
        robot_pose[:3, 3] += origin_offset
    
    return robot_pose


def scale_dual_arm_poses(
    left_human: np.ndarray,
    right_human: np.ndarray,
    human_arm_length: float = 0.60,
    robot_arm_length: float = 0.50,
    left_offset: Optional[np.ndarray] = None,
    right_offset: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale both arm poses for dual-arm teleoperation.
    
    Args:
        left_human: 4x4 transform for left wrist.
        right_human: 4x4 transform for right wrist.
        human_arm_length: Human arm length in meters.
        robot_arm_length: Robot arm length in meters.
        left_offset: Optional offset for left arm origin.
        right_offset: Optional offset for right arm origin.
    
    Returns:
        Tuple of (left_robot_pose, right_robot_pose).
    """
    left_robot = scale_wrist_pose(left_human, human_arm_length, robot_arm_length, left_offset)
    right_robot = scale_wrist_pose(right_human, human_arm_length, robot_arm_length, right_offset)
    
    return left_robot, right_robot


def clamp_to_workspace(
    pose: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray
) -> np.ndarray:
    """
    Clamp wrist position to robot workspace limits.
    
    Ensures the target position is reachable by clipping to
    a bounding box defined by workspace limits.
    
    Args:
        pose: 4x4 transformation matrix.
        workspace_min: [x_min, y_min, z_min] in meters.
        workspace_max: [x_max, y_max, z_max] in meters.
    
    Returns:
        Clamped 4x4 transformation matrix.
    """
    clamped = pose.copy()
    clamped[:3, 3] = np.clip(pose[:3, 3], workspace_min, workspace_max)
    return clamped


def compute_arm_length_from_urdf(
    robot_model,
    shoulder_link: str,
    wrist_link: str
) -> float:
    """
    Estimate arm length from URDF by computing distance between
    shoulder and wrist in the zero configuration.
    
    Args:
        robot_model: RobotModel instance.
        shoulder_link: Name of shoulder link.
        wrist_link: Name of wrist/end-effector link.
    
    Returns:
        Estimated arm length in meters.
    """
    import jax.numpy as jnp
    
    # Get pose at zero configuration
    zero_angles = jnp.zeros(robot_model.num_joints)
    ee_pose = robot_model.forward_kinematics(zero_angles)
    
    # The translation gives approximate arm length
    # (This is a simplification - actual arm length depends on structure)
    arm_length = float(jnp.linalg.norm(ee_pose.position))
    
    return arm_length


class ArmScaler:
    """
    Stateful arm scaler with calibration support.
    
    Handles arm length calibration and workspace clamping.
    
    Example:
        >>> scaler = ArmScaler(robot_arm_length=0.50)
        >>> scaler.calibrate_human_arm(0.62)  # Measured from VR
        >>> 
        >>> for vr_frame in tracking:
        >>>     robot_pose = scaler.scale(vr_frame.wrist_pose)
    """
    
    def __init__(
        self,
        robot_arm_length: float = 0.50,
        human_arm_length: float = 0.60,
        workspace_min: Optional[np.ndarray] = None,
        workspace_max: Optional[np.ndarray] = None
    ):
        """
        Initialize arm scaler.
        
        Args:
            robot_arm_length: Robot arm length in meters.
            human_arm_length: Initial human arm length estimate.
            workspace_min: Optional workspace lower bounds [x, y, z].
            workspace_max: Optional workspace upper bounds [x, y, z].
        """
        self.robot_arm_length = robot_arm_length
        self.human_arm_length = human_arm_length
        self.workspace_min = workspace_min
        self.workspace_max = workspace_max
    
    def calibrate_human_arm(self, measured_length: float) -> None:
        """Update human arm length from measurement."""
        self.human_arm_length = measured_length
        print(f"[ArmScaler] Calibrated human arm length: {measured_length:.3f}m")
        print(f"[ArmScaler] Scale factor: {self.scale_factor:.3f}")
    
    @property
    def scale_factor(self) -> float:
        """Current scaling factor."""
        return self.robot_arm_length / self.human_arm_length
    
    def scale(self, human_pose: np.ndarray) -> np.ndarray:
        """
        Scale a human wrist pose to robot workspace.
        
        Args:
            human_pose: 4x4 transformation from VR.
        
        Returns:
            Scaled 4x4 transformation for robot.
        """
        robot_pose = scale_wrist_pose(
            human_pose,
            self.human_arm_length,
            self.robot_arm_length
        )
        
        if self.workspace_min is not None and self.workspace_max is not None:
            robot_pose = clamp_to_workspace(
                robot_pose,
                self.workspace_min,
                self.workspace_max
            )
        
        return robot_pose
