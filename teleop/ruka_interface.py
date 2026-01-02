"""
RukaRobotInterface: Integration between OpenTeleVision and RUKA Hand

This module provides the interface to control the RUKA hand using hand tracking
data from Apple Vision Pro via OpenTeleVision.

The RUKA hand is underactuated:
    - 15 Degrees of Freedom (joints in the kinematic model)
    - 11 Dynamixel motors (tendon-driven actuation)
    
The RUKAOperator class handles this mapping internally using per-finger LSTM
controllers that learn the non-linear tendon dynamics.
"""

import sys
import numpy as np
from pathlib import Path

# Add RUKA package to path
RUKA_PATH = Path(__file__).parent.parent / "RUKA"
sys.path.insert(0, str(RUKA_PATH))

from ruka_hand.control.operator import RUKAOperator
from ruka_hand.utils.vectorops import calculate_fingertips, calculate_joint_angles


class RukaRobotInterface:
    """
    Interface between OpenTeleVision and RUKA hand.
    
    Takes hand landmarks from Apple Vision Pro and drives RUKA motors
    through pre-trained LSTM controllers. Handles the 15 DoF → 11 motor
    underactuation mapping internally via RUKAOperator.
    
    Attributes:
        hand_type (str): 'left' or 'right' hand
        operator (RUKAOperator): RUKA control operator with LSTM controllers
        
    Example:
        >>> interface = RukaRobotInterface(hand_type="right")
        >>> # keypoints from OpenTeleVision (25 landmarks, 3D positions)
        >>> interface.control(keypoints)
        >>> interface.close()
    """
    
    # Mapping from AVP 25-point landmarks to RUKA's 5x5 keypoint format
    # AVP format: [wrist, thumb(4), index(4), middle(4), ring(4), pinky(4), extra(4)]
    # RUKA format: [5 fingers][5 keypoints per finger][3D position]
    AVP_TO_RUKA_INDICES = {
        'thumb':  [0, 1, 2, 3, 4],     # wrist + 4 thumb joints
        'index':  [0, 5, 6, 7, 8],     # wrist + 4 index joints  
        'middle': [0, 9, 10, 11, 12],  # wrist + 4 middle joints
        'ring':   [0, 13, 14, 15, 16], # wrist + 4 ring joints
        'pinky':  [0, 17, 18, 19, 20], # wrist + 4 pinky joints
    }
    
    def __init__(
        self,
        hand_type: str = "right",
        moving_average_limit: int = 2,
        fingertip_overshoot_ratio: float = 0.0,
        joint_angle_overshoot_ratio: float = 0.0,
    ):
        """
        Initialize the RUKA robot interface.
        
        Args:
            hand_type: 'left' or 'right' hand
            moving_average_limit: Smoothing window size (higher = smoother but more lag)
            fingertip_overshoot_ratio: Amplification for fingertip movements (0.0 = off)
            joint_angle_overshoot_ratio: Amplification for joint angles (0.0 = off)
        """
        self.hand_type = hand_type
        self.moving_average_limit = moving_average_limit
        
        print(f"[RukaRobotInterface] Initializing {hand_type} hand...")
        print(f"[RukaRobotInterface] Loading LSTM controllers from checkpoints...")
        
        # Initialize the RUKA operator (loads LSTM controllers for each finger)
        self.operator = RUKAOperator(
            hand_type=hand_type,
            moving_average_limit=moving_average_limit,
            fingertip_overshoot_ratio=fingertip_overshoot_ratio,
            joint_angle_overshoot_ratio=joint_angle_overshoot_ratio,
        )
        
        print(f"[RukaRobotInterface] {hand_type.capitalize()} hand initialized successfully!")
        
    def _convert_avp_to_ruka_keypoints(self, avp_landmarks: np.ndarray) -> np.ndarray:
        """
        Convert AVP 25-point landmarks to RUKA's (5, 5, 3) keypoint format.
        
        Args:
            avp_landmarks: Shape (25, 3) - 25 hand landmarks from Apple Vision Pro
            
        Returns:
            keypoints: Shape (5, 5, 3) - 5 fingers, 5 keypoints each, 3D positions
        """
        keypoints = np.zeros((5, 5, 3))
        
        for finger_idx, (finger_name, indices) in enumerate(self.AVP_TO_RUKA_INDICES.items()):
            for joint_idx, avp_idx in enumerate(indices):
                keypoints[finger_idx, joint_idx] = avp_landmarks[avp_idx]
                
        return keypoints
    
    def control(self, keypoints: np.ndarray) -> None:
        """
        Primary control method - takes AVP landmarks and drives RUKA motors.
        
        The method:
        1. Converts AVP landmarks (25x3) to RUKA format (5x5x3)
        2. Passes to RUKAOperator which computes fingertip/joint angle features
        3. Per-finger LSTM controllers predict motor commands
        4. HandController sends commands to Dynamixel motors
        
        Args:
            keypoints: Shape (25, 3) - 25 hand landmarks from AVP/OpenTeleVision
                      Each row is [x, y, z] position in meters, relative to wrist
        """
        if keypoints.shape != (25, 3):
            raise ValueError(f"Expected keypoints shape (25, 3), got {keypoints.shape}")
            
        # Convert to RUKA's expected format
        ruka_keypoints = self._convert_avp_to_ruka_keypoints(keypoints)
        
        # Drive the motors through RUKA operator
        # This handles: keypoints → fingertips/joint_angles → LSTM → motor commands
        self.operator.step(ruka_keypoints)
        
    def control_raw(self, ruka_keypoints: np.ndarray) -> None:
        """
        Control with pre-formatted RUKA keypoints (bypasses conversion).
        
        Args:
            ruka_keypoints: Shape (5, 5, 3) - Already in RUKA's expected format
        """
        if ruka_keypoints.shape != (5, 5, 3):
            raise ValueError(f"Expected keypoints shape (5, 5, 3), got {ruka_keypoints.shape}")
            
        self.operator.step(ruka_keypoints)
    
    def reset(self) -> None:
        """
        Reset the hand to tensioned (fully open) position.
        
        Useful for:
        - Initial calibration
        - Returning to safe state
        - Between teleoperation sessions
        """
        print("[RukaRobotInterface] Resetting hand to tensioned position...")
        self.operator.reset()
        print("[RukaRobotInterface] Reset complete.")
        
    def close(self) -> None:
        """
        Clean shutdown of the RUKA hand.
        
        Closes the serial connection to Dynamixel motors.
        Always call this when done to properly release hardware resources.
        """
        print("[RukaRobotInterface] Closing connection...")
        self.operator.controller.close()
        print("[RukaRobotInterface] Connection closed.")
        
    def get_motor_positions(self) -> np.ndarray:
        """
        Get current motor positions.
        
        Returns:
            positions: Shape (11,) - Current position of each motor in Dynamixel units
        """
        return np.array(self.operator.controller.hand.read_pos())
    
    def get_joint_info(self) -> dict:
        """
        Get information about the RUKA hand's joint/motor configuration.
        
        Returns:
            dict with:
                - num_joints: 15 (kinematic DoF)
                - num_motors: 11 (actual actuators)
                - finger_to_motors: Mapping of finger names to motor IDs
        """
        from ruka_hand.utils.constants import FINGER_NAMES_TO_MOTOR_IDS
        
        return {
            'num_joints': 15,
            'num_motors': 11,
            'finger_to_motors': FINGER_NAMES_TO_MOTOR_IDS,
            'input_type': self.operator.controller.input_type,
        }


# Convenience function for quick testing
def test_ruka_interface():
    """Quick test of the RukaRobotInterface."""
    print("Testing RukaRobotInterface...")
    
    # Create interface
    interface = RukaRobotInterface(hand_type="right")
    
    # Print joint info
    info = interface.get_joint_info()
    print(f"Joint info: {info}")
    
    # Get current positions
    positions = interface.get_motor_positions()
    print(f"Current motor positions: {positions}")
    
    # Reset to open position
    interface.reset()
    
    # Clean up
    interface.close()
    
    print("Test complete!")


if __name__ == "__main__":
    test_ruka_interface()
