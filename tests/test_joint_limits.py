#!/usr/bin/env python3
"""
Test suite for joint limit parsing in RobotModel.

This specifically tests the fix for the bug where:
  `joint.limit.lower or -jnp.pi`
fails when the URDF specifies a limit of 0.0 (falsy in Python).

The fix uses explicit `is None` checks instead of the `or` operator.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
from teleop.ik.robot import RobotModel


def create_test_urdf(lower: float, upper: float) -> str:
    """Create a minimal URDF with specified joint limits."""
    return f"""<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="{lower}" upper="{upper}" effort="10" velocity="1"/>
  </joint>
</robot>
"""


def create_continuous_urdf() -> str:
    """Create a URDF with a continuous (no-limit) joint."""
    return """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  
  <joint name="joint1" type="continuous">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""


class TestJointLimitParsing(unittest.TestCase):
    """Test cases for joint limit parsing in RobotModel."""
    
    def test_zero_lower_limit(self):
        """
        CRITICAL TEST: Lower limit of 0.0 should be parsed as 0.0, not -pi.
        
        This was the original bug: `0.0 or -jnp.pi` returns -jnp.pi
        because 0.0 is falsy in Python.
        """
        urdf_content = create_test_urdf(lower=0.0, upper=1.5)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name
        
        try:
            robot = RobotModel(urdf_path, base_link="base_link", end_effector_link="link1")
            lower, upper = robot.joint_limits
            
            # The lower limit should be exactly 0.0, not -pi
            self.assertAlmostEqual(float(lower[0]), 0.0, places=5,
                msg=f"Lower limit should be 0.0, got {float(lower[0])}")
            self.assertAlmostEqual(float(upper[0]), 1.5, places=5)
        finally:
            Path(urdf_path).unlink()
    
    def test_zero_upper_limit(self):
        """
        Upper limit of 0.0 should be parsed as 0.0, not +pi.
        """
        urdf_content = create_test_urdf(lower=-1.5, upper=0.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name
        
        try:
            robot = RobotModel(urdf_path, base_link="base_link", end_effector_link="link1")
            lower, upper = robot.joint_limits
            
            self.assertAlmostEqual(float(lower[0]), -1.5, places=5)
            # The upper limit should be exactly 0.0, not +pi
            self.assertAlmostEqual(float(upper[0]), 0.0, places=5,
                msg=f"Upper limit should be 0.0, got {float(upper[0])}")
        finally:
            Path(urdf_path).unlink()
    
    def test_both_limits_zero(self):
        """
        Both limits set to 0.0 (a locked joint).
        """
        urdf_content = create_test_urdf(lower=0.0, upper=0.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name
        
        try:
            robot = RobotModel(urdf_path, base_link="base_link", end_effector_link="link1")
            lower, upper = robot.joint_limits
            
            self.assertAlmostEqual(float(lower[0]), 0.0, places=5)
            self.assertAlmostEqual(float(upper[0]), 0.0, places=5)
        finally:
            Path(urdf_path).unlink()
    
    def test_negative_limits(self):
        """
        Negative limits should be parsed correctly.
        """
        urdf_content = create_test_urdf(lower=-2.0, upper=-0.5)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name
        
        try:
            robot = RobotModel(urdf_path, base_link="base_link", end_effector_link="link1")
            lower, upper = robot.joint_limits
            
            self.assertAlmostEqual(float(lower[0]), -2.0, places=5)
            self.assertAlmostEqual(float(upper[0]), -0.5, places=5)
        finally:
            Path(urdf_path).unlink()
    
    def test_positive_limits(self):
        """
        Normal positive limits should work as before.
        """
        urdf_content = create_test_urdf(lower=0.5, upper=2.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name
        
        try:
            robot = RobotModel(urdf_path, base_link="base_link", end_effector_link="link1")
            lower, upper = robot.joint_limits
            
            self.assertAlmostEqual(float(lower[0]), 0.5, places=5)
            self.assertAlmostEqual(float(upper[0]), 2.0, places=5)
        finally:
            Path(urdf_path).unlink()
    
    def test_continuous_joint_defaults_to_pi(self):
        """
        Continuous joints (no limit tag) should default to ±pi.
        """
        urdf_content = create_continuous_urdf()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name
        
        try:
            robot = RobotModel(urdf_path, base_link="base_link", end_effector_link="link1")
            lower, upper = robot.joint_limits
            
            # Continuous joints should get ±pi as defaults
            self.assertAlmostEqual(float(lower[0]), -jnp.pi, places=5)
            self.assertAlmostEqual(float(upper[0]), jnp.pi, places=5)
        finally:
            Path(urdf_path).unlink()


class TestJointLimitClipping(unittest.TestCase):
    """Test that IK solver respects the parsed limits."""
    
    def test_ik_respects_zero_limit(self):
        """
        IK solver should clip solutions to the actual limits, including 0.0.
        """
        from teleop.ik import IKSolver, SE3Pose
        
        # Create a robot with a 0.0 lower limit
        urdf_content = create_test_urdf(lower=0.0, upper=1.5)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name
        
        try:
            robot = RobotModel(urdf_path, base_link="base_link", end_effector_link="link1")
            solver = IKSolver(robot)
            
            # Get a target pose
            target = robot.forward_kinematics(jnp.array([0.5]))
            
            # Solve IK
            result = solver.solve(target)
            
            # The solution should be clipped to [0.0, 1.5]
            self.assertGreaterEqual(float(result.joint_angles[0]), 0.0,
                msg="Solution should not go below lower limit of 0.0")
            self.assertLessEqual(float(result.joint_angles[0]), 1.5,
                msg="Solution should not exceed upper limit of 1.5")
        finally:
            Path(urdf_path).unlink()


if __name__ == "__main__":
    print("=" * 70)
    print("  Joint Limit Parsing Test Suite")
    print("=" * 70)
    print("\nThis tests the fix for the bug where `or -jnp.pi` fails")
    print("when a URDF specifies a limit of 0.0.\n")
    
    unittest.main(verbosity=2)
