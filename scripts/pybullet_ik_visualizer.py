#!/usr/bin/env python3
"""
PyBullet IK Visualizer - Interactive testing tool for IK solvers.

This script provides real-time visualization of IK solving using PyBullet.
It supports any URDF with configurable base and end-effector links.

Features:
- Interactive target point control via keyboard/sliders
- Real-time IK solving using JAX-based solvers
- Visual feedback with target/end-effector markers
- Support for single and dual-arm configurations
- Performance metrics display

Usage:
    # H1 Inspire (default)
    python scripts/pybullet_ik_visualizer.py

    # Custom URDF
    python scripts/pybullet_ik_visualizer.py --urdf path/to/robot.urdf \
        --base-link torso --ee-link wrist

    # Both arms
    python scripts/pybullet_ik_visualizer.py --dual-arm

Controls:
    WASD    - Move target in XY plane
    Q/E     - Move target up/down (Z axis)
    Tab     - Switch arm (dual-arm mode)
    R       - Reset to home position
    Space   - Toggle continuous solve mode
    Esc     - Exit
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("ERROR: PyBullet not installed. Run: pip install pybullet")
    sys.exit(1)

import jax.numpy as jnp

from teleop.ik import RobotModel, SE3Pose
from teleop.ik.smooth_solver import SmoothIKSolver


# ============================================================================
# Robot Configurations (Extensible)
# ============================================================================

@dataclass
class RobotConfig:
    """Configuration for a robot arm."""
    urdf_path: str
    base_link: str
    end_effector_link: str
    initial_target: Tuple[float, float, float] = (0.4, 0.3, 0.8)
    home_pose: Optional[List[float]] = None
    description: str = "Robot arm"


# Pre-configured robots (add your own here!)
ROBOT_PRESETS: Dict[str, RobotConfig] = {
    "h1_left": RobotConfig(
        urdf_path=str(REPO_ROOT / "assets/h1_inspire/urdf/h1_inspire_fixed_torso.urdf"),
        base_link="torso_link",
        end_effector_link="left_wrist_yaw_link",
        initial_target=(0.35, 0.4, 0.6),
        description="H1 Inspire Left Arm (7 DOF)"
    ),
    "h1_right": RobotConfig(
        urdf_path=str(REPO_ROOT / "assets/h1_inspire/urdf/h1_inspire_fixed_torso.urdf"),
        base_link="torso_link",
        end_effector_link="right_wrist_yaw_link",
        initial_target=(0.35, -0.4, 0.6),
        description="H1 Inspire Right Arm (7 DOF)"
    ),
}


# ============================================================================
# PyBullet Visualization Class
# ============================================================================

class IKVisualizer:
    """
    Interactive IK visualization using PyBullet.
    
    This class provides a real-time visualization environment for testing
    and debugging IK solvers with any URDF-based robot.
    """
    
    def __init__(
        self,
        config: RobotConfig,
        solver_type: str = "smooth",
        show_gui: bool = True
    ):
        """
        Initialize the visualizer.
        
        Args:
            config: Robot configuration specifying URDF and link names.
            solver_type: Type of IK solver ("smooth" or "basic").
            show_gui: Whether to show the PyBullet GUI.
        """
        self.config = config
        self.target_pos = np.array(config.initial_target)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self.continuous_solve = True
        self.move_speed = 0.01  # meters per key press
        
        # Initialize PyBullet
        self._init_pybullet(show_gui)
        
        # Load robot
        self._load_robot()
        
        # Initialize IK solver
        self._init_solver(solver_type)
        
        # Create visual markers
        self._create_markers()
        
        # Create GUI sliders
        self._create_sliders()
        
        # Performance tracking
        self.solve_times: List[float] = []
        self.last_solve_error = 0.0
    
    def _init_pybullet(self, show_gui: bool):
        """Initialize PyBullet physics simulation."""
        if show_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # No gravity for manipulation testing
        
        # Add ground plane for reference
        p.loadURDF("plane.urdf")
        
        # Set camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )
    
    def _load_robot(self):
        """Load the robot URDF into PyBullet."""
        urdf_path = Path(self.config.urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        # Load with fixed base (upper torso testing)
        self.robot_id = p.loadURDF(
            str(urdf_path),
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # Build joint info mapping
        # PyBullet getJointInfo returns:
        # (jointIndex, jointName, jointType, qIndex, uIndex, flags, jointDamping,
        #  jointFriction, jointLowerLimit, jointUpperLimit, jointMaxForce, 
        #  jointMaxVelocity, linkName, jointAxis, parentFramePos, parentFrameOrn, parentIndex)
        self.joint_info: Dict[str, dict] = {}
        self.joint_name_to_idx: Dict[str, int] = {}
        self.controllable_joints: List[int] = []
        
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            link_name = info[12].decode('utf-8')  # Child link name
            
            self.joint_info[joint_name] = {
                'index': i,
                'type': joint_type,
                'lower': info[8],
                'upper': info[9],
                'link_name': link_name,
                'parent_index': info[16]  # Parent link index (int)
            }
            self.joint_name_to_idx[joint_name] = i
            
            if joint_type != p.JOINT_FIXED:
                self.controllable_joints.append(i)
        
        print(f"Loaded robot with {len(self.controllable_joints)} controllable joints")
    
    def _init_solver(self, solver_type: str):
        """Initialize the JAX-based IK solver."""
        print(f"Initializing IK solver for chain: {self.config.base_link} -> {self.config.end_effector_link}")
        
        self.robot_model = RobotModel(
            self.config.urdf_path,
            base_link=self.config.base_link,
            end_effector_link=self.config.end_effector_link
        )
        
        print(f"IK chain has {self.robot_model.num_joints} joints: {self.robot_model.joint_names}")
        
        # Map IK solver joint names to PyBullet indices
        # Joint names in RobotModel are just the joint names, we need to find them
        self.ik_to_pybullet: List[int] = []
        for joint_name in self.robot_model.joint_names:
            if joint_name in self.joint_name_to_idx:
                self.ik_to_pybullet.append(self.joint_name_to_idx[joint_name])
            else:
                # Try adding "_joint" suffix
                joint_name_with_suffix = joint_name + "_joint"
                if joint_name_with_suffix in self.joint_name_to_idx:
                    self.ik_to_pybullet.append(self.joint_name_to_idx[joint_name_with_suffix])
                else:
                    print(f"WARNING: Could not find PyBullet joint for IK joint '{joint_name}'")
                    self.ik_to_pybullet.append(-1)
        
        # Create solver
        self.solver = SmoothIKSolver(
            self.robot_model,
            position_weight=50.0,
            orientation_weight=1.0,
            regularization_weight=0.02,
            smoothness_weight=0.1,
            max_iterations=30,
            tolerance=1e-3,
            enable_filter=True
        )
        
        print("IK solver initialized successfully")
    
    def _create_markers(self):
        """Create visual markers for target and end-effector."""
        # Target marker (red sphere)
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[1, 0, 0, 0.8]
        )
        self.target_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_pos
        )
        
        # End-effector marker (green sphere)
        ee_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.025,
            rgbaColor=[0, 1, 0, 0.8]
        )
        self.ee_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=ee_visual,
            basePosition=[0, 0, 0]
        )
        
        # Coordinate frame at target (optional, helps with orientation)
        self._draw_frame(self.target_pos, size=0.1)
    
    def _draw_frame(self, pos: np.ndarray, size: float = 0.1):
        """Draw a coordinate frame at position."""
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB for XYZ
        for i, color in enumerate(colors):
            end = pos.copy()
            end[i] += size
            p.addUserDebugLine(pos, end, color, lineWidth=2, lifeTime=0)
    
    def _create_sliders(self):
        """Create GUI sliders for target position."""
        self.sliders = {}
        self.sliders['x'] = p.addUserDebugParameter("Target X", -1.0, 1.0, self.target_pos[0])
        self.sliders['y'] = p.addUserDebugParameter("Target Y", -1.0, 1.0, self.target_pos[1])
        self.sliders['z'] = p.addUserDebugParameter("Target Z", 0.0, 1.5, self.target_pos[2])
    
    def update_target_from_sliders(self):
        """Update target position from GUI sliders."""
        try:
            self.target_pos[0] = p.readUserDebugParameter(self.sliders['x'])
            self.target_pos[1] = p.readUserDebugParameter(self.sliders['y'])
            self.target_pos[2] = p.readUserDebugParameter(self.sliders['z'])
            p.resetBasePositionAndOrientation(self.target_marker, self.target_pos, [0, 0, 0, 1])
        except p.error:
            # Window was closed
            pass
    
    def handle_keyboard(self) -> bool:
        """
        Handle keyboard input.
        
        Returns:
            False if user pressed escape (exit), True otherwise.
        """
        keys = p.getKeyboardEvents()
        
        for key, state in keys.items():
            if state & p.KEY_WAS_TRIGGERED:
                if key == ord('w'):
                    self.target_pos[0] += self.move_speed
                elif key == ord('s'):
                    self.target_pos[0] -= self.move_speed
                elif key == ord('a'):
                    self.target_pos[1] += self.move_speed
                elif key == ord('d'):
                    self.target_pos[1] -= self.move_speed
                elif key == ord('q'):
                    self.target_pos[2] += self.move_speed
                elif key == ord('e'):
                    self.target_pos[2] -= self.move_speed
                elif key == ord('r'):
                    self.reset()
                elif key == ord(' '):
                    self.continuous_solve = not self.continuous_solve
                    print(f"Continuous solve: {self.continuous_solve}")
                elif key == 27:  # ESC key
                    return False
        
        # Update target marker position
        p.resetBasePositionAndOrientation(self.target_marker, self.target_pos, [0, 0, 0, 1])
        return True
    
    def solve_and_apply(self):
        """Run IK solver and apply solution to robot."""
        # Create target pose
        target = SE3Pose(
            position=jnp.array(self.target_pos),
            quaternion=jnp.array(self.target_quat)
        )
        
        # Solve IK
        start_time = time.perf_counter()
        result = self.solver.solve(target, compute_error=True)
        solve_time = (time.perf_counter() - start_time) * 1000
        
        self.solve_times.append(solve_time)
        if len(self.solve_times) > 100:
            self.solve_times.pop(0)
        
        self.last_solve_error = result.position_error
        
        # Apply to PyBullet
        joint_angles = np.array(result.joint_angles)
        for i, pb_idx in enumerate(self.ik_to_pybullet):
            if pb_idx >= 0 and i < len(joint_angles):
                p.setJointMotorControl2(
                    self.robot_id,
                    pb_idx,
                    p.POSITION_CONTROL,
                    targetPosition=float(joint_angles[i]),
                    force=100
                )
        
        # Update end-effector marker using FK
        ee_pose = self.robot_model.forward_kinematics(jnp.array(joint_angles))
        p.resetBasePositionAndOrientation(
            self.ee_marker,
            np.array(ee_pose.position),
            [0, 0, 0, 1]
        )
        
        return result
    
    def reset(self):
        """Reset robot to home position and target to initial."""
        self.target_pos = np.array(self.config.initial_target)
        p.resetBasePositionAndOrientation(self.target_marker, self.target_pos, [0, 0, 0, 1])
        
        # Reset solver state
        self.solver.reset()
        
        # Reset joint positions
        for joint_idx in self.controllable_joints:
            p.resetJointState(self.robot_id, joint_idx, 0)
        
        print("Reset to home position")
    
    def print_status(self):
        """Print current status to console."""
        avg_time = np.mean(self.solve_times) if self.solve_times else 0
        hz = 1000 / avg_time if avg_time > 0 else 0
        
        status = (
            f"\rTarget: [{self.target_pos[0]:.3f}, {self.target_pos[1]:.3f}, {self.target_pos[2]:.3f}] | "
            f"Error: {self.last_solve_error*100:.2f}cm | "
            f"Solve: {avg_time:.2f}ms ({hz:.0f}Hz) | "
            f"{'[SOLVING]' if self.continuous_solve else '[PAUSED]'}"
        )
        print(status, end='', flush=True)
    
    def run(self):
        """Main visualization loop."""
        print("\n" + "=" * 60)
        print("  PyBullet IK Visualizer")
        print("=" * 60)
        print(f"Robot: {self.config.description}")
        print(f"Chain: {self.config.base_link} -> {self.config.end_effector_link}")
        print(f"Joints: {self.robot_model.num_joints}")
        print("\nControls:")
        print("  WASD  - Move target in XY")
        print("  Q/E   - Move target up/down")
        print("  R     - Reset position")
        print("  Space - Toggle solving")
        print("  Esc   - Exit")
        print("=" * 60 + "\n")
        
        try:
            while p.isConnected():
                # Handle input
                if not self.handle_keyboard():
                    break
                
                # Update from sliders
                self.update_target_from_sliders()
                
                # Solve IK
                if self.continuous_solve:
                    self.solve_and_apply()
                
                # Step simulation
                p.stepSimulation()
                
                # Print status
                self.print_status()
                
                # Rate limit
                time.sleep(1/240)
                
        except (KeyboardInterrupt, p.error):
            pass
        finally:
            print("\n\nExiting...")
            if p.isConnected():
                p.disconnect()
            print("Visualization ended.")


# ============================================================================
# Dual-Arm Visualizer
# ============================================================================

class DualArmIKVisualizer:
    """Dual-arm IK visualization with shared PyBullet instance."""
    
    def __init__(self, left_config: RobotConfig, right_config: RobotConfig):
        """Initialize dual-arm visualizer."""
        self.left_config = left_config
        self.right_config = right_config
        self.active_arm = "left"
        
        # State (initialize before markers)
        self.targets = {
            "left": np.array(left_config.initial_target),
            "right": np.array(right_config.initial_target)
        }
        self.move_speed = 0.01
        self.continuous_solve = True
        self.solve_times: List[float] = []
        
        # Initialize PyBullet once
        self.physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.loadURDF("plane.urdf")
        
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )
        
        # Load robot (shared)
        self._load_robot()
        
        # Initialize both solvers
        self._init_solvers()
        
        # Create markers for both arms
        self._create_markers()
    
    def _load_robot(self):
        """Load robot URDF."""
        urdf_path = Path(self.left_config.urdf_path)
        self.robot_id = p.loadURDF(
            str(urdf_path),
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Build joint mapping
        self.joint_name_to_idx: Dict[str, int] = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode('utf-8')
            self.joint_name_to_idx[name] = i
    
    def _init_solvers(self):
        """Initialize IK solvers for both arms."""
        self.solvers = {}
        self.robot_models = {}
        self.ik_to_pybullet = {}
        
        for arm, config in [("left", self.left_config), ("right", self.right_config)]:
            model = RobotModel(
                config.urdf_path,
                base_link=config.base_link,
                end_effector_link=config.end_effector_link
            )
            self.robot_models[arm] = model
            self.solvers[arm] = SmoothIKSolver(model, enable_filter=True)
            
            # Build joint mapping
            self.ik_to_pybullet[arm] = []
            for jn in model.joint_names:
                if jn in self.joint_name_to_idx:
                    self.ik_to_pybullet[arm].append(self.joint_name_to_idx[jn])
                else:
                    self.ik_to_pybullet[arm].append(-1)
            
            print(f"{arm.capitalize()} arm: {model.num_joints} joints")
    
    def _create_markers(self):
        """Create markers for both arms."""
        self.target_markers = {}
        self.ee_markers = {}
        
        colors = {"left": [1, 0, 0, 0.8], "right": [0, 0, 1, 0.8]}  # Red left, Blue right
        
        for arm in ["left", "right"]:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=colors[arm])
            self.target_markers[arm] = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis, basePosition=self.targets[arm]
            )
            
            ee_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.8])
            self.ee_markers[arm] = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=ee_vis, basePosition=[0, 0, 0]
            )
        
        # Create sliders for both arms
        self.sliders = {}
        for arm in ["left", "right"]:
            prefix = "L" if arm == "left" else "R"
            self.sliders[arm] = {
                'x': p.addUserDebugParameter(f"{prefix} Target X", -1.0, 1.0, self.targets[arm][0]),
                'y': p.addUserDebugParameter(f"{prefix} Target Y", -1.0, 1.0, self.targets[arm][1]),
                'z': p.addUserDebugParameter(f"{prefix} Target Z", 0.0, 1.5, self.targets[arm][2])
            }
    
    def update_targets_from_sliders(self):
        """Update target positions from GUI sliders."""
        try:
            for arm in ["left", "right"]:
                self.targets[arm][0] = p.readUserDebugParameter(self.sliders[arm]['x'])
                self.targets[arm][1] = p.readUserDebugParameter(self.sliders[arm]['y'])
                self.targets[arm][2] = p.readUserDebugParameter(self.sliders[arm]['z'])
                p.resetBasePositionAndOrientation(
                    self.target_markers[arm], self.targets[arm], [0, 0, 0, 1]
                )
        except p.error:
            pass
    
    def handle_keyboard(self) -> bool:
        """Handle keyboard input."""
        keys = p.getKeyboardEvents()
        
        for key, state in keys.items():
            if state & p.KEY_WAS_TRIGGERED:
                target = self.targets[self.active_arm]
                
                if key == ord('w'):
                    target[0] += self.move_speed
                elif key == ord('s'):
                    target[0] -= self.move_speed
                elif key == ord('a'):
                    target[1] += self.move_speed
                elif key == ord('d'):
                    target[1] -= self.move_speed
                elif key == ord('q'):
                    target[2] += self.move_speed
                elif key == ord('e'):
                    target[2] -= self.move_speed
                elif key == ord('\t'):
                    self.active_arm = "right" if self.active_arm == "left" else "left"
                    print(f"\nActive arm: {self.active_arm.upper()}")
                elif key == ord('r'):
                    self.reset()
                elif key == ord(' '):
                    self.continuous_solve = not self.continuous_solve
                elif key == 27:  # ESC key
                    return False
        
        # Update markers
        for arm in ["left", "right"]:
            p.resetBasePositionAndOrientation(
                self.target_markers[arm], self.targets[arm], [0, 0, 0, 1]
            )
        
        return True
    
    def solve_and_apply(self):
        """Solve IK for both arms and apply."""
        start = time.perf_counter()
        
        for arm in ["left", "right"]:
            target = SE3Pose(
                position=jnp.array(self.targets[arm]),
                quaternion=jnp.array([1.0, 0.0, 0.0, 0.0])
            )
            
            result = self.solvers[arm].solve(target)
            angles = np.array(result.joint_angles)
            
            for i, pb_idx in enumerate(self.ik_to_pybullet[arm]):
                if pb_idx >= 0 and i < len(angles):
                    p.setJointMotorControl2(
                        self.robot_id, pb_idx, p.POSITION_CONTROL,
                        targetPosition=float(angles[i]), force=100
                    )
            
            # Update EE marker
            ee_pose = self.robot_models[arm].forward_kinematics(jnp.array(angles))
            p.resetBasePositionAndOrientation(
                self.ee_markers[arm], np.array(ee_pose.position), [0, 0, 0, 1]
            )
        
        solve_time = (time.perf_counter() - start) * 1000
        self.solve_times.append(solve_time)
        if len(self.solve_times) > 100:
            self.solve_times.pop(0)
    
    def reset(self):
        """Reset both arms."""
        self.targets["left"] = np.array(self.left_config.initial_target)
        self.targets["right"] = np.array(self.right_config.initial_target)
        self.solvers["left"].reset()
        self.solvers["right"].reset()
        print("\nReset to home positions")
    
    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("  Dual-Arm IK Visualizer")
        print("=" * 60)
        print("Controls: WASD/QE move, Tab switch arm, R reset, Esc exit")
        print("Use sliders to adjust L/R targets independently")
        print(f"Active arm: {self.active_arm.upper()}")
        print("=" * 60 + "\n")
        
        try:
            while p.isConnected():
                if not self.handle_keyboard():
                    break
                
                # Update from sliders
                self.update_targets_from_sliders()
                
                if self.continuous_solve:
                    self.solve_and_apply()
                
                p.stepSimulation()
                
                avg = np.mean(self.solve_times) if self.solve_times else 0
                print(f"\rActive: {self.active_arm.upper()} | Solve: {avg:.1f}ms", end='', flush=True)
                
                time.sleep(1/240)
                
        except (KeyboardInterrupt, p.error):
            pass
        finally:
            print("\n\nExiting...")
            if p.isConnected():
                p.disconnect()
            print("Done.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive PyBullet IK Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--preset", "-p",
        choices=list(ROBOT_PRESETS.keys()),
        default="h1_left",
        help="Use a pre-configured robot (default: h1_left)"
    )
    
    parser.add_argument(
        "--urdf",
        type=str,
        help="Path to custom URDF file (overrides preset)"
    )
    
    parser.add_argument(
        "--base-link",
        type=str,
        default="torso_link",
        help="Base link name for IK chain"
    )
    
    parser.add_argument(
        "--ee-link",
        type=str,
        help="End-effector link name for IK chain"
    )
    
    parser.add_argument(
        "--dual-arm",
        action="store_true",
        help="Enable dual-arm mode (H1 only)"
    )
    
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available robot presets and exit"
    )
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        print("\nAvailable Robot Presets:")
        print("-" * 40)
        for name, config in ROBOT_PRESETS.items():
            print(f"  {name}: {config.description}")
        print()
        return
    
    # Dual-arm mode
    if args.dual_arm:
        viz = DualArmIKVisualizer(
            ROBOT_PRESETS["h1_left"],
            ROBOT_PRESETS["h1_right"]
        )
        viz.run()
        return
    
    # Single arm mode
    if args.urdf:
        # Custom URDF
        if not args.ee_link:
            print("ERROR: --ee-link required when using custom URDF")
            sys.exit(1)
        
        config = RobotConfig(
            urdf_path=args.urdf,
            base_link=args.base_link,
            end_effector_link=args.ee_link,
            description="Custom Robot"
        )
    else:
        # Use preset
        config = ROBOT_PRESETS[args.preset]
    
    # Run visualizer
    viz = IKVisualizer(config)
    viz.run()


if __name__ == "__main__":
    main()
