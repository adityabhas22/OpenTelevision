# TeleOp System Design: Custom Upper Torso Robot

> **Purpose:** This document consolidates research on implementing VR teleoperation for a custom upper-torso robot with 5DOF arms and RUKA hands. It serves as a complete reference for developers and AI agents working on this project.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Existing Codebase Analysis](#2-existing-codebase-analysis)
3. [Inverse Kinematics (IK) Deep Dive](#3-inverse-kinematics-ik-deep-dive)
4. [Hand Retargeting vs Arm IK](#4-hand-retargeting-vs-arm-ik)
5. [JAX for Real-Time IK](#5-jax-for-real-time-ik)
6. [Hardware Integration Architecture](#6-hardware-integration-architecture)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Code Reference Map](#8-code-reference-map)
9. [External Resources](#9-external-resources)

---

## 1. System Overview

### What We're Building

A VR teleoperation system that allows a human operator wearing a Meta Quest or Apple Vision Pro to control:
- **5DOF Arms** (position control via Inverse Kinematics)
- **RUKA Dexterous Hands** (gesture control via Retargeting)

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VR HEADSET                                     │
│  (Quest 3 / Apple Vision Pro)                                               │
│                                                                             │
│  Outputs via WebXR:                                                         │
│  • Head Pose (4x4 matrix)                                                   │
│  • Left/Right Wrist Pose (4x4 matrix each)                                  │
│  • Left/Right Hand Landmarks (25 joints × 3D coords = 75 floats each)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ WebSocket (wss://)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HOST COMPUTER                                     │
│  (Linux PC / Jetson / Mac)                                                  │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                       │
│  │  OpenTeleVision      │───▶│  Preprocessor        │                       │
│  │  (Vuer WebXR Server) │    │  (Coord Transform)   │                       │
│  │                      │    │  Y-up → Z-up         │                       │
│  │  - Receives hand     │    │  World → Wrist-rel   │                       │
│  │    tracking data     │    │                      │                       │
│  │  - Shared memory     │    │                      │                       │
│  └──────────────────────┘    └──────────┬───────────┘                       │
│                                         │                                   │
│                    ┌────────────────────┴────────────────────┐              │
│                    ▼                                         ▼              │
│  ┌──────────────────────────────┐      ┌──────────────────────────────┐     │
│  │  ARM IK SOLVER               │      │  HAND RETARGETING            │     │
│  │  (JAX-based)                 │      │  (dex-retargeting / RUKA)    │     │
│  │                              │      │                              │     │
│  │  Input: Wrist Pose (SE3)     │      │  Input: 25 landmarks         │     │
│  │  Output: 5 joint angles      │      │  Output: Finger joint angles │     │
│  │                              │      │  (via LSTM controller)       │     │
│  │  ⚠️ REQUIRES URDF            │      │  ✅ Already implemented      │     │
│  └──────────────────────────────┘      └──────────────────────────────┘     │
│                    │                                         │              │
│                    └────────────────────┬────────────────────┘              │
│                                         ▼                                   │
│                          ┌──────────────────────────────┐                   │
│                          │  UDP Command Sender          │                   │
│                          │  (struct-packed binary)      │                   │
│                          │  60-100 Hz                   │                   │
│                          └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ UDP over Ethernet
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROBOT HARDWARE                                    │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                       │
│  │  Microcontroller     │───▶│  Motor Drivers       │                       │
│  │  (Teensy 4.1/ESP32)  │    │  (CAN Bus / RS485)   │                       │
│  │                      │    │                      │                       │
│  │  - UDP receiver      │    │  - 5DOF arm motors   │                       │
│  │  - Safety watchdog   │    │  - RUKA Dynamixels   │                       │
│  │  - 1kHz control loop │    │                      │                       │
│  └──────────────────────┘    └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Critical Blocker: URDF

> **⚠️ The URDF (Unified Robot Description Format) file for the 5DOF arms is the critical missing piece.**

Without it:
- Cannot compute Forward Kinematics (FK)
- Cannot solve Inverse Kinematics (IK)
- Cannot simulate before deploying
- Cannot use standard robotics tools

---

## 2. Existing Codebase Analysis

### Repository Structure

```
/home/aditya/TeleVision/
├── teleop/                         # OpenTeleVision core
│   ├── TeleVision.py               # WebXR server (Vuer-based)
│   ├── Preprocessor.py             # Coordinate transforms
│   ├── constants_vuer.py           # Transform matrices
│   ├── teleop_hand.py              # Isaac Gym simulation example
│   ├── inspire_hand.yml            # dex-retargeting config
│   ├── ruka_interface.py           # RUKA motor driver
│   └── teleop_ruka.py              # RUKA teleop entry point
│
├── RUKA/                           # RUKA Hand submodule
│   └── ruka_hand/
│       ├── control/
│       │   ├── controller.py       # HandController (LSTM)
│       │   └── operator.py         # RUKAOperator
│       ├── teleoperation/
│       │   └── avp_teleoperator.py # AVP→RUKA integration ✅
│       └── utils/
│           └── vectorops.py        # Joint angle calculation
│
└── assets/
    ├── h1_inspire/                 # Unitree H1 URDF (reference)
    └── inspire_hand/               # Inspire Hand URDF
```

### Key Files Explained

#### `teleop/TeleVision.py` - The WebXR Server

```python
class OpenTeleVision:
    """
    Runs a Vuer-based WebXR server that connects to VR headsets.
    
    Properties available after connection:
    - left_hand / right_hand: 4x4 wrist transformation matrices
    - left_landmarks / right_landmarks: (25, 3) finger joint positions
    - head_matrix: 4x4 head pose matrix
    """
    
    # Communication via shared memory for zero-copy performance
    self.left_hand_shared = Array('d', 16, lock=True)  # 4x4 matrix
    self.right_hand_shared = Array('d', 16, lock=True)
    self.left_landmarks_shared = Array('d', 75, lock=True)  # 25 joints × 3
    self.right_landmarks_shared = Array('d', 75, lock=True)
```

#### `teleop/Preprocessor.py` - Coordinate Transforms

```python
# WebXR uses Y-up coordinate system
# Robots typically use Z-up
# This matrix converts between them:

grd_yup2grd_zup = np.array([
    [0, 0, -1, 0],
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

class VuerPreprocessor:
    def process(self, tv):
        # Returns:
        # - head_mat: Head pose in robot frame
        # - rel_left/right_wrist_mat: Wrist poses relative to head
        # - rel_left/right_fingers: Finger landmarks relative to wrist
```

#### `RUKA/ruka_hand/teleoperation/avp_teleoperator.py` - Hand Teleop

This file is the **existing working implementation** for RUKA hand control via AVP:

```python
class AVPTeleoperator:
    """
    Apple Vision Pro Teleoperator for RUKA hand.
    
    Data Flow:
        AVP (Safari WebXR) 
            → OpenTeleVision Server (shared memory)
            → VuerPreprocessor (coordinate transform)
            → AVPTeleoperator (keypoint mapping)
            → RUKAOperator (LSTM controllers)
            → Dynamixel Motors
    """
    
    # WebXR joint indices for RUKA's expected format
    HAND_JOINTS = {
        'thumb':  [0, 1, 2, 3, 4],       # wrist, metacarpal, proximal, distal, tip
        'index':  [0, 6, 7, 8, 9],       
        'middle': [0, 11, 12, 13, 14],   
        'ring':   [0, 16, 17, 18, 19],   
        'pinky':  [0, 21, 22, 23, 24],   
    }
```

---

## 3. Inverse Kinematics (IK) Deep Dive

### What is IK?

Given a desired end-effector pose (position + orientation), find joint angles that achieve it.

```
Forward Kinematics:  Joint Angles → End-Effector Pose  (Easy, deterministic)
Inverse Kinematics:  End-Effector Pose → Joint Angles  (Hard, may have 0, 1, or many solutions)
```

### Mathematical Formulation

IK is typically solved as an optimization problem:

```
minimize:    ||FK(θ) - Target_pose||²
subject to:  θ_min ≤ θ ≤ θ_max  (joint limits)
```

Where:
- `θ` = vector of joint angles
- `FK(θ)` = forward kinematics function (defined by URDF)
- `Target_pose` = desired SE(3) pose from VR wrist tracking

### IK Solver Options

| Solver | Speed | Ease of Use | GPU Support | Notes |
|--------|-------|-------------|-------------|-------|
| **SciPy (scipy.optimize)** | ~10-50 Hz | Easy | No | Good for prototyping |
| **JAX + JAXOpt** | ~1000+ Hz | Medium | Yes | KScale's choice |
| **PyBullet IK** | ~100 Hz | Easy | No | Requires physics sim |
| **MoveIt! (ROS)** | ~100 Hz | Complex | No | Industry standard |
| **IKFast (OpenRAVE)** | ~10000 Hz | Hard | No | Analytical, robot-specific |

### Why JAX for IK?

JAX provides **automatic differentiation**, which means:

1. You write the forward kinematics function once
2. JAX automatically computes gradients (∂FK/∂θ)
3. Gradient-based optimization becomes trivial and fast

**Without auto-diff (SciPy):**
```python
# Finite differences: evaluate FK 2N times for N joints
gradient[i] = (fk(θ + ε*eᵢ) - fk(θ - ε*eᵢ)) / (2ε)
```

**With auto-diff (JAX):**
```python
# Single backward pass through computation graph
gradient = jax.grad(loss_fn)(θ)
```

---

## 4. Hand Retargeting vs Arm IK

### Why They're Different

| Aspect | Arm (IK) | Hand (Retargeting) |
|--------|----------|-------------------|
| **Goal** | Exact pose matching | Gesture/shape preservation |
| **DOF** | 5-7 joints | 15-20+ joints |
| **Constraint** | End-effector in SE(3) | Finger curl ratios |
| **Problem** | "Place wrist HERE" | "Make fingers look LIKE MINE" |

### The Embodiment Gap Problem

Human hands and robot hands have different:
- Link lengths
- Joint ranges
- Thumb attachment points

If you tried exact position matching, the robot would break itself trying to reach impossible positions.

### Retargeting Solution

Instead of absolute positions, match **relative vectors**:

```
Human: Vector from palm to index-tip = v1
Robot: Find joint angles where robot's palm-to-index-tip ≈ k*v1

(k = scaling factor for different hand sizes)
```

### dex-retargeting Library

Used in `teleop_hand.py`:

```python
# Config file (inspire_hand.yml) specifies:
target_link_human_indices: [[0,0,0,0,0], [4,9,14,19,24]]  # wrist + 5 tips
target_task_link_names: ["R_thumb_tip", "R_index_tip", ...]
scaling_factor: 1.1

# Usage:
retargeting = RetargetingConfig.from_dict(cfg).build()
robot_qpos = retargeting.retarget(human_fingertips)
```

### RUKA's Approach (Different)

RUKA doesn't use dex-retargeting. It uses **learned LSTM controllers**:

```python
# RUKAOperator.step() flow:
1. Receive keypoints (5 fingers × 5 joints × 3D = 75 values)
2. Calculate fingertip positions and joint angles from keypoints
3. Feed to trained LSTM model
4. LSTM outputs motor commands directly
```

---

## 5. JAX for Real-Time IK

### JAX Libraries for URDF Parsing

| Library | Purpose | URL |
|---------|---------|-----|
| **jax-ik** | URDF→JAX FK, differentiable IK | github.com/google-deepmind/jax-ik |
| **jaxsim** | Full physics engine with URDF | github.com/ami-iit/jaxsim |
| **kinjax** | Lightweight kinematics | github.com/liruilong940607/kinjax |

### Conceptual JAX IK Solver

```python
import jax
import jax.numpy as jnp
from jax import grad, jit

# 1. Parse URDF into JAX-compatible FK function
# (Library-specific, but conceptually:)
def forward_kinematics(joint_angles):
    """
    Computes end-effector pose from joint angles.
    
    Args:
        joint_angles: jnp.array of shape (5,) for 5DOF arm
    
    Returns:
        position: jnp.array of shape (3,) - XYZ
        quaternion: jnp.array of shape (4,) - WXYZ
    """
    # Chain multiplication of transformation matrices
    # Each joint contributes a rotation based on its angle
    T = jnp.eye(4)
    for i, (axis, angle) in enumerate(zip(joint_axes, joint_angles)):
        T = T @ rotation_matrix(axis, angle) @ translation_matrix(link_lengths[i])
    
    position = T[:3, 3]
    quaternion = rotation_matrix_to_quaternion(T[:3, :3])
    return position, quaternion

# 2. Define loss function
def ik_loss(joint_angles, target_pos, target_quat):
    current_pos, current_quat = forward_kinematics(joint_angles)
    
    pos_error = jnp.sum((current_pos - target_pos) ** 2)
    
    # Quaternion distance (1 - |q1·q2| handles double-cover)
    quat_error = 1.0 - jnp.abs(jnp.dot(current_quat, target_quat))
    
    return pos_error + 0.1 * quat_error

# 3. Compile gradient function
ik_loss_grad = jit(grad(ik_loss))

# 4. Solve IK (simple gradient descent)
def solve_ik(target_pos, target_quat, initial_guess, num_iters=50, lr=0.1):
    joints = initial_guess
    
    for _ in range(num_iters):
        grads = ik_loss_grad(joints, target_pos, target_quat)
        joints = joints - lr * grads
        joints = jnp.clip(joints, joint_limits_min, joint_limits_max)
    
    return joints
```

### Performance Expectations

| Setup | Expected Hz |
|-------|-------------|
| JAX CPU (M1 Mac) | ~500 Hz |
| JAX GPU (RTX 3080) | ~5000 Hz |
| SciPy CPU | ~50 Hz |

For teleoperation, you need **≥60 Hz** for smooth control. JAX easily achieves this.

---

## 6. Hardware Integration Architecture

### Recommended Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST COMPUTER                            │
│  OS: Ubuntu 22.04 / macOS                                   │
│  Hardware: PC with GPU or Jetson Orin                       │
│                                                             │
│  Software:                                                  │
│  - Python 3.10+                                             │
│  - JAX (with CUDA if GPU)                                   │
│  - OpenTeleVision (Vuer, aiortc)                            │
│  - RUKA libraries                                           │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ UDP (Ethernet preferred)
                         │ Port: 8888
                         │ Packet: struct { float joints[5]; uint8 gripper; }
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 BRIDGE MICROCONTROLLER                      │
│  Hardware: Teensy 4.1 (600MHz, native Ethernet)             │
│  Alternative: ESP32-S3 (WiFi, but adds latency)             │
│                                                             │
│  Responsibilities:                                          │
│  - Receive UDP packets                                      │
│  - Validate joint limits                                    │
│  - Safety watchdog (stop if no packet for 100ms)            │
│  - Send commands to motor drivers                           │
│  - Run 1kHz PID loops if needed                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ CAN Bus (preferred) or RS-485
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MOTOR DRIVERS                            │
│                                                             │
│  For 5DOF Arm (likely):                                     │
│  - Robstride motors with CAN interface                      │
│  - ODrive controllers                                       │
│  - Custom servo drivers                                     │
│                                                             │
│  For RUKA Hand:                                             │
│  - Dynamixel servos (already integrated)                    │
│  - Connected via USB/UART                                   │
└─────────────────────────────────────────────────────────────┘
```

### Communication Protocol

**Packet Format (Binary struct, not JSON for speed):**

```python
# Python sender
import struct
import socket

PACKET_FORMAT = '<5fB'  # Little-endian: 5 floats + 1 byte
# Total size: 5*4 + 1 = 21 bytes

def send_command(sock, joint_angles, gripper):
    packet = struct.pack(PACKET_FORMAT, *joint_angles, gripper)
    sock.sendto(packet, (ROBOT_IP, 8888))
```

```cpp
// C++ receiver (Teensy/ESP32)
struct __attribute__((packed)) CommandPacket {
    float joints[5];
    uint8_t gripper;
};

void onUdpPacket(AsyncUDPPacket packet) {
    if (packet.length() == sizeof(CommandPacket)) {
        CommandPacket* cmd = (CommandPacket*)packet.data();
        setJointTargets(cmd->joints);
        setGripper(cmd->gripper);
        lastPacketTime = millis();
    }
}

void loop() {
    // Safety: Stop if no command in 100ms
    if (millis() - lastPacketTime > 100) {
        emergencyStop();
    }
}
```

### Why UDP over TCP?

| Aspect | UDP | TCP |
|--------|-----|-----|
| **Latency** | ~1ms | ~10-50ms (retransmits) |
| **Packet loss** | Drops old packets | Blocks waiting for retransmit |
| **For teleop** | ✅ Perfect | ❌ Causes stuttering |

For real-time control, **an old command is useless**. UDP drops stale data, which is exactly what we want.

---

## 7. Implementation Roadmap

### Phase 1: URDF Creation (Blocked)

**Inputs needed:**
- CAD files or physical measurements of arm
- Joint types (revolute/prismatic)
- Joint limits (min/max angles)
- Link lengths and offsets

**Output:** `my_5dof_robot.urdf`

**Tips:**
- Start with "stick figure" URDF (no meshes)
- Verify with RViz or PyBullet before adding complexity
- KScale's K-Bot URDF can serve as reference

### Phase 2: IK Development

1. **Parse URDF** → FK function
2. **Implement IK solver** (start with SciPy, upgrade to JAX)
3. **Test in simulation** (Isaac Gym or MuJoCo)
4. **Verify tracking accuracy** (move hand, see robot follow)

### Phase 3: Hardware Integration

1. **Set up MCU bridge** (UDP receiver)
2. **Wire motor drivers** (CAN or PWM)
3. **Implement safety watchdog**
4. **Tune PID gains** for smooth motion

### Phase 4: Full System Integration

1. **Combine arm IK + hand retargeting**
2. **Add stereo video feedback** (GStreamer/WebRTC)
3. **Tune latency** (target <100ms glass-to-glass)
4. **User testing** and iteration

---

## 8. Code Reference Map

### Files You'll Modify/Create

| File | Purpose | Status |
|------|---------|--------|
| `teleop/arm_ik.py` | JAX-based IK solver | To create |
| `teleop/urdf_parser.py` | Load URDF into JAX | To create |
| `teleop/arm_teleop.py` | Arm teleop loop | To create |
| `teleop/full_teleop.py` | Combined arm+hand control | To create |
| `firmware/udp_bridge/` | MCU firmware | To create |
| `assets/my_robot.urdf` | Robot description | **BLOCKED** |

### Files You'll Use As-Is

| File | Purpose |
|------|---------|
| `teleop/TeleVision.py` | WebXR server (unchanged) |
| `teleop/Preprocessor.py` | Coordinate transforms (unchanged) |
| `RUKA/ruka_hand/teleoperation/avp_teleoperator.py` | Hand control (reference) |
| `RUKA/ruka_hand/control/operator.py` | Motor interface (reference) |

### Useful Patterns to Copy

**From `avp_teleoperator.py`:**
```python
# Coordinate transform (Y-up to Z-up)
GRD_YUP2GRD_ZUP = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])
transformed = (GRD_YUP2GRD_ZUP @ coords.T).T
```

**From `teleop_hand.py`:**
```python
# Main teleop loop structure
while True:
    # 1. Get VR data
    head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
    
    # 2. Send to simulation/robot
    left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
    
    # 3. Update video stream
    np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
```

---

## 9. External Resources

### Documentation

| Resource | URL |
|----------|-----|
| OpenTeleVision Paper | [arxiv.org/abs/2407.01512](https://arxiv.org/abs/2407.01512) |
| KScale Teleop Docs | [docs.kscale.dev/robots/k-bot/teleop](https://docs.kscale.dev/robots/k-bot/teleop) |
| KScale VR Teleop Repo | [github.com/kscalelabs/kbot_vr_teleop](https://github.com/kscalelabs/kbot_vr_teleop) |
| dex-retargeting | [github.com/dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) |
| JAX Quickstart | [jax.readthedocs.io](https://jax.readthedocs.io) |
| URDF Tutorial | [wiki.ros.org/urdf](http://wiki.ros.org/urdf) |

### Libraries to Install

```bash
# Core dependencies
pip install jax jaxlib  # Add [cuda12] for GPU
pip install jaxopt
pip install vuer aiortc aiohttp-cors
pip install urdf-parser-py

# Optional
pip install mujoco  # For simulation
pip install rerun-sdk  # For visualization (KScale uses this)
```

### Reference Implementations

1. **KScale's signaling server** - How they handle IK offloading
2. **OpenTeleVision's `teleop_hand.py`** - Complete teleop loop example
3. **RUKA's `avp_teleoperator.py`** - Hand tracking integration

---

## Appendix A: WebXR Hand Joint Indices

```
WebXR provides 25 joints per hand:

Index | Joint Name
------|------------
0     | Wrist
1-4   | Thumb (metacarpal, proximal, distal, tip)
5-9   | Index (metacarpal, proximal, intermediate, distal, tip)
10-14 | Middle (metacarpal, proximal, intermediate, distal, tip)
15-19 | Ring (metacarpal, proximal, intermediate, distal, tip)
20-24 | Pinky (metacarpal, proximal, intermediate, distal, tip)

Fingertip indices: [4, 9, 14, 19, 24]
```

---

## Appendix B: Coordinate Frame Conventions

```
WebXR (VR Headset):     Robot (Typical):
    Y                       Z
    │                       │
    │                       │
    └────X                  └────X
   /                       /
  Z                       Y

Transform: Apply grd_yup2grd_zup matrix
```

---

## Appendix C: Quick Start Checklist

- [ ] Clone repo: `git clone https://github.com/adityabhas22/OpenTelevision.git`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Set up SSL certs for WebXR (see README)
- [ ] **Create URDF for 5DOF arm** ← BLOCKER
- [ ] Implement JAX FK from URDF
- [ ] Implement IK solver
- [ ] Test in simulation
- [ ] Set up MCU bridge
- [ ] Wire motors
- [ ] Full system test

---

*Document generated from research session on 2026-01-02. For questions, refer to the conversation history or contact the development team.*
