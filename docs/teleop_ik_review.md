# TeleOp / IK Review

This document captures the current issues in the `@teleop` and `@ik` subsystems, plus proposed solutions to unblock custom upper‑torso teleoperation (dual 5‑DOF arms, RUKA hands, fixed torso).

---

## 1. Joint Limit Parsing Bug

- **Issue:** `teleop/ik/robot.py` uses `joint.limit.lower or -jnp.pi` (same for `upper`). When a URDF sets the limit to `0.0`, Python treats it as falsy and replaces the bound with `±π`, so zero-bounded joints clip incorrectly and IK solutions can drift far outside hardware limits.
- **Fix:** Check for `None` explicitly before falling back. Example:
  ```python
  lower = joint.limit.lower if joint.limit.lower is not None else -jnp.pi
  upper = joint.limit.upper if joint.limit.upper is not None else jnp.pi
  ```
  Store these in Python lists, then convert to `jnp.array`.
- **Follow-up:** Consider also capturing `velocity`/`effort` fields as metadata for future torque limits.

---

## 2. Hard-Coded Frame Transforms

- **Issue:** `teleop/constants_vuer.py` and `teleop/Preprocessor.py` assume Inspire Hands: fixed `grd_yup2grd_zup` rotation, `hand2inspire` roll/pitch swap, static offsets between head and wrists, and ignore head rotation except for cameras. This prevents retargeting to other URDFs or torso geometries.
- **Fix:** Introduce a calibration/config layer:
  1. Calibrate `T_base^head` once (neutral stance).
  2. Store per-arm shoulder transforms (`T_base^{shoulder_L/R}`) and tool transforms (wrist → flange → RUKA adapter) in YAML.
  3. Each frame: VR wrist pose (head frame) → multiply by `T_base^head` → multiply by tool transform → pass to IK.
  4. Replace the static offsets (`[-0.6, 0, 1.6]`) with config-driven vectors.
- **Benefit:** Any URDF can be dropped in by editing config; calibration errors are isolated; head orientation naturally carries through once torso yaw is unlocked.

---

## 3. Arm Scaling & Workspace Clamping

- **Issue:** `teleop/utils/scaling.py` exists but is not integrated. Current teleop adds ad‑hoc offsets, so the upper torso robot will frequently request unreachable poses when the operator reaches beyond the robot’s bubble.
- **Fix:** Use `ArmScaler` with real measurements:
  - Compute robot arm length from URDF (`compute_arm_length_from_urdf`) or manual measurements.
  - Calibrate user arm length once per session; update `ArmScaler`.
  - Define workspace bounds for each arm (box or ellipsoid) and clamp targets before IK.
- **Benefit:** Prevents solver thrashing near limits, keeps targets in reachable subspace, and gives a single knob to adjust human/robot scale factors.

---

## 4. Dual Arm Solver Trustworthiness

- **Issue:** `teleop/ik/dual_arm_solver.py` always returns `left_converged=True` and `right_converged=True` without verifying residuals. No joint limit margin is enforced beyond clipping, so solutions near limits can snap abruptly.
- **Fixes:**
  1. After each `SmoothIKSolver` call, compute FK and compare to target; propagate `pos_error`, `ori_error`, and convergence flags up to the caller.
  2. Add limit margins to the residual (not just final clipping) by expanding `_make_residual_fn` to penalize `q` inside a configurable buffer from the hardware limit.
  3. Expose failure statuses so the teleop loop can hold last good angles or trigger a replan instead of sending bad commands.
- **Benefit:** Keeps both arms synchronized, prevents silent IK failures, and allows per-arm fallback policies.

---

## 5. TeleVision Stale Data Handling

- **Issue:** `teleop/TeleVision.py` swallows all exceptions and writes into shared memory even when incoming packets are malformed. No timestamps/validity flags exist, so downstream loops can solve IK on frames that are minutes old if tracking drops.
- **Fix:** Track a monotonic timestamp alongside each pose/landmark array. Add `valid` bits per stream. Downstream code should reject frames older than ~50‑100 ms and pause IK/hardware commands when data is stale.
- **Benefit:** Prevents robots from freezing in odd configurations when VR tracking glitches.

---

## 6. Pose Calibration Workflow

- **Issue:** There is no tooling to align VR head/wrist poses with the robot’s physical origin, so every transform is implicit.
- **Fix:** Build a CLI (`scripts/calibrate_teleop.py`) that:
  1. Prompts the operator to stand in neutral posture.
  2. Samples head pose and both wrist poses for a few seconds.
  3. Solves for `T_base^head` and shoulder offsets (either user-input or detected).
  4. Writes the calibration into the robot config file.
- **Benefit:** Reproducible calibrations, easier debugging, and simpler onboarding for new robot embodiments.

---

## 7. Simulation Path

- **Issue:** Isaac Gym harness (`teleop/teleop_hand.py`) is Inspire-specific and heavy; no simple way to test new URDF arms.
- **Recommendation:** Start with MuJoCo (Python package `mujoco` or `dm_control`):
  - Load your URDF torso and arms; run a small viewer to check FK matches IK solutions.
  - Once IK is validated, port the same URDF into Isaac or another GPU sim if you need stereo rendering.
- **Benefit:** Fast iteration without CUDA setup; easier debugging of link frames and tool transforms.

---

## 8. Hardware Control Stack Suggestion

- **Current Gap:** No defined bridge between PC and actuators for the dual-arm torso (RUKA hand already uses its own LSTM pipeline).
- **Suggestion:** Use a Teensy 4.1 or STM32 board as a UDP bridge:
  - Packet: `struct { float q_left[5]; float q_right[5]; float q_ruka[11]; uint32 flags; }`.
  - 200 Hz send rate from host; MCU interpolates to 1 kHz motor loops.
  - Watchdog stops motors if no packet for 100 ms; include ESTOP flag in the packet.
  - For future extensibility, wrap this in ROS 2 (`ros2_control`) if you need standardized tooling.

---

## 9. Action Items

1. Patch joint limit parsing in `RobotModel`.
2. Add robot-specific config files (base/EE link names, tool transforms, shoulder offsets, rest poses).
3. Implement calibration script to populate the config.
4. Integrate `ArmScaler` and workspace clamping into the teleop loop.
5. Refactor `VuerPreprocessor` to use calibrated transforms instead of hard-coded matrices.
6. Extend dual-arm solver to return convergence metrics and enforce limit margins in the residuals.
7. Add timestamps/validity flags to shared-memory VR data; update consumers to enforce freshness.
8. Spin up a MuJoCo-based sim harness for the torso URDF.
9. Define the hardware command protocol (UDP bridge) even before the motor drivers are ready.

This checklist can drive the next iteration before adding your custom URDF and hardware.
