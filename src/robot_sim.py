"""Unitree G1 humanoid simulator with MuJoCo.

Supports natural language actions: walk, wave, bow, squat, kick, dance, etc.
Optimized for real-time demo rendering.
"""

import os
import numpy as np
import mujoco
from typing import Dict, List, Tuple

# ── G1 actuator mapping (29 DOF) ────────────────────────────────────────────
# Left leg:  [0] hip_pitch [1] hip_roll [2] hip_yaw [3] knee [4] ankle_pitch [5] ankle_roll
# Right leg: [6] hip_pitch [7] hip_roll [8] hip_yaw [9] knee [10] ankle_pitch [11] ankle_roll
# Waist:     [12] yaw [13] roll [14] pitch
# Left arm:  [15] shoulder_pitch [16] shoulder_roll [17] shoulder_yaw [18] elbow
#            [19] wrist_roll [20] wrist_pitch [21] wrist_yaw
# Right arm: [22] shoulder_pitch [23] shoulder_roll [24] shoulder_yaw [25] elbow
#            [26] wrist_roll [27] wrist_pitch [28] wrist_yaw

L_HIP_P, L_HIP_R, L_HIP_Y, L_KNEE, L_ANKLE_P, L_ANKLE_R = 0, 1, 2, 3, 4, 5
R_HIP_P, R_HIP_R, R_HIP_Y, R_KNEE, R_ANKLE_P, R_ANKLE_R = 6, 7, 8, 9, 10, 11
WAIST_Y, WAIST_R, WAIST_P = 12, 13, 14
L_SH_P, L_SH_R, L_SH_Y, L_ELBOW = 15, 16, 17, 18
L_WR_R, L_WR_P, L_WR_Y = 19, 20, 21
R_SH_P, R_SH_R, R_SH_Y, R_ELBOW = 22, 23, 24, 25
R_WR_R, R_WR_P, R_WR_Y = 26, 27, 28

# Stand keyframe (from MuJoCo Menagerie)
STAND = np.zeros(29, dtype=np.float64)
STAND[L_SH_P] = 0.2;  STAND[R_SH_P] = 0.2
STAND[L_SH_R] = 0.2;  STAND[R_SH_R] = -0.2
STAND[L_ELBOW] = 1.28; STAND[R_ELBOW] = 1.28


def _lerp(a, b, t):
    """Linear interpolation clamped to [0,1]."""
    t = max(0.0, min(1.0, t))
    return a * (1 - t) + b * t


# ── Gait functions: (time, phase_fraction) → 29-dim control targets ─────────

def _stand(t, frac):
    return STAND.copy()


def _walk(t, frac, speed=1.0):
    """Bipedal walk: conservative gait tuned for stability."""
    ctrl = STAND.copy()
    phase = t * speed * 2.5

    s = np.sin(phase)
    c = np.cos(phase)

    # Conservative leg swing — small amplitude for stability
    hip_amp = 0.18
    knee_amp = 0.25
    ankle_amp = 0.12

    ctrl[L_HIP_P] = -hip_amp * s
    ctrl[L_KNEE] = knee_amp * max(s, 0)
    ctrl[L_ANKLE_P] = ankle_amp * s * 0.8

    ctrl[R_HIP_P] = hip_amp * s
    ctrl[R_KNEE] = knee_amp * max(-s, 0)
    ctrl[R_ANKLE_P] = -ankle_amp * s * 0.8

    # Lateral weight shift for balance
    ctrl[L_HIP_R] = 0.06 * c
    ctrl[R_HIP_R] = 0.06 * c
    ctrl[L_ANKLE_R] = -0.03 * c
    ctrl[R_ANKLE_R] = -0.03 * c

    # Arm counterswing
    arm_amp = 0.2
    ctrl[L_SH_P] = 0.2 + arm_amp * s
    ctrl[R_SH_P] = 0.2 - arm_amp * s
    ctrl[L_ELBOW] = 1.0
    ctrl[R_ELBOW] = 1.0

    return ctrl


def _run(t, frac):
    return _walk(t, frac, speed=2.0)


def _walk_backward(t, frac):
    return _walk(t, frac, speed=-0.8)


def _turn_left(t, frac):
    ctrl = STAND.copy()
    phase = t * 3.0
    # Yaw the waist and hips asymmetrically
    ctrl[WAIST_Y] = 0.3 * np.sin(phase)
    ctrl[L_HIP_Y] = 0.15 * np.sin(phase)
    ctrl[R_HIP_Y] = -0.15 * np.sin(phase)
    # Slight leg motion to shuffle
    ctrl[L_HIP_P] = -0.15 * np.sin(phase)
    ctrl[R_HIP_P] = 0.15 * np.sin(phase)
    ctrl[L_KNEE] = 0.2 * max(np.sin(phase), 0)
    ctrl[R_KNEE] = 0.2 * max(-np.sin(phase), 0)
    return ctrl


def _turn_right(t, frac):
    ctrl = _turn_left(t, frac)
    ctrl[WAIST_Y] *= -1
    ctrl[L_HIP_Y], ctrl[R_HIP_Y] = ctrl[R_HIP_Y], ctrl[L_HIP_Y]
    return ctrl


def _wave(t, frac):
    """Big enthusiastic wave with right hand."""
    ctrl = STAND.copy()
    phase = t * 5.0

    # Raise arm smoothly
    raise_alpha = min(frac * 3.0, 1.0)
    ctrl[R_SH_P] = _lerp(0.2, -2.0, raise_alpha)      # shoulder way up
    ctrl[R_SH_R] = _lerp(-0.2, -0.8, raise_alpha)      # out to side
    ctrl[R_SH_Y] = _lerp(0, 0.3, raise_alpha)
    ctrl[R_ELBOW] = _lerp(1.28, 0.5, raise_alpha)       # partially extended
    ctrl[R_WR_P] = 0.6 * np.sin(phase)                   # wave wrist
    ctrl[R_WR_Y] = 0.3 * np.sin(phase * 1.5)

    # Left arm relaxed at side
    ctrl[L_SH_P] = 0.3
    ctrl[L_ELBOW] = 1.0

    # Slight body engagement
    ctrl[WAIST_Y] = 0.08 * np.sin(phase * 0.5)
    ctrl[WAIST_R] = 0.04 * np.sin(phase * 0.3)
    return ctrl


def _bow(t, frac):
    """Bow forward."""
    ctrl = STAND.copy()
    # Smooth bow curve
    bow_angle = 0.5 * np.sin(frac * np.pi)  # 0 → 0.5 → 0
    ctrl[WAIST_P] = bow_angle
    # Slight knee bend for balance
    ctrl[L_KNEE] = 0.15 * bow_angle
    ctrl[R_KNEE] = 0.15 * bow_angle
    ctrl[L_HIP_P] = -0.1 * bow_angle
    ctrl[R_HIP_P] = -0.1 * bow_angle
    # Arms at sides
    ctrl[L_SH_P] = 0.4
    ctrl[R_SH_P] = 0.4
    ctrl[L_ELBOW] = 0.3
    ctrl[R_ELBOW] = 0.3
    return ctrl


def _squat(t, frac):
    """Squat down and up."""
    depth = 0.8 * np.sin(frac * np.pi)  # 0 → deep → 0
    ctrl = STAND.copy()
    ctrl[L_HIP_P] = -depth * 0.6
    ctrl[R_HIP_P] = -depth * 0.6
    ctrl[L_KNEE] = depth
    ctrl[R_KNEE] = depth
    ctrl[L_ANKLE_P] = -depth * 0.3
    ctrl[R_ANKLE_P] = -depth * 0.3
    # Arms forward for balance
    ctrl[L_SH_P] = -0.3 * depth
    ctrl[R_SH_P] = -0.3 * depth
    ctrl[L_ELBOW] = 0.5
    ctrl[R_ELBOW] = 0.5
    return ctrl


def _kick(t, frac):
    """Right leg kick — tiny motion, maximum stability."""
    ctrl = STAND.copy()

    if frac < 0.3:
        # Prepare: bend standing knee slightly
        alpha = frac / 0.3
        ctrl[L_KNEE] = 0.1 * alpha
    elif frac < 0.6:
        # Kick: small hip flexion only
        alpha = (frac - 0.3) / 0.3
        kick = 0.25 * np.sin(alpha * np.pi)  # smooth up-down
        ctrl[R_HIP_P] = -kick
        ctrl[L_KNEE] = 0.1
    else:
        # Return to stand
        alpha = (frac - 0.6) / 0.4
        ctrl[L_KNEE] = 0.1 * (1 - alpha)

    # Arms for counterbalance
    ctrl[L_SH_P] = -0.2
    ctrl[R_SH_P] = 0.3
    ctrl[L_ELBOW] = 0.8
    ctrl[R_ELBOW] = 0.8
    return ctrl


def _raise_arms(t, frac):
    """Raise both arms overhead."""
    ctrl = STAND.copy()
    lift = min(frac * 2.0, 1.0)  # quick raise
    ctrl[L_SH_P] = _lerp(0.2, -2.5, lift)
    ctrl[R_SH_P] = _lerp(0.2, -2.5, lift)
    ctrl[L_SH_R] = _lerp(0.2, 0.0, lift)
    ctrl[R_SH_R] = _lerp(-0.2, 0.0, lift)
    ctrl[L_ELBOW] = _lerp(1.28, 0.2, lift)
    ctrl[R_ELBOW] = _lerp(1.28, 0.2, lift)
    return ctrl


def _dance(t, frac):
    """Expressive upper-body dance — legs stay stable."""
    ctrl = STAND.copy()
    phase = t * 3.5

    # Gentle weight shift (small for stability)
    ctrl[L_HIP_R] = 0.04 * np.sin(phase)
    ctrl[R_HIP_R] = 0.04 * np.sin(phase)

    # Waist groove — main expression
    ctrl[WAIST_Y] = 0.25 * np.sin(phase * 0.5)
    ctrl[WAIST_R] = 0.12 * np.sin(phase)
    ctrl[WAIST_P] = 0.08 * np.sin(phase * 0.7)

    # Subtle knee bounce
    ctrl[L_KNEE] = 0.12 * max(np.sin(phase), 0)
    ctrl[R_KNEE] = 0.12 * max(-np.sin(phase), 0)
    ctrl[L_ANKLE_P] = -0.05 * max(np.sin(phase), 0)
    ctrl[R_ANKLE_P] = -0.05 * max(-np.sin(phase), 0)

    # Arm choreography — big expressive movements
    ctrl[L_SH_P] = -0.5 + 0.8 * np.sin(phase * 0.5)
    ctrl[R_SH_P] = -0.5 + 0.8 * np.sin(phase * 0.5 + np.pi)
    ctrl[L_SH_R] = 0.5 + 0.3 * np.sin(phase)
    ctrl[R_SH_R] = -0.5 - 0.3 * np.sin(phase)
    ctrl[L_ELBOW] = 0.8 + 0.4 * np.sin(phase * 2)
    ctrl[R_ELBOW] = 0.8 + 0.4 * np.sin(phase * 2 + np.pi)

    return ctrl


def _sit(t, frac):
    """Sit/crouch down."""
    alpha = min(frac * 2.0, 1.0)
    ctrl = STAND.copy()
    ctrl[L_HIP_P] = _lerp(0, -1.2, alpha)
    ctrl[R_HIP_P] = _lerp(0, -1.2, alpha)
    ctrl[L_KNEE] = _lerp(0, 2.0, alpha)
    ctrl[R_KNEE] = _lerp(0, 2.0, alpha)
    ctrl[L_ANKLE_P] = _lerp(0, -0.6, alpha)
    ctrl[R_ANKLE_P] = _lerp(0, -0.6, alpha)
    ctrl[L_SH_P] = 0.4
    ctrl[R_SH_P] = 0.4
    ctrl[L_ELBOW] = 0.5
    ctrl[R_ELBOW] = 0.5
    return ctrl


# ── Action registry ──────────────────────────────────────────────────────────

ACTION_REGISTRY = {
    "stand": {
        "fn": _stand,
        "duration": 1.0,
        "description": "Stand upright",
    },
    "walk_forward": {
        "fn": _walk,
        "duration": 3.0,
        "description": "Walk forward",
    },
    "walk_backward": {
        "fn": _walk_backward,
        "duration": 2.0,
        "description": "Walk backward",
    },
    "run": {
        "fn": _run,
        "duration": 3.0,
        "description": "Run forward",
    },
    "turn_left": {
        "fn": _turn_left,
        "duration": 2.0,
        "description": "Turn left",
    },
    "turn_right": {
        "fn": _turn_right,
        "duration": 2.0,
        "description": "Turn right",
    },
    "wave": {
        "fn": _wave,
        "duration": 2.5,
        "description": "Wave hello",
    },
    "bow": {
        "fn": _bow,
        "duration": 2.5,
        "description": "Bow forward",
    },
    "squat": {
        "fn": _squat,
        "duration": 2.5,
        "description": "Squat down and up",
    },
    "kick": {
        "fn": _kick,
        "duration": 2.0,
        "description": "Kick with right leg",
    },
    "raise_arms": {
        "fn": _raise_arms,
        "duration": 2.0,
        "description": "Raise both arms overhead",
    },
    "dance": {
        "fn": _dance,
        "duration": 4.0,
        "description": "Dance expressively",
    },
    "sit": {
        "fn": _sit,
        "duration": 2.0,
        "description": "Sit/crouch down",
    },
}


# ── Simulator ────────────────────────────────────────────────────────────────

class G1Simulator:
    """MuJoCo-based Unitree G1 humanoid simulator with offscreen rendering."""

    def __init__(self, image_size: Tuple[int, int] = (640, 480)):
        self.image_size = image_size

        # Load G1 scene (floor, skybox, lighting)
        from robot_descriptions import g1_mj_description
        scene_path = os.path.join(
            os.path.dirname(g1_mj_description.MJCF_PATH), "scene.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.model.opt.timestep = 0.002  # 500Hz physics
        self.data = mujoco.MjData(self.model)

        # Offscreen renderer
        self.renderer = mujoco.Renderer(self.model, image_size[1], image_size[0])

        # Camera: 3/4 front view
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = 150
        self.camera.elevation = -15
        self.camera.distance = 3.0
        self.camera.lookat[:] = [0.0, 0.0, 0.75]

        self.scene_option = mujoco.MjvOption()

        print(f"G1 humanoid initialized: {self.model.nu} actuators, "
              f"dt={self.model.opt.timestep}")

    def reset(self):
        """Reset to standing keyframe."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def render_frame(self) -> np.ndarray:
        """Render and return [H, W, 3] uint8. Camera tracks the robot."""
        # Track pelvis position (body 0 = world, body 1 = pelvis)
        pelvis_pos = self.data.xpos[1].copy()
        self.camera.lookat[:] = [pelvis_pos[0], pelvis_pos[1], 0.75]
        self.renderer.update_scene(self.data, self.camera, self.scene_option)
        return self.renderer.render().copy()

    def execute_actions(
        self,
        action_sequence: List[Dict],
        fps: int = 30,
    ) -> Dict[str, np.ndarray]:
        """Execute action sequence, return frames + metadata.

        Optimized: only renders at target FPS, physics runs at 500Hz.
        """
        self.reset()

        frames = []
        actions_log = []
        action_labels = []

        steps_per_frame = max(1, int(1.0 / (fps * self.model.opt.timestep)))

        for act_idx, action_spec in enumerate(action_sequence):
            action_name = action_spec["action"]
            if action_name not in ACTION_REGISTRY:
                continue

            # Reset to standing before each action to prevent accumulated falls
            if act_idx > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
                # Settle for a moment
                for _ in range(50):
                    self.data.ctrl[:29] = STAND
                    mujoco.mj_step(self.model, self.data)

            entry = ACTION_REGISTRY[action_name]
            action_fn = entry["fn"]
            duration = action_spec.get("duration", entry["duration"])
            total_steps = int(duration / self.model.opt.timestep)

            for step in range(total_steps):
                t = step * self.model.opt.timestep
                frac = step / max(total_steps - 1, 1)

                targets = action_fn(t, frac)
                self.data.ctrl[:29] = targets

                mujoco.mj_step(self.model, self.data)

                if step % steps_per_frame == 0:
                    frames.append(self.render_frame())
                    actions_log.append(targets.copy())
                    action_labels.append(action_name)

        if not frames:
            frames.append(self.render_frame())
            actions_log.append(STAND.copy())
            action_labels.append("stand")

        return {
            "frames": np.stack(frames, axis=0),
            "actions": np.array(actions_log),
            "action_labels": action_labels,
        }


def get_available_actions() -> Dict[str, str]:
    return {k: v["description"] for k, v in ACTION_REGISTRY.items()}


def test_simulator():
    """Generate test videos of all actions."""
    import imageio

    os.makedirs("outputs", exist_ok=True)
    sim = G1Simulator(image_size=(640, 480))

    # All-actions showcase
    sequence = [
        {"action": "stand", "duration": 0.5},
        {"action": "wave", "duration": 2.0},
        {"action": "walk_forward", "duration": 3.0},
        {"action": "turn_left", "duration": 1.5},
        {"action": "kick", "duration": 2.0},
        {"action": "dance", "duration": 3.0},
        {"action": "bow", "duration": 2.0},
        {"action": "sit", "duration": 1.5},
    ]

    print("Executing showcase sequence...")
    result = sim.execute_actions(sequence, fps=30)
    print(f"Generated {result['frames'].shape[0]} frames")

    path = "outputs/g1_showcase.mp4"
    writer = imageio.get_writer(path, fps=30)
    for frame in result["frames"]:
        writer.append_data(frame)
    writer.close()
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    test_simulator()
