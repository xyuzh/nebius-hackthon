"""Genesis Franka Panda environment for robotic manipulation data collection."""

import numpy as np
import os

import genesis as gs

_gs_initialized = False


def _ensure_genesis_init():
    """Initialize Genesis exactly once per process."""
    global _gs_initialized
    if not _gs_initialized:
        gs.init(backend=gs.gpu)
        _gs_initialized = True


class FrankaPandaEnv:
    """Franka Panda arm environment in Genesis physics simulator."""

    def __init__(self, image_size=(640, 480), dt=0.01, max_steps=200, headless=True):
        self.image_size = tuple(image_size)
        self.dt = dt
        self.max_steps = max_steps
        self.headless = headless

        _ensure_genesis_init()

        self.scene = gs.Scene(
            dt=self.dt,
            renderer=gs.renderers.Rasterizer(),
        )

        # Ground plane
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Franka Panda robot
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )

        # Target object - a small cube on the table
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.5, 0.0, 0.02),
            ),
        )

        # Camera for rendering
        self.cam = self.scene.add_camera(
            res=self.image_size,
            pos=(1.5, 1.5, 1.2),
            lookat=(0.0, 0.0, 0.4),
            fov=40,
        )

        self.scene.build()

        # Get joint info
        self.num_dofs = self.robot.n_dofs
        # Franka has 9 DOFs: 7 arm joints + 2 gripper fingers
        self.arm_dof_indices = list(range(7))
        self.gripper_dof_indices = [7, 8]

        # Home position for the arm
        self.home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])

    def reset(self, target_pos=None):
        """Reset environment with optional random target position."""
        self.step_count = 0

        # Randomize target position if not provided
        if target_pos is None:
            target_pos = np.array([
                0.4 + np.random.uniform(-0.15, 0.15),
                0.0 + np.random.uniform(-0.15, 0.15),
                0.02,
            ])

        self.target_pos = target_pos

        # Reset robot to home position
        self.robot.set_qpos(self.home_qpos)

        self.scene.step()
        return self._get_obs()

    def _get_obs(self):
        """Get current observation (camera frame)."""
        # Genesis camera API varies by version — try multiple approaches
        render_result = self.cam.render()

        frame = None
        if render_result is not None:
            # render() returned data directly (some versions)
            if isinstance(render_result, tuple):
                frame = render_result[0]  # (rgb, depth, ...) tuple
            else:
                frame = render_result
        else:
            # render() was void — retrieve via accessor methods
            for method in ('get_color', 'get_rgb', 'get_image'):
                if hasattr(self.cam, method):
                    frame = getattr(self.cam, method)()
                    break

        if frame is None:
            raise RuntimeError("Could not retrieve camera frame from Genesis")

        return self._to_rgb_uint8(frame)

    @staticmethod
    def _to_rgb_uint8(frame):
        """Convert any frame format to [H, W, 3] uint8 numpy array."""
        # Torch tensor -> numpy
        if hasattr(frame, 'cpu'):
            frame = frame.cpu().numpy()
        # RGBA -> RGB
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        # Float [0,1] -> uint8 [0,255]
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def step(self, action):
        """Step the environment with joint position targets."""
        # Set joint position targets
        self.robot.control_dofs_position(action)
        self.scene.step()
        self.step_count += 1

        obs = self._get_obs()
        done = self.step_count >= self.max_steps
        return obs, done

    def get_scripted_action(self, phase, t):
        """Generate scripted manipulation trajectory.

        Phases:
        0: Reach above the object
        1: Lower to grasp
        2: Close gripper
        3: Lift object
        """
        qpos = np.array(self.home_qpos, dtype=np.float32)

        if phase == 0:
            # Reach: interpolate toward a pre-grasp pose above the target
            alpha = min(t / 50.0, 1.0)
            reach_qpos = np.array([0.0, 0.2, 0.0, -1.8, 0.0, 2.0, 0.785])
            qpos[:7] = self.home_qpos[:7] * (1 - alpha) + reach_qpos * alpha
            qpos[7:9] = 0.04  # gripper open

        elif phase == 1:
            # Lower: move down toward the object
            alpha = min(t / 40.0, 1.0)
            pre_grasp = np.array([0.0, 0.2, 0.0, -1.8, 0.0, 2.0, 0.785])
            grasp_qpos = np.array([0.0, 0.4, 0.0, -1.5, 0.0, 2.2, 0.785])
            qpos[:7] = pre_grasp * (1 - alpha) + grasp_qpos * alpha
            qpos[7:9] = 0.04  # gripper open

        elif phase == 2:
            # Close gripper
            qpos[:7] = np.array([0.0, 0.4, 0.0, -1.5, 0.0, 2.2, 0.785])
            qpos[7:9] = 0.0  # gripper closed

        elif phase == 3:
            # Lift
            alpha = min(t / 50.0, 1.0)
            grasp_qpos = np.array([0.0, 0.4, 0.0, -1.5, 0.0, 2.2, 0.785])
            lift_qpos = np.array([0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.785])
            qpos[:7] = grasp_qpos * (1 - alpha) + lift_qpos * alpha
            qpos[7:9] = 0.0  # gripper closed

        return qpos

    def collect_rollout(self, seed=None):
        """Collect a full rollout with scripted policy.

        Returns:
            dict with keys:
                frames: np.ndarray [T, H, W, 3] uint8
                actions: np.ndarray [T, action_dim] float32
        """
        if seed is not None:
            np.random.seed(seed)

        obs = self.reset()
        frames = [obs]
        actions = []

        # Phase durations
        phase_steps = [50, 40, 30, 80]  # reach, lower, grasp, lift = 200 total
        phase = 0
        phase_t = 0

        for step in range(self.max_steps - 1):
            # Get scripted action for current phase
            action = self.get_scripted_action(phase, phase_t)
            actions.append(action)

            obs, done = self.step(action)
            frames.append(obs)

            phase_t += 1
            if phase < len(phase_steps) - 1 and phase_t >= phase_steps[phase]:
                phase += 1
                phase_t = 0

            if done:
                break

        return {
            "frames": np.stack(frames, axis=0),  # [T, H, W, 3]
            "actions": np.array(actions, dtype=np.float32),  # [T-1, action_dim]
        }


def test_env():
    """Test the environment by collecting a single rollout and saving video."""
    import imageio

    os.makedirs("outputs", exist_ok=True)

    env = FrankaPandaEnv()
    print("Collecting test rollout...")
    rollout = env.collect_rollout(seed=42)

    print(f"Frames shape: {rollout['frames'].shape}")
    print(f"Actions shape: {rollout['actions'].shape}")

    # Save as video
    output_path = "outputs/test_rollout.mp4"
    writer = imageio.get_writer(output_path, fps=24)
    for frame in rollout["frames"]:
        writer.append_data(frame)
    writer.close()
    print(f"Saved test rollout to {output_path}")


if __name__ == "__main__":
    test_env()
