"""Ray-distributed parallel rollout collection from Genesis environments."""

import os
import numpy as np
import ray
import time
from typing import Dict, List

from src.genesis_env import FrankaPandaEnv


@ray.remote(num_gpus=0.5)
class GenesisWorker:
    """Ray actor that owns a Genesis environment and collects rollouts."""

    def __init__(self, worker_id: int, image_size=(640, 480), dt=0.01, max_steps=200):
        self.worker_id = worker_id
        self.env = FrankaPandaEnv(
            image_size=image_size,
            dt=dt,
            max_steps=max_steps,
        )
        print(f"[Worker {worker_id}] Genesis environment initialized")

    def collect_rollouts(self, num_rollouts: int, base_seed: int = 0) -> List[Dict]:
        """Collect multiple rollouts with different random seeds."""
        rollouts = []
        for i in range(num_rollouts):
            seed = base_seed + self.worker_id * num_rollouts + i
            t0 = time.time()
            rollout = self.env.collect_rollout(seed=seed)
            elapsed = time.time() - t0
            print(f"[Worker {self.worker_id}] Rollout {i+1}/{num_rollouts} "
                  f"({rollout['frames'].shape[0]} frames, {elapsed:.1f}s)")
            rollouts.append(rollout)
        return rollouts


def collect_dataset(config: dict) -> Dict[str, np.ndarray]:
    """Collect a dataset of rollouts using distributed Ray workers.

    Args:
        config: Configuration dict with env and collection settings.

    Returns:
        Dictionary with:
            frames: np.ndarray [N, T, H, W, 3] uint8 - all rollout frames
            actions: np.ndarray [N, T-1, action_dim] float32 - all actions
    """
    env_cfg = config["env"]
    coll_cfg = config["collection"]

    num_workers = coll_cfg["num_workers"]
    rollouts_per_worker = coll_cfg["rollouts_per_worker"]

    print(f"Spawning {num_workers} Genesis workers, "
          f"{rollouts_per_worker} rollouts each...")

    # Create workers
    workers = [
        GenesisWorker.remote(
            worker_id=i,
            image_size=env_cfg["image_size"],
            dt=env_cfg["dt"],
            max_steps=env_cfg["max_steps"],
        )
        for i in range(num_workers)
    ]

    # Dispatch collection tasks
    futures = [
        w.collect_rollouts.remote(rollouts_per_worker, base_seed=42)
        for w in workers
    ]

    # Gather results
    print("Waiting for all workers to finish...")
    t0 = time.time()
    all_results = ray.get(futures)
    elapsed = time.time() - t0

    # Flatten: list of list of rollouts -> single list
    all_rollouts = []
    for worker_rollouts in all_results:
        all_rollouts.extend(worker_rollouts)

    total_rollouts = len(all_rollouts)
    total_frames = sum(r["frames"].shape[0] for r in all_rollouts)
    print(f"Collected {total_rollouts} rollouts ({total_frames} total frames) "
          f"in {elapsed:.1f}s")

    # Stack into arrays
    frames = np.stack([r["frames"] for r in all_rollouts], axis=0)
    actions = np.stack([r["actions"] for r in all_rollouts], axis=0)

    dataset = {"frames": frames, "actions": actions}

    # Save to disk
    save_dir = "/tmp/genesis_dataset"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "frames.npy"), frames)
    np.save(os.path.join(save_dir, "actions.npy"), actions)
    print(f"Dataset saved to {save_dir}/ "
          f"(frames: {frames.shape}, actions: {actions.shape})")

    return dataset


def test_collection():
    """Test data collection with minimal config."""
    ray.init(ignore_reinit_error=True)

    config = {
        "env": {
            "image_size": [640, 480],
            "dt": 0.01,
            "max_steps": 200,
        },
        "collection": {
            "num_workers": 2,
            "rollouts_per_worker": 2,
        },
    }

    dataset = collect_dataset(config)
    print(f"Test dataset - frames: {dataset['frames'].shape}, "
          f"actions: {dataset['actions'].shape}")
    ray.shutdown()


if __name__ == "__main__":
    test_collection()
