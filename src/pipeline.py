"""End-to-end pipeline: Genesis data collection → Cosmos prediction → Video rendering.

This is the main entry point for the Ray Job. It orchestrates:
1. Parallel data collection from Genesis physics simulator
2. Cosmos world model inference (robot "dreams")
3. Side-by-side comparison video rendering
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np

import ray


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(config: dict):
    """Execute the full pipeline.

    Stage 1: Collect rollout data from Genesis (distributed)
    Stage 2: Run Cosmos world model predictions (GPU)
    Stage 3: Render comparison videos
    """
    output_dir = config["video"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    total_t0 = time.time()

    # =========================================================================
    # Stage 1: Data Collection from Genesis
    # =========================================================================
    print("=" * 70)
    print("STAGE 1: Collecting rollout data from Genesis physics simulator")
    print("=" * 70)

    from src.data_collector import collect_dataset

    t0 = time.time()
    dataset = collect_dataset(config)
    stage1_time = time.time() - t0

    num_rollouts = dataset["frames"].shape[0]
    num_frames = dataset["frames"].shape[1]
    print(f"\n✓ Stage 1 complete: {num_rollouts} rollouts × {num_frames} frames "
          f"({stage1_time:.1f}s)")

    # Save a few sample rollouts as standalone videos
    import imageio
    for i in range(min(3, num_rollouts)):
        sample_path = os.path.join(output_dir, f"genesis_rollout_{i:02d}.mp4")
        writer = imageio.get_writer(sample_path, fps=config["video"]["fps"])
        for frame in dataset["frames"][i]:
            writer.append_data(frame)
        writer.close()
        print(f"  Saved sample rollout: {sample_path}")

    # =========================================================================
    # Stage 2: Cosmos World Model Predictions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Running Cosmos world model predictions ('Robot Dreams')")
    print("=" * 70)

    from src.cosmos_inference import cosmos_predict_batch

    t0 = time.time()
    predictions = cosmos_predict_batch(dataset, config)
    stage2_time = time.time() - t0

    print(f"\n✓ Stage 2 complete: {len(predictions)} dream sequences "
          f"({stage2_time:.1f}s)")

    # =========================================================================
    # Stage 3: Video Rendering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Rendering comparison videos (Reality vs Dreams)")
    print("=" * 70)

    from src.render_video import render_comparison_videos

    t0 = time.time()
    render_comparison_videos(dataset, predictions, config)
    stage3_time = time.time() - t0

    print(f"\n✓ Stage 3 complete: Videos saved to {output_dir}/ "
          f"({stage3_time:.1f}s)")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - total_t0
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Stage 1 (Genesis Data Collection): {stage1_time:.1f}s")
    print(f"  Stage 2 (Cosmos World Model):      {stage2_time:.1f}s")
    print(f"  Stage 3 (Video Rendering):         {stage3_time:.1f}s")
    print(f"  Total:                             {total_time:.1f}s")
    print(f"\nOutputs:")

    # List output files
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {fpath} ({size_mb:.1f} MB)")

    print(f"\n🎬 Demo video: {output_dir}/world_model_demo.mp4")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Teaching Robots to Dream - World Model Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override number of Genesis workers"
    )
    parser.add_argument(
        "--num-rollouts", type=int, default=None,
        help="Override rollouts per worker"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.num_workers is not None:
        config["collection"]["num_workers"] = args.num_workers
    if args.num_rollouts is not None:
        config["collection"]["rollouts_per_worker"] = args.num_rollouts
    if args.output_dir is not None:
        config["video"]["output_dir"] = args.output_dir

    # Initialize Ray
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": ".",
            "excludes": ["outputs/", ".git/", "__pycache__/"],
        },
    )

    print("Teaching Robots to Dream")
    print(f"Ray initialized: {ray.cluster_resources()}")
    print(f"Config: {config}\n")

    try:
        output_dir = run_pipeline(config)
        print(f"\nSuccess! Check {output_dir}/ for results.")
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
