#!/bin/bash
# Submit the world model pipeline as a Ray Job on Anyscale
#
# Prerequisites:
#   - anyscale CLI installed and authenticated
#   - HF_TOKEN set for Cosmos model download
#
# Usage:
#   bash scripts/submit_ray_job.sh

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN for Hugging Face model access}"

JOB_NAME="robotics-world-model-$(date +%Y%m%d-%H%M%S)"
IMAGE_URI="anyscale/ray:2.44.1-slim-py312-cu128"

echo "╔══════════════════════════════════════════════╗"
echo "║   Teaching Robots to Dream - Ray Job Submit  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Job name:  ${JOB_NAME}"
echo "Image:     ${IMAGE_URI}"
echo ""

# ── Submit Job ─────────────────────────────────────────────────────────────
anyscale job submit \
  --name "${JOB_NAME}" \
  --image-uri "${IMAGE_URI}" \
  --working-dir . \
  --env-var "HF_TOKEN=${HF_TOKEN}" \
  --env-var "PYTHONPATH=." \
  --compute-config '{
    "head_node": {
      "instance_type": "g5.4xlarge"
    },
    "worker_nodes": [
      {
        "instance_type": "g5.2xlarge",
        "min_nodes": 2,
        "max_nodes": 4
      }
    ]
  }' \
  --pip-packages "genesis-world torch>=2.4 transformers>=4.52 accelerate>=1.7 diffusers>=0.34 sentencepiece protobuf moviepy>=2.0 imageio[ffmpeg] numpy Pillow pyyaml tqdm huggingface_hub" \
  -- python -m src.pipeline --config configs/default.yaml

echo ""
echo "Job submitted! Monitor with: anyscale job logs ${JOB_NAME}"
