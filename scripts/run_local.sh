#!/bin/bash
# Run the pipeline locally for development/testing
#
# Usage:
#   bash scripts/run_local.sh                  # full pipeline
#   bash scripts/run_local.sh --num-workers 1  # minimal test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_DIR}"

echo "╔══════════════════════════════════════════════╗"
echo "║   Teaching Robots to Dream - Local Run       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Install dependencies if needed
if ! python -c "import genesis" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Create output directory
mkdir -p outputs

# Run pipeline
echo "Starting pipeline..."
python -m src.pipeline --config configs/default.yaml "$@"

echo ""
echo "Done! Check outputs/ for results."
