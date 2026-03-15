#!/bin/bash
# Launch the Gradio web app locally
#
# Usage:
#   bash scripts/run_app.sh
#   OPENAI_API_KEY=sk-... bash scripts/run_app.sh  # with LLM prompt parsing
#   ANTHROPIC_API_KEY=sk-... bash scripts/run_app.sh  # alternative LLM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "========================================"
echo "  Teaching Robots to Dream"
echo "  Unitree Go2 + World Model + Chat"
echo "========================================"
echo ""

if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "LLM: OpenAI (advanced prompt parsing)"
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "LLM: Anthropic Claude (advanced prompt parsing)"
else
    echo "LLM: None (keyword matching - set OPENAI_API_KEY or ANTHROPIC_API_KEY for smarter parsing)"
fi
echo ""

# Install deps if needed
if ! .venv/bin/python -c "import gradio" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install --python .venv/bin/python -r requirements.txt
fi

mkdir -p outputs

echo "Starting web app on http://localhost:7860 ..."
echo ""
exec .venv/bin/python -m src.app
