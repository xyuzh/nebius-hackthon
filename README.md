# Teaching Robots to Dream

**Chat with a Unitree G1 humanoid robot.** Type a command in natural language, watch it execute in real-time physics simulation, and see a trained world model's prediction side-by-side.

> Nebius Hackathon 2026 | Robotics Track | [github.com/unitreerobotics](https://github.com/unitreerobotics)

## Quick Start

```bash
# Install
uv pip install --python .venv/bin/python -r requirements.txt

# Launch (trains world model on first run ~8 min, then instant)
PYTHONPATH=. .venv/bin/python -m src.app

# Open http://localhost:7860
```

## What It Does

1. **You type** a command: "Wave hello", "Dance", "Walk forward"
2. **MuJoCo simulates** the Unitree G1 humanoid executing the action (500Hz physics)
3. **Trained U-Net world model** predicts what it thinks will happen (next-frame prediction)
4. **Side-by-side streams** in real-time: Reality vs Prediction, with live SSIM metrics

## Architecture

```
User: "Dance"
    │
    ▼
┌──────────────┐    ┌────────────────────┐
│ LLM / Keyword │───→│ Action Sequence     │
│ Parser        │    │ [dance(4s)]         │
└──────────────┘    └────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
   ┌──────────────────┐         ┌──────────────────┐
   │ MuJoCo Physics    │         │ U-Net World Model │
   │ Unitree G1 (29DOF)│         │ 3.3M params       │
   │ 500Hz, 15fps render│         │ 128×128 trained   │
   └────────┬─────────┘         └────────┬─────────┘
            │                            │
            └──────────┬─────────────────┘
                       ▼
            ┌──────────────────┐
            │ Side-by-Side      │
            │ Streaming @ 15fps │──→ Gradio Web App
            │ + SSIM metrics    │
            └──────────────────┘
```

## World Model

- **Architecture**: U-Net with skip connections (encoder-decoder with skip connections for sharp predictions)
- **Parameters**: 3,281,379
- **Training**: 20 epochs on 702 frames at 128×128, MSE loss
- **Final loss**: 0.00038
- **Prediction**: Teacher-forced next-frame prediction from 4 context frames

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Robot | **Unitree G1** humanoid (29-DOF) from [unitreerobotics](https://github.com/unitreerobotics) |
| Physics | **MuJoCo** (500Hz, 15fps render) via MuJoCo Menagerie |
| World Model | **U-Net** ConvLSTM-style video predictor, trained on sim data |
| NLP | **GPT-4o-mini** / keyword parser for prompt → actions |
| Web UI | **Gradio** with real-time frame streaming |
| Orchestration | **Ray/Anyscale** for GPU-scale training & inference |

## Robot Actions (13)

`stand`, `walk_forward`, `walk_backward`, `run`, `turn_left`, `turn_right`,
`wave`, `bow`, `squat`, `kick`, `raise_arms`, `dance`, `sit`

## Project Structure

```
src/
├── app.py               # Gradio web app (streaming)
├── robot_sim.py          # G1 simulator + 13 action gaits
├── prompt_to_action.py   # NLP → action sequences
├── train_world_model.py  # U-Net training pipeline
├── world_model.py        # Inference + dream effects
├── render_video.py       # Video rendering utilities
├── cosmos_inference.py   # Cosmos (GPU cloud)
├── data_collector.py     # Ray distributed collection
├── genesis_env.py        # Genesis (GPU cloud)
└── pipeline.py           # Ray Job entrypoint
```

## Hackathon Team

Built at Nebius Hackathon, March 15, 2026.
