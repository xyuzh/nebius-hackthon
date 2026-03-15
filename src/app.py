"""Teaching Robots to Dream — Gradio Web App.

Pre-renders animated side-by-side VIDEOS on startup (main thread, MuJoCo safe).
Buttons instantly serve the pre-rendered mp4 — real animated simulation.
"""

import os
import time
import numpy as np
import json
import imageio
import gradio as gr
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional, Dict

from src.robot_sim import G1Simulator, get_available_actions, ACTION_REGISTRY, STAND
from src.prompt_to_action import parse_prompt_keywords, describe_sequence
from src.render_video import _get_font


# ── Globals ──────────────────────────────────────────────────────────────────

_stats: Optional[Dict] = None
_video_cache: Dict[str, str] = {}  # action_name -> path to mp4


def _ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = 6.5025, 58.5225
    ma, mb = a.mean(), b.mean()
    return float(((2*ma*mb+C1)*(2*((a-ma)*(b-mb)).mean()+C2))/((ma**2+mb**2+C1)*(a.var()+b.var()+C2)))


def _panel(real, dream, label, ssim_v, prog):
    """Render one side-by-side frame."""
    H, W = real.shape[:2]
    if dream.shape[:2] != (H, W):
        dream = np.array(Image.fromarray(dream).resize((W, H)))
    g, bh = 6, 28
    cw, ch = W*2+g, H+bh
    c = np.full((ch, cw, 3), 12, dtype=np.uint8)
    c[:H, :W] = real
    c[:H, W+g:] = dream
    c[:H, W:W+g] = 25
    img = Image.fromarray(c)
    d = ImageDraw.Draw(img)
    ft = _get_font(13)
    d.text((6, H+5), "Reality (MuJoCo)", fill=(100,255,100), font=ft)
    d.text((W+g+6, H+5), "World Model", fill=(100,180,255), font=ft)
    if label:
        d.text((cw//2-40, H+5), label, fill=(255,220,100), font=ft)
    d.text((cw-120, H+5), f"SSIM:{ssim_v:.2f}", fill=(170,170,170), font=ft)
    py = H+22
    d.rectangle([6, py, cw-6, py+3], fill=(35,35,45))
    d.rectangle([6, py, 6+int((cw-12)*prog), py+3], fill=(80,180,255))
    return np.array(img)


def _predict(model, ctx, img_size=128):
    import torch
    if model is None:
        return np.array(Image.fromarray(ctx[-1]).filter(ImageFilter.GaussianBlur(6)))
    n = model.n_context
    fs = ctx[-n:]
    H, W = fs[0].shape[:2]
    proc = []
    for f in fs:
        a = np.array(Image.fromarray(f).resize((img_size, img_size)), dtype=np.float32) / 255.0
        proc.append(a.transpose(2, 0, 1))
    with torch.no_grad():
        t = torch.from_numpy(np.stack(proc)).unsqueeze(0)
        p = model(t)[0].numpy()
    out = (p.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return np.array(Image.fromarray(out).resize((W, H), Image.BILINEAR))


def precache_videos():
    """Pre-render animated mp4 for every action. Runs on main thread."""
    global _video_cache, _stats

    os.makedirs("outputs/videos", exist_ok=True)

    sim = G1Simulator(image_size=(480, 360))

    # Load model
    model = None
    if os.path.exists("outputs/world_model.pt"):
        try:
            import torch
            from src.train_world_model import VideoPredictor
            model = VideoPredictor(n_context=4)
            model.load_state_dict(
                torch.load("outputs/world_model.pt", weights_only=True, map_location="cpu")
            )
            model.eval()
            print("World model loaded")
        except Exception as e:
            print(f"Model load failed: {e}")

    if os.path.exists("outputs/training_stats.json"):
        with open("outputs/training_stats.json") as f:
            _stats = json.load(f)

    print("Pre-rendering action videos...")
    for aname, entry in ACTION_REGISTRY.items():
        t0 = time.time()
        vid_path = f"outputs/videos/{aname}.mp4"

        rollout = sim.execute_actions(
            [{"action": aname, "duration": entry["duration"]}], fps=15
        )
        frames = rollout["frames"]
        n = frames.shape[0]
        nc = min(4, max(2, n // 4))

        writer = imageio.get_writer(vid_path, fps=15, macro_block_size=1)

        # Context frames (same on both sides)
        real_buf = []
        for i in range(min(nc, n)):
            real_buf.append(frames[i])
            p = _panel(frames[i], frames[i], "context", 1.0, i / max(n-1, 1))
            writer.append_data(p)

        # Prediction frames (teacher-forced)
        for i in range(nc, n):
            real = frames[i]
            real_buf.append(real)
            dream = _predict(model, real_buf)
            sv = _ssim(real, dream)
            p = _panel(real, dream, aname, sv, i / max(n-1, 1))
            writer.append_data(p)

        # Hold last frame for 0.5s
        for _ in range(8):
            writer.append_data(p)

        writer.close()
        _video_cache[aname] = vid_path
        elapsed = time.time() - t0
        print(f"  {aname}: {n} frames -> {vid_path} ({elapsed:.1f}s)")

    print(f"Pre-rendered {len(_video_cache)} action videos")


def get_video(prompt: str) -> Optional[str]:
    """Return video for prompt. Cached actions are instant. Custom combos use subprocess."""
    if not prompt or not prompt.strip():
        return _video_cache.get("wave") or _video_cache.get("stand")

    actions = parse_prompt_keywords(prompt.strip())
    if not actions:
        return _video_cache.get("stand")

    # Single cached action -> instant
    if len(actions) == 1 and actions[0]["action"] in _video_cache:
        return _video_cache[actions[0]["action"]]

    # Multi-action or custom -> render via subprocess (MuJoCo safe)
    return _render_via_subprocess(actions)


def _render_via_subprocess(actions) -> str:
    """Render custom action sequence in a subprocess (avoids MuJoCo thread deadlock)."""
    import subprocess, tempfile, json

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="outputs/videos")
    tmp.close()

    actions_json = json.dumps(actions)
    python = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv", "bin", "python")
    if not os.path.exists(python):
        python = "python"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        result = subprocess.run(
            [python, "-m", "src.render_worker", actions_json, tmp.name],
            capture_output=True, text=True, timeout=30, env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if result.returncode == 0 and os.path.exists(tmp.name):
            return tmp.name
        else:
            print(f"Render subprocess failed: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("Render subprocess timed out")
    except Exception as e:
        print(f"Render error: {e}")

    # Fallback to first cached action
    first = actions[0]["action"] if actions else "stand"
    return _video_cache.get(first, _video_cache.get("stand"))


# ── Gradio UI ────────────────────────────────────────────────────────────────

def create_app():
    default_video = _video_cache.get("wave") or list(_video_cache.values())[0]

    with gr.Blocks(title="Teaching Robots to Dream") as app:
        gr.Markdown(
            "# Teaching Robots to Dream\n"
            "**Unitree G1 Humanoid** — click an action to watch the robot "
            "in simulation (left) vs trained world model prediction (right)."
        )

        with gr.Row():
            with gr.Column(scale=3):
                vid_out = gr.Video(
                    label="Reality vs World Model Prediction",
                    value=default_video,
                    autoplay=True,
                    loop=True,
                    height=440,
                )

            with gr.Column(scale=2):
                txt = gr.Textbox(label="Command", placeholder="e.g. walk forward")
                txt.submit(fn=get_video, inputs=txt, outputs=vid_out)

                gr.Markdown("**Actions:**")
                with gr.Row():
                    for ex in ["Wave", "Walk forward", "Dance", "Kick", "Bow"]:
                        b = gr.Button(ex, size="sm")
                        b.click(fn=get_video, inputs=gr.State(ex), outputs=vid_out)
                with gr.Row():
                    for ex in ["Squat", "Run", "Raise arms", "Sit down", "Turn left"]:
                        b = gr.Button(ex, size="sm")
                        b.click(fn=get_video, inputs=gr.State(ex), outputs=vid_out)
                with gr.Row():
                    for ex in ["Walk backward", "Turn right", "Stand"]:
                        b = gr.Button(ex, size="sm")
                        b.click(fn=get_video, inputs=gr.State(ex), outputs=vid_out)

                if os.path.exists("outputs/training_stats.png"):
                    gr.Image(
                        value="outputs/training_stats.png",
                        label="World Model Training", height=160,
                    )

                with gr.Accordion("Model Info", open=False):
                    info = ""
                    if _stats:
                        info = (
                            f"**U-Net World Model**: {_stats['model_params']:,} params\n\n"
                            f"- {_stats['epochs']} epochs on {_stats['total_frames']} frames\n"
                            f"- Resolution: {_stats['img_size']}x{_stats['img_size']}\n"
                            f"- Final loss: {_stats['final_loss']:.5f}\n\n"
                        )
                    info += "**Robot**: Unitree G1 (29-DOF humanoid)\n\n"
                    info += "**Actions:** " + ", ".join(
                        f"`{n}`" for n in get_available_actions()
                    )
                    gr.Markdown(info)

    return app


def main():
    os.makedirs("outputs", exist_ok=True)
    precache_videos()
    print("\nLaunching at http://localhost:7860")
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
