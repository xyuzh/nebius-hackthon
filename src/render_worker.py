"""Subprocess worker for MuJoCo rendering.

MuJoCo OpenGL deadlocks in threads on macOS. This script runs as a subprocess
to render simulation frames on the main thread, writing output to a temp video.
"""

import sys
import os
import json
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFilter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robot_sim import G1Simulator, ACTION_REGISTRY, STAND
from src.render_video import _get_font


def _ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = 6.5025, 58.5225
    ma, mb = a.mean(), b.mean()
    return float(((2*ma*mb+C1)*(2*((a-ma)*(b-mb)).mean()+C2))/((ma**2+mb**2+C1)*(a.var()+b.var()+C2)))


def _panel(real, dream, label, ssim_v, prog):
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


def render_actions(actions_json: str, output_path: str):
    """Render action sequence to video file."""
    import torch
    actions = json.loads(actions_json)

    sim = G1Simulator(image_size=(480, 360))

    model = None
    if os.path.exists("outputs/world_model.pt"):
        from src.train_world_model import VideoPredictor
        model = VideoPredictor(n_context=4)
        model.load_state_dict(
            torch.load("outputs/world_model.pt", weights_only=True, map_location="cpu")
        )
        model.eval()

    rollout = sim.execute_actions(actions, fps=15)
    frames = rollout["frames"]
    labels = rollout["action_labels"]
    n = frames.shape[0]
    nc = min(4, max(2, n // 4))

    writer = imageio.get_writer(output_path, fps=15, macro_block_size=1)

    real_buf = []
    for i in range(min(nc, n)):
        real_buf.append(frames[i])
        p = _panel(frames[i], frames[i], "context", 1.0, i / max(n-1, 1))
        writer.append_data(p)

    for i in range(nc, n):
        real = frames[i]
        real_buf.append(real)
        dream = _predict(model, real_buf)
        sv = _ssim(real, dream)
        li = min(i, len(labels)-1)
        p = _panel(real, dream, labels[li], sv, i / max(n-1, 1))
        writer.append_data(p)

    for _ in range(8):
        writer.append_data(p)

    writer.close()
    print(f"OK:{n}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: render_worker.py <actions_json> <output_path>")
        sys.exit(1)
    render_actions(sys.argv[1], sys.argv[2])
