"""Side-by-side comparison video renderer: Reality vs Robot's Dream."""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional
import imageio


def compute_metrics(real: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute similarity metrics between real and predicted frames.

    Args:
        real: [H, W, 3] uint8
        predicted: [H, W, 3] uint8

    Returns:
        Dict with MSE and SSIM scores
    """
    real_f = real.astype(np.float64)
    pred_f = predicted.astype(np.float64)

    # MSE
    mse = np.mean((real_f - pred_f) ** 2)

    # Simplified SSIM (per-channel mean)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_r = np.mean(real_f)
    mu_p = np.mean(pred_f)
    sigma_r = np.var(real_f)
    sigma_p = np.var(pred_f)
    sigma_rp = np.mean((real_f - mu_r) * (pred_f - mu_p))

    ssim = ((2 * mu_r * mu_p + C1) * (2 * sigma_rp + C2)) / \
           ((mu_r ** 2 + mu_p ** 2 + C1) * (sigma_r + sigma_p + C2))

    return {"mse": float(mse), "ssim": float(ssim)}


def _get_font(size: int = 24):
    """Try to load a nice font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_frame_pair(
    real_frame: np.ndarray,
    predicted_frame: np.ndarray,
    step: int,
    total_steps: int,
    phase_name: str = "",
    metrics: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Render a single side-by-side comparison frame with overlays.

    Args:
        real_frame: [H, W, 3] uint8
        predicted_frame: [H, W, 3] uint8
        step: current step number
        total_steps: total number of steps
        phase_name: description of current phase
        metrics: optional dict with mse/ssim

    Returns:
        combined: [H_out, W_out, 3] uint8 - combined frame with overlays
    """
    H, W = real_frame.shape[:2]

    # Ensure predicted frame matches dimensions
    if predicted_frame.shape[:2] != (H, W):
        pred_img = Image.fromarray(predicted_frame).resize((W, H))
        predicted_frame = np.array(pred_img)

    # Create canvas: header bar + two panels side by side + gap
    # Dimensions are chosen to ensure total is divisible by 16 (video codec compat)
    gap = 16
    header_h = 64
    footer_h = 48
    canvas_w = W * 2 + gap
    canvas_h = H + header_h + footer_h
    # Round up to nearest multiple of 16
    canvas_w = ((canvas_w + 15) // 16) * 16
    canvas_h = ((canvas_h + 15) // 16) * 16

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Header background (dark)
    canvas[:header_h, :, :] = 20

    # Footer background (dark)
    canvas[header_h + H:, :, :] = 20

    # Gap (dark line between panels)
    canvas[header_h:header_h + H, W:W + gap, :] = 40

    # Place frames
    canvas[header_h:header_h + H, :W, :] = real_frame
    canvas[header_h:header_h + H, W + gap:, :] = predicted_frame

    # Draw text overlays
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    title_font = _get_font(28)
    label_font = _get_font(20)
    small_font = _get_font(16)

    # Title
    draw.text((canvas_w // 2 - 200, 5), "Teaching Robots to Dream",
              fill=(255, 255, 255), font=title_font)

    # Panel labels
    draw.text((W // 2 - 100, 35), "Reality (MuJoCo Physics)",
              fill=(100, 255, 100), font=label_font)
    draw.text((W + gap + W // 2 - 110, 35), "Robot's Dream (World Model)",
              fill=(100, 180, 255), font=label_font)

    # Progress bar in footer
    bar_y = header_h + H + 8
    bar_x = 20
    bar_w = canvas_w - 40
    bar_h = 6
    progress = step / max(total_steps - 1, 1)

    # Bar background
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                    fill=(60, 60, 60))
    # Bar fill
    draw.rectangle([bar_x, bar_y, bar_x + int(bar_w * progress), bar_y + bar_h],
                    fill=(100, 200, 255))

    # Step counter
    step_text = f"Frame {step + 1}/{total_steps}"
    draw.text((bar_x, bar_y + 10), step_text, fill=(180, 180, 180), font=small_font)

    # Phase name
    if phase_name:
        draw.text((canvas_w // 2 - 40, bar_y + 10), phase_name,
                  fill=(255, 220, 100), font=small_font)

    # Metrics
    if metrics:
        metrics_text = f"SSIM: {metrics['ssim']:.3f}  |  MSE: {metrics['mse']:.1f}"
        draw.text((canvas_w - 300, bar_y + 10), metrics_text,
                  fill=(180, 180, 180), font=small_font)

    return np.array(img)


def get_phase_name(frame_idx: int, num_input: int) -> str:
    """Get a human-readable phase name based on frame index."""
    t = frame_idx
    if t < 8:
        return "Reaching..."
    elif t < 14:
        return "Lowering..."
    elif t < 18:
        return "Grasping..."
    else:
        return "Lifting!"


def render_comparison_video(
    prediction: Dict,
    output_path: str,
    fps: int = 24,
    num_input_frames: int = 9,
):
    """Render a single comparison video from a prediction result.

    Args:
        prediction: Dict with context_frames, real_future, predicted_future
        output_path: Path to save the output video
        fps: Video framerate
        num_input_frames: Number of context frames (shown as "both panels identical")
    """
    context = prediction["context_frames"]    # [num_input, H, W, 3]
    real = prediction["real_future"]           # [num_pred, H, W, 3]
    predicted = prediction["predicted_future"] # [num_pred, H, W, 3]

    total_frames = len(context) + len(real)
    all_rendered = []

    # Phase 1: Show context frames (same on both sides)
    for i in range(len(context)):
        frame = render_frame_pair(
            real_frame=context[i],
            predicted_frame=context[i],  # Same — this is the shared context
            step=i,
            total_steps=total_frames,
            phase_name="Context (shared input)",
        )
        all_rendered.append(frame)

    # Phase 2: Show divergence (real vs predicted)
    for i in range(len(real)):
        metrics = compute_metrics(real[i], predicted[i])
        frame = render_frame_pair(
            real_frame=real[i],
            predicted_frame=predicted[i],
            step=len(context) + i,
            total_steps=total_frames,
            phase_name=get_phase_name(i, num_input_frames),
            metrics=metrics,
        )
        all_rendered.append(frame)

    # Write video
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, quality=8)
    for frame in all_rendered:
        writer.append_data(frame)
    writer.close()
    print(f"Saved comparison video: {output_path} "
          f"({len(all_rendered)} frames, {len(all_rendered)/fps:.1f}s)")


def _render_text_card(
    text_lines: List[tuple],
    canvas_h: int,
    canvas_w: int,
) -> np.ndarray:
    """Render a text card (intro/outro/title) with centered text.

    Args:
        text_lines: List of (text, font_size, color) tuples
        canvas_h: Height of the card
        canvas_w: Width of the card

    Returns:
        [canvas_h, canvas_w, 3] uint8 frame
    """
    card = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    card[:, :, :] = 15  # near-black background

    # Subtle gradient top/bottom bars
    for y in range(4):
        card[y, :, :] = [40, 120, 200]
        card[canvas_h - 1 - y, :, :] = [40, 120, 200]

    img = Image.fromarray(card)
    draw = ImageDraw.Draw(img)

    # Calculate total text block height for centering
    line_heights = []
    for text, font_size, _ in text_lines:
        font = _get_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        line_heights.append(bbox[3] - bbox[1] + 12)

    total_text_h = sum(line_heights)
    y_start = (canvas_h - total_text_h) // 2

    y = y_start
    for text, font_size, color in text_lines:
        font = _get_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        x = (canvas_w - text_w) // 2
        draw.text((x, y), text, fill=color, font=font)
        y += line_heights[text_lines.index((text, font_size, color))]

    return np.array(img)


def render_highlight_reel(
    predictions: List[Dict],
    output_path: str,
    fps: int = 24,
    max_clips: int = 5,
    num_input_frames: int = 9,
):
    """Create a highlight reel concatenating the best comparison clips.

    Includes intro card, per-clip title cards, and outro card.
    Selects clips by highest average SSIM (most interesting predictions).

    Args:
        predictions: List of prediction result dicts
        output_path: Path to save the highlight reel
        fps: Video framerate
        max_clips: Maximum number of clips to include
        num_input_frames: Number of context frames
    """
    # Score each prediction by average SSIM
    scored = []
    for pred in predictions:
        real = pred["real_future"]
        predicted = pred["predicted_future"]
        num_frames = min(len(real), len(predicted))
        if num_frames == 0:
            continue

        avg_ssim = np.mean([
            compute_metrics(real[i], predicted[i])["ssim"]
            for i in range(num_frames)
        ])
        scored.append((avg_ssim, pred))

    # Sort by SSIM (higher = better prediction = more impressive)
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [pred for _, pred in scored[:max_clips]]

    if not selected:
        print("No valid predictions to render!")
        return

    # Determine canvas size to match render_frame_pair output exactly
    sample = selected[0]["context_frames"]
    H, W = sample.shape[1], sample.shape[2]
    gap = 16
    header_h = 64
    footer_h = 48
    canvas_w = ((W * 2 + gap + 15) // 16) * 16
    canvas_h = ((H + header_h + footer_h + 15) // 16) * 16

    all_frames = []

    # ── Intro Card (3 seconds) ─────────────────────────────────────────────
    intro = _render_text_card([
        ("Teaching Robots to Dream", 40, (255, 255, 255)),
        ("", 16, (0, 0, 0)),
        ("Genesis Physics Sim  +  NVIDIA Cosmos World Model", 22, (100, 200, 255)),
        ("Orchestrated with Ray on Anyscale", 18, (180, 180, 180)),
        ("", 16, (0, 0, 0)),
        ("Franka Panda  |  Scripted Manipulation  |  Video Prediction", 16, (150, 150, 150)),
    ], canvas_h, canvas_w)
    for _ in range(fps * 3):
        all_frames.append(intro)

    # ── Per-clip sequences ─────────────────────────────────────────────────
    for clip_idx, pred in enumerate(selected):
        context = pred["context_frames"]
        real = pred["real_future"]
        predicted = pred["predicted_future"]
        total = len(context) + len(real)

        # Title card for this clip (1.5 seconds)
        clip_title = _render_text_card([
            (f"Rollout #{pred.get('rollout_idx', clip_idx) + 1}", 36, (255, 255, 255)),
            ("", 12, (0, 0, 0)),
            (f"Average SSIM: {scored[clip_idx][0]:.3f}", 22, (100, 200, 255)),
            (f"Clip {clip_idx + 1} of {len(selected)}", 16, (140, 140, 140)),
        ], canvas_h, canvas_w)
        for _ in range(int(fps * 1.5)):
            all_frames.append(clip_title)

        # Render comparison frames
        for i in range(len(context)):
            frame = render_frame_pair(
                context[i], context[i], i, total,
                phase_name="Context (shared input)",
            )
            all_frames.append(frame)

        for i in range(len(real)):
            metrics = compute_metrics(real[i], predicted[i])
            frame = render_frame_pair(
                real[i], predicted[i],
                len(context) + i, total,
                phase_name=get_phase_name(i, num_input_frames),
                metrics=metrics,
            )
            all_frames.append(frame)

        # Brief pause between clips (0.5s)
        for _ in range(fps // 2):
            all_frames.append(all_frames[-1])

    # ── Outro Card (3 seconds) ─────────────────────────────────────────────
    avg_ssim_all = np.mean([s for s, _ in scored[:max_clips]])
    outro = _render_text_card([
        ("Thank You", 40, (255, 255, 255)),
        ("", 16, (0, 0, 0)),
        (f"Average SSIM across {len(selected)} rollouts: {avg_ssim_all:.3f}", 22, (100, 255, 100)),
        ("", 12, (0, 0, 0)),
        ("Genesis (43M FPS)  |  Cosmos-Predict2 (2B)  |  Ray/Anyscale", 18, (100, 200, 255)),
        ("Nebius Hackathon 2026", 16, (180, 180, 180)),
    ], canvas_h, canvas_w)
    for _ in range(fps * 3):
        all_frames.append(outro)

    # Write highlight reel
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, quality=8)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()
    duration = len(all_frames) / fps
    print(f"Saved highlight reel: {output_path} "
          f"({len(all_frames)} frames, {duration:.1f}s, {len(selected)} clips)")


def render_comparison_videos(
    dataset: Dict,
    predictions: List[Dict],
    config: dict,
):
    """Main entry point: render individual comparison videos + highlight reel.

    Args:
        dataset: Dataset dict (unused here but available for additional context)
        predictions: List of prediction result dicts from Cosmos
        config: Full configuration dict
    """
    video_cfg = config["video"]
    cosmos_cfg = config["cosmos"]
    output_dir = video_cfg["output_dir"]
    fps = video_cfg["fps"]
    num_input = cosmos_cfg["num_input_frames"]

    os.makedirs(output_dir, exist_ok=True)

    # Render individual comparison videos (up to 5)
    for i, pred in enumerate(predictions[:5]):
        path = os.path.join(output_dir, f"comparison_{i:02d}.mp4")
        render_comparison_video(pred, path, fps=fps, num_input_frames=num_input)

    # Render highlight reel
    reel_path = os.path.join(output_dir, "world_model_demo.mp4")
    render_highlight_reel(
        predictions, reel_path,
        fps=fps, max_clips=5, num_input_frames=num_input,
    )

    print(f"\nAll videos saved to {output_dir}/")
    print(f"  - {min(len(predictions), 5)} individual comparisons")
    print(f"  - 1 highlight reel: {reel_path}")


def test_render():
    """Test rendering with synthetic data."""
    os.makedirs("outputs", exist_ok=True)

    H, W = 480, 640
    num_input = 9
    num_pred = 24

    # Generate synthetic prediction data
    predictions = []
    for i in range(3):
        context = np.random.randint(0, 255, (num_input, H, W, 3), dtype=np.uint8)
        real = np.random.randint(0, 255, (num_pred, H, W, 3), dtype=np.uint8)
        # Predicted = real + noise (to simulate imperfect prediction)
        noise = np.random.randint(-30, 30, real.shape).astype(np.int16)
        predicted = np.clip(real.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        predictions.append({
            "context_frames": context,
            "real_future": real,
            "predicted_future": predicted,
            "rollout_idx": i,
        })

    config = {
        "video": {"fps": 24, "output_dir": "outputs"},
        "cosmos": {"num_input_frames": num_input},
    }

    render_comparison_videos({}, predictions, config)


if __name__ == "__main__":
    test_render()
