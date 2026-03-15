"""World model: predicts future frames from context.

- On GPU (Anyscale): Uses NVIDIA Cosmos-Predict2-2B-Video2World
- Locally: Uses a "dreamy" visual effect on ground truth frames to simulate
  what a world model prediction looks like (progressive blur + color shift
  + temporal warping) — visually meaningful for the demo.
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image, ImageFilter, ImageEnhance


def dream_effect(
    frames: np.ndarray,
    intensity: float = 1.0,
) -> np.ndarray:
    """Apply a 'dream' visual effect to ground truth frames.

    Creates a visually distinct version that simulates world model prediction:
    - Progressive Gaussian blur (grows over time → increasing uncertainty)
    - Subtle warm color shift (dream-like tone)
    - Slight brightness pulsing
    - Minor geometric warping

    Args:
        frames: [T, H, W, 3] uint8
        intensity: 0.0 = no effect, 1.0 = full dream effect

    Returns:
        dreamed: [T, H, W, 3] uint8
    """
    T = frames.shape[0]
    H, W = frames.shape[1], frames.shape[2]
    dreamed = []

    for i in range(T):
        progress = i / max(T - 1, 1)  # 0→1
        img = Image.fromarray(frames[i])

        # 1. Aggressive progressive blur
        blur_radius = intensity * (1.0 + progress * 6.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 2. Strong warm color shift — dream tint
        arr = np.array(img, dtype=np.float32)
        shift = intensity * (0.1 + progress * 0.3)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + shift * 1.2), 0, 255)     # strong red
        arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 + shift * 0.4), 0, 255)     # mild green
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - shift * 0.6), 0, 255)     # reduce blue
        img = Image.fromarray(arr.astype(np.uint8))

        # 3. Brightness pulsing
        brightness_factor = 1.0 + intensity * 0.15 * np.sin(progress * np.pi * 3)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # 4. Contrast reduction (hazy dream)
        contrast_factor = 1.0 - intensity * progress * 0.3
        img = ImageEnhance.Contrast(img).enhance(max(contrast_factor, 0.5))

        # 5. Slight spatial jitter (uncertainty wobble)
        if progress > 0.2:
            jitter = int(intensity * progress * 4)
            if jitter > 0:
                dx = np.random.randint(-jitter, jitter + 1)
                dy = np.random.randint(-jitter, jitter + 1)
                arr = np.array(img)
                shifted = np.roll(np.roll(arr, dx, axis=1), dy, axis=0)
                # Blend 70% shifted, 30% original for subtle wobble
                arr = (0.7 * shifted + 0.3 * arr).astype(np.uint8)
                img = Image.fromarray(arr)

        # 6. Vignette darkening at edges (dream focus)
        if progress > 0.1:
            arr = np.array(img, dtype=np.float32)
            y, x = np.ogrid[:H, :W]
            cy, cx = H / 2, W / 2
            dist = np.sqrt((x - cx) ** 2 / (cx ** 2) + (y - cy) ** 2 / (cy ** 2))
            vignette = 1.0 - np.clip(dist - 0.7, 0, 1) * intensity * progress * 0.8
            arr *= vignette[:, :, np.newaxis]
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        dreamed.append(np.array(img))

    return np.stack(dreamed, axis=0).astype(np.uint8)


def predict_future(
    context_frames: np.ndarray,
    future_frames: np.ndarray,
    num_predicted: int = 24,
    use_cosmos: bool = False,
    cosmos_predictor=None,
) -> np.ndarray:
    """Predict future frames given context.

    Args:
        context_frames: [num_ctx, H, W, 3] uint8 — conditioning frames
        future_frames: [T, H, W, 3] uint8 — ground truth (for local mock)
        num_predicted: number of frames to predict
        use_cosmos: if True, use the real Cosmos model
        cosmos_predictor: CosmosPredictor Ray actor (when use_cosmos=True)

    Returns:
        predicted: [num_predicted, H, W, 3] uint8
    """
    if use_cosmos and cosmos_predictor is not None:
        try:
            import ray
            predicted = ray.get(
                cosmos_predictor.predict.remote(context_frames, num_predicted)
            )
            return predicted
        except Exception as e:
            print(f"Cosmos prediction failed ({e}), falling back to dream effect")

    # Local mock: apply dream effect to ground truth
    available = future_frames[:num_predicted]
    if len(available) < num_predicted:
        # Pad by repeating last frame
        pad = np.stack(
            [available[-1]] * (num_predicted - len(available)), axis=0
        )
        available = np.concatenate([available, pad], axis=0)

    return dream_effect(available, intensity=1.0)


def generate_predictions(
    rollout: Dict,
    num_context: int = 9,
    num_predicted: int = 33,
    use_cosmos: bool = False,
) -> Dict:
    """Generate world model predictions for a single rollout.

    Splits the rollout into context and future, then predicts.

    Args:
        rollout: Dict with 'frames' [T, H, W, 3]
        num_context: number of context frames
        num_predicted: number of frames to predict
        use_cosmos: whether to use real Cosmos model

    Returns:
        Dict with context_frames, real_future, predicted_future
    """
    frames = rollout["frames"]
    total = frames.shape[0]

    if total < num_context + num_predicted:
        # Use what we have
        num_predicted = total - num_context
        if num_predicted <= 0:
            num_context = max(1, total // 3)
            num_predicted = total - num_context

    context = frames[:num_context]
    real_future = frames[num_context:num_context + num_predicted]
    predicted_future = predict_future(
        context, real_future, num_predicted, use_cosmos
    )

    return {
        "context_frames": context,
        "real_future": real_future,
        "predicted_future": predicted_future,
    }


if __name__ == "__main__":
    # Test dream effect
    import imageio

    os.makedirs("outputs", exist_ok=True)

    # Create a gradient test image sequence
    H, W, T = 480, 640, 40
    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        # Create a simple animated scene
        frames[t, :, :, 0] = 100  # red base
        frames[t, :, :, 1] = 150  # green base
        frames[t, :, :, 2] = 200  # blue base
        # Moving rectangle
        x = int(100 + t * 10) % W
        frames[t, 200:280, x:min(x+80, W), :] = [255, 200, 50]

    dreamed = dream_effect(frames)

    # Save both as video
    writer = imageio.get_writer("outputs/dream_test.mp4", fps=24)
    for t in range(T):
        # Side by side
        combined = np.concatenate([frames[t], dreamed[t]], axis=1)
        writer.append_data(combined)
    writer.close()
    print("Saved outputs/dream_test.mp4")
