"""Frame lengths and crop boxes for the BasicVSR++ and MiniMax H3 passes."""

from __future__ import annotations

import math

H3_FPS = 24.0
MIN_FRAMES = 5
MAX_SAMPLE_FRAMES = 362
FRAME_MOD = 17
FRAME_REM = 5
RESTORATION_SIZE = 256


def align_down(count: int) -> int:
    """Largest count <= ``count`` with ``count % 17 == 5``, or 0 when that is under 5."""
    if count < MIN_FRAMES:
        return 0
    drop = (count % FRAME_MOD - FRAME_REM) % FRAME_MOD
    aligned = count - drop
    return aligned if aligned >= MIN_FRAMES else 0


def align_up(count: int) -> int:
    count = max(MIN_FRAMES, int(count))
    while count % FRAME_MOD != FRAME_REM:
        count += 1
    return count


def context_frames(seconds: float) -> int:
    seconds = min(15.0, max(1.0, float(seconds)))
    return min(align_up(round(seconds * H3_FPS)), MAX_SAMPLE_FRAMES)


def fit_context(seconds: float, sample_frames: int) -> tuple[int, int]:
    requested = context_frames(seconds)
    available = max(0, int(sample_frames))
    fitted = align_down(min(available, requested, MAX_SAMPLE_FRAMES))
    if fitted < MIN_FRAMES:
        fitted = align_down(available)
    overlap = 22 if fitted > 22 else 0
    if overlap >= fitted:
        overlap = 0
    return fitted, overlap


def frames_at_h3_fps(frame_count: int, source_fps: float) -> list[int]:
    """Source indices for a constant 24 fps timeline."""
    frame_count = int(frame_count)
    if frame_count <= 0:
        return []
    if source_fps <= H3_FPS + 0.05:
        return list(range(frame_count))
    out_count = max(1, int(round(frame_count * H3_FPS / float(source_fps))))
    picked: list[int] = []
    last = -1
    for slot in range(out_count):
        index = min(frame_count - 1, (slot * frame_count) // out_count)
        if index != last:
            picked.append(index)
            last = index
    return picked


def split_samples(frame_indices: list[int]) -> list[list[int]]:
    windows: list[list[int]] = []
    start = 0
    total = len(frame_indices)
    while start < total:
        if total - start <= MAX_SAMPLE_FRAMES:
            count = align_down(total - start)
            if count >= MIN_FRAMES:
                windows.append(frame_indices[start : start + count])
            break
        count = align_down(MAX_SAMPLE_FRAMES)
        windows.append(frame_indices[start : start + count])
        start += count
    return windows


def generation_size(width: int, height: int, side: int) -> tuple[int, int]:
    """Keep the crop's shape at the pixel count of a ``side`` square."""
    side = max(512, min(1280, int(side)))
    megapixels = (side / 1024) ** 2
    total = megapixels * 1024 * 1024
    scale = math.sqrt(total / (width * height))
    out_w = max(32, round(width * scale / 32) * 32)
    out_h = max(32, round(height * scale / 32) * 32)
    return out_w, out_h


def expand_box(x1: int, y1: int, x2: int, y2: int, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    """Grow a detection toward the 256 restoration square, staying inside the frame."""
    width = x2 - x1
    height = y2 - y1
    border = max(20, int(max(width, height) * 0.06))
    x1 = max(0, x1 - border)
    y1 = max(0, y1 - border)
    x2 = min(frame_w, x2 + border)
    y2 = min(frame_h, y2 + border)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    scale = min(1.0, RESTORATION_SIZE / width, RESTORATION_SIZE / height)
    missing_w = int((RESTORATION_SIZE - width * scale) / scale) if scale > 0 else 0
    missing_h = int((RESTORATION_SIZE - height * scale) / scale) if scale > 0 else 0
    budget_w = width
    budget_h = height
    left_room = x1
    right_room = frame_w - x2
    top_room = y1
    bottom_room = frame_h - y2
    grow_x = min(left_room, right_room, missing_w // 2, budget_w)
    extra_left = min(left_room - grow_x, missing_w - grow_x * 2, budget_w - grow_x)
    extra_right = min(right_room - grow_x, missing_w - grow_x * 2 - extra_left, budget_w - grow_x - extra_left)
    grow_y = min(top_room, bottom_room, missing_h // 2, budget_h)
    extra_top = min(top_room - grow_y, missing_h - grow_y * 2, budget_h - grow_y)
    extra_bottom = min(bottom_room - grow_y, missing_h - grow_y * 2 - extra_top, budget_h - grow_y - extra_top)
    x1 = max(0, x1 - math.floor(grow_x / 2) - extra_left)
    x2 = min(frame_w, x2 + math.ceil(grow_x / 2) + extra_right)
    y1 = max(0, y1 - math.floor(grow_y / 2) - extra_top)
    y2 = min(frame_h, y2 + math.ceil(grow_y / 2) + extra_bottom)
    return int(x1), int(y1), int(x2), int(y2)


def stable_crop(
    boxes: list[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """One even rectangle covering every restored box, plus a small margin."""
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    pad = max(32, int(round(0.06 * max(x2 - x1, y2 - y1))))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame_w, x2 + pad)
    y2 = min(frame_h, y2 + pad)
    x1 -= x1 % 2
    y1 -= y1 % 2
    if x2 % 2:
        x2 = x2 + 1 if x2 + 1 <= frame_w else x2 - 1
    if y2 % 2:
        y2 = y2 + 1 if y2 + 1 <= frame_h else y2 - 1
    if x2 <= x1 or y2 <= y1:
        return 0, 0, frame_w - frame_w % 2, frame_h - frame_h % 2
    return x1, y1, x2, y2


def letterbox(image, size: int = RESTORATION_SIZE):
    """Fit ``image`` inside a square. Returns the square and the content rectangle."""
    import cv2

    height, width = image.shape[:2]
    scale = size / max(height, width)
    fitted_w = max(1, int(round(width * scale)))
    fitted_h = max(1, int(round(height * scale)))
    fitted = cv2.resize(image, (fitted_w, fitted_h), interpolation=cv2.INTER_AREA)
    import numpy as np

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - fitted_w) // 2
    y0 = (size - fitted_h) // 2
    canvas[y0 : y0 + fitted_h, x0 : x0 + fitted_w] = fitted
    return canvas, (x0, y0, fitted_w, fitted_h)


def edge_fade(height: int, width: int, falloff: int):
    import numpy as np

    falloff = max(1, min(falloff, height // 2, width // 2))
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).clip(max=falloff).astype(np.float32) / falloff
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).clip(max=falloff).astype(np.float32) / falloff
    y = y * y * (3 - 2 * y)
    x = x * x * (3 - 2 * x)
    return (y[:, None] * x[None, :])[:, :, None]
