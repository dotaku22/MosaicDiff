"""BasicVSR++ on the mosaic, then MiniMax H3 on one locked crop."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from mosaicdiff.geometry import (
    H3_FPS,
    edge_fade,
    expand_box,
    fit_context,
    frames_at_h3_fps,
    generation_size,
    letterbox,
    split_samples,
    stable_crop,
)
from mosaicdiff.paths import nodes_dir, worker_script
from mosaicdiff.settings import Settings
from mosaicdiff.videoio import VideoWriter, copy_audio, open_capture
from mosaicdiff.vsr import restore_squares

PROMPT = "Restore the vulva or penis the reference video"
CHUNK = 90
OVERLAP = 4


class Cancelled(Exception):
    pass


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    number = 1
    while True:
        candidate = path.with_name(f"{stem} ({number}){suffix}")
        if not candidate.exists():
            return candidate
        number += 1


def process_video(
    source: Path,
    destination: Path,
    settings: Settings,
    log,
    progress,
    cancel,
) -> Path:
    missing = settings.missing()
    if missing:
        listed = ", ".join(label for label, _path in missing)
        raise FileNotFoundError(f"Missing model files: {listed}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = unique_path(destination)
    import torch

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("MosaicDiff needs an NVIDIA GPU")

    from mosaicdiff.detect import Detector
    from mosaicdiff.vsr import load_restorer

    log(f"Detecting mosaic in {source.name}")
    detector = Detector(settings.resolved("detector"), device)
    log(f"Detector model on {detector.provider}")
    try:
        boxes, width, height, fps, _count = _detect(source, detector, log, progress, cancel)
    finally:
        detector.close()
        torch.cuda.empty_cache()

    present = [box for box in boxes if box is not None]
    if not present:
        raise RuntimeError(f"No mosaic found in {source.name}")

    grown = []
    for box in boxes:
        if box is None:
            grown.append(None)
        else:
            grown.append(expand_box(*box, height, width))

    log(f"BasicVSR++ on {len(present)} frames")
    restorer = load_restorer(settings.resolved("vsr"), device)
    vsr_path = destination.with_name(destination.stem + ".vsr" + destination.suffix)
    try:
        _restore_video(source, vsr_path, grown, restorer, device, fps, progress, cancel)
    finally:
        del restorer
        torch.cuda.empty_cache()

    crop = stable_crop([box for box in grown if box is not None], width, height)
    log(f"H3 crop {crop[2] - crop[0]}x{crop[3] - crop[1]} at {crop[0]},{crop[1]}")
    try:
        _h3(source, vsr_path, destination, settings, crop, grown, fps, log, progress, cancel)
    except Cancelled:
        vsr_path.unlink(missing_ok=True)
        raise
    except Exception:
        log(f"H3 stopped. The BasicVSR++ video is still at {vsr_path.name}")
        raise
    vsr_path.unlink(missing_ok=True)
    progress(1.0, "Done")
    return destination


def _check(cancel) -> None:
    if cancel.is_set():
        raise Cancelled()


def _detect(source, detector, log, progress, cancel):
    capture, width, height, fps, count = open_capture(source)
    boxes = []
    batch = []
    seen = 0
    try:
        while True:
            _check(cancel)
            ok, frame = capture.read()
            if not ok:
                break
            batch.append(frame)
            seen += 1
            if len(batch) == detector.batch:
                boxes.extend(detector.best_boxes(batch))
                batch = []
                if count:
                    progress(0.25 * min(1.0, seen / count), "Detecting")
        if batch:
            boxes.extend(detector.best_boxes(batch))
    finally:
        capture.release()
    log(f"{sum(box is not None for box in boxes)} of {len(boxes)} frames have a mosaic")
    return boxes, width, height, fps, len(boxes)


def _restore_video(source, vsr_path, boxes, model, device, fps, progress, cancel) -> None:
    capture, width, height, _fps, count = open_capture(source)
    writer = VideoWriter(vsr_path, width, height, Fraction(fps).limit_denominator(1000))
    hold: list[tuple[np.ndarray, tuple[int, int, int, int] | None]] = []
    frame_index = 0
    skip_head = 0
    try:
        while True:
            _check(cancel)
            ok, frame = capture.read()
            if not ok:
                break
            hold.append((frame, boxes[frame_index] if frame_index < len(boxes) else None))
            frame_index += 1
            if len(hold) >= CHUNK:
                _flush_chunk(hold, model, device, writer, keep_tail=OVERLAP, skip_head=skip_head)
                skip_head = OVERLAP
                if count:
                    progress(0.25 + 0.30 * min(1.0, frame_index / count), "BasicVSR++")
        if hold:
            _check(cancel)
            _flush_chunk(hold, model, device, writer, keep_tail=0, skip_head=skip_head)
    finally:
        capture.release()
        writer.close()


def _flush_chunk(hold, model, device, writer, keep_tail: int, skip_head: int) -> None:
    if not hold:
        return
    indexed = [(index, frame, box) for index, (frame, box) in enumerate(hold) if box is not None]
    restored: dict[int, np.ndarray] = {}
    if indexed:
        squares = []
        metas = []
        for _index, frame, box in indexed:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            square, meta = letterbox(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            squares.append(square)
            metas.append((_index, box, meta))
        if squares:
            output = restore_squares(model, squares, device)
            for (index, box, meta), square in zip(metas, output):
                restored[index] = _unletterbox(square, box, meta)
    end = len(hold) if keep_tail <= 0 else max(skip_head, len(hold) - keep_tail)
    for index in range(skip_head, end):
        frame, box = hold[index]
        patch = restored.get(index)
        if patch is not None and box is not None:
            frame = _paste(frame, patch, box)
        writer.write(frame)
    del hold[:end]


def _unletterbox(square_rgb: np.ndarray, box, meta) -> np.ndarray:
    x0, y0, fitted_w, fitted_h = meta
    content = square_rgb[y0 : y0 + fitted_h, x0 : x0 + fitted_w]
    bgr = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = box
    return cv2.resize(bgr, (max(1, x2 - x1), max(1, y2 - y1)), interpolation=cv2.INTER_LANCZOS4)


def _paste(frame: np.ndarray, patch: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    fitted = patch
    if fitted.shape[1] != x2 - x1 or fitted.shape[0] != y2 - y1:
        fitted = cv2.resize(fitted, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    short = min(y2 - y1, x2 - x1)
    falloff = max(24, min(round(short * 0.06), short // 6))
    fade = edge_fade(y2 - y1, x2 - x1, falloff)
    base = frame.copy()
    region = base[y1:y2, x1:x2].astype(np.float32)
    mixed = region * (1.0 - fade) + fitted.astype(np.float32) * fade
    base[y1:y2, x1:x2] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return base


def _h3(source, vsr_path, destination, settings: Settings, crop, boxes, fps, log, progress, cancel) -> None:
    present = [index for index, box in enumerate(boxes) if box is not None]
    first, last = present[0], present[-1]
    span = frames_at_h3_fps(last - first + 1, fps)
    sampled = [first + index for index in span]
    windows = split_samples(sampled)
    if not windows:
        raise RuntimeError("The restored section is shorter than 5 frames at 24 fps")
    gen_w, gen_h = generation_size(crop[2] - crop[0], crop[3] - crop[1], settings.h3_resolution)
    log(f"MiniMax H3, {len(windows)} sample(s), {gen_w}x{gen_h}")
    progress(0.58, "MiniMax H3")
    with tempfile.TemporaryDirectory(prefix="mosaicdiff-", dir=str(destination.parent)) as temp_name:
        temp = Path(temp_name)
        specs = []
        for number, indices in enumerate(windows):
            _check(cancel)
            context, overlap = fit_context(settings.h3_seconds, len(indices))
            out_dir = temp / f"window_{number:03d}"
            out_dir.mkdir()
            specs.append(
                {
                    "frame_indices": indices,
                    "width": gen_w,
                    "height": gen_h,
                    "crop": list(crop),
                    "out_dir": str(out_dir),
                    "context_frames": context,
                    "context_overlap": overlap,
                }
            )
            log(f"Sample {number + 1}: {len(indices)} frames, context {context}")
        job = {
            "comfy_root": str(settings.resolved("comfy_root")),
            "nodes_dir": str(nodes_dir()),
            "unet": str(settings.resolved("unet")),
            "lora": str(settings.resolved("lora")),
            "clip": str(settings.resolved("clip")),
            "video_vae": str(settings.resolved("vae")),
            "prompt": PROMPT,
            "seed": 0,
            "steps": 8,
            "video": str(vsr_path),
            "windows": specs,
        }
        job_path = temp / "job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        _run_comfy(settings.resolved("comfy_python"), job_path, log, cancel)
        progress(0.9, "Writing")
        pastes = {}
        for spec in specs:
            out_dir = Path(spec["out_dir"])
            for order, frame_idx in enumerate(spec["frame_indices"]):
                image = out_dir / f"{order:06d}.png"
                if image.is_file():
                    pastes[int(frame_idx)] = image
        output_fps = H3_FPS if fps > H3_FPS + 0.05 else fps
        _write_output(vsr_path, destination, pastes, crop, output_fps, fps, len(boxes))
        if settings.compare:
            compare_path = destination.with_name(destination.stem + "_compare" + destination.suffix)
            try:
                _write_compare(source, destination, compare_path, fps)
                log(f"Comparison {compare_path.name}")
            except Exception as exc:
                compare_path.unlink(missing_ok=True)
                log(f"Comparison was not written: {exc}")
    with_audio = destination.with_name(destination.stem + ".audio" + destination.suffix)
    try:
        if copy_audio(destination, source, with_audio):
            with_audio.replace(destination)
    except Exception as exc:
        with_audio.unlink(missing_ok=True)
        log(f"Audio was not copied: {exc}")
    log(f"Wrote {destination.name}")


def _run_comfy(python: Path, job_path: Path, log, cancel) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        [str(python), "-s", str(worker_script()), "--job", str(job_path)],
        cwd=str(job_path.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            _check(cancel)
            text = line.rstrip()
            if text and "%|" not in text:
                log(text)
    except Cancelled:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"MiniMax H3 process exited with status {code}")


def _write_output(vsr_path, destination, pastes, crop, output_fps, source_fps, frame_count: int) -> None:
    capture, width, height, _fps, _count = open_capture(vsr_path)
    keep = set(frames_at_h3_fps(frame_count, source_fps)) if output_fps == H3_FPS else None
    writer = VideoWriter(destination, width, height, Fraction(output_fps).limit_denominator(1000))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if keep is not None and index not in keep:
                index += 1
                continue
            image_path = pastes.get(index)
            if image_path is not None:
                patch = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if patch is not None:
                    frame = _paste(frame, patch, crop)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.close()


def _write_compare(source, restored, destination, source_fps) -> None:
    original, _w, _h, _fps, original_count = open_capture(source)
    result, width, height, _fps, restored_count = open_capture(restored)
    if abs(source_fps - H3_FPS) > 0.05 and original_count != restored_count:
        indices = frames_at_h3_fps(original_count, source_fps)
    else:
        indices = list(range(min(original_count, restored_count)))
    wanted = set(indices)
    writer = VideoWriter(destination, width * 2, height, Fraction(H3_FPS if len(indices) != original_count else source_fps).limit_denominator(1000))
    index = 0
    written = 0
    try:
        while written < len(indices):
            ok, frame = original.read()
            if not ok:
                break
            if index in wanted:
                ok, right = result.read()
                if not ok:
                    break
                left = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                pair = np.concatenate((left, right[:height, :width]), axis=1)
                _label(pair, "with mosaic", 16)
                _label(pair, "without mosaic", width + 16)
                writer.write(pair)
                written += 1
            index += 1
    finally:
        original.release()
        result.release()
        writer.close()


def _label(image: np.ndarray, text: str, x: int) -> None:
    cv2.putText(image, text, (x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
