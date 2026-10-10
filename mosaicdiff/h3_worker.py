"""Run the tested MiniMax H3 ref2va graph. Executed with ComfyUI's Python, not Jasna's.

One process loads the UNET, the Decensore LoRA, the Qwen3-VL text encoder, and
the video VAE, then restores every region in the job file. Sampling matches
``VSR_restore.json``: euler, simple scheduler, 8 steps, denoise 1, BasicGuider
(no CFG), ``ref_image_size`` match, and the Decensore LoRA at strength 1.
The lightning LoRA in that graph is switched off. MiniMax H3 Context Windows
splits a long region into overlapping pieces during that one sample.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def _boot() -> None:
    import logging

    import comfy.options

    comfy.options.args_parsing = False
    import comfy.cli_args as cli_args

    # ComfyUI calls exit(-1) while importing comfy.ldm.modules.attention when
    # this flag is set and the sageattention package is missing.
    cli_args.args.use_sage_attention = importlib.util.find_spec("sageattention") is not None
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    # The allocator is chosen on the first torch import. ComfyUI sets this
    # before it loads any model.
    if os.name == "nt" and cli_args.args.cuda_device is None and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"
    import cuda_malloc  # noqa: F401

    # Comfy Kitchen's INT8 attention stands in for sage attention when the
    # sageattention package is absent. Checked after cuda_malloc so the torch
    # import it triggers sees the allocator choice. MOSAICDIFF_CK_ATTENTION=0
    # keeps PyTorch attention for comparison.
    if not cli_args.args.use_sage_attention and os.environ.get("MOSAICDIFF_CK_ATTENTION", "1") != "0":
        cli_args.args.use_ck_attention = _ck_attention_available()

    if not cli_args.enables_dynamic_vram():
        return
    import comfy_aimdo.control

    headroom = None if cli_args.args.reserve_vram is None else int(cli_args.args.reserve_vram * 1024**3)
    try:
        comfy_aimdo.control.init(
            simple_vram_headroom=headroom,
            nvml_pressure=not getattr(cli_args.args, "disable_nvml_pressure", False),
        )
    except TypeError:
        try:
            comfy_aimdo.control.init(simple_vram_headroom=headroom)
        except TypeError:
            comfy_aimdo.control.init()


def _ck_attention_available() -> bool:
    """True when comfy_kitchen ships its INT8 attention kernel for this card.

    Older comfy-kitchen (0.2.26, the one ComfyUI 0.30 pins) has no
    int8_attention_is_available, and that ComfyUI has no --use-ck-attention
    either, so the flag stays off there and PyTorch attention is used.
    """
    try:
        import comfy_kitchen

        return bool(comfy_kitchen.int8_attention_is_available())
    except (ImportError, AttributeError):
        return False


def _enable_dynamic_vram() -> None:
    """Use the same weight loader as the ComfyUI window.

    The text encoder is 16 GB and the UNET is 21 GB. The legacy loader keeps
    those weights in system memory and copies one layer onto the card per
    operation, which is why conditioning and the first sample step sit for
    minutes. ComfyUI's dynamic loader keeps the quantized weights on the card.
    """
    import comfy.cli_args as cli_args
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    import comfy_aimdo.control

    args = cli_args.args
    supported = comfy.model_management.is_nvidia() or (
        comfy.model_management.is_amd() and comfy.model_management.rocm_version >= (7, 14)
    )
    if not (args.enable_dynamic_vram or (cli_args.enables_dynamic_vram() and supported)):
        print("Dynamic VRAM off", flush=True)
        return
    if (not args.enable_dynamic_vram) and comfy.model_management.torch_version_numeric < (2, 8):
        print("Dynamic VRAM needs PyTorch 2.8 or newer", flush=True)
        return
    devices = list(comfy.model_management.get_all_torch_devices())
    extra = int(args.vram_headroom * 1024**3)
    try:
        ready = comfy_aimdo.control.init_devices((device.index, extra) for device in devices)
    except TypeError:
        ready = comfy_aimdo.control.init_devices(device.index for device in devices)
    if not ready:
        print("Dynamic VRAM failed to start", flush=True)
        return
    try:
        comfy_aimdo.control.set_log_info()
    except AttributeError:
        pass
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    print("Dynamic VRAM enabled", flush=True)


def _load(job: dict):
    _enable_dynamic_vram()
    import torch

    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        BasicScheduler,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    from nodes import VAEDecode
    from PIL import Image

    print("Loading MiniMax H3 text encoder", flush=True)
    clip = comfy.sd.load_clip(
        ckpt_paths=[job["clip"]],
        embedding_directory=None,
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    print("Loading MiniMax H3 video VAE", flush=True)
    vae_sd, vae_metadata = comfy.utils.load_torch_file(job["video_vae"], return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_metadata)
    vae.throw_exception_if_invalid()
    print("Loading MiniMax H3 UNET", flush=True)
    model = comfy.sd.load_diffusion_model(job["unet"], model_options={})
    print("Loading Decensore LoRA", flush=True)
    lora, lora_metadata = comfy.utils.load_torch_file(job["lora"], safe_load=True, return_metadata=True)
    model, _clip = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0, lora_metadata=lora_metadata)
    print(
        f"Loaders: text {type(clip.patcher).__name__}, UNET {type(model).__name__}",
        flush=True,
    )
    _watch_progress()
    _watch_vae(vae)
    _watch_text(clip)
    return {
        "torch": torch,
        "Image": Image,
        "clip": clip,
        "vae": vae,
        "base_model": model,
        "decode": VAEDecode(),
        "reference": MiniMaxH3ReferenceToVideo,
        "guider": BasicGuider,
        "scheduler": BasicScheduler,
        "sampler": KSamplerSelect,
        "noise": RandomNoise,
        "sample": SamplerCustomAdvanced,
    }


def _bundled_node(job: dict, folder: str, filename: str) -> Path:
    nodes = str(job.get("nodes_dir") or "").strip()
    if nodes:
        path = Path(nodes) / folder / filename
        if path.is_file():
            return path
    return Path(job["comfy_root"]) / "custom_nodes" / folder / filename


def _apply_context(model, job: dict):
    path = _bundled_node(job, "ComfyUI-H3-ContextWindows", "nodes.py")
    if not path.is_file():
        raise RuntimeError(f"MiniMax H3 Context Windows node not found: {path}")
    spec = importlib.util.spec_from_file_location("mosaicdiff_h3_context_windows", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MiniMax H3 Context Windows node could not be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    length = int(job.get("context_frames", 124))
    overlap = int(job.get("context_overlap", 22))
    return module.MiniMaxH3ContextWindows.execute(model, length, overlap, "pyramid")[0]


def _gpu_line() -> str:
    import torch

    if not torch.cuda.is_available():
        return "GPU n/a"
    free, total = torch.cuda.mem_get_info()
    return f"{(total - free) / 1024 ** 3:.1f} GiB used, {free / 1024 ** 3:.1f} GiB free"


def _free_other_models() -> None:
    """Give the VAE the card. Comfy unloads the diffusion model and text encoder
    before a VAE node; leaving them resident makes this VAE miss its normal
    encode and fall back to the slow tiled path.
    """
    import comfy.model_management as model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache()


def _watch_progress() -> None:
    import comfy.utils

    last = {"step": None}

    def hook(current, total, preview=None, node_id=None):
        step = int(current)
        if step == last["step"]:
            return
        last["step"] = step
        print(f"Sampling step {step}/{int(total)}", flush=True)

    comfy.utils.set_progress_bar_global_hook(hook)


def _watch_vae(vae) -> None:
    encode = vae.encode
    decode = vae.decode

    def encode_logged(pixel_samples, *args, **kwargs):
        print(f"VAE encode {tuple(pixel_samples.shape)} ({_gpu_line()})", flush=True)
        _free_other_models()
        print(f"VAE encode ready ({_gpu_line()})", flush=True)
        started = time.perf_counter()
        out = encode(pixel_samples, *args, **kwargs)
        print(f"VAE encode finished in {time.perf_counter() - started:.1f}s", flush=True)
        return out

    def decode_logged(samples, *args, **kwargs):
        print(f"VAE decode {tuple(samples.shape)} ({_gpu_line()})", flush=True)
        _free_other_models()
        print(f"VAE decode ready ({_gpu_line()})", flush=True)
        started = time.perf_counter()
        out = decode(samples, *args, **kwargs)
        print(f"VAE decode finished in {time.perf_counter() - started:.1f}s", flush=True)
        return out

    vae.encode = encode_logged
    vae.decode = decode_logged


def _watch_text(clip) -> None:
    encode = clip.encode_from_tokens_scheduled

    def encode_logged(*args, **kwargs):
        print(f"Text encoder ({_gpu_line()})", flush=True)
        _free_other_models()
        print(f"Text encoder ready ({_gpu_line()})", flush=True)
        started = time.perf_counter()
        out = encode(*args, **kwargs)
        print(f"Text encoder finished in {time.perf_counter() - started:.1f}s", flush=True)
        return out

    clip.encode_from_tokens_scheduled = encode_logged


def _read_frames(torch, image_cls, paths: list[str]):
    import numpy as np

    frames = []
    for path in paths:
        image = image_cls.open(path).convert("RGB")
        frames.append(torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).float().div_(255.0))
    return torch.stack(frames, dim=0)


def _pin_reference_border(loaded, latent, frames, width: int, height: int, mask_path: str | None) -> None:
    """Encode the reference into the sample and stop the sampler editing its border.

    MiniMax rewrites the whole frame, including the edges, and that rewrite runs
    late. A noise mask of 0 keeps those latent pixels on the reference frame.
    """
    torch = loaded["torch"]
    import comfy.nested_tensor
    import comfy.utils

    video, audio = latent["samples"].unbind()
    batch = comfy.utils.common_upscale(frames.movedim(-1, 1), width, height, "lanczos", "disabled")
    encoded = loaded["vae"].encode(batch.movedim(1, -1))
    if tuple(encoded.shape) != tuple(video.shape):
        raise RuntimeError(
            f"Reference latent is {tuple(encoded.shape)}, generation latent is {tuple(video.shape)}"
        )
    encoded = encoded.to(device=video.device, dtype=video.dtype)
    lat_h, lat_w = encoded.shape[-2], encoded.shape[-1]
    mask = _denoise_mask(torch, mask_path, encoded.shape[2], lat_h, lat_w)
    audio_mask = torch.zeros((1, 1, audio.shape[-2], audio.shape[-1]), dtype=torch.float32)
    latent["samples"] = comfy.nested_tensor.NestedTensor((encoded, audio))
    latent["noise_mask"] = comfy.nested_tensor.NestedTensor((mask, audio_mask))
    held = float((mask[0, 0, 0] <= 0).float().mean())
    print(f"Noise mask holds {held:.0%} of each frame, including the border", flush=True)


def _denoise_mask(torch, mask_path: str | None, frames: int, height: int, width: int):
    import cv2

    if mask_path:
        image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = None
    if image is None:
        border = max(2, min(height, width) // 8)
        mask = torch.ones((1, 1, frames, height, width), dtype=torch.float32)
        mask[:, :, :, :border, :] = 0
        mask[:, :, :, -border:, :] = 0
        mask[:, :, :, :, :border] = 0
        mask[:, :, :, :, -border:] = 0
        return mask
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    plane = torch.from_numpy(small.astype("float32") / 255.0).view(1, 1, 1, height, width)
    return plane.expand(1, 1, frames, height, width).contiguous()


def _rtx_node(job: dict):
    path = _bundled_node(job, "comfyui_nvidia_rtx_nodes", "__init__.py")
    if not path.is_file():
        raise RuntimeError(f"RTX Video Super Resolution node not found: {path}")
    spec = importlib.util.spec_from_file_location("mosaicdiff_rtx_video_superres", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"RTX Video Super Resolution node could not be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rtx_fit(job: dict, images, gen_w: int, gen_h: int, crop_w: int, crop_h: int):
    """Upscale the H3 frames with RTX Video Super Resolution, then fit the crop."""
    import comfy.utils

    cover = max(crop_w / gen_w, crop_h / gen_h, 2.0)
    scale = min(4.0, cover)
    print(
        f"RTX upscale {gen_w}x{gen_h} by {scale:.2f}, then fit {crop_w}x{crop_h}",
        flush=True,
    )
    _free_other_models()
    module = _rtx_node(job)
    upscaled = module.RTXVideoSuperResolution.execute(
        images,
        {"resize_type": module.UpscaleType.SCALE_BY, "scale": scale, "width": 0, "height": 0},
        "ULTRA",
    )[0]
    fitted = comfy.utils.common_upscale(
        upscaled.movedim(-1, 1),
        crop_w,
        crop_h,
        "lanczos",
        "disabled",
    ).movedim(1, -1)
    del upscaled
    return fitted


def _read_crop(frame, crop: list[int]):
    x1, y1, x2, y2 = (int(value) for value in crop)
    image = frame[y1:y2, x1:x2]
    if image.size == 0:
        raise RuntimeError(f"Empty H3 crop {x1},{y1}-{x2},{y2}")
    return image.copy()


def _stream_windows(loaded, job: dict) -> None:
    """Read the VSR video once, crop each sample, and restore it before reading on.

    Crops are taken here so the full frames are not written out as images.
    """
    import cv2

    windows = list(job["windows"])
    pending = [
        {"window": window, "need": set(int(frame) for frame in window["frame_indices"]), "got": {}}
        for window in windows
    ]
    capture = cv2.VideoCapture(job["video"])
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {job['video']}")
    frame_idx = 0
    try:
        while pending:
            ok, frame = capture.read()
            if not ok:
                break
            ready = []
            for item in pending:
                if frame_idx not in item["need"]:
                    continue
                item["got"][frame_idx] = _read_crop(frame, item["window"]["crop"])
                if len(item["got"]) == len(item["need"]):
                    ready.append(item)
            for item in ready:
                number = windows.index(item["window"]) + 1
                print(f"Sample {number}/{len(windows)}", flush=True)
                _restore_window(loaded, job, item["window"], item["got"])
                item["got"].clear()
                pending.remove(item)
            frame_idx += 1
    finally:
        capture.release()
    if pending:
        missing = sorted(pending[0]["need"] - set(pending[0]["got"]))[:8]
        raise RuntimeError(f"VSR video ended before H3 frame(s) {missing}")


def _restore_window(loaded, job: dict, window: dict, crops: dict | None = None) -> None:
    torch = loaded["torch"]
    started = time.perf_counter()
    if crops is None:
        print(f"Reading {len(window['frames'])} frames", flush=True)
        frames = _read_frames(torch, loaded["Image"], window["frames"])
    else:
        import numpy as np

        print(f"Reading {len(window['frame_indices'])} frames", flush=True)
        ordered = []
        for frame_idx in window["frame_indices"]:
            bgr = crops[int(frame_idx)]
            rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).float().div_(255.0)
            ordered.append(rgb)
        frames = torch.stack(ordered, dim=0)
    length = int(frames.shape[0])
    width = int(window["width"])
    height = int(window["height"])
    loaded["model"] = _apply_context(
        loaded["base_model"],
        {
            "comfy_root": job["comfy_root"],
            "nodes_dir": job.get("nodes_dir"),
            "context_frames": int(window.get("context_frames", job.get("context_frames", 124))),
            "context_overlap": int(window.get("context_overlap", job.get("context_overlap", 22))),
        },
    )
    print(
        f"Conditioning {length} frames, generation {width}x{height}, reference {frames.shape[2]}x{frames.shape[1]}",
        flush=True,
    )
    conditioned = loaded["reference"].execute(
        loaded["clip"],
        job["prompt"],
        width,
        height,
        length,
        "match",
        vae=loaded["vae"],
        audio_vae=None,
        ref_videos={"ref_video_0": frames},
    )
    positive, latent = conditioned[0], conditioned[1]
    _pin_reference_border(loaded, latent, frames, width, height, window.get("mask"))
    print(f"Sampling {int(job['steps'])} steps", flush=True)
    guider = loaded["guider"].execute(loaded["model"], positive)[0]
    sigmas = loaded["scheduler"].execute(loaded["model"], "simple", int(job["steps"]), 1.0)[0]
    sampler = loaded["sampler"].execute("euler")[0]
    noise = loaded["noise"].execute(int(job["seed"]))[0]
    sampled = loaded["sample"].execute(noise, guider, sampler, sigmas, latent)[0]
    images = loaded["decode"].decode(loaded["vae"], sampled)[0]
    if images.shape[0] != length:
        raise RuntimeError(f"H3 decoded {images.shape[0]} frames, expected {length}")
    crop = window.get("crop")
    if crop:
        crop_w = int(crop[2]) - int(crop[0])
        crop_h = int(crop[3]) - int(crop[1])
    else:
        crop_w, crop_h = width, height
    images = _rtx_fit(job, images, width, height, crop_w, crop_h)
    out_dir = Path(window["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {length} frames", flush=True)
    for index in range(length):
        array = images[index].clamp(0, 1).mul(255).round().to(dtype=torch.uint8).cpu().numpy()
        loaded["Image"].fromarray(array).save(out_dir / f"{index:06d}.png")
    print(f"Window finished in {time.perf_counter() - started:.1f}s", flush=True)
    del frames, positive, latent, sampled, images
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    os.chdir(job["comfy_root"])
    sys.path.insert(0, job["comfy_root"])
    _boot()
    print(f"Comfy process started ({_gpu_line()})", flush=True)
    loaded = _load(job)
    windows = job["windows"]
    # Gradient tracking is off, same as a Comfy node, so the VAE and text encoder
    # do not keep activations. inference_mode is stronger than that and marks the
    # weights so Comfy cannot move the quantized UNET onto the GPU.
    with loaded["torch"].no_grad():
        if job.get("video"):
            _stream_windows(loaded, job)
        else:
            for index, window in enumerate(windows, start=1):
                print(f"Sample {index}/{len(windows)}", flush=True)
                _restore_window(loaded, job, window)
    print("MiniMax H3 pass finished", flush=True)


if __name__ == "__main__":
    main()
