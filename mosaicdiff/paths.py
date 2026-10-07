"""Where MosaicDiff keeps settings, and where it looks for model files."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR_NAME = "MosaicDiff"

# Names used when someone drops the weights next to the installed program.
BUNDLED_NAMES = {
    "vsr": "basicvsr.pth",
    "detector": "rfdetr.onnx",
    "unet": "unet.safetensors",
    "lora": "lora.safetensors",
    "clip": "clip.safetensors",
    "vae": "vae.safetensors",
}

# ComfyUI stays an install. Weight files are resolved from models\ next to the program.
LOCAL_DEFAULTS = {
    "comfy_python": Path(r"E:\Workspace\ume52\scripts\venv\Scripts\python.exe"),
    "comfy_root": Path(r"E:\Workspace\ume52\ComfyUI"),
}

MODEL_LABELS = {
    "vsr": "BasicVSR++ checkpoint",
    "detector": "Mosaic detector",
    "unet": "Eros Max",
    "lora": "H3 LoRA (place this yourself)",
    "clip": "H3 text encoder",
    "vae": "H3 video VAE",
    "comfy_python": "ComfyUI Python",
    "comfy_root": "ComfyUI folder",
}


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def worker_script() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS")) / "mosaicdiff" / "h3_worker.py"
    return Path(__file__).resolve().parent / "h3_worker.py"


def settings_path() -> Path:
    """The shipped program keeps its own settings beside the exe.

    A shared AppData file would point it at the development folders above Shipped.
    """
    if is_frozen():
        return install_dir() / "settings.json"
    root = Path.home() / "AppData" / "Roaming" / APP_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root / "settings.json"


def models_dir() -> Path:
    return install_dir() / "models"


def nodes_dir() -> Path:
    """Custom Comfy nodes shipped with MosaicDiff, not taken from the Comfy install."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS")) / "comfy_nodes"
    return Path(__file__).resolve().parents[1] / "comfy_nodes"


def default_output_dir() -> Path:
    return install_dir() / "Output"


def bundled_model(key: str) -> Path | None:
    name = BUNDLED_NAMES.get(key)
    if name is None:
        return None
    path = models_dir() / name
    return path if path.is_file() else None
