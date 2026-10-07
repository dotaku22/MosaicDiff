"""Find a ComfyUI install and make sure the RTX upscaler can import."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from mosaicdiff.paths import LOCAL_DEFAULTS
from mosaicdiff.settings import Settings


def _is_comfy_root(path: Path) -> bool:
    return (path / "main.py").is_file() and (path / "comfy").is_dir()


def _python_near(comfy_root: Path) -> Path | None:
    candidates = (
        comfy_root.parent / "python_embeded" / "python.exe",
        comfy_root.parent / "venv" / "Scripts" / "python.exe",
        comfy_root / "venv" / "Scripts" / "python.exe",
        comfy_root.parent / "scripts" / "venv" / "Scripts" / "python.exe",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _search_roots() -> list[Path]:
    home = Path.home()
    seeds = [
        LOCAL_DEFAULTS["comfy_root"],
        home / "ComfyUI",
        home / "Documents" / "ComfyUI",
        home / "Desktop" / "ComfyUI_windows_portable" / "ComfyUI",
        home / "Downloads" / "ComfyUI_windows_portable" / "ComfyUI",
        Path(r"C:\ComfyUI"),
        Path(r"C:\ComfyUI_windows_portable\ComfyUI"),
        Path(r"D:\ComfyUI"),
        Path(r"D:\ComfyUI_windows_portable\ComfyUI"),
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for seed in seeds:
        path = seed if _is_comfy_root(seed) else seed / "ComfyUI"
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not _is_comfy_root(resolved):
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


def comfy_ready(settings: Settings) -> bool:
    root = settings.resolved("comfy_root")
    python = settings.resolved("comfy_python")
    return _is_comfy_root(root) and python.is_file()


def comfy_instructions(settings: Settings) -> str:
    root = settings.resolved("comfy_root")
    python = settings.resolved("comfy_python")
    return (
        "ComfyUI is required for the Eros Max pass, and MosaicDiff could not find it.\n"
        "Install a current ComfyUI. The Windows portable build is fine, and the folder can be anywhere.\n"
        "It needs to be new enough to include the built-in MiniMax H3 nodes.\n"
        "Press Models and set both of these:\n"
        "ComfyUI folder: the folder that contains main.py and the comfy folder.\n"
        "ComfyUI Python: python.exe from that install. In a portable build it is python_embeded\\python.exe next to the ComfyUI folder.\n"
        "MosaicDiff also looks in your user folder, Documents, Desktop, Downloads, and C:\\ or D:\\. "
        "If ComfyUI is in one of those places, press Start again.\n"
        "The extra nodes are already in MosaicDiff. You do not download those.\n"
        f"Looked for the folder at {root}\n"
        f"Looked for Python at {python}"
    )


def discover_comfy(settings: Settings, log) -> None:
    """Fill a missing Comfy folder or Python from a portable layout."""
    root = settings.resolved("comfy_root")
    python = settings.resolved("comfy_python")
    changed = False
    if not _is_comfy_root(root):
        found = _search_roots()
        if found:
            root = found[0]
            settings.paths["comfy_root"] = str(root)
            changed = True
            log(f"ComfyUI folder: {root}")
    if not python.is_file():
        guessed = _python_near(settings.resolved("comfy_root"))
        if guessed is not None:
            settings.paths["comfy_python"] = str(guessed)
            changed = True
            log(f"ComfyUI Python: {guessed}")
    if changed:
        settings.save()


def ensure_rtx_package(settings: Settings, log, cancel: threading.Event) -> None:
    """Install nvidia-vfx into the Comfy environment when the RTX node needs it."""
    python = settings.resolved("comfy_python")
    if not python.is_file():
        return
    if cancel.is_set():
        return
    probe = subprocess.run(
        [str(python), "-c", "import nvvfx"],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return
    log("Installing the NVIDIA video effects package into ComfyUI")
    if cancel.is_set():
        return
    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "nvidia-vfx"],
        capture_output=True,
        text=True,
    )
    if installed.returncode != 0:
        detail = (installed.stderr or installed.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "pip failed"
        raise RuntimeError(f"Could not install nvidia-vfx for the RTX upscaler: {tail}")
    log("NVIDIA video effects package installed")
