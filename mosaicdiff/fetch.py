"""Download public weights into models\\ when they are not already there.

BasicVSR++ is the plain checkpoint. MosaicDiff does not build a TensorRT
engine for it. The detector is the RF-DETR ONNX model and also runs as-is.
The Eros Max file is the H3 diffusion model. The LoRA is not downloaded;
that file has to be placed by hand.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from mosaicdiff.paths import BUNDLED_NAMES, MODEL_LABELS, models_dir
from mosaicdiff.settings import Settings

# repo, filename inside the repo, optional git revision
_REMOTE = {
    "vsr": (
        "ladaapp/lada",
        "lada_mosaic_restoration_model_generic_v1.2.pth",
        "3bfd69ffc21518bde80ba6b61696d51efd0a398b",
    ),
    "unet": (
        "TenStrip/10Eros-Max",
        "10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors",
        None,
    ),
    "clip": (
        "Comfy-Org/MiniMax-H3",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        None,
    ),
    "vae": (
        "Comfy-Org/MiniMax-H3",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        None,
    ),
}


class FetchError(RuntimeError):
    pass


_TQDM_LOCK = threading.RLock()
_LOG_EVERY_SECONDS = 3.0


def _size(num: float) -> str:
    value = float(max(0, num))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _eta(seconds: float) -> str:
    remaining = max(0, int(seconds))
    if remaining < 60:
        return f"{remaining}s"
    minutes, _seconds = divmod(remaining, 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


@dataclass
class _Progress:
    log: callable
    label: str
    progress: callable | None = None
    announced: bool = False
    seen: set = field(default_factory=set)

    def tqdm(self):
        log = self.log
        label = self.label
        state = self

        class Bar:
            def __init__(self, *args, total=None, initial=0, desc=None, **kwargs) -> None:
                self.total = None if total is None else int(total)
                self.n = int(initial or 0)
                self.desc = "" if desc is None else str(desc)
                self.started = time.monotonic()
                self.last_log = 0.0
                self.last_bar = 0.0
                self.last_n = self.n
                self.last_time = self.started
                if self.total and not state.announced:
                    state.announced = True
                    log(f"{label}: {_size(self.total)} to download")
                if self.n:
                    self._report(force=True)

            def update(self, n=1) -> None:
                self.n += 0 if n is None else int(n)
                if self.n < 0:
                    self.n = 0
                done = self.total is not None and self.n >= self.total
                if done or time.monotonic() - self.last_log >= _LOG_EVERY_SECONDS:
                    self._report(force=True)
                else:
                    self._move_bar()

            def _phase(self) -> str:
                text = self.desc.lower()
                if "download" in text:
                    return "received"
                if "reconstruct" in text:
                    return "saved"
                return ""

            def _move_bar(self) -> None:
                if state.progress is None or not self.total or self._phase() == "received":
                    return
                now = time.monotonic()
                if now - self.last_bar < 0.25 and self.n < self.total:
                    return
                self.last_bar = now
                state.progress(min(1.0, self.n / self.total))

            def _report(self, force: bool = False) -> None:
                now = time.monotonic()
                elapsed = max(now - self.last_time, 0.001)
                gained = self.n - self.last_n
                speed = gained / elapsed if gained > 0 else 0.0
                self.last_log = now
                self.last_n = self.n
                self.last_time = now
                self._move_bar()
                phase = self._phase()
                speed_text = f"{_size(speed)}/s" if speed else "starting"
                if self.total:
                    percent = 100.0 * self.n / self.total
                    eta = f", about {_eta((self.total - self.n) / speed)} left" if speed and self.n < self.total else ""
                    prefix = f"{phase} " if phase else ""
                    log(f"{label}: {prefix}{_size(self.n)} / {_size(self.total)} ({percent:.1f}%) at {speed_text}{eta}")
                elif self.n:
                    prefix = f"{phase} " if phase else ""
                    log(f"{label}: {prefix}{_size(self.n)} at {speed_text}")

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                self.close()

            def close(self) -> None:
                if self.n and id(self) not in state.seen:
                    state.seen.add(id(self))
                    self._report(force=True)

            @classmethod
            def get_lock(cls):
                return _TQDM_LOCK

            @classmethod
            def set_lock(cls, lock) -> None:
                pass

            def refresh(self, *args, **kwargs) -> None:
                pass

            def set_description(self, *args, **kwargs) -> None:
                pass

            def set_postfix_str(self, *args, **kwargs) -> None:
                pass

        return Bar


def _download(key: str, destination: Path, log, cancel: threading.Event, progress=None) -> None:
    repo, filename, revision = _REMOTE[key]
    label = MODEL_LABELS[key]
    if cancel.is_set():
        raise FetchError("Stopped before the download finished.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = destination.parent / "_incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {label} from {repo}")
    from huggingface_hub import hf_hub_download

    saved = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            local_dir=str(incoming),
            tqdm_class=_Progress(log, label, progress).tqdm(),
        )
    )
    if cancel.is_set():
        raise FetchError("Stopped before the download finished.")
    if saved.resolve() != destination.resolve():
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.move(str(saved), temporary)
        temporary.replace(destination)
    log(f"Saved {destination.name}")


def _use_detector_model(settings: Settings, log) -> None:
    """Point at the ONNX file. An old TensorRT engine path is not the model."""
    current = settings.resolved("detector")
    if current.suffix.lower() == ".onnx" and current.is_file():
        return
    destination = models_dir() / BUNDLED_NAMES["detector"]
    if not destination.is_file():
        raise FetchError(
            f"Place the mosaic detector at {destination}. "
            "That file is rfdetr-v6.onnx. MosaicDiff runs it directly, so no TensorRT engine is required."
        )
    settings.paths["detector"] = str(destination)
    settings.save()
    log(f"Using detector model {destination.name}")


def ensure_weights(settings: Settings, log, cancel: threading.Event, progress=None) -> None:
    """Fill any missing public weights. Leave the LoRA for the user."""
    changed = False
    for key in _REMOTE:
        current = settings.resolved(key)
        if current.is_file():
            continue
        destination = models_dir() / BUNDLED_NAMES[key]
        if not destination.is_file():
            _download(key, destination, log, cancel, progress)
        settings.paths[key] = str(destination)
        changed = True
    if changed:
        settings.save()

    lora = settings.resolved("lora")
    if not lora.is_file():
        expected = models_dir() / BUNDLED_NAMES["lora"]
        raise FetchError(f"Place your LoRA at {expected}")

    _use_detector_model(settings, log)

    missing = settings.missing()
    if missing:
        lines = "\n".join(f"{label}: {path}" for label, path in missing)
        raise FetchError(f"These files are still missing:\n{lines}")
