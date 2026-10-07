"""Saved window settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mosaicdiff.paths import (
    BUNDLED_NAMES,
    LOCAL_DEFAULTS,
    MODEL_LABELS,
    default_output_dir,
    is_frozen,
    models_dir,
    settings_path,
)

PATH_KEYS = (
    "vsr",
    "detector",
    "unet",
    "lora",
    "clip",
    "vae",
    "comfy_python",
    "comfy_root",
)


@dataclass
class Settings:
    output_dir: str = ""
    h3_seconds: int = 5
    h3_resolution: int = 800
    compare: bool = False
    paths: dict[str, str] = field(default_factory=dict)

    def resolved(self, key: str) -> Path:
        # The shipped exe only reads weights from the models folder beside it.
        if is_frozen() and key in BUNDLED_NAMES:
            return models_dir() / BUNDLED_NAMES[key]
        chosen = (self.paths.get(key) or "").strip()
        if chosen:
            return Path(chosen)
        name = BUNDLED_NAMES.get(key)
        if name is not None:
            return models_dir() / name
        return LOCAL_DEFAULTS[key]

    def missing(self) -> list[tuple[str, Path]]:
        gone = []
        for key in PATH_KEYS:
            path = self.resolved(key)
            if not path.exists():
                gone.append((MODEL_LABELS[key], path))
        return gone

    def save(self) -> None:
        payload = {
            "output_dir": self.output_dir,
            "h3_seconds": self.h3_seconds,
            "h3_resolution": self.h3_resolution,
            "compare": self.compare,
            "paths": self.paths,
        }
        settings_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "Settings":
        path = settings_path()
        if not path.is_file():
            settings = cls()
            if is_frozen():
                settings.output_dir = str(default_output_dir())
            return settings
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            settings = cls()
            if is_frozen():
                settings.output_dir = str(default_output_dir())
            return settings
        settings = cls(
            output_dir=str(data.get("output_dir") or ""),
            h3_seconds=int(data.get("h3_seconds") or 5),
            h3_resolution=int(data.get("h3_resolution") or 800),
            compare=False,
            paths={key: str(value) for key, value in dict(data.get("paths") or {}).items()},
        )
        settings.h3_seconds = min(15, max(1, settings.h3_seconds))
        settings.h3_resolution = min(1280, max(512, settings.h3_resolution - settings.h3_resolution % 32))
        if is_frozen():
            settings.output_dir = str(default_output_dir())
            for key in BUNDLED_NAMES:
                settings.paths.pop(key, None)
        return settings
