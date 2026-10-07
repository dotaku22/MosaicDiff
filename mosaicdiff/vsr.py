"""BasicVSR++ on 256-pixel crops.

This always runs the plain checkpoint. MosaicDiff does not compile BasicVSR++.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_restorer(checkpoint: Path, device: torch.device):
    from mosaicdiff.basicvsr.inference import load_model

    return load_model(None, str(checkpoint), device, True)


def restore_squares(model, squares: list[np.ndarray], device: torch.device) -> list[np.ndarray]:
    """Restore RGB 256 squares. Returns RGB uint8 squares in the same order."""
    frames = [torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1) for image in squares]
    with torch.inference_mode():
        stacked = torch.stack(frames).to(device=device, dtype=torch.float16).div_(255.0)
        restored = model(inputs=stacked.unsqueeze(0)).squeeze(0)
    array = (
        restored.clamp(0, 1)
        .mul(255)
        .round()
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [array[index] for index in range(array.shape[0])]
