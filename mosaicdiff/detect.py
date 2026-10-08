"""Mosaic boxes from the RF-DETR ONNX model.

The model runs as-is. MosaicDiff does not compile a TensorRT engine for it,
so the same file works on any graphics card.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_PROVIDER_ORDER = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)


def _cuda_libraries() -> None:
    """The CUDA provider loads cuDNN from the libraries shipped with PyTorch."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    lib = Path(torch.__file__).resolve().parent / "lib"
    if lib.is_dir():
        os.add_dll_directory(str(lib))


def _providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    chosen = [name for name in _PROVIDER_ORDER if name in available]
    return chosen or ["CPUExecutionProvider"]


class Detector:
    def __init__(self, model_path: Path, device: torch.device, score_threshold: float = 0.35) -> None:
        import onnxruntime as ort

        model_path = Path(model_path)
        if model_path.suffix.lower() != ".onnx":
            raise RuntimeError(
                f"The detector needs the ONNX model, not a compiled engine: {model_path}"
            )
        self.device = device
        self.score_threshold = float(score_threshold)
        _cuda_libraries()
        self.session = ort.InferenceSession(str(model_path), providers=_providers())
        self.provider = self.session.get_providers()[0]
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = tuple(model_input.shape)
        batch = shape[0] if shape else None
        if isinstance(batch, int) and batch > 0:
            self.batch = batch
            self._fixed_batch = True
        else:
            self.batch = 4
            self._fixed_batch = False
        resolution = shape[-1] if shape else 576
        self.resolution = int(resolution) if isinstance(resolution, int) and resolution > 0 else 576
        self.output_names = [item.name for item in self.session.get_outputs()]

    def close(self) -> None:
        self.session = None

    def best_boxes(self, frames_bgr: list[np.ndarray]) -> list[tuple[int, int, int, int] | None]:
        if not frames_bgr:
            return []
        height, width = frames_bgr[0].shape[:2]
        batch = self._preprocess(frames_bgr)
        outputs = self._infer(batch)
        boxes, scores = self._decode(outputs, (height, width), len(frames_bgr))
        chosen: list[tuple[int, int, int, int] | None] = []
        for frame_boxes, frame_scores in zip(boxes, scores):
            if len(frame_boxes) == 0:
                chosen.append(None)
                continue
            areas = (frame_boxes[:, 2] - frame_boxes[:, 0]) * (frame_boxes[:, 3] - frame_boxes[:, 1])
            pick = int(np.argmax(areas * frame_scores))
            x1, y1, x2, y2 = frame_boxes[pick]
            chosen.append((int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))))
        return chosen

    def _preprocess(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        import cv2

        images = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
            images.append(resized)
        array = np.stack(images).astype(np.float32) / 255.0
        array = np.transpose(array, (0, 3, 1, 2))
        mean = np.asarray(_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        array = (array - mean) / std
        count = array.shape[0]
        if self._fixed_batch and count < self.batch:
            pad = np.repeat(array[-1:], self.batch - count, axis=0)
            array = np.concatenate((array, pad), axis=0)
        return np.ascontiguousarray(array)

    def _infer(self, batch: np.ndarray) -> dict[str, torch.Tensor]:
        values = self.session.run(self.output_names, {self.input_name: batch})
        return {name: torch.from_numpy(np.asarray(value)) for name, value in zip(self.output_names, values)}

    def _decode(self, outputs, target_hw, count: int):
        boxes_name = next(
            name
            for name in self.output_names
            if outputs[name].ndim == 3 and (outputs[name].shape[-1] == 4 or "box" in name.lower() or name.lower() == "dets")
        )
        masks_name = next(name for name in self.output_names if outputs[name].ndim == 4)
        logits_name = next(name for name in self.output_names if name not in {boxes_name, masks_name})
        pred_boxes = outputs[boxes_name][:count].float()
        pred_logits = outputs[logits_name][:count].float()
        batch, _queries, classes = pred_logits.shape
        prob = pred_logits.sigmoid()
        k = min(16, prob.shape[1] * classes)
        top_values, top_indexes = torch.topk(prob.view(batch, -1), k, dim=1)
        top_boxes = top_indexes // classes
        center_x, center_y, width, height = pred_boxes.unbind(-1)
        corners = torch.stack(
            (center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height),
            dim=-1,
        )
        corners = corners.gather(1, top_boxes.unsqueeze(-1).expand(batch, k, 4))
        frame_h, frame_w = target_hw
        corners = corners * corners.new_tensor((frame_w, frame_h, frame_w, frame_h))
        valid = top_values > self.score_threshold
        boxes_cpu = corners.detach().cpu().numpy()
        valid_cpu = valid.detach().cpu().numpy()
        scores_cpu = top_values.detach().cpu().numpy()
        box_list = [boxes_cpu[index][valid_cpu[index]] for index in range(batch)]
        score_list = [scores_cpu[index][valid_cpu[index]] for index in range(batch)]
        return box_list, score_list
