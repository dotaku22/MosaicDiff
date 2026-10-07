"""Context windows for MiniMax H3.

Same idea as Wan / LTXV Context Windows: patch the model, keep using
SamplerCustomAdvanced. Each step is split into overlapping latent windows
and pyramid-blended, so a long AV latent is never run as one sequence.

H3 audio time is the last axis, not the video time axis, and every window
has to stay on the 17k+5 frame grid. The stock window node does neither.
"""

from __future__ import annotations

import comfy.conds
import comfy.context_windows
from comfy.context_windows import (
    IndexListContextHandler,
    IndexListContextWindow,
    WindowingState,
)
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io

CATEGORY = "model/patch/minimax"


def align_h3_frames(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def frames_to_latent(n: int) -> int:
    """Pixel frames on the 17k+5 grid -> video latent length (5k+2)."""
    n = align_h3_frames(n)
    if n <= 5:
        return 2
    return ((n - 5) // 17) * 5 + 2


def frames_before(latent_index: int) -> int:
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(int(latent_index)))


def _latent_overlap(index: int, vt: int, win0: int, win1: int):
    """Latent frames of a clip that starts at pixel `index` and overlaps [win0, win1).

    The slice start is pulled back onto the 1-4-4-4-4 phase so the cut still
    matches how the clip was encoded.
    """
    if vt <= 0:
        return None
    if index + frames_before(vt) <= win0 or index >= win1:
        return None
    rel0 = win0 - index
    a = 0
    if rel0 > 0:
        while a < vt and frames_before(a + 1) <= rel0:
            a += 1
        a -= a % 5
    b = a
    rel1 = win1 - index
    while b < vt and frames_before(b) < rel1:
        b += 1
    b = min(vt, max(b, a + 1))
    if a >= b or a >= vt:
        return None
    return a, b


def _clip_keyframe(keyframe: dict, win0: int, win1: int):
    """Return a copy of a guide whose latents cover only this window, or None."""
    item = dict(keyframe)
    index = int(item.get("resolved_frame_index", 0))
    latent = item.get("latent")
    audio = item.get("audio_latent")
    vt = int(latent.shape[2]) if latent is not None else 0
    at = int(audio.shape[-1]) if audio is not None else 0

    video_range = _latent_overlap(index, vt, win0, win1) if latent is not None else None
    if video_range is not None:
        a, b = video_range
        item["latent"] = latent[:, :, a:b]
        item["resolved_frame_index"] = index + frames_before(a)
        if audio is not None:
            a0 = max(0, min(at, int(round(FRAME_RESCALE * frames_before(a)))))
            a1 = max(a0, min(at, int(round(FRAME_RESCALE * frames_before(b)))))
            if a1 > a0:
                item["audio_latent"] = audio[..., a0:a1]
            else:
                item.pop("audio_latent", None)
        return item

    if audio is None or at <= 0:
        return None
    lo = FRAME_RESCALE * (win0 - index)
    hi = FRAME_RESCALE * (win1 - index)
    a0 = max(0, min(at, int(lo)))
    a1 = max(a0, min(at, int(round(hi + 1e-6))))
    if a1 <= a0:
        return None
    item.pop("latent", None)
    item["audio_latent"] = audio[..., a0:a1]
    item["resolved_frame_index"] = index + int(round(a0 / FRAME_RESCALE))
    return item


def audio_indices_for_video(video_indices: list[int], video_total: int, audio_total: int) -> list[int]:
    """Audio latent steps that cover the same time as these video latent tokens.

    A contiguous window always gets the same count for the same pixel span.
    This avoids needless shape variation between equal-length windows.
    """
    if audio_total <= 0 or not video_indices:
        return []
    total_frames = max(1, frames_before(video_total))
    start = int(video_indices[0])
    end = int(video_indices[-1]) + 1
    if list(video_indices) != list(range(start, end)):
        return _audio_indices_loose(video_indices, total_frames, audio_total)
    span = frames_before(end) - frames_before(start)
    count = max(1, int(round(span * audio_total / total_frames)))
    count = min(count, audio_total)
    a0 = int(round(frames_before(start) * audio_total / total_frames))
    if a0 + count > audio_total:
        a0 = audio_total - count
    return list(range(max(0, a0), max(0, a0) + count))


def _audio_indices_loose(video_indices: list[int], total_frames: int, audio_total: int) -> list[int]:
    chosen = []
    seen = set()
    for v in video_indices:
        f0 = frames_before(v)
        f1 = frames_before(v + 1)
        a0 = min(int(round(f0 * audio_total / total_frames)), audio_total - 1)
        a1 = min(int(round(f1 * audio_total / total_frames)), audio_total)
        if a1 <= a0:
            a1 = min(audio_total, a0 + 1)
        for a in range(a0, a1):
            if a not in seen:
                seen.add(a)
                chosen.append(a)
    return chosen or [0]


def window_video_reference(ref, start, length, total):
    """Window a timeline-matched V2V reference; preserve independent references."""
    latent = ref.get("latent")
    if ref.get("kind") not in ("video", "video_audio") or latent is None:
        return ref
    if latent.shape[2] != total:
        return ref
    item = dict(ref)
    item["latent"] = latent[:, :, start:start + length].contiguous()
    item["latent_t"] = item["latent"].shape[2]
    audio = ref.get("audio_latent")
    if audio is not None:
        indices = audio_indices_for_video(list(range(start, start + length)), total, audio.shape[-1])
        item["audio_latent"] = audio[..., indices[0]:indices[-1] + 1].contiguous()
        item["ref_audio_t"] = item["audio_latent"].shape[-1]
    else:
        item["ref_audio_t"] = 0
    return item


class H3WindowingState(WindowingState):
    def prepare_window(self, window: IndexListContextWindow, model) -> IndexListContextWindow:
        if not self.is_multimodal:
            return window
        video = self.latents[0]
        video_total = video.shape[self.dim]
        audio = self.latents[1]
        audio_total = audio.shape[-1]
        audio_idxs = audio_indices_for_video(window.index_list, video_total, audio_total)
        ratio = audio_total / video_total if video_total else 1
        audio_overlap = max(round(window.context_overlap * ratio), 0)
        modality_windows = {
            1: IndexListContextWindow(
                audio_idxs,
                dim=audio.ndim - 1,
                total_frames=audio_total,
                context_overlap=audio_overlap,
            )
        }
        return IndexListContextWindow(
            window.index_list,
            dim=self.dim,
            total_frames=video.shape[self.dim],
            modality_windows=modality_windows,
            context_overlap=window.context_overlap,
        )


class H3ContextHandler(IndexListContextHandler):
    def _build_window_state(self, x_in, conds, model):
        base = super()._build_window_state(x_in, conds, model)
        return H3WindowingState(
            latents=base.latents,
            guide_latents=base.guide_latents,
            guide_entries=base.guide_entries,
            keyframe_idxs=base.keyframe_idxs,
            latent_shapes=base.latent_shapes,
            dim=base.dim,
            is_multimodal=base.is_multimodal,
            temporal_downscale_ratio=base.temporal_downscale_ratio,
        )

    def evaluate_context_windows(self, *args, **kwargs):
        import logging
        import time
        started = time.perf_counter()
        result = super().evaluate_context_windows(*args, **kwargs)
        logging.info("H3 window completed in %.2fs", time.perf_counter() - started)
        return result

    def combine_context_window_results(self, x_in, sub_conds_out, sub_conds, window, window_idx, total_windows, timestep, conds_final, counts_final, biases_final):
        # Stock combine always weights self.dim. For H3 audio that axis is the
        # stereo pair (size 2); the window lives on the last axis.
        dim = window.dim
        self._retarget_fuse_buffers(x_in, dim, counts_final, biases_final)
        saved = self.dim
        self.dim = dim
        try:
            return super().combine_context_window_results(
                x_in, sub_conds_out, sub_conds, window, window_idx, total_windows, timestep,
                conds_final, counts_final, biases_final,
            )
        finally:
            self.dim = saved

    @staticmethod
    def _retarget_fuse_buffers(x_in, dim, counts_final, biases_final):
        import torch
        if not counts_final or counts_final[0].shape[dim] == x_in.shape[dim]:
            return
        shape = comfy.context_windows.get_shape_for_dim(x_in, dim)
        for i, counts in enumerate(counts_final):
            fill = torch.ones if float(counts.flatten()[0]) != 0.0 else torch.zeros
            counts_final[i] = fill(shape, device=x_in.device, dtype=counts.dtype)
            if i < len(biases_final) and len(biases_final[i]) != x_in.shape[dim]:
                biases_final[i] = [0.0] * int(x_in.shape[dim])

    def get_resized_cond(self, cond_in, x_in, window, device=None):
        resized = super().get_resized_cond(cond_in, x_in, window, device)
        if not resized:
            return resized
        audio_window = window.get_window_for_modality(1) if window.modality_windows else None
        start = int(window.index_list[0])
        offset = frames_before(start)
        window_frames = frames_before(start + len(window.index_list)) - offset
        for cond in resized:
            model_conds = cond.get("model_conds")
            if not isinstance(model_conds, dict):
                continue
            self._shift_payload(model_conds, offset, window_frames,
                                start, len(window.index_list), x_in.shape[2])
            if audio_window is not None:
                self._slice_audio_mask(model_conds, audio_window, device)
        return resized

    @staticmethod
    def _shift_payload(model_conds: dict, offset: int, window_frames: int,
                       start=0, length=None, total=None) -> None:
        payload_cond = model_conds.get("minimax_payload")
        if payload_cond is None or not isinstance(getattr(payload_cond, "cond", None), dict):
            return
        payload = dict(payload_cond.cond)
        payload.pop("layout", None)
        win0 = offset
        win1 = offset + window_frames
        kept = []
        for keyframe in payload.get("keyframes") or []:
            clipped = _clip_keyframe(keyframe, win0, win1)
            if clipped is None:
                # Retain still-image appearance guides in later windows.
                latent = keyframe.get("latent")
                if latent is None or latent.shape[2] != 1:
                    continue
                clipped = dict(keyframe)
                clipped.pop("audio_latent", None)
                clipped["resolved_frame_index"] = win0
            clipped["resolved_frame_index"] = int(clipped["resolved_frame_index"]) - win0
            kept.append(clipped)
        # Refs are not on the target timeline. They ride in every window.
        # cond_*_latents must list the same tensors PackedLayout will count,
        # keyframes first and then refs, or the packed rows disagree.
        refs = [ref for ref in (payload.get("refs") or []) if isinstance(ref, dict)]
        if length is not None and total is not None:
            # Video 1 is the driving clip. Later videos are identity references,
            # even when their duration happens to match the output duration.
            import logging
            video_number = 0
            windowed_refs = []
            for ref in refs:
                if ref.get("kind") in ("video", "video_audio"):
                    video_number += 1
                    original = ref
                    if ref.get("h3_window_role", "source" if video_number == 1 else "identity") == "source":
                        ref = window_video_reference(ref, start, length, total)
                    logging.info("H3 Video %d reference: %s, latent frames %s -> %s",
                                 video_number, "windowed" if ref is not original else "full",
                                 original.get("latent_t"), ref.get("latent_t"))
                windowed_refs.append(ref)
            refs = windowed_refs
        payload["refs"] = refs
        payload["keyframes"] = kept
        payload["cond_video_latents"] = (
            [kf["latent"] for kf in kept if kf.get("latent") is not None]
            + [ref["latent"] for ref in refs if ref.get("latent") is not None]
        )
        payload["cond_audio_latents"] = (
            [kf["audio_latent"] for kf in kept if kf.get("audio_latent") is not None]
            + [ref["audio_latent"] for ref in refs if ref.get("audio_latent") is not None]
        )
        import logging
        video_tokens = sum(z.shape[2] * ((z.shape[3] + 1) // 2) * ((z.shape[4] + 1) // 2)
                           for z in payload["cond_video_latents"])
        logging.info("H3 conditioning at frame %d: %d guides, %d references, %d visual tokens",
                     offset, len(kept), len(refs), video_tokens)
        model_conds["minimax_payload"] = comfy.conds.CONDConstant(payload)

    @staticmethod
    def _slice_audio_mask(model_conds: dict, audio_window: IndexListContextWindow, device) -> None:
        mask_cond = model_conds.get("audio_denoise_mask")
        if mask_cond is None or not isinstance(getattr(mask_cond, "cond", None), torch_tensor_type()):
            return
        mask = mask_cond.cond
        if mask.shape[-1] == audio_window.total_frames:
            model_conds["audio_denoise_mask"] = mask_cond._copy_with(
                audio_window.get_tensor(mask, device)
            )


def torch_tensor_type():
    import torch
    return torch.Tensor


class MiniMaxH3ContextWindows(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ContextWindows",
            display_name="MiniMax H3 Context Windows",
            category=CATEGORY,
            description="Split a long MiniMax H3 AV latent into overlapping windows during sampling. Use with SamplerCustomAdvanced. Length and overlap are video frames on the 17k+5 grid.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "context_length",
                    default=124,
                    min=5,
                    max=362,
                    step=17,
                    tooltip="Window length in video frames. Snapped up to the H3 grid (5, 22, 39, ... 124, 243).",
                ),
                io.Int.Input(
                    "context_overlap",
                    default=22,
                    min=0,
                    max=124,
                    step=17,
                    tooltip="Overlap in video frames, also snapped to the H3 grid. 22 is about one second. 0 disables the blend.",
                ),
                io.Combo.Input(
                    "fuse_method",
                    options=comfy.context_windows.ContextFuseMethods.LIST_STATIC,
                    default=comfy.context_windows.ContextFuseMethods.PYRAMID,
                    tooltip="How overlapping windows are blended. Pyramid weights the edges down.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, context_length, context_overlap, fuse_method):
        length_frames = align_h3_frames(context_length)
        overlap_frames = 0 if int(context_overlap) <= 0 else align_h3_frames(context_overlap)
        if overlap_frames >= length_frames:
            raise ValueError(
                f"context_overlap ({overlap_frames} frames) must be shorter than context_length ({length_frames} frames)."
            )
        latent_length = frames_to_latent(length_frames)
        latent_overlap = 0 if overlap_frames == 0 else frames_to_latent(overlap_frames)
        model = model.clone()
        model.model_options["context_handler"] = H3ContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(
                comfy.context_windows.ContextSchedules.STATIC_STANDARD
            ),
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=latent_length,
            context_overlap=latent_overlap,
            context_stride=1,
            closed_loop=False,
            dim=2,
            freenoise=False,
            causal_window_fix=False,
        )
        comfy.context_windows.create_prepare_sampling_wrapper(model)
        print(
            f"MiniMax H3 Context Windows: {length_frames} frames "
            f"({latent_length} latent) overlap {overlap_frames} frames ({latent_overlap} latent)"
        )
        return io.NodeOutput(model)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ContextWindows": MiniMaxH3ContextWindows,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ContextWindows": "MiniMax H3 Context Windows",
}
