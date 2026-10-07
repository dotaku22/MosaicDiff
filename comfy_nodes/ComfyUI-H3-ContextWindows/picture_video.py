"""Experimental Picture 1 label with temporal identity-reference encoding."""
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo


class PictureVideoClip:
    def __init__(self, clip):
        self.clip = clip

    def tokenize(self, prompt, minimax_ref_items):
        # Tokenize blocks independently to keep the source numbered Video 1.
        # Only the identity presentation header changes; temporal vision blocks stay intact.
        tokenizer = self.clip.tokenizer.qwen3vl_32b
        old = tokenizer.tokenize_with_weights("<Video 1>: ", disable_weights=True)[0]
        new = tokenizer.tokenize_with_weights("<Picture 1>: ", disable_weights=True)[0]
        entries = []
        for index, item in enumerate(minimax_ref_items):
            block = self.clip.tokenize("", minimax_ref_items=[item])["qwen3vl_32b"][0]
            if index == 0:
                if [x[0] for x in block[:len(old)]] != [x[0] for x in old]:
                    raise ValueError("Unexpected H3 reference header; cannot label identity video as Picture 1.")
                block = new + block[len(old):]
            entries.extend(block)
        if prompt:
            entries.extend(self.clip.tokenize(prompt)["qwen3vl_32b"][0])
        return {"qwen3vl_32b": [entries]}

    def encode_from_tokens_scheduled(self, tokens):
        return self.clip.encode_from_tokens_scheduled(tokens)


class MiniMaxH3PictureVideoReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PictureVideoReference",
            display_name="MiniMax H3 Reference to Video (Picture Video Experimental)",
            category="model/conditioning/minimax",
            description="Experimental: a temporal face reference presented as Picture 1. Source motion is Video 1. Both image batches must be 24 fps. LoRA compatibility is not guaranteed.",
            inputs=[io.Clip.Input("clip"), io.Vae.Input("vae"),
                    io.Image.Input("ref_image_1", tooltip="Face reference VIDEO batch, at least 5 frames at 24 fps."),
                    io.Image.Input("source_video", tooltip="Driving video at 24 fps; presented as Video 1."),
                    io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                    io.Int.Input("width", default=512, min=32, max=8192, step=32),
                    io.Int.Input("height", default=512, min=32, max=8192, step=32),
                    io.Int.Input("length", default=124, min=5, max=3600, step=17)],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()])

    @classmethod
    def execute(cls, clip, vae, ref_image_1, source_video, prompt, width, height, length):
        if ref_image_1.shape[0] < 5:
            raise ValueError("ref_image_1 requires a video batch with at least 5 frames.")
        result = MiniMaxH3ReferenceToVideo.execute(
            clip=PictureVideoClip(clip), vae=vae, prompt=prompt,
            width=width, height=height, length=length,
            ref_videos={"ref_video_1": ref_image_1, "ref_video_2": source_video})
        positive, latent = result.result
        copied = []
        for embedding, metadata in positive:
            metadata = dict(metadata)
            refs = [dict(ref) for ref in metadata.get("minimax_refs", [])]
            if len(refs) != 2:
                raise ValueError("Expected identity and source video reference blocks.")
            refs[0]["h3_window_role"] = "identity"
            refs[1]["h3_window_role"] = "source"
            metadata["minimax_refs"] = refs
            copied.append([embedding, metadata])
        return io.NodeOutput(copied, latent)
