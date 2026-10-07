Public weights are downloaded into this folder on Start when they are missing.
The LoRA is not downloaded. Place that file here yourself.

basicvsr.pth          BasicVSR++ checkpoint (plain weights, not a compiled engine)
rfdetr.onnx           mosaic detector model (runs as-is on any graphics card)
unet.safetensors      Eros Max H3 (TenStrip/10Eros-Max, TURBO hybrid beta5 int8)
lora.safetensors      your H3 LoRA
clip.safetensors      MiniMax H3 text encoder
vae.safetensors       MiniMax H3 video VAE

ComfyUI is still required for the H3 pass. Point ComfyUI Python and
the ComfyUI folder at that install from the Models button.
The other PC needs an NVIDIA GPU and a current driver.

Licenses for the detector, BasicVSR++, and the bundled libraries are in
the licenses folder next to the program. Start with licenses\NOTICES.txt.
