# MosaicDiff

MosaicDiff restores mosaic in a video with the plain BasicVSR++ checkpoint, then repaints that region with Eros Max on MiniMax H3.

It is a separate program. It does not run Jasna.

## Run it here

`run.bat` uses the Jasna virtual environment when that environment is next to this folder. Finished videos go to `Output`. If a public weight file is missing, Start downloads it into `models\`. Place your own LoRA there as `lora.safetensors`. Open **Models** if a path is wrong.

## Download the Windows program

The ready-to-run build is on the [Releases](https://github.com/dotaku22/MosaicDiff/releases) page. Download `MosaicDiff-windows.7z`, open it with [7-Zip](https://www.7-zip.org/), and run `MosaicDiff.exe`.

That download does not include ComfyUI. Licenses for the shipped pieces are in the `licenses` folder. Start with `licenses\NOTICES.txt`.

On the other PC:

- An NVIDIA GPU and a current driver.
- A current ComfyUI install, the normal one from the ComfyUI site or the portable package. MosaicDiff looks for it in the usual folders. If it cannot find it, set **ComfyUI Python** and **ComfyUI folder** in **Models**.
- Your LoRA, saved as `models\lora.safetensors`. BasicVSR++, Eros Max, the text encoder, and the video VAE download on first Start when they are missing.

The H3 context-window node and the RTX upscaler ship in `comfy_nodes` and are loaded from there. The other PC does not install those itself. The first H3 run installs the NVIDIA video effects package into that ComfyUI Python if it is not already there. ComfyUI itself still has to include the built-in MiniMax H3 nodes, so use a current ComfyUI.

## What the window does

Add videos and press Start. Finished files are written to `Output`. Two sliders matter: how many seconds H3 samples at once, and the H3 resolution. BasicVSR++ runs from its checkpoint, and the mosaic detector runs from its ONNX model. Neither is compiled to TensorRT.
