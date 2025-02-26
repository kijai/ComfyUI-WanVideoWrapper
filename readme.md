# ComfyUI wrapper nodes for [WanVideo](https://github.com/Wan-Video/Wan2.1)

# WORK IN PROGRESS

# Installation
1. Clone this repo into `custom_nodes` folder.
2. Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in ComfyUI_windows_portable -folder:

  `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt`

## Models

https://huggingface.co/Kijai/WanVideo_comfy/tree/main

Text encoders to `ComfyUI/models/text_encoders`

Transformer to `ComfyUI/models/diffusion_models`

Vae to `ComfyUI/models/vae`

Right now I have only ran the I2V model successfully.

This test was 512x512x81

~16GB used with 20/40 blocks offloaded

https://github.com/user-attachments/assets/fa6d0a4f-4a4d-4de5-84a4-877cc37b715f

