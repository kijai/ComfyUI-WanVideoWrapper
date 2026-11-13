import torch
from ..utils import log
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class WanVideoAddDualControlEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Base image embeds to extend with LongVie dual (dense + sparse) control latents — connect from an image-embeds producer (e.g. WanVideoImageToVideoEncode)"}),
                    "vae": ("WANVAE", {"tooltip": "Wan VAE used to encode dense/sparse/prev control videos into latents — connect from WanVideoVAELoader"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Multiplier on the dual (dense + sparse) control embedding injected alongside the base image embeds; 1.0 is baseline, 0 disables"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the embedding application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the embedding application"}),
                    "first_frame_noise_level": ("FLOAT", {"default": 0.925926, "min": 0.0, "max": 1.0, "step": 0.000001, "tooltip": "Noise level for the first frame when using previous frames"}),
                },
                "optional": {
                    "dense": ("IMAGE", {"tooltip": "Per-frame dense control video (typically inverted depth maps); colors are inverted internally to match comfy depth convention"}),
                    "sparse": ("IMAGE", {"tooltip": "Sparse control signal (tracks) video input"}),
                    "prev_images": ("IMAGE", {"tooltip": "Previous frames for temporal consistency, default is 8 frames"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, vae, strength, start_percent, end_percent, first_frame_noise_level, dense=None, sparse=None, prev_images=None):
        updated = dict(embeds)
        updated.setdefault("dual_control", {})

        if dense is None and sparse is None:
            raise ValueError("At least one of dense or sparse inputs must be provided.")

        num_frames = dense.shape[0] if dense is not None else sparse.shape[0]
        height = dense.shape[1] if dense is not None else sparse.shape[1]
        width = dense.shape[2] if dense is not None else sparse.shape[2]
        msk = torch.ones(1, num_frames, height//8, width//8, device=device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)

        dense_input_latent = sparse_input_latent = None

        vae.to(device)
        if dense is not None:
            dense_images = 1 - dense[..., :3] # Invert colors for depth to match the usual range in comfy
            dense_images = dense_images.permute(3, 0, 1, 2) * 2 - 1
            dense_video_latent = vae.encode([dense_images.to(device, vae.dtype)], device, tiled=False)
            dense_first = (dense_images[:, :1]).to(device, vae.dtype)
            vae_input_dense = torch.cat([dense_first, torch.zeros(3, num_frames-1, height, width, device=device, dtype=vae.dtype)], dim=1)
            dense_concat_latent = vae.encode([vae_input_dense], device, tiled=False)
            dense_concat_latent = torch.cat([msk, dense_concat_latent], dim=1)
            dense_input_latent = torch.cat([dense_video_latent, dense_concat_latent],dim=1)
        if sparse is not None:
            sparse_images = sparse[..., :3].permute(3, 0, 1, 2) * 2 - 1
            sparse_video_latent = vae.encode([sparse_images.to(device, vae.dtype)], device, tiled=False)
            sparse_first = (sparse_images[:, :1]).to(device, vae.dtype)
            vae_input_sparse = torch.cat([sparse_first, torch.zeros(3, num_frames-1, height, width, device=device, dtype=vae.dtype)], dim=1)
            sparse_concat_latent = vae.encode([vae_input_sparse], device, tiled=False)
            sparse_concat_latent = torch.cat([msk, sparse_concat_latent], dim=1)
            sparse_input_latent = torch.cat([sparse_video_latent, sparse_concat_latent],dim=1)

        if prev_images is not None:
            prev_images = prev_images[..., :3].permute(3, 0, 1, 2) * 2 - 1
            prev_video_latent = vae.encode([prev_images.to(device, vae.dtype)], device, tiled=False)
            updated["dual_control"]["prev_latent"] = prev_video_latent[0]

        vae.to(offload_device)
        updated["dual_control"]["dense_input_latent"] = dense_input_latent
        updated["dual_control"]["sparse_input_latent"] = sparse_input_latent
        updated["dual_control"]["strength"] = strength
        updated["dual_control"]["start_percent"] = start_percent
        updated["dual_control"]["end_percent"] = end_percent
        updated["dual_control"]["first_frame_noise_level"] = first_frame_noise_level
        return (updated,)


NODE_CLASS_MAPPINGS = {
    "WanVideoAddDualControlEmbeds": WanVideoAddDualControlEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddDualControlEmbeds": "WanVideo Add Dual Control Embeds",
    }
