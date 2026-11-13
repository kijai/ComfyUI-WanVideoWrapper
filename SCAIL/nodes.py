import torch
from ..utils import log
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class WanVideoAddSCAILReferenceEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Base image embeds to extend with the SCAIL reference latent — connect from a WanVideo*Embeds producer"}),
                    "vae": ("WANVAE", {"tooltip": "Wan VAE used to encode the reference image into a latent — connect from WanVideoVAELoader"}),
                    "ref_image": ("IMAGE", {"tooltip": "Single reference appearance image to encode and inject as the SCAIL identity/style cue"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Multiplier on the encoded SCAIL reference latent injected into the diffusion conditioning; 0 disables the reference"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the embedding application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the embedding application"}),
                },
                "optional": {
                    "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Optional CLIP vision embeds of the reference image used as additional context — connect from WanVideoClipVisionEncode"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, vae, ref_image, strength, start_percent, end_percent, clip_embeds=None):
        updated = dict(embeds)

        vae.to(device)
        ref_image_in = (ref_image[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
        ref_latent = vae.encode([ref_image_in], device, tiled=False)[0]
        log.info(f"SCAIL ref_latent shape: {ref_latent.shape}")

        ref_mask = torch.ones_like(ref_latent[:4])
        ref_latent = torch.cat([ref_latent, ref_mask], dim=0)
        vae.to(offload_device)

        updated.setdefault("scail_embeds", {})
        updated["scail_embeds"]["ref_latent_pos"] = ref_latent * strength
        updated["scail_embeds"]["ref_latent_neg"] = torch.zeros_like(ref_latent)
        updated["scail_embeds"]["ref_start_percent"] = start_percent
        updated["scail_embeds"]["ref_end_percent"] = end_percent
        updated["clip_context"] = clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None

        return (updated,)

class WanVideoAddSCAILPoseEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Base image embeds to extend with SCAIL pose conditioning — connect from a WanVideo*Embeds producer"}),
                    "vae": ("WANVAE", {"tooltip": "Wan VAE used to encode the pose-image sequence into a latent — connect from WanVideoVAELoader"}),
                    "pose_images": ("IMAGE", {"tooltip": "Per-frame pose-stick images covering the entire output video; encoded by the VAE into the SCAIL pose latent"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Multiplier on the SCAIL pose latent injected into the diffusion conditioning; 0 disables pose control"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the pose control application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the pose control application"}),
                },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, vae, pose_images, strength, start_percent=0.0, end_percent=1.0):
        updated = dict(embeds)

        vae.to(device)
        pose_images_in = (pose_images[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
        pose_latent = vae.encode([pose_images_in], device, tiled=False)[0]
        pose_mask = torch.ones_like(pose_latent[:4])
        pose_latent = torch.cat([pose_latent, pose_mask], dim=0)
        log.info(f"SCAIL pose_latent shape: {pose_latent.shape}")

        vae.to(offload_device)

        updated.setdefault("scail_embeds", {})
        updated["scail_embeds"]["pose_latent"] = pose_latent
        updated["scail_embeds"]["pose_strength"] = strength
        updated["scail_embeds"]["pose_start_percent"] = start_percent
        updated["scail_embeds"]["pose_end_percent"] = end_percent

        return (updated,)


NODE_CLASS_MAPPINGS = {
    "WanVideoAddSCAILPoseEmbeds": WanVideoAddSCAILPoseEmbeds,
    "WanVideoAddSCAILReferenceEmbeds": WanVideoAddSCAILReferenceEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddSCAILReferenceEmbeds": "WanVideo Add SCAIL Reference Embeds",
    "WanVideoAddSCAILPoseEmbeds": "WanVideo Add SCAIL Pose Embeds",
    }