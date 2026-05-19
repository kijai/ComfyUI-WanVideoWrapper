import torch
from ..utils import log
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class WanVideoAddOneToAllReferenceEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Base image embeds to extend with the OneToAll reference latent — connect from a WanVideo*Embeds producer"}),
                    "vae": ("WANVAE", {"tooltip": "Wan VAE used to encode the reference image (and mask) into a latent — connect from WanVideoVAELoader"}),
                    "ref_image": ("IMAGE", {"tooltip": "Single reference appearance image to encode and inject as the OneToAll identity/style cue"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Multiplier on the OneToAll reference latent injected into the diffusion conditioning; 0 disables the reference"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the embedding application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the embedding application"}),
                },
                "optional": {
                    "ref_mask": ("MASK", {"tooltip": "Optional mask over the reference image marking the region to keep; encoded alongside the reference for masked identity transfer"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, vae, ref_image, strength, start_percent, end_percent, ref_mask=None):
        updated = dict(embeds)

        ref_latent = ref_latent_empty = None
        vae.to(device)
        ref_image_in = (ref_image[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
        ref_latent = vae.encode([ref_image_in], device, tiled=False)
        ref_mask_in = None
        if ref_mask is not None:
            ref_mask_in = (ref_mask.unsqueeze(0).repeat(3, 1, 1, 1) * 2 - 1.).to(device, vae.dtype)
        else:
            ref_mask_in = torch.zeros_like(ref_image_in)-1
        ref_mask_latent = vae.encode([ref_mask_in], device, tiled=False)

        if ref_mask is not None and not torch.all(ref_mask == 0):
            ref_latent_empty = vae.encode([torch.zeros_like(ref_image_in)-1], device, tiled=False)
        else:
            ref_latent_empty = ref_mask_latent

        vae.to(offload_device)

        updated.setdefault("one_to_all_embeds", {})
        updated["one_to_all_embeds"]["ref_latent_pos"] = torch.cat([ref_latent, ref_latent_empty], dim=1)
        updated["one_to_all_embeds"]["ref_latent_neg"] = torch.cat([ref_latent_empty, ref_latent_empty], dim=1)
        updated["one_to_all_embeds"]["ref_strength"] = strength
        updated["one_to_all_embeds"]["ref_start_percent"] = start_percent
        updated["one_to_all_embeds"]["ref_end_percent"] = end_percent

        return (updated,)

class WanVideoAddOneToAllPoseEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Base image embeds to extend with OneToAll pose-controlnet conditioning — connect from a WanVideo*Embeds producer"}),
                    "pose_images": ("IMAGE", {"tooltip": "Per-frame pose-stick images covering the full output video; used as the OneToAll pose controlnet driving signal"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Multiplier on the OneToAll pose controlnet feature injected into the diffusion conditioning; 0 disables pose control"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the pose control application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the pose control application"}),
                },
                "optional": {
                    "pose_prefix_image": ("IMAGE", {"tooltip": "Optional single image used as the pose-conditioning prefix frame; defaults to the first frame of pose_images when omitted"}),
                    "pose_cfg_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "CFG scale for the pose control, has no effect if main cfg scale is 1.0"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, pose_images, strength, pose_prefix_image=None, start_percent=0.0, end_percent=1.0, pose_cfg_scale=1.5):
        updated = dict(embeds)
        updated.setdefault("one_to_all_embeds", {})
        pose_images_in = pose_images[..., :3].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1 # 1 B H W C -> B C 1 H W
        updated["one_to_all_embeds"]["pose_images"] = pose_images_in
        if pose_prefix_image is not None:
            updated["one_to_all_embeds"]["pose_prefix_image"] = pose_prefix_image.unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1 # 1 B H W C -> B C 1 H W
        else:
            updated["one_to_all_embeds"]["pose_prefix_image"] = pose_images_in[:, :, :1]

        updated["one_to_all_embeds"]["controlnet_strength"] = strength
        updated["one_to_all_embeds"]["controlnet_start_percent"] = start_percent
        updated["one_to_all_embeds"]["controlnet_end_percent"] = end_percent
        updated["one_to_all_embeds"]["pose_cfg_scale"] = pose_cfg_scale

        return (updated,)

class WanVideoAddOneToAllExtendEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Base image embeds to extend for OneToAll long-video continuation — connect from a WanVideo*Embeds producer"}),
                    "prev_latents": ("LATENT", {"tooltip": "Previous latents to be used to continue generation"}),
                    "window_size": ("INT", {"default": 81, "min": 1, "max": 256, "step": 1, "tooltip": "Total number of frames in this extension window (including overlap with previous segment); the wrapper slices pose_images[frames_processed-overlap : frames_processed-overlap+window_size]" }),
                    "overlap": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1, "tooltip": "Number of overlapping frames between previous and new frames" }),
                    "frames_processed": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Number of frames already processed in the video" }),
                    "if_not_enough_frames": (["pad_with_last", "error"], {"default": "pad_with_last", "tooltip": "What to do if there are not enough frames in pose_images for the window"}),
                },
                "optional": {
                    "pose_images": ("IMAGE", {"tooltip": "Per-frame pose-stick images covering the entire video; the node slices pose_images[frames_processed-overlap : frames_processed-overlap+window_size] for this segment"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "IMAGE",)
    RETURN_NAMES = ("image_embeds", "pose_slice",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, prev_latents, if_not_enough_frames, window_size=81, overlap=5, frames_processed=0, pose_images=None):
        updated = dict(embeds)
        updated.setdefault("one_to_all_embeds", {})
        updated["one_to_all_embeds"]["prev_latents"] = prev_latents["samples"][0]
        if pose_images is not None:
            pose_images_in = pose_images.clone()[..., :3]
            start = max(0, frames_processed - overlap)
            end = start + window_size
            log.info(f"Extracting pose images from {start} to {end}")
            if start >= pose_images_in.shape[0]:
                raise ValueError(f"start index {start} exceeds pose images length {pose_images_in.shape[0]}")
            if end > pose_images_in.shape[0]:
                if if_not_enough_frames == "pad_with_last":
                    padding_needed = end - pose_images_in.shape[0]
                    pose_images_in = torch.cat([pose_images_in, pose_images_in[-1:].repeat(padding_needed, 1, 1, 1)], dim=0)
                    log.info(f"Not enough frames, padding with {padding_needed} frames to reach {end} total frames")
                else:
                    raise ValueError(f"end index {end} exceeds pose images length {pose_images.shape[0]}")
            pose_slice = pose_images_in[start:end]
        else:
            pose_slice = torch.zeros((1, 64, 64, 3))

        return (updated, pose_slice)


NODE_CLASS_MAPPINGS = {
    "WanVideoAddOneToAllReferenceEmbeds": WanVideoAddOneToAllReferenceEmbeds,
    "WanVideoAddOneToAllPoseEmbeds": WanVideoAddOneToAllPoseEmbeds,
    "WanVideoAddOneToAllExtendEmbeds": WanVideoAddOneToAllExtendEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddOneToAllReferenceEmbeds": "WanVideo Add OneToAll Reference Embeds",
    "WanVideoAddOneToAllPoseEmbeds": "WanVideo Add OneToAll Pose Embeds",
    "WanVideoAddOneToAllExtendEmbeds": "WanVideo Add OneToAll Extend Embeds",
    }
