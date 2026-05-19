import os
import torch
from ..utils import log
import numpy as np

import comfy.model_management as mm
from comfy.utils import load_torch_file
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

from .resampler import Resampler

class LoadLynxResampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from 'ComfyUI/models/diffusion_models'"}),
                "precision": (["fp32", "bf16", "fp16"], {"default": "fp16", "tooltip": "Computation/storage dtype for the Lynx resampler weights; fp16 is the safe default, fp32 is most accurate but uses more VRAM"}),
            },
        }

    RETURN_TYPES = ("LYNXRESAMPLER",)
    RETURN_NAMES = ("resampler", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model_name, precision):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        resampler_sd = load_torch_file(model_path, safe_load=True)

        output_dim = resampler_sd["proj_out.weight"].shape[0]

        resampler = Resampler(
            depth=4,
            dim=1280,
            dim_head=64,
            embedding_dim=512,
            ff_mult=4,
            heads=20,
            num_queries=16,
            output_dim=output_dim,
            dtype=dtype,
        ).eval()
        resampler.to(offload_device, dtype)
        resampler.load_state_dict(resampler_sd, strict=True)

        return resampler,


class LynxInsightFaceCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source portrait image; the first frame is run through InsightFace landmark detection and aligned into a 112×112 IP-face crop plus a 256×256 reference crop"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("ip_image", "ref_image")
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, image, image_size=112):
        from .face.face_encoder import get_landmarks_from_image
        from .face.face_utils import align_face
        from insightface.utils import face_align

        image_np = (image[0].numpy() * 255).astype(np.uint8)
        landmarks = get_landmarks_from_image(image_np)

        in_image = np.array(image_np)
        landmark = np.array(landmarks)

        ip_face_aligned = face_align.norm_crop(in_image, landmark=landmark, image_size=112)
        ref_face_aligned = align_face(in_image, landmark, extend_face_crop=True, face_size=256)

        ip_face_aligned = torch.from_numpy(ip_face_aligned).unsqueeze(0).float() / 255.0
        ref_face_aligned = torch.from_numpy(ref_face_aligned).unsqueeze(0).float() / 255.0

        ip_face_aligned = (ip_face_aligned - ip_face_aligned.min()) / (ip_face_aligned.max() - ip_face_aligned.min())
        ref_face_aligned = (ref_face_aligned - ref_face_aligned.min()) / (ref_face_aligned.max() - ref_face_aligned.min())
        ref_face_aligned = ref_face_aligned[:, :,  :, [2, 1, 0]]  # BGR to RGB

        return ip_face_aligned, ref_face_aligned


class LynxEncodeFaceIP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resampler": ("LYNXRESAMPLER", {"tooltip": "Loaded Lynx resampler that maps ArcFace embeddings into the Wan IP-adapter conditioning space — connect from Load Lynx Resampler"}),
                "ip_image": ("IMAGE", {"tooltip": "Aligned 112×112 IP-face crop (from LynxInsightFaceCrop) to encode as the identity reference for the IP-adapter; range is normalized to [-1,1] before ArcFace embedding"}),
            },
        }

    RETURN_TYPES = ("LYNXIP",)
    RETURN_NAMES = ("lynx_face_embeds",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, resampler, ip_image):
        from .face.face_encoder import FaceEncoderArcFace

        image_in = ip_image.permute(0, 3, 1, 2).to(device) * 2 - 1  # to [-1, 1]

        # Face embedding via ArcFace
        face_encoder = FaceEncoderArcFace()
        face_encoder.init_encoder_model(device)
        arcface_embed = face_encoder(image_in).to(device, resampler.dtype)[0]

        arcface_embed = arcface_embed.reshape([1, -1, 512])

        resampler.to(device)
        ip_x = resampler(arcface_embed)
        ip_x_uncond = resampler(arcface_embed * 0)
        resampler.to(offload_device)

        ip_x= ip_x.to(resampler.dtype)

        out_dict = {
            'ip_x': ip_x,
            'ip_x_uncond': ip_x_uncond,
        }

        return out_dict,

class DrawArcFaceLandmarks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lynx_face_embeds": ("LYNXIP", {"tooltip": "Lynx face IP embeddings (carries ArcFace landmarks used for the overlay) — connect from Lynx Encode Face IP"}),
                "image": ("IMAGE", {"tooltip": "Input images for the model"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Same socket as the required image — kept for backward compatibility; wire the source image here"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("landmarked_image", )
    FUNCTION = "draw"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Draw face landmarks on an image for visualization/debugging"

    def draw(self, lynx_face_embeds, image):
        import cv2
        landmarks = lynx_face_embeds['landmarks']
        image_np = image[0].numpy() * 255

        for (x, y) in landmarks:
            cv2.circle(image_np, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

        image_out = torch.from_numpy(image_np / 255).unsqueeze(0).float()

        return image_out,

class WanVideoAddLynxEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Existing Wan image-embeds bundle to extend with Lynx identity conditioning — connect from WanVideoImageToVideoEncode / WanVideoEmptyEmbeds / any other embeds producer"}),
                    "ip_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the IP-adapter ArcFace identity injection (lynx_ip_embeds); higher = stronger face lock, 1.0 is the trained default"}),
                    "ref_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the reference-image latent feature (ref_image path); higher = more appearance match, 1.0 is the trained default"}),
                    "lynx_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "If above 1.0 and main cfg_scale is above 1.0, run extra pass, default value 2.0"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start of the denoising schedule (0–1) at which the Lynx face/ref conditioning becomes active"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End of the denoising schedule (0–1) after which the Lynx face/ref conditioning is dropped"}),
                },
                "optional": {
                    "vae": ("WANVAE", {"tooltip": "VAE model, only needed if ref_image is provided"}),
                    "lynx_ip_embeds": ("LYNXIP", {"tooltip": "Lynx face IP embeddings produced from a cropped face image — connect from Lynx Encode Face IP"}),
                    "ref_image": ("IMAGE", {"tooltip": "Optional reference portrait image (typically the 256×256 ref_image from LynxInsightFaceCrop); VAE-encoded into a latent that feeds the ref_scale pathway. Requires vae and ref_text_embed to be wired."}),
                    "ref_text_embed": ("WANVIDEOTEXTEMBEDS", {"tooltip": "Text embeddings paired with ref_image (e.g. describing the reference subject); required whenever ref_image is wired — connect from WanVideoTextEncode/Cached"}),
                    "ref_blocks_to_use": ("STRING", {"default": "", "forceInput": True, "tooltip": "Comma-separated list of block indices and ranges to use for reference feature, e.g. '0-20, 25, 28, 35-39'. If empty, use all blocks."}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, ip_scale, ref_scale, start_percent, end_percent, lynx_cfg_scale, vae=None, lynx_ip_embeds=None, ref_image=None, ref_text_embed=None, ref_blocks_to_use=""):
        if ref_image is not None and ref_text_embed is None:
            raise ValueError("If ref_image is provided, ref_text_embed must also be provided.")
        if ref_image is not None:
            vae.to(device)
            ref_image_in = (ref_image[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
            ref_latent = vae.encode([ref_image_in], device, tiled=False, sample=True)
            ref_latent_uncond = vae.encode([torch.zeros_like(ref_image_in)], device, tiled=False, sample=True)
            vae.to(offload_device)
        if ref_blocks_to_use.strip() == "":
            ref_blocks_to_use = None
        else:
            # Parse comma-separated blocks and ranges
            blocks = []
            for item in ref_blocks_to_use.split(","):
                item = item.strip()
                if "-" in item and not item.startswith("-"):
                    # Handle range like "0-20" or "35-39"
                    try:
                        start, end = item.split("-", 1)
                        start, end = int(start.strip()), int(end.strip())
                        blocks.extend(list(range(start, end + 1)))
                    except ValueError:
                        print(f"Invalid range format: {item}")
                elif item.isdigit():
                    # Handle single number
                    blocks.append(int(item))
                else:
                    print(f"Invalid block specification: {item}")
            ref_blocks_to_use = sorted(list(set(blocks)))  # Remove duplicates and sort
            print("Using ref blocks:", ref_blocks_to_use)
            
        new_entry = {
            "ip_x": lynx_ip_embeds["ip_x"] if lynx_ip_embeds is not None else None,
            "ip_x_uncond": lynx_ip_embeds["ip_x_uncond"] if lynx_ip_embeds is not None else None,
            "ref_latent": ref_latent if ref_image is not None else None,
            "ref_latent_uncond": ref_latent_uncond if ref_image is not None else None,
            "ref_text_embed": ref_text_embed if ref_text_embed is not None else None,
            "ip_scale": ip_scale,
            "ref_scale": ref_scale,
            "cfg_scale": lynx_cfg_scale,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "ref_blocks_to_use": ref_blocks_to_use,
        }

        updated = dict(embeds)
        updated["lynx_embeds"] = new_entry
        return (updated,)

NODE_CLASS_MAPPINGS = {
    "LoadLynxResampler": LoadLynxResampler,
    "LynxEncodeFaceIP": LynxEncodeFaceIP,
    "DrawArcFaceLandmarks": DrawArcFaceLandmarks,
    "WanVideoAddLynxEmbeds": WanVideoAddLynxEmbeds,
    "LynxInsightFaceCrop": LynxInsightFaceCrop,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLynxResampler": "Load Lynx Resampler",
    "LynxEncodeFaceIP": "Lynx Encode Face IP",
    "DrawArcFaceLandmarks": "Draw ArcFace Landmarks",
    "WanVideoAddLynxEmbeds": "WanVideo Add Lynx Embeds",
    "LynxInsightFaceCrop": "Lynx InsightFace Crop",
}
