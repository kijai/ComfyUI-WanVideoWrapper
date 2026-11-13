import folder_paths
import torch

from comfy.utils import load_torch_file
import comfy.model_management as mm
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class WanVideoAddFlashVSRInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Base image embeds to extend with FlashVSR low-res conditioning — connect from a WanVideo*Embeds producer"}),
                    "images": ("IMAGE", {"tooltip": "Per-frame low-quality / low-resolution source video to super-resolve; passed through FlashVSR as the LQ conditioning signal"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Multiplier on the FlashVSR low-res conditioning latent; 1.0 = full super-resolution guidance, 0 disables it"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, images, strength):
        updated = dict(embeds)
        updated["flashvsr_LQ_images"] = images
        updated["flashvsr_strength"] = strength
        return (updated,)


class WanVideoFlashVSRDecoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16", "tooltip": "Compute dtype the FlashVSR TCDecoder loads at; bf16 default matches the released weights"}
                ),
            }
        }

    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision):
        from .TCDecoder import build_tcdecoder
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = folder_paths.get_full_path("vae", model_name)
        sd = load_torch_file(model_path, safe_load=True)

        TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16+768, dtype=dtype)
        TCDecoder.load_state_dict(sd, strict=True)
        TCDecoder.to(dtype)

        return (TCDecoder,)

NODE_CLASS_MAPPINGS = {
    "WanVideoAddFlashVSRInput": WanVideoAddFlashVSRInput,
    "WanVideoFlashVSRDecoderLoader": WanVideoFlashVSRDecoderLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddFlashVSRInput": "WanVideo Add FlashVSR Input",
    "WanVideoFlashVSRDecoderLoader": "WanVideo FlashVSR Decoder Loader",
}
