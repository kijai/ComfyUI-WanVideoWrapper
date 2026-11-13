import torch
from comfy import model_management as mm
import gc

def rope_params_mocha(max_seq_len, dim, theta=10000, L_test=25, k=0, start=0):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    if k > 0:
        print(f"RifleX: Using {k}th freq")
        inv_theta_pow[k-1] = 0.9 * 2 * torch.pi / L_test
        
    freqs = torch.outer(torch.arange(start, max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.autocast(device_type=mm.get_autocast_device(mm.get_torch_device()), enabled=False)
def rope_apply_mocha(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    split_size_1 = c - 2 * (c // 3)
    split_size_2 = c // 3
    split_size_3 = c // 3
    freqs_0, freqs_1, freqs_2 = freqs.split([split_size_1, split_size_2, split_size_3], dim=1)

    batch_size = grid_sizes.size(0)
    outputs = []
    
    for i in range(batch_size):
        f, h, w = grid_sizes[i].tolist()
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].reshape(seq_len, n, -1, 2))

        sf = (f - 2) // 2

        freq0_start, freq0_end = 1, 1 + sf
        freq1_start, freq1_end = 1, 1 + h
        freq2_start, freq2_end = 1, 1 + w
        
        repeat_freqs = torch.cat([
            freqs_0[freq0_start:freq0_end].view(sf, 1, 1, -1).expand(sf, h, w, -1),
            freqs_1[freq1_start:freq1_end].view(1, h, 1, -1).expand(sf, h, w, -1),
            freqs_2[freq2_start:freq2_end].view(1, 1, w, -1).expand(sf, h, w, -1)
        ], dim=-1)

        mask_freqs = torch.cat([
            freqs_0[1:2].view(1, 1, 1, -1).expand(1, h, w, -1),
            freqs_1[freq1_start:freq1_end].view(1, h, 1, -1).expand(1, h, w, -1),
            freqs_2[freq2_start:freq2_end].view(1, 1, w, -1).expand(1, h, w, -1)
        ], dim=-1)

        img_freqs = torch.cat([
            freqs_0[0:1].view(1, 1, 1, -1).expand(1, h, w, -1),
            freqs_1[freq1_start:freq1_end].view(1, h, 1, -1).expand(1, h, w, -1),
            freqs_2[freq2_start:freq2_end].view(1, 1, w, -1).expand(1, h, w, -1)
        ], dim=-1)
    
        if f == 2 * sf + 2:
            freqs_i = torch.cat([repeat_freqs, repeat_freqs, mask_freqs, img_freqs], dim=0)
        else:
            bias_freqs = torch.cat([
                freqs_0[0:1].view(1, 1, 1, -1).expand(1, h, w, -1),
                freqs_1[(h+1):(2*h+1)].view(1, h, 1, -1).expand(1, h, w, -1),
                freqs_2[(w+1):(2*w+1)].view(1, 1, w, -1).expand(1, h, w, -1)
            ], dim=-1)
            freqs_i = torch.cat([repeat_freqs, repeat_freqs, mask_freqs, img_freqs, bias_freqs], dim=0)
        
        freqs_i = freqs_i.reshape(f * h * w, 1, -1).to(x.device)
        
        # Apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]], dim=0)

        outputs.append(x_i)
    
    return torch.stack(outputs).to(x.dtype)


device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class MochaEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("WANVAE", {"tooltip": "Loaded Wan VAE used to encode input_video and ref images into latent space — connect from WanVideoVAELoader"}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload the VAE to CPU after encoding the reference / input video latents; reduces VRAM at the cost of an extra device transfer"}),
                "input_video": ("IMAGE", {"tooltip": "Source video frames (B,H,W,C) to be edited; range-shifted to [-1,1] and VAE-encoded as the base latent stream for MoCha"}),
                "mask": ("MASK", {"tooltip": "Per-frame edit mask aligned with input_video; downsampled to latent resolution and binarized so masked regions are regenerated, unmasked regions are preserved"}),
                "ref1": ("IMAGE", {"tooltip": "Primary reference image used as the identity / appearance anchor; VAE-encoded and concatenated to the latent stream"}),
            },
            "optional": {
                "ref2": ("IMAGE", {"tooltip": "Optional second reference image; VAE-encoded and appended after ref1 for multi-reference identity blending"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            }
        }
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)

    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Input for MoCha model: https://github.com/Orange-3DV-Team/MoCha"

    def process(self, vae, force_offload, input_video, mask, ref1, ref2=None, tiled_vae=False):
        W = input_video.shape[2]
        H = input_video.shape[1]
        F = input_video.shape[0]

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        F = (F - 1) // 4 * 4 + 1
        input_video = input_video.clone()[: F]

        mm.soft_empty_cache()
        gc.collect()
        vae.to(device)

        input_video = input_video.clone()[..., :3].to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)
        ref1 = ref1.clone()[..., :3].to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)
        if ref2 is not None:
            ref2 = ref2.clone()[..., :3].to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)


        latents = vae.encode(input_video * 2.0 - 1.0, device, tiled=tiled_vae)
        
        ref_latents = vae.encode(ref1 * 2.0 - 1.0, device, tiled=tiled_vae)
        num_refs = 1
        if ref2 is not None:
            ref2_latents = vae.encode(ref2 * 2.0 - 1.0, device, tiled=tiled_vae)
            ref_latents = torch.cat([ref_latents, ref2_latents], dim=2)
            num_refs = 2
        

        input_latent_mask = torch.nn.functional.interpolate(mask.unsqueeze(1).to(vae.dtype), size=(lat_h, lat_w), mode='nearest').unsqueeze(1)
        input_latent_mask = input_latent_mask.repeat(1, 16, 1, 1, 1).to(device, vae.dtype)

        input_latent_mask[input_latent_mask <= 0.5] = 0
        input_latent_mask[input_latent_mask > 0.5] = 1
        input_latent_mask[input_latent_mask == 0] = -1

        mocha_embeds = torch.cat([latents, input_latent_mask, ref_latents], dim=2)
        mocha_embeds = mocha_embeds[0]

        target_shape = (16, (F - 1) // 4 + 1, lat_h, lat_w)

        seq_len = (target_shape[1] * 2 + 1 + num_refs) * (target_shape[2] * target_shape[3] // 4)

        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "seq_len": seq_len,
            "mocha_embeds": mocha_embeds,
            "num_frames": F,
            "target_shape": target_shape,
            "mocha_num_refs": num_refs,
        }
        
        return (image_embeds,)
        

NODE_CLASS_MAPPINGS = {
    "MochaEmbeds": MochaEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "MochaEmbeds": "Mocha Embeds",
    }