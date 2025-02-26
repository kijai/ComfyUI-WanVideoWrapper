import os
import torch
import gc
from .utils import log, print_memory
import numpy as np
import math
from tqdm import tqdm

from .wanvideo.modules.clip import CLIPModel
from .wanvideo.modules.model import WanModel, rope_params
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, save_torch_file, ProgressBar
import comfy.model_base
import comfy.latent_formats

script_directory = os.path.dirname(os.path.abspath(__file__))

def add_noise_to_reference_video(image, ratio=None):
    if ratio is None:
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
        sigma = torch.exp(sigma).to(image.dtype)
    else:
        sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio

    image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image

class WanVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 40, "step": 1, "tooltip": "Number of double blocks to swap"}),
            },
        }
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Settings for block swapping, reduces VRAM use by swapping blocks to CPU memory"

    def setargs(self, **kwargs):
        return (kwargs, )

# class WanVideoEnhanceAVideo:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "weight": ("FLOAT", {"default": 2.0, "min": 0, "max": 100, "step": 0.01, "tooltip": "The feta Weight of the Enhance-A-Video"}),
#                 "blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable Enhance-A-Video for selected blocks"}),
#                 "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply Enhance-A-Video"}),
#                 "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply Enhance-A-Video"}),
#             },
#         }
#     RETURN_TYPES = ("FETAARGS",)
#     RETURN_NAMES = ("feta_args",)
#     FUNCTION = "setargs"
#     CATEGORY = "WanVideoWrapper"
#     DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"

#     def setargs(self, **kwargs):
#         return (kwargs, )


# class WanVideoTeaCache:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
#                                             "tooltip": "Higher values will make TeaCache more aggressive, faster, but may cause artifacts"}),
#             },
#         }
#     RETURN_TYPES = ("TEACACHEARGS",)
#     RETURN_NAMES = ("teacache_args",)
#     FUNCTION = "process"
#     CATEGORY = "WanVideoWrapper"
#     DESCRIPTION = "TeaCache settings for WanVideo to speed up inference"

#     def process(self, rel_l1_thresh):
#         teacache_args = {
#             "rel_l1_thresh": rel_l1_thresh,
#         }
#         return (teacache_args,)


class WanVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

from comfy.latent_formats import LatentFormat


class WanVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo #todo better values
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        # Don't know what this is. Value taken from ComfyUI Mochi model.
        self.memory_usage_factor = 2.0
        # denoiser is handled by extension
        self.unet_config["disable_unet_model_creation"] = True


#region Model loading
class WanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6", "torchao_int4", "torchao_int8"], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("WANCOMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("WANVIDLORA", {"default": None}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device,  quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None):
        transformer = None
        mm.unload_all_models()
        mm.soft_empty_cache()
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        manual_offloading = True
        transformer_load_device = device if load_device == "main_device" else offload_device

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        dim = sd["patch_embedding.weight"].shape[0]
        in_channels = sd["patch_embedding.weight"].shape[1]
        print("in_channels: ", in_channels)
        ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]
        model_type = "i2v" if in_channels == 36 else "t2v"
        num_heads = 40 if dim == 5120 else 12
        num_layers = 40 if dim == 5120 else 30

        log.info(f"Model type: {model_type}, num_heads: {num_heads}, num_layers: {num_layers}")

        TRANSFORMER_CONFIG= {
            "dim": dim,
            "ffn_dim": ffn_dim,
            "eps": 1e-06,
            "freq_dim": 256,
            "in_dim": in_channels,
            "model_type": model_type,
            "out_dim": 16,
            "text_len": 512,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attention_mode": attention_mode,
            "main_device": device,
            "offload_device": offload_device,
        }

        with init_empty_weights():
            transformer = WanModel(**TRANSFORMER_CONFIG)
        transformer.eval()

        comfy_model = WanVideoModel(
            WanVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )


        if not "torchao" in quantization:
            log.info("Using accelerate to load and assign model weights to device...")
            if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_scaled":
                dtype = torch.float8_e4m3fn
            elif quantization == "fp8_e5m2":
                dtype = torch.float8_e5m2
            else:
                dtype = base_dtype
            params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb", "modulation"}
            for name, param in transformer.named_parameters():
                #print("Assigning Parameter name: ", name)
                dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])

            comfy_model.diffusion_model = transformer
            comfy_model.load_device = transformer_load_device
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)

            del sd
            gc.collect()
            mm.soft_empty_cache()

            if lora is not None:
                from comfy.sd import load_lora_for_models
                for l in lora:
                    log.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                    lora_path = l["path"]
                    lora_strength = l["strength"]
                    lora_sd = load_torch_file(lora_path, safe_load=True)

                    # for k in lora_sd.keys():
                    #   print(k)

                    patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)

            comfy.model_management.load_models_gpu([patcher])

            if load_device == "offload_device":
                patcher.model.diffusion_model.to(offload_device)

            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                #params_to_keep.update({"ffn"})
                print(params_to_keep)
                convert_fp8_linear(patcher.model.diffusion_model, base_dtype, params_to_keep=params_to_keep)

            #compile
            if compile_args is not None:
                torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
                if compile_args["compile_transformer_blocks"]:
                    for i, block in enumerate(patcher.model.diffusion_model.blocks):
                        patcher.model.diffusion_model.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
        elif "torchao" in quantization:
            try:
                from torchao.quantization import (
                quantize_,
                fpx_weight_only,
                float8_dynamic_activation_float8_weight,
                int8_dynamic_activation_int8_weight,
                int8_weight_only,
                int4_weight_only
            )
            except:
                raise ImportError("torchao is not installed")

            # def filter_fn(module: nn.Module, fqn: str) -> bool:
            #     target_submodules = {'attn1', 'ff'} # avoid norm layers, 1.5 at least won't work with quantized norm1 #todo: test other models
            #     if any(sub in fqn for sub in target_submodules):
            #         return isinstance(module, nn.Linear)
            #     return False

            if "fp6" in quantization:
                quant_func = fpx_weight_only(3, 2)
            elif "int4" in quantization:
                quant_func = int4_weight_only()
            elif "int8" in quantization:
                quant_func = int8_weight_only()
            elif "fp8dq" in quantization:
                quant_func = float8_dynamic_activation_float8_weight()
            elif 'fp8dqrow' in quantization:
                from torchao.quantization.quant_api import PerRow
                quant_func = float8_dynamic_activation_float8_weight(granularity=PerRow())
            elif 'int8dq' in quantization:
                quant_func = int8_dynamic_activation_int8_weight()

            log.info(f"Quantizing model with {quant_func}")
            comfy_model.diffusion_model = transformer
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)

            if lora is not None:
                from comfy.sd import load_lora_for_models
                for l in lora:
                    log.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                    lora_path = l["path"]
                    lora_strength = l["strength"]
                    lora_sd = load_torch_file(lora_path, safe_load=True)

                    #for k in lora_sd.keys():
                    #   print(k)

                    patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)

            comfy.model_management.load_models_gpu([patcher])

            for i, block in enumerate(patcher.model.diffusion_model.blocks):
                log.info(f"Quantizing block {i}")
                for name, _ in block.named_parameters(prefix=f"blocks.{i}"):
                    #print(f"Parameter name: {name}")
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=transformer_load_device, dtype=base_dtype, value=sd[name])
                if compile_args is not None:
                    patcher.model.diffusion_model.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                quantize_(block, quant_func)
                print(block)
                #block.to(offload_device)
            for name, param in patcher.model.diffusion_model.named_parameters():
                if "blocks" not in name:
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=transformer_load_device, dtype=base_dtype, value=sd[name])

            manual_offloading = False # to disable manual .to(device) calls
            log.info(f"Quantized transformer blocks to {quantization}")
            for name, param in patcher.model.diffusion_model.named_parameters():
                print(name, param.dtype)
                #param.data = param.data.to(self.vae_dtype).to(device)

            del sd
            mm.soft_empty_cache()

        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["manual_offloading"] = manual_offloading
        patcher.model["quantization"] = "disabled"
        patcher.model["block_swap_args"] = block_swap_args

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)

        return (patcher,)

#region load VAE

class WanVideoVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
            }
        }

    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision, compile_args=None):
        from .wanvideo.wan_video_vae import WanVideoVAE

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        #with open(os.path.join(script_directory, 'configs', 'hy_vae_config.json')) as f:
        #    vae_config = json.load(f)
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        has_model_prefix = any(k.startswith("model.") for k in vae_sd.keys())
        if not has_model_prefix:
            vae_sd = {f"model.{k}": v for k, v in vae_sd.items()}

        vae = WanVideoVAE(dtype=dtype)
        vae.load_state_dict(vae_sd)
        vae.eval()
        vae.to(device = offload_device, dtype = dtype)


        return (vae,)



class WanVideoTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_transformer_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),

            },
        }
    RETURN_TYPES = ("WANCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_transformer_blocks": compile_transformer_blocks,
        }

        return (compile_args, )

#region TextEncode

class LoadWanVideoT5TextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
                 "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
            },
            "optional": {
                "load_device": (["main_device", "offload_device"], {"default": "offload_device"}),
            }
        }

    RETURN_TYPES = ("WANTEXTENCODER",)
    RETURN_NAMES = ("wan_t5_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan text_encoder model from 'ComfyUI/models/LLM'"

    def loadmodel(self, model_name, precision, load_device="offload_device"):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        text_encoder_load_device = device if load_device == "main_device" else offload_device

        tokenizer_path = os.path.join(script_directory, "configs", "T5_tokenizer")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_path = folder_paths.get_full_path("text_encoders", model_name)
        sd = load_torch_file(model_path, safe_load=True)

        T5_text_encoder = T5EncoderModel(
            text_len=512,
            dtype=dtype,
            device=text_encoder_load_device,
            state_dict=sd,
            tokenizer_path=tokenizer_path,
        )

        return (T5_text_encoder,)

class LoadWanVideoClipTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
                 "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            },
            "optional": {
                "load_device": (["main_device", "offload_device"], {"default": "offload_device"}),
            }
        }

    RETURN_TYPES = ("WANCLIP",)
    RETURN_NAMES = ("wan_clip_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan text_encoder model from 'ComfyUI/models/LLM'"

    def loadmodel(self, model_name, precision, load_device="offload_device"):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        text_encoder_load_device = device if load_device == "main_device" else offload_device

        tokenizer_path = os.path.join(script_directory, "configs", "clip")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_path = folder_paths.get_full_path("text_encoders", model_name)
        sd = load_torch_file(model_path, safe_load=True)

        clip_model = CLIPModel(dtype=dtype, device=text_encoder_load_device, state_dict=sd, tokenizer_path=tokenizer_path)

        return (clip_model,)

class WanVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "t5": ("WANTEXTENCODER",),
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, t5, positive_prompt, negative_prompt,force_offload=True):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        t5.model.to(device)

        context = t5([positive_prompt], device)
        context_null = t5([negative_prompt], device)
        context = [t.to(device) for t in context]
        context_null = [t.to(device) for t in context_null]

        if force_offload:
            t5.model.to(offload_device)


        prompt_embeds_dict = {
                "prompt_embeds": context,
                "negative_prompt_embeds": context_null,
            }
        return (prompt_embeds_dict,)

#region clip image encode
class WanVideoImageClipEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("WANCLIP",),
            "image": ("IMAGE", {"tooltip": "Image to encode"}),
            "vae": ("WANVAE",),
            "generation_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the image to encode"}),
            "generation_height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 5, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, clip, vae, image, num_frames, generation_width, generation_height, force_offload=True):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        patch_size = (1, 2, 2)
        vae_stride = (4, 8, 8)
        sp_size = 1 #no parallelism
        H, W = image.shape[1], image.shape[2]
        max_area = generation_width * generation_height

        from comfy.clip_vision import clip_preprocess
        pixel_values = clip_preprocess(image.to(device), size=224, mean=self.image_mean, std=self.image_std, crop=True).float()
        clip.model.to(device)
        clip_context = clip.visual(pixel_values)
        if force_offload:
            clip.model.to(offload_device)

        aspect_ratio = H / W
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
            patch_size[1] * patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
            patch_size[2] * patch_size[2])
        h = lat_h * vae_stride[1]
        w = lat_w * vae_stride[2]

        msk = torch.ones(1, num_frames, lat_h, lat_w, device=device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        max_seq_len = ((num_frames - 1) // vae_stride[0] + 1) * lat_h * lat_w // (
            patch_size[1] * patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / sp_size)) * sp_size

        vae.to(device)

        image = image.to(device = device, dtype = vae.dtype) * 2 - 1

        y = vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, num_frames-1, h, w, device=device)
            ],
                         dim=1).to(image)
        ],device)[0]
        y = torch.concat([msk, y])
        vae.to(offload_device)

        image_embeds = {
            "image_embeds": y,
            "clip_context": clip_context,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
        }

        return (image_embeds,)

class WanVideoEmptyEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height):

        patch_size = (1, 2, 2)
        vae_stride = (4, 8, 8)

        target_shape = (16, (num_frames - 1) // vae_stride[0] + 1,
                        height // vae_stride[1],
                        width // vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (patch_size[1] * patch_size[2]) *
                            target_shape[1])

        embeds = {
            "max_seq_len": seq_len,
            "target_shape": target_shape,
            "num_frames": num_frames
        }

        return (embeds,)


#region Sampler
class WanVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "scheduler": (["unipc", "dpm++", "dpm++_sde"],
                    {
                        "default": 'dpm++'
                    }),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 4. Allows for new frames to be generated after without looping"}),


            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                #"image_cond_latents": ("LATENT", {"tooltip": "init Latents to use for image2video process"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                #"riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 4. Allows for new frames to be generated after 129 without looping"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, model, text_embeds, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index, force_offload=True, samples=None, denoise_strength=1.0):
        patcher = model
        model = model.model
        transformer = model.diffusion_model

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if model["block_swap_args"] is not None:
            for name, param in transformer.named_parameters():
                if "block" not in name:
                    param.data = param.data.to(device)

            transformer.block_swap(
                model["block_swap_args"]["blocks_to_swap"] - 1 ,
            )
        else:
            if model["manual_offloading"]:
                transformer.to(device)


        # # Initialize TeaCache if enabled
        # if teacache_args is not None:
        #     # Check if dimensions have changed since last run
        #     if (not hasattr(transformer, 'last_dimensions') or
        #             transformer.last_dimensions != (height, width, num_frames) or
        #             not hasattr(transformer, 'last_frame_count') or
        #             transformer.last_frame_count != num_frames):
        #         # Reset TeaCache state on dimension change
        #         transformer.cnt = 0
        #         transformer.accumulated_rel_l1_distance = 0
        #         transformer.previous_modulated_input = None
        #         transformer.previous_residual = None
        #         transformer.last_dimensions = (height, width, num_frames)
        #         transformer.last_frame_count = num_frames

        #     transformer.enable_teacache = True
        #     transformer.num_steps = steps
        #     transformer.rel_l1_thresh = teacache_args["rel_l1_thresh"]
        # else:
        #     transformer.enable_teacache = False

        mm.soft_empty_cache()
        gc.collect()

        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        #for name, param in transformer.named_parameters():
        #    print(name, param.data.device)

        steps = int(steps/denoise_strength)

        if scheduler == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif 'dpm++' in scheduler:
            if scheduler == 'dpm++_sde':
                algorithm_type = "sde-dpmsolver++"
            else:
                algorithm_type = "dpmsolver++"
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False,
                algorithm_type= algorithm_type)
            sampling_sigmas = get_sampling_sigmas(steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        if denoise_strength < 1.0:
            steps = int(steps * denoise_strength)
            timesteps = timesteps[-(steps + 1):]

        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)
        if transformer.model_type == "i2v":
            noise = torch.randn(
                16,
                (image_embeds["num_frames"] - 1) // 4 + 1,
                image_embeds["lat_h"],
                image_embeds["lat_w"],
                dtype=torch.float32,
                generator=seed_g,
                device=torch.device("cpu"))
            seq_len = image_embeds["max_seq_len"]
        else: #t2v
            target_shape = image_embeds["target_shape"]
            seq_len = image_embeds["max_seq_len"]
            noise = torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    generator=seed_g)

        if samples is not None:
            latent_timestep = timesteps[:1].to(noise)
            noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * samples["samples"].squeeze(0).to(noise)

        latent = noise.to(device)

        d = transformer.dim // transformer.num_heads
        freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6), L_test=latent.shape[2], k=riflex_freq_index),
            rope_params(1024, 2 * (d // 6), L_test=latent.shape[2], k=riflex_freq_index),
            rope_params(1024, 2 * (d // 6), L_test=latent.shape[2], k=riflex_freq_index)
        ],
        dim=1)

        if not isinstance(cfg, list):
            cfg = [cfg] * (steps +1)
        print(cfg)

        base_args = {
            'clip_fea': image_embeds.get('clip_context', None),
            'seq_len': seq_len,
            'device': device,
            'freqs': freqs,
        }
        if transformer.model_type == "i2v":
            base_args.update({
                 'y': [image_embeds["image_embeds"]],
            })

        arg_c = base_args.copy()
        arg_c.update({'context': [text_embeds["prompt_embeds"][0]]})

        arg_null = base_args.copy()
        arg_null.update({'context': text_embeds["negative_prompt_embeds"]})

        pbar = ProgressBar(steps)

        from latent_preview import prepare_callback
        callback = prepare_callback(patcher, steps)

        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=model["dtype"], enabled=True):
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(device)

                noise_pred_cond = transformer(
                    latent_model_input, t=timestep, **arg_c)[0].to(offload_device)
                if cfg[i] != 1.0:
                    noise_pred_uncond = transformer(
                        latent_model_input, t=timestep, **arg_null)[0].to(offload_device)

                    noise_pred = noise_pred_uncond + cfg[i] * (
                        noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                latent = latent.to(offload_device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(device)]

                if callback is not None:
                    callback_latent = (latent_model_input[0].cpu() - noise_pred * t.cpu() / 1000).detach().permute(1,0,2,3)
                    callback(i, callback_latent, None, steps)
                else:
                    pbar.update(1)
                del latent_model_input, timestep

        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        print_memory(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        return ({
            "samples": x0[0].unsqueeze(0).cpu()
            },)

#region VideoDecode
class WanVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper"

    def decode(self, vae, samples, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()
        latents = samples["samples"]
        vae.to(device)

        latents = latents.to(device = device, dtype = vae.dtype)

        mm.soft_empty_cache()

        image = vae.decode(latents, device=device, tiled=enable_vae_tiling, tile_size=(tile_x, tile_y), tile_stride=(tile_stride_x, tile_stride_y))[0]
        print(image.shape)
        print(image.min(), image.max())
        vae.to(offload_device)

        image = (image - image.min()) / (image.max() - image.min())
        image = torch.clamp(image, 0.0, 1.0)
        image = image.permute(1, 2, 3, 0).cpu().float()

        return (image,)

#region VideoEncode
class WanVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                    "optional": {
                        "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for leapfusion I2V where some noise can add motion and give sharper results"}),
                        "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for leapfusion I2V where lower values allow for more motion"}),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, vae, image, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, noise_aug_strength=0.0, latent_strength=1.0):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        generator = torch.Generator(device=torch.device("cpu"))#.manual_seed(seed)
        vae.to(device)

        image = (image.clone() * 2.0 - 1.0).to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        if noise_aug_strength > 0.0:
            image = add_noise_to_reference_video(image, ratio=noise_aug_strength)

        latents = vae.encode(image, device=device, tiled=enable_vae_tiling, tile_size=(tile_x, tile_y), tile_stride=(tile_stride_x, tile_stride_y))#.latent_dist.sample(generator)
        if latent_strength != 1.0:
            latents *= latent_strength
        #latents = latents * vae.config.scaling_factor

        vae.to(offload_device)
        print("encoded latents shape",latents.shape)


        return ({"samples": latents},)

class WanVideoLatentPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "min_val": ("FLOAT", {"default": -0.15, "min": -1.0, "max": 0.0, "step": 0.0001}),
                 "max_val": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.0001}),
                 "r_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.0001}),
                 "g_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.0001}),
                 "b_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.0001}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("images", "latent_rgb_factors",)
    FUNCTION = "sample"
    CATEGORY = "WanVideoWrapper"

    def sample(self, samples, seed, min_val, max_val, r_bias, g_bias, b_bias):
        mm.soft_empty_cache()

        latents = samples["samples"].clone()
        print("in sample", latents.shape)
        #latent_rgb_factors =[[-0.02531045419704009, -0.00504800612542497, 0.13293717293982546], [-0.03421835830845858, 0.13996708548892614, -0.07081038680118075], [0.011091819063647063, -0.03372949685846012, -0.0698232210116172], [-0.06276524604742019, -0.09322986677909442, 0.01826383612148913], [0.021290659938126788, -0.07719530444034409, -0.08247812477766273], [0.04401102991215147, -0.0026401932105894754, -0.01410913586718443], [0.08979717602613707, 0.05361221258740831, 0.11501425309699129], [0.04695121980405198, -0.13053491609675175, 0.05025986885867986], [-0.09704684176098193, 0.03397687417738002, -0.1105886644677771], [0.14694697234804935, -0.12316902186157716, 0.04210404546699645], [0.14432470831243552, -0.002580008133591355, -0.08490676947390643], [0.051502750076553944, -0.10071695490292451, -0.01786223610178095], [-0.12503276881774464, 0.08877830923879379, 0.1076584501927316], [-0.020191205513213406, -0.1493425056303128, -0.14289740371758308], [-0.06470138952271293, -0.07410426095060325, 0.00980804676890873], [0.11747671720735695, 0.10916082743849789, -0.12235599365235904]]
        latent_rgb_factors = [
        [0.000159, -0.000223, 0.001299],
        [0.000566, 0.000786, 0.001948],
        [0.001531, -0.000337, 0.000863],
        [0.001887, 0.002190, 0.002117],
        [0.002032, 0.000782, -0.000512],
        [0.001634, 0.001260, 0.001685],
        [0.001360, -0.000292, 0.000189],
        [0.001410, 0.000769, 0.001935],
        [-0.000365, 0.000211, 0.000397],
        [-0.000091, 0.001333, 0.001812],
        [0.000201, 0.001866, 0.000546],
        [0.001889, 0.000544, -0.000237],
        [0.001779, 0.000022, 0.001764],
        [0.001456, 0.000431, 0.001574],
        [0.001791, 0.001738, -0.000121],
        [-0.000034, -0.000405, 0.000708]
    ]

        import random
        random.seed(seed)
        #latent_rgb_factors = [[random.uniform(min_val, max_val) for _ in range(3)] for _ in range(16)]
        #latent_rgb_factors = [[0.1 for _ in range(3)] for _ in range(16)]
        out_factors = latent_rgb_factors
        print(latent_rgb_factors)

        latent_rgb_factors_bias = [-0.0011, 0.0, -0.0002]
        #latent_rgb_factors_bias = [r_bias, g_bias, b_bias]

        latent_rgb_factors = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)
        latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)
        print(latent_rgb_factors)

        print("latent_rgb_factors", latent_rgb_factors.shape)

        latent_images = []
        for t in range(latents.shape[2]):
            latent = latents[:, :, t, :, :]
            latent = latent[0].permute(1, 2, 0)
            latent_image = torch.nn.functional.linear(
                latent,
                latent_rgb_factors,
                bias=latent_rgb_factors_bias
            )
            latent_images.append(latent_image)
        latent_images = torch.stack(latent_images, dim=0)
        print("latent_images", latent_images.shape)
        latent_images_min = latent_images.min()
        latent_images_max = latent_images.max()
        latent_images = (latent_images - latent_images_min) / (latent_images_max - latent_images_min)

        return (latent_images.float().cpu(), out_factors)


class WanVideoLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("loras"),
                {"tooltip": "LORA models are expected to be in ComfyUI/models/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks":("SELECTEDBLOCKS", ),
            }
        }

    RETURN_TYPES = ("WANVIDLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora, strength, blocks=None, prev_lora=None, fuse_lora=False):
        loras_list = []

        lora = {
            "path": folder_paths.get_full_path("loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
            "fuse_lora": fuse_lora,
            "blocks": blocks
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora)
        return (loras_list,)


NODE_CLASS_MAPPINGS = {
    "WanVideoSampler": WanVideoSampler,
    "WanVideoDecode": WanVideoDecode,
    "WanVideoTextEncode": WanVideoTextEncode,
    "WanVideoModelLoader": WanVideoModelLoader,
    "WanVideoVAELoader": WanVideoVAELoader,
    "LoadWanVideoT5TextEncoder": LoadWanVideoT5TextEncoder,
    "WanVideoImageClipEncode": WanVideoImageClipEncode,
    "LoadWanVideoClipTextEncoder": LoadWanVideoClipTextEncoder,
    "WanVideoEncode": WanVideoEncode,
    "WanVideoBlockSwap": WanVideoBlockSwap,
    "WanVideoTorchCompileSettings": WanVideoTorchCompileSettings,
    "WanVideoLatentPreview": WanVideoLatentPreview,
    "WanVideoEmptyEmbeds": WanVideoEmptyEmbeds,
    "WanVideoLoraSelect": WanVideoLoraSelect,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoSampler": "WanVideo Sampler",
    "WanVideoDecode": "WanVideo Decode",
    "WanVideoTextEncode": "WanVideo TextEncode",
    "WanVideoTextImageEncode": "WanVideo TextImageEncode (IP2V)",
    "WanVideoModelLoader": "WanVideo Model Loader",
    "WanVideoVAELoader": "WanVideo VAE Loader",
    "LoadWanVideoT5TextEncoder": "Load WanVideo T5 TextEncoder",
    "WanVideoImageClipEncode": "WanVideo ImageClip Encode",
    "LoadWanVideoClipTextEncoder": "Load WanVideo Clip TextEncoder",
    "WanVideoEncode": "WanVideo Encode",
    "WanVideoBlockSwap": "WanVideo BlockSwap",
    "WanVideoTorchCompileSettings": "WanVideo Torch Compile Settings",
    "WanVideoLatentPreview": "WanVideo Latent Preview",
    "WanVideoEmptyEmbeds": "WanVideo Empty Embeds",
    "WanVideoLoraSelect": "WanVideo Lora Select",
    }
