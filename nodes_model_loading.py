import torch
import torch.nn as nn
import os, gc, uuid
from .utils import log, apply_lora
import numpy as np
from tqdm import tqdm
import re

from .wanvideo.modules.model import WanModel, LoRALinearLayer, WanRMSNorm
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.modules.clip import CLIPModel
from .wanvideo.wan_video_vae import WanVideoVAE, WanVideoVAE38
from .custom_linear import _replace_linear

from accelerate import init_empty_weights
from .utils import set_module_tensor_to_device, get_module_memory_mb_per_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar
import comfy.model_base
from comfy.sd import load_lora_for_models
try:
    from .gguf.gguf import _replace_with_gguf_linear, GGUFParameter
    from gguf import GGMLQuantizationType
except Exception:
    pass

script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

try:
    from server import PromptServer
except Exception:
    PromptServer = None

attention_modes = ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn", "sageattn_3", "radial_sage_attention", "sageattn_compiled",
                    "sageattn_ultravico", "comfy"]

#from city96's gguf nodes
def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        log.warning(f"Unknown file list already present on key {key}: {base}")
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])

try:
    from comfy.latent_formats import Wan21, Wan22
    latent_format = Wan21
except: #for backwards compatibility
    log.warning("WARNING: Wan21 latent format not found, update ComfyUI for better live video preview")
    from comfy.latent_formats import HunyuanVideo
    latent_format = HunyuanVideo

class WanVideoModel(torch.nn.Module):
    def __init__(self, model_config, transformer, device=None):
        super().__init__()
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.device = device
        self.current_patcher = None
        self.diffusion_model = transformer
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

class WanVideoModelConfig:
    def __init__(self, latent_format=latent_format):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = latent_format

def filter_state_dict_by_blocks(state_dict, blocks_mapping, layer_filter=[]):
    filtered_dict = {}

    if isinstance(layer_filter, str):
        layer_filters = [layer_filter] if layer_filter else []
    else:
        # Filter out empty strings
        layer_filters = [f for f in layer_filter if f] if layer_filter else []

    #print("layer_filter: ", layer_filters)

    for key in state_dict:
        if not any(filter_str in key for filter_str in layer_filters):
            if 'blocks.' in key:

                block_pattern = key.split('diffusion_model.')[1].split('.', 2)[0:2]
                block_key = f'{block_pattern[0]}.{block_pattern[1]}.'

                if block_key in blocks_mapping:
                    filtered_dict[key] = state_dict[key]
            else:
                filtered_dict[key] = state_dict[key]

    for key in filtered_dict:
        print(key)

    #from safetensors.torch import save_file
    #save_file(filtered_dict, "filtered_state_dict_2.safetensors")

    return filtered_dict

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        # aitoolkit/lycoris format
        if k.startswith("lycoris_blocks_"):
            k = k.replace("lycoris_blocks_", "blocks.")
            k = k.replace("_cross_attn_", ".cross_attn.")
            k = k.replace("_self_attn_", ".self_attn.")
            k = k.replace("_ffn_net_0_proj", ".ffn.0")
            k = k.replace("_ffn_net_2", ".ffn.2")
            k = k.replace("to_out_0", "o")
        # Diffusers format
        if k.startswith('transformer.'):
            k = k.replace('transformer.', 'diffusion_model.')
        if k.startswith('pipe.dit.'): #unianimate-dit/diffsynth
            k = k.replace('pipe.dit.', 'diffusion_model.')
        if k.startswith('blocks.'):
            k = k.replace('blocks.', 'diffusion_model.blocks.')
        if k.startswith('vace_blocks.'):
            k = k.replace('vace_blocks.', 'diffusion_model.vace_blocks.')
        k = k.replace('.default.', '.')
        k = k.replace('.diff_m', '.modulation.diff')
        k = k.replace('base_model.model.', 'diffusion_model.')

        # Fun LoRA format
        if k.startswith('lora_unet__'):
            # Split into main path and weight type parts
            parts = k.split('.')
            main_part = parts[0]  # e.g. lora_unet__blocks_0_cross_attn_k
            weight_type = '.'.join(parts[1:]) if len(parts) > 1 else None  # e.g. lora_down.weight

            # Process the main part - convert from underscore to dot format
            if 'blocks_' in main_part:
                # Extract components
                components = main_part[len('lora_unet__'):].split('_')

                # Start with diffusion_model
                new_key = "diffusion_model"

                # Add blocks.N
                if components[0] == 'blocks':
                    new_key += f".blocks.{components[1]}"

                    # Handle different module types
                    idx = 2
                    if idx < len(components):
                        if components[idx] == 'self' and idx+1 < len(components) and components[idx+1] == 'attn':
                            new_key += ".self_attn"
                            idx += 2
                        elif components[idx] == 'cross' and idx+1 < len(components) and components[idx+1] == 'attn':
                            new_key += ".cross_attn"
                            idx += 2
                        elif components[idx] == 'ffn':
                            new_key += ".ffn"
                            idx += 1

                    # Add the component (k, q, v, o) and handle img suffix
                    if idx < len(components):
                        component = components[idx]
                        idx += 1

                        # Check for img suffix
                        if idx < len(components) and components[idx] == 'img':
                            component += '_img'
                            idx += 1

                        new_key += f".{component}"

                # Handle weight type
                if weight_type:
                    if weight_type == 'alpha':
                        new_key += '.alpha'
                    elif weight_type == 'lora_down.weight' or weight_type == 'lora_down':
                        new_key += '.lora_A.weight'
                    elif weight_type == 'lora_up.weight' or weight_type == 'lora_up':
                        new_key += '.lora_B.weight'
                    else:
                        # Keep original weight type if not matching our patterns
                        new_key += f'.{weight_type}'
                        # Add .weight suffix if missing
                        if not new_key.endswith('.weight'):
                            new_key += '.weight'

                k = new_key
            else:
                # For other lora_unet__ formats (head, embeddings, etc.)
                new_key = main_part.replace('lora_unet__', 'diffusion_model.')

                # Fix specific component naming patterns
                new_key = new_key.replace('_self_attn', '.self_attn')
                new_key = new_key.replace('_cross_attn', '.cross_attn')
                new_key = new_key.replace('_ffn', '.ffn')
                new_key = new_key.replace('blocks_', 'blocks.')
                new_key = new_key.replace('head_head', 'head.head')
                new_key = new_key.replace('img_emb', 'img_emb')
                new_key = new_key.replace('text_embedding', 'text.embedding')
                new_key = new_key.replace('time_embedding', 'time.embedding')
                new_key = new_key.replace('time_projection', 'time.projection')

                # Replace remaining underscores with dots
                parts = new_key.split('.')
                final_parts = []
                for part in parts:
                    if part in ['img_emb', 'self_attn', 'cross_attn']:
                        final_parts.append(part)
                    else:
                        final_parts.append(part.replace('_', '.'))
                new_key = '.'.join(final_parts)

                # Handle weight type
                if weight_type:
                    if weight_type == 'alpha':
                        new_key += '.alpha'
                    elif weight_type == 'lora_down.weight' or weight_type == 'lora_down':
                        new_key += '.lora_A.weight'
                    elif weight_type == 'lora_up.weight' or weight_type == 'lora_up':
                        new_key += '.lora_B.weight'
                    else:
                        new_key += f'.{weight_type}'
                        if not new_key.endswith('.weight'):
                            new_key += '.weight'

                k = new_key

            # Handle special embedded components
            special_components = {
                'time.projection': 'time_projection',
                'img.emb': 'img_emb',
                'text.emb': 'text_emb',
                'time.emb': 'time_emb',
            }
            for old, new in special_components.items():
                if old in k:
                    k = k.replace(old, new)

        # Fix diffusion.model -> diffusion_model
        if k.startswith('diffusion.model.'):
            k = k.replace('diffusion.model.', 'diffusion_model.')

        # Finetrainer format
        if '.attn1.' in k:
            k = k.replace('.attn1.', '.cross_attn.')
            k = k.replace('.to_k.', '.k.')
            k = k.replace('.to_q.', '.q.')
            k = k.replace('.to_v.', '.v.')
            k = k.replace('.to_out.0.', '.o.')
        elif '.attn2.' in k:
            k = k.replace('.attn2.', '.cross_attn.')
            k = k.replace('.to_k.', '.k.')
            k = k.replace('.to_q.', '.q.')
            k = k.replace('.to_v.', '.v.')
            k = k.replace('.to_out.0.', '.o.')

        if "img_attn.proj" in k:
            k = k.replace("img_attn.proj", "img_attn_proj")
        if "img_attn.qkv" in k:
            k = k.replace("img_attn.qkv", "img_attn_qkv")
        if "txt_attn.proj" in k:
            k = k.replace("txt_attn.proj", "txt_attn_proj")
        if "txt_attn.qkv" in k:
            k = k.replace("txt_attn.qkv", "txt_attn_qkv")
        new_sd[k] = v
    return new_sd

def compensate_rs_lora_format(lora_sd):
    rank = lora_sd["base_model.model.blocks.0.cross_attn.k.lora_A.weight"].shape[0]
    alpha = torch.tensor(rank * rank // rank ** 0.5)
    log.info(f"Detected rank stabilized peft lora format with rank {rank}, setting alpha to {alpha} to compensate.")
    new_sd = {}
    for k, v in lora_sd.items():
        if k.endswith(".lora_A.weight"):
            new_sd[k] = v
            new_k = k.replace(".lora_A.weight", ".alpha")
            new_sd[new_k] = alpha
        else:
            new_sd[k] = v
    return new_sd

class WanVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 48, "step": 1, "tooltip": "Number of transformer blocks to swap, the 14B model has 40, while the 1.3B and 5B models have 30 blocks. LongCat-video has 48"}),
                "offload_img_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload img_emb to offload_device"}),
                "offload_txt_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload time_emb to offload_device"}),
            },
            "optional": {
                "use_non_blocking": ("BOOLEAN", {"default": False, "tooltip": "Use non-blocking memory transfer for offloading, reserves more RAM but is faster"}),
                "vace_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1, "tooltip": "Number of VACE blocks to swap, the VACE model has 15 blocks"}),
                "prefetch_blocks": ("INT", {"default": 0, "min": 0, "max": 40, "step": 1, "tooltip": "Number of blocks to prefetch ahead, can speed up processing but increases memory usage. 1 is usually enough to offset speed loss from block swapping, use the debug option to confirm it for your system"}),
                "block_swap_debug": ("BOOLEAN", {"default": False, "tooltip": "Enable debug logging for block swapping"}),
            },
        }
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Settings for block swapping, reduces VRAM use by swapping blocks to CPU memory"

    def setargs(self, **kwargs):
        return (kwargs, )

class WanVideoVRAMManagement:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "offload_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Percentage of parameters to offload"}),
            },
        }
    RETURN_TYPES = ("VRAM_MANAGEMENTARGS",)
    RETURN_NAMES = ("vram_management_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Alternative offloading method from DiffSynth-Studio, more aggressive in reducing memory use than block swapping, but can be slower"

    def setargs(self, **kwargs):
        return (kwargs, )

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
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": True, "tooltip": "Compile only the transformer blocks, usually enough and can make compilation faster and less error prone"}),
            },
            "optional": {
                "dynamo_recompile_limit": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.recompile_limit"}),
                "force_parameter_static_shapes": ("BOOLEAN", {"default": False, "tooltip": "torch._dynamo.config.force_parameter_static_shapes"}),
                "allow_unmerged_lora_compile": ("BOOLEAN", {"default": False, "tooltip": "Allow LoRA application to be compiled with torch.compile to avoid graph breaks, causes issues with some LoRAs, mostly dynamic ones"}),
            },
        }
    RETURN_TYPES = ("WANCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "set_args"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch > 2.7.0 is recommended"

    def set_args(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks_only, dynamo_recompile_limit=128,
                 force_parameter_static_shapes=True, allow_unmerged_lora_compile=False):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
            "compile_transformer_blocks_only": compile_transformer_blocks_only,
            "force_parameter_static_shapes": force_parameter_static_shapes,
            "allow_unmerged_lora_compile": allow_unmerged_lora_compile,
        }

        return (compile_args, )

class WanVideoLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("loras"),
                {"tooltip": "LORA models are expected to be in ComfyUI/models/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks":("SELECTEDBLOCKS", ),
                "low_mem_load": ("BOOLEAN", {"default": False, "tooltip": "Load the LORA model with less VRAM usage, slower loading. This affects ALL LoRAs, not just the current one. No effect if merge_loras is False"}),
                "merge_loras": ("BOOLEAN", {"default": True, "tooltip": "Merge LoRAs into the model, otherwise they are loaded on the fly. Always disabled for GGUF and scaled fp8 models. This affects ALL LoRAs, not just the current one"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("WANVIDLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora, strength, unique_id, blocks={}, prev_lora=None, low_mem_load=False, merge_loras=True):
        if not merge_loras:
            low_mem_load = False  # Unmerged LoRAs don't need low_mem_load
        loras_list = []

        if not isinstance(strength, list):
            strength = round(strength, 4)
            if strength == 0.0:
                if prev_lora is not None:
                    loras_list.extend(prev_lora)
                return (loras_list,)

        try:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora)
        except Exception:
            lora_path = lora

        # Load metadata from the safetensors file
        metadata = {}
        try:
            from safetensors.torch import safe_open
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
        except Exception as e:
            log.info(f"Could not load metadata from {lora}: {e}")

        if unique_id and PromptServer is not None:
            try:
                if metadata:
                    # Build table rows for metadata
                    metadata_rows = ""
                    for key, value in metadata.items():
                        # Format value - handle special cases
                        if isinstance(value, dict):
                            formatted_value = "<pre>" + "\n".join([f"{k}: {v}" for k, v in value.items()]) + "</pre>"
                        elif isinstance(value, (list, tuple)):
                            formatted_value = "<pre>" + "\n".join([str(item) for item in value]) + "</pre>"
                        else:
                            formatted_value = str(value)
                        metadata_rows += f"<tr><td><b>{key}</b></td><td>{formatted_value}</td></tr>"
                    PromptServer.instance.send_progress_text(
                        f"<details>"
                        f"<summary><b>Metadata</b></summary>"
                        f"<table border='0' cellpadding='3'>"
                        f"<tr><td colspan='2'><b>Metadata</b></td></tr>"
                        f"{metadata_rows}"
                        f"</table>"
                        f"</details>",
                        unique_id
                    )
            except Exception as e:
                log.warning(f"Error displaying metadata: {e}")
                pass

        lora = {
            "path": lora_path,
            "strength": strength,
            "name": os.path.splitext(lora)[0],
            "blocks": blocks.get("selected_blocks", {}),
            "layer_filter": blocks.get("layer_filter", ""),
            "low_mem_load": low_mem_load,
            "merge_loras": merge_loras,
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora)
        return (loras_list,)

class WanVideoLoraSelectByName(WanVideoLoraSelect):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora_name": ("STRING", {"default": "", "multiline": False, "tooltip": "Lora filename to load"}),
               "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks":("SELECTEDBLOCKS", ),
                "low_mem_load": ("BOOLEAN", {"default": False, "tooltip": "Load the LORA model with less VRAM usage, slower loading. This affects ALL LoRAs, not just the current one. No effect if merge_loras is False"}),
                "merge_loras": ("BOOLEAN", {"default": True, "tooltip": "Merge LoRAs into the model, otherwise they are loaded on the fly. Always disabled for GGUF and scaled fp8 models. This affects ALL LoRAs, not just the current one"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def getlorapath(self, lora_name, strength, unique_id, blocks={}, prev_lora=None, low_mem_load=False, merge_loras=True):
        lora_list = folder_paths.get_filename_list("loras")
        lora_path = "none"
        for lora in lora_list:
            if lora_name in lora:
                lora_path = lora
                log.info(f"Found LoRA file: {lora_path}")
        return super().getlorapath(
            lora_path, strength, unique_id, blocks=blocks, prev_lora=prev_lora, low_mem_load=low_mem_load, merge_loras=merge_loras
        )

class WanVideoLoraSelectMulti:
    @classmethod
    def INPUT_TYPES(s):
        lora_files = folder_paths.get_filename_list("loras")
        lora_files = ["none"] + lora_files  # Add "none" as the first option
        return {
            "required": {
               "lora_0": (lora_files, {"default": "none"}),
                "strength_0": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "lora_1": (lora_files, {"default": "none"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "lora_2": (lora_files, {"default": "none"}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "lora_3": (lora_files, {"default": "none"}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "lora_4": (lora_files, {"default": "none"}),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks":("SELECTEDBLOCKS", ),
                "low_mem_load": ("BOOLEAN", {"default": False, "tooltip": "Load the LORA model with less VRAM usage, slower loading. No effect if merge_loras is False"}),
                "merge_loras": ("BOOLEAN", {"default": True, "tooltip": "Merge LoRAs into the model, otherwise they are loaded on the fly. Always disabled for GGUF and scaled fp8 models. This affects ALL LoRAs, not just the current one"}),

            }
        }

    RETURN_TYPES = ("WANVIDLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora_0, strength_0, lora_1, strength_1, lora_2, strength_2,
                lora_3, strength_3, lora_4, strength_4, blocks={}, prev_lora=None,
                low_mem_load=False, merge_loras=True):
        if not merge_loras:
            low_mem_load = False  # Unmerged LoRAs don't need low_mem_load
        loras_list = list(prev_lora) if prev_lora else []
        lora_inputs = [
            (lora_0, strength_0),
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
            (lora_4, strength_4)
        ]
        for lora_name, strength in lora_inputs:
            s = round(strength, 4) if not isinstance(strength, list) else strength
            if not lora_name or lora_name == "none" or s == 0.0:
                continue
            loras_list.append({
                "path": folder_paths.get_full_path_or_raise("loras", lora_name),
                "strength": s,
                "name": os.path.splitext(lora_name)[0],
                "blocks": blocks.get("selected_blocks", {}),
                "layer_filter": blocks.get("layer_filter", ""),
                "low_mem_load": low_mem_load,
                "merge_loras": merge_loras,
            })
        if len(loras_list) == 0:
            return None,
        return (loras_list,)

class WanVideoVACEModelSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vace_model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' VACE model to use when not using model that has it included"}),
            },
        }

    RETURN_TYPES = ("VACEPATH",)
    RETURN_NAMES = ("extra_model", )
    FUNCTION = "getvacepath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "VACE model to use when not using model that has it included, loaded from 'ComfyUI/models/diffusion_models'"

    def getvacepath(self, vace_model):
        vace_model = [{"path": folder_paths.get_full_path_or_raise("diffusion_models", vace_model)}]
        return (vace_model,)

class WanVideoExtraModelSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "extra_model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' path to extra state dict to add to the main model"}),
            },
            "optional": {
                "prev_model":("VACEPATH", {"default": None, "tooltip": "For loading multiple extra models"}),
            },
        }

    RETURN_TYPES = ("VACEPATH",)
    RETURN_NAMES = ("extra_model", )
    FUNCTION = "getmodelpath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Extra model to load and add to the main model, ie. VACE or MTV Crafter 'ComfyUI/models/diffusion_models'"

    def getmodelpath(self, extra_model, prev_model=None):
        extra_model = {"path": folder_paths.get_full_path_or_raise("diffusion_models", extra_model)}
        if prev_model is not None and isinstance(prev_model, list):
            extra_model_list = prev_model + [extra_model]
        else:
            extra_model_list = [extra_model]
        return (extra_model_list,)

class WanVideoLoraBlockEdit:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {}
        argument = ("BOOLEAN", {"default": True})

        for i in range(40):
            arg_dict["blocks.{}.".format(i)] = argument

        return {"required": arg_dict, "optional": {"layer_filter": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("SELECTEDBLOCKS", )
    RETURN_NAMES = ("blocks", )
    OUTPUT_TOOLTIPS = ("The modified lora model",)
    FUNCTION = "select"

    CATEGORY = "WanVideoWrapper"

    def select(self, layer_filter=[], **kwargs):
        selected_blocks = {k: v for k, v in kwargs.items() if v is True and isinstance(v, bool)}
        print("Selected blocks LoRA: ", selected_blocks)
        selected = {
            "selected_blocks": selected_blocks,
            "layer_filter": [x.strip() for x in layer_filter.split(",")]
        }
        return (selected,)

def model_lora_keys_unet(model, key_map={}):
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        k = k.replace("_orig_mod.", "")
        if k.startswith("diffusion_model."):
            if k.endswith(".weight"):
                key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k
                key_map["{}".format(k[:-len(".weight")])] = k #generic lora format without any weird key names
            else:
                key_map["{}".format(k)] = k #generic lora format for not .weight without any weird key names

    diffusers_keys = comfy.utils.unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key
            key_map["lycoris_{}".format(key_lora)] = unet_key #simpletuner lycoris format

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(p, k[:-len(".weight")].replace(".to_", ".processor.to_"))
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key

    return key_map

def add_patches(patcher, patches, strength_patch=1.0, strength_model=1.0):
    with patcher.use_ejected():
        p = set()
        model_sd = patcher.model.state_dict()
        for k in patches:
            offset = None
            function = None
            if isinstance(k, str):
                key = k
            else:
                offset = k[1]
                key = k[0]
                if len(k) > 2:
                    function = k[2]

            # Check for key, or key with '._orig_mod' inserted after block number, in model_sd
            key_in_sd = key in model_sd
            key_orig_mod = None
            if not key_in_sd:
                # Try to insert '._orig_mod' after the block number if pattern matches
                parts = key.split('.')
                # Look for 'blocks', block number, then insert
                try:
                    idx = parts.index('blocks')
                    if idx + 1 < len(parts):
                        # Only if the next part is a number
                        if parts[idx+1].isdigit():
                            new_parts = parts[:idx+2] + ['_orig_mod'] + parts[idx+2:]
                            key_orig_mod = '.'.join(new_parts)
                except ValueError:
                    pass
            key_orig_mod_in_sd = key_orig_mod is not None and key_orig_mod in model_sd
            if key_in_sd or key_orig_mod_in_sd:
                actual_key = key if key_in_sd else key_orig_mod
                p.add(k)
                current_patches = patcher.patches.get(actual_key, [])
                current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                patcher.patches[actual_key] = current_patches

        patcher.patches_uuid = uuid.uuid4()
        return list(p)

def load_lora_for_models_mod(model, lora, strength_model):
    key_map = {}
    if model is not None:
        key_map = model_lora_keys_unet(model.model, key_map)

    loaded = comfy.lora.load_lora(lora, key_map)

    new_modelpatcher = model.clone()
    k = add_patches(new_modelpatcher, loaded, strength_model)
    k = set(k)
    for x in loaded:
        if (x not in k):
            log.warning("NOT LOADED {}".format(x))

    return (new_modelpatcher)

class WanVideoSetLoRAs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "model": ("WANVIDEOMODEL", ),
            },
            "optional": {
                "lora": ("WANVIDLORA", ),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "setlora"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True
    DESCRIPTION = "Sets the LoRA weights to be used directly in linear layers of the model, this does NOT merge LoRAs"

    def setlora(self, model, lora=None):
        if lora is None:
            return (model,)

        patcher = model.clone()

        merge_loras = False
        for l in lora:
            merge_loras = l.get("merge_loras", True)
        if merge_loras is True:
            raise ValueError("Set LoRA node does not use low_mem_load and can't merge LoRAs, disable 'merge_loras' in the LoRA select node.")

        patcher.model_options['transformer_options']["lora_scheduling_enabled"] = False
        for l in lora:
            log.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
            lora_path = l["path"]
            lora_strength = l["strength"]
            if isinstance(lora_strength, list):
                if merge_loras:
                    raise ValueError("LoRA strength should be a single value when merge_loras=True")
                patcher.model_options['transformer_options']["lora_scheduling_enabled"] = True
            if lora_strength == 0:
                log.warning(f"LoRA {lora_path} has strength 0, skipping...")
                continue
            lora_sd = load_torch_file(lora_path, safe_load=True)
            if "dwpose_embedding.0.weight" in lora_sd: #unianimate
                raise NotImplementedError("Unianimate LoRA patching is not implemented in this node.")
            if "base_model.model.blocks.0.cross_attn.k.lora_A.weight" in lora_sd: # assume rs_lora
                lora_sd = compensate_rs_lora_format(lora_sd)

            lora_sd = standardize_lora_key_format(lora_sd)
            if l["blocks"]:
                lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"], l.get("layer_filter", []))

            # Filter out any LoRA keys containing 'img' if the base model state_dict has no 'img' keys
            if not any('img' in k for k in model.model.diffusion_model.state_dict().keys()):
                lora_sd = {k: v for k, v in lora_sd.items() if 'img' not in k}

            if "diffusion_model.patch_embedding.lora_A.weight" in lora_sd:
                raise NotImplementedError("Control LoRA patching is not implemented in this node.")

            patcher = load_lora_for_models_mod(patcher, lora_sd, lora_strength)

            del lora_sd

        return (patcher,)

def rename_fuser_block(name):
    # map fuser blocks to main blocks
    new_name = name
    if "face_adapter.fuser_blocks." in name:
        match = re.search(r'face_adapter\.fuser_blocks\.(\d+)\.', name)
        if match:
            fuser_block_num = int(match.group(1))
            main_block_num = fuser_block_num * 5
            new_name = name.replace(f"face_adapter.fuser_blocks.{fuser_block_num}.", f"blocks.{main_block_num}.fuser_block.")
    return new_name

def load_weights(transformer, sd=None, weight_dtype=None, base_dtype=None,
                 transformer_load_device=None, block_swap_args=None, gguf=False, reader=None, patcher=None, compile_args=None):
    params_to_keep = {"time_in", "patch_embedding", "time_", "modulation", "text_embedding",
                      "adapter", "add", "ref_conv", "casual_audio_encoder", "cond_encoder", "frame_packer", "audio_proj_glob", "face_encoder", "fuser_block"}
    param_count = sum(1 for _ in transformer.named_parameters())
    pbar = ProgressBar(param_count)
    block_idx = vace_block_idx = None

    if gguf:
        log.info("Using GGUF to load and assign model weights to device...")

        # Prepare sd from GGUF readers

        # handle possible non-GGUF weights
        extra_sd = {}
        for key, value in sd.items():
            if value.device != torch.device("meta"):
                extra_sd[key] = value

        sd = {}
        all_tensors = []
        for r in reader:
            all_tensors.extend(r.tensors)
        for tensor in all_tensors:
            name = rename_fuser_block(tensor.name)
            if "glob" not in name and "multitalk_audio_proj" not in name and "audio_proj" in name:
                name = name.replace("audio_proj", "multitalk_audio_proj")
            load_device = device
            if "vace_blocks." in name:
                try:
                    vace_block_idx = int(name.split("vace_blocks.")[1].split(".")[0])
                except Exception:
                    vace_block_idx = None
            elif "blocks." in name and "face" not in name:
                try:
                    block_idx = int(name.split("blocks.")[1].split(".")[0])
                except Exception:
                    block_idx = None

            if block_swap_args is not None:
                if block_idx is not None:
                    if block_idx >= len(transformer.blocks) - block_swap_args.get("blocks_to_swap", 0):
                        load_device = offload_device
                elif vace_block_idx is not None:
                    if vace_block_idx >= len(transformer.vace_blocks) - block_swap_args.get("vace_blocks_to_swap", 0):
                        load_device = offload_device

            is_gguf_quant = tensor.tensor_type not in [GGMLQuantizationType.F32, GGMLQuantizationType.F16]
            weights = torch.from_numpy(tensor.data.copy()).to(load_device)
            sd[name] = GGUFParameter(weights, quant_type=tensor.tensor_type) if is_gguf_quant else weights
        sd.update(extra_sd)
        del all_tensors, extra_sd

        if not getattr(transformer, "gguf_patched", False):
            transformer = _replace_with_gguf_linear(
                transformer, base_dtype, sd, patches=patcher.patches, compile_args=compile_args
            )
            transformer.gguf_patched = True
    else:
        log.info("Loading and assigning model weights to device...")
    named_params = transformer.named_parameters()

    for name, param in tqdm(named_params,
            desc=f"Loading transformer parameters to {transformer_load_device}",
            total=param_count,
            leave=True):
        block_idx = vace_block_idx = None
        if name.startswith("vace_blocks."):
            try:
                vace_block_idx = int(name.split("vace_blocks.")[1].split(".")[0])
            except Exception:
                vace_block_idx = None
        elif name.startswith("blocks.") and "face" not in name and "controlnet_blocks." not in name:
            try:
                block_idx = int(name.split("blocks.")[1].split(".")[0])
            except Exception:
                block_idx = None

        if "loras" in name or "uni3c" in name:
            continue

        # GGUF: skip GGUFParameter params
        if gguf and isinstance(param, GGUFParameter):
            continue

        key = name.replace("_orig_mod.", "")
        value=sd[key]
        keep_fp32 = ["patch_embedding", "motion_encoder", "condition_embedding"]

        if gguf:
            dtype_to_use = torch.float32 if "patch_embedding" in name or "motion_encoder" in name else base_dtype
        else:
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else weight_dtype
            dtype_to_use = weight_dtype if value.dtype == weight_dtype else dtype_to_use
            scale_key = key.replace(".weight", ".scale_weight")
            if scale_key in sd:
                dtype_to_use = value.dtype
            if "bias" in name or "img_emb" in name:
                dtype_to_use = base_dtype
            if any(k in name for k in keep_fp32):
                dtype_to_use = torch.float32
            if "modulation" in name or "norm" in name:
                dtype_to_use = value.dtype if value.dtype == torch.float32 else base_dtype

        load_device = transformer_load_device
        if block_swap_args is not None:
            load_device = device
            if block_idx is not None:
                if block_idx >= len(transformer.blocks) - block_swap_args.get("blocks_to_swap", 0):
                    load_device = offload_device
            elif vace_block_idx is not None:
                if vace_block_idx >= len(transformer.vace_blocks) - block_swap_args.get("vace_blocks_to_swap", 0):
                    load_device = offload_device
        # Set tensor to device
        set_module_tensor_to_device(transformer, name, device=load_device, dtype=dtype_to_use, value=value)
        pbar.update(1)

    #[print(name, param.device, param.dtype) for name, param in transformer.named_parameters()]
    memory_on_device = get_module_memory_mb_per_device(transformer)
    log.info("-" * 25)
    log.info("Transformer weights loaded:")
    for dev, mem_mb in memory_on_device.items():
        log.info(f"Device: {dev:8s} | Memory: {mem_mb:,.2f} MB")

    if hasattr(pbar, "_last_sent_value"):
        pbar._last_sent_value = -1
    pbar.update_absolute(0)

def patch_control_lora(transformer, device):
    log.info("Control-LoRA detected, patching model...")

    in_cls = transformer.patch_embedding.__class__ # nn.Conv3d
    old_in_dim = transformer.in_dim # 16
    new_in_dim = 32

    new_in = in_cls(
        new_in_dim,
        transformer.patch_embedding.out_channels,
        transformer.patch_embedding.kernel_size,
        transformer.patch_embedding.stride,
        transformer.patch_embedding.padding,
    ).to(device=device, dtype=torch.float32)

    new_in.weight.zero_()
    new_in.bias.zero_()

    new_in.weight[:, :old_in_dim].copy_(transformer.patch_embedding.weight)
    new_in.bias.copy_(transformer.patch_embedding.bias)

    transformer.patch_embedding = new_in
    transformer.expanded_patch_embedding = new_in

def patch_stand_in_lora(transformer, lora_sd, transformer_load_device, base_dtype, lora_strength):
    if "diffusion_model.blocks.0.self_attn.q_loras.down.weight" in lora_sd:
        log.info("Stand-In LoRA detected")
        for block in transformer.blocks:
            block.self_attn.q_loras = LoRALinearLayer(transformer.dim, transformer.dim, rank=128, device=transformer_load_device, dtype=base_dtype, strength=lora_strength)
            block.self_attn.k_loras = LoRALinearLayer(transformer.dim, transformer.dim, rank=128, device=transformer_load_device, dtype=base_dtype, strength=lora_strength)
            block.self_attn.v_loras = LoRALinearLayer(transformer.dim, transformer.dim, rank=128, device=transformer_load_device, dtype=base_dtype, strength=lora_strength)
            for lora in [block.self_attn.q_loras, block.self_attn.k_loras, block.self_attn.v_loras]:
                for param in lora.parameters():
                    param.requires_grad = False
        for name, param in transformer.named_parameters():
            if "lora" in name:
                param.data.copy_(lora_sd["diffusion_model." + name].to(param.device, dtype=param.dtype))

def add_lora_weights(patcher, lora, base_dtype, merge_loras=False):
    unianimate_sd = None
    control_lora=False
    #spacepxl's control LoRA patch
    for l in lora:
        log.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
        lora_path = l["path"]
        lora_strength = l["strength"]
        if isinstance(lora_strength, list):
            if merge_loras:
                raise ValueError("LoRA strength should be a single value when merge_loras=True")
            patcher.model.diffusion_model.lora_scheduling_enabled = True
        if lora_strength == 0:
            log.warning(f"LoRA {lora_path} has strength 0, skipping...")
            continue
        lora_sd = load_torch_file(lora_path, safe_load=True)
        if "dwpose_embedding.0.weight" in lora_sd: #unianimate
            from .unianimate.nodes import update_transformer
            log.info("Unianimate LoRA detected, patching model...")
            patcher.model.diffusion_model, unianimate_sd = update_transformer(patcher.model.diffusion_model, lora_sd)
        if "base_model.model.blocks.0.cross_attn.k.lora_A.weight" in lora_sd: # assume rs_lora
                lora_sd = compensate_rs_lora_format(lora_sd)
        lora_sd = standardize_lora_key_format(lora_sd)

        if l["blocks"]:
            lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"], l.get("layer_filter", []))

        # Filter out any LoRA keys containing 'img' if the base model state_dict has no 'img' keys
        #if not any('img' in k for k in sd.keys()):
        #    lora_sd = {k: v for k, v in lora_sd.items() if 'img' not in k}

        if "diffusion_model.patch_embedding.lora_A.weight" in lora_sd:
            control_lora = True
        #stand-in LoRA patch
        if "diffusion_model.blocks.0.self_attn.q_loras.down.weight" in lora_sd:
            patch_stand_in_lora(patcher.model.diffusion_model, lora_sd, device, base_dtype, lora_strength)
        # normal LoRA patch
        else:
            patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)

        del lora_sd
    return patcher, control_lora, unianimate_sd

class WanVideoSetAttentionModeOverride:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
                "attention_mode": (attention_modes, {"default": "sdpa"}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Step to start applying the attention mode override"}),
                "end_step": ("INT", {"default": 10000, "min": 1, "max": 10000, "step": 1, "tooltip": "Step to end applying the attention mode override"}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Print verbose info about attention mode override during generation"}),
            },
            "optional": {
                "blocks":("INT", {"forceInput": True} ),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "getmodelpath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Override the attention mode for the model for specific step and/or block range"

    def getmodelpath(self, model, attention_mode, start_step, end_step, verbose, blocks=None):
        model_clone = model.clone()
        attention_mode_override = {
            "mode": attention_mode,
            "start_step": start_step,
            "end_step": end_step,
            "verbose": verbose,
        }
        if blocks is not None:
            attention_mode_override["blocks"] = blocks
        model_clone.model_options['transformer_options']["attention_mode_override"] = attention_mode_override

        return (model_clone,)


class WanVideoUltraVicoSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
                "alpha": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Alpha value for the decay, higher values mean slower decay"}),
            },
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "getmodelpath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Set UltraVico parameters, attention mode still needs to be set to sageattn_ultravico, https://github.com/thu-ml/DiT-Extrapolation"

    def getmodelpath(self, model, alpha):
        model_clone = model.clone()
        model_clone.model_options['transformer_options']["ultravico_alpha"] = alpha

        return (model_clone,)


#region Model loading
class WanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16", "fp16_fast"], {"default": "bf16"}),
            "quantization": (["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e4m3fn_scaled", "fp8_e4m3fn_scaled_fast", "fp8_e5m2", "fp8_e5m2_fast", "fp8_e5m2_scaled", "fp8_e5m2_scaled_fast"], {"default": "disabled",
                            "tooltip": "Optional quantization method, 'disabled' acts as autoselect based by weights. Scaled modes only work with matching weights, _fast modes (fp8 matmul) require CUDA compute capability >= 8.9 (NVIDIA 4000 series and up), e4m3fn generally can not be torch.compiled on compute capability < 8.9 (3000 series and under)"}),
            "load_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            },
            "optional": {
                "attention_mode": (attention_modes, {"default": "sdpa"}),
                "compile_args": ("WANCOMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("WANVIDLORA", {"default": None}),
                "vram_management_args": ("VRAM_MANAGEMENTARGS", {"default": None, "tooltip": "Alternative offloading method from DiffSynth-Studio, more aggressive in reducing memory use than block swapping, but can be slower"}),
                "extra_model": ("VACEPATH", {"default": None, "tooltip": "Extra model to add to the main model, ie. VACE or MTV Crafter"}),
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {"default": None, "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"}),
                "multitalk_model": ("MULTITALKMODEL", {"default": None, "tooltip": "Multitalk model"}),
                "fantasyportrait_model": ("FANTASYPORTRAITMODEL", {"default": None, "tooltip": "FantasyPortrait model"}),
                "rms_norm_function": (["default", "pytorch"], {"default": "default", "tooltip": "RMSNorm function to use, 'pytorch' is the new native torch RMSNorm, which is faster (when not using torch.compile mostly) but changes results slightly. 'default' is the original WanRMSNorm"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device,  quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, vram_management_args=None, extra_model=None, vace_model=None,
                  fantasytalking_model=None, multitalk_model=None, fantasyportrait_model=None, rms_norm_function="default"):
        assert not (vram_management_args is not None and block_swap_args is not None), "Can't use both block_swap_args and vram_management_args at the same time"
        if vace_model is not None:
            extra_model = vace_model
        lora_low_mem_load = merge_loras = False
        if lora is not None:
            merge_loras = any(l.get("merge_loras", True) for l in lora)
            lora_low_mem_load = any(l.get("low_mem_load", False) for l in lora)

        transformer = None
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        gguf = False
        if model.endswith(".gguf"):
            if quantization != "disabled":
                raise ValueError("Quantization should be disabled when loading GGUF models.")
            quantization = "gguf"
            gguf = True
            if merge_loras is True:
                raise ValueError("GGUF models do not support LoRA merging, please disable merge_loras in the LoRA select node.")

        transformer_load_device = device if load_device == "main_device" else offload_device
        if lora is not None and not merge_loras:
            transformer_load_device = offload_device

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        if base_precision == "fp16_fast":
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise ValueError("torch.backends.cuda.matmul.allow_fp16_accumulation is not available in this version of torch, requires torch 2.7.0.dev2025 02 26 nightly minimum currently")
        else:
            try:
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                    torch.backends.cuda.matmul.allow_fp16_accumulation = False
            except Exception:
                pass


        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)

        gguf_reader = None
        if not gguf:
            sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)
        else:
            gguf_reader=[]
            from .gguf.gguf import load_gguf
            sd, reader = load_gguf(model_path)
            gguf_reader.append(reader)

        # Ovi
        extra_audio_model = False
        if any(key.startswith("video_model.") for key in sd.keys()):
            sd = {key.replace("video_model.", "", 1).replace("modulation.modulation", "modulation"): value for key, value in sd.items()}
        if any(key.startswith("audio_model.") for key in sd.keys()) and any(key.startswith("blocks.") for key in sd.keys()):
            extra_audio_model = True


        is_wananimate = "pose_patch_embedding.weight" in sd
        # rename WanAnimate face fuser block keys to insert into main blocks instead
        if is_wananimate:
            for key in list(sd.keys()):
                new_key = rename_fuser_block(key)
                if new_key != key:
                    sd[new_key] = sd.pop(key)

        is_scaled_fp8 = False

        if quantization == "disabled":
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float8_e4m3fn:
                        quantization = "fp8_e4m3fn"
                        if "scaled_fp8" in sd:
                            is_scaled_fp8 = True
                            quantization = "fp8_e4m3fn_scaled"
                        break
                    elif v.dtype == torch.float8_e5m2:
                        quantization = "fp8_e5m2"
                        if "scaled_fp8" in sd:
                            is_scaled_fp8 = True
                            quantization = "fp8_e5m2_scaled"
                        break

        scale_weights = {}
        if "fp8" in quantization:
            for k, v in sd.items():
                if k.endswith(".scale_weight") or k.endswith(".weight_scale"):
                    is_scaled_fp8 = True
                    break

        if is_scaled_fp8 and "scaled" not in quantization:
            quantization = quantization + "_scaled"

        if torch.cuda.is_available():
            #only warning for now
            major, minor = torch.cuda.get_device_capability(device)
            log.info(f"CUDA Compute Capability: {major}.{minor}")
            if compile_args is not None and "e4" in quantization and (major, minor) < (8, 9):
                log.warning("WARNING: Torch.compile with fp8_e4m3fn weights on CUDA compute capability < 8.9 may not be supported. Please use fp8_e5m2, GGUF or higher precision instead, or check the latest triton version that adds support for older architectures https://github.com/woct0rdho/triton-windows/releases/tag/v3.5.0-windows.post21")

        if is_scaled_fp8 and "scaled" not in quantization:
            raise ValueError("The model is a scaled fp8 model, please set quantization to '_scaled'")
        if not is_scaled_fp8 and "scaled" in quantization:
            raise ValueError("The model is not a scaled fp8 model, please disable '_scaled' in quantization")

        if "vace_blocks.0.after_proj.weight" in sd and not "patch_embedding.weight" in sd:
            raise ValueError("You are attempting to load a VACE module as a WanVideo model, instead you should use the vace_model input and matching T2V base model")

        # currently this can be VACE, MTV-Crafter, Lynx or Ovi-audio weights
        if extra_model is not None:
            for _model in extra_model:
                log.info(f"Loading extra model: {_model['path']}")
                if gguf:
                    if not _model["path"].endswith(".gguf"):
                        raise ValueError("With GGUF main model the extra model must also be GGUF quantized, if the main model already has VACE included, you can disconnect the extra module loader")
                    extra_sd, extra_reader = load_gguf(_model["path"])
                    gguf_reader.append(extra_reader)
                    del extra_reader
                else:
                    if _model["path"].endswith(".gguf"):
                        raise ValueError("With GGUF extra model the main model must also be GGUF quantized model")
                    extra_sd = load_torch_file(_model["path"], device=transformer_load_device, safe_load=True)
                if "audio_model.patch_embedding.0.weight" in extra_sd:
                    extra_audio_model = True
                sd.update(extra_sd)
                del extra_sd

        first_key = next(iter(sd))
        if first_key.startswith("audio_model.") and not extra_audio_model:
            sd = {key.replace("audio_model.", "", 1): value for key, value in sd.items()}
        if first_key.startswith("model.diffusion_model."):
            sd = {key.replace("model.diffusion_model.", "", 1): value for key, value in sd.items()}
        elif first_key.startswith("model."):
            sd = {key.replace("model.", "", 1): value for key, value in sd.items()}

        if "patch_embedding.weight" in sd:
            dim = sd["patch_embedding.weight"].shape[0]
            in_channels = sd["patch_embedding.weight"].shape[1]
        elif "patch_embedding.0.weight" in sd:
            dim = sd["patch_embedding.0.weight"].shape[0]
            in_channels = sd["patch_embedding.0.weight"].shape[1]
        else:
            raise ValueError("No patch_embedding weight found, is the selected model a full WanVideo model?")

        in_features = sd["blocks.0.self_attn.k.weight"].shape[1]
        out_features = sd["blocks.0.self_attn.k.weight"].shape[0]
        log.info(f"Detected model in_channels: {in_channels}")

        if "blocks.0.ffn.0.bias" in sd:
            ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]
            ffn2_dim = sd["blocks.0.ffn.2.weight"].shape[1]
        else:
            ffn_dim = sd["blocks.0.ffn.w1.weight"].shape[0]
            ffn2_dim = sd["blocks.0.ffn.w1.weight"].shape[1]

        patch_size=(1, 2, 2)
        if "patch_embedding.0.weight" in sd:
            patch_size = [1]

        is_humo = "audio_proj.audio_proj_glob_1.layer.weight" in sd
        is_wananimate = "pose_patch_embedding.weight" in sd

        #lynx
        lynx_ip_layers = lynx_ref_layers = None
        if "blocks.0.self_attn.ref_adapter.to_k_ref.weight" in sd:
            log.info("Lynx full reference adapter detected")
            lynx_ref_layers = "full"
        if "blocks.0.cross_attn.ip_adapter.registers" in sd:
            log.info("Lynx full IP adapter detected")
            lynx_ip_layers = "full"
        elif "blocks.0.cross_attn.ip_adapter.to_v_ip.weight" in sd:
            log.info("Lynx lite IP adapter detected")
            lynx_ip_layers = "lite"

        model_type = "t2v"
        if not "text_embedding.0.weight" in sd:
            model_type = "no_cross_attn" #minimaxremover
        elif "model_type.Wan2_1-FLF2V-14B-720P" in sd or "img_emb.emb_pos" in sd or "flf2v" in model.lower():
            model_type = "fl2v"
        if "blocks.0.cross_attn.k_img.weight" in sd:
            model_type = "i2v"
        elif in_channels == 16:
            model_type = "t2v"
        elif "control_adapter.conv.weight" in sd:
            model_type = "t2v"
        if "audio_injector.injector.0.k.weight" in sd:
            model_type = "s2v"

        out_dim = 16
        if dim == 5120: #14B
            num_heads = 40
            num_layers = 40
        elif dim == 3072: #5B
            num_heads = 24
            num_layers = 30
            out_dim = 48
            model_type = "t2v" #5B no img crossattn
        elif dim == 4096: #longcat
            num_heads = 32
            num_layers = 48
        else: #1.3B
            num_heads = 12
            num_layers = 30

        vace_layers, vace_in_dim = None, None
        if "vace_blocks.0.after_proj.weight" in sd:
            if in_channels != 16:
                raise ValueError("VACE only works properly with T2V models.")
            model_type = "t2v"
            if dim == 5120:
                vace_layers = [0, 5, 10, 15, 20, 25, 30, 35]
            else:
                vace_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
            vace_in_dim = 96

        log.info(f"Model cross attention type: {model_type}, num_heads: {num_heads}, num_layers: {num_layers}")

        teacache_coefficients_map = {
            "1_3B": {
                "e": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
                "e0": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            },
            "14B": {
                "e": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
                "e0": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            },
            "i2v_480": {
                "e": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
                "e0": [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            },
            "i2v_720":{
                "e": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
                "e0": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
            },
            # Placeholders until TeaCache for Wan2.2 is obtained
            "14B_2.2": {
                "e": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
                "e0": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            },
            "i2v_14B_2.2":{
                "e": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
                "e0": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
            },
        }

        magcache_ratios_map = {
            "1_3B": np.array([1.0]*2+[1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939]),
            "14B": np.array([1.0]*2+[1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]),
            "i2v_480": np.array([1.0]*2+[0.98783, 0.98993, 0.97559, 0.97593, 0.98311, 0.98319, 0.98202, 0.98225, 0.9888, 0.98878, 0.98762, 0.98759, 0.98957, 0.98971, 0.99052, 0.99043, 0.99383, 0.99384, 0.98857, 0.9886, 0.99065, 0.99068, 0.98845, 0.98847, 0.99057, 0.99057, 0.98957, 0.98961, 0.98601, 0.9861, 0.98823, 0.98823, 0.98756, 0.98759, 0.98808, 0.98814, 0.98721, 0.98724, 0.98571, 0.98572, 0.98543, 0.98544, 0.98157, 0.98165, 0.98411, 0.98413, 0.97952, 0.97953, 0.98149, 0.9815, 0.9774, 0.97742, 0.97825, 0.97826, 0.97355, 0.97361, 0.97085, 0.97087, 0.97056, 0.97055, 0.96588, 0.96587, 0.96113, 0.96124, 0.9567, 0.95681, 0.94961, 0.94969, 0.93973, 0.93988, 0.93217, 0.93224, 0.91878, 0.91896, 0.90955, 0.90954, 0.92617, 0.92616]),
            "i2v_720": np.array([1.0]*2+[0.99428, 0.99498, 0.98588, 0.98621, 0.98273, 0.98281, 0.99018, 0.99023, 0.98911, 0.98917, 0.98646, 0.98652, 0.99454, 0.99456, 0.9891, 0.98909, 0.99124, 0.99127, 0.99102, 0.99103, 0.99215, 0.99212, 0.99515, 0.99515, 0.99576, 0.99572, 0.99068, 0.99072, 0.99097, 0.99097, 0.99166, 0.99169, 0.99041, 0.99042, 0.99201, 0.99198, 0.99101, 0.99101, 0.98599, 0.98603, 0.98845, 0.98844, 0.98848, 0.98851, 0.98862, 0.98857, 0.98718, 0.98719, 0.98497, 0.98497, 0.98264, 0.98263, 0.98389, 0.98393, 0.97938, 0.9794, 0.97535, 0.97536, 0.97498, 0.97499, 0.973, 0.97301, 0.96827, 0.96828, 0.96261, 0.96263, 0.95335, 0.9534, 0.94649, 0.94655, 0.93397, 0.93414, 0.91636, 0.9165, 0.89088, 0.89109, 0.8679, 0.86768]),
            "14B_2.2": np.array([1.0]*2+[0.99505, 0.99389, 0.99441, 0.9957, 0.99558, 0.99551, 0.99499, 0.9945, 0.99534, 0.99548, 0.99468, 0.9946, 0.99463, 0.99458, 0.9946, 0.99453, 0.99408, 0.99404, 0.9945, 0.99441, 0.99409, 0.99398, 0.99403, 0.99397, 0.99382, 0.99377, 0.99349, 0.99343, 0.99377, 0.99378, 0.9933, 0.99328, 0.99303, 0.99301, 0.99217, 0.99216, 0.992, 0.99201, 0.99201, 0.99202, 0.99133, 0.99132, 0.99112, 0.9911, 0.99155, 0.99155, 0.98958, 0.98957, 0.98959, 0.98958, 0.98838, 0.98835, 0.98826, 0.98825, 0.9883, 0.98828, 0.98711, 0.98709, 0.98562, 0.98561, 0.98511, 0.9851, 0.98414, 0.98412, 0.98284, 0.98282, 0.98104, 0.98101, 0.97981, 0.97979, 0.97849, 0.97849, 0.97557, 0.97554, 0.97398, 0.97395, 0.97171, 0.97166, 0.96917, 0.96913, 0.96511, 0.96507, 0.96263, 0.96257, 0.95839, 0.95835, 0.95483, 0.95475, 0.94942, 0.94936, 0.9468, 0.94678, 0.94583, 0.94594, 0.94843, 0.94872, 0.96949, 0.97015]),
            "i2v_14B_2.2": np.array([1.0]*2+[0.99512, 0.99559, 0.99559, 0.99561, 0.99595, 0.99577, 0.99512, 0.99512, 0.99546, 0.99534, 0.99543, 0.99531, 0.99496, 0.99491, 0.99504, 0.99499, 0.99444, 0.99449, 0.99481, 0.99481, 0.99435, 0.99435, 0.9943, 0.99431, 0.99411, 0.99406, 0.99373, 0.99376, 0.99413, 0.99405, 0.99363, 0.99359, 0.99335, 0.99331, 0.99244, 0.99243, 0.99229, 0.99229, 0.99239, 0.99236, 0.99163, 0.9916, 0.99149, 0.99151, 0.99191, 0.99192, 0.9898, 0.98981, 0.9899, 0.98987, 0.98849, 0.98849, 0.98846, 0.98846, 0.98861, 0.98861, 0.9874, 0.98738, 0.98588, 0.98589, 0.98539, 0.98534, 0.98444, 0.98439, 0.9831, 0.98309, 0.98119, 0.98118, 0.98001, 0.98, 0.97862, 0.97859, 0.97555, 0.97558, 0.97392, 0.97388, 0.97152, 0.97145, 0.96871, 0.9687, 0.96435, 0.96434, 0.96129, 0.96127, 0.95639, 0.95638, 0.95176, 0.95175, 0.94446, 0.94452, 0.93972, 0.93974, 0.93575, 0.9359, 0.93537, 0.93552, 0.96655, 0.96616]),
        }

        model_variant = "14B" #default to this
        if model_type == "i2v" or model_type == "fl2v":
            if "480" in model or "fun" in model.lower() or "a2" in model.lower() or "540" in model: #just a guess for the Fun model for now...
                model_variant = "i2v_480"
            elif "720" in model:
                model_variant = "i2v_720"
        elif model_type == "t2v":
            model_variant = "14B"

        if dim == 1536:
            model_variant = "1_3B"
        if dim == 3072:
            log.info("5B model detected, no Teacache or MagCache coefficients available, consider using EasyCache for this model")

        if "high" in model.lower() or "low" in model.lower():
            if "i2v" in model.lower():
                model_variant = "i2v_14B_2.2"
            else:
                model_variant = "14B_2.2"

        log.info(f"Model variant detected: {model_variant}")

        TRANSFORMER_CONFIG= {
            "dim": dim,
            "in_features": in_features,
            "out_features": out_features,
            "patch_size": patch_size,
            "ffn_dim": ffn_dim,
            "ffn2_dim": ffn2_dim,
            "eps": 1e-06,
            "freq_dim": 256,
            "in_dim": in_channels,
            "model_type": model_type,
            "out_dim": out_dim,
            "text_len": 512,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attention_mode": attention_mode,
            "rope_func": "comfy",
            "main_device": device,
            "offload_device": offload_device,
            "dtype": base_dtype,
            "teacache_coefficients": teacache_coefficients_map[model_variant],
            "magcache_ratios": magcache_ratios_map[model_variant],
            "vace_layers": vace_layers,
            "vace_in_dim": vace_in_dim,
            "inject_sample_info": True if "fps_embedding.weight" in sd else False,
            "add_ref_conv": True if "ref_conv.weight" in sd else False,
            "in_dim_ref_conv": sd["ref_conv.weight"].shape[1] if "ref_conv.weight" in sd else None,
            "add_control_adapter": True if "control_adapter.conv.weight" in sd else False,
            "use_motion_attn": True if "blocks.0.motion_attn.k.weight" in sd else False,
            "enable_adain": True if "audio_injector.injector_adain_layers.0.linear.weight" in sd else False,
            "cond_dim": sd["cond_encoder.weight"].shape[1] if "cond_encoder.weight" in sd else 0,
            "zero_timestep": model_type == "s2v",
            "humo_audio": is_humo,
            "is_wananimate": is_wananimate,
            "rms_norm_function": rms_norm_function,
            "lynx_ip_layers": lynx_ip_layers,
            "lynx_ref_layers": lynx_ref_layers,
            "is_longcat": dim == 4096,

        }

        with init_empty_weights():
            transformer = WanModel(**TRANSFORMER_CONFIG).eval()

        if extra_audio_model:
            log.info("Ovi extra audio model detected, initializing...")
            TRANSFORMER_CONFIG.update({
                "patch_size": [1],
                "in_dim": 20,
                "out_dim": 20,
                })

            with init_empty_weights():
                transformer.audio_model = WanModel(**TRANSFORMER_CONFIG).eval()

            from .wanvideo.modules.model import WanLayerNorm

            for block in transformer.blocks:
                block.cross_attn.k_fusion = nn.Linear(block.dim, block.dim)
                block.cross_attn.v_fusion = nn.Linear(block.dim, block.dim)
                block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(block.dim, elementwise_affine=True)
                block.cross_attn.norm_k_fusion = WanRMSNorm(block.dim, eps=1e-6) if block.qk_norm else nn.Identity()

            for block in transformer.audio_model.blocks:
                block.cross_attn.k_fusion = nn.Linear(block.dim, block.dim)
                block.cross_attn.v_fusion = nn.Linear(block.dim, block.dim)
                block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(block.dim, elementwise_affine=True)
                block.cross_attn.norm_k_fusion = WanRMSNorm(block.dim, eps=1e-6) if block.qk_norm else nn.Identity()

        #ReCamMaster
        if "blocks.0.cam_encoder.weight" in sd:
            log.info("ReCamMaster model detected, patching model...")
            for block in transformer.blocks:
                block.cam_encoder = nn.Linear(12, dim)
                block.projector = nn.Linear(dim, dim)
                block.cam_encoder.weight.data.zero_()
                block.cam_encoder.bias.data.zero_()
                block.projector.weight = nn.Parameter(torch.eye(dim))
                block.projector.bias = nn.Parameter(torch.zeros(dim))

        # FantasyTalking https://github.com/Fantasy-AMAP
        if fantasytalking_model is not None:
            log.info("FantasyTalking model detected, patching model...")
            context_dim = fantasytalking_model["sd"]["proj_model.proj.weight"].shape[0]
            for block in transformer.blocks:
                block.cross_attn.k_proj = nn.Linear(context_dim, dim, bias=False)
                block.cross_attn.v_proj = nn.Linear(context_dim, dim, bias=False)
            sd.update(fantasytalking_model["sd"])

        # FantasyPortrait https://github.com/Fantasy-AMAP/fantasy-portrait/
        if fantasyportrait_model is not None and "blocks.0.cross_attn.emo_k_proj.weight" not in sd:
            log.info("FantasyPortrait model detected, patching model...")
            context_dim = fantasyportrait_model["sd"]["ip_adapter.blocks.0.cross_attn.ip_adapter_single_stream_k_proj.weight"].shape[1]

            with init_empty_weights():
                for block in transformer.blocks:
                    block.cross_attn.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, dim, bias=False)
                    block.cross_attn.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, dim, bias=False)
            ip_adapter_sd = {}
            for k, v in fantasyportrait_model["sd"].items():
                if k.startswith("ip_adapter."):
                    ip_adapter_sd[k.replace("ip_adapter.", "")] = v
            sd.update(ip_adapter_sd)
            del ip_adapter_sd

        # FlashPortrait
        if "blocks.0.cross_attn.emo_k_proj.weight" in sd:
            log.info("FlashPortrait model detected, patching model...")
            context_dim = sd["blocks.0.cross_attn.emo_k_proj.weight"].shape[1]

            sd = {k.replace("emo_k_proj", "ip_adapter_single_stream_k_proj"): v for k, v in sd.items()}
            sd = {k.replace("emo_v_proj", "ip_adapter_single_stream_v_proj"): v for k, v in sd.items()}

            with init_empty_weights():
                for block in transformer.blocks:
                    block.cross_attn.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, dim, bias=False)
                    block.cross_attn.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, dim, bias=False)

        # LongCat Avatar
        proj1_key = "multitalk_audio_proj.proj1.weight" if "multitalk_audio_proj.proj1.weight" in sd \
                    else "multitalk_audio_proj.proj1.weight_int8" if "multitalk_audio_proj.proj1.weight_int8" in sd \
                    else None
        if proj1_key is not None and ("blocks.0.audio_cross_attn.q_norm.weight" in sd or "blocks.0.audio_cross_attn.q_norm.weight_int8" in sd):
            log.info("MultiTalk/InfiniteTalk model detected, patching model...")
            from .multitalk.multitalk import AudioProjModel
            from .wanvideo.modules.model import WanLayerNorm
            from .LongCat.layers import SingleStreamAttention

            # Detect LongCat-Avatar audio encoder variant from proj1 input dim:
            #   v1.0 (wav2vec2): seq_len * blocks * channels = 5 * 12 * 768 = 46080
            #   v1.5 (whisper):  seq_len * blocks * channels = 5 *  5 * 1280 = 32000
            proj1_in = sd[proj1_key].shape[1]
            if proj1_in == 32000:
                audio_proj_blocks, audio_proj_channels = 5, 1280
                log.info("LongCat-Avatar-1.5 (Whisper) audio proj detected")
            else:
                audio_proj_blocks, audio_proj_channels = 12, 768

            for block in transformer.blocks:
                with init_empty_weights():
                    if "blocks.0.audio_modulation.1.weight" in sd:
                        block.audio_modulation = nn.Sequential(nn.SiLU(), nn.Linear(512, 3 * dim, bias=True))
                    block.norm_x = WanLayerNorm(dim, transformer.eps, elementwise_affine=True)
                    block.audio_cross_attn = SingleStreamAttention(
                            dim=dim,
                            encoder_hidden_states_dim=768,
                            num_heads=num_heads,
                        qkv_bias=True,
                        qk_norm=True,
                        class_range=24,
                        class_interval=4,
                        attention_mode=attention_mode,
                    )
                    multitalk_proj_model = AudioProjModel(blocks=audio_proj_blocks, channels=audio_proj_channels)
            transformer.multitalk_audio_proj = multitalk_proj_model
        # SkyreelsV3
        elif "blocks.1.audio_cross_attn.kv_linear.weight" in sd and "audio_proj.proj1.weight" in sd:
            sd = {k.replace("audio_proj", "multitalk_audio_proj"): v for k, v in sd.items()}
            # init audio module
            from .multitalk.multitalk import SingleStreamMultiAttention, AudioProjModel
            from .wanvideo.modules.model import WanLayerNorm

            for block in transformer.blocks:
                with init_empty_weights():
                    block.norm_x = WanLayerNorm(dim, transformer.eps, elementwise_affine=True)
                    block.audio_cross_attn = SingleStreamMultiAttention(dim=dim, num_heads=num_heads, attention_mode=attention_mode)

            transformer.multitalk_audio_proj = AudioProjModel()
        elif multitalk_model is not None:
            multitalk_model_type = multitalk_model.get("model_type", "MultiTalk")
            log.info(f"{multitalk_model_type} detected, patching model...")

            multitalk_model_path = multitalk_model["model_path"]
            if multitalk_model_path.endswith(".gguf") and not gguf:
                raise ValueError("Multitalk/InfiniteTalk model is a GGUF model, main model also has to be a GGUF model.")
            if "scaled" in multitalk_model and gguf:
                raise ValueError("fp8 scaled Multitalk/InfiniteTalk model can't be used with GGUF main model")

            # init audio module
            from .multitalk.multitalk import SingleStreamMultiAttention
            from .wanvideo.modules.model import WanLayerNorm

            for block in transformer.blocks:
                with init_empty_weights():
                    block.norm_x = WanLayerNorm(dim, transformer.eps, elementwise_affine=True)
                    block.audio_cross_attn = SingleStreamMultiAttention(dim=dim, num_heads=num_heads, attention_mode=attention_mode)
            transformer.multitalk_audio_proj = multitalk_model["proj_model"]
            transformer.multitalk_model_type = multitalk_model_type

            extra_model_path = multitalk_model["model_path"]
            extra_sd = {}
            if multitalk_model_path.endswith(".gguf"):
                extra_sd_temp, extra_reader = load_gguf(extra_model_path)
                gguf_reader.append(extra_reader)
                del extra_reader
            else:
                extra_sd_temp = load_torch_file(extra_model_path, device=transformer_load_device, safe_load=True)

            for k, v in extra_sd_temp.items():
                extra_sd[k.replace("audio_proj.", "multitalk_audio_proj.")] = v

            sd.update(extra_sd)
            del extra_sd

        sd = {k.replace(".weight_scale", ".scale_weight"): v for k, v in sd.items()}

        # FlashVSR
        if "LQ_proj_in.norm1.gamma" in sd:
            log.info("FlashVSR model detected, patching model...")
            from .FlashVSR.LQ_proj_model import Buffer_LQ4x_Proj
            transformer.LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)

        # Additional cond latents
        if "add_conv_in.weight" in sd:
            inner_dim = sd["add_conv_in.weight"].shape[0]
            add_cond_in_dim = sd["add_conv_in.weight"].shape[1]
            attn_cond_in_dim = sd["attn_conv_in.weight"].shape[1]
            transformer.add_conv_in = nn.Conv3d(add_cond_in_dim, inner_dim, kernel_size=transformer.patch_size, stride=transformer.patch_size)
            transformer.add_proj = nn.Linear(inner_dim, inner_dim)
            transformer.attn_conv_in = nn.Conv3d(attn_cond_in_dim, inner_dim, kernel_size=transformer.patch_size, stride=transformer.patch_size)

        # Bindweave text_projection
        if "text_projection.0.weight" in sd:
            log.info("Bindweave model detected, adding text_projection to the model")
            text_dim = sd["text_projection.0.weight"].shape[0]
            transformer.text_projection = nn.Sequential(nn.Linear(sd["text_projection.0.weight"].shape[1], text_dim), nn.GELU(approximate='tanh'), nn.Linear(text_dim, text_dim))

        latent_format=Wan22 if dim == 3072 else Wan21
        comfy_model = WanVideoModel(WanVideoModelConfig(latent_format=latent_format), device=device, transformer=transformer)

        # SteadyDancer
        if "condition_embedding_align.cross_attn.in_proj_bias" in sd:
            from .steadydancer.mobilenetv2_dcd import DYModule
            from .steadydancer.small_archs import PoseRefNetNoBNV3, FactorConv3d
            in_dim_c = 16
            transformer.patch_embedding_fuse = nn.Conv3d(in_channels + in_dim_c + in_dim_c, dim, kernel_size=patch_size, stride=patch_size) # x, fused pose, aligned pose
            transformer.patch_embedding_ref_c = nn.Conv3d(in_dim_c, dim, kernel_size=patch_size, stride=patch_size) # ref_c
            transformer.condition_embedding_spatial = DYModule(inp=in_dim_c, oup=in_dim_c) # Spatial Structure Adaptive Extractor
            transformer.condition_embedding_temporal = nn.Sequential( # Temporal Motion Coherence Module
                FactorConv3d(in_channels=in_dim_c, out_channels=in_dim_c, kernel_size=(3, 3, 3), stride=1), nn.SiLU(),
                FactorConv3d(in_channels=in_dim_c, out_channels=in_dim_c, kernel_size=(3, 3, 3), stride=1), nn.SiLU(),
                FactorConv3d(in_channels=in_dim_c, out_channels=in_dim_c, kernel_size=(3, 3, 3), stride=1), nn.SiLU())
            transformer.condition_embedding_align = PoseRefNetNoBNV3(in_channels_x=16, in_channels_c=16, hidden_dim=128, num_heads=8) # Frame-wise Attention Alignment Unit

        # SCAIL
        if "patch_embedding_pose.weight" in sd:
            log.info("SCAIL model detected, patching model...")
            pose_dim = sd["patch_embedding_pose.weight"].shape[1]
            transformer.patch_embedding_pose = nn.Conv3d(pose_dim, dim, kernel_size=patch_size, stride=patch_size)

        if "image_to_cond.conv_in.bias" in sd:
            # One-to-all
            from .onetoall.controlnet import MiniHunyuanEncoder, MiniEncoder2D
            from .onetoall.refextractor_2d import WanRefextractor, WanAttentionBlock

            controlnet_layers = len({k.split(".")[2] for k in sd if k.startswith("controlnet.blocks.")})
            refextractor_layers = len({k.split(".")[2] for k in sd if k.startswith("refextractor.blocks.")})
            log.info(f"{controlnet_layers} One-to-all controlnet layers and {refextractor_layers} refextractor layers detected, patching model...")

            with init_empty_weights():
                transformer.image_to_cond = MiniEncoder2D(
                    in_channels = sd["image_to_cond.conv_in.bias"].shape[0],
                    out_channels = in_channels,
                    down_block_types= ("DownEncoderBlockInflated","DownEncoderBlockInflated","DownEncoderBlockInflated"),
                    block_out_channels=(16, 16, 16),
                    norm_num_groups = 4,
                    layers_per_block = 1,
                    spatial_compression_ratio=1
                )

                transformer.input_hint_block = MiniHunyuanEncoder(
                    in_channels=3,
                    out_channels=in_channels,
                    block_out_channels=(16, 16, 16, 16),
                    norm_num_groups=4,
                    layers_per_block=1,
                    spatial_compression_ratio=16
                )

                transformer.controlnet = nn.Module()
                transformer.controlnet.blocks = nn.ModuleList([WanAttentionBlock(in_features, out_features, ffn_dim, ffn2_dim, num_heads) for _ in range(controlnet_layers)])
                transformer.controlnet_zero = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(controlnet_layers)])
                transformer.refextractor = WanRefextractor(
                    patch_size=(1, 2, 2), in_dim=sd["refextractor.patch_embedding.weight"].shape[1],
                    dim=dim, in_features=in_features, out_features=out_features, ffn_dim=ffn_dim, ffn2_dim=ffn2_dim,
                        num_heads=num_heads, num_layers=refextractor_layers)

                for block in transformer.blocks:
                    block.ref_attn_k_img = nn.Linear(in_features, out_features)
                    block.ref_attn_v_img = nn.Linear(in_features, out_features)
                    block.ref_attn_norm_k_img = WanRMSNorm(out_features, eps=1e-6)

        if "blocks.0.control_blocks_dense.cross_attn.k.weight" in sd:
            log.info("LongVie2 model detected, patching model...")
            from .LongVie2.modules import WanModelDualControl
            control_layers = 12
            with init_empty_weights():
                dual_controller = WanModelDualControl(dim=5120, ffn_dim=13824, eps=1e-06, num_heads=40, control_layers=control_layers)
                for b in range(control_layers):
                    transformer.blocks[b].control_blocks_dense = dual_controller.control_blocks_dense[b]
                    transformer.blocks[b].control_blocks_sparse = dual_controller.control_blocks_sparse[b]
                    transformer.blocks[b].control_combine_linears = dual_controller.control_combine_linears[b]
                transformer.dual_controller = nn.Module()
                transformer.dual_controller.control_initial_combine_linear_dense = dual_controller.control_initial_combine_linear_dense
                transformer.dual_controller.control_initial_combine_linear_sparse = dual_controller.control_initial_combine_linear_sparse
                transformer.dual_controller.control_t_mod = dual_controller.control_t_mod
                transformer.dual_controller.control_text_linear = dual_controller.control_text_linear
                transformer.dual_controller_freqs = dual_controller.freqs


        comfy_model.diffusion_model = transformer
        comfy_model.load_device = transformer_load_device
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
        patcher.model.is_patched = False

        scale_weights = {}
        if "fp8" in quantization:
            for k, v in sd.items():
                if k.endswith(".scale_weight"):
                    scale_weights[k] = v.to(device, base_dtype)

        if quantization in ["fp8_e4m3fn", "fp8_e4m3fn_fast"]:
            weight_dtype = torch.float8_e4m3fn
        elif quantization in ["fp8_e5m2", "fp8_e5m2_fast"]:
            weight_dtype = torch.float8_e5m2
        else:
            weight_dtype = base_dtype

        params_to_keep = {"norm", "bias", "time_in", "patch_embedding", "time_", "img_emb", "modulation", "text_embedding", "adapter", "add", "ref_conv", "audio_proj"}

        control_lora = False

        if not merge_loras and control_lora:
            log.warning("Control-LoRA patching is only supported with merge_loras=True")

        if lora is not None:
            patcher, control_lora, unianimate_sd = add_lora_weights(patcher, lora, base_dtype, merge_loras=merge_loras)
            if unianimate_sd is not None:
                log.info("Merging UniAnimate weights to the model...")
                sd.update(unianimate_sd)
                del unianimate_sd

        if not gguf:
            if lora is not None and merge_loras:
                if not lora_low_mem_load:
                    load_weights(transformer, sd, weight_dtype, base_dtype, transformer_load_device)

                if control_lora:
                    patch_control_lora(patcher.model.diffusion_model, device)
                    patcher.model.is_patched = True

                log.info("Merging LoRA to the model...")
                patcher = apply_lora(
                    patcher, device, transformer_load_device, params_to_keep=params_to_keep, dtype=weight_dtype, base_dtype=base_dtype, state_dict=sd,
                    low_mem_load=lora_low_mem_load, control_lora=control_lora, scale_weights=scale_weights)
                if not control_lora:
                    scale_weights.clear()
                    patcher.patches.clear()
                transformer.patched_linear = False
                sd = None
            elif "scaled" in quantization or lora is not None:
                transformer = _replace_linear(transformer, base_dtype, sd, scale_weights=scale_weights, compile_args=compile_args)
                transformer.patched_linear = True

        if "fast" in quantization:
            if lora is not None and not merge_loras:
                raise NotImplementedError("fp8_fast is not supported with unmerged LoRAs")
            from .fp8_optimization import convert_fp8_linear
            convert_fp8_linear(transformer, base_dtype, params_to_keep, scale_weight_keys=scale_weights)

        if vram_management_args is not None:
            if gguf:
                raise ValueError("GGUF models don't support vram management")
            from .diffsynth.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
            from .wanvideo.modules.model import WanLayerNorm

            total_params_in_model = sum(p.numel() for p in patcher.model.diffusion_model.parameters())
            log.info(f"Total number of parameters in the loaded model: {total_params_in_model}")

            offload_percent = vram_management_args["offload_percent"]
            offload_params = int(total_params_in_model * offload_percent)
            params_to_keep = total_params_in_model - offload_params
            log.info(f"Selected params to offload: {offload_params}")

            enable_vram_management(
                patcher.model.diffusion_model,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    WanLayerNorm: AutoWrappedModule,
                    WanRMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=weight_dtype,
                    offload_device=offload_device,
                    onload_dtype=weight_dtype,
                    onload_device=device,
                    computation_dtype=base_dtype,
                    computation_device=device,
                ),
                max_num_param=params_to_keep,
                overflow_module_config = dict(
                    offload_dtype=weight_dtype,
                    offload_device=offload_device,
                    onload_dtype=weight_dtype,
                    onload_device=offload_device,
                    computation_dtype=base_dtype,
                    computation_device=device,
                ),
                compile_args = compile_args,
            )

        if merge_loras and lora is not None:
            # Skip offloading if load_device is main_device (for unified memory systems like AMD Strix Halo)
            if load_device != "main_device":
                log.info(f"Moving diffusion model from {patcher.model.diffusion_model.device} to {offload_device}")
                patcher.model.diffusion_model.to(offload_device)
                gc.collect()
                mm.soft_empty_cache()
            else:
                log.info(f"Skipping offload (load_device=main_device, keeping model on {patcher.model.diffusion_model.device})")

        patcher.model["base_dtype"] = base_dtype
        patcher.model["weight_dtype"] = weight_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["quantization"] = quantization
        patcher.model["auto_cpu_offload"] = True if vram_management_args is not None else False
        patcher.model["control_lora"] = control_lora
        patcher.model["compile_args"] = compile_args
        patcher.model["gguf_reader"] = gguf_reader
        patcher.model["fp8_matmul"] = "fast" in quantization
        patcher.model["scale_weights"] = scale_weights
        patcher.model["sd"] = sd
        patcher.model["lora"] = lora

        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = block_swap_args
        patcher.model_options["transformer_options"]["merge_loras"] = merge_loras

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)
        return (patcher,)

# class WanVideoSaveModel:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "model": ("WANVIDEOMODEL", {"tooltip": "WANVideo model to save"}),
#                 "output_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Path to save the model"}),
#             },
#         }

#     RETURN_TYPES = ()
#     FUNCTION = "savemodel"
#     CATEGORY = "WanVideoWrapper"
#     DESCRIPTION = "Saves the model including merged LoRAs and quantization to diffusion_models/WanVideoWrapperSavedModels"
#     OUTPUT_NODE = True

#     def savemodel(self, model, output_path):
#         from safetensors.torch import save_file
#         model_sd = model.model.diffusion_model.state_dict()
#         for k in model_sd.keys():
#             print("key:", k, "shape:", model_sd[k].shape, "dtype:", model_sd[k].dtype, "device:", model_sd[k].device)
#         model_sd
#         model_name = os.path.basename(model.model["model_name"])
#         if not output_path:
#             output_path = os.path.join(folder_paths.models_dir, "diffusion_models", "WanVideoWrapperSavedModels", "saved_" + model_name)
#         else:
#             output_path = os.path.join(output_path, model_name)
#         log.info(f"Saving model to {output_path}")
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         save_file(model_sd, output_path)
#         return ()

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
                "compile_args": ("WANCOMPILEARGS", ),
                "use_cpu_cache": ("BOOLEAN", {"default": False, "tooltip": "Reduces VRAM usage, but slows the VAE down a lot"}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Enables memory usage logging when using the model"}),
            }
        }

    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision, compile_args=None, use_cpu_cache=False, verbose=False):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = folder_paths.get_full_path_or_raise("vae", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        has_model_prefix = any(k.startswith("model.") for k in vae_sd.keys())
        if not has_model_prefix:
            vae_sd = {f"model.{k}": v for k, v in vae_sd.items()}

        dim = vae_sd["model.decoder.conv1.bias"].shape[0]
        if dim == 96:
            log.info("Detected lightVAE model with 75% pruning")
            pruning_rate = 0.75
        else:
            pruning_rate = 0.0

        if vae_sd["model.conv2.weight"].shape[0] == 16:
            vae = WanVideoVAE(dtype=dtype, pruning_rate=pruning_rate, cpu_cache=use_cpu_cache, verbose=verbose)
        elif vae_sd["model.conv2.weight"].shape[0] == 48:
            vae = WanVideoVAE38(dtype=dtype, pruning_rate=pruning_rate, cpu_cache=use_cpu_cache, verbose=verbose)

        vae.load_state_dict(vae_sd)
        del vae_sd
        vae.eval()
        vae.to(device=offload_device, dtype=dtype)
        if compile_args is not None:
            vae.model.decoder = torch.compile(vae.model.decoder, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        return (vae,)

class WanVideoTinyVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae_approx"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae_approx'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
                "parallel": ("BOOLEAN", {"default": False, "tooltip": "uses more memory but is faster"}),
            }
        }

    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan VAE model from 'ComfyUI/models/vae_approx'"

    def loadmodel(self, model_name, precision, parallel=False):
        from .taehv import TAEHV

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = folder_paths.get_full_path_or_raise("vae_approx", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        vae = TAEHV(vae_sd, parallel=parallel, dtype=dtype, model_name=model_name)

        vae.to(device=offload_device, dtype=dtype)

        return (vae,)

class LoadWanVideoT5TextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
                "precision": (["fp32", "bf16"],
                    {"default": "bf16"}
                ),
            },
            "optional": {
                "load_device": (["main_device", "offload_device"], {"default": "offload_device"}),
                "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            }
        }

    RETURN_TYPES = ("WANTEXTENCODER",)
    RETURN_NAMES = ("wan_t5_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan text_encoder model from 'ComfyUI/models/LLM'"

    def loadmodel(self, model_name, precision, load_device="offload_device", quantization="disabled"):
        text_encoder_load_device = device if load_device == "main_device" else offload_device

        tokenizer_path = os.path.join(script_directory, "configs", "T5_tokenizer")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_path = folder_paths.get_full_path_or_raise("text_encoders", model_name)
        sd = load_torch_file(model_path, safe_load=True)

        if quantization == "disabled":
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float8_e4m3fn:
                        quantization = "fp8_e4m3fn"
                        break

        if "token_embedding.weight" not in sd and "shared.weight" not in sd:
            raise ValueError("Invalid T5 text encoder model, this node expects the 'umt5-xxl' model")
        if "scaled_fp8" in sd:
            raise ValueError("Invalid T5 text encoder model, fp8 scaled is not supported by this node")

        # Convert state dict keys from T5 format to the expected format
        if "shared.weight" in sd:
            log.info("Converting T5 text encoder model to the expected format...")
            converted_sd = {}

            for key, value in sd.items():
                # Handle encoder block patterns
                if key.startswith('encoder.block.'):
                    parts = key.split('.')
                    block_num = parts[2]

                    # Self-attention components
                    if 'layer.0.SelfAttention' in key:
                        if key.endswith('.k.weight'):
                            new_key = f"blocks.{block_num}.attn.k.weight"
                        elif key.endswith('.o.weight'):
                            new_key = f"blocks.{block_num}.attn.o.weight"
                        elif key.endswith('.q.weight'):
                            new_key = f"blocks.{block_num}.attn.q.weight"
                        elif key.endswith('.v.weight'):
                            new_key = f"blocks.{block_num}.attn.v.weight"
                        elif 'relative_attention_bias' in key:
                            new_key = f"blocks.{block_num}.pos_embedding.embedding.weight"
                        else:
                            new_key = key

                    # Layer norms
                    elif 'layer.0.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm1.weight"
                    elif 'layer.1.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm2.weight"

                    # Feed-forward components
                    elif 'layer.1.DenseReluDense' in key:
                        if 'wi_0' in key:
                            new_key = f"blocks.{block_num}.ffn.gate.0.weight"
                        elif 'wi_1' in key:
                            new_key = f"blocks.{block_num}.ffn.fc1.weight"
                        elif 'wo' in key:
                            new_key = f"blocks.{block_num}.ffn.fc2.weight"
                        else:
                            new_key = key
                    else:
                        new_key = key
                elif key == "shared.weight":
                    new_key = "token_embedding.weight"
                elif key == "encoder.final_layer_norm.weight":
                    new_key = "norm.weight"
                else:
                    new_key = key
                converted_sd[new_key] = value
            sd = converted_sd

        T5_text_encoder = T5EncoderModel(
            text_len=512,
            dtype=dtype,
            device=text_encoder_load_device,
            state_dict=sd,
            tokenizer_path=tokenizer_path,
            quantization=quantization
        )
        text_encoder = {
            "model": T5_text_encoder,
            "dtype": dtype,
            "name": model_name,
        }

        return (text_encoder,)

class LoadWanVideoClipTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("clip_vision") + folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/clip_vision'"}),
                 "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            },
            "optional": {
                "load_device": (["main_device", "offload_device"], {"default": "offload_device"}),
            }
        }

    RETURN_TYPES = ("CLIP_VISION",)
    RETURN_NAMES = ("wan_clip_vision", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan clip_vision model from 'ComfyUI/models/clip_vision'"

    def loadmodel(self, model_name, precision, load_device="offload_device"):
        text_encoder_load_device = device if load_device == "main_device" else offload_device

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_path = folder_paths.get_full_path_or_raise("clip_vision", model_name)
        # We also support legacy setups where the model is in the text_encoders folder
        if model_path is None:
            model_path = folder_paths.get_full_path_or_raise("text_encoders", model_name)
        sd = load_torch_file(model_path, safe_load=True)
        if "log_scale" not in sd:
            raise ValueError("Invalid CLIP model, this node expectes the 'open-clip-xlm-roberta-large-vit-huge-14' model")

        clip_model = CLIPModel(dtype=dtype, device=device, state_dict=sd)
        clip_model.model.to(text_encoder_load_device)
        del sd

        return (clip_model,)

NODE_CLASS_MAPPINGS = {
    "WanVideoModelLoader": WanVideoModelLoader,
    "WanVideoVAELoader": WanVideoVAELoader,
    "WanVideoLoraSelect": WanVideoLoraSelect,
    "WanVideoLoraSelectByName": WanVideoLoraSelectByName,
    "WanVideoSetLoRAs": WanVideoSetLoRAs,
    "WanVideoLoraBlockEdit": WanVideoLoraBlockEdit,
    "WanVideoTinyVAELoader": WanVideoTinyVAELoader,
    "WanVideoVACEModelSelect": WanVideoVACEModelSelect,
    "WanVideoExtraModelSelect": WanVideoExtraModelSelect,
    "WanVideoLoraSelectMulti": WanVideoLoraSelectMulti,
    "WanVideoBlockSwap": WanVideoBlockSwap,
    "WanVideoVRAMManagement": WanVideoVRAMManagement,
    "WanVideoTorchCompileSettings": WanVideoTorchCompileSettings,
    "LoadWanVideoT5TextEncoder": LoadWanVideoT5TextEncoder,
    "LoadWanVideoClipTextEncoder": LoadWanVideoClipTextEncoder,
    "WanVideoSetAttentionModeOverride": WanVideoSetAttentionModeOverride,
    "WanVideoUltraVicoSettings": WanVideoUltraVicoSettings,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoModelLoader": "WanVideo Model Loader",
    "WanVideoVAELoader": "WanVideo VAE Loader",
    "WanVideoLoraSelect": "WanVideo Lora Select",
    "WanVideoLoraSelectByName": "WanVideo Lora Select By Name",
    "WanVideoSetLoRAs": "WanVideo Set LoRAs",
    "WanVideoLoraBlockEdit": "WanVideo Lora Block Edit",
    "WanVideoTinyVAELoader": "WanVideo Tiny VAE Loader",
    "WanVideoVACEModelSelect": "WanVideo VACE Module Select",
    "WanVideoExtraModelSelect": "WanVideo Extra Model Select",
    "WanVideoLoraSelectMulti": "WanVideo Lora Select Multi",
    "WanVideoBlockSwap": "WanVideo Block Swap",
    "WanVideoVRAMManagement": "WanVideo VRAM Management",
    "WanVideoTorchCompileSettings": "WanVideo Torch Compile Settings",
    "LoadWanVideoT5TextEncoder": "WanVideo T5 Text Encoder Loader",
    "LoadWanVideoClipTextEncoder": "WanVideo CLIP Text Encoder Loader",
    "WanVideoSetAttentionModeOverride": "WanVideo Set Attention Mode Override",
    "WanVideoUltraVicoSettings": "WanVideo UltraVico Settings"
    }
