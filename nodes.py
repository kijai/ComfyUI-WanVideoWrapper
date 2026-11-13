import os, gc, math
import torch
import torch.nn.functional as F
import hashlib
from tqdm import tqdm

from .utils import(log, clip_encode_image_tiled, add_noise_to_reference_video, set_module_tensor_to_device)
from .taehv import TAEHV

from comfy import model_management as mm
from comfy.utils import ProgressBar, common_upscale
from comfy.clip_vision import clip_preprocess, ClipVisionModel
import folder_paths
from .scail_pose2_mask_contract import (
    build_disabled_samples_payload,
    scail_pose2_mask_disables_samples,
)

script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)
class WanVideoEnhanceAVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 2.0, "min": 0, "max": 100, "step": 0.01, "tooltip": "The feta Weight of the Enhance-A-Video"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply Enhance-A-Video"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply Enhance-A-Video"}),
            },
        }
    RETURN_TYPES = ("FETAARGS",)
    RETURN_NAMES = ("feta_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"

    def setargs(self, **kwargs):
        return (kwargs, )

class WanVideoSetBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
               },
            "optional": {
                "block_swap_args": ("BLOCKSWAPARGS", ),
               }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, block_swap_args=None):
        if block_swap_args is None:
            return (model,)
        patcher = model.clone()
        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = block_swap_args     

        return (patcher,)

class WanVideoSetRadialAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
                "dense_attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "sparse_sage_attention",
                    ], {"default": "sageattn", "tooltip": "The attention mode for dense attention"}),
                "dense_blocks": ("INT",  {"default": 1, "min": 0, "max": 40, "step": 1, "tooltip": "Number of blocks to apply normal attention to"}),
                "dense_vace_blocks": ("INT",  {"default": 1, "min": 0, "max": 15, "step": 1, "tooltip": "Number of vace blocks to apply normal attention to"}),
                "dense_timesteps": ("INT",  {"default": 2, "min": 0, "max": 100, "step": 1, "tooltip": "The step to start applying sparse attention"}),
                "decay_factor": ("FLOAT",  {"default": 0.2, "min": 0, "max": 1, "step": 0.01, "tooltip": "Controls how quickly the attention window shrinks as the distance between frames increases in the sparse attention mask."}),
                "block_size":([128, 64], {"default": 128, "tooltip": "Radial attention block size, larger blocks are faster but restricts usable dimensions more."}),
               }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Sets radial attention parameters, dense attention refers to normal attention"

    def loadmodel(self, model, dense_attention_mode, dense_blocks, dense_vace_blocks, dense_timesteps, decay_factor, block_size):
        if "radial" not in model.model.diffusion_model.attention_mode:
            raise Exception("Enable radial attention first in the model loader.")
            
        patcher = model.clone()
        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}

        patcher.model_options["transformer_options"]["dense_attention_mode"] = dense_attention_mode
        patcher.model_options["transformer_options"]["dense_blocks"] = dense_blocks
        patcher.model_options["transformer_options"]["dense_vace_blocks"] = dense_vace_blocks
        patcher.model_options["transformer_options"]["dense_timesteps"] = dense_timesteps
        patcher.model_options["transformer_options"]["decay_factor"] = decay_factor
        patcher.model_options["transformer_options"]["block_size"] = block_size

        return (patcher,)

class WanVideoBlockList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blocks": ("STRING",  {"default": "1", "multiline":True}),
               }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("block_list", )
    FUNCTION = "create_list"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Comma separated list of blocks to apply block swap to, can also use ranges like '0-5' or '0,2,3-5' etc., can be connected to the dense_blocks input of 'WanVideoSetRadialAttention' node"

    def create_list(self, blocks):
        block_list = []
        for line in blocks.splitlines():
            for part in line.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    try:
                        start, end = map(int, part.split("-", 1))
                        block_list.extend(range(start, end + 1))
                    except Exception:
                        raise ValueError(f"Invalid range: '{part}'")
                else:
                    try:
                        block_list.append(int(part))
                    except Exception:
                        raise ValueError(f"Invalid integer: '{part}'")
        return (block_list,)



# In-memory cache for prompt extender output
_extender_cache = {}

cache_dir = os.path.join(script_directory, 'text_embed_cache')

def get_cache_path(prompt):
    cache_key = prompt.strip()
    cache_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, f"{cache_hash}.pt")

def get_cached_text_embeds(positive_prompt, negative_prompt):
    
    os.makedirs(cache_dir, exist_ok=True)

    context = None
    context_null = None

    pos_cache_path = get_cache_path(positive_prompt)
    neg_cache_path = get_cache_path(negative_prompt)

    # Try to load positive prompt embeds
    if os.path.exists(pos_cache_path):
        try:
            log.info(f"Loading prompt embeds from cache: {pos_cache_path}")
            context = torch.load(pos_cache_path)
        except Exception as e:
            log.warning(f"Failed to load cache: {e}, will re-encode.")

    # Try to load negative prompt embeds
    if os.path.exists(neg_cache_path):
        try:
            log.info(f"Loading prompt embeds from cache: {neg_cache_path}")
            context_null = torch.load(neg_cache_path)
        except Exception as e:
            log.warning(f"Failed to load cache: {e}, will re-encode.")

    return context, context_null

class WanVideoTextEncodeCached:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
            "precision": (["fp32", "bf16"],
                    {"default": "bf16"}
                ),
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "use_disk_cache": ("BOOLEAN", {"default": True, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
            "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            },
            "optional": {
                "extender_args": ("WANVIDEOPROMPTEXTENDER_ARGS", {"tooltip": "Use this node to extend the prompt with additional text."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "WANVIDEOTEXTEMBEDS", "STRING")
    RETURN_NAMES = ("text_embeds", "negative_text_embeds", "positive_prompt")
    OUTPUT_TOOLTIPS = ("The text embeddings for both prompts", "The text embeddings for the negative prompt only (for NAG)", "Positive prompt to display prompt extender results")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = """Encodes text prompts into text embeddings. This node loads and completely unloads the T5 after done,  
leaving no VRAM or RAM imprint. If prompts have been cached before T5 is not loaded at all.  
negative output is meant to be used with NAG, it contains only negative prompt embeddings.  

Additionally you can provide a Qwen LLM model to extend the positive prompt with either one  
of the original Wan templates or a custom system prompt.  
"""


    def process(self, model_name, precision, positive_prompt, negative_prompt, quantization='disabled', use_disk_cache=True, device="gpu", extender_args=None):
        from .nodes_model_loading import LoadWanVideoT5TextEncoder
        pbar = ProgressBar(3)

        echoshot = True if "[1]" in positive_prompt else False

        # Handle prompt extension with in-memory cache
        orig_prompt = positive_prompt
        if extender_args is not None:
            extender_key = (orig_prompt, str(extender_args))
            if extender_key in _extender_cache:
                positive_prompt = _extender_cache[extender_key]
                log.info(f"Loaded extended prompt from in-memory cache: {positive_prompt}")
            else:
                from .qwen.qwen import QwenLoader, WanVideoPromptExtender
                log.info("Using WanVideoPromptExtender to process prompts")
                qwen, = QwenLoader().load(
                    extender_args["model"], 
                    load_device="main_device" if device == "gpu" else "cpu", 
                    precision=precision)
                positive_prompt, = WanVideoPromptExtender().generate(
                    qwen=qwen,
                    max_new_tokens=extender_args["max_new_tokens"],
                    prompt=orig_prompt,
                    device=device,
                    force_offload=False,
                    custom_system_prompt=extender_args["system_prompt"],
                    seed=extender_args["seed"]
                )
                log.info(f"Extended positive prompt: {positive_prompt}")
                _extender_cache[extender_key] = positive_prompt
                del qwen
            pbar.update(1)

        # Now check disk cache using the (possibly extended) prompt
        if use_disk_cache:
            context, context_null = get_cached_text_embeds(positive_prompt, negative_prompt)
            if context is not None and context_null is not None:
                return{
                    "prompt_embeds": context,
                    "negative_prompt_embeds": context_null,
                    "echoshot": echoshot,
                },{"prompt_embeds": context_null}, positive_prompt

        t5, = LoadWanVideoT5TextEncoder().loadmodel(model_name, precision, "main_device", quantization)
        pbar.update(1)

        prompt_embeds_dict, = WanVideoTextEncode().process(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            t5=t5,
            force_offload=False,
            model_to_offload=None,
            use_disk_cache=use_disk_cache,
            device=device
        )
        pbar.update(1)
        del t5
        mm.soft_empty_cache()
        gc.collect() 
        return (prompt_embeds_dict, {"prompt_embeds": prompt_embeds_dict["negative_prompt_embeds"]}, positive_prompt)

#region TextEncode
class WanVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "t5": ("WANTEXTENCODER",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
                "use_disk_cache": ("BOOLEAN", {"default": False, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
                "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompts into text embeddings. For rudimentary prompt travel you can input multiple prompts separated by '|', they will be equally spread over the video length"


    def process(self, positive_prompt, negative_prompt, t5=None, force_offload=True, model_to_offload=None, use_disk_cache=False, device="gpu"):
        if t5 is None and not use_disk_cache:
            raise ValueError("T5 encoder is required for text encoding. Please provide a valid T5 encoder or enable disk cache.")

        echoshot = True if "[1]" in positive_prompt else False

        if use_disk_cache:
            context, context_null = get_cached_text_embeds(positive_prompt, negative_prompt)
            if context is not None and context_null is not None:
                return{
                    "prompt_embeds": context,
                    "negative_prompt_embeds": context_null,
                    "echoshot": echoshot,
                },
            
        if t5 is None:
            raise ValueError("No cached text embeds found for prompts, please provide a T5 encoder.")

        if model_to_offload is not None and device == "gpu":
            try:
                log.info(f"Moving video model to {offload_device}")
                model_to_offload.model.to(offload_device)
            except Exception:
                pass

        encoder = t5["model"]
        dtype = t5["dtype"]
        
        positive_prompts = []
        all_weights = []

        # Split positive prompts and process each with weights
        if "|" in positive_prompt:
            log.info("Multiple positive prompts detected, splitting by '|'")
            positive_prompts_raw = [p.strip() for p in positive_prompt.split('|')]
        elif "[1]" in positive_prompt:
            log.info("Multiple positive prompts detected, splitting by [#] and enabling EchoShot")
            import re
            segments = re.split(r'\[\d+\]', positive_prompt)
            positive_prompts_raw = [segment.strip() for segment in segments if segment.strip()]
            assert len(positive_prompts_raw) > 1 and len(positive_prompts_raw) < 7, 'Input shot num must between 2~6 !'
        else:
            positive_prompts_raw = [positive_prompt.strip()]
            
        for p in positive_prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)

        mm.soft_empty_cache()

        if device == "gpu":
            device_to = mm.get_torch_device()
        else:
            device_to = torch.device("cpu")

        if encoder.quantization == "fp8_e4m3fn":
            cast_dtype = torch.float8_e4m3fn
        else:
            cast_dtype = encoder.dtype

        params_to_keep = {'norm', 'pos_embedding', 'token_embedding'}
        if hasattr(encoder, 'state_dict'):
            model_state_dict = encoder.state_dict
        else:
            model_state_dict = encoder.model.state_dict()

        params_list = list(encoder.model.named_parameters())
        pbar = tqdm(params_list, desc="Loading T5 parameters", leave=True)
        for name, param in pbar:
            dtype_to_use = dtype if any(keyword in name for keyword in params_to_keep) else cast_dtype
            value = model_state_dict[name]
            set_module_tensor_to_device(encoder.model, name, device=device_to, dtype=dtype_to_use, value=value)
        del model_state_dict
        if hasattr(encoder, 'state_dict'):
            del encoder.state_dict
            mm.soft_empty_cache()
            gc.collect()

        with torch.autocast(device_type=mm.get_autocast_device(device_to), dtype=encoder.dtype, enabled=encoder.quantization != 'disabled'):
            # Encode positive if not loaded from cache
            if use_disk_cache and context is not None:
                pass
            else:
                context = encoder(positive_prompts, device_to)
                # Apply weights to embeddings if any were extracted
                for i, weights in enumerate(all_weights):
                    for text, weight in weights.items():
                        log.info(f"Applying weight {weight} to prompt: {text}")
                        if len(weights) > 0:
                            context[i] = context[i] * weight

            # Encode negative if not loaded from cache
            if use_disk_cache and context_null is not None:
                pass
            else:
                context_null = encoder([negative_prompt], device_to)

        if force_offload:
            encoder.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        prompt_embeds_dict = {
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
            "echoshot": echoshot,
        }

        # Save each part to its own cache file if needed
        if use_disk_cache:
            pos_cache_path = get_cache_path(positive_prompt)
            neg_cache_path = get_cache_path(negative_prompt)
            try:
                if not os.path.exists(pos_cache_path):
                    torch.save(context, pos_cache_path)
                    log.info(f"Saved prompt embeds to cache: {pos_cache_path}")
            except Exception as e:
                log.warning(f"Failed to save cache: {e}")
            try:
                if not os.path.exists(neg_cache_path):
                    torch.save(context_null, neg_cache_path)
                    log.info(f"Saved prompt embeds to cache: {neg_cache_path}")
            except Exception as e:
                log.warning(f"Failed to save cache: {e}")

        return (prompt_embeds_dict,)
    
    def parse_prompt_weights(self, prompt):
        """Extract text and weights from prompts with (text:weight) format"""
        import re
        
        # Parse all instances of (text:weight) in the prompt
        pattern = r'\((.*?):([\d\.]+)\)'
        matches = re.findall(pattern, prompt)
        
        # Replace each match with just the text part
        cleaned_prompt = prompt
        weights = {}
        
        for match in matches:
            text, weight = match
            orig_text = f"({text}:{weight})"
            cleaned_prompt = cleaned_prompt.replace(orig_text, text)
            weights[text] = float(weight)
            
        return cleaned_prompt, weights
    
class WanVideoTextEncodeSingle:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "t5": ("WANTEXTENCODER",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
                "use_disk_cache": ("BOOLEAN", {"default": False, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
                "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompt into text embedding."

    def process(self, prompt, t5=None, force_offload=True, model_to_offload=None, use_disk_cache=False, device="gpu"):
        # Unified cache logic: use a single cache file per unique prompt
        encoded = None
        echoshot = True if "[1]" in prompt else False
        if use_disk_cache:
            cache_dir = os.path.join(script_directory, 'text_embed_cache')
            os.makedirs(cache_dir, exist_ok=True)
            def get_cache_path(prompt):
                cache_key = prompt.strip()
                cache_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
                return os.path.join(cache_dir, f"{cache_hash}.pt")
            cache_path = get_cache_path(prompt)
            if os.path.exists(cache_path):
                try:
                    log.info(f"Loading prompt embeds from cache: {cache_path}")
                    encoded = torch.load(cache_path)
                except Exception as e:
                    log.warning(f"Failed to load cache: {e}, will re-encode.")

        if t5 is None and encoded is None:
            raise ValueError("No cached text embeds found for prompts, please provide a T5 encoder.")

        if encoded is None:
            try:
                if model_to_offload is not None and device == "gpu":
                    log.info(f"Moving video model to {offload_device}")
                    model_to_offload.model.to(offload_device)
                    mm.soft_empty_cache()
            except Exception:
                pass

            encoder = t5["model"]
            dtype = t5["dtype"]

            if device == "gpu":
                device_to = mm.get_torch_device()
            else:
                device_to = torch.device("cpu")

            if encoder.quantization == "fp8_e4m3fn":
                cast_dtype = torch.float8_e4m3fn
            else:
                cast_dtype = encoder.dtype
            params_to_keep = {'norm', 'pos_embedding', 'token_embedding'}
            for name, param in encoder.model.named_parameters():
                dtype_to_use = dtype if any(keyword in name for keyword in params_to_keep) else cast_dtype
                value = encoder.state_dict[name] if hasattr(encoder, 'state_dict') else encoder.model.state_dict()[name]
                set_module_tensor_to_device(encoder.model, name, device=device_to, dtype=dtype_to_use, value=value)
            if hasattr(encoder, 'state_dict'):
                del encoder.state_dict
                mm.soft_empty_cache()
                gc.collect()
            with torch.autocast(device_type=mm.get_autocast_device(device_to), dtype=encoder.dtype, enabled=encoder.quantization != 'disabled'):
                encoded = encoder([prompt], device_to)

            if force_offload:
                encoder.model.to(offload_device)
                mm.soft_empty_cache()

            # Save to cache if enabled
            if use_disk_cache:
                try:
                    if not os.path.exists(cache_path):
                        torch.save(encoded, cache_path)
                        log.info(f"Saved prompt embeds to cache: {cache_path}")
                except Exception as e:
                    log.warning(f"Failed to save cache: {e}")

        prompt_embeds_dict = {
            "prompt_embeds": encoded,
            "negative_prompt_embeds": None,
            "echoshot": echoshot
        }
        return (prompt_embeds_dict,)
    
class WanVideoApplyNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "original_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "nag_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "inplace": ("BOOLEAN", {"default": True, "tooltip": "If true, modifies tensors in place to save memory. Leads to different numerical results which may change the output slightly."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Adds NAG prompt embeds to original prompt embeds: 'https://github.com/ChenDarYen/Normalized-Attention-Guidance'"

    def process(self, original_text_embeds, nag_text_embeds, nag_scale, nag_tau, nag_alpha, inplace=True):
        prompt_embeds_dict_copy = original_text_embeds.copy()
        prompt_embeds_dict_copy.update({
                "nag_prompt_embeds": nag_text_embeds["prompt_embeds"],
                "nag_params": {
                    "nag_scale": nag_scale,
                    "nag_tau": nag_tau,
                    "nag_alpha": nag_alpha,
                    "inplace": inplace,
                }
            })
        return (prompt_embeds_dict_copy,)
    
class WanVideoTextEmbedBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("CONDITIONING",),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Bridge between ComfyUI native text embedding and WanVideoWrapper text embedding"

    def process(self, positive, negative=None):
        prompt_embeds_dict = {
                "prompt_embeds": positive[0][0].to(device),
                "negative_prompt_embeds": negative[0][0].to(device) if negative is not None else None,
            }
        return (prompt_embeds_dict,)
    
#region clip vision
class WanVideoClipVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "image_1": ("IMAGE", {"tooltip": "Image to encode"}),
            "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}), 
            "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}),
            "crop": (["center", "disabled"], {"default": "center", "tooltip": "Crop image to 224x224 before encoding"}),
            "combine_embeds": (["average", "sum", "concat", "batch"], {"default": "average", "tooltip": "Method to combine multiple clip embeds"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_2": ("IMAGE", ),
                "negative_image": ("IMAGE", {"tooltip": "image to use for uncond"}),
                "tiles": ("INT", {"default": 0, "min": 0, "max": 16, "step": 2, "tooltip": "Use matteo's tiled image encoding for improved accuracy"}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ratio of the tile average"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_CLIPEMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, clip_vision, image_1, strength_1, strength_2, force_offload, crop, combine_embeds, image_2=None, negative_image=None, tiles=0, ratio=1.0):
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

        if image_2 is not None:
            image = torch.cat([image_1, image_2], dim=0)
        else:
            image = image_1

        clip_vision.model.to(device)
        
        negative_clip_embeds = None

        if tiles > 0:
            log.info("Using tiled image encoding")
            clip_embeds = clip_encode_image_tiled(clip_vision, image.to(device), tiles=tiles, ratio=ratio)
            if negative_image is not None:
                negative_clip_embeds = clip_encode_image_tiled(clip_vision, negative_image.to(device), tiles=tiles, ratio=ratio)
        else:
            if isinstance(clip_vision, ClipVisionModel):
                clip_embeds = clip_vision.encode_image(image).penultimate_hidden_states.to(device)
                if negative_image is not None:
                    negative_clip_embeds = clip_vision.encode_image(negative_image).penultimate_hidden_states.to(device)
            else:
                pixel_values = clip_preprocess(image.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                clip_embeds = clip_vision.visual(pixel_values)
                if negative_image is not None:
                    pixel_values = clip_preprocess(negative_image.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                    negative_clip_embeds = clip_vision.visual(pixel_values)
    
        log.info(f"Clip embeds shape: {clip_embeds.shape}, dtype: {clip_embeds.dtype}")

        weighted_embeds = []
        weighted_embeds.append(clip_embeds[0:1] * strength_1)

        # Handle all additional embeddings
        if clip_embeds.shape[0] > 1:
            weighted_embeds.append(clip_embeds[1:2] * strength_2)
            
            if clip_embeds.shape[0] > 2:
                for i in range(2, clip_embeds.shape[0]):
                    weighted_embeds.append(clip_embeds[i:i+1])  # Add as-is without strength modifier
            
            # Combine all weighted embeddings
            if combine_embeds == "average":
                clip_embeds = torch.mean(torch.stack(weighted_embeds), dim=0)
            elif combine_embeds == "sum":
                clip_embeds = torch.sum(torch.stack(weighted_embeds), dim=0)
            elif combine_embeds == "concat":
                clip_embeds = torch.cat(weighted_embeds, dim=1)
            elif combine_embeds == "batch":
                clip_embeds = torch.cat(weighted_embeds, dim=0)
        else:
            clip_embeds = weighted_embeds[0]
                

        log.info(f"Combined clip embeds shape: {clip_embeds.shape}")
        
        if force_offload:
            clip_vision.model.to(offload_device)
            mm.soft_empty_cache()

        clip_embeds_dict = {
            "clip_embeds": clip_embeds,
            "negative_clip_embeds": negative_clip_embeds
        }

        return (clip_embeds_dict,)
        
class WanVideoRealisDanceLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ref_latent": ("LATENT", {"tooltip": "Reference image to encode"}),
            "pose_cond_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the SMPL model"}),
            "pose_cond_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the SMPL model"}),
            },
            "optional": {
                "smpl_latent": ("LATENT", {"tooltip": "SMPL pose image to encode"}),
                "hamer_latent": ("LATENT", {"tooltip": "Hamer hand pose image to encode"}),
            },
        }

    RETURN_TYPES = ("ADD_COND_LATENTS",)
    RETURN_NAMES = ("add_cond_latents",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, ref_latent, pose_cond_start_percent, pose_cond_end_percent, hamer_latent=None, smpl_latent=None):
        if smpl_latent is None and hamer_latent is None:
            raise Exception("At least one of smpl_latent or hamer_latent must be provided")
        if smpl_latent is None:
            smpl = torch.zeros_like(hamer_latent["samples"])
        else:
            smpl = smpl_latent["samples"]
        if hamer_latent is None:
            hamer = torch.zeros_like(smpl_latent["samples"])
        else:
            hamer = hamer_latent["samples"]

        pose_latent = torch.cat((smpl, hamer), dim=1)
        
        add_cond_latents = {
            "ref_latent": ref_latent["samples"],
            "pose_latent": pose_latent,
            "pose_cond_start_percent": pose_cond_start_percent,
            "pose_cond_end_percent": pose_cond_end_percent,
        }

        return (add_cond_latents,)

    
class WanVideoAddStandInLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "ip_image_latent": ("LATENT", {"tooltip": "Reference image to encode"}),
                    "freq_offset": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "EXPERIMENTAL: RoPE frequency offset between the reference and rest of the sequence"}),
                    #"start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent to apply the ref "}),
                    #"end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent to apply the ref "}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, ip_image_latent, freq_offset):
        # Prepare the new extra latent entry
        new_entry = {
            "ip_image_latent": ip_image_latent["samples"],
            "freq_offset": freq_offset,
            #"ip_start_percent": start_percent,
            #"ip_end_percent": end_percent,
        }    

        # Return a new dict with updated extra_latents
        updated = dict(embeds)
        updated["standin_input"] = new_entry
        return (updated,)
    
class WanVideoAddBindweaveEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "reference_latents": ("LATENT", {"tooltip": "Reference image to encode"}),
                }, 
                "optional": {
                    "ref_masks": ("MASK", {"tooltip": "Reference mask to encode"}),
                    "qwenvl_embeds_pos": ("QWENVL_EMBEDS", {"tooltip": "Qwen-VL image embeddings for the reference image"}),
                    "qwenvl_embeds_neg": ("QWENVL_EMBEDS", {"tooltip": "Qwen-VL image embeddings for the reference image"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "LATENT", "MASK",)
    RETURN_NAMES = ("image_embeds", "image_embed_preview", "mask_preview",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, reference_latents, ref_masks=None, qwenvl_embeds_pos=None, qwenvl_embeds_neg=None):
        updated = dict(embeds)
        image_embeds = embeds["image_embeds"]
        max_refs = 4
        num_refs = reference_latents["samples"].shape[0]
        pad = torch.zeros(image_embeds.shape[0], max_refs-num_refs, image_embeds.shape[2], image_embeds.shape[3], device=image_embeds.device, dtype=image_embeds.dtype)
        if num_refs < max_refs:
            image_embeds = torch.cat([pad, image_embeds], dim=1)
        ref_latents = [ref_latent for ref_latent in reference_latents["samples"]]
        image_embeds = torch.cat([*ref_latents, image_embeds], dim=1)
        
        mask = embeds.get("mask", None)
        if mask is not None:
            mask_pad = torch.zeros(mask.shape[0], max_refs-num_refs, mask.shape[2], mask.shape[3], device=mask.device, dtype=mask.dtype)
            if num_refs < max_refs:
                mask = torch.cat([mask_pad, mask], dim=1)
            if ref_masks is not None:
                ref_mask_ = common_upscale(ref_masks.unsqueeze(1), mask.shape[3], mask.shape[2], "nearest", "disabled").movedim(0,1)
                ref_mask_ = torch.cat([ref_mask_, torch.zeros(3, ref_mask_.shape[1], ref_mask_.shape[2], ref_mask_.shape[3], device=ref_mask_.device, dtype=ref_mask_.dtype)])
                mask = torch.cat([ref_mask_, mask], dim=1)
            else:
                mask = torch.cat([torch.ones(mask.shape[0], num_refs, mask.shape[2], mask.shape[3], device=mask.device, dtype=mask.dtype), mask], dim=1)

            updated["mask"] = mask

        clip_embeds = updated.get("clip_context", None)
        if clip_embeds is not None:
            B, T, C = clip_embeds.shape
            target_len = max_refs * 257  # 4 * 257 = 1028
            if T < target_len:
                pad = torch.zeros(B, target_len - T, C, device=clip_embeds.device, dtype=clip_embeds.dtype)
                padded_embeds = torch.cat([clip_embeds, pad], dim=1)
                log.info(f"Padded clip embeds from {clip_embeds.shape} to {padded_embeds.shape} for Bindweave")
                updated["clip_context"] = padded_embeds
            else:
                updated["clip_context"] = clip_embeds

        updated["image_embeds"] = image_embeds
        updated["qwenvl_embeds_pos"] = qwenvl_embeds_pos
        updated["qwenvl_embeds_neg"] = qwenvl_embeds_neg
        return (updated, {"samples": image_embeds.unsqueeze(0)}, mask[0].float())
    
class TextImageEncodeQwenVL():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip": ("CLIP",),
                    "prompt": ("STRING", {"default": "", "multiline": True}),
                }, 
                "optional": {
                    "image": ("IMAGE", ),
                }
        }

    RETURN_TYPES = ("QWENVL_EMBEDS",)
    RETURN_NAMES = ("qwenvl_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(cls, clip, prompt, image=None):
        if image is None:
            input_images = []
            llama_template = None
        else:
            input_images = [image[:, :, :, :3]]

            llama_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

        tokens = clip.tokenize(prompt, images=input_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        print("Qwen-VL embeds shape:", conditioning[0][0].shape)
        return (conditioning[0][0],)

class WanVideoAddMTVMotion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "mtv_crafter_motion": ("MTVCRAFTERMOTION",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the MTV motion"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent to apply the ref "}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent to apply the ref "}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, mtv_crafter_motion, strength, start_percent, end_percent):
        # Prepare the new extra latent entry
        new_entry = {
            "mtv_motion_tokens": mtv_crafter_motion["mtv_motion_tokens"],
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "global_mean": mtv_crafter_motion["global_mean"],
            "global_std": mtv_crafter_motion["global_std"]
        }

        # Return a new dict with updated extra_latents
        updated = dict(embeds)
        updated["mtv_crafter_motion"] = new_entry
        return (updated,)

class WanVideoAddStoryMemLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "memory_images": ("IMAGE",),
                    "rope_negative_offset": ("BOOLEAN", {"default": False, "tooltip": "Use positive RoPE frequency offset for the memory latents"}),
                    "rope_negative_offset_frames": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1, "tooltip": "RoPE frequency offset for the memory latents"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, vae, embeds, memory_images, rope_negative_offset, rope_negative_offset_frames):
        updated = dict(embeds)
        story_mem_latents, = WanVideoEncodeLatentBatch().encode(vae, memory_images)
        updated["story_mem_latents"] = story_mem_latents["samples"].squeeze(2).permute(1, 0, 2, 3)  # [C, T, H, W]
        updated["rope_negative_offset_frames"] = rope_negative_offset_frames if rope_negative_offset else 0
        return (updated,)


class WanVideoSVIProEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "anchor_samples": ("LATENT", {"tooltip": "Initial start image encoded"}),
                    "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
                },
                "optional": {
                    "prev_samples": ("LATENT", {"tooltip": "Last latent from previous generation"}),
                    "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "Number of latents used to continue"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, anchor_samples, num_frames, prev_samples=None, motion_latent_count=1):

        anchor_latent = anchor_samples["samples"][0].clone()

        C, T, H, W = anchor_latent.shape

        total_latents = (num_frames - 1) // 4 + 1
        device = anchor_latent.device
        dtype = anchor_latent.dtype

        if prev_samples is None or motion_latent_count == 0:
            padding_size = total_latents - anchor_latent.shape[1]
            padding = torch.zeros(C, padding_size, H, W, dtype=dtype, device=device)
            y = torch.concat([anchor_latent, padding], dim=1)
        else:
            prev_latent = prev_samples["samples"][0].clone()
            motion_latent = prev_latent[:, -motion_latent_count:]
            padding_size = total_latents - anchor_latent.shape[1] - motion_latent.shape[1]
            padding = torch.zeros(C, padding_size, H, W, dtype=dtype, device=device)
            y = torch.concat([anchor_latent, motion_latent, padding], dim=1)

        msk = torch.ones(1, num_frames, H, W, device=device, dtype=dtype)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, H, W)
        msk = msk.transpose(1, 2)[0]

        image_embeds = {
            "image_embeds": y,
            "num_frames": num_frames,
            "lat_h": H,
            "lat_w": W,
            "mask": msk
        }

        return (image_embeds,)

#region I2V encode
class WanVideoImageToVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for I2V where some noise can add motion and give sharper results"}),
            "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("WANVAE",),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "start_image": ("IMAGE", {"tooltip": "Image to encode"}),
                "end_image": ("IMAGE", {"tooltip": "end frame"}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Control signal for the Fun -model"}),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True, "tooltip": "Enable when using official FLF2V or Fun model"}),
                "temporal_mask": ("MASK", {"tooltip": "mask"}),
                "extra_latents": ("LATENT", {"tooltip": "Extra latents to add to the input front, used for Skyreels A2 reference images"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "add_cond_latents": ("ADD_COND_LATENTS", {"advanced": True, "tooltip": "Additional cond latents WIP"}),
                "augment_empty_frames": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "EXPERIMENTAL: Augment empty frames with the difference to the start image to force more motion"}),
                "empty_frame_pad_image": ("IMAGE", {"tooltip": "Use this image to pad empty frames instead of gray, used with SVI-shot and SVI 2.0 LoRAs"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, width, height, num_frames, force_offload, noise_aug_strength, 
                start_latent_strength, end_latent_strength, start_image=None, end_image=None, control_embeds=None, fun_or_fl2v_model=False,
                temporal_mask=None, extra_latents=None, clip_embeds=None, tiled_vae=False, add_cond_latents=None, vae=None, augment_empty_frames=0.0, empty_frame_pad_image=None):

        if vae is None:
            raise ValueError("VAE is required for image encoding.")
        H = height
        W = width

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_frames = ((num_frames - 1) // 4) * 4 + 1
        two_ref_images = start_image is not None and end_image is not None

        if start_image is None and end_image is not None:
            fun_or_fl2v_model = True # end image alone only works with this option

        base_frames = num_frames + (1 if two_ref_images and not fun_or_fl2v_model else 0)
        if temporal_mask is None:
            mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device, dtype=vae.dtype)
            if start_image is not None:
                mask[:, 0:start_image.shape[0]] = 1  # First frame
            if end_image is not None:
                mask[:, -end_image.shape[0]:] = 1  # End frame if exists
        else:
            mask = common_upscale(temporal_mask.unsqueeze(1).to(device), lat_w, lat_h, "nearest", "disabled").squeeze(1)
            if mask.shape[0] > base_frames:
                mask = mask[:base_frames]
            elif mask.shape[0] < base_frames:
                mask = torch.cat([mask, torch.zeros(base_frames - mask.shape[0], lat_h, lat_w, device=device)])
            mask = mask.unsqueeze(0).to(device, vae.dtype)

        pixel_mask = mask.clone()

        # Repeat first frame and optionally end frame
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1) # T, C, H, W
        if end_image is not None and not fun_or_fl2v_model:
            end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1) # T, C, H, W
            mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        else:
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)

        # Reshape mask into groups of 4 frames
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w) # 1, T, C, H, W
        mask = mask.movedim(1, 2)[0]# C, T, H, W

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            start_image = start_image[..., :3]
            if start_image.shape[1] != H or start_image.shape[2] != W:
                resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_start_image = start_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_start_image = resized_start_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_start_image = add_noise_to_reference_video(resized_start_image, ratio=noise_aug_strength)

        if end_image is not None:
            end_image = end_image[..., :3]
            if end_image.shape[1] != H or end_image.shape[2] != W:
                resized_end_image = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_end_image = end_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_end_image = resized_end_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_end_image = add_noise_to_reference_video(resized_end_image, ratio=noise_aug_strength)

        # Concatenate image with zero frames and encode
        if start_image is not None and end_image is None:
            zero_frames = torch.zeros(3, num_frames-start_image.shape[0], H, W, device=device, dtype=vae.dtype)
            concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames], dim=1)
            del resized_start_image, zero_frames
        elif start_image is None and end_image is not None:
            zero_frames = torch.zeros(3, num_frames-end_image.shape[0], H, W, device=device, dtype=vae.dtype)
            concatenated = torch.cat([zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
            del zero_frames
        elif start_image is None and end_image is None:
            concatenated = torch.zeros(3, num_frames, H, W, device=device, dtype=vae.dtype)
        else:
            if fun_or_fl2v_model:
                zero_frames = torch.zeros(3, num_frames-(start_image.shape[0]+end_image.shape[0]), H, W, device=device, dtype=vae.dtype)
            else:
                zero_frames = torch.zeros(3, num_frames-1, H, W, device=device, dtype=vae.dtype)
            concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
            del resized_start_image, zero_frames

        if empty_frame_pad_image is not None:
            pad_img = empty_frame_pad_image.clone()[..., :3]
            if pad_img.shape[1] != H or pad_img.shape[2] != W:
                pad_img = common_upscale(pad_img.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
            pad_img = (pad_img.movedim(-1, 0) * 2 - 1).to(device, dtype=vae.dtype)

            num_pad_frames = pad_img.shape[1]
            num_target_frames = concatenated.shape[1]
            if num_pad_frames < num_target_frames:
                pad_img = torch.cat([pad_img, pad_img[:, -1:].expand(-1, num_target_frames - num_pad_frames, -1, -1)], dim=1)
            else:
                pad_img = pad_img[:, :num_target_frames]

            frame_is_empty = (pixel_mask[0].mean(dim=(-2, -1)) < 0.5)[:concatenated.shape[1]].clone()
            if start_image is not None:
                frame_is_empty[:start_image.shape[0]] = False
            if end_image is not None:
                frame_is_empty[-end_image.shape[0]:] = False

            concatenated[:, frame_is_empty] = pad_img[:, frame_is_empty]

        mm.soft_empty_cache()
        gc.collect()

        vae.to(device)
        y = vae.encode([concatenated], device, end_=(end_image is not None and not fun_or_fl2v_model),tiled=tiled_vae)[0]
        del concatenated

        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            y = torch.cat([samples, y], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, 0:samples.shape[1]]), mask], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        y[:, :1] *= start_latent_strength
        y[:, -1:] *= end_latent_strength
        if augment_empty_frames > 0.0:
            frame_is_empty = (mask[0].mean(dim=(-2, -1)) < 0.5).view(1, -1, 1, 1)
            y = y[:, :1] + (y - y[:, :1]) * ((augment_empty_frames+1) * frame_is_empty + ~frame_is_empty)

        # Calculate maximum sequence length
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        frames_per_stride = (num_frames - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1)
        max_seq_len = frames_per_stride * patches_per_frame

        if add_cond_latents is not None:
            add_cond_latents["ref_latent_neg"] = vae.encode(torch.zeros(1, 3, 1, H, W, device=device, dtype=vae.dtype), device)

        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "image_embeds": y.cpu(),
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "end_image": resized_end_image if end_image is not None else None,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            "add_cond_latents": add_cond_latents,
            "mask": mask.cpu()
        }

        return (image_embeds,)

# region WanAnimate
class WanVideoAnimateEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            "frame_window_size": ("INT", {"default": 77, "min": 1, "max": 10000, "step": 1, "tooltip": "Number of frames to use for temporal attention window"}),
            "colormatch": (
            [
                'disabled',
                'mkl',
                'hm',
                'reinhard',
                'mvgd',
                'hm-mvgd-hm',
                'hm-mkl-hm',
            ], {
               "default": 'disabled', "tooltip": "Color matching method to use between the windows"
            },),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the pose"}),
            "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the face"}),
            },
            "optional": {
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "ref_images": ("IMAGE", {"tooltip": "Image to encode"}),
                "pose_images": ("IMAGE", {"tooltip": "end frame"}),
                "face_images": ("IMAGE", {"tooltip": "end frame"}),
                "bg_images": ("IMAGE", {"tooltip": "background images"}),
                "mask": ("MASK", {"tooltip": "mask"}),
                "start_ref_image": ("IMAGE", {"tooltip": "start ref image"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, force_offload, frame_window_size, colormatch, pose_strength, face_strength,
                ref_images=None, pose_images=None, face_images=None, clip_embeds=None, tiled_vae=False, bg_images=None, mask=None, start_ref_image=None):
        
        W = (width // 16) * 16
        H = (height // 16) * 16

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_refs = ref_images.shape[0] if ref_images is not None else 0
        num_frames = ((num_frames - 1) // 4) * 4 + 1

        looping = num_frames > frame_window_size or start_ref_image is not None

        if num_frames < frame_window_size:
            frame_window_size = num_frames

        target_shape = (16, (num_frames - 1) // 4 + 1 + num_refs, lat_h, lat_w)
        latent_window_size = ((frame_window_size - 1) // 4)

        if not looping:
            num_frames = num_frames + num_refs * 4
        else:
            latent_window_size = latent_window_size + 1

        mm.soft_empty_cache()
        gc.collect()
        vae.to(device)
        # Resize and rearrange the input image dimensions
        pose_latents = ref_latent = None
        if pose_images is not None:
            pose_images = pose_images[..., :3]
            if pose_images.shape[1] != H or pose_images.shape[2] != W:
                resized_pose_images = common_upscale(pose_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_pose_images = pose_images.permute(3, 0, 1, 2) # C, T, H, W
            resized_pose_images = resized_pose_images * 2 - 1
            if not looping:
                pose_latents = vae.encode([resized_pose_images.to(device, vae.dtype)], device,tiled=tiled_vae)
                pose_latents = pose_latents.to(offload_device)
            
                if pose_latents.shape[2] < latent_window_size:
                    log.info(f"WanAnimate: Padding pose latents from {pose_latents.shape} to length {latent_window_size}")
                    pad_len = latent_window_size - pose_latents.shape[2]
                    pad = torch.zeros(pose_latents.shape[0], pose_latents.shape[1], pad_len, pose_latents.shape[3], pose_latents.shape[4], device=pose_latents.device, dtype=pose_latents.dtype)
                    pose_latents = torch.cat([pose_latents, pad], dim=2)
                del resized_pose_images
            else:
                resized_pose_images = resized_pose_images.to(offload_device, dtype=vae.dtype)            

        bg_latents = None
        if bg_images is not None:
            if bg_images.shape[1] != H or bg_images.shape[2] != W:
                resized_bg_images = common_upscale(bg_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_bg_images = bg_images.permute(3, 0, 1, 2) # C, T, H, W
            resized_bg_images = (resized_bg_images[:3] * 2 - 1)

        if not looping:
            if bg_images is None:
                resized_bg_images = torch.zeros(3, num_frames - num_refs, H, W, device=device, dtype=vae.dtype)
            bg_latents = vae.encode([resized_bg_images.to(device, vae.dtype)], device,tiled=tiled_vae)[0].to(offload_device)
            del resized_bg_images
        elif bg_images is not None:
            resized_bg_images = resized_bg_images.to(offload_device, dtype=vae.dtype)

        if ref_images is not None:
            if ref_images.shape[1] != H or ref_images.shape[2] != W:
                resized_ref_images = common_upscale(ref_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_ref_images = ref_images.permute(3, 0, 1, 2) # C, T, H, W
            resized_ref_images = resized_ref_images[:3] * 2 - 1

            ref_latent = vae.encode([resized_ref_images.to(device, vae.dtype)], device,tiled=tiled_vae)[0]
            msk = torch.zeros(4, 1, lat_h, lat_w, device=device, dtype=vae.dtype)
            msk[:, :num_refs] = 1
            ref_latent_masked = torch.cat([msk, ref_latent], dim=0).to(offload_device) # 4+C 1 H W

            if mask is None:
                bg_mask = torch.zeros(1, num_frames, lat_h, lat_w, device=offload_device, dtype=vae.dtype)
            else:
                bg_mask = 1 - mask[:num_frames]
                if bg_mask.shape[0] < num_frames and not looping:
                    bg_mask = torch.cat([bg_mask, bg_mask[-1:].repeat(num_frames - bg_mask.shape[0], 1, 1)], dim=0)
                bg_mask = common_upscale(bg_mask.unsqueeze(1), lat_w, lat_h, "nearest", "disabled").squeeze(1)
                bg_mask = bg_mask.unsqueeze(-1).permute(3, 0, 1, 2).to(offload_device, vae.dtype) # C, T, H, W
            
            if bg_images is None and looping:
                bg_mask[:, :num_refs] = 1
            bg_mask_mask_repeated = torch.repeat_interleave(bg_mask[:, 0:1], repeats=4, dim=1) # T, C, H, W
            bg_mask = torch.cat([bg_mask_mask_repeated, bg_mask[:, 1:]], dim=1)
            bg_mask = bg_mask.view(1, bg_mask.shape[1] // 4, 4, lat_h, lat_w) # 1, T, C, H, W
            bg_mask = bg_mask.movedim(1, 2)[0]# C, T, H, W

            if not looping:
                bg_latents_masked = torch.cat([bg_mask[:, :bg_latents.shape[1]], bg_latents], dim=0)
                del bg_mask, bg_latents
                ref_latent = torch.cat([ref_latent_masked, bg_latents_masked], dim=1)
            else:
                ref_latent = ref_latent_masked

        if face_images is not None:
            face_images = face_images[..., :3]
            if face_images.shape[1] != 512 or face_images.shape[2] != 512:
                resized_face_images = common_upscale(face_images.movedim(-1, 1), 512, 512, "lanczos", "center").movedim(0, 1)
            else:
                resized_face_images = face_images.permute(3, 0, 1, 2) # B, C, T, H, W
            resized_face_images = (resized_face_images * 2 - 1).unsqueeze(0)
            resized_face_images = resized_face_images.to(offload_device, dtype=vae.dtype)

        if start_ref_image is not None:
            if start_ref_image.shape[1] != H or start_ref_image.shape[2] != W:
                resized_start_ref_image = common_upscale(start_ref_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_start_ref_image = start_ref_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_start_ref_image = resized_start_ref_image[:3] * 2 - 1

        seq_len = math.ceil((target_shape[2] * target_shape[3]) / 4 * target_shape[1])
        
        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": seq_len,
            "pose_latents": pose_latents,
            "pose_images": resized_pose_images if pose_images is not None and looping else None,
            "bg_images": resized_bg_images if bg_images is not None and looping else None,
            "ref_masks": bg_mask if mask is not None and looping else None,
            "is_masked": mask is not None,
            "ref_latent": ref_latent,
            "ref_image": resized_ref_images if ref_images is not None else None,
            "start_ref_image": resized_start_ref_image if start_ref_image is not None else None,
            "face_pixels": resized_face_images if face_images is not None else None,
            "num_frames": num_frames,
            "target_shape": target_shape,
            "frame_window_size": frame_window_size,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "vae": vae,
            "colormatch": colormatch,
            "looping": looping,
            "pose_strength": pose_strength,
            "face_strength": face_strength,
        }

        return (image_embeds,)

# region UniLumos
class WanVideoUniLumosEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            },
            "optional": {
                "foreground_latents": ("LATENT", {"tooltip": "Video foreground latents"}),
                "background_latents": ("LATENT", {"tooltip": "Video background latents"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, foreground_latents=None, background_latents=None):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
        }
        if foreground_latents is not None:
            embeds["foreground_latents"] = foreground_latents["samples"][0]
        else:
            embeds["foreground_latents"] = torch.zeros(target_shape[0], target_shape[1], target_shape[2], target_shape[3], device=torch.device("cpu"), dtype=torch.float32)
        if background_latents is not None:
            embeds["background_latents"] = background_latents["samples"][0]
        else:
            embeds["background_latents"] = torch.zeros(target_shape[0], target_shape[1], target_shape[2], target_shape[3], device=torch.device("cpu"), dtype=torch.float32)

        return (embeds,)
    
class WanVideoEmptyEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            },
            "optional": {
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "control signal for the Fun -model"}),
                "extra_latents": ("LATENT", {"tooltip": "First latent to use for the Pusa -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, control_embeds=None, extra_latents=None):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
        }
        if extra_latents is not None:
            embeds["extra_latents"] = [{
                "samples": extra_latents["samples"],
                "index": 0,
            }]

        return (embeds,)
    
class WanVideoAddExtraLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "extra_latents": ("LATENT",),
                    "latent_index": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1, "tooltip": "Index to insert the extra latents at in latent space"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, extra_latents, latent_index):
        # Prepare the new extra latent entry
        new_entry = {
            "samples": extra_latents["samples"],
            "index": latent_index,
        }
        # Get previous extra_latents list, or start a new one
        prev_extra_latents = embeds.get("extra_latents", None)
        if prev_extra_latents is None:
            extra_latents_list = [new_entry]
        elif isinstance(prev_extra_latents, list):
            extra_latents_list = prev_extra_latents + [new_entry]
        else:
            extra_latents_list = [prev_extra_latents, new_entry]

        # Return a new dict with updated extra_latents
        updated = dict(embeds)
        updated["extra_latents"] = extra_latents_list
        return (updated,)
    
class WanVideoAddLucyEditLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "extra_latents": ("LATENT",),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, extra_latents):
        updated = dict(embeds)
        updated["extra_channel_latents"] = extra_latents["samples"]
        return (updated,)

class WanVideoMiniMaxRemoverEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
            "mask_latents": ("LATENT", {"tooltip": "Encoded latents to use as mask"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, latents, mask_latents):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "minimax_latents": latents["samples"].squeeze(0),
            "minimax_mask_latents": mask_latents["samples"].squeeze(0),
        }
    
        return (embeds,)
    
# region phantom
class WanVideoPhantomEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "phantom_latent_1": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
            
            "phantom_cfg_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "CFG scale for the extra phantom cond pass"}),
            "phantom_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the phantom model"}),
            "phantom_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the phantom model"}),
            },
            "optional": {
                "phantom_latent_2": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "phantom_latent_3": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "phantom_latent_4": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "vace_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "VACE embeds"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, phantom_cfg_scale, phantom_start_percent, phantom_end_percent, phantom_latent_1, phantom_latent_2=None, phantom_latent_3=None, phantom_latent_4=None, vace_embeds=None):
        samples = phantom_latent_1["samples"].squeeze(0)
        if phantom_latent_2 is not None:
            samples = torch.cat([samples, phantom_latent_2["samples"].squeeze(0)], dim=1)
        if phantom_latent_3 is not None:
            samples = torch.cat([samples, phantom_latent_3["samples"].squeeze(0)], dim=1)
        if phantom_latent_4 is not None:
            samples = torch.cat([samples, phantom_latent_4["samples"].squeeze(0)], dim=1)
        C, T, H, W = samples.shape

        log.info(f"Phantom latents shape: {samples.shape}")

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        H * 8 // VAE_STRIDE[1],
                        W * 8 // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "phantom_latents": samples,
            "phantom_cfg_scale": phantom_cfg_scale,
            "phantom_start_percent": phantom_start_percent,
            "phantom_end_percent": phantom_end_percent,
        }
        if vace_embeds is not None:
            vace_input = {
                "vace_context": vace_embeds["vace_context"],
                "vace_scale": vace_embeds["vace_scale"],
                "has_ref": vace_embeds["has_ref"],
                "vace_start_percent": vace_embeds["vace_start_percent"],
                "vace_end_percent": vace_embeds["vace_end_percent"],
                "vace_seq_len": vace_embeds["vace_seq_len"],
                "additional_vace_inputs": vace_embeds["additional_vace_inputs"],
                }
            embeds.update(vace_input)
    
        return (embeds,)
    
class WanVideoControlEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
            },
            "optional": {
                "fun_ref_image": ("LATENT", {"tooltip": "Reference latent for the Fun 1.1 -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, latents, start_percent, end_percent, fun_ref_image=None):
        samples = latents["samples"].squeeze(0)
        C, T, H, W = samples.shape

        num_frames = (T - 1) * 4 + 1
        seq_len = math.ceil((H * W) / 4 * ((num_frames - 1) // 4 + 1))
      
        embeds = {
            "max_seq_len": seq_len,
            "target_shape": samples.shape,
            "num_frames": num_frames,
            "control_embeds": {
                "control_images": samples,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "fun_ref_image": fun_ref_image["samples"][:,:, 0] if fun_ref_image is not None else None,
            }
        }
    
        return (embeds,)
    
class WanVideoAddControlEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("WANVIDIMAGE_EMBEDS",),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            },
            "optional": {
                "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
                "fun_ref_image": ("LATENT", {"tooltip": "Reference latent for the Fun 1.1 -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, embeds, start_percent, end_percent, fun_ref_image=None, latents=None):      
        new_entry = {
            "control_images": latents["samples"].squeeze(0) if latents is not None else None,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "fun_ref_image": fun_ref_image["samples"][:,:, 0] if fun_ref_image is not None else None,
        }

        updated = dict(embeds)
        updated["control_embeds"] = new_entry

        return (updated,)
    
class WanVideoAddPusaNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("WANVIDIMAGE_EMBEDS",),
            "noise_multipliers": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Noise multipliers for Pusa, can be a list of floats"}),
            "noisy_steps": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "Number steps to apply the extra noise"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Adds latent and timestep noise multipliers when using flowmatch_pusa"

    def add(self, embeds, noise_multipliers, noisy_steps):
        updated = dict(embeds)
        updated["pusa_noise_multipliers"] = noise_multipliers
        updated["pusa_noisy_steps"] = noisy_steps

        return (updated,)
    
class WanVideoSLG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "blocks": ("STRING", {"default": "10", "tooltip": "Blocks to skip uncond on, separated by comma, index starts from 0"}),
            "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            },
        }

    RETURN_TYPES = ("SLGARGS", )
    RETURN_NAMES = ("slg_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Skips uncond on the selected blocks"

    def process(self, blocks, start_percent, end_percent):
        slg_block_list = [int(x.strip()) for x in blocks.split(",")]

        slg_args = {
            "blocks": slg_block_list,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }
        return (slg_args,)

#region VACE
class WanVideoVACEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply VACE"}),
            "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply VACE"}),
            },
            "optional": {
                "input_frames": ("IMAGE",),
                "ref_images": ("IMAGE",),
                "input_masks": ("MASK",),
                "prev_vace_embeds": ("WANVIDIMAGE_EMBEDS",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, strength, vace_start_percent, vace_end_percent, input_frames=None, ref_images=None, input_masks=None, prev_vace_embeds=None, tiled_vae=False):
        width = (width // 16) * 16
        height = (height // 16) * 16

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        # vace context encode
        if input_frames is None:
            input_frames = torch.zeros((1, 3, num_frames, height, width), device=device, dtype=vae.dtype)
        else:
            input_frames = input_frames.clone()[:num_frames, :, :, :3]
            input_frames = common_upscale(input_frames.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            input_frames = input_frames.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
            input_frames = input_frames * 2 - 1
        if input_masks is None:
            input_masks = torch.ones_like(input_frames, device=device)
        else:
            log.info(f"input_masks shape: {input_masks.shape}")
            input_masks = input_masks[:num_frames]
            input_masks = common_upscale(input_masks.clone().unsqueeze(1), width, height, "nearest-exact", "disabled").squeeze(1)
            input_masks = input_masks.to(vae.dtype).to(device)
            input_masks = input_masks.unsqueeze(-1).unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(1, 3, 1, 1, 1) # B, C, T, H, W

        if ref_images is not None:
            ref_images = ref_images.clone()[..., :3]
            # Create padded image
            if ref_images.shape[0] > 1:
                ref_images = torch.cat([ref_images[i] for i in range(ref_images.shape[0])], dim=1).unsqueeze(0)
            
            B, H, W, C = ref_images.shape
            current_aspect = W / H
            target_aspect = width / height
            if current_aspect > target_aspect:
                # Image is wider than target, pad height
                new_h = int(W / target_aspect)
                pad_h = (new_h - H) // 2
                padded = torch.ones(ref_images.shape[0], new_h, W, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, pad_h:pad_h+H, :, :] = ref_images
                ref_images = padded
            elif current_aspect < target_aspect:
                # Image is taller than target, pad width
                new_w = int(H * target_aspect)
                pad_w = (new_w - W) // 2
                padded = torch.ones(ref_images.shape[0], H, new_w, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, :, pad_w:pad_w+W, :] = ref_images
                ref_images = padded
            ref_images = common_upscale(ref_images.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
            
            ref_images = ref_images.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3).unsqueeze(0)
            ref_images = ref_images * 2 - 1

        vae = vae.to(device)
        z0 = self.vace_encode_frames(vae, input_frames, ref_images, masks=input_masks, tiled_vae=tiled_vae)
        
        m0 = self.vace_encode_masks(input_masks, ref_images)
        z = self.vace_latent(z0, m0)
        vae.to(offload_device)

        vace_input = {
            "vace_context": z,
            "vace_scale": strength,
            "has_ref": ref_images is not None,
            "num_frames": num_frames,
            "target_shape": target_shape,
            "vace_start_percent": vace_start_percent,
            "vace_end_percent": vace_end_percent,
            "vace_seq_len": math.ceil((z[0].shape[2] * z[0].shape[3]) / 4 * z[0].shape[1]),
            "additional_vace_inputs": [],
        }

        if prev_vace_embeds is not None:
            if "additional_vace_inputs" in prev_vace_embeds and prev_vace_embeds["additional_vace_inputs"]:
                vace_input["additional_vace_inputs"] = prev_vace_embeds["additional_vace_inputs"].copy()
            vace_input["additional_vace_inputs"].append(prev_vace_embeds)
    
        return (vace_input,)
    
    def vace_encode_frames(self, vae, frames, ref_images, masks=None, tiled_vae=False):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        pbar = ProgressBar(len(frames))
        if masks is None:
            latents = vae.encode(frames, device=device, tiled=tiled_vae)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            del frames
            inactive = vae.encode(inactive, device=device, tiled=tiled_vae)
            reactive = vae.encode(reactive, device=device, tiled=tiled_vae)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
            del inactive, reactive
        
        
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                else:
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
            pbar.update(1)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        pbar = ProgressBar(len(masks))
        for mask, refs in zip(masks, ref_images):
            _c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // VAE_STRIDE[0])
            height = 2 * (int(height) // (VAE_STRIDE[1] * 2))
            width = 2 * (int(width) // (VAE_STRIDE[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, VAE_STRIDE[1], width, VAE_STRIDE[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                VAE_STRIDE[1] * VAE_STRIDE[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
            pbar.update(1)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]


#region context options
class WanVideoContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context_schedule": (["uniform_standard", "uniform_looped", "static_standard"],),
            "context_frames": ("INT", {"default": 81, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of pixel frames in the context, NOTE: the latent space has 4 frames in 1"} ),
            "context_stride": ("INT", {"default": 4, "min": 4, "max": 100, "step": 1, "tooltip": "Context stride as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "context_overlap": ("INT", {"default": 16, "min": 4, "max": 100, "step": 1, "tooltip": "Context overlap as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "freenoise": ("BOOLEAN", {"default": True, "tooltip": "Shuffle the noise"}),
            "verbose": ("BOOLEAN", {"default": False, "tooltip": "Print debug output"}),
            },
            "optional": {
                "fuse_method": (["linear", "pyramid"], {"default": "linear", "tooltip": "Window weight function: linear=ramps at edges only, pyramid=triangular weights peaking in middle"}),
                "reference_latent": ("LATENT", {"tooltip": "Image to be used as init for I2V models for windows where first frame is not the actual first frame. Mostly useful with MAGREF model"}),
            }
        }

    RETURN_TYPES = ("WANVIDCONTEXT", )
    RETURN_NAMES = ("context_options",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Context options for WanVideo, allows splitting the video into context windows and attemps blending them for longer generations than the model and memory otherwise would allow."

    def process(self, context_schedule, context_frames, context_stride, context_overlap, freenoise, verbose, image_cond_start_step=6, image_cond_window_count=2, vae=None, fuse_method="linear", reference_latent=None):
        context_options = {
            "context_schedule":context_schedule,
            "context_frames":context_frames,
            "context_stride":context_stride,
            "context_overlap":context_overlap,
            "freenoise":freenoise,
            "verbose":verbose,
            "fuse_method":fuse_method,
            "reference_latent":reference_latent["samples"] if reference_latent is not None else None,
        }

        return (context_options,)

class WanVideoLoopArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "shift_skip": ("INT", {"default": 6, "min": 0, "tooltip": "Skip step of latent shift"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the looping effect"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the looping effect"}),
            },
        }

    RETURN_TYPES = ("LOOPARGS", )
    RETURN_NAMES = ("loop_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Looping through latent shift as shown in https://github.com/YisuiTT/Mobius/"

    def process(self, **kwargs):
        return (kwargs,)

class WanVideoExperimentalArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "video_attention_split_steps": ("STRING", {"default": "", "tooltip": "Steps to split self attention when using multiple prompts"}),
                "cfg_zero_star": ("BOOLEAN", {"default": False, "tooltip": "https://github.com/WeichenFan/CFG-Zero-star"}),
                "use_zero_init": ("BOOLEAN", {"default": False}),
                "zero_star_steps": ("INT", {"default": 0, "min": 0, "tooltip": "Steps to split self attention when using multiple prompts"}),
                "use_fresca": ("BOOLEAN", {"default": False, "tooltip": "https://github.com/WikiChao/FreSca"}),
                "fresca_scale_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fresca_scale_high": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fresca_freq_cutoff": ("INT", {"default": 20, "min": 0, "max": 10000, "step": 1}),
                "use_tcfg": ("BOOLEAN", {"default": False, "tooltip": "https://arxiv.org/abs/2503.18137 TCFG: Tangential Damping Classifier-free Guidance. CFG artifacts reduction."}),
                "raag_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Alpha value for RAAG, 1.0 is default, 0.0 is disabled."}),
                "bidirectional_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable bidirectional sampling, based on https://github.com/ff2416/WanFM"}),
                "temporal_score_rescaling": ("BOOLEAN", {"default": False, "tooltip": "Enable temporal score rescaling: https://github.com/temporalscorerescaling/TSR/"}),
                "tsr_k": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "The sampling temperature"}),
                "tsr_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How early TSR steer the sampling process"}),
            },
        }

    RETURN_TYPES = ("EXPERIMENTALARGS", )
    RETURN_NAMES = ("exp_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Experimental stuff"
    EXPERIMENTAL = True

    def process(self, **kwargs):
        return (kwargs,)
    
class WanVideoFreeInitArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "freeinit_num_iters": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Number of FreeInit iterations"}),
                "freeinit_method": (["butterworth", "ideal", "gaussian", "none"], {"default": "ideal", "tooltip": "Frequency filter type"}),
                "freeinit_n": ("INT", {"default": 4, "min": 1, "max": 10, "tooltip": "Butterworth filter order (only for butterworth)"}),
                "freeinit_d_s": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Spatial filter cutoff"}),
                "freeinit_d_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Temporal filter cutoff"}),
            },
        }

    RETURN_TYPES = ("FREEINITARGS", )
    RETURN_NAMES = ("freeinit_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/TianxingWu/FreeInit; FreeInit, a concise yet effective method to improve temporal consistency of videos generated by diffusion models"
    EXPERIMENTAL = True

    def process(self, **kwargs):
        return (kwargs,)

rope_functions = ["default", "comfy", "comfy_chunked"]
class WanVideoRoPEFunction:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "rope_function": (rope_functions, {"default": "comfy"}),
                "ntk_scale_f": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "ntk_scale_h": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "ntk_scale_w": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (rope_functions, )
    RETURN_NAMES = ("rope_function",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True

    def process(self, rope_function, ntk_scale_f, ntk_scale_h, ntk_scale_w):
        if ntk_scale_f != 1.0 or ntk_scale_h != 1.0 or ntk_scale_w != 1.0:
            rope_func_dict = {
                "rope_function": rope_function,
                "ntk_scale_f": ntk_scale_f,
                "ntk_scale_h": ntk_scale_h,
                "ntk_scale_w": ntk_scale_w,
            }
            return (rope_func_dict,)
        return (rope_function,)

#region TTM
class WanVideoAddTTMLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("WANVIDIMAGE_EMBEDS",),
            "reference_latents": ("LATENT", {"tooltip": "Latents used as reference for TTM"}),
            "mask": ("MASK", {"tooltip": "Mask used for TTM"}),
            "start_step": ("INT", {"default": 0, "min": -1, "max": 1000, "step": 1, "tooltip": "Start step for whole denoising process"}),
            "end_step": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1, "tooltip": "The step to stop applying TTM"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds", )
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/time-to-move/TTM"

    def add(self, embeds, reference_latents, mask, start_step, end_step):

        if end_step < max(0, start_step):
            raise ValueError(f"`end_step` ({end_step}) must be >= `start_step` ({start_step}).")

        mask_sampled = mask[::4]
        mask_sampled = mask_sampled.unsqueeze(1).unsqueeze(0)  # [1, T, 1, H, W]

        vae_upscale_factor = 8
        if reference_latents["samples"].shape[1] == 48:
            vae_upscale_factor = 16

        # Upsample spatially to latent resolution
        H_latent = mask_sampled.shape[-2] // vae_upscale_factor
        W_latent = mask_sampled.shape[-1] // vae_upscale_factor
        mask_latent = F.interpolate(
            mask_sampled.float(),
            size=(mask_sampled.shape[2], H_latent, W_latent),
            mode="nearest"
        )

        updated = dict(embeds)
        updated["ttm_reference_latents"] = reference_latents["samples"].squeeze(0)
        updated["ttm_mask"] = mask_latent.squeeze(0).movedim(1, 0)  # [T, 1, H, W]
        updated["ttm_start_step"] = start_step
        updated["ttm_end_step"] = end_step

        return (updated,)

#region VideoDecode
class WanVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": (
                        "Drastically reduces memory use but will introduce seams at tile stride boundaries. "
                        "The location and number of seams is dictated by the tile stride size. "
                        "The visibility of seams can be controlled by increasing the tile size. "
                        "Seams become less obvious at 1.5x stride and are barely noticeable at 2x stride size. "
                        "Which is to say if you use a stride width of 160, the seams are barely noticeable with a tile width of 320."
                    )}),
                    "tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile width in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile height in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride width in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride height in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    },
                    "optional": {
                        "normalization": (["default", "minmax", "none"], {"advanced": True}),
                    }
                }

    @classmethod
    def VALIDATE_INPUTS(s, tile_x, tile_y, tile_stride_x, tile_stride_y):
        if tile_x <= tile_stride_x:
            return "Tile width must be larger than the tile stride width."
        if tile_y <= tile_stride_y:
            return "Tile height must be larger than the tile stride height."
        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper"

    def decode(self, vae, samples, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, normalization="default"):
        mm.soft_empty_cache()
        video = samples.get("video", None)
        if video is not None:
            video.clamp_(-1.0, 1.0)
            video.add_(1.0).div_(2.0)
            return video.cpu().float(),
        latents = samples["samples"].clone()
        end_image = samples.get("end_image", None)
        has_ref = samples.get("has_ref", False)
        drop_last = samples.get("drop_last", False)
        is_looped = samples.get("looped", False)

        flashvsr_LQ_images = samples.get("flashvsr_LQ_images", None)

        vae.to(device)

        latents = latents.to(device = device, dtype = vae.dtype)

        mm.soft_empty_cache()

        if has_ref:
            latents = latents[:, :, 1:]
        if drop_last:
            latents = latents[:, :, :-1]

        if type(vae).__name__ == "TAEHV":
            images = vae.decode_video(latents.permute(0, 2, 1, 3, 4), cond=flashvsr_LQ_images.to(vae.dtype) if flashvsr_LQ_images is not None else None)[0].permute(1, 0, 2, 3)
            images = torch.clamp(images, 0.0, 1.0)
            images = images.permute(1, 2, 3, 0).cpu().float()
            return (images,)
        else:
            images = vae.decode(latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//8, tile_y//8), tile_stride=(tile_stride_x//8, tile_stride_y//8))[0]


        images = images.cpu().float()

        if normalization != "none":
            if normalization == "minmax":
                images.sub_(images.min()).div_(images.max() - images.min())
            else:
                images.clamp_(-1.0, 1.0)
                images.add_(1.0).div_(2.0)

        if is_looped:
            temp_latents = torch.cat([latents[:, :, -3:]] + [latents[:, :, :2]], dim=2)
            temp_images = vae.decode(temp_latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))[0]
            temp_images = temp_images.cpu().float()
            temp_images = (temp_images - temp_images.min()) / (temp_images.max() - temp_images.min())
            images = torch.cat([temp_images[:, 9:].to(images), images[:, 5:]], dim=1)

        if end_image is not None: 
            images = images[:, 0:-1]


        vae.to(offload_device)
        mm.soft_empty_cache()

        images.clamp_(0.0, 1.0)

        return (images.permute(1, 2, 3, 0),)

#region VideoEncode
class WanVideoEncodeLatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "images": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes a batch of images individually to create a latent video batch where each video is a single frame, useful for I2V init purposes, for example as multiple context window inits"

    def encode(self, vae, images, enable_vae_tiling=False, tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128, latent_strength=1.0):
        vae.to(device)

        images = images.clone()

        B, H, W, C = images.shape
        if W % 16 != 0 or H % 16 != 0:
            new_height = (H // 16) * 16
            new_width = (W // 16) * 16
            log.warning(f"Image size {W}x{H} is not divisible by 16, resizing to {new_width}x{new_height}")
            images = common_upscale(images.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)

        if images.shape[-1] == 4:
            images = images[..., :3]
        images = images.to(vae.dtype).to(device) * 2.0 - 1.0

        latent_list = []
        for img in images:
            if enable_vae_tiling and tile_x is not None:
                latent = vae.encode(img.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3), device=device, tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))
            else:
                latent = vae.encode(img.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3), device=device, tiled=enable_vae_tiling)

            if latent_strength != 1.0:
                latent *= latent_strength
            latent_list.append(latent.squeeze(0).cpu())
        latents_out = torch.stack(latent_list, dim=0)

        log.info(f"WanVideoEncode: Encoded latents shape {latents_out.shape}")
        vae.to(offload_device)
        mm.soft_empty_cache()

        return ({"samples": latents_out},)

class WanVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "driving_video": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                    "optional": {
                        "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for leapfusion I2V where some noise can add motion and give sharper results"}),
                        "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for leapfusion I2V where lower values allow for more motion"}),
                        "mask": ("MASK", ),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, vae, driving_video, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, noise_aug_strength=0.0, latent_strength=1.0, mask=None):
        if scail_pose2_mask_disables_samples(mask):
            payload = build_disabled_samples_payload(mask)
            log.info(
                "WanVideoEncode: SCAIL-Pose2 disabled samples path "
                f"condition_mode={payload['scail_pose2_condition_mode']} "
                f"reason={payload['scail_pose2_disable_reason']}"
            )
            return (payload,)

        vae.to(device)

        image = driving_video.clone()

        B, H, W, C = image.shape
        if W % 16 != 0 or H % 16 != 0:
            new_height = (H // 16) * 16
            new_width = (W // 16) * 16
            log.warning(f"Image size {W}x{H} is not divisible by 16, resizing to {new_width}x{new_height}")
            image = common_upscale(image.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)

        if image.shape[-1] == 4:
            image = image[..., :3]
        image = image.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W        

        if noise_aug_strength > 0.0:
            image = add_noise_to_reference_video(image, ratio=noise_aug_strength)

        if isinstance(vae, TAEHV):
            latents = vae.encode_video(image.permute(0, 2, 1, 3, 4), parallel=False)# B, T, C, H, W
            latents = latents.permute(0, 2, 1, 3, 4)
        else:
            latents = vae.encode(image * 2.0 - 1.0, device=device, tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))

            vae.to(offload_device)
        if latent_strength != 1.0:
            latents *= latent_strength

        latents = latents.cpu()

        log.info(f"WanVideoEncode: Encoded latents shape {latents.shape}")
        mm.soft_empty_cache()

        return ({"samples": latents, "noise_mask": mask},)

NODE_CLASS_MAPPINGS = {
    "WanVideoDecode": WanVideoDecode,
    "WanVideoTextEncode": WanVideoTextEncode,
    "WanVideoTextEncodeSingle": WanVideoTextEncodeSingle,
    "WanVideoClipVisionEncode": WanVideoClipVisionEncode,
    "WanVideoImageToVideoEncode": WanVideoImageToVideoEncode,
    "WanVideoEncode": WanVideoEncode,
    "WanVideoEncodeLatentBatch": WanVideoEncodeLatentBatch,
    "WanVideoEmptyEmbeds": WanVideoEmptyEmbeds,
    "WanVideoEnhanceAVideo": WanVideoEnhanceAVideo,
    "WanVideoContextOptions": WanVideoContextOptions,
    "WanVideoTextEmbedBridge": WanVideoTextEmbedBridge,
    "WanVideoControlEmbeds": WanVideoControlEmbeds,
    "WanVideoSLG": WanVideoSLG,
    "WanVideoLoopArgs": WanVideoLoopArgs,
    "WanVideoSetBlockSwap": WanVideoSetBlockSwap,
    "WanVideoExperimentalArgs": WanVideoExperimentalArgs,
    "WanVideoVACEEncode": WanVideoVACEEncode,
    "WanVideoPhantomEmbeds": WanVideoPhantomEmbeds,
    "WanVideoRealisDanceLatents": WanVideoRealisDanceLatents,
    "WanVideoApplyNAG": WanVideoApplyNAG,
    "WanVideoMiniMaxRemoverEmbeds": WanVideoMiniMaxRemoverEmbeds,
    "WanVideoFreeInitArgs": WanVideoFreeInitArgs,
    "WanVideoSetRadialAttention": WanVideoSetRadialAttention,
    "WanVideoBlockList": WanVideoBlockList,
    "WanVideoTextEncodeCached": WanVideoTextEncodeCached,
    "WanVideoAddExtraLatent": WanVideoAddExtraLatent,
    "WanVideoAddStandInLatent": WanVideoAddStandInLatent,
    "WanVideoAddControlEmbeds": WanVideoAddControlEmbeds,
    "WanVideoAddMTVMotion": WanVideoAddMTVMotion,
    "WanVideoRoPEFunction": WanVideoRoPEFunction,
    "WanVideoAddPusaNoise": WanVideoAddPusaNoise,
    "WanVideoAnimateEmbeds": WanVideoAnimateEmbeds,
    "WanVideoAddLucyEditLatents": WanVideoAddLucyEditLatents,
    "WanVideoAddBindweaveEmbeds": WanVideoAddBindweaveEmbeds,
    "TextImageEncodeQwenVL": TextImageEncodeQwenVL,
    "WanVideoUniLumosEmbeds": WanVideoUniLumosEmbeds,
    "WanVideoAddTTMLatents": WanVideoAddTTMLatents,
    "WanVideoAddStoryMemLatents": WanVideoAddStoryMemLatents,
    "WanVideoSVIProEmbeds": WanVideoSVIProEmbeds,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoDecode": "WanVideo Decode",
    "WanVideoTextEncode": "WanVideo TextEncode",
    "WanVideoTextEncodeSingle": "WanVideo TextEncodeSingle",
    "WanVideoTextImageEncode": "WanVideo TextImageEncode (IP2V)",
    "WanVideoClipVisionEncode": "WanVideo ClipVision Encode",
    "WanVideoImageToVideoEncode": "WanVideo ImageToVideo Encode",
    "WanVideoEncode": "WanVideo Encode",
    "WanVideoEncodeLatentBatch": "WanVideo Encode Latent Batch",
    "WanVideoEmptyEmbeds": "WanVideo Empty Embeds",
    "WanVideoEnhanceAVideo": "WanVideo Enhance-A-Video",
    "WanVideoContextOptions": "WanVideo Context Options",
    "WanVideoTextEmbedBridge": "WanVideo TextEmbed Bridge",
    "WanVideoControlEmbeds": "WanVideo Control Embeds",
    "WanVideoSLG": "WanVideo SLG",
    "WanVideoLoopArgs": "WanVideo Loop Args",
    "WanVideoSetBlockSwap": "WanVideo Set BlockSwap",
    "WanVideoExperimentalArgs": "WanVideo Experimental Args",
    "WanVideoVACEEncode": "WanVideo VACE Encode",
    "WanVideoPhantomEmbeds": "WanVideo Phantom Embeds",
    "WanVideoRealisDanceLatents": "WanVideo RealisDance Latents",
    "WanVideoApplyNAG": "WanVideo Apply NAG",
    "WanVideoMiniMaxRemoverEmbeds": "WanVideo MiniMax Remover Embeds",
    "WanVideoFreeInitArgs": "WanVideo Free Init Args",
    "WanVideoSetRadialAttention": "WanVideo Set Radial Attention",
    "WanVideoBlockList": "WanVideo Block List",
    "WanVideoTextEncodeCached": "WanVideo TextEncode Cached",
    "WanVideoAddExtraLatent": "WanVideo Add Extra Latent",
    "WanVideoAddStandInLatent": "WanVideo Add StandIn Latent",
    "WanVideoAddControlEmbeds": "WanVideo Add Control Embeds",
    "WanVideoAddMTVMotion": "WanVideo MTV Crafter Motion",
    "WanVideoRoPEFunction": "WanVideo RoPE Function",
    "WanVideoAddPusaNoise": "WanVideo Add Pusa Noise",
    "WanVideoAnimateEmbeds": "WanVideo Animate Embeds",
    "WanVideoAddLucyEditLatents": "WanVideo Add LucyEdit Latents",
    "WanVideoAddBindweaveEmbeds": "WanVideo Add Bindweave Embeds",
    "WanVideoUniLumosEmbeds": "WanVideo UniLumos Embeds",
    "WanVideoAddTTMLatents": "WanVideo Add TTMLatents",
    "WanVideoAddStoryMemLatents": "WanVideo Add StoryMem Latents",
    "WanVideoSVIProEmbeds": "WanVideo SVIPro Embeds",
}
