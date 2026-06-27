import os, gc, math, copy
import torch
import numpy as np
from tqdm import tqdm
import inspect
from PIL import Image
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from .wanvideo.schedulers.fm_solvers import get_sampling_sigmas, retrieve_timesteps
from .wanvideo.modules.model import rope_params
from .custom_linear import remove_lora_from_module, set_lora_params, _replace_linear
from .wanvideo.schedulers import get_scheduler, scheduler_list
from .gguf.gguf import set_lora_params_gguf
from .multitalk.multitalk import timestep_transform, add_noise
from .utils import(log, print_memory, apply_lora, fourier_filter, optimized_scale, setup_radial_attention,
                   compile_model, dict_to_device, tangential_projection, get_raag_guidance, temporal_score_rescaling)
from .cache_methods.cache_methods import cache_report
from .nodes_model_loading import load_weights
from .enhance_a_video.globals import set_enhance_weight, set_num_frames
from .WanMove.trajectory import replace_feature
from contextlib import nullcontext

from comfy import model_management as mm
from comfy.utils import ProgressBar, load_torch_file
from comfy.cli_args import args, LatentPreviewMethod

script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

rope_functions = ["default", "comfy", "comfy_chunked"]

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

try:
    from .gguf.gguf import GGUFParameter
except:
    pass

class MetaParameter(torch.nn.Parameter):
    def __new__(cls, dtype, quant_type=None):
        data = torch.empty(0, dtype=dtype)
        self = torch.nn.Parameter(data, requires_grad=False)
        self.quant_type = quant_type
        return self

def offload_transformer(transformer, remove_lora=True):
    transformer.teacache_state.clear_all()
    transformer.magcache_state.clear_all()
    transformer.easycache_state.clear_all()

    if transformer.patched_linear:
        for name, param in transformer.named_parameters():
            if "loras" in name or "controlnet" in name:
                continue
            module = transformer
            subnames = name.split('.')
            for subname in subnames[:-1]:
                module = getattr(module, subname)
            attr_name = subnames[-1]
            if param.data.is_floating_point():
                meta_param = torch.nn.Parameter(torch.empty_like(param.data, device='meta'), requires_grad=False)
                setattr(module, attr_name, meta_param)
            elif isinstance(param.data, GGUFParameter):
                quant_type = getattr(param, 'quant_type', None)
                setattr(module, attr_name, MetaParameter(param.data.dtype, quant_type))
            else:
                pass
        if remove_lora:
            remove_lora_from_module(transformer)
    else:
        transformer.to(offload_device)

    for block in transformer.blocks:
        block.kv_cache = None
        if transformer.audio_model is not None and hasattr(block, 'audio_block'):
            block.audio_block = None

    mm.soft_empty_cache()
    gc.collect()


def init_blockswap(transformer, block_swap_args, model):
    if not transformer.patched_linear:
        if block_swap_args is not None:
            for name, param in transformer.named_parameters():
                if "block" not in name or "control_adapter" in name or "face" in name:
                    param.data = param.data.to(device)
                elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                    param.data = param.data.to(offload_device)
                elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                    param.data = param.data.to(offload_device)

            transformer.block_swap(
                block_swap_args["blocks_to_swap"] - 1 ,
                block_swap_args["offload_txt_emb"],
                block_swap_args["offload_img_emb"],
                vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", None),
            )
        elif model["auto_cpu_offload"]:
            for module in transformer.modules():
                if hasattr(module, "offload"):
                    module.offload()
                if hasattr(module, "onload"):
                    module.onload()
            for block in transformer.blocks:
                block.modulation = torch.nn.Parameter(block.modulation.to(device))
            transformer.head.modulation = torch.nn.Parameter(transformer.head.modulation.to(device))
        else:
            transformer.to(device)

class WanVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (scheduler_list, {"default": "unipc",}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),
            },
            "optional": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feta_args": ("FETAARGS", ),
                "context_options": ("WANVIDCONTEXT", ),
                "cache_args": ("CACHEARGS", ),
                "flowedit_args": ("FLOWEDITARGS", ),
                "batched_cfg": ("BOOLEAN", {"default": False, "tooltip": "Batch cond and uncond for faster sampling, possibly faster on some hardware, uses more memory"}),
                "slg_args": ("SLGARGS", ),
                "rope_function": (rope_functions, {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile. Chunked version has reduced peak VRAM usage when not using torch.compile"}),
                "loop_args": ("LOOPARGS", ),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "sigmas": ("SIGMAS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS", ),
                "uni3c_embeds": ("UNI3C_EMBEDS", ),
                "multitalk_embeds": ("MULTITALK_EMBEDS", ),
                "freeinit_args": ("FREEINITARGS", ),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Start step for the sampling, 0 means full sampling, otherwise samples only from this step"}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1, "tooltip": "End step for the sampling, -1 means full sampling, otherwise samples only until this step"}),
                "add_noise_to_samples": ("BOOLEAN", {"default": False, "tooltip": "Add noise to the samples before sampling, needed for video2video sampling when starting from clean video"}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT",)
    RETURN_NAMES = ("samples", "denoised_samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, model, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index, text_embeds=None,
        force_offload=True, samples=None, feta_args=None, denoise_strength=1.0, context_options=None,
        cache_args=None, teacache_args=None, flowedit_args=None, batched_cfg=False, slg_args=None, rope_function="default", loop_args=None,
        experimental_args=None, sigmas=None, unianimate_poses=None, fantasytalking_embeds=None, uni3c_embeds=None, multitalk_embeds=None, freeinit_args=None, start_step=0, end_step=-1, add_noise_to_samples=False):

        patcher = model
        model = model.model
        transformer = model.diffusion_model

        dtype = model["base_dtype"]
        weight_dtype = model["weight_dtype"]
        fp8_matmul = model["fp8_matmul"]
        gguf_reader = model["gguf_reader"]
        control_lora = model["control_lora"]

        vae = image_embeds.get("vae", None)
        tiled_vae = image_embeds.get("tiled_vae", False)

        transformer_options = patcher.model_options.get("transformer_options", None)
        merge_loras = transformer_options["merge_loras"]

        block_swap_args = transformer_options.get("block_swap_args", None)
        if block_swap_args is not None:
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", False)
            transformer.blocks_to_swap = block_swap_args.get("blocks_to_swap", 0)
            transformer.vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", 0)
            transformer.prefetch_blocks = block_swap_args.get("prefetch_blocks", 0)
            transformer.block_swap_debug = block_swap_args.get("block_swap_debug", False)
            transformer.offload_img_emb = block_swap_args.get("offload_img_emb", False)
            transformer.offload_txt_emb = block_swap_args.get("offload_txt_emb", False)

        is_5b = transformer.out_dim == 48
        vae_upscale_factor = 16 if is_5b else 8

        # Load weights
        if transformer.audio_model is not None:
            for block in transformer.blocks:
                if hasattr(block, 'audio_block'):
                    block.audio_block = None

        if not transformer.patched_linear and patcher.model["sd"] is not None and len(patcher.patches) != 0 and gguf_reader is None:
            transformer = _replace_linear(transformer, dtype, patcher.model["sd"], compile_args=model["compile_args"])
            transformer.patched_linear = True
        if patcher.model["sd"] is not None and gguf_reader is None:
            load_weights(patcher.model.diffusion_model, patcher.model["sd"], weight_dtype, base_dtype=dtype, transformer_load_device=device,
                         block_swap_args=block_swap_args, compile_args=model["compile_args"])

        if gguf_reader is not None: #handle GGUF
            load_weights(transformer, patcher.model["sd"], base_dtype=dtype, transformer_load_device=device, patcher=patcher, gguf=True,
                         reader=gguf_reader, block_swap_args=block_swap_args, compile_args=model["compile_args"])
            set_lora_params_gguf(transformer, patcher.patches)
            transformer.patched_linear = True
        elif len(patcher.patches) != 0: #handle patched linear layers (unmerged loras, fp8 scaled)
            log.info(f"Using {len(patcher.patches)} LoRA weight patches for WanVideo model")
            if not merge_loras and fp8_matmul:
                raise NotImplementedError("FP8 matmul with unmerged LoRAs is not supported")
            set_lora_params(transformer, patcher.patches)
        else:
            remove_lora_from_module(transformer) #clear possible unmerged lora weights

        transformer.lora_scheduling_enabled = transformer_options.get("lora_scheduling_enabled", False)

        #torch.compile
        if model["auto_cpu_offload"] is False:
            transformer = compile_model(transformer, model["compile_args"])

        multitalk_sampling = image_embeds.get("multitalk_sampling", False)

        if multitalk_sampling and context_options is not None:
            raise Exception("context_options are not compatible or necessary with 'WanVideoImageToVideoMultiTalk' node, since it's already an alternative method that creates the video in a loop.")

        if not multitalk_sampling and scheduler == "multitalk":
            raise Exception("multitalk scheduler is only for multitalk sampling when using ImagetoVideoMultiTalk -node")

        if text_embeds == None:
            text_embeds = {
                "prompt_embeds": [],
                "negative_prompt_embeds": [],
            }
        else:
            text_embeds = dict_to_device(text_embeds, device)

        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)

        #region Scheduler
        if denoise_strength < 1.0:
            if start_step != 0:
                raise ValueError("start_step must be 0 when denoise_strength is used")
            start_step = steps - int(steps * denoise_strength) - 1
            add_noise_to_samples = True #for now to not break old workflows

        sample_scheduler = None
        if isinstance(scheduler, dict):
            sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
            timesteps = scheduler["timesteps"]
            start_step = scheduler.get("start_step", start_step)
        elif scheduler != "multitalk":
            sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas, log_timesteps=True)
            log.info(f"sigmas: {sample_scheduler.sigmas}")
        else:
            timesteps = torch.tensor([1000, 750, 500, 250], device=device)

        total_steps = steps
        steps = len(timesteps)

        is_pusa = "pusa" in sample_scheduler.__class__.__name__.lower()

        scheduler_step_args = {"generator": seed_g}
        step_sig = inspect.signature(sample_scheduler.step)
        for arg in list(scheduler_step_args.keys()):
            if arg not in step_sig.parameters:
                scheduler_step_args.pop(arg)

        # Ovi
        if transformer.audio_model is not None: # temporary workaround (...nothing more permanent)
            for i, block in enumerate(transformer.blocks):
                block.audio_block = transformer.audio_model.blocks[i]
            sample_scheduler_ovi = copy.deepcopy(sample_scheduler)
            rope_function = "default" # comfy rope not implemented for ovi model yet
        ovi_negative_text_embeds = text_embeds.get("ovi_negative_prompt_embeds", None)
        ovi_audio_cfg = text_embeds.get("ovi_audio_cfg", None)
        if ovi_audio_cfg is not None:
            if not isinstance(ovi_audio_cfg, list):
                ovi_audio_cfg = [ovi_audio_cfg] * (steps + 1)

        if isinstance(cfg, list):
            if steps < len(cfg):
                log.info(f"Received {len(cfg)} cfg values, but only {steps} steps. Slicing cfg list to match steps.")
                cfg = cfg[:steps]
            elif steps > len(cfg):
                log.info(f"Received only {len(cfg)} cfg values, but {steps} steps. Extending cfg list to match steps.")
                cfg.extend([cfg[-1]] * (steps - len(cfg)))
            log.info(f"Using per-step cfg list: {cfg}")
        else:
            cfg = [cfg] * (steps + 1)

        control_latents = control_camera_latents = clip_fea = clip_fea_neg = end_image = recammaster = camera_embed = unianim_data = mocha_embeds = image_cond_neg =None
        vace_data = vace_context = vace_scale = None
        fun_or_fl2v_model = drop_last = False
        phantom_latents = fun_ref_image = ATI_tracks = None
        add_cond = attn_cond = attn_cond_neg = noise_pred_flipped = None
        humo_audio = humo_audio_neg = None
        has_ref = image_embeds.get("has_ref", False)

        #I2V
        image_cond = image_embeds.get("image_embeds", None)
        if image_cond is not None:
            if transformer.in_dim == 16:
                raise ValueError("T2V (text to video) model detected, encoded images only work with I2V (Image to video) models")
            elif transformer.in_dim not in [48, 32]: # fun 2.1 models don't use the mask
                image_cond_mask = image_embeds.get("mask", None)
                if image_cond_mask is not None:
                    image_cond = torch.cat([image_cond_mask, image_cond])
            else:
                image_cond[:, 1:] = 0

            log.info(f"image_cond shape: {image_cond.shape}")

            #ATI tracks
            if transformer_options is not None:
                ATI_tracks = transformer_options.get("ati_tracks", None)
                if ATI_tracks is not None:
                    from .ATI.motion_patch import patch_motion
                    topk = transformer_options.get("ati_topk", 2)
                    temperature = transformer_options.get("ati_temperature", 220.0)
                    ati_start_percent = transformer_options.get("ati_start_percent", 0.0)
                    ati_end_percent = transformer_options.get("ati_end_percent", 1.0)
                    image_cond_ati = patch_motion(ATI_tracks.to(image_cond.device, image_cond.dtype), image_cond, topk=topk, temperature=temperature)
                    log.info(f"ATI tracks shape: {ATI_tracks.shape}")

            add_cond_latents = image_embeds.get("add_cond_latents", None)
            if add_cond_latents is not None:
                add_cond = add_cond_latents["pose_latent"]
                attn_cond = add_cond_latents["ref_latent"]
                attn_cond_neg = add_cond_latents["ref_latent_neg"]
                add_cond_start_percent = add_cond_latents["pose_cond_start_percent"]
                add_cond_end_percent = add_cond_latents["pose_cond_end_percent"]

            end_image = image_embeds.get("end_image", None)
            fun_or_fl2v_model = image_embeds.get("fun_or_fl2v_model", False)

            noise = torch.randn( #C, T, H, W
                48 if is_5b else 16,
                (image_embeds["num_frames"] - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1),
                image_embeds["lat_h"],
                image_embeds["lat_w"],
                dtype=torch.float32,
                generator=seed_g,
                device=torch.device("cpu"))

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                if transformer.in_dim not in [148, 52, 48, 36, 32]:
                    raise ValueError("Control signal only works with Fun-Control model")

                control_latents = control_embeds.get("control_images", None)
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
                control_camera_latents = control_embeds.get("control_camera_latents", None)
                if control_camera_latents is not None:
                    if transformer.control_adapter is None:
                        raise ValueError("Control camera latents are only supported with Fun-Control-Camera model")
                    control_camera_start_percent = control_embeds.get("control_camera_start_percent", 0.0)
                    control_camera_end_percent = control_embeds.get("control_camera_end_percent", 1.0)

            drop_last = image_embeds.get("drop_last", False)
        else: #t2v
            target_shape = image_embeds.get("target_shape", None)
            if target_shape is None:
                raise ValueError("Empty image embeds must be provided for T2V models")

            # VACE
            vace_context = image_embeds.get("vace_context", None)
            vace_scale = image_embeds.get("vace_scale", None)
            if not isinstance(vace_scale, list):
                vace_scale = [vace_scale] * (steps+1)
            vace_start_percent = image_embeds.get("vace_start_percent", 0.0)
            vace_end_percent = image_embeds.get("vace_end_percent", 1.0)
            vace_seqlen = image_embeds.get("vace_seq_len", None)

            vace_additional_embeds = image_embeds.get("additional_vace_inputs", [])
            if vace_context is not None:
                vace_data = [
                    {"context": vace_context,
                     "scale": vace_scale,
                     "start": vace_start_percent,
                     "end": vace_end_percent,
                     "seq_len": vace_seqlen
                     }
                ]
                if len(vace_additional_embeds) > 0:
                    for i in range(len(vace_additional_embeds)):
                        if vace_additional_embeds[i].get("has_ref", False):
                            has_ref = True
                        vace_scale = vace_additional_embeds[i]["vace_scale"]
                        if not isinstance(vace_scale, list):
                            vace_scale = [vace_scale] * (steps+1)
                        vace_data.append({
                            "context": vace_additional_embeds[i]["vace_context"],
                            "scale": vace_scale,
                            "start": vace_additional_embeds[i]["vace_start_percent"],
                            "end": vace_additional_embeds[i]["vace_end_percent"],
                            "seq_len": vace_additional_embeds[i]["vace_seq_len"]
                        })

            noise = torch.randn(
                    48 if is_5b else 16,
                    target_shape[1] + 1 if has_ref else target_shape[1],
                    target_shape[2] // 2 if is_5b else target_shape[2], #todo make this smarter
                    target_shape[3] // 2 if is_5b else target_shape[3], #todo make this smarter
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    generator=seed_g)

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

            recammaster = image_embeds.get("recammaster", None)
            if recammaster is not None:
                camera_embed = recammaster.get("camera_embed", None)
                recam_latents = recammaster.get("source_latents", None)
                orig_noise_len = noise.shape[1]
                log.info(f"RecamMaster camera embed shape: {camera_embed.shape}")
                log.info(f"RecamMaster source video shape: {recam_latents.shape}")
                seq_len *= 2

            if image_embeds.get("mocha_embeds", None) is not None:
                mocha_embeds = image_embeds.get("mocha_embeds", None)
                mocha_num_refs = image_embeds.get("mocha_num_refs", 0)
                orig_noise_len = noise.shape[1]
                seq_len = image_embeds.get("seq_len", seq_len)
                log.info(f"MoCha embeds shape: {mocha_embeds.shape}")

            # Fun control and control lora
            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                control_latents = control_embeds.get("control_images", None)
                if control_latents is not None:
                    control_latents = control_latents.to(device)

                control_camera_latents = control_embeds.get("control_camera_latents", None)
                if control_camera_latents is not None:
                    if transformer.control_adapter is None:
                        raise ValueError("Control camera latents are only supported with Fun-Control-Camera model")
                    control_camera_start_percent = control_embeds.get("control_camera_start_percent", 0.0)
                    control_camera_end_percent = control_embeds.get("control_camera_end_percent", 1.0)

                if control_lora:
                    image_cond = control_latents.to(device)
                    if not patcher.model.is_patched:
                        log.info("Re-loading control LoRA...")
                        patcher = apply_lora(patcher, device, device, low_mem_load=False, control_lora=True)
                        patcher.model.is_patched = True
                else:
                    if transformer.in_dim not in [148, 48, 36, 32, 52]:
                        raise ValueError("Control signal only works with Fun-Control model")
                    image_cond = torch.zeros_like(noise).to(device) #fun control
                    if transformer.in_dim in [148, 52] or transformer.control_adapter is not None: #fun 2.2 control
                        mask_latents = torch.tile(
                            torch.zeros_like(noise[:1]), [4, 1, 1, 1]
                        )
                        masked_video_latents_input = torch.zeros_like(noise)
                        image_cond = torch.cat([mask_latents, masked_video_latents_input], dim=0).to(device)
                    clip_fea = None
                    fun_ref_image = control_embeds.get("fun_ref_image", None)
                    if fun_ref_image is not None:
                        if transformer.ref_conv.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                            raise ValueError("Fun-Control reference image won't work with this specific fp8_scaled model, it's been fixed in latest version of the model")
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
            else:
                if transformer.in_dim in [148, 52]: #fun inp
                    mask_latents = torch.tile(
                        torch.zeros_like(noise[:1]), [4, 1, 1, 1]
                    )
                    masked_video_latents_input = torch.zeros_like(noise)
                    image_cond = torch.cat([mask_latents, masked_video_latents_input], dim=0).to(device)

            # Phantom inputs
            phantom_latents = image_embeds.get("phantom_latents", None)
            phantom_cfg_scale = image_embeds.get("phantom_cfg_scale", None)
            if not isinstance(phantom_cfg_scale, list):
                phantom_cfg_scale = [phantom_cfg_scale] * (steps +1)
            phantom_start_percent = image_embeds.get("phantom_start_percent", 0.0)
            phantom_end_percent = image_embeds.get("phantom_end_percent", 1.0)

        # CLIP image features
        clip_fea = image_embeds.get("clip_context", None)
        if clip_fea is not None:
            clip_fea = clip_fea.to(dtype)
        clip_fea_neg = image_embeds.get("negative_clip_context", None)
        if clip_fea_neg is not None:
            clip_fea_neg = clip_fea_neg.to(dtype)

        num_frames = image_embeds.get("num_frames", 0)

        #HuMo inputs
        humo_audio = image_embeds.get("humo_audio_emb", None)
        humo_audio_neg = image_embeds.get("humo_audio_emb_neg", None)
        humo_reference_count = image_embeds.get("humo_reference_count", 0)

        if humo_audio is not None:
            from .HuMo.nodes import get_audio_emb_window
            if not multitalk_sampling:
                humo_audio, _ = get_audio_emb_window(humo_audio, num_frames, frame0_idx=0)
                zero_audio_pad = torch.zeros(humo_reference_count, *humo_audio.shape[1:]).to(humo_audio.device)
                humo_audio = torch.cat([humo_audio, zero_audio_pad], dim=0)
                humo_audio_neg = torch.zeros_like(humo_audio, dtype=humo_audio.dtype, device=humo_audio.device)
            humo_audio = humo_audio.to(device, dtype)

        if humo_audio_neg is not None:
            humo_audio_neg = humo_audio_neg.to(device, dtype)
        humo_audio_scale = image_embeds.get("humo_audio_scale", 1.0)
        humo_image_cond = image_embeds.get("humo_image_cond", None)
        humo_image_cond_neg = image_embeds.get("humo_image_cond_neg", None)

        pos_latent = neg_latent = None

        # Ovi
        noise_audio = latent_ovi = seq_len_ovi = None
        if transformer.audio_model is not None:
            noise_audio = samples.get("latent_ovi_audio", None) if samples is not None else None
            if noise_audio is not None:
                if not torch.any(noise_audio):
                    noise_audio = torch.randn(noise_audio.shape, device=torch.device("cpu"), dtype=torch.float32, generator=seed_g)
                else:
                    noise_audio = noise_audio.squeeze().movedim(0, 1).to(device, dtype)
            else:
                noise_audio = torch.randn((157, 20), device=torch.device("cpu"), dtype=torch.float32, generator=seed_g)  # T C
            log.info(f"Ovi audio latent shape: {noise_audio.shape}")
            latent_ovi = noise_audio
            seq_len_ovi = noise_audio.shape[0]

        if transformer.dim == 1536 and humo_image_cond is not None: #small humo model
            #noise = torch.cat([noise[:, :-humo_reference_count], humo_image_cond[4:, -humo_reference_count:]], dim=1)
            pos_latent = humo_image_cond[4:, -humo_reference_count:].to(device, dtype)
            neg_latent = torch.zeros_like(pos_latent)
            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])
            humo_image_cond = humo_image_cond_neg = None

        humo_audio_cfg_scale = image_embeds.get("humo_audio_cfg_scale", 1.0)
        humo_start_percent = image_embeds.get("humo_start_percent", 0.0)
        humo_end_percent = image_embeds.get("humo_end_percent", 1.0)
        if not isinstance(humo_audio_cfg_scale, list):
            humo_audio_cfg_scale = [humo_audio_cfg_scale] * (steps + 1)

        # region WanAnim inputs
        frame_window_size = image_embeds.get("frame_window_size", 77)
        wananimate_loop = image_embeds.get("looping", False)
        if wananimate_loop and context_options is not None:
            raise Exception("context_options are not compatible or necessary with WanAnim looping, since it creates the video in a loop.")
        wananim_pose_latents = image_embeds.get("pose_latents", None)
        wananim_pose_strength = image_embeds.get("pose_strength", 1.0)
        wananim_face_strength = image_embeds.get("face_strength", 1.0)
        wananim_face_pixels = image_embeds.get("face_pixels", None)
        wananim_ref_masks = image_embeds.get("ref_masks", None)
        wananim_is_masked = image_embeds.get("is_masked", False)
        if not wananimate_loop: # create zero face pixels if mask is provided without face pixels, as masking seems to require face input to work properly
            if wananim_face_pixels is None and wananim_is_masked:
                if context_options is None:
                    wananim_face_pixels = torch.zeros(1, 3, num_frames-1, 512, 512, dtype=torch.float32, device=offload_device)
                else:
                    wananim_face_pixels = torch.zeros(1, 3, context_options["context_frames"]-1, 512, 512, dtype=torch.float32, device=device)

        if image_cond is None:
            image_cond = image_embeds.get("ref_latent", None)
            has_ref = image_cond is not None or has_ref

        latent_video_length = noise.shape[1]

        # Initialize FreeInit filter if enabled
        freq_filter = None
        if freeinit_args is not None:
            from .freeinit.freeinit_utils import get_freq_filter, freq_mix_3d
            filter_shape = list(noise.shape)  # [batch, C, T, H, W]
            freq_filter = get_freq_filter(
                filter_shape,
                device=device,
                filter_type=freeinit_args.get("freeinit_method", "butterworth"),
                n=freeinit_args.get("freeinit_n", 4) if freeinit_args.get("freeinit_method", "butterworth") == "butterworth" else None,
                d_s=freeinit_args.get("freeinit_s", 1.0),
                d_t=freeinit_args.get("freeinit_t", 1.0)
            )
            if samples is not None:
                saved_generator_state = samples.get("generator_state", None)
                if saved_generator_state is not None:
                    seed_g.set_state(saved_generator_state)

        # UniAnimate
        if unianimate_poses is not None:
            transformer.dwpose_embedding.to(device, dtype)
            dwpose_data = unianimate_poses["pose"].to(device, dtype)
            dwpose_data = torch.cat([dwpose_data[:,:,:1].repeat(1,1,3,1,1), dwpose_data], dim=2)
            dwpose_data = transformer.dwpose_embedding(dwpose_data)
            log.info(f"UniAnimate pose embed shape: {dwpose_data.shape}")
            if not multitalk_sampling:
                if dwpose_data.shape[2] > latent_video_length:
                    log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is longer than the video length {latent_video_length}, truncating")
                    dwpose_data = dwpose_data[:,:, :latent_video_length]
                elif dwpose_data.shape[2] < latent_video_length:
                    log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is shorter than the video length {latent_video_length}, padding with last pose")
                    pad_len = latent_video_length - dwpose_data.shape[2]
                    pad = dwpose_data[:,:,:1].repeat(1,1,pad_len,1,1)
                    dwpose_data = torch.cat([dwpose_data, pad], dim=2)

            random_ref_dwpose_data = None
            if image_cond is not None:
                transformer.randomref_embedding_pose.to(device, dtype)
                random_ref_dwpose = unianimate_poses.get("ref", None)
                if random_ref_dwpose is not None:
                    random_ref_dwpose_data = transformer.randomref_embedding_pose(
                        random_ref_dwpose.to(device, dtype)
                        ).unsqueeze(2).to(dtype) # [1, 20, 104, 60]
                del random_ref_dwpose

            unianim_data = {
                "dwpose": dwpose_data,
                "random_ref": random_ref_dwpose_data.squeeze(0) if random_ref_dwpose_data is not None else None,
                "strength": unianimate_poses["strength"],
                "start_percent": unianimate_poses["start_percent"],
                "end_percent": unianimate_poses["end_percent"]
            }

        # FantasyTalking
        audio_proj = multitalk_audio_embeds = None
        audio_scale = 1.0
        if fantasytalking_embeds is not None:
            audio_proj = fantasytalking_embeds["audio_proj"].to(device)
            audio_scale = fantasytalking_embeds["audio_scale"]
            audio_cfg_scale = fantasytalking_embeds["audio_cfg_scale"]
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps +1)
            log.info(f"Audio proj shape: {audio_proj.shape}")


        # MultiTalk
        multitalk_audio_embeds = audio_emb_slice = audio_features_in = None
        multitalk_embeds = image_embeds.get("multitalk_embeds", multitalk_embeds)

        if multitalk_embeds is not None:
            audio_emb_slice = multitalk_embeds.get("audio_emb_slice", None) # if already sliced
            # Handle single or multiple speaker embeddings
            if audio_emb_slice is None:
                audio_features_in = multitalk_embeds.get("audio_features", None)
            if audio_features_in is not None:
                if isinstance(audio_features_in, list):
                    multitalk_audio_embeds = [emb.to(device, dtype) for emb in audio_features_in]
                else:
                    # keep backward-compatibility with single tensor input
                    multitalk_audio_embeds = [audio_features_in.to(device, dtype)]

                shapes = [tuple(e.shape) for e in multitalk_audio_embeds]
                log.info(f"Multitalk audio features shapes (per speaker): {shapes}")

            audio_scale = multitalk_embeds.get("audio_scale", 1.0)
            audio_cfg_scale = multitalk_embeds.get("audio_cfg_scale", 1.0)
            ref_target_masks = multitalk_embeds.get("ref_target_masks", None)
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps + 1)

        # FantasyPortrait
        fantasy_portrait_input = None
        fantasy_portrait_embeds = image_embeds.get("portrait_embeds", None)
        if fantasy_portrait_embeds is not None:
            log.info("Using FantasyPortrait embeddings")
            fantasy_portrait_input = fantasy_portrait_embeds.copy()
            portrait_cfg = fantasy_portrait_input.get("cfg_scale", 1.0)
            if not isinstance(portrait_cfg, list):
                portrait_cfg = [portrait_cfg] * (steps + 1)

        # MiniMax Remover
        minimax_latents = minimax_mask_latents = None
        minimax_latents = image_embeds.get("minimax_latents", None)
        minimax_mask_latents = image_embeds.get("minimax_mask_latents", None)
        if minimax_latents is not None:
            log.info(f"minimax_latents: {minimax_latents.shape}")
            log.info(f"minimax_mask_latents: {minimax_mask_latents.shape}")
            minimax_latents = minimax_latents.to(device, dtype)
            minimax_mask_latents = minimax_mask_latents.to(device, dtype)

        # Context windows
        is_looped = False
        context_reference_latent = None
        if context_options is not None:
            if context_options["context_frames"] <= num_frames:
                context_schedule = context_options["context_schedule"]
                context_frames =  (context_options["context_frames"] - 1) // 4 + 1
                context_stride = context_options["context_stride"] // 4
                context_overlap = context_options["context_overlap"] // 4
                context_reference_latent = context_options.get("reference_latent", None)

                # Get total number of prompts
                num_prompts = len(text_embeds["prompt_embeds"])
                log.info(f"Number of prompts: {num_prompts}")
                # Calculate which section this context window belongs to
                section_size = (latent_video_length / num_prompts) if num_prompts != 0 else 1
                log.info(f"Section size: {section_size}")
                is_looped = context_schedule == "uniform_looped"

                if mocha_embeds is not None:
                    seq_len = (context_frames * 2 + 1 + mocha_num_refs) * (noise.shape[2] * noise.shape[3] // 4)
                else:
                    seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * context_frames)
                log.info(f"context window seq len: {seq_len}")

                if context_options["freenoise"]:
                    log.info("Applying FreeNoise")
                    # code from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
                    delta = context_frames - context_overlap
                    for start_idx in range(0, latent_video_length-context_frames, delta):
                        place_idx = start_idx + context_frames
                        if place_idx >= latent_video_length:
                            break
                        end_idx = place_idx - 1

                        if end_idx + delta >= latent_video_length:
                            final_delta = latent_video_length - place_idx
                            list_idx = torch.tensor(list(range(start_idx,start_idx+final_delta)), device=torch.device("cpu"), dtype=torch.long)
                            list_idx = list_idx[torch.randperm(final_delta, generator=seed_g)]
                            noise[:, place_idx:place_idx + final_delta, :, :] = noise[:, list_idx, :, :]
                            break
                        list_idx = torch.tensor(list(range(start_idx,start_idx+delta)), device=torch.device("cpu"), dtype=torch.long)
                        list_idx = list_idx[torch.randperm(delta, generator=seed_g)]
                        noise[:, place_idx:place_idx + delta, :, :] = noise[:, list_idx, :, :]

                log.info(f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap")
                from .context_windows.context import get_context_scheduler, create_window_mask, WindowTracker
                self.window_tracker = WindowTracker(verbose=context_options["verbose"])
                context = get_context_scheduler(context_schedule)
            else:
                log.info("Context frames is larger than total num_frames, disabling context windows")
                context_options = None

        #MTV Crafter
        mtv_input = image_embeds.get("mtv_crafter_motion", None)
        mtv_motion_tokens = None
        if mtv_input is not None:
            from .MTV.mtv import prepare_motion_embeddings
            log.info("Using MTV Crafter embeddings")
            mtv_start_percent = mtv_input.get("start_percent", 0.0)
            mtv_end_percent = mtv_input.get("end_percent", 1.0)
            mtv_strength = mtv_input.get("strength", 1.0)
            mtv_motion_tokens = mtv_input.get("mtv_motion_tokens", None)
            if not isinstance(mtv_strength, list):
                mtv_strength = [mtv_strength] * (steps + 1)
            d = transformer.dim // transformer.num_heads
            mtv_freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)
            motion_rotary_emb = prepare_motion_embeddings(
                latent_video_length if context_options is None else context_frames,
                24, mtv_input["global_mean"], [mtv_input["global_std"]], device=device)
            log.info(f"mtv_motion_rotary_emb: {motion_rotary_emb[0].shape}")
            mtv_freqs = mtv_freqs.to(device, dtype)

        #region S2V
        s2v_audio_input = s2v_ref_latent = s2v_pose = s2v_ref_motion = None
        framepack = False
        s2v_audio_embeds = image_embeds.get("audio_embeds", None)
        if s2v_audio_embeds is not None:
            log.info(f"Using S2V audio embeddings")
            framepack = s2v_audio_embeds.get("enable_framepack", False)
            if framepack and context_options is not None:
                raise ValueError("S2V framepack and context windows cannot be used at the same time")

            s2v_audio_input = s2v_audio_embeds.get("audio_embed_bucket", None)
            if s2v_audio_input is not None:
                #s2v_audio_input = s2v_audio_input[..., 0:image_embeds["num_frames"]]
                s2v_audio_input = s2v_audio_input.to(device, dtype)
            s2v_audio_scale = s2v_audio_embeds["audio_scale"]
            s2v_ref_latent = s2v_audio_embeds.get("ref_latent", None)
            if s2v_ref_latent is not None:
                s2v_ref_latent = s2v_ref_latent.to(device, dtype)
            s2v_ref_motion = s2v_audio_embeds.get("ref_motion", None)
            if s2v_ref_motion is not None:
                s2v_ref_motion = s2v_ref_motion.to(device, dtype)
            s2v_pose = s2v_audio_embeds.get("pose_latent", None)
            if s2v_pose is not None:
                s2v_pose = s2v_pose.to(device, dtype)
            s2v_pose_start_percent = s2v_audio_embeds.get("pose_start_percent", 0.0)
            s2v_pose_end_percent = s2v_audio_embeds.get("pose_end_percent", 1.0)
            s2v_num_repeat = s2v_audio_embeds.get("num_repeat", 1)
            vae = s2v_audio_embeds.get("vae", None)

        # vid2vid
        noise_mask=original_image=None
        if samples is not None and not multitalk_sampling and not wananimate_loop:
            saved_generator_state = samples.get("generator_state", None)
            if saved_generator_state is not None:
                seed_g.set_state(saved_generator_state)
            input_samples = samples.get("samples", None)
            if input_samples is not None:
                input_samples = input_samples.squeeze(0).to(noise)
                if input_samples.shape[1] != noise.shape[1]:
                    input_samples = torch.cat([input_samples[:, :1].repeat(1, noise.shape[1] - input_samples.shape[1], 1, 1), input_samples], dim=1)

                if add_noise_to_samples:
                    latent_timestep = timesteps[:1].to(noise)
                    noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples
                else:
                    noise = input_samples

                noise_mask = samples.get("noise_mask", None)
                if noise_mask is not None:
                    log.info(f"Latent noise_mask shape: {noise_mask.shape}")
                    original_image = samples.get("original_image", None)
                    if original_image is None:
                        original_image = input_samples
                    if len(noise_mask.shape) == 4:
                        noise_mask = noise_mask.squeeze(1)
                    if noise_mask.shape[0] < noise.shape[1]:
                        noise_mask = noise_mask.repeat(noise.shape[1] // noise_mask.shape[0], 1, 1)

                    noise_mask = torch.nn.functional.interpolate(
                        noise_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                        size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                        mode='trilinear',
                        align_corners=False
                    ).repeat(1, noise.shape[0], 1, 1, 1)

        # extra latents (Pusa) and 5b
        latents_to_insert = add_index = noise_multipliers = None
        extra_latents = image_embeds.get("extra_latents", None)
        clean_latent_indices = []
        noise_multiplier_list = image_embeds.get("pusa_noise_multipliers", None)
        if noise_multiplier_list is not None:
            if len(noise_multiplier_list) != latent_video_length:
                noise_multipliers = torch.zeros(latent_video_length)
            else:
                noise_multipliers = torch.tensor(noise_multiplier_list)
                log.info(f"Using Pusa noise multipliers: {noise_multipliers}")
        if extra_latents is not None and transformer.multitalk_model_type.lower() != "infinitetalk":
            if noise_multiplier_list is not None:
                noise_multiplier_list = list(noise_multiplier_list) + [1.0] * (len(clean_latent_indices) - len(noise_multiplier_list))
            for i, entry in enumerate(extra_latents):
                add_index = entry["index"]
                num_extra_frames = entry["samples"].shape[2]
                # Handle negative indices
                if add_index < 0:
                    add_index = noise.shape[1] + add_index
                add_index = max(0, min(add_index, noise.shape[1] - num_extra_frames))
                if start_step == 0:
                    noise[:, add_index:add_index+num_extra_frames] = entry["samples"].to(noise)
                    log.info(f"Adding extra samples to latent indices {add_index} to {add_index+num_extra_frames-1}")
                clean_latent_indices.extend(range(add_index, add_index+num_extra_frames))
            if noise_multipliers is not None and len(noise_multiplier_list) != latent_video_length:
                for i, idx in enumerate(clean_latent_indices):
                    noise_multipliers[idx] = noise_multiplier_list[i]
                log.info(f"Using Pusa noise multipliers: {noise_multipliers}")

        # lucy edit
        extra_channel_latents = image_embeds.get("extra_channel_latents", None)
        if extra_channel_latents is not None:
            extra_channel_latents = extra_channel_latents[0].to(noise)

        # FlashVSR
        flashvsr_LQ_latent = LQ_images = None
        flashvsr_LQ_images = image_embeds.get("flashvsr_LQ_images", None)
        flashvsr_strength = image_embeds.get("flashvsr_strength", 1.0)
        if flashvsr_LQ_images is not None:
            flashvsr_LQ_images = flashvsr_LQ_images[:num_frames]
            first_frame = flashvsr_LQ_images[:1]
            last_frame = flashvsr_LQ_images[-1:].repeat(3, 1, 1, 1)
            flashvsr_LQ_images = torch.cat([first_frame, flashvsr_LQ_images, last_frame], dim=0)
            LQ_images = flashvsr_LQ_images.unsqueeze(0).movedim(-1, 1).to(dtype) * 2 - 1
            if context_options is None:
                flashvsr_LQ_latent = transformer.LQ_proj_in(LQ_images.to(device))
                log.info(f"flashvsr_LQ_latent: {flashvsr_LQ_latent[0].shape}")
                seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

        latent = noise

        # LongCat-Avatar
        longcat_ref_latent = None
        longcat_num_ref_latents = longcat_num_cond_latents = 0
        longcat_avatar_options = image_embeds.get("longcat_avatar_options", None)

        if longcat_avatar_options is not None:
            longcat_ref_latent = longcat_avatar_options.get("longcat_ref_latent", None)
            if longcat_ref_latent is not None:
                log.info(f"LongCat-Avatar reference latent shape: {longcat_ref_latent.shape}")
                latent = torch.cat([longcat_ref_latent.to(latent), latent], dim=1)
                seq_len = math.ceil((latent.shape[2] * latent.shape[3]) / 4 * latent.shape[1])
                insert_len = longcat_ref_latent.shape[1]
                clean_latent_indices = list(range(0, insert_len)) + [i + insert_len for i in clean_latent_indices]
                longcat_num_ref_latents = longcat_ref_latent.shape[1]
                latent_video_length += insert_len
            longcat_num_cond_latents = len(clean_latent_indices)
            log.info(f"LongCat num_cond_latents: {longcat_num_cond_latents} num_ref_latents: {longcat_num_ref_latents}")
        audio_stride = 2 if transformer.is_longcat else 1

        #controlnet
        controlnet_latents = controlnet = None
        if transformer_options is not None:
            controlnet = transformer_options.get("controlnet", None)
            if controlnet is not None:
                self.controlnet = controlnet["controlnet"]
                controlnet_start = controlnet["controlnet_start"]
                controlnet_end = controlnet["controlnet_end"]
                controlnet_latents = controlnet["control_latents"]
                controlnet["controlnet_weight"] = controlnet["controlnet_strength"]
                controlnet["controlnet_stride"] = controlnet["control_stride"]

        #uni3c
        uni3c_data = uni3c_data_input = None
        if uni3c_embeds is not None:
            transformer.uni3c_controlnet = uni3c_embeds["controlnet"]
            render_latent = uni3c_embeds["render_latent"].to(device)
            uni3c_data = uni3c_embeds.copy()
            if render_latent.shape != noise.shape:
                render_latent = torch.nn.functional.interpolate(render_latent, size=(noise.shape[1], noise.shape[2], noise.shape[3]), mode='trilinear', align_corners=False)
            uni3c_data["render_latent"] = render_latent

        # Enhance-a-video (feta)
        if feta_args is not None and latent_video_length > 1:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            set_num_frames(latent_video_length) if context_options is None else set_num_frames(context_frames)
            enhance_enabled = True
        else:
            feta_args = None
            enhance_enabled = False

        # EchoShot https://github.com/D2I-ai/EchoShot
        echoshot = False
        shot_len = None
        if text_embeds is not None:
            echoshot = text_embeds.get("echoshot", False)
        if echoshot:
            shot_num = len(text_embeds["prompt_embeds"])
            shot_len = [latent_video_length//shot_num] * (shot_num-1)
            shot_len.append(latent_video_length-sum(shot_len))
            rope_function = "default" #echoshot does not support comfy rope function
            log.info(f"Number of shots in prompt: {shot_num}, Shot token lengths: {shot_len}")

        # Bindweave
        qwenvl_embeds_pos = image_embeds.get("qwenvl_embeds_pos", None)
        qwenvl_embeds_neg = image_embeds.get("qwenvl_embeds_neg", None)

        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()

        #blockswap init
        init_blockswap(transformer, block_swap_args, model)

        # Initialize Cache if enabled
        previous_cache_states = None
        transformer.enable_teacache = transformer.enable_magcache = transformer.enable_easycache = False
        cache_args = teacache_args if teacache_args is not None else cache_args #for backward compatibility on old workflows
        if cache_args is not None:
            from .cache_methods.cache_methods import set_transformer_cache_method
            transformer = set_transformer_cache_method(transformer, timesteps, cache_args)

            # Initialize cache state
            if samples is not None:
                previous_cache_states = samples.get("cache_states", None)
                if previous_cache_states is not None:
                    log.info("Using cache states from previous sampler")
                    self.cache_state = previous_cache_states["cache_state"]
                    transformer.easycache_state = previous_cache_states["easycache_state"]
                    transformer.magcache_state = previous_cache_states["magcache_state"]
                    transformer.teacache_state = previous_cache_states["teacache_state"]

        if previous_cache_states is None:
            self.cache_state = [None, None]
            if phantom_latents is not None:
                log.info(f"Phantom latents shape: {phantom_latents.shape}")
                self.cache_state = [None, None, None]
            self.cache_state_source = [None, None]
            self.cache_states_context = []

        # Skip layer guidance (SLG)
        if slg_args is not None:
            assert batched_cfg is not None, "Batched cfg is not supported with SLG"
            transformer.slg_blocks = slg_args["blocks"]
            transformer.slg_start_percent = slg_args["start_percent"]
            transformer.slg_end_percent = slg_args["end_percent"]
        else:
            transformer.slg_blocks = None

        # Setup radial attention
        if transformer.attention_mode == "radial_sage_attention":
            setup_radial_attention(transformer, transformer_options, latent, seq_len, latent_video_length, context_options=context_options)

        # FlowEdit setup
        if flowedit_args is not None:
            source_embeds = flowedit_args["source_embeds"]
            source_embeds = dict_to_device(source_embeds, device)
            source_image_embeds = flowedit_args.get("source_image_embeds", image_embeds)
            source_image_cond = source_image_embeds.get("image_embeds", None)
            source_clip_fea = source_image_embeds.get("clip_fea", clip_fea)
            if source_image_cond is not None:
                source_image_cond = source_image_cond.to(dtype)
            skip_steps = flowedit_args["skip_steps"]
            drift_steps = flowedit_args["drift_steps"]
            source_cfg = flowedit_args["source_cfg"]
            if not isinstance(source_cfg, list):
                source_cfg = [source_cfg] * (steps +1)
            drift_cfg = flowedit_args["drift_cfg"]
            if not isinstance(drift_cfg, list):
                drift_cfg = [drift_cfg] * (steps +1)

            x_init = samples["samples"].clone().squeeze(0).to(device)
            x_tgt = samples["samples"].squeeze(0).to(device)

            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=flowedit_args["drift_flow_shift"],
                use_dynamic_shifting=False)

            sampling_sigmas = get_sampling_sigmas(steps, flowedit_args["drift_flow_shift"])

            drift_timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)

            if drift_steps > 0:
                drift_timesteps = torch.cat([drift_timesteps, torch.tensor([0]).to(drift_timesteps.device)]).to(drift_timesteps.device)
                timesteps[-drift_steps:] = drift_timesteps[-drift_steps:]

        # Experimental args
        use_cfg_zero_star = use_tangential = use_fresca = bidirectional_sampling = use_tsr = False
        raag_alpha = 0.0
        transformer.video_attention_split_steps = []
        if experimental_args is not None:
            video_attention_split_steps = experimental_args.get("video_attention_split_steps", [])
            if video_attention_split_steps:
                transformer.video_attention_split_steps = [int(x.strip()) for x in video_attention_split_steps.split(",")]

            use_zero_init = experimental_args.get("use_zero_init", True)
            use_cfg_zero_star = experimental_args.get("cfg_zero_star", False)
            use_tangential = experimental_args.get("use_tcfg", False)
            zero_star_steps = experimental_args.get("zero_star_steps", 0)
            raag_alpha = experimental_args.get("raag_alpha", 0.0)

            use_fresca = experimental_args.get("use_fresca", False)
            if use_fresca:
                fresca_scale_low = experimental_args.get("fresca_scale_low", 1.0)
                fresca_scale_high = experimental_args.get("fresca_scale_high", 1.25)
                fresca_freq_cutoff = experimental_args.get("fresca_freq_cutoff", 20)

            bidirectional_sampling = experimental_args.get("bidirectional_sampling", False)
            if bidirectional_sampling:
                sample_scheduler_flipped = copy.deepcopy(sample_scheduler)
            use_tsr = experimental_args.get("temporal_score_rescaling", False)
            tsr_k = experimental_args.get("tsr_k", 1.0)
            tsr_sigma = experimental_args.get("tsr_sigma", 1.0)

        # Rotary positional embeddings (RoPE)

        # RoPE base freq scaling as used with CineScale
        ntk_alphas = [1.0, 1.0, 1.0]
        if isinstance(rope_function, dict):
            ntk_alphas = rope_function["ntk_scale_f"], rope_function["ntk_scale_h"], rope_function["ntk_scale_w"]
            rope_function = rope_function["rope_function"]

        # Stand-In
        standin_input = image_embeds.get("standin_input", None)
        if standin_input is not None:
            rope_function = "comfy" # only works with this currently

        freqs = None

        log.info(f"Rope function: {rope_function}")

        riflex_freq_index = 0 if riflex_freq_index is None else riflex_freq_index
        transformer.rope_embedder.k = None
        transformer.rope_embedder.num_frames = None
        d = transformer.dim // transformer.num_heads

        if mocha_embeds is not None:
            from .mocha.nodes import rope_params_mocha
            log.info("Using Mocha RoPE")
            rope_function = 'mocha'

            freqs = torch.cat([
                rope_params_mocha(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index, start=-1),
                rope_params_mocha(1024, 2 * (d // 6), start=-1),
                rope_params_mocha(1024, 2 * (d // 6), start=-1)
            ],
            dim=1)
        elif "default" in rope_function or bidirectional_sampling: # original RoPE
            freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)
        elif "comfy" in rope_function: # comfy's rope
            transformer.rope_embedder.k = riflex_freq_index
            transformer.rope_embedder.num_frames = latent_video_length

        transformer.rope_func = rope_function
        for block in transformer.blocks:
            block.rope_func = rope_function
        if transformer.vace_layers is not None:
            for block in transformer.vace_blocks:
                block.rope_func = rope_function

        # Lynx
        lynx_ref_buffer = None
        lynx_embeds = image_embeds.get("lynx_embeds", None)
        if lynx_embeds is not None:
            if lynx_embeds.get("ip_x", None) is not None:
                if transformer.blocks[0].cross_attn.ip_adapter is None:
                    raise ValueError("Lynx IP embeds provided, but the no lynx ip adapter layers found in the model.")
            lynx_embeds = lynx_embeds.copy()
            log.info("Using Lynx embeddings", lynx_embeds)
            lynx_ref_latent = lynx_embeds.get("ref_latent", None)
            lynx_ref_latent_uncond = lynx_embeds.get("ref_latent_uncond", None)
            lynx_ref_text_embed = lynx_embeds.get("ref_text_embed", None)
            lynx_ref_text_embed = dict_to_device(lynx_ref_text_embed, device)
            lynx_cfg_scale = lynx_embeds.get("cfg_scale", 1.0)
            if not isinstance(lynx_cfg_scale, list):
                lynx_cfg_scale = [lynx_cfg_scale] * (steps + 1)

            if lynx_ref_latent is not None:
                if transformer.blocks[0].self_attn.ref_adapter is None:
                    raise ValueError("Lynx reference provided, but the no lynx reference adapter layers found in the model.")
                lynx_ref_latent = lynx_ref_latent[0]
                lynx_ref_latent_uncond = lynx_ref_latent_uncond[0]
                lynx_embeds["ref_feature_extractor"] = True
                log.info(f"Lynx ref latent shape: {lynx_ref_latent.shape}")
                log.info("Extracting Lynx ref cond buffer...")
                if transformer.in_dim == 36:
                    mask_latents = torch.tile(torch.zeros_like(lynx_ref_latent[:1]), [4, 1, 1, 1])
                    empty_image_cond = torch.cat([mask_latents, torch.zeros_like(lynx_ref_latent)], dim=0).to(device)
                    lynx_ref_input = torch.cat([lynx_ref_latent, empty_image_cond], dim=0)
                else:
                    lynx_ref_input = lynx_ref_latent
                lynx_ref_buffer = transformer(
                    [lynx_ref_input.to(device, dtype)],
                    torch.tensor([0], device=device),
                    lynx_ref_text_embed["prompt_embeds"],
                    seq_len=math.ceil((lynx_ref_latent.shape[2] * lynx_ref_latent.shape[3]) / 4 * lynx_ref_latent.shape[1]),
                    lynx_embeds=lynx_embeds
                )
                log.info(f"Extracted {len(lynx_ref_buffer)} cond ref buffers")
                if any(not math.isclose(c, 1.0) for c in cfg):
                    log.info("Extracting Lynx ref uncond buffer...")
                    if transformer.in_dim == 36:
                        lynx_ref_input_uncond = torch.cat([lynx_ref_latent_uncond, empty_image_cond], dim=0)
                    else:
                        lynx_ref_input_uncond = lynx_ref_latent_uncond
                    lynx_ref_buffer_uncond = transformer(
                        [lynx_ref_input_uncond.to(device, dtype)],
                        torch.tensor([0], device=device),
                        lynx_ref_text_embed["prompt_embeds"],
                        seq_len=math.ceil((lynx_ref_latent.shape[2] * lynx_ref_latent.shape[3]) / 4 * lynx_ref_latent.shape[1]),
                        lynx_embeds=lynx_embeds,
                        is_uncond=True
                    )
                    log.info(f"Extracted {len(lynx_ref_buffer_uncond)} uncond ref buffers")

            if lynx_embeds.get("ip_x", None) is not None:
                lynx_embeds["ip_x"] = lynx_embeds["ip_x"].to(device, dtype)
                lynx_embeds["ip_x_uncond"] = lynx_embeds["ip_x_uncond"].to(device, dtype)
            lynx_embeds["ref_feature_extractor"] = False
            lynx_embeds["ref_latent"] = lynx_embeds["ref_text_embed"] = None
            lynx_embeds["ref_buffer"] = lynx_ref_buffer
            lynx_embeds["ref_buffer_uncond"] = lynx_ref_buffer_uncond if not math.isclose(cfg[0], 1.0) else None
            mm.soft_empty_cache()

        # UniLumos
        foreground_latents = image_embeds.get("foreground_latents", None)
        if foreground_latents is not None:
            log.info(f"UniLumos foreground latent input shape: {foreground_latents.shape}")
            foreground_latents = foreground_latents.to(device, dtype)
        background_latents = image_embeds.get("background_latents", None)
        if background_latents is not None:
            log.info(f"UniLumos background latent input shape: {background_latents.shape}")
            background_latents = background_latents.to(device, dtype)

        #Time-to-move (TTM)
        ttm_start_step = 0
        ttm_reference_latents = image_embeds.get("ttm_reference_latents", None)
        if ttm_reference_latents is not None:
            motion_mask = image_embeds["ttm_mask"].to(device, dtype)
            ttm_start_step = max(image_embeds["ttm_start_step"] - start_step, 0)
            ttm_end_step = image_embeds["ttm_end_step"] - start_step

            if ttm_start_step > steps:
                raise ValueError("TTM start step is beyond the total number of steps")

            if ttm_end_step > ttm_start_step:
                log.info("Using Time-to-move (TTM)")
                log.info(f"TTM reference latents shape: {ttm_reference_latents.shape}")
                log.info(f"TTM motion mask shape: {motion_mask.shape}")
                log.info(f"Applying TTM from step {ttm_start_step} to {ttm_end_step}")

                latent = add_noise(ttm_reference_latents, noise, timesteps[ttm_start_step].to(noise.device)).to(latent)

        # SteadyDancer
        sdancer_embeds = image_embeds.get("sdancer_embeds", None)
        sdancer_data = sdancer_input = None
        if sdancer_embeds is not None:
            log.info("Using SteadyDancer embeddings:")
            for k, v in sdancer_embeds.items():
                log.info(f"  {k}: {v.shape if isinstance(v, torch.Tensor) else v}")
            sdancer_data = sdancer_embeds.copy()
            sdancer_data = dict_to_device(sdancer_data, device, dtype)

        # One-to-all-Animation
        one_to_all_embeds = image_embeds.get("one_to_all_embeds", None)
        one_to_all_data = prev_latents = None
        latents_to_not_step = 0
        if one_to_all_embeds is not None:
            log.info("Using One-to-All embeddings:")
            for k, v in one_to_all_embeds.items():
                log.info(f"  {k}: {v.shape if isinstance(v, torch.Tensor) else v}")
            one_to_all_data = one_to_all_embeds.copy()
            one_to_all_data = dict_to_device(one_to_all_data, device, dtype)
            if one_to_all_embeds.get("pose_images") is not None:
                transformer.input_hint_block.to(device)
                pose_images_in = one_to_all_data.pop("pose_images")
                pose_images = transformer.input_hint_block(pose_images_in)
                if one_to_all_embeds.get("ref_latent_pos") is not None:
                    pose_prefix_image = transformer.input_hint_block(one_to_all_data.pop("pose_prefix_image"))
                    pose_images = torch.cat([pose_prefix_image, pose_images],dim=2)
                one_to_all_data["controlnet_tokens"] = pose_images.flatten(2).transpose(1, 2)
                transformer.input_hint_block.to(offload_device)

                one_to_all_pose_cfg_scale = one_to_all_embeds.get("pose_cfg_scale", 1.0)
                if not isinstance(one_to_all_pose_cfg_scale, list):
                    one_to_all_pose_cfg_scale = [one_to_all_pose_cfg_scale] * (steps + 1)

            prev_latents = one_to_all_data.get("prev_latents", None)
            if prev_latents is not None:
                log.info(f"Using previous latents for One-to-All Animation with shape: {prev_latents.shape}")
                latent[:, :prev_latents.shape[1]] = prev_latents.to(latent)
                one_to_all_data["token_replace"] = True
                latents_to_not_step = prev_latents.shape[1]
                one_to_all_data["num_latent_frames_to_replace"] = latents_to_not_step

        # SCAIL
        scail_embeds = image_embeds.get("scail_embeds", None)
        scail_data = None
        if scail_embeds is not None:
            log.info("Using SCAIL embeddings:")
            for k, v in scail_embeds.items():
                log.info(f"  {k}: {v.shape if isinstance(v, torch.Tensor) else v}")
            scail_data = scail_embeds.copy()
            scail_data = dict_to_device(scail_data, device, dtype)


        # WanMove
        wanmove_embeds = None
        if image_cond is not None:
            wanmove_embeds = image_embeds.get("wanmove_embeds", None)
            if wanmove_embeds is not None:
                track_pos = wanmove_embeds["track_pos"]
                if any(not math.isclose(c, 1.0) for c in cfg):
                    image_cond_neg = torch.cat([image_embeds["mask"], image_cond])
                if context_options is None:
                    image_cond = replace_feature(image_cond.unsqueeze(0).clone(), track_pos.unsqueeze(0), wanmove_embeds.get("strength", 1.0))[0]

        #region model pred
        def predict_with_cfg(z, cfg_scale, positive_embeds, negative_embeds, timestep, idx, image_cond=None, clip_fea=None,
                             control_latents=None, vace_data=None, unianim_data=None, audio_proj=None, control_camera_latents=None,
                             add_cond=None, cache_state=None, context_window=None, multitalk_audio_embeds=None, fantasy_portrait_input=None, reverse_time=False,
                             mtv_motion_tokens=None, s2v_audio_input=None, s2v_ref_motion=None, s2v_motion_frames=[1, 0], s2v_pose=None,
                             humo_image_cond=None, humo_image_cond_neg=None, humo_audio=None, humo_audio_neg=None, wananim_pose_latents=None,
                             wananim_face_pixels=None, uni3c_data=None, latent_model_input_ovi=None, flashvsr_LQ_latent=None,):
            nonlocal transformer
            nonlocal audio_cfg_scale

            autocast_enabled = ("fp8" in model["quantization"] and not transformer.patched_linear)
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype) if autocast_enabled else nullcontext():

                if use_cfg_zero_star and (idx <= zero_star_steps) and use_zero_init:
                    return z*0, None

                nonlocal patcher
                current_step_percentage = idx / len(timesteps)
                control_lora_enabled = False
                image_cond_input = None
                if control_embeds is not None and control_camera_latents is None:
                    if control_lora:
                        control_lora_enabled = True
                    else:
                        if ((control_start_percent <= current_step_percentage <= control_end_percent) or \
                            (control_end_percent > 0 and idx == 0 and current_step_percentage >= control_start_percent)) and \
                            (control_latents is not None):
                            image_cond_input = torch.cat([control_latents.to(z), image_cond.to(z)])
                        else:
                            image_cond_input = torch.cat([torch.zeros_like(noise, device=device, dtype=dtype), image_cond.to(z)])
                        if fun_ref_image is not None:
                            fun_ref_input = fun_ref_image.to(z)
                        else:
                            fun_ref_input = torch.zeros_like(z, dtype=z.dtype)[:, 0].unsqueeze(1)

                    if control_lora:
                        if not control_start_percent <= current_step_percentage <= control_end_percent:
                            control_lora_enabled = False
                            if patcher.model.is_patched:
                                log.info("Unloading LoRA...")
                                patcher.unpatch_model(device)
                                patcher.model.is_patched = False
                        else:
                            image_cond_input = control_latents.to(z)
                            if not patcher.model.is_patched:
                                log.info("Loading LoRA...")
                                patcher = apply_lora(patcher, device, device, low_mem_load=False, control_lora=True)
                                patcher.model.is_patched = True

                elif ATI_tracks is not None and ((ati_start_percent <= current_step_percentage <= ati_end_percent) or
                              (ati_end_percent > 0 and idx == 0 and current_step_percentage >= ati_start_percent)):
                    image_cond_input = image_cond_ati.to(z)
                elif humo_image_cond is not None:
                    humo_image_cond_neg_input = None
                    if context_window is not None:
                        image_cond_input = humo_image_cond[:, context_window].to(z)
                        humo_image_cond_neg_input = humo_image_cond_neg[:, context_window].to(z)
                        if humo_reference_count > 0:
                            image_cond_input[:, -humo_reference_count:] = humo_image_cond[:, -humo_reference_count:]
                            humo_image_cond_neg_input[:, -humo_reference_count:] = humo_image_cond_neg[:, -humo_reference_count:]
                    else:
                        if image_cond is not None:
                            image_cond_input = image_cond.to(z)
                            if humo_reference_count > 0:
                                image_cond_input = torch.cat([image_cond_input, humo_image_cond[:, -humo_reference_count:].to(z)], dim=1)
                                humo_image_cond_neg_input = torch.cat([image_cond_input, humo_image_cond_neg[:, -humo_reference_count:].to(z)], dim=1)
                        else:
                            image_cond_input = humo_image_cond.to(z)
                            humo_image_cond_neg_input = humo_image_cond_neg.to(z)

                elif image_cond is not None:
                    if reverse_time: # Flip the image condition
                        image_cond_input = torch.cat([
                            torch.flip(image_cond[:4], dims=[1]),
                            torch.flip(image_cond[4:], dims=[1])
                        ]).to(z)
                    else:
                        image_cond_input = image_cond.to(z)

                if control_camera_latents is not None:
                    if (control_camera_start_percent <= current_step_percentage <= control_camera_end_percent) or \
                            (control_end_percent > 0 and idx == 0 and current_step_percentage >= control_camera_start_percent):
                        control_camera_input = control_camera_latents.to(device, dtype)
                    else:
                        control_camera_input = None

                if recammaster is not None:
                    z = torch.cat([z, recam_latents.to(z)], dim=1)

                if mocha_embeds is not None:
                    if context_window is not None and mocha_embeds.shape[2] != context_frames:
                        latent_frames = len(context_window)
                        # [latent_frames, 1 mask frame, mocha_num_refs]
                        latent_end = latent_frames
                        mask_end = latent_end + 1
                        partial_latents = mocha_embeds[:, context_window]  # windowed latents
                        mask_frame = mocha_embeds[:, latent_end:mask_end]  # single mask frame
                        ref_frames = mocha_embeds[:, -mocha_num_refs:]     # reference frames

                        partial_mocha_embeds = torch.cat([partial_latents, mask_frame, ref_frames], dim=1)
                        z = torch.cat([z, partial_mocha_embeds.to(z)], dim=1)
                    else:
                        z = torch.cat([z, mocha_embeds.to(z)], dim=1)

                if mtv_input is not None:
                    if ((mtv_start_percent <= current_step_percentage <= mtv_end_percent) or \
                            (mtv_end_percent > 0 and idx == 0 and current_step_percentage >= mtv_start_percent)):
                        mtv_motion_tokens = mtv_motion_tokens.to(z)
                        mtv_motion_rotary_emb = motion_rotary_emb

                use_phantom = False
                phantom_ref = None
                if phantom_latents is not None:
                    if (phantom_start_percent <= current_step_percentage <= phantom_end_percent) or \
                        (phantom_end_percent > 0 and idx == 0 and current_step_percentage >= phantom_start_percent):
                        phantom_ref = phantom_latents.to(z)
                        use_phantom = True
                        if cache_state is not None and len(cache_state) != 3:
                            cache_state.append(None)

                if controlnet_latents is not None:
                    if (controlnet_start <= current_step_percentage < controlnet_end):
                        self.controlnet.to(device)
                        controlnet_states = self.controlnet(
                            hidden_states=z.unsqueeze(0).to(device, self.controlnet.dtype),
                            timestep=timestep,
                            encoder_hidden_states=positive_embeds[0].unsqueeze(0).to(device, self.controlnet.dtype),
                            attention_kwargs=None,
                            controlnet_states=controlnet_latents.to(device, self.controlnet.dtype),
                            return_dict=False,
                        )[0]
                        if isinstance(controlnet_states, (tuple, list)):
                            controlnet["controlnet_states"] = [x.to(z) for x in controlnet_states]
                        else:
                            controlnet["controlnet_states"] = controlnet_states.to(z)

                add_cond_input = None
                if add_cond is not None:
                    if (add_cond_start_percent <= current_step_percentage <= add_cond_end_percent) or \
                        (add_cond_end_percent > 0 and idx == 0 and current_step_percentage >= add_cond_start_percent):
                        add_cond_input = add_cond

                if minimax_latents is not None:
                    if context_window is not None:
                        z = torch.cat([z, minimax_latents[:, context_window], minimax_mask_latents[:, context_window]], dim=0)
                    else:
                        z = torch.cat([z, minimax_latents, minimax_mask_latents], dim=0)

                multitalk_audio_input = None
                if audio_emb_slice is not None:
                    multitalk_audio_input = audio_emb_slice.to(z)
                elif not multitalk_sampling and multitalk_audio_embeds is not None:
                    audio_embedding = multitalk_audio_embeds
                    audio_embs = []
                    indices = (torch.arange(4 + 1) - 2) * 1
                    human_num = len(audio_embedding)
                    # split audio with window size
                    audio_end_idx = latent_video_length * 4 + 1 if add_cond is not None else (latent_video_length-1) * 4 + 1
                    audio_end_idx = audio_end_idx * audio_stride
                    if context_window is None:
                        for human_idx in range(human_num):
                            center_indices = torch.arange(0, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
                            center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0] - 1)

                            audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                            audio_embs.append(audio_emb)
                    else:
                        for human_idx in range(human_num):
                            audio_start = (context_window[0] * 4) * audio_stride
                            audio_end = (context_window[-1] * 4 + 1) * audio_stride
                            #print("audio_start: ", audio_start, "audio_end: ", audio_end)
                            center_indices = torch.arange(audio_start, audio_end, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
                            center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0] - 1)
                            audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                            audio_embs.append(audio_emb)
                    multitalk_audio_input = torch.concat(audio_embs, dim=0).to(dtype)

                elif multitalk_sampling and multitalk_audio_embeds is not None:
                    multitalk_audio_input = multitalk_audio_embeds

                if context_window is not None and uni3c_data is not None and uni3c_data["render_latent"].shape[2] != context_frames:
                    uni3c_data_input = {"render_latent": uni3c_data["render_latent"][:, :, context_window]}
                    for k in uni3c_data:
                        if k != "render_latent":
                            uni3c_data_input[k] = uni3c_data[k]
                else:
                    uni3c_data_input = uni3c_data

                if context_window is not None and sdancer_data is not None and sdancer_data["cond_pos"].shape[1] != context_frames:
                    sdancer_input = sdancer_data.copy()
                    sdancer_input["cond_pos"] = sdancer_data["cond_pos"][:, context_window]
                    sdancer_input["cond_neg"] = sdancer_data["cond_neg"][:, context_window] if sdancer_data.get("cond_neg", None) is not None else None
                else:
                    sdancer_input = sdancer_data

                if s2v_pose is not None:
                    if not ((s2v_pose_start_percent <= current_step_percentage <= s2v_pose_end_percent) or \
                            (s2v_pose_end_percent > 0 and idx == 0 and current_step_percentage >= s2v_pose_start_percent)):
                        s2v_pose = None


                if humo_audio is not None and ((humo_start_percent <= current_step_percentage <= humo_end_percent) or \
                            (humo_end_percent > 0 and idx == 0 and current_step_percentage >= humo_start_percent)):
                    if context_window is None:
                        humo_audio_input = humo_audio
                        humo_audio_input_neg = humo_audio_neg if humo_audio_neg is not None else None
                    else:
                        humo_audio_input = humo_audio[context_window].to(z)
                        if humo_audio_neg is not None:
                            humo_audio_input_neg = humo_audio_neg[context_window].to(z)
                        else:
                            humo_audio_input_neg = None
                else:
                    humo_audio_input = humo_audio_input_neg = None

                if extra_channel_latents is not None:
                    if context_window is not None:
                        extra_channel_latents_input = extra_channel_latents[:, context_window].to(z)
                    else:
                        extra_channel_latents_input = extra_channel_latents.to(z)
                    z = torch.cat([z, extra_channel_latents_input])

                if "rcm" in sample_scheduler.__class__.__name__.lower():
                    c_in = 1 / (torch.cos(timestep) + torch.sin(timestep))
                    c_noise = (torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))) * 1000
                    z = z * c_in
                    timestep = c_noise

                if image_cond is not None:
                    self.noise_front_pad_num = image_cond_input.shape[1] - z.shape[1]
                    if self.noise_front_pad_num > 0:
                        pad = torch.zeros((z.shape[0], self.noise_front_pad_num, z.shape[2], z.shape[3]), dtype=z.dtype, device=z.device)
                        z = torch.concat([pad, z], dim=1)
                        nonlocal seq_len
                        seq_len = math.ceil((z.shape[2] * z.shape[3]) / 4 * z.shape[1])
                else:
                    self.noise_front_pad_num = 0

                if background_latents is not None or foreground_latents is not None:
                    z = torch.cat([z, foreground_latents.to(z), background_latents.to(z)], dim=0)

                scail_data_in = None
                if scail_data is not None:
                    ref_concat_mask = torch.zeros_like(z[:4])
                    z = torch.cat([z, ref_concat_mask])
                    if context_window is not None:
                        scail_data_in = scail_data.copy()
                        scail_data_in["pose_latent"] = scail_data["pose_latent"][:, context_window]
                    else:
                        scail_data_in = scail_data

                if wanmove_embeds is not None and context_window is not None:
                    image_cond_input = replace_feature(image_cond_input.unsqueeze(0), track_pos[:, context_window].unsqueeze(0), wanmove_embeds.get("strength", 1.0))[0]

                base_params = {
                    'x': [z], # latent
                    'y': [image_cond_input] if image_cond_input is not None else None, # image cond
                    'clip_fea': clip_fea, # clip features
                    'seq_len': seq_len, # sequence length
                    'device': device, # main device
                    'freqs': freqs, # rope freqs
                    't': timestep, # current timestep
                    'is_uncond': False, # is unconditional
                    'current_step': idx, # current step
                    'current_step_percentage': current_step_percentage, # current step percentage
                    'last_step': len(timesteps) - 1 == idx, # is last step
                    'control_lora_enabled': control_lora_enabled, # control lora toggle for patch embed selection
                    'enhance_enabled': enhance_enabled, # enhance-a-video toggle
                    'camera_embed': camera_embed, # recammaster embedding
                    'unianim_data': unianim_data, # unianimate input
                    'fun_ref': fun_ref_input if fun_ref_image is not None else None, # Fun model reference latent
                    'fun_camera': control_camera_input if control_camera_latents is not None else None, # Fun model camera embed
                    'audio_proj': audio_proj if fantasytalking_embeds is not None else None, # FantasyTalking audio projection
                    'audio_scale': audio_scale, # FantasyTalking audio scale
                    "uni3c_data": uni3c_data_input, # Uni3C input
                    "controlnet": controlnet, # TheDenk's controlnet input
                    "add_cond": add_cond_input, # additional conditioning input
                    "nag_params": text_embeds.get("nag_params", {}), # normalized attention guidance
                    "nag_context": text_embeds.get("nag_prompt_embeds", None), # normalized attention guidance context
                    "multitalk_audio": multitalk_audio_input, # Multi/InfiniteTalk audio input
                    "ref_target_masks": ref_target_masks if multitalk_audio_embeds is not None else None, # Multi/InfiniteTalk reference target masks
                    "inner_t": [shot_len] if shot_len else None, # inner timestep for EchoShot
                    "standin_input": standin_input, # Stand-in reference input
                    "fantasy_portrait_input": fantasy_portrait_input, # Fantasy portrait input
                    "phantom_ref": phantom_ref, # Phantom reference input
                    "reverse_time": reverse_time, # Reverse RoPE toggle
                    "ntk_alphas": ntk_alphas, # RoPE freq scaling values
                    "mtv_motion_tokens": mtv_motion_tokens if mtv_input is not None else None, # MTV-Crafter motion tokens
                    "mtv_motion_rotary_emb": mtv_motion_rotary_emb if mtv_input is not None else None, # MTV-Crafter RoPE
                    "mtv_strength": mtv_strength[idx] if mtv_input is not None else 1.0, # MTV-Crafter scaling
                    "mtv_freqs": mtv_freqs if mtv_input is not None else None, # MTV-Crafter extra RoPE freqs
                    "s2v_audio_input": s2v_audio_input, # official speech-to-video audio input
                    "s2v_ref_latent": s2v_ref_latent, # speech-to-video reference latent
                    "s2v_ref_motion": s2v_ref_motion, # speech-to-video reference motion latent
                    "s2v_audio_scale": s2v_audio_scale if s2v_audio_input is not None else 1.0, # speech-to-video audio scale
                    "s2v_pose": s2v_pose if s2v_pose is not None else None, # speech-to-video pose control
                    "s2v_motion_frames": s2v_motion_frames, # speech-to-video motion frames,
                    "humo_audio": humo_audio, # humo audio input
                    "humo_audio_scale": humo_audio_scale if humo_audio is not None else 1,
                    "wananim_pose_latents": wananim_pose_latents.to(device) if wananim_pose_latents is not None else None, # WanAnimate pose latents
                    "wananim_face_pixel_values": wananim_face_pixels.to(device, torch.float32) if wananim_face_pixels is not None else None, # WanAnimate face images
                    "wananim_pose_strength": wananim_pose_strength,
                    "wananim_face_strength": wananim_face_strength,
                    "lynx_embeds": lynx_embeds, # Lynx face and reference embeddings
                    "x_ovi": [latent_model_input_ovi.to(z)] if latent_model_input_ovi is not None else None, # Audio latent model input for Ovi
                    "seq_len_ovi": seq_len_ovi, # Audio latent model sequence length for Ovi
                    "ovi_negative_text_embeds": ovi_negative_text_embeds, # Audio latent model negative text embeds for Ovi
                    "flashvsr_LQ_latent": flashvsr_LQ_latent, # FlashVSR LQ latent for upsampling
                    "flashvsr_strength": flashvsr_strength, # FlashVSR strength
                    "longcat_num_cond_latents": longcat_num_cond_latents,
                    "longcat_num_ref_latents": longcat_num_ref_latents,
                    "longcat_avatar_options": longcat_avatar_options, # LongCat avatar attention options
                    "sdancer_input": sdancer_input, # SteadyDancer input
                    "one_to_all_input": one_to_all_data, # One-to-All input
                    "one_to_all_controlnet_strength": one_to_all_data["controlnet_strength"] if one_to_all_data is not None else 0.0,
                    "scail_input": scail_data_in, # SCAIL input
                }

                batch_size = 1

                if not math.isclose(cfg_scale, 1.0):
                    if negative_embeds is None:
                        raise ValueError("Negative embeddings must be provided for CFG scale > 1.0")
                    if len(positive_embeds) > 1:
                        negative_embeds = negative_embeds * len(positive_embeds)

                try:
                    if not batched_cfg:
                        #conditional (positive) pass
                        if pos_latent is not None: # for humo
                            base_params['x'] = [torch.cat([z[:, :-humo_reference_count], pos_latent], dim=1)]
                        base_params["add_text_emb"] = qwenvl_embeds_pos.to(device) if qwenvl_embeds_pos is not None else None # QwenVL embeddings for Bindweave
                        noise_pred_cond, noise_pred_ovi, cache_state_cond = transformer(
                            context=positive_embeds,
                            pred_id=cache_state[0] if cache_state else None,
                            vace_data=vace_data, attn_cond=attn_cond,
                            **base_params
                        )
                        noise_pred_cond = noise_pred_cond[0]
                        noise_pred_ovi = noise_pred_ovi[0] if noise_pred_ovi is not None else None
                        if math.isclose(cfg_scale, 1.0):
                            if use_fresca:
                                noise_pred_cond = fourier_filter(noise_pred_cond, fresca_scale_low, fresca_scale_high, fresca_freq_cutoff)
                            if fantasy_portrait_input is not None and not math.isclose(portrait_cfg[idx], 1.0):
                                base_params["fantasy_portrait_input"] = None
                                noise_pred_no_portrait, noise_pred_ovi, cache_state_uncond = transformer(context=positive_embeds, pred_id=cache_state[0] if cache_state else None,
                                vace_data=vace_data, attn_cond=attn_cond, **base_params)
                                return noise_pred_no_portrait[0] + portrait_cfg[idx] * (noise_pred_cond - noise_pred_no_portrait[0]), noise_pred_ovi, [cache_state_cond, cache_state_uncond]
                            elif multitalk_audio_input is not None and not math.isclose(audio_cfg_scale[idx], 1.0):
                                base_params['multitalk_audio'] = torch.zeros_like(multitalk_audio_input)[-1:]
                                noise_pred_uncond_audio, _, cache_state_uncond = transformer(
                                context=positive_embeds, pred_id=cache_state[0] if cache_state else None,
                                vace_data=vace_data, attn_cond=attn_cond, **base_params)
                                return noise_pred_uncond_audio[0] + audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_uncond_audio[0]), noise_pred_ovi, [cache_state_cond, cache_state_uncond]
                            else:
                                return noise_pred_cond, noise_pred_ovi, [cache_state_cond]

                        #unconditional (negative) pass
                        base_params['is_uncond'] = True
                        base_params['clip_fea'] = clip_fea_neg if clip_fea_neg is not None else clip_fea
                        base_params["add_text_emb"] = qwenvl_embeds_neg.to(device) if qwenvl_embeds_neg is not None else None # QwenVL embeddings for Bindweave
                        base_params['y'] = [image_cond_neg.to(z)] if image_cond_neg is not None else base_params['y']
                        if wananim_face_pixels is not None:
                            base_params['wananim_face_pixel_values'] = torch.zeros_like(wananim_face_pixels).to(device, torch.float32) - 1
                        if humo_audio_input_neg is not None:
                            base_params['humo_audio'] = humo_audio_input_neg
                        if neg_latent is not None:
                            base_params['x'] = [torch.cat([z[:, :-humo_reference_count], neg_latent], dim=1)]

                        noise_pred_uncond_text, noise_pred_ovi_uncond, cache_state_uncond = transformer(
                            context=negative_embeds if humo_audio_input_neg is None else positive_embeds, #ti #t
                            pred_id=cache_state[1] if cache_state else None,
                            vace_data=vace_data, attn_cond=attn_cond_neg,
                            **base_params)
                        noise_pred_uncond_text = noise_pred_uncond_text[0]
                        noise_pred_ovi_uncond = noise_pred_ovi_uncond[0] if noise_pred_ovi_uncond is not None else None

                        # HuMo
                        if not math.isclose(humo_audio_cfg_scale[idx], 1.0):
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            if humo_image_cond is not None and humo_audio_input_neg is not None:
                                if t > 980 and humo_image_cond_neg_input is not None: # use image cond for first timesteps
                                    base_params['y'] = [humo_image_cond_neg_input]

                                noise_pred_humo_audio_uncond, _, cache_state_humo = transformer(
                                context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                                **base_params)

                                noise_pred = (noise_pred_uncond_text + humo_audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_humo_audio_uncond[0])
                                            + (cfg_scale - 2.0) * (noise_pred_humo_audio_uncond[0] - noise_pred_uncond_text))
                                return noise_pred, None, [cache_state_cond, cache_state_uncond, cache_state_humo]
                            elif humo_audio_input is not None:
                                if cache_state is not None and len(cache_state) != 4:
                                    cache_state.append(None)
                                # audio
                                noise_pred_humo_null, _, cache_state_humo = transformer(
                                context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                                **base_params)
                                # negative
                                if humo_audio_input is not None:
                                    base_params['humo_audio'] = humo_audio_input
                                noise_pred_humo_audio, _, cache_state_humo2 = transformer(
                                context=positive_embeds, pred_id=cache_state[3] if cache_state else None, vace_data=None,
                                **base_params)
                                noise_pred = (humo_audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_humo_audio[0])
                                    + cfg_scale * (noise_pred_humo_audio[0] - noise_pred_uncond_text)
                                    + cfg_scale * (noise_pred_uncond_text - noise_pred_humo_null[0])
                                    + noise_pred_humo_null[0])
                                return noise_pred, None, [cache_state_cond, cache_state_uncond, cache_state_humo, cache_state_humo2]

                        #phantom
                        if use_phantom and not math.isclose(phantom_cfg_scale[idx], 1.0):
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            noise_pred_phantom, _, cache_state_phantom = transformer(
                            context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                            **base_params)

                            noise_pred = (noise_pred_uncond_text + phantom_cfg_scale[idx] * (noise_pred_phantom[0] - noise_pred_uncond_text)
                                          + cfg_scale * (noise_pred_cond - noise_pred_phantom[0]))
                            return noise_pred, None,[cache_state_cond, cache_state_uncond, cache_state_phantom]
                        # audio cfg (fantasytalking and multitalk)
                        if (fantasytalking_embeds is not None or multitalk_audio_input is not None):
                            if not math.isclose(audio_cfg_scale[idx], 1.0):
                                if cache_state is not None and len(cache_state) != 3:
                                    cache_state.append(None)

                                base_params['audio_proj'] = None
                                base_params['multitalk_audio'] = torch.zeros_like(multitalk_audio_input)[-1:] if multitalk_audio_input is not None else None
                                base_params['is_uncond'] = False
                                noise_pred_uncond_audio, _, cache_state_audio = transformer(
                                    context=negative_embeds,
                                    pred_id=cache_state[2] if cache_state else None,
                                    vace_data=vace_data,
                                    **base_params)
                                noise_pred_uncond_audio = noise_pred_uncond_audio[0]

                                noise_pred = noise_pred_uncond_audio + cfg_scale * (
                                    (noise_pred_cond - noise_pred_uncond_text)
                                    + audio_cfg_scale[idx] * (noise_pred_uncond_text - noise_pred_uncond_audio))
                                return noise_pred, None,[cache_state_cond, cache_state_uncond, cache_state_audio]
                        # lynx
                        if lynx_embeds is not None and not math.isclose(lynx_cfg_scale[idx], 1.0):
                            base_params['is_uncond'] = False
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            noise_pred_lynx, _, cache_state_lynx = transformer(
                            context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                            **base_params)

                            noise_pred = (noise_pred_uncond_text + lynx_cfg_scale[idx] * (noise_pred_lynx[0] - noise_pred_uncond_text)
                                          + cfg_scale * (noise_pred_cond - noise_pred_lynx[0]))
                            return noise_pred, None, [cache_state_cond, cache_state_uncond, cache_state_lynx]
                        # one-to-all
                        if one_to_all_data is not None and not math.isclose(one_to_all_pose_cfg_scale[idx], 1.0):
                            tqdm.write("One-to-All pose CFG pass...")
                            base_params['is_uncond'] = False
                            base_params['one_to_all_controlnet_strength'] = 0.0
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            noise_pred_pose_uncond, _, cache_state_ref = transformer(
                            context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                            **base_params)

                            noise_pred = (noise_pred_uncond_text + one_to_all_pose_cfg_scale[idx] * (noise_pred_pose_uncond[0] - noise_pred_uncond_text)
                                          + cfg_scale * (noise_pred_cond - noise_pred_pose_uncond[0]))
                            return noise_pred, None, [cache_state_cond, cache_state_uncond, cache_state_ref]

                    #batched
                    else:
                        base_params['z'] = [z] * 2
                        base_params['y'] = [image_cond_input] * 2 if image_cond_input is not None else None
                        base_params['clip_fea'] = torch.cat([clip_fea, clip_fea], dim=0)
                        cache_state_uncond = None
                        [noise_pred_cond, noise_pred_uncond], _, cache_state_cond = transformer(
                            context=positive_embeds + negative_embeds, is_uncond=False,
                            pred_id=cache_state[0] if cache_state else None,
                            **base_params
                        )
                except Exception as e:
                    log.error(f"Error during model prediction: {e}")
                    if force_offload:
                        if not model["auto_cpu_offload"]:
                            offload_transformer(transformer)
                    raise e

                #https://github.com/WeichenFan/CFG-Zero-star/
                alpha = 1.0
                if use_cfg_zero_star:
                    alpha = optimized_scale(
                        noise_pred_cond.view(batch_size, -1),
                        noise_pred_uncond.view(batch_size, -1)
                    ).view(batch_size, 1, 1, 1)

                noise_pred_uncond_text = noise_pred_uncond_text * alpha

                if use_tangential:
                    noise_pred_uncond_text = tangential_projection(noise_pred_cond, noise_pred_uncond_text)

                # RAAG (RATIO-aware Adaptive Guidance)
                if raag_alpha > 0.0:
                    cfg_scale = get_raag_guidance(noise_pred_cond, noise_pred_uncond_text, cfg_scale, raag_alpha)
                    log.info(f"RAAG modified cfg: {cfg_scale}")

                #https://github.com/WikiChao/FreSca
                if use_fresca:
                    filtered_cond = fourier_filter(noise_pred_cond - noise_pred_uncond, fresca_scale_low, fresca_scale_high, fresca_freq_cutoff)
                    noise_pred = noise_pred_uncond_text + cfg_scale * filtered_cond * alpha
                else:
                    noise_pred = noise_pred_uncond_text + cfg_scale * (noise_pred_cond - noise_pred_uncond_text)
                del noise_pred_uncond_text, noise_pred_cond

                if latent_model_input_ovi is not None:
                    if ovi_audio_cfg is None:
                        audio_cfg_scale = cfg_scale - 1.0 if cfg_scale > 4.0 else cfg_scale
                    else:
                        audio_cfg_scale = ovi_audio_cfg[idx]
                    noise_pred_ovi = noise_pred_ovi_uncond + audio_cfg_scale * (noise_pred_ovi - noise_pred_ovi_uncond)

                return noise_pred, noise_pred_ovi, [cache_state_cond, cache_state_uncond]

        if args.preview_method in [LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB]: #default for latent2rgb
            from latent_preview import prepare_callback
        else:
            from .latent_preview import prepare_callback #custom for tiny VAE previews
        callback = prepare_callback(patcher, len(timesteps))

        if not multitalk_sampling and not framepack and not wananimate_loop:
            log.info(f"Input sequence length: {seq_len}")
            log.info(f"Sampling {(latent_video_length-1) * 4 + 1} frames at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps-ttm_start_step} steps")

        intermediate_device = device

        # Differential diffusion prep
        masks = None
        if not multitalk_sampling and samples is not None and noise_mask is not None:
            thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
            thresholds = thresholds.reshape(-1, 1, 1, 1, 1).to(device)
            masks = (1-noise_mask.repeat(len(timesteps), 1, 1, 1, 1).to(device)) > thresholds

        latent_shift_loop = False
        if loop_args is not None:
            latent_shift_loop = is_looped = True
            latent_skip = loop_args["shift_skip"]
            latent_shift_start_percent = loop_args["start_percent"]
            latent_shift_end_percent = loop_args["end_percent"]
            shift_idx = 0

        #clear memory before sampling
        mm.soft_empty_cache()
        gc.collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        # Main sampling loop with FreeInit iterations
        iterations = freeinit_args.get("freeinit_num_iters", 3) if freeinit_args is not None else 1
        current_latent = latent
        initial_noise_saved = None

        for iter_idx in range(iterations):

            # FreeInit noise reinitialization (after first iteration)
            if freeinit_args is not None and iter_idx > 0:
                # restart scheduler for each iteration
                sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas)

                # Re-apply start_step and end_step logic to timesteps and sigmas
                if end_step != -1:
                    timesteps = timesteps[:end_step]
                    sample_scheduler.sigmas = sample_scheduler.sigmas[:end_step+1]
                if start_step > 0:
                    timesteps = timesteps[start_step:]
                    sample_scheduler.sigmas = sample_scheduler.sigmas[start_step:]
                if hasattr(sample_scheduler, 'timesteps'):
                    sample_scheduler.timesteps = timesteps

                # Diffuse current latent to t=999
                diffuse_timesteps = torch.full((noise.shape[0],), 999, device=device, dtype=torch.long)
                z_T = add_noise(
                    current_latent.to(device),
                    initial_noise_saved.to(device),
                    diffuse_timesteps
                )

                # Generate new random noise
                z_rand = torch.randn(z_T.shape, dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))
                # Apply frequency mixing
                current_latent = (freq_mix_3d(z_T.to(torch.float32), z_rand.to(device), LPF=freq_filter)).to(dtype)

            # Store initial noise for first iteration
            if freeinit_args is not None and iter_idx == 0:
                initial_noise_saved = current_latent.detach().clone()
                if input_samples is not None:
                    current_latent = input_samples.to(device)
                    continue

            # Reset per-iteration states
            self.cache_state = [None, None]
            self.cache_state_source = [None, None]
            self.cache_states_context = []
            if context_options is not None:
                self.window_tracker = WindowTracker(verbose=context_options["verbose"])

            # Set latent for denoising
            latent = current_latent

            if is_pusa and clean_latent_indices:
                pusa_noisy_steps = image_embeds.get("pusa_noisy_steps", -1)
                if pusa_noisy_steps == -1:
                    pusa_noisy_steps = len(timesteps)
            try:
                pbar = ProgressBar(len(timesteps) - ttm_start_step)
                #region main loop start
                for idx, t in enumerate(tqdm(timesteps[ttm_start_step:], disable=multitalk_sampling or wananimate_loop)):
                    if flowedit_args is not None:
                        if idx < skip_steps:
                            continue

                    if bidirectional_sampling:
                        latent_flipped = torch.flip(latent, dims=[1])
                        latent_model_input_flipped = latent_flipped.to(device)

                    #InfiniteTalk first frame handling
                    if (extra_latents is not None
                        and not multitalk_sampling
                        and transformer.multitalk_model_type=="InfiniteTalk"):
                        for entry in extra_latents:
                            add_index = entry["index"]
                            num_extra_frames = entry["samples"].shape[2]
                            latent[:, add_index:add_index+num_extra_frames] = entry["samples"].to(latent)

                    latent_model_input = latent.to(device)
                    latent_model_input_ovi = latent_ovi.to(device) if latent_ovi is not None else None

                    current_step_percentage = idx / len(timesteps)

                    timestep = torch.tensor([t]).to(device)
                    if is_pusa or ((is_5b or transformer.is_longcat) and clean_latent_indices):
                        orig_timestep = timestep
                        timestep = timestep.unsqueeze(1).repeat(1, latent_video_length)
                        if extra_latents is not None:
                            if clean_latent_indices and noise_multipliers is not None:
                                if is_pusa:
                                    scheduler_step_args["cond_frame_latent_indices"] = clean_latent_indices
                                    scheduler_step_args["noise_multipliers"] = noise_multipliers
                                for latent_idx in clean_latent_indices:
                                    timestep[:, latent_idx] = timestep[:, latent_idx] * noise_multipliers[latent_idx]
                                    # add noise for conditioning frames if multiplier > 0
                                    if idx < pusa_noisy_steps and noise_multipliers[latent_idx] > 0:
                                        latent_size = (1, latent.shape[0], latent.shape[1], latent.shape[2], latent.shape[3])
                                        noise_for_cond = torch.randn(latent_size, generator=seed_g, device=torch.device("cpu"))
                                        timestep_cond = torch.ones_like(timestep) * timestep.max()
                                        if is_pusa:
                                            latent[:, latent_idx:latent_idx+1] = sample_scheduler.add_noise_for_conditioning_frames(
                                                latent[:, latent_idx:latent_idx+1].to(device),
                                                noise_for_cond[:, :, latent_idx:latent_idx+1].to(device),
                                                timestep_cond[:, latent_idx:latent_idx+1].to(device),
                                                noise_multiplier=noise_multipliers[latent_idx])
                            else:
                                timestep[:, clean_latent_indices] = 0
                            #print("timestep: ", timestep)

                    ### latent shift
                    if latent_shift_loop:
                        if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                            latent_model_input = torch.cat([latent_model_input[:, shift_idx:]] + [latent_model_input[:, :shift_idx]], dim=1)

                    #enhance-a-video
                    enhance_enabled = False
                    if feta_args is not None and feta_start_percent <= current_step_percentage <= feta_end_percent:
                        enhance_enabled = True

                    #flow-edit
                    if flowedit_args is not None:
                        sigma = t / 1000.0
                        sigma_prev = (timesteps[idx + 1] if idx < len(timesteps) - 1 else timesteps[-1]) / 1000.0
                        noise = torch.randn(x_init.shape, generator=seed_g, device=torch.device("cpu"))
                        if idx < len(timesteps) - drift_steps:
                            cfg = drift_cfg

                        zt_src = (1-sigma) * x_init + sigma * noise.to(t)
                        zt_tgt = x_tgt + zt_src - x_init

                        #source
                        if idx < len(timesteps) - drift_steps:
                            if context_options is not None:
                                counter = torch.zeros_like(zt_src, device=intermediate_device)
                                vt_src = torch.zeros_like(zt_src, device=intermediate_device)
                                context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                                for c in context_queue:
                                    window_id = self.window_tracker.get_window_id(c)

                                    if cache_args is not None:
                                        current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                                    else:
                                        current_teacache = None

                                    prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                                    if context_options["verbose"]:
                                        log.info(f"Prompt index: {prompt_index}")

                                    if len(source_embeds["prompt_embeds"]) > 1:
                                        positive = source_embeds["prompt_embeds"][prompt_index]
                                    else:
                                        positive = source_embeds["prompt_embeds"]

                                    partial_img_emb = None
                                    if source_image_cond is not None:
                                        partial_img_emb = source_image_cond[:, c, :, :]
                                        partial_img_emb[:, 0, :, :] = source_image_cond[:, 0, :, :].to(intermediate_device)

                                    partial_zt_src = zt_src[:, c, :, :]
                                    vt_src_context, _, new_teacache = predict_with_cfg(
                                        partial_zt_src, cfg[idx],
                                        positive, source_embeds["negative_prompt_embeds"],
                                        timestep, idx, partial_img_emb, control_latents,
                                        source_clip_fea, current_teacache)

                                    if cache_args is not None:
                                        self.window_tracker.cache_states[window_id] = new_teacache

                                    window_mask = create_window_mask(vt_src_context, c, latent_video_length, context_overlap)
                                    vt_src[:, c, :, :] += vt_src_context * window_mask
                                    counter[:, c, :, :] += window_mask
                                vt_src /= counter
                            else:
                                vt_src, _, self.cache_state_source = predict_with_cfg(
                                    zt_src, cfg[idx],
                                    source_embeds["prompt_embeds"],
                                    source_embeds["negative_prompt_embeds"],
                                    timestep, idx, source_image_cond,
                                    source_clip_fea, control_latents,
                                    cache_state=self.cache_state_source)
                        else:
                            if idx == len(timesteps) - drift_steps:
                                x_tgt = zt_tgt
                            zt_tgt = x_tgt
                            vt_src = 0
                        #target
                        if context_options is not None:
                            counter = torch.zeros_like(zt_tgt, device=intermediate_device)
                            vt_tgt = torch.zeros_like(zt_tgt, device=intermediate_device)
                            context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                            for c in context_queue:
                                window_id = self.window_tracker.get_window_id(c)

                                if cache_args is not None:
                                    current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                                else:
                                    current_teacache = None

                                prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                                if context_options["verbose"]:
                                    log.info(f"Prompt index: {prompt_index}")

                                if len(text_embeds["prompt_embeds"]) > 1:
                                    positive = text_embeds["prompt_embeds"][prompt_index]
                                else:
                                    positive = text_embeds["prompt_embeds"]

                                partial_img_emb = None
                                partial_control_latents = None
                                if image_cond is not None:
                                    partial_img_emb = image_cond[:, c, :, :]
                                    partial_img_emb[:, 0, :, :] = image_cond[:, 0, :, :].to(intermediate_device)
                                if control_latents is not None:
                                    partial_control_latents = control_latents[:, c, :, :]

                                partial_zt_tgt = zt_tgt[:, c, :, :]
                                vt_tgt_context, _, new_teacache = predict_with_cfg(
                                    partial_zt_tgt, cfg[idx],
                                    positive, text_embeds["negative_prompt_embeds"],
                                    timestep, idx, partial_img_emb, partial_control_latents,
                                    clip_fea, current_teacache)

                                if cache_args is not None:
                                    self.window_tracker.cache_states[window_id] = new_teacache

                                window_mask = create_window_mask(vt_tgt_context, c, latent_video_length, context_overlap)
                                vt_tgt[:, c, :, :] += vt_tgt_context * window_mask
                                counter[:, c, :, :] += window_mask
                            vt_tgt /= counter
                        else:
                            vt_tgt, _,self.cache_state = predict_with_cfg(
                                zt_tgt, cfg[idx],
                                text_embeds["prompt_embeds"],
                                text_embeds["negative_prompt_embeds"],
                                timestep, idx, image_cond, clip_fea, control_latents,
                                cache_state=self.cache_state)
                        v_delta = vt_tgt - vt_src
                        x_tgt = x_tgt.to(torch.float32)
                        v_delta = v_delta.to(torch.float32)
                        x_tgt = x_tgt + (sigma_prev - sigma) * v_delta
                        x0 = x_tgt
                    #region context windowing
                    elif context_options is not None:
                        counter = torch.zeros_like(latent_model_input, device=intermediate_device)
                        noise_pred = torch.zeros_like(latent_model_input, device=intermediate_device)
                        context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                        fraction_per_context = 1.0 / len(context_queue)
                        context_pbar = ProgressBar(steps)
                        step_start_progress = idx

                        # Validate all context windows before processing
                        max_idx = latent_model_input.shape[1] if latent_model_input.ndim > 1 else 0
                        for window_indices in context_queue:
                            if not all(0 <= idx < max_idx for idx in window_indices):
                                raise ValueError(f"Invalid context window indices {window_indices} for latent_model_input with shape {latent_model_input.shape}")

                        for i, c in enumerate(context_queue):
                            window_id = self.window_tracker.get_window_id(c)

                            if cache_args is not None:
                                current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                            else:
                                current_teacache = None

                            prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                            if context_options["verbose"]:
                                log.info(f"Prompt index: {prompt_index}")

                            # Use the appropriate prompt for this section
                            if len(text_embeds["prompt_embeds"]) > 1:
                                positive = [text_embeds["prompt_embeds"][prompt_index]]
                            else:
                                positive = text_embeds["prompt_embeds"]

                            partial_img_emb = None
                            partial_control_latents = None
                            if image_cond is not None:
                                partial_img_emb = image_cond[:, c]
                                if c[0] != 0 and context_reference_latent is not None:
                                    if context_reference_latent.shape[0] == 1: #only single extra init latent
                                        new_init_image = context_reference_latent[0, :, 0].to(intermediate_device)
                                        # Concatenate the first 4 channels of partial_img_emb with new_init_image to match the required shape
                                        partial_img_emb[:, 0] = torch.cat([image_cond[:4, 0], new_init_image], dim=0)
                                    elif context_reference_latent.shape[0] > 1:
                                        num_extra_inits = context_reference_latent.shape[0]
                                        section_size = (latent_video_length / num_extra_inits)
                                        extra_init_index = min(int(max(c) / section_size), num_extra_inits - 1)
                                        if context_options["verbose"]:
                                            log.info(f"extra init image index: {extra_init_index}")
                                        new_init_image = context_reference_latent[extra_init_index, :, 0].to(intermediate_device)
                                        partial_img_emb[:, 0] = torch.cat([image_cond[:4, 0], new_init_image], dim=0)
                                else:
                                    new_init_image = image_cond[:, 0].to(intermediate_device)
                                    partial_img_emb[:, 0] = new_init_image

                                if control_latents is not None:
                                    partial_control_latents = control_latents[:, c]

                            partial_control_camera_latents = None
                            if control_camera_latents is not None:
                                partial_control_camera_latents = control_camera_latents[:, :, c]

                            partial_vace_context = None
                            if vace_data is not None:
                                window_vace_data = []
                                for vace_entry in vace_data:
                                    partial_context = vace_entry["context"][0][:, c]
                                    if has_ref:
                                        if c[0] != 0 and context_reference_latent is not None:
                                            if context_reference_latent.shape[0] == 1: #only single extra init latent
                                                partial_context[16:32, :1] = context_reference_latent[0, :, :1].to(intermediate_device)
                                            elif context_reference_latent.shape[0] > 1:
                                                num_extra_inits = context_reference_latent.shape[0]
                                                section_size = (latent_video_length / num_extra_inits)
                                                extra_init_index = min(int(max(c) / section_size), num_extra_inits - 1)
                                                if context_options["verbose"]:
                                                    log.info(f"extra init image index: {extra_init_index}")
                                                partial_context[16:32, :1] = context_reference_latent[extra_init_index, :, :1].to(intermediate_device)
                                        else:
                                            partial_context[:, 0] = vace_entry["context"][0][:, 0]

                                    window_vace_data.append({
                                        "context": [partial_context],
                                        "scale": vace_entry["scale"],
                                        "start": vace_entry["start"],
                                        "end": vace_entry["end"],
                                        "seq_len": vace_entry["seq_len"]
                                    })

                                partial_vace_context = window_vace_data

                            partial_audio_proj = None
                            if fantasytalking_embeds is not None:
                                partial_audio_proj = audio_proj[:, c]

                            partial_fantasy_portrait_input = None
                            if fantasy_portrait_input is not None:
                                partial_fantasy_portrait_input = fantasy_portrait_input.copy()
                                partial_fantasy_portrait_input["adapter_proj"] = fantasy_portrait_input["adapter_proj"][:, c]

                            partial_latent_model_input = latent_model_input[:, c]
                            if latents_to_insert is not None and c[0] != 0:
                                partial_latent_model_input[:, :1] = latents_to_insert

                            partial_unianim_data = None
                            if unianim_data is not None:
                                partial_dwpose = dwpose_data[:, :, c]
                                partial_unianim_data = {
                                    "dwpose": partial_dwpose,
                                    "random_ref": unianim_data["random_ref"],
                                    "strength": unianimate_poses["strength"],
                                    "start_percent": unianimate_poses["start_percent"],
                                    "end_percent": unianimate_poses["end_percent"]
                                }

                            partial_mtv_motion_tokens = None
                            if mtv_input is not None:
                                start_token_index = c[0] * 24
                                end_token_index = (c[-1] + 1) * 24
                                partial_mtv_motion_tokens = mtv_motion_tokens[:, start_token_index:end_token_index, :]
                                if context_options["verbose"]:
                                    log.info(f"context window: {c}")
                                    log.info(f"motion_token_indices: {start_token_index}-{end_token_index}")

                            partial_s2v_audio_input = None
                            if s2v_audio_input is not None:
                                audio_start = c[0] * 4
                                audio_end = c[-1] * 4 + 1
                                center_indices = torch.arange(audio_start, audio_end, 1)
                                center_indices = torch.clamp(center_indices, min=0, max=s2v_audio_input.shape[-1] - 1)
                                partial_s2v_audio_input = s2v_audio_input[..., center_indices]

                            partial_s2v_pose = None
                            if s2v_pose is not None:
                                partial_s2v_pose = s2v_pose[:, :, c].to(device, dtype)

                            partial_add_cond = None
                            if add_cond is not None:
                                partial_add_cond = add_cond[:, :, c].to(device, dtype)

                            partial_wananim_face_pixels = partial_wananim_pose_latents = None
                            if wananim_face_pixels is not None and partial_wananim_face_pixels is None:
                                start = c[0] * 4
                                end = c[-1] * 4
                                center_indices = torch.arange(start, end, 1)
                                center_indices = torch.clamp(center_indices, min=0, max=wananim_face_pixels.shape[2] - 1)
                                partial_wananim_face_pixels = wananim_face_pixels[:, :, center_indices].to(device, dtype)
                            if wananim_pose_latents is not None:
                                start = c[0]
                                end = c[-1]
                                center_indices = torch.arange(start, end, 1)
                                center_indices = torch.clamp(center_indices, min=0, max=wananim_pose_latents.shape[2] - 1)
                                partial_wananim_pose_latents = wananim_pose_latents[:, :, center_indices][:, :, :context_frames-1].to(device, dtype)

                            partial_flashvsr_LQ_latent = None
                            if LQ_images is not None:
                                start = c[0] * 4
                                end = c[-1] * 4 + 1 + 4
                                center_indices = torch.arange(start, end, 1)
                                center_indices = torch.clamp(center_indices, min=0, max=LQ_images.shape[2] - 1)
                                partial_flashvsr_LQ_images = LQ_images[:, :, center_indices].to(device)
                                partial_flashvsr_LQ_latent = transformer.LQ_proj_in(partial_flashvsr_LQ_images)

                            if len(timestep.shape) != 1:
                                partial_timestep = timestep[:, c]
                                partial_timestep[:, :1] = 0
                            else:
                                partial_timestep = timestep

                            orig_model_input_frames = partial_latent_model_input.shape[1]

                            noise_pred_context, _, new_teacache = predict_with_cfg(
                                partial_latent_model_input,
                                cfg[idx], positive,
                                text_embeds["negative_prompt_embeds"],
                                partial_timestep, idx, partial_img_emb, clip_fea, partial_control_latents, partial_vace_context, partial_unianim_data,partial_audio_proj,
                                partial_control_camera_latents, partial_add_cond, current_teacache, context_window=c, fantasy_portrait_input=partial_fantasy_portrait_input,
                                mtv_motion_tokens=partial_mtv_motion_tokens, s2v_audio_input=partial_s2v_audio_input, s2v_motion_frames=[1, 0], s2v_pose=partial_s2v_pose,
                                humo_image_cond=humo_image_cond, humo_image_cond_neg=humo_image_cond_neg, humo_audio=humo_audio, humo_audio_neg=humo_audio_neg,
                                wananim_face_pixels=partial_wananim_face_pixels, wananim_pose_latents=partial_wananim_pose_latents, multitalk_audio_embeds=multitalk_audio_embeds,
                                uni3c_data=uni3c_data, flashvsr_LQ_latent=partial_flashvsr_LQ_latent)

                            if cache_args is not None:
                                self.window_tracker.cache_states[window_id] = new_teacache

                            if mocha_embeds is not None:
                                noise_pred_context = noise_pred_context[:, :orig_model_input_frames]

                            window_mask = create_window_mask(noise_pred_context, c, noise.shape[1], context_overlap, looped=is_looped, window_type=context_options["fuse_method"])
                            noise_pred[:, c] += noise_pred_context * window_mask
                            counter[:, c] += window_mask
                            context_pbar.update_absolute(step_start_progress + (i + 1) * fraction_per_context, len(timesteps))
                        noise_pred /= counter
                    #region multitalk
                    elif multitalk_sampling:
                        mode = image_embeds.get("multitalk_mode", "multitalk")
                        if mode == "auto":
                            mode = transformer.multitalk_model_type.lower()
                        log.info(f"Multitalk mode: {mode}")
                        cond_frame = None
                        offload = image_embeds.get("force_offload", False)
                        offloaded = False
                        tiled_vae = image_embeds.get("tiled_vae", False)
                        frame_num = clip_length = image_embeds.get("frame_window_size", 81)

                        clip_embeds = image_embeds.get("clip_context", None)
                        if clip_embeds is not None:
                            clip_embeds = clip_embeds.to(dtype)
                        colormatch = image_embeds.get("colormatch", "disabled")
                        motion_frame = image_embeds.get("motion_frame", 25)
                        target_w = image_embeds.get("target_w", None)
                        target_h = image_embeds.get("target_h", None)
                        original_images = cond_image = image_embeds.get("multitalk_start_image", None)
                        if original_images is None:
                            original_images = torch.zeros([noise.shape[0], 1, target_h, target_w], device=device)

                        output_path = image_embeds.get("output_path", "")
                        img_counter = 0

                        if len(multitalk_embeds['audio_features'])==2 and (multitalk_embeds['ref_target_masks'] is None):
                            face_scale = 0.1
                            x_min, x_max = int(target_h * face_scale), int(target_h * (1 - face_scale))
                            lefty_min, lefty_max = int((target_w//2) * face_scale), int((target_w//2) * (1 - face_scale))
                            righty_min, righty_max = int((target_w//2) * face_scale + (target_w//2)), int((target_w//2) * (1 - face_scale) + (target_w//2))
                            human_mask1, human_mask2 = (torch.zeros([target_h, target_w]) for _ in range(2))
                            human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                            human_mask2[x_min:x_max, righty_min:righty_max] = 1
                            background_mask = torch.where((human_mask1 + human_mask2) > 0, torch.tensor(0), torch.tensor(1))
                            human_masks = [human_mask1, human_mask2, background_mask]
                            ref_target_masks = torch.stack(human_masks, dim=0)
                            multitalk_embeds['ref_target_masks'] = ref_target_masks

                        gen_video_list = []
                        is_first_clip = True
                        arrive_last_frame = False
                        cur_motion_frames_num = 1
                        audio_start_idx = iteration_count = step_iteration_count = 0
                        audio_end_idx = (audio_start_idx + clip_length) * audio_stride
                        indices = (torch.arange(4 + 1) - 2) * 1
                        current_condframe_index = 0

                        audio_embedding = multitalk_audio_embeds
                        human_num = len(audio_embedding)
                        audio_embs = None
                        cond_frame = None

                        uni3c_data = uni3c_data_input = None
                        if uni3c_embeds is not None:
                            transformer.controlnet = uni3c_embeds["controlnet"]
                            uni3c_data = uni3c_embeds.copy()

                        encoded_silence = None

                        try:
                            silence_path = os.path.join(script_directory, "multitalk", "encoded_silence.safetensors")
                            encoded_silence = load_torch_file(silence_path)["audio_emb"].to(dtype)
                        except:
                             log.warning("No encoded silence file found, padding with end of audio embedding instead.")

                        total_frames = len(audio_embedding[0])
                        estimated_iterations = total_frames // (frame_num - motion_frame) + 1
                        callback = prepare_callback(patcher, estimated_iterations)

                        if frame_num >= total_frames:
                            arrive_last_frame = True
                            estimated_iterations = 1

                        log.info(f"Sampling {total_frames} frames in {estimated_iterations} windows, at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")

                        while True: # start video generation iteratively
                            self.cache_state = [None, None]

                            cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)
                            if mode == "infinitetalk":
                                cond_image = original_images[:, :, current_condframe_index:current_condframe_index+1] if cond_image is not None else None
                            if multitalk_embeds is not None:
                                audio_embs = []
                                # split audio with window size
                                for human_idx in range(human_num):
                                    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
                                    center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0]-1)
                                    audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                                    audio_embs.append(audio_emb)
                                audio_embs = torch.concat(audio_embs, dim=0).to(dtype)

                            h, w = (cond_image.shape[-2], cond_image.shape[-1]) if cond_image is not None else (target_h, target_w)
                            lat_h, lat_w = h // VAE_STRIDE[1], w // VAE_STRIDE[2]
                            seq_len = ((frame_num - 1) // VAE_STRIDE[0] + 1) * lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
                            latent_frame_num = (frame_num - 1) // 4 + 1

                            noise = torch.randn(
                                16, latent_frame_num,
                                lat_h, lat_w, dtype=torch.float32, device=torch.device("cpu"), generator=seed_g).to(device)

                            # Calculate the correct latent slice based on current iteration
                            if is_first_clip:
                                latent_start_idx = 0
                                latent_end_idx = noise.shape[1]
                            else:
                                new_frames_per_iteration = frame_num - motion_frame
                                new_latent_frames_per_iteration = ((new_frames_per_iteration - 1) // 4 + 1)
                                latent_start_idx = iteration_count * new_latent_frames_per_iteration
                                latent_end_idx = latent_start_idx + noise.shape[1]

                            if samples is not None:
                                noise_mask = samples.get("noise_mask", None)
                                input_samples = samples["samples"]
                                if input_samples is not None:
                                    input_samples = input_samples.squeeze(0).to(noise)
                                    # Check if we have enough frames in input_samples
                                    if latent_end_idx > input_samples.shape[1]:
                                        # We need more frames than available - pad the input_samples at the end
                                        pad_length = latent_end_idx - input_samples.shape[1]
                                        last_frame = input_samples[:, -1:].repeat(1, pad_length, 1, 1)
                                        input_samples = torch.cat([input_samples, last_frame], dim=1)
                                    input_samples = input_samples[:, latent_start_idx:latent_end_idx]
                                    if noise_mask is not None:
                                        original_image = input_samples.to(device)

                                    assert input_samples.shape[1] == noise.shape[1], f"Slice mismatch: {input_samples.shape[1]} vs {noise.shape[1]}"

                                    if add_noise_to_samples:
                                        latent_timestep = timesteps[0]
                                        noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples
                                    else:
                                        noise = input_samples

                                # diff diff prep
                                if noise_mask is not None:
                                    if len(noise_mask.shape) == 4:
                                        noise_mask = noise_mask.squeeze(1)
                                    if audio_end_idx > noise_mask.shape[0]:
                                        noise_mask = noise_mask.repeat(audio_end_idx // noise_mask.shape[0], 1, 1)
                                    noise_mask = noise_mask[audio_start_idx:audio_end_idx]
                                    noise_mask = torch.nn.functional.interpolate(
                                        noise_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                                        size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                                        mode='trilinear',
                                        align_corners=False
                                    ).repeat(1, noise.shape[0], 1, 1, 1)

                                    thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
                                    thresholds = thresholds.reshape(-1, 1, 1, 1, 1).to(device)
                                    masks = (1-noise_mask.repeat(len(timesteps), 1, 1, 1, 1).to(device)) > thresholds

                            # zero padding and vae encode for img cond
                            if cond_image is not None or cond_frame is not None:
                                cond_ = cond_image if (is_first_clip or humo_image_cond is None) else cond_frame
                                cond_frame_num = cond_.shape[2]
                                video_frames = torch.zeros(1, 3, frame_num-cond_frame_num, target_h, target_w, device=device, dtype=vae.dtype)
                                padding_frames_pixels_values = torch.concat([cond_.to(device, vae.dtype), video_frames], dim=2)

                                # encode
                                vae.to(device)
                                y = vae.encode(padding_frames_pixels_values, device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]

                                if mode == "multitalk":
                                    latent_motion_frames = y[:, :cur_motion_frames_latent_num] # C T H W
                                else:
                                    cond_ = cond_image if is_first_clip else cond_frame
                                    latent_motion_frames = vae.encode(cond_.to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]

                                vae.to(offload_device)

                                #motion_frame_index = cur_motion_frames_latent_num if mode == "infinitetalk" else 1
                                msk = torch.zeros(4, latent_frame_num, lat_h, lat_w, device=device, dtype=dtype)
                                msk[:, :1] = 1
                                y = torch.cat([msk, y]) # 4+C T H W
                                mm.soft_empty_cache()
                            else:
                                y = None
                                latent_motion_frames = noise[:, :1]

                            partial_humo_cond_input = partial_humo_cond_neg_input = partial_humo_audio = partial_humo_audio_neg = None
                            if humo_image_cond is not None:
                                partial_humo_cond_input = humo_image_cond[:, :latent_frame_num]
                                partial_humo_cond_neg_input = humo_image_cond_neg[:, :latent_frame_num]
                                if y is not None:
                                    partial_humo_cond_input[:, :1] = y[:, :1]
                                if humo_reference_count > 0:
                                    partial_humo_cond_input[:, -humo_reference_count:] = humo_image_cond[:, -humo_reference_count:]
                                    partial_humo_cond_neg_input[:, -humo_reference_count:] = humo_image_cond_neg[:, -humo_reference_count:]

                            if humo_audio is not None:
                                if is_first_clip:
                                    audio_embs = None

                                partial_humo_audio, _ = get_audio_emb_window(humo_audio, frame_num, frame0_idx=audio_start_idx)
                                #zero_audio_pad = torch.zeros(humo_reference_count, *partial_humo_audio.shape[1:], device=partial_humo_audio.device, dtype=partial_humo_audio.dtype)
                                partial_humo_audio[-humo_reference_count:] = 0
                                partial_humo_audio_neg = torch.zeros_like(partial_humo_audio, device=partial_humo_audio.device, dtype=partial_humo_audio.dtype)

                            if scheduler == "multitalk":
                                timesteps = list(np.linspace(1000, 1, steps, dtype=np.float32))
                                timesteps.append(0.)
                                timesteps = [torch.tensor([t], device=device) for t in timesteps]
                                timesteps = [timestep_transform(t, shift=shift, num_timesteps=1000) for t in timesteps]
                            else:
                                if isinstance(scheduler, dict):
                                    sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
                                    timesteps = scheduler["timesteps"]
                                else:
                                    sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, total_steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas)
                                timesteps = [torch.tensor([float(t)], device=device) for t in timesteps] + [torch.tensor([0.], device=device)]

                            # sample videos
                            latent = noise

                            # injecting motion frames
                            if not is_first_clip and mode == "multitalk":
                                latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                                motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                                add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                                latent[:, :add_latent.shape[1]] = add_latent

                            if offloaded:
                                # Load weights
                                if transformer.patched_linear and gguf_reader is None:
                                    load_weights(patcher.model.diffusion_model, patcher.model["sd"], weight_dtype, base_dtype=dtype, transformer_load_device=device, block_swap_args=block_swap_args)
                                elif gguf_reader is not None: #handle GGUF
                                    load_weights(transformer, patcher.model["sd"], base_dtype=dtype, transformer_load_device=device, patcher=patcher, gguf=True, reader=gguf_reader, block_swap_args=block_swap_args)
                                #blockswap init
                                init_blockswap(transformer, block_swap_args, model)

                            # Use the appropriate prompt for this section
                            if len(text_embeds["prompt_embeds"]) > 1:
                                prompt_index = min(iteration_count, len(text_embeds["prompt_embeds"]) - 1)
                                positive = [text_embeds["prompt_embeds"][prompt_index]]
                                log.info(f"Using prompt index: {prompt_index}")
                            else:
                                positive = text_embeds["prompt_embeds"]

                            # uni3c slices
                            if uni3c_embeds is not None:
                                vae.to(device)
                                # Pad original_images if needed
                                num_frames = original_images.shape[2]
                                if audio_end_idx > num_frames:
                                    pad_len = audio_end_idx - num_frames
                                    last_frame = original_images[:, :, -1:].repeat(1, 1, pad_len, 1, 1)
                                    padded_images = torch.cat([original_images, last_frame], dim=2)
                                else:
                                    padded_images = original_images
                                render_latent = vae.encode(
                                    padded_images[:, :, audio_start_idx:audio_end_idx].to(device, vae.dtype),
                                    device=device, tiled=tiled_vae
                                ).to(dtype)

                                vae.to(offload_device)
                                uni3c_data['render_latent'] = render_latent

                            # unianimate slices
                            partial_unianim_data = None
                            if unianim_data is not None:
                                partial_dwpose = dwpose_data[:, :, latent_start_idx:latent_end_idx]
                                partial_unianim_data = {
                                    "dwpose": partial_dwpose,
                                    "random_ref": unianim_data["random_ref"],
                                    "strength": unianimate_poses["strength"],
                                    "start_percent": unianimate_poses["start_percent"],
                                    "end_percent": unianimate_poses["end_percent"]
                                }

                            # fantasy portrait slices
                            partial_fantasy_portrait_input = None
                            if fantasy_portrait_input is not None:
                                adapter_proj = fantasy_portrait_input["adapter_proj"]
                                if latent_end_idx > adapter_proj.shape[1]:
                                    pad_len = latent_end_idx - adapter_proj.shape[1]
                                    last_frame = adapter_proj[:, -1:, :, :].repeat(1, pad_len, 1, 1)
                                    padded_proj = torch.cat([adapter_proj, last_frame], dim=1)
                                else:
                                    padded_proj = adapter_proj
                                partial_fantasy_portrait_input = fantasy_portrait_input.copy()
                                partial_fantasy_portrait_input["adapter_proj"] = padded_proj[:, latent_start_idx:latent_end_idx]

                            mm.soft_empty_cache()
                            gc.collect()
                            # sampling loop
                            sampling_pbar = tqdm(total=len(timesteps)-1, desc=f"Sampling audio indices {audio_start_idx}-{audio_end_idx}", position=0, leave=True)
                            for i in range(len(timesteps)-1):
                                timestep = timesteps[i]
                                latent_model_input = latent.to(device)
                                if mode == "infinitetalk":
                                    if humo_image_cond is None or not is_first_clip:
                                        latent_model_input[:, :cur_motion_frames_latent_num] = latent_motion_frames

                                noise_pred, _, self.cache_state = predict_with_cfg(
                                    latent_model_input, cfg[min(i, len(timesteps)-1)], positive, text_embeds["negative_prompt_embeds"],
                                    timestep, i, y, clip_embeds, control_latents, None, partial_unianim_data, audio_proj, control_camera_latents, add_cond,
                                    cache_state=self.cache_state, multitalk_audio_embeds=audio_embs, fantasy_portrait_input=partial_fantasy_portrait_input,
                                    humo_image_cond=partial_humo_cond_input, humo_image_cond_neg=partial_humo_cond_neg_input, humo_audio=partial_humo_audio, humo_audio_neg=partial_humo_audio_neg,
                                    uni3c_data = uni3c_data)

                                if callback is not None:
                                    callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                    callback(step_iteration_count, callback_latent, None, estimated_iterations*(len(timesteps)-1))
                                    del callback_latent

                                sampling_pbar.update(1)
                                step_iteration_count += 1

                                # update latent
                                if use_tsr:
                                    noise_pred = temporal_score_rescaling(noise_pred, latent, timestep, tsr_k, tsr_sigma)
                                if scheduler == "multitalk":
                                    noise_pred = -noise_pred
                                    dt = (timesteps[i] - timesteps[i + 1]) / 1000
                                    latent = latent + noise_pred * dt[:, None, None, None]
                                else:
                                    latent = sample_scheduler.step(noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0).to(noise_pred.device), **scheduler_step_args)[0].squeeze(0)
                                del noise_pred, latent_model_input, timestep

                                # differential diffusion inpaint
                                if masks is not None:
                                    if i < len(timesteps) - 1:
                                        image_latent = add_noise(original_image.to(device), noise.to(device), timesteps[i+1])
                                        mask = masks[i].to(latent)
                                        latent = image_latent * mask + latent * (1-mask)

                                # injecting motion frames
                                if not is_first_clip and mode == "multitalk":
                                    latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                                    motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                                    add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                                    latent[:, :add_latent.shape[1]] = add_latent
                                else:
                                    if humo_image_cond is None or not is_first_clip:
                                        latent[:, :cur_motion_frames_latent_num] = latent_motion_frames

                            del noise, latent_motion_frames
                            if offload:
                                offload_transformer(transformer, remove_lora=False)
                                offloaded = True
                            if humo_image_cond is not None and humo_reference_count > 0:
                                latent = latent[:,:-humo_reference_count]
                            vae.to(device)
                            videos = vae.decode(latent.unsqueeze(0).to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False)[0].cpu()

                            vae.to(offload_device)

                            sampling_pbar.close()

                            # optional color correction (less relevant for InfiniteTalk)
                            if colormatch != "disabled":
                                videos = videos.permute(1, 2, 3, 0).float().numpy()
                                from color_matcher import ColorMatcher
                                cm = ColorMatcher()
                                cm_result_list = []
                                for img in videos:
                                    if mode == "multitalk":
                                        cm_result = cm.transfer(src=img, ref=original_images[0].permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
                                    else:
                                        cm_result = cm.transfer(src=img, ref=cond_image[0].permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
                                    cm_result_list.append(torch.from_numpy(cm_result).to(vae.dtype))

                                videos = torch.stack(cm_result_list, dim=0).permute(3, 0, 1, 2)

                            # optionally save generated samples to disk
                            if output_path:
                                video_np = videos.clamp(-1.0, 1.0).add(1.0).div(2.0).mul(255).cpu().float().numpy().transpose(1, 2, 3, 0).astype('uint8')
                                num_frames_to_save = video_np.shape[0] if is_first_clip else video_np.shape[0] - cur_motion_frames_num
                                log.info(f"Saving {num_frames_to_save} generated frames to {output_path}")
                                start_idx = 0 if is_first_clip else cur_motion_frames_num
                                for i in range(start_idx, video_np.shape[0]):
                                    im = Image.fromarray(video_np[i])
                                    im.save(os.path.join(output_path, f"frame_{img_counter:05d}.png"))
                                    img_counter += 1
                            else:
                                gen_video_list.append(videos if is_first_clip else videos[:, cur_motion_frames_num:])

                            current_condframe_index += 1
                            iteration_count += 1

                            # decide whether is done
                            if arrive_last_frame:
                                break

                            # update next condition frames
                            is_first_clip = False
                            cur_motion_frames_num = motion_frame

                            cond_ = videos[:, -cur_motion_frames_num:].unsqueeze(0)
                            if mode == "infinitetalk":
                                cond_frame = cond_
                            else:
                                cond_image = cond_

                            del videos, latent

                            # Repeat audio emb
                            if multitalk_embeds is not None:
                                audio_start_idx += (frame_num - cur_motion_frames_num - humo_reference_count)
                                audio_end_idx = audio_start_idx + clip_length
                                if audio_end_idx >= len(audio_embedding[0]):
                                    arrive_last_frame = True
                                    miss_lengths = []
                                    source_frames = []
                                    for human_inx in range(human_num):
                                        source_frame = len(audio_embedding[human_inx])
                                        source_frames.append(source_frame)
                                        if audio_end_idx >= len(audio_embedding[human_inx]):
                                            log.warning(f"Audio embedding for subject {human_inx} not long enough: {len(audio_embedding[human_inx])}, need {audio_end_idx}, padding...")
                                            miss_length = audio_end_idx - len(audio_embedding[human_inx]) + 3
                                            log.warning(f"Padding length: {miss_length}")
                                            if encoded_silence is not None:
                                                add_audio_emb = encoded_silence[-1*miss_length:]
                                            else:
                                                add_audio_emb = torch.flip(audio_embedding[human_inx][-1*miss_length:], dims=[0])
                                            audio_embedding[human_inx] = torch.cat([audio_embedding[human_inx], add_audio_emb.to(device, dtype)], dim=0)
                                            miss_lengths.append(miss_length)
                                        else:
                                            miss_lengths.append(0)
                                if mode == "infinitetalk" and current_condframe_index >= original_images.shape[2]:
                                    last_frame = original_images[:, :, -1:, :, :]
                                    miss_length   = 1
                                    original_images = torch.cat([original_images, last_frame.repeat(1, 1, miss_length, 1, 1)], dim=2)

                        if not output_path:
                            gen_video_samples = torch.cat(gen_video_list, dim=1)
                        else:
                            gen_video_samples = torch.zeros(3, 1, 64, 64) # dummy output

                        if force_offload:
                            if not model["auto_cpu_offload"]:
                                offload_transformer(transformer)
                        try:
                            print_memory(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        except:
                            pass
                        return {"video": gen_video_samples.permute(1, 2, 3, 0), "output_path": output_path},
                    # region framepack loop
                    elif framepack:
                        framepack_out = []
                        ref_motion_image = None
                        #infer_frames = image_embeds["num_frames"]
                        infer_frames = s2v_audio_embeds.get("frame_window_size", 80)
                        motion_frames = infer_frames - 7 #73 default
                        lat_motion_frames = (motion_frames + 3) // 4
                        lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames

                        step_iteration_count = 0
                        total_frames = s2v_audio_input.shape[-1]

                        s2v_motion_frames = [motion_frames, lat_motion_frames]

                        noise = torch.randn( #C, T, H, W
                            48 if is_5b else 16,
                                lat_target_frames,
                                target_shape[2],
                                target_shape[3],
                                dtype=torch.float32,
                                generator=seed_g,
                                device=torch.device("cpu"))

                        seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

                        if ref_motion_image is None:
                            ref_motion_image = torch.zeros(
                                [1, 3, motion_frames, latent.shape[2]*vae_upscale_factor, latent.shape[3]*vae_upscale_factor],
                                dtype=vae.dtype,
                                device=device)
                        videos_last_frames = ref_motion_image

                        if s2v_pose is not None:
                            pose_cond_list = []
                            for r in range(s2v_num_repeat):
                                pose_start = r * (infer_frames // 4)
                                pose_end = pose_start + (infer_frames // 4)

                                cond_lat = s2v_pose[:, :, pose_start:pose_end]

                                pad_len = (infer_frames // 4) - cond_lat.shape[2]
                                if pad_len > 0:
                                    pad = -torch.ones(cond_lat.shape[0], cond_lat.shape[1], pad_len, cond_lat.shape[3], cond_lat.shape[4], device=cond_lat.device, dtype=cond_lat.dtype)
                                    cond_lat = torch.cat([cond_lat, pad], dim=2)
                                pose_cond_list.append(cond_lat.cpu())

                        log.info(f"Sampling {total_frames} frames in {s2v_num_repeat} windows, at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")
                        # sample
                        for r in range(s2v_num_repeat):

                            mm.soft_empty_cache()
                            gc.collect()
                            if ref_motion_image is not None:
                                vae.to(device)
                                ref_motion = vae.encode(ref_motion_image.to(vae.dtype), device=device, pbar=False).to(dtype)[0]

                                vae.to(offload_device)

                            left_idx = r * infer_frames
                            right_idx = r * infer_frames + infer_frames

                            s2v_audio_input_slice = s2v_audio_input[..., left_idx:right_idx]
                            if s2v_audio_input_slice.shape[-1] < (right_idx - left_idx):
                                pad_len = (right_idx - left_idx) - s2v_audio_input_slice.shape[-1]
                                pad_shape = list(s2v_audio_input_slice.shape)
                                pad_shape[-1] = pad_len
                                pad = torch.zeros(pad_shape, device=s2v_audio_input_slice.device, dtype=s2v_audio_input_slice.dtype)
                                log.info(f"Padding s2v_audio_input_slice from {s2v_audio_input_slice.shape[-1]} to {right_idx - left_idx}")
                                s2v_audio_input_slice = torch.cat([s2v_audio_input_slice, pad], dim=-1)

                            if ref_motion_image is not None:
                                input_motion_latents = ref_motion.clone().unsqueeze(0)
                            else:
                                input_motion_latents = None

                            s2v_pose_slice = None
                            if s2v_pose is not None:
                                s2v_pose_slice = pose_cond_list[r].to(device)

                            if isinstance(scheduler, dict):
                                sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
                                timesteps = scheduler["timesteps"]
                            else:
                                sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, total_steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas)

                            latent = noise.to(device)
                            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling audio indices {left_idx}-{right_idx}", position=0)):
                                latent_model_input = latent.to(device)
                                timestep = torch.tensor([t]).to(device)
                                noise_pred, _, self.cache_state = predict_with_cfg(
                                    latent_model_input,
                                    cfg[idx],
                                    text_embeds["prompt_embeds"],
                                    text_embeds["negative_prompt_embeds"],
                                    timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                                    cache_state=self.cache_state, fantasy_portrait_input=fantasy_portrait_input, mtv_motion_tokens=mtv_motion_tokens,
                                    s2v_audio_input=s2v_audio_input_slice, s2v_ref_motion=input_motion_latents, s2v_motion_frames=s2v_motion_frames, s2v_pose=s2v_pose_slice)

                                latent = sample_scheduler.step(
                                        noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0),
                                        **scheduler_step_args)[0].squeeze(0)
                                if callback is not None:
                                    callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                    callback(step_iteration_count, callback_latent, None, s2v_num_repeat*(len(timesteps)))
                                    del callback_latent
                                step_iteration_count += 1
                                del latent_model_input, noise_pred


                            vae.to(device)
                            decode_latents = torch.cat([ref_motion.unsqueeze(0), latent.unsqueeze(0)], dim=2)
                            image = vae.decode(decode_latents.to(device, vae.dtype), device=device, pbar=False)[0]
                            del decode_latents
                            image = image.unsqueeze(0)[:, :, -infer_frames:]
                            if r == 0:
                                image = image[:, :, 3:]

                            framepack_out.append(image.cpu())

                            overlap_frames_num = min(motion_frames, image.shape[2])

                            videos_last_frames = torch.cat([
                                videos_last_frames[:, :, overlap_frames_num:],
                                image[:, :, -overlap_frames_num:]], dim=2).to(device, vae.dtype)

                            ref_motion_image = videos_last_frames

                        vae.to(offload_device)

                        mm.soft_empty_cache()
                        gen_video_samples = torch.cat(framepack_out, dim=2).squeeze(0).permute(1, 2, 3, 0)

                        if force_offload:
                            if not model["auto_cpu_offload"]:
                                offload_transformer(transformer)
                        try:
                            print_memory(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        except:
                            pass
                        return {"video": gen_video_samples},
                    # region wananimate loop
                    elif wananimate_loop:
                        # calculate frame counts
                        total_frames = num_frames
                        refert_num = 1

                        real_clip_len = frame_window_size - refert_num
                        last_clip_num = (total_frames - refert_num) % real_clip_len
                        extra = 0 if last_clip_num == 0 else real_clip_len - last_clip_num
                        target_len = total_frames + extra
                        estimated_iterations = target_len // real_clip_len
                        target_latent_len = (target_len - 1) // 4 + estimated_iterations
                        latent_window_size = (frame_window_size - 1) // 4 + 1

                        from .utils import tensor_pingpong_pad

                        ref_latent = image_embeds.get("ref_latent", None)
                        ref_images = image_embeds.get("ref_image", None)
                        bg_images = image_embeds.get("bg_images", None)
                        pose_images = image_embeds.get("pose_images", None)

                        current_ref_images = face_images = face_images_in = None

                        if wananim_face_pixels is not None:
                            face_images = tensor_pingpong_pad(wananim_face_pixels, target_len)
                            log.info(f"WanAnimate: Face input {wananim_face_pixels.shape} padded to shape {face_images.shape}")
                        if wananim_ref_masks is not None:
                            ref_masks_in = tensor_pingpong_pad(wananim_ref_masks, target_latent_len)
                            log.info(f"WanAnimate: Ref masks {wananim_ref_masks.shape} padded to shape {ref_masks_in.shape}")
                        if bg_images is not None:
                            bg_images_in = tensor_pingpong_pad(bg_images, target_len)
                            log.info(f"WanAnimate: BG images {bg_images.shape} padded to shape {bg_images.shape}")
                        if pose_images is not None:
                            pose_images_in = tensor_pingpong_pad(pose_images, target_len)
                            log.info(f"WanAnimate: Pose images {pose_images.shape} padded to shape {pose_images_in.shape}")

                        # init variables
                        offloaded = False

                        colormatch = image_embeds.get("colormatch", "disabled")
                        output_path = image_embeds.get("output_path", "")
                        offload = image_embeds.get("force_offload", False)

                        lat_h, lat_w = noise.shape[2], noise.shape[3]
                        start = start_latent = img_counter = step_iteration_count = iteration_count = 0
                        end = frame_window_size
                        end_latent = latent_window_size


                        callback = prepare_callback(patcher, estimated_iterations)
                        log.info(f"Sampling {total_frames} frames in {estimated_iterations} windows, at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")

                        # outer WanAnimate loop
                        gen_video_list = []
                        while True:
                            if start + refert_num >= total_frames:
                                break

                            mm.soft_empty_cache()

                            mask_reft_len = 0 if start == 0 else refert_num

                            self.cache_state = [None, None]

                            noise = torch.randn(16, latent_window_size + 1, lat_h, lat_w, dtype=torch.float32, device=torch.device("cpu"), generator=seed_g).to(device)
                            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

                            if current_ref_images is not None or bg_images is not None or ref_latent is not None:
                                if offload:
                                    offload_transformer(transformer, remove_lora=False)
                                    offloaded = True
                                vae.to(device)
                                if wananim_ref_masks is not None:
                                    msk = ref_masks_in[:, start_latent:end_latent].to(device, dtype)
                                else:
                                    msk = torch.zeros(4, latent_window_size, lat_h, lat_w, device=device, dtype=dtype)
                                if bg_images is not None:
                                    bg_image_slice = bg_images_in[:, start:end].to(device)
                                else:
                                    bg_image_slice = torch.zeros(3, frame_window_size-refert_num, lat_h * 8, lat_w * 8, device=device, dtype=vae.dtype)
                                if mask_reft_len == 0:
                                    temporal_ref_latents = vae.encode([bg_image_slice], device,tiled=tiled_vae)[0]
                                else:
                                    concatenated = torch.cat([current_ref_images.to(device, dtype=vae.dtype), bg_image_slice[:, mask_reft_len:]], dim=1)
                                    temporal_ref_latents = vae.encode([concatenated.to(device, vae.dtype)], device,tiled=tiled_vae, pbar=False)[0]
                                    msk[:, :mask_reft_len] = 1

                                if msk.shape[1] != temporal_ref_latents.shape[1]:
                                    if temporal_ref_latents.shape[1] < msk.shape[1]:
                                        pad_len = msk.shape[1] - temporal_ref_latents.shape[1]
                                        pad_tensor = temporal_ref_latents[:, -1:].repeat(1, pad_len, 1, 1)
                                        temporal_ref_latents = torch.cat([temporal_ref_latents, pad_tensor], dim=1)
                                    else:
                                        temporal_ref_latents = temporal_ref_latents[:, :msk.shape[1]]

                                if ref_latent is not None:
                                    temporal_ref_latents = torch.cat([msk, temporal_ref_latents], dim=0) # 4+C T H W
                                    image_cond_in = torch.cat([ref_latent.to(device), temporal_ref_latents], dim=1) # 4+C T+trefs H W
                                    del temporal_ref_latents, msk, bg_image_slice
                                else:
                                    image_cond_in = torch.cat([torch.tile(torch.zeros_like(noise[:1]), [4, 1, 1, 1]), torch.zeros_like(noise)], dim=0).to(device)
                            else:
                                image_cond_in = torch.cat([torch.tile(torch.zeros_like(noise[:1]), [4, 1, 1, 1]), torch.zeros_like(noise)], dim=0).to(device)

                            pose_input_slice = None
                            if pose_images is not None:
                                vae.to(device)
                                pose_image_slice = pose_images_in[:, start:end].to(device)
                                pose_input_slice = vae.encode([pose_image_slice], device,tiled=tiled_vae, pbar=False).to(dtype)

                            vae.to(offload_device)

                            if wananim_face_pixels is None and wananim_ref_masks is not None:
                                face_images_in = torch.zeros(1, 3, frame_window_size, 512, 512, device=device, dtype=torch.float32)
                            elif wananim_face_pixels is not None:
                                face_images_in = face_images[:, :, start:end].to(device, torch.float32) if face_images is not None else None

                            if samples is not None:
                                input_samples = samples["samples"]
                                if input_samples is not None:
                                    input_samples = input_samples.squeeze(0).to(noise)
                                    # Check if we have enough frames in input_samples
                                    if latent_end_idx > input_samples.shape[1]:
                                        # We need more frames than available - pad the input_samples at the end
                                        pad_length = latent_end_idx - input_samples.shape[1]
                                        last_frame = input_samples[:, -1:].repeat(1, pad_length, 1, 1)
                                        input_samples = torch.cat([input_samples, last_frame], dim=1)
                                    input_samples = input_samples[:, latent_start_idx:latent_end_idx]
                                    if noise_mask is not None:
                                        original_image = input_samples.to(device)

                                    assert input_samples.shape[1] == noise.shape[1], f"Slice mismatch: {input_samples.shape[1]} vs {noise.shape[1]}"

                                    if add_noise_to_samples:
                                        latent_timestep = timesteps[0]
                                        noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples
                                    else:
                                        noise = input_samples

                                # diff diff prep
                                noise_mask = samples.get("noise_mask", None)
                                if noise_mask is not None:
                                    if len(noise_mask.shape) == 4:
                                        noise_mask = noise_mask.squeeze(1)
                                    if noise_mask.shape[0] < noise.shape[1]:
                                        noise_mask = noise_mask.repeat(noise.shape[1] // noise_mask.shape[0], 1, 1)
                                    else:
                                        noise_mask = noise_mask[start_latent:end_latent]
                                    noise_mask = torch.nn.functional.interpolate(
                                        noise_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                                        size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                                        mode='trilinear',
                                        align_corners=False
                                    ).repeat(1, noise.shape[0], 1, 1, 1)

                                    thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
                                    thresholds = thresholds.reshape(-1, 1, 1, 1, 1).to(device)
                                    masks = (1-noise_mask.repeat(len(timesteps), 1, 1, 1, 1).to(device)) > thresholds

                            if isinstance(scheduler, dict):
                                sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
                                timesteps = scheduler["timesteps"]
                            else:
                                sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, total_steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas)

                            # sample videos
                            latent = noise

                            if offloaded:
                                # Load weights
                                if transformer.patched_linear and gguf_reader is None:
                                    load_weights(patcher.model.diffusion_model, patcher.model["sd"], weight_dtype, base_dtype=dtype, transformer_load_device=device, block_swap_args=block_swap_args)
                                elif gguf_reader is not None: #handle GGUF
                                    load_weights(transformer, patcher.model["sd"], base_dtype=dtype, transformer_load_device=device, patcher=patcher, gguf=True, reader=gguf_reader, block_swap_args=block_swap_args)
                                #blockswap init
                                init_blockswap(transformer, block_swap_args, model)

                            # Use the appropriate prompt for this section
                            if len(text_embeds["prompt_embeds"]) > 1:
                                prompt_index = min(iteration_count, len(text_embeds["prompt_embeds"]) - 1)
                                positive = [text_embeds["prompt_embeds"][prompt_index]]
                                log.info(f"Using prompt index: {prompt_index}")
                            else:
                                positive = text_embeds["prompt_embeds"]

                            # uni3c slices
                            uni3c_data_input = None
                            if uni3c_embeds is not None:
                                render_latent = uni3c_embeds["render_latent"][:,:,start_latent:end_latent].to(device)
                                if render_latent.shape[2] < noise.shape[1]:
                                    render_latent = torch.nn.functional.interpolate(render_latent, size=(noise.shape[1], noise.shape[2], noise.shape[3]), mode='trilinear', align_corners=False)
                                uni3c_data_input = {"render_latent": render_latent}
                                for k in uni3c_data:
                                    if k != "render_latent":
                                        uni3c_data_input[k] = uni3c_data[k]

                            mm.soft_empty_cache()
                            gc.collect()
                            # inner WanAnimate sampling loop
                            sampling_pbar = tqdm(total=len(timesteps), desc=f"Frames {start}-{end}", position=0, leave=True)
                            for i in range(len(timesteps)):
                                timestep = timesteps[i]
                                latent_model_input = latent.to(device)

                                noise_pred, _, self.cache_state = predict_with_cfg(
                                    latent_model_input, cfg[min(i, len(timesteps)-1)], positive, text_embeds["negative_prompt_embeds"],
                                    timestep, i, cache_state=self.cache_state, image_cond=image_cond_in, clip_fea=clip_fea, wananim_face_pixels=face_images_in,
                                    wananim_pose_latents=pose_input_slice, uni3c_data=uni3c_data_input,
                                 )
                                if callback is not None:
                                    callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                    callback(step_iteration_count, callback_latent, None, estimated_iterations*(len(timesteps)))
                                    del callback_latent

                                sampling_pbar.update(1)
                                step_iteration_count += 1

                                if use_tsr:
                                    noise_pred = temporal_score_rescaling(noise_pred, latent, timestep, tsr_k, tsr_sigma)

                                latent = sample_scheduler.step(noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0).to(noise_pred.device), **scheduler_step_args)[0].squeeze(0)
                                del noise_pred, latent_model_input, timestep

                                # differential diffusion inpaint
                                if masks is not None:
                                    if i < len(timesteps) - 1:
                                        image_latent = add_noise(original_image.to(device), noise.to(device), timesteps[i+1])
                                        mask = masks[i].to(latent)
                                        latent = image_latent * mask + latent * (1-mask)

                            del noise
                            if offload:
                                offload_transformer(transformer, remove_lora=False)
                                offloaded = True

                            vae.to(device)
                            videos = vae.decode(latent[:, 1:].unsqueeze(0).to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False)[0].cpu()
                            del latent

                            if start != 0:
                                videos = videos[:, refert_num:]

                            sampling_pbar.close()

                            # optional color correction
                            if colormatch != "disabled":
                                videos = videos.permute(1, 2, 3, 0).float().numpy()
                                from color_matcher import ColorMatcher
                                cm = ColorMatcher()
                                cm_result_list = []
                                for img in videos:
                                    cm_result = cm.transfer(src=img, ref=ref_images.permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
                                    cm_result_list.append(torch.from_numpy(cm_result).to(vae.dtype))
                                videos = torch.stack(cm_result_list, dim=0).permute(3, 0, 1, 2)
                                del cm_result_list

                            current_ref_images = videos[:, -refert_num:].clone().detach()

                            # optionally save generated samples to disk
                            if output_path:
                                video_np = videos.clamp(-1.0, 1.0).add(1.0).div(2.0).mul(255).cpu().float().numpy().transpose(1, 2, 3, 0).astype('uint8')
                                num_frames_to_save = video_np.shape[0] if is_first_clip else video_np.shape[0] - cur_motion_frames_num
                                log.info(f"Saving {num_frames_to_save} generated frames to {output_path}")
                                start_idx = 0 if is_first_clip else cur_motion_frames_num
                                for i in range(start_idx, video_np.shape[0]):
                                    im = Image.fromarray(video_np[i])
                                    im.save(os.path.join(output_path, f"frame_{img_counter:05d}.png"))
                                    img_counter += 1
                            else:
                                gen_video_list.append(videos)

                            del videos

                            iteration_count += 1
                            start += frame_window_size - refert_num
                            end += frame_window_size - refert_num
                            start_latent += latent_window_size - ((refert_num - 1)// 4 + 1)
                            end_latent += latent_window_size - ((refert_num - 1)// 4 + 1)

                        if not output_path:
                            gen_video_samples = torch.cat(gen_video_list, dim=1)
                        else:
                            gen_video_samples = torch.zeros(3, 1, 64, 64) # dummy output

                        if force_offload:
                            vae.to(offload_device)
                            if not model["auto_cpu_offload"]:
                                offload_transformer(transformer)
                        try:
                            print_memory(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        except:
                            pass
                        return {"video": gen_video_samples.permute(1, 2, 3, 0), "output_path": output_path},

                    #region normal inference
                    else:
                        noise_pred, noise_pred_ovi, self.cache_state = predict_with_cfg(
                            latent_model_input,
                            cfg[idx], text_embeds["prompt_embeds"], text_embeds["negative_prompt_embeds"],
                            timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                            cache_state=self.cache_state, fantasy_portrait_input=fantasy_portrait_input, multitalk_audio_embeds=multitalk_audio_embeds, mtv_motion_tokens=mtv_motion_tokens, s2v_audio_input=s2v_audio_input,
                            humo_image_cond=humo_image_cond, humo_image_cond_neg=humo_image_cond_neg, humo_audio=humo_audio, humo_audio_neg=humo_audio_neg,
                            wananim_face_pixels=wananim_face_pixels, wananim_pose_latents=wananim_pose_latents, uni3c_data = uni3c_data, latent_model_input_ovi=latent_model_input_ovi, flashvsr_LQ_latent=flashvsr_LQ_latent,
                        )
                        if bidirectional_sampling:
                            noise_pred_flipped, _,self.cache_state = predict_with_cfg(
                            latent_model_input_flipped,
                            cfg[idx], text_embeds["prompt_embeds"], text_embeds["negative_prompt_embeds"],
                            timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                            cache_state=self.cache_state, fantasy_portrait_input=fantasy_portrait_input, mtv_motion_tokens=mtv_motion_tokens,reverse_time=True)

                    if latent_shift_loop:
                        #reverse latent shift
                        if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                            noise_pred = torch.cat([noise_pred[:, latent_video_length - shift_idx:]] + [noise_pred[:, :latent_video_length - shift_idx]], dim=1)
                            shift_idx = (shift_idx + latent_skip) % latent_video_length


                    if flowedit_args is None:
                        latent = latent.to(intermediate_device)

                        if self.noise_front_pad_num > 0:
                            noise_pred = noise_pred[:, self.noise_front_pad_num:]

                        if use_tsr:
                            noise_pred = temporal_score_rescaling(noise_pred, latent, timestep, tsr_k, tsr_sigma)

                        if transformer.is_longcat:
                            noise_pred = -noise_pred

                        if len(timestep.shape) != 1 and clean_latent_indices and not is_pusa: #5b and longcat, skip clean latents for scheduler step
                            step_process_indices = [i for i in range(latent.shape[1]) if i not in clean_latent_indices]
                            latent[:, step_process_indices] = sample_scheduler.step(noise_pred[:, step_process_indices].unsqueeze(0), orig_timestep,
                                                            latent[:, step_process_indices].unsqueeze(0), **scheduler_step_args)[0].squeeze(0)
                        else:
                            if latents_to_not_step > 0:
                                raw_latent = latent[:, :latents_to_not_step]
                                noise_pred_in = noise_pred[:, latents_to_not_step:]
                                latent = latent[:, latents_to_not_step:]
                            elif recammaster is not None or mocha_embeds is not None:
                                noise_pred_in = noise_pred[:, :orig_noise_len]
                                latent = latent[:, :orig_noise_len]
                            else:
                                noise_pred_in = noise_pred
                            latent = sample_scheduler.step(noise_pred_in.unsqueeze(0), timestep, latent.unsqueeze(0), **scheduler_step_args)[0].squeeze(0)
                            if noise_pred_flipped is not None:
                                latent_backwards = sample_scheduler_flipped.step(noise_pred_flipped.unsqueeze(0), timestep, latent_flipped.unsqueeze(0), **scheduler_step_args)[0].squeeze(0)
                                latent_backwards = torch.flip(latent_backwards, dims=[1])
                                latent = latent * 0.5 + latent_backwards * 0.5
                            if latents_to_not_step > 0:
                                latent = torch.cat([raw_latent, latent], dim=1)

                        if latent_ovi is not None:
                            latent_ovi = sample_scheduler_ovi.step(noise_pred_ovi.unsqueeze(0), t, latent_ovi.to(device).unsqueeze(0), **scheduler_step_args)[0].squeeze(0)

                        #InfiniteTalk first frame handling
                        if (extra_latents is not None
                            and not multitalk_sampling
                            and transformer.multitalk_model_type=="InfiniteTalk"):
                            for entry in extra_latents:
                                add_index = entry["index"]
                                num_extra_frames = entry["samples"].shape[2]
                                latent[:, add_index:add_index+num_extra_frames] = entry["samples"].to(latent)

                        # differential diffusion inpaint
                        if masks is not None:
                            if idx < len(timesteps) - 1:
                                noise_timestep = timesteps[idx+1]
                                image_latent = sample_scheduler.scale_noise(
                                    original_image.to(device), torch.tensor([noise_timestep]), noise.to(device)
                                )
                                mask = masks[idx].to(latent)
                                latent = image_latent * mask + latent * (1-mask)

                        # TTM
                        if ttm_reference_latents is not None and (idx + ttm_start_step) < ttm_end_step:
                            if idx + ttm_start_step + 1 < len(sample_scheduler.all_timesteps):
                                noisy_latents = add_noise(ttm_reference_latents, noise, sample_scheduler.all_timesteps[idx + ttm_start_step + 1].to(noise.device)).to(latent)
                                latent = latent * (1 - motion_mask) + noisy_latents * motion_mask
                            else:
                                latent = latent * (1 - motion_mask) + ttm_reference_latents.to(latent) * motion_mask

                        if freeinit_args is not None:
                            current_latent = latent.clone()

                        if callback is not None:
                            if recammaster is not None or mocha_embeds is not None:
                                callback_latent = (latent_model_input[:, :orig_noise_len].to(device) - noise_pred[:, :orig_noise_len].to(device) * t.to(device) / 1000).detach()
                            #elif phantom_latents is not None:
                            #    callback_latent = (latent_model_input[:,:-phantom_latents.shape[1]].to(device) - noise_pred[:,:-phantom_latents.shape[1]].to(device) * t.to(device) / 1000).detach()
                            elif humo_reference_count > 0:
                                callback_latent = (latent_model_input[:,:-humo_reference_count].to(device) - noise_pred[:,:-humo_reference_count].to(device) * t.to(device) / 1000).detach()
                            elif "rcm" in sample_scheduler.__class__.__name__.lower():
                                callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device)).detach()
                            else:
                                callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach()
                            callback(idx, callback_latent.permute(1,0,2,3), None, len(timesteps))
                        else:
                            pbar.update(1)
                    else:
                        if callback is not None:
                            callback_latent = (zt_tgt.to(device) - vt_tgt.to(device) * t.to(device) / 1000).detach()
                            callback(idx, callback_latent.permute(1,0,2,3), None, len(timesteps))
                        else:
                            pbar.update(1)
            except Exception as e:
                log.error(f"Error during sampling: {e}")
                if force_offload:
                    if not model["auto_cpu_offload"]:
                        offload_transformer(transformer)
                raise e

        if phantom_latents is not None:
            latent = latent[:,:-phantom_latents.shape[1]]
        if humo_reference_count > 0:
            latent = latent[:,:-humo_reference_count]
        if longcat_ref_latent is not None:
            latent = latent[:, longcat_ref_latent.shape[1]:]

        cache_states = None
        if cache_args is not None:
            cache_report(transformer, cache_args)
            if end_step != -1 and end_step < total_steps:
                cache_states = {
                    "cache_state": self.cache_state,
                    "easycache_state": transformer.easycache_state,
                    "teacache_state": transformer.teacache_state,
                    "magcache_state": transformer.magcache_state,
                }

        if force_offload:
            if not model["auto_cpu_offload"]:
                offload_transformer(transformer)

        try:
            print_memory(device)
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        return ({
            "samples": latent.unsqueeze(0).cpu(),
            "looped": is_looped,
            "end_image": end_image if not fun_or_fl2v_model else None,
            "has_ref": has_ref,
            "drop_last": drop_last,
            "generator_state": seed_g.get_state(),
            "original_image": original_image.cpu() if original_image is not None else None,
            "cache_states": cache_states,
            "latent_ovi_audio": latent_ovi.unsqueeze(0).transpose(1, 2).cpu() if latent_ovi is not None else None,
            "flashvsr_LQ_images": LQ_images,
        },{
            "samples": callback_latent.unsqueeze(0).cpu() if callback is not None else None,
        })

class WanVideoSamplerSettings(WanVideoSampler):
    RETURN_TYPES = ("SAMPLER_ARGS",)
    RETURN_NAMES = ("sampler_inputs", )
    DESCRIPTION = "Node to output all settings and inputs for the WanVideoSamplerFromSettings -node"
    def process(self, *args, **kwargs):
        import inspect
        params = inspect.signature(WanVideoSampler.process).parameters
        args_dict = {name: kwargs.get(name, param.default if param.default is not inspect.Parameter.empty else None)
                     for name, param in params.items() if name != "self"}
        return args_dict,

class WanVideoSamplerFromSettings(WanVideoSampler):
    DESCRIPTION = "Utility node with no other functionality than to look cleaner, useful for the live preview as the main sampler node has become a messy monster"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_inputs": ("SAMPLER_ARGS",),},
        }

    def process(self, sampler_inputs):
        return super().process(**sampler_inputs)


class WanVideoSamplerExtraArgs():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),
                "feta_args": ("FETAARGS", ),
                "context_options": ("WANVIDCONTEXT", ),
                "cache_args": ("CACHEARGS", ),
                "slg_args": ("SLGARGS", ),
                "rope_function": (rope_functions, {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile. Chunked version has reduced peak VRAM usage when not using torch.compile"}),
                "loop_args": ("LOOPARGS", ),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS", ),
                "uni3c_embeds": ("UNI3C_EMBEDS", ),
                "multitalk_embeds": ("MULTITALK_EMBEDS", ),
            }
        }
    RETURN_TYPES = ("WANVIDSAMPLEREXTRAARGS",)
    RETURN_NAMES = ("extra_args", )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, *args, **kwargs):
        return kwargs,


class WanVideoSamplerv2(WanVideoSampler):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": ("WANVIDEOSCHEDULER",),
            },
            "optional": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "add_noise_to_samples": ("BOOLEAN", {"default": False, "tooltip": "Add noise to the samples before sampling, needed for video2video sampling when starting from clean video"}),
                "extra_args": ("WANVIDSAMPLEREXTRAARGS", ),
            }
        }

    def process(self, *args, extra_args=None, **kwargs):
        import inspect
        params = inspect.signature(WanVideoSampler.process).parameters
        args_dict = {name: kwargs.get(name, param.default if param.default is not inspect.Parameter.empty else None)
                     for name, param in params.items() if name != "self"}

        if extra_args is not None:
            args_dict.update(extra_args)
        else:
            args_dict["rope_function"] = "comfy"

        return super().process(**args_dict)


class WanVideoScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "scheduler": (scheduler_list, {"default": "unipc"}),
                "steps": ("INT", {"default": 30, "min": 1, "tooltip": "Number of steps for the scheduler"}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "start_step": ("INT", {"default": 0, "min": 0, "tooltip": "Starting step for the scheduler"}),
                "end_step": ("INT", {"default": -1, "min": -1, "tooltip": "Ending step for the scheduler"})
            },
            "optional": {
                "sigmas": ("SIGMAS", ),
                "enhance_hf": ("BOOLEAN", {"default": False, "tooltip": "Enhanced high-frequency denoising schedule"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("SIGMAS", "INT", "FLOAT", scheduler_list, "INT", "INT",)
    RETURN_NAMES = ("sigmas", "steps", "shift", "scheduler", "start_step", "end_step")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True

    def process(self, scheduler, steps, start_step, end_step, shift, unique_id, sigmas=None, enhance_hf=False):
        sample_scheduler, timesteps, start_idx, end_idx = get_scheduler(
            scheduler, steps, start_step, end_step, shift, device, sigmas=sigmas, log_timesteps=True, enhance_hf=enhance_hf)

        scheduler_dict = {
            "sample_scheduler": sample_scheduler,
            "timesteps": timesteps,
        }

        try:
            from server import PromptServer
            import io
            import base64
            import matplotlib.pyplot as plt
        except:
            PromptServer = None
        if unique_id and PromptServer is not None:
            try:
                # Plot sigmas and save to a buffer
                sigmas_np = sample_scheduler.full_sigmas.cpu().numpy()
                if not np.isclose(sigmas_np[-1], 0.0, atol=1e-6):
                    sigmas_np = np.append(sigmas_np, 0.0)
                buf = io.BytesIO()
                fig = plt.figure(facecolor='#353535')
                ax = fig.add_subplot(111)
                ax.set_facecolor('#353535')  # Set axes background color
                x_values = range(0, len(sigmas_np))
                ax.plot(x_values, sigmas_np)
                # Annotate each sigma value
                ax.scatter(x_values, sigmas_np, color='white', s=20, zorder=3)  # Small dots at each sigma
                for x, y in zip(x_values, sigmas_np):
                    # Show all annotations if few steps, or just show split step annotations
                    show_annotation = len(sigmas_np) <= 10
                    is_split_step = (start_idx > 0 and x == start_idx) or (end_idx != -1 and x == end_idx + 1)

                    if show_annotation or is_split_step:
                        color = 'orange'
                        if is_split_step:
                            color = 'yellow'
                        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(10, 1), ha='center', color=color, fontsize=12)
                ax.set_xticks(x_values)
                ax.set_title("Sigmas", color='white')           # Title font color
                ax.set_xlabel("Step", color='white')            # X label font color
                ax.set_ylabel("Sigma Value", color='white')     # Y label font color
                ax.tick_params(axis='x', colors='white', labelsize=10)        # X tick color
                ax.tick_params(axis='y', colors='white', labelsize=10)        # Y tick color
                # Add split point if end_step is defined
                end_idx += 1
                if end_idx != -1 and 0 <= end_idx < len(sigmas_np) - 1:
                    ax.axvline(end_idx, color='red', linestyle='--', linewidth=2, label='end_step split')
                # Add split point if start_step is defined
                if start_idx > 0 and 0 <= start_idx < len(sigmas_np):
                    ax.axvline(start_idx, color='green', linestyle='--', linewidth=2, label='start_step split')
                if (end_idx != -1 and 0 <= end_idx < len(sigmas_np)) or (start_idx > 0 and 0 <= start_idx < len(sigmas_np)):
                    handles, labels = ax.get_legend_handles_labels()
                    if labels:
                        ax.legend()
                if start_idx < end_idx and 0 <= start_idx < len(sigmas_np) and 0 < end_idx < len(sigmas_np):
                    ax.axvspan(start_idx, end_idx, color='lightblue', alpha=0.1, label='Sampled Range')
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()

                # Send as HTML img tag with base64 data
                html_img = f"<img src='data:image/png;base64,{img_base64}' alt='Sigmas Plot' style='max-width:100%; height:100%; overflow:hidden; display:block;'>"
                PromptServer.instance.send_progress_text(html_img, unique_id)
            except Exception as e:
                log.error(f"Failed to send sigmas plot: {e}")
                pass

        return (sigmas, steps, shift, scheduler_dict, start_step, end_step)

class WanVideoSchedulerv2(WanVideoScheduler):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "scheduler": (scheduler_list, {"default": "unipc"}),
                "steps": ("INT", {"default": 30, "min": 1, "tooltip": "Number of steps for the scheduler"}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "start_step": ("INT", {"default": 0, "min": 0, "tooltip": "Starting step for the scheduler"}),
                "end_step": ("INT", {"default": -1, "min": -1, "tooltip": "Ending step for the scheduler"})
            },
            "optional": {
                "sigmas": ("SIGMAS", ),
                "enhance_hf": ("BOOLEAN", {"default": False, "tooltip": "Enhanced high-frequency denoising schedule"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("WANVIDEOSCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True

    def process(self, *args, **kwargs):
        sigmas, steps, shift, scheduler_dict, start_step, end_step = super().process(*args, **kwargs)
        return scheduler_dict,

NODE_CLASS_MAPPINGS = {
    "WanVideoSampler": WanVideoSampler,
    "WanVideoSamplerSettings": WanVideoSamplerSettings,
    "WanVideoSamplerFromSettings": WanVideoSamplerFromSettings,
    "WanVideoSamplerv2": WanVideoSamplerv2,
    "WanVideoSamplerExtraArgs": WanVideoSamplerExtraArgs,
    "WanVideoScheduler": WanVideoScheduler,
    "WanVideoSchedulerv2": WanVideoSchedulerv2,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoSampler": "WanVideo Sampler",
    "WanVideoSamplerSettings": "WanVideo Sampler Settings",
    "WanVideoSamplerFromSettings": "WanVideo Sampler From Settings",
    "WanVideoSamplerv2": "WanVideo Sampler v2",
    "WanVideoSamplerExtraArgs": "WanVideoSampler v2 Extra Args",
    "WanVideoScheduler": "WanVideo Scheduler",
    "WanVideoSchedulerv2": "WanVideo Scheduler v2",
}
