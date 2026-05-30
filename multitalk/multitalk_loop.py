import torch
import os
import gc
from PIL import Image
import numpy as np
from ..latent_preview import prepare_callback
from ..wanvideo.schedulers import get_scheduler
from .multitalk import timestep_transform, add_noise
from ..utils import log, print_memory, temporal_score_rescaling, offload_transformer, init_blockswap, match_and_blend_colors
from comfy.utils import load_torch_file
from ..nodes_model_loading import load_weights
from ..HuMo.nodes import get_audio_emb_window
import comfy.model_management as mm
from tqdm import tqdm
import copy

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)
vae_upscale_factor = 8
script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

def multitalk_loop(self, **kwargs):
    # Unpack kwargs into local variables
    (latent, total_steps, steps, start_step, end_step, shift, cfg, denoise_strength,
     sigmas, weight_dtype, transformer, patcher, block_swap_args, model, vae, dtype,
     scheduler, scheduler_step_args, text_embeds, image_embeds, multitalk_embeds,
     multitalk_audio_embeds, unianim_data, dwpose_data, unianimate_poses, uni3c_embeds,
     humo_image_cond, humo_image_cond_neg, humo_audio, humo_reference_count,
     add_noise_to_samples, audio_stride, use_tsr, tsr_k, tsr_sigma, fantasy_portrait_input,
     noise, timesteps, force_offload, add_cond, control_latents, audio_proj,
     control_camera_latents, samples, masks, seed_g, gguf_reader, predict_func
    ) = (kwargs.get(k) for k in (
        'latent', 'total_steps', 'steps', 'start_step', 'end_step', 'shift', 'cfg',
        'denoise_strength', 'sigmas', 'weight_dtype', 'transformer', 'patcher',
        'block_swap_args', 'model', 'vae', 'dtype', 'scheduler', 'scheduler_step_args',
        'text_embeds', 'image_embeds', 'multitalk_embeds', 'multitalk_audio_embeds',
        'unianim_data', 'dwpose_data', 'unianimate_poses', 'uni3c_embeds',
        'humo_image_cond', 'humo_image_cond_neg', 'humo_audio', 'humo_reference_count',
        'add_noise_to_samples', 'audio_stride', 'use_tsr', 'tsr_k', 'tsr_sigma',
        'fantasy_portrait_input', 'noise', 'timesteps', 'force_offload', 'add_cond',
        'control_latents', 'audio_proj', 'control_camera_latents', 'samples', 'masks',
        'seed_g', 'gguf_reader', 'predict_with_cfg'
    ))

    mode = image_embeds.get("multitalk_mode", "multitalk")
    if mode == "auto":
        mode = transformer.multitalk_model_type.lower()
    elif mode == "skyreelsv3":
        num_pseudo_frames = 5
        pseudo_frames = reference_keyframes = None
        keyframe_index = 0
        reference_video = image_embeds.get("reference_video", None)
    log.info(f"Multitalk mode: {mode}")
    drop_frames = image_embeds.get("drop_frames", 0)
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
    original_images = image_embeds.get("multitalk_start_image", None)
    cond_image = original_images.clone() if original_images is not None else None
    original_color_reference = cond_image.clone() if cond_image is not None else None
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

    uni3c_data = None
    if uni3c_embeds is not None:
        transformer.controlnet = uni3c_embeds["controlnet"]
        uni3c_data = uni3c_embeds.copy()

    encoded_silence = None

    try:
        silence_path = os.path.join(script_directory, "encoded_silence.safetensors")
        encoded_silence = load_torch_file(silence_path)["audio_emb"].to(dtype)
    except Exception:
            log.warning("No encoded silence file found, padding with end of audio embedding instead.")

    total_frames = len(audio_embedding[0])
    estimated_iterations = total_frames // (frame_num - motion_frame - drop_frames) + 1
    callback = prepare_callback(patcher, estimated_iterations)

    # If reference_video is provided, extract keyframes from it
    if mode == "skyreelsv3" and reference_video is not None:
        ref_video_length = reference_video.shape[1]  # (C, T, H, W)
        if colormatch == "reinhard_torch":
            reference_video = match_and_blend_colors(reference_video, original_color_reference, 1.0)

        if ref_video_length >= total_frames:
            # Reference is long enough - extract keyframes at the expected positions
            segment_interval = frame_num - motion_frame - drop_frames
            generate_idx = []
            current_idx = frame_num - 1
            while current_idx < total_frames:
                generate_idx.append(min(current_idx, ref_video_length - 1))
                current_idx += segment_interval
        else:
            # Calculate target indices then map to reference video
            audio_length = total_frames
            generate_idx_target = [0]
            segment_interval = frame_num - motion_frame - drop_frames
            current_idx = frame_num - 1
            while current_idx < audio_length - 1:
                generate_idx_target.append(current_idx)
                current_idx += segment_interval
            if generate_idx_target[-1] != audio_length - 1:
                generate_idx_target.append(audio_length - 1)

            # Map target indices to reference video
            generate_idx_target = np.array(generate_idx_target, dtype=np.int16)
            original_max = generate_idx_target[-1]
            original_min = generate_idx_target[0]
            if original_max > original_min:
                generate_idx_float = (generate_idx_target.astype(np.float64) - original_min) * (ref_video_length - 1) / (original_max - original_min)
                generate_idx = np.clip(np.round(generate_idx_float), 0, ref_video_length - 1).astype(np.int32).tolist()
            else:
                generate_idx = [0]

            generate_idx = generate_idx[1:]
            log.info(f"Reference video ({ref_video_length} frames) mapped to target ({total_frames} frames). Keyframe indices: {generate_idx}")

        # Extract keyframes from reference video
        # reference_video shape: (C, T, H, W) from nodes.py processing
        # Select keyframes and add batch dimension: (C, num_keyframes, H, W) -> (1, C, num_keyframes, H, W)
        selected_keyframes = reference_video[:, generate_idx]  # (C, num_keyframes, H, W)
        reference_keyframes = selected_keyframes.unsqueeze(0).cpu()  # (1, C, num_keyframes, H, W)
        log.info(f"Extracted {len(generate_idx)} keyframes from provided reference video at indices {generate_idx}, shape: {reference_keyframes.shape}")
        log.info(f"Reference video total frames: {reference_video.shape[1]}, will generate {total_frames} total frames with {estimated_iterations} windows")

    if frame_num >= total_frames:
        arrive_last_frame = True
        estimated_iterations = 1

    log.info(f"Sampling {total_frames} frames in {estimated_iterations} windows, at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")

    while True: # start video generation iteratively
        self.cache_state = [None, None]

        if mode == "skyreelsv3" and reference_keyframes is not None:
            clamped_index = min(keyframe_index, reference_keyframes.shape[2] - 1) # Clamp keyframe_index to reuse last keyframe if we run out
            pseudo_frames = reference_keyframes[:, :, clamped_index:clamped_index+1].repeat(1, 1, num_pseudo_frames, 1, 1) # Use one keyframe and repeat it 5 times
            log.info(f"Window {iteration_count}: using keyframe {clamped_index}/{reference_keyframes.shape[2]-1} for pseudo frames.")
            keyframe_index += 1
        else:
            pseudo_frames = None

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
            audio_embs = torch.cat(audio_embs, dim=0).to(dtype)

        h, w = (cond_image.shape[-2], cond_image.shape[-1]) if cond_image is not None else (target_h, target_w)
        lat_h, lat_w = h // VAE_STRIDE[1], w // VAE_STRIDE[2]
        latent_frame_num = (frame_num - 1) // 4 + 1

        noise = torch.randn(16, latent_frame_num, lat_h, lat_w, dtype=torch.float32, device=torch.device("cpu"), generator=seed_g).to(device)

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

            # Prepare pseudo frames if enabled and available from reference_video
            if mode == "skyreelsv3" and pseudo_frames is not None:
                video_frames = torch.zeros(1, 3, frame_num-cond_frame_num-num_pseudo_frames, target_h, target_w, device=device, dtype=vae.dtype)
                padding_frames_pixels_values = torch.cat([cond_.to(device, vae.dtype), video_frames, pseudo_frames.to(device, vae.dtype)], dim=2)
            else:
                video_frames = torch.zeros(1, 3, frame_num-cond_frame_num, target_h, target_w, device=device, dtype=vae.dtype)
                padding_frames_pixels_values = torch.cat([cond_.to(device, vae.dtype), video_frames], dim=2)

            # encode
            vae.to(device)
            y = vae.encode(padding_frames_pixels_values, device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]

            if mode == "infinitetalk":
                cond_ = cond_image if is_first_clip else cond_frame
                latent_motion_frames = vae.encode(cond_.to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]
            else:
                latent_motion_frames = y[:, :cur_motion_frames_latent_num] # C T H W

            vae.to(offload_device)

            #motion_frame_index = cur_motion_frames_latent_num if mode == "infinitetalk" else 1
            if mode == "skyreelsv3" and pseudo_frames is not None:
                # create mask in pixel space, then transform
                msk_pixel = torch.ones(1, frame_num, lat_h, lat_w, device=device)
                msk_pixel[:, cur_motion_frames_num : -num_pseudo_frames] = 0
                msk_pixel = torch.cat([
                    torch.repeat_interleave(msk_pixel[:, 0:1], repeats=4, dim=1),
                    msk_pixel[:, 1:],
                ], dim=1)
                msk_pixel = msk_pixel.view(1, msk_pixel.shape[1] // 4, 4, lat_h, lat_w)
                msk = msk_pixel.transpose(1, 2).squeeze(0).to(dtype)  # 4 T H W
            else:
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
                sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, total_steps, start_step, end_step, shift, device, transformer.dim, denoise_strength, sigmas=sigmas)
            timesteps = [torch.tensor([float(t)], device=device) for t in timesteps] + [torch.tensor([0.], device=device)]

        # sample videos
        latent = noise

        # injecting motion frames
        if not is_first_clip and mode != "infinitetalk":
            latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
            motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
            add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
            latent[:, :add_latent.shape[1]] = add_latent
            del motion_add_noise, add_latent

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

            noise_pred, _, self.cache_state = predict_func(
                latent_model_input, cfg[min(i, len(timesteps)-1)], positive, text_embeds["negative_prompt_embeds"],
                timestep, i, y, clip_embeds, control_latents, None, partial_unianim_data, audio_proj, control_camera_latents, add_cond,
                cache_state=self.cache_state, multitalk_audio_embeds=audio_embs, fantasy_portrait_input=partial_fantasy_portrait_input,
                humo_image_cond=partial_humo_cond_input, humo_image_cond_neg=partial_humo_cond_neg_input, humo_audio=partial_humo_audio, humo_audio_neg=partial_humo_audio_neg,
                uni3c_data = uni3c_data)

            if callback is not None:
                callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * timestep.to(device) / 1000).detach().permute(1,0,2,3)
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
            if not is_first_clip and mode != "infinitetalk":
                latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                latent[:, :add_latent.shape[1]] = add_latent
                del motion_add_noise, add_latent
            elif mode == "infinitetalk":
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

        # crop drop_frames from end if enabled
        if mode == "skyreelsv3" and drop_frames > 0 and not arrive_last_frame:
            videos = videos[:, :-drop_frames]

        # optional color correction (less relevant for InfiniteTalk)
        if colormatch != "disabled":
            if colormatch == "reinhard_torch":
                videos = match_and_blend_colors(videos, original_color_reference, 1.0)
            else:
                videos = videos.permute(1, 2, 3, 0).float().numpy()
                from color_matcher import ColorMatcher
                cm = ColorMatcher()
                cm_result_list = []
                for img in videos:
                    if mode == "infinitetalk":
                        cm_result = cm.transfer(src=img, ref=cond_image[0].permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
                    else:
                        cm_result = cm.transfer(src=img, ref=original_images[0].permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
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
            audio_start_idx += (frame_num - cur_motion_frames_num - humo_reference_count - drop_frames)
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
    except Exception:
        pass
    return {"video": gen_video_samples.permute(1, 2, 3, 0), "output_path": output_path},
