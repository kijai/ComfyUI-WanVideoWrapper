import folder_paths
from comfy import model_management as mm
from comfy.utils import load_torch_file, common_upscale
from accelerate import init_empty_weights
import torch
from ..utils import log, set_module_tensor_to_device
import os
import json
import datetime
import scipy.signal as ss
import numpy as np

script_directory = os.path.dirname(os.path.abspath(__file__))
folder_paths.add_model_folder_path("wav2vec2", os.path.join(folder_paths.models_dir, "wav2vec2"))

class Wav2VecModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("wav2vec2"), {"tooltip": "These models are loaded from the 'ComfyUI/models/wav2vec2' -folder",}),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            },
           
        }
    

    RETURN_TYPES = ("WAV2VECMODEL",)
    RETURN_NAMES = ("wav2vec_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device):
        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
        from ..multitalk.wav2vec2 import Wav2Vec2Model as MultiTalkWav2Vec2Model
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if load_device == "offload_device":
            transfomer_load_device = offload_device
        else:
            transfomer_load_device = device

        config_path = os.path.join(script_directory, "wav2vec2_config.json")
        wav2vec2_config = Wav2Vec2Config(**json.load(open(config_path)))

        with init_empty_weights():
            wav2vec2 = MultiTalkWav2Vec2Model(wav2vec2_config).eval()

        feature_extractor_config = {
            "do_normalize": False,
            "feature_size": 1,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": False,
            "sampling_rate": 16000
        }
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor(**feature_extractor_config)

        model_path = folder_paths.get_full_path_or_raise("wav2vec2", model)
        sd = load_torch_file(model_path, device=transfomer_load_device, safe_load=True)

        for name, param in wav2vec2.named_parameters():
            key = "wav2vec2." + name
            if "original0" in name:
                key = "wav2vec2.encoder.pos_conv_embed.conv.weight_g"
            elif "original1" in name:
                key = "wav2vec2.encoder.pos_conv_embed.conv.weight_v"
            value=sd[key]
            set_module_tensor_to_device(wav2vec2, name, device=offload_device, dtype=base_dtype, value=value)

        wav2vec2_model = {
            "feature_extractor": wav2vec_feature_extractor,
            "model": wav2vec2,
            "dtype": base_dtype,
            "model_type": "tencent",
        }

        return (wav2vec2_model,)
    
class MultiTalkModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
            },
        }

    RETURN_TYPES = ("MULTITALKMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision=None):
        from .multitalk import AudioProjModel
        
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)

        audio_window=5
        intermediate_dim=512
        output_dim=768
        context_tokens=32
        vae_scale=4
        norm_output_audio = True

        with init_empty_weights():
            multitalk_proj_model = AudioProjModel(
                    seq_len=audio_window,
                    seq_len_vf=audio_window+vae_scale-1,
                    intermediate_dim=intermediate_dim,
                    output_dim=output_dim,
                    context_tokens=context_tokens,
                    norm_output_audio=norm_output_audio,
            )

        multitalk = {
            "proj_model": multitalk_proj_model,
            "model_path": model_path,
            "model_type": "InfiniteTalk" if "infinite" in model.lower() else "MultiTalk",
        }

        return (multitalk,)

def loudness_norm(audio_array, sr=16000, lufs=-23):
    try:
        import pyloudnorm
    except Exception:
        raise ImportError("pyloudnorm package is not installed")
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyloudnorm.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def _add_noise_floor(audio, noise_db=-45):
        noise_amp = 10 ** (noise_db / 20)
        noise = np.random.randn(len(audio)) * noise_amp
        return audio + noise

def _smooth_transients(audio, sr=16000):
        b, a = ss.butter(3, 3000 / (sr/2))
        return ss.lfilter(b, a, audio)
    
class MultiTalkWav2VecEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "wav2vec_model": ("WAV2VECMODEL",),
            "audio_1": ("AUDIO",),
            "normalize_loudness": ("BOOLEAN", {"default": True, "tooltip": "Normalize the audio loudness to -23 LUFS"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1, "tooltip": "The total frame count to generate."}),
            "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1}),
            "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the audio conditioning"}),
            "audio_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "When not 1.0, an extra model pass without audio conditioning is done: slower inference but more motion is allowed"}),
            "multi_audio_type": (["para", "add"], {"default": "para", "tooltip": "'para' overlay speakers in parallel, 'add' concatenate sequentially"}),
        },
            "optional" : {
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "ref_target_masks": ("MASK", {"tooltip": "Per-speaker semantic mask(s) in pixel space. Supply one mask per speaker (plus optional background) to guide mouth assignment"}),
                "add_noise_floor": ("BOOLEAN", {"default": False, "tooltip": "Add a low-level noise floor to the audio to reduce silent gaps"}),
                "smooth_transients": ("BOOLEAN", {"default": False, "tooltip": "Apply a low-pass filter to the audio to smooth out transients"}),
            }
        }

    RETURN_TYPES = ("MULTITALK_EMBEDS", "AUDIO", "INT", )
    RETURN_NAMES = ("multitalk_embeds", "audio", "num_frames", )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, wav2vec_model, normalize_loudness, fps, num_frames, audio_1, audio_scale, audio_cfg_scale, multi_audio_type, audio_2=None, audio_3=None, audio_4=None,
                ref_target_masks=None, add_noise_floor=False, smooth_transients=False):
        model_type = wav2vec_model["model_type"]
        if not "tencent" in model_type.lower():
            raise ValueError("Only tencent wav2vec2 models supported by MultiTalk")
        import torchaudio
        import numpy as np
        from einops import rearrange

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = wav2vec_model["dtype"]
        wav2vec2 = wav2vec_model["model"]
        wav2vec2_feature_extractor = wav2vec_model["feature_extractor"]

        sr = 16000

        audio_inputs = [audio_1, audio_2, audio_3, audio_4]
        audio_inputs = [a for a in audio_inputs if a is not None]

        multitalk_audio_features = []
        seq_lengths = []
        audio_outputs = []  # for debugging / optional saving – choose first as return

        for audio in audio_inputs:
            audio_input = audio["waveform"]
            sample_rate = audio["sample_rate"]

            if sample_rate != 16000:
                audio_input = torchaudio.functional.resample(audio_input, sample_rate, sr)
            audio_input = audio_input[0][0]

            start_time = 0
            end_time = num_frames / fps

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            try:
                audio_segment = audio_input[start_sample:end_sample]
            except Exception:
                audio_segment = audio_input

            audio_segment = audio_segment.numpy()

            if normalize_loudness:
                audio_segment = loudness_norm(audio_segment, sr=sr)
            if add_noise_floor:
                audio_segment = _add_noise_floor(audio_segment, noise_db=-45)
            if smooth_transients:
                audio_segment = _smooth_transients(audio_segment, sr=sr)

            audio_feature = np.squeeze(
                wav2vec2_feature_extractor(audio_segment, sampling_rate=sr).input_values
            )

            audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
            audio_feature = audio_feature.unsqueeze(0)

            # audio encoder
            audio_duration = len(audio_segment) / sr
            video_length = audio_duration * fps

            wav2vec2.to(device)
            embeddings = wav2vec2(audio_feature.to(dtype), seq_len=int(video_length), output_hidden_states=True)
            wav2vec2.to(offload_device)

            if len(embeddings) == 0:
                print("Fail to extract audio embedding for one speaker")
                continue

            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

            multitalk_audio_features.append(audio_emb.cpu().detach())
            seq_lengths.append(audio_emb.shape[0])

            waveform_tensor = torch.from_numpy(audio_segment).float().cpu().unsqueeze(0).unsqueeze(0)  # (B, C, N)
            audio_outputs.append({"waveform": waveform_tensor, "sample_rate": sr})

        log.info("[MultiTalk] --- Raw speaker lengths (samples) ---")
        for idx, ao in enumerate(audio_outputs):
            log.info(f"  speaker {idx+1}: {ao['waveform'].shape[-1]} samples (shape: {ao['waveform'].shape})")

        # Pad / combine depending on multi_audio_type
        if len(multitalk_audio_features) > 1:
            if multi_audio_type == "para":
                max_len = max(seq_lengths)
                padded = []
                for emb in multitalk_audio_features:
                    if emb.shape[0] < max_len:
                        pad = torch.zeros(max_len - emb.shape[0], *emb.shape[1:], dtype=emb.dtype)
                        emb = torch.cat([emb, pad], dim=0)
                    padded.append(emb)
                multitalk_audio_features = padded
            elif multi_audio_type == "add":
                total_len = sum(seq_lengths)
                full_list = []
                offset = 0
                for emb, length in zip(multitalk_audio_features, seq_lengths):
                    full = torch.zeros(total_len, *emb.shape[1:], dtype=emb.dtype)
                    full[offset:offset+length] = emb
                    full_list.append(full)
                    offset += length
                multitalk_audio_features = full_list

        # if audio_encoder_output is not None:
        #     all_layers = audio_encoder_output["encoded_audio_all_layers"]
        #     audio_feat = torch.stack(all_layers, dim=0).squeeze(1)[1:]  # shape: [num_layers, T, 512]
        #     audio_feat = audio_feat.movedim(0, 1)
        #     print("audio_feat mean", audio_feat.mean())
        #     print("audio_feat min max", audio_feat.min(), audio_feat.max())
        #     multitalk_audio_features.append(audio_feat.cpu().detach())

        # fallback
        if len(multitalk_audio_features) == 0:
            raise RuntimeError("No valid audio embeddings extracted, please check inputs")

        multitalk_embeds = {
            "audio_features": multitalk_audio_features,
            "audio_scale": audio_scale,
            "audio_cfg_scale": audio_cfg_scale,
            "ref_target_masks": ref_target_masks
        }

        if len(audio_outputs) == 1: # single speaker
            out_audio = audio_outputs[0]
        else: # multi speaker
            if multi_audio_type == "para":
                # Overlay speakers in parallel – mix waveforms to same length (max len)
                max_len = max([a["waveform"].shape[-1] for a in audio_outputs])
                mixed = torch.zeros(1, 1, max_len, dtype=audio_outputs[0]["waveform"].dtype)
                for a in audio_outputs:
                    w = a["waveform"]
                    if w.shape[-1] < max_len:
                        w = torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
                    mixed += w
                out_audio = {"waveform": mixed, "sample_rate": sr}
            else:  # "add" – sequential concatenate with silent padding for other speakers
                total_len = sum([a["waveform"].shape[-1] for a in audio_outputs])
                mixed = torch.zeros(1, 1, total_len, dtype=audio_outputs[0]["waveform"].dtype)
                offset = 0
                for a in audio_outputs:
                    w = a["waveform"]
                    mixed[:, :, offset:offset + w.shape[-1]] += w
                    offset += w.shape[-1]
                out_audio = {"waveform": mixed, "sample_rate": sr}

        # Calculate actual frames based on audio duration
        actual_num_frames = num_frames
        if len(audio_outputs) > 0:
            if multi_audio_type == "para":
                # For parallel mode, use the longest audio duration
                max_audio_duration = max([ao["waveform"].shape[-1] / sr for ao in audio_outputs])
                actual_frames_from_audio = int(max_audio_duration * fps)
            else:  # "add"
                # For sequential mode, use the total audio duration
                total_audio_duration = sum([ao["waveform"].shape[-1] / sr for ao in audio_outputs])
                actual_frames_from_audio = int(total_audio_duration * fps)
            
            # Use the smaller of requested frames or actual audio frames
            actual_num_frames = min(num_frames, actual_frames_from_audio)
            
            if actual_frames_from_audio < num_frames:
                log.info(f"[MultiTalk] Audio duration ({actual_frames_from_audio} frames) is shorter than requested ({num_frames} frames). Using {actual_num_frames} frames.")

        # Debug: log final mixed audio length and mode
        total_samples_raw = sum([ao["waveform"].shape[-1] for ao in audio_outputs])
        log.info(f"[MultiTalk] total raw duration = {total_samples_raw/sr:.3f}s")
        log.info(f"[MultiTalk] multi_audio_type={multi_audio_type} | final waveform shape={out_audio['waveform'].shape} | length={out_audio['waveform'].shape[-1]} samples | seconds={out_audio['waveform'].shape[-1]/sr:.3f}s (expected {'sum' if multi_audio_type=='add' else 'max'} of raw)")

        return (multitalk_embeds, out_audio, actual_num_frames)
    
class MultiTalkSilentEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1, "tooltip": "The total frame count to generate."}),
        },
        }

    RETURN_TYPES = ("MULTITALK_EMBEDS", )
    RETURN_NAMES = ("multitalk_embeds", )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames):
        silence_path = os.path.join(script_directory, "encoded_silence.safetensors")
        encoded_silence = load_torch_file(silence_path)["audio_emb"]
       
        target_frames = num_frames
        repeats = (target_frames + encoded_silence.shape[0] - 1) // encoded_silence.shape[0]
        repeated = encoded_silence.repeat(repeats, 1, 1)
        repeated = repeated[:target_frames]
        multitalk_embeds = {
            "audio_features": repeated,
            "audio_scale": 1.0,
            "audio_cfg_scale": 1.0,
            "ref_target_masks": None
        }
        return (multitalk_embeds,)


class WanVideoImageToVideoMultiTalk:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the generation"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the generation"}),
            "frame_window_size": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "The number of frames to process at once, should be a value the model is generally good at."}),
            "motion_frame": ("INT", {"default": 25, "min": 1, "max": 10000, "step": 1, "tooltip": "Driven frame length used in the long video generation. Basically the overlap length."}),
            "force_offload": ("BOOLEAN", {"default": False, "tooltip": "Whether to force offload the model within the loop for VAE operations, enable if you encounter memory issues."}),
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
            },
            "optional": {
                "start_image": ("IMAGE", {"tooltip": "Images to encode"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "mode": ([
                    "auto",
                    "multitalk",
                    "infinitetalk"
                ], {"default": "auto", "tooltip": "The sampling strategy to use in the long video generation loop, should match the model used"}),
                "output_path": ("STRING", {"default": "", "tooltip": "If set, will save each window's resulting frames to this folder, also DISABLES returning the final video tensor to save memory"}),

            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING",)
    RETURN_NAMES = ("image_embeds", "output_path")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Enables Multi/InfiniteTalk long video generation sampling method, the video is created in windows with overlapping frames. Not compatible or necessary to be used with context windows and many other features besides Multi/InfiniteTalk."

    def process(self, vae, width, height, frame_window_size, motion_frame, force_offload, colormatch, start_image=None, tiled_vae=False, clip_embeds=None, mode="multitalk", output_path=""):

        H = height
        W = width
        VAE_STRIDE = (4, 8, 8)
        
        num_frames = ((frame_window_size - 1) // 4) * 4 + 1

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            resized_start_image = resized_start_image * 2 - 1
            resized_start_image = resized_start_image.unsqueeze(0)
        
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        if output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"{timestamp}_{mode}_output")
            os.makedirs(output_path, exist_ok=True)

        image_embeds = {
            "multitalk_sampling": True,
            "multitalk_start_image": resized_start_image if start_image is not None else None,
            "frame_window_size": num_frames,
            "motion_frame": motion_frame,
            "target_h": H,
            "target_w": W,
            "tiled_vae": tiled_vae,
            "force_offload": force_offload,
            "vae": vae,
            "target_shape": target_shape,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "colormatch": colormatch,
            "multitalk_mode": mode,
            "output_path": output_path
        }

        return (image_embeds, output_path)

class WanVideoImageToVideoSkyreelsv3_audio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the generation"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the generation"}),
            "frame_window_size": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "The number of frames to process at once, should be a value the model is generally good at."}),
            "motion_frame": ("INT", {"default": 5, "min": 1, "max": 10000, "step": 1, "tooltip": "Driven frame length used in the long video generation. Basically the overlap length."}),
            "drop_frames": ("INT", {"default": 12, "min": 0, "max": 10000, "step": 1, "tooltip": "Additional frames to drop when advancing the audio window. Higher values = less overlap = faster generation but potentially less smooth transitions."}),
            "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            "force_offload": ("BOOLEAN", {"default": False, "tooltip": "Whether to force offload the model within the loop for VAE operations, enable if you encounter memory issues."}),
            "colormatch": (
            [
                'disabled',
                'reinhard_torch',
                'mkl',
                'hm',
                'reinhard',
                'mvgd',
                'hm-mvgd-hm',
                'hm-mkl-hm',
            ], {
               "default": 'disabled', "tooltip": "Color matching method to use between the windows"
            },),
            },
            "optional": {
                "start_image": ("IMAGE", {"tooltip": "Images to encode"}),
                "reference_video": ("IMAGE", {"tooltip": "Optional: Pre-generated reference video to use for keyframes instead of extracting from first generation. Should be color-matched to source image."}),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "output_path": ("STRING", {"default": "", "tooltip": "If set, will save each window's resulting frames to this folder, also DISABLES returning the final video tensor to save memory"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING",)
    RETURN_NAMES = ("image_embeds", "output_path")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Enables Multi/InfiniteTalk long video generation sampling method, the video is created in windows with overlapping frames. Not compatible or necessary to be used with context windows and many other features besides Multi/InfiniteTalk."

    def process(self, vae, width, height, frame_window_size, motion_frame, drop_frames, force_offload, colormatch, start_image=None,
                tiled_vae=False, clip_embeds=None, mode="multitalk", output_path="", reference_video=None):

        H, W = height, width
        num_frames = ((frame_window_size - 1) // 4) * 4 + 1

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            resized_start_image = resized_start_image * 2 - 1
            resized_start_image = resized_start_image.unsqueeze(0)

        target_shape = (16, (num_frames - 1) // 4 + 1, height // 8, width // 8)

        if output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"{timestamp}_{mode}_output")
            os.makedirs(output_path, exist_ok=True)

        processed_reference_video = None
        if reference_video is not None:
            processed_reference_video = common_upscale(reference_video.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            processed_reference_video = processed_reference_video * 2 - 1

        image_embeds = {
            "multitalk_sampling": True,
            "multitalk_start_image": resized_start_image if start_image is not None else None,
            "frame_window_size": num_frames,
            "motion_frame": motion_frame,
            "drop_frames": drop_frames,
            "use_pseudo_frames": True,
            "reference_video": processed_reference_video,
            "target_h": H,
            "target_w": W,
            "tiled_vae": tiled_vae,
            "force_offload": force_offload,
            "vae": vae,
            "target_shape": target_shape,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "colormatch": colormatch,
            "multitalk_mode": "skyreelsv3",
            "output_path": output_path
        }

        return (image_embeds, output_path)

NODE_CLASS_MAPPINGS = {
    "MultiTalkModelLoader": MultiTalkModelLoader,
    "MultiTalkWav2VecEmbeds": MultiTalkWav2VecEmbeds,
    "WanVideoImageToVideoMultiTalk": WanVideoImageToVideoMultiTalk,
    "Wav2VecModelLoader": Wav2VecModelLoader,
    "MultiTalkSilentEmbeds": MultiTalkSilentEmbeds,
    "WanVideoImageToVideoSkyreelsv3_audio": WanVideoImageToVideoSkyreelsv3_audio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiTalkModelLoader": "Multi/InfiniteTalk Model Loader",
    "MultiTalkWav2VecEmbeds": "Multi/InfiniteTalk Wav2vec2 Embeds",
    "WanVideoImageToVideoMultiTalk": "WanVideo Long I2V Multi/InfiniteTalk",
    "Wav2VecModelLoader": "Wav2vec2 Model Loader",
    "MultiTalkSilentEmbeds": "MultiTalk Silent Embeds",
    "WanVideoImageToVideoSkyreelsv3_audio": "WanVideo Long SkyReelsV3 A2V",
}