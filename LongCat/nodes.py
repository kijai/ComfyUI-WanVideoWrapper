import torch
import torch.nn.functional as F
from ..utils import log
import comfy.model_management as mm
from comfy_api.latest import io

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


class WanVideoLongCatAvatarExtendEmbeds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanVideoLongCatAvatarExtendEmbeds",
            category="WanVideoWrapper",
            inputs=[
                io.Latent.Input("prev_latents", tooltip="Full previous latents to be used to continue generation, continuation frames are selected based on 'overlap' parameter"),
                io.Custom("MULTITALK_EMBEDS").Input("audio_embeds", tooltip="Full length audio embeddings"),
                io.Int.Input("num_frames", default=93, min=1, max=256, step=1, tooltip="Number of new frames to generate"),
                io.Int.Input("overlap", default=13, min=0, max=16, step=1, tooltip="Number of overlapping frames from previous latents for video continuation, set to 0 for T2V"),
                io.Int.Input("frames_processed", default=0, min=0, max=10000, step=1, tooltip="Number of frames already processed in the video, used to select audio features"),
                io.Combo.Input("if_not_enough_audio", ["pad_with_start", "mirror_from_end"], default="pad_with_start", tooltip="What to do if there are not enough frames in pose_images for the window"),
                io.Int.Input("ref_frame_index", default=10, min=0, max=1000, step=1, tooltip="Values between 0 - 24 ensures better consistency, while selecting other ranges (e.g., -10 or 30) helps reduce repeated actions"),
                io.Int.Input("ref_mask_frame_range", default=3, min=0, max=20, step=1, tooltip="Larger range can further help mitigate repeated actions, but excessively large values may introduce artifacts"),
                io.Latent.Input("ref_latent", optional=True, tooltip="Reference latent used for consistency, generally should be either the init image, or first latent from first generation"),
                io.Latent.Input("samples", optional=True, tooltip="For the sampler 'samples' input, used for slicing samples per window for vid2vid"),
                io.Custom("IMAGE").Input("prev_images", optional=True, tooltip="LongCat-Avatar-1.5: decoded frames from the previous segment. When provided together with `vae`, the trailing `overlap` frames are re-encoded through the VAE and used as the overlap conditioning (matches v1.5's use_vcond=False behavior). Leave disconnected for v1.0."),
                io.Custom("WANVAE").Input("vae", optional=True, tooltip="LongCat-Avatar-1.5: VAE used to re-encode `prev_images` for the overlap region. Only used when `prev_images` is also provided."),
            ],
            outputs=[
                io.Custom("WANVIDIMAGE_EMBEDS").Output(display_name="image_embeds", tooltip="Embeds for WanVideo LongCat Avatar generation"),
                io.Latent.Output(display_name="samples_slice", tooltip="Sliced latent samples for the new frames"),
            ],
        )

    @classmethod
    def execute(cls, prev_latents, audio_embeds, num_frames, overlap, if_not_enough_audio, frames_processed, ref_frame_index, ref_mask_frame_range, ref_latent=None, samples=None, prev_images=None, vae=None) -> io.NodeOutput:

        new_audio_embed = audio_embeds.copy()

        audio_features = torch.stack(new_audio_embed["audio_features"])
        num_audio_features = audio_features.shape[1]
        if audio_features.shape[1] < frames_processed + num_frames:
            deficit = frames_processed + num_frames - audio_features.shape[1]
            if if_not_enough_audio == "pad_with_start":
                pad = audio_features[:, :1].repeat(1, deficit, 1, 1)
                audio_features = torch.cat([audio_features, pad], dim=1)
            elif if_not_enough_audio == "mirror_from_end":
                to_add = audio_features[:, -deficit:, :].flip(dims=[1])
                audio_features = torch.cat([audio_features, to_add], dim=1)
            log.warning(f"Not enough audio features, padded with strategy '{if_not_enough_audio}' from {num_audio_features} to {audio_features.shape[1]} frames")

        ref_target_masks = new_audio_embed.get("ref_target_masks", None)
        if ref_target_masks is not None:
            new_audio_embed["ref_target_masks"] = ref_target_masks[:, frames_processed:frames_processed+num_frames, :]

        prev_samples = prev_latents["samples"].clone()
        if overlap != 0:
            latent_overlap = (overlap - 1) // 4 + 1
            if prev_images is not None and vae is not None:
                # LongCat-Avatar-1.5 path: re-encodes instead of just slicing
                img = prev_images[-overlap:]
                if img.shape[-1] == 4:
                    img = img[..., :3]
                img = img.to(vae.dtype).to(device) * 2.0 - 1.0
                img = img.permute(3, 0, 1, 2).unsqueeze(0).contiguous() # [T, H, W, C] -> [B, C, T, H, W]
                vae.to(device)
                prev_samples = vae.encode(img, device=device).to(prev_samples)
                vae.to(offload_device)
                mm.soft_empty_cache()
                log.info(f"Re-encoded {overlap} overlap frames -> latent shape {tuple(prev_samples.shape)}")
            else:
                prev_samples = prev_samples[:, :, -latent_overlap:]

        ref_sample = None
        if ref_latent is not None:
            ref_sample = ref_latent["samples"][0, :, :1].clone()
            log.info(f"Previous latents shape: {prev_samples.shape}, using last {latent_overlap} latent frames for overlap.")

        new_latent_frames = (num_frames - 1) // 4 + 1
        target_shape = (16, new_latent_frames, prev_samples.shape[-2], prev_samples.shape[-1])

        audio_stride = new_audio_embed.get("audio_stride", 2)
        indices = torch.arange(2 * 2 + 1) - 2

        if frames_processed == 0:
            audio_start_idx = 0
        else:
            audio_start_idx = (frames_processed - overlap) * audio_stride
        audio_end_idx = audio_start_idx + num_frames * audio_stride

        log.info(f"Extracting audio embeddings from index {audio_start_idx} to {audio_end_idx}")

        audio_embs = []
        for human_idx in range(len(audio_features)):
            center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=audio_features[human_idx].shape[0] - 1)

            audio_emb = audio_features[human_idx][center_indices].unsqueeze(0).to(device)
            audio_embs.append(audio_emb)
        audio_emb = torch.cat(audio_embs, dim=0)

        new_audio_embed["audio_features"] = None
        new_audio_embed["audio_emb_slice"] = audio_emb

        longcat_avatar_options = {
            "longcat_ref_latent": ref_sample,
            "ref_frame_index": ref_frame_index,
            "ref_mask_frame_range": ref_mask_frame_range,
        }

        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "extra_latents": [{"samples": prev_samples, "index": 0}] if overlap != 0 else None,
            "multitalk_embeds": new_audio_embed,
            "longcat_avatar_options": longcat_avatar_options,
        }

        samples_slice = None
        if samples is not None:
            latent_start_index = (frames_processed - 1) // 4 + 1 if frames_processed > 0 else 0
            latent_end_index = latent_start_index + new_latent_frames
            samples_slice = samples.copy()
            samples_slice["samples"] = samples["samples"][:, :, latent_start_index:latent_end_index].clone()

        return io.NodeOutput(embeds, samples_slice)


class LongCatAvatarWhisperEmbeds:
    """Audio embeds for LongCat-Video-Avatar-1.5 (Whisper-large-v3).

    Produces a MULTITALK_EMBEDS dict whose audio_features are shaped [T, 5, 1280]
    (5 grouped Whisper layers, 1280-d hidden state), matching the audio stream
    the v1.5 AudioProjModel expects. audio_stride is set to 1 to signal v1.5
    timing to the consumer nodes (vs. 2 for the v1.0 wav2vec2 path).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "whisper_model": ("WHISPERMODEL",),
                "audio_1": ("AUDIO",),
                "normalize_loudness": ("BOOLEAN", {"default": True, "tooltip": "Normalize audio loudness to -23 LUFS before encoding (matches the v1.5 reference pipeline)"}),
                "num_frames": ("INT", {"default": 93, "min": 1, "max": 10000, "step": 1, "tooltip": "Total frame count to generate; bounds how much audio is consumed"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1, "tooltip": "Target video fps. LongCat-Video-Avatar-1.5 is trained at 25 fps."}),
                "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the audio conditioning"}),
                "audio_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "When not 1.0, an extra model pass without audio conditioning is done"}),
                "multi_audio_type": (["para", "add"], {"default": "para", "tooltip": "'para' overlays speakers in parallel (equal length); 'add' concatenates speakers sequentially with silence padding"}),
            },
            "optional": {
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "ref_target_masks": ("MASK", {"tooltip": "Per-speaker semantic mask(s) in pixel space, one per speaker"}),
            },
        }

    RETURN_TYPES = ("MULTITALK_EMBEDS", "AUDIO", "INT",)
    RETURN_NAMES = ("multitalk_embeds", "audio", "num_frames",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, whisper_model, audio_1, normalize_loudness, num_frames, fps,
                audio_scale, audio_cfg_scale, multi_audio_type,
                audio_2=None, audio_3=None, audio_4=None, ref_target_masks=None):
        import torchaudio
        import numpy as np
        from ..multitalk.nodes import loudness_norm

        model = whisper_model["model"]
        feature_extractor = whisper_model["feature_extractor"]
        dtype = whisper_model["dtype"]

        sr = 16000
        MEL_CHUNK = 750 * 640  # 480000 samples = 30s at 16kHz; matches Whisper's chunk_length
        ENC_CHUNK = 3000       # encoder window in mel frames
        ENC_FPS = 50           # whisper encoder output frames per second

        def linear_interp(features, output_len):
            features = features.transpose(1, 2)  # [B, D, T]
            out = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
            return out.transpose(1, 2)

        audio_inputs = [a for a in [audio_1, audio_2, audio_3, audio_4] if a is not None]

        audio_features_list = []
        seq_lengths = []
        audio_outputs = []

        end_time = num_frames / float(fps)
        end_sample = int(end_time * sr)

        for audio in audio_inputs:
            audio_input = audio["waveform"]
            sample_rate = audio["sample_rate"]
            if sample_rate != sr:
                audio_input = torchaudio.functional.resample(audio_input, sample_rate, sr)
            audio_input = audio_input[0][0]
            audio_segment = audio_input[:end_sample].cpu().numpy().astype(np.float32)

            if normalize_loudness:
                audio_segment = loudness_norm(audio_segment, sr=sr)

            audio_duration = len(audio_segment) / sr
            video_length = int(audio_duration * fps)
            if video_length < 1:
                continue

            mel_chunks = []
            for i in range(0, len(audio_segment), MEL_CHUNK):
                mel = feature_extractor(audio_segment[i:i + MEL_CHUNK], sampling_rate=sr,
                                        return_tensors="pt").input_features
                mel_chunks.append(mel)
            mel_features = torch.cat(mel_chunks, dim=-1).to(device=device, dtype=dtype)

            model.to(device)
            enc_chunks = []
            with torch.no_grad():
                for i in range(0, mel_features.shape[-1], ENC_CHUNK):
                    chunk = mel_features[:, :, i:i + ENC_CHUNK]
                    chunk_hs = model.encoder(chunk, output_hidden_states=True).hidden_states
                    enc_chunks.append(torch.stack(chunk_hs, dim=2))  # [1, T_enc, n_layers+1, D]
            model.to(offload_device)

            audio_prompts = torch.cat(enc_chunks, dim=1)
            audio_prompts = audio_prompts[:, :video_length * 2]

            feat0 = linear_interp(audio_prompts[:, :, 0:8].mean(dim=2), video_length)
            feat1 = linear_interp(audio_prompts[:, :, 8:16].mean(dim=2), video_length)
            feat2 = linear_interp(audio_prompts[:, :, 16:24].mean(dim=2), video_length)
            feat3 = linear_interp(audio_prompts[:, :, 24:32].mean(dim=2), video_length)
            feat4 = linear_interp(audio_prompts[:, :, 32], video_length)
            audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]  # [T, 5, 1280]

            audio_features_list.append(audio_emb.cpu().detach())
            seq_lengths.append(audio_emb.shape[0])

            waveform_tensor = torch.from_numpy(audio_segment).float().unsqueeze(0).unsqueeze(0)
            audio_outputs.append({"waveform": waveform_tensor, "sample_rate": sr})

        if len(audio_features_list) == 0:
            raise RuntimeError("No valid Whisper audio embeddings extracted, please check inputs")

        if len(audio_features_list) > 1:
            if multi_audio_type == "para":
                max_len = max(seq_lengths)
                padded = []
                for emb in audio_features_list:
                    if emb.shape[0] < max_len:
                        pad = torch.zeros(max_len - emb.shape[0], *emb.shape[1:], dtype=emb.dtype)
                        emb = torch.cat([emb, pad], dim=0)
                    padded.append(emb)
                audio_features_list = padded
            else:  # "add"
                total_len = sum(seq_lengths)
                full_list = []
                offset = 0
                for emb, length in zip(audio_features_list, seq_lengths):
                    full = torch.zeros(total_len, *emb.shape[1:], dtype=emb.dtype)
                    full[offset:offset + length] = emb
                    full_list.append(full)
                    offset += length
                audio_features_list = full_list

        multitalk_embeds = {
            "audio_features": audio_features_list,
            "audio_scale": audio_scale,
            "audio_cfg_scale": audio_cfg_scale,
            "ref_target_masks": ref_target_masks,
            "audio_stride": 1,
            "audio_encoder_type": "whisper",
        }

        if len(audio_outputs) == 1:
            out_audio = audio_outputs[0]
        elif multi_audio_type == "para":
            max_len = max(a["waveform"].shape[-1] for a in audio_outputs)
            mixed = torch.zeros(1, 1, max_len, dtype=audio_outputs[0]["waveform"].dtype)
            for a in audio_outputs:
                w = a["waveform"]
                if w.shape[-1] < max_len:
                    w = F.pad(w, (0, max_len - w.shape[-1]))
                mixed += w
            out_audio = {"waveform": mixed, "sample_rate": sr}
        else:
            total_len = sum(a["waveform"].shape[-1] for a in audio_outputs)
            mixed = torch.zeros(1, 1, total_len, dtype=audio_outputs[0]["waveform"].dtype)
            offset = 0
            for a in audio_outputs:
                w = a["waveform"]
                mixed[:, :, offset:offset + w.shape[-1]] += w
                offset += w.shape[-1]
            out_audio = {"waveform": mixed, "sample_rate": sr}

        return (multitalk_embeds, out_audio, num_frames)


NODE_CLASS_MAPPINGS = {
    "WanVideoLongCatAvatarExtendEmbeds": WanVideoLongCatAvatarExtendEmbeds,
    "LongCatAvatarWhisperEmbeds": LongCatAvatarWhisperEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLongCatAvatarExtendEmbeds": "WanVideo LongCat Avatar Extend Embeds",
    "LongCatAvatarWhisperEmbeds": "LongCat Avatar Whisper Embeds (v1.5)",
    }