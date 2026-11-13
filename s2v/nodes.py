import folder_paths
import math
import torch
import torch.nn.functional as F
import numpy as np

def get_sample_indices(original_fps,
                       total_frames,
                       target_fps,
                       num_sample,
                       fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if not fixed_start is None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    start_time = start_frame / original_fps

    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices

def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    features: shape=[1, T, 512]
    input_fps: fps for audio, f_a
    output_fps: fps for video, f_m
    output_len: video length
    """
    features = features.transpose(1, 2)  # [1, 512, T]
    seq_len = features.shape[2] / float(input_fps)  # T/f_a
    if output_len is None:
        output_len = int(seq_len * output_fps)  # f_m*T/f_a
    output_features = F.interpolate(
        features, size=output_len, align_corners=True,
        mode='linear')  # [1, 512, output_len]
    return output_features.transpose(1, 2)  # [1, output_len, 512]

class WanVideoAddS2VEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Existing Wan image-embeds bundle to extend with S2V audio/pose conditioning — connect from WanVideoImageToVideoEncode / WanVideoEmptyEmbeds / any other embeds producer"}),
                    "frame_window_size": ("INT", {"default": 80, "min": 1, "max": 100000, "step": 1, "tooltip": "Number of frames per S2V audio window; audio embeddings are bucketed to this size and stepped across the generation"}),
                    "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Strength of the audio conditioning applied to the cross-attention; higher = more pronounced audio-driven motion, 1.0 is the trained default"}),
                    "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start of the denoising schedule (0–1) at which the pose_latent conditioning becomes active"}),
                    "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End of the denoising schedule (0–1) after which the pose_latent conditioning is dropped"})
                },
                "optional": {
                    "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT", {"tooltip": "Encoded audio features from a Wan 2.2 S2V audio encoder; carries the multi-layer hidden states used to drive audio-conditioned motion"}),
                    "ref_latent": ("LATENT", {"tooltip": "Optional VAE-encoded reference image latent for identity anchoring; produced by WanVideoEncode or WanVideoEncodeLatentBatch"}),
                    "pose_latent": ("LATENT", {"tooltip": "Optional VAE-encoded pose-control video latent (e.g. DWPose render); gated by pose_start_percent / pose_end_percent during sampling"}),
                    "vae": ("WANVAE", {"tooltip": "Loaded Wan VAE; carried through the embeds bundle so the sampler can decode S2V-internal latents on demand — connect from WanVideoVAELoader"}),
                    "enable_framepack": ("BOOLEAN", {"default": False, "tooltip": "Enable Framepack sampling loop, not compatible with context windows"})
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "INT",)
    RETURN_NAMES = ("image_embeds", "audio_frame_count")
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, frame_window_size, audio_encoder_output=None, audio_scale=1.0, ref_latent=None, pose_latent=None, vae=None, pose_start_percent=0.0, pose_end_percent=1.0, enable_framepack=False):
        audio_frame_count=0
        if audio_encoder_output is not None:
            all_layers = audio_encoder_output["encoded_audio_all_layers"]
            audio_feat = torch.stack(all_layers, dim=0).squeeze(1)  # shape: [num_layers, T, 512]

            print("audio_feat in", audio_feat.shape)
            input_fps = 50 # determined by the model itself
            output_fps = 30 # determined by the model itself
            bucket_fps = 16 # target fps for the generation

            if input_fps != output_fps:
                audio_feat = linear_interpolation(audio_feat, input_fps=input_fps, output_fps=output_fps)
            print("audio_feat after interpolation", audio_feat.shape)

            audio_feat = audio_feat[:, :embeds["num_frames"] * output_fps // bucket_fps, :]
            print("audio_feat after trim", audio_feat.shape)

            self.video_rate = output_fps

            audio_embed_bucket, num_repeat = self.get_audio_embed_bucket_fps(
                audio_feat,
                fps=bucket_fps,
                batch_frames=frame_window_size
            )
            print("audio_embed_bucket", audio_embed_bucket.shape)

            audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
            if len(audio_embed_bucket.shape) == 3:
                audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
            elif len(audio_embed_bucket.shape) == 4:
                audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)

            audio_frame_count = audio_embed_bucket.shape[-1]

            print("audio_embed_bucket", audio_embed_bucket.shape)

        new_entry = {
            "audio_embed_bucket": audio_embed_bucket if audio_encoder_output is not None else None,
            "num_repeat": num_repeat if audio_encoder_output is not None else None,
            "ref_latent": ref_latent["samples"] if ref_latent is not None else None,
            "pose_latent": pose_latent["samples"] if pose_latent is not None else None,
            "audio_scale": audio_scale,
            "vae": vae,
            "pose_start_percent": pose_start_percent,
            "pose_end_percent": pose_end_percent,
            "enable_framepack": enable_framepack,
            "frame_window_size": frame_window_size
        }
        updated = dict(embeds)
        updated["audio_embeds"] = new_entry
        return (updated, audio_frame_count)
    
    def get_audio_embed_bucket_fps(self, audio_embed, fps=16, batch_frames=81, m=0):
        num_layers, audio_frame_num, audio_dim = audio_embed.shape

        if num_layers > 1:
            return_all_layers = True
        else:
            return_all_layers = False

        scale = self.video_rate / fps

        min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1

        bucket_num = min_batch_num * batch_frames
        padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * self.video_rate) - audio_frame_num
        batch_idx = get_sample_indices(
            original_fps=self.video_rate,
            total_frames=audio_frame_num + padd_audio_num,
            target_fps=fps,
            num_sample=bucket_num,
            fixed_start=0)
        batch_audio_eb = []
        audio_sample_stride = int(self.video_rate / fps)
        for bi in batch_idx:
            if bi < audio_frame_num:

                chosen_idx = list(
                    range(bi - m * audio_sample_stride,
                          bi + (m + 1) * audio_sample_stride,
                          audio_sample_stride))
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [
                    audio_frame_num - 1 if c >= audio_frame_num else c
                    for c in chosen_idx
                ]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(
                        start_dim=-2, end_dim=-1)
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                frame_audio_embed = \
                torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device) if not return_all_layers \
                    else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
            batch_audio_eb.append(frame_audio_embed)
        batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb],
                                   dim=0)

        return batch_audio_eb, min_batch_num


    
NODE_CLASS_MAPPINGS = {
    "WanVideoAddS2VEmbeds": WanVideoAddS2VEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddS2VEmbeds": "WanVideo Add S2V Embeds",
}