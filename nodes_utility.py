import torch
import torch.nn.functional as F
import numpy as np
from comfy.utils import common_upscale
from comfy import model_management
from tqdm import tqdm
from .utils import log
from einops import rearrange

try:
    from server import PromptServer
except Exception:
    PromptServer = None

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

main_device = model_management.get_torch_device()
offload_device = model_management.unet_offload_device()

class WanVideoImageResizeToClosest:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Image to resize"}),
            "generation_width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "generation_height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "aspect_ratio_preservation": (["keep_input", "stretch_to_new", "crop_to_new"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", )
    RETURN_NAMES = ("image","width","height",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Resizes image to the closest supported resolution based on aspect ratio and max pixels, according to the original code"

    def process(self, image, generation_width, generation_height, aspect_ratio_preservation ):

        H, W = image.shape[1], image.shape[2]
        max_area = generation_width * generation_height

        crop = "disabled"

        if aspect_ratio_preservation == "keep_input":
            aspect_ratio = H / W
        elif aspect_ratio_preservation == "stretch_to_new" or aspect_ratio_preservation == "crop_to_new":
            aspect_ratio = generation_height / generation_width
            if aspect_ratio_preservation == "crop_to_new":
                crop = "center"

        lat_h = round(
        np.sqrt(max_area * aspect_ratio) // VAE_STRIDE[1] //
        PATCH_SIZE[1] * PATCH_SIZE[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // VAE_STRIDE[2] //
            PATCH_SIZE[2] * PATCH_SIZE[2])
        h = lat_h * VAE_STRIDE[1]
        w = lat_w * VAE_STRIDE[2]

        resized_image = common_upscale(image.movedim(-1, 1), w, h, "lanczos", crop).movedim(1, -1)

        return (resized_image, w, h)

class ExtractStartFramesForContinuations:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video_frames": ("IMAGE", {"tooltip": "Input video frames to extract the start frames from."}),
                "num_frames": ("INT", {"default": 10, "min": 1, "max": 1024, "step": 1, "tooltip": "Number of frames to get from the start of the video."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("start_frames",)
    FUNCTION = "get_start_frames"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Extracts the first N frames from a video sequence for continuations."

    def get_start_frames(self, input_video_frames, num_frames):
        if input_video_frames is None or input_video_frames.shape[0] == 0:
            log.warning("Input video frames are empty. Returning an empty tensor.")
            if input_video_frames is not None:
                return (torch.empty((0,) + input_video_frames.shape[1:], dtype=input_video_frames.dtype),)
            else:
                # Return a tensor with 4 dimensions, as expected for an IMAGE type.
                return (torch.empty((0, 64, 64, 3), dtype=torch.float32),)

        total_frames = input_video_frames.shape[0]
        num_to_get = min(num_frames, total_frames)

        if num_to_get < num_frames:
            log.warning(f"Requested {num_frames} frames, but input video only has {total_frames} frames. Returning first {num_to_get} frames.")

        start_frames = input_video_frames[:num_to_get]

        return (start_frames.cpu().float(),)

class WanVideoVACEStartToEndFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "empty_frame_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "White level of empty frame to use"}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "control_images": ("IMAGE",),
                "inpaint_mask": ("MASK", {"tooltip": "Inpaint mask to use for the empty frames"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Index to start from"}),
                "end_index": ("INT", {"default": -1, "min": -10000, "max": 10000, "step": 1, "tooltip": "Index to end at"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to create start/end frame batch and masks for VACE"

    def process(self, num_frames, empty_frame_level, start_image=None, end_image=None, control_images=None, inpaint_mask=None, start_index=0, end_index=-1):

        if start_image is None and end_image is None and control_images is not None:
            if control_images.shape[0] >= num_frames:
                control_images = control_images[:num_frames]
            elif control_images.shape[0] < num_frames:
                # padd with empty_frame_level frames
                padding = torch.ones((num_frames - control_images.shape[0], control_images.shape[1], control_images.shape[2], control_images.shape[3]), device=control_images.device) * empty_frame_level
                control_images = torch.cat([control_images, padding], dim=0)
            return (control_images.cpu().float(), torch.zeros_like(control_images[:, :, :, 0]).cpu().float())
        B, H, W, C = start_image.shape if start_image is not None else end_image.shape
        device = start_image.device if start_image is not None else end_image.device

        # Convert negative end_index to positive
        if end_index < 0:
            end_index = num_frames + end_index

        # Create output batch with empty frames
        out_batch = torch.ones((num_frames, H, W, 3), device=device) * empty_frame_level

        # Create mask tensor with proper dimensions
        masks = torch.ones((num_frames, H, W), device=device)

        # Pre-process all images at once to avoid redundant work
        if end_image is not None and (end_image.shape[1] != H or end_image.shape[2] != W):
            end_image = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)

        if control_images is not None and (control_images.shape[1] != H or control_images.shape[2] != W):
            control_images = common_upscale(control_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)

        # Place start image at start_index
        if start_image is not None:
            frames_to_copy = min(start_image.shape[0], num_frames - start_index)
            if frames_to_copy > 0:
                out_batch[start_index:start_index + frames_to_copy] = start_image[:frames_to_copy]
                masks[start_index:start_index + frames_to_copy] = 0

        # Place end image at end_index
        if end_image is not None:
            # Calculate where to start placing end images
            end_start = end_index - end_image.shape[0] + 1
            if end_start < 0:  # Handle case where end images won't all fit
                end_image = end_image[abs(end_start):]
                end_start = 0

            frames_to_copy = min(end_image.shape[0], num_frames - end_start)
            if frames_to_copy > 0:
                out_batch[end_start:end_start + frames_to_copy] = end_image[:frames_to_copy]
                masks[end_start:end_start + frames_to_copy] = 0

        # Apply control images to remaining frames that don't have start or end images
        if control_images is not None:
            # Create a mask of frames that are still empty (mask == 1)
            empty_frames = masks.sum(dim=(1, 2)) > 0.5 * H * W

            if empty_frames.any():
                # Only apply control images where they exist
                control_length = control_images.shape[0]
                for frame_idx in range(num_frames):
                    if empty_frames[frame_idx] and frame_idx < control_length:
                        out_batch[frame_idx] = control_images[frame_idx]

        # Apply inpaint mask if provided
        if inpaint_mask is not None:
            inpaint_mask = common_upscale(inpaint_mask.unsqueeze(1), W, H, "nearest-exact", "disabled").squeeze(1).to(device)

            # Handle different mask lengths efficiently
            if inpaint_mask.shape[0] > num_frames:
                inpaint_mask = inpaint_mask[:num_frames]
            elif inpaint_mask.shape[0] < num_frames:
                repeat_factor = (num_frames + inpaint_mask.shape[0] - 1) // inpaint_mask.shape[0]  # Ceiling division
                inpaint_mask = inpaint_mask.repeat(repeat_factor, 1, 1)[:num_frames]

            # Apply mask in one operation
            masks = inpaint_mask * masks

        return (out_batch.cpu().float(), masks.cpu().float())


class CreateCFGScheduleFloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 30, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of steps to schedule cfg for"} ),
            "cfg_scale_start": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "cfg_scale_end": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "interpolation": (["linear", "ease_in", "ease_out"], {"default": "linear", "tooltip": "Interpolation method to use for the cfg scale"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "Start percent of the steps to apply cfg"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "End percent of the steps to apply cfg"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float_list",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to generate a list of floats that can be used to schedule cfg scale for the steps, outside the set range cfg is set to 1.0"

    def process(self, steps, cfg_scale_start, cfg_scale_end, interpolation, start_percent, end_percent, unique_id):

        # Create a list of floats for the cfg schedule
        cfg_list = [1.0] * steps
        start_idx = min(int(steps * start_percent), steps - 1)
        end_idx = min(int(steps * end_percent), steps - 1)

        for i in range(start_idx, end_idx + 1):
            if i >= steps:
                break

            if end_idx == start_idx:
                t = 0
            else:
                t = (i - start_idx) / (end_idx - start_idx)

            if interpolation == "linear":
                factor = t
            elif interpolation == "ease_in":
                factor = t * t
            elif interpolation == "ease_out":
                factor = t * (2 - t)

            cfg_list[i] = round(cfg_scale_start + factor * (cfg_scale_end - cfg_scale_start), 2)

        # If start_percent > 0, always include the first step
        if start_percent > 0:
            cfg_list[0] = 1.0

        if unique_id and PromptServer is not None:
            try:
                PromptServer.instance.send_progress_text(
                    f"{cfg_list}",
                    unique_id
                )
            except Exception:
                pass

        return (cfg_list,)

class CreateScheduleFloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 30, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of steps to schedule cfg for"} ),
            "start_value": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "end_value": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "default_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.01, "tooltip": "Default value to use for the steps"}),
            "interpolation": (["linear", "ease_in", "ease_out"], {"default": "linear", "tooltip": "Interpolation method to use for the cfg scale"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "Start percent of the steps to apply cfg"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "End percent of the steps to apply cfg"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float_list",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to generate a list of floats that can be used to schedule things like cfg and lora scale per step"

    def process(self, steps, start_value, end_value, default_value,interpolation, start_percent, end_percent, unique_id):

        # Create a list of floats for the cfg schedule
        cfg_list = [default_value] * steps
        start_idx = min(int(steps * start_percent), steps - 1)
        end_idx = min(int(steps * end_percent), steps - 1)

        for i in range(start_idx, end_idx + 1):
            if i >= steps:
                break

            if end_idx == start_idx:
                t = 0
            else:
                t = (i - start_idx) / (end_idx - start_idx)

            if interpolation == "linear":
                factor = t
            elif interpolation == "ease_in":
                factor = t * t
            elif interpolation == "ease_out":
                factor = t * (2 - t)

            cfg_list[i] = round(start_value + factor * (end_value - start_value), 2)

        # If start_percent > 0, always include the first step
        if start_percent > 0:
            cfg_list[0] = default_value

        if unique_id and PromptServer is not None:
            try:
                PromptServer.instance.send_progress_text(
                    f"{cfg_list}",
                    unique_id
                )
            except Exception:
                pass

        return (cfg_list,)


class DummyComfyWanModelObject:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "shift": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Sigma shift value"}),
            }
        }

    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model",)
    FUNCTION = "create"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to create empty Wan model to use with BasicScheduler -node to get sigmas"

    def create(self, shift):
        from comfy.model_sampling import ModelSamplingDiscreteFlow
        class DummyModel:
            def get_model_object(self, name):
                if name == "model_sampling":
                    model_sampling = ModelSamplingDiscreteFlow()
                    model_sampling.set_parameters(shift=shift)
                    return model_sampling
                return None
        return (DummyModel(),)

class WanVideoLatentReScale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "samples": ("LATENT",),
                    "direction": (["comfy_to_wrapper", "wrapper_to_comfy"], {"tooltip": "Direction to rescale latents, from comfy to wrapper or vice versa"}),
                }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Rescale latents to match the expected range for encoding or decoding between native ComfyUI VAE and the WanVideoWrapper VAE."

    def encode(self, samples, direction):
        samples = samples.copy()
        latents = samples["samples"]

        if latents.shape[1] == 48:
            mean = [
                    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
                    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
                    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
                    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
                    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
                    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
                ]
            std = [
                    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
                    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
                    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
                    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
                    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744
                ]
        else:
            mean = [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ]
            std = [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ]
        mean = torch.tensor(mean).view(1, latents.shape[1], 1, 1, 1)
        std = torch.tensor(std).view(1, latents.shape[1], 1, 1, 1)
        inv_std = (1.0 / std).view(1, latents.shape[1], 1, 1, 1)
        if direction == "comfy_to_wrapper":
            latents = (latents - mean.to(latents)) * inv_std.to(latents)
        elif direction == "wrapper_to_comfy":
            latents = latents / inv_std.to(latents) + mean.to(latents)

        samples["samples"] = latents

        return (samples,)

class WanVideoSigmaToStep:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "sigma": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("step",)
    FUNCTION = "convert"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Simply passes a float value as an integer, used to set start/end steps with sigma threshold"

    def convert(self, sigma):
        return (sigma,)

class NormalizeAudioLoudness:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
            "lufs": ("FLOAT", {"default": -23.0, "min": -100.0, "max": 0.0, "step": 0.1, "tool": "Loudness Units relative to Full Scale, higher LUFS values (closer to 0) mean louder audio. Lower LUFS values (more negative) mean quieter audio."}),
           },
        }

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio", )
    FUNCTION = "normalize"
    CATEGORY = "WanVideoWrapper"

    def normalize(self, audio, lufs):
        audio_input = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if audio_input.dim() == 3:
            audio_input = audio_input.squeeze(0)
        audio_input_np = audio_input.detach().transpose(0, 1).numpy().astype(np.float32)
        audio_input_np = np.ascontiguousarray(audio_input_np)
        normalized_audio = self.loudness_norm(audio_input_np, sr=sample_rate, lufs=lufs)

        out_audio = {"waveform": torch.from_numpy(normalized_audio).transpose(0, 1).unsqueeze(0).float(), "sample_rate": sample_rate}

        return (out_audio, )

    def loudness_norm(self, audio_array, sr=16000, lufs=-23):
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

class WanVideoPassImagesFromSamples:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "samples": ("LATENT",),
                }
                }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "output_path",)
    OUTPUT_TOOLTIPS = ("Decoded images from the samples dictionary", "Output path if provided in the samples dictionary",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Gets possible already decoded images from the samples dictionary, used with Multi/InfiniteTalk sampling"

    def decode(self, samples):
        video = samples.get("video", None)
        video.clamp_(-1.0, 1.0)
        video.add_(1.0).div_(2.0)
        return video.cpu().float(), samples.get("output_path", "")


class FaceMaskFromPoseKeypoints:
    @classmethod
    def INPUT_TYPES(s):
        input_types = {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
                "person_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Index of the person to start with"}),
            }
        }
        return input_types
    RETURN_TYPES = ("MASK",)
    FUNCTION = "createmask"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def createmask(self, pose_kps, person_index):
        pose_frames = pose_kps
        prev_center = None
        np_frames = []
        for i, pose_frame in enumerate(pose_frames):
            selected_idx, prev_center = self.select_closest_person(pose_frame, person_index if i == 0 else prev_center)
            np_frames.append(self.draw_kps(pose_frame, selected_idx))

        if not np_frames:
            # Handle case where no frames were processed
            log.warning("No valid pose frames found, returning empty mask")
            return (torch.zeros((1, 64, 64), dtype=torch.float32),)

        np_frames = np.stack(np_frames, axis=0)
        tensor = torch.from_numpy(np_frames).float() / 255.
        log.info(f"tensor.shape: {tensor.shape}")
        tensor = tensor[:, :, :, 0]
        return (tensor,)

    def select_closest_person(self, pose_frame, prev_center_or_index):
        people = pose_frame["people"]
        if not people:
            return -1, None

        centers = []
        valid_people_indices = []

        for idx, person in enumerate(people):
            # Check if face keypoints exist and are valid
            if "face_keypoints_2d" not in person or not person["face_keypoints_2d"]:
                continue

            kps = np.array(person["face_keypoints_2d"])
            if len(kps) == 0:
                continue

            n = len(kps) // 3
            if n == 0:
                continue

            facial_kps = rearrange(kps, "(n c) -> n c", n=n, c=3)[:, :2]

            # Check if we have valid coordinates (not all zeros)
            if np.all(facial_kps == 0):
                continue

            center = facial_kps.mean(axis=0)

            # Check if center is valid (not NaN or infinite)
            if np.isnan(center).any() or np.isinf(center).any():
                continue

            centers.append(center)
            valid_people_indices.append(idx)

        if not centers:
            return -1, None

        if isinstance(prev_center_or_index, (int, np.integer)):
            # First frame: use person_index, but map to valid people
            if 0 <= prev_center_or_index < len(valid_people_indices):
                idx = valid_people_indices[prev_center_or_index]
                return idx, centers[prev_center_or_index]
            elif valid_people_indices:
                # Fallback to first valid person
                idx = valid_people_indices[0]
                return idx, centers[0]
            else:
                return -1, None
        elif prev_center_or_index is not None:
            # Find closest to previous center
            prev_center = np.array(prev_center_or_index)
            dists = [np.linalg.norm(center - prev_center) for center in centers]
            min_idx = int(np.argmin(dists))
            actual_idx = valid_people_indices[min_idx]
            return actual_idx, centers[min_idx]
        else:
            # prev_center_or_index is None, fallback to first valid person
            if valid_people_indices:
                idx = valid_people_indices[0]
                return idx, centers[0]
            else:
                return -1, None

    def draw_kps(self, pose_frame, person_index):
        import cv2
        width, height = pose_frame["canvas_width"], pose_frame["canvas_height"]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        people = pose_frame["people"]

        if person_index < 0 or person_index >= len(people):
            return canvas  # Out of bounds, return blank

        person = people[person_index]

        # Check if face keypoints exist and are valid
        if "face_keypoints_2d" not in person or not person["face_keypoints_2d"]:
            return canvas  # No face keypoints, return blank

        face_kps_data = person["face_keypoints_2d"]
        if len(face_kps_data) == 0:
            return canvas  # Empty keypoints, return blank

        n = len(face_kps_data) // 3
        if n < 17:  # Need at least 17 points for outer contour
            return canvas  # Not enough keypoints, return blank

        facial_kps = rearrange(np.array(face_kps_data), "(n c) -> n c", n=n, c=3)[:, :2]

        # Check if we have valid coordinates (not all zeros)
        if np.all(facial_kps == 0):
            return canvas  # All keypoints are zero, return blank

        # Check for NaN or infinite values
        if np.isnan(facial_kps).any() or np.isinf(facial_kps).any():
            return canvas  # Invalid coordinates, return blank

        # Check for negative coordinates or coordinates that would create streaks
        if np.any(facial_kps < 0):
            return canvas  # Negative coordinates, likely bad detection

        # Check if coordinates are reasonable (not too close to edges which might indicate bad detection)
        min_margin = 5  # Minimum distance from edges
        if (np.any(facial_kps[:, 0] < min_margin) or
            np.any(facial_kps[:, 1] < min_margin) or
            np.any(facial_kps[:, 0] > width - min_margin) or
            np.any(facial_kps[:, 1] > height - min_margin)):
            # Check if this looks like a streak to corner (many points near 0,0)
            corner_points = np.sum((facial_kps[:, 0] < min_margin) & (facial_kps[:, 1] < min_margin))
            if corner_points > 3:  # Too many points near corner, likely bad detection
                return canvas

        facial_kps = facial_kps.astype(np.int32)

        # Ensure coordinates are within canvas bounds
        facial_kps[:, 0] = np.clip(facial_kps[:, 0], 0, width - 1)
        facial_kps[:, 1] = np.clip(facial_kps[:, 1], 0, height - 1)

        part_color = (255, 255, 255)
        outer_contour = facial_kps[:17]

        # Additional validation for the contour before drawing
        # Check if contour points are too spread out (indicating bad detection)
        if len(outer_contour) >= 3:
            # Calculate bounding box of the contour
            min_x, min_y = np.min(outer_contour, axis=0)
            max_x, max_y = np.max(outer_contour, axis=0)
            contour_width = max_x - min_x
            contour_height = max_y - min_y

            # If contour spans more than 80% of canvas, likely bad detection
            if (contour_width > 0.8 * width or contour_height > 0.8 * height):
                return canvas

            # Check if we have a valid contour (at least 3 unique points)
            unique_points = np.unique(outer_contour, axis=0)
            if len(unique_points) >= 3:
                # Final check: ensure the contour is reasonable
                # Calculate area to see if it's too large or too small
                contour_area = cv2.contourArea(outer_contour)
                canvas_area = width * height

                # If contour is less than 0.1% or more than 50% of canvas, skip
                if 0.001 * canvas_area <= contour_area <= 0.5 * canvas_area:
                    cv2.fillPoly(canvas, pts=[outer_contour], color=part_color)

        return canvas


class DrawGaussianNoiseOnImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "mask": ("MASK", ),
                  },
                  "optional": {
                    "device": (["cpu", "gpu"], {"default": "cpu", "tooltip": "Device to use for processing"}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = "Fills the background (masked area) with Gaussian noise sampled using the mean and variance of the subject (unmasked) region."

    def apply(self, image, mask, device="cpu", seed=0):
        B, H, W, C = image.shape
        BM, HM, WM = mask.shape

        processing_device = main_device if device == "gpu" else torch.device("cpu")

        in_masks = mask.clone().to(processing_device)
        in_images = image.clone().to(processing_device)

        # Resize mask to match image dimensions
        if HM != H or WM != W:
            in_masks = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        # Match batch sizes
        if B > BM:
            in_masks = in_masks.repeat((B + BM - 1) // BM, 1, 1)[:B]
        elif BM > B:
            in_masks = in_masks[:B]

        output_images = []

        # Set random seed for reproducibility
        generator = torch.Generator(device=processing_device).manual_seed(seed)

        for i in tqdm(range(B), desc="DrawGaussianNoiseOnImage batch"):
            curr_mask = in_masks[i]
            img_idx = min(i, B - 1)
            curr_image = in_images[img_idx]

            # Expand mask to 3 channels
            mask_expanded = curr_mask.unsqueeze(-1).expand(-1, -1, 3)

            # Calculate mean and std per channel from the subject region (where mask is 1)
            subject_mask = mask_expanded > 0.5

            # Initialize noise tensor
            noise = torch.zeros_like(curr_image)

            for c in range(C):
                channel = curr_image[:, :, c]
                channel_mask = subject_mask[:, :, c]

                if channel_mask.sum() > 0:
                    # Get subject pixels
                    subject_pixels = channel[channel_mask]

                    # Calculate statistics
                    mean = subject_pixels.mean()
                    std = subject_pixels.std()

                    # Generate Gaussian noise for this channel
                    noise[:, :, c] = torch.normal(mean=mean.item(), std=std.item(),
                                                  size=(H, W), generator=generator,
                                                  device=processing_device)

            # Clamp noise to valid range
            noise = torch.clamp(noise, 0.0, 1.0)

            # Apply: keep subject, fill background with noise
            masked_image = curr_image * mask_expanded + noise * (1 - mask_expanded)
            output_images.append(masked_image)

        # If no masks were processed, return empty tensor
        if not output_images:
            return (torch.zeros((0, H, W, 3), dtype=image.dtype),)

        out_rgb = torch.stack(output_images, dim=0).cpu()

        return (out_rgb, )


class WanVideoPreviewEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                }
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("image_embeds", "mask",)
    FUNCTION = "get"
    CATEGORY = "WanVideoWrapper"

    def get(self, embeds):
        latents = embeds.get("image_embeds", None)
        mask = embeds.get("mask", None)
        if mask is not None:
            mask = mask[0].float().cpu()
        return ({"samples": latents.unsqueeze(0)}, mask)

NODE_CLASS_MAPPINGS = {
    "WanVideoImageResizeToClosest": WanVideoImageResizeToClosest,
    "WanVideoVACEStartToEndFrame": WanVideoVACEStartToEndFrame,
    "ExtractStartFramesForContinuations": ExtractStartFramesForContinuations,
    "CreateCFGScheduleFloatList": CreateCFGScheduleFloatList,
    "DummyComfyWanModelObject": DummyComfyWanModelObject,
    "WanVideoLatentReScale": WanVideoLatentReScale,
    "CreateScheduleFloatList": CreateScheduleFloatList,
    "WanVideoSigmaToStep": WanVideoSigmaToStep,
    "NormalizeAudioLoudness": NormalizeAudioLoudness,
    "WanVideoPassImagesFromSamples": WanVideoPassImagesFromSamples,
    "FaceMaskFromPoseKeypoints": FaceMaskFromPoseKeypoints,
    "DrawGaussianNoiseOnImage": DrawGaussianNoiseOnImage,
    "WanVideoPreviewEmbeds": WanVideoPreviewEmbeds,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoImageResizeToClosest": "WanVideo Image Resize To Closest",
    "WanVideoVACEStartToEndFrame": "WanVideo VACE Start To End Frame",
    "ExtractStartFramesForContinuations": "Extract Start Frames For Continuations",
    "CreateCFGScheduleFloatList": "Create CFG Schedule Float List",
    "DummyComfyWanModelObject": "Dummy Comfy Wan Model Object",
    "WanVideoLatentReScale": "WanVideo Latent ReScale",
    "CreateScheduleFloatList": "Create Schedule Float List",
    "WanVideoSigmaToStep": "WanVideo Sigma To Step",
    "NormalizeAudioLoudness": "Normalize Audio Loudness",
    "WanVideoPassImagesFromSamples": "WanVideo Pass Images From Samples",
    "FaceMaskFromPoseKeypoints": "Face Mask From Pose Keypoints",
    "DrawGaussianNoiseOnImage": "Draw Gaussian Noise On Image",
    "WanVideoPreviewEmbeds": "WanVideo Preview Embeds",
}
