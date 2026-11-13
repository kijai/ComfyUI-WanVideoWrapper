import json
import torch
import torchvision.transforms.functional as TF
from ..utils import log
from .trajectory import create_pos_feature_map, draw_tracks_on_video, replace_feature
import os
from comfy import model_management as mm
device = mm.get_torch_device()
script_directory = os.path.dirname(os.path.abspath(__file__))

VAE_STRIDE = (4, 8, 8)  # t, h, w

class WanVideoWanDrawWanMoveTracks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", {"tooltip": "Per-frame video frames to overlay the WanMove tracks onto for visualization"}),
                    "tracks": ("TRACKS", {"tooltip": "WanMove tracks dictionary (track_path + track_visibility) — connect from WanVideoAddWanMoveTracks or compatible producer"}),
                },
                "optional": {
                    "line_resolution": ("INT", {"default": 24, "min": 4, "max": 64, "step": 1, "tooltip": "Number of points to use for each line segment"}),
                    "circle_size": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1, "tooltip": "Size of the circle to draw for each track point"}),
                    "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Opacity of the circle to draw for each track point"}),
                    "line_width": ("INT", {"default": 14, "min": 1, "max": 50, "step": 1, "tooltip": "Pixel width of the path line drawn between consecutive track points across frames"}),
                }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "WanVideoWrapper"

    def execute(self, images, tracks, line_resolution=24, circle_size=10, opacity=0.5, line_width=14):
        if tracks is None or "track_path" not in tracks:
            log.warning("WanVideoWanDrawWanMoveTracks: No tracks provided.")
            return (images.float().cpu(), )
        track = tracks["track_path"].unsqueeze(0)
        track_visibility = tracks["track_visibility"].unsqueeze(0)
        images_in = images * 255.0
        if images_in.shape[0] != track.shape[1]:
            repeat_count = track.shape[1] // images.shape[0]
            images_in = images_in.repeat(repeat_count, 1, 1, 1)
        track_video = draw_tracks_on_video(images_in, track, track_visibility, track_frame=line_resolution, circle_size=circle_size, opacity=opacity, line_width=line_width)
        track_video = torch.stack([TF.to_tensor(frame) for frame in track_video], dim=0).movedim(1, -1)

        return (track_video.float().cpu(), )


class WanVideoAddWanMoveTracks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Base image embeds to extend with WanMove track-position guidance — connect from a WanVideo*Embeds producer"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Multiplier on the WanMove track-position feature map injected into the diffusion conditioning; 0 disables motion control"}),
                },
                "optional": {
                    "track_mask": ("MASK", {"tooltip": "Optional per-frame mask whose nonzero pixels mark visible track points; takes precedence over track visibility from a TRACKS input"}),
                    "track_coords": ("STRING", {"forceInput": True, "tooltip": "JSON string or list of JSON strings representing the tracks"}),
                    "tracks": ("TRACKS", {"tooltip": "Alternatively use Comfy Tracks dictionary"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "TRACKS")
    RETURN_NAMES = ("image_embeds", "tracks")
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, image_embeds, track_coords=None, tracks=None, strength=1.0, track_mask=None):
        updated = dict(image_embeds)

        track_visibility = None

        target_shape = image_embeds.get("target_shape")
        if target_shape is not None:
            height = target_shape[2] * VAE_STRIDE[1]
            width = target_shape[3] * VAE_STRIDE[2]
        else:
            height = image_embeds["lat_h"] * VAE_STRIDE[1]
            width = image_embeds["lat_w"] * VAE_STRIDE[2]
        num_frames = image_embeds["num_frames"]

        if track_coords is not None:
            tracks_data = parse_json_tracks(track_coords)
            track_list = [
                [[track[frame]['x'], track[frame]['y']] for track in tracks_data]
                for frame in range(len(tracks_data[0]))
            ]
            track = torch.tensor(track_list, dtype=torch.float32, device=device)  # shape: (frames, num_tracks, 2)
        elif tracks is not None and "track_path" in tracks:
            track = tracks["track_path"]
            if track_mask is None:
                track_visibility = tracks.get("track_visibility", None)
        track = track[:num_frames]

        num_tracks = track.shape[-2]
        if track_visibility is None:
            if track_mask is None:
                track_visibility = torch.ones((num_frames, num_tracks), dtype=torch.bool, device=device)
            else:
                track_visibility = (track_mask > 0).any(dim=(1, 2)).unsqueeze(-1)
        feature_map, track_pos = create_pos_feature_map(track, track_visibility, VAE_STRIDE, height, width, 16, track_num=num_tracks, device=device)

        updated.setdefault("wanmove_embeds", {})
        updated["wanmove_embeds"]["track_pos"] = track_pos
        updated["wanmove_embeds"]["strength"] = strength

        tracks_dict = {
            "track_path": track,
            "track_visibility": track_visibility,
        }

        return (updated, tracks_dict,)


def parse_json_tracks(tracks):
    tracks_data = []
    try:
        # If tracks is a string, try to parse it as JSON
        if isinstance(tracks, str):
            parsed = json.loads(tracks.replace("'", '"'))
            tracks_data.extend(parsed)
        else:
            # If tracks is a list of strings, parse each one
            for track_str in tracks:
                parsed = json.loads(track_str.replace("'", '"'))
                tracks_data.append(parsed)

        # Check if we have a single track (dict with x,y) or a list of tracks
        if tracks_data and isinstance(tracks_data[0], dict) and 'x' in tracks_data[0]:
            # Single track detected, wrap it in a list
            tracks_data = [tracks_data]
        elif tracks_data and isinstance(tracks_data[0], list) and tracks_data[0] and isinstance(tracks_data[0][0], dict) and 'x' in tracks_data[0][0]:
            # Already a list of tracks, nothing to do
            pass
        else:
            # Unexpected format
            log.warning(f"Warning: Unexpected track format: {type(tracks_data[0])}")

    except json.JSONDecodeError as e:
        log.warning(f"Error parsing tracks JSON: {e}")
        tracks_data = []

    return tracks_data

import node_helpers

class WanMove_native:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("CONDITIONING", {"tooltip": "Positive CONDITIONING containing a concat_latent_image whose feature map will be overridden by the WanMove track positions"}),
            "track_coords": ("STRING", {"forceInput": True, "tooltip": "JSON string or list of JSON strings representing the tracks"}),
            },
            "optional": {
                "track_mask": ("MASK", {"tooltip": "Optional per-frame mask whose nonzero pixels define track visibility; when omitted all tracks are treated as visible"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "TRACKS")
    RETURN_NAMES = ("positive", "tracks")
    FUNCTION = "patchcond"
    CATEGORY = "WanVideoWrapper"
    DEPRECATED = True

    def patchcond(self, positive, track_coords, track_mask=None):

        concat_latent_image = positive[0][1]["concat_latent_image"]
        B, C, T, H, W = concat_latent_image.shape
        num_frames = (T-1) * 4 + 1
        width = W * 8
        height = H * 8

        tracks_data = parse_json_tracks(track_coords)
        track_list = [
            [[track[frame]['x'], track[frame]['y']] for track in tracks_data]
            for frame in range(len(tracks_data[0]))
        ]
        track = torch.tensor(track_list, dtype=torch.float32, device=device)  # shape: (frames, num_tracks, 2)
        track = track[:num_frames]

        num_tracks = track.shape[-2]
        if track_mask is None:
            track_visibility = torch.ones((num_frames, num_tracks), dtype=torch.bool, device=device)
        else:
            track_visibility = (track_mask > 0).any(dim=(1, 2)).unsqueeze(-1)

        feature_map, track_pos = create_pos_feature_map(track, track_visibility, VAE_STRIDE, height, width, 16, track_num=num_tracks, device=device)
        wanmove_cond = replace_feature(concat_latent_image, track_pos.unsqueeze(0))
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": wanmove_cond})

        tracks_dict = {
            "track_path": track,
            "track_visibility": track_visibility,
        }
        return (positive, tracks_dict)


NODE_CLASS_MAPPINGS = {
    "WanVideoAddWanMoveTracks": WanVideoAddWanMoveTracks,
    "WanVideoWanDrawWanMoveTracks": WanVideoWanDrawWanMoveTracks,
    "WanMove_native": WanMove_native,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddWanMoveTracks": "WanVideo Add WanMove Tracks",
    "WanVideoWanDrawWanMoveTracks": "WanVideo Draw WanMove Tracks",
    "WanMove_native": "WanMove Native",
    }
