import os
import torch
import numpy as np
from ..utils import log

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

alignment_model_path = os.path.join(script_directory, "models", "face_landmark.onnx")
det_model_path = os.path.join(script_directory, "models", "face_det.onnx")

from .model import PortraitAdapter
from .pd_fgc.pdf import get_drive_expression_pd_fgc, det_landmarks, FanEncoder
from .pd_fgc.camer import CameraDemo
from .pd_fgc.face_align import FaceAlignment

def load_pd_fgc_model(state_dict, providers):
    face_aligner = CameraDemo(
        face_alignment_module=FaceAlignment(
            providers=providers,
            alignment_model_path=alignment_model_path,
            det_model_path=det_model_path,
        ),
        reset=False,
    )

    pd_fpg_motion = FanEncoder()
    m, u = pd_fpg_motion.load_state_dict(state_dict, strict=False)
    pd_fpg_motion = pd_fpg_motion.eval()

    return face_aligner, pd_fpg_motion


def get_emo_feature(frame_list, face_aligner, pd_fpg_motion, device):
    

    comfy_pbar = ProgressBar(3)
    _, landmark_list, rect_list = det_landmarks(face_aligner, frame_list, comfy_pbar)

    
    # Fill missing landmarks and rects with previous valid one
    last_valid_landmark = None
    last_valid_rect = None
    for i in range(len(landmark_list)):
        if landmark_list[i] is None:
            landmark_list[i] = last_valid_landmark
        else:
            last_valid_landmark = landmark_list[i]
        if rect_list[i] is None:
            rect_list[i] = last_valid_rect
        else:
            last_valid_rect = rect_list[i]

    # Forward fill for leading None values
    if landmark_list[0] is None:
        first_valid = next((l for l in landmark_list if l is not None), None)
        for i in range(len(landmark_list)):
            if landmark_list[i] is None:
                landmark_list[i] = first_valid
            else:
                break
    if rect_list[0] is None:
        first_valid = next((r for r in rect_list if r is not None), None)
        for i in range(len(rect_list)):
            if rect_list[i] is None:
                rect_list[i] = first_valid
            else:
                break

    emo_list = get_drive_expression_pd_fgc(pd_fpg_motion, frame_list, landmark_list, device)
    comfy_pbar.update(1)

    #emo_feat_list = []
    head_emo_feat_list = []
    for emo in emo_list:
        headpose_emb = emo["headpose_emb"]
        eye_embed = emo["eye_embed"]
        emo_embed = emo["emo_embed"]
        mouth_feat = emo["mouth_feat"]

        emo_feat = torch.cat([eye_embed, emo_embed, mouth_feat], dim=1)
        head_emo_feat = torch.cat([headpose_emb, emo_feat], dim=1)

        #emo_feat_list.append(emo_feat)
        head_emo_feat_list.append(head_emo_feat)

    #emo_feat_all = torch.cat(emo_feat_list, dim=0).unsqueeze(0)
    head_emo_feat_all = torch.cat(head_emo_feat_list, dim=0).unsqueeze(0)

    return head_emo_feat_all, rect_list, landmark_list

class FantasyPortraitFaceDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "portrait_model": ("FANTASYPORTRAITMODEL", {"tooltip": "Loaded FantasyPortrait adapter model that turns detected face features into cross-attention projections — connect from FantasyPortrait Model Loader"}),
                "images": ("IMAGE", {"tooltip": "Driving-video frames containing the face to track; each frame is fed to the face detector + landmark model to extract per-frame emotion features"}),
            },
            "optional": {
                "adapter_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Overall strength of the portrait adapter projection (head pose + eyes + emotion + mouth combined); 1.0 is the trained default"}),
                "mouth_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the mouth-feature channel only; raise to exaggerate lip motion, lower to mute it"}),
                "emo_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the eye + emotion-embedding channels; raise to exaggerate expression, lower for a more neutral face"}),
                "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device to run the face detector / landmark ONNX models on; cuda is faster but uses the same GPU as diffusion"}),
            }
        }

    RETURN_TYPES = ("PORTRAIT_EMBEDS", "BBOX", "LANDMARKS")
    RETURN_NAMES = ("portrait_embeds", "bbox", "landmarks")
    FUNCTION = "detect"
    CATEGORY = "WanVideoWrapper"

    def detect(self, images, portrait_model, adapter_scale=1.0, mouth_scale=1.0, emo_scale=1.0, device="cuda"):
        B, H, W, C = images.shape
        num_frames = ((B - 1) // 4) * 4 + 1
        images = images.clone()[:num_frames]
        
        def tensor_batch_to_numpy_list(images):
            images = images.detach().cpu()
            numpy_list = []
            for img in images:
                # img shape: (H, W, C)
                img = img.numpy()
                img = img[..., :3]
                img = (img * 255).clip(0, 255)
                img = img.astype(np.uint8)
                numpy_list.append(img)
            return numpy_list


        numpy_list = tensor_batch_to_numpy_list(images)

        pd_fpg_sd = {}
        for k, v in portrait_model["sd"].items():
            if k.startswith("pd_fpg."):
                pd_fpg_sd[k.replace("pd_fpg.", "")] = v

        if device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        face_aligner, pd_fpg_motion = load_pd_fgc_model(pd_fpg_sd, providers)

        pd_fpg_motion.to(device)
        head_emo_feat_all, rect_list, landmark_list = get_emo_feature(numpy_list, face_aligner, pd_fpg_motion, device=device)
        log.info(f"FantasyPortraitFaceDetector: input frames: {num_frames}")
        log.info(f"FantasyPortraitFaceDetector: features extracted for {head_emo_feat_all.shape[1]} frames")
        pd_fpg_motion.to(offload_device)

        portrait_model = portrait_model["proj_model"]

        portrait_model.to(device)
        adapter_proj = portrait_model.get_adapter_proj(head_emo_feat_all.to(device, dtype=portrait_model.dtype), adapter_scale=adapter_scale, mouth_scale=mouth_scale, emo_scale=emo_scale)
        portrait_model.to(offload_device)

        pos_idx_range = portrait_model.split_audio_adapter_sequence(adapter_proj.size(1), num_frames=num_frames)
        proj_split, context_lens = portrait_model.split_tensor_with_padding(adapter_proj, pos_idx_range, expand_length=0)

        return (proj_split, rect_list, landmark_list)
    
class LandmarksToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "landmarks": ("LANDMARKS", {"default": [], "tooltip": "Per-frame 2D facial landmark coordinates produced by FantasyPortrait Face Detector"}),
            "width": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1, "tooltip": "Output canvas width in pixels (ignored if an input image is wired — landmarks are drawn on top of it)"}),
            "height": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1, "tooltip": "Output canvas height in pixels (ignored if an input image is wired — landmarks are drawn on top of it)"}),
           
            },
            "optional": {
                "image": ("IMAGE",  {"tooltip": "Optional background image(s) to draw the landmarks on top of; when wired, width/height are ignored and the image's resolution is used instead"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("keypoints_image",)
    FUNCTION = "drawkeypoints"
    CATEGORY = "LivePortrait"

    def drawkeypoints(self, landmarks, width=512, height=512, image=None):
        import cv2
        if image is not None:
            image = image.detach().cpu().numpy() * 255
            
        keypoints_img_list = []
        pbar = ProgressBar(len(landmarks))
        for i, lmk in enumerate(landmarks):
            if len(lmk) > 0:
                if image is None:
                    keypoints_image = np.zeros((height, width, 3), dtype=np.uint8) * 255
                else:
                    keypoints_image = image[i].copy()
                for (x, y) in lmk:
                    cv2.circle(keypoints_image, (int(x), int(y)), radius=2, thickness=-1, color=(255,255,255))
            else:
                keypoints_image = np.zeros((height, width, 3), dtype=np.uint8) * 255
            keypoints_img_list.append(keypoints_image)
            pbar.update(1)

        keypoints_img_tensor = (
            torch.stack([torch.from_numpy(np_array) for np_array in keypoints_img_list]) / 255).float()

        return (keypoints_img_tensor,)

class WanVideoAddFantasyPortrait:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Existing Wan image-embeds bundle to extend with portrait conditioning — connect from WanVideoImageToVideoEncode / WanVideoEmptyEmbeds / any other embeds producer"}),
                    "portrait_embeds": ("PORTRAIT_EMBEDS", {"tooltip": "Per-frame portrait projections extracted from a driving video — connect from FantasyPortrait Face Detector"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the portrait conditioning applied to the cross-attention; 1.0 is the trained default, higher = stronger face-driving"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start of the denoising schedule (0–1) at which the portrait embedding becomes active"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End of the denoising schedule (0–1) after which the portrait embedding is dropped"}),
                    "portrait_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01, "tooltip": "When != 1.0, an extra model pass without the portrait embedding is done for portrait-specific CFG; slower but allows fine control over how much the face drives motion"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, portrait_embeds, strength, start_percent=0.0, end_percent=1.0, portrait_cfg=1.0):
        new_entry = {
            "adapter_proj": portrait_embeds,
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "cfg_scale": portrait_cfg,
        }

        updated = dict(embeds)
        updated["portrait_embeds"] = new_entry
        return (updated,)

class FantasyPortraitModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16", "tooltip": "Computation/storage dtype for the portrait adapter weights; fp16 is the safe default, fp32 is most accurate but uses more VRAM"}),
            },
        }

    RETURN_TYPES = ("FANTASYPORTRAITMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        adapter_in_dim = sd["proj_model.norm.weight"].shape[0]

        with init_empty_weights():
            fantasyportrait_proj_adapter = PortraitAdapter(adapter_in_dim=adapter_in_dim, adapter_proj_dim=adapter_in_dim, dtype=base_dtype)

        for name, param in fantasyportrait_proj_adapter.named_parameters():
            set_module_tensor_to_device(fantasyportrait_proj_adapter, name, device=offload_device, dtype=base_dtype, value=sd[name])

        fantasyportrait = {
            "proj_model": fantasyportrait_proj_adapter,
            "sd": sd,
        }

        return (fantasyportrait,)


NODE_CLASS_MAPPINGS = {
    "FantasyPortraitModelLoader": FantasyPortraitModelLoader,
    "FantasyPortraitFaceDetector": FantasyPortraitFaceDetector,
    "WanVideoAddFantasyPortrait": WanVideoAddFantasyPortrait,
    "LandmarksToImage": LandmarksToImage,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "FantasyPortraitModelLoader": "FantasyPortrait Model Loader",
    "FantasyPortraitFaceDetector": "FantasyPortrait Face Detector",
    "WanVideoAddFantasyPortrait": "WanVideo Add Fantasy Portrait",
    "LandmarksToImage": "Landmarks to Image",
    }
