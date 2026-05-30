import os
import torch
from ..utils import log
import numpy as np

import comfy.model_management as mm
from comfy.utils import load_torch_file
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

local_model_path = os.path.join(folder_paths.models_dir, "nlf", "nlf_l_multi_0.3.2.torchscript")
folder_paths.add_model_folder_path("nlf", os.path.join(folder_paths.models_dir, "nlf"))

from .motion4d import SMPL_VQVAE, VectorQuantizer, Encoder, Decoder

def check_jit_script_function():
    if torch.jit.script.__name__ != "script":
        # Get more details about what modified it
        module = torch.jit.script.__module__
        qualname = getattr(torch.jit.script, '__qualname__', 'unknown')
        code_file = None
        try:
            code_file = torch.jit.script.__code__.co_filename
            code_line = torch.jit.script.__code__.co_firstlineno
            log.warning(f"torch.jit.script has been modified by another custom node.\n"
                    f"  Function name: {torch.jit.script.__name__}\n"
                    f"  Module: {module}\n"
                    f"  Qualified name: {qualname}\n"
                    f"  Defined in: {code_file}:{code_line}\n"
                    f"This may cause issues with the NLF model.")
        except Exception:
            log.warning("--------------------------------")
            log.warning(f"torch.jit.script function is: {torch.jit.script.__name__} from module {module}, "
                    f"this has been modified by another custom node. This may cause issues with the NLF model.")
            log.warning("--------------------------------")

model_list = [
    "https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript",
    "https://github.com/isarandi/nlf/releases/download/v0.2.2/nlf_l_multi_0.2.2.torchscript",
]

class DownloadAndLoadNLFModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": (model_list, {"default": "https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript"}),
             },
             "optional": {
                 "warmup": ("BOOLEAN", {"default": True, "tooltip": "Whether to warmup the model after loading"}),
             },
        }

    RETURN_TYPES = ("NLFMODEL",)
    RETURN_NAMES = ("nlf_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, url, warmup=True):
        if url not in model_list:
            raise ValueError(f"URL {url} is not in the list of allowed models.")
        check_jit_script_function()

        if not os.path.exists(local_model_path):
            log.info(f"Downloading NLF model to: {local_model_path}")
            import requests
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            response = requests.get(url)
            if response.status_code == 200:
                with open(local_model_path, "wb") as f:
                    f.write(response.content)
            else:
                print("Failed to download file:", response.status_code)

        model = torch.jit.load(local_model_path).eval()

        if warmup:
            log.info("Warming up NLF model...")
            dummy_input = torch.zeros(1, 3, 256, 256, device=device)
            jit_profiling_prev_state = torch._C._jit_set_profiling_executor(True)
            try:
                for _ in range(2):
                    _ = model.detect_smpl_batched(dummy_input)
            finally:
                torch._C._jit_set_profiling_executor(jit_profiling_prev_state)

            log.info("NLF model warmed up")

        model = model.to(offload_device)

        return (model,)

class LoadNLFModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_model": (folder_paths.get_filename_list("nlf"), {"tooltip": "These models are loaded from the 'ComfyUI/models/nlf' -folder",}),

            },
             "optional": {
                "warmup": ("BOOLEAN", {"default": True, "tooltip": "Whether to warmup the model after loading"}),
             },
        }

    RETURN_TYPES = ("NLFMODEL",)
    RETURN_NAMES = ("nlf_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, nlf_model, warmup=True):
        check_jit_script_function()
        model = torch.jit.load(folder_paths.get_full_path_or_raise("nlf", nlf_model)).eval()

        if warmup:
            log.info("Warming up NLF model...")
            dummy_input = torch.zeros(1, 3, 256, 256, device=device)
            jit_profiling_prev_state = torch._C._jit_set_profiling_executor(True)
            try:
                for _ in range(2):
                    _ = model.detect_smpl_batched(dummy_input)
            finally:
                torch._C._jit_set_profiling_executor(jit_profiling_prev_state)
            log.info("NLF model warmed up")

        model = model.to(offload_device)

        return model,

class LoadVQVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
        }

    RETURN_TYPES = ("VQVAE",)
    RETURN_NAMES = ("vqvae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model_name):
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        # Get motion tokenizer
        motion_encoder = Encoder(
            in_channels=3,
            mid_channels=[128, 512],
            out_channels=3072,
            downsample_time=[2, 2],
            downsample_joint=[1, 1]
        )
        motion_quant = VectorQuantizer(nb_code=8192, code_dim=3072)
        motion_decoder = Decoder(
            in_channels=3072,
            mid_channels=[512, 128],
            out_channels=3,
            upsample_rate=2.0,
            frame_upsample_rate=[2.0, 2.0],
            joint_upsample_rate=[1.0, 1.0]
        )

        vqvae = SMPL_VQVAE(motion_encoder, motion_decoder, motion_quant).to(device)
        vqvae.load_state_dict(vae_sd, strict=True)

        return vqvae,

class MTVCrafterEncodePoses:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vqvae": ("VQVAE", {"tooltip": "VQVAE model"}),
                "poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            },
        }

    RETURN_TYPES = ("MTVCRAFTERMOTION", "NLFPRED")
    RETURN_NAMES = ("mtvcrafter_motion", "pose_results")
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, vqvae, poses):

        global_mean = np.load(os.path.join(script_directory, "data", "mean.npy")) #global_mean.shape: (24, 3)
        global_std = np.load(os.path.join(script_directory, "data", "std.npy"))

        smpl_poses = []
        for pose in poses['joints3d_nonparam'][0]:
            smpl_poses.append(pose[0].cpu().numpy())
        smpl_poses = np.array(smpl_poses)

        norm_poses = torch.tensor((smpl_poses - global_mean) / global_std).unsqueeze(0)
        print(f"norm_poses shape: {norm_poses.shape}, dtype: {norm_poses.dtype}")

        vqvae.to(device)
        motion_tokens, vq_loss = vqvae(norm_poses.to(device), return_vq=True)

        recon_motion = vqvae(norm_poses.to(device))[0][0].to(dtype=torch.float32).cpu().detach() * global_std + global_mean
        vqvae.to(offload_device)

        poses_dict = {
            'mtv_motion_tokens': motion_tokens,
            'global_mean': global_mean,
            'global_std': global_std
        }

        return poses_dict, recon_motion


class NLFPredict:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("NLFMODEL",),
            "images": ("IMAGE", {"tooltip": "Input images for the model"}),
            },
            "optional": {
                "per_batch": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1, "tooltip": "How many images to process at once. -1 means all at once."}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "BBOX",)
    RETURN_NAMES = ("pose_results", "bboxes")
    FUNCTION = "predict"
    CATEGORY = "WanVideoWrapper"

    def predict(self, model, images, per_batch=-1):

        check_jit_script_function()
        model = model.to(device)

        num_images = images.shape[0]

        # Determine batch size
        if per_batch == -1:
            batch_size = num_images
        else:
            batch_size = per_batch

        # Initialize result containers
        all_boxes = []
        all_joints3d_nonparam = []

        # Process in batches
        for i in range(0, num_images, batch_size):
            end_idx = min(i + batch_size, num_images)
            batch_images = images[i:end_idx]

            jit_profiling_prev_state = torch._C._jit_set_profiling_executor(True)
            try:
                pred = model.detect_smpl_batched(batch_images.permute(0, 3, 1, 2).to(device))
            finally:
                torch._C._jit_set_profiling_executor(jit_profiling_prev_state)

            # Collect boxes and joints from this batch
            if 'boxes' in pred:
                all_boxes.extend(pred['boxes'])
            if 'joints3d_nonparam' in pred:
                all_joints3d_nonparam.extend(pred['joints3d_nonparam'])

        model = model.to(offload_device)

        # Move collected results to offload device
        all_boxes = [box.to(offload_device) for box in all_boxes]
        all_joints3d_nonparam = [joints.to(offload_device) for joints in all_joints3d_nonparam]

        # Maintain the original nested format: wrap in a list to match expected structure
        pose_results = {
            'joints3d_nonparam': [all_joints3d_nonparam],
        }

        # Convert bboxes to list format: [x_min, y_min, x_max, y_max] for each detection
        # Each box tensor is shape (1, 5) with [x_min, y_min, x_max, y_max, confidence]
        formatted_boxes = []
        for box in all_boxes:
            # Handle empty detections (no person detected in frame)
            if box.numel() == 0 or box.shape[0] == 0:
                formatted_boxes.append([0.0, 0.0, 0.0, 0.0])
            else:
                # Extract first 4 values (x_min, y_min, x_max, y_max), drop confidence
                bbox_values = box[0, :4].cpu().tolist()
                formatted_boxes.append(bbox_values)

        return (pose_results, formatted_boxes)

class DrawNLFPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            },
            "optional": {
                "stick_width": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "Stick width multiplier"}),
                "point_radius": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "Point radius for drawing the pose"}),
                "style": (["original", "scail"], {"default": "original", "tooltip": "style of the pose drawing"}),
            }
    }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "predict"
    CATEGORY = "WanVideoWrapper"

    def predict(self, poses, width, height, stick_width=1.0, point_radius=2, style="original"):
        from .draw_pose import get_control_conditions

        if isinstance(poses, dict):
            pose_input = poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in poses else poses
        else:
            pose_input = poses

        control_conditions = get_control_conditions(pose_input, height, width, stick_width=stick_width, point_radius=point_radius, style=style)

        return (control_conditions,)

NODE_CLASS_MAPPINGS = {
    "LoadNLFModel": LoadNLFModel,
    "DownloadAndLoadNLFModel": DownloadAndLoadNLFModel,
    "NLFPredict": NLFPredict,
    "DrawNLFPoses": DrawNLFPoses,
    "LoadVQVAE": LoadVQVAE,
    "MTVCrafterEncodePoses": MTVCrafterEncodePoses
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadNLFModel": "Load NLF Model",
    "DownloadAndLoadNLFModel": "(Download)Load NLF Model",
    "NLFPredict": "NLF Predict",
    "DrawNLFPoses": "Draw NLF Poses",
    "LoadVQVAE": "Load VQVAE",
    "MTVCrafterEncodePoses": "MTV Crafter Encode Poses"
}
