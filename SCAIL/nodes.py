import torch
import torch.nn.functional as F
import logging
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
# IMPORTANT: keep SCAIL nodes off wrapper utils; utils pulls broad Comfy runtime at import time.
log = logging.getLogger(__name__)

SCAIL2_PAYLOAD_KIND = "wanvideo_scail2_condition_adapter"
SCAIL2_PAYLOAD_SCHEMA_NAME = "scail_pose2.wanvideo_scail2_payload"
SCAIL2_PAYLOAD_VERSION = 1
SCAIL2_EMBEDS_KEY = "scail2_embeds"
SCAIL_V1_EMBEDS_KEY = "scail_embeds"
BACKGROUND_INDEX = -1
SCAIL2_STRENGTH_KEYS = (
    "ref_image",
    "ref_mask",
    "condition_video",
    "driving_mask",
)


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _require_scail2_payload(condition):
    if not isinstance(condition, dict):
        raise ValueError("SCAIL-2 condition must be a SCAIL2_WANVIDEO_PAYLOAD dict")
    if condition.get("kind") != SCAIL2_PAYLOAD_KIND:
        raise ValueError("Unsupported SCAIL-2 payload kind")
    if condition.get("version") != SCAIL2_PAYLOAD_VERSION:
        raise ValueError("Unsupported SCAIL-2 payload version")
    schema = condition.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("SCAIL-2 payload is missing schema metadata")
    if schema.get("name") != SCAIL2_PAYLOAD_SCHEMA_NAME:
        raise ValueError("Unsupported SCAIL-2 payload schema")
    if schema.get("version") != SCAIL2_PAYLOAD_VERSION:
        raise ValueError("Unsupported SCAIL-2 payload schema version")
    native = schema.get("native_wrapper", {})
    if native.get("embeds_key") != SCAIL2_EMBEDS_KEY:
        raise ValueError("SCAIL-2 payload targets an unsupported wrapper embeds key")
    return condition


def _scail2_strength(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    strength = float(value)
    if strength < 0.0 or strength > 10.0:
        raise ValueError(f"{name} must be between 0.0 and 10.0")
    return strength


def _scail2_strengths(
    *,
    ref_image_strength,
    ref_mask_strength,
    condition_video_strength,
    driving_mask_strength,
):
    return {
        "ref_image": _scail2_strength("ref_image_strength", ref_image_strength),
        "ref_mask": _scail2_strength("ref_mask_strength", ref_mask_strength),
        "condition_video": _scail2_strength(
            "condition_video_strength",
            condition_video_strength,
        ),
        "driving_mask": _scail2_strength("driving_mask_strength", driving_mask_strength),
    }


def _latent_frame_count(num_frames):
    return (int(num_frames) - 1) // 4 + 1


def _validate_target_shape(embeds, payload):
    target_shape = embeds.get("target_shape")
    if target_shape is None:
        return
    if len(target_shape) != 4:
        raise ValueError("WANVIDIMAGE_EMBEDS target_shape must be (channels, frames, height, width)")
    dimensions = payload.get("dimensions", {})
    width = int(dimensions["width"])
    height = int(dimensions["height"])
    num_frames = int(dimensions["num_frames"])
    _, latent_frames, latent_h, latent_w = target_shape
    if int(latent_w) * 8 != width or int(latent_h) * 8 != height:
        raise ValueError("SCAIL-2 payload dimensions do not match image_embeds target_shape")
    if int(latent_frames) != _latent_frame_count(num_frames):
        raise ValueError("SCAIL-2 payload frame count does not match image_embeds target_shape")


def _image_to_cthw(image):
    return (image[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device)


def _resize_cthw_spatial(image_cthw, height, width):
    frames = image_cthw.permute(1, 0, 2, 3)
    resized = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
    return resized.permute(1, 0, 2, 3)


def _resize_bhwc_spatial(image, height, width):
    image = image[..., :3]
    if int(image.shape[1]) == int(height) and int(image.shape[2]) == int(width):
        return image
    frames = image.permute(0, 3, 1, 2)
    resized = F.interpolate(frames, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).contiguous()


def _mask_indices_to_reference_alpha(mask_indices, *, height, width, name):
    if mask_indices is None:
        return None
    if hasattr(mask_indices, "detach"):
        indices = mask_indices.detach().to(device)
    else:
        indices = torch.as_tensor(mask_indices, device=device)
    if indices.ndim == 2:
        indices = indices.unsqueeze(0)
    if indices.ndim != 3:
        raise ValueError(f"{name} mask indices must have shape [frames, height, width]")
    if int(indices.shape[0]) <= 0 or int(indices.shape[1]) <= 0 or int(indices.shape[2]) <= 0:
        raise ValueError(f"{name} mask indices must be non-empty")
    alpha = (indices[:1] != BACKGROUND_INDEX).to(dtype=torch.float32).unsqueeze(1)
    if int(alpha.shape[-2]) != int(height) or int(alpha.shape[-1]) != int(width):
        alpha = F.interpolate(alpha, size=(int(height), int(width)), mode="nearest")
    return alpha.permute(0, 2, 3, 1).contiguous()


def _prepare_reference_image(image, mask_indices, *, replace_flag, height, width, name):
    if image is None:
        raise ValueError(f"{name} is required for SCAIL-2 condition embeds")
    resized = _resize_bhwc_spatial(image, height, width)
    if replace_flag and mask_indices is not None:
        alpha = _mask_indices_to_reference_alpha(
            mask_indices,
            height=height,
            width=width,
            name=name,
        )
        alpha = alpha.to(resized.device, resized.dtype)
        if int(alpha.shape[0]) != int(resized.shape[0]):
            alpha = alpha[:1].expand(int(resized.shape[0]), -1, -1, -1)
        resized = resized * alpha
    return resized


def _prepare_condition_video(image, mask_indices, *, replace_flag, height, width):
    if image is None:
        raise ValueError("pose is required for SCAIL-2 condition embeds")
    return _resize_bhwc_spatial(image, height, width)


def _encode_image_batch(vae, image, *, name, spatial_size=None):
    if image is None:
        raise ValueError(f"{name} is required for SCAIL-2 condition embeds")
    image_cthw = _image_to_cthw(image).to(device, vae.dtype)
    if spatial_size is not None:
        image_cthw = _resize_cthw_spatial(image_cthw, spatial_size[0], spatial_size[1])
    latent = vae.encode([image_cthw], device, tiled=False)[0]
    log.info(f"SCAIL-2 {name} latent shape: {latent.shape}")
    return latent


def _runtime_mask_to_scail2_tensor(mask, *, name):
    data = _field(mask, "data")
    if data is None:
        raise ValueError(f"{name} runtime mask is missing data")
    tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.ndim == 5:
        return tensor[0].permute(1, 0, 2, 3).contiguous()
    if tensor.ndim == 4:
        return tensor.contiguous()
    raise ValueError(f"{name} runtime mask must be 4D or 5D")


def _additional_ref_image(additional_ref):
    image = _field(additional_ref, "image")
    if image is None:
        raise ValueError("additional reference is missing image")
    return image


def _additional_ref_mask_indices(additional_ref):
    return _field(additional_ref, "mask_indices")

class WanVideoAddSCAILReferenceEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "vae": ("WANVAE", {"tooltip": "VAE model"}),
                    "ref_image": ("IMAGE",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the reference embedding"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the embedding application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the embedding application"}),
                },
                "optional": {
                    "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, vae, ref_image, strength, start_percent, end_percent, clip_embeds=None):
        updated = dict(embeds)

        vae.to(device)
        ref_image_in = (ref_image[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
        ref_latent = vae.encode([ref_image_in], device, tiled=False)[0]
        log.info(f"SCAIL ref_latent shape: {ref_latent.shape}")

        ref_mask = torch.ones_like(ref_latent[:4])
        ref_latent = torch.cat([ref_latent, ref_mask], dim=0)
        vae.to(offload_device)

        updated.setdefault("scail_embeds", {})
        updated["scail_embeds"]["ref_latent_pos"] = ref_latent * strength
        updated["scail_embeds"]["ref_latent_neg"] = torch.zeros_like(ref_latent)
        updated["scail_embeds"]["ref_start_percent"] = start_percent
        updated["scail_embeds"]["ref_end_percent"] = end_percent
        updated["clip_context"] = clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None

        return (updated,)

class WanVideoAddSCAILPoseEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "vae": ("WANVAE", {"tooltip": "VAE model"}),
                    "pose_images": ("IMAGE", {"tooltip": "Pose images for the entire video"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the pose control"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the pose control application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the pose control application"}),
                },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, vae, pose_images, strength, start_percent=0.0, end_percent=1.0):
        updated = dict(embeds)

        vae.to(device)
        pose_images_in = (pose_images[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
        pose_latent = vae.encode([pose_images_in], device, tiled=False)[0]
        pose_mask = torch.ones_like(pose_latent[:4])
        pose_latent = torch.cat([pose_latent, pose_mask], dim=0)
        log.info(f"SCAIL pose_latent shape: {pose_latent.shape}")

        vae.to(offload_device)

        updated.setdefault("scail_embeds", {})
        updated["scail_embeds"]["pose_latent"] = pose_latent
        updated["scail_embeds"]["pose_strength"] = strength
        updated["scail_embeds"]["pose_start_percent"] = start_percent
        updated["scail_embeds"]["pose_end_percent"] = end_percent

        return (updated,)


class WanVideoAddSCAIL2ConditionEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "condition": ("SCAIL2_WANVIDEO_PAYLOAD",),
                    "vae": ("WANVAE", {"tooltip": "VAE model"}),
                    "ref_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength multiplier for SCAIL-2 reference image latents"}),
                    "ref_mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength multiplier for SCAIL-2 reference mask embeddings"}),
                    "condition_video_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength multiplier for SCAIL-2 condition video latents"}),
                    "driving_mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength multiplier for SCAIL-2 driving mask embeddings"}),
                },
                "optional": {
                    "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(
        self,
        embeds,
        condition,
        vae,
        ref_image_strength=1.0,
        ref_mask_strength=1.0,
        condition_video_strength=1.0,
        driving_mask_strength=1.0,
        clip_embeds=None,
    ):
        payload = _require_scail2_payload(condition)
        updated = dict(embeds)
        if updated.get(SCAIL_V1_EMBEDS_KEY) is not None:
            raise ValueError("SCAIL-2 native embeds cannot be combined with v1 scail_embeds")
        if updated.get(SCAIL2_EMBEDS_KEY) is not None:
            raise ValueError("image_embeds already contains scail2_embeds")
        _validate_target_shape(updated, payload)

        source_condition = payload.get("condition")
        dimensions = payload["dimensions"]
        width = int(dimensions["width"])
        height = int(dimensions["height"])
        replace_flag = bool(payload["replace_flag"])
        clip_context = clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None
        strengths = _scail2_strengths(
            ref_image_strength=ref_image_strength,
            ref_mask_strength=ref_mask_strength,
            condition_video_strength=condition_video_strength,
            driving_mask_strength=driving_mask_strength,
        )

        vae.to(device)
        ref_image = _prepare_reference_image(
            _field(source_condition, "ref_image"),
            _field(source_condition, "ref_mask_indices"),
            replace_flag=replace_flag,
            height=height,
            width=width,
            name="reference",
        )
        ref_latent = _encode_image_batch(
            vae,
            ref_image,
            name="reference",
        )
        pose_latent = _encode_image_batch(
            vae,
            _prepare_condition_video(
                _field(source_condition, "pose_video"),
                _field(source_condition, "driving_mask_indices"),
                replace_flag=replace_flag,
                height=height,
                width=width,
            ),
            name="pose",
            spatial_size=(height // 2, width // 2),
        )
        additional_ref_latents = []
        for index, additional_ref in enumerate(payload.get("additional_references") or ()):
            additional_ref_image = _prepare_reference_image(
                _additional_ref_image(additional_ref),
                _additional_ref_mask_indices(additional_ref),
                replace_flag=replace_flag,
                height=height,
                width=width,
                name=f"additional_reference_{index}",
            )
            additional_ref_latents.append(
                _encode_image_batch(
                    vae,
                    additional_ref_image,
                    name=f"additional_reference_{index}",
                )
            )
        vae.to(offload_device)

        runtime_masks = payload["runtime_masks"]
        additional_ref_masks = [
            _runtime_mask_to_scail2_tensor(mask, name=f"additional_reference_{index}")
            for index, mask in enumerate(runtime_masks.get("additional_references") or ())
        ]
        scail2_embeds = {
            "schema": payload["schema"],
            "mode": payload["mode"],
            "replace_flag": payload["replace_flag"],
            "dimensions": payload["dimensions"],
            "segment": payload.get("segment"),
            "source": payload["source"],
            "strengths": strengths,
            "ref_latents": [ref_latent],
            "ref_masks": [
                _runtime_mask_to_scail2_tensor(runtime_masks["reference"], name="reference")
            ],
            "pose_latents": [pose_latent],
            "driving_masks": [
                _runtime_mask_to_scail2_tensor(runtime_masks["driving"], name="driving")
            ],
            "additional_ref_latents": additional_ref_latents or None,
            "additional_ref_masks": additional_ref_masks or None,
            "clip_context": clip_context,
        }
        if clip_context is not None:
            # IMPORTANT: WanVideoSampler reads CLIP image conditioning from the
            # top-level image_embeds key, not from the nested SCAIL-2 payload.
            updated["clip_context"] = clip_context
        updated[SCAIL2_EMBEDS_KEY] = scail2_embeds
        return (updated,)


NODE_CLASS_MAPPINGS = {
    "WanVideoAddSCAILPoseEmbeds": WanVideoAddSCAILPoseEmbeds,
    "WanVideoAddSCAILReferenceEmbeds": WanVideoAddSCAILReferenceEmbeds,
    "WanVideoAddSCAIL2ConditionEmbeds": WanVideoAddSCAIL2ConditionEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddSCAILReferenceEmbeds": "WanVideo Add SCAIL Reference Embeds",
    "WanVideoAddSCAILPoseEmbeds": "WanVideo Add SCAIL Pose Embeds",
    "WanVideoAddSCAIL2ConditionEmbeds": "WanVideo Add SCAIL-2 Condition Embeds",
    }
