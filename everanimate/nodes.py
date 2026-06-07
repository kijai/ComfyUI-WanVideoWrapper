"""
EverAnimate integration for ComfyUI-WanVideoWrapper.

Implements the input convention from "EverAnimate: Minute-Scale Human Animation
via Latent Flow Restoration" (arXiv:2605.15042), targeting Wan2.2-Animate-14B
with the rank-32 LoRA released by EPFL VITA.

Phase 1 (this file): WanVideoEverAnimateEmbeds — drop-in replacement for
WanVideoAnimateEmbeds in single-window mode. Produces ref_latent with the
[anchor x N, motion x M, zeros] composition expected by the EverAnimate LoRA.

Phase 2 (later): wananimate_loop branch in nodes_sampler.py for auto long
generation with per-clip anchor reselection and latent-space chunk carry.
"""

import torch
import gc
import math
from comfy import model_management as mm
from comfy.utils import common_upscale

from ..utils import log

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


# ─────────────────────────────────────────────────────────────────────────────
# Anchor selection helpers (latent space)
# ─────────────────────────────────────────────────────────────────────────────

def _pad_or_trim_anchor_slots(slices, target_count):
    """Replicate the last slice to fill, or trim if oversupplied.

    Matches everanimate_inference.py::_normalize_anchor_latent_count.
    Inputs: list of [C, 1, h, w] latent tensors.
    """
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(slices) == 0:
        raise ValueError("at least one anchor latent slice required")
    out = list(slices[:target_count])
    while len(out) < target_count:
        out.append(out[-1].clone())
    return out


def _build_clip0_anchor_stack(user_anchor_latent, num_anchors, mode):
    """Build the N-slot anchor stack for clip 0 (no previous chunk available).

    For Phase 1 (single-window) we only support "repeat_user": fill all N
    slots with the user's reference image latent. This matches EverAnimate's
    --use_repeat_anchor inference flag.

    Pose-triangle and random-plus-user modes require multiple source frames,
    which we will introduce in Phase 2 when we have the pose video in scope
    inside the sampler loop.

    user_anchor_latent: [C, 1, h, w]
    returns:            [C, N, h, w]
    """
    if mode != "repeat_user":
        log.warning(
            f"EverAnimate: anchor_select_mode='{mode}' not yet implemented in "
            f"Phase 1; falling back to 'repeat_user' for clip 0."
        )
    return user_anchor_latent.repeat(1, num_anchors, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 node — single-window EverAnimate embeds
# ─────────────────────────────────────────────────────────────────────────────

class WanVideoEverAnimateEmbeds:
    """Embeds builder for Wan2.2-Animate + EverAnimate LoRA, single-window mode.

    Drop-in replacement for WanVideoAnimateEmbeds when num_frames equals
    frame_window_size (no internal long-gen loop). Reconstructs ref_latent
    with the EverAnimate input convention:

        ref_latent = Concat_ch( mask_4ch , y_16ch )

        y_16ch = Concat_t( anchor_latents [C, N, h, w],
                           motion_latent  [C, M, h, w],
                           zero_padding   [C, T - N - M, h, w] )

    where N=num_video_anchor_latents (default 4) and M=num_motion_latents
    (default 1). The 4-channel mask sets 1 on the N anchor slots and 0 on
    the rest; this signals the LoRA that those positions are
    identity/motion memory rather than denoise targets.

    For clip 0 (prev_last_latent and video_anchor_latent both None), motion
    is filled with zeros and all N anchors are filled with the user ref
    image latent (mode='repeat_user'). For continuation clips, the caller
    supplies those latents externally (Phase 2 will provide a state node;
    Phase 1 leaves them optional for early testing).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("WANVAE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "frame_window_size": ("INT", {"default": 77, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "RGB frames per sampler window. Default 77 matches EverAnimate's "
                               "trained clip length (frames_per_clip=77 → 20 content latents). "
                               "Set num_frames > frame_window_size to trigger auto long-generation "
                               "(sampler loops with N-slot anchor + M-slot motion + latent carry)."}),
                "colormatch": (
                    ["disabled", "mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"],
                    {"default": "disabled"},
                ),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "num_video_anchor_latents": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "EverAnimate N — number of identity-memory anchor latents prepended to y. "
                               "Paper uses 4; LoRA was trained with this value."}),
                "num_motion_latents": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1,
                    "tooltip": "EverAnimate M — number of motion-memory slots after the anchors. "
                               "Holds prev chunk's last latent at continuation; zeros at clip 0."}),
                "anchor_select_mode": (["repeat_user", "pose_triangle", "random_plus_user"],
                    {"default": "repeat_user",
                     "tooltip": "Phase 1 only implements 'repeat_user'. Other modes activate in Phase 2 "
                                "when previous-chunk source frames are available."}),
            },
            "optional": {
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS",),
                "ref_images": ("IMAGE",),
                "pose_images": ("IMAGE",),
                "face_images": ("IMAGE",),
                "bg_images": ("IMAGE",),
                "mask": ("MASK",),
                "tiled_vae": ("BOOLEAN", {"default": False}),
                "prev_last_latent": ("LATENT", {
                    "tooltip": "Previous chunk's last M latents [C, M, h, w]. Leave empty for clip 0."}),
                "video_anchor_latent": ("LATENT", {
                    "tooltip": "Pre-selected N anchor latents [C, N, h, w] from a previous chunk. "
                               "Leave empty for clip 0 to use 'repeat_user' on ref_images."}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/EverAnimate"

    def process(self, vae, width, height, num_frames, force_offload, frame_window_size,
                colormatch, pose_strength, face_strength,
                num_video_anchor_latents, num_motion_latents, anchor_select_mode,
                ref_images=None, pose_images=None, face_images=None, clip_embeds=None,
                tiled_vae=False, bg_images=None, mask=None,
                prev_last_latent=None, video_anchor_latent=None):

        # ── Geometry ────────────────────────────────────────────────────────
        W = (width // 16) * 16
        H = (height // 16) * 16
        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        N = num_video_anchor_latents
        M = num_motion_latents

        num_frames = ((num_frames - 1) // 4) * 4 + 1

        if num_frames < frame_window_size:
            frame_window_size = num_frames

        looping = num_frames > frame_window_size

        # In Kijai's WanVideoAnimateEmbeds num_refs is the count of ref_image
        # latents prepended; in EverAnimate convention it's the number of
        # anchor slots N. Single-window extends num_frames by N*4 to host the
        # anchor slots; looping mode lets the sampler handle that per-iter.
        target_shape = (16, (num_frames - 1) // 4 + 1 + N, lat_h, lat_w)
        if looping:
            extended_num_frames = num_frames
            latent_window_size = (frame_window_size - 1) // 4 + 1
        else:
            extended_num_frames = num_frames + N * 4
            latent_window_size = (frame_window_size - 1) // 4

        # ── VAE prep ────────────────────────────────────────────────────────
        mm.soft_empty_cache()
        gc.collect()
        vae.to(device)

        # ── Pose handling ──────────────────────────────────────────────────
        # Single-window: pre-encode the full clip into pose_latents.
        # Looping: store resized RGB tensor; sampler slices+encodes per window.
        pose_latents = None
        resized_pose_images = None
        if pose_images is not None:
            pose_images = pose_images[..., :3]
            if pose_images.shape[1] != H or pose_images.shape[2] != W:
                resized_pose_images = common_upscale(
                    pose_images.movedim(-1, 1), W, H, "lanczos", "disabled"
                ).movedim(0, 1)
            else:
                resized_pose_images = pose_images.permute(3, 0, 1, 2)
            resized_pose_images = resized_pose_images * 2 - 1
            if not looping:
                pose_latents = vae.encode(
                    [resized_pose_images.to(device, vae.dtype)], device, tiled=tiled_vae
                )
                pose_latents = pose_latents.to(offload_device)
                if pose_latents.shape[2] < latent_window_size:
                    pad_len = latent_window_size - pose_latents.shape[2]
                    pad = torch.zeros(
                        pose_latents.shape[0], pose_latents.shape[1], pad_len,
                        pose_latents.shape[3], pose_latents.shape[4],
                        device=pose_latents.device, dtype=pose_latents.dtype,
                    )
                    pose_latents = torch.cat([pose_latents, pad], dim=2)
                del resized_pose_images
                resized_pose_images = None
            else:
                resized_pose_images = resized_pose_images.to(offload_device, dtype=vae.dtype)

        # ── BG handling ─────────────────────────────────────────────────────
        # Single-window: pre-encode bg (or zeros) for the full window.
        # Looping: store RGB tensor (or None); sampler builds zeros per-iter.
        bg_latents = None
        resized_bg_images = None
        if bg_images is not None:
            if bg_images.shape[1] != H or bg_images.shape[2] != W:
                resized_bg_images = common_upscale(
                    bg_images.movedim(-1, 1), W, H, "lanczos", "disabled"
                ).movedim(0, 1)
            else:
                resized_bg_images = bg_images.permute(3, 0, 1, 2)
            resized_bg_images = (resized_bg_images[:3] * 2 - 1)

        if not looping:
            if bg_images is None:
                resized_bg_images = torch.zeros(
                    3, extended_num_frames - N, H, W, device=device, dtype=vae.dtype
                )
            bg_latents = vae.encode(
                [resized_bg_images.to(device, vae.dtype)], device, tiled=tiled_vae
            )[0].to(offload_device)
            del resized_bg_images
            resized_bg_images = None
        elif bg_images is not None:
            resized_bg_images = resized_bg_images.to(offload_device, dtype=vae.dtype)

        # ── User ref image → single-frame latent ────────────────────────────
        if ref_images is None:
            raise ValueError("EverAnimate: ref_images is required (the identity anchor)")

        if ref_images.shape[1] != H or ref_images.shape[2] != W:
            resized_ref_images = common_upscale(
                ref_images.movedim(-1, 1), W, H, "lanczos", "disabled"
            ).movedim(0, 1)
        else:
            resized_ref_images = ref_images.permute(3, 0, 1, 2)
        resized_ref_images = resized_ref_images[:3] * 2 - 1

        user_ref_latent = vae.encode(
            [resized_ref_images.to(device, vae.dtype)], device, tiled=tiled_vae
        )[0]  # [C=16, num_refs_in, h, w]

        # Take only the first temporal slice to make it [C, 1, h, w].
        user_anchor = user_ref_latent[:, :1]

        # ── Build the N anchor slots ────────────────────────────────────────
        if video_anchor_latent is not None:
            anchor_stack = video_anchor_latent["samples"] if isinstance(video_anchor_latent, dict) else video_anchor_latent
            if anchor_stack.dim() == 5:
                anchor_stack = anchor_stack.squeeze(0)
            if anchor_stack.shape[1] != N:
                log.warning(
                    f"EverAnimate: provided video_anchor_latent has T={anchor_stack.shape[1]} "
                    f"but num_video_anchor_latents={N}; normalising."
                )
                slices = [anchor_stack[:, i:i + 1] for i in range(anchor_stack.shape[1])]
                slices = _pad_or_trim_anchor_slots(slices, N)
                anchor_stack = torch.cat(slices, dim=1)
            anchor_stack = anchor_stack.to(device=device, dtype=user_anchor.dtype)
        else:
            anchor_stack = _build_clip0_anchor_stack(user_anchor, N, anchor_select_mode)

        # ── Build the M motion slot(s) ──────────────────────────────────────
        if prev_last_latent is not None and M > 0:
            motion_stack = prev_last_latent["samples"] if isinstance(prev_last_latent, dict) else prev_last_latent
            if motion_stack.dim() == 5:
                motion_stack = motion_stack.squeeze(0)
            if motion_stack.shape[1] != M:
                motion_stack = motion_stack[:, -M:]
                if motion_stack.shape[1] < M:
                    pad = motion_stack[:, -1:].repeat(1, M - motion_stack.shape[1], 1, 1)
                    motion_stack = torch.cat([motion_stack, pad], dim=1)
            motion_stack = motion_stack.to(device=device, dtype=user_anchor.dtype)
        elif M > 0:
            motion_stack = torch.zeros(
                user_anchor.shape[0], M, lat_h, lat_w, device=device, dtype=user_anchor.dtype
            )
        else:
            motion_stack = None

        # ── 4-channel mask + anchor packing ─────────────────────────────────
        # Layout (channel-axis): 4 mask channels + 16 VAE-latent channels.
        # Anchor slots: mask=1 (identity memory, fully observed)
        # Motion slots: mask=1 (previous-chunk latent, observed)
        # BG / denoise region: mask=0 (model generates)
        msk_anchor = torch.ones(4, N, lat_h, lat_w, device=device, dtype=vae.dtype)
        ref_latent_masked = torch.cat([msk_anchor, anchor_stack.to(vae.dtype)], dim=0)
        # ref_latent_masked: [20, N, h, w]

        if looping:
            # In looping mode the sampler builds the bg + motion region per
            # window. We only ship the anchor stack so it stays fixed across
            # iterations (Phase 2.1 will allow per-iter anchor reselection).
            ref_latent = ref_latent_masked.to(offload_device)
            del bg_latents
            bg_latents = None
        else:
            # Single-window: build the bg_mask and pack motion + bg slots.
            if mask is None:
                bg_mask_raw = torch.zeros(
                    1, extended_num_frames, lat_h, lat_w, device=device, dtype=vae.dtype
                )
            else:
                bg_mask_raw = 1 - mask[:extended_num_frames]
                if bg_mask_raw.shape[0] < extended_num_frames:
                    bg_mask_raw = torch.cat([
                        bg_mask_raw,
                        bg_mask_raw[-1:].repeat(extended_num_frames - bg_mask_raw.shape[0], 1, 1),
                    ], dim=0)
                bg_mask_raw = common_upscale(
                    bg_mask_raw.unsqueeze(1), lat_w, lat_h, "nearest", "disabled"
                ).squeeze(1)
                bg_mask_raw = bg_mask_raw.unsqueeze(-1).permute(3, 0, 1, 2).to(device, vae.dtype)

            bg_mask_first = torch.repeat_interleave(bg_mask_raw[:, 0:1], repeats=4, dim=1)
            bg_mask = torch.cat([bg_mask_first, bg_mask_raw[:, 1:]], dim=1)
            bg_mask = bg_mask.view(1, bg_mask.shape[1] // 4, 4, lat_h, lat_w)
            bg_mask = bg_mask.movedim(1, 2)[0]  # [4, T_lat, h, w]

            bg_mask_for_bg = bg_mask[:, :bg_latents.shape[1]]
            bg_latents_masked = torch.cat([bg_mask_for_bg, bg_latents.to(device)], dim=0)
            # bg_latents_masked: [20, T_bg, h, w]

            if motion_stack is not None and M > 0:
                if bg_latents_masked.shape[1] < M:
                    raise ValueError(
                        f"EverAnimate: bg region is too short "
                        f"({bg_latents_masked.shape[1]} latents) to host M={M} motion slots."
                    )
                bg_latents_masked[:4, :M] = 1.0
                bg_latents_masked[4:, :M] = motion_stack.to(bg_latents_masked.dtype)

            ref_latent = torch.cat(
                [ref_latent_masked, bg_latents_masked], dim=1
            ).to(offload_device)
            # ref_latent: [20, N + T_bg, h, w]
            del bg_latents_masked, bg_latents, bg_mask
        del ref_latent_masked

        # ── Face & start_ref handling (face encoded later in sampler) ──────
        resized_face_images = None
        if face_images is not None:
            face_images = face_images[..., :3]
            if face_images.shape[1] != 512 or face_images.shape[2] != 512:
                resized_face_images = common_upscale(
                    face_images.movedim(-1, 1), 512, 512, "lanczos", "center"
                ).movedim(0, 1)
            else:
                resized_face_images = face_images.permute(3, 0, 1, 2)
            resized_face_images = (resized_face_images * 2 - 1).unsqueeze(0)
            resized_face_images = resized_face_images.to(offload_device, dtype=vae.dtype)

        seq_len = math.ceil((target_shape[2] * target_shape[3]) / 4 * target_shape[1])

        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            # Standard WANVIDIMAGE_EMBEDS keys (sampler reads these)
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": seq_len,
            "pose_latents": pose_latents,  # single-window only; looping uses pose_images
            "pose_images": resized_pose_images if (pose_images is not None and looping) else None,
            "bg_images": resized_bg_images if (bg_images is not None and looping) else None,
            "ref_masks": None,
            "is_masked": mask is not None,
            "ref_latent": ref_latent,
            "ref_image": resized_ref_images if ref_images is not None else None,
            "start_ref_image": None,
            "face_pixels": resized_face_images if face_images is not None else None,
            "num_frames": extended_num_frames,
            "target_shape": target_shape,
            "frame_window_size": frame_window_size,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "vae": vae,
            "colormatch": colormatch,
            "looping": looping,
            "pose_strength": pose_strength,
            "face_strength": face_strength,
            "force_offload": force_offload,
            # EverAnimate metadata — sampler everanimate_mode branch reads these
            "everanimate_mode": True,
            "num_video_anchor_latents": N,
            "num_motion_latents": M,
            "anchor_select_mode": anchor_select_mode,
        }

        return (image_embeds,)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "WanVideoEverAnimateEmbeds": WanVideoEverAnimateEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoEverAnimateEmbeds": "WanVideo EverAnimate Embeds",
}
