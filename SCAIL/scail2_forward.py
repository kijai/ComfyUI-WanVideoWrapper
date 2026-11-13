from __future__ import annotations

from typing import Any


REF_SPATIAL_SHIFT = 120.0
POSE_SPATIAL_SHIFT = 120.0
SCAIL2_HISTORY_CHANNELS = 4
SCAIL2_STRENGTH_DEFAULTS = {
    "ref_image": 1.0,
    "ref_mask": 1.0,
    "condition_video": 1.0,
    "driving_mask": 1.0,
}


def as_scail2_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _scail2_strength(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"SCAIL-2 strength {name} must be a number")
    strength = float(value)
    if strength < 0.0 or strength > 10.0:
        raise ValueError(f"SCAIL-2 strength {name} must be between 0.0 and 10.0")
    return strength


def scail2_strengths(scail2_input: dict[str, Any] | None) -> dict[str, float]:
    if scail2_input is None:
        return dict(SCAIL2_STRENGTH_DEFAULTS)
    raw = scail2_input.get("strengths") or {}
    if not isinstance(raw, dict):
        raise ValueError("SCAIL-2 strengths must be a dict")
    return {
        name: _scail2_strength(name, raw.get(name, default))
        for name, default in SCAIL2_STRENGTH_DEFAULTS.items()
    }


def scale_scail2_items(items: list[Any], strength: float) -> list[Any]:
    if float(strength) == 1.0:
        return items
    return [item * float(strength) for item in items]


def shape_of(value: Any, *, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{name} must expose a shape")
    return tuple(int(part) for part in shape)


def patch_embedding_input_channels(patch_embedding: Any) -> int | None:
    weight = getattr(patch_embedding, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None or len(tuple(shape)) < 2:
        return None
    return int(shape[1])


def scail2_history_channels_needed(
    *,
    latent_channels: int,
    patch_channels: int | None,
) -> int:
    if patch_channels is None or patch_channels == latent_channels:
        return 0
    if patch_channels - latent_channels == SCAIL2_HISTORY_CHANNELS:
        return SCAIL2_HISTORY_CHANNELS
    raise ValueError(
        "SCAIL-2 main latent channel mismatch: "
        f"patch embedding expects {patch_channels}, got {latent_channels}"
    )


def append_scail2_history_channels(
    latent: Any,
    *,
    patch_embedding: Any,
    name: str = "SCAIL-2 main latent",
    fill_value: float = 0.0,
) -> Any:
    latent_shape = shape_of(latent, name=name)
    if len(latent_shape) != 4:
        raise ValueError(f"{name} must be CTHW, got {latent_shape}")
    needed = scail2_history_channels_needed(
        latent_channels=latent_shape[0],
        patch_channels=patch_embedding_input_channels(patch_embedding),
    )
    if needed == 0:
        return latent

    # Official ComfyUI SCAIL-2 supplies these 4 history-mask channels before
    # the 20-channel patch embedding; zero is the official fallback without a mask.
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime requires torch
        raise RuntimeError("torch is required to append SCAIL-2 history channels") from exc
    history_shape = (needed, *latent_shape[1:])
    if float(fill_value) == 0.0:
        history = latent.new_zeros(history_shape)
    else:
        history = latent.new_full(history_shape, float(fill_value))
    return torch.cat([latent, history], dim=0)


def mark_scail2_prefix_history_channels(
    latent: Any,
    *,
    prefix_frames: int,
    patch_embedding: Any,
    name: str = "SCAIL-2 main latent",
    fill_value: float = 1.0,
) -> Any:
    latent_shape = shape_of(latent, name=name)
    if len(latent_shape) != 4:
        raise ValueError(f"{name} must be CTHW, got {latent_shape}")
    prefix_frames = int(prefix_frames)
    if prefix_frames <= 0:
        return latent
    if prefix_frames > latent_shape[1]:
        raise ValueError(
            f"{name} prefix_frames={prefix_frames} exceeds latent frame count {latent_shape[1]}"
        )

    patch_channels = patch_embedding_input_channels(patch_embedding)
    if patch_channels is not None and patch_channels != latent_shape[0]:
        raise ValueError(
            "SCAIL-2 main latent channel mismatch after history append: "
            f"patch embedding expects {patch_channels}, got {latent_shape[0]}"
        )
    needed = SCAIL2_HISTORY_CHANNELS
    if latent_shape[0] < needed:
        raise ValueError(f"{name} has fewer channels than SCAIL-2 history channels")

    marked = latent.clone()
    marked[-needed:, :prefix_frames] = float(fill_value)
    return marked


def latent_frames(items: list[Any], *, name: str) -> int:
    return sum(shape_of(item, name=name)[1] for item in items)


def patch_count(frames: int, height: int, width: int, patch_size: tuple[int, int, int]) -> int:
    patch_t, patch_h, patch_w = patch_size
    t = (frames + (patch_t // 2)) // patch_t
    h = (height + (patch_h // 2)) // patch_h
    w = (width + (patch_w // 2)) // patch_w
    return t * h * w


def scail2_rope_shifts(*, replace_flag: bool, additional_ref_count: int) -> dict[str, dict[str, float]]:
    base_video_shift = 0 if replace_flag else 1
    return {
        "t": {
            "additional_ref": 0,
            "ref": additional_ref_count,
            "pose": base_video_shift + additional_ref_count,
            "video": base_video_shift + additional_ref_count,
        },
        "h": {
            "additional_ref": REF_SPATIAL_SHIFT if replace_flag else 0.0,
            "ref": REF_SPATIAL_SHIFT if replace_flag else 0.0,
            "pose": 0.0,
            "video": 0.0,
        },
        "w": {
            "additional_ref": 0.0,
            "ref": 0.0,
            "pose": POSE_SPATIAL_SHIFT,
            "video": 0.0,
        },
    }


def _require_pair(left: list[Any], right: list[Any], *, left_name: str, right_name: str) -> None:
    if left and not right:
        raise ValueError(f"{right_name} is required when {left_name} is provided")
    if right and not left:
        raise ValueError(f"{right_name} requires {left_name}")
    if left and right and len(left) != len(right):
        raise ValueError(f"{left_name} and {right_name} must have the same item count")


def build_scail2_forward_plan(
    scail2_input: dict[str, Any],
    *,
    video_shape: tuple[int, int, int, int],
    patch_size: tuple[int, int, int],
) -> dict[str, Any]:
    if scail2_input is None:
        raise ValueError("scail2_input is required")

    video_shape = tuple(int(part) for part in video_shape)
    if len(video_shape) != 4:
        raise ValueError(f"video_shape must be (C, F, H, W), got {video_shape}")

    _channels, video_frames, height, width = video_shape
    ref_latents = as_scail2_list(scail2_input.get("ref_latents"))
    ref_masks = as_scail2_list(scail2_input.get("ref_masks"))
    pose_latents = as_scail2_list(scail2_input.get("pose_latents"))
    driving_masks = as_scail2_list(scail2_input.get("driving_masks"))
    additional_ref_latents = as_scail2_list(scail2_input.get("additional_ref_latents"))
    additional_ref_masks = as_scail2_list(scail2_input.get("additional_ref_masks"))

    _require_pair(ref_latents, ref_masks, left_name="ref_latents", right_name="ref_masks")
    _require_pair(
        additional_ref_latents,
        additional_ref_masks,
        left_name="additional_ref_latents",
        right_name="additional_ref_masks",
    )

    if pose_latents and driving_masks:
        pose_shape = shape_of(pose_latents[0], name="pose_latents")
        mask_shape = shape_of(driving_masks[0], name="driving_masks")
        if pose_shape[1:] != mask_shape[1:]:
            raise ValueError(
                "pose_latents and driving_masks must share temporal/spatial shape, "
                f"got {pose_shape[1:]} and {mask_shape[1:]}"
            )

    control_shape = None
    if pose_latents:
        control_shape = shape_of(pose_latents[0], name="pose_latents")
    elif driving_masks:
        control_shape = shape_of(driving_masks[0], name="driving_masks")

    additional_ref_frames = latent_frames(additional_ref_latents, name="additional_ref_latents")
    ref_frames = latent_frames(ref_latents, name="ref_latents")
    prefix_frames = additional_ref_frames + ref_frames
    main_length = patch_count(video_frames + prefix_frames, height, width, patch_size)
    additional_ref_length = patch_count(additional_ref_frames, height, width, patch_size) if additional_ref_frames else 0
    ref_length = patch_count(ref_frames, height, width, patch_size) if ref_frames else 0
    control_length = 0
    if control_shape is not None:
        control_length = patch_count(control_shape[1], control_shape[2], control_shape[3], patch_size)

    additional_ref_count = (additional_ref_frames + (patch_size[0] // 2)) // patch_size[0]
    replace_flag = bool(scail2_input.get("replace_flag", False))
    rope_shifts = scail2_rope_shifts(
        replace_flag=replace_flag,
        additional_ref_count=additional_ref_count,
    )

    return {
        "replace_flag": replace_flag,
        "video_shape": video_shape,
        "video_frames": video_frames,
        "height": height,
        "width": width,
        "prefix_frames": prefix_frames,
        "additional_ref_frames": additional_ref_frames,
        "ref_frames": ref_frames,
        "additional_ref_count": additional_ref_count,
        "main_length": main_length,
        "additional_ref_length": additional_ref_length,
        "ref_length": ref_length,
        "control_length": control_length,
        "total_length": main_length + control_length,
        "has_pose": bool(pose_latents),
        "has_control_mask": bool(driving_masks),
        "has_ref_mask_stream": bool(ref_masks or additional_ref_masks),
        "control_shape": control_shape,
        "rope_shifts": rope_shifts,
        "cache_key": (
            replace_flag,
            video_shape,
            tuple(patch_size),
            prefix_frames,
            additional_ref_frames,
            ref_frames,
            control_shape,
            tuple(
                (axis, tuple(sorted(values.items())))
                for axis, values in sorted(rope_shifts.items())
            ),
        ),
    }
