from __future__ import annotations

from typing import Any, Callable, MutableMapping


SCAIL_V1_EMBEDS_KEY = "scail_embeds"
SCAIL2_EMBEDS_KEY = "scail2_embeds"
SCAIL2_MODEL_ARG = "scail2_input"


def prepare_scail2_data(
    image_embeds: MutableMapping[str, Any],
    *,
    dict_to_device: Callable[[dict[str, Any], Any, Any], dict[str, Any]],
    device: Any,
    dtype: Any,
) -> dict[str, Any] | None:
    """Extract native SCAIL-2 embeds and move copied data to sampler device."""

    scail2_embeds = image_embeds.get(SCAIL2_EMBEDS_KEY)
    if scail2_embeds is None:
        return None

    if image_embeds.get(SCAIL_V1_EMBEDS_KEY) is not None:
        raise ValueError(
            "SCAIL-2 native scail2_embeds cannot be combined with v1 scail_embeds"
        )

    if not isinstance(scail2_embeds, dict):
        raise TypeError("image_embeds['scail2_embeds'] must be a dict")

    return dict_to_device(scail2_embeds.copy(), device, dtype)


def scail2_context_window_input(
    scail2_data: dict[str, Any] | None,
    context_window: Any,
) -> dict[str, Any] | None:
    if scail2_data is None:
        return None

    if context_window is None:
        return scail2_data

    validate_scail2_context_frame_map(scail2_data, context_window)

    sliced = scail2_data.copy()
    sliced["pose_latents"] = _slice_temporal_field(
        scail2_data.get("pose_latents"),
        context_window,
        field_name="pose_latents",
    )
    sliced["driving_masks"] = _slice_temporal_field(
        scail2_data.get("driving_masks"),
        context_window,
        field_name="driving_masks",
    )
    return sliced


def validate_scail2_context_frame_map(
    scail2_data: dict[str, Any],
    context_window: Any,
) -> None:
    pose_counts = _temporal_frame_counts(
        scail2_data.get("pose_latents"),
        field_name="pose_latents",
    )
    mask_counts = _temporal_frame_counts(
        scail2_data.get("driving_masks"),
        field_name="driving_masks",
    )
    _validate_field_counts_consistent("pose_latents", pose_counts)
    _validate_field_counts_consistent("driving_masks", mask_counts)
    if pose_counts and mask_counts and sorted(set(pose_counts)) != sorted(set(mask_counts)):
        raise ValueError(
            "SCAIL-2 context frame map mismatch: "
            f"pose_latents frames={list(pose_counts)} "
            f"driving_masks frames={list(mask_counts)}"
        )
    for field_name, counts in (
        ("pose_latents", pose_counts),
        ("driving_masks", mask_counts),
    ):
        for frame_count in sorted(set(counts)):
            _validate_context_indices(
                context_window,
                frame_count=frame_count,
                field_name=field_name,
            )


def _temporal_frame_counts(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(
            count
            for item in value
            for count in _temporal_frame_counts(item, field_name=field_name)
        )
    if isinstance(value, tuple):
        return tuple(
            count
            for item in value
            for count in _temporal_frame_counts(item, field_name=field_name)
        )
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError(f"SCAIL-2 {field_name} tensors must expose CTHW-like shape")
    return (int(shape[1]),)


def _validate_field_counts_consistent(field_name: str, counts: tuple[int, ...]) -> None:
    if len(set(counts)) > 1:
        raise ValueError(
            f"SCAIL-2 context frame map mismatch within {field_name}: "
            f"frames={list(counts)}"
        )


def _validate_context_indices(
    context_window: Any,
    *,
    frame_count: int,
    field_name: str,
) -> None:
    if isinstance(context_window, slice):
        _context_window_length(context_window, frame_count)
        return
    if isinstance(context_window, (str, bytes)):
        raise TypeError("SCAIL-2 context_window must be a sequence of frame indices")
    length = _context_window_length(context_window, frame_count)
    for raw_index in context_window:
        index = int(raw_index)
        if index < -frame_count or index >= frame_count:
            raise ValueError(
                "SCAIL-2 context_window index out of range: "
                f"field={field_name} index={index} "
                f"frame_count={frame_count} context_length={length}"
            )


def _slice_temporal_field(
    value: Any,
    context_window: Any,
    *,
    field_name: str,
) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [
            _slice_temporal_tensor(item, context_window, field_name=field_name)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _slice_temporal_tensor(item, context_window, field_name=field_name)
            for item in value
        )
    return _slice_temporal_tensor(value, context_window, field_name=field_name)


def _slice_temporal_tensor(tensor: Any, context_window: Any, *, field_name: str) -> Any:
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError(f"SCAIL-2 {field_name} tensors must expose CTHW-like shape")

    expected_frames = _context_window_length(context_window, int(shape[1]))
    try:
        sliced = tensor[:, context_window]
    except Exception as exc:
        raise TypeError(
            f"SCAIL-2 {field_name} cannot be sliced by wrapper context_window"
        ) from exc

    sliced_shape = getattr(sliced, "shape", None)
    if sliced_shape is None or len(sliced_shape) < 2:
        raise ValueError(
            f"SCAIL-2 {field_name} context slicing must preserve time dimension"
        )
    if int(sliced_shape[1]) != expected_frames:
        raise ValueError(
            f"SCAIL-2 {field_name} context slice has {sliced_shape[1]} frames, "
            f"expected {expected_frames}"
        )
    return sliced


def _context_window_length(context_window: Any, frame_count: int) -> int:
    if isinstance(context_window, slice):
        return len(range(*context_window.indices(frame_count)))
    if isinstance(context_window, (str, bytes)):
        raise TypeError("SCAIL-2 context_window must be a sequence of frame indices")
    try:
        length = len(context_window)
    except TypeError as exc:
        raise TypeError(
            "SCAIL-2 context_window must be a sequence of frame indices"
        ) from exc
    if length <= 0:
        raise ValueError("SCAIL-2 context_window must contain at least one frame")
    return int(length)


def add_scail2_model_param(
    base_params: dict[str, Any],
    scail2_data_in: dict[str, Any] | None,
) -> dict[str, Any]:
    if scail2_data_in is not None:
        base_params[SCAIL2_MODEL_ARG] = scail2_data_in
    return base_params
