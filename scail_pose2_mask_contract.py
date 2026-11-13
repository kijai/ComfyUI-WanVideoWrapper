"""SCAIL-Pose2 replacement noise-mask conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCAIL_POSE2_CONDITION_MODE_ATTR = "scail_pose2_condition_mode"
SCAIL_POSE2_MASK_ROLE_ATTR = "scail_pose2_mask_role"
SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE = "replacement_denoise_mask"
SCAIL_POSE2_DISABLE_SAMPLES_ATTR = "scail_pose2_disable_samples"
SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR = "scail_pose2_disable_samples_reason"
SCAIL_POSE2_SAMPLES_DISABLED_KEY = "scail_pose2_samples_disabled"
SCAIL_POSE2_DISABLE_REASON_KEY = "scail_pose2_disable_reason"
SCAIL_POSE2_CONDITION_MODE_KEY = "scail_pose2_condition_mode"
SCAIL_POSE2_DISABLED_REASON_UNSPECIFIED = "unspecified"
SCAIL_POSE2_CONDITION_MODE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class NoiseMaskLatentContract:
    original_shape: tuple[int, ...]
    prepared_shape: tuple[int, ...]
    latent_shape: tuple[int, int, int]
    channel_count: int
    interpolation_mode: str
    scail_pose2_replacement: bool
    frame_policy: str
    subject_ratio: float
    preserve_ratio: float
    pre_grow_subject_ratio: float
    latent_grow_pixels: int
    latent_temporal_grow_frames: int

    def to_log_string(self) -> str:
        return (
            "noise_mask_latent_contract "
            f"original_shape={self.original_shape} "
            f"prepared_shape={self.prepared_shape} "
            f"latent_shape={self.latent_shape} "
            f"channels={self.channel_count} "
            f"mode={self.interpolation_mode} "
            f"scail_pose2_replacement={self.scail_pose2_replacement} "
            f"frame_policy={self.frame_policy} "
            f"latent_grow_pixels={self.latent_grow_pixels} "
            f"latent_temporal_grow_frames={self.latent_temporal_grow_frames} "
            f"pre_grow_subject_ratio={self.pre_grow_subject_ratio:.6f} "
            f"subject_ratio={self.subject_ratio:.6f} "
            f"preserve_ratio={self.preserve_ratio:.6f}"
        )


@dataclass(frozen=True)
class SamplesInitializationContract:
    add_noise_to_samples: bool
    scail_pose2_replacement: bool
    mask_aware: bool
    subject_ratio: float
    preserve_ratio: float
    subject_source: str
    preserve_source: str

    def to_log_string(self) -> str:
        return (
            "samples_initialization_contract "
            f"add_noise_to_samples={self.add_noise_to_samples} "
            f"scail_pose2_replacement={self.scail_pose2_replacement} "
            f"mask_aware={self.mask_aware} "
            f"subject_source={self.subject_source} "
            f"preserve_source={self.preserve_source} "
            f"subject_ratio={self.subject_ratio:.6f} "
            f"preserve_ratio={self.preserve_ratio:.6f}"
        )


@dataclass(frozen=True)
class SamplesWindowAlignmentContract:
    original_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    source_latent_frame_count: int
    output_frame_count: int
    frame_policy: str
    start_latent: int | None
    end_latent: int | None

    def to_log_string(self) -> str:
        return (
            "samples_window_alignment_contract "
            f"original_shape={self.original_shape} "
            f"output_shape={self.output_shape} "
            f"source_latent_frame_count={self.source_latent_frame_count} "
            f"output_frame_count={self.output_frame_count} "
            f"frame_policy={self.frame_policy} "
            f"start_latent={self.start_latent} "
            f"end_latent={self.end_latent}"
        )


@dataclass(frozen=True)
class DisabledSamplesPayloadContext:
    condition_mode: str
    reason: str

    def to_log_fragment(self) -> str:
        return f"condition_mode={self.condition_mode} reason={self.reason}"


def is_scail_pose2_replacement_noise_mask(noise_mask: Any) -> bool:
    return (
        getattr(noise_mask, SCAIL_POSE2_CONDITION_MODE_ATTR, None) == "replacement"
        and getattr(noise_mask, SCAIL_POSE2_MASK_ROLE_ATTR, None)
        == SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE
    )


def scail_pose2_mask_disables_samples(mask: Any) -> bool:
    """Return whether SCAIL-Pose2 metadata disables the samples path."""

    return bool(getattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, False))


def build_disabled_samples_payload(mask: Any) -> dict[str, Any]:
    """Build a LATENT payload that downstream samplers intentionally ignore."""

    reason = getattr(
        mask,
        SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR,
        SCAIL_POSE2_DISABLED_REASON_UNSPECIFIED,
    )
    condition_mode = getattr(
        mask,
        SCAIL_POSE2_CONDITION_MODE_ATTR,
        SCAIL_POSE2_CONDITION_MODE_UNKNOWN,
    )
    return {
        "samples": None,
        "noise_mask": None,
        SCAIL_POSE2_SAMPLES_DISABLED_KEY: True,
        SCAIL_POSE2_DISABLE_REASON_KEY: reason,
        SCAIL_POSE2_CONDITION_MODE_KEY: condition_mode,
    }


def samples_payload_is_disabled(samples: Any) -> bool:
    """Return whether a LATENT samples payload is explicitly disabled."""

    return bool(
        isinstance(samples, dict)
        and samples.get(SCAIL_POSE2_SAMPLES_DISABLED_KEY, False)
    )


def disabled_samples_log_context(
    samples: Any,
) -> DisabledSamplesPayloadContext | None:
    """Return public-safe log context for an intentionally disabled samples path."""

    if not samples_payload_is_disabled(samples):
        return None
    return DisabledSamplesPayloadContext(
        condition_mode=str(
            samples.get(
                SCAIL_POSE2_CONDITION_MODE_KEY,
                SCAIL_POSE2_CONDITION_MODE_UNKNOWN,
            )
        ),
        reason=str(
            samples.get(
                SCAIL_POSE2_DISABLE_REASON_KEY,
                SCAIL_POSE2_DISABLED_REASON_UNSPECIFIED,
            )
        ),
    )


def normalize_samples_payload_for_sampler(
    samples: Any,
) -> tuple[Any | None, DisabledSamplesPayloadContext | None]:
    """Normalize the sampler input before any vid2vid samples branch runs."""

    context = disabled_samples_log_context(samples)
    if context is not None:
        return None, context
    return samples, None


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in getattr(value, "shape", ()))


def _positive_latent_shape(latent_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    parsed = tuple(int(part) for part in latent_shape)
    if len(parsed) != 3 or any(part <= 0 for part in parsed):
        raise ValueError("latent_shape must be a positive (frames, height, width) tuple")
    return parsed


def _positive_channel_count(channel_count: Any) -> int:
    parsed = int(channel_count)
    if parsed <= 0:
        raise ValueError("channel_count must be positive")
    return parsed


def _non_negative_grow_pixels(latent_grow_pixels: Any) -> int:
    parsed = int(latent_grow_pixels)
    if parsed < 0:
        raise ValueError("latent_grow_pixels must be non-negative")
    return parsed


def align_samples_to_latent_window(
    input_samples: Any,
    *,
    target_frame_count: Any,
    start_latent: int | None = None,
    end_latent: int | None = None,
) -> tuple[Any, SamplesWindowAlignmentContract]:
    """Align `[C,T,H,W]` samples to the active latent window."""

    original_shape = _shape_tuple(input_samples)
    if len(original_shape) != 4:
        raise ValueError("input_samples must have shape [C,T,H,W]")
    target_frames = int(target_frame_count)
    if target_frames <= 0:
        raise ValueError("target_frame_count must be positive")

    source_frames = int(input_samples.shape[1])
    output = input_samples
    policy = "direct"
    parsed_start = int(start_latent) if start_latent is not None else None
    parsed_end = int(end_latent) if end_latent is not None else None

    if parsed_start is not None and parsed_end is not None:
        if source_frames == target_frames:
            policy = f"direct_already_windowed_{parsed_start}_{parsed_end}"
        else:
            if parsed_start < 0 or parsed_end <= parsed_start or parsed_end > source_frames:
                raise ValueError(
                    "sample latent window is out of range, "
                    f"got {parsed_start}:{parsed_end} for {source_frames} frames"
                )
            output = input_samples[:, parsed_start:parsed_end]
            policy = f"slice_{parsed_start}_{parsed_end}"

    if int(output.shape[1]) != target_frames:
        raise ValueError(
            "sample latent window frame count mismatch, "
            f"got {int(output.shape[1])} frames for target {target_frames}"
        )

    contract = SamplesWindowAlignmentContract(
        original_shape=original_shape,
        output_shape=_shape_tuple(output),
        source_latent_frame_count=source_frames,
        output_frame_count=int(output.shape[1]),
        frame_policy=policy,
        start_latent=parsed_start,
        end_latent=parsed_end,
    )
    return output, contract


def resize_noise_mask_for_latents(
    noise_mask: Any,
    *,
    latent_shape: tuple[int, int, int],
    channel_count: Any,
    start_latent: int | None = None,
    end_latent: int | None = None,
    source_latent_frame_count: int | None = None,
    latent_grow_pixels: Any = 0,
    latent_temporal_grow_frames: Any = 0,
) -> tuple[Any, NoiseMaskLatentContract]:
    """Resize a sampler `noise_mask` to `[1, C, T, H, W]` latent mask shape."""

    target_frames, target_height, target_width = _positive_latent_shape(latent_shape)
    channels = _positive_channel_count(channel_count)
    grow_pixels = _non_negative_grow_pixels(latent_grow_pixels)
    temporal_grow_frames = _non_negative_grow_pixels(latent_temporal_grow_frames)
    original_shape = _shape_tuple(noise_mask)
    scail_pose2_replacement = is_scail_pose2_replacement_noise_mask(noise_mask)
    prepared = noise_mask

    if len(prepared.shape) == 4:
        prepared = prepared.squeeze(1)
    if len(prepared.shape) != 3:
        raise ValueError("noise_mask must have shape [T,H,W] or [T,1,H,W]")

    frame_policy = "direct"
    if (
        start_latent is not None
        and end_latent is not None
        and source_latent_frame_count is not None
        and int(prepared.shape[0]) != int(source_latent_frame_count)
    ):
        source_frames = int(source_latent_frame_count)
        if source_frames <= 0:
            raise ValueError("source_latent_frame_count must be positive")
        prepared = _resize_prepared_mask(
            prepared,
            target_frames=source_frames,
            target_height=target_height,
            target_width=target_width,
            scail_pose2_replacement=scail_pose2_replacement,
        )
        prepared = prepared[int(start_latent):int(end_latent)]
        frame_policy = (
            f"resize_full_{source_frames}_then_slice_"
            f"{int(start_latent)}_{int(end_latent)}"
        )
    elif prepared.shape[0] < target_frames:
        repeat_count = max(1, target_frames // int(prepared.shape[0]))
        if repeat_count > 1:
            prepared = prepared.repeat(repeat_count, 1, 1)
            frame_policy = f"repeat_x{repeat_count}"
        else:
            frame_policy = "interpolate_time"
    elif start_latent is not None and end_latent is not None:
        prepared = prepared[int(start_latent):int(end_latent)]
        frame_policy = f"slice_{int(start_latent)}_{int(end_latent)}"

    prepared_shape = _shape_tuple(prepared)
    interpolation_mode = "conservative_area" if scail_pose2_replacement else "trilinear"
    resized_3d = _resize_prepared_mask(
        prepared,
        target_frames=target_frames,
        target_height=target_height,
        target_width=target_width,
        scail_pose2_replacement=scail_pose2_replacement,
    )
    pre_grow_subject_ratio = float(resized_3d.float().mean().item())
    if scail_pose2_replacement and grow_pixels > 0:
        resized_3d = _grow_spatial_binary_mask(resized_3d, grow_pixels)
    if scail_pose2_replacement and temporal_grow_frames > 0:
        resized_3d = _grow_temporal_binary_mask(resized_3d, temporal_grow_frames)
    resized = resized_3d.unsqueeze(0).unsqueeze(0)
    subject_ratio = float(resized.float().mean().item())
    contract = NoiseMaskLatentContract(
        original_shape=original_shape,
        prepared_shape=prepared_shape,
        latent_shape=(target_frames, target_height, target_width),
        channel_count=channels,
        interpolation_mode=interpolation_mode,
        scail_pose2_replacement=scail_pose2_replacement,
        frame_policy=frame_policy,
        subject_ratio=subject_ratio,
        preserve_ratio=1.0 - subject_ratio,
        pre_grow_subject_ratio=pre_grow_subject_ratio,
        latent_grow_pixels=grow_pixels if scail_pose2_replacement else 0,
        latent_temporal_grow_frames=(
            temporal_grow_frames if scail_pose2_replacement else 0
        ),
    )
    return resized.repeat(1, channels, 1, 1, 1), contract


def _resize_prepared_mask(
    prepared: Any,
    *,
    target_frames: int,
    target_height: int,
    target_width: int,
    scail_pose2_replacement: bool,
) -> Any:
    import torch.nn.functional as F

    view = prepared.unsqueeze(0).unsqueeze(0)
    if scail_pose2_replacement:
        binary = (prepared >= 0.5).to(dtype=prepared.dtype)
        resized = F.interpolate(
            binary.unsqueeze(0).unsqueeze(0),
            size=(target_frames, target_height, target_width),
            mode="area",
        )
        resized = (resized > 0.0).to(dtype=prepared.dtype)
    else:
        resized = F.interpolate(
            view,
            size=(target_frames, target_height, target_width),
            mode="trilinear",
            align_corners=False,
        )
    return resized.squeeze(0).squeeze(0)


def _grow_spatial_binary_mask(mask: Any, grow_pixels: int) -> Any:
    import torch.nn.functional as F

    kernel = grow_pixels * 2 + 1
    view = mask.unsqueeze(0).unsqueeze(0).float()
    grown = F.max_pool3d(
        view,
        kernel_size=(1, kernel, kernel),
        stride=1,
        padding=(0, grow_pixels, grow_pixels),
    )
    return (grown.squeeze(0).squeeze(0) >= 0.5).to(dtype=mask.dtype)


def _grow_temporal_binary_mask(mask: Any, grow_frames: int) -> Any:
    import torch.nn.functional as F

    kernel = grow_frames * 2 + 1
    view = mask.unsqueeze(0).unsqueeze(0).float()
    grown = F.max_pool3d(
        view,
        kernel_size=(kernel, 1, 1),
        stride=1,
        padding=(grow_frames, 0, 0),
    )
    return (grown.squeeze(0).squeeze(0) >= 0.5).to(dtype=mask.dtype)


def apply_samples_to_noise(
    noise: Any,
    input_samples: Any,
    *,
    noise_mask: Any | None,
    timestep: Any,
    add_noise_to_samples: bool,
    scail_pose2_replacement: bool,
) -> tuple[Any, SamplesInitializationContract]:
    """Apply input samples to sampler noise with SCAIL-Pose2 mask awareness."""

    if input_samples is None:
        raise ValueError("input_samples must not be None")
    if tuple(input_samples.shape) != tuple(noise.shape):
        raise ValueError(
            "input_samples and noise must share shape, "
            f"got {tuple(input_samples.shape)} and {tuple(noise.shape)}"
        )

    if add_noise_to_samples:
        scale = _noise_timestep_scale(timestep, noise)
        initialized_from_samples = noise * scale + (1.0 - scale) * input_samples
        sample_source = "noised_samples"
    else:
        initialized_from_samples = input_samples
        sample_source = "samples"

    if scail_pose2_replacement and noise_mask is not None:
        subject_mask = _mask_like_noise(noise_mask, noise)
        preserve_mask = 1.0 - subject_mask
        initialized = initialized_from_samples * preserve_mask + noise * subject_mask
        subject_ratio = float(subject_mask.float().mean().item())
        return initialized, SamplesInitializationContract(
            add_noise_to_samples=bool(add_noise_to_samples),
            scail_pose2_replacement=True,
            mask_aware=True,
            subject_ratio=subject_ratio,
            preserve_ratio=1.0 - subject_ratio,
            subject_source="random_noise",
            preserve_source=sample_source,
        )

    return initialized_from_samples, SamplesInitializationContract(
        add_noise_to_samples=bool(add_noise_to_samples),
        scail_pose2_replacement=bool(scail_pose2_replacement),
        mask_aware=False,
        subject_ratio=0.0,
        preserve_ratio=1.0,
        subject_source=sample_source,
        preserve_source=sample_source,
    )


def _noise_timestep_scale(timestep: Any, noise: Any) -> Any:
    import torch

    if torch.is_tensor(timestep):
        timestep_value = timestep.to(device=noise.device, dtype=noise.dtype).reshape(-1)[0]
    else:
        timestep_value = torch.tensor(timestep, device=noise.device, dtype=noise.dtype)
    return timestep_value / 1000.0


def _mask_like_noise(noise_mask: Any, noise: Any) -> Any:
    mask = noise_mask
    if len(mask.shape) == len(noise.shape) + 1 and int(mask.shape[0]) == 1:
        mask = mask.squeeze(0)
    if len(mask.shape) != len(noise.shape):
        raise ValueError(
            "noise_mask must have shape [C,T,H,W] or [1,C,T,H,W], "
            f"got {tuple(noise_mask.shape)} for noise {tuple(noise.shape)}"
        )
    if int(mask.shape[0]) == 1 and int(noise.shape[0]) != 1:
        mask = mask.repeat(int(noise.shape[0]), 1, 1, 1)
    if tuple(mask.shape) != tuple(noise.shape):
        raise ValueError(
            "noise_mask and noise must share broadcasted shape, "
            f"got {tuple(mask.shape)} and {tuple(noise.shape)}"
        )
    return mask.to(device=noise.device, dtype=noise.dtype).clamp(0.0, 1.0)
