from __future__ import annotations

from typing import Any


SCAIL2_MASK_CHANNELS = 28


def _conv3d_shape(sd: dict[str, Any], key: str, *, dim: int, patch_size: tuple[int, int, int]) -> tuple[int, ...]:
    weight = sd[key]
    shape = getattr(weight, "shape", None)
    if shape is None:
        raise ValueError(f"{key} must expose a Conv3d weight shape")

    shape = tuple(int(value) for value in shape)
    if len(shape) != 5:
        raise ValueError(f"{key} must be a 5D Conv3d weight, got shape {shape}")
    if shape[0] != dim:
        raise ValueError(f"{key} output channels {shape[0]} do not match model dim {dim}")
    if shape[2:] != tuple(patch_size):
        raise ValueError(f"{key} kernel shape {shape[2:]} does not match patch_size {patch_size}")
    return shape


def apply_scail_loader_patches(
    transformer: Any,
    sd: dict[str, Any],
    nn: Any,
    *,
    dim: int,
    patch_size: tuple[int, int, int],
    log: Any,
) -> None:
    transformer.scail2_enabled = False
    transformer.scail2_mask_dim = None

    if "patch_embedding_pose.weight" in sd:
        log.info("SCAIL model detected, patching model...")
        pose_shape = _conv3d_shape(
            sd,
            "patch_embedding_pose.weight",
            dim=dim,
            patch_size=patch_size,
        )
        pose_dim = pose_shape[1]
        transformer.patch_embedding_pose = nn.Conv3d(
            pose_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    if "patch_embedding_mask.weight" in sd:
        log.info("SCAIL-2 mask embedding detected, patching model...")
        mask_shape = _conv3d_shape(
            sd,
            "patch_embedding_mask.weight",
            dim=dim,
            patch_size=patch_size,
        )
        mask_dim = mask_shape[1]
        if mask_dim != SCAIL2_MASK_CHANNELS:
            raise ValueError(
                "patch_embedding_mask.weight must use 28 input channels for SCAIL-2, "
                f"got {mask_dim}"
            )
        transformer.patch_embedding_mask = nn.Conv3d(
            mask_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        transformer.scail2_enabled = True
        transformer.scail2_mask_dim = mask_dim
