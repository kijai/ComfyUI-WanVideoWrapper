from __future__ import annotations

import importlib.util
import unittest

from scail_pose2_mask_contract import (
    SCAIL_POSE2_CONDITION_MODE_ATTR,
    SCAIL_POSE2_DISABLE_SAMPLES_ATTR,
    SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR,
    SCAIL_POSE2_MASK_ROLE_ATTR,
    SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
    build_disabled_samples_payload,
    is_scail_pose2_replacement_noise_mask,
    normalize_samples_payload_for_sampler,
    resize_noise_mask_for_latents,
    samples_payload_is_disabled,
    scail_pose2_mask_disables_samples,
)


class _DummyMask:
    pass


class ScailPose2SamplesDisableContractTests(unittest.TestCase):
    def test_non_replacement_mask_metadata_builds_disabled_samples_payload(self) -> None:
        mask = _DummyMask()
        setattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, True)
        setattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR, "non_replacement_mode")
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "animation")

        payload = build_disabled_samples_payload(mask)

        self.assertTrue(scail_pose2_mask_disables_samples(mask))
        self.assertTrue(samples_payload_is_disabled(payload))
        self.assertIsNone(payload["samples"])
        self.assertIsNone(payload["noise_mask"])
        self.assertTrue(payload["scail_pose2_samples_disabled"])
        self.assertEqual("animation", payload["scail_pose2_condition_mode"])
        self.assertEqual("non_replacement_mode", payload["scail_pose2_disable_reason"])

    def test_replacement_and_generic_payloads_are_not_disabled(self) -> None:
        replacement_mask = _DummyMask()
        setattr(replacement_mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(
            replacement_mask,
            SCAIL_POSE2_MASK_ROLE_ATTR,
            SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
        )
        generic_mask = _DummyMask()

        self.assertFalse(scail_pose2_mask_disables_samples(replacement_mask))
        self.assertFalse(scail_pose2_mask_disables_samples(generic_mask))
        self.assertFalse(samples_payload_is_disabled({"samples": object()}))
        self.assertFalse(samples_payload_is_disabled(None))

    def test_sampler_normalizer_ignores_disabled_payload_with_log_context(self) -> None:
        mask = _DummyMask()
        setattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, True)
        setattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR, "non_replacement_mode")
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "animation")
        payload = build_disabled_samples_payload(mask)

        normalized, context = normalize_samples_payload_for_sampler(payload)

        self.assertIsNone(normalized)
        self.assertIsNotNone(context)
        self.assertEqual("animation", context.condition_mode)
        self.assertEqual("non_replacement_mode", context.reason)
        self.assertIn("condition_mode=animation", context.to_log_fragment())
        self.assertIn("reason=non_replacement_mode", context.to_log_fragment())

    def test_sampler_normalizer_preserves_normal_payload(self) -> None:
        payload = {"samples": object(), "noise_mask": object()}

        normalized, context = normalize_samples_payload_for_sampler(payload)

        self.assertIs(normalized, payload)
        self.assertIsNone(context)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
class ScailPose2MaskContractTests(unittest.TestCase):
    def test_tagged_replacement_mask_uses_conservative_binary_latent_conversion(self) -> None:
        import torch

        mask = torch.tensor([[[0.0, 1.0], [0.49, 0.51]]], dtype=torch.float32)
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(1, 2, 2),
            channel_count=3,
            latent_grow_pixels=0,
        )

        self.assertTrue(is_scail_pose2_replacement_noise_mask(mask))
        self.assertEqual((1, 3, 1, 2, 2), tuple(latent_mask.shape))
        self.assertEqual("conservative_area", contract.interpolation_mode)
        self.assertTrue(contract.scail_pose2_replacement)
        self.assertEqual(0.0, float(latent_mask[0, 0, 0, 0, 0].item()))
        self.assertEqual(1.0, float(latent_mask[0, 0, 0, 0, 1].item()))
        self.assertEqual(0.0, float(latent_mask[0, 0, 0, 1, 0].item()))
        self.assertEqual(1.0, float(latent_mask[0, 0, 0, 1, 1].item()))
        self.assertAlmostEqual(0.5, contract.subject_ratio)
        self.assertAlmostEqual(0.5, contract.preserve_ratio)
        self.assertEqual(0, contract.latent_grow_pixels)
        self.assertAlmostEqual(0.5, contract.pre_grow_subject_ratio)

    def test_thin_replacement_subject_survives_latent_downsampling(self) -> None:
        import torch

        mask = torch.zeros((1, 8, 8), dtype=torch.float32)
        mask[:, :, 7] = 1.0
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(1, 1, 1),
            channel_count=1,
            latent_grow_pixels=0,
        )

        self.assertEqual("conservative_area", contract.interpolation_mode)
        self.assertEqual(1.0, float(latent_mask[0, 0, 0, 0, 0].item()))
        self.assertAlmostEqual(1.0, contract.subject_ratio)
        self.assertAlmostEqual(0.0, contract.preserve_ratio)

    def test_untagged_mask_keeps_trilinear_policy(self) -> None:
        import torch

        mask = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(2, 1, 2),
            channel_count=2,
        )

        self.assertFalse(is_scail_pose2_replacement_noise_mask(mask))
        self.assertEqual((1, 2, 2, 1, 2), tuple(latent_mask.shape))
        self.assertEqual("trilinear", contract.interpolation_mode)
        self.assertFalse(contract.scail_pose2_replacement)

    def test_frame_slice_policy_is_reported(self) -> None:
        import torch

        mask = torch.ones((4, 2, 2), dtype=torch.float32)
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(2, 1, 1),
            channel_count=1,
            start_latent=1,
            end_latent=3,
        )

        self.assertEqual((1, 1, 2, 1, 1), tuple(latent_mask.shape))
        self.assertEqual("slice_1_3", contract.frame_policy)

    def test_context_slice_resizes_full_source_timeline_before_slice(self) -> None:
        import torch

        mask = torch.zeros((8, 2, 2), dtype=torch.float32)
        mask[4:] = 1.0
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(2, 1, 1),
            channel_count=1,
            start_latent=1,
            end_latent=3,
            source_latent_frame_count=4,
        )

        self.assertEqual((1, 1, 2, 1, 1), tuple(latent_mask.shape))
        self.assertEqual("resize_full_4_then_slice_1_3", contract.frame_policy)

    def test_replacement_latent_grow_is_spatial_only(self) -> None:
        import torch

        mask = torch.zeros((2, 5, 5), dtype=torch.float32)
        mask[:, 2, 2] = 1.0
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(2, 5, 5),
            channel_count=1,
            latent_grow_pixels=1,
        )

        self.assertEqual((1, 1, 2, 5, 5), tuple(latent_mask.shape))
        self.assertEqual(1, contract.latent_grow_pixels)
        self.assertAlmostEqual(1.0 / 25.0, contract.pre_grow_subject_ratio)
        self.assertAlmostEqual(9.0 / 25.0, contract.subject_ratio)
        self.assertEqual(9.0, float(latent_mask[0, 0, 0].sum().item()))
        self.assertEqual(9.0, float(latent_mask[0, 0, 1].sum().item()))

    def test_replacement_temporal_grow_is_bounded_and_reported(self) -> None:
        import torch

        mask = torch.zeros((5, 1, 1), dtype=torch.float32)
        mask[2, 0, 0] = 1.0
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(5, 1, 1),
            channel_count=1,
            latent_grow_pixels=0,
            latent_temporal_grow_frames=1,
        )

        self.assertEqual(1, contract.latent_temporal_grow_frames)
        self.assertAlmostEqual(1.0 / 5.0, contract.pre_grow_subject_ratio)
        self.assertAlmostEqual(3.0 / 5.0, contract.subject_ratio)
        self.assertEqual([0.0, 1.0, 1.0, 1.0, 0.0], latent_mask[0, 0, :, 0, 0].tolist())
        self.assertIn("latent_temporal_grow_frames=1", contract.to_log_string())

    def test_temporal_grow_is_ignored_for_untagged_masks(self) -> None:
        import torch

        mask = torch.zeros((5, 1, 1), dtype=torch.float32)
        mask[2, 0, 0] = 1.0

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(5, 1, 1),
            channel_count=1,
            latent_grow_pixels=0,
            latent_temporal_grow_frames=1,
        )

        self.assertFalse(contract.scail_pose2_replacement)
        self.assertEqual(0, contract.latent_temporal_grow_frames)
        self.assertEqual([0.0, 0.0, 1.0, 0.0, 0.0], latent_mask[0, 0, :, 0, 0].tolist())


if __name__ == "__main__":
    unittest.main()
