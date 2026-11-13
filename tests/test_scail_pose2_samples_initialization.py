from __future__ import annotations

import importlib.util
import unittest

from scail_pose2_mask_contract import (
    SCAIL_POSE2_CONDITION_MODE_ATTR,
    SCAIL_POSE2_MASK_ROLE_ATTR,
    SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
    align_samples_to_latent_window,
    apply_samples_to_noise,
    resize_noise_mask_for_latents,
)


def _tag_replacement_mask(mask):
    setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
    setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)
    return mask


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
class ScailPose2SamplesInitializationTests(unittest.TestCase):
    def test_align_samples_to_latent_window_slices_and_preserves_source_count(self) -> None:
        import torch

        samples = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1)

        sliced, contract = align_samples_to_latent_window(
            samples,
            target_frame_count=2,
            start_latent=2,
            end_latent=4,
        )

        self.assertEqual((1, 2, 1, 1), tuple(sliced.shape))
        self.assertEqual([2.0, 3.0], sliced[0, :, 0, 0].tolist())
        self.assertEqual(6, contract.source_latent_frame_count)
        self.assertEqual(2, contract.output_frame_count)
        self.assertEqual("slice_2_4", contract.frame_policy)
        self.assertIn("source_latent_frame_count=6", contract.to_log_string())

    def test_align_samples_to_latent_window_accepts_already_windowed_samples(self) -> None:
        import torch

        samples = torch.arange(2, dtype=torch.float32).view(1, 2, 1, 1)

        aligned, contract = align_samples_to_latent_window(
            samples,
            target_frame_count=2,
            start_latent=2,
            end_latent=4,
        )

        self.assertTrue(torch.equal(samples, aligned))
        self.assertEqual(2, contract.source_latent_frame_count)
        self.assertEqual("direct_already_windowed_2_4", contract.frame_policy)

    def test_align_samples_to_latent_window_rejects_mismatched_slice(self) -> None:
        import torch

        samples = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1)

        with self.assertRaisesRegex(ValueError, "sample latent window"):
            align_samples_to_latent_window(
                samples,
                target_frame_count=3,
                start_latent=2,
                end_latent=4,
            )

    def test_context_window_subject_uses_noise_after_sample_and_mask_slicing(self) -> None:
        import torch

        samples = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1)
        noise = torch.full((1, 2, 1, 1), 10.0, dtype=torch.float32)
        full_mask = _tag_replacement_mask(torch.zeros((12, 1, 1), dtype=torch.float32))
        full_mask[6:8, 0, 0] = 1.0

        sliced_samples, sample_contract = align_samples_to_latent_window(
            samples,
            target_frame_count=2,
            start_latent=2,
            end_latent=4,
        )
        latent_mask, mask_contract = resize_noise_mask_for_latents(
            full_mask,
            latent_shape=(2, 1, 1),
            channel_count=1,
            start_latent=2,
            end_latent=4,
            source_latent_frame_count=sample_contract.source_latent_frame_count,
        )
        initialized, init_contract = apply_samples_to_noise(
            noise,
            sliced_samples,
            noise_mask=latent_mask,
            timestep=torch.tensor([0.0], dtype=torch.float32),
            add_noise_to_samples=False,
            scail_pose2_replacement=mask_contract.scail_pose2_replacement,
        )

        self.assertEqual("slice_2_4", sample_contract.frame_policy)
        self.assertEqual("resize_full_6_then_slice_2_4", mask_contract.frame_policy)
        self.assertEqual([0.0, 1.0], latent_mask[0, 0, :, 0, 0].tolist())
        self.assertEqual(2.0, float(initialized[0, 0, 0, 0].item()))
        self.assertEqual(10.0, float(initialized[0, 1, 0, 0].item()))
        self.assertEqual("random_noise", init_contract.subject_source)

    def test_replacement_subject_keeps_random_noise_when_add_noise_is_enabled(self) -> None:
        import torch

        noise = torch.full((2, 1, 2, 2), 10.0, dtype=torch.float32)
        input_samples = torch.full_like(noise, 100.0)
        mask = _tag_replacement_mask(
            torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        )
        latent_mask, _ = resize_noise_mask_for_latents(
            mask,
            latent_shape=(1, 2, 2),
            channel_count=2,
            latent_grow_pixels=0,
        )

        initialized, contract = apply_samples_to_noise(
            noise,
            input_samples,
            noise_mask=latent_mask,
            timestep=torch.tensor([500.0], dtype=torch.float32),
            add_noise_to_samples=True,
            scail_pose2_replacement=True,
        )

        self.assertTrue(contract.mask_aware)
        self.assertTrue(contract.scail_pose2_replacement)
        self.assertEqual("random_noise", contract.subject_source)
        self.assertEqual("noised_samples", contract.preserve_source)
        self.assertIn("subject_source=random_noise", contract.to_log_string())
        self.assertIn("preserve_source=noised_samples", contract.to_log_string())
        self.assertAlmostEqual(0.25, contract.subject_ratio)
        self.assertEqual(10.0, float(initialized[0, 0, 0, 0].item()))
        self.assertEqual(55.0, float(initialized[0, 0, 0, 1].item()))
        self.assertEqual(55.0, float(initialized[1, 0, 1, 1].item()))

    def test_replacement_subject_keeps_random_noise_when_add_noise_is_disabled(self) -> None:
        import torch

        noise = torch.full((1, 1, 2, 2), 7.0, dtype=torch.float32)
        input_samples = torch.full_like(noise, 99.0)
        mask = _tag_replacement_mask(
            torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], dtype=torch.float32)
        )
        latent_mask, _ = resize_noise_mask_for_latents(
            mask,
            latent_shape=(1, 2, 2),
            channel_count=1,
            latent_grow_pixels=0,
        )

        initialized, contract = apply_samples_to_noise(
            noise,
            input_samples,
            noise_mask=latent_mask,
            timestep=torch.tensor([0.0], dtype=torch.float32),
            add_noise_to_samples=False,
            scail_pose2_replacement=True,
        )

        self.assertTrue(contract.mask_aware)
        self.assertEqual("random_noise", contract.subject_source)
        self.assertEqual("samples", contract.preserve_source)
        self.assertEqual(7.0, float(initialized[0, 0, 0, 1].item()))
        self.assertEqual(99.0, float(initialized[0, 0, 0, 0].item()))
        self.assertEqual(99.0, float(initialized[0, 0, 1, 1].item()))

    def test_generic_masks_keep_existing_full_sample_initialization(self) -> None:
        import torch

        noise = torch.full((1, 1, 2, 2), 10.0, dtype=torch.float32)
        input_samples = torch.full_like(noise, 100.0)
        latent_mask = torch.tensor(
            [[[[[1.0, 0.0], [0.0, 0.0]]]]],
            dtype=torch.float32,
        )

        initialized, contract = apply_samples_to_noise(
            noise,
            input_samples,
            noise_mask=latent_mask,
            timestep=torch.tensor([500.0], dtype=torch.float32),
            add_noise_to_samples=True,
            scail_pose2_replacement=False,
        )

        self.assertFalse(contract.mask_aware)
        self.assertEqual("noised_samples", contract.subject_source)
        self.assertEqual("noised_samples", contract.preserve_source)
        self.assertEqual(55.0, float(initialized[0, 0, 0, 0].item()))
        self.assertEqual(55.0, float(initialized[0, 0, 1, 1].item()))


if __name__ == "__main__":
    unittest.main()
