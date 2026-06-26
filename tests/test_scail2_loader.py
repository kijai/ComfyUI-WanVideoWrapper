from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_loader_module():
    spec = importlib.util.spec_from_file_location(
        "scail2_loader_under_test",
        ROOT / "SCAIL" / "scail2_loader.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Weight:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeConv3d:
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)


class FakeNN:
    Conv3d = FakeConv3d


class FakeLog:
    def __init__(self) -> None:
        self.messages = []

    def info(self, message) -> None:
        self.messages.append(str(message))


class Transformer:
    pass


class SCAIL2LoaderTests(unittest.TestCase):
    def test_v1_pose_embedding_still_patches_without_scail2_mask(self) -> None:
        module = import_loader_module()
        transformer = Transformer()

        module.apply_scail_loader_patches(
            transformer,
            {"patch_embedding_pose.weight": Weight((5120, 16, 1, 2, 2))},
            FakeNN,
            dim=5120,
            patch_size=(1, 2, 2),
            log=FakeLog(),
        )

        self.assertEqual(16, transformer.patch_embedding_pose.in_channels)
        self.assertEqual(5120, transformer.patch_embedding_pose.out_channels)
        self.assertFalse(transformer.scail2_enabled)
        self.assertIsNone(transformer.scail2_mask_dim)

    def test_scail2_mask_embedding_patches_with_28_channels(self) -> None:
        module = import_loader_module()
        transformer = Transformer()

        module.apply_scail_loader_patches(
            transformer,
            {
                "patch_embedding_pose.weight": Weight((5120, 16, 1, 2, 2)),
                "patch_embedding_mask.weight": Weight((5120, 28, 1, 2, 2)),
            },
            FakeNN,
            dim=5120,
            patch_size=(1, 2, 2),
            log=FakeLog(),
        )

        self.assertEqual(28, transformer.patch_embedding_mask.in_channels)
        self.assertEqual(5120, transformer.patch_embedding_mask.out_channels)
        self.assertTrue(transformer.scail2_enabled)
        self.assertEqual(28, transformer.scail2_mask_dim)

    def test_non_scail_model_sets_negative_capability_metadata(self) -> None:
        module = import_loader_module()
        transformer = Transformer()

        module.apply_scail_loader_patches(
            transformer,
            {},
            FakeNN,
            dim=5120,
            patch_size=(1, 2, 2),
            log=FakeLog(),
        )

        self.assertFalse(transformer.scail2_enabled)
        self.assertIsNone(transformer.scail2_mask_dim)
        self.assertFalse(hasattr(transformer, "patch_embedding_pose"))
        self.assertFalse(hasattr(transformer, "patch_embedding_mask"))

    def test_rejects_mask_embedding_with_wrong_channel_count(self) -> None:
        module = import_loader_module()

        with self.assertRaisesRegex(ValueError, "28 input channels"):
            module.apply_scail_loader_patches(
                Transformer(),
                {"patch_embedding_mask.weight": Weight((5120, 27, 1, 2, 2))},
                FakeNN,
                dim=5120,
                patch_size=(1, 2, 2),
                log=FakeLog(),
            )

    def test_rejects_inconsistent_mask_embedding_shape(self) -> None:
        module = import_loader_module()

        with self.assertRaisesRegex(ValueError, "5D Conv3d"):
            module.apply_scail_loader_patches(
                Transformer(),
                {"patch_embedding_mask.weight": Weight((5120, 28))},
                FakeNN,
                dim=5120,
                patch_size=(1, 2, 2),
                log=FakeLog(),
            )

        with self.assertRaisesRegex(ValueError, "patch_size"):
            module.apply_scail_loader_patches(
                Transformer(),
                {"patch_embedding_mask.weight": Weight((5120, 28, 1, 1, 1))},
                FakeNN,
                dim=5120,
                patch_size=(1, 2, 2),
                log=FakeLog(),
            )


if __name__ == "__main__":
    unittest.main()
