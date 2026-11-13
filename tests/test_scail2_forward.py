from __future__ import annotations

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_forward_module():
    spec = importlib.util.spec_from_file_location(
        "scail2_forward_under_test",
        ROOT / "SCAIL" / "scail2_forward.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakePatchEmbedding:
    def __init__(self, in_channels):
        self.weight = FakeTensor((8, in_channels, 1, 2, 2))


class FakeLatent:
    def __init__(self, shape, *, fill=1.0, parts=(), writes=()):
        self.shape = tuple(shape)
        self.fill = fill
        self.parts = tuple(parts)
        self.writes = tuple(writes)

    def new_zeros(self, shape):
        return FakeLatent(shape, fill=0.0)

    def new_full(self, shape, fill_value):
        return FakeLatent(shape, fill=float(fill_value))

    def clone(self):
        return FakeLatent(self.shape, fill=self.fill, parts=self.parts, writes=self.writes)

    def __setitem__(self, key, value):
        self.writes = (*self.writes, (key, value))

    def __mul__(self, value):
        return FakeLatent(self.shape, fill=self.fill * float(value), parts=(self,))


class FakeTorchModule:
    @staticmethod
    def cat(items, dim=0):
        items = tuple(items)
        shape = list(items[0].shape)
        shape[dim] = sum(item.shape[dim] for item in items)
        return FakeLatent(tuple(shape), parts=items)


@contextmanager
def fake_torch_module():
    sentinel = object()
    previous = sys.modules.get("torch", sentinel)
    sys.modules["torch"] = FakeTorchModule()
    try:
        yield
    finally:
        if previous is sentinel:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous


def plan(scail2_input, video_shape=(16, 2, 4, 4)):
    module = import_forward_module()
    return module.build_scail2_forward_plan(
        scail2_input,
        video_shape=video_shape,
        patch_size=(1, 2, 2),
    )


class SCAIL2ForwardPlanTests(unittest.TestCase):
    def test_history_channels_are_appended_for_twenty_channel_patch_embedding(self) -> None:
        module = import_forward_module()
        embedding = FakePatchEmbedding(20)
        latent = FakeLatent((16, 2, 4, 4))

        with fake_torch_module():
            expanded = module.append_scail2_history_channels(
                latent,
                patch_embedding=embedding,
            )

        self.assertEqual((20, 2, 4, 4), tuple(expanded.shape))
        self.assertIs(latent, expanded.parts[0])
        self.assertEqual((4, 2, 4, 4), expanded.parts[1].shape)
        self.assertEqual(0.0, expanded.parts[1].fill)

    def test_history_channels_can_be_filled_for_reference_or_pose_markers(self) -> None:
        module = import_forward_module()
        embedding = FakePatchEmbedding(20)
        latent = FakeLatent((16, 2, 4, 4))

        with fake_torch_module():
            expanded = module.append_scail2_history_channels(
                latent,
                patch_embedding=embedding,
                fill_value=1.0,
            )

        self.assertEqual((20, 2, 4, 4), tuple(expanded.shape))
        self.assertEqual((4, 2, 4, 4), expanded.parts[1].shape)
        self.assertEqual(1.0, expanded.parts[1].fill)

    def test_reference_prefix_history_channels_are_marked(self) -> None:
        module = import_forward_module()
        embedding = FakePatchEmbedding(20)
        latent = FakeLatent((20, 5, 4, 4))

        marked = module.mark_scail2_prefix_history_channels(
            latent,
            prefix_frames=2,
            patch_embedding=embedding,
        )

        self.assertIsNot(latent, marked)
        self.assertEqual((20, 5, 4, 4), marked.shape)
        self.assertEqual(1, len(marked.writes))
        self.assertEqual((slice(-4, None, None), slice(None, 2, None)), marked.writes[0][0])
        self.assertEqual(1.0, marked.writes[0][1])

    def test_history_channels_are_noop_when_channels_already_match(self) -> None:
        module = import_forward_module()
        embedding = FakePatchEmbedding(20)
        latent = FakeLatent((20, 2, 4, 4))

        self.assertIs(
            latent,
            module.append_scail2_history_channels(
                latent,
                patch_embedding=embedding,
            ),
        )

    def test_history_channels_reject_unexpected_channel_gap(self) -> None:
        module = import_forward_module()
        embedding = FakePatchEmbedding(20)
        latent = FakeLatent((15, 2, 4, 4))

        with self.assertRaisesRegex(ValueError, "expects 20, got 15"):
            module.append_scail2_history_channels(
                latent,
                patch_embedding=embedding,
            )

    def test_mask_only_control_is_not_dropped(self) -> None:
        result = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "driving_masks": [FakeTensor((28, 2, 2, 2))],
            }
        )

        self.assertFalse(result["has_pose"])
        self.assertTrue(result["has_control_mask"])
        self.assertGreater(result["control_length"], 0)
        self.assertEqual(result["main_length"] + result["control_length"], result["total_length"])

    def test_pose_only_control_is_planned_without_mask_control(self) -> None:
        result = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "pose_latents": [FakeTensor((16, 2, 2, 2))],
            }
        )

        self.assertTrue(result["has_pose"])
        self.assertFalse(result["has_control_mask"])
        self.assertEqual((16, 2, 2, 2), result["control_shape"])
        self.assertGreater(result["control_length"], 0)

    def test_pose_and_mask_share_control_shape(self) -> None:
        result = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "pose_latents": [FakeTensor((16, 2, 2, 2))],
                "driving_masks": [FakeTensor((28, 2, 2, 2))],
            }
        )

        self.assertTrue(result["has_pose"])
        self.assertTrue(result["has_control_mask"])
        self.assertEqual((16, 2, 2, 2), result["control_shape"])

        with self.assertRaisesRegex(ValueError, "share temporal/spatial"):
            plan(
                {
                    "pose_latents": [FakeTensor((16, 2, 2, 2))],
                    "driving_masks": [FakeTensor((28, 3, 2, 2))],
                }
            )

    def test_replace_flag_changes_rope_shifts_and_cache_key(self) -> None:
        animation = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "pose_latents": [FakeTensor((16, 2, 2, 2))],
                "replace_flag": False,
            }
        )
        replacement = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "pose_latents": [FakeTensor((16, 2, 2, 2))],
                "replace_flag": True,
            }
        )

        self.assertNotEqual(animation["cache_key"], replacement["cache_key"])
        self.assertEqual(1, animation["rope_shifts"]["t"]["video"])
        self.assertEqual(0, replacement["rope_shifts"]["t"]["video"])
        self.assertEqual(0.0, animation["rope_shifts"]["h"]["ref"])
        self.assertEqual(120.0, replacement["rope_shifts"]["h"]["ref"])

    def test_additional_references_affect_order_and_lengths(self) -> None:
        result = plan(
            {
                "additional_ref_latents": [
                    FakeTensor((16, 1, 4, 4)),
                    FakeTensor((16, 1, 4, 4)),
                ],
                "additional_ref_masks": [
                    FakeTensor((28, 1, 4, 4)),
                    FakeTensor((28, 1, 4, 4)),
                ],
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "driving_masks": [FakeTensor((28, 2, 2, 2))],
            }
        )

        self.assertEqual(3, result["prefix_frames"])
        self.assertEqual(2, result["additional_ref_count"])
        self.assertGreater(result["additional_ref_length"], 0)
        self.assertEqual(2, result["rope_shifts"]["t"]["ref"])
        self.assertEqual(3, result["rope_shifts"]["t"]["video"])

    def test_additional_reference_masks_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "additional_ref_masks is required"):
            plan({"additional_ref_latents": [FakeTensor((16, 1, 4, 4))]})

        with self.assertRaisesRegex(ValueError, "additional_ref_masks requires"):
            plan({"additional_ref_masks": [FakeTensor((28, 1, 4, 4))]})

    def test_scail2_strengths_default_and_partial_override(self) -> None:
        module = import_forward_module()

        self.assertEqual(
            {
                "ref_image": 1.0,
                "ref_mask": 1.0,
                "condition_video": 1.0,
                "driving_mask": 1.0,
            },
            module.scail2_strengths({}),
        )
        self.assertEqual(
            {
                "ref_image": 1.5,
                "ref_mask": 1.0,
                "condition_video": 0.25,
                "driving_mask": 1.0,
            },
            module.scail2_strengths(
                {"strengths": {"ref_image": 1.5, "condition_video": 0.25}}
            ),
        )

    def test_scail2_strengths_validate_range_and_shape(self) -> None:
        module = import_forward_module()

        with self.assertRaisesRegex(ValueError, "must be a dict"):
            module.scail2_strengths({"strengths": 1.0})
        with self.assertRaisesRegex(ValueError, "ref_image"):
            module.scail2_strengths({"strengths": {"ref_image": -0.01}})
        with self.assertRaisesRegex(ValueError, "driving_mask"):
            module.scail2_strengths({"strengths": {"driving_mask": 10.01}})
        with self.assertRaisesRegex(ValueError, "ref_mask"):
            module.scail2_strengths({"strengths": {"ref_mask": True}})

    def test_scale_scail2_items_preserves_default_and_scales_non_default(self) -> None:
        module = import_forward_module()
        latent = FakeLatent((16, 1, 4, 4), fill=2.0)

        self.assertIs(module.scale_scail2_items([latent], 1.0)[0], latent)

        scaled = module.scale_scail2_items([latent], 1.5)[0]
        self.assertIsNot(scaled, latent)
        self.assertEqual(3.0, scaled.fill)
        self.assertIs(scaled.parts[0], latent)

    def test_model_keeps_v1_and_v2_forward_branches_distinct(self) -> None:
        model_source = (ROOT / "wanvideo" / "modules" / "model.py").read_text(encoding="utf-8")

        self.assertIn("scail_input=None,  # SCAIL pose", model_source)
        self.assertIn("scail2_input=None,  # SCAIL-2 native pose/mask conditioning", model_source)
        self.assertIn("# SCAIL pose", model_source)
        self.assertIn("if scail_input is not None:", model_source)
        self.assertIn("if scail2_input is not None:", model_source)
        self.assertIn("append_scail2_history_channels", model_source)
        self.assertIn("mark_scail2_prefix_history_channels", model_source)
        self.assertIn("scail2_strengths", model_source)
        self.assertIn("scale_scail2_items", model_source)

    def test_model_pads_scail2_pose_latents_before_pose_patch_embedding(self) -> None:
        model_source = (ROOT / "wanvideo" / "modules" / "model.py").read_text(encoding="utf-8")

        self.assertIn("append_scail2_history_channels(\n                    scail2_pose_latents", model_source)
        self.assertIn("patch_embedding=self.patch_embedding_pose", model_source)
        self.assertIn('name="SCAIL-2 pose latent"', model_source)
        self.assertIn("fill_value=1.0", model_source)

    def test_model_marks_scail2_reference_prefix_before_patch_embedding(self) -> None:
        model_source = (ROOT / "wanvideo" / "modules" / "model.py").read_text(encoding="utf-8")

        self.assertIn("mark_scail2_prefix_history_channels(", model_source)
        self.assertIn('prefix_frames=scail2_plan["prefix_frames"]', model_source)
        self.assertIn('name="SCAIL-2 main latent"', model_source)

    def test_model_applies_scail2_strengths_to_native_groups(self) -> None:
        model_source = (ROOT / "wanvideo" / "modules" / "model.py").read_text(encoding="utf-8")

        self.assertIn("strengths = scail2_strengths(scail2_input)", model_source)
        self.assertIn('scale_scail2_items(ref_latents, strengths["ref_image"])', model_source)
        self.assertIn('strengths["ref_mask"]', model_source)
        self.assertIn('strengths["condition_video"]', model_source)
        self.assertIn('strengths["driving_mask"]', model_source)


if __name__ == "__main__":
    unittest.main()
