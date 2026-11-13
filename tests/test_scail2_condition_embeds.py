from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        if isinstance(key, tuple) and key and key[-1] == slice(None, 3, None):
            return FakeTensor((*self.shape[:-1], 3))
        if key == 0:
            return FakeTensor(self.shape[1:])
        raise TypeError(f"unsupported fake tensor index: {key!r}")

    def permute(self, *dims):
        return FakeTensor(tuple(self.shape[index] for index in dims))

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def __mul__(self, other):
        return FakeTensor(self.shape)

    def __sub__(self, other):
        return FakeTensor(self.shape)


def install_torch_stub() -> None:
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.device = lambda name: name
    torch.as_tensor = lambda data, dtype=None: data if isinstance(data, FakeTensor) else FakeTensor(())

    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")

    def interpolate(tensor, size, mode=None, align_corners=None):
        frames, channels, _height, _width = tensor.shape
        return FakeTensor((frames, channels, int(size[0]), int(size[1])))

    functional.interpolate = interpolate
    nn.functional = functional
    torch.nn = nn

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional


def install_comfy_stub() -> None:
    install_torch_stub()
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: "cpu"
    model_management.unet_offload_device = lambda: "cpu"
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management


def install_comfy_stub_with_real_torch() -> None:
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: "cpu"
    model_management.unet_offload_device = lambda: "cpu"
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management


def import_scail_nodes():
    preserved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "torch"
        or name.startswith("torch.")
        or name == "comfy"
        or name.startswith("comfy.")
    }
    install_comfy_stub()
    module_name = "wan_scail_nodes_under_test"
    sys.modules.pop(module_name, None)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            ROOT / "SCAIL" / "nodes.py",
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name in tuple(sys.modules):
            if (
                name == "torch"
                or name.startswith("torch.")
                or name == "comfy"
                or name.startswith("comfy.")
            ):
                sys.modules.pop(name, None)
        sys.modules.update(preserved_modules)


def import_scail_nodes_with_real_torch():
    for name in ("comfy", "comfy.model_management"):
        sys.modules.pop(name, None)
    import torch  # noqa: F401

    install_comfy_stub_with_real_torch()
    module_name = "wan_scail_nodes_real_torch_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "SCAIL" / "nodes.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeVAE:
    dtype = "float32"

    def __init__(self) -> None:
        self.to_calls = []
        self.encode_calls = []

    def to(self, target):
        self.to_calls.append(str(target))
        return self

    def encode(self, images, device, tiled=False):
        image = images[0]
        self.encode_calls.append(
            {
                "shape": tuple(image.shape),
                "device": str(device),
                "tiled": tiled,
            }
        )
        _channels, frames, height, width = image.shape
        latent_h = max(height // 8, 1)
        latent_w = max(width // 8, 1)
        return [FakeTensor((16, frames, latent_h, latent_w))]


class RecordingVAE(FakeVAE):
    def __init__(self) -> None:
        super().__init__()
        import torch

        self.dtype = torch.float32

    def encode(self, images, device, tiled=False):
        image = images[0]
        self.encode_calls.append(
            {
                "shape": tuple(image.shape),
                "device": str(device),
                "tiled": tiled,
                "image": image.detach().cpu().clone(),
            }
        )
        _channels, frames, height, width = image.shape
        latent_h = max(height // 8, 1)
        latent_w = max(width // 8, 1)
        return [FakeTensor((16, frames, latent_h, latent_w))]


def image_batch(frames: int, height: int = 8, width: int = 8):
    return FakeTensor((frames, height, width, 3))


def runtime_mask(latent_frames: int):
    data = FakeTensor((1, latent_frames, 28, 1, 1))
    return {
        "data": data,
        "comfy_shape": tuple(data.shape),
        "scail2_shape": (28, latent_frames, 1, 1),
    }


def payload(
    *,
    include_additional: bool = True,
    ref_height: int = 8,
    ref_width: int = 8,
    additional_height: int = 8,
    additional_width: int = 8,
):
    additional_ref = {"image": image_batch(1, additional_height, additional_width)}
    additional_refs = [additional_ref] if include_additional else []
    additional_masks = [runtime_mask(1)] if include_additional else []
    condition = {
        "ref_image": image_batch(1, ref_height, ref_width),
        "pose_video": image_batch(5),
        "additional_references": additional_refs,
    }
    return {
        "kind": "wanvideo_scail2_condition_adapter",
        "version": 1,
        "schema": {
            "name": "scail_pose2.wanvideo_scail2_payload",
            "version": 1,
            "native_wrapper": {
                "embeds_key": "scail2_embeds",
            },
        },
        "condition": condition,
        "mode": "replacement",
        "replace_flag": True,
        "dimensions": {
            "width": 8,
            "height": 8,
            "num_frames": 5,
        },
        "source": {
            "source_kind": "unit_test",
        },
        "runtime_masks": {
            "reference": runtime_mask(1),
            "driving": runtime_mask(2),
            "additional_references": additional_masks,
        },
        "additional_references": additional_refs,
    }


def real_runtime_mask(latent_frames: int):
    import torch

    data = torch.zeros((1, latent_frames, 28, 1, 1), dtype=torch.float32)
    return {
        "data": data,
        "comfy_shape": tuple(data.shape),
        "scail2_shape": (28, latent_frames, 1, 1),
    }


def real_replacement_payload():
    import torch

    ref_mask_indices = torch.full((1, 8, 8), -1, dtype=torch.int8)
    ref_mask_indices[:, :, :4] = 3
    driving_mask_indices = torch.zeros((5, 8, 8), dtype=torch.int8)
    driving_mask_indices[:, :, :4] = 3
    add_mask_indices = torch.full((1, 8, 8), -1, dtype=torch.int8)
    add_mask_indices[:, :4, :] = 3
    additional_ref = {
        "image": torch.ones((1, 8, 8, 3), dtype=torch.float32),
        "mask_indices": add_mask_indices,
    }
    condition = {
        "ref_image": torch.ones((1, 8, 8, 3), dtype=torch.float32),
        "ref_mask_indices": ref_mask_indices,
        "pose_video": torch.ones((5, 8, 8, 3), dtype=torch.float32),
        "driving_mask_indices": driving_mask_indices,
        "additional_references": [additional_ref],
    }
    return {
        "kind": "wanvideo_scail2_condition_adapter",
        "version": 1,
        "schema": {
            "name": "scail_pose2.wanvideo_scail2_payload",
            "version": 1,
            "native_wrapper": {
                "embeds_key": "scail2_embeds",
            },
        },
        "condition": condition,
        "mode": "replacement",
        "replace_flag": True,
        "dimensions": {
            "width": 8,
            "height": 8,
            "num_frames": 5,
        },
        "source": {
            "source_kind": "unit_test",
        },
        "runtime_masks": {
            "reference": real_runtime_mask(1),
            "driving": real_runtime_mask(2),
            "additional_references": [real_runtime_mask(1)],
        },
        "additional_references": [additional_ref],
    }


def real_condition_video_leak_payload(*, replace_flag: bool = True):
    import torch

    driving_mask_indices = torch.zeros((5, 8, 8), dtype=torch.int8)
    driving_mask_indices[:, :, :4] = 3
    pose_video = torch.ones((5, 8, 8, 3), dtype=torch.float32)
    pose_video[:, :, :4] = 0.0
    condition = {
        "ref_image": torch.ones((1, 8, 8, 3), dtype=torch.float32),
        "ref_mask_indices": torch.zeros((1, 8, 8), dtype=torch.int8),
        "pose_video": pose_video,
        "driving_mask_indices": driving_mask_indices,
        "additional_references": [],
    }
    return {
        "kind": "wanvideo_scail2_condition_adapter",
        "version": 1,
        "schema": {
            "name": "scail_pose2.wanvideo_scail2_payload",
            "version": 1,
            "native_wrapper": {
                "embeds_key": "scail2_embeds",
            },
        },
        "condition": condition,
        "mode": "replacement" if replace_flag else "animation",
        "replace_flag": replace_flag,
        "dimensions": {
            "width": 8,
            "height": 8,
            "num_frames": 5,
        },
        "source": {
            "source_kind": "unit_test",
        },
        "runtime_masks": {
            "reference": real_runtime_mask(1),
            "driving": real_runtime_mask(2),
            "additional_references": [],
        },
        "additional_references": [],
    }


def real_condition_video_structure_payload():
    import torch

    driving_mask_indices = torch.zeros((5, 8, 8), dtype=torch.int8)
    driving_mask_indices[:, :, :4] = 3
    pose_video = torch.ones((5, 8, 8, 3), dtype=torch.float32)
    pose_video[:, :, :2] = 0.05
    pose_video[:, :, 2:4] = 0.95
    condition = {
        "ref_image": torch.ones((1, 8, 8, 3), dtype=torch.float32),
        "ref_mask_indices": torch.zeros((1, 8, 8), dtype=torch.int8),
        "pose_video": pose_video,
        "driving_mask_indices": driving_mask_indices,
        "additional_references": [],
    }
    return {
        "kind": "wanvideo_scail2_condition_adapter",
        "version": 1,
        "schema": {
            "name": "scail_pose2.wanvideo_scail2_payload",
            "version": 1,
            "native_wrapper": {
                "embeds_key": "scail2_embeds",
            },
        },
        "condition": condition,
        "mode": "replacement",
        "replace_flag": True,
        "dimensions": {
            "width": 8,
            "height": 8,
            "num_frames": 5,
        },
        "source": {
            "source_kind": "unit_test",
        },
        "runtime_masks": {
            "reference": real_runtime_mask(1),
            "driving": real_runtime_mask(2),
            "additional_references": [],
        },
        "additional_references": [],
    }


class WanVideoAddSCAIL2ConditionEmbedsTests(unittest.TestCase):
    def test_node_contract_is_registered(self) -> None:
        module = import_scail_nodes()

        self.assertIn("WanVideoAddSCAIL2ConditionEmbeds", module.NODE_CLASS_MAPPINGS)
        self.assertIn("WanVideoAddSCAILPoseEmbeds", module.NODE_CLASS_MAPPINGS)
        self.assertIn("WanVideoAddSCAILReferenceEmbeds", module.NODE_CLASS_MAPPINGS)

        node_cls = module.NODE_CLASS_MAPPINGS["WanVideoAddSCAIL2ConditionEmbeds"]
        self.assertEqual(("WANVIDIMAGE_EMBEDS",), node_cls.RETURN_TYPES)
        self.assertEqual(("image_embeds",), node_cls.RETURN_NAMES)
        self.assertEqual(
            "SCAIL2_WANVIDEO_PAYLOAD",
            node_cls.INPUT_TYPES()["required"]["condition"][0],
        )
        required = node_cls.INPUT_TYPES()["required"]
        self.assertIn("ref_image_strength", required)
        self.assertIn("ref_mask_strength", required)
        self.assertIn("condition_video_strength", required)
        self.assertIn("driving_mask_strength", required)
        self.assertEqual(1.0, required["ref_image_strength"][1]["default"])
        self.assertEqual(1.0, required["ref_mask_strength"][1]["default"])
        self.assertEqual(1.0, required["condition_video_strength"][1]["default"])
        self.assertEqual(1.0, required["driving_mask_strength"][1]["default"])

    def test_add_materializes_scail2_embeds(self) -> None:
        module = import_scail_nodes()
        vae = FakeVAE()
        embeds = {
            "target_shape": (16, 2, 1, 1),
            "num_frames": 5,
        }
        clip_embeds = {"clip_embeds": object()}

        result = module.WanVideoAddSCAIL2ConditionEmbeds().add(
            embeds,
            payload(),
            vae,
            clip_embeds=clip_embeds,
        )[0]

        self.assertIsNot(result, embeds)
        self.assertNotIn("scail_embeds", result)
        self.assertIn("scail2_embeds", result)

        scail2 = result["scail2_embeds"]
        self.assertTrue(scail2["replace_flag"])
        self.assertEqual("replacement", scail2["mode"])
        self.assertEqual([(16, 1, 1, 1)], [x.shape for x in scail2["ref_latents"]])
        self.assertEqual([(16, 5, 1, 1)], [x.shape for x in scail2["pose_latents"]])
        self.assertEqual([(28, 1, 1, 1)], [x.shape for x in scail2["ref_masks"]])
        self.assertEqual([(28, 2, 1, 1)], [x.shape for x in scail2["driving_masks"]])
        self.assertEqual([(16, 1, 1, 1)], [x.shape for x in scail2["additional_ref_latents"]])
        self.assertEqual([(28, 1, 1, 1)], [x.shape for x in scail2["additional_ref_masks"]])
        self.assertIs(scail2["clip_context"], clip_embeds["clip_embeds"])
        self.assertIs(result["clip_context"], clip_embeds["clip_embeds"])
        self.assertIsNone(scail2["segment"])
        self.assertEqual({"source_kind": "unit_test"}, scail2["source"])
        self.assertEqual(
            {
                "ref_image": 1.0,
                "ref_mask": 1.0,
                "condition_video": 1.0,
                "driving_mask": 1.0,
            },
            scail2["strengths"],
        )
        self.assertEqual((3, 1, 8, 8), vae.encode_calls[0]["shape"])
        self.assertEqual((3, 5, 4, 4), vae.encode_calls[1]["shape"])

    def test_add_stores_non_default_strength_metadata(self) -> None:
        module = import_scail_nodes()
        result = module.WanVideoAddSCAIL2ConditionEmbeds().add(
            {
                "target_shape": (16, 2, 1, 1),
                "num_frames": 5,
            },
            payload(),
            FakeVAE(),
            ref_image_strength=1.4,
            ref_mask_strength=0.7,
            condition_video_strength=0.5,
            driving_mask_strength=1.8,
        )[0]

        self.assertEqual(
            {
                "ref_image": 1.4,
                "ref_mask": 0.7,
                "condition_video": 0.5,
                "driving_mask": 1.8,
            },
            result["scail2_embeds"]["strengths"],
        )

    def test_rejects_invalid_strength_metadata(self) -> None:
        module = import_scail_nodes()

        with self.assertRaisesRegex(ValueError, "ref_image_strength"):
            module.WanVideoAddSCAIL2ConditionEmbeds().add(
                {"target_shape": (16, 2, 1, 1), "num_frames": 5},
                payload(),
                FakeVAE(),
                ref_image_strength=-0.1,
            )

    def test_references_are_resized_to_payload_dimensions_before_encode(self) -> None:
        module = import_scail_nodes()
        vae = FakeVAE()
        embeds = {
            "target_shape": (16, 2, 1, 1),
            "num_frames": 5,
        }

        module.WanVideoAddSCAIL2ConditionEmbeds().add(
            embeds,
            payload(ref_height=16, ref_width=12, additional_height=10, additional_width=14),
            vae,
        )

        self.assertEqual((3, 1, 8, 8), vae.encode_calls[0]["shape"])
        self.assertEqual((3, 5, 4, 4), vae.encode_calls[1]["shape"])
        self.assertEqual((3, 1, 8, 8), vae.encode_calls[2]["shape"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_replacement_mode_composites_reference_images_before_encode(self) -> None:
        module = import_scail_nodes_with_real_torch()
        vae = RecordingVAE()
        embeds = {
            "target_shape": (16, 2, 1, 1),
            "num_frames": 5,
        }

        module.WanVideoAddSCAIL2ConditionEmbeds().add(
            embeds,
            real_replacement_payload(),
            vae,
        )

        primary_ref = vae.encode_calls[0]["image"]
        additional_ref = vae.encode_calls[2]["image"]
        self.assertEqual(1.0, float(primary_ref[0, 0, 0, 0].item()))
        self.assertEqual(-1.0, float(primary_ref[0, 0, 0, 7].item()))
        self.assertEqual(1.0, float(additional_ref[0, 0, 0, 0].item()))
        self.assertEqual(-1.0, float(additional_ref[0, 0, 7, 0].item()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_replacement_mode_preserves_raw_condition_video_before_pose_encode(
        self,
    ) -> None:
        module = import_scail_nodes_with_real_torch()
        vae = RecordingVAE()

        module.WanVideoAddSCAIL2ConditionEmbeds().add(
            {
                "target_shape": (16, 2, 1, 1),
                "num_frames": 5,
            },
            real_condition_video_leak_payload(replace_flag=True),
            vae,
        )

        encoded_pose_input = vae.encode_calls[1]["image"]
        subject_region = encoded_pose_input[:, :, :, :2]
        background_region = encoded_pose_input[:, :, :, 2:]
        self.assertLess(float(subject_region[0].max().item()), -0.9)
        self.assertLess(float(subject_region[1].max().item()), -0.9)
        self.assertLess(float(subject_region[2].max().item()), -0.9)
        self.assertGreater(float(background_region[0].min().item()), 0.9)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_replacement_mode_preserves_raw_subject_rgb_gradient(self) -> None:
        module = import_scail_nodes_with_real_torch()
        vae = RecordingVAE()

        module.WanVideoAddSCAIL2ConditionEmbeds().add(
            {
                "target_shape": (16, 2, 1, 1),
                "num_frames": 5,
            },
            real_condition_video_structure_payload(),
            vae,
        )

        encoded_pose_input = vae.encode_calls[1]["image"]
        subject_column_left = encoded_pose_input[:, :, :, 0]
        subject_column_right = encoded_pose_input[:, :, :, 1]
        self.assertLess(float(subject_column_left.max().item()), -0.85)
        self.assertGreater(float(subject_column_right.min().item()), 0.85)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_animation_mode_does_not_sanitize_condition_video(self) -> None:
        module = import_scail_nodes_with_real_torch()
        vae = RecordingVAE()

        module.WanVideoAddSCAIL2ConditionEmbeds().add(
            {
                "target_shape": (16, 2, 1, 1),
                "num_frames": 5,
            },
            real_condition_video_leak_payload(replace_flag=False),
            vae,
        )

        encoded_pose_input = vae.encode_calls[1]["image"]
        self.assertLess(float(encoded_pose_input[:, :, :, :2].min().item()), -0.9)

    def test_rejects_invalid_payload(self) -> None:
        module = import_scail_nodes()
        bad_payload = payload()
        bad_payload["schema"] = {"name": "wrong", "version": 1, "native_wrapper": {"embeds_key": "scail2_embeds"}}

        with self.assertRaisesRegex(ValueError, "schema"):
            module.WanVideoAddSCAIL2ConditionEmbeds().add(
                {"target_shape": (16, 2, 1, 1)},
                bad_payload,
                FakeVAE(),
            )

    def test_rejects_existing_scail_embeds(self) -> None:
        module = import_scail_nodes()

        with self.assertRaisesRegex(ValueError, "v1 scail_embeds"):
            module.WanVideoAddSCAIL2ConditionEmbeds().add(
                {"target_shape": (16, 2, 1, 1), "scail_embeds": {}},
                payload(),
                FakeVAE(),
            )

    def test_rejects_target_shape_mismatch(self) -> None:
        module = import_scail_nodes()

        with self.assertRaisesRegex(ValueError, "dimensions"):
            module.WanVideoAddSCAIL2ConditionEmbeds().add(
                {"target_shape": (16, 2, 1, 2)},
                payload(),
                FakeVAE(),
            )

    def test_scail2_example_workflow_contains_strength_widgets(self) -> None:
        path = (
            ROOT
            / "example_workflows"
            / "wanvideo_2_1_14B_SCAIL2_replacement_and_animate_dual_mode_example_01.json"
        )
        workflow = json.loads(path.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in workflow["nodes"]
            if node.get("type") == "WanVideoAddSCAIL2ConditionEmbeds"
        ]

        self.assertTrue(nodes)
        for node in nodes:
            self.assertEqual(4, len(node["widgets_values"]))
            for value in node["widgets_values"]:
                self.assertIsInstance(value, (int, float))
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 10.0)


if __name__ == "__main__":
    unittest.main()
