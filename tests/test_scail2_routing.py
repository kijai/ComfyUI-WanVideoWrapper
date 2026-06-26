from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_routing_module():
    spec = importlib.util.spec_from_file_location(
        "scail2_routing_under_test",
        ROOT / "SCAIL" / "scail2_routing.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTemporalTensor:
    def __init__(self, shape, name="tensor"):
        self.shape = tuple(shape)
        self.name = name

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"unsupported fake tensor index: {key!r}")
        channel_index, frame_index = key
        if channel_index != slice(None):
            raise TypeError(f"unsupported fake tensor channel index: {key!r}")
        if isinstance(frame_index, slice):
            frame_count = len(range(*frame_index.indices(self.shape[1])))
        else:
            frame_count = len(frame_index)
        return FakeTemporalTensor(
            (self.shape[0], frame_count, *self.shape[2:]),
            name=f"{self.name}_sliced",
        )


class SCAIL2RoutingTests(unittest.TestCase):
    def test_prepare_returns_none_without_scail2_embeds(self) -> None:
        module = import_routing_module()

        self.assertIsNone(
            module.prepare_scail2_data(
                {"target_shape": (16, 2, 1, 1)},
                dict_to_device=lambda data, device, dtype: data,
                device="cuda",
                dtype="float16",
            )
        )

    def test_prepare_copies_and_moves_scail2_embeds(self) -> None:
        module = import_routing_module()
        original = {"pose_latents": object()}
        image_embeds = {"scail2_embeds": original}
        calls = []

        def fake_dict_to_device(data, device, dtype):
            calls.append((data, device, dtype))
            data["moved"] = True
            return data

        result = module.prepare_scail2_data(
            image_embeds,
            dict_to_device=fake_dict_to_device,
            device="cuda:0",
            dtype="bfloat16",
        )

        self.assertIsNot(result, original)
        self.assertEqual(1, len(calls))
        self.assertEqual("cuda:0", calls[0][1])
        self.assertEqual("bfloat16", calls[0][2])
        self.assertNotIn("moved", original)
        self.assertIs(result["pose_latents"], original["pose_latents"])

    def test_prepare_rejects_v1_and_v2_together(self) -> None:
        module = import_routing_module()

        with self.assertRaisesRegex(ValueError, "v1 scail_embeds"):
            module.prepare_scail2_data(
                {"scail_embeds": {}, "scail2_embeds": {}},
                dict_to_device=lambda data, device, dtype: data,
                device="cuda",
                dtype="float16",
            )

    def test_prepare_rejects_non_dict_scail2_embeds(self) -> None:
        module = import_routing_module()

        with self.assertRaisesRegex(TypeError, "must be a dict"):
            module.prepare_scail2_data(
                {"scail2_embeds": object()},
                dict_to_device=lambda data, device, dtype: data,
                device="cuda",
                dtype="float16",
            )

    def test_context_window_none_preserves_scail2_native_payload(self) -> None:
        module = import_routing_module()
        scail2_data = {"pose_latents": object()}

        self.assertIs(scail2_data, module.scail2_context_window_input(scail2_data, None))

    def test_context_window_slices_temporal_controls_only(self) -> None:
        module = import_routing_module()
        ref_latents = [FakeTemporalTensor((16, 1, 2, 2), name="ref_latents")]
        ref_masks = [FakeTemporalTensor((28, 1, 2, 2), name="ref_masks")]
        additional_ref_latents = [
            FakeTemporalTensor((16, 1, 2, 2), name="additional_ref_latents")
        ]
        additional_ref_masks = [
            FakeTemporalTensor((28, 1, 2, 2), name="additional_ref_masks")
        ]
        pose_latents = [FakeTemporalTensor((16, 5, 2, 2), name="pose_latents")]
        driving_masks = [FakeTemporalTensor((28, 5, 2, 2), name="driving_masks")]
        scail2_data = {
            "schema": {"version": 1},
            "ref_latents": ref_latents,
            "ref_masks": ref_masks,
            "additional_ref_latents": additional_ref_latents,
            "additional_ref_masks": additional_ref_masks,
            "pose_latents": pose_latents,
            "driving_masks": driving_masks,
        }

        result = module.scail2_context_window_input(scail2_data, [0, 2, 4])

        self.assertIsNot(result, scail2_data)
        self.assertIs(result["schema"], scail2_data["schema"])
        self.assertIs(result["ref_latents"], ref_latents)
        self.assertIs(result["ref_masks"], ref_masks)
        self.assertIs(result["additional_ref_latents"], additional_ref_latents)
        self.assertIs(result["additional_ref_masks"], additional_ref_masks)
        self.assertIs(scail2_data["pose_latents"], pose_latents)
        self.assertIs(scail2_data["driving_masks"], driving_masks)
        self.assertEqual((16, 3, 2, 2), result["pose_latents"][0].shape)
        self.assertEqual((28, 3, 2, 2), result["driving_masks"][0].shape)
        self.assertEqual((16, 5, 2, 2), scail2_data["pose_latents"][0].shape)
        self.assertEqual((28, 5, 2, 2), scail2_data["driving_masks"][0].shape)

    def test_context_window_accepts_slice_objects(self) -> None:
        module = import_routing_module()
        scail2_data = {
            "pose_latents": [FakeTemporalTensor((16, 5, 2, 2))],
            "driving_masks": [FakeTemporalTensor((28, 5, 2, 2))],
        }

        result = module.scail2_context_window_input(scail2_data, slice(1, 5, 2))

        self.assertEqual((16, 2, 2, 2), result["pose_latents"][0].shape)
        self.assertEqual((28, 2, 2, 2), result["driving_masks"][0].shape)

    def test_context_window_rejects_pose_mask_frame_count_mismatch(self) -> None:
        module = import_routing_module()
        scail2_data = {
            "pose_latents": [FakeTemporalTensor((16, 5, 2, 2), name="pose_latents")],
            "driving_masks": [FakeTemporalTensor((28, 4, 2, 2), name="driving_masks")],
        }

        with self.assertRaisesRegex(ValueError, "pose_latents frames=.*driving_masks"):
            module.scail2_context_window_input(scail2_data, [0, 1, 2])

    def test_context_window_rejects_out_of_range_indices_before_slicing(self) -> None:
        module = import_routing_module()
        scail2_data = {
            "pose_latents": [FakeTemporalTensor((16, 5, 2, 2), name="pose_latents")],
            "driving_masks": [FakeTemporalTensor((28, 5, 2, 2), name="driving_masks")],
        }

        with self.assertRaisesRegex(ValueError, "field=pose_latents.*frame_count=5"):
            module.scail2_context_window_input(scail2_data, [0, 4, 5])

    def test_context_window_rejects_scalar_window(self) -> None:
        module = import_routing_module()
        scail2_data = {"pose_latents": [FakeTemporalTensor((16, 5, 2, 2))]}

        with self.assertRaisesRegex(TypeError, "sequence of frame indices"):
            module.scail2_context_window_input(scail2_data, 0)

    def test_model_params_receive_scail2_input_only_when_present(self) -> None:
        module = import_routing_module()
        base_params = {"scail_input": None}
        scail2_data = {"pose_latents": object()}

        result = module.add_scail2_model_param(base_params, scail2_data)

        self.assertIs(result, base_params)
        self.assertIs(result["scail2_input"], scail2_data)

        no_scail2 = {"scail_input": None}
        module.add_scail2_model_param(no_scail2, None)
        self.assertNotIn("scail2_input", no_scail2)


if __name__ == "__main__":
    unittest.main()
