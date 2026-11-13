from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "readme.md"
EXAMPLE = (
    ROOT
    / "example_workflows"
    / "wanvideo_2_1_14B_SCAIL2_replacement_and_animate_dual_mode_example_01.json"
)


def _load_example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


class Scail2ExampleWorkflowTests(unittest.TestCase):
    def test_readme_documents_scail_pose2_dual_mode_samples_behavior(self) -> None:
        source = README.read_text(encoding="utf-8")

        self.assertIn("SCAIL-Pose2 dual-mode samples", source)
        self.assertIn("`replacement` keeps", source)
        self.assertIn("`animation`", source)
        self.assertIn("`WanVideoEncode.samples`", source)
        self.assertIn("`WanVideoSampler.samples`", source)
        self.assertIn("`add_noise_to_samples` only matters", source)
        self.assertNotIn(".planning/", source)
        self.assertNotIn("reference/docs", source)

    def test_replacement_example_uses_raw_driving_video_condition_route(self) -> None:
        data = _load_example()
        nodes = {node["id"]: node for node in data["nodes"]}
        links = {link[0]: link for link in data["links"]}

        self.assertNotIn(
            "SCAILPose2ReplacementConditionVideo",
            {node["type"] for node in data["nodes"]},
        )

        condition = nodes[447]
        driving_input = next(
            item for item in condition["inputs"] if item["name"] == "driving_video"
        )
        driving_link = links[driving_input["link"]]

        self.assertEqual([856, 478, 0, 447, 4, "IMAGE"], driving_link)
        self.assertEqual("Get_driving_video", nodes[478]["title"])

    def test_replacement_example_wires_denoise_mask_to_encode_samples_path(self) -> None:
        data = _load_example()
        nodes = {node["id"]: node for node in data["nodes"]}
        links = {link[0]: link for link in data["links"]}

        mask_node = nodes[454]
        mask_output = next(item for item in mask_node["outputs"] if item["name"] == "mask")
        mask_link = links[mask_output["links"][0]]

        self.assertEqual("SCAILPose2ReplacementDenoiseMask", mask_node["type"])
        self.assertEqual([836, 454, 0, 475, 2, "MASK"], mask_link)
        self.assertEqual("WanVideoEncode", nodes[475]["type"])

    def test_replacement_example_keeps_encode_samples_connected_to_sampler(self) -> None:
        data = _load_example()
        nodes = {node["id"]: node for node in data["nodes"]}
        links = {link[0]: link for link in data["links"]}

        condition = nodes[447]
        encode = nodes[475]
        sampler = nodes[348]
        encode_samples_output = next(
            item for item in encode["outputs"] if item["name"] == "samples"
        )
        sampler_samples_input = next(
            item for item in sampler["inputs"] if item["name"] == "samples"
        )
        samples_link = links[encode_samples_output["links"][0]]

        self.assertEqual("replacement", condition["widgets_values"][0])
        self.assertEqual("WanVideoEncode", encode["type"])
        self.assertEqual("WanVideoSamplerv2", sampler["type"])
        self.assertEqual(sampler_samples_input["link"], encode_samples_output["links"][0])
        self.assertEqual([834, 475, 0, 348, 4, "LATENT"], samples_link)


if __name__ == "__main__":
    unittest.main()
