from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_nlf_bbox_module():
    spec = importlib.util.spec_from_file_location(
        "mtv_nlf_bbox_under_test",
        ROOT / "MTV" / "nlf_bbox.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MTVNLFPredictBBoxTests(unittest.TestCase):
    def test_formatter_preserves_multi_person_candidates(self) -> None:
        module = import_nlf_bbox_module()

        formatted = module.format_nlf_detected_boxes(
            [
                [[1, 2, 4, 6, 0.9], [10, 20, 15, 26, 0.8]],
                [[7, 8, 8, 10, 0.7]],
                [],
            ]
        )

        self.assertEqual(
            [[1.0, 2.0, 4.0, 6.0], [10.0, 20.0, 15.0, 26.0]],
            formatted[0],
        )
        self.assertEqual([7.0, 8.0, 8.0, 10.0], formatted[1])
        self.assertEqual([0.0, 0.0, 0.0, 0.0], formatted[2])


if __name__ == "__main__":
    unittest.main()
