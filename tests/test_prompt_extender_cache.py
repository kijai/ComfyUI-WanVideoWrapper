"""Offline regression tests for the cached text node's prompt extension.

Run with ``python -m unittest discover -s tests -p 'test_prompt_extender_cache.py'``.
The production class is compiled directly from nodes.py so the cache behavior
is exercised without importing unrelated ComfyUI/CUDA nodes. Model loading,
generation and the later disk-cache lookup are isolated test doubles.
"""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


NODES_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
PACKAGE = "_wan_prompt_cache_tests"


def load_cached_node(cache_lookup):
    module = ast.parse(NODES_PATH.read_text(encoding="utf-8"), filename=str(NODES_PATH))
    node = next(
        item
        for item in module.body
        if isinstance(item, ast.ClassDef) and item.name == "WanVideoTextEncodeCached"
    )
    namespace = {
        "__name__": f"{PACKAGE}.nodes",
        "__package__": PACKAGE,
        "_extender_cache": {},
        "ProgressBar": Mock(),
        "log": Mock(),
        "get_cached_text_embeds": cache_lookup,
    }
    code = compile(ast.Module(body=[node], type_ignores=[]), str(NODES_PATH), "exec")
    exec(code, namespace)
    return namespace["WanVideoTextEncodeCached"]()


class PromptExtenderCacheTests(unittest.TestCase):
    def setUp(self):
        self.load_qwen = Mock(
            side_effect=lambda model, load_device, precision: (
                {"model": model, "precision": precision, "load_device": load_device},
            )
        )
        self.generate = Mock(
            side_effect=lambda **args: (
                f"{args['prompt']} [{args['qwen']['precision']}, "
                f"{args['device']}, seed={args['seed']}]",
            )
        )
        self.load_t5 = Mock(side_effect=AssertionError("disk-cache hit must not load T5"))
        self.cache_lookup = Mock(return_value=("positive embeddings", "negative embeddings"))
        package = ModuleType(PACKAGE)
        package.__path__ = []
        qwen_package = ModuleType(f"{PACKAGE}.qwen")
        qwen_package.__path__ = []
        qwen_module = ModuleType(f"{PACKAGE}.qwen.qwen")
        qwen_module.QwenLoader = lambda: SimpleNamespace(load=self.load_qwen)
        qwen_module.WanVideoPromptExtender = lambda: SimpleNamespace(generate=self.generate)
        loading_module = ModuleType(f"{PACKAGE}.nodes_model_loading")
        loading_module.LoadWanVideoT5TextEncoder = lambda: SimpleNamespace(loadmodel=self.load_t5)
        modules = {
            PACKAGE: package,
            f"{PACKAGE}.qwen": qwen_package,
            f"{PACKAGE}.qwen.qwen": qwen_module,
            f"{PACKAGE}.nodes_model_loading": loading_module,
        }
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.node = load_cached_node(self.cache_lookup)
        self.extender_args = {
            "model": "qwen-test",
            "max_new_tokens": 16,
            "system_prompt": "Expand the prompt.",
            "seed": 7,
        }

    def process(self, **overrides):
        args = {
            "model_name": "t5-test",
            "precision": "bf16",
            "positive_prompt": "a walking cat",
            "negative_prompt": "blur",
            "device": "gpu",
            "extender_args": self.extender_args,
        }
        args.update(overrides)
        result = self.node.process(**args)
        self.load_t5.assert_not_called()
        return result

    def test_identical_settings_reuse_generated_prompt(self):
        first = self.process()
        second = self.process()
        self.assertEqual(first, second)
        self.assertEqual(self.generate.call_count, 1)
        self.load_qwen.assert_called_once_with(
            "qwen-test", load_device="main_device", precision="bf16"
        )

    def test_precision_change_regenerates_before_disk_cache_lookup(self):
        first = self.process()
        second = self.process(precision="fp32")
        self.assertEqual(self.generate.call_count, 2)
        self.assertNotEqual(first[2], second[2])
        self.load_qwen.assert_called_with(
            "qwen-test", load_device="main_device", precision="fp32"
        )
        self.cache_lookup.assert_called_with(
            second[2], "blur", ("t5-test", "fp32", "disabled", "gpu")
        )

    def test_device_change_regenerates_before_disk_cache_lookup(self):
        first = self.process()
        second = self.process(device="cpu")
        self.assertEqual(self.generate.call_count, 2)
        self.assertNotEqual(first[2], second[2])
        self.load_qwen.assert_called_with(
            "qwen-test", load_device="cpu", precision="bf16"
        )
        self.cache_lookup.assert_called_with(
            second[2], "blur", ("t5-test", "bf16", "disabled", "cpu")
        )

    def test_returning_to_original_settings_reuses_original_prompt(self):
        first = self.process()
        changed = self.process(precision="fp32", device="cpu")
        restored = self.process()
        self.assertNotEqual(first[2], changed[2])
        self.assertEqual(first[2], restored[2])
        self.assertEqual(self.generate.call_count, 2)

    def test_changing_extender_seed_still_regenerates(self):
        first = self.process()
        second = self.process(extender_args={**self.extender_args, "seed": 9})
        self.assertNotEqual(first[2], second[2])
        self.assertEqual(self.generate.call_count, 2)

    def test_t5_only_changes_reuse_the_expanded_prompt(self):
        first = self.process()
        second = self.process(model_name="another-t5", quantization="fp8_e4m3fn")
        self.assertEqual(first[2], second[2])
        self.assertEqual(self.generate.call_count, 1)
        self.cache_lookup.assert_called_with(
            second[2], "blur", ("another-t5", "bf16", "fp8_e4m3fn", "gpu")
        )

    def test_no_extender_preserves_prompt_without_loading_qwen(self):
        result = self.process(extender_args=None)
        self.assertEqual(result[2], "a walking cat")
        self.generate.assert_not_called()
        self.load_qwen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
