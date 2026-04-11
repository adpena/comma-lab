from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "research" / "monte_carlo_layer_scale_search.py"


def load_module():
    spec = importlib.util.spec_from_file_location("monte_carlo_layer_scale_search", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MonteCarloLayerScaleSearchTests(unittest.TestCase):
    def test_apply_layer_scales_keeps_shape_and_scales_requested_layers(self) -> None:
        mod = load_module()
        base_state = {
            "conv1.weight": torch.ones(2, 2, 3, 3),
            "conv1.bias": torch.ones(2),
            "conv2.weight": torch.ones(2, 2, 3, 3),
            "conv2.bias": torch.ones(2),
            "conv3.weight": torch.ones(3, 2, 3, 3),
            "conv3.bias": torch.ones(3),
        }
        theta = np.array([1.0, -1.0, 0.5, 0.0, -0.5, 0.25], dtype=np.float32)
        scaled = mod.apply_layer_scales(base_state, theta, scale_width=0.10)
        self.assertEqual(tuple(scaled["conv1.weight"].shape), (2, 2, 3, 3))
        self.assertAlmostEqual(float(scaled["conv1.weight"][0, 0, 0, 0]), 1.10, places=6)
        self.assertAlmostEqual(float(scaled["conv1.bias"][0]), 0.90, places=6)
        self.assertAlmostEqual(float(scaled["conv3.bias"][0]), 1.025, places=6)

    def test_metadata_strategy_key_does_not_replace_runtime_variant(self) -> None:
        mod = load_module()
        meta = mod.normalize_postfilter_meta({"variant": "saliency_weighted", "hidden": 32, "kernel": 3})
        meta["search_strategy"] = "mc_layer_scale"
        self.assertEqual(meta["variant"], "saliency_weighted")
        self.assertEqual(meta["search_strategy"], "mc_layer_scale")

    def test_parser_accepts_init_meta_path(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "seed.json"
            meta_path.write_text(json.dumps({"theta": [0, 1, 2, 3, 4, 5]}))
            args = mod.build_arg_parser().parse_args(["--init-meta", str(meta_path)])
            self.assertEqual(args.init_meta, meta_path)


if __name__ == "__main__":
    unittest.main()
