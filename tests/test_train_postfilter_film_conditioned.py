from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_film_conditioned.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_film_conditioned", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainPostfilterFilmConditionedTests(unittest.TestCase):
    def test_film_model_preserves_shape(self) -> None:
        mod = load_module()
        model = mod.FiLMQATPostFilter(hidden=8, kernel=3)
        x = torch.rand(2, 3, 16, 16) * 255.0
        y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_film_descriptor_has_three_features(self) -> None:
        mod = load_module()
        model = mod.FiLMQATPostFilter(hidden=8, kernel=3)
        x = torch.rand(2, 3, 16, 16) * 255.0
        desc = model._descriptor(x)
        self.assertEqual(tuple(desc.shape), (2, 3))
