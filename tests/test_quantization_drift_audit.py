from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "quantization_drift_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("quantization_drift_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QuantizationDriftAuditTests(unittest.TestCase):
    def test_dequantize_int8_state_file_recovers_expected_tensor_shapes(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            int8_path = root / "toy_int8.pt"
            state = {
                "conv.weight.q": torch.tensor([[[[1, -1], [2, -2]]]], dtype=torch.int8),
                "conv.weight.s": torch.tensor(0.5),
                "conv.bias": torch.tensor([0.25], dtype=torch.float32),
                "__meta__": {"variant": "saliency_weighted", "hidden": 1, "kernel": 3, "alpha": 20.0},
            }
            torch.save(state, int8_path)

            meta, recovered = mod.dequantize_int8_state_file(int8_path)
            self.assertEqual(meta["variant"], "saliency_weighted")
            self.assertEqual(tuple(recovered["conv.weight"].shape), (1, 1, 2, 2))
            self.assertTrue(torch.equal(recovered["conv.bias"], torch.tensor([0.25])))

    def test_layerwise_drift_report_and_aggregate_metrics(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fp32_path = root / "toy_fp32.pt"
            int8_path = root / "toy_int8.pt"
            meta_path = root / "toy_best_meta.json"

            fp32_state = {
                "conv.weight": torch.tensor([[[[0.5, -0.5], [1.0, -1.0]]]], dtype=torch.float32),
                "conv.bias": torch.tensor([0.25], dtype=torch.float32),
            }
            torch.save(fp32_state, fp32_path)

            int8_state = {
                "conv.weight.q": torch.tensor([[[[1, -1], [2, -2]]]], dtype=torch.int8),
                "conv.weight.s": torch.tensor(0.5),
                "conv.bias": torch.tensor([0.20], dtype=torch.float32),
                "__meta__": {"variant": "saliency_weighted", "hidden": 1, "kernel": 3, "alpha": 20.0},
            }
            torch.save(int8_state, int8_path)
            meta_path.write_text(json.dumps({"fp32_path": str(fp32_path), "int8_path": str(int8_path), "epoch": 3, "scorer": 1.23}))

            report = mod.audit_best_meta(meta_path)
            self.assertEqual(report["meta_path"], str(meta_path))
            self.assertEqual(report["layer_count"], 2)
            self.assertGreater(report["aggregate"]["max_abs"], 0.0)
            self.assertGreaterEqual(report["layers"][0]["max_abs"], report["layers"][1]["max_abs"])


if __name__ == "__main__":
    unittest.main()
