# SPDX-License-Identifier: MIT
from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessTinyFrameSelfCompressTests(unittest.TestCase):
    def _build_model(self):
        from tac.lossless.tiny_frame_predictor import TinyFramePredictorConfig, build_tiny_frame_predictor

        config = TinyFramePredictorConfig(
            context_frames=4,
            positions=8,
            vocab_size=16,
            embed_dim=4,
            hidden_dim=8,
            mixer_layers=1,
        )
        return build_tiny_frame_predictor(config)

    def test_select_tiny_frame_linear_layers_orders_largest_first(self) -> None:
        from tac.lossless.tiny_frame_self_compress import select_tiny_frame_linear_layers

        model = self._build_model()

        selected = select_tiny_frame_linear_layers(model, layer_names=None, largest_layers=3)
        names = tuple(name for name, _ in selected)

        self.assertEqual(names[0], "output_projection")
        self.assertEqual(len(selected), 3)
        self.assertGreaterEqual(selected[0][1].weight.numel(), selected[1][1].weight.numel())
        self.assertGreaterEqual(selected[1][1].weight.numel(), selected[2][1].weight.numel())

    def test_export_tiny_frame_self_compression_defaults_to_output_projection(self) -> None:
        import torch

        from tac.lossless.tiny_frame_self_compress import export_tiny_frame_self_compression

        model = self._build_model()
        with torch.no_grad():
            weight = torch.arange(model.output_projection.weight.numel(), dtype=torch.float32)
            model.output_projection.weight.copy_(weight.reshape_as(model.output_projection.weight) / 8.0)
            bias = torch.linspace(-1.5, 1.5, steps=model.output_projection.bias.numel(), dtype=torch.float32)
            model.output_projection.bias.copy_(bias)

        artifact = export_tiny_frame_self_compression(model)
        repeated = export_tiny_frame_self_compression(model)
        summary = artifact.summary
        layer = summary["layers"][0]

        self.assertEqual(layer["name"], "output_projection")
        self.assertEqual(summary["layer_count"], 1)
        self.assertEqual(summary["target_layer_names"], ["output_projection"])
        self.assertEqual(layer["weight_values"], 16 * 12)
        self.assertEqual(layer["row_scale_bytes"], 16 * 2)
        self.assertEqual(layer["packed_weight_bytes"], 96)
        self.assertEqual(layer["weight_payload_bytes"], 128)
        self.assertEqual(layer["bias_values"], 16)
        self.assertEqual(layer["bias_payload_bytes"], 32)
        self.assertEqual(layer["total_payload_bytes"], 160)
        self.assertEqual(summary["payload_bytes"], 160)
        self.assertEqual(summary["total_bytes"], len(artifact.data))
        self.assertEqual(summary, repeated.summary)
        self.assertEqual(artifact.data, repeated.data)

    def test_export_tiny_frame_self_compression_can_include_multiple_largest_layers(self) -> None:
        from tac.lossless.tiny_frame_self_compress import export_tiny_frame_self_compression

        model = self._build_model()

        artifact = export_tiny_frame_self_compression(model, layer_names=None, largest_layers=2, weight_bits=3)
        summary = artifact.summary
        names = [layer["name"] for layer in summary["layers"]]

        self.assertEqual(summary["layer_count"], 2)
        self.assertEqual(names[0], "output_projection")
        self.assertEqual(names[1], "mixer.0.ff.0")
        self.assertEqual(summary["target_layer_names"], names)
        self.assertEqual(summary["payload_bytes"], sum(layer["total_payload_bytes"] for layer in summary["layers"]))
        self.assertEqual(summary["total_bytes"], len(artifact.data))

    def test_tiny_frame_self_compression_byte_count_matches_export_and_artifact_path(self) -> None:
        from tac.lossless.tiny_frame_self_compress import (
            export_tiny_frame_self_compression,
            tiny_frame_self_compression_byte_count,
        )

        model = self._build_model()
        artifact = export_tiny_frame_self_compression(model)

        self.assertEqual(
            tiny_frame_self_compression_byte_count(model),
            artifact.summary["total_bytes"],
        )
        self.assertEqual(
            tiny_frame_self_compression_byte_count(model),
            tiny_frame_self_compression_byte_count(model),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "tiny_frame.selfc"
            artifact_path.write_bytes(artifact.data)

            self.assertEqual(
                tiny_frame_self_compression_byte_count(artifact_path=artifact_path),
                len(artifact.data),
            )


if __name__ == "__main__":
    unittest.main()
