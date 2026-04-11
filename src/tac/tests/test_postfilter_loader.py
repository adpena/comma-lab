from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "submissions" / "robust_current" / "inflate_postfilter.py"
PIXELSHUFFLE_MODULE_PATH = MODULE_PATH  # inflate_postfilter.py contains PixelShuffleDilatedPostFilter


def load_module():
    spec = importlib.util.spec_from_file_location("inflate_postfilter", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pixelshuffle_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_pixelshuffle_dilated", PIXELSHUFFLE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def quantize_state(state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
    state: dict[str, object] = {}
    for name, param in state_dict.items():
        p = param.detach().cpu().float()
        scale = p.abs().max() / 127.0
        if float(scale) == 0.0:
            scale = torch.tensor(1.0)
        quantized = (p / scale).round().clamp(-128, 127).to(torch.int8)
        state[name + ".q"] = quantized
        state[name + ".s"] = scale
    return state


class PostfilterLoaderTests(unittest.TestCase):
    def test_loader_supports_per_channel_weights_and_fp32_biases(self) -> None:
        mod = load_module()
        model = mod.PostFilter(hidden=4, kernel=3)
        state: dict[str, object] = {"__meta__": {"variant": "residual", "hidden": 4, "kernel": 3}}
        for name, param in model.state_dict().items():
            p = param.detach().cpu().float()
            if name.endswith("bias"):
                state[name] = p
                continue
            flattened = p.reshape(p.shape[0], -1)
            scale = flattened.abs().max(dim=1).values / 127.0
            scale[scale == 0] = 1.0
            shape = [p.shape[0]] + [1] * (p.ndim - 1)
            quantized = (p / scale.view(*shape)).round().clamp(-128, 127).to(torch.int8)
            state[name + ".q"] = quantized
            state[name + ".s"] = scale

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "candidate_pc.pt"
            torch.save(state, weights)
            loaded = mod.load_postfilter_int8(str(weights), device="cpu")

        self.assertEqual(loaded.conv1.out_channels, 4)
        self.assertTrue(torch.allclose(loaded.conv1.bias, model.conv1.bias))

    def test_loader_supports_metadata_driven_hidden_size(self) -> None:
        mod = load_module()
        model = mod.PostFilter(hidden=24, kernel=3)
        state = quantize_state(model.state_dict())
        state["__meta__"] = {"variant": "residual", "hidden": 24, "kernel": 3}

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "candidate.pt"
            torch.save(state, weights)
            loaded = mod.load_postfilter_int8(str(weights), device="cpu")

        self.assertEqual(loaded.conv1.out_channels, 24)
        self.assertEqual(loaded.conv2.in_channels, 24)
        self.assertEqual(loaded.conv2.out_channels, 24)

    def test_loader_supports_depthwise_variant_metadata(self) -> None:
        mod = load_module()
        model = mod.DepthwisePostFilter(hidden=12, kernel=3)
        state = quantize_state(model.state_dict())
        state["__meta__"] = {"variant": "depthwise", "hidden": 12, "kernel": 3}

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "candidate_dw.pt"
            torch.save(state, weights)
            loaded = mod.load_postfilter_int8(str(weights), device="cpu")

        self.assertIsInstance(loaded, mod.DepthwisePostFilter)
        self.assertEqual(loaded.pw_in.out_channels, 12)
        self.assertEqual(loaded.dw.groups, 12)

    def test_loader_supports_luma_variant_metadata(self) -> None:
        mod = load_module()
        model = mod.LumaPostFilter(hidden=10, kernel=3)
        state = quantize_state(model.state_dict())
        state["__meta__"] = {"variant": "luma", "hidden": 10, "kernel": 3}

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "candidate_luma.pt"
            torch.save(state, weights)
            loaded = mod.load_postfilter_int8(str(weights), device="cpu")

        self.assertIsInstance(loaded, mod.LumaPostFilter)
        self.assertEqual(loaded.conv1.in_channels, 1)
        self.assertEqual(loaded.conv1.out_channels, 10)

    def test_loader_treats_saliency_weighted_variant_as_residual(self) -> None:
        mod = load_module()
        model = mod.PostFilter(hidden=16, kernel=3)
        state = quantize_state(model.state_dict())
        state["__meta__"] = {"variant": "saliency_weighted", "hidden": 16, "kernel": 3, "alpha": 10.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "candidate_saliency.pt"
            torch.save(state, weights)
            loaded = mod.load_postfilter_int8(str(weights), device="cpu")

        self.assertIsInstance(loaded, mod.PostFilter)
        self.assertEqual(loaded.conv1.out_channels, 16)

    def test_loader_supports_film_conditioned_variant_metadata(self) -> None:
        mod = load_module()
        model = mod.FiLMPostFilter(hidden=12, kernel=3)
        state = quantize_state(model.state_dict())
        state["__meta__"] = {"variant": "film_conditioned", "hidden": 12, "kernel": 3}

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "candidate_film.pt"
            torch.save(state, weights)
            loaded = mod.load_postfilter_int8(str(weights), device="cpu")

        self.assertIsInstance(loaded, mod.FiLMPostFilter)
        self.assertEqual(loaded.conv1.out_channels, 12)

    def test_loader_keeps_legacy_default_without_metadata(self) -> None:
        mod = load_module()
        model = mod.PostFilter(hidden=16, kernel=3)
        state = quantize_state(model.state_dict())

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "legacy.pt"
            torch.save(state, weights)
            loaded = mod.load_postfilter_int8(str(weights), device="cpu")

        self.assertEqual(loaded.conv1.out_channels, 16)

    def test_loader_supports_pixelshuffle_dilated_variant_metadata(self) -> None:
        mod = load_module()
        pixel_mod = load_pixelshuffle_module()
        model = pixel_mod.PixelShuffleDilatedPostFilter(hidden=8, kernel=3)
        state = quantize_state(model.state_dict())
        state["__meta__"] = {"variant": "pixelshuffle_dilated", "hidden": 8, "kernel": 3, "alpha": 20.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "candidate_psd.pt"
            torch.save(state, weights)
            loaded = mod.load_postfilter_int8(str(weights), device="cpu")

        # pixelshuffle_dilated maps to PSDPostFilter (canonical alias)
        self.assertEqual(loaded.conv1.in_channels, 12)
        self.assertEqual(loaded.conv4.out_channels, 12)
        # Verify forward pass works at correct shape
        x = torch.rand(1, 3, 32, 32) * 255
        with torch.no_grad():
            out = loaded(x)
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_loader_inferrs_pixelshuffle_dilated_from_state_layout_when_meta_is_wrong(self) -> None:
        mod = load_module()
        pixel_mod = load_pixelshuffle_module()
        model = pixel_mod.PixelShuffleDilatedPostFilter(hidden=8, kernel=3)
        state = quantize_state(model.state_dict())
        state["__meta__"] = {"variant": "saliency_weighted", "hidden": 8, "kernel": 3, "alpha": 20.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "candidate_psd_badmeta.pt"
            torch.save(state, weights)
            loaded = mod.load_postfilter_int8(str(weights), device="cpu")

        # When meta says saliency_weighted but state layout has conv4 (PSD),
        # the loader should detect the state layout and load correctly
        self.assertEqual(loaded.conv1.in_channels, 12)
        self.assertEqual(loaded.conv4.out_channels, 12)
        x = torch.rand(1, 3, 32, 32) * 255
        with torch.no_grad():
            out = loaded(x)
        self.assertEqual(out.shape, (1, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
