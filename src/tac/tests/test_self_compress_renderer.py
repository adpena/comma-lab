"""Lane S engineering tests — Self-Compressing renderer codec.

Covers:
  * SelfCompressingConv2d STE round-trip with the extended kwargs
    (stride / groups / padding_mode) added in 2026-04-27.
  * The post-construction swap helper correctly swaps eligible Conv2d,
    skips ConvTranspose2d, protects scorer-sensitive name patterns.
  * Renderer with SC codec applied builds AND forwards (smoke).
  * SCv1 export → load round-trip is byte-stable AND output-stable.
  * SCv1 byte count is smaller than FP4-QAT for the same arch at
    target_bits ≤ 4.
  * inflate_renderer.py dispatches SCv1 magic to the right loader.
  * Per-channel bit allocation: when a profile sets target_bits, the
    Lagrangian penalty actually drives the bit-depth down (sanity-check
    that the compute_renderer_rate_penalty hook is wired correctly).

These tests replace the role of an end-to-end auth eval — every claim
the engineering report makes is pinned by a test here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INFLATE_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"


# ── SelfCompressingConv2d primitives ─────────────────────────────────────


class TestSelfCompressingConv2dExtendedKwargs:
    """The Lane S swap needs SC convs with stride / groups / padding_mode."""

    def test_stride_2_works(self):
        from tac.self_compress import SelfCompressingConv2d
        conv = SelfCompressingConv2d(
            in_channels=8, out_channels=16, kernel_size=3,
            stride=2, padding=1,
        )
        x = torch.randn(2, 8, 32, 32)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16)

    def test_dilation_works(self):
        from tac.self_compress import SelfCompressingConv2d
        conv = SelfCompressingConv2d(
            in_channels=8, out_channels=8, kernel_size=3,
            stride=1, padding=2, dilation=2,
        )
        x = torch.randn(2, 8, 16, 16)
        y = conv(x)
        assert y.shape == (2, 8, 16, 16)

    def test_grouped_works(self):
        from tac.self_compress import SelfCompressingConv2d
        conv = SelfCompressingConv2d(
            in_channels=8, out_channels=8, kernel_size=3,
            stride=1, padding=1, groups=8,  # depthwise
        )
        x = torch.randn(2, 8, 16, 16)
        y = conv(x)
        assert y.shape == (2, 8, 16, 16)
        # Verify weight_bits accounts for groups (fan_in = in_per_group * k * k = 9)
        assert int(conv.weight_numel()) == 8 * 1 * 3 * 3

    def test_replicate_padding_works(self):
        from tac.self_compress import SelfCompressingConv2d
        conv = SelfCompressingConv2d(
            in_channels=4, out_channels=4, kernel_size=3,
            stride=1, padding=1, padding_mode="replicate",
        )
        x = torch.randn(1, 4, 8, 8)
        y = conv(x)
        assert y.shape == (1, 4, 8, 8)

    def test_int4_round_trip_via_ste(self):
        """STE forward at fixed 4-bit gives different weights than FP32 forward
        but gradient still flows."""
        from tac.self_compress import SelfCompressingConv2d
        conv = SelfCompressingConv2d(8, 16, 3, padding=1, init_bits=4.0)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        y = conv(x)
        y.sum().backward()
        assert x.grad is not None and x.grad.abs().sum().item() > 0
        # Mean bits start at 4
        assert abs(conv.effective_bits_per_weight() - 4.0) < 0.01


# ── Swap helper ──────────────────────────────────────────────────────────


class TestSwapRendererConvsWithSelfCompress:
    def _build(self, **kwargs):
        from tac.renderer import build_renderer
        torch.manual_seed(7)
        return build_renderer(
            num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
            motion_hidden=32, depth=1, use_zoom_flow=True, pose_dim=6, **kwargs,
        )

    def test_protected_layers_stay_fp32(self):
        from tac.self_compress import (
            swap_renderer_convs_with_self_compress,
        )
        m = self._build()
        diag = swap_renderer_convs_with_self_compress(m, init_bits=8.0)
        # Every protected name should appear in diag["protected"], not "swapped"
        # The dilated-h64 baseline arch has these specific layers protected:
        expected_protected = {"renderer.head", "motion.head", "renderer.fuse_conv"}
        actual_protected = set(diag["protected"])
        # All expected items must be present (other arch flavours may add more)
        missing = expected_protected - actual_protected
        assert not missing, f"Expected protected layers missing: {missing}"
        # And none of the protected names should appear in swapped
        for p in actual_protected:
            assert p not in diag["swapped"]

    def test_transposed_convs_skipped(self):
        from tac.self_compress import swap_renderer_convs_with_self_compress
        m = self._build()
        diag = swap_renderer_convs_with_self_compress(m, init_bits=8.0)
        # The dilated-h64 arch has 1 ConvTranspose2d (renderer.up_conv); it
        # MUST appear in skipped (transposed)
        skipped_names = " ".join(diag["skipped"])
        assert "transposed" in skipped_names, (
            f"ConvTranspose2d not in skipped: {diag['skipped']}"
        )

    def test_swap_preserves_param_count_within_overhead(self):
        """The bit_depth.bits tensors add a small overhead (one float per
        SC output channel) but no other params should change."""
        from tac.self_compress import (
            swap_renderer_convs_with_self_compress, list_self_compress_layers,
        )
        m = self._build()
        n_before = sum(p.numel() for p in m.parameters())
        swap_renderer_convs_with_self_compress(m, init_bits=8.0)
        n_after = sum(p.numel() for p in m.parameters())
        # Overhead = sum of channels in SC layers
        sc_layers = list_self_compress_layers(m)
        overhead = sum(layer.out_channels for _, layer in sc_layers)
        assert n_after - n_before == overhead, (
            f"Param delta {n_after - n_before} != expected overhead {overhead}"
        )

    def test_swap_preserves_forward_at_init(self):
        """At init_bits=8 the SC quantization barely affects the output."""
        from tac.self_compress import swap_renderer_convs_with_self_compress
        m = self._build()
        m.eval()
        mask_t = torch.randint(0, 5, (1, 32, 48))
        mask_t1 = torch.randint(0, 5, (1, 32, 48))
        ego_flow = torch.zeros(1, 2, 32, 48)
        pose = torch.zeros(1, 6)
        with torch.no_grad():
            out_before = m(mask_t, mask_t1, ego_flow=ego_flow, pose=pose)
        swap_renderer_convs_with_self_compress(m, init_bits=8.0)
        m.eval()
        with torch.no_grad():
            out_after = m(mask_t, mask_t1, ego_flow=ego_flow, pose=pose)
        # 8-bit STE quantization vs fp32 gives small but non-zero diff.
        # Bound at 5.0 (max RGB level); typical < 1.0.
        diff = (out_before - out_after).abs().max().item()
        assert diff < 5.0, f"8-bit SC perturbed forward by {diff} (>5)"


# ── SCv1 export / load ──────────────────────────────────────────────────


def _build_swapped_renderer(seed: int = 7, bit_choices=(0, 2, 2, 3, 3, 4)):
    """Helper: build a dilated-h64 renderer with SC swap and a varied
    per-channel bit distribution mimicking post-training state."""
    import random
    from tac.renderer import build_renderer
    from tac.self_compress import (
        swap_renderer_convs_with_self_compress, list_self_compress_layers,
    )
    torch.manual_seed(seed)
    random.seed(seed + 1)
    m = build_renderer(
        num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
        motion_hidden=32, depth=1, use_zoom_flow=True, pose_dim=6,
    )
    swap_renderer_convs_with_self_compress(m, init_bits=8.0)
    for _name, layer in list_self_compress_layers(m):
        with torch.no_grad():
            for i in range(layer.bit_depth.bits.shape[0]):
                layer.bit_depth.bits[i] = float(random.choice(bit_choices))
    m.eval()
    return m


class TestSCv1ExportLoad:
    def test_round_trip_byte_exact_forward(self, tmp_path):
        from tac.renderer_export import (
            export_self_compressed_renderer, load_self_compressed_renderer,
        )
        m = _build_swapped_renderer()
        bin_path = tmp_path / "renderer.bin"
        export_self_compressed_renderer(m, bin_path)
        assert bin_path.read_bytes()[:4] == b"SCv1"

        m2 = load_self_compressed_renderer(bin_path)

        mask_t = torch.randint(0, 5, (1, 32, 48))
        mask_t1 = torch.randint(0, 5, (1, 32, 48))
        ego_flow = torch.zeros(1, 2, 32, 48)
        pose = torch.zeros(1, 6)
        with torch.no_grad():
            o1 = m(mask_t, mask_t1, ego_flow=ego_flow, pose=pose)
            o2 = m2(mask_t, mask_t1, ego_flow=ego_flow, pose=pose)
        # Round-trip is byte-exact because we re-quantize at the exact
        # learned bit-depth and the inner Conv2d already holds those values.
        diff = (o1 - o2).abs().max().item()
        assert diff < 1e-3, f"SCv1 round-trip diff {diff} exceeds 1e-3"

    def test_byte_stable_repeated_export(self, tmp_path):
        """Re-exporting the same model must give the same bytes (no
        nondeterminism in the packer)."""
        from tac.renderer_export import export_self_compressed_renderer
        m = _build_swapped_renderer()
        bin_a = tmp_path / "a.bin"
        bin_b = tmp_path / "b.bin"
        export_self_compressed_renderer(m, bin_a)
        export_self_compressed_renderer(m, bin_b)
        assert bin_a.read_bytes() == bin_b.read_bytes(), (
            "SCv1 export is not byte-deterministic (caller may rely on this "
            "for reproducible archives)"
        )

    def test_smaller_than_fp4_for_low_target_bits(self, tmp_path):
        """At avg ~2.5 SC bits/weight, SCv1 must be smaller than the same
        arch's FP4 export (which is 4 bits/weight)."""
        from tac.renderer_export import (
            export_self_compressed_renderer,
            export_asymmetric_checkpoint_fp4,
        )
        # SC version (mean ~2.5)
        m_sc = _build_swapped_renderer(bit_choices=(0, 2, 2, 2, 3, 3))
        sc_path = tmp_path / "sc.bin"
        sc_bytes = export_self_compressed_renderer(m_sc, sc_path)

        # FP4 version of the SAME baseline (no SC swap)
        from tac.renderer import build_renderer
        torch.manual_seed(7)
        m_fp4 = build_renderer(
            num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
            motion_hidden=32, depth=1, use_zoom_flow=True, pose_dim=6,
        )
        m_fp4.eval()
        fp4_path = tmp_path / "fp4.bin"
        fp4_bytes = export_asymmetric_checkpoint_fp4(m_fp4, fp4_path)

        # SC must be smaller. Allow a 5% buffer since the SC header has
        # per-channel bit-depth metadata that fp4 doesn't carry; the SC body
        # must overcome that overhead AND beat fp4's flat 4 bits/param.
        assert sc_bytes < fp4_bytes, (
            f"SC bytes ({sc_bytes:,}) not smaller than FP4 ({fp4_bytes:,}); "
            f"check that protected layers aren't dominating the SCv1 byte count."
        )

    def test_detect_checkpoint_type_returns_self_compress_v1(self, tmp_path):
        from tac.renderer_export import (
            export_self_compressed_renderer, detect_checkpoint_type,
        )
        m = _build_swapped_renderer()
        bin_path = tmp_path / "r.bin"
        export_self_compressed_renderer(m, bin_path)
        fmt = detect_checkpoint_type(bin_path)
        assert fmt == "self_compress_v1", (
            f"detect_checkpoint_type returned {fmt!r} for SCv1"
        )

    def test_load_any_renderer_dispatches_scv1(self, tmp_path):
        from tac.renderer_export import (
            export_self_compressed_renderer, load_any_renderer_checkpoint,
        )
        m = _build_swapped_renderer()
        bin_path = tmp_path / "r.bin"
        export_self_compressed_renderer(m, bin_path)
        loaded = load_any_renderer_checkpoint(bin_path)
        from tac.renderer import AsymmetricPairGenerator
        assert isinstance(loaded, AsymmetricPairGenerator)


# ── inflate_renderer.py dispatch ─────────────────────────────────────────


def _import_inflate_renderer():
    spec = importlib.util.spec_from_file_location(
        "_lane_s_inflate_renderer_under_test",
        INFLATE_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestInflateRendererSourceContainsSCv1Dispatch:
    """Source-level regression checks for inflate_renderer.py SCv1 dispatch.
    Cheap regression: catches refactors that silently drop the dispatch."""

    def test_source_has_SCv1_branch(self):
        src = INFLATE_PATH.read_text()
        assert 'magic == b"SCv1"' in src, (
            "inflate_renderer.py must dispatch on b'SCv1' magic for Lane S"
        )
        assert "load_self_compressed_renderer" in src, (
            "inflate_renderer.py SCv1 branch must call "
            "load_self_compressed_renderer"
        )

    def test_docstring_lists_scv1_format(self):
        src = INFLATE_PATH.read_text()
        assert "SCv1" in src, "Lane S SCv1 format must be documented in _load_renderer docstring"

    def test_scv1_branch_imports_from_tac(self):
        src = INFLATE_PATH.read_text()
        idx = src.index('magic == b"SCv1"')
        block = src[idx:idx + 2000]
        assert "from tac.renderer_export import load_self_compressed_renderer" in block, (
            "SCv1 branch must import the canonical loader from tac"
        )
        assert "RuntimeError" in block, (
            "SCv1 branch must raise RuntimeError on ImportError (no silent fallback)"
        )


class TestInflateRendererLoadsSCv1:
    def test_load_scv1_via_inflate_loader(self, tmp_path):
        from tac.renderer_export import export_self_compressed_renderer
        m = _build_swapped_renderer()
        bin_path = tmp_path / "renderer.bin"
        export_self_compressed_renderer(m, bin_path)
        assert bin_path.read_bytes()[:4] == b"SCv1"

        inflate_mod = _import_inflate_renderer()
        loaded = inflate_mod._load_renderer(str(bin_path), "cpu")
        # Forward returns (B, 2, H, W, 3) HWC pair — same contract as ASYM/FP4A.
        mask_t = torch.randint(0, 5, (1, 32, 48))
        mask_t1 = torch.randint(0, 5, (1, 32, 48))
        ego_flow = torch.zeros(1, 2, 32, 48)
        pose = torch.zeros(1, 6)
        with torch.no_grad():
            out = loaded(mask_t, mask_t1, ego_flow=ego_flow, pose=pose)
        assert out.shape == (1, 2, 32, 48, 3)
        assert torch.isfinite(out).all()
        assert out.min().item() >= -1e-3
        assert out.max().item() <= 255.0 + 1e-3


# ── Lagrangian rate penalty ──────────────────────────────────────────────


class TestRendererRatePenalty:
    def test_compute_rate_penalty_zero_when_under_target(self):
        """When avg bits/weight is well below target, the ReLU-penalty is 0."""
        from tac.self_compress import (
            swap_renderer_convs_with_self_compress, list_self_compress_layers,
            compute_renderer_rate_penalty,
        )
        from tac.renderer import build_renderer
        m = build_renderer(
            num_classes=5, embed_dim=6, base_ch=36, mid_ch=60, motion_hidden=32,
            depth=1, use_zoom_flow=True, pose_dim=6,
        )
        swap_renderer_convs_with_self_compress(m, init_bits=2.0)
        # Force every channel to 2 bits — well under target=4.0
        for _name, layer in list_self_compress_layers(m):
            with torch.no_grad():
                layer.bit_depth.bits.fill_(2.0)
        pen = compute_renderer_rate_penalty(m, target_bits_per_weight=4.0, lambda_rate=1.0)
        assert pen.item() == 0.0

    def test_rate_penalty_positive_when_over_target(self):
        from tac.self_compress import (
            swap_renderer_convs_with_self_compress, list_self_compress_layers,
            compute_renderer_rate_penalty,
        )
        from tac.renderer import build_renderer
        m = build_renderer(
            num_classes=5, embed_dim=6, base_ch=36, mid_ch=60, motion_hidden=32,
            depth=1, use_zoom_flow=True, pose_dim=6,
        )
        swap_renderer_convs_with_self_compress(m, init_bits=8.0)
        for _name, layer in list_self_compress_layers(m):
            with torch.no_grad():
                layer.bit_depth.bits.fill_(8.0)
        pen = compute_renderer_rate_penalty(m, target_bits_per_weight=2.5, lambda_rate=1.0)
        # Excess = (8 - 2.5) bits/weight on average → penalty > 0
        assert pen.item() > 0.0
        # And penalty must be differentiable wrt the bit_depth tensor
        pen.backward()
        any_grad = False
        for _name, layer in list_self_compress_layers(m):
            if layer.bit_depth.bits.grad is not None and layer.bit_depth.bits.grad.abs().sum() > 0:
                any_grad = True
                break
        assert any_grad, "rate penalty did not produce gradient on bit_depth"

    def test_rate_penalty_drives_bits_down_one_step(self):
        """A single optimizer step of rate-only loss must reduce avg bits."""
        from tac.self_compress import (
            swap_renderer_convs_with_self_compress, list_self_compress_layers,
            compute_renderer_rate_penalty, renderer_average_bits_per_weight,
        )
        from tac.renderer import build_renderer
        m = build_renderer(
            num_classes=5, embed_dim=6, base_ch=36, mid_ch=60, motion_hidden=32,
            depth=1, use_zoom_flow=True, pose_dim=6,
        )
        swap_renderer_convs_with_self_compress(m, init_bits=8.0)
        for _name, layer in list_self_compress_layers(m):
            with torch.no_grad():
                layer.bit_depth.bits.fill_(8.0)

        bit_params = []
        for _name, layer in list_self_compress_layers(m):
            bit_params.append(layer.bit_depth.bits)
        opt = torch.optim.SGD(bit_params, lr=1.0)

        avg_before = renderer_average_bits_per_weight(m)
        opt.zero_grad()
        pen = compute_renderer_rate_penalty(
            m, target_bits_per_weight=2.5, lambda_rate=1.0,
        )
        pen.backward()
        opt.step()
        with torch.no_grad():
            for p in bit_params:
                p.clamp_(0.0, 8.0)
        avg_after = renderer_average_bits_per_weight(m)
        assert avg_after < avg_before, (
            f"Rate penalty did not reduce bits: {avg_before:.2f} → {avg_after:.2f}"
        )


# ── Profile validation ──────────────────────────────────────────────────


class TestSelfCompressProfileSanity:
    def test_self_compress_renderer_smoke_profile_exists(self):
        from tac.profiles import PROFILES
        assert "self_compress_renderer_smoke" in PROFILES
        prof = PROFILES["self_compress_renderer_smoke"]
        assert prof["use_self_compress_codec"] is True
        assert "self_compress_target_bits" in prof
        assert "self_compress_lambda_end" in prof

    def test_self_compress_renderer_full_profile_exists(self):
        from tac.profiles import PROFILES
        assert "self_compress_renderer_full" in PROFILES
        prof = PROFILES["self_compress_renderer_full"]
        assert prof["use_self_compress_codec"] is True
        # Full profile must enforce eval_roundtrip (CLAUDE.md non-negotiable)
        assert prof["eval_roundtrip"] is True
        # Full profile must match dilated-h64 baseline arch byte-for-byte
        for k, expected in [
            ("base_ch", 36), ("mid_ch", 60), ("motion_hidden", 32),
            ("embed_dim", 6), ("depth", 1), ("pose_dim", 6),
        ]:
            assert prof[k] == expected, (
                f"self_compress_renderer_full[{k}]={prof[k]} != "
                f"dilated-h64 baseline {expected}"
            )

    def test_train_renderer_resolves_self_compress_args(self):
        """Sanity check: parse_args wires the SC profile keys into args."""
        from tac.experiments.train_renderer import parse_args
        args = parse_args([
            "--profile", "self_compress_renderer_smoke",
            "--tag", "unit_test",
            "--no-auth-eval-on-best",
        ])
        assert args.use_self_compress_codec is True
        assert abs(args.self_compress_target_bits - 4.0) < 0.01
        assert abs(args.self_compress_init_bits - 8.0) < 0.01
        assert args.self_compress_lambda_end > 0
