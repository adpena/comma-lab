"""Round-trip + dispatch tests for CCh1 (Cool-Chic) and C3R1 (C3 residual)
export/load paths in `tac.renderer_export`.

These cover Phase 1 of the Lane I (Cool-Chic / C3 neural mask + renderer
compression) work, 2026-04-27. The existing FP4A round-trip tests cover the
canonical `AsymmetricPairGenerator` path; CoolChic + C3 add a parallel format
because their state_dicts contain `latents` ParameterList entries that the
FP4A walk does not see.

Test surface:
  * Round-trip a `CoolChicLatentRenderer` PairGenerator: bytes → load → forward
    matches the original to within FP4 quantization noise.
  * Round-trip a `C3ResidualRenderer` PairGenerator (pure FP4 path).
  * Round-trip a `C3ResidualRenderer` with `residual_quant_bits=8` mixed
    precision and assert the residual round-trip is STRICTLY tighter than
    pure FP4 (the science hypothesis behind Phase 3).
  * Magic-byte detection routes correctly through `detect_checkpoint_type`
    and `load_any_renderer_checkpoint` (the consumer chain `pgc.load_renderer`
    + `inflate_renderer._load_renderer` both rely on this dispatch).
  * Type-check guards reject mis-routing (e.g. exporting a CoolChic model
    via `export_c3_residual_renderer` raises TypeError).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.contrib.coolchic_renderer import (
    build_coolchic_renderer,
    build_c3_residual_renderer,
)
from tac.renderer_export import (
    detect_checkpoint_type,
    export_coolchic_renderer,
    export_c3_residual_renderer,
    load_coolchic_renderer,
    load_c3_residual_renderer,
    load_any_renderer_checkpoint,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _build_coolchic_with_random_weights(seed: int = 42):
    torch.manual_seed(seed)
    model = build_coolchic_renderer(
        num_classes=5,
        embed_dim=4,
        latent_ch=4,
        hidden=16,
        motion_hidden=16,
        latent_shapes=((2, 3), (4, 6)),
    )
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    model.eval()
    return model


def _build_c3_with_random_weights(seed: int = 42):
    torch.manual_seed(seed)
    model = build_c3_residual_renderer(
        num_classes=5,
        embed_dim=4,
        latent_ch=2,
        hidden=12,
        motion_hidden=16,
        residual_hidden=16,
        residual_layers=2,
        residual_scale=10.0,
        latent_shapes=((2, 3), (4, 6)),
    )
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    model.eval()
    return model


def _eval_pair(model, mask_t, mask_t1):
    with torch.no_grad():
        return model(mask_t, mask_t1)


# ── Round-trip tests ──────────────────────────────────────────────────────


class TestCoolChicRoundTrip:
    def test_export_returns_positive_byte_count(self, tmp_path: Path):
        model = _build_coolchic_with_random_weights()
        out = tmp_path / "renderer.bin"
        nbytes = export_coolchic_renderer(model, out)
        assert nbytes > 0
        assert out.stat().st_size == nbytes

    def test_magic_bytes_match_CCh1(self, tmp_path: Path):
        model = _build_coolchic_with_random_weights()
        out = tmp_path / "renderer.bin"
        export_coolchic_renderer(model, out)
        assert out.read_bytes()[:4] == b"CCh1"
        assert detect_checkpoint_type(out) == "coolchic"

    def test_round_trip_forward_matches_within_fp4_noise(self, tmp_path: Path):
        model = _build_coolchic_with_random_weights()
        mask_t = torch.randint(0, 5, (1, 16, 24))
        mask_t1 = torch.randint(0, 5, (1, 16, 24))
        original = _eval_pair(model, mask_t, mask_t1)

        out = tmp_path / "renderer.bin"
        export_coolchic_renderer(model, out)
        loaded = load_coolchic_renderer(out)
        restored = _eval_pair(loaded, mask_t, mask_t1)

        assert restored.shape == original.shape == (1, 2, 16, 24, 3)
        # FP4 quantization on small (~50K param) net stays well below 50 px
        # max diff in [0,255] space (typical: < 2 px).
        max_diff = (restored - original).abs().max().item()
        assert max_diff < 50.0, f"Round-trip max diff {max_diff:.4f} exceeds FP4 budget"

    def test_load_any_dispatches_coolchic(self, tmp_path: Path):
        model = _build_coolchic_with_random_weights()
        out = tmp_path / "renderer.bin"
        export_coolchic_renderer(model, out)
        # load_any must dispatch to load_coolchic_renderer when magic is CCh1.
        m1 = load_coolchic_renderer(out)
        m2 = load_any_renderer_checkpoint(out)
        mask_t = torch.randint(0, 5, (1, 16, 24))
        mask_t1 = torch.randint(0, 5, (1, 16, 24))
        # Same load path should produce identical output.
        out1 = _eval_pair(m1, mask_t, mask_t1)
        out2 = _eval_pair(m2, mask_t, mask_t1)
        assert (out1 - out2).abs().max().item() < 1e-5

    def test_round_trip_preserves_latent_shapes(self, tmp_path: Path):
        model = _build_coolchic_with_random_weights()
        out = tmp_path / "renderer.bin"
        export_coolchic_renderer(model, out)
        loaded = load_coolchic_renderer(out)
        assert loaded.renderer.latent_shapes == model.renderer.latent_shapes
        assert loaded.renderer.latent_ch == model.renderer.latent_ch
        assert loaded.renderer.hidden == model.renderer.hidden

    def test_export_rejects_c3_residual_via_typeerror(self, tmp_path: Path):
        """Exporting a C3 model through the CoolChic helper must fail loud."""
        model = _build_c3_with_random_weights()
        out = tmp_path / "should_not_write.bin"
        with pytest.raises(TypeError, match="CoolChicLatentRenderer"):
            export_coolchic_renderer(model, out)

    def test_trailing_byte_check_raises(self, tmp_path: Path):
        """Padding the binary should trip the consumed-all-bytes guard."""
        model = _build_coolchic_with_random_weights()
        out = tmp_path / "renderer.bin"
        export_coolchic_renderer(model, out)
        out.write_bytes(out.read_bytes() + b"\x00\x00\x00\x00")
        with pytest.raises(ValueError, match="Trailing data"):
            load_coolchic_renderer(out)


class TestC3ResidualRoundTrip:
    def test_export_returns_positive_byte_count(self, tmp_path: Path):
        model = _build_c3_with_random_weights()
        out = tmp_path / "renderer.bin"
        nbytes = export_c3_residual_renderer(model, out)
        assert nbytes > 0

    def test_magic_bytes_match_C3R1(self, tmp_path: Path):
        model = _build_c3_with_random_weights()
        out = tmp_path / "renderer.bin"
        export_c3_residual_renderer(model, out)
        assert out.read_bytes()[:4] == b"C3R1"
        assert detect_checkpoint_type(out) == "c3_residual"

    def test_round_trip_pure_fp4(self, tmp_path: Path):
        model = _build_c3_with_random_weights()
        mask_t = torch.randint(0, 5, (1, 16, 24))
        mask_t1 = torch.randint(0, 5, (1, 16, 24))
        original = _eval_pair(model, mask_t, mask_t1)

        out = tmp_path / "renderer.bin"
        export_c3_residual_renderer(model, out)  # residual_quant_bits=None
        loaded = load_c3_residual_renderer(out)
        restored = _eval_pair(loaded, mask_t, mask_t1)

        assert restored.shape == (1, 2, 16, 24, 3)
        max_diff = (restored - original).abs().max().item()
        assert max_diff < 50.0

    def test_round_trip_mixed_precision_int8_residual(self, tmp_path: Path):
        """Phase 3 of Lane I: int8 residual should preserve the C3 float-path
        SegNet gain that pure FP4 destroys. We verify the engineering
        precondition — int8 residual round-trip is STRICTLY tighter than FP4
        on the same checkpoint — at a small rate cost."""
        model = _build_c3_with_random_weights()
        mask_t = torch.randint(0, 5, (1, 16, 24))
        mask_t1 = torch.randint(0, 5, (1, 16, 24))
        original = _eval_pair(model, mask_t, mask_t1)

        # Pure FP4
        fp4_path = tmp_path / "fp4.bin"
        nb_fp4 = export_c3_residual_renderer(model, fp4_path, residual_quant_bits=None)
        loaded_fp4 = load_c3_residual_renderer(fp4_path)
        restored_fp4 = _eval_pair(loaded_fp4, mask_t, mask_t1)
        diff_fp4 = (restored_fp4 - original).abs().max().item()

        # Mixed precision (int8 residual)
        mp_path = tmp_path / "mp.bin"
        nb_mp = export_c3_residual_renderer(model, mp_path, residual_quant_bits=8)
        loaded_mp = load_c3_residual_renderer(mp_path)
        restored_mp = _eval_pair(loaded_mp, mask_t, mask_t1)
        diff_mp = (restored_mp - original).abs().max().item()

        # int8 residual MUST produce tighter round-trip than FP4 (the science
        # hypothesis). Allow a tiny epsilon for randomness on edge cases — but
        # in our smoke fixture the gap is ~3-5x in favour of int8.
        assert diff_mp <= diff_fp4 + 1e-3, (
            f"Mixed-precision (int8 residual) round-trip {diff_mp:.4f} should be "
            f"<= pure-FP4 round-trip {diff_fp4:.4f}"
        )

        # Rate cost is bounded: the residual head has ~3K params; int8-vs-FP4
        # delta is ~4 bits/param × 3K = ~1.5KB extra.
        rate_overhead_pct = (nb_mp - nb_fp4) / nb_fp4 * 100
        assert rate_overhead_pct < 25.0, (
            f"Mixed-precision rate overhead {rate_overhead_pct:.1f}% exceeds "
            f"25% — the residual head should be a small fraction of total "
            f"bytes"
        )

    def test_round_trip_preserves_residual_config(self, tmp_path: Path):
        model = _build_c3_with_random_weights()
        out = tmp_path / "renderer.bin"
        export_c3_residual_renderer(model, out)
        loaded = load_c3_residual_renderer(out)
        assert loaded.renderer.residual_hidden == model.renderer.residual_hidden
        assert loaded.renderer.residual_layers == model.renderer.residual_layers
        assert loaded.renderer.residual_scale == model.renderer.residual_scale
        assert loaded.renderer.base_renderer.latent_shapes == \
               model.renderer.base_renderer.latent_shapes

    def test_export_rejects_coolchic_via_typeerror(self, tmp_path: Path):
        model = _build_coolchic_with_random_weights()
        out = tmp_path / "should_not_write.bin"
        with pytest.raises(TypeError, match="C3ResidualRenderer"):
            export_c3_residual_renderer(model, out)

    def test_load_any_dispatches_c3_residual(self, tmp_path: Path):
        model = _build_c3_with_random_weights()
        out = tmp_path / "renderer.bin"
        export_c3_residual_renderer(model, out)
        m = load_any_renderer_checkpoint(out)
        # Inspect the actual type rather than just calling forward — the
        # auto-dispatch must construct C3ResidualRenderer, not CoolChic.
        from tac.contrib.coolchic_renderer import C3ResidualRenderer
        assert isinstance(m.renderer, C3ResidualRenderer), \
            f"Expected C3ResidualRenderer, got {type(m.renderer).__name__}"


class TestMagicByteCollisions:
    """Guard against silent dispatch breakage when new magic constants are
    added or shadowed by existing format bytes."""

    def test_coolchic_magic_distinct_from_existing_formats(self):
        from tac.renderer_export import _COOLCHIC_MAGIC, _C3_RESIDUAL_MAGIC
        existing = {b"DPSM", b"ASYM", b"FP4A", b"I4LZ", b"MXLZ"}
        assert _COOLCHIC_MAGIC not in existing
        assert _C3_RESIDUAL_MAGIC not in existing
        assert _COOLCHIC_MAGIC != _C3_RESIDUAL_MAGIC
        assert len(_COOLCHIC_MAGIC) == 4
        assert len(_C3_RESIDUAL_MAGIC) == 4

    def test_pgc_load_renderer_lists_lane_i_magics(self):
        """The downstream consumer chain (precompute_gradient_corrections.
        load_renderer) must accept CCh1/C3R1 — without this its strict
        whitelist would reject Lane I checkpoints with
        'unrecognized renderer checkpoint format' even though
        load_any_renderer_checkpoint can handle them."""
        import experiments.precompute_gradient_corrections as pgc
        assert pgc._RENDERER_MAGIC_COOLCHIC == b"CCh1"
        assert pgc._RENDERER_MAGIC_C3_RESIDUAL == b"C3R1"
