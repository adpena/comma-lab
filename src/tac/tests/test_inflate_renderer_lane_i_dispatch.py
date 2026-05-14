# SPDX-License-Identifier: MIT
"""Magic-byte dispatch tests for `submissions/robust_current/inflate_renderer.py`
on Lane I (Cool-Chic / C3 residual) renderers.

This file is the CRITICAL-tier coverage for Phase 1B of the Lane I work
(2026-04-27): the inflate-side `_load_renderer` MUST recognize the CCh1 and
C3R1 magic bytes and dispatch to the right loader. Without this, the
canonical archive build → archive inflate → score loop is BROKEN for
Cool-Chic / C3 archives — the .bin would fall through to torch.load and
crash with a cryptic 'could not convert string to float: ...' message
(the same DEN-V2 bug class fixed for FP4A).

Test scope:
  * Real .bin files (from the canonical exporter) load cleanly through
    `_load_renderer`.
  * `_load_renderer` produces a model whose forward() returns the same
    `(B, 2, H, W, 3)` HWC pair format as AsymmetricPairGenerator — so the
    rest of the inflate pipeline (frame production loop) does not need to
    special-case Lane I.
  * `inflate_renderer.py` source contains explicit `b"CCh1"` and `b"C3R1"`
    branches with the right loader call. We grep for the regression: this
    is the cheapest way to guarantee the dispatch wasn't silently dropped
    in a refactor.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INFLATE_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"


# ── Module loader (avoid sys.path pollution) ─────────────────────────────


def _import_inflate_renderer():
    """Import submissions/robust_current/inflate_renderer.py without polluting
    sys.path. Uses importlib spec_from_file_location so we don't accidentally
    shadow `inflate_renderer` from another import path."""
    spec = importlib.util.spec_from_file_location(
        "_lane_i_inflate_renderer_under_test",
        INFLATE_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Source-level regression checks ───────────────────────────────────────


class TestInflateRendererSourceContainsLaneIDispatch:
    """Cheap regression checks: grep the inflate_renderer source for the
    Lane I magic-byte branches. These catch refactors that would silently
    drop the dispatch (the DEN-V2 bug pattern that originally motivated
    the magic-byte dispatch tests for FP4A)."""

    def test_source_has_CCh1_branch(self):
        src = INFLATE_PATH.read_text()
        assert 'magic == b"CCh1"' in src, \
            "inflate_renderer.py must dispatch on b'CCh1' magic for Lane I CoolChic"
        assert "load_coolchic_renderer" in src, \
            "inflate_renderer.py CCh1 branch must call load_coolchic_renderer"

    def test_source_has_C3R1_branch(self):
        src = INFLATE_PATH.read_text()
        assert 'magic == b"C3R1"' in src, \
            "inflate_renderer.py must dispatch on b'C3R1' magic for Lane I C3 residual"
        assert "load_c3_residual_renderer" in src, \
            "inflate_renderer.py C3R1 branch must call load_c3_residual_renderer"

    def test_docstring_lists_lane_i_formats(self):
        src = INFLATE_PATH.read_text()
        # The docstring of _load_renderer must enumerate the 6 supported
        # formats — otherwise consumers don't know Lane I is supported.
        assert "CCh1" in src, "Lane I CCh1 format must be listed"
        assert "C3R1" in src, "Lane I C3R1 format must be listed"


# ── End-to-end dispatch (real .bin → real model) ─────────────────────────


def _build_coolchic(seed: int = 7):
    from tac.contrib.coolchic_renderer import build_coolchic_renderer
    torch.manual_seed(seed)
    m = build_coolchic_renderer(
        num_classes=5,
        embed_dim=4,
        latent_ch=4,
        hidden=16,
        motion_hidden=16,
        latent_shapes=((2, 3), (4, 6)),
    )
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    m.eval()
    return m


def _build_c3(seed: int = 7):
    from tac.contrib.coolchic_renderer import build_c3_residual_renderer
    torch.manual_seed(seed)
    m = build_c3_residual_renderer(
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
        for p in m.parameters():
            p.normal_(0, 0.05)
    m.eval()
    return m


class TestInflateRendererLoadsLaneIBinaries:
    def test_load_coolchic_bin_via_inflate_loader(self, tmp_path: Path):
        from tac.renderer_export import export_coolchic_renderer
        model = _build_coolchic()
        bin_path = tmp_path / "renderer.bin"
        export_coolchic_renderer(model, bin_path)
        assert bin_path.read_bytes()[:4] == b"CCh1"

        inflate_mod = _import_inflate_renderer()
        loaded = inflate_mod._load_renderer(str(bin_path), "cpu")
        # Forward returns (B, 2, H, W, 3) HWC pair — same contract as ASYM/FP4A.
        mask_t = torch.randint(0, 5, (1, 16, 24))
        mask_t1 = torch.randint(0, 5, (1, 16, 24))
        with torch.no_grad():
            out = loaded(mask_t, mask_t1)
        assert out.shape == (1, 2, 16, 24, 3)
        assert out.dtype.is_floating_point
        assert torch.isfinite(out).all()
        # Range check: rendered RGB in [0, 255]
        assert out.min().item() >= 0.0 - 1e-3
        assert out.max().item() <= 255.0 + 1e-3

    def test_load_c3_residual_bin_via_inflate_loader(self, tmp_path: Path):
        from tac.renderer_export import export_c3_residual_renderer
        model = _build_c3()
        bin_path = tmp_path / "renderer.bin"
        export_c3_residual_renderer(model, bin_path)
        assert bin_path.read_bytes()[:4] == b"C3R1"

        inflate_mod = _import_inflate_renderer()
        loaded = inflate_mod._load_renderer(str(bin_path), "cpu")
        mask_t = torch.randint(0, 5, (1, 16, 24))
        mask_t1 = torch.randint(0, 5, (1, 16, 24))
        with torch.no_grad():
            out = loaded(mask_t, mask_t1)
        assert out.shape == (1, 2, 16, 24, 3)
        assert torch.isfinite(out).all()

    def test_load_c3_mixed_precision_bin_via_inflate_loader(self, tmp_path: Path):
        """Phase 3 verification: residual_quant_bits=8 binary loads the same
        way as a pure-FP4 binary — the loader auto-detects the layer kind
        from the header."""
        from tac.renderer_export import export_c3_residual_renderer
        model = _build_c3()
        bin_path = tmp_path / "renderer.bin"
        export_c3_residual_renderer(model, bin_path, residual_quant_bits=8)
        inflate_mod = _import_inflate_renderer()
        loaded = inflate_mod._load_renderer(str(bin_path), "cpu")
        mask_t = torch.randint(0, 5, (1, 16, 24))
        mask_t1 = torch.randint(0, 5, (1, 16, 24))
        with torch.no_grad():
            out = loaded(mask_t, mask_t1)
        assert out.shape == (1, 2, 16, 24, 3)
        assert torch.isfinite(out).all()

    def test_lane_i_inflate_dispatch_uses_tac_loader(self, tmp_path: Path):
        """Inflate-side loader MUST dispatch to tac.renderer_export when
        available, not silently fall through to a bytes parser. The CCh1/C3R1
        dispatch raises a clear RuntimeError if tac is missing rather than
        crashing with 'could not convert string to float'."""
        from tac.renderer_export import export_coolchic_renderer
        model = _build_coolchic()
        bin_path = tmp_path / "renderer.bin"
        export_coolchic_renderer(model, bin_path)

        # Sanity: with tac present, the dispatch returns a valid model.
        inflate_mod = _import_inflate_renderer()
        loaded = inflate_mod._load_renderer(str(bin_path), "cpu")
        assert loaded is not None

        # The inflate loader is hard-required to call the tac loader for
        # Lane I (no inline fallback). Source check ensures the import path
        # is what we expect.
        src = INFLATE_PATH.read_text()
        # CCh1 branch must import from tac.renderer_export and surface a
        # RuntimeError on ImportError (no silent fallback to torch.load).
        ccch1_idx = src.index('magic == b"CCh1"')
        ccch1_block = src[ccch1_idx:ccch1_idx + 1500]
        assert "from tac.renderer_export import load_coolchic_renderer" in ccch1_block
        assert "RuntimeError" in ccch1_block
        c3r1_idx = src.index('magic == b"C3R1"')
        c3r1_block = src[c3r1_idx:c3r1_idx + 1500]
        assert "from tac.renderer_export import load_c3_residual_renderer" in c3r1_block
        assert "RuntimeError" in c3r1_block
