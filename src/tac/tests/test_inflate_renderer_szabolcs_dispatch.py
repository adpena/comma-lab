# SPDX-License-Identifier: MIT
"""Magic-byte dispatch tests for SZv1 routing in inflate_renderer.py.

Mirrors ``test_inflate_renderer_lane_i_dispatch.py`` but for Lane SZ:

    * Source contains the ``b"SZv1"`` branch and calls
      ``load_szabolcs_renderer``.
    * The branch raises a clear ``RuntimeError`` if tac is missing (no silent
      fallback to ``torch.load`` which would crash with a cryptic message).
    * End-to-end: a real SZv1 binary loads through ``_load_renderer`` and
      returns a runnable model.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from tac.contrib.szabolcs_renderer import (
    SzabolcsRenderer,
    build_szabolcs_renderer,
    encode_luma_to_probability_map,
)
from tac.szabolcs_archive import pack_szabolcs_archive

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INFLATE_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"


def _import_inflate_renderer():
    spec = importlib.util.spec_from_file_location(
        "_lane_sz_inflate_renderer_under_test", INFLATE_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Source-level checks ───────────────────────────────────────────────────


class TestInflateRendererSourceContainsSZv1Dispatch:
    def test_source_has_SZv1_branch(self):
        src = INFLATE_PATH.read_text()
        assert 'magic == b"SZv1"' in src, (
            "inflate_renderer.py must dispatch on b'SZv1' magic for Lane SZ"
        )
        assert "load_szabolcs_renderer" in src, (
            "inflate_renderer.py SZv1 branch must call load_szabolcs_renderer"
        )

    def test_szv1_branch_imports_from_tac(self):
        src = INFLATE_PATH.read_text()
        idx = src.index('magic == b"SZv1"')
        block = src[idx:idx + 1500]
        assert "from tac.contrib.szabolcs_renderer import load_szabolcs_renderer" in block
        # Hard error on missing tac — no silent fallback.
        assert "RuntimeError" in block

    def test_docstring_lists_SZv1_format(self):
        src = INFLATE_PATH.read_text()
        # _load_renderer docstring should enumerate SZv1 in its supported
        # formats list.
        assert "SZv1" in src


# ── End-to-end dispatch ───────────────────────────────────────────────────


def _build_szv1_bin(tmp_path: Path) -> Path:
    bundle = build_szabolcs_renderer(
        hidden=8, num_blocks=2, max_frame_index=64,
        shared_latent_height=8, shared_latent_width=10,
        quiet=True,
    )
    m = bundle.model
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    out = tmp_path / "renderer.bin"
    pack_szabolcs_archive(m, output_path=out)
    return out


class TestInflateRendererLoadsSZv1Binary:
    def test_load_szv1_via_inflate_loader(self, tmp_path: Path):
        bin_path = _build_szv1_bin(tmp_path)
        assert bin_path.read_bytes()[:4] == b"SZv1"

        inflate_mod = _import_inflate_renderer()
        loaded = inflate_mod._load_renderer(str(bin_path), "cpu")
        # Returned model has the szabolcs forward signature.
        assert isinstance(loaded, SzabolcsRenderer)
        # Sanity-run forward.
        torch.manual_seed(0)
        luma = torch.randint(0, 256, (1, loaded.h, loaded.w), dtype=torch.float32)
        prob = encode_luma_to_probability_map(luma)
        idx = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            out = loaded(prob, idx)
        assert out.shape == (1, 3, loaded.h, loaded.w)
        assert torch.isfinite(out).all()
        assert out.min().item() >= 0.0
        assert out.max().item() <= 255.0

    def test_szv1_rejects_bad_bytes(self, tmp_path: Path):
        # Write a SZv1-magic header but garbage afterwards.
        bad = tmp_path / "bad.bin"
        bad.write_bytes(b"SZv1" + b"\x00" * 64)
        inflate_mod = _import_inflate_renderer()
        with pytest.raises(Exception):
            inflate_mod._load_renderer(str(bad), "cpu")
