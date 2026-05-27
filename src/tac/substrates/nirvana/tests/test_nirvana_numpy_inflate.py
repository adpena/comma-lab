# SPDX-License-Identifier: MIT
"""Numpy-portable inflate parity for nirvana (8th MLX-first directive).

Proves the shipped inflate runtime decodes the NRV1-v2 archive with NO torch
import (numpy + brotli + PIL only) AND that the numpy patch-decode + stitch
forward matches the torch NirvanaSubstrate.forward to fp16-roundtrip tolerance
(Catalog #369: consumes the REAL trained decoder + patch embeddings, not a
synthetic frame base).
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import numpy as np
import torch

from tac.substrates.nirvana.architecture import NirvanaConfig, NirvanaSubstrate
from tac.substrates.nirvana.archive import (
    NRV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
    parse_archive_numpy,
)
from tac.substrates.nirvana.inflate import inflate_one_video

_INFLATE_PATH = Path(__file__).resolve().parents[1] / "inflate.py"
_ARCHIVE_PATH = Path(__file__).resolve().parents[1] / "archive.py"


def _cfg() -> NirvanaConfig:
    return NirvanaConfig(
        latent_dim=6,
        patch_embed_dim=4,
        patch_grid_h=2,
        patch_grid_w=2,
        embed_dim=12,
        decoder_channels=(10, 8, 6),
        sin_frequency=30.0,
        num_upsample_blocks=2,
        initial_patch_grid_h=3,
        initial_patch_grid_w=4,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _meta(cfg: NirvanaConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_patch_grid_h": cfg.initial_patch_grid_h,
        "initial_patch_grid_w": cfg.initial_patch_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


def _build_model_and_blob() -> tuple[NirvanaSubstrate, bytes]:
    torch.manual_seed(0)
    cfg = _cfg()
    model = NirvanaSubstrate(cfg).eval()
    with torch.no_grad():
        model.latents.normal_(std=0.5)
        model.patch_embeddings.normal_(std=0.5)
    blob = pack_archive(
        model.state_dict(),
        model.latents.detach(),
        _meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )
    return model, blob


def test_schema_version_is_v2_numpy_bridge() -> None:
    assert NRV1_SCHEMA_VERSION == 2


def test_inflate_module_has_no_torch_import() -> None:
    tree = ast.parse(_INFLATE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("torch", "mlx")), alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(("torch", "mlx")), node.module


def test_parse_archive_numpy_is_torch_free_in_code() -> None:
    tree = ast.parse(_ARCHIVE_PATH.read_text(encoding="utf-8"))
    targets = {"parse_archive_numpy", "_deserialize_numpy_state_dict"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name):
                    assert inner.id != "torch", node.name
                if isinstance(inner, ast.Attribute) and isinstance(
                    inner.value, ast.Name
                ):
                    assert inner.value.id != "torch", node.name


def test_numpy_parse_matches_torch_parse_exactly() -> None:
    _model, blob = _build_model_and_blob()
    at = parse_archive(blob)
    an = parse_archive_numpy(blob)
    assert an.schema_version == NRV1_SCHEMA_VERSION
    assert np.abs(at.latents.numpy() - an.latents).max() < 1e-5
    for key in an.decoder_state_dict:
        torch_w = at.decoder_state_dict[key].numpy()
        assert np.abs(torch_w - an.decoder_state_dict[key]).max() < 1e-5, key


def test_numpy_decode_matches_torch_model() -> None:
    """Numpy patch-decode + stitch parity vs torch NirvanaSubstrate.forward."""
    model, blob = _build_model_and_blob()
    cfg = _cfg()

    with torch.no_grad():
        idx = torch.tensor([0], dtype=torch.long)
        t_rgb0, t_rgb1 = model(idx)
    t_rgb0 = t_rgb0[0].permute(1, 2, 0).numpy()  # (H, W, 3)
    t_rgb1 = t_rgb1[0].permute(1, 2, 0).numpy()

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "0"
        inflate_one_video(blob, out)
        from PIL import Image

        # Compare the rendered PNG (uint8) to the torch frame quantized the same way.
        np_f0 = np.asarray(Image.open(out / "0.png"))
        np_f1 = np.asarray(Image.open(out / "1.png"))
    t_f0_u8 = (np.clip(t_rgb0, 0, 1) * 255.0).round().clip(0, 255).astype(np.uint8)
    t_f1_u8 = (np.clip(t_rgb1, 0, 1) * 255.0).round().clip(0, 255).astype(np.uint8)
    # uint8 rounding tolerance: allow <=1 LSB on a tiny fraction of pixels
    assert np.abs(t_f0_u8.astype(int) - np_f0.astype(int)).max() <= 1
    assert np.abs(t_f1_u8.astype(int) - np_f1.astype(int)).max() <= 1


def test_inflate_one_video_writes_pngs() -> None:
    _model, blob = _build_model_and_blob()
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "0"
        inflate_one_video(blob, out)
        for frame_idx in range(cfg.num_pairs * 2):
            png = out / f"{frame_idx}.png"
            assert png.is_file(), png
            assert png.stat().st_size > 0
