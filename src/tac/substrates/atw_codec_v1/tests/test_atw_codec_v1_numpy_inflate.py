# SPDX-License-Identifier: MIT
"""Numpy-portable inflate parity for atw_codec_v1 (8th MLX-first directive).

Proves the shipped inflate runtime decodes the ATW1 archive with NO torch
import (numpy + brotli only) AND that the numpy decode forward matches the
torch ATWCodec ``reconstruct_from_wz_residual`` to fp16-roundtrip tolerance
(Catalog #369: consumes the REAL trained decoder + WZ side-info head, not a
synthetic frame base).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from tac.substrates.atw_codec_v1.archive import (
    pack_archive,
    parse_archive,
    parse_archive_numpy,
)
from tac.substrates.atw_codec_v1.architecture import ATWCodec, ATWCodecConfig
from tac.substrates.atw_codec_v1.inflate import _decode_pair, _wz_predict

_INFLATE_PATH = Path(__file__).resolve().parents[1] / "inflate.py"
_ARCHIVE_PATH = Path(__file__).resolve().parents[1] / "archive.py"


def _cfg() -> ATWCodecConfig:
    return ATWCodecConfig(
        latent_dim=6,
        encoder_input_channels=3,
        encoder_hidden_dim=8,
        decoder_embed_dim=8,
        decoder_initial_grid_h=3,
        decoder_initial_grid_w=4,
        decoder_channels=(6, 5, 4, 3, 2, 2),
        decoder_num_upsample_blocks=3,
        num_pairs=4,
        output_height=24,
        output_width=32,
        scorer_class_prior_dim=5,
        wz_head_hidden_dim=8,
        wz_head_enabled=True,
    )


def _meta(cfg: ATWCodecConfig) -> dict[str, object]:
    return {
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
    }


def _build_model_and_blob() -> tuple[ATWCodec, bytes]:
    torch.manual_seed(0)
    cfg = _cfg()
    model = ATWCodec(cfg).eval()
    with torch.no_grad():
        model.latents.normal_(std=0.5)
        model.scorer_class_prior_table.normal_(std=0.5)
    blob = pack_archive(
        model.encoder.state_dict(),
        model.decoder.state_dict(),
        model.wz_side_info_head.state_dict(),
        model.latents.detach(),  # at inflate time, latents store z_residual
        model.scorer_class_prior_table.detach(),
        _meta(cfg),
    )
    return model, blob


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
    assert an.schema_version == at.schema_version
    # latent residual dequant exact (same int8 dequant math)
    assert np.abs(at.latent_residual.numpy() - an.latent_residual).max() < 1e-5
    # class prior table exact (same fp16 bytes)
    assert (
        np.abs(
            at.scorer_class_prior_table.numpy() - an.scorer_class_prior_table
        ).max()
        < 1e-5
    )
    # decoder weights exact (same fp16 blob)
    for key in an.decoder_state_dict:
        torch_w = at.decoder_state_dict[key].numpy()
        assert np.abs(torch_w - an.decoder_state_dict[key]).max() < 1e-5, key


def test_numpy_decode_matches_torch_model() -> None:
    """Numpy decode parity vs torch reconstruct_from_wz_residual (decoder output)."""
    cfg = _cfg()
    model, blob = _build_model_and_blob()
    arc = parse_archive_numpy(blob)

    # torch reference: reconstruct pair 0 from z_residual (sigmoid'd decoder out).
    with torch.no_grad():
        idx = torch.tensor([0], dtype=torch.long)
        z_res = model.latents[idx]
        t_rgb0, t_rgb1 = model.reconstruct_from_wz_residual(idx, z_res)
    # NCHW (1,3,H,W) -> NHWC for comparison
    t_rgb0 = t_rgb0.permute(0, 2, 3, 1).numpy()
    t_rgb1 = t_rgb1.permute(0, 2, 3, 1).numpy()

    # numpy decode
    z_residual = arc.latent_residual[0:1]
    class_prior = arc.scorer_class_prior_table[0:1]
    z = z_residual + _wz_predict(class_prior, arc.wz_side_info_head_state_dict)
    n_rgb0, n_rgb1 = _decode_pair(
        z, arc.decoder_state_dict,
        embed_dim=cfg.decoder_embed_dim, grid_h=cfg.decoder_initial_grid_h,
        grid_w=cfg.decoder_initial_grid_w, num_blocks=cfg.decoder_num_upsample_blocks,
        out_h=cfg.output_height, out_w=cfg.output_width,
    )
    assert np.abs(t_rgb0 - n_rgb0).max() < 1e-4
    assert np.abs(t_rgb1 - n_rgb1).max() < 1e-4


def test_wz_residual_is_operationally_consumed() -> None:
    """Catalog #220: nonzero class prior changes the decode (no-op detector)."""
    cfg = _cfg()
    _model, blob = _build_model_and_blob()
    arc = parse_archive_numpy(blob)
    z_res = arc.latent_residual[0:1]
    cp = arc.scorer_class_prior_table[0:1]
    z_with = z_res + _wz_predict(cp, arc.wz_side_info_head_state_dict)
    # zero class prior -> different latent -> different frame
    z_without = z_res + _wz_predict(
        np.zeros_like(cp), arc.wz_side_info_head_state_dict
    )
    rgb_with, _ = _decode_pair(
        z_with, arc.decoder_state_dict,
        embed_dim=cfg.decoder_embed_dim, grid_h=cfg.decoder_initial_grid_h,
        grid_w=cfg.decoder_initial_grid_w, num_blocks=cfg.decoder_num_upsample_blocks,
        out_h=cfg.output_height, out_w=cfg.output_width,
    )
    rgb_without, _ = _decode_pair(
        z_without, arc.decoder_state_dict,
        embed_dim=cfg.decoder_embed_dim, grid_h=cfg.decoder_initial_grid_h,
        grid_w=cfg.decoder_initial_grid_w, num_blocks=cfg.decoder_num_upsample_blocks,
        out_h=cfg.output_height, out_w=cfg.output_width,
    )
    assert np.abs(rgb_with - rgb_without).max() > 1e-6
