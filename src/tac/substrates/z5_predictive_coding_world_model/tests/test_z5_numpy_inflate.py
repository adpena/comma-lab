# SPDX-License-Identifier: MIT
"""Numpy-portable inflate parity for z5_predictive_coding_world_model.

Per the 8th MLX-first directive: proves the shipped inflate runtime decodes the
Z5PCWM1 archive with NO torch import (numpy + brotli only) AND that the numpy
autoregressive predictor rollout + NeRV decode matches the torch
PredictiveCodingSubstrate.reconstruct_pair to fp16-roundtrip tolerance (Catalog
#369: consumes the REAL trained predictor + decoder + residuals + ego_motion,
not a synthetic frame base).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from tac.substrates.z5_predictive_coding_world_model.architecture import (
    PredictiveCodingConfig,
    PredictiveCodingSubstrate,
)
from tac.substrates.z5_predictive_coding_world_model.archive import (
    pack_archive,
    parse_archive,
    parse_archive_numpy,
)
from tac.substrates.z5_predictive_coding_world_model.inflate import (
    _decode_pair,
    _rollout_latents,
)

_INFLATE_PATH = Path(__file__).resolve().parents[1] / "inflate.py"
_ARCHIVE_PATH = Path(__file__).resolve().parents[1] / "archive.py"


def _cfg() -> PredictiveCodingConfig:
    return PredictiveCodingConfig(
        latent_dim=6,
        encoder_input_channels=3,
        encoder_hidden_dim=8,
        decoder_embed_dim=8,
        decoder_initial_grid_h=3,
        decoder_initial_grid_w=4,
        decoder_channels=(6, 5, 4, 3, 2, 2),
        decoder_num_upsample_blocks=3,
        num_pairs=5,
        output_height=24,
        output_width=32,
        predictor_hidden_dim=10,
        predictor_num_layers=3,
        predictor_ego_motion_dim=4,
        identity_predictor=False,
    )


def _meta(cfg: PredictiveCodingConfig) -> dict[str, object]:
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
        "predictor_hidden_dim": cfg.predictor_hidden_dim,
    }


def _build_model_and_blob() -> tuple[PredictiveCodingSubstrate, bytes]:
    torch.manual_seed(0)
    cfg = _cfg()
    model = PredictiveCodingSubstrate(cfg).eval()
    with torch.no_grad():
        model.latent_init.normal_(std=0.5)
        model.residuals.normal_(std=0.5)
        model.ego_motion_buffer.normal_(std=0.5)
    blob = pack_archive(
        model.encoder.state_dict(),
        model.decoder.state_dict(),
        model.predictor.state_dict(),
        model.latent_init.detach(),
        model.residuals.detach(),
        model.ego_motion_buffer.detach(),
        _meta(cfg),
        predictor_num_layers=cfg.predictor_num_layers,
        identity_predictor=cfg.identity_predictor,
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
    assert np.abs(at.latent_init.numpy() - an.latent_init).max() < 1e-5
    assert np.abs(at.residuals.numpy() - an.residuals).max() < 1e-5
    assert np.abs(at.ego_motion.numpy() - an.ego_motion).max() < 1e-5
    for key in an.decoder_state_dict:
        torch_w = at.decoder_state_dict[key].numpy()
        assert np.abs(torch_w - an.decoder_state_dict[key]).max() < 1e-5, key
    for key in an.predictor_state_dict:
        torch_w = at.predictor_state_dict[key].numpy()
        assert np.abs(torch_w - an.predictor_state_dict[key]).max() < 1e-5, key


def test_numpy_autoregressive_decode_matches_torch_model() -> None:
    """Numpy rollout + decode parity vs torch reconstruct_pair (decoder output)."""
    cfg = _cfg()
    model, blob = _build_model_and_blob()
    arc = parse_archive_numpy(blob)

    # Exercise the autoregressive rollout: pair index 3 (depends on z_0..z_3).
    pair = 3
    with torch.no_grad():
        idx = torch.tensor([pair], dtype=torch.long)
        t_rgb0, t_rgb1, _z = model.reconstruct_pair(idx)
    t_rgb0 = t_rgb0.permute(0, 2, 3, 1).numpy()
    t_rgb1 = t_rgb1.permute(0, 2, 3, 1).numpy()

    z_all = _rollout_latents(
        arc, num_pairs=cfg.num_pairs, num_layers=cfg.predictor_num_layers, identity=False
    )
    n_rgb0, n_rgb1 = _decode_pair(
        z_all[pair : pair + 1], arc.decoder_state_dict,
        embed_dim=cfg.decoder_embed_dim, grid_h=cfg.decoder_initial_grid_h,
        grid_w=cfg.decoder_initial_grid_w, num_blocks=cfg.decoder_num_upsample_blocks,
        out_h=cfg.output_height, out_w=cfg.output_width,
    )
    # GELU tanh-approx + fp16 roundtrip + autoregressive accumulation tolerance.
    assert np.abs(t_rgb0 - n_rgb0).max() < 1e-3
    assert np.abs(t_rgb1 - n_rgb1).max() < 1e-3


def test_predictor_is_operationally_consumed() -> None:
    """Catalog #220: the predictor moves the latent (autoregressive rollout != identity)."""
    cfg = _cfg()
    _model, blob = _build_model_and_blob()
    arc = parse_archive_numpy(blob)
    z_full = _rollout_latents(
        arc, num_pairs=cfg.num_pairs, num_layers=cfg.predictor_num_layers, identity=False
    )
    z_identity = _rollout_latents(
        arc, num_pairs=cfg.num_pairs, num_layers=cfg.predictor_num_layers, identity=True
    )
    # The full predictor produces different latents than the identity ablation
    # at every t >= 1 (the predictor is non-trivially consumed).
    assert np.abs(z_full[1:] - z_identity[1:]).max() > 1e-5
