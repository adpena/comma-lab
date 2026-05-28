# SPDX-License-Identifier: MIT
"""Z6-v2 MLX renderer + MLX→PyTorch bridge canonical test suite.

Mirrors sister :mod:`tac.substrates.pact_nerv_selector_v3.tests.test_pact_nerv_selector_v3_mlx_renderer_and_bridge`
per INDIVIDUALLY-FRACTAL discipline. Skips when MLX is unavailable (non-Apple
CI) per the canonical try/except import guard in mlx_renderer.py.
"""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx", reason="MLX is not available on this host")
import mlx.core as mx  # noqa: E402

from tac.substrates.z6_v2_cargo_cult_unwind.architecture import (  # noqa: E402
    Z6V2Config,
    Z6V2Substrate,
)
from tac.substrates.z6_v2_cargo_cult_unwind.archive import (  # noqa: E402
    pack_archive,
    parse_archive,
)
from tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate import (  # noqa: E402
    pack_archive_from_exported_state_dict,
    z6_v2_meta_from_config,
)
from tac.substrates.z6_v2_cargo_cult_unwind.mlx_renderer import (  # noqa: E402
    MLX_EVIDENCE_GRADE,
    SCHEMA_VERSION,
    Z6V2SubstrateMLX,
)


def test_mlx_renderer_schema_version():
    assert SCHEMA_VERSION == "z6_v2_cargo_cult_unwind_mlx_renderer_v1"


def test_mlx_renderer_evidence_grade_non_promotable():
    """Per Catalog #192/#317/#341 canonical non-promotable marker."""
    assert MLX_EVIDENCE_GRADE == "[macOS-MLX research-signal]"


def test_mlx_renderer_param_count_matches_pytorch_within_init():
    """MLX + PyTorch sisters land same param count (same architecture)."""
    cfg = Z6V2Config(num_pairs=32)
    pyt_n = Z6V2Substrate(cfg).num_parameters()
    mlx_n = Z6V2SubstrateMLX(cfg).num_parameters()
    # Same architecture; exact match expected.
    assert pyt_n == mlx_n, f"PyTorch={pyt_n} != MLX={mlx_n}"


def test_mlx_renderer_forward_shape_call_b2chw_255():
    """Forward returns (B, 2, 3, H, W) in [0, 255] per canonical convention."""
    cfg = Z6V2Config(num_pairs=8)
    model = Z6V2SubstrateMLX(cfg)
    idx = mx.array([0, 1, 2, 3], dtype=mx.int32)
    out = model(idx)
    mx.eval(out)
    assert tuple(int(s) for s in out.shape) == (4, 2, 3, 384, 512)
    out_min = float(mx.min(out))
    out_max = float(mx.max(out))
    assert out_min >= 0.0, f"output min {out_min} < 0"
    assert out_max <= 255.0, f"output max {out_max} > 255"


def test_mlx_renderer_state_dict_keys_match_pytorch_sister():
    """Bridge invariant: MLX export_state_dict matches PyTorch state_dict keys."""
    cfg = Z6V2Config(num_pairs=8)
    pyt = Z6V2Substrate(cfg)
    pyt_keys = set(pyt.state_dict().keys())
    mlx_model = Z6V2SubstrateMLX(cfg)
    mlx_keys = set(mlx_model.export_state_dict().keys())
    in_pyt_only = pyt_keys - mlx_keys
    in_mlx_only = mlx_keys - pyt_keys
    assert not in_pyt_only, f"PyTorch-only keys: {sorted(in_pyt_only)[:5]}"
    assert not in_mlx_only, f"MLX-only keys: {sorted(in_mlx_only)[:5]}"


def test_mlx_state_dict_weight_layout_matches_pytorch_conv2d():
    """Bridge invariant: Conv2d weights stored as (out, in, kH, kW)."""
    cfg = Z6V2Config(num_pairs=4)
    pyt = Z6V2Substrate(cfg)
    pyt_sd = pyt.state_dict()
    mlx_model = Z6V2SubstrateMLX(cfg)
    mlx_sd = mlx_model.export_state_dict()
    for key in pyt_sd:
        if "dsc" in key and "weight" in key:
            assert pyt_sd[key].shape == tuple(mlx_sd[key].shape), (
                f"shape mismatch {key}: PYT={pyt_sd[key].shape} MLX={mlx_sd[key].shape}"
            )


def test_z6_v2_meta_from_config_canonical_keys():
    cfg = Z6V2Config(num_pairs=4)
    meta = z6_v2_meta_from_config(cfg)
    expected_keys = {
        "embed_dim", "initial_grid_h", "initial_grid_w", "decoder_channels",
        "sin_frequency", "num_upsample_blocks", "output_height", "output_width",
        "rao_ballard_level_boundary", "film_generator_depth",
        "film_hidden_width", "cooperative_receiver_beta",
    }
    assert set(meta.keys()) == expected_keys


def test_pack_archive_from_exported_state_dict_roundtrip():
    """MLX export + canonical pack_archive + parse_archive roundtrip."""
    cfg = Z6V2Config(num_pairs=8)
    mlx_model = Z6V2SubstrateMLX(cfg)
    exported = mlx_model.export_state_dict()
    bin_bytes = pack_archive_from_exported_state_dict(
        exported_state_dict=exported, cfg=cfg
    )
    arc = parse_archive(bin_bytes)
    assert arc.latents.shape == (8, 24)
    assert arc.ego_vecs.shape == (8, 6)
    # Decoder state dict keys preserved.
    decoder_keys = set(arc.decoder_state_dict.keys())
    expected_decoder_keys = set(exported.keys()) - {"latents", "ego_vecs"}
    assert decoder_keys == expected_decoder_keys, (
        f"missing decoder keys: {expected_decoder_keys - decoder_keys}"
    )


def test_mlx_renderer_no_pose_dim_attribute_not_ia3_modulation():
    """Z6-v2 distinguishing primitive is FiLM γ+β NOT IA3 γ-only.

    Sanity check: the MLX renderer does NOT carry an `ia3_mods` attribute
    (that's IA3 sister's distinguishing primitive).
    """
    cfg = Z6V2Config(num_pairs=4)
    mlx_model = Z6V2SubstrateMLX(cfg)
    assert not hasattr(mlx_model, "ia3_mods"), (
        "Z6-v2 should NOT have ia3_mods attribute; that's PACT-NeRV-IA3's primitive"
    )
    # Z6-v2 has ego_vecs (NOT ego_poses which is the IA3-style name).
    assert hasattr(mlx_model, "ego_vecs"), (
        "Z6-v2 MLX renderer must expose ego_vecs (FoE ego-motion vector)"
    )
