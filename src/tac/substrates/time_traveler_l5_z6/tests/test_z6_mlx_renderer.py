# SPDX-License-Identifier: MIT
"""Z6 MLX-local renderer tests — L0 SCAFFOLD per OVERNIGHT Path 3 candidate #D.

Test groups (>=10 per Catalog #229 PV discipline):

- (A) Renderer construction + parameter breakdown: 3 tests
- (B) State-dict export → PyTorch parity: 4 tests
- (C) Auxiliary buffer export: 2 tests
- (D) Canonical Provenance manifest: 3 tests
- (E) MLX → PyTorch .pt bridge: 2 tests
- (F) MLX → Z6PCWM1 archive bridge + canonical PyTorch inflate roundtrip: 3 tests
- (G) L0 SCAFFOLD scope guards (depth>=2 + identity_predictor refused): 2 tests

All tests SKIP on non-Apple-Silicon / no-MLX CI per the canonical MLX guard
pattern (Catalog #1 / CLAUDE.md "MLX portable-local-substrate authority").
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module on non-Apple-Silicon / no-MLX CI
mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import torch  # noqa: E402

from tac.substrates.time_traveler_l5_z6.architecture import (  # noqa: E402
    Z6PredictiveCodingConfig,
    Z6PredictiveCodingSubstrate,
)
from tac.substrates.time_traveler_l5_z6.archive import parse_archive  # noqa: E402
from tac.substrates.time_traveler_l5_z6.inflate import (  # noqa: E402
    _read_single_member_archive_bytes,
    inflate_one_video,
)
from tac.substrates.time_traveler_l5_z6.mlx_export_bridge import (  # noqa: E402
    Z6_MLX_ARCHIVE_BUILD_SCHEMA,
    Z6_MLX_TO_PYTORCH_EXPORT_SCHEMA,
    build_z6_pytorch_pt_from_mlx_renderer,
    build_z6pcwm1_archive_from_mlx_renderer,
)
from tac.substrates.time_traveler_l5_z6.mlx_renderer import (  # noqa: E402
    EVIDENCE_GRADE,
    EVIDENCE_TAG,
    LANE_ID,
    SCHEMA_VERSION,
    Z6PredictiveCodingMLXRenderer,
)


def _small_cfg() -> Z6PredictiveCodingConfig:
    """Tiny config sized for fast tests (~10K params, 24x32 output)."""
    return Z6PredictiveCodingConfig(
        latent_dim=8,
        num_pairs=4,
        output_height=24,
        output_width=32,
        decoder_num_upsample_blocks=3,
        decoder_channels=(12, 8, 6),
        decoder_embed_dim=16,
        predictor_depth=1,
    )


# ---------------------------------------------------------------------------
# (A) Renderer construction + parameter breakdown
# ---------------------------------------------------------------------------


def test_a01_mlx_renderer_constructs_with_default_small_cfg() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    assert r.cfg.latent_dim == cfg.latent_dim
    assert r.cfg.num_pairs == cfg.num_pairs


def test_a02_mlx_renderer_parameter_breakdown_returns_canonical_keys() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    breakdown = r.num_parameters_breakdown()
    assert set(breakdown.keys()) == {
        "encoder", "decoder", "predictor", "latent_init", "residuals", "total",
    }
    assert all(isinstance(v, int) and v > 0 for k, v in breakdown.items() if k != "total")
    expected_total = (
        breakdown["encoder"] + breakdown["decoder"] + breakdown["predictor"]
        + breakdown["latent_init"] + breakdown["residuals"]
    )
    assert breakdown["total"] == expected_total


def test_a03_mlx_renderer_reconstruct_pair_returns_expected_shapes() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    pair_indices = mx.array([0, 1, 2, 3], dtype=mx.int32)
    rgb_0, rgb_1, z = r.reconstruct_pair(pair_indices)
    assert tuple(int(d) for d in rgb_0.shape) == (4, cfg.output_height, cfg.output_width, 3)
    assert tuple(int(d) for d in rgb_1.shape) == (4, cfg.output_height, cfg.output_width, 3)
    assert tuple(int(d) for d in z.shape) == (4, cfg.latent_dim)


# ---------------------------------------------------------------------------
# (B) State-dict export → PyTorch parity
# ---------------------------------------------------------------------------


def test_b01_mlx_state_dict_keys_match_pytorch_substrate_keys() -> None:
    """MLX-exported state_dict keys must be a subset of PyTorch state_dict keys."""
    cfg = _small_cfg()
    mlx_sub = Z6PredictiveCodingMLXRenderer(cfg)
    torch_sub = Z6PredictiveCodingSubstrate(cfg)
    mlx_keys = set(mlx_sub.export_state_dict().keys())
    torch_keys = set(torch_sub.state_dict().keys())
    # All MLX keys must be in PyTorch's state_dict (no extra keys MLX-side)
    assert not mlx_keys - torch_keys, f"MLX has extra keys not in PyTorch: {mlx_keys - torch_keys}"
    # PyTorch has 3 extra keys (latent_init, residuals, ego_motion_buffer) by design.
    assert torch_keys - mlx_keys == {"latent_init", "residuals", "ego_motion_buffer"}


def test_b02_mlx_state_dict_shapes_match_pytorch_substrate_shapes() -> None:
    """Per-key MLX shapes must EXACTLY match the PyTorch sister's shapes."""
    cfg = _small_cfg()
    mlx_sub = Z6PredictiveCodingMLXRenderer(cfg)
    torch_sub = Z6PredictiveCodingSubstrate(cfg)
    mlx_sd = mlx_sub.export_state_dict()
    torch_sd = torch_sub.state_dict()
    for k in mlx_sd:
        mlx_shape = mlx_sd[k].shape
        torch_shape = tuple(torch_sd[k].shape)
        assert mlx_shape == torch_shape, (
            f"shape mismatch for {k}: mlx={mlx_shape} torch={torch_shape}"
        )


def test_b03_mlx_state_dict_loads_into_pytorch_substrate() -> None:
    """Critical contract: MLX state_dict must load into PyTorch via load_state_dict."""
    cfg = _small_cfg()
    mlx_sub = Z6PredictiveCodingMLXRenderer(cfg)
    mlx_sd_np = mlx_sub.export_state_dict()
    torch_sd_load = {
        k: torch.from_numpy(v.copy()).float() for k, v in mlx_sd_np.items()
    }
    target = Z6PredictiveCodingSubstrate(cfg)
    res = target.load_state_dict(torch_sd_load, strict=False)
    # Only the 3 buffers/params (loaded separately via aux export) should be missing
    assert set(res.missing_keys) == {"latent_init", "residuals", "ego_motion_buffer"}
    assert res.unexpected_keys == []


def test_b04_mlx_state_dict_conv_weights_in_pytorch_oihw_layout() -> None:
    """Exported Conv2d weights must be PyTorch (out, in, kH, kW) not MLX (out, kH, kW, in)."""
    cfg = _small_cfg()
    mlx_sub = Z6PredictiveCodingMLXRenderer(cfg)
    mlx_sd = mlx_sub.export_state_dict()
    # Encoder stem: PyTorch shape is (hidden_dim, input_channels, 3, 3)
    expected_stem_shape = (
        cfg.encoder_hidden_dim, cfg.encoder_input_channels, 3, 3,
    )
    assert mlx_sd["encoder.stem.weight"].shape == expected_stem_shape


# ---------------------------------------------------------------------------
# (C) Auxiliary buffer export
# ---------------------------------------------------------------------------


def test_c01_export_auxiliary_buffers_returns_canonical_keys() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    aux = r.export_auxiliary_buffers()
    assert set(aux.keys()) == {"latent_init", "residuals", "ego_motion"}
    assert aux["latent_init"].shape == (cfg.latent_dim,)
    assert aux["residuals"].shape == (cfg.num_pairs, cfg.latent_dim)
    assert aux["ego_motion"].shape == (cfg.num_pairs, cfg.predictor_ego_motion_dim)


def test_c02_export_auxiliary_buffers_dtype_float32() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    aux = r.export_auxiliary_buffers()
    for k, v in aux.items():
        assert v.dtype == np.float32, f"aux {k} has dtype {v.dtype}; expected float32"


# ---------------------------------------------------------------------------
# (D) Canonical Provenance manifest
# ---------------------------------------------------------------------------


def test_d01_state_dict_manifest_carries_non_promotable_markers() -> None:
    """Per Catalog #287/#323: every export carries axis+grade+non-promotable markers."""
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    mf = r.export_state_dict_manifest()
    assert mf.evidence_grade == "macOS-MLX research-signal"
    assert mf.score_claim is False
    assert mf.promotion_eligible is False
    assert mf.ready_for_exact_eval_dispatch is False


def test_d02_state_dict_manifest_schema_version_is_canonical() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    mf = r.export_state_dict_manifest()
    assert mf.schema_version == SCHEMA_VERSION
    assert mf.substrate_id == "time_traveler_l5_z6"


def test_d03_canonical_evidence_tag_matches_module_constant() -> None:
    """Sanity: the module-level EVIDENCE_TAG must match what manifests would emit."""
    assert EVIDENCE_TAG == "[macOS-MLX research-signal]"
    assert EVIDENCE_GRADE == "macOS-MLX research-signal"
    assert LANE_ID == "lane_z6_predictive_coding_mlx_scaffold_20260526"


# ---------------------------------------------------------------------------
# (E) MLX → PyTorch .pt bridge
# ---------------------------------------------------------------------------


def test_e01_build_z6_pytorch_pt_from_mlx_renderer_produces_loadable_pt() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    with tempfile.TemporaryDirectory() as tmpd:
        pt_path = Path(tmpd) / "test_z6_mlx.pt"
        mf = build_z6_pytorch_pt_from_mlx_renderer(r, pt_path)
        assert pt_path.is_file()
        assert mf["z6_mlx_export_schema_version"] == Z6_MLX_TO_PYTORCH_EXPORT_SCHEMA
        assert mf["promotion_eligible"] is False
        assert mf["lane_id"] == LANE_ID
        # Load the .pt and confirm it's a valid PyTorch state_dict
        loaded = torch.load(pt_path, weights_only=True)
        assert isinstance(loaded, dict)
        assert "encoder.stem.weight" in loaded


def test_e02_build_z6_pytorch_pt_refuses_overwrite_by_default() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    with tempfile.TemporaryDirectory() as tmpd:
        pt_path = Path(tmpd) / "test_z6_mlx.pt"
        build_z6_pytorch_pt_from_mlx_renderer(r, pt_path)
        with pytest.raises(FileExistsError):
            build_z6_pytorch_pt_from_mlx_renderer(r, pt_path)
        # overwrite=True succeeds
        build_z6_pytorch_pt_from_mlx_renderer(r, pt_path, overwrite=True)


# ---------------------------------------------------------------------------
# (F) MLX → Z6PCWM1 archive bridge + canonical PyTorch inflate roundtrip
# ---------------------------------------------------------------------------


def test_f01_build_z6pcwm1_archive_from_mlx_renderer_produces_parseable_archive() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    with tempfile.TemporaryDirectory() as tmpd:
        archive_path = Path(tmpd) / "0.bin"
        mf = build_z6pcwm1_archive_from_mlx_renderer(r, archive_path)
        assert archive_path.is_file()
        assert mf["schema_version"] == Z6_MLX_ARCHIVE_BUILD_SCHEMA
        assert mf["promotion_eligible"] is False
        # Parse the archive bytes back via canonical PyTorch parse_archive
        arc = parse_archive(archive_path.read_bytes())
        assert arc.latent_init.shape == (cfg.latent_dim,)
        assert arc.residuals.shape == (cfg.num_pairs, cfg.latent_dim)
        assert arc.ego_motion.shape == (cfg.num_pairs, cfg.predictor_ego_motion_dim)


def test_f02_mlx_archive_meta_carries_non_promotable_markers() -> None:
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    with tempfile.TemporaryDirectory() as tmpd:
        archive_path = Path(tmpd) / "0.bin"
        build_z6pcwm1_archive_from_mlx_renderer(r, archive_path)
        arc = parse_archive(archive_path.read_bytes())
        assert arc.meta["mlx_training_evidence_grade"] == "macOS-MLX research-signal"
        assert arc.meta["mlx_training_score_claim"] is False
        assert arc.meta["mlx_training_promotion_eligible"] is False
        assert arc.meta["mlx_training_lane_id"] == LANE_ID


def test_f03_mlx_archive_inflates_via_canonical_pytorch_inflate_runtime() -> None:
    """Critical end-to-end contract: MLX-built archive inflates through PyTorch."""
    cfg = _small_cfg()
    r = Z6PredictiveCodingMLXRenderer(cfg)
    with tempfile.TemporaryDirectory() as tmpd:
        archive_dir = Path(tmpd)
        archive_path = archive_dir / "0.bin"
        build_z6pcwm1_archive_from_mlx_renderer(r, archive_path)
        raw_out = archive_dir / "test_inflate.raw"
        # Read via canonical helper
        archive_bytes = _read_single_member_archive_bytes(archive_dir)
        frames = inflate_one_video(archive_bytes, raw_out, device="cpu")
        # Expect 2 frames per pair × num_pairs
        assert frames == cfg.num_pairs * 2
        # Camera resolution: 874 × 1164 × 3 bytes per frame
        assert raw_out.stat().st_size == frames * 874 * 1164 * 3


# ---------------------------------------------------------------------------
# (G) L0 SCAFFOLD scope guards
# ---------------------------------------------------------------------------


def test_g01_predictor_depth_gte_2_raises_not_implemented() -> None:
    """L0 SCAFFOLD only supports depth=1; depth>=2 deferred to follow-on subagent."""
    cfg = Z6PredictiveCodingConfig(
        latent_dim=8, num_pairs=4, predictor_depth=3,
    )
    with pytest.raises(NotImplementedError, match="predictor_depth=1"):
        Z6PredictiveCodingMLXRenderer(cfg)


def test_g02_identity_predictor_true_raises_not_implemented() -> None:
    """L0 SCAFFOLD doesn't implement the identity-predictor ablation probe."""
    cfg = Z6PredictiveCodingConfig(
        latent_dim=8, num_pairs=4, identity_predictor=True,
    )
    with pytest.raises(NotImplementedError, match="identity_predictor"):
        Z6PredictiveCodingMLXRenderer(cfg)
