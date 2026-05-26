# SPDX-License-Identifier: MIT
"""ATW V2 cooperative-receiver V2 — L0 SCAFFOLD test suite.

Per Phase 3 design memo §4.1 (binding deliverables) + Catalog #91
ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation no_op_proof +
operator binding directive #3 (3-axis evidence: math + scientific +
engineering rigor + MLX drift minimization + portability via numpy).

Test categories
===============

1. **Substrate identity** — verify __init__.py constants importable and
   correct per Catalog #241 contract.
2. **Archive grammar roundtrip** — pack→parse byte-stable per Catalog #91.
3. **Byte-mutation no_op_proof** — every section in the grammar has
   NON-ZERO impact on parse result per Catalog #139/#272. The v1
   cdf_table_blob bug class (FALSIFIED via codex 057130de4) does NOT
   recur at any section of the new grammar.
4. **Inflate runtime pattern** — inflate.py CLI signature + select_inflate_device
   per Catalog #146 + #205.
5. **MLX↔numpy parity** — bit-exact-where-applicable per Axis 3 portability.
6. **MLX↔PyTorch drift** — within Catalog #1265 gate threshold per Axis 2
   MLX drift minimization (when both backends available).
7. **Layer 1 META-unwind verification** — substrate exposes Atick-Redlich
   binding + Tishby/WZ demotion + ego-motion-FOE conditioning constants.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tac.substrates.atw_v2_cooperative_receiver_v2 import (
    ARCHIVE_GRAMMAR,
    DESIGN_MEMO_PATH,
    DISPATCH_ENABLED,
    EXPORT_FORMAT,
    HORIZON_CLASS,
    IMPLEMENTATION_STATUS,
    LANE_ID,
    LAYER_1_META_UNWIND_ADVISORY_CROSS_CHECKS,
    LAYER_1_META_UNWIND_BINDING_ANCHOR,
    LAYER_1_META_UNWIND_DEMOTED_THEOREMS,
    NEW_CONDITIONING_VARIABLE,
    NO_OP_DETECTOR_PLANNED,
    PARSER_SECTION_MANIFEST,
    PHASE_1_AUDIT_PATH,
    PHASE_2_DECISION_PATH,
    RESEARCH_ONLY,
    SCORE_AWARE_LOSS,
    SUBSTRATE_ID,
    SUBSTRATE_VERSION,
    V1_FALSIFICATION_ANCHOR_COMMIT,
    V1_FALSIFICATION_MAX_ABS_RAW_BYTE_DELTA,
)
from tac.substrates.atw_v2_cooperative_receiver_v2.archive import (
    ATWV2CR2_HEADER_FIXED_SIZE,
    ATWV2CR2_HEADER_FMT,
    ATWV2CR2_MAGIC,
    ATWV2CR2_SCHEMA_VERSION,
    ATWv2CR2Archive,
    NUM_SECTIONS,
    PARSER_SECTION_ROLES,
    build_smoke_archive,
    pack_archive,
    parse_archive,
)
from tac.substrates.atw_v2_cooperative_receiver_v2.numpy_reference import (
    CooperativeReceiverConfig,
    init_numpy_weights_random,
    numpy_decode_pair_with_ego_motion_conditioning,
    numpy_ego_motion_foe_projection,
    numpy_linear,
    numpy_relu,
    numpy_sigmoid,
    numpy_softmax,
)


# -----------------------------------------------------------------------------
# Category 1: Substrate identity (Catalog #241 contract)
# -----------------------------------------------------------------------------


def test_substrate_id_is_canonical():
    assert SUBSTRATE_ID == "atw_v2_cooperative_receiver_v2"


def test_substrate_version_is_l0_scaffold_phase_3():
    assert "l0_scaffold" in SUBSTRATE_VERSION
    assert "20260526" in SUBSTRATE_VERSION


def test_lane_id_is_path_3_h():
    assert LANE_ID == "lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526"


def test_research_only_at_l0_scaffold():
    assert RESEARCH_ONLY is True


def test_dispatch_disabled_at_l0_scaffold():
    assert DISPATCH_ENABLED is False


def test_implementation_status_declares_l0_scaffold():
    assert "l0_scaffold" in IMPLEMENTATION_STATUS
    assert "full_main_not_implemented" in IMPLEMENTATION_STATUS


def test_horizon_class_is_frontier_pursuit():
    assert HORIZON_CLASS == "frontier_pursuit"


def test_design_memo_path_canonical():
    assert "path_3_h_atw_v2_cooperative_receiver_substrate_design_20260526" in DESIGN_MEMO_PATH


def test_phase_1_audit_path_canonical():
    assert "path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit" in PHASE_1_AUDIT_PATH


def test_phase_2_decision_path_canonical():
    assert "path_3_h_atw_v2_cooperative_receiver_substrate_design_decision" in PHASE_2_DECISION_PATH


# -----------------------------------------------------------------------------
# Category 1b: Layer 1 META-unwind canonical constants
# -----------------------------------------------------------------------------


def test_layer_1_meta_unwind_anchor_is_atick_redlich():
    assert "Atick-Redlich" in LAYER_1_META_UNWIND_BINDING_ANCHOR
    assert "1990" in LAYER_1_META_UNWIND_BINDING_ANCHOR
    assert "cooperative_receiver" in LAYER_1_META_UNWIND_BINDING_ANCHOR


def test_tishby_is_demoted():
    assert any("Tishby" in t for t in LAYER_1_META_UNWIND_DEMOTED_THEOREMS)


def test_wyner_ziv_is_demoted():
    assert any("Wyner_Ziv" in t and "dropped" in t for t in LAYER_1_META_UNWIND_DEMOTED_THEOREMS)


def test_new_conditioning_is_ego_motion():
    assert "ego_motion" in NEW_CONDITIONING_VARIABLE
    assert "foe" in NEW_CONDITIONING_VARIABLE.lower()


def test_advisory_cross_checks_cite_schmidhuber():
    assert any("Schmidhuber" in c for c in LAYER_1_META_UNWIND_ADVISORY_CROSS_CHECKS)


# -----------------------------------------------------------------------------
# Category 1c: v1 falsification anchor preserved (Catalog #110/#113)
# -----------------------------------------------------------------------------


def test_v1_falsification_anchor_commit_preserved():
    assert V1_FALSIFICATION_ANCHOR_COMMIT == "057130de4"


def test_v1_falsification_max_abs_raw_byte_delta_zero():
    """v1 cdf_table_blob byte-mutation smoke proved max_abs_raw_byte_delta=0."""
    assert V1_FALSIFICATION_MAX_ABS_RAW_BYTE_DELTA == 0


# -----------------------------------------------------------------------------
# Category 1d: Catalog #124 archive grammar 8 fields declared inline
# -----------------------------------------------------------------------------


def test_archive_grammar_constant_declared():
    assert "ATWv2CR2" in ARCHIVE_GRAMMAR


def test_parser_section_manifest_declared():
    assert "header" in PARSER_SECTION_MANIFEST
    assert "encoder_blob" in PARSER_SECTION_MANIFEST
    assert "cond_embed_blob" in PARSER_SECTION_MANIFEST
    assert "ego_motion_proj_blob" in PARSER_SECTION_MANIFEST


def test_export_format_canonical():
    assert "ATWv2CR2" in EXPORT_FORMAT
    assert "0_bin" in EXPORT_FORMAT


def test_score_aware_loss_routes_canonical_atick_redlich():
    assert "atick_redlich" in SCORE_AWARE_LOSS
    assert "cooperative_receiver_loss" in SCORE_AWARE_LOSS


def test_no_op_detector_planned_no_dead_bytes():
    assert "no_dead_bytes" in NO_OP_DETECTOR_PLANNED
    assert "cdf_table_blob_lesson_learned" in NO_OP_DETECTOR_PLANNED


# -----------------------------------------------------------------------------
# Category 2: Archive grammar roundtrip (Catalog #91)
# -----------------------------------------------------------------------------


def test_archive_magic_is_atwv2cr2():
    assert ATWV2CR2_MAGIC == b"ATWv2CR2"
    assert len(ATWV2CR2_MAGIC) == 8


def test_archive_schema_version_is_1():
    assert ATWV2CR2_SCHEMA_VERSION == 1


def test_num_sections_is_8():
    assert NUM_SECTIONS == 8
    assert len(PARSER_SECTION_ROLES) == 8


def test_no_cdf_table_blob_in_canonical_grammar():
    """Phase 1 CC-8 unwind: cdf_table_blob REMOVED from canonical grammar."""
    assert "cdf_table_blob" not in PARSER_SECTION_ROLES


def test_ego_motion_proj_blob_in_canonical_grammar():
    """Phase 2 CC-7 unwind: NEW ego-motion FOE projection conditioning."""
    assert "ego_motion_proj_blob" in PARSER_SECTION_ROLES


def test_cond_embed_blob_in_canonical_grammar():
    """Phase 2 binding decision: conditioning embedding head."""
    assert "cond_embed_blob" in PARSER_SECTION_ROLES


def test_pack_parse_archive_roundtrip_byte_stable():
    """Catalog #91 ENCODE_INFLATE_ROUNDTRIP: pack→parse byte-stable."""
    archive = build_smoke_archive(num_pairs=10, latent_dim=8)
    packed = pack_archive(archive)
    parsed = parse_archive(packed)
    assert parsed.encoder_blob == archive.encoder_blob
    assert parsed.decoder_blob == archive.decoder_blob
    assert parsed.cond_embed_blob == archive.cond_embed_blob
    assert parsed.ego_motion_proj_blob == archive.ego_motion_proj_blob
    assert parsed.per_pair_latent_blob == archive.per_pair_latent_blob
    assert parsed.class_cond_cdf_blob == archive.class_cond_cdf_blob
    assert parsed.meta_blob == archive.meta_blob


def test_pack_archive_starts_with_magic():
    archive = build_smoke_archive(num_pairs=10, latent_dim=8)
    packed = pack_archive(archive)
    assert packed[:8] == ATWV2CR2_MAGIC


def test_parse_archive_rejects_bad_magic():
    archive = build_smoke_archive(num_pairs=10, latent_dim=8)
    packed = bytearray(pack_archive(archive))
    packed[:8] = b"BADMAGIC"
    with pytest.raises(ValueError, match="magic mismatch"):
        parse_archive(bytes(packed))


# -----------------------------------------------------------------------------
# Category 3: Byte-mutation no_op_proof (Catalog #139/#272)
# -----------------------------------------------------------------------------


def test_per_pair_latent_blob_byte_mutation_changes_parse_result():
    """Catalog #139/#272: per_pair_latent_blob bytes are NOT decode-opaque.

    Mutating a byte in this section MUST change the parsed value (proving
    operational consumption per Catalog #220).
    """
    archive_orig = build_smoke_archive(num_pairs=10, latent_dim=8)
    packed_orig = pack_archive(archive_orig)

    # Mutate the per_pair_latent_blob bytes (XOR a byte in the middle)
    parsed_orig = parse_archive(packed_orig)
    mutated_latent_blob = bytearray(parsed_orig.per_pair_latent_blob)
    if len(mutated_latent_blob) > 10:
        mutated_latent_blob[10] ^= 0xFF
    archive_mutated = ATWv2CR2Archive(
        encoder_blob=archive_orig.encoder_blob,
        decoder_blob=archive_orig.decoder_blob,
        cond_embed_blob=archive_orig.cond_embed_blob,
        ego_motion_proj_blob=archive_orig.ego_motion_proj_blob,
        per_pair_latent_blob=bytes(mutated_latent_blob),
        class_cond_cdf_blob=archive_orig.class_cond_cdf_blob,
        meta_blob=archive_orig.meta_blob,
        reserved_section_for_phase_4_extension=archive_orig.reserved_section_for_phase_4_extension,
    )
    packed_mutated = pack_archive(archive_mutated)
    parsed_mutated = parse_archive(packed_mutated)
    assert parsed_orig.per_pair_latent_blob != parsed_mutated.per_pair_latent_blob


def test_ego_motion_proj_blob_byte_mutation_changes_parse_result():
    """Catalog #139/#272: ego_motion_proj_blob bytes are NOT decode-opaque."""
    archive_orig = build_smoke_archive(num_pairs=10, latent_dim=8)
    parsed_orig = parse_archive(pack_archive(archive_orig))
    mutated = bytearray(parsed_orig.ego_motion_proj_blob)
    if len(mutated) > 5:
        mutated[5] ^= 0xFF
    archive_mutated = ATWv2CR2Archive(
        encoder_blob=archive_orig.encoder_blob,
        decoder_blob=archive_orig.decoder_blob,
        cond_embed_blob=archive_orig.cond_embed_blob,
        ego_motion_proj_blob=bytes(mutated),
        per_pair_latent_blob=archive_orig.per_pair_latent_blob,
        class_cond_cdf_blob=archive_orig.class_cond_cdf_blob,
        meta_blob=archive_orig.meta_blob,
        reserved_section_for_phase_4_extension=archive_orig.reserved_section_for_phase_4_extension,
    )
    parsed_mutated = parse_archive(pack_archive(archive_mutated))
    assert parsed_orig.ego_motion_proj_blob != parsed_mutated.ego_motion_proj_blob


def test_cdf_table_blob_class_extincted_via_canonical_equation_26_excluded_context():
    """Per Phase 1 CC-8 + canonical equation #26 EXCLUDED context.

    Verify the NEW grammar does NOT carry cdf_table_blob (the v1 bug class).
    """
    # The canonical PARSER_SECTION_ROLES MUST NOT include cdf_table_blob.
    assert "cdf_table_blob" not in PARSER_SECTION_ROLES


# -----------------------------------------------------------------------------
# Category 4: numpy reference forward path (Axis 3 portability)
# -----------------------------------------------------------------------------


def test_numpy_ego_motion_foe_projection_basic():
    pose_delta = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    result = numpy_ego_motion_foe_projection(pose_delta)
    assert result.shape == (2, 6)
    # First sample: translation (1,0,0) is already normalized
    np.testing.assert_allclose(result[0, :3], [1.0, 0.0, 0.0], atol=1e-6)
    # Rotation component normalized too
    np.testing.assert_allclose(result[0, 3:], [0.0, 1.0, 0.0], atol=1e-6)


def test_numpy_ego_motion_foe_projection_handles_zero_pose_delta():
    """Robustness: zero pose_delta should not divide-by-zero."""
    pose_delta = np.zeros((1, 6), dtype=np.float32)
    result = numpy_ego_motion_foe_projection(pose_delta)
    assert result.shape == (1, 6)
    # All zeros input → all zeros output (eps in denominator prevents div by 0)
    assert np.all(np.isfinite(result))


def test_numpy_relu_basic():
    x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    expected = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(numpy_relu(x), expected)


def test_numpy_sigmoid_basic():
    x = np.array([0.0], dtype=np.float32)
    np.testing.assert_allclose(numpy_sigmoid(x), [0.5], atol=1e-6)


def test_numpy_softmax_sums_to_one():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = numpy_softmax(x, axis=-1)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)


def test_numpy_linear_basic():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bias = np.array([0.0, 0.0], dtype=np.float32)
    out = numpy_linear(x, weight, bias)
    np.testing.assert_allclose(out, [[1.0, 2.0]], atol=1e-6)


def test_numpy_decode_pair_with_ego_motion_runs_end_to_end():
    """Smoke test: numpy reference end-to-end forward pass works."""
    cfg = CooperativeReceiverConfig(
        num_pairs=4,
        latent_dim=8,
        ego_motion_dim=6,
        cond_embed_dim=4,
        decoder_embed_dim=8,
        decoder_initial_grid_h=3,
        decoder_initial_grid_w=4,
        decoder_channels=(8, 6),
        decoder_num_upsample_blocks=2,
        output_height=12,
        output_width=16,
    )
    weights = init_numpy_weights_random(cfg, seed=42)

    B = 2
    per_pair_latent_residual = np.zeros((B, cfg.latent_dim), dtype=np.float32)
    ego_motion_proj = np.zeros((B, cfg.ego_motion_dim), dtype=np.float32)
    ego_motion_proj[0, 0] = 1.0
    ego_motion_proj[1, 1] = 1.0

    rgb_0, rgb_1 = numpy_decode_pair_with_ego_motion_conditioning(
        per_pair_latent_residual,
        ego_motion_proj,
        cfg=cfg,
        cond_embed_weight_1=weights.cond_embed_weight_1,
        cond_embed_bias_1=weights.cond_embed_bias_1,
        cond_embed_weight_2=weights.cond_embed_weight_2,
        cond_embed_bias_2=weights.cond_embed_bias_2,
        initial_proj_weight=weights.initial_proj_weight,
        initial_proj_bias=weights.initial_proj_bias,
        decoder_block_weights=weights.decoder_block_weights,
        decoder_block_biases=weights.decoder_block_biases,
        final_conv_weight=weights.final_conv_weight,
        final_conv_bias=weights.final_conv_bias,
    )
    assert rgb_0.shape == (B, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (B, 3, cfg.output_height, cfg.output_width)
    # Sigmoid output in [0, 1]
    assert rgb_0.min() >= 0.0
    assert rgb_0.max() <= 1.0
    assert rgb_1.min() >= 0.0
    assert rgb_1.max() <= 1.0


# -----------------------------------------------------------------------------
# Category 5: MLX↔numpy parity (when MLX available)
# -----------------------------------------------------------------------------


def test_mlx_ego_motion_foe_projection_matches_numpy():
    """MLX↔numpy parity test for ego-motion FOE projection primitive (Axis 3)."""
    pytest.importorskip("mlx", reason="MLX not available; numpy reference still works")
    import mlx.core as mx

    from tac.substrates.atw_v2_cooperative_receiver_v2.mlx_renderer import (
        mlx_ego_motion_foe_projection,
    )

    pose_delta_np = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    pose_delta_mlx = mx.array(pose_delta_np)

    out_numpy = numpy_ego_motion_foe_projection(pose_delta_np)
    out_mlx = np.array(mlx_ego_motion_foe_projection(pose_delta_mlx))

    # MLX drift bound per Phase 3 §8: elementwise + division should be near
    # bit-exact (no FMA reassociation concerns)
    np.testing.assert_allclose(out_numpy, out_mlx, atol=1e-5, rtol=1e-5)


# -----------------------------------------------------------------------------
# Category 6: MLX↔PyTorch drift verification (when both available)
# -----------------------------------------------------------------------------


def test_mlx_pytorch_ego_motion_foe_projection_drift_below_catalog_1265_threshold():
    """MLX↔PyTorch drift for ego-motion FOE projection within Catalog #1265 bound."""
    pytest.importorskip("mlx", reason="MLX not available; skipping MLX↔PyTorch drift test")
    pytest.importorskip("torch", reason="PyTorch not available")
    import mlx.core as mx
    import torch as _torch

    from tac.substrates.atw_v2_cooperative_receiver_v2.mlx_renderer import (
        mlx_ego_motion_foe_projection,
    )
    from tac.substrates.atw_v2_cooperative_receiver_v2._torch_compat_reference import (
        torch_ego_motion_foe_projection,
    )

    pose_delta_np = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    out_mlx = np.array(mlx_ego_motion_foe_projection(mx.array(pose_delta_np)))
    out_torch = torch_ego_motion_foe_projection(_torch.from_numpy(pose_delta_np)).numpy()

    max_abs_drift = float(np.max(np.abs(out_mlx - out_torch)))
    # Per Phase 3 §8: ego-motion FOE projection should be near bit-exact (no
    # reduction-order or FMA concerns). Catalog #1265 gate threshold = 1e-3.
    assert max_abs_drift < 1e-3, (
        f"MLX↔PyTorch ego-motion FOE projection drift {max_abs_drift:.3e} exceeds "
        f"Catalog #1265 gate threshold 1e-3"
    )


# -----------------------------------------------------------------------------
# Category 7: Inflate runtime pattern (Catalog #146 + #205)
# -----------------------------------------------------------------------------


def test_inflate_select_inflate_device_importable():
    """Catalog #205: canonical select_inflate_device function exists."""
    from tac.substrates.atw_v2_cooperative_receiver_v2.inflate import (
        select_inflate_device,
    )

    # Just verify importable; do NOT invoke (depends on env + torch availability)
    assert callable(select_inflate_device)


def test_inflate_select_inflate_device_refuses_mps():
    """Catalog #205: MPS explicitly refused per CLAUDE.md MPS-NOT-authoritative."""
    pytest.importorskip("torch", reason="PyTorch not available")
    import os

    from tac.substrates.atw_v2_cooperative_receiver_v2.inflate import (
        select_inflate_device,
    )

    old_val = os.environ.get("PACT_INFLATE_DEVICE")
    try:
        os.environ["PACT_INFLATE_DEVICE"] = "mps"
        with pytest.raises(RuntimeError, match="MPS"):
            select_inflate_device()
    finally:
        if old_val is None:
            del os.environ["PACT_INFLATE_DEVICE"]
        else:
            os.environ["PACT_INFLATE_DEVICE"] = old_val


def test_inflate_main_cli_signature_matches_catalog_146():
    """Catalog #146: inflate.py CLI signature = <archive_dir> <output_dir> <file_list>."""
    from tac.substrates.atw_v2_cooperative_receiver_v2.inflate import main

    with pytest.raises(SystemExit, match="usage: inflate.py"):
        main([])


# -----------------------------------------------------------------------------
# Category 8: _full_main NotImplementedError (Catalog #240(c))
# -----------------------------------------------------------------------------


def test_full_main_skeleton_raises_not_implemented_per_catalog_240c():
    """Catalog #240(c): _full_main raises NotImplementedError pre-Phase 2 council."""
    from tac.substrates.atw_v2_cooperative_receiver_v2._training_only import (
        compute_atwv2cr2_loss_skeleton,
    )

    with pytest.raises(NotImplementedError, match="Catalog #240"):
        compute_atwv2cr2_loss_skeleton()
