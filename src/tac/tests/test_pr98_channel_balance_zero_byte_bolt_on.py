# SPDX-License-Identifier: MIT
"""Tests for L28 PR98 zero-byte decode-side channel-balance bolt-on.

Slot LL landing per Slot DD canonical highest-EV-shortest-WC RANK 1 finding.
Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" +
"HNeRV / leaderboard-implementation parity discipline" L28.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.codec.pr98_channel_balance_zero_byte_bolt_on import (
    CANONICAL_EQUATION_CANDIDATE_ID,
    PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL,
    PR98_L28_ARCHIVE_BYTES_DELTA,
    PR98_L28_CANONICAL_SOURCE_LINE_RANGE,
    PR98_L28_CANONICAL_SOURCE_PATH,
    PR98_L28_EXPECTED_SCORE_DELTA_BAND,
    PROVENANCE_MODEL_ID,
    Pr98ChannelBalanceConfig,
    apply_pr98_channel_balance_to_decoded_pair,
    apply_pr98_channel_balance_to_decoded_pair_torch,
    build_axis_decomposition_for_pr98_bolt_on,
    list_candidate_substrates_for_l28_application,
    verify_pr98_channel_balance_byte_stable,
)

# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------


def test_canonical_offsets_match_pr101_inflate_lines_49_51_exactly() -> None:
    """Canonical L28 offsets MUST match PR101 hnerv_ft_microcodec inflate.py:49-51 verbatim.

    PR101 lines 49-51 canonical reference:
        up[:, 0, 0].sub_(1.0)   # frame_0 RED   -= 1.0
        up[:, 0, 2].sub_(1.0)   # frame_0 BLUE  -= 1.0
        up[:, 1, 1].sub_(1.0)   # frame_1 GREEN -= 1.0
    """
    expected = (
        (0, 0, 1.0),
        (0, 2, 1.0),
        (1, 1, 1.0),
    )
    assert expected == PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL


def test_canonical_archive_bytes_delta_is_zero_per_slot_dd_canonical() -> None:
    """L28 is canonical ZERO-BYTE bolt-on per Slot DD canonical highest-EV-shortest-WC RANK 1."""
    assert PR98_L28_ARCHIVE_BYTES_DELTA == 0


def test_canonical_score_delta_band_per_slot_dd_canonical() -> None:
    """L28 canonical score delta band: -0.0005 to -0.0001 per PR98 third-prize anchor."""
    lo, hi = PR98_L28_EXPECTED_SCORE_DELTA_BAND
    assert lo == -0.0005
    assert hi == -0.0001
    # Both negative (score-lowering)
    assert lo < 0
    assert hi < 0


def test_canonical_source_line_range_is_49_51() -> None:
    """L28 canonical source-of-truth: PR101 hnerv_ft_microcodec inflate.py:49-51."""
    assert PR98_L28_CANONICAL_SOURCE_LINE_RANGE == "49-51"


def test_canonical_source_path_points_to_pr101_hnerv_ft_microcodec() -> None:
    """Canonical reference path is PR101 hnerv_ft_microcodec inflate.py."""
    assert "public_pr101_intake" in PR98_L28_CANONICAL_SOURCE_PATH
    assert "hnerv_ft_microcodec/inflate.py" in PR98_L28_CANONICAL_SOURCE_PATH


def test_provenance_model_id_canonical_tac_codec_namespace() -> None:
    """PROVENANCE_MODEL_ID is canonical tac.codec.pr98... per Catalog #323."""
    assert PROVENANCE_MODEL_ID == "tac.codec.pr98_channel_balance_zero_byte_bolt_on"


def test_canonical_equation_candidate_id_has_v1_suffix() -> None:
    """Canonical equation candidate ID follows registry convention with _v1 suffix."""
    assert CANONICAL_EQUATION_CANDIDATE_ID.endswith("_v1")
    assert "pr98" in CANONICAL_EQUATION_CANDIDATE_ID
    assert "zero_byte" in CANONICAL_EQUATION_CANDIDATE_ID


# ---------------------------------------------------------------------------
# Pr98ChannelBalanceConfig
# ---------------------------------------------------------------------------


def test_config_construction_accepts_canonical_substrate_id() -> None:
    cfg = Pr98ChannelBalanceConfig(substrate_id="v14_v2_dqs1")
    assert cfg.substrate_id == "v14_v2_dqs1"
    assert cfg.offsets == PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL
    assert cfg.clamp_min == 0.0
    assert cfg.clamp_max == 255.0
    assert cfg.apply_in_place is True


def test_config_rejects_empty_substrate_id() -> None:
    with pytest.raises(ValueError, match="substrate_id"):
        Pr98ChannelBalanceConfig(substrate_id="")


def test_config_rejects_whitespace_only_substrate_id() -> None:
    with pytest.raises(ValueError, match="substrate_id"):
        Pr98ChannelBalanceConfig(substrate_id="   ")


def test_config_rejects_non_string_substrate_id() -> None:
    with pytest.raises(ValueError, match="substrate_id"):
        Pr98ChannelBalanceConfig(substrate_id=42)  # type: ignore[arg-type]


def test_config_rejects_empty_offsets_tuple() -> None:
    with pytest.raises(ValueError, match="offsets"):
        Pr98ChannelBalanceConfig(substrate_id="test", offsets=())


def test_config_rejects_invalid_pair_frame_index() -> None:
    with pytest.raises(ValueError, match="pair_frame_index"):
        Pr98ChannelBalanceConfig(
            substrate_id="test",
            offsets=((2, 0, 1.0),),  # 2 is invalid; only 0 or 1
        )


def test_config_rejects_invalid_rgb_channel_index() -> None:
    with pytest.raises(ValueError, match="rgb_channel_index"):
        Pr98ChannelBalanceConfig(
            substrate_id="test",
            offsets=((0, 3, 1.0),),  # 3 is invalid; only 0/1/2
        )


def test_config_rejects_malformed_offset_entry() -> None:
    with pytest.raises(ValueError, match="3-tuple"):
        Pr98ChannelBalanceConfig(
            substrate_id="test",
            offsets=((0, 0),),  # type: ignore[arg-type] (2-tuple not 3-tuple)
        )


def test_config_rejects_inverted_clamp_range() -> None:
    with pytest.raises(ValueError, match="clamp_max"):
        Pr98ChannelBalanceConfig(
            substrate_id="test", clamp_min=255.0, clamp_max=0.0
        )


def test_config_is_frozen_per_catalog_287_sister_discipline() -> None:
    """Per Catalog #287 sister discipline + canonical immutable contract."""
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    with pytest.raises((AttributeError, TypeError)):
        cfg.substrate_id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# apply_pr98_channel_balance_to_decoded_pair
# ---------------------------------------------------------------------------


def test_canonical_subtraction_frame_0_red_channel() -> None:
    """frame_0 RED channel (idx 0, 0) is decremented by 1.0 per L28 canonical."""
    rgb = np.full((1, 2, 3, 4, 4), 128.0, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    out = apply_pr98_channel_balance_to_decoded_pair(rgb.copy(), cfg)
    assert (out[:, 0, 0] == 127.0).all()


def test_canonical_subtraction_frame_0_blue_channel() -> None:
    """frame_0 BLUE channel (idx 0, 2) is decremented by 1.0 per L28 canonical."""
    rgb = np.full((1, 2, 3, 4, 4), 128.0, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    out = apply_pr98_channel_balance_to_decoded_pair(rgb.copy(), cfg)
    assert (out[:, 0, 2] == 127.0).all()


def test_canonical_subtraction_frame_1_green_channel() -> None:
    """frame_1 GREEN channel (idx 1, 1) is decremented by 1.0 per L28 canonical."""
    rgb = np.full((1, 2, 3, 4, 4), 128.0, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    out = apply_pr98_channel_balance_to_decoded_pair(rgb.copy(), cfg)
    assert (out[:, 1, 1] == 127.0).all()


def test_unaffected_channels_remain_unchanged() -> None:
    """Channels NOT in canonical L28 offsets MUST remain unchanged."""
    rgb = np.full((1, 2, 3, 4, 4), 128.0, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    out = apply_pr98_channel_balance_to_decoded_pair(rgb.copy(), cfg)
    # Unaffected: frame_0 GREEN (0,1), frame_1 RED (1,0), frame_1 BLUE (1,2)
    assert (out[:, 0, 1] == 128.0).all()
    assert (out[:, 1, 0] == 128.0).all()
    assert (out[:, 1, 2] == 128.0).all()


def test_in_place_mutation_default() -> None:
    """apply_in_place=True (default) mutates the input array."""
    rgb = np.full((1, 2, 3, 4, 4), 128.0, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test", apply_in_place=True)
    out = apply_pr98_channel_balance_to_decoded_pair(rgb, cfg)
    # Output is the same array (in-place mutation)
    assert out is rgb
    # frame_0 RED was mutated
    assert (rgb[:, 0, 0] == 127.0).all()


def test_apply_in_place_false_returns_new_array() -> None:
    """apply_in_place=False returns a new array; input unchanged."""
    rgb = np.full((1, 2, 3, 4, 4), 128.0, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test", apply_in_place=False)
    out = apply_pr98_channel_balance_to_decoded_pair(rgb, cfg)
    # Output is a different array
    assert out is not rgb
    # Input unchanged
    assert (rgb[:, 0, 0] == 128.0).all()
    # Output mutated
    assert (out[:, 0, 0] == 127.0).all()


def test_rejects_wrong_ndim_shape() -> None:
    rgb_4d = np.zeros((2, 3, 4, 4), dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    with pytest.raises(ValueError, match="5-D"):
        apply_pr98_channel_balance_to_decoded_pair(rgb_4d, cfg)


def test_rejects_wrong_pair_axis_size() -> None:
    rgb = np.zeros((1, 3, 3, 4, 4), dtype=np.float32)  # axis 1 = 3, should be 2
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    with pytest.raises(ValueError, match="axis 1 must be 2"):
        apply_pr98_channel_balance_to_decoded_pair(rgb, cfg)


def test_rejects_wrong_channel_axis_size() -> None:
    rgb = np.zeros((1, 2, 4, 4, 4), dtype=np.float32)  # axis 2 = 4, should be 3
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    with pytest.raises(ValueError, match="axis 2 must be 3"):
        apply_pr98_channel_balance_to_decoded_pair(rgb, cfg)


def test_canonical_clamp_to_0_255_per_pr101_inflate_line_54() -> None:
    """Per PR101 inflate.py:54 canonical clamp BEFORE round-and-uint8 cast.

    Values below 0 after subtraction are clamped to 0; values above 255
    are clamped to 255. The clamp preserves the canonical PR101 contract.
    """
    rgb = np.full((1, 2, 3, 4, 4), 0.5, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    out = apply_pr98_channel_balance_to_decoded_pair(rgb.copy(), cfg)
    # frame_0 RED was 0.5 - 1.0 = -0.5; clamped to 0
    assert (out[:, 0, 0] == 0.0).all()


def test_default_config_when_none_passed() -> None:
    """When config=None, helper uses default canonical config."""
    rgb = np.full((1, 2, 3, 4, 4), 128.0, dtype=np.float32)
    out = apply_pr98_channel_balance_to_decoded_pair(rgb.copy(), None)
    # Canonical L28 applied
    assert (out[:, 0, 0] == 127.0).all()
    assert (out[:, 0, 2] == 127.0).all()
    assert (out[:, 1, 1] == 127.0).all()


def test_canonical_pr101_batch_shape_16_pairs() -> None:
    """Canonical PR101 batch shape per inflate.py:39-44: batch up to 16 pairs."""
    rgb = np.full((16, 2, 3, 874, 1164), 128.0, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test", apply_in_place=False)
    out = apply_pr98_channel_balance_to_decoded_pair(rgb, cfg)
    assert out.shape == (16, 2, 3, 874, 1164)
    # Spot-check a single pair
    assert (out[0, 0, 0] == 127.0).all()


def test_multi_pair_batch_all_pairs_get_canonical_subtraction() -> None:
    """All pairs in a batch get the canonical L28 subtraction (broadcast)."""
    rgb = np.full((4, 2, 3, 8, 8), 200.0, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="test")
    out = apply_pr98_channel_balance_to_decoded_pair(rgb.copy(), cfg)
    # Every pair gets the canonical -1.0 on frame_0 R/B + frame_1 G
    for pair_idx in range(4):
        assert (out[pair_idx, 0, 0] == 199.0).all()
        assert (out[pair_idx, 0, 2] == 199.0).all()
        assert (out[pair_idx, 1, 1] == 199.0).all()


def test_torch_helper_matches_numpy_helper_semantics() -> None:
    """Torch runtimes must route through the same L28 contract as numpy runtimes."""
    torch = pytest.importorskip("torch")

    np_rgb = np.full((2, 2, 3, 3, 4), 128.0, dtype=np.float32)
    torch_rgb = torch.from_numpy(np_rgb.copy())
    cfg = Pr98ChannelBalanceConfig(substrate_id="torch_test", apply_in_place=False)

    np_out = apply_pr98_channel_balance_to_decoded_pair(np_rgb, cfg)
    torch_out = apply_pr98_channel_balance_to_decoded_pair_torch(torch_rgb, cfg)

    assert torch_out is not torch_rgb
    assert np.array_equal(torch_out.numpy(), np_out)
    assert torch.all(torch_rgb == 128.0)


def test_torch_helper_mutates_in_place_by_default() -> None:
    """PR101-style torch use keeps in-place sub_ semantics by default."""
    torch = pytest.importorskip("torch")

    rgb = torch.full((1, 2, 3, 2, 2), 200.0)
    out = apply_pr98_channel_balance_to_decoded_pair_torch(
        rgb,
        Pr98ChannelBalanceConfig(substrate_id="torch_test"),
    )

    assert out is rgb
    assert torch.all(rgb[:, 0, 0] == 199.0)
    assert torch.all(rgb[:, 0, 1] == 200.0)
    assert torch.all(rgb[:, 0, 2] == 199.0)
    assert torch.all(rgb[:, 1, 0] == 200.0)
    assert torch.all(rgb[:, 1, 1] == 199.0)
    assert torch.all(rgb[:, 1, 2] == 200.0)


def test_torch_helper_rejects_wrong_shape() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="5-D"):
        apply_pr98_channel_balance_to_decoded_pair_torch(
            torch.zeros((2, 3, 4, 4)),
            Pr98ChannelBalanceConfig(substrate_id="torch_test"),
        )


# ---------------------------------------------------------------------------
# verify_pr98_channel_balance_byte_stable
# ---------------------------------------------------------------------------


def test_byte_stability_holds_for_canonical_input() -> None:
    """L28 bolt-on is byte-stable across repeated invocations on same input."""
    rgb = np.full((2, 2, 3, 8, 8), 128.0, dtype=np.float32)
    assert verify_pr98_channel_balance_byte_stable(rgb) is True


def test_byte_stability_with_explicit_config() -> None:
    rgb = np.full((1, 2, 3, 4, 4), 100.0, dtype=np.float32)
    cfg = Pr98ChannelBalanceConfig(substrate_id="byte_stability_test")
    assert verify_pr98_channel_balance_byte_stable(rgb, cfg) is True


def test_byte_stability_at_clamp_edge() -> None:
    """Byte-stability holds even when subtraction triggers clamp at 0."""
    rgb = np.full((1, 2, 3, 4, 4), 0.5, dtype=np.float32)
    assert verify_pr98_channel_balance_byte_stable(rgb) is True


def test_byte_stability_at_upper_clamp_edge() -> None:
    rgb = np.full((1, 2, 3, 4, 4), 255.0, dtype=np.float32)
    assert verify_pr98_channel_balance_byte_stable(rgb) is True


# ---------------------------------------------------------------------------
# build_axis_decomposition_for_pr98_bolt_on
# ---------------------------------------------------------------------------


def test_axis_decomposition_archive_bytes_delta_is_zero() -> None:
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="v14_v2_dqs1",
        current_archive_bytes=174000,
        current_d_pose=3.4e-5,
    )
    assert axis["predicted_archive_bytes_delta"] == 0


def test_axis_decomposition_has_canonical_provenance() -> None:
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    provenance = axis["canonical_provenance"]
    assert isinstance(provenance, dict)
    assert provenance["evidence_grade"] == "predicted"
    assert provenance["promotion_eligible"] is False
    assert provenance["score_claim_valid"] is False
    # canonical_helper_invocation points to the canonical builder per
    # Catalog #323; PROVENANCE_MODEL_ID is the predictor identifier
    # surfaced via source_path
    assert (
        "tac.provenance.builders.build_provenance_for_predicted"
        in provenance["canonical_helper_invocation"]
    )
    assert PROVENANCE_MODEL_ID in provenance["source_path"]


def test_axis_decomposition_tier_a_canonical_routing_markers() -> None:
    """Per Catalog #341: axis_tag=[predicted] + promotable=False structural."""
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    assert axis["axis_tag"] == "[predicted]"


def test_axis_decomposition_default_per_axis_split() -> None:
    """Default seg/pose split is 50%/50% of -0.0003 mid-band per Slot DD."""
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    assert axis["predicted_d_seg_delta"] == -0.00015
    assert axis["predicted_d_pose_delta"] == -0.00015


def test_axis_decomposition_accepts_explicit_per_axis_deltas() -> None:
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test",
        current_archive_bytes=100000,
        current_d_pose=0.001,
        predicted_d_seg_delta=-0.0004,
        predicted_d_pose_delta=-0.0001,
    )
    assert axis["predicted_d_seg_delta"] == -0.0004
    assert axis["predicted_d_pose_delta"] == -0.0001


def test_axis_decomposition_substrate_id_threaded_through() -> None:
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="v14_v2_dqs1",
        current_archive_bytes=174000,
        current_d_pose=3.4e-5,
    )
    assert axis["substrate_id"] == "v14_v2_dqs1"


def test_axis_decomposition_carries_canonical_equation_candidate() -> None:
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    assert axis["canonical_equation_candidate"] == CANONICAL_EQUATION_CANDIDATE_ID


def test_axis_decomposition_carries_canonical_source_reference() -> None:
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    assert "PR101" in axis["canonical_source"]
    assert PR98_L28_CANONICAL_SOURCE_LINE_RANGE in axis["canonical_source"]


def test_axis_decomposition_deterministic_for_same_input() -> None:
    """Identical context produces identical canonical Provenance hash."""
    axis_a = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    axis_b = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    assert axis_a["canonical_provenance"]["source_sha256"] == (
        axis_b["canonical_provenance"]["source_sha256"]
    )


# ---------------------------------------------------------------------------
# list_candidate_substrates_for_l28_application
# ---------------------------------------------------------------------------


def test_candidate_enumeration_includes_v14_v2_dqs1() -> None:
    candidates = list_candidate_substrates_for_l28_application()
    substrate_ids = [c["substrate_id"] for c in candidates]
    assert "v14_v2_dqs1" in substrate_ids


def test_candidate_enumeration_includes_fec6_canonical_frontier() -> None:
    candidates = list_candidate_substrates_for_l28_application()
    substrate_ids = [c["substrate_id"] for c in candidates]
    assert "fec6_canonical_frontier_pointer" in substrate_ids


def test_candidate_enumeration_includes_pr106_format0d() -> None:
    candidates = list_candidate_substrates_for_l28_application()
    substrate_ids = [c["substrate_id"] for c in candidates]
    assert "pr106_format0d" in substrate_ids


def test_candidate_enumeration_includes_nscs06_v8_stacked() -> None:
    candidates = list_candidate_substrates_for_l28_application()
    substrate_ids = [c["substrate_id"] for c in candidates]
    assert any("nscs06" in sid for sid in substrate_ids)


def test_candidate_enumeration_all_have_canonical_score_delta_band() -> None:
    """Every candidate has the canonical L28 score delta band per Slot DD."""
    candidates = list_candidate_substrates_for_l28_application()
    for candidate in candidates:
        band = candidate["estimated_score_delta_band"]
        assert band[0] == -0.0005
        assert band[1] == -0.0001


def test_candidate_enumeration_all_operator_routable_paired_cuda() -> None:
    """Per Catalog #246: all candidates are operator-routable for paired-CUDA."""
    candidates = list_candidate_substrates_for_l28_application()
    for candidate in candidates:
        assert candidate["operator_routable_paired_cuda_ratification"] is True


def test_v14_v2_dqs1_carries_canonical_archive_sha_prefix() -> None:
    """V14-V2 DQS1 sub-frontier canonical sha per Slot DD: 7a0da5d0fc327cba."""
    candidates = list_candidate_substrates_for_l28_application()
    dqs1 = next(c for c in candidates if c["substrate_id"] == "v14_v2_dqs1")
    assert dqs1["canonical_archive_sha_prefix"] == "7a0da5d0fc327cba"


def test_fec6_carries_canonical_archive_sha_prefix() -> None:
    """fec6 canonical frontier sha per Catalog #343: b7106c9bdbb8."""
    candidates = list_candidate_substrates_for_l28_application()
    fec6 = next(
        c for c in candidates if c["substrate_id"] == "fec6_canonical_frontier_pointer"
    )
    assert fec6["canonical_archive_sha_prefix"] == "b7106c9bdbb8"


def test_pr106_carries_canonical_archive_sha_prefix() -> None:
    """PR106 format0d canonical CUDA frontier sha per Catalog #343: 9cb989cef519."""
    candidates = list_candidate_substrates_for_l28_application()
    pr106 = next(c for c in candidates if c["substrate_id"] == "pr106_format0d")
    assert pr106["canonical_archive_sha_prefix"] == "9cb989cef519"


# ---------------------------------------------------------------------------
# Cathedral consumer integration (Catalog #335)
# ---------------------------------------------------------------------------


def test_cathedral_consumer_imports_cleanly() -> None:
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    assert pr98_channel_balance_consumer.CONSUMER_NAME == "pr98_channel_balance_consumer"
    assert pr98_channel_balance_consumer.CONSUMER_VERSION == "0.1.0"


def test_cathedral_consumer_satisfies_catalog_335_contract() -> None:
    from tac.cathedral.consumer_contract import validate_consumer_module
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    verdict = validate_consumer_module(pr98_channel_balance_consumer)
    assert verdict.contract_compliant, (
        f"Contract violations: {verdict.validation_errors}"
    )


def test_cathedral_consumer_hook_4_active() -> None:
    from tac.cathedral.consumer_contract import HookNumber
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in (
        pr98_channel_balance_consumer.CONSUMER_HOOK_NUMBERS
    )


def test_cathedral_consumer_hook_5_active() -> None:
    from tac.cathedral.consumer_contract import HookNumber
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in (
        pr98_channel_balance_consumer.CONSUMER_HOOK_NUMBERS
    )


def test_cathedral_consumer_consume_candidate_pr_95_family_match() -> None:
    """Per Slot DD: L28 applies to PR-95-family decoder substrates."""
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    candidate = {"substrate_id": "v14_v2_dqs1"}
    contribution = pr98_channel_balance_consumer.consume_candidate(candidate)
    assert "L28 PR98 zero-byte" in contribution["rationale"]
    assert "applicable" in contribution["rationale"]


def test_cathedral_consumer_consume_candidate_non_pr_95_no_match() -> None:
    """Per Catalog #341: non-applicable substrate gets observability-only annotation."""
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    candidate = {"substrate_id": "completely_unrelated_substrate_xyz"}
    contribution = pr98_channel_balance_consumer.consume_candidate(candidate)
    assert "not structurally applicable" in contribution["rationale"]


def test_cathedral_consumer_tier_a_canonical_routing_markers() -> None:
    """Per Catalog #341: contributions ALWAYS carry Tier A markers."""
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    for substrate_id in ("v14_v2_dqs1", "fec6", "completely_unrelated"):
        candidate = {"substrate_id": substrate_id}
        contribution = pr98_channel_balance_consumer.consume_candidate(candidate)
        assert contribution["predicted_delta_adjustment"] == 0.0
        assert contribution["axis_tag"] == "[predicted]"
        assert contribution["promotable"] is False


def test_cathedral_consumer_update_from_anchor_no_op() -> None:
    """Per consumer docstring: refit lives at canonical equation surface, not here."""
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    # Must not raise on any anchor input
    pr98_channel_balance_consumer.update_from_anchor({"any": "anchor"})
    pr98_channel_balance_consumer.update_from_anchor(None)


def test_cathedral_consumer_handles_candidate_without_substrate_token() -> None:
    """Missing substrate_id field defaults to non-applicable annotation."""
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    candidate = {"unrelated_field": "value"}
    contribution = pr98_channel_balance_consumer.consume_candidate(candidate)
    assert contribution["axis_tag"] == "[predicted]"
    assert contribution["promotable"] is False


def test_cathedral_consumer_recognizes_lane_id_token() -> None:
    """Consumer can resolve substrate via lane_id field (canonical sister field)."""
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    candidate = {"lane_id": "lane_pr101_frame_exploit_selector_fec6_clean"}
    contribution = pr98_channel_balance_consumer.consume_candidate(candidate)
    # Should match: "fec6" and "pr101" are PR-95-family applicable tokens
    assert "applicable" in contribution["rationale"]


def test_cathedral_consumer_case_insensitive_token_match() -> None:
    from tac.cathedral_consumers import pr98_channel_balance_consumer

    candidate = {"substrate_id": "PR101_UPPERCASE_VARIANT"}
    contribution = pr98_channel_balance_consumer.consume_candidate(candidate)
    assert "applicable" in contribution["rationale"]


# ---------------------------------------------------------------------------
# Canonical Provenance integration (Catalog #323)
# ---------------------------------------------------------------------------


def test_canonical_provenance_uses_pr98_model_id() -> None:
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    helper_invocation = axis["canonical_provenance"]["canonical_helper_invocation"]
    assert "tac.provenance.builders.build_provenance_for_predicted" in helper_invocation


def test_canonical_provenance_grade_is_predicted() -> None:
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    assert axis["canonical_provenance"]["evidence_grade"] == "predicted"


def test_canonical_provenance_axis_tag_predicted() -> None:
    axis = build_axis_decomposition_for_pr98_bolt_on(
        substrate_id="test", current_archive_bytes=100000, current_d_pose=0.001
    )
    assert axis["canonical_provenance"]["measurement_axis"] == "[predicted]"


# ---------------------------------------------------------------------------
# Live-repo regression guards
# ---------------------------------------------------------------------------


def test_canonical_helper_module_importable_from_canonical_path() -> None:
    """Helper module is importable at canonical tac.codec.* namespace."""
    import importlib

    module = importlib.import_module(
        "tac.codec.pr98_channel_balance_zero_byte_bolt_on"
    )
    assert hasattr(module, "apply_pr98_channel_balance_to_decoded_pair")
    assert hasattr(module, "Pr98ChannelBalanceConfig")
    assert hasattr(module, "verify_pr98_channel_balance_byte_stable")
    assert hasattr(module, "build_axis_decomposition_for_pr98_bolt_on")
    assert hasattr(module, "list_candidate_substrates_for_l28_application")


def test_cathedral_consumer_module_importable_from_canonical_path() -> None:
    """Cathedral consumer module is importable at canonical
    tac.cathedral_consumers.* namespace per Catalog #335 auto-discovery."""
    import importlib

    module = importlib.import_module(
        "tac.cathedral_consumers.pr98_channel_balance_consumer"
    )
    assert hasattr(module, "CONSUMER_NAME")
    assert hasattr(module, "CONSUMER_VERSION")
    assert hasattr(module, "CONSUMER_HOOK_NUMBERS")
    assert hasattr(module, "update_from_anchor")
    assert hasattr(module, "consume_candidate")


def test_canonical_helper_public_api_exported_via_all() -> None:
    """All canonical helper public API surfaces are in __all__."""
    from tac.codec import pr98_channel_balance_zero_byte_bolt_on as mod

    expected = {
        "PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL",
        "PR98_L28_EXPECTED_SCORE_DELTA_BAND",
        "PR98_L28_ARCHIVE_BYTES_DELTA",
        "PR98_L28_CANONICAL_SOURCE_LINE_RANGE",
        "PR98_L28_CANONICAL_SOURCE_PATH",
        "PROVENANCE_MODEL_ID",
        "CANONICAL_EQUATION_CANDIDATE_ID",
        "Pr98ChannelBalanceConfig",
        "apply_pr98_channel_balance_to_decoded_pair",
        "verify_pr98_channel_balance_byte_stable",
        "build_axis_decomposition_for_pr98_bolt_on",
        "list_candidate_substrates_for_l28_application",
    }
    assert expected.issubset(set(mod.__all__))


def test_cathedral_consumer_public_api_exported_via_all() -> None:
    from tac.cathedral_consumers import pr98_channel_balance_consumer as mod

    expected = {
        "CONSUMER_NAME",
        "CONSUMER_VERSION",
        "CONSUMER_HOOK_NUMBERS",
        "update_from_anchor",
        "consume_candidate",
    }
    assert expected.issubset(set(mod.__all__))
