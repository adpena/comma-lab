"""Tests for the PR98 decode-side nudge primitive.

Covers:
- The 3 slots match PR98's source byte-faithfully:
  ``up[:, 0, 0].sub_(1.0)``, ``up[:, 0, 2].sub_(1.0)``, ``up[:, 1, 1].sub_(1.0)``
- target_modes is locked to ``("contest_one_video_replay",)`` only;
  ``contest_generalized`` / ``production_generalized`` are refused at
  construction time.
- byte-cost stipulation: composition gating refuses without explicit eval.
- Catalog #91 roundtrip-style invariant: forward application then negate-back
  recovers the original tensor exactly.
- Tensor shape validation: refuses non-(B, 2, 3, H, W) shapes.
- Score-claim discipline: SCORE_CLAIM, PROMOTION_ELIGIBLE,
  READY_FOR_EXACT_EVAL_DISPATCH all False.
"""
from __future__ import annotations

import math

import pytest
import torch

from tac.packet_compiler.pr98_decode_side_nudge import (
    DEPLOYMENT_TARGET,
    ESTIMATED_BYTE_COST_AS_SIDECAR_BYTES,
    ESTIMATED_RUNTIME_CODE_OVERHEAD_LOC,
    PR98_NUDGE_SLOTS,
    PROMOTION_ELIGIBLE,
    READY_FOR_EXACT_EVAL_DISPATCH,
    SCORE_CLAIM,
    SOURCE_PORT_REFERENCE,
    TARGET_MODES,
    PR98DecodeSideNudge,
    apply_pr98_decode_side_nudge_inplace,
)


# ── Source byte-faithfulness ────────────────────────────────────────────────


def test_pr98_nudge_slots_match_inflate_py_source_byte_faithfully() -> None:
    """The 3 slots are exactly PR98's hnerv_muon_finetuned/inflate.py:55-57.

        up[:, 0, 0].sub_(1.0)  -> (0, 0, -1.0)
        up[:, 0, 2].sub_(1.0)  -> (0, 2, -1.0)
        up[:, 1, 1].sub_(1.0)  -> (1, 1, -1.0)
    """
    expected = (
        (0, 0, -1.0),
        (0, 2, -1.0),
        (1, 1, -1.0),
    )
    assert PR98_NUDGE_SLOTS == expected


def test_pr98_nudge_dataclass_default_matches_module_constant() -> None:
    nudge = PR98DecodeSideNudge()
    assert nudge.slots == PR98_NUDGE_SLOTS


def test_source_port_reference_points_at_pr98_inflate_py() -> None:
    assert "public_pr98_intake" in SOURCE_PORT_REFERENCE
    assert "hnerv_muon_finetuned_from_pr95" in SOURCE_PORT_REFERENCE
    assert "inflate.py:55-57" in SOURCE_PORT_REFERENCE


# ── target_modes lockdown ───────────────────────────────────────────────────


def test_target_modes_default_is_contest_one_video_replay_only() -> None:
    assert TARGET_MODES == ("contest_one_video_replay",)
    assert PR98DecodeSideNudge().target_modes == ("contest_one_video_replay",)


def test_pr98_nudge_refuses_contest_generalized_at_construction() -> None:
    with pytest.raises(ValueError, match="contest_generalized"):
        PR98DecodeSideNudge(
            target_modes=("contest_one_video_replay", "contest_generalized")
        )


def test_pr98_nudge_refuses_production_generalized_at_construction() -> None:
    with pytest.raises(ValueError, match="production_generalized"):
        PR98DecodeSideNudge(target_modes=("production_generalized",))


def test_deployment_target_is_t4_contest_runtime() -> None:
    assert DEPLOYMENT_TARGET == "t4_contest_runtime"
    assert PR98DecodeSideNudge().deployment_target == "t4_contest_runtime"


# ── Byte-cost stipulation ───────────────────────────────────────────────────


def test_estimated_byte_cost_sidecar_approximates_28kb_at_600_pairs() -> None:
    """For PR98's 600 frame-pair archive the cost is roughly the council target."""
    nudge = PR98DecodeSideNudge()
    cost = nudge.estimated_byte_cost_as_sidecar(n_pairs=600)
    # 3 × 600 × ceil(log2(600*2*3))/8 + 6 ≈ 12-byte addressing overhead bound.
    address_bits = math.ceil(math.log2(600 * 2 * 3))
    expected = 6 + math.ceil(3 * 600 * address_bits / 8)
    assert cost == expected
    # Sanity: not zero, not absurd.
    assert 1_000 < cost < 100_000


def test_estimated_byte_cost_sidecar_rejects_zero_pairs() -> None:
    nudge = PR98DecodeSideNudge()
    with pytest.raises(ValueError, match="n_pairs"):
        nudge.estimated_byte_cost_as_sidecar(n_pairs=0)


def test_refuse_composition_without_byte_cost_evaluation_raises_on_none() -> None:
    nudge = PR98DecodeSideNudge()
    with pytest.raises(ValueError, match="byte-cost"):
        nudge.refuse_composition_without_byte_cost_evaluation(None)


def test_refuse_composition_without_byte_cost_evaluation_refuses_negative() -> None:
    nudge = PR98DecodeSideNudge()
    with pytest.raises(ValueError, match=">= 0"):
        nudge.refuse_composition_without_byte_cost_evaluation(-1)


def test_refuse_composition_passes_with_nonneg_int() -> None:
    nudge = PR98DecodeSideNudge()
    # Returns None, doesn't raise.
    assert nudge.refuse_composition_without_byte_cost_evaluation(0) is None
    assert nudge.refuse_composition_without_byte_cost_evaluation(28_000) is None


def test_module_constants_estimated_byte_cost_and_loc_are_reasonable() -> None:
    """The exported module constants match the council's binding stipulation."""
    # ~28 KB target.
    assert 20_000 < ESTIMATED_BYTE_COST_AS_SIDECAR_BYTES < 50_000
    # PR98's hardcoded nudge is 3 LOC.
    assert ESTIMATED_RUNTIME_CODE_OVERHEAD_LOC == 3


# ── In-place nudge application ──────────────────────────────────────────────


def test_apply_pr98_nudge_subtracts_one_from_three_slots() -> None:
    """Forward application subtracts 1.0 from exactly the 3 PR98 slots."""
    rng = torch.Generator().manual_seed(0)
    tensor = torch.randn((2, 2, 3, 8, 8), generator=rng) * 50.0 + 128.0
    original = tensor.clone()
    apply_pr98_decode_side_nudge_inplace(tensor)
    # Frame 0, channel 0: subtracted 1.0
    assert torch.allclose(tensor[:, 0, 0], original[:, 0, 0] - 1.0)
    # Frame 0, channel 2: subtracted 1.0
    assert torch.allclose(tensor[:, 0, 2], original[:, 0, 2] - 1.0)
    # Frame 1, channel 1: subtracted 1.0
    assert torch.allclose(tensor[:, 1, 1], original[:, 1, 1] - 1.0)
    # Untouched: frame 0 channel 1, frame 1 channels 0 and 2.
    assert torch.equal(tensor[:, 0, 1], original[:, 0, 1])
    assert torch.equal(tensor[:, 1, 0], original[:, 1, 0])
    assert torch.equal(tensor[:, 1, 2], original[:, 1, 2])


def test_apply_pr98_nudge_is_in_place_returns_same_tensor() -> None:
    tensor = torch.zeros((1, 2, 3, 4, 4))
    out = apply_pr98_decode_side_nudge_inplace(tensor)
    assert out is tensor


def test_apply_pr98_nudge_roundtrip_with_negated_offset_recovers_original() -> None:
    """Catalog #91 roundtrip invariant: apply + apply-negated returns to start."""
    rng = torch.Generator().manual_seed(7)
    tensor = torch.randn((1, 2, 3, 4, 4), generator=rng) * 100.0
    original = tensor.clone()
    apply_pr98_decode_side_nudge_inplace(tensor)
    # Now apply +1.0 offsets to negate the -1.0 of PR98.
    inverse = PR98DecodeSideNudge(
        slots=tuple(
            (frame_in_pair, channel, -offset)
            for frame_in_pair, channel, offset in PR98_NUDGE_SLOTS
        )
    )
    apply_pr98_decode_side_nudge_inplace(tensor, nudge=inverse)
    assert torch.allclose(tensor, original)


def test_apply_pr98_nudge_refuses_wrong_shape() -> None:
    tensor = torch.zeros((4, 4))  # rank 2
    with pytest.raises(ValueError, match=r"\(B, 2, 3, H, W\)"):
        apply_pr98_decode_side_nudge_inplace(tensor)


def test_apply_pr98_nudge_refuses_wrong_frame_or_channel_dims() -> None:
    tensor = torch.zeros((1, 3, 3, 4, 4))  # frames=3 not 2
    with pytest.raises(ValueError, match=r"\(B, 2, 3, H, W\)"):
        apply_pr98_decode_side_nudge_inplace(tensor)
    tensor2 = torch.zeros((1, 2, 4, 4, 4))  # channels=4 not 3
    with pytest.raises(ValueError, match=r"\(B, 2, 3, H, W\)"):
        apply_pr98_decode_side_nudge_inplace(tensor2)


def test_apply_pr98_nudge_refuses_non_tensor_input() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        apply_pr98_decode_side_nudge_inplace([0.0, 1.0, 2.0])


# ── Slot validation ─────────────────────────────────────────────────────────


def test_pr98_nudge_refuses_out_of_range_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_in_pair"):
        PR98DecodeSideNudge(slots=((2, 0, -1.0),))


def test_pr98_nudge_refuses_out_of_range_channel_index() -> None:
    with pytest.raises(ValueError, match="channel"):
        PR98DecodeSideNudge(slots=((0, 5, -1.0),))


def test_pr98_nudge_refuses_malformed_slot_tuple() -> None:
    with pytest.raises(ValueError, match="frame_in_pair, channel, offset"):
        PR98DecodeSideNudge(slots=((0, 0),))  # missing offset


# ── Score-claim discipline ──────────────────────────────────────────────────


def test_score_claim_flags_are_all_false() -> None:
    assert SCORE_CLAIM is False
    assert PROMOTION_ELIGIBLE is False
    assert READY_FOR_EXACT_EVAL_DISPATCH is False
