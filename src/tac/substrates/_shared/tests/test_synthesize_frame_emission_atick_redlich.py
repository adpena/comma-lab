# SPDX-License-Identifier: MIT
"""Tests for the SHARED SYNTHESIZE_FRAME emission helper.

Per Slot R landing memo + design memo at
``.omx/research/synthesize_frame_emission_atick_redlich_shared_substrate_module_design_20260529.md``.

Coverage:
- Canonical contract (frozen dataclass invariants + Tier A markers per
  Catalog #341 + per-axis decomposition per Catalog #356 + Provenance
  per Catalog #323).
- Per-substrate adapter correctness (Protocol satisfaction).
- Byte-stability per Catalog #266 (deterministic byte output).
- Public API integration end-to-end.
- Sister cargo-cult unwind A8 (Protocol-based 4th sister cell).
"""

from __future__ import annotations

import struct

import pytest
import torch
import torch.nn as nn

from tac.codec.cooperative_receiver.atick_redlich import (
    AtickRedlichWeights,
    CooperativeReceiverOutput,
)
from tac.substrates._shared.synthesize_frame_emission_atick_redlich import (
    PREDICTED_BAND_PER_SUBSTRATE,
    PROVENANCE_MODEL_ID,
    RECOGNIZED_FRAMEWORK_BACKENDS,
    RECOGNIZED_SUBSTRATE_IDS,
    AtickRedlichReceiver,
    SubstrateGrammarAdapter,
    SynthesizeFrameEmissionConfig,
    SynthesizeFrameEmissionPerPairResult,
    build_atick_redlich_cooperative_receiver_for_substrate,
    synthesize_frame_emission_per_pair,
    verify_synthesize_frame_emission_byte_stability,
)


# ---------------------------------------------------------------------------
# Stand-in scorer fixtures (mirror canonical pattern in
# src/tac/codec/cooperative_receiver/tests/test_atick_redlich.py)
# ---------------------------------------------------------------------------

class _StandinSegScorer(nn.Module):
    """Upstream-contract SegNet stand-in with ``preprocess_input``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=1, bias=False)

    def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_btchw.shape
        return x_btchw[:, -1]

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


class _StandinPoseScorer(nn.Module):
    """Upstream-contract PoseNet stand-in returning a pose dict."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 6, bias=False)

    def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_btchw.shape
        flat = x_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
        flat6 = flat.expand(-1, 6, -1, -1)
        flat12 = flat6.reshape(b, t * 6, h, w)
        return flat12[:, :, ::2, ::2]

    def forward(self, x_b12hw: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pose": self.proj(x_b12hw.flatten(2).mean(dim=2))}


def _toy_pair(batch: int = 1, h: int = 32, w: int = 48, seed: int = 0):
    """Deterministic RGB pair fixture for tests."""
    gen = torch.Generator().manual_seed(seed)
    rgb_0 = (torch.rand(batch, 3, h, w, generator=gen) * 255.0).requires_grad_(True)
    rgb_1 = (torch.rand(batch, 3, h, w, generator=gen) * 255.0).requires_grad_(True)
    gt_0 = torch.rand(batch, 3, h, w, generator=gen) * 255.0
    gt_1 = torch.rand(batch, 3, h, w, generator=gen) * 255.0
    return rgb_0, rgb_1, gt_0, gt_1


def _toy_injector(archive: bytes, metadata: bytes) -> tuple[bytes, int]:
    """Toy per-substrate adapter injector for tests.

    Concatenates archive + metadata as the canonical injection; returns
    bytes added to the archive (signed; positive = archive grew).
    """
    return archive + metadata, len(metadata)


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------

def test_recognized_substrate_ids_include_slot_l_3_cells() -> None:
    """Slot L 3 symposia cells are all in the canonical recognized set."""
    for cell in ("pr110", "pr101", "dqs1"):
        assert cell in RECOGNIZED_SUBSTRATE_IDS


def test_recognized_substrate_ids_include_protocol_test() -> None:
    """``protocol_test`` placeholder is admitted for Protocol-conformance tests."""
    assert "protocol_test" in RECOGNIZED_SUBSTRATE_IDS


def test_recognized_framework_backends_per_8th_mlx_first() -> None:
    """3 backends recognized per 8th MLX-first standing directive."""
    assert RECOGNIZED_FRAMEWORK_BACKENDS == frozenset({"pytorch", "mlx", "numpy"})


def test_predicted_band_per_substrate_matches_slot_l_symposia() -> None:
    """Predicted ΔS bands match the 3 Slot L symposia §6 declarations."""
    assert PREDICTED_BAND_PER_SUBSTRATE["pr110"] == (-0.0025, 0.0015)
    # PR101 EXPANDED upper bound per Assumption-Adversary binding revision
    # for HNeRV parity L3 grammar overhead.
    assert PREDICTED_BAND_PER_SUBSTRATE["pr101"] == (-0.0025, 0.0040)
    assert PREDICTED_BAND_PER_SUBSTRATE["dqs1"] == (-0.0025, 0.0015)


def test_provenance_model_id_canonical() -> None:
    """Provenance model_id is the canonical helper self-reference."""
    assert PROVENANCE_MODEL_ID == (
        "tac.substrates._shared.synthesize_frame_emission_atick_redlich.v1"
    )


# ---------------------------------------------------------------------------
# SynthesizeFrameEmissionConfig (frozen dataclass invariants)
# ---------------------------------------------------------------------------

def test_config_default_construction_for_each_slot_l_cell() -> None:
    """Config constructs cleanly for all 3 Slot L cells with defaults."""
    for cell in ("pr110", "pr101", "dqs1"):
        cfg = SynthesizeFrameEmissionConfig(substrate_id=cell)
        assert cfg.substrate_id == cell
        assert isinstance(cfg.weights, AtickRedlichWeights)
        assert cfg.per_pair_metadata_budget_bytes == 80
        assert cfg.framework_agnostic_backend == "pytorch"
        assert cfg.apply_eval_roundtrip is True
        assert cfg.random_seed == 42


def test_config_predicted_band_per_substrate() -> None:
    """``predicted_band()`` returns the canonical Slot L symposium band."""
    cfg_pr110 = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    assert cfg_pr110.predicted_band() == (-0.0025, 0.0015)
    cfg_pr101 = SynthesizeFrameEmissionConfig(substrate_id="pr101")
    assert cfg_pr101.predicted_band() == (-0.0025, 0.0040)
    cfg_protocol = SynthesizeFrameEmissionConfig(substrate_id="protocol_test")
    assert cfg_protocol.predicted_band() is None


def test_config_rejects_unrecognized_substrate_id() -> None:
    """Future 4th cells must be added to ``RECOGNIZED_SUBSTRATE_IDS`` first."""
    with pytest.raises(ValueError, match="not in RECOGNIZED_SUBSTRATE_IDS"):
        SynthesizeFrameEmissionConfig(substrate_id="pr106")


def test_config_rejects_empty_substrate_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SynthesizeFrameEmissionConfig(substrate_id="")


def test_config_rejects_non_str_substrate_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SynthesizeFrameEmissionConfig(substrate_id=42)  # type: ignore[arg-type]


def test_config_rejects_non_atick_redlich_weights() -> None:
    with pytest.raises(ValueError, match="AtickRedlichWeights"):
        SynthesizeFrameEmissionConfig(
            substrate_id="pr110",
            weights={"beta_seg": 100.0},  # type: ignore[arg-type]
        )


def test_config_rejects_negative_metadata_budget() -> None:
    with pytest.raises(ValueError, match="per_pair_metadata_budget_bytes must be >= 0"):
        SynthesizeFrameEmissionConfig(
            substrate_id="pr110", per_pair_metadata_budget_bytes=-1
        )


def test_config_rejects_unrecognized_framework_backend() -> None:
    with pytest.raises(ValueError, match="framework_agnostic_backend"):
        SynthesizeFrameEmissionConfig(
            substrate_id="pr110", framework_agnostic_backend="jax"
        )


def test_config_rejects_apply_eval_roundtrip_false() -> None:
    """CLAUDE.md non-negotiable: eval_roundtrip=False is forbidden."""
    with pytest.raises(ValueError, match="eval_roundtrip"):
        SynthesizeFrameEmissionConfig(
            substrate_id="pr110", apply_eval_roundtrip=False
        )


def test_config_rejects_bool_metadata_budget() -> None:
    """``bool`` is a subclass of ``int``; reject explicitly."""
    with pytest.raises(ValueError, match="per_pair_metadata_budget_bytes must be int"):
        SynthesizeFrameEmissionConfig(
            substrate_id="pr110",
            per_pair_metadata_budget_bytes=True,  # type: ignore[arg-type]
        )


def test_config_frozen_dataclass_invariant() -> None:
    """Frozen dataclass — cannot mutate fields."""
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    with pytest.raises(Exception):
        cfg.substrate_id = "pr101"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AtickRedlichReceiver (per-substrate adapter wrapper)
# ---------------------------------------------------------------------------

def test_receiver_construction_succeeds() -> None:
    seg = _StandinSegScorer()
    pose = _StandinPoseScorer()
    receiver = AtickRedlichReceiver(
        substrate_id="pr110",
        seg_scorer=seg,
        pose_scorer=pose,
        archive_grammar_slot_injector=_toy_injector,
    )
    assert receiver.substrate_id == "pr110"
    assert receiver.seg_scorer is seg
    assert receiver.pose_scorer is pose
    assert receiver.archive_grammar_slot_injector is _toy_injector


def test_receiver_rejects_unrecognized_substrate_id() -> None:
    seg = _StandinSegScorer()
    pose = _StandinPoseScorer()
    with pytest.raises(ValueError, match="not in RECOGNIZED_SUBSTRATE_IDS"):
        AtickRedlichReceiver(
            substrate_id="pr106",
            seg_scorer=seg,
            pose_scorer=pose,
            archive_grammar_slot_injector=_toy_injector,
        )


def test_receiver_rejects_seg_scorer_missing_preprocess_input() -> None:
    pose = _StandinPoseScorer()

    class _NoPreprocessSeg:
        pass

    with pytest.raises(ValueError, match="seg_scorer must expose preprocess_input"):
        AtickRedlichReceiver(
            substrate_id="pr110",
            seg_scorer=_NoPreprocessSeg(),
            pose_scorer=pose,
            archive_grammar_slot_injector=_toy_injector,
        )


def test_receiver_rejects_pose_scorer_missing_preprocess_input() -> None:
    seg = _StandinSegScorer()

    class _NoPreprocessPose:
        pass

    with pytest.raises(ValueError, match="pose_scorer must expose preprocess_input"):
        AtickRedlichReceiver(
            substrate_id="pr110",
            seg_scorer=seg,
            pose_scorer=_NoPreprocessPose(),
            archive_grammar_slot_injector=_toy_injector,
        )


def test_receiver_rejects_non_callable_injector() -> None:
    seg = _StandinSegScorer()
    pose = _StandinPoseScorer()
    with pytest.raises(ValueError, match="must be callable"):
        AtickRedlichReceiver(
            substrate_id="pr110",
            seg_scorer=seg,
            pose_scorer=pose,
            archive_grammar_slot_injector="not_callable",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# SubstrateGrammarAdapter Protocol (sister cargo-cult unwind A8)
# ---------------------------------------------------------------------------

def test_toy_injector_satisfies_protocol_shape() -> None:
    """Toy callable + canonical signature satisfies the Protocol return shape."""
    result = _toy_injector(b"ARCHIVE", b"META")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bytes)
    assert isinstance(result[1], int)
    assert result == (b"ARCHIVEMETA", 4)


def test_protocol_class_adapter_satisfies_runtime_checkable() -> None:
    """Class-based adapter satisfies the SubstrateGrammarAdapter Protocol."""

    class _AdapterImpl:
        substrate_id = "protocol_test"

        def inject_synthesize_frame_metadata(
            self, archive_bytes: bytes, synthetic_frame_metadata_bytes: bytes
        ) -> tuple[bytes, int]:
            return archive_bytes + synthetic_frame_metadata_bytes, len(
                synthetic_frame_metadata_bytes
            )

    adapter = _AdapterImpl()
    assert isinstance(adapter, SubstrateGrammarAdapter)


# ---------------------------------------------------------------------------
# build_atick_redlich_cooperative_receiver_for_substrate factory
# ---------------------------------------------------------------------------

def test_factory_returns_frozen_receiver() -> None:
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "dqs1",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    assert isinstance(receiver, AtickRedlichReceiver)
    assert receiver.substrate_id == "dqs1"
    # Frozen — cannot mutate.
    with pytest.raises(Exception):
        receiver.substrate_id = "pr110"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# End-to-end synthesize_frame_emission_per_pair
# ---------------------------------------------------------------------------

def test_emission_returns_per_pair_result_with_tier_a_markers() -> None:
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=11)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    assert isinstance(result, SynthesizeFrameEmissionPerPairResult)
    # Tier A canonical-routing markers per Catalog #341.
    assert result.canonical_routing_markers["predicted_delta_adjustment"] == 0.0
    assert result.canonical_routing_markers["promotable"] is False
    assert result.canonical_routing_markers["axis_tag"] == "[predicted]"
    assert result.canonical_routing_markers["score_claim"] is False
    assert (
        result.canonical_routing_markers["ready_for_exact_eval_dispatch"] is False
    )


def test_emission_frame_bytes_match_expected_size() -> None:
    """Synthetic frame bytes = B * H * W * 3 uint8 (canonical contest layout)."""
    h, w = 32, 48
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(batch=1, h=h, w=w, seed=22)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr101")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr101",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=5,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    expected = 1 * h * w * 3
    assert len(result.frame_0_bytes) == expected
    assert len(result.frame_1_bytes) == expected


def test_emission_metadata_bytes_match_budget() -> None:
    """Metadata bytes are bounded to ``per_pair_metadata_budget_bytes``."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=33)
    cfg = SynthesizeFrameEmissionConfig(
        substrate_id="pr110", per_pair_metadata_budget_bytes=128
    )
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    assert len(result.per_pair_metadata_bytes) == 128


def test_emission_metadata_header_format_canonical() -> None:
    """Metadata header is canonical ``<IIddd`` per design memo §6."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=44)
    cfg = SynthesizeFrameEmissionConfig(
        substrate_id="pr110",
        per_pair_metadata_budget_bytes=80,
        random_seed=12345,
    )
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=7,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    header_len = struct.calcsize("<IIddd")
    assert len(result.per_pair_metadata_bytes) >= header_len
    unpacked = struct.unpack(
        "<IIddd", result.per_pair_metadata_bytes[:header_len]
    )
    pair_idx, seed, seg_term, pose_term, pose_sqrt = unpacked
    assert pair_idx == 7
    assert seed == 12345
    assert seg_term == pytest.approx(
        float(result.atick_redlich_loss.seg_term.detach().cpu().item())
    )
    assert pose_term == pytest.approx(
        float(result.atick_redlich_loss.pose_term.detach().cpu().item())
    )
    assert pose_sqrt == pytest.approx(
        float(result.atick_redlich_loss.pose_sqrt.detach().cpu().item())
    )


def test_emission_axis_decomposition_carries_provenance() -> None:
    """AxisDecomposition includes canonical Provenance per Catalog #356/#323."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=55)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="dqs1")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "dqs1",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    decomp = result.predicted_axis_decomposition
    # AxisDecomposition.as_dict() canonical keys per Catalog #356.
    assert "predicted_d_seg_delta" in decomp
    assert "predicted_d_pose_delta" in decomp
    assert "predicted_archive_bytes_delta" in decomp
    assert decomp["axis_tag"] == "[predicted]"
    # Canonical Provenance per Catalog #323.
    assert "canonical_provenance" in decomp
    prov = decomp["canonical_provenance"]
    assert isinstance(prov, dict)
    # evidence_grade is the canonical ProvenanceEvidenceGrade.PREDICTED enum
    # value ("predicted"); the axis_tag "[predicted]" lives on
    # AxisDecomposition.axis_tag directly (verified above).
    assert prov.get("evidence_grade") == "predicted"
    assert prov.get("promotion_eligible") is False
    # canonical_helper_invocation cites the Provenance builder (canonical
    # per builders.py:307); model_id (encoded in source_path) cites THIS
    # SHARED helper per design memo §6 PROVENANCE_MODEL_ID constant.
    assert prov.get("canonical_helper_invocation") == (
        "tac.provenance.builders.build_provenance_for_predicted"
    )
    source_path = prov.get("source_path") or ""
    assert PROVENANCE_MODEL_ID in source_path
    assert "synthesize_frame_emission_atick_redlich" in source_path


def test_emission_axis_decomposition_seg_delta_sign() -> None:
    """``predicted_d_seg_delta`` is non-positive (negative = improvement)."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=66)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    assert result.predicted_axis_decomposition["predicted_d_seg_delta"] <= 0.0
    assert result.predicted_axis_decomposition["predicted_d_pose_delta"] <= 0.0
    # archive bytes delta is the metadata bytes count (positive; archive grows).
    assert (
        result.predicted_axis_decomposition["predicted_archive_bytes_delta"]
        == len(result.per_pair_metadata_bytes)
    )


def test_emission_atick_redlich_loss_is_canonical_output() -> None:
    """Returned loss IS the canonical CooperativeReceiverOutput dataclass."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=77)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    loss = result.atick_redlich_loss
    assert isinstance(loss, CooperativeReceiverOutput)
    assert loss.cooperative_loss.requires_grad  # gradient flows
    assert loss.seg_term.numel() == 1
    assert loss.pose_term.numel() == 1


def test_emission_gradient_flows_into_rgb_inputs() -> None:
    """Atick-Redlich loss gradient backprops into the predicted RGB inputs."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=88)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    result.atick_redlich_loss.cooperative_loss.backward()
    assert rgb_0.grad is not None
    assert rgb_1.grad is not None
    assert torch.isfinite(rgb_0.grad).all()
    assert torch.isfinite(rgb_1.grad).all()


def test_emission_rejects_substrate_id_mismatch() -> None:
    """receiver.substrate_id != config.substrate_id is refused."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=99)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr101",  # mismatched
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    with pytest.raises(ValueError, match="substrate_id="):
        synthesize_frame_emission_per_pair(
            per_pair_index=0,
            rgb_0=rgb_0,
            rgb_1=rgb_1,
            gt_rgb_0=gt_0,
            gt_rgb_1=gt_1,
            cooperative_receiver=receiver,
            config=cfg,
        )


# ---------------------------------------------------------------------------
# Byte-stability (Catalog #266 + Dim 7 deterministic reproducibility)
# ---------------------------------------------------------------------------

def test_byte_stability_same_inputs_same_output() -> None:
    """Same inputs + config + receiver yields byte-identical synthetic frames."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=111)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    stable = verify_synthesize_frame_emission_byte_stability(
        result,
        rerun_config=cfg,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
    )
    assert stable is True


def test_byte_stability_different_seed_yields_different_metadata() -> None:
    """Different random_seed yields different metadata padding (not header)."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=222)
    cfg_a = SynthesizeFrameEmissionConfig(
        substrate_id="pr110", random_seed=1, per_pair_metadata_budget_bytes=128
    )
    cfg_b = SynthesizeFrameEmissionConfig(
        substrate_id="pr110", random_seed=2, per_pair_metadata_budget_bytes=128
    )
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result_a = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg_a,
    )
    result_b = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg_b,
    )
    # Different seeds → header seed field differs, so full metadata differs.
    assert result_a.per_pair_metadata_bytes != result_b.per_pair_metadata_bytes
    # Frames are deterministic from RGB inputs alone (same here).
    assert result_a.frame_0_bytes == result_b.frame_0_bytes
    assert result_a.frame_1_bytes == result_b.frame_1_bytes


def test_byte_stability_different_pair_index_yields_different_metadata() -> None:
    """Different per_pair_index → different header → different metadata bytes."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=333)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result_0 = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    result_1 = synthesize_frame_emission_per_pair(
        per_pair_index=1,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    assert result_0.per_pair_metadata_bytes != result_1.per_pair_metadata_bytes


# ---------------------------------------------------------------------------
# SynthesizeFrameEmissionPerPairResult invariants
# ---------------------------------------------------------------------------

def test_result_rejects_negative_pair_index() -> None:
    """Result dataclass refuses negative per_pair_index."""
    seg = _StandinSegScorer()
    pose = _StandinPoseScorer()
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=444)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=seg,
        pose_scorer=pose,
        archive_grammar_slot_injector=_toy_injector,
    )
    # Get a real CooperativeReceiverOutput first.
    real_result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    with pytest.raises(ValueError, match="per_pair_index"):
        SynthesizeFrameEmissionPerPairResult(
            per_pair_index=-1,
            frame_0_bytes=real_result.frame_0_bytes,
            frame_1_bytes=real_result.frame_1_bytes,
            per_pair_metadata_bytes=real_result.per_pair_metadata_bytes,
            atick_redlich_loss=real_result.atick_redlich_loss,
            predicted_axis_decomposition=real_result.predicted_axis_decomposition,
            canonical_routing_markers=real_result.canonical_routing_markers,
        )


def test_result_rejects_non_zero_predicted_delta_adjustment() -> None:
    """Catalog #341 Tier A invariant: predicted_delta_adjustment MUST be 0.0."""
    seg = _StandinSegScorer()
    pose = _StandinPoseScorer()
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=555)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=seg,
        pose_scorer=pose,
        archive_grammar_slot_injector=_toy_injector,
    )
    real_result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    bad_markers = dict(real_result.canonical_routing_markers)
    bad_markers["predicted_delta_adjustment"] = 0.05  # forbidden non-zero
    with pytest.raises(ValueError, match="predicted_delta_adjustment"):
        SynthesizeFrameEmissionPerPairResult(
            per_pair_index=0,
            frame_0_bytes=real_result.frame_0_bytes,
            frame_1_bytes=real_result.frame_1_bytes,
            per_pair_metadata_bytes=real_result.per_pair_metadata_bytes,
            atick_redlich_loss=real_result.atick_redlich_loss,
            predicted_axis_decomposition=real_result.predicted_axis_decomposition,
            canonical_routing_markers=bad_markers,
        )


def test_result_rejects_promotable_true() -> None:
    """Catalog #341 Tier A invariant: promotable MUST be False."""
    seg = _StandinSegScorer()
    pose = _StandinPoseScorer()
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=666)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=seg,
        pose_scorer=pose,
        archive_grammar_slot_injector=_toy_injector,
    )
    real_result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    bad_markers = dict(real_result.canonical_routing_markers)
    bad_markers["promotable"] = True  # forbidden
    with pytest.raises(ValueError, match="promotable"):
        SynthesizeFrameEmissionPerPairResult(
            per_pair_index=0,
            frame_0_bytes=real_result.frame_0_bytes,
            frame_1_bytes=real_result.frame_1_bytes,
            per_pair_metadata_bytes=real_result.per_pair_metadata_bytes,
            atick_redlich_loss=real_result.atick_redlich_loss,
            predicted_axis_decomposition=real_result.predicted_axis_decomposition,
            canonical_routing_markers=bad_markers,
        )


def test_result_rejects_non_predicted_axis_tag() -> None:
    """Catalog #341 Tier A invariant: axis_tag MUST be '[predicted]'."""
    seg = _StandinSegScorer()
    pose = _StandinPoseScorer()
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=777)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="pr110")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "pr110",
        seg_scorer=seg,
        pose_scorer=pose,
        archive_grammar_slot_injector=_toy_injector,
    )
    real_result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    bad_markers = dict(real_result.canonical_routing_markers)
    bad_markers["axis_tag"] = "[contest-CPU]"  # forbidden for scaffold
    with pytest.raises(ValueError, match="axis_tag"):
        SynthesizeFrameEmissionPerPairResult(
            per_pair_index=0,
            frame_0_bytes=real_result.frame_0_bytes,
            frame_1_bytes=real_result.frame_1_bytes,
            per_pair_metadata_bytes=real_result.per_pair_metadata_bytes,
            atick_redlich_loss=real_result.atick_redlich_loss,
            predicted_axis_decomposition=real_result.predicted_axis_decomposition,
            canonical_routing_markers=bad_markers,
        )


# ---------------------------------------------------------------------------
# Sister 4th cell extension regression (cargo-cult unwind A8)
# ---------------------------------------------------------------------------

def test_protocol_test_substrate_id_admits_future_4th_cell_pattern() -> None:
    """Future 4th sister cells use the Protocol; protocol_test admits the pattern.

    Per cargo-cult unwind A8: the per-substrate adapter pattern must NOT
    require modifying the SHARED helper signature when a 4th sister cell
    lands (e.g. PR106 × SYNTHESIZE_FRAME). The ``protocol_test`` substrate
    ID is the canonical regression guard.
    """
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=888)
    cfg = SynthesizeFrameEmissionConfig(substrate_id="protocol_test")
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        "protocol_test",
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    # Helper succeeds without per-substrate adapter implementation modifying
    # the SHARED helper.
    assert isinstance(result, SynthesizeFrameEmissionPerPairResult)
    # protocol_test has no canonical predicted_band per Slot L symposia.
    assert cfg.predicted_band() is None


# ---------------------------------------------------------------------------
# Multi-substrate end-to-end smoke (PR110 + PR101 + DQS1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("substrate_id", ["pr110", "pr101", "dqs1"])
def test_emission_smoke_per_slot_l_cell(substrate_id: str) -> None:
    """Smoke regression: emission succeeds for each of the 3 Slot L cells."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair(seed=999)
    cfg = SynthesizeFrameEmissionConfig(substrate_id=substrate_id)
    receiver = build_atick_redlich_cooperative_receiver_for_substrate(
        substrate_id,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        archive_grammar_slot_injector=_toy_injector,
    )
    result = synthesize_frame_emission_per_pair(
        per_pair_index=0,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        cooperative_receiver=receiver,
        config=cfg,
    )
    assert result.per_pair_index == 0
    assert isinstance(result.atick_redlich_loss, CooperativeReceiverOutput)
    # Predicted band for this substrate matches canonical Slot L symposium.
    band = cfg.predicted_band()
    assert band == PREDICTED_BAND_PER_SUBSTRATE[substrate_id]
