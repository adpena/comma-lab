# SPDX-License-Identifier: MIT
"""Packet grammar registry for cooperative-receiver substrate work.

This module is the single small registry that keeps new first-principles
packet magics visible to xray, compiler manifests, and integration reports.
It intentionally carries no score authority and imports no scorer code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CooperativeReceiverPacketGrammar:
    """One byte-level packet grammar emitted by a cooperative-receiver lane."""

    magic: bytes
    xray_label: str
    substrate_class: str
    archive_version: str
    campaign_id: str
    source_module: str
    compiler_stage: str
    notes: str


COOPERATIVE_RECEIVER_PACKET_GRAMMARS: tuple[CooperativeReceiverPacketGrammar, ...] = (
    CooperativeReceiverPacketGrammar(
        magic=b"DFL1",
        xray_label="renderer_payload_dfl1_native_v1",
        substrate_class="renderer_payload_dfl1_native_packet",
        archive_version="renderer_payload_dfl1_fixed3_v1",
        campaign_id="codex_renderer_payload_dfl1_native_20260525",
        source_module="submissions.robust_current.unpack_renderer_payload",
        compiler_stage="renderer_mask_pose_native_deflate_pack",
        notes=(
            "Source-runtime-native renderer payload carrying renderer.bin, "
            "masks.mkv, and optimized_poses.pt as fixed-order raw ZIP deflate "
            "streams; parser smoke only until full-frame parity and exact eval"
        ),
    ),
    CooperativeReceiverPacketGrammar(
        magic=b"TT5L",
        xray_label="time_traveler_l5_v1",
        substrate_class="time_traveler_l5_packet",
        archive_version="time_traveler_l5_tt5l_v1",
        campaign_id="time_traveler_world_model_substrate",
        source_module="tac.substrates.time_traveler_l5_autonomy.archive",
        compiler_stage="representation_prediction_quantization_arithmetic_pack",
        notes="monolithic Time-Traveler L5 archive grammar",
    ),
    CooperativeReceiverPacketGrammar(
        magic=b"SBO1",
        xray_label="sabor_boundary_only_renderer_v1",
        substrate_class="sabor_boundary_only_renderer_packet",
        archive_version="sabor_boundary_only_sbo1_v1",
        campaign_id="sabor_boundary_only_renderer",
        source_module="tac.substrates.sabor_boundary_only_renderer.archive",
        compiler_stage="segmentation_boundary_prediction_pack",
        notes="stable-argmax boundary-only renderer grammar",
    ),
    CooperativeReceiverPacketGrammar(
        magic=b"S2SB",
        xray_label="s2sbs_byte_stuffing_archive_v1",
        substrate_class="s2sbs_byte_stuffing_packet",
        archive_version="s2sbs_s2s1_v1",
        campaign_id="s2sbs_hf_byte_stuffing",
        source_module="tac.substrates.s2sbs_byte_stuffing.archive",
        compiler_stage="side_information_hf_residual_pack",
        notes="stride-2-stem high-frequency byte-stuffing grammar",
    ),
    CooperativeReceiverPacketGrammar(
        magic=b"CMLR",
        xray_label="coord_mlp_residual_sidecar_v1",
        substrate_class="coord_mlp_residual_sidecar_packet",
        archive_version="coord_mlp_residual_sidecar_v1",
        campaign_id="h15_coord_mlp_residual_sidecar_pr103_on_pr106",
        source_module="tac.substrates.coord_mlp_residual_sidecar.archive",
        compiler_stage="external_residual_sidecar_pack",
        notes="Coord-MLP residual basis sidecar grammar",
    ),
    CooperativeReceiverPacketGrammar(
        magic=b"DPW1",
        xray_label="driving_prior_world_model_v1",
        substrate_class="driving_prior_world_model_packet",
        archive_version="driving_prior_world_model_dpw1_v1",
        campaign_id="driving_prior_world_model_substrate",
        source_module="tac.substrates.driving_prior_world_model.archive",
        compiler_stage="driving_prior_prediction_residual_pack",
        notes="deterministic 2032 driving-prior world-model scaffold grammar",
    ),
    # ── A1 + sidecar composition grammars (solver-stack wire-in sweep 2026-05-13) ──
    # These are sidecar magic bytes that appear INSIDE an A1-host packet (the host
    # carries A1's existing decoder; the sidecar adds residual or pose detail).
    # xray classifies the sidecar magic as a distinct substrate class so the
    # composition lane is visible in autopilot dispatch ranking.
    CooperativeReceiverPacketGrammar(
        magic=b"LPA1",
        xray_label="a1_plus_lapose_sidecar_v1",
        substrate_class="a1_plus_lapose_composition_packet",
        archive_version="a1_plus_lapose_lpa1_v1",
        campaign_id="a1_plus_lapose_composition",
        source_module="tac.substrates.a1_plus_lapose.archive",
        compiler_stage="pose_axis_foveal_rgb_residual_pack",
        notes="A1 + LAPose foveal RGB residual sidecar; pose-axis target at PR106 2.71x marginal regime",
    ),
    CooperativeReceiverPacketGrammar(
        magic=b"WAV1",
        xray_label="a1_plus_wavelet_residual_sidecar_v1",
        substrate_class="a1_plus_wavelet_residual_composition_packet",
        archive_version="a1_plus_wavelet_residual_wav1_v1",
        campaign_id="a1_plus_wavelet_residual_retarget",
        source_module="tac.substrates.a1_plus_wavelet_residual.archive",
        compiler_stage="seg_axis_db4_idwt_detail_band_pack",
        notes="A1 + DB4 IDWT single-level detail-band wavelet residual sidecar; seg-axis edge sharpening",
    ),
    CooperativeReceiverPacketGrammar(
        magic=b"FGS1",
        xray_label="frame0_grain_selector_sidecar_v1",
        substrate_class="frame0_postdecode_selector_packet",
        archive_version="frame0_grain_selector_fgs1_v1",
        campaign_id="hdm8_frame0_postdecode_selector",
        source_module="submissions.hdm8_film_grain_sidecar.inflate",
        compiler_stage="postdecode_scorer_aware_selector_pack",
        notes=(
            "Archive-charged deterministic first-frame postdecode selector; "
            "stacks after any RGB decoder and must remain scorer-free at inflate time"
        ),
    ),
    CooperativeReceiverPacketGrammar(
        magic=b"FES1",
        xray_label="frame_exploit_selector_sidecar_v1",
        substrate_class="frame_exploit_selector_sidecar_packet",
        archive_version="frame_exploit_selector_fes1_v1",
        campaign_id="frame_exploit_segnet_posenet_selector",
        source_module="submissions.frame_exploit_selector_sidecar.inflate",
        compiler_stage="postdecode_pairwise_frame0_selector_pack",
        notes=(
            "Archive-charged PR106 sidecar selector derived from full-600 "
            "SegNet/PoseNet proxy rows; exact-CUDA gated before any score claim"
        ),
    ),
)


def xray_magic_signatures() -> tuple[tuple[bytes, str], ...]:
    """Return ``(magic, label)`` rows consumed by static xray classification."""

    return tuple((row.magic, row.xray_label) for row in COOPERATIVE_RECEIVER_PACKET_GRAMMARS)


def xray_substrate_classes() -> tuple[str, ...]:
    """Return xray substrate-class identifiers for the packet grammars."""

    return tuple(row.substrate_class for row in COOPERATIVE_RECEIVER_PACKET_GRAMMARS)


def compiler_hook_rows() -> list[dict[str, object]]:
    """Return JSON-safe compiler/grammar hook rows for integration manifests."""

    return [
        {
            "magic_hex": row.magic.hex(),
            "magic_ascii": row.magic.decode("ascii"),
            "xray_label": row.xray_label,
            "substrate_class": row.substrate_class,
            "archive_version": row.archive_version,
            "campaign_id": row.campaign_id,
            "source_module": row.source_module,
            "compiler_stage": row.compiler_stage,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "notes": row.notes,
        }
        for row in COOPERATIVE_RECEIVER_PACKET_GRAMMARS
    ]


def _validate_unique_four_byte_magics() -> None:
    seen: dict[bytes, str] = {}
    for row in COOPERATIVE_RECEIVER_PACKET_GRAMMARS:
        if len(row.magic) != 4:
            raise ImportError(f"{row.substrate_class}: magic must be exactly 4 bytes")
        if row.magic in seen:
            raise ImportError(
                f"duplicate cooperative-receiver magic {row.magic!r}: "
                f"{seen[row.magic]} and {row.substrate_class}"
            )
        seen[row.magic] = row.substrate_class


_validate_unique_four_byte_magics()


__all__ = [
    "COOPERATIVE_RECEIVER_PACKET_GRAMMARS",
    "CooperativeReceiverPacketGrammar",
    "compiler_hook_rows",
    "xray_magic_signatures",
    "xray_substrate_classes",
]
