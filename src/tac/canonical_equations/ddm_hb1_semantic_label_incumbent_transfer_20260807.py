# SPDX-License-Identifier: MIT
"""HB1 semantic-label byte incumbent and transfer-admissibility law."""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_hb1_semantic_label_incumbent_transfer_v1"
SOURCE_ARTIFACT = ".omx/research/ddm_hb1_20260806/RECEIPT.md"
CONTEST_UNCOMPRESSED_BYTES = 37_545_489


def semantic_label_transfer_status(
    *,
    incumbent_bytes: int,
    external_stream_bytes: int,
    external_model_bytes: int,
    trained_on_target_payload: bool,
    decode_equality_on_target_payload: bool,
) -> dict[str, bool | float | int | str]:
    """Classify whether an external semantic coder can beat the target incumbent."""

    if incumbent_bytes <= 0 or external_stream_bytes < 0 or external_model_bytes < 0:
        raise ValueError("byte counts must be positive for incumbent and non-negative for external")
    external_total = int(external_stream_bytes) + int(external_model_bytes)
    delta_bytes = external_total - int(incumbent_bytes)
    proof_complete = bool(trained_on_target_payload and decode_equality_on_target_payload)
    return {
        "external_total_bytes": external_total,
        "delta_bytes_if_transfer_were_valid": delta_bytes,
        "delta_s_rate_if_transfer_were_valid": 25.0 * delta_bytes / CONTEST_UNCOMPRESSED_BYTES,
        "transfer_admissible": proof_complete,
        "incumbent_stands": not proof_complete or delta_bytes >= 0,
        "reason": (
            "external_coder_trained_and_decode_equal_on_target_payload"
            if proof_complete
            else "external_anchor_not_trained_or_decode_equal_on_target_payload"
        ),
    }


def semantic_label_s_rate(bytes_count: int) -> float:
    """Contest rate contribution for a semantic-label payload byte count."""

    if bytes_count < 0:
        raise ValueError("bytes_count must be non-negative")
    return 25.0 * int(bytes_count) / CONTEST_UNCOMPRESSED_BYTES


def build_ddm_hb1_semantic_label_incumbent_transfer_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        SOURCE_ARTIFACT,
        reactivation_criteria=(
            "append an anchor when HPAC or SMEVR is trained on OUR tq1c/GT label payloads "
            "and exact decode equality is attached"
        ),
        measurement_axis="[byte-only scorer-free]",
        hardware_substrate="macos_arm64_no_cuda",
        captured_at_utc="2026-08-06T19:13:03Z",
    )
    tq1c_transfer = semantic_label_transfer_status(
        incumbent_bytes=142_001,
        external_stream_bytes=116_980,
        external_model_bytes=15_164,
        trained_on_target_payload=False,
        decode_equality_on_target_payload=False,
    )
    anchor = EmpiricalAnchor(
        anchor_id="hb1_pr130_hpac_external_anchor_not_transferable_to_our_labels_20260806",
        measurement_utc="2026-08-06T19:13:03Z",
        inputs={
            "tq1c_incumbent_bytes": 142_001,
            "gt_lstars_incumbent_bytes": 173_617,
            "external_pr130_stream_bytes": 116_980,
            "external_pr130_packed_model_bytes": 15_164,
            "trained_on_our_payload": False,
            "decode_equality_on_our_payload": False,
        },
        predicted_output={
            "tq1c_transfer": tq1c_transfer,
            "adopt_external_pr130_hpac": False,
            "incumbent": "PP1 KT temporal context-arith",
        },
        empirical_output={
            "tq1c_incumbent_s_rate": semantic_label_s_rate(142_001),
            "gt_lstars_incumbent_s_rate": semantic_label_s_rate(173_617),
            "pr130_external_total_bytes": 132_144,
            "pr130_external_delta_s_if_valid_for_tq1c": tq1c_transfer[
                "delta_s_rate_if_transfer_were_valid"
            ],
            "hpac_on_our_payload_measured": False,
            "smevr_on_full_label_map_measured": False,
        },
        residual=0.0,
        source_artifact=SOURCE_ARTIFACT,
        measurement_method=(
            "HB1 byte-only source inspection of TK1/PP1 incumbent rows and PR130 retained "
            "HPAC stream/model accounting; no scorer execution"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="HB1 semantic-label incumbent transfer admissibility",
        one_line_summary=(
            "PP1 KT remains the measured incumbent for OUR tq1c/GT label payloads; an "
            "external PR130 HPAC byte advantage is not transferable without target-payload "
            "training and exact decode equality."
        ),
        latex_form=r"S_{rate}=25B/37545489,\quad adopt \Rightarrow trained_{target}\wedge equal_{decode}",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_hb1_semantic_label_incumbent_transfer_20260807:"
            "semantic_label_transfer_status"
        ),
        domain_of_validity={
            "included": [
                "n600 5-class semantic label planes",
                "byte-only coder race rows with explicit payload identity and equality evidence",
            ],
            "excluded": [
                "SegNet/PoseNet score claims",
                "external HPAC payloads not trained on OUR label object",
                "SMEVR full-label-map adoption before implementation and equality proof",
            ],
            "verdict_scope": "FORMULATION: semantic-label byte race transfer boundary",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "incumbent_bytes": "bytes",
            "external_stream_bytes": "bytes",
            "external_model_bytes": "bytes",
            "trained_on_target_payload": "bool",
            "decode_equality_on_target_payload": "bool",
        },
        units_out={
            "transfer_admissible": "bool",
            "incumbent_stands": "bool",
            "delta_s_rate_if_transfer_were_valid": "contest S rate units",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"hb1_transfer_boundary_residual": 0.0},
        last_calibration_utc="2026-08-06T19:13:03Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "HB1 HPAC resume fire order",
            "semantic stream byte-race reranker",
            "PR130 intake transfer audit",
        ),
        canonical_producers=(
            ".omx/research/ddm_hb1_20260806/RECEIPT.md",
            ".omx/research/ddm_hb1_20260806/BYTE_RACE_TABLE.md",
            ".omx/research/ddm_tk1_20260806/semantic_stream_race.json",
        ),
        provenance=provenance,
    )


def populate_ddm_hb1_semantic_label_incumbent_transfer_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_ddm_hb1_semantic_label_incumbent_transfer_v1()
    register_canonical_equation(
        eq,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="ddm_cq1 registration: HB1 semantic-label incumbent transfer boundary",
    )
    return eq


__all__ = [
    "CONTEST_UNCOMPRESSED_BYTES",
    "EQUATION_ID",
    "SOURCE_ARTIFACT",
    "build_ddm_hb1_semantic_label_incumbent_transfer_v1",
    "populate_ddm_hb1_semantic_label_incumbent_transfer_v1",
    "semantic_label_s_rate",
    "semantic_label_transfer_status",
]
