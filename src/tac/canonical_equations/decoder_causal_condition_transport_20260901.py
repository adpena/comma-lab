# SPDX-License-Identifier: MIT
"""Decoder-causal conditioning transport law for lossless coded schedules.

This is an operational-domain extension of
``wyner_ziv_decoder_side_information_conditional_entropy_savings_v1``.  The
parent law prices decoder side information.  This extension states the causal
admission condition that must hold *before* a coder may treat conditioning as
free: the decoder must be able to reproduce the exact equivalence class that
selects the integer CDF, alphabet, group membership, or parse action at the
moment that choice is consumed.

The source anchors are DCC1's bounded receiver census, QX3's exact 510,404-byte
bridge for an encoder-only context, and GMF1's 3/3 source closure for SFP1's
encoder-only schedule labels.  The law is scorer-free and makes no score claim.
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "decoder_causal_condition_transport_v1"
PARENT_EQUATION_ID = (
    "wyner_ziv_decoder_side_information_conditional_entropy_savings_v1"
)
SOURCE_MEMO = (
    ".omx/research/ddm_dcc1_decoder_causal_conditioning_verdict_20260901.md"
)


def receiver_causal_context_is_free(
    *,
    exact_equivalence_class_reproducible: bool,
    available_before_consumption: bool,
    side_message_bytes: int = 0,
) -> bool:
    """Return whether a declared conditioning class is genuinely free.

    ``side_message_bytes`` is deliberately part of the decision.  A counted
    message can make a schedule receiver-closed, but it does not make the
    conditioning free.
    """

    if side_message_bytes < 0:
        raise ValueError("side_message_bytes must be non-negative")
    return bool(
        exact_equivalence_class_reproducible
        and available_before_consumption
        and side_message_bytes == 0
    )


def transport_floor_bytes(conditional_entropy_bits: float) -> int:
    """Optimistic byte floor for carrying a missing context equivalence class.

    This is the parent Wyner-Ziv conditional-entropy floor only.  Finite coder,
    model, grammar, header, and archive costs are additional.
    """

    value = float(conditional_entropy_bits)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("conditional_entropy_bits must be finite and non-negative")
    return math.ceil(value / 8.0)


def build_decoder_causal_condition_transport_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        SOURCE_MEMO,
        reactivation_criteria=(
            "append an anchor when a new receiver-causal schedule is measured, or "
            "when a purported byte win uses a context unavailable at its exact "
            "decoder consumption time"
        ),
        measurement_axis="[source proof + scorer-free exact receiver/rate receipts]",
        hardware_substrate="source_inspection_and_macos_cpu_exact_coder",
        captured_at_utc="2026-09-01T19:39:40Z",
    )
    qx3 = EmpiricalAnchor(
        anchor_id="qx3_encoder_only_c1_context_requires_exact_bridge_20260901",
        measurement_utc="2026-09-01T02:21:50Z",
        inputs={
            "coded_field_bytes": 117_964_800,
            "decoder_context_mismatch_sites": 1_669_798,
            "declared_unclosed_section_bytes": 22_661,
            "context_source": "encoder_only_C1_baseline",
        },
        predicted_output={
            "free_conditioning": False,
            "required_repair": "transmit_or_replace_context",
        },
        empirical_output={
            "free_conditioning": False,
            "exact_bridge_bytes": 510_404,
            "complete_archive_bytes": 624_296,
            "complete_archive_sha256": (
                "5be6693516348f2a25c87fcea65f205477f339d6090c64636ef1c4b98531901c"
            ),
        },
        residual=0.0,
        source_artifact=".omx/research/ddm_qx3_receiver_closure_20260831.md",
        measurement_method=(
            "full-n600 exact receiver mismatch census, real-coder bridge encode, "
            "complete archive construction, and parse-back"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    gmf1 = EmpiricalAnchor(
        anchor_id="gmf1_sfp1_three_encoder_only_schedule_contexts_20260901",
        measurement_utc="2026-09-01T19:19:11Z",
        inputs={
            "proposal_denominator": 3,
            "context_fields": [
                "source_class",
                "target_class",
                "boundary_distance",
                "position_cell",
            ],
            "stored_side_stream": False,
        },
        predicted_output={
            "receiver_causal_contexts": 1,
            "receiver_noncausal_contexts": 3,
            "fit_admissible": False,
        },
        empirical_output={
            "proposals_source_closed": 3,
            "proposal_denominator": 3,
            "receiver_causal_contexts": ["position_cell"],
            "receiver_noncausal_contexts": [
                "source_class",
                "target_class",
                "boundary_distance",
            ],
        },
        residual=0.0,
        source_artifact=(
            ".omx/research/ddm_gmf1_fitted_crossgroup_gm_verdict_20260901.md"
        ),
        measurement_method=(
            "source inspection of SFP1 field construction and receiver contract, "
            "cross-checked against the 3/3 JBP1 executable-key blocker"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Decoder-causal conditioning transport",
        one_line_summary=(
            "Conditioning is free only when the decoder reproduces the exact "
            "CDF/parse equivalence class before use; otherwise its transport is counted."
        ),
        latex_form=(
            r"H(E_i(C_i)\mid D_{<i},p_i)=0\ \mathrm{for\ free\ conditioning};\ "
            r"B_T\geq\lceil H(E(C)\mid D,p)/8\rceil"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.decoder_causal_condition_transport_20260901:"
            "transport_floor_bytes"
        ),
        domain_of_validity={
            "parent_equation_id": PARENT_EQUATION_ID,
            "extension_kind": "operational_domain_extension_not_new_gate",
            "included": [
                "lossless arithmetic/range/ANS schedules whose integer CDF, alphabet, "
                "group membership, or parse action depends on a context class",
                "decoder-native contexts derived from decoded prefix state and public position",
                "counted side messages delivered before their first use",
            ],
            "excluded": [
                "correlated proxies that do not select the exact same integer CDF class",
                "encoder-only scorer, target, pre-edit-field, or hidden schedule labels",
                "entropy estimates promoted to physical bytes without a real coder",
            ],
            "verdict_scope": "FORMULATION admission and optimistic transport floor",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "conditional_entropy_bits": "bits",
            "decoder_state": "causal prefix state",
            "position": "public deterministic coordinates",
        },
        units_out={
            "free_conditioning": "bool",
            "transport_floor_bytes": "bytes",
        },
        empirical_anchors=(qx3, gmf1),
        predicted_vs_empirical_residual={
            "qx3_receiver_closure_prediction": 0.0,
            "gmf1_source_closure_prediction": 0.0,
        },
        last_calibration_utc="2026-09-01T19:39:40Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "task #1374 SCMDL causal-state/model builder",
            "task #1182 fitted cross-group G/M schedule",
            "QX decoder-native event grammar successors",
            "archive parser and exact parse-back review",
        ),
        canonical_producers=(
            SOURCE_MEMO,
            ".omx/research/ddm_qx3_receiver_closure_20260831.md",
            ".omx/research/ddm_qx4_decodable_conditioning_reprice_20260901.md",
            ".omx/research/ddm_gmf1_fitted_crossgroup_gm_verdict_20260901.md",
        ),
        provenance=provenance,
    )


def populate_decoder_causal_condition_transport_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append the extension to the canonical registry under its existing lock."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_decoder_causal_condition_transport_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DCC1 Catalog #344 waiver retirement; post-#400 Catalog #299 "
            "operational-domain extension of the Wyner-Ziv decoder-side-information law"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "PARENT_EQUATION_ID",
    "SOURCE_MEMO",
    "build_decoder_causal_condition_transport_v1",
    "populate_decoder_causal_condition_transport_v1",
    "receiver_causal_context_is_free",
    "transport_floor_bytes",
]
