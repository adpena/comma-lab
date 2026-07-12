# SPDX-License-Identifier: MIT
"""Measured lossless component-delta rate law for task #452.

The connected-component transform is a clean-room integer-stream codec.  The
defect, tube-algebra, twisted-module, and finite-gauging language is explicitly
an analogy to Benjamin, Lam, and Luo (2026), *Chiral Tube Algebras I:
Topological Defect Lines, Twisted Modules, and Finite Gauging*,
arXiv:2607.07786, Sections 1.1 and 1.3.  No CFT theorem is imported as a
compression theorem.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "defect_network_component_delta_rate_v1"
_UTC = "2026-07-12T23:41:38Z"
_MEMO = ".omx/research/codex_findings_defect_network_tube_rate_code_20260712_codex.md"
_RECEIPT = (
    "experiments/results/defect_network_tube_rate_code_20260712T225958Z/"
    "measurement_receipt.json"
)
_AXIS = (
    "[macOS-CPU advisory . deterministic numpy standalone-section rate probe . NON-PROMOTABLE]"
)

INCUMBENT_BYTES = 1_010_237
CANDIDATE_BYTES = 1_003_855
BYTES_SAVED = INCUMBENT_BYTES - CANDIDATE_BYTES
INCUMBENT_RESIDUAL_STREAM_BYTES = 993_897
COMPONENT_STREAM_BYTES = 996_246
COMPONENT_STREAM_DELTA_BYTES = COMPONENT_STREAM_BYTES - INCUMBENT_RESIDUAL_STREAM_BYTES
RATE_TERM_DELTA_IF_RECEIVER_CLOSED = 25.0 * (CANDIDATE_BYTES - INCUMBENT_BYTES) / 37_545_489


def build_defect_network_component_delta_rate_v1() -> CanonicalEquation:
    """Build the exact-byte, receiver-conditional task #452 rate law."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=(
            "re-measure after a receiver-consumed phase-carrier A/B or on a new exact phase section; "
            "do not infer d_seg or d_pose from this rate-only anchor"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="apple_macos_cpu_numpy_reference",
        captured_at_utc=_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="defect_network_gt_n600_component_delta_20260712",
        measurement_utc=_UTC,
        inputs={
            "n_pairs": 600,
            "phase_residual_count": 1_287_364,
            "connected_component_count": 151_175,
            "connectivity": 8,
            "class_roles": {"road": 0, "lane": 1, "undriv": 2},
            "gt_cache_sha256": (
                "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
            ),
        },
        predicted_output={
            "gate": (
                "candidate_bytes < incumbent_bytes with authenticated shared geometry and "
                "bit-identical residual and phase decode"
            ),
        },
        empirical_output={
            "incumbent_bytes": INCUMBENT_BYTES,
            "candidate_bytes": CANDIDATE_BYTES,
            "bytes_saved": BYTES_SAVED,
            "incumbent_residual_stream_bytes": INCUMBENT_RESIDUAL_STREAM_BYTES,
            "component_stream_bytes": COMPONENT_STREAM_BYTES,
            "component_stream_delta_bytes": COMPONENT_STREAM_DELTA_BYTES,
            "rate_term_delta_if_receiver_closed": RATE_TERM_DELTA_IF_RECEIVER_CLOSED,
            "residual_roundtrip_bit_identical": True,
            "phase_field_roundtrip_bit_identical": True,
            "rate_code_subverdict": "GO",
            "rate_code_scope": "standalone section under shared GT-cache geometry",
            "defect_mechanism_subverdict": "NO-GO_HEADER_DEDUPLICATION_CONFOUND",
            "overall_verdict": "NEEDS-MORE_RECEIVER_GEOMETRY_AND_CONSUMPTION_UNMEASURED",
        },
        residual=0.0,
        source_artifact=_RECEIPT,
        measurement_method=(
            "lossless PHAS1 decode; GT-cache-derived 8-connected ground components shared "
            "out-of-band by both codecs; one first "
            "residual plus raster-order component deltas; deterministic stream-scheme selection; "
            "full residual and phase-field equality checks"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=0.0,
        noise_floor_provenance="deterministic exact serialized-byte and array equality comparison",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Lossless defect-component section-rate attribution law",
        one_line_summary=(
            "On gt_n600, the exact standalone candidate section is 6,382 bytes smaller only because its "
            "header removes redundant counts; its component streams are 2,349 bytes larger."
        ),
        latex_form=(
            r"r_C=(r_{C,0},\Delta r_{C,1},\ldots,\Delta r_{C,|C|-1}),\quad "
            r"B_{\rm save}=B_{\rm PHAS1}-B_{\rm DTUB1}=6382\ {\rm bytes},\quad "
            r"B_{\rm component\ streams}-B_{\rm residual}=2349\ {\rm bytes},\quad "
            r"\Delta S_{\rm rate}=25(B_{\rm DTUB1}-B_{\rm PHAS1})/37545489"
        ),
        python_callable_module_path=(
            "tac.boundary_math.defect_network_rate_code:encode_defect_tube_recode"
        ),
        domain_of_validity={
            "scope_level": "instance/formulation",
            "included": [
                "gt_n600 standalone PHAS1 section",
                "shared out-of-band GT-cache-derived 8-connected road/lane/undriv active components",
                "lossless residual and decoded phase-field equality",
                "separate stream-byte attribution from container-header bytes",
            ],
            "excluded": [
                "receiver-consumed through-R phase-effect A/B",
                "witness-receiver derivability of the GT-cache geometry",
                "d_seg or d_pose equality claim",
                "contest score, archive promotion, or pointer movement",
                "tube-algebra or finite-gauging theorem used as compression authority",
            ],
            "authority": _AXIS,
        },
        units_in={"residual": "quantized phase-grid units", "section_bytes": "standalone bytes"},
        units_out={"bytes_saved": "standalone section bytes", "rate_term_delta": "conditional score units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"exact_decode_mismatch_count": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tools.probe_defect_network_rate_code",),
        canonical_producers=("tac.boundary_math.defect_network_rate_code",),
        provenance=provenance,
    )


def populate_defect_network_component_delta_rate_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append the measured equation through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_defect_network_component_delta_rate_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "task452; rate-code GO; overall NEEDS-MORE because PHAS1 is not receiver-consumed; "
            "research_only; score_claim=false"
        ),
    )
    return equation


__all__ = [
    "BYTES_SAVED",
    "CANDIDATE_BYTES",
    "COMPONENT_STREAM_BYTES",
    "COMPONENT_STREAM_DELTA_BYTES",
    "EQUATION_ID",
    "INCUMBENT_BYTES",
    "INCUMBENT_RESIDUAL_STREAM_BYTES",
    "RATE_TERM_DELTA_IF_RECEIVER_CLOSED",
    "build_defect_network_component_delta_rate_v1",
    "populate_defect_network_component_delta_rate_v1",
]
