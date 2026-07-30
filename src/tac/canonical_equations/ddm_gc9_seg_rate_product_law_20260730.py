"""ddm_gc9 — the seg×rate PRODUCT law c = (100·d_seg)·rate (canonical equation + 3 measured anchors).

The gc9 convocation (memo ``.omx/research/ddm_gc9_from_here_convocation_20260730.md``, commit
``1f41de9c79``) reframed the two-endpoint "seg+rate ≈ 0.77 wall" as a SUM-reading error: the
measured campaign invariant is the product ``c = (100*d_seg) * (25*bytes/37_545_489)``.  Three
receiver-realized points custody it: burn-1 c=0.14764, QA24 c=0.08815 (SMEVR-priced), and the
knee-B truncation c=0.11640 (FEED-wr1gb) — burns move c DOWN (×0.597/burn measured over 2 burns);
post-hoc deletion moves c UP.  Bar arithmetic: 0.172141 needs c ≤ ~5.05e-3 (DERIVED under the
Contrarian-flagged HEURISTIC pose≈0.03 + hyperbola-slide assumptions — that derivation is
context, never an anchor).  Consumers: gc9 fork decision table, ax1 §10 composed-stack
arithmetic, burn-3 stop/continue telemetry (Δlog c per burn).

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED.  Advisory; score_claim=False; the anchors
are measured (d_seg, bytes) pairs from receiver-realized custody, never score/pointer claims.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_gc9_seg_rate_product_law_v1"

SOURCE_ARTIFACT = ".omx/research/ddm_gc9_from_here_convocation_20260730.md"
CONTEST_UNCOMPRESSED_BYTES = 37_545_489

MEASURED_SLOPE_PER_BURN = 0.597  # c ratio QA24/burn-1 (2 burn points; slope, not a guarantee)
BAR_C_HEURISTIC = 5.05e-3  # DERIVED (pose~0.03 + hyperbola-slide HEURISTIC per Contrarian dissent)


def product_c(d_seg: float, counted_bytes: int) -> float:
    """The product invariant c = (100*d_seg) * (25*counted_bytes/37,545,489)."""
    return (100.0 * float(d_seg)) * (25.0 * float(counted_bytes) / CONTEST_UNCOMPRESSED_BYTES)


def _anchor(anchor_id: str, d_seg: float, counted_bytes: int, c_recorded: float,
            method: str, source: str, provenance) -> EmpiricalAnchor:
    c = product_c(d_seg, counted_bytes)
    return EmpiricalAnchor(
        anchor_id=anchor_id,
        measurement_utc="2026-07-30T00:00:00Z",
        inputs={"d_seg": d_seg, "counted_bytes": counted_bytes},
        predicted_output={"c": c_recorded},
        empirical_output={"c": c},
        residual=abs(c - c_recorded),
        source_artifact=source,
        measurement_method=method,
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
    )


def build_seg_rate_product_law_v1() -> CanonicalEquation:
    """Build the product-law equation + its three receiver-realized measured anchors."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=SOURCE_ARTIFACT,
        reactivation_criteria=(
            "each new receiver-realized (d_seg, counted_bytes) endpoint appends an anchor; "
            "re-fit the per-burn slope on >=3 burn points; retire the hyperbola-slide bar "
            "heuristic the moment a measured pose term replaces the pose~0.03 assumption"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="apple_macos_cpu_torch",
    )
    anchors = (
        _anchor(
            "gc9_product_burn1_pfs1_d1_20260729", 0.00389011, 569_996, 0.14764,
            "burn-1 pfs1 D1 composed archive (the wr1 gate REF row: d_seg 0.00389011, 569,996 B)",
            ".omx/research/ddm_pb1_postburn_completion_20260729.md", provenance,
        ),
        _anchor(
            "gc9_product_qa24_smevr_20260730", 0.0052766, 250_898, 0.08815,
            "QA24 endpoint receiver-realized d_seg (archive sha e7640dee, deploy parity -1e-6) "
            "x SMEVR-priced token bytes (tr1_window_receipt)",
            "/Volumes/VertigoDataTier/pact/ddm_782_qa24_endpoint_20260730/realized_verdict/"
            "p1_receiver_realized_verdict.json", provenance,
        ),
        _anchor(
            "gc9_product_kneeB_gate_20260730", 0.01001419, 174_578, 0.11640,
            "wr1 Gate-B staged exact composed gate (600-sample evaluate path, FEED-wr1gb)",
            "/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_kneeB_realized_gate_receipt.json",
            provenance,
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="seg-rate product invariant c",
        one_line_summary=(
            "c=(100*d_seg)*(25*bytes/37545489): burns move c down (x0.597/burn measured), "
            "post-hoc deletion moves c up (knee-B 0.1164 > QA24 0.08815); bar needs c<=~5.05e-3 "
            "(heuristic derivation, labeled)."
        ),
        latex_form=r"c=(100\,d_{seg})\cdot\frac{25\,B}{37545489}",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_gc9_seg_rate_product_law_20260730:product_c"),
        domain_of_validity={
            "included": [
                "cross-burn campaign telemetry (delta log c per burn)",
                "gc9 fork decision table + ax1 composed-stack arithmetic",
                "parent-selection comparisons at receiver-realized custody",
            ],
            "excluded": [
                "score, promotion, or pointer movement",
                "the bar-c value as authority (pose~0.03 hyperbola-slide is a HEURISTIC, "
                "Contrarian dissent recorded in the gc9 memo)",
                "points without receiver-realized custody (proxy/training-loss d_seg)",
            ],
            "authority": "[macOS-CPU advisory]",
        },
        units_in={"d_seg": "argmax disagreement fraction", "counted_bytes": "archive bytes"},
        units_out={"c": "dimensionless product of the two S terms"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"max_anchor_residual": max(
            a.residual for a in anchors)},
        last_calibration_utc="2026-07-30T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "gc9 fork decision table (.omx/research/ddm_gc9_from_here_convocation_20260730.md)",
            "ax1 composed burn-3 stack arithmetic (ddm_ax1_all_axes_derivation_20260730.md)",
        ),
        canonical_producers=(
            "tools.pb1_receiver_realized_verdict",
            "experiments.stage_wr1_realized_gate",
        ),
        provenance=provenance,
    )


def populate_seg_rate_product_law_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append the product law through the locked registry helper (the gc9 op-routable #3 export)."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_seg_rate_product_law_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "gc9 product law (MAIN boundary export, op-routable #3); [macOS-CPU advisory]; "
            "score_claim=false; consumers = fork table + burn-3 delta-log-c telemetry"
        ),
    )
    return equation
