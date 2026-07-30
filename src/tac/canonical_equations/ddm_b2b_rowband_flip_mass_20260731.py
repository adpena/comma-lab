"""ddm_b2b — QA84 row-band flip-mass foveation constant (canonical equation + measured anchor).

The QA84 variable-cell grammar (``tac.witness_dsl.qa84_rowband_grammar_20260731``) consumes a
LOAD-BEARING measured design constant: **72.1-72.7%% of flip-prone mass lives in render rows
160-240** (the op1 foveation gate PASSED at the pre-registered >=50%% criterion; QA74 typed the
flip mass from the frozen CPU-torch SegNet gt cache). This module custodies that number so the
grammar's band bounds resolve through the value-provenance ladder (MEASURED anchor), not a bare
constant (MAIN per-leg drift-detector EQUATIONS leg on the ddm_b2b QA84 landing).

Registration note: this module DEFINES the canonical equation + its measured anchor (touching the
canonical_equations leg). The locked-registry ``populate_*`` append + ``__init__`` export is the
named landing/operator follow-up (kept out of this build arm to avoid mutating the shared locked
registry while a sister burn holds the run slot). ``build_rowband_flip_mass_foveation_v1()`` is
importable + self-validating now; the QA84 Lever's ``constant_refs`` points at ``EQUATION_ID``.

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED. Advisory; score_claim=False; the anchor is
a measured spatial FACT (flip-mass fraction), never a score/pointer claim.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "rowband_flip_mass_foveation_band_v1"

#: The QA74-typed source custody (frozen CPU SegNet gt cache flip mass) surfaced via the census.
SOURCE_ARTIFACT = ".omx/research/ddm_gd1_generic_default_census_20260731.md"
GT_CACHE_ARTIFACT = "experiments/results/mlx_fleet_gt_cache/gt_n96.npz"

#: MEASURED: fraction of flip-prone mass inside render rows [160, 240] (op1 foveation gate).
FLIP_MASS_FRACTION_IN_BAND = 0.721
FOVEATION_GATE_CRITERION = 0.50
BAND_RENDER_ROWS = (160, 240)


def flip_band_render_rows() -> tuple[int, int]:
    """The measured maximum-flip-mass render-row band the QA84 foveation grammar centers on."""
    return BAND_RENDER_ROWS


def build_rowband_flip_mass_foveation_v1() -> CanonicalEquation:
    """Build the canonical row-band flip-mass foveation equation + its measured anchor."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=SOURCE_ARTIFACT,
        reactivation_criteria=(
            "re-type the flip mass per render-row band from the frozen CPU SegNet argmax on a "
            "fresh gt cache; if the >=50%% concentration band shifts, re-center the QA84 grammar "
            "band bounds and re-run the matched-bytes d_seg race"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="apple_macos_cpu_torch",
    )
    anchor = EmpiricalAnchor(
        anchor_id="rowband_flip_mass_op1_foveation_gate_qa74_20260731",
        measurement_utc="2026-07-31T00:00:00Z",
        inputs={
            "band_render_rows": list(BAND_RENDER_ROWS),
            "gate_criterion_min_fraction": FOVEATION_GATE_CRITERION,
            "source_gt_cache": GT_CACHE_ARTIFACT,
            "typing": "QA74 per-render-row flip-mass typing (op1 foveation gate)",
        },
        predicted_output={"foveation_band": list(BAND_RENDER_ROWS),
                          "gate_passes_if_fraction_ge": FOVEATION_GATE_CRITERION},
        empirical_output={"flip_mass_fraction_in_band": FLIP_MASS_FRACTION_IN_BAND,
                          "gate_passed": FLIP_MASS_FRACTION_IN_BAND >= FOVEATION_GATE_CRITERION,
                          "range_across_typings": [0.721, 0.727]},
        residual=0.0,
        source_artifact=SOURCE_ARTIFACT,
        measurement_method=(
            "op1 foveation gate on QA74-typed per-render-row flip mass from the frozen CPU-torch "
            "SegNet argmax (gt cache); flip-prone = bottom GT-margin decile pixels"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Row-band flip-mass foveation constant",
        one_line_summary=(
            "72.1-72.7%% of flip-prone mass lives in render rows 160-240 (op1 foveation gate "
            ">=50%% PASSED); the QA84 grammar makes that band FINE (D8) and the bulk COARSE (D16)."
        ),
        latex_form=r"\Pr[\mathrm{flip}\in\mathrm{rows}[160,240]]=0.721\ge 0.5",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_b2b_rowband_flip_mass_20260731:flip_band_render_rows"),
        domain_of_validity={
            "included": [
                "QA84 RowBandGrammar band-bound derivation on the tr1 render frame (384x512)",
                "op1 foveation-gate spatial concentration decisions",
            ],
            "excluded": [
                "claim that foveation improves d_seg (that is the burn-2 matched-bytes race)",
                "score, promotion, or pointer movement",
                "in-band azimuthal (column) sparsity (separate QA74 g4 custody)",
            ],
            "authority": "[macOS-CPU advisory]",
        },
        units_in={"band_render_rows": "render-frame row index (0..383)"},
        units_out={"flip_mass_fraction": "dimensionless fraction of flip-prone mass"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"foveation_band_concentration": 0.0},
        last_calibration_utc="2026-07-31T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.qa84_rowband_grammar_20260731.RowBandGrammar",
            "tac.witness_dsl.spec_tr1_renderer_20260728.lever_token_rowband",
        ),
        canonical_producers=("tools.ddm_b2b_segnet_field_pass",),
        provenance=provenance,
    )


def populate_rowband_flip_mass_foveation_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append the measured equation through the locked registry helper (the b2b owed export)."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_rowband_flip_mass_foveation_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "QA84 row-band foveation constant (b2b equations leg, MAIN consolidation export); "
            "[macOS-CPU advisory]; score_claim=false; consumer = RowBandGrammar band bounds"
        ),
    )
    return equation


__all__ = [
    "BAND_RENDER_ROWS",
    "EQUATION_ID",
    "FLIP_MASS_FRACTION_IN_BAND",
    "FOVEATION_GATE_CRITERION",
    "build_rowband_flip_mass_foveation_v1",
    "flip_band_render_rows",
    "populate_rowband_flip_mass_foveation_v1",
]
