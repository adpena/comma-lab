# SPDX-License-Identifier: MIT
"""Canonical law for RG1 stream composition and bounded polygon projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.optimization.ddm_rg1_receiver_grammar import project_polygon_center
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_rg1_bounded_receiver_composition_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / (
    ".omx/research/ddm_rg1_receiver_grammar_extension_20260724T080402Z/"
    "ddm_ms6_receiver_support_measurement_receipt.json"
)


def rg1_bounded_center_projection(
    *,
    center: int,
    relative_coordinates: Sequence[int],
    extent: int,
) -> int:
    """Evaluate the RG1 integer projection used by the receiver probe."""

    return project_polygon_center(center, relative_coordinates, extent)


def _evaluate(inputs: Mapping[str, Any]) -> int:
    if set(inputs) != {"center", "relative_coordinates", "extent"}:
        raise ValueError("inputs must contain center, relative_coordinates, and extent")
    relative = inputs["relative_coordinates"]
    if not isinstance(relative, Sequence) or isinstance(relative, (str, bytes)):
        raise ValueError("relative_coordinates must be a sequence")
    return rg1_bounded_center_projection(
        center=inputs["center"],
        relative_coordinates=relative,
        extent=inputs["extent"],
    )


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_rg1_bounded_receiver_composition_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the scoped RG1 law from the assignment-bound measurement receipt."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Rebuild after any base, grammar, scorer, PF2, or assignment-table "
            "SHA changes; never infer contest authority from this advisory axis."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_threads4_batch32",
        captured_at_utc="2026-07-24T09:00:00Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM RG1 bounded receiver composition",
        one_line_summary=(
            "Compose counted Lane productions before typed post-solve corrections "
            "and project polygon centers onto their exact in-grid interval."
        ),
        latex_form=(
            r"\Theta_{\mathrm{RG1}}=C_{\mathrm{post}}\circ P_{\mathrm{Lane}}"
            r"(\Theta_0),\quad P_0=C_0=I,\quad "
            r"c^\star=\Pi_{[-\min r,\;N-1-\max r]}(c)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_rg1_bounded_receiver_composition_20260724:"
            "rg1_bounded_center_projection"
        ),
        domain_of_validity={
            "base": "SHA-bound V19C nested carrier",
            "composition_order": [
                "sealed V13/V19C base",
                "counted Lane program production",
                "typed post-solve correction",
                "inherited raster and exact R",
            ],
            "inactive_identity": "P_0=C_0=I; compiler returns exact base bytes",
            "projection_interval": "[-min(relative), extent-1-max(relative)]",
            "stream_types": ["SKELETON", "CONNECTION", "FIBER", "GAUGE", "RESIDUAL"],
            "research_only": True,
            "score_claim": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "verdict_scope": "INSTANCE_EXTENDED_GRAMMAR_RG1",
            "excluded": [
                "contest score or frontier movement",
                "silent substitution for the original infeasible coordinate",
                "MS4 eligibility without complete exact G3 pair-bucket coverage",
            ],
        },
        units_in={
            "center": "integer_raster_coordinate",
            "relative_coordinates": "integer_polygon_offsets",
            "extent": "integer_scorer_extent",
        },
        units_out={"projected_center": "integer_raster_coordinate"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T09:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_rg1_receiver_grammar",
            "tools.measure_ddm_ms6_receiver_support",
            "tools.summarize_ddm_ms6_receiver_support",
        ),
        canonical_producers=(
            "tac.optimization.ddm_rg1_receiver_grammar",
            "tools.measure_ddm_ms6_receiver_support",
        ),
        provenance=provenance,
    )


def populate_ddm_rg1_bounded_receiver_composition_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append RG1 through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_rg1_bounded_receiver_composition_v1(
        source_receipt=source_receipt,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "RG1 ordered typed-stream composition and bounded polygon projection; "
            "advisory measurement only; score_claim=false; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_ddm_rg1_bounded_receiver_composition_v1",
    "populate_ddm_rg1_bounded_receiver_composition_v1",
    "rg1_bounded_center_projection",
]
