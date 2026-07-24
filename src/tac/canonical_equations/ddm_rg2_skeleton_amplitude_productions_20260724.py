# SPDX-License-Identifier: MIT
"""Canonical law for RG2 receiver-derived SKELETON amplitude coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_rg2_skeleton_amplitude_productions_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / (
    ".omx/research/ddm_rg2_skeleton_amplitude_productions_20260724T094305Z/"
    "ddm_ms6_receiver_support_measurement_receipt.json"
)


def select_skeleton_amplitude_row_band(
    support_mass_by_row: Sequence[int],
    *,
    band_height: int = 64,
) -> int:
    """Return the earliest maximum-mass receiver row band.

    The production compiler first derives a binary receiver support mask from
    the sealed base. This pure law consumes its per-row masses; it never reads
    scorer labels, PF2 membership, or ground-truth pixels.
    """

    if isinstance(band_height, bool) or band_height <= 0:
        raise ValueError("band_height must be a positive integer")
    if not support_mass_by_row or len(support_mass_by_row) % band_height:
        raise ValueError("support rows must form complete equal-height bands")
    masses = []
    for start in range(0, len(support_mass_by_row), band_height):
        band = support_mass_by_row[start : start + band_height]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in band):
            raise ValueError("support masses must be nonnegative integers")
        masses.append(sum(band))
    if not any(masses):
        raise ValueError("typed class pair has no receiver support")
    return max(range(len(masses)), key=masses.__getitem__)


def _evaluate(inputs: Mapping[str, Any]) -> int:
    if set(inputs) not in (
        {"support_mass_by_row"},
        {"support_mass_by_row", "band_height"},
    ):
        raise ValueError("inputs must contain support_mass_by_row and optional band_height")
    rows = inputs["support_mass_by_row"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("support_mass_by_row must be a sequence")
    return select_skeleton_amplitude_row_band(
        rows,
        band_height=inputs.get("band_height", 64),
    )


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_rg2_skeleton_amplitude_productions_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the instance-scoped RG2 amplitude law from its receipt."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Rebuild after any base, grammar, scorer, PF2, RG1 table, row-band, "
            "or assignment SHA changes; never infer contest authority from this "
            "advisory axis."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_threads4_batch32",
        captured_at_utc="2026-07-24T10:20:00Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM RG2 receiver-derived SKELETON amplitude production",
        one_line_summary=(
            "Address one signed SKELETON amplitude quantum at the earliest "
            "maximum-mass receiver-derived row band, after RG1 and before R."
        ),
        latex_form=(
            r"b^\star=\min\arg\max_b\sum_{r\in B_b,x}\chi_{p,a,b,s}(r,x),"
            r"\quad q\in\{-1,+1\},\quad "
            r"\Theta_{\mathrm{RG2}}=A_{p,a,b,s,t,b^\star,q}"
            r"\circ\mathcal{R}_{\mathrm{camera}}\circ C_{\mathrm{RG1}}"
            r"\circ P_{\mathrm{RG1}}(\Theta_0),"
            r"\quad A_{\varnothing}=I"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_rg2_skeleton_amplitude_productions_20260724:"
            "select_skeleton_amplitude_row_band"
        ),
        domain_of_validity={
            "base": "SHA-bound V19C nested carrier",
            "address": (
                "exact pair x ordered class pair x boundary-or-cell stratum x "
                "temporal class x receiver-derived 64-row band"
            ),
            "composition_order": [
                "sealed V13/V19C base",
                "RG1 Lane production",
                "RG1 typed correction",
                "inherited raster to camera surface",
                "RG2 receiver-derived SKELETON amplitude mask",
                "evaluator-owned exact R",
            ],
            "inactive_identity": "A_empty=I; compiler returns exact input bytes",
            "quantum": "exactly one signed integer quantum in {-1,+1}",
            "stream_type": "TypedStreamTag SKELETON/L3_raster",
            "address_authority": "sealed receiver masks only; no scorer/PF2/GT spatial input",
            "research_only": True,
            "score_claim": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "verdict_scope": "INSTANCE_EXTENDED_GRAMMAR_RG2",
            "excluded": [
                "contest score or frontier movement",
                "invented address when both typed class roles have zero support",
                "MS4 eligibility without complete exact G3 pair-bucket coverage",
                "RG3 iteration",
            ],
        },
        units_in={
            "support_mass_by_row": "nonnegative receiver-support cells per raster row",
            "band_height": "raster rows",
        },
        units_out={"row_band": "zero-based receiver row-band index"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T10:20:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_rg1_receiver_grammar",
            "tools.build_ddm_rg2_skeleton_amplitude_assignments",
            "tools.measure_ddm_ms6_receiver_support",
            "tools.summarize_ddm_ms6_receiver_support",
        ),
        canonical_producers=(
            "tac.optimization.ddm_rg1_receiver_grammar",
            "tools.measure_ddm_ms6_receiver_support",
        ),
        provenance=provenance,
    )


def populate_ddm_rg2_skeleton_amplitude_productions_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append RG2 through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_rg2_skeleton_amplitude_productions_v1(
        source_receipt=source_receipt,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "RG2 counted receiver-derived SKELETON amplitude law; "
            "advisory measurement only; score_claim=false; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_ddm_rg2_skeleton_amplitude_productions_v1",
    "populate_ddm_rg2_skeleton_amplitude_productions_v1",
    "select_skeleton_amplitude_row_band",
]
