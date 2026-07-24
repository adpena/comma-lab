# SPDX-License-Identifier: MIT
"""Canonical law for terminal RG3 receiver residual-family coordinates."""

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

EQUATION_ID = "ddm_rg3_residual_family_productions_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / (
    ".omx/research/ddm_rg3_residual_family_productions_20260724T110418Z/"
    "ddm_ms6_receiver_support_measurement_receipt.json"
)
CLASS_BIRTH = "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION"
FINER_EVENT = "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK"
FISHER_STRATUM = "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK"
FAMILIES = (CLASS_BIRTH, FINER_EVENT, FISHER_STRATUM)


def _validated_rows(values: Sequence[int | float], *, label: str) -> list[float]:
    if len(values) != 384:
        raise ValueError(f"{label} must contain exactly 384 raster-row masses")
    rows = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{label} must contain nonnegative finite numbers")
        number = float(value)
        if number == float("inf") or number != number:
            raise ValueError(f"{label} must contain nonnegative finite numbers")
        rows.append(number)
    return rows


def _earliest_max_band(values: Sequence[float], *, start: int, stop: int, width: int) -> int:
    masses = [
        sum(values[offset : offset + width])
        for offset in range(start, stop, width)
    ]
    if not any(masses):
        raise ValueError("selected receiver support has zero mass")
    return max(range(len(masses)), key=masses.__getitem__)


def select_rg3_residual_address(
    receiver_support_mass_by_row: Sequence[int | float],
    *,
    family: str,
    row_band: int | None = None,
    fisher_weighted_support_mass_by_row: Sequence[int | float] | None = None,
) -> dict[str, Any]:
    """Select the counted RG3 row/16-row-subband address and magnitude alphabet.

    ``fisher_weighted_support_mass_by_row`` is an offline assignment statistic
    computed as receiver support times ``0.5*sech(margin/2)^2``. The receiver
    consumes only the selected integer fine-band symbol; neither margins nor
    scorer state are shipped.
    """

    if family not in FAMILIES:
        raise ValueError(f"unknown RG3 residual family: {family!r}")
    support = _validated_rows(
        receiver_support_mass_by_row,
        label="receiver_support_mass_by_row",
    )
    if family == CLASS_BIRTH:
        if row_band is not None:
            raise ValueError("class-birth row_band must be receiver-derived")
        selected_row = _earliest_max_band(support, start=0, stop=384, width=64)
        fine_source = support
        magnitudes = (1,)
    else:
        if isinstance(row_band, bool) or not isinstance(row_band, int) or not 0 <= row_band < 6:
            raise ValueError("finer/Fisher row_band must be an integer in [0, 5]")
        selected_row = row_band
        magnitudes = (1, 2)
        if family == FISHER_STRATUM:
            if fisher_weighted_support_mass_by_row is None:
                raise ValueError("Fisher family requires weighted receiver-support mass")
            fine_source = _validated_rows(
                fisher_weighted_support_mass_by_row,
                label="fisher_weighted_support_mass_by_row",
            )
        else:
            if fisher_weighted_support_mass_by_row is not None:
                raise ValueError("Fisher weights are valid only for the Fisher family")
            fine_source = support
    start = selected_row * 64
    fine_band = _earliest_max_band(
        fine_source,
        start=start,
        stop=start + 64,
        width=16,
    )
    return {
        "row_band": selected_row,
        "fine_band": fine_band,
        "signed_magnitude_alphabet": magnitudes,
        "fisher_margin_field_shipped": False,
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "receiver_support_mass_by_row",
        "family",
        "row_band",
        "fisher_weighted_support_mass_by_row",
    }
    if not set(inputs) <= allowed or not {
        "receiver_support_mass_by_row",
        "family",
    } <= set(inputs):
        raise ValueError("RG3 inputs differ from the canonical callable contract")
    return select_rg3_residual_address(
        inputs["receiver_support_mass_by_row"],
        family=inputs["family"],
        row_band=inputs.get("row_band"),
        fisher_weighted_support_mass_by_row=inputs.get(
            "fisher_weighted_support_mass_by_row"
        ),
    )


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_rg3_residual_family_productions_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the instance-scoped terminal RG3 residual-family law."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Rebuild after any base, grammar, scorer, PF2, RG2 table, margin "
            "tensor, residual assignment, fine-band, or checkpoint SHA changes."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_threads4_batch32",
        captured_at_utc="2026-07-24T11:30:00Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM RG3 terminal receiver residual-family production",
        one_line_summary=(
            "Choose a receiver-derived 16-row residual address, Fisher-weighted "
            "for cell strata, then test the counted signed magnitude alphabet."
        ),
        latex_form=(
            r"F(m)=\tfrac12\operatorname{sech}^2(m/2),\quad "
            r"f^\star=\min\arg\max_f\sum_{(r,x)\in B_{b,f}}\chi(r,x)w(r,x),"
            r"\quad w=1\ {\rm or}\ F(m),\quad |q|\in\{1\}\ {\rm or}\ \{1,2\},"
            r"\quad A_{\varnothing}=I"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_rg3_residual_family_productions_20260724:"
            "select_rg3_residual_address"
        ),
        domain_of_validity={
            "base": "SHA-bound V19C nested carrier",
            "families": list(FAMILIES),
            "address": "pair x class pair x stratum x temporal class x 64-row band x 16-row fine band",
            "inactive_identity": "A_empty=I; compiler returns exact input bytes",
            "class_birth_magnitude_alphabet": [1],
            "finer_and_fisher_magnitude_alphabet": [1, 2],
            "fisher_metric": "categorical Fisher trace = 0.5*sech(margin/2)^2",
            "fisher_payload_policy": "offline address selection only; no margin/scorer field shipped",
            "stream_type": "TypedStreamTag SKELETON/L3_raster",
            "research_only": True,
            "score_claim": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "verdict_scope": "INSTANCE_EXTENDED_GRAMMAR_RG3",
            "excluded": [
                "contest score or frontier movement",
                "Fourier residual basis",
                "RG4 iteration",
                "MS4 rerun without exact G3 pair-bucket coverage",
            ],
        },
        units_in={
            "receiver_support_mass_by_row": "receiver-support cells per raster row",
            "fisher_weighted_support_mass_by_row": "dimensionless Fisher-weighted support per row",
            "row_band": "zero-based 64-row band index",
        },
        units_out={
            "row_band": "zero-based 64-row band index",
            "fine_band": "zero-based 16-row subband within row band",
            "signed_magnitude_alphabet": "absolute integer quanta; direction is separately counted",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T11:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_rg1_receiver_grammar",
            "tools.build_ddm_rg3_residual_family_assignments",
            "tools.measure_ddm_ms6_receiver_support",
            "tools.summarize_ddm_ms6_receiver_support",
        ),
        canonical_producers=(
            "tools.build_ddm_rg3_residual_family_assignments",
            "tools.measure_ddm_ms6_receiver_support",
        ),
        provenance=provenance,
    )


def populate_ddm_rg3_residual_family_productions_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append RG3 through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_rg3_residual_family_productions_v1(
        source_receipt=source_receipt,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "RG3 terminal residual-family law; Fisher-margin address selection; "
            "advisory only; score_claim=false; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "CLASS_BIRTH",
    "EQUATION_ID",
    "FINER_EVENT",
    "FISHER_STRATUM",
    "build_ddm_rg3_residual_family_productions_v1",
    "populate_ddm_rg3_residual_family_productions_v1",
    "select_rg3_residual_address",
]
