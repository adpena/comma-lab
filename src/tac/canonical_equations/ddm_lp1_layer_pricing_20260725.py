# SPDX-License-Identifier: MIT
"""Canonical accounting laws for DDM #669(b+c) layer pricing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_lp1_deepest_home_context_waterfill_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / (
    ".omx/research/ddm_lp1_layer_pricing_20260725T031654Z/"
    "ddm_lp1_layer_pricing_receipt.json"
)
LAYER_ORDER = (
    "L1_program",
    "L2_chart_grammar",
    "L3_RGB_realization",
    "L4_scorer_feature",
)
RATE_DUAL = 25 / 37_545_489


def price_context_and_allocate(
    context_races: Sequence[Mapping[str, Any]],
    *,
    measured_receiver_closed_marginals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select same-object contexts and reverse-waterfill measured marginals.

    A marginal row must supply exact ``bytes`` and ``joint_score_delta``.
    Negative joint score after the exact byte charge is the sole admission
    criterion.  Unmeasured planning reserves are not accepted as inputs.
    """

    context_rows: list[dict[str, Any]] = []
    for index, row in enumerate(context_races):
        required = {
            "explicit_bytes",
            "contextual_bytes",
            "context_parameter_bytes",
            "same_object",
        }
        if set(row) != required:
            raise ValueError(f"context_races[{index}] keys differ")
        explicit = row["explicit_bytes"]
        contextual = row["contextual_bytes"]
        parameters = row["context_parameter_bytes"]
        same_object = row["same_object"]
        for label, value in (
            ("explicit_bytes", explicit),
            ("contextual_bytes", contextual),
            ("context_parameter_bytes", parameters),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"context_races[{index}].{label} must be exact bytes")
        if not isinstance(same_object, bool):
            raise ValueError(f"context_races[{index}].same_object must be boolean")
        savings = explicit - contextual - parameters
        keep = same_object and savings > 0
        context_rows.append(
            {
                "savings_bytes": savings,
                "keep_context": keep,
                "selected_bytes": contextual + parameters if keep else explicit,
            }
        )

    admitted: list[dict[str, Any]] = []
    for index, row in enumerate(measured_receiver_closed_marginals):
        if set(row) != {"bytes", "joint_score_delta", "stream_id"}:
            raise ValueError(
                f"measured_receiver_closed_marginals[{index}] keys differ"
            )
        byte_count = row["bytes"]
        delta = row["joint_score_delta"]
        stream_id = row["stream_id"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError("marginal bytes must be a positive exact integer")
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(delta)
        ):
            raise ValueError("marginal joint_score_delta must be finite numeric")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("marginal stream_id must be a nonempty string")
        if delta < 0:
            admitted.append(
                {
                    "stream_id": stream_id,
                    "bytes": byte_count,
                    "joint_score_delta": float(delta),
                }
            )
    admitted.sort(key=lambda row: (row["joint_score_delta"] / row["bytes"], row["stream_id"]))
    return {
        "context_rows": context_rows,
        "admitted_marginals": admitted,
        "allocated_bytes": sum(row["bytes"] for row in admitted),
        "rate_dual_score_per_byte": RATE_DUAL,
        "unmeasured_reserves_allocate_zero": True,
    }


def deepest_surviving_layer(layer_survival: Mapping[str, bool]) -> str:
    """Return the deepest measured surviving layer, failing on holes."""

    if set(layer_survival) != set(LAYER_ORDER):
        raise ValueError("layer survival keys differ from the sealed L1-L4 ladder")
    seen_failure = False
    deepest: str | None = None
    for layer in LAYER_ORDER:
        survives = layer_survival[layer]
        if not isinstance(survives, bool):
            raise ValueError("layer survival values must be exact booleans")
        if survives and seen_failure:
            raise ValueError("layer survival cannot resume after an earlier failure")
        if survives:
            deepest = layer
        else:
            seen_failure = True
    if deepest is None:
        raise ValueError("stream has no measured surviving layer")
    return deepest


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if set(inputs) != {"context_races", "measured_receiver_closed_marginals"}:
        raise ValueError("LP1 equation inputs differ")
    return price_context_and_allocate(
        inputs["context_races"],
        measured_receiver_closed_marginals=inputs[
            "measured_receiver_closed_marginals"
        ],
    )


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_lp1_deepest_home_context_waterfill_v1(
    *, source_receipt: Path = RECEIPT
) -> CanonicalEquation:
    """Build the instance-scoped LP1 accounting law."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Rebuild after any C1, G4, DM1, DM2, DM4, PF, MS7, scorer, "
            "receiver grammar, or coder-owned same-object receipt SHA changes."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="receipt_only_no_new_execution",
        captured_at_utc="2026-07-25T03:16:54Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM LP1 deepest-home context waterfill",
        one_line_summary=(
            "Keep a decoder-derived context only on the same semantic object "
            "after charging video-derived parameters; allocate only measured "
            "receiver-closed negative joint-score marginals."
        ),
        latex_form=(
            r"h(s)=\max\{\ell\leq4:F_{\ell\to5}(s)\ne0\};\quad "
            r"B_{\rm ctx}=B_{\rm innovation}+B_{\rm video\ params};\quad "
            r"\Delta S_i<0\Rightarrow b_i\ {\rm admitted}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_lp1_layer_pricing_20260725:"
            "price_context_and_allocate"
        ),
        domain_of_validity={
            "instance": "SHA-bound C1/G4/DM1/DM2/DM4/PF/MS7 receipts",
            "layers": list(LAYER_ORDER),
            "types": ["GAUGE", "FIBER", "RESIDUAL", "CONTEXT", "PROGRAM"],
            "generic_receiver_and_context_code_bytes": 0,
            "video_derived_context_parameter_bytes": "counted",
            "same_object_required": True,
            "research_only": True,
            "score_claim": False,
            "verdict_scope": (
                "receipt accounting only; no coder-choice supersession, global "
                "minimum description, contest score, promotion, or pointer mutation"
            ),
        },
        units_in={
            "context_races": "exact parseback-equivalent bytes",
            "measured_receiver_closed_marginals": "bytes and joint score units",
        },
        units_out={
            "allocated_bytes": "bytes",
            "rate_dual_score_per_byte": "score units per byte",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-25T03:16:54Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_lp1_layer_pricing",
            "future MAIN-reviewed R6 C1 waterfill",
        ),
        canonical_producers=("tools.build_ddm_lp1_layer_pricing",),
        provenance=provenance,
    )


def populate_ddm_lp1_deepest_home_context_waterfill_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the LP1 law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_lp1_deepest_home_context_waterfill_v1(
        source_receipt=source_receipt,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "#669(b+c) deepest-home and context accounting; no coder-choice "
            "supersession; score_claim=false; pointer unchanged; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "LAYER_ORDER",
    "RATE_DUAL",
    "build_ddm_lp1_deepest_home_context_waterfill_v1",
    "deepest_surviving_layer",
    "populate_ddm_lp1_deepest_home_context_waterfill_v1",
    "price_context_and_allocate",
]
