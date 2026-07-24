# SPDX-License-Identifier: MIT
"""Canonical accounting law for the DDM #669c solved-value prices."""

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

EQUATION_ID = "ddm_dm1_semantic_record_price_and_rehome_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / (
    ".omx/research/ddm_dm1_25_row_solved_value_pricing_20260724T123443Z/"
    "ddm_dm1_25_row_solved_value_pricing_receipt.json"
)
CODECS = ("zlib9", "lzma9", "context_arithmetic")


def _prices(value: Mapping[str, Any], *, label: str) -> dict[str, int]:
    if set(value) != set(CODECS):
        raise ValueError(f"{label} must contain exactly the sealed three coders")
    output: dict[str, int] = {}
    for codec in CODECS:
        raw = value[codec]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"{label}.{codec} must be a positive exact byte count")
        output[codec] = raw
    return output


def account_dm1_semantic_prices(
    row_prices: Sequence[Mapping[str, Any]],
    *,
    joint_prices: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply exact independent and joint byte accounting to 25 semantic rows.

    ``row_prices`` and ``joint_prices`` are container byte counts after exact
    parseback.  This callable deliberately performs no score/slack arithmetic:
    semantic records are not receiver-closed archive streams.
    """

    if len(row_prices) != 25:
        raise ValueError("row_prices must contain exactly the registered 25 rows")
    rows = [_prices(value, label=f"row_prices[{index}]") for index, value in enumerate(row_prices)]
    joint = _prices(joint_prices, label="joint_prices")
    independent_by_codec = {
        codec: sum(row[codec] for row in rows)
        for codec in CODECS
    }
    independent_best = sum(min(row.values()) for row in rows)
    joint_codec = min(CODECS, key=lambda codec: (joint[codec], CODECS.index(codec)))
    return {
        "independent_best_per_row_bytes": independent_best,
        "independent_by_fixed_codec_bytes": independent_by_codec,
        "joint_best_bytes": joint[joint_codec],
        "joint_best_codec": joint_codec,
        "joint_savings_vs_independent_best_bytes": (
            independent_best - joint[joint_codec]
        ),
        "score_slack_arithmetic_permitted": False,
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if set(inputs) != {"row_prices", "joint_prices"}:
        raise ValueError("DM1 price inputs differ from the canonical callable contract")
    return account_dm1_semantic_prices(
        inputs["row_prices"],
        joint_prices=inputs["joint_prices"],
    )


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_dm1_semantic_record_price_and_rehome_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the instance-scoped #669c semantic price/re-homing law."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Rebuild after any solved-plane, PF2 event index, demand ledger, "
            "frozen SegNet, semantic-record grammar, or coder implementation SHA changes."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_threads4_pair_streaming",
        captured_at_utc="2026-07-24T12:34:43Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM #669c semantic-record price and deepest surviving home",
        one_line_summary=(
            "Price 25 exact semantic records independently and jointly; assign "
            "boundary placement to SKELETON/L4 and within-support choice to FIBER/L4."
        ),
        latex_form=(
            r"P_i=\min_{c\in\{\mathrm{z9,xz9,ctx}\}}"
            r"|C_c(r_i)|,\quad P_{\rm ind}=\sum_{i=1}^{25}P_i,\quad "
            r"P_{\rm joint}=\min_c|C_c(r_1\Vert\cdots\Vert r_{25})|;"
            r"\quad h(\partial\Omega)=({\rm SKELETON},L4),\ "
            r"h(v\mid\Omega)=({\rm FIBER},L4)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_dm1_solved_value_pricing_20260724:"
            "account_dm1_semantic_prices"
        ),
        domain_of_validity={
            "instance": "SHA-bound IS1 n600 solved planes and registered 25 PF2 rows",
            "row_count": 25,
            "boundary_rows": 16,
            "cell_rows": 9,
            "coders": list(CODECS),
            "parseback": "exact canonical semantic record and coder container",
            "boundary_home": "SKELETON/L4_scorer_feature; L3 is realization debt",
            "cell_home": "FIBER/L4_scorer_feature with SHA-bound external support",
            "connection_home": "NULL; no same-bucket delta-xi=1 comparator",
            "research_only": True,
            "score_claim": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "verdict_scope": (
                "semantic record only; excludes receiver-closed RGB, Pose, exact "
                "evaluator, archive bytes, score slack, promotion, and frontier movement"
            ),
        },
        units_in={
            "row_prices": "25 x exact coder-container bytes",
            "joint_prices": "exact coder-container bytes",
        },
        units_out={
            "independent_best_per_row_bytes": "bytes",
            "joint_best_bytes": "bytes",
            "joint_savings_vs_independent_best_bytes": "bytes",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T12:34:43Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_dm1_solved_value_pricing",
            "future MAIN-reviewed L3 receiver preimage compiler",
        ),
        canonical_producers=(
            "tools.measure_ddm_dm1_solved_value_pricing",
        ),
        provenance=provenance,
    )


def populate_ddm_dm1_semantic_record_price_and_rehome_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append #669c through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_dm1_semantic_record_price_and_rehome_v1(
        source_receipt=source_receipt,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "#669c exact semantic-record accounting; receiver price remains NULL; "
            "score_claim=false; pointer unchanged; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "CODECS",
    "EQUATION_ID",
    "account_dm1_semantic_prices",
    "build_ddm_dm1_semantic_record_price_and_rehome_v1",
    "populate_ddm_dm1_semantic_record_price_and_rehome_v1",
]
