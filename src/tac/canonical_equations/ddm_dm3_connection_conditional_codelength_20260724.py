# SPDX-License-Identifier: MIT
"""Canonical exact-byte law for the DDM DM3 CONNECTION discriminator."""

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

EQUATION_ID = "ddm_dm3_heldout_connection_conditional_codelength_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / (
    ".omx/research/ddm_dm3_connection1_conditional_codelength_"
    "20260724T135912Z/ddm_dm3_connection1_conditional_codelength_receipt.json"
)
HISTORY_FAMILIES = ("identity", "xi_advected", "affine_tracked")


def _exact_bytes(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative exact integer byte count")
    return value


def account_dm3_connection_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact held-out static-versus-history byte rows.

    Each row must already have passed exact static and history parseback.  The
    callable performs no score, slack, or receiver arithmetic.
    """

    if len(rows) != 36:
        raise ValueError("rows must contain the registered 36 eligible buckets")
    seen: set[str] = set()
    static = program = residual = 0
    positive = 0
    family_counts = dict.fromkeys(HISTORY_FAMILIES, 0)
    decomposition: dict[tuple[str, str, int], dict[str, int]] = {}
    for index, row in enumerate(rows):
        bucket_id = row.get("bucket_id")
        if not isinstance(bucket_id, str) or not bucket_id or bucket_id in seen:
            raise ValueError(f"rows[{index}].bucket_id must be unique and nonempty")
        seen.add(bucket_id)
        family = row.get("winning_history_family")
        if family not in HISTORY_FAMILIES:
            raise ValueError(f"rows[{index}] introduced a history family")
        b_static = _exact_bytes(row.get("B_static"), f"rows[{index}].B_static")
        b_program = _exact_bytes(
            row.get("B_history_program"),
            f"rows[{index}].B_history_program",
        )
        b_residual = _exact_bytes(
            row.get("B_residual"),
            f"rows[{index}].B_residual",
        )
        delta = b_static - b_program - b_residual
        if row.get("delta_B_connection") != delta:
            raise ValueError(f"rows[{index}] delta_B_connection is inconsistent")
        bucket_type = row.get("bucket_type")
        stratum = row.get("stratum")
        support_size = row.get("support_size")
        if support_size is None:
            later_support = row.get("later_support")
            if isinstance(later_support, Mapping):
                support_size = later_support.get("count")
        if not isinstance(bucket_type, str) or not bucket_type:
            raise ValueError(f"rows[{index}].bucket_type must be nonempty")
        if stratum not in {"boundary", "cell"}:
            raise ValueError(f"rows[{index}].stratum is outside the sealed contract")
        if (
            isinstance(support_size, bool)
            or not isinstance(support_size, int)
            or support_size <= 0
        ):
            raise ValueError(f"rows[{index}].support_size must be positive")
        key = (bucket_type, stratum, support_size)
        group = decomposition.setdefault(
            key,
            {
                "row_count": 0,
                "B_static": 0,
                "B_history_program": 0,
                "B_residual": 0,
                "delta_B_connection": 0,
            },
        )
        group["row_count"] += 1
        group["B_static"] += b_static
        group["B_history_program"] += b_program
        group["B_residual"] += b_residual
        group["delta_B_connection"] += delta
        static += b_static
        program += b_program
        residual += b_residual
        positive += delta > 0
        family_counts[family] += 1
    return {
        "B_static": static,
        "B_history_program": program,
        "B_residual": residual,
        "B_history_total": program + residual,
        "delta_B_connection": static - program - residual,
        "positive_bucket_count": positive,
        "nonpositive_bucket_count": len(rows) - positive,
        "winning_family_counts": family_counts,
        "decomposition": [
            {
                "bucket_type": key[0],
                "stratum": key[1],
                "support_size": key[2],
                **decomposition[key],
            }
            for key in sorted(decomposition)
        ],
        "score_slack_arithmetic_permitted": False,
        "receiver_archive_bytes_inferred": False,
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if set(inputs) != {"rows"}:
        raise ValueError("DM3 equation inputs differ from the callable contract")
    return account_dm3_connection_rows(inputs["rows"])


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_dm3_heldout_connection_conditional_codelength_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the bucket-scoped held-out exact-byte law."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Rebuild after any solved-plane, PF2 index, frozen SegNet, DM1 "
            "record/coder, history-program grammar, or holdout-policy SHA changes."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_threads4_pair_streaming",
        captured_at_utc="2026-07-24T14:22:09Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM DM3 held-out CONNECTION conditional codelength",
        one_line_summary=(
            "Charge generic history selector/state plus exact residual and compare "
            "against a static DM1 record on one held-out transition per eligible bucket."
        ),
        latex_form=(
            r"\Delta B_{\rm conn}(b)="
            r"B_{\rm static}(r_{p+1}^b)-"
            r"\left(B_{\rm program}(h_{\neg p}^b)+"
            r"B_{\rm residual}(r_{p+1}^b\mid r_p^b,h_{\neg p}^b)\right),"
            r"\quad p\notin{\rm fit}(h_{\neg p}^b)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations."
            "ddm_dm3_connection_conditional_codelength_20260724:"
            "account_dm3_connection_rows"
        ),
        domain_of_validity={
            "instance": "SHA-bound IS1 n600 solved planes plus PF2 occupied supports",
            "represented_buckets": 37,
            "eligible_buckets": 36,
            "eligible_consecutive_transitions": 8_602,
            "heldout_policy": "lower median consecutive transition per bucket",
            "fit_policy": "all other same-bucket consecutive transitions",
            "history_families": list(HISTORY_FAMILIES),
            "coders": ["zlib9", "lzma9", "context_arithmetic"],
            "external_context": (
                "FIBER retains the sealed DM1 external SHA-bound target support; "
                "boundary SKELETON placement remains counted"
            ),
            "aggregate_anchor": {
                "B_static": 7_049,
                "B_history_program": 624,
                "B_residual": 5_237,
                "delta_B_connection": 1_188,
                "positive_buckets": 32,
                "nonpositive_buckets": 4,
                "identity_selected": 34,
                "xi_advected_selected": 0,
                "affine_tracked_selected": 2,
            },
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "verdict_scope": (
                "one deterministic held-out fold per eligible bucket; not an "
                "all-fold estimate, receiver price, archive delta, Pose result, "
                "contest score, family minimum, or promotion claim"
            ),
        },
        units_in={"rows": "36 exact charged bucket-level byte rows"},
        units_out={
            "B_static": "bytes",
            "B_history_program": "bytes",
            "B_residual": "bytes",
            "delta_B_connection": "bytes",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T14:22:09Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "DC1 temporal-latent selector after MAIN review",
            "#688 program-family acquisition after MAIN review",
            "future all-fold DDM CONNECTION probe",
        ),
        canonical_producers=(
            "tools.measure_ddm_dm3_connection_conditional_codelength",
        ),
        provenance=provenance,
    )


def populate_ddm_dm3_heldout_connection_conditional_codelength_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append DM3 through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_dm3_heldout_connection_conditional_codelength_v1(
        source_receipt=source_receipt,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DM3 +1188 B held-out aggregate over 36 buckets; identity wins "
            "34/36; semantic only; score_claim=false; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "HISTORY_FAMILIES",
    "account_dm3_connection_rows",
    "build_ddm_dm3_heldout_connection_conditional_codelength_v1",
    "populate_ddm_dm3_heldout_connection_conditional_codelength_v1",
]
