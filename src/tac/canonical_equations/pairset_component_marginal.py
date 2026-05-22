# SPDX-License-Identifier: MIT
"""Canonical equation for pairset component marginal score decomposition."""
from __future__ import annotations

from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.optimization.pairset_component_marginal import (
    PAIRSET_COMPONENT_MARGINAL_SCORE_DECOMPOSITION_EQUATION_ID,
    build_component_score_delta_payload,
    component_score_delta,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

_OBSERVATION_FEEDBACK_MEMO = (
    ".omx/research/codex_findings_dqs1_pairset_observation_feedback_"
    "20260522T164706Z_codex.md"
)
_RANK019_HARDENING_MEMO = (
    ".omx/research/codex_findings_dqs1_rank019_queue_worker_drift_hardening_"
    "20260522T201951Z_codex.md"
)
_PREDICTED_PLACEHOLDER_SHA = "0" * 64


def pairset_component_marginal_score_delta(
    *,
    segnet_delta: float = 0.0,
    posenet_delta: float = 0.0,
    rate_delta: float | None = None,
    archive_byte_delta: float | None = None,
) -> float:
    """Return score delta = SegNet delta + PoseNet delta + rate delta."""

    return component_score_delta(
        segnet_delta=segnet_delta,
        posenet_delta=posenet_delta,
        rate_delta=rate_delta,
        archive_byte_delta=archive_byte_delta,
    )


def pairset_component_marginal_payload(
    **kwargs: Any,
) -> dict[str, Any]:
    """Return the full equation payload with status and false-authority fields."""

    return build_component_score_delta_payload(**kwargs)


def _predicted_provenance():
    return build_provenance_for_predicted(
        model_id="pairset_component_marginal_score_decomposition.v1",
        inputs_sha256=_PREDICTED_PLACEHOLDER_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )


def _anchor_provenance(
    axis: str,
    hardware_substrate: str,
    *,
    sidecar_path: str = _OBSERVATION_FEEDBACK_MEMO,
):
    return build_provenance_for_research_sidecar(
        sidecar_path=sidecar_path,
        reactivation_criteria=(
            "pairset_component_marginal_score_decomposition_empirical_anchor"
        ),
        measurement_axis=axis,
        hardware_substrate=hardware_substrate,
    )


def build_pairset_component_marginal_score_decomposition_v1() -> CanonicalEquation:
    """Build the pairset component marginal score-decomposition equation.

    The equation formalizes the exact component ledger observed on DQS1
    drop-one candidates:

        score_delta = segnet_delta + posenet_delta + rate_delta

    A one-byte archive saving has ``rate_delta = -25 / 37_545_489``. A
    drop-one candidate is component-safe only when the scorer penalty is
    smaller than that rate credit.
    """

    cpu_predicted = pairset_component_marginal_score_delta(
        segnet_delta=0.0,
        posenet_delta=0.0,
        rate_delta=-0.00000066585895312,
    )
    cpu_penalty_predicted = pairset_component_marginal_score_delta(
        segnet_delta=0.000001,
        posenet_delta=0.0,
        rate_delta=-0.00000066585895312,
    )
    cuda_predicted = pairset_component_marginal_score_delta(
        segnet_delta=0.000002,
        posenet_delta=0.0,
        rate_delta=-0.00000066585895312,
    )
    cpu_empirical = -0.00000066585895312
    cpu_penalty_empirical = 0.00000033414104686
    cuda_empirical = 0.00000133414104686
    cpu_anchor = EmpiricalAnchor(
        anchor_id="dqs1_pair0371_drop_one_component_cpu_20260522",
        measurement_utc="2026-05-22T18:14:50Z",
        inputs={
            "candidate_id": "pairset_drop_one_rank021_pair0371",
            "pair_index": 371,
            "axis": "[contest-CPU]",
            "segnet_delta": 0.0,
            "posenet_delta": 0.0,
            "rate_delta": -0.00000066585895312,
        },
        predicted_output={"score_delta": cpu_predicted},
        empirical_output={"observed_delta_vs_axis_baseline": cpu_empirical},
        residual=abs(cpu_predicted - cpu_empirical),
        source_artifact=_OBSERVATION_FEEDBACK_MEMO,
        measurement_method="dqs1_drop_one_pair0371_contest_cpu_component_delta",
        provenance=_anchor_provenance("[contest-CPU]", "linux_x86_64_cpu"),
    )
    cuda_anchor = EmpiricalAnchor(
        anchor_id="dqs1_pair0371_drop_one_component_cuda_t4_20260522",
        measurement_utc="2026-05-22T18:25:46Z",
        inputs={
            "candidate_id": "pairset_drop_one_rank021_pair0371",
            "pair_index": 371,
            "axis": "[contest-CUDA T4]",
            "segnet_delta": 0.000002,
            "posenet_delta": 0.0,
            "rate_delta": -0.00000066585895312,
        },
        predicted_output={"score_delta": cuda_predicted},
        empirical_output={"observed_delta_vs_axis_baseline": cuda_empirical},
        residual=abs(cuda_predicted - cuda_empirical),
        source_artifact=_OBSERVATION_FEEDBACK_MEMO,
        measurement_method="dqs1_drop_one_pair0371_contest_cuda_t4_component_delta",
        provenance=_anchor_provenance("[contest-CUDA T4]", "linux_x86_64_t4"),
    )
    rank026_cpu_penalty_anchor = EmpiricalAnchor(
        anchor_id="dqs1_pair0320_drop_one_component_cpu_penalty_20260522",
        measurement_utc="2026-05-22T18:41:57Z",
        inputs={
            "candidate_id": "pairset_drop_one_rank026_pair0320",
            "pair_index": 320,
            "axis": "[contest-CPU]",
            "segnet_delta": 0.000001,
            "posenet_delta": 0.0,
            "rate_delta": -0.00000066585895312,
        },
        predicted_output={"score_delta": cpu_penalty_predicted},
        empirical_output={"observed_delta_vs_axis_baseline": cpu_penalty_empirical},
        residual=abs(cpu_penalty_predicted - cpu_penalty_empirical),
        source_artifact=_OBSERVATION_FEEDBACK_MEMO,
        measurement_method="dqs1_drop_one_pair0320_contest_cpu_component_delta",
        provenance=_anchor_provenance("[contest-CPU]", "linux_x86_64_cpu"),
    )
    rank027_cpu_penalty_anchor = EmpiricalAnchor(
        anchor_id="dqs1_pair0378_drop_one_component_cpu_penalty_20260522",
        measurement_utc="2026-05-22T19:07:00Z",
        inputs={
            "candidate_id": "pairset_drop_one_rank027_pair0378",
            "pair_index": 378,
            "axis": "[contest-CPU]",
            "segnet_delta": 0.000001,
            "posenet_delta": 0.0,
            "rate_delta": -0.00000066585895312,
        },
        predicted_output={"score_delta": cpu_penalty_predicted},
        empirical_output={"observed_delta_vs_axis_baseline": cpu_penalty_empirical},
        residual=abs(cpu_penalty_predicted - cpu_penalty_empirical),
        source_artifact=_OBSERVATION_FEEDBACK_MEMO,
        measurement_method="dqs1_drop_one_pair0378_contest_cpu_component_delta",
        provenance=_anchor_provenance("[contest-CPU]", "linux_x86_64_cpu"),
    )
    rank031_cpu_penalty_anchor = EmpiricalAnchor(
        anchor_id="dqs1_pair0296_drop_one_component_cpu_penalty_20260522",
        measurement_utc="2026-05-22T19:23:06Z",
        inputs={
            "candidate_id": "pairset_drop_one_rank031_pair0296",
            "pair_index": 296,
            "axis": "[contest-CPU]",
            "segnet_delta": 0.000001,
            "posenet_delta": 0.0,
            "rate_delta": -0.00000066585895312,
        },
        predicted_output={"score_delta": cpu_penalty_predicted},
        empirical_output={"observed_delta_vs_axis_baseline": cpu_penalty_empirical},
        residual=abs(cpu_penalty_predicted - cpu_penalty_empirical),
        source_artifact=_OBSERVATION_FEEDBACK_MEMO,
        measurement_method="dqs1_drop_one_pair0296_contest_cpu_component_delta",
        provenance=_anchor_provenance("[contest-CPU]", "linux_x86_64_cpu"),
    )
    rank020_cpu_penalty_anchor = EmpiricalAnchor(
        anchor_id="dqs1_pair0430_drop_one_component_cpu_penalty_20260522",
        measurement_utc="2026-05-22T19:36:56Z",
        inputs={
            "candidate_id": "pairset_drop_one_rank020_pair0430",
            "pair_index": 430,
            "axis": "[contest-CPU]",
            "segnet_delta": 0.000001,
            "posenet_delta": 0.0,
            "rate_delta": -0.00000066585895312,
        },
        predicted_output={"score_delta": cpu_penalty_predicted},
        empirical_output={"observed_delta_vs_axis_baseline": cpu_penalty_empirical},
        residual=abs(cpu_penalty_predicted - cpu_penalty_empirical),
        source_artifact=_OBSERVATION_FEEDBACK_MEMO,
        measurement_method="dqs1_drop_one_pair0430_contest_cpu_component_delta",
        provenance=_anchor_provenance("[contest-CPU]", "linux_x86_64_cpu"),
    )
    rank022_cpu_penalty_anchor = EmpiricalAnchor(
        anchor_id="dqs1_pair0167_drop_one_component_cpu_penalty_20260522",
        measurement_utc="2026-05-22T19:52:59Z",
        inputs={
            "candidate_id": "pairset_drop_one_rank022_pair0167",
            "pair_index": 167,
            "axis": "[contest-CPU]",
            "segnet_delta": 0.000001,
            "posenet_delta": 0.0,
            "rate_delta": -0.00000066585895312,
        },
        predicted_output={"score_delta": cpu_penalty_predicted},
        empirical_output={"observed_delta_vs_axis_baseline": cpu_penalty_empirical},
        residual=abs(cpu_penalty_predicted - cpu_penalty_empirical),
        source_artifact=_OBSERVATION_FEEDBACK_MEMO,
        measurement_method="dqs1_drop_one_pair0167_contest_cpu_component_delta",
        provenance=_anchor_provenance("[contest-CPU]", "linux_x86_64_cpu"),
    )
    rank019_cpu_penalty_anchor = EmpiricalAnchor(
        anchor_id="dqs1_pair0151_drop_one_component_cpu_penalty_20260522",
        measurement_utc="2026-05-22T20:10:43Z",
        inputs={
            "candidate_id": "pairset_drop_one_rank019_pair0151",
            "pair_index": 151,
            "axis": "[contest-CPU]",
            "segnet_delta": 0.000001,
            "posenet_delta": 0.0,
            "rate_delta": -0.00000066585895312,
        },
        predicted_output={"score_delta": cpu_penalty_predicted},
        empirical_output={"observed_delta_vs_axis_baseline": cpu_penalty_empirical},
        residual=abs(cpu_penalty_predicted - cpu_penalty_empirical),
        source_artifact=_RANK019_HARDENING_MEMO,
        measurement_method="dqs1_drop_one_pair0151_contest_cpu_component_delta",
        provenance=_anchor_provenance(
            "[contest-CPU]",
            "linux_x86_64_cpu",
            sidecar_path=_RANK019_HARDENING_MEMO,
        ),
    )
    return CanonicalEquation(
        equation_id=PAIRSET_COMPONENT_MARGINAL_SCORE_DECOMPOSITION_EQUATION_ID,
        name="Pairset component marginal score decomposition",
        one_line_summary=(
            "Drop-pair score delta equals SegNet delta plus PoseNet delta plus "
            "rate delta; rate credit must exceed scorer penalty."
        ),
        latex_form=(
            r"\Delta S_{\mathrm{drop}} = \Delta S_{\mathrm{seg}} + "
            r"\Delta S_{\mathrm{pose}} + 25 \Delta B / 37{,}545{,}489"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pairset_component_marginal:"
            "pairset_component_marginal_score_delta"
        ),
        domain_of_validity={
            "candidate_family": ["decoder_q_selective_dqs1"],
            "operation": ["drop_one", "drop_two"],
            "axes": ["[contest-CPU]", "[contest-CUDA T4]"],
            "identity_required": [
                "candidate_id",
                "selected_pair_indices",
                "archive_sha256",
                "runtime_sha256",
            ],
        },
        units_in={
            "segnet_delta": "float_score_units",
            "posenet_delta": "float_score_units",
            "rate_delta": "float_score_units",
            "archive_byte_delta": "float_archive_bytes",
        },
        units_out={"score_delta": "float_score_units_negative_is_better"},
        empirical_anchors=(
            cpu_anchor,
            cuda_anchor,
            rank026_cpu_penalty_anchor,
            rank027_cpu_penalty_anchor,
            rank031_cpu_penalty_anchor,
            rank020_cpu_penalty_anchor,
            rank022_cpu_penalty_anchor,
            rank019_cpu_penalty_anchor,
        ),
        predicted_vs_empirical_residual={
            cpu_anchor.measurement_method: cpu_anchor.residual,
            cuda_anchor.measurement_method: cuda_anchor.residual,
            rank026_cpu_penalty_anchor.measurement_method: (
                rank026_cpu_penalty_anchor.residual
            ),
            rank027_cpu_penalty_anchor.measurement_method: (
                rank027_cpu_penalty_anchor.residual
            ),
            rank031_cpu_penalty_anchor.measurement_method: (
                rank031_cpu_penalty_anchor.residual
            ),
            rank020_cpu_penalty_anchor.measurement_method: (
                rank020_cpu_penalty_anchor.residual
            ),
            rank022_cpu_penalty_anchor.measurement_method: (
                rank022_cpu_penalty_anchor.residual
            ),
            rank019_cpu_penalty_anchor.measurement_method: (
                rank019_cpu_penalty_anchor.residual
            ),
        },
        last_calibration_utc="2026-05-22T20:10:43Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.cross_family_candidate_portfolio",
            "tac.xray.pairset_component_marginal",
            "tac.master_gradient_consumers",
        ),
        canonical_producers=(
            "tools/plan_cross_family_candidate_portfolio.py",
            "tools/recover_modal_auth_eval.py",
        ),
        provenance=_predicted_provenance(),
    )


__all__ = [
    "build_pairset_component_marginal_score_decomposition_v1",
    "pairset_component_marginal_payload",
    "pairset_component_marginal_score_delta",
]
