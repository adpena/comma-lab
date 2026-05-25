# SPDX-License-Identifier: MIT
"""Canonical equations for MLX/PyTorch drift propagation.

These equations preserve local MLX parity evidence as planner-visible,
non-promotional system intelligence. They do not create score authority:
every anchor emitted here remains ``[macOS-MLX research-signal]`` and carries
false-authority blockers until paired contest CPU/CUDA evidence exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
    InvalidEquationError,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1"
DEFAULT_BELOW_SCORER_PRECISION_THRESHOLD = 0.001


def mlx_pytorch_full_decoder_downstream_scorer_drift_bound(
    *,
    aggregate_contest_score_drift_units: float,
    precision_threshold: float = DEFAULT_BELOW_SCORER_PRECISION_THRESHOLD,
) -> dict[str, Any]:
    """Classify a local MLX/PyTorch downstream drift measurement.

    The output is an engineering-bridge verdict only. It is intentionally
    non-promotional because the measurement axis is not contest CPU/CUDA.
    """

    drift = _finite_float(
        aggregate_contest_score_drift_units,
        "aggregate_contest_score_drift_units",
    )
    threshold = _finite_float(precision_threshold, "precision_threshold")
    if threshold <= 0.0:
        raise ValueError("precision_threshold must be > 0")

    if drift < threshold:
        verdict = "BELOW_SCORER_PRECISION"
    elif drift < threshold * 10.0:
        verdict = "AT_SCORER_PRECISION_BOUNDARY"
    else:
        verdict = "ABOVE_SCORER_PRECISION"

    return {
        "equation_id": EQUATION_ID,
        "aggregate_contest_score_drift_units": drift,
        "precision_threshold": threshold,
        "normalized_precision_fraction": drift / threshold,
        "verdict": verdict,
        "evidence_grade": "macOS-MLX research-signal",
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "macos_mlx_research_signal_not_contest_authority",
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
        ],
    }


def build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1(
    result_payload: Mapping[str, Any],
    *,
    source_artifact: str,
    captured_at_utc: str | None = None,
) -> CanonicalEquation:
    """Build the full-decoder downstream scorer-drift canonical equation."""

    if not isinstance(result_payload, Mapping):
        raise InvalidEquationError("result_payload must be a mapping")
    if not source_artifact:
        raise InvalidEquationError("source_artifact must be non-empty")

    aggregate = _finite_float(
        result_payload.get("aggregate_contest_score_drift_units"),
        "aggregate_contest_score_drift_units",
    )
    threshold = _stage5_precision_threshold(result_payload)
    verdict_payload = mlx_pytorch_full_decoder_downstream_scorer_drift_bound(
        aggregate_contest_score_drift_units=aggregate,
        precision_threshold=threshold,
    )
    measurement_utc = _measurement_utc(result_payload, captured_at_utc)
    inputs_sha = _sha256_json(
        {
            "archive_zip_sha256": result_payload.get("archive_zip_sha256"),
            "covered_pair_indices_sha256": result_payload.get(
                "covered_pair_indices_sha256"
            ),
            "n_pairs_actual": result_payload.get("n_pairs_actual"),
            "scorer_input_mode": result_payload.get("scorer_input_mode"),
            "checkpoint_mode": result_payload.get("checkpoint_mode"),
        }
    )
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=source_artifact,
        reactivation_criteria=(
            "rerun on wider PR95-class pair windows and paired contest "
            "CPU/CUDA before promotion or hardware-sensitive rank/kill"
        ),
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate=str(
            (result_payload.get("provenance") or {}).get(
                "hardware_substrate",
                "darwin_arm64_apple_silicon",
            )
        ),
        captured_at_utc=measurement_utc,
    )
    anchor = EmpiricalAnchor(
        anchor_id=_anchor_id(result_payload, measurement_utc),
        measurement_utc=measurement_utc,
        inputs={
            "archive_zip_sha256": result_payload.get("archive_zip_sha256"),
            "n_pairs_actual": int(result_payload.get("n_pairs_actual") or 0),
            "covered_pair_window": list(
                result_payload.get("covered_pair_window") or []
            ),
            "covered_pair_indices_sha256": result_payload.get(
                "covered_pair_indices_sha256"
            ),
            "checkpoint_mode": result_payload.get("checkpoint_mode"),
            "scorer_input_mode": result_payload.get("scorer_input_mode"),
            "posenet_sha256": result_payload.get("posenet_sha256"),
            "segnet_sha256": result_payload.get("segnet_sha256"),
        },
        predicted_output={
            "verdict": "BELOW_SCORER_PRECISION",
            "aggregate_contest_score_drift_units_upper_bound": threshold,
        },
        empirical_output={
            "aggregate_verdict": result_payload.get("aggregate_verdict"),
            "aggregate_contest_score_drift_units": aggregate,
            "selfcomp_mackay_theoretical_prediction_verified": bool(
                result_payload.get(
                    "selfcomp_mackay_theoretical_prediction_verified",
                    False,
                )
            ),
            "stage_digest": _stage_digest(result_payload),
        },
        residual=verdict_payload["normalized_precision_fraction"],
        source_artifact=source_artifact,
        measurement_method=(
            "pr95_mlx_pytorch_full_decoder_downstream_scorer_drift_5_stage"
        ),
        provenance=provenance,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="MLX/PyTorch full-decoder downstream scorer drift propagation",
        one_line_summary=(
            "PR95-class MLX/PyTorch decoder drift is bounded below local "
            "scorer precision only as macOS-MLX research signal."
        ),
        latex_form=(
            r"\Delta S_{\mathrm{MLX}\to\mathrm{Torch}} = "
            r"100 d_{\mathrm{seg}} + \sqrt{10 d_{\mathrm{pose}}} "
            r"< \epsilon_{\mathrm{local}}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.mlx_pytorch_drift:"
            "mlx_pytorch_full_decoder_downstream_scorer_drift_bound"
        ),
        domain_of_validity={
            "substrate_family": "PR95-class HNeRV decoder",
            "framework_pair": "MLX vs PyTorch",
            "measurement_axis": "[macOS-MLX research-signal]",
            "scorer_input_modes": ["contest_uint8", "decoder_float"],
            "promotion_authority": False,
            "requires_paired_contest_cpu_cuda_for_promotion": True,
        },
        units_in={
            "aggregate_contest_score_drift_units": "float_score_units",
            "precision_threshold": "float_score_units",
        },
        units_out={
            "normalized_precision_fraction": "float_fraction",
            "verdict": "str",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "normalized_precision_fraction": verdict_payload[
                "normalized_precision_fraction"
            ]
        },
        last_calibration_utc=measurement_utc,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift",
            "tools.run_pr95_mlx_long_training",
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
        ),
        canonical_producers=(
            "tools.measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift",
        ),
        provenance=build_provenance_for_predicted(
            model_id=EQUATION_ID,
            inputs_sha256=inputs_sha,
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="darwin_arm64_apple_silicon",
            captured_at_utc=measurement_utc,
        ),
    )


def build_equation_from_result_json(path: str | Path) -> CanonicalEquation:
    """Load a measurement JSON and build the canonical equation."""

    result_path = Path(path)
    payload = json.loads(result_path.read_text())
    return build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1(
        payload,
        source_artifact=str(result_path),
    )


def _stage5_precision_threshold(payload: Mapping[str, Any]) -> float:
    for stage in payload.get("stages") or ():
        if not isinstance(stage, Mapping):
            continue
        if stage.get("stage_name") != "stage_5_contest_score_aggregation":
            continue
        extra = stage.get("extra")
        if isinstance(extra, Mapping) and "precision_lo" in extra:
            return _finite_float(extra["precision_lo"], "stage_5.precision_lo")
    return DEFAULT_BELOW_SCORER_PRECISION_THRESHOLD


def _stage_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    digest: dict[str, Any] = {}
    for stage in payload.get("stages") or ():
        if not isinstance(stage, Mapping):
            continue
        name = str(stage.get("stage_name") or "")
        if not name:
            continue
        digest[name] = {
            "metric": stage.get("metric"),
            "max_abs": stage.get("max_abs"),
            "mean_abs": stage.get("mean_abs"),
            "rms": stage.get("rms"),
            "extra": _selected_stage_extra(name, stage.get("extra")),
        }
    return digest


def _selected_stage_extra(name: str, extra: Any) -> dict[str, Any]:
    if not isinstance(extra, Mapping):
        return {}
    keep_by_stage = {
        "stage_2_uint8_quantization_at_inflate": (
            "flipped_pixels",
            "total_pixels",
            "flipped_fraction",
        ),
        "stage_3_segnet_forward": (
            "argmax_flip_pixels",
            "total_pixels",
            "argmax_flip_fraction",
        ),
        "stage_5_contest_score_aggregation": (
            "aggregate_contest_delta_units",
            "verdict_per_stage",
            "precision_lo",
            "precision_hi",
        ),
    }
    keys = keep_by_stage.get(name, ())
    return {key: extra.get(key) for key in keys if key in extra}


def _measurement_utc(
    payload: Mapping[str, Any],
    captured_at_utc: str | None,
) -> str:
    if captured_at_utc:
        return captured_at_utc
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("captured_at_utc")
        if isinstance(value, str) and value:
            return value
    return datetime.now(UTC).isoformat()


def _anchor_id(payload: Mapping[str, Any], measurement_utc: str) -> str:
    compact = (
        measurement_utc.replace("-", "")
        .replace(":", "")
        .replace("+0000", "Z")
        .replace("+00:00", "Z")
    )
    compact = "".join(ch for ch in compact if ch.isalnum())
    n_pairs = int(payload.get("n_pairs_actual") or 0)
    mode = str(payload.get("scorer_input_mode") or "unknown")
    return f"pr95_full_decoder_downstream_{mode}_{n_pairs}_pairs_{compact}"


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return number


__all__ = [
    "DEFAULT_BELOW_SCORER_PRECISION_THRESHOLD",
    "EQUATION_ID",
    "build_equation_from_result_json",
    "build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1",
    "mlx_pytorch_full_decoder_downstream_scorer_drift_bound",
]
