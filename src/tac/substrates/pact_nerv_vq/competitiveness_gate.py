# SPDX-License-Identifier: MIT
"""Fail-closed competitiveness gate for PACT-NeRV-VQ artifacts.

The gate separates a real byte win from a score-lowering candidate. PACT-VQ can
stay alive as a retraining family, but exact spend needs distortion evidence,
not just a smaller decoder packet.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY

PACT_VQ_COMPETITIVENESS_GATE_SCHEMA = "pact_nerv_vq_competitiveness_gate.v1"
PACT_VQ_COMPETITIVENESS_GATE_AXIS_TAG = "[macOS-MLX research-signal]"


class PactVqCompetitivenessGateError(RuntimeError):
    """Raised when a PACT-VQ competitiveness input is not usable."""


@dataclass(frozen=True)
class ScorePoint:
    """Minimal score summary extracted from a full-video replay response."""

    score: float
    d_seg: float
    d_pose: float
    rate_score: float
    archive_bytes: int
    response_path: str
    evidence_tag: str
    score_axis: str
    max_pairs: int
    n_samples: int

    @property
    def nonrate_score(self) -> float:
        return self.score - self.rate_score

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "nonrate_score": self.nonrate_score,
            "avg_segnet_dist": self.d_seg,
            "avg_posenet_dist": self.d_pose,
            "rate_score": self.rate_score,
            "archive_bytes": self.archive_bytes,
            "response_path": self.response_path,
            "evidence_tag": self.evidence_tag,
            "score_axis": self.score_axis,
            "max_pairs": self.max_pairs,
            "n_samples": self.n_samples,
        }


def build_pact_vq_competitiveness_gate(
    *,
    codec_sweep_report: Mapping[str, Any],
    source_replay_profile: Mapping[str, Any],
    best_codec_replay_profile: Mapping[str, Any],
    exact_spend_score_threshold: float = 1.0,
    max_segnet_dist_for_exact_spend: float = 0.02,
    max_posenet_dist_for_exact_spend: float = 1.0,
) -> dict[str, Any]:
    """Classify whether the current PACT-VQ candidate deserves exact spend.

    The input replays are expected to be full-video MLX research-signal reports.
    This function deliberately never turns them into contest authority.
    """

    if exact_spend_score_threshold <= 0:
        raise PactVqCompetitivenessGateError(
            "exact_spend_score_threshold must be positive"
        )
    if max_segnet_dist_for_exact_spend <= 0:
        raise PactVqCompetitivenessGateError(
            "max_segnet_dist_for_exact_spend must be positive"
        )
    if max_posenet_dist_for_exact_spend <= 0:
        raise PactVqCompetitivenessGateError(
            "max_posenet_dist_for_exact_spend must be positive"
        )

    _require_false_authority(codec_sweep_report, label="codec_sweep_report")
    _require_false_authority(source_replay_profile, label="source_replay_profile")
    _require_false_authority(
        best_codec_replay_profile,
        label="best_codec_replay_profile",
    )

    best_variant = _mapping(codec_sweep_report.get("best_variant"), "best_variant")
    best_decoder_codec = str(best_variant.get("decoder_codec") or "")
    if not best_decoder_codec:
        raise PactVqCompetitivenessGateError("best_variant.decoder_codec missing")
    receiver_proof_passed = best_variant.get("receiver_proof_passed") is True
    source = _baseline_score_point(source_replay_profile)
    best = _baseline_score_point(best_codec_replay_profile)
    _require_mlx_axis(source, label="source_replay_profile")
    _require_mlx_axis(best, label="best_codec_replay_profile")

    score_delta = best.score - source.score
    rate_delta = best.rate_score - source.rate_score
    nonrate_delta = best.nonrate_score - source.nonrate_score
    archive_byte_delta = best.archive_bytes - source.archive_bytes
    bytes_saved = archive_byte_delta < 0
    rate_fraction = _safe_ratio(best.rate_score, best.score)
    distortion_to_rate_score_ratio = _safe_ratio(best.nonrate_score, best.rate_score)
    rate_only_replay = abs(nonrate_delta) <= 1e-9 and rate_delta < 0
    preserve_rate_primitive = receiver_proof_passed and bytes_saved
    distortion_failed = (
        best.score > exact_spend_score_threshold
        or best.d_seg > max_segnet_dist_for_exact_spend
        or best.d_pose > max_posenet_dist_for_exact_spend
    )
    exact_spend_candidate = (
        receiver_proof_passed
        and not distortion_failed
        and best.score <= exact_spend_score_threshold
        and best.d_seg <= max_segnet_dist_for_exact_spend
        and best.d_pose <= max_posenet_dist_for_exact_spend
    )

    sweep_blockers = [str(item) for item in codec_sweep_report.get("blockers") or []]
    best_variant_blockers = [str(item) for item in best_variant.get("blockers") or []]
    replay_attached = bool(best.response_path) and bool(source.response_path)
    replay_supersedes_stale_blocker = replay_attached and (
        "full_video_mlx_scorer_replay_not_attached" in sweep_blockers
        or "full_video_mlx_scorer_replay_not_attached" in best_variant_blockers
    )

    blockers = [
        "mlx_local_replay_is_false_authority",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if not receiver_proof_passed:
        blockers.append("best_codec_receiver_proof_not_passed")
    if distortion_failed:
        blockers.append("pact_vq_distortion_not_competitive_at_current_fit")
    if rate_only_replay:
        blockers.append("best_codec_improves_rate_only_distortion_unchanged")
    if replay_supersedes_stale_blocker:
        blockers.append("codec_sweep_report_needs_replay_attachment")
    blockers = _ordered_unique(blockers)

    if exact_spend_candidate:
        verdict = "LOCAL_CANDIDATE_FOR_EXACT_SPEND_TRIAGE"
    elif preserve_rate_primitive and distortion_failed:
        verdict = "PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION"
    else:
        verdict = "NO_GO_CURRENT_ARTIFACT_REQUIRES_REPAIR"

    recommended_next_actions = [
        "preserve_best_codec_as_rate_primitive_for_full_stack_portfolio",
        "do_not_spend_exact_eval_on_rate_only_artifact_before_fit_gate",
        "keep_pact_vq_family_active_as_conditional_retraining_lane",
        "optimize_order_dependent_full_stack_interaction_before_design_retirement",
        "train_score_faithful_decoder_codebook_or_reduce_decoder_codebook_bytes",
        "attach_full_video_mlx_replay_to_codec_sweep_adjudication",
        "promote_only_after_receiver_proof_plus_full_video_distortion_gate",
    ]
    if exact_spend_candidate:
        recommended_next_actions = [
            "run_claimed_exact_cpu_cuda_pair_replay_before_any_score_claim",
            "keep_mlx_gate_false_authority_until_auth_axis_payload_passes",
        ]

    return {
        "schema": PACT_VQ_COMPETITIVENESS_GATE_SCHEMA,
        "axis_tag": PACT_VQ_COMPETITIVENESS_GATE_AXIS_TAG,
        "family": "pact_nerv_vq",
        "verdict": verdict,
        "abandon_family": False,
        "abandon_current_artifact": False if preserve_rate_primitive else not exact_spend_candidate,
        "demote_for_full_stack_portfolio": False if preserve_rate_primitive else not exact_spend_candidate,
        "exact_axis_blocked": not exact_spend_candidate,
        "preserve_rate_primitive": preserve_rate_primitive,
        "exact_spend_candidate": exact_spend_candidate,
        "best_decoder_codec": best_decoder_codec,
        "best_codec_receiver_proof_passed": receiver_proof_passed,
        "source": source.as_jsonable(),
        "best_codec": best.as_jsonable(),
        "deltas": {
            "score": score_delta,
            "nonrate_score": nonrate_delta,
            "rate_score": rate_delta,
            "archive_bytes": archive_byte_delta,
        },
        "rate_axis": {
            "rate_fraction_of_best_score": rate_fraction,
            "distortion_to_rate_score_ratio": distortion_to_rate_score_ratio,
            "rate_only_replay": rate_only_replay,
            "best_archive_bytes_from_codec_sweep": _safe_int(
                best_variant.get("archive_bytes")
            ),
            "best_archive_sha256_from_codec_sweep": best_variant.get("archive_sha256"),
            "bytes_saved_vs_source_replay": -archive_byte_delta if bytes_saved else 0,
        },
        "thresholds": {
            "exact_spend_score_threshold": exact_spend_score_threshold,
            "max_segnet_dist_for_exact_spend": max_segnet_dist_for_exact_spend,
            "max_posenet_dist_for_exact_spend": max_posenet_dist_for_exact_spend,
        },
        "integration_gaps": {
            "codec_sweep_blockers": sweep_blockers,
            "best_variant_blockers": best_variant_blockers,
            "full_video_replay_attached_in_gate": replay_attached,
            "replay_supersedes_stale_sweep_blocker": replay_supersedes_stale_blocker,
        },
        "blockers": blockers,
        "recommended_next_actions": recommended_next_actions,
        **FALSE_AUTHORITY,
    }


def build_pact_vq_competitiveness_gate_from_paths(
    *,
    codec_sweep_report_path: str | Path,
    source_replay_profile_path: str | Path,
    best_codec_replay_profile_path: str | Path,
    exact_spend_score_threshold: float = 1.0,
    max_segnet_dist_for_exact_spend: float = 0.02,
    max_posenet_dist_for_exact_spend: float = 1.0,
) -> dict[str, Any]:
    """Load JSON inputs and build the PACT-VQ competitiveness gate."""

    return build_pact_vq_competitiveness_gate(
        codec_sweep_report=_load_json(codec_sweep_report_path),
        source_replay_profile=_load_json(source_replay_profile_path),
        best_codec_replay_profile=_load_json(best_codec_replay_profile_path),
        exact_spend_score_threshold=exact_spend_score_threshold,
        max_segnet_dist_for_exact_spend=max_segnet_dist_for_exact_spend,
        max_posenet_dist_for_exact_spend=max_posenet_dist_for_exact_spend,
    )


def _baseline_score_point(profile: Mapping[str, Any]) -> ScorePoint:
    row = _baseline_variant_row(profile)
    response_path = str(row.get("mlx_response") or "")
    response = _load_json(response_path) if response_path else profile
    score = _safe_float(response.get("canonical_score"))
    if score is None:
        score = _safe_float(response.get("score_recomputed_from_components"))
    d_seg = _safe_float(response.get("avg_segnet_dist"))
    d_pose = _safe_float(response.get("avg_posenet_dist"))
    rate = _safe_float(response.get("score_rate_contribution"))
    archive_bytes = _safe_int(response.get("archive_size_bytes"))
    if archive_bytes == 0:
        archive_bytes = _safe_int(row.get("archive_zip_bytes"))
    missing = []
    if score is None:
        missing.append("canonical_score")
    if d_seg is None:
        missing.append("avg_segnet_dist")
    if d_pose is None:
        missing.append("avg_posenet_dist")
    if rate is None:
        missing.append("score_rate_contribution")
    if archive_bytes <= 0:
        missing.append("archive_size_bytes")
    if missing:
        raise PactVqCompetitivenessGateError(
            f"baseline replay response missing {', '.join(missing)}"
        )
    return ScorePoint(
        score=score,
        d_seg=d_seg,
        d_pose=d_pose,
        rate_score=rate,
        archive_bytes=archive_bytes,
        response_path=response_path,
        evidence_tag=str(response.get("evidence_tag") or ""),
        score_axis=str(response.get("score_axis") or ""),
        max_pairs=_safe_int(response.get("max_pairs")),
        n_samples=_safe_int(response.get("n_samples")),
    )


def _baseline_variant_row(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = profile.get("variant_rows")
    if not isinstance(rows, list):
        raise PactVqCompetitivenessGateError("profile.variant_rows missing")
    for row in rows:
        if isinstance(row, Mapping) and row.get("variant_id") == "baseline":
            return row
    raise PactVqCompetitivenessGateError("profile baseline variant missing")


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve(strict=False)
    if not p.is_file():
        raise PactVqCompetitivenessGateError(f"JSON input missing: {p}")
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PactVqCompetitivenessGateError(f"JSON input is not object: {p}")
    return payload


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for field in (
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if payload.get(field) is True:
            raise PactVqCompetitivenessGateError(
                f"{label}.{field} is true; local gate cannot consume authority rows"
            )


def _require_mlx_axis(point: ScorePoint, *, label: str) -> None:
    if point.score_axis != PACT_VQ_COMPETITIVENESS_GATE_AXIS_TAG:
        raise PactVqCompetitivenessGateError(
            f"{label}.score_axis must equal {PACT_VQ_COMPETITIVENESS_GATE_AXIS_TAG!r}; "
            f"got {point.score_axis!r}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PactVqCompetitivenessGateError(f"{label} missing or not object")
    return value


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_ratio(num: float, den: float) -> float | None:
    if den == 0:
        return None
    out = num / den
    return out if isfinite(out) else None


def _ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
