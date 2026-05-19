# SPDX-License-Identifier: MIT
"""MPS-axis score-response probe for PR101-family archives (codex op7 item #4).

Codex op7 iteration item #4 (2026-05-19): the score_response_matrix.md for
`pr101-op7-rank1-raw-byte-delta-same-length` shows local_cpu + local_mps
target rows BOTH False (scaffolded but not run). The cross-device sensitivity
data that would tell us per-byte response stability across MPS/CUDA/CPU was
NEVER MEASURED. This module adds the canonical local-MPS forward path so the
op7 anchor (and future PR101 candidates) get a 3-axis comparison instead of
just contest-CUDA / contest-CPU.

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #1 + Catalog
#192 + Catalog #317: MPS scores are `evidence_grade=MPS-research-signal`,
`score_claim=False`, `promotion_eligible=False`. This module SHALL NOT
produce a promotable contest-axis claim. The MPS axis is a free-but-noisy
research signal that complements (not replaces) the paid contest-CUDA +
contest-CPU pair.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE": promotion still requires Linux x86_64 +
NVIDIA. This module's outputs feed cathedral autopilot RANKING (Catalog
#317/#341 MPS-prescreen consumer routing) but never PROMOTION.

Per slot 9 `tac.mps_diagnostic.drift_predictor.predict_drift`: the
Cauchy-Schwarz upper bound on the MPS-vs-CUDA gap is PREDICTED BEFORE the
empirical probe runs; this module compares predicted vs measured to validate
the formalization memo §3.3 bound.

Sister modules:
- `tac.scorer.load_default_scorers(device="mps")`: hardware-aware loader.
- `tac.mps_diagnostic.drift_predictor.predict_drift`: predicted MPS gap.
- `tac.master_gradient_post_brotli_decompress`: corrected per-byte grain.
- `tac.master_gradient_iterative_refinement`: pass-N consumer.

Empirical anchor: PR101 op7 baseline sha b83bf348... (178258 archive bytes,
contest-CPU 0.19285) + candidate sha 30826b37... (contest-CPU regressed
+0.00169). Predicted MPS-vs-CUDA aggregate gap from slot 9 predictor is
~0.072% for the PR101 architecture (empirical CUDA score is 0.20533 from the
sister cathedral consumer wiring).

[verified-against:experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/score_response_matrix.md]
"""
from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional


__all__ = [
    "AUTH_EVAL_SCHEMA",
    "MPS_AXIS_TAG",
    "MPS_HARDWARE_SUBSTRATE_PREFIX",
    "MpsAxisProbeError",
    "MpsScoreResponseRecord",
    "build_cross_device_comparison_table",
    "build_mps_axis_provenance",
    "build_mps_research_signal_record",
    "validate_cauchy_schwarz_bound_on_measured_gap",
]


MPS_AXIS_TAG: str = "[MPS-research-signal]"
MPS_HARDWARE_SUBSTRATE_PREFIX: str = "macos_arm64"
AUTH_EVAL_SCHEMA: str = "tac_mps_axis_score_response_probe_v1"


class MpsAxisProbeError(RuntimeError):
    """Raised when MPS-axis probe arguments or computation are malformed."""


@dataclass(frozen=True)
class MpsScoreResponseRecord:
    """One MPS-axis score-response measurement on a single PR101 archive.

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #287/#323: every field
    is tagged with MPS_AXIS_TAG; score_claim_valid=False;
    promotion_eligible=False. The record is research-signal only.
    """

    archive_path: str
    archive_sha256: str
    archive_bytes: int
    seg_term: float
    pose_term: float
    rate_term: float
    score: float
    n_samples: int
    runtime_tree_sha256: Optional[str]
    measurement_axis: str = MPS_AXIS_TAG
    hardware_substrate: str = MPS_HARDWARE_SUBSTRATE_PREFIX
    score_claim_valid: bool = False
    promotion_eligible: bool = False
    measurement_utc: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.archive_sha256, str) or len(self.archive_sha256) != 64:
            raise MpsAxisProbeError(
                f"archive_sha256 must be a 64-char hex sha256; got len="
                f"{len(self.archive_sha256) if isinstance(self.archive_sha256, str) else type(self.archive_sha256).__name__}"
            )
        if self.archive_bytes <= 0:
            raise MpsAxisProbeError("archive_bytes must be > 0")
        if self.n_samples <= 0:
            raise MpsAxisProbeError("n_samples must be > 0")
        if self.measurement_axis != MPS_AXIS_TAG:
            raise MpsAxisProbeError(
                f"measurement_axis MUST be {MPS_AXIS_TAG!r}; got "
                f"{self.measurement_axis!r} — per CLAUDE.md 'MPS auth eval is NOISE'"
            )
        if self.score_claim_valid is not False:
            raise MpsAxisProbeError(
                "score_claim_valid MUST be False for MPS-axis records — "
                "per CLAUDE.md 'MPS auth eval is NOISE' non-negotiable"
            )
        if self.promotion_eligible is not False:
            raise MpsAxisProbeError(
                "promotion_eligible MUST be False for MPS-axis records — "
                "per Catalog #192 macos_cpu / mps non-promotion discipline"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "seg_term": self.seg_term,
            "pose_term": self.pose_term,
            "rate_term": self.rate_term,
            "score": self.score,
            "n_samples": self.n_samples,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "measurement_axis": self.measurement_axis,
            "hardware_substrate": self.hardware_substrate,
            "score_claim_valid": self.score_claim_valid,
            "promotion_eligible": self.promotion_eligible,
            "measurement_utc": self.measurement_utc,
        }


def build_mps_axis_provenance(
    archive_sha256: str,
    source_path: str,
    captured_at_utc: str | None = None,
):
    """Build the canonical Provenance row for an MPS-axis record.

    Routes through `tac.provenance.builders.build_provenance_for_mps_proxy`
    per Catalog #323 + CLAUDE.md "MPS auth eval is NOISE".

    Args:
        archive_sha256: the archive's sha256.
        source_path: path to the source (durable; not /tmp).
        captured_at_utc: optional UTC timestamp.

    Returns:
        Frozen `tac.provenance.Provenance` with kind=ADVISORY_NON_PROMOTABLE.
    """
    from tac.provenance import build_provenance_for_mps_proxy

    return build_provenance_for_mps_proxy(
        artifact_sha256=archive_sha256,
        source_path=source_path,
        captured_at_utc=captured_at_utc,
    )


def build_mps_research_signal_record(
    archive_path: Path,
    *,
    seg_term: float,
    pose_term: float,
    rate_term: float,
    score: float,
    n_samples: int,
    runtime_tree_sha256: Optional[str] = None,
) -> MpsScoreResponseRecord:
    """Build a frozen MpsScoreResponseRecord with auto-computed archive sha+bytes.

    Args:
        archive_path: path to the archive on disk.
        seg_term: 100 * d_seg from upstream/evaluate.py.
        pose_term: sqrt(10 * d_pose) from upstream/evaluate.py.
        rate_term: 25 * archive_bytes / 37545489 from upstream/evaluate.py.
        score: total scorer output.
        n_samples: number of frame pairs evaluated.
        runtime_tree_sha256: optional sha256 over the inflate.sh runtime tree.

    Returns:
        MpsScoreResponseRecord with all invariants enforced.
    """
    if not archive_path.exists():
        raise MpsAxisProbeError(f"archive_path does not exist: {archive_path}")
    raw = archive_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    return MpsScoreResponseRecord(
        archive_path=str(archive_path),
        archive_sha256=sha,
        archive_bytes=len(raw),
        seg_term=float(seg_term),
        pose_term=float(pose_term),
        rate_term=float(rate_term),
        score=float(score),
        n_samples=int(n_samples),
        runtime_tree_sha256=runtime_tree_sha256,
        measurement_utc=datetime.now(UTC).isoformat(),
    )


def build_cross_device_comparison_table(
    baseline_records: Mapping[str, Mapping[str, float]],
    candidate_records: Mapping[str, Mapping[str, float]],
    *,
    baseline_archive_sha256: str,
    candidate_archive_sha256: str,
) -> dict[str, Any]:
    """Side-by-side MPS-vs-CUDA-vs-CPU comparison.

    For each axis present in both `baseline_records` and `candidate_records`,
    computes the Δseg / Δpose / Δrate / Δtotal between candidate and baseline,
    PLUS the cross-device comparison MPS-vs-CPU and MPS-vs-CUDA so the
    operator can audit per-byte response stability.

    Args:
        baseline_records: {axis: {seg_term, pose_term, rate_term, score, ...}}
            for the baseline archive across each measurement axis.
        candidate_records: same shape for the candidate.
        baseline_archive_sha256: baseline sha (for the comparison header).
        candidate_archive_sha256: candidate sha.

    Returns:
        Cross-device comparison dict, serializable to JSON.
    """
    axes = sorted(set(baseline_records.keys()) & set(candidate_records.keys()))
    per_axis: list[dict[str, Any]] = []
    for axis in axes:
        b = baseline_records[axis]
        c = candidate_records[axis]
        seg_delta = float(c.get("seg_term", 0.0)) - float(b.get("seg_term", 0.0))
        pose_delta = float(c.get("pose_term", 0.0)) - float(b.get("pose_term", 0.0))
        rate_delta = float(c.get("rate_term", 0.0)) - float(b.get("rate_term", 0.0))
        total_delta = float(c.get("score", 0.0)) - float(b.get("score", 0.0))
        per_axis.append(
            {
                "axis": axis,
                "baseline_score": float(b.get("score", 0.0)),
                "candidate_score": float(c.get("score", 0.0)),
                "seg_term_delta": seg_delta,
                "pose_term_delta": pose_delta,
                "rate_term_delta": rate_delta,
                "total_delta": total_delta,
                "n_samples": int(b.get("n_samples", 0)),
            }
        )

    # Cross-device drift on the candidate's score (apples-to-apples per
    # archive bytes): MPS vs CPU and MPS vs CUDA aggregate gap fraction
    cross_device: dict[str, Any] = {}
    cpu_axis = next(
        (a for a in axes if "contest_cpu" in a or "macos_cpu" in a or "local_cpu" in a), None
    )
    cuda_axis = next(
        (a for a in axes if "contest_cuda" in a or "local_cuda" in a), None
    )
    mps_axis = next((a for a in axes if "mps" in a.lower()), None)
    if mps_axis and cuda_axis:
        c_mps = float(candidate_records[mps_axis].get("score", 0.0))
        c_cuda = float(candidate_records[cuda_axis].get("score", 0.0))
        denom = max(abs(c_cuda), 1e-12)
        cross_device["mps_vs_cuda_aggregate_fraction"] = abs(c_mps - c_cuda) / denom
        cross_device["mps_vs_cuda_absolute_gap"] = abs(c_mps - c_cuda)
    if mps_axis and cpu_axis:
        c_mps = float(candidate_records[mps_axis].get("score", 0.0))
        c_cpu = float(candidate_records[cpu_axis].get("score", 0.0))
        denom = max(abs(c_cpu), 1e-12)
        cross_device["mps_vs_cpu_aggregate_fraction"] = abs(c_mps - c_cpu) / denom
        cross_device["mps_vs_cpu_absolute_gap"] = abs(c_mps - c_cpu)

    return {
        "schema": AUTH_EVAL_SCHEMA,
        "baseline_archive_sha256": baseline_archive_sha256,
        "candidate_archive_sha256": candidate_archive_sha256,
        "axes_compared": axes,
        "per_axis": per_axis,
        "cross_device_drift_on_candidate": cross_device,
        "evidence_discipline": {
            "mps_axis_tag": MPS_AXIS_TAG,
            "mps_score_claim_valid": False,
            "mps_promotion_eligible": False,
            "claude_md_anchor": (
                "CLAUDE.md 'MPS auth eval is NOISE' + 'Submission auth eval — "
                "BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE'"
            ),
        },
    }


def validate_cauchy_schwarz_bound_on_measured_gap(
    predicted_gap_upper_bound: float, measured_gap: float
) -> dict[str, Any]:
    """Validate that the empirical MPS-vs-CUDA gap respects slot 9's bound.

    Per `tac.mps_diagnostic.drift_predictor.cauchy_schwarz_upper_bound`, the
    empirical gap MUST be <= the predicted upper bound (modulo finite-sample
    noise). A violation means either (a) the architecture features fed to
    `predict_drift` were wrong, or (b) the calibration anchors used are stale.

    Args:
        predicted_gap_upper_bound: from `DriftPrediction.predicted_aggregate_gap_upper_bound`.
        measured_gap: empirical fraction (|mps_score - cuda_score| / |cuda_score|).

    Returns:
        Verdict dict with "bound_satisfied" boolean + "headroom" + actionable text.
    """
    if predicted_gap_upper_bound < 0:
        raise MpsAxisProbeError(
            f"predicted_gap_upper_bound must be >=0; got {predicted_gap_upper_bound}"
        )
    if measured_gap < 0:
        raise MpsAxisProbeError(f"measured_gap must be >=0; got {measured_gap}")
    bound_satisfied = measured_gap <= predicted_gap_upper_bound
    headroom = predicted_gap_upper_bound - measured_gap
    if bound_satisfied:
        verdict = "BOUND_SATISFIED"
        action = (
            "Empirical MPS-vs-CUDA gap respects slot 9's Cauchy-Schwarz upper "
            "bound; no model recalibration required."
        )
    else:
        verdict = "BOUND_VIOLATED"
        action = (
            "Empirical MPS-vs-CUDA gap EXCEEDS slot 9's Cauchy-Schwarz upper "
            "bound by "
            f"{abs(headroom):.6e}; either (a) ArchitectureFeatures fed to "
            "predict_drift were wrong, or (b) CalibrationAnchors are stale. "
            "Operator-routable: re-fit calibration anchors + re-run predictor."
        )
    return {
        "verdict": verdict,
        "predicted_gap_upper_bound": float(predicted_gap_upper_bound),
        "measured_gap": float(measured_gap),
        "headroom": float(headroom),
        "bound_satisfied": bound_satisfied,
        "action_recommendation": action,
    }
