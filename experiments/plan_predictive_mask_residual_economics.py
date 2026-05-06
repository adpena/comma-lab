#!/usr/bin/env python3
"""Plan PMG residual economics against sub-0.24 geometry targets.

This local-only planner reads existing PMG/CMG3 archive manifests and computes
whether row-span-plus-residual repair can plausibly satisfy both the C067 byte
target and a decoded-geometry trust threshold. It does not build archives,
launch jobs, load scorer networks, or make score claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "predictive_mask_residual_economics_v1"
TOOL = "experiments/plan_predictive_mask_residual_economics.py"
REPORT_NAME = "predictive_mask_residual_economics.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/predictive_mask_residual_economics_20260502"
)

FRONTIER_SCORE = 0.31561703078448233
FRONTIER_BYTES = 276_214
SUB024_TARGET_BYTES_UNCHANGED_DISTORTION = 162_650
GEOMETRY_TRUST_DISAGREEMENT_FRACTION = 0.001
MASK_STREAM_BYTES = 219_472
ORIGINAL_VIDEO_BYTES = 37_545_489

DEFAULT_MANIFESTS = (
    REPO_ROOT / "experiments/results/pmg_hotspot_candidate_c067_stride8_lzma_20260502/build_manifest.json",
    REPO_ROOT / "experiments/results/pmg_hotspot_candidate_c067_stride4_lzma_20260502/build_manifest.json",
    REPO_ROOT / "experiments/results/pmg_hotspot_candidate_c067_protect_top10_20260502/build_manifest.json",
    REPO_ROOT / "experiments/results/pmg_hotspot_candidate_c067_protect_top80_20260502/build_manifest.json",
    REPO_ROOT / "experiments/results/pmg_hotspot_candidate_c067_protect_top600_20260502/build_manifest.json",
)


class PlannerError(ValueError):
    """Raised for malformed residual-economics inputs."""


@dataclass(frozen=True)
class PmgPoint:
    path: Path
    candidate_id: str
    archive_bytes: int
    archive_sha256: str
    payload_bytes: int
    final_disagreement: float
    residual_pixels_touched: int
    protected_pair_count: int | None

    @property
    def archive_delta_vs_frontier(self) -> int:
        return int(self.archive_bytes - FRONTIER_BYTES)

    @property
    def score_rate_delta_vs_frontier(self) -> float:
        return 25.0 * float(self.archive_delta_vs_frontier) / float(ORIGINAL_VIDEO_BYTES)

    def as_dict(self, repo_root: Path) -> dict[str, Any]:
        return {
            "manifest_path": _display_path(self.path, repo_root),
            "candidate_id": self.candidate_id,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "archive_delta_vs_frontier": self.archive_delta_vs_frontier,
            "formula_only_rate_delta_vs_frontier": self.score_rate_delta_vs_frontier,
            "payload_bytes": self.payload_bytes,
            "final_disagreement_fraction": self.final_disagreement,
            "residual_pixels_touched": self.residual_pixels_touched,
            "protected_pair_count": self.protected_pair_count,
            "score_claim": False,
        }


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PlannerError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannerError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise PlannerError(f"{field} must be finite")
    return out


def _int_value(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerError(f"{field} must be an integer")
    return int(value)


def _str_value(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlannerError(f"{field} must be a nonempty string")
    return value


def load_pmg_point(path: Path) -> PmgPoint:
    payload = _read_json(path)
    output_archive = payload.get("output_archive")
    cmg3 = payload.get("pmg_hotspot_cmg3")
    if not isinstance(output_archive, dict):
        raise PlannerError(f"{path}: missing output_archive object")
    if not isinstance(cmg3, dict):
        raise PlannerError(f"{path}: missing pmg_hotspot_cmg3 object")
    pair_protection = cmg3.get("pair_protection")
    protected_pair_count: int | None = None
    if isinstance(pair_protection, dict):
        protected = pair_protection.get("protected_pair_indices")
        if isinstance(protected, list):
            protected_pair_count = len(protected)
    return PmgPoint(
        path=path,
        candidate_id=_str_value(
            payload.get("pmg_hotspot_plan", {}).get("candidate_id")
            if isinstance(payload.get("pmg_hotspot_plan"), dict)
            else payload.get("candidate_id"),
            field=f"{path}: candidate_id",
        ),
        archive_bytes=_int_value(output_archive.get("bytes"), field=f"{path}: archive bytes"),
        archive_sha256=_str_value(output_archive.get("sha256"), field=f"{path}: archive sha256"),
        payload_bytes=_int_value(cmg3.get("payload_bytes"), field=f"{path}: payload_bytes"),
        final_disagreement=_finite_float(
            cmg3.get("final_pixel_disagreement_vs_source_fraction"),
            field=f"{path}: final disagreement",
        ),
        residual_pixels_touched=_int_value(
            cmg3.get("residual_pixels_touched"), field=f"{path}: residual pixels"
        ),
        protected_pair_count=protected_pair_count,
    )


def _sorted_points(points: list[PmgPoint]) -> list[PmgPoint]:
    return sorted(
        points,
        key=lambda point: (
            point.final_disagreement,
            point.archive_bytes,
            point.payload_bytes,
            point.candidate_id,
        ),
    )


def lower_envelope(points: list[PmgPoint]) -> list[PmgPoint]:
    """Return points that are not byte- and disagreement-dominated."""
    envelope: list[PmgPoint] = []
    for point in sorted(points, key=lambda p: (p.final_disagreement, p.archive_bytes)):
        dominated = any(
            other.archive_bytes <= point.archive_bytes
            and other.final_disagreement <= point.final_disagreement
            and (
                other.archive_bytes < point.archive_bytes
                or other.final_disagreement < point.final_disagreement
            )
            for other in points
        )
        if not dominated:
            envelope.append(point)
    return envelope


def interpolate_archive_bytes_for_threshold(points: list[PmgPoint], threshold: float) -> dict[str, Any]:
    """Linearly interpolate archive bytes for a disagreement threshold.

    The interpolation is a local planning estimate over measured manifest
    points, not a proof of monotonicity.
    """
    ordered = _sorted_points(points)
    if not ordered:
        raise PlannerError("at least one PMG point is required")
    passing = [point for point in ordered if point.final_disagreement <= threshold]
    if passing:
        best = min(passing, key=lambda p: p.archive_bytes)
        return {
            "threshold": threshold,
            "estimated_archive_bytes": best.archive_bytes,
            "method": "observed_point",
            "bracket": [best.as_dict(REPO_ROOT)],
            "planning_only": True,
        }
    lower = min(ordered, key=lambda p: p.final_disagreement)
    return {
        "threshold": threshold,
        "estimated_archive_bytes": lower.archive_bytes,
        "method": "unreached_threshold_best_observed",
        "best_observed_disagreement_fraction": lower.final_disagreement,
        "bracket": [lower.as_dict(REPO_ROOT)],
        "planning_only": True,
    }


def build_plan(
    *,
    manifest_paths: list[Path],
    output_json: Path,
    command: list[str] | None = None,
    repo_root: Path = REPO_ROOT,
    geometry_threshold: float = GEOMETRY_TRUST_DISAGREEMENT_FRACTION,
) -> dict[str, Any]:
    points = [load_pmg_point(path) for path in manifest_paths if path.exists()]
    missing = [path for path in manifest_paths if not path.exists()]
    if not points:
        raise PlannerError("no readable PMG manifests were provided")
    envelope = lower_envelope(points)
    threshold_estimate = interpolate_archive_bytes_for_threshold(envelope, geometry_threshold)
    exact_reconstruction = interpolate_archive_bytes_for_threshold(envelope, 0.0)
    byte_target_passes = [
        point for point in points if point.archive_bytes <= SUB024_TARGET_BYTES_UNCHANGED_DISTORTION
    ]
    geometry_target_passes = [
        point for point in points if point.final_disagreement <= geometry_threshold
    ]
    joint_passes = [
        point
        for point in points
        if point.archive_bytes <= SUB024_TARGET_BYTES_UNCHANGED_DISTORTION
        and point.final_disagreement <= geometry_threshold
    ]
    best_byte = min(points, key=lambda p: p.archive_bytes)
    best_geometry = min(points, key=lambda p: (p.final_disagreement, p.archive_bytes))
    decision = (
        "residual_rowspan_not_sub024_viable"
        if not joint_passes
        else "residual_rowspan_has_local_candidate_requiring_exact_cuda"
    )
    plan = {
        "schema": SCHEMA,
        "producer": TOOL,
        "command": command or [],
        "score_claim": False,
        "promotion_eligible": False,
        "remote_jobs_dispatched": False,
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "frontier": {
            "score": FRONTIER_SCORE,
            "archive_bytes": FRONTIER_BYTES,
            "sub024_target_archive_bytes_at_unchanged_distortion": (
                SUB024_TARGET_BYTES_UNCHANGED_DISTORTION
            ),
            "mask_stream_bytes": MASK_STREAM_BYTES,
        },
        "inputs": {
            "manifest_count": len(points),
            "missing_manifests": [_display_path(path, repo_root) for path in missing],
            "manifests_sha256": {
                _display_path(point.path, repo_root): _sha256_file(point.path)
                for point in points
            },
        },
        "points": [point.as_dict(repo_root) for point in _sorted_points(points)],
        "lower_envelope": [point.as_dict(repo_root) for point in envelope],
        "best_byte_point": best_byte.as_dict(repo_root),
        "best_geometry_point": best_geometry.as_dict(repo_root),
        "thresholds": {
            "geometry_trust_disagreement_fraction": geometry_threshold,
            "sub024_archive_bytes": SUB024_TARGET_BYTES_UNCHANGED_DISTORTION,
        },
        "threshold_estimates": {
            "geometry_trust": threshold_estimate,
            "exact_mask_reconstruction": exact_reconstruction,
        },
        "pass_counts": {
            "byte_target_only": len(byte_target_passes),
            "geometry_target_only": len(geometry_target_passes),
            "joint_byte_and_geometry": len(joint_passes),
        },
        "decision": decision,
        "ranked_next_actions": [
            {
                "rank": 1,
                "action_id": "learned_geometry_preserving_mask_decoder",
                "status": "highest_ev",
                "reason": (
                    "row-span residual coding has byte headroom only while disagreement "
                    "remains far outside geometry trust; exact reconstruction is byte-regressive"
                ),
            },
            {
                "rank": 2,
                "action_id": "use_residual_economics_as_training_loss_budget",
                "status": "local_design_input",
                "reason": (
                    "the learned decoder must model geometry, not memorize per-pixel residuals; "
                    "residual bytes define the fallback budget"
                ),
            },
            {
                "rank": 3,
                "action_id": "remote_exact_eval",
                "status": "blocked" if not joint_passes else "requires_claim_and_cuda",
                "reason": "no PMG residual point currently satisfies both sub-0.24 bytes and geometry trust",
            },
        ],
    }
    _write_json(output_json, plan)
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--geometry-threshold", type=float, default=GEOMETRY_TRUST_DISAGREEMENT_FRACTION)
    parser.add_argument("--manifest", action="append", type=Path, dest="manifests")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = args.output_json or (args.output_dir / REPORT_NAME)
    manifests = list(args.manifests) if args.manifests else list(DEFAULT_MANIFESTS)
    plan = build_plan(
        manifest_paths=manifests,
        output_json=output_json,
        command=[str(Path(sys.argv[0]).as_posix()), *sys.argv[1:]],
        geometry_threshold=float(args.geometry_threshold),
    )
    print(
        json.dumps(
            {
                "decision": plan["decision"],
                "joint_passes": plan["pass_counts"]["joint_byte_and_geometry"],
                "best_byte": plan["best_byte_point"]["archive_bytes"],
                "best_geometry": plan["best_geometry_point"]["archive_bytes"],
                "score_claim": False,
                "output_json": str(output_json),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
