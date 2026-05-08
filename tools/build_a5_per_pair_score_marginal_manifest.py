#!/usr/bin/env python3
"""Build advisory per-pair score-marginal evidence for the PR101 A5 packet."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.codec.frame_conditional_bit_budget import unpack_frame_conditional_q_bits  # noqa: E402
from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

SCHEMA = "pr101_a5_per_pair_score_marginals.v1"


class A5ScoreMarginalManifestError(ValueError):
    """Raised when an A5 score-marginal input is malformed."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5-manifest", type=Path, required=True)
    parser.add_argument("--candidate-archive-manifest", type=Path, required=True)
    parser.add_argument("--pair-difficulty-map", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def build_manifest(
    *,
    a5_manifest_path: Path,
    candidate_archive_manifest_path: Path,
    pair_difficulty_map_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    a5_manifest_path = _resolve(a5_manifest_path, repo_root)
    candidate_archive_manifest_path = _resolve(candidate_archive_manifest_path, repo_root)
    pair_difficulty_map_path = _resolve(pair_difficulty_map_path, repo_root)
    a5_manifest = _load_json_object(a5_manifest_path)
    candidate_manifest = _load_json_object(candidate_archive_manifest_path)
    pair_difficulty = _load_json_object(pair_difficulty_map_path)

    n_pairs = _positive_int(a5_manifest.get("n_pairs"), "a5_manifest.n_pairs")
    candidate_archive = _mapping(candidate_manifest.get("candidate_archive"), "candidate_archive")
    archive_path = _resolve(Path(str(candidate_archive.get("path"))), repo_root)
    member_manifest = _mapping(
        candidate_manifest.get("archive_member_manifest"), "archive_member_manifest"
    )
    q_bits = _decode_candidate_q_bits(archive_path, member_manifest, n_pairs=n_pairs)
    score, pose, seg = _pair_score_vectors(pair_difficulty, n_pairs=n_pairs)

    score_norm = _normalise_nonnegative(score)
    pose_norm = _normalise_nonnegative(pose)
    seg_norm = _normalise_nonnegative(seg)
    blended = _normalise_nonnegative(score_norm + pose_norm + seg_norm)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "marginal_evidence_available": True,
        "per_pair_score_marginals_ready": True,
        "evidence_grade": "[macOS-MPS advisory pair-difficulty proxy]",
        "evidence_semantics": (
            "advisory score-domain routing evidence only; exact CUDA and contest-CPU "
            "eval remain required before any score or promotion claim"
        ),
        "n_pairs": n_pairs,
        "source_artifacts": {
            "a5_manifest": _artifact(a5_manifest_path, repo_root),
            "candidate_archive_manifest": _artifact(
                candidate_archive_manifest_path, repo_root
            ),
            "pair_difficulty_map": _artifact(pair_difficulty_map_path, repo_root),
            "candidate_archive": _artifact(archive_path, repo_root),
        },
        "candidate_archive": {
            "path": repo_relative(archive_path, repo_root),
            "bytes": candidate_archive.get("bytes"),
            "sha256": candidate_archive.get("sha256"),
        },
        "q_bits_sideinfo": {
            "sha256": sha256_bytes(
                _candidate_sideinfo_bytes(archive_path, member_manifest)
            ),
            "q_bits_min": int(q_bits.min()),
            "q_bits_max": int(q_bits.max()),
            "q_bits_mean": float(np.mean(q_bits)),
            "q_bits_sha256": sha256_bytes(q_bits.tobytes()),
        },
        "per_pair_score_marginals": _round_list(blended),
        "per_pair_score_proxy_raw": _round_list(score),
        "per_pair_pose_proxy_raw": _round_list(pose),
        "per_pair_seg_proxy_raw": _round_list(seg),
        "per_pair_q_bits": [int(value) for value in q_bits.tolist()],
        "alignment": _alignment_report(q_bits.astype(np.float64), score, pose, seg),
        "dispatch_interpretation": {
            "a5_schedule_has_positive_score_alignment": bool(
                _pearson(q_bits, score) > 0.0
            ),
            "a5_schedule_has_positive_seg_alignment": bool(_pearson(q_bits, seg) > 0.0),
            "a5_schedule_has_positive_pose_alignment": bool(_pearson(q_bits, pose) > 0.0),
            "exact_eval_required": True,
        },
    }
    payload["manifest_sha256_excluding_self"] = sha256_bytes(json_text(payload).encode())
    return payload


def _decode_candidate_q_bits(
    archive_path: Path, member_manifest: dict[str, Any], *, n_pairs: int
) -> np.ndarray:
    sideinfo = _candidate_sideinfo_bytes(archive_path, member_manifest)
    q_bits = unpack_frame_conditional_q_bits(sideinfo, n_pairs=n_pairs)
    if q_bits.size != n_pairs:
        raise A5ScoreMarginalManifestError(
            f"decoded q_bits length {q_bits.size} != n_pairs {n_pairs}"
        )
    return q_bits


def _candidate_sideinfo_bytes(archive_path: Path, member_manifest: dict[str, Any]) -> bytes:
    member_name = str(member_manifest.get("member_name") or "x")
    offset = _positive_int(
        member_manifest.get("q_bits_sideinfo_offset"),
        "archive_member_manifest.q_bits_sideinfo_offset",
        allow_zero=True,
    )
    size = _positive_int(
        member_manifest.get("q_bits_sideinfo_bytes"),
        "archive_member_manifest.q_bits_sideinfo_bytes",
    )
    with zipfile.ZipFile(archive_path) as zf:
        payload = zf.read(member_name)
    end = offset + size
    if end > len(payload):
        raise A5ScoreMarginalManifestError(
            f"q_bits side-info slice {offset}:{end} exceeds member length {len(payload)}"
        )
    return payload[offset:end]


def _pair_score_vectors(payload: dict[str, Any], *, n_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = payload.get("pairs_by_difficulty")
    if not isinstance(rows, list):
        raise A5ScoreMarginalManifestError("pair_difficulty_map.pairs_by_difficulty missing")
    score = np.full(n_pairs, np.nan, dtype=np.float64)
    pose = np.full(n_pairs, np.nan, dtype=np.float64)
    seg = np.full(n_pairs, np.nan, dtype=np.float64)
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = _positive_int(row.get("pair_idx"), "pairs_by_difficulty[].pair_idx", allow_zero=True)
        if idx >= n_pairs:
            raise A5ScoreMarginalManifestError(f"pair_idx {idx} >= n_pairs {n_pairs}")
        score[idx] = _finite_float(row.get("score"), "pairs_by_difficulty[].score")
        pose[idx] = _finite_float(
            row.get("pose_contribution"), "pairs_by_difficulty[].pose_contribution"
        )
        seg[idx] = _finite_float(
            row.get("seg_contribution"), "pairs_by_difficulty[].seg_contribution"
        )
    if not (np.isfinite(score).all() and np.isfinite(pose).all() and np.isfinite(seg).all()):
        raise A5ScoreMarginalManifestError(
            "pair difficulty map must cover every pair with finite score/pose/seg values"
        )
    return score, pose, seg


def _alignment_report(
    q_bits: np.ndarray, score: np.ndarray, pose: np.ndarray, seg: np.ndarray
) -> dict[str, Any]:
    low = q_bits <= 2
    high = q_bits >= 8
    return {
        "q_bits_vs_score_pearson": _pearson(q_bits, score),
        "q_bits_vs_pose_pearson": _pearson(q_bits, pose),
        "q_bits_vs_seg_pearson": _pearson(q_bits, seg),
        "low_q_bits_pair_count": int(low.sum()),
        "high_q_bits_pair_count": int(high.sum()),
        "score_mean_low_q_bits_le_2": float(score[low].mean()) if low.any() else None,
        "score_mean_high_q_bits_ge_8": float(score[high].mean()) if high.any() else None,
        "pose_mean_low_q_bits_le_2": float(pose[low].mean()) if low.any() else None,
        "pose_mean_high_q_bits_ge_8": float(pose[high].mean()) if high.any() else None,
        "seg_mean_low_q_bits_le_2": float(seg[low].mean()) if low.any() else None,
        "seg_mean_high_q_bits_ge_8": float(seg[high].mean()) if high.any() else None,
    }


def _normalise_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    min_value = float(values.min())
    shifted = values - min_value
    total = float(shifted.sum())
    if total <= 0.0:
        return np.ones_like(values) / float(values.size)
    return shifted / total


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size != right.size or left.size == 0:
        raise A5ScoreMarginalManifestError("correlation inputs must have matching nonzero length")
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _round_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 12) for value in values.tolist()]


def _artifact(path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise A5ScoreMarginalManifestError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise A5ScoreMarginalManifestError(f"{label} must be a JSON object")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise A5ScoreMarginalManifestError(f"{label} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise A5ScoreMarginalManifestError(f"{label} must be an integer") from exc
    if out < 0 or (out == 0 and not allow_zero):
        raise A5ScoreMarginalManifestError(f"{label} must be positive")
    return out


def _finite_float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise A5ScoreMarginalManifestError(f"{label} must be numeric") from exc
    if not np.isfinite(out):
        raise A5ScoreMarginalManifestError(f"{label} must be finite")
    return out


def _resolve(path: Path, repo_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        payload = build_manifest(
            a5_manifest_path=args.a5_manifest,
            candidate_archive_manifest_path=args.candidate_archive_manifest,
            pair_difficulty_map_path=args.pair_difficulty_map,
            repo_root=REPO_ROOT,
        )
        payload = attach_tool_run_manifest(
            payload,
            tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
            argv=raw_argv,
            input_paths=[
                args.a5_manifest,
                args.candidate_archive_manifest,
                args.pair_difficulty_map,
            ],
            repo_root=REPO_ROOT,
            output_path=args.json_out,
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"FATAL: A5 score marginal manifest rejected: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
