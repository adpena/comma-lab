#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare public pose manifolds against the C102 QP1 frontier.

This is a local planning tool. It decodes QP1-compatible pose streams from the
C102 parent and public PR75/PR77/PR65 sources where available, forms sparse
low-dimensional velocity-column proposal bases, and ranks candidate families by
byte cost, proxy benefit, risk, and break-even against the target score.

It does not build archives, run exact eval, claim score, or dispatch remote
work. Every emitted candidate has ``exact_eval_readiness=false``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import brotli
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.qp1_pose_codec import QP1_MAGIC, VELOCITY_SCALE, decode_qp1, encode_qp1


TOOL = "experiments/compare_public_pose_manifolds.py"
SCHEMA_VERSION = 1
EXPECTED_PAIRS = 600
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
TARGET_SCORE = 0.31

DEFAULT_C102_ARCHIVE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip"
)
DEFAULT_C102_EVAL = DEFAULT_C102_ARCHIVE.parent / "contest_auth_eval.json"
DEFAULT_C102_TRACE = DEFAULT_C102_ARCHIVE.parent / "component_trace.json"
DEFAULT_PR75_ARCHIVE = REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr75/archive.zip"
DEFAULT_PR75_DECODED_POSE = REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr75/pose_q_br.decoded"
DEFAULT_PR77_ARCHIVE = REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip"
DEFAULT_PR77_DECODED_POSE = REPO_ROOT / (
    "experiments/results/top_submission_reverse_engineering_20260503_pr77/unpacked/optimized_poses.qp1"
)
DEFAULT_PR65_ARCHIVE = REPO_ROOT / (
    "experiments/results/top_submission_delta_reverse_engineering_20260503/"
    "sources/pr65_henosis_archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/public_pose_manifold_compare_20260503_worker"
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/public_pose_manifold_compare_20260503_worker.md"


class PublicPoseManifoldError(ValueError):
    """Raised when input custody or deterministic planning fails closed."""


@dataclass(frozen=True)
class PoseSourceSpec:
    label: str
    archive_path: Path | None
    decoded_pose_path: Path | None = None
    kind: str = "qp1_archive"


@dataclass(frozen=True)
class PoseSource:
    label: str
    available: bool
    reason: str | None
    archive_path: str | None
    archive_bytes: int | None
    archive_sha256: str | None
    pose_stream_bytes: int | None
    pose_stream_sha256: str | None
    pose_words_sha256: str | None
    pose_float32_sha256: str | None
    word_count: int | None
    words: np.ndarray | None
    decoded_by: str | None
    risk_flags: tuple[str, ...]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _safe_zip_member(name: str) -> str:
    parts = Path(name).parts
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or len(parts) != 1
        or any(part in {"", ".", ".."} for part in parts)
        or name.startswith(".")
        or name == "__MACOSX"
    ):
        raise PublicPoseManifoldError(f"unsafe ZIP member path: {name!r}")
    return name


def _zip_inventory(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as zf:
        out = []
        seen: set[str] = set()
        for info in zf.infolist():
            if info.is_dir():
                continue
            _safe_zip_member(info.filename)
            if info.filename in seen:
                raise PublicPoseManifoldError(f"duplicate archive member: {info.filename}")
            seen.add(info.filename)
            out.append(
                {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                }
            )
        return out


def decode_qp1_words(payload: bytes) -> list[int]:
    """Decode QP1 bytes to the quantized velocity-column words."""

    if not payload.startswith(QP1_MAGIC):
        raise PublicPoseManifoldError(f"bad QP1 magic: {payload[:4]!r}")
    if len(payload) < 5:
        raise PublicPoseManifoldError("QP1 payload too short")
    words = [struct.unpack_from("<H", payload, 3)[0]]
    cursor = 5
    while cursor < len(payload):
        shift = 0
        acc = 0
        while True:
            if cursor >= len(payload):
                raise PublicPoseManifoldError("truncated QP1 VLQ")
            byte = payload[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        delta = (acc >> 1) ^ -(acc & 1)
        words.append((words[-1] + delta) & 0xFFFF)
    return words


def _read_qp1_from_archive(path: Path) -> tuple[bytes, str]:
    """Read the optimized pose stream from supported local archive formats."""

    try:
        from experiments.build_qp1_pose_active_subspace_candidates import load_archive_parts

        parts = load_archive_parts(path)
        return brotli.decompress(parts.pose_br), f"builder:{parts.payload_format}"
    except Exception as first_exc:
        try:
            from experiments.plan_c091_pose_manifold_bigmove import parse_source_archive

            parsed = parse_source_archive("public_pose_compare", path)
            return parsed.decoded["optimized_poses.qp1"], f"runtime:{parsed.slices.payload_format}"
        except Exception as second_exc:
            raise PublicPoseManifoldError(
                f"could not decode QP1 pose from {path}: {first_exc}; {second_exc}"
            ) from second_exc


def _read_pr65_as_qp1(path: Path) -> tuple[bytes, str]:
    from experiments.plan_pr65_henosis_stream_transfer import (
        decode_pr65_p1d1_pose,
        parse_pr65_henosis_archive,
    )

    parsed = parse_pr65_henosis_archive(path, expected_sha256=None)
    poses = decode_pr65_p1d1_pose(parsed["_segments_bytes"]["pose"])
    return encode_qp1(poses), "pr65_p1d1_reencoded_to_qp1_velocity_col0"


def load_pose_source(spec: PoseSourceSpec, *, expected_len: int | None = EXPECTED_PAIRS) -> PoseSource:
    risk_flags: list[str] = ["planning_only_not_score_evidence"]
    if spec.kind == "pr65":
        risk_flags.extend(
            [
                "pr65_direct_transfer_or_qpost_exact_negatives_exist",
                "pr65_reencoded_to_qp1_velocity_only_for_basis_comparison",
            ]
        )
    elif spec.label.upper() in {"PR75", "PR77"}:
        risk_flags.extend(
            [
                "public_replay_components_do_not_promote_locally",
                "public_difference_basis_not_direct_stream_copy",
            ]
        )
    if spec.archive_path is None or not spec.archive_path.exists():
        return PoseSource(
            label=spec.label,
            available=False,
            reason="archive_missing",
            archive_path=str(spec.archive_path) if spec.archive_path is not None else None,
            archive_bytes=None,
            archive_sha256=None,
            pose_stream_bytes=None,
            pose_stream_sha256=None,
            pose_words_sha256=None,
            pose_float32_sha256=None,
            word_count=None,
            words=None,
            decoded_by=None,
            risk_flags=tuple(risk_flags),
        )

    archive_bytes = spec.archive_path.stat().st_size
    archive_sha = _sha256_path(spec.archive_path)
    try:
        _zip_inventory(spec.archive_path)
        if spec.decoded_pose_path is not None and spec.decoded_pose_path.exists():
            raw = spec.decoded_pose_path.read_bytes()
            decoded_by = f"decoded_pose_path:{spec.decoded_pose_path}"
            if not raw.startswith(QP1_MAGIC):
                raise PublicPoseManifoldError(
                    f"{spec.decoded_pose_path} does not start with QP1 magic"
                )
        elif spec.kind == "pr65":
            raw, decoded_by = _read_pr65_as_qp1(spec.archive_path)
        else:
            raw, decoded_by = _read_qp1_from_archive(spec.archive_path)
        words = np.asarray(decode_qp1_words(raw), dtype=np.int32)
        if expected_len is not None and len(words) != expected_len:
            raise PublicPoseManifoldError(
                f"{spec.label} pose word count {len(words)} != expected {expected_len}"
            )
        poses = decode_qp1(raw).astype("<f4", copy=False)
        return PoseSource(
            label=spec.label,
            available=True,
            reason=None,
            archive_path=str(spec.archive_path),
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha,
            pose_stream_bytes=len(raw),
            pose_stream_sha256=_sha256_bytes(raw),
            pose_words_sha256=_sha256_bytes(words.astype("<i4", copy=False).tobytes()),
            pose_float32_sha256=_sha256_bytes(poses.tobytes()),
            word_count=int(len(words)),
            words=words,
            decoded_by=decoded_by,
            risk_flags=tuple(risk_flags),
        )
    except Exception as exc:
        return PoseSource(
            label=spec.label,
            available=False,
            reason=str(exc),
            archive_path=str(spec.archive_path),
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha,
            pose_stream_bytes=None,
            pose_stream_sha256=None,
            pose_words_sha256=None,
            pose_float32_sha256=None,
            word_count=None,
            words=None,
            decoded_by=None,
            risk_flags=tuple([*risk_flags, "decode_unavailable"]),
        )


def _source_summary(source: PoseSource) -> dict[str, Any]:
    return {
        "label": source.label,
        "available": source.available,
        "reason": source.reason,
        "archive_path": source.archive_path,
        "archive_bytes": source.archive_bytes,
        "archive_sha256": source.archive_sha256,
        "pose_stream_bytes": source.pose_stream_bytes,
        "pose_stream_sha256": source.pose_stream_sha256,
        "pose_words_sha256": source.pose_words_sha256,
        "pose_float32_sha256": source.pose_float32_sha256,
        "word_count": source.word_count,
        "decoded_by": source.decoded_by,
        "risk_flags": list(source.risk_flags),
    }


def load_anchor_eval(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    prov = payload.get("provenance", {}) if isinstance(payload.get("provenance"), Mapping) else {}
    archive_sha = prov.get("archive_sha256")
    out = {
        "path": str(path),
        "archive_bytes": int(payload["archive_size_bytes"]),
        "archive_sha256": archive_sha,
        "score_recomputed_from_components": float(
            payload.get("score_recomputed_from_components", payload.get("canonical_score"))
        ),
        "avg_posenet_dist": float(payload["avg_posenet_dist"]),
        "avg_segnet_dist": float(payload["avg_segnet_dist"]),
        "n_samples": int(payload["n_samples"]),
        "device": prov.get("device"),
        "gpu_model": prov.get("gpu_model"),
        "gpu_t4_match": prov.get("gpu_t4_match"),
        "score_claim": False,
    }
    if out["n_samples"] != EXPECTED_PAIRS:
        raise PublicPoseManifoldError(f"anchor eval sample count must be {EXPECTED_PAIRS}")
    return out


def load_trace(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text())
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise PublicPoseManifoldError(f"{path} missing samples list")
    out: dict[int, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping) or "pair_index" not in sample:
            continue
        pair = int(sample["pair_index"])
        pose = float(sample.get("score_pose_contribution_first_order", 0.0))
        seg = float(sample.get("score_seg_contribution_exact", 0.0))
        combined = float(sample.get("score_combined_contribution_first_order", pose + seg))
        out[pair] = {
            "pair_index": pair,
            "frame_indices": list(sample.get("frame_indices", [2 * pair, 2 * pair + 1])),
            "pose_score_contribution": pose,
            "seg_score_contribution": seg,
            "combined_score_contribution": combined,
        }
    if len(out) == 0:
        raise PublicPoseManifoldError(f"{path} yielded no component-trace samples")
    return out


def _moving_average(values: np.ndarray, radius: int = 3) -> np.ndarray:
    padded = np.pad(values.astype(np.float64), (radius, radius), mode="edge")
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / float(2 * radius + 1)
    return np.convolve(padded, kernel, mode="valid")


def _dct_lowpass(values: np.ndarray, keep: int = 12) -> np.ndarray:
    x = values.astype(np.float64)
    n = x.size
    if n == 0:
        return x
    keep = max(1, min(int(keep), n))
    rows = np.arange(n, dtype=np.float64) + 0.5
    coeffs = []
    for k in range(keep):
        basis = np.cos(math.pi * rows * k / n)
        scale = 1.0 / n if k == 0 else 2.0 / n
        coeffs.append(scale * float(np.dot(x, basis)))
    recon = np.zeros_like(x)
    for k, coeff in enumerate(coeffs):
        recon += coeff * np.cos(math.pi * rows * k / n)
    return recon


def _normalize_delta(delta: np.ndarray) -> np.ndarray:
    delta = delta.astype(np.float64)
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs <= 0.0:
        return delta
    return delta / max_abs


def _basis_rows(c102: PoseSource, references: Sequence[PoseSource]) -> list[dict[str, Any]]:
    if c102.words is None:
        raise PublicPoseManifoldError("C102 pose source is unavailable")
    source = c102.words.astype(np.float64)
    smooth = _moving_average(source, radius=3)
    dct = _dct_lowpass(source, keep=12)
    bases: list[dict[str, Any]] = [
        {
            "basis_id": "velocity_col0_neighbor_smooth",
            "basis_type": "velocity_col0_smooth",
            "delta_q": smooth - source,
            "basis_source_labels": ["C102"],
            "risk_flags": ["local_smoothness_proxy", "qp1_velocity_col0_only"],
            "alpha": 0.35,
            "top_pairs": 48,
            "max_abs_delta_q": 5,
            "trust": 0.030,
            "byte_overhead": 16,
        },
        {
            "basis_id": "dct_lowfreq_residual_pull",
            "basis_type": "fourier_dct_smooth",
            "delta_q": dct - source,
            "basis_source_labels": ["C102"],
            "risk_flags": ["dct_lowfreq_proxy", "can_oversmooth_hard_pose_events"],
            "alpha": 0.30,
            "top_pairs": 64,
            "max_abs_delta_q": 4,
            "trust": 0.024,
            "byte_overhead": 24,
        },
    ]

    public_deltas: list[np.ndarray] = []
    for ref in references:
        if not ref.available or ref.words is None:
            continue
        delta = ref.words.astype(np.float64) - source
        if not np.any(delta):
            bases.append(
                {
                    "basis_id": f"public_difference_{ref.label.lower()}_noop",
                    "basis_type": "public_difference",
                    "delta_q": delta,
                    "basis_source_labels": ["C102", ref.label],
                    "risk_flags": [*ref.risk_flags, "no_nonzero_delta"],
                    "alpha": 0.20,
                    "top_pairs": 0,
                    "max_abs_delta_q": 0,
                    "trust": 0.0,
                    "byte_overhead": 0,
                }
            )
            continue
        public_deltas.append(_normalize_delta(delta))
        bases.append(
            {
                "basis_id": f"public_difference_{ref.label.lower()}",
                "basis_type": "public_difference_sparse",
                "delta_q": delta,
                "basis_source_labels": ["C102", ref.label],
                "risk_flags": list(ref.risk_flags),
                "alpha": 0.20 if ref.label.upper() != "PR65" else 0.125,
                "top_pairs": 48 if ref.label.upper() != "PR65" else 40,
                "max_abs_delta_q": 6 if ref.label.upper() != "PR65" else 5,
                "trust": 0.032 if ref.label.upper() != "PR65" else 0.026,
                "byte_overhead": 32,
            }
        )
    if public_deltas:
        combined = np.zeros_like(source)
        for delta in public_deltas:
            combined += delta
        combined += 0.35 * _normalize_delta(smooth - source)
        bases.append(
            {
                "basis_id": "active_subspace_proxy_public_plus_smooth",
                "basis_type": "active_subspace_proxy",
                "delta_q": combined,
                "basis_source_labels": ["C102", *[r.label for r in references if r.available]],
                "risk_flags": [
                    "active_subspace_proxy_not_fitted_gradient",
                    "public_direct_transfer_negatives_do_not_promote",
                    "qp1_velocity_col0_only",
                ],
                "alpha": 4.0,
                "top_pairs": 96,
                "max_abs_delta_q": 8,
                "trust": 0.040,
                "byte_overhead": 48,
            }
        )
    return bases


def _ranked_pairs_for_basis(
    trace: Mapping[int, Mapping[str, Any]],
    delta_q: np.ndarray,
    *,
    max_abs_delta_q: int,
    alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    abs_delta = np.abs(delta_q.astype(np.float64))
    max_basis = float(abs_delta.max()) if abs_delta.size else 0.0
    for pair, raw in enumerate(delta_q):
        if raw == 0:
            continue
        delta = int(round(float(raw) * alpha))
        if delta == 0:
            delta = 1 if raw > 0 else -1
        delta = max(-max_abs_delta_q, min(max_abs_delta_q, delta))
        if delta == 0:
            continue
        sample = trace.get(pair, {})
        pose = float(sample.get("pose_score_contribution", 0.0))
        seg = float(sample.get("seg_score_contribution", 0.0))
        combined = float(sample.get("combined_score_contribution", pose + seg))
        magnitude_bonus = 0.0 if max_basis <= 0.0 else 0.12 * abs(float(raw)) / max_basis
        rank_score = combined + 0.35 * pose + magnitude_bonus * max(combined, 1e-12)
        rows.append(
            {
                "pair_index": pair,
                "frame_indices": sample.get("frame_indices", [2 * pair, 2 * pair + 1]),
                "raw_basis_delta_q": float(raw),
                "selected_delta_q": int(delta),
                "rank_score": float(rank_score),
                "pose_score_contribution": pose,
                "seg_score_contribution": seg,
                "combined_score_contribution": combined,
            }
        )
    rows.sort(key=lambda row: (-float(row["rank_score"]), int(row["pair_index"])))
    return rows


def _candidate_from_basis(
    *,
    c102: PoseSource,
    anchor_eval: Mapping[str, Any],
    trace: Mapping[int, Mapping[str, Any]],
    basis: Mapping[str, Any],
) -> dict[str, Any]:
    if c102.words is None:
        raise PublicPoseManifoldError("C102 source words are unavailable")
    selected_all = _ranked_pairs_for_basis(
        trace,
        np.asarray(basis["delta_q"], dtype=np.float64),
        max_abs_delta_q=int(basis["max_abs_delta_q"]),
        alpha=float(basis["alpha"]),
    )
    top_pairs = int(basis["top_pairs"])
    selected = selected_all[:top_pairs] if top_pairs > 0 else []
    changed_count = len(selected)
    abs_mean = (
        sum(abs(int(row["selected_delta_q"])) for row in selected) / changed_count
        if changed_count
        else 0.0
    )
    selected_trace_mass = sum(float(row["combined_score_contribution"]) for row in selected)
    pose_trace_mass = sum(float(row["pose_score_contribution"]) for row in selected)
    seg_trace_mass = sum(float(row["seg_score_contribution"]) for row in selected)
    magnitude_factor = min(1.0, abs_mean / max(1.0, float(basis["max_abs_delta_q"]))) if changed_count else 0.0
    expected_benefit = selected_trace_mass * float(basis["trust"]) * (0.5 + 0.5 * magnitude_factor)
    estimated_payload_bytes = int(basis["byte_overhead"]) + changed_count * 2
    estimated_rate_cost = estimated_payload_bytes * RATE_SCORE_PER_BYTE
    source_score = float(anchor_eval["score_recomputed_from_components"])
    break_even_required = max(0.0, source_score + estimated_rate_cost - TARGET_SCORE)
    risk_flags = sorted(
        {
            "planning_only",
            "exact_eval_readiness_false",
            "no_archive_built",
            *[str(flag) for flag in basis.get("risk_flags", [])],
        }
    )
    margin = expected_benefit - break_even_required
    return {
        "candidate_id": f"c102_qp1_{basis['basis_id']}_top{changed_count:03d}",
        "source_label": c102.label,
        "source_archive_sha256": c102.archive_sha256,
        "source_archive_bytes": c102.archive_bytes,
        "pose_stream_sha256": c102.pose_stream_sha256,
        "pose_stream_bytes": c102.pose_stream_bytes,
        "basis_id": basis["basis_id"],
        "basis_type": basis["basis_type"],
        "basis_source_labels": list(basis["basis_source_labels"]),
        "selected_pair_count": changed_count,
        "selected_pairs": [int(row["pair_index"]) for row in selected],
        "selected_coefs": [
            {
                "pair_index": int(row["pair_index"]),
                "delta_q": int(row["selected_delta_q"]),
                "delta_velocity": float(row["selected_delta_q"]) / VELOCITY_SCALE,
                "raw_basis_delta_q": float(row["raw_basis_delta_q"]),
            }
            for row in selected
        ],
        "selected_pair_records": selected,
        "estimated_payload_bytes": estimated_payload_bytes,
        "estimated_rate_score_cost": estimated_rate_cost,
        "selected_trace_mass": selected_trace_mass,
        "selected_pose_trace_mass": pose_trace_mass,
        "selected_seg_trace_mass": seg_trace_mass,
        "expected_benefit_proxy": expected_benefit,
        "break_even_score_gain_required": break_even_required,
        "proxy_margin_vs_target": margin,
        "risk_flags": risk_flags,
        "exact_eval_readiness": False,
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_criteria": [
            "convert this plan into a byte-closed archive that changes only the intended QP1 pose stream",
            "prove decoded mask, renderer, and action streams are preserved or intentionally changed with custody",
            "exact_eval_readiness must be recomputed by the builder and set true only after local closure gates",
            "claim a non-conflicting lane with tools/claim_lane_dispatch.py before any remote exact eval",
            "run archive.zip -> inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda",
        ],
    }


def _write_candidates_csv(path: Path, candidates: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "candidate_id",
        "source_archive_sha256",
        "source_archive_bytes",
        "pose_stream_sha256",
        "pose_stream_bytes",
        "basis_id",
        "selected_pairs",
        "selected_coefs",
        "expected_benefit_proxy",
        "break_even_score_gain_required",
        "proxy_margin_vs_target",
        "estimated_payload_bytes",
        "risk_flags",
        "exact_eval_readiness",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "rank": candidate.get("rank"),
                    "candidate_id": candidate["candidate_id"],
                    "source_archive_sha256": candidate["source_archive_sha256"],
                    "source_archive_bytes": candidate["source_archive_bytes"],
                    "pose_stream_sha256": candidate["pose_stream_sha256"],
                    "pose_stream_bytes": candidate["pose_stream_bytes"],
                    "basis_id": candidate["basis_id"],
                    "selected_pairs": json.dumps(candidate["selected_pairs"], sort_keys=True),
                    "selected_coefs": json.dumps(candidate["selected_coefs"], sort_keys=True),
                    "expected_benefit_proxy": candidate["expected_benefit_proxy"],
                    "break_even_score_gain_required": candidate["break_even_score_gain_required"],
                    "proxy_margin_vs_target": candidate["proxy_margin_vs_target"],
                    "estimated_payload_bytes": candidate["estimated_payload_bytes"],
                    "risk_flags": json.dumps(candidate["risk_flags"], sort_keys=True),
                    "exact_eval_readiness": candidate["exact_eval_readiness"],
                }
            )


def _ledger_text(plan: Mapping[str, Any]) -> str:
    top = plan["ranked_candidates"][:3]
    lines = [
        "# Public Pose Manifold Compare - 2026-05-03 Worker",
        "",
        "## Scope",
        "",
        "- Tool: `experiments/compare_public_pose_manifolds.py`.",
        "- Scope: local C102-native QP1 pose-manifold comparison and planning only.",
        "- Remote dispatch: `false`.",
        "- Score claim: `false`.",
        "- Archive built: `false`.",
        "- Exact-eval readiness: `false` for every emitted row.",
        "",
        "## Anchor",
        "",
        f"- C102 archive bytes: `{plan['anchor_eval']['archive_bytes']}`.",
        f"- C102 archive SHA-256: `{plan['anchor_eval']['archive_sha256']}`.",
        f"- C102 score: `{plan['anchor_eval']['score_recomputed_from_components']}`.",
        f"- C102 PoseNet: `{plan['anchor_eval']['avg_posenet_dist']}`.",
        f"- C102 SegNet: `{plan['anchor_eval']['avg_segnet_dist']}`.",
        "",
        "## Sources",
        "",
    ]
    for source in plan["pose_sources"]:
        lines.append(
            f"- `{source['label']}` available=`{source['available']}` "
            f"archive_bytes=`{source['archive_bytes']}` pose_bytes=`{source['pose_stream_bytes']}` "
            f"reason=`{source['reason']}`."
        )
    lines.extend(["", "## Top Planning Rows", ""])
    if not top:
        lines.append("- No nonzero candidate rows were emitted.")
    for candidate in top:
        lines.append(
            f"- `{candidate['candidate_id']}`: pairs=`{candidate['selected_pair_count']}`, "
            f"estimated_payload_bytes=`{candidate['estimated_payload_bytes']}`, "
            f"expected_benefit_proxy=`{candidate['expected_benefit_proxy']}`, "
            f"break_even_required=`{candidate['break_even_score_gain_required']}`, "
            f"margin=`{candidate['proxy_margin_vs_target']}`, readiness=`false`."
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Plan JSON: `{plan['artifacts']['plan_json']}`.",
            f"- Candidate CSV: `{plan['artifacts']['candidate_csv']}`.",
            f"- Pose sources JSON: `{plan['artifacts']['pose_sources_json']}`.",
            "",
            "## Dispatch Boundary",
            "",
            "Any future exact-eval candidate must be built by a separate byte-closed archive builder, "
            "must pass local stream-closure gates, and must claim the lane before dispatch. "
            "This artifact is not sufficient for a score, rank, promotion, or retirement claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_public_pose_manifold_compare(
    *,
    c102_archive: Path = DEFAULT_C102_ARCHIVE,
    c102_eval: Path = DEFAULT_C102_EVAL,
    c102_trace: Path = DEFAULT_C102_TRACE,
    pr75_archive: Path | None = DEFAULT_PR75_ARCHIVE,
    pr75_decoded_pose: Path | None = DEFAULT_PR75_DECODED_POSE,
    pr77_archive: Path | None = DEFAULT_PR77_ARCHIVE,
    pr77_decoded_pose: Path | None = DEFAULT_PR77_DECODED_POSE,
    pr65_archive: Path | None = DEFAULT_PR65_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ledger_md: Path | None = DEFAULT_LEDGER,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_eval = load_anchor_eval(c102_eval)
    trace = load_trace(c102_trace)
    c102 = load_pose_source(PoseSourceSpec("C102", c102_archive), expected_len=EXPECTED_PAIRS)
    if not c102.available:
        raise PublicPoseManifoldError(f"C102 pose source unavailable: {c102.reason}")
    if c102.archive_sha256 != anchor_eval["archive_sha256"]:
        raise PublicPoseManifoldError(
            f"C102 archive SHA mismatch: archive={c102.archive_sha256} eval={anchor_eval['archive_sha256']}"
        )
    references = [
        load_pose_source(PoseSourceSpec("PR75", pr75_archive, pr75_decoded_pose), expected_len=EXPECTED_PAIRS),
        load_pose_source(PoseSourceSpec("PR77", pr77_archive, pr77_decoded_pose), expected_len=EXPECTED_PAIRS),
        load_pose_source(PoseSourceSpec("PR65", pr65_archive, None, "pr65"), expected_len=EXPECTED_PAIRS),
    ]
    bases = _basis_rows(c102, references)
    candidates = [
        _candidate_from_basis(c102=c102, anchor_eval=anchor_eval, trace=trace, basis=basis)
        for basis in bases
        if int(basis.get("top_pairs", 0)) > 0
    ]
    candidates.sort(
        key=lambda item: (
            -float(item["proxy_margin_vs_target"]),
            -float(item["expected_benefit_proxy"]),
            int(item["estimated_payload_bytes"]),
            item["candidate_id"],
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    sources_json = output_dir / "pose_sources.json"
    plan_json = output_dir / "pose_manifold_compare_plan.json"
    candidate_csv = output_dir / "candidate_rankings.csv"
    plan = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_built": False,
        "remote_dispatch": {"dispatched": False, "dispatch_state_touched": False},
        "target_score": TARGET_SCORE,
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "anchor_eval": anchor_eval,
        "pose_sources": [_source_summary(c102), *[_source_summary(item) for item in references]],
        "basis_count": len(bases),
        "basis_summaries": [
            {
                "basis_id": basis["basis_id"],
                "basis_type": basis["basis_type"],
                "basis_source_labels": list(basis["basis_source_labels"]),
                "nonzero_delta_count": int(np.count_nonzero(np.asarray(basis["delta_q"]))),
                "max_abs_delta_q": float(np.max(np.abs(np.asarray(basis["delta_q"])))) if len(basis["delta_q"]) else 0.0,
                "risk_flags": list(basis.get("risk_flags", [])),
            }
            for basis in bases
        ],
        "ranked_candidates": candidates,
        "top_exact_candidate_designs": candidates[:3],
        "artifacts": {
            "plan_json": str(plan_json),
            "candidate_csv": str(candidate_csv),
            "pose_sources_json": str(sources_json),
            "ledger_md": str(ledger_md) if ledger_md is not None else None,
        },
        "determinism": {
            "sort_keys": [
                "proxy_margin_vs_target desc",
                "expected_benefit_proxy desc",
                "estimated_payload_bytes asc",
                "candidate_id asc",
            ],
            "wall_clock_fields": False,
        },
    }
    _write_json(sources_json, plan["pose_sources"])
    _write_candidates_csv(candidate_csv, candidates)
    _write_json(plan_json, plan)
    if ledger_md is not None:
        ledger_md.parent.mkdir(parents=True, exist_ok=True)
        ledger_md.write_text(_ledger_text(plan))
    return plan


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c102-archive", type=Path, default=DEFAULT_C102_ARCHIVE)
    parser.add_argument("--c102-eval", type=Path, default=DEFAULT_C102_EVAL)
    parser.add_argument("--c102-trace", type=Path, default=DEFAULT_C102_TRACE)
    parser.add_argument("--pr75-archive", type=Path, default=DEFAULT_PR75_ARCHIVE)
    parser.add_argument("--pr75-decoded-pose", type=Path, default=DEFAULT_PR75_DECODED_POSE)
    parser.add_argument("--pr77-archive", type=Path, default=DEFAULT_PR77_ARCHIVE)
    parser.add_argument("--pr77-decoded-pose", type=Path, default=DEFAULT_PR77_DECODED_POSE)
    parser.add_argument("--pr65-archive", type=Path, default=DEFAULT_PR65_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_public_pose_manifold_compare(
        c102_archive=args.c102_archive,
        c102_eval=args.c102_eval,
        c102_trace=args.c102_trace,
        pr75_archive=args.pr75_archive,
        pr75_decoded_pose=args.pr75_decoded_pose,
        pr77_archive=args.pr77_archive,
        pr77_decoded_pose=args.pr77_decoded_pose,
        pr65_archive=args.pr65_archive,
        output_dir=args.output_dir,
        ledger_md=None if args.no_ledger else args.ledger_md,
    )
    print(
        json.dumps(
            {
                "plan_json": plan["artifacts"]["plan_json"],
                "candidate_csv": plan["artifacts"]["candidate_csv"],
                "candidate_count": len(plan["ranked_candidates"]),
                "top_candidates": [
                    {
                        "rank": item["rank"],
                        "candidate_id": item["candidate_id"],
                        "expected_benefit_proxy": item["expected_benefit_proxy"],
                        "break_even_score_gain_required": item["break_even_score_gain_required"],
                        "exact_eval_readiness": item["exact_eval_readiness"],
                    }
                    for item in plan["ranked_candidates"][:3]
                ],
                "score_claim": plan["score_claim"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
