# SPDX-License-Identifier: MIT
"""Shared custody checks for contest exact-eval evidence.

This module is deliberately narrow: it validates archive/runtime identity,
axis-labelled metric fields, and official score-formula closure. It does not
decide whether a substrate should be promoted, ranked, or dispatched; callers
keep their substrate-specific blocker labels and policy gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTEST_REFERENCE_BYTES = 37_545_489
CONTEST_EXACT_SAMPLE_COUNT = 600
SCORE_FORMULA_TOLERANCE = 1e-9
CONTEST_EXACT_AXES = frozenset({"contest_cpu", "contest_cuda"})
CUDA_DEVICE_TOKENS = frozenset({"a10", "a100", "cuda", "gpu", "h100", "l4", "t4"})
CPU_DEVICE_TOKENS = frozenset({"cpu", "x86", "x86_64"})
CONTEST_CPU_REQUIRED_OS_TOKENS = frozenset({"debian", "linux", "ubuntu"})
CONTEST_CPU_FORBIDDEN_TOKENS = frozenset({
    "aarch64",
    "ane",
    "apple",
    "arm",
    "arm64",
    "darwin",
    "m1",
    "m2",
    "m3",
    "m4",
    "mac",
    "macbook",
    "macbookpro",
    "macos",
    "metal",
    "mlx",
    "mps",
    "osx",
})
NEGATED_DEVICE_TOKENS = frozenset({
    "disabled",
    "false",
    "no",
    "non",
    "not",
    "off",
    "unavailable",
    "without",
})

_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class ExactEvalEvidenceValidation:
    """Structured validation result for one exact-eval evidence row."""

    blockers: tuple[str, ...]
    annotations: tuple[str, ...] = ()
    archive_sha256: str = ""
    runtime_tree_sha256: str = ""
    score: float | None = None
    seg_dist: float | None = None
    pose_dist: float | None = None
    archive_bytes: int | None = None
    n_samples: int | None = None


def is_sha256_hex(value: object) -> bool:
    """Return true only for literal 64-hex SHA-256 strings."""

    return isinstance(value, str) and bool(_SHA256_HEX_RE.fullmatch(value.strip()))


def normalize_sha256(value: object) -> str:
    """Return lowercase SHA-256 text, or an empty string for invalid input."""

    if not is_sha256_hex(value):
        return ""
    return str(value).strip().lower()


def finite_float(value: object) -> float | None:
    """Coerce numeric JSON values to finite float, excluding booleans."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        out = float(value)
        return out if math.isfinite(out) else None
    return None


def positive_int(value: object) -> int | None:
    """Coerce positive JSON integer values, excluding booleans and strings."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _clean_text(value: object) -> str:
    """Return stripped string text for JSON scalar fields."""

    return str(value or "").strip()


def _contains_token(value: str, tokens: frozenset[str]) -> bool:
    """Return true only for non-negated device/hardware capability tokens."""

    parts = re.findall(r"[a-z0-9]+", value.lower())
    for idx, part in enumerate(parts):
        if part not in tokens:
            continue
        prev_token = parts[idx - 1] if idx > 0 else ""
        next_token = parts[idx + 1] if idx + 1 < len(parts) else ""
        if prev_token in NEGATED_DEVICE_TOKENS or next_token in NEGATED_DEVICE_TOKENS:
            continue
        return True
    return False


def _contains_any_token(value: str, tokens: frozenset[str]) -> bool:
    parts = set(re.findall(r"[a-z0-9]+", value.lower()))
    return bool(parts & tokens)


def contains_non_negated_device_token(value: str, tokens: frozenset[str]) -> bool:
    """Return true when ``value`` contains a non-negated hardware/device token."""

    return _contains_token(value, tokens)


def contains_forbidden_contest_cpu_token(value: str) -> bool:
    """Return true when CPU-axis evidence leaks macOS/MPS or CUDA/GPU semantics."""

    return _contains_any_token(value, CONTEST_CPU_FORBIDDEN_TOKENS) or _contains_token(
        value,
        CUDA_DEVICE_TOKENS,
    )


def is_contest_cpu_device_text(value: str) -> bool:
    """Return true for Linux/x86 CPU axis text with no CUDA/GPU/macOS leakage."""

    return (
        _contains_token(value, CPU_DEVICE_TOKENS)
        and _contains_any_token(value, CONTEST_CPU_REQUIRED_OS_TOKENS)
        and not contains_forbidden_contest_cpu_token(value)
    )


def is_contest_cuda_device_text(value: str) -> bool:
    """Return true for CUDA/GPU axis text, respecting negated tokens."""

    return _contains_token(value, CUDA_DEVICE_TOKENS)


def is_contest_cuda_inflate_device_text(value: str) -> bool:
    """Return true for CUDA-axis inflate policy text.

    Modal CUDA auth-eval uses ``--inflate-device auto`` as the canonical
    contest-CUDA path when the runtime may select the compliant CUDA inflate
    backend. Keep this narrower than generic device text: ``auto`` is accepted
    for CUDA inflate policy only, never as CPU evidence or eval-device evidence.
    """

    text = _clean_text(value).lower()
    return text == "auto" or is_contest_cuda_device_text(text)


def _auth_eval_command_has_expected_shape(command: str, semantic_axis: str) -> bool:
    """Return true for recognizable contest auth-eval/evaluate invocations."""

    lowered = command.lower()
    parts = re.findall(r"[a-z0-9_./-]+", lowered)
    has_entrypoint = any(
        token.endswith("contest_auth_eval")
        or token.endswith("contest_auth_eval.py")
        or token.endswith("evaluate.py")
        for token in parts
    )
    if not has_entrypoint:
        return False
    if semantic_axis == "contest_cuda":
        return is_contest_cuda_device_text(lowered)
    if semantic_axis == "contest_cpu":
        return _contains_token(lowered, CPU_DEVICE_TOKENS)
    return True


def _is_transient_path_text(path_text: str) -> bool:
    return (
        path_text.startswith("/tmp/")
        or path_text.startswith("/var/tmp/")
        or path_text.startswith("/private/tmp/")
    )


def _resolve_evidence_path(value: object, base_dir: Path) -> tuple[Path | None, str | None]:
    """Resolve an evidence path relative to ``base_dir`` for custody checks."""

    path_text = _clean_text(value).removeprefix("file:").strip()
    if not path_text:
        return None, "missing"
    if _is_transient_path_text(path_text):
        return None, "transient"
    resolved_base = base_dir.resolve()
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = resolved_base / path
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_base)
    except ValueError:
        return resolved_path, "outside_base_dir"
    return resolved_path, None


def _json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_aggregate_sha256(path: Path) -> str:
    payload = _json_mapping(path)
    if payload is None:
        return ""
    direct = normalize_sha256(payload.get("aggregate_sha256"))
    if direct:
        return direct
    nested = payload.get("payload")
    if isinstance(nested, Mapping):
        return normalize_sha256(nested.get("aggregate_sha256"))
    return ""


def contest_score(seg_dist: float, pose_dist: float, archive_bytes: int) -> float:
    """Compute the official contest score formula."""

    return (
        100.0 * float(seg_dist)
        + math.sqrt(10.0 * float(pose_dist))
        + 25.0 * int(archive_bytes) / CONTEST_REFERENCE_BYTES
    )


def score_formula_matches(
    *,
    score: float,
    seg_dist: float,
    pose_dist: float,
    archive_bytes: int,
    tolerance: float = SCORE_FORMULA_TOLERANCE,
) -> bool:
    """Return true when ``score`` matches the official formula."""

    return abs(
        float(score)
        - contest_score(
            seg_dist=float(seg_dist),
            pose_dist=float(pose_dist),
            archive_bytes=int(archive_bytes),
        )
    ) <= tolerance


def extract_archive_sha256(mapping: Mapping[str, Any]) -> str:
    """Extract the first valid archive SHA from common candidate/eval fields."""

    for key in (
        "candidate_archive_sha256",
        "archive_sha256",
        "expected_archive_sha256",
        "exact_archive_sha256",
        "archive_sha",
    ):
        value = mapping.get(key)
        if is_sha256_hex(value):
            return str(value).strip().lower()
    return ""


def extract_runtime_tree_sha256(mapping: Mapping[str, Any]) -> str:
    """Extract runtime-tree SHA from top-level or nested custody manifests."""

    for key in (
        "candidate_runtime_tree_sha256",
        "runtime_tree_sha256",
        "expected_runtime_tree_sha256",
        "inflate_runtime_tree_sha256",
    ):
        value = mapping.get(key)
        if is_sha256_hex(value):
            return str(value).strip().lower()
    for outer_key in (
        "runtime_manifest",
        "inflate_runtime_manifest",
        "runtime_custody",
        "provenance",
    ):
        nested = mapping.get(outer_key)
        if not isinstance(nested, Mapping):
            continue
        value = nested.get("runtime_tree_sha256")
        if is_sha256_hex(value):
            return str(value).strip().lower()
        inflate_manifest = nested.get("inflate_runtime_manifest")
        if isinstance(inflate_manifest, Mapping):
            value = inflate_manifest.get("runtime_tree_sha256")
            if is_sha256_hex(value):
                return str(value).strip().lower()
    return ""


def validate_exact_eval_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_axis: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_runtime_tree_sha256: str | None = None,
    require_artifact_path: bool = False,
    require_hardware: bool = True,
    require_auth_eval_command: bool = True,
    require_log_path: bool = True,
    require_devices: bool = False,
    require_inflated_outputs_manifest: bool = False,
    require_raw_output_aggregate_sha256: bool = False,
    annotation_prefix: str = "",
    artifact_base_dir: Path | str | None = None,
) -> ExactEvalEvidenceValidation:
    """Validate one axis-labelled exact-eval evidence row.

    Blocker names are generic suffixes such as ``score_missing`` or
    ``score_formula_mismatch``. Callers translate them into their public
    blocker vocabulary so existing ledgers and tests remain stable.
    """

    blockers: list[str] = []
    annotations: list[str] = []
    prefix = f"{annotation_prefix}_" if annotation_prefix else ""

    axis = _clean_text(evidence.get("axis"))
    if expected_axis is not None:
        if not axis:
            blockers.append("axis_missing")
            annotations.append(f"{prefix}axis_missing")
        elif axis != expected_axis:
            blockers.append("axis_mismatch")
            annotations.append(f"{prefix}axis_mismatch:{axis!r}!={expected_axis!r}")

    archive_sha = normalize_sha256(evidence.get("archive_sha256"))
    runtime_sha = normalize_sha256(evidence.get("runtime_tree_sha256"))
    expected_archive_sha = normalize_sha256(expected_archive_sha256)
    expected_runtime_sha = normalize_sha256(expected_runtime_tree_sha256)

    if not archive_sha:
        blockers.append("archive_sha_invalid")
    elif expected_archive_sha and archive_sha != expected_archive_sha:
        blockers.append("archive_sha_mismatch")
    if not runtime_sha:
        blockers.append("runtime_tree_sha_invalid")
    elif expected_runtime_sha and runtime_sha != expected_runtime_sha:
        blockers.append("runtime_tree_sha_mismatch")

    n_samples = positive_int(evidence.get("n_samples"))
    archive_bytes = positive_int(evidence.get("archive_bytes"))
    seg_dist = finite_float(evidence.get("seg_dist"))
    pose_dist = finite_float(evidence.get("pose_dist"))
    score = finite_float(evidence.get("score"))

    if n_samples is None:
        blockers.append("n_samples_missing")
    elif (expected_axis or axis) in CONTEST_EXACT_AXES and n_samples != CONTEST_EXACT_SAMPLE_COUNT:
        blockers.append("n_samples_not_contest_exact")
    if archive_bytes is None:
        blockers.append("archive_bytes_missing")
    if seg_dist is None or seg_dist < 0.0:
        blockers.append("seg_dist_missing")
    if pose_dist is None or pose_dist < 0.0:
        blockers.append("pose_dist_missing")
    if score is None:
        blockers.append("score_missing")

    semantic_axis = expected_axis or axis
    hardware = _clean_text(evidence.get("hardware"))
    inflate_device = _clean_text(evidence.get("inflate_device"))
    eval_device = _clean_text(evidence.get("eval_device"))

    if require_hardware and not hardware:
        blockers.append("hardware_missing")
    elif semantic_axis == "contest_cuda" and hardware and not is_contest_cuda_device_text(hardware):
        blockers.append("hardware_not_cuda")
    elif semantic_axis == "contest_cpu" and hardware and not is_contest_cpu_device_text(hardware):
        blockers.append("hardware_not_contest_cpu")
    if require_devices:
        if not inflate_device:
            blockers.append("inflate_device_missing")
        elif semantic_axis == "contest_cuda" and not is_contest_cuda_inflate_device_text(
            inflate_device
        ):
            blockers.append("inflate_device_not_cuda")
        elif semantic_axis == "contest_cpu":
            if not _contains_token(inflate_device, CPU_DEVICE_TOKENS):
                blockers.append("inflate_device_not_cpu")
            elif contains_forbidden_contest_cpu_token(inflate_device):
                blockers.append("inflate_device_not_contest_cpu")
        if not eval_device:
            blockers.append("eval_device_missing")
        elif semantic_axis == "contest_cuda" and not is_contest_cuda_device_text(eval_device):
            blockers.append("eval_device_not_cuda")
        elif semantic_axis == "contest_cpu":
            if not _contains_token(eval_device, CPU_DEVICE_TOKENS):
                blockers.append("eval_device_not_cpu")
            elif contains_forbidden_contest_cpu_token(eval_device):
                blockers.append("eval_device_not_contest_cpu")
    if require_auth_eval_command:
        auth_eval_command = _clean_text(evidence.get("auth_eval_command"))
        if not auth_eval_command:
            blockers.append("auth_eval_command_missing")
        elif semantic_axis in CONTEST_EXACT_AXES and not _auth_eval_command_has_expected_shape(
            auth_eval_command,
            semantic_axis,
        ):
            blockers.append("auth_eval_command_unrecognized")
        elif semantic_axis == "contest_cpu" and contains_forbidden_contest_cpu_token(
            auth_eval_command
        ):
            blockers.append("auth_eval_command_not_contest_cpu")
    if require_log_path and not _clean_text(evidence.get("log_path")):
        blockers.append("log_path_missing")
    if require_artifact_path and not _clean_text(evidence.get("artifact_path")):
        blockers.append("artifact_path_missing")

    raw_output_aggregate_sha = normalize_sha256(
        evidence.get("raw_output_aggregate_sha256")
        or evidence.get("inflated_output_aggregate_sha256")
    )
    inflated_manifest_sha = normalize_sha256(
        evidence.get("inflated_outputs_manifest_sha256")
        or evidence.get("inflated_output_manifest_sha256")
    )
    inflated_manifest_path_text = _clean_text(
        evidence.get("inflated_outputs_manifest_path")
        or evidence.get("inflated_output_manifest_path")
    )
    if require_raw_output_aggregate_sha256 and not raw_output_aggregate_sha:
        blockers.append("raw_output_aggregate_sha_invalid")
    if require_inflated_outputs_manifest:
        if not inflated_manifest_path_text:
            blockers.append("inflated_outputs_manifest_path_missing")
        if not inflated_manifest_sha:
            blockers.append("inflated_outputs_manifest_sha_invalid")

    base_dir = Path(artifact_base_dir) if artifact_base_dir is not None else None
    if base_dir is not None:
        if require_log_path and "log_path_missing" not in blockers:
            log_path, log_path_error = _resolve_evidence_path(
                evidence.get("log_path"),
                base_dir,
            )
            if log_path_error == "transient":
                blockers.append("log_path_transient")
            elif log_path_error == "outside_base_dir":
                blockers.append("log_path_outside_base_dir")
            elif log_path is None or not log_path.is_file():
                blockers.append("log_path_file_missing")
        if require_artifact_path and "artifact_path_missing" not in blockers:
            artifact_path, artifact_path_error = _resolve_evidence_path(
                evidence.get("artifact_path"),
                base_dir,
            )
            if artifact_path_error == "transient":
                blockers.append("artifact_path_transient")
            elif artifact_path_error == "outside_base_dir":
                blockers.append("artifact_path_outside_base_dir")
            elif artifact_path is None or not artifact_path.is_file():
                blockers.append("artifact_path_file_missing")
        if (
            require_inflated_outputs_manifest
            and "inflated_outputs_manifest_path_missing" not in blockers
        ):
            manifest_path, manifest_path_error = _resolve_evidence_path(
                inflated_manifest_path_text,
                base_dir,
            )
            if manifest_path_error == "transient":
                blockers.append("inflated_outputs_manifest_path_transient")
            elif manifest_path_error == "outside_base_dir":
                blockers.append("inflated_outputs_manifest_path_outside_base_dir")
            elif manifest_path is None or not manifest_path.is_file():
                blockers.append("inflated_outputs_manifest_file_missing")
            else:
                if inflated_manifest_sha and _sha256_file(manifest_path) != inflated_manifest_sha:
                    blockers.append("inflated_outputs_manifest_sha_mismatch")
                manifest_aggregate_sha = _manifest_aggregate_sha256(manifest_path)
                if not manifest_aggregate_sha:
                    blockers.append("inflated_outputs_manifest_aggregate_missing")
                elif (
                    raw_output_aggregate_sha
                    and manifest_aggregate_sha != raw_output_aggregate_sha
                ):
                    blockers.append("inflated_outputs_manifest_aggregate_mismatch")

    if (
        score is not None
        and seg_dist is not None
        and pose_dist is not None
        and archive_bytes is not None
    ):
        recomputed = contest_score(seg_dist, pose_dist, archive_bytes)
        if abs(score - recomputed) > SCORE_FORMULA_TOLERANCE:
            blockers.append("score_formula_mismatch")
            annotations.append(f"{prefix}score_formula_mismatch:{score:.12g}!={recomputed:.12g}")

    return ExactEvalEvidenceValidation(
        blockers=tuple(dict.fromkeys(blockers)),
        annotations=tuple(annotations),
        archive_sha256=archive_sha,
        runtime_tree_sha256=runtime_sha,
        score=score,
        seg_dist=seg_dist,
        pose_dist=pose_dist,
        archive_bytes=archive_bytes,
        n_samples=n_samples,
    )


__all__ = [
    "CONTEST_EXACT_SAMPLE_COUNT",
    "CONTEST_REFERENCE_BYTES",
    "SCORE_FORMULA_TOLERANCE",
    "ExactEvalEvidenceValidation",
    "contest_score",
    "extract_archive_sha256",
    "extract_runtime_tree_sha256",
    "finite_float",
    "is_sha256_hex",
    "normalize_sha256",
    "positive_int",
    "score_formula_matches",
    "validate_exact_eval_evidence",
]
