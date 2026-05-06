"""Readiness guards for engineered correction atoms.

Engineered corrections are score-affecting bytes. This module validates the
packed sparse-correction contract before a correction artifact can be treated
as locally actionable. It never runs scorers, never dispatches GPU work, and
never makes a score claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.precompute_gradient_corrections import (
    pack_sparse_corrections,
    unpack_sparse_corrections,
)

SUPPORTED_QBITS = (4, 8, 16)
WIRE_CHANNELS = 3


class EngineeredCorrectionReadinessError(ValueError):
    """Raised when a correction artifact is unsafe or not locally actionable."""


@dataclasses.dataclass(frozen=True)
class CorrectionReadinessReport:
    """Deterministic readiness report for one correction artifact."""

    ready_for_local_patch: bool
    ready_for_exact_eval_dispatch: bool
    score_claim: bool
    dispatch_attempted: bool
    packed_bytes: int
    packed_sha256: str
    n_kept: int
    n_total: int
    quantize_bits: int
    shape: tuple[int, int, int, int]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockers": list(self.blockers),
            "dispatch_attempted": self.dispatch_attempted,
            "n_kept": self.n_kept,
            "n_total": self.n_total,
            "packed_bytes": self.packed_bytes,
            "packed_sha256": self.packed_sha256,
            "quantize_bits": self.quantize_bits,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "ready_for_local_patch": self.ready_for_local_patch,
            "score_claim": self.score_claim,
            "shape": list(self.shape),
            "warnings": list(self.warnings),
        }


def audit_sparse_corrections(
    sparse_data: dict[str, Any],
    *,
    max_packed_bytes: int,
    compression: str = "zlib",
    require_nonzero: bool = True,
    manifest: dict[str, Any] | None = None,
) -> CorrectionReadinessReport:
    """Validate sparse correction data and its packed byte contract."""
    blockers: list[str] = []
    warnings: list[str] = []
    manifest = {} if manifest is None else dict(manifest)
    score_claim = bool(manifest.get("score_claim", False))
    dispatch_attempted = bool(manifest.get("dispatch_attempted", False))
    if score_claim:
        blockers.append("manifest_score_claim_true")
    if dispatch_attempted:
        blockers.append("manifest_dispatch_attempted_true")
    if max_packed_bytes <= 0:
        blockers.append("max_packed_bytes_must_be_positive")

    shape = _shape_tuple(sparse_data.get("shape"), blockers)
    qbits = _int_field(sparse_data, "quantize_bits", blockers)
    n_total = _int_field(sparse_data, "n_total", blockers)
    n_kept = _int_field(sparse_data, "n_kept", blockers)
    scale = _float_field(sparse_data, "scale", blockers)
    if qbits not in SUPPORTED_QBITS:
        blockers.append(f"unsupported_quantize_bits_{qbits}")
    if scale is not None and (not math.isfinite(scale) or scale <= 0.0):
        blockers.append("scale_must_be_finite_positive")
    if shape is not None:
        expected_total = int(np.prod(shape[:3]))
        if n_total is not None and n_total != expected_total:
            blockers.append(
                f"n_total_mismatch_expected_{expected_total}_got_{n_total}"
            )
        if shape[-1] != WIRE_CHANNELS:
            blockers.append(
                f"wire_format_requires_{WIRE_CHANNELS}_channels_got_{shape[-1]}"
            )

    indices = np.asarray(sparse_data.get("indices", []))
    values = np.asarray(sparse_data.get("values", []))
    if n_kept is not None:
        if indices.shape != (n_kept,):
            blockers.append(f"indices_shape_mismatch_{indices.shape}_expected_{(n_kept,)}")
        if values.shape != (n_kept, WIRE_CHANNELS):
            blockers.append(
                f"values_shape_mismatch_{values.shape}_expected_{(n_kept, WIRE_CHANNELS)}"
            )
    if n_kept == 0 and require_nonzero:
        blockers.append("no_corrections_selected")
    if indices.size:
        if np.any(indices < 0):
            blockers.append("negative_indices")
        if n_total is not None and np.any(indices >= n_total):
            blockers.append("indices_out_of_bounds")
        if len(np.unique(indices)) != len(indices):
            blockers.append("duplicate_indices")
    if values.size:
        if qbits in (4, 8) and values.dtype != np.int8:
            blockers.append(f"int_corrections_must_be_int8_got_{values.dtype}")
        if qbits == 4 and np.any(np.abs(values.astype(np.int16)) > 7):
            blockers.append("int4_corrections_out_of_range")
        if qbits == 16 and values.dtype != np.float16:
            blockers.append(f"fp16_corrections_must_be_float16_got_{values.dtype}")
        if not np.all(np.isfinite(values.astype(np.float32))):
            blockers.append("correction_values_must_be_finite")
        if require_nonzero and not np.any(values):
            blockers.append("all_correction_values_zero")

    packed = b""
    packed_sha = ""
    pack_blockers = tuple(
        blocker
        for blocker in blockers
        if blocker not in {"manifest_score_claim_true", "manifest_dispatch_attempted_true"}
        and blocker != "max_packed_bytes_must_be_positive"
    )
    if not pack_blockers:
        try:
            packed = pack_sparse_corrections(sparse_data, compression=compression)
            roundtrip = unpack_sparse_corrections(
                packed,
                compressed=(compression == "zlib"),
            )
            _assert_roundtrip_equal(sparse_data, roundtrip)
        except Exception as exc:
            blockers.append(f"pack_roundtrip_failed:{type(exc).__name__}:{exc}")
    if packed:
        packed_sha = hashlib.sha256(packed).hexdigest()
        if len(packed) > max_packed_bytes:
            blockers.append(
                f"packed_bytes_exceed_cap_{len(packed)}_gt_{max_packed_bytes}"
            )
    else:
        packed_sha = hashlib.sha256(b"").hexdigest()

    if qbits in (4, 8) and scale is not None and scale > 255.0:
        warnings.append("large_scale_may_clip_pixels_after_apply")

    return CorrectionReadinessReport(
        ready_for_local_patch=not blockers,
        ready_for_exact_eval_dispatch=False,
        score_claim=False,
        dispatch_attempted=False,
        packed_bytes=len(packed),
        packed_sha256=packed_sha,
        n_kept=int(n_kept or 0),
        n_total=int(n_total or 0),
        quantize_bits=int(qbits or 0),
        shape=shape or (0, 0, 0, 0),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def audit_corrections_bin(
    path: str | Path,
    *,
    max_packed_bytes: int,
    compressed: bool = True,
    manifest_path: str | Path | None = None,
) -> CorrectionReadinessReport:
    """Validate an existing packed ``gradient_corrections.bin`` artifact."""
    artifact = Path(path)
    payload = artifact.read_bytes()
    sparse = unpack_sparse_corrections(payload, compressed=compressed)
    manifest = None
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    report = audit_sparse_corrections(
        sparse,
        max_packed_bytes=max_packed_bytes,
        compression="zlib" if compressed else "none",
        manifest=manifest,
    )
    if report.packed_sha256 != hashlib.sha256(payload).hexdigest():
        blockers = (*report.blockers, "packed_payload_not_canonical")
        return dataclasses.replace(
            report,
            ready_for_local_patch=False,
            packed_bytes=len(payload),
            packed_sha256=hashlib.sha256(payload).hexdigest(),
            blockers=blockers,
        )
    return report


def detector_cost_atom_from_correction_report(
    report: CorrectionReadinessReport,
    *,
    atom_id: str,
    detector_capacity: float | None = None,
    positive_scorer_sensitivity: float | None = None,
    evidence_grade: str = "planning",
) -> dict[str, Any]:
    """Convert a correction readiness report into a detector-cost atom row.

    The returned mapping is intended for
    ``tac.uniward_delta.build_detector_cost_manifest``. It carries only charged
    byte and optional optimizer-feedback fields; it does not make a score or
    dispatch claim.
    """
    if not atom_id:
        raise EngineeredCorrectionReadinessError("atom_id must be nonempty")
    atom = {
        "atom_id": str(atom_id),
        "atom_kind": "engineered_sparse_correction",
        "stream_role": "sidecar_or_correction_stream",
        "charged_bytes": int(report.packed_bytes),
        "n_kept": int(report.n_kept),
        "quantize_bits": int(report.quantize_bits),
        "packed_sha256": report.packed_sha256,
        "evidence_grade": evidence_grade,
        "score_claim": False,
        "dispatch_attempted": False,
    }
    if detector_capacity is not None:
        atom["detector_capacity"] = float(detector_capacity)
    if positive_scorer_sensitivity is not None:
        atom["positive_scorer_sensitivity"] = float(positive_scorer_sensitivity)
    return atom


def _shape_tuple(value: Any, blockers: list[str]) -> tuple[int, int, int, int] | None:
    try:
        shape = tuple(int(part) for part in value)
    except Exception:
        blockers.append("shape_must_be_4d_integer_sequence")
        return None
    if len(shape) != 4 or any(part <= 0 for part in shape):
        blockers.append(f"shape_must_be_positive_4d_got_{shape}")
        return None
    return shape  # type: ignore[return-value]


def _int_field(payload: dict[str, Any], key: str, blockers: list[str]) -> int | None:
    try:
        value = int(payload[key])
    except Exception:
        blockers.append(f"{key}_must_be_integer")
        return None
    if value < 0:
        blockers.append(f"{key}_must_be_nonnegative")
    return value


def _float_field(payload: dict[str, Any], key: str, blockers: list[str]) -> float | None:
    try:
        return float(payload[key])
    except Exception:
        blockers.append(f"{key}_must_be_float")
        return None


def _assert_roundtrip_equal(source: dict[str, Any], roundtrip: dict[str, Any]) -> None:
    if tuple(source["shape"]) != tuple(roundtrip["shape"]):
        raise EngineeredCorrectionReadinessError("roundtrip shape mismatch")
    for key in ("quantize_bits", "n_kept", "n_total"):
        if int(source[key]) != int(roundtrip[key]):
            raise EngineeredCorrectionReadinessError(f"roundtrip {key} mismatch")
    if not np.array_equal(np.asarray(source["indices"]), np.asarray(roundtrip["indices"])):
        raise EngineeredCorrectionReadinessError("roundtrip indices mismatch")
    if not np.array_equal(np.asarray(source["values"]), np.asarray(roundtrip["values"])):
        raise EngineeredCorrectionReadinessError("roundtrip values mismatch")


__all__ = [
    "SUPPORTED_QBITS",
    "WIRE_CHANNELS",
    "CorrectionReadinessReport",
    "EngineeredCorrectionReadinessError",
    "audit_corrections_bin",
    "audit_sparse_corrections",
    "detector_cost_atom_from_correction_report",
]
