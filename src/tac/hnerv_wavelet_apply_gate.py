# SPDX-License-Identifier: MIT
"""Readiness gate for score-relevant WR01 wavelet apply transforms.

WR01 sidechannels are now byte-closed and runtime-consumed, but the current
runtime mode is explicit no-op. This module computes the exact contest-rate
penalty and the minimum component improvement required before a future WR01
apply transform can be considered for archive preflight or exact CUDA eval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.repo_io import read_json

SCHEMA_VERSION = 1
AUDIT_NAME = "hnerv_wavelet_apply_gate"
CONTEST_ORIGINAL_BYTES = 37_545_489


class HnervWaveletApplyGateError(ValueError):
    """Raised when WR01 apply-gate inputs are invalid."""


def build_wavelet_apply_gate(
    *,
    sidechannel_manifest: Mapping[str, Any],
    stacked_metadata: Mapping[str, Any] | None = None,
    baseline_pose_dist: float | None = None,
    required_component_margin: float = 0.0,
) -> dict[str, Any]:
    """Return a fail-closed readiness report for a WR01 apply transform.

    Round 4 R4-B note (2026-05-06, 80% confidence): this gate is
    **fail-closed by design**. Four blockers in `_dispatch_blockers`
    (`requires_reviewed_wr01_apply_transform`,
    `requires_component_benefit_evidence_over_break_even`,
    `requires_archive_manifest_preflight`, `requires_exact_cuda_auth_eval`)
    are unconditionally appended on every call. `ready_for_exact_eval_dispatch`
    and `ready_for_archive_preflight` are therefore always `False`. This is
    INTENTIONAL — the gate is a documented "next required evidence" report,
    not a pass/fail oracle. Operators must REMOVE blockers from the report
    by hand only after the corresponding evidence has been produced (a
    council Round-N CLEAN review for the apply_transform; a contest-CUDA
    score that exceeds break-even; a fresh archive manifest preflight; a
    CUDA auth-eval JSON pinned to the candidate archive bytes).

    Code that consumes this report should NOT use `not gate["dispatch_blockers"]`
    as a clearance check — the list is never empty in this module's emitted
    output. Use the `next_required_evidence` field as a checklist and
    confirm each item via the corresponding evidence artifact before
    dispatch. The gate's value is in the cross-cut of byte-delta gating
    + component-margin + WR01-specific runtime mode + manifest schema
    enforcement, not in producing a green-light boolean.

    ``required_component_margin`` is an additional score-space safety margin
    beyond the byte-rate break-even. A positive value makes the gate stricter.
    """

    if required_component_margin < 0:
        raise HnervWaveletApplyGateError("required_component_margin must be non-negative")
    stacked = dict(stacked_metadata or {})
    # Round 2 R2-2 fix (2026-05-06, 88% confidence): apply_transform manifest
    # uses the renamed `candidate_archive_byte_delta_vs_source_estimate` key.
    # Old sidechannel manifests use `candidate_archive_byte_delta`. Read both.
    #
    # Round 3 R3-A fix (2026-05-06, 92% regression): the `or` idiom is unsafe
    # for integers because zero is falsy. A real `candidate_archive_byte_delta = 0`
    # (valid: wavelet transform produces same-size archive) was silently
    # dropped to the fallback key. Use explicit `is None` so the primary key's
    # zero is preserved.
    _byte_delta_raw = sidechannel_manifest.get("candidate_archive_byte_delta")
    if _byte_delta_raw is None:
        _byte_delta_raw = sidechannel_manifest.get("candidate_archive_byte_delta_vs_source_estimate")
    # Round 5 R5-2 fix (2026-05-06, 82%): _safe_int returns None for
    # non-integer floats (e.g. JSON-rounded 3.14). The downstream "missing
    # archive byte delta" error message would then be misleading — the key
    # IS present, just with the wrong type. Detect explicitly and raise a
    # clear diagnostic instead of letting the generic "missing" path fire.
    if isinstance(_byte_delta_raw, float) and not _byte_delta_raw.is_integer():
        raise HnervWaveletApplyGateError(
            f"candidate_archive_byte_delta is non-integer float: {_byte_delta_raw}"
        )
    candidate_delta = _safe_int(_byte_delta_raw)
    stacked_delta = _safe_int(stacked.get("delta_bytes_vs_pr106_zip"))
    archive_byte_delta = stacked_delta if stacked_delta is not None else candidate_delta
    if archive_byte_delta is None:
        raise HnervWaveletApplyGateError("missing archive byte delta in WR01 inputs")

    sidechannel_bytes = _safe_int(sidechannel_manifest.get("wavelet_sidechannel_bytes"))
    decoded = sidechannel_manifest.get("decoded_wavelet_sidechannel")
    decoded_atoms = _decoded_atom_count(decoded)
    # Round 3 R3-B fix (2026-05-06, 85% confidence): use explicit `is None`
    # guards instead of `or` for both runtime_proof and runtime_mode. Same
    # antipattern class as R2-2: `or` discards falsy-but-valid values
    # (empty mapping, empty string). For runtime_proof, an empty dict {} is
    # falsy but is a real proof object; for runtime_mode, future valid modes
    # could be falsy strings.
    runtime_proof = stacked.get("wavelet_runtime_consumption_proof")
    if runtime_proof is None:
        runtime_proof = sidechannel_manifest.get("runtime_consumption_proof")
    runtime_mode = stacked.get("wavelet_runtime_mode")
    if runtime_mode is None:
        runtime_mode = "candidate_sidechannel_only"
    rate_score_delta = 25.0 * float(archive_byte_delta) / float(CONTEST_ORIGINAL_BYTES)
    break_even_score = max(0.0, rate_score_delta) + float(required_component_margin)
    min_seg_dist_reduction = break_even_score / 100.0
    pose_break_even = _pose_break_even(baseline_pose_dist, break_even_score)

    blockers = _dispatch_blockers(
        archive_byte_delta=archive_byte_delta,
        decoded_atoms=decoded_atoms,
        runtime_proof=runtime_proof,
        runtime_mode=str(runtime_mode),
        break_even_score=break_even_score,
        sidechannel_manifest=sidechannel_manifest,
        stacked_metadata=stacked,
    )
    return {
        "audit": AUDIT_NAME,
        "schema_version": SCHEMA_VERSION,
        "score_claim": False,
        "dispatch_attempted": False,
        "sidechannel_manifest_path": sidechannel_manifest.get("manifest_path"),
        "candidate_archive_sha256": sidechannel_manifest.get("candidate_archive_sha256"),
        "stacked_archive_path": stacked.get("archive_path"),
        "stacked_archive_sha256": stacked.get("archive_sha256"),
        "source_archive_sha256": sidechannel_manifest.get("source_archive_sha256"),
        "archive_byte_delta": archive_byte_delta,
        "rate_score_delta": rate_score_delta,
        "required_component_margin": float(required_component_margin),
        "break_even_score_improvement": break_even_score,
        "min_required_seg_dist_reduction": min_seg_dist_reduction,
        "baseline_pose_dist": baseline_pose_dist,
        "pose_break_even": pose_break_even,
        "wavelet_sidechannel_bytes": sidechannel_bytes,
        "decoded_atom_count": decoded_atoms,
        "runtime_mode": runtime_mode,
        "runtime_consumption_proof": runtime_proof,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": blockers,
        "next_required_evidence": [
            "reviewed_wr01_apply_transform",
            "byte_closed_candidate_with_transform_mode_not_noop",
            "component_response_or_exact_cuda_evidence_exceeding_break_even",
            "archive_manifest_preflight",
            "exact_cuda_auth_eval",
        ],
    }


def build_wavelet_apply_gate_from_paths(
    *,
    sidechannel_manifest_path: str | Path,
    stacked_metadata_path: str | Path | None = None,
    baseline_pose_dist: float | None = None,
    required_component_margin: float = 0.0,
) -> dict[str, Any]:
    """Load JSON inputs and return :func:`build_wavelet_apply_gate` output."""

    manifest = _read_mapping(Path(sidechannel_manifest_path))
    if "manifest_path" not in manifest:
        manifest = dict(manifest)
        manifest["manifest_path"] = str(sidechannel_manifest_path)
    stacked = _read_mapping(Path(stacked_metadata_path)) if stacked_metadata_path else None
    return build_wavelet_apply_gate(
        sidechannel_manifest=manifest,
        stacked_metadata=stacked,
        baseline_pose_dist=baseline_pose_dist,
        required_component_margin=required_component_margin,
    )


def _dispatch_blockers(
    *,
    archive_byte_delta: int,
    decoded_atoms: int,
    runtime_proof: Any,
    runtime_mode: str,
    break_even_score: float,
    sidechannel_manifest: Mapping[str, Any],
    stacked_metadata: Mapping[str, Any],
) -> list[str]:
    blockers = [
        "requires_reviewed_wr01_apply_transform",
        "requires_component_benefit_evidence_over_break_even",
        "requires_archive_manifest_preflight",
        "requires_exact_cuda_auth_eval",
    ]
    if archive_byte_delta > 0:
        blockers.append("wr01_rate_penalty_must_be_recovered_by_distortion")
    if decoded_atoms <= 0:
        blockers.append("wr01_has_no_decoded_atoms")
    # Round 2 R2-7 fix (2026-05-06, 80% confidence): `runtime_consumed` is
    # tautologically True for any non-empty plan because the underlying proof
    # function only verifies decode roundtrip (Round 1 BUG #3 finding). The
    # gate previously blocked only on `runtime_mode == "explicit_noop_consume_only"`
    # which is NOT the default — the default `"candidate_sidechannel_only"`
    # leaves both gates passable. Add an explicit "runtime mode must prove
    # actual atom application" blocker that fires for the default mode too,
    # so the gate cannot be tautologically passed by a plan that only roundtrips
    # decode without an inflate path applying atoms.
    if not isinstance(runtime_proof, Mapping) or runtime_proof.get("runtime_consumed") is not True:
        blockers.append("wr01_runtime_consumption_not_proven")
    # Only modes that PROVE actual atom application clear this blocker.
    # `candidate_sidechannel_only` (default) does NOT prove application — only
    # that the candidate archive carries the sidechannel bytes. Real
    # application is gated on `wr01_apply_transform_complete` or
    # `inflate_path_applies_wr01_atoms` (set by the apply_transform pipeline).
    _RUNTIME_MODES_THAT_PROVE_APPLICATION = (
        "wr01_apply_transform_complete",
        "inflate_path_applies_wr01_atoms",
    )
    if runtime_mode == "explicit_noop_consume_only":
        blockers.append("wr01_runtime_mode_is_explicit_noop")
    elif runtime_mode not in _RUNTIME_MODES_THAT_PROVE_APPLICATION:
        blockers.append(f"wr01_runtime_mode_does_not_prove_application:{runtime_mode}")
    if sidechannel_manifest.get("score_claim") is not False:
        blockers.append("sidechannel_manifest_score_claim_not_false")
    if stacked_metadata and stacked_metadata.get("score_claim") is not False:
        blockers.append("stacked_metadata_score_claim_not_false")
    # Round 3 R3-C fix (2026-05-06, 82% confidence): `<= 0` was wrong. A
    # zero break-even score is the most permissive valid case — it means
    # zero rate penalty (wavelet transform produced same-size archive) and
    # zero required margin, so any positive component improvement clears
    # the gate. Block only on negative break-even, which is mathematically
    # impossible given `max(0.0, rate_score_delta) + max(0, margin)` but
    # documents the invariant.
    if break_even_score < 0:
        blockers.append("invalid_negative_break_even_target")
    return list(dict.fromkeys(blockers))


def _pose_break_even(baseline_pose_dist: float | None, break_even_score: float) -> dict[str, Any] | None:
    if baseline_pose_dist is None:
        return None
    if baseline_pose_dist < 0:
        raise HnervWaveletApplyGateError("baseline_pose_dist must be non-negative")
    baseline_pose_term = math.sqrt(10.0 * baseline_pose_dist)
    required_pose_term = max(0.0, baseline_pose_term - break_even_score)
    required_pose_dist = (required_pose_term * required_pose_term) / 10.0
    return {
        "baseline_pose_term": baseline_pose_term,
        "required_pose_term": required_pose_term,
        "required_pose_dist": required_pose_dist,
        "min_required_pose_dist_reduction": max(0.0, baseline_pose_dist - required_pose_dist),
    }


def _decoded_atom_count(decoded: Any) -> int:
    if isinstance(decoded, Mapping):
        value = decoded.get("total_atom_count")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _read_mapping(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise HnervWaveletApplyGateError(f"{path} is not a JSON object")
    return data


__all__ = [
    "AUDIT_NAME",
    "CONTEST_ORIGINAL_BYTES",
    "SCHEMA_VERSION",
    "HnervWaveletApplyGateError",
    "build_wavelet_apply_gate",
    "build_wavelet_apply_gate_from_paths",
]
