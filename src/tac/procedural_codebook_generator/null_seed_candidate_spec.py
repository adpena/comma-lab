# SPDX-License-Identifier: MIT
"""Typed null-seed candidate specs for packet-lowering work.

[verified-against: .omx/research/null_codebook_replacement_plan_fec6_20260520T223631Z_codex.md]
[verified-against: src/tac/procedural_codebook_generator/null_replacement_plan.py]
[verified-against: src/tac/master_gradient_operator_plan.py CandidateModificationSpec discipline]

The null-byte replacement plan ranks spans whose master-gradient magnitude is
near zero. That is not enough authority to shrink an archive: a seed
replacement still needs either byte-for-byte reconstruction of parser-valid
payload bytes or an explicit runtime adapter plus exact eval.

This module is the lowering bridge between the planning surface and future
packet builders. It binds a selected null span to archive/member custody,
verifies the original slice SHA empirically, emits deterministic in-archive
seed material, and records whether the seed-derived bytes directly reconstruct
the original span. When they do not, the spec remains a blocked runtime-adapter
candidate rather than a score or promotion claim.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.authority_contract import apply_false_authority_contract

from .null_replacement_plan import SCHEMA as PLAN_SCHEMA
from .seed_derived_codebook import (
    DEFAULT_GENERATOR_KIND,
    SUPPORTED_GENERATOR_KINDS,
    derive_codebook_from_seed,
)

SPEC_SCHEMA = "null_seed_candidate_spec_v1"
SEED_DERIVATION_DOMAIN = b"tac.null_seed_candidate_spec.seed.v1"
COORDINATE_SYSTEM = "packetir_inner_member_section_span"


class NullSeedCandidateSpecError(ValueError):
    """Raised when a null-seed candidate cannot be tied to archive custody."""


def build_null_seed_candidate_spec(
    plan: Mapping[str, Any],
    *,
    candidate_id: str | None = None,
    candidate_rank: int = 1,
    archive_zip_path: str | Path | None = None,
    inner_member_name: str | None = None,
    seed_material: bytes | None = None,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
    output_dtype: str | np.dtype = "uint8",
) -> dict[str, Any]:
    """Build a fail-closed typed spec for one null-seed candidate.

    The returned payload is suitable for downstream packet builders, LL
    planning, and review. It is deliberately not dispatchable: the spec only
    proves archive custody plus seed/reconstruction status.
    """

    _require_plan_schema(plan)
    candidate = _select_candidate(
        plan, candidate_id=candidate_id, candidate_rank=candidate_rank
    )
    if generator_kind not in SUPPORTED_GENERATOR_KINDS:
        raise NullSeedCandidateSpecError(
            f"generator_kind {generator_kind!r} is not supported; expected one of "
            f"{sorted(SUPPORTED_GENERATOR_KINDS)}"
        )
    dtype = np.dtype(output_dtype)
    if dtype.itemsize != 1:
        raise NullSeedCandidateSpecError(
            "null-span candidate specs currently require a byte-sized dtype; "
            f"got {dtype}"
        )

    archive_path = _resolve_archive_path(plan, archive_zip_path)
    member_name, member_payload, member_zip = _read_zip_member(
        archive_path, inner_member_name or _plan_inner_member_name(plan)
    )
    archive_custody = _archive_custody(archive_path)
    member_custody = _member_custody(member_name, member_payload, member_zip)
    _verify_plan_member_sha(plan, member_payload)

    start, end = _candidate_range(candidate)
    if end > len(member_payload):
        raise NullSeedCandidateSpecError(
            f"candidate range {start}:{end} exceeds member bytes {len(member_payload)}"
        )
    original_span = member_payload[start:end]
    original_sha = hashlib.sha256(original_span).hexdigest()
    expected_sha = candidate.get("original_sha256")
    if expected_sha is not None and str(expected_sha).lower() != original_sha:
        raise NullSeedCandidateSpecError(
            "candidate original_sha256 mismatch: "
            f"expected={expected_sha}, actual={original_sha}"
        )

    expected_seed_len = _positive_int(candidate.get("seed_bytes"), "candidate.seed_bytes")
    seed = (
        _derive_default_seed(candidate, original_sha, expected_seed_len)
        if seed_material is None
        else bytes(seed_material)
    )
    if len(seed) != expected_seed_len:
        raise NullSeedCandidateSpecError(
            f"seed_material length {len(seed)} does not match candidate seed_bytes "
            f"{expected_seed_len}"
        )

    derived = derive_codebook_from_seed(
        seed,
        output_shape=(len(original_span),),
        dtype=dtype,
        generator_kind=generator_kind,  # type: ignore[arg-type]
    ).tobytes()
    derived_sha = hashlib.sha256(derived).hexdigest()
    reconstructs_original = derived == original_span
    blockers = _spec_blockers(candidate, reconstructs_original=reconstructs_original)

    spec: dict[str, Any] = {
        "schema": SPEC_SCHEMA,
        "spec_id": f"null_seed_candidate::{candidate['candidate_id']}",
        "source_plan_schema": plan.get("schema"),
        "source_candidate_id": candidate["candidate_id"],
        "source_candidate_rank": _candidate_rank(plan, candidate),
        "coordinate_system": COORDINATE_SYSTEM,
        "raw_archive_byte_coordinates_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "research_only": True,
        "archive_custody": archive_custody,
        "member_custody": member_custody,
        "null_span": {
            "section": candidate.get("section"),
            "range": [start, end],
            "original_bytes": len(original_span),
            "original_sha256": original_sha,
            "plan_original_sha256": expected_sha,
            "source": candidate.get("source"),
            "net_saved_inner_bytes": candidate.get("net_saved_inner_bytes"),
            "predicted_rate_delta_upper_bound": candidate.get(
                "predicted_rate_delta_upper_bound"
            ),
        },
        "seed_replacement": {
            "authority_mode": "archive_seeded",
            "seed_bytes": len(seed),
            "seed_sha256": hashlib.sha256(seed).hexdigest(),
            "seed_hex": seed.hex(),
            "seed_derivation": (
                "operator_supplied"
                if seed_material is not None
                else "sha256(domain || candidate_id || original_span_sha256)"
            ),
            "generator_kind": generator_kind,
            "output_shape": [len(original_span)],
            "output_dtype": str(dtype),
            "derived_payload_sha256": derived_sha,
            "seed_reconstructs_original_payload": reconstructs_original,
        },
        "direct_replacement_ready": reconstructs_original,
        "runtime_adapter_required": not reconstructs_original,
        "candidate_modification_spec": _candidate_modification_spec(
            archive_custody=archive_custody,
            member_custody=member_custody,
            candidate=candidate,
            blockers=blockers,
        ),
        "required_next_proofs": [
            "runtime_adapter_decodes_seeded_span" if not reconstructs_original else "none",
            "seed_mutation_changes_inflated_frames_or_expected_null_region",
            "full_frame_inflate_output_parity_or_exact_candidate_delta_report",
            "contest_cpu_and_cuda_exact_eval_on_candidate_archive",
        ],
        "blockers": blockers,
        "verdict": (
            "direct_seed_reconstruction_ready"
            if reconstructs_original
            else "blocked_until_runtime_adapter_and_exact_eval"
        ),
    }
    return apply_false_authority_contract(
        spec,
        preserve_dispatch_ready=False,
        reason="null_seed_candidate_spec_requires_runtime_adapter_and_exact_eval",
    )


def render_null_seed_candidate_spec_markdown(spec: Mapping[str, Any]) -> str:
    """Render a compact review memo for a null-seed candidate spec."""

    span = _require_mapping(spec.get("null_span"), "null_span")
    seed = _require_mapping(spec.get("seed_replacement"), "seed_replacement")
    archive = _require_mapping(spec.get("archive_custody"), "archive_custody")
    member = _require_mapping(spec.get("member_custody"), "member_custody")
    lines = [
        "# Null-Seed Candidate Spec",
        "",
        f"- Schema: `{spec.get('schema')}`",
        f"- Spec: `{spec.get('spec_id')}`",
        f"- Verdict: `{spec.get('verdict')}`",
        f"- Score claim: `{str(spec.get('score_claim')).lower()}`",
        f"- Promotion eligible: `{str(spec.get('promotion_eligible')).lower()}`",
        f"- Archive: `{archive.get('path')}`",
        f"- Archive SHA-256: `{archive.get('sha256')}`",
        f"- Member: `{member.get('name')}`",
        f"- Member SHA-256: `{member.get('sha256')}`",
        f"- Span: `{span.get('range')}` in `{span.get('section')}`",
        f"- Original bytes: `{span.get('original_bytes')}`",
        f"- Original span SHA-256: `{span.get('original_sha256')}`",
        f"- Seed bytes: `{seed.get('seed_bytes')}`",
        f"- Generator: `{seed.get('generator_kind')}`",
        "- Seed reconstructs original payload: "
        f"`{str(seed.get('seed_reconstructs_original_payload')).lower()}`",
        f"- Direct replacement ready: `{str(spec.get('direct_replacement_ready')).lower()}`",
        f"- Runtime adapter required: `{str(spec.get('runtime_adapter_required')).lower()}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = spec.get("blockers")
    if isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes)):
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- `blockers_missing_or_malformed`")
    lines.extend(
        [
            "",
            "This spec is a lowering target, not a score artifact. It proves",
            "archive/member custody and seed reconstruction status only.",
            "",
        ]
    )
    return "\n".join(lines)


def _require_plan_schema(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise NullSeedCandidateSpecError(
            f"expected plan schema {PLAN_SCHEMA!r}; got {plan.get('schema')!r}"
        )


def _select_candidate(
    plan: Mapping[str, Any], *, candidate_id: str | None, candidate_rank: int
) -> Mapping[str, Any]:
    candidates = plan.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise NullSeedCandidateSpecError("plan.candidates must be a sequence")
    rows = [row for row in candidates if isinstance(row, Mapping)]
    if candidate_id is not None:
        for row in rows:
            if row.get("candidate_id") == candidate_id:
                return row
        raise NullSeedCandidateSpecError(f"candidate_id not found: {candidate_id}")
    if candidate_rank <= 0:
        raise NullSeedCandidateSpecError("candidate_rank must be positive")
    index = candidate_rank - 1
    if index >= len(rows):
        raise NullSeedCandidateSpecError(
            f"candidate_rank {candidate_rank} exceeds candidate count {len(rows)}"
        )
    return rows[index]


def _candidate_rank(plan: Mapping[str, Any], candidate: Mapping[str, Any]) -> int:
    rows = plan.get("candidates")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return 0
    for idx, row in enumerate(rows, start=1):
        if isinstance(row, Mapping) and row.get("candidate_id") == candidate.get(
            "candidate_id"
        ):
            return idx
    return 0


def _candidate_range(candidate: Mapping[str, Any]) -> tuple[int, int]:
    value = candidate.get("range")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise NullSeedCandidateSpecError("candidate.range must be [start, end]")
    start = _nonnegative_int(value[0], "candidate.range[0]")
    end = _positive_int(value[1], "candidate.range[1]")
    if end <= start:
        raise NullSeedCandidateSpecError("candidate range must be non-empty")
    original_bytes = _positive_int(candidate.get("original_bytes"), "candidate.original_bytes")
    if original_bytes != end - start:
        raise NullSeedCandidateSpecError(
            "candidate.original_bytes must equal candidate range length"
        )
    return start, end


def _resolve_archive_path(
    plan: Mapping[str, Any], archive_zip_path: str | Path | None
) -> Path:
    raw = archive_zip_path
    if raw is None:
        input_paths = _require_mapping(plan.get("input_paths"), "input_paths")
        raw = input_paths.get("archive_zip")
    if raw is None:
        raise NullSeedCandidateSpecError(
            "archive_zip_path missing and plan.input_paths.archive_zip absent"
        )
    path = Path(raw)
    if not path.is_file():
        raise NullSeedCandidateSpecError(f"archive zip does not exist: {path}")
    return path


def _plan_inner_member_name(plan: Mapping[str, Any]) -> str | None:
    input_paths = plan.get("input_paths")
    if not isinstance(input_paths, Mapping):
        return None
    value = input_paths.get("inner_member_name")
    return None if value is None else str(value)


def _read_zip_member(
    archive_path: Path, member_name: str | None
) -> tuple[str, bytes, zipfile.ZipInfo]:
    with zipfile.ZipFile(archive_path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if member_name is None:
            if len(infos) != 1:
                raise NullSeedCandidateSpecError(
                    f"{archive_path} has {len(infos)} file members; pass inner_member_name"
                )
            info = infos[0]
        else:
            matches = [info for info in infos if info.filename == member_name]
            if len(matches) != 1:
                raise NullSeedCandidateSpecError(
                    f"{archive_path} expected exactly one member {member_name!r}"
                )
            info = matches[0]
        return info.filename, zf.read(info), info


def _archive_custody(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "provenance_kind": "archive_zip_custody",
    }


def _member_custody(
    name: str, payload: bytes, info: zipfile.ZipInfo
) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(payload),
        "compressed_bytes": int(info.compress_size),
        "compress_type": int(info.compress_type),
        "crc32": f"{int(info.CRC):08x}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "provenance_kind": "archive_member_custody",
    }


def _verify_plan_member_sha(plan: Mapping[str, Any], member_payload: bytes) -> None:
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        return
    expected = inputs.get("inner_bytes_sha256")
    if expected is None:
        return
    actual = hashlib.sha256(member_payload).hexdigest()
    if str(expected).lower() != actual:
        raise NullSeedCandidateSpecError(
            "plan inputs.inner_bytes_sha256 mismatch: "
            f"expected={expected}, actual={actual}"
        )


def _derive_default_seed(
    candidate: Mapping[str, Any], original_sha: str, seed_len: int
) -> bytes:
    out = bytearray()
    counter = 0
    candidate_id = str(candidate.get("candidate_id", "unknown")).encode("utf-8")
    original = original_sha.encode("ascii")
    while len(out) < seed_len:
        h = hashlib.sha256()
        h.update(SEED_DERIVATION_DOMAIN)
        h.update(counter.to_bytes(4, "little"))
        h.update(candidate_id)
        h.update(b"\0")
        h.update(original)
        out.extend(h.digest())
        counter += 1
    return bytes(out[:seed_len])


def _spec_blockers(
    candidate: Mapping[str, Any], *, reconstructs_original: bool
) -> list[str]:
    blockers = [
        "full_frame_inflate_output_parity_missing",
        "contest_cpu_exact_eval_missing",
        "contest_cuda_exact_eval_missing",
    ]
    if not reconstructs_original:
        blockers.extend(
            [
                "seed_reconstruction_mismatch",
                "runtime_adapter_not_materialized",
                "seed_mutation_frame_delta_proof_missing",
            ]
        )
    section = str(candidate.get("section", "")).lower()
    if "source_payload" in section and not reconstructs_original:
        blockers.append("source_payload_seed_substitution_parse_risk")
    if "selector" in section and not reconstructs_original:
        blockers.append("selector_seed_adapter_required")
    return _unique(blockers)


def _candidate_modification_spec(
    *,
    archive_custody: Mapping[str, Any],
    member_custody: Mapping[str, Any],
    candidate: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "spec_id": f"candidate_modification::{candidate['candidate_id']}",
        "source_archive_path": archive_custody.get("path"),
        "source_archive_sha256": archive_custody.get("sha256"),
        "source_archive_bytes": archive_custody.get("bytes"),
        "source_member_name": member_custody.get("name"),
        "source_member_sha256": member_custody.get("sha256"),
        "operator_id": f"null_seed_replacement::{candidate['candidate_id']}",
        "section_name": candidate.get("section"),
        "section_role": "master_gradient_null_span",
        "mutation_grain": "archive_seeded_procedural_replacement",
        "mutation_operator": "seed_derive_null_span_candidate",
        "axis_label": "[predicted-rate-only-upper-bound]",
        "response_matrix_columns": (
            "seg_dist_delta",
            "pose_dist_delta",
            "rate_bytes_delta",
        ),
        "packet_proofs_required": (
            "runtime_adapter_decodes_seeded_span",
            "seed_mutation_frame_delta_proof",
            "updated_zip_headers",
            "updated_zip_crc",
            "inflate_success_proof",
            "exact_cuda_eval_on_candidate_archive",
        ),
        "packet_proofs_available": False,
        "ready_for_operator_probe": False,
        "ready_for_provider_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "coordinate_system": COORDINATE_SYSTEM,
        "raw_archive_byte_coordinates_allowed": False,
        "blockers": tuple(blockers),
        "rationale": (
            "null-gradient rank is planning evidence only",
            "seed replacement must be parser-valid or runtime-adapter backed",
            "candidate remains non-dispatchable until exact eval closes the loop",
        ),
    }


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NullSeedCandidateSpecError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NullSeedCandidateSpecError(f"{name} must be a positive integer")
    return int(value)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NullSeedCandidateSpecError(f"{name} must be a mapping")
    return value


def _unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "COORDINATE_SYSTEM",
    "SPEC_SCHEMA",
    "NullSeedCandidateSpecError",
    "build_null_seed_candidate_spec",
    "render_null_seed_candidate_spec_markdown",
]
