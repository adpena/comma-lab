# SPDX-License-Identifier: MIT
"""PR101/FEC6 frontier PacketIR authority matrix.

This joins the current FEC6 frontier archive, exact-eval evidence, and
parser/profile artifacts into one non-promotional audit surface. It does not
build candidates, dispatch evals, or upgrade the archive to a score claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tac.packet_compiler.deterministic_compiler import SCHEMA_VERSION as DETERMINISTIC_COMPILER_SCHEMA
from tac.packet_compiler.pr101_fec6_candidate_queue import (
    PR101_FEC6_BYTE_ACCOUNTING_SCHEMA,
    PR101_FEC6_CANDIDATE_QUEUE_SCHEMA,
)
from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json

# Canonical golden_vectors schema per
# tac.packet_compiler.deterministic_compiler._golden_vectors and
# tac.packet_compiler.deterministic_compiler._parser_section_manifest.  Per
# codex adversarial review 2026-05-19 F2: the v1 manifest-row only checked
# `isinstance(_, Mapping)` so empty `{}` passed as valid identity evidence.
# v2 enforces the canonical schema fields.
_GOLDEN_VECTORS_SCHEMA_VERSION = "deterministic_golden_vectors.v1"
_PARSER_SECTION_MANIFEST_SCHEMA_VERSION = (
    "deterministic_parser_section_manifest.v1"
)
_GOLDEN_VECTOR_MEMBER_REQUIRED_KEYS: tuple[str, ...] = (
    "name",
    "payload_sha256",
    "compressed_payload_sha256",
    "compress_type",
    "uncompressed_bytes",
    "compressed_bytes",
    "data_offset",
)
_PARSER_SECTION_MANIFEST_LIST_KEYS: tuple[str, ...] = (
    "section_names",
    "lengths",
    "section_sha256s",
    "offsets",
    "compress_types",
)


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in value)


def _validate_parser_section_manifest(
    manifest: Any,
) -> list[str]:
    """Return per-blocker list for the canonical parser_section_manifest.

    Refuses any state that the v1 isinstance(Mapping) check let through:
    ``{}`` empty dict, missing schema_version, section_count <= 0, list
    lengths that disagree with section_count, non-string section names,
    non-hex section_sha256s, negative lengths/offsets.
    """

    blockers: list[str] = []
    if not isinstance(manifest, Mapping):
        return ["deterministic_compiler_parser_manifest_missing"]
    if manifest.get("schema_version") != _PARSER_SECTION_MANIFEST_SCHEMA_VERSION:
        blockers.append("deterministic_compiler_parser_manifest_schema_mismatch")
    section_count = manifest.get("section_count")
    if not isinstance(section_count, int) or section_count < 1:
        blockers.append("deterministic_compiler_parser_manifest_section_count_invalid")
        section_count = None
    for key in _PARSER_SECTION_MANIFEST_LIST_KEYS:
        value = manifest.get(key)
        if not isinstance(value, list):
            blockers.append(
                f"deterministic_compiler_parser_manifest_{key}_not_list"
            )
            continue
        if not value:
            blockers.append(
                f"deterministic_compiler_parser_manifest_{key}_empty"
            )
            continue
        if section_count is not None and len(value) != section_count:
            blockers.append(
                f"deterministic_compiler_parser_manifest_{key}_length_mismatch"
            )
    names = manifest.get("section_names")
    if isinstance(names, list) and names:
        for name in names:
            if not isinstance(name, str) or not name:
                blockers.append(
                    "deterministic_compiler_parser_manifest_section_name_invalid"
                )
                break
    section_shas = manifest.get("section_sha256s")
    if isinstance(section_shas, list) and section_shas:
        for sha in section_shas:
            if not _is_sha256_hex(sha):
                blockers.append(
                    "deterministic_compiler_parser_manifest_section_sha_invalid"
                )
                break
    lengths = manifest.get("lengths")
    if isinstance(lengths, list) and lengths:
        for length in lengths:
            if not isinstance(length, int) or length < 0:
                blockers.append(
                    "deterministic_compiler_parser_manifest_length_negative_or_nonint"
                )
                break
    offsets = manifest.get("offsets")
    if isinstance(offsets, list) and offsets:
        for offset in offsets:
            if not isinstance(offset, int) or offset < 0:
                blockers.append(
                    "deterministic_compiler_parser_manifest_offset_negative_or_nonint"
                )
                break
    return blockers


def _validate_golden_vectors(
    vectors: Any,
    *,
    expected_archive_sha256: str,
    expected_runtime_tree_sha256: str | None = None,
) -> list[str]:
    """Return per-blocker list for the canonical golden_vectors manifest.

    Refuses any state that the v1 isinstance(Mapping) check let through:
    ``{}`` empty dict, missing schema_version, empty member_vectors list,
    archive_sha256/runtime_tree_sha256 mismatch vs the top-level manifest
    they accompany, member_vectors entries lacking required keys, member
    payload_sha256/compressed_payload_sha256 not valid hex.
    """

    blockers: list[str] = []
    if not isinstance(vectors, Mapping):
        return ["deterministic_compiler_golden_vectors_missing"]
    if vectors.get("schema_version") != _GOLDEN_VECTORS_SCHEMA_VERSION:
        blockers.append("deterministic_compiler_golden_vectors_schema_mismatch")
    if vectors.get("mode") not in ("identity", "canonicalize", "optimize"):
        blockers.append("deterministic_compiler_golden_vectors_mode_invalid")
    archive_sha = str(vectors.get("archive_sha256") or "").strip().lower()
    if not _is_sha256_hex(archive_sha):
        blockers.append("deterministic_compiler_golden_vectors_archive_sha_invalid")
    elif expected_archive_sha256 and archive_sha != expected_archive_sha256.lower():
        blockers.append(
            "deterministic_compiler_golden_vectors_archive_sha_mismatch"
        )
    runtime_tree_sha = str(vectors.get("runtime_tree_sha256") or "").strip().lower()
    if not _is_sha256_hex(runtime_tree_sha):
        blockers.append(
            "deterministic_compiler_golden_vectors_runtime_tree_sha_invalid"
        )
    elif (
        expected_runtime_tree_sha256
        and runtime_tree_sha != expected_runtime_tree_sha256.lower()
    ):
        blockers.append(
            "deterministic_compiler_golden_vectors_runtime_tree_sha_mismatch"
        )
    members = vectors.get("member_vectors")
    if not isinstance(members, list):
        blockers.append(
            "deterministic_compiler_golden_vectors_member_vectors_not_list"
        )
    elif not members:
        blockers.append(
            "deterministic_compiler_golden_vectors_member_vectors_empty"
        )
    else:
        for member in members:
            if not isinstance(member, Mapping):
                blockers.append(
                    "deterministic_compiler_golden_vectors_member_entry_not_object"
                )
                break
            missing_keys = [
                key
                for key in _GOLDEN_VECTOR_MEMBER_REQUIRED_KEYS
                if key not in member
            ]
            if missing_keys:
                blockers.append(
                    "deterministic_compiler_golden_vectors_member_missing_keys"
                )
                break
            if not _is_sha256_hex(member.get("payload_sha256")):
                blockers.append(
                    "deterministic_compiler_golden_vectors_member_payload_sha_invalid"
                )
                break
            if not _is_sha256_hex(member.get("compressed_payload_sha256")):
                blockers.append(
                    "deterministic_compiler_golden_vectors_member_compressed_payload_sha_invalid"
                )
                break
    return blockers

PR101_FRONTIER_PACKETIR_MATRIX_SCHEMA = "pr101_fec6_frontier_packetir_matrix_v1"
PR101_FRONTIER_PACKETIR_MATRIX_DEFAULT_JSON = (
    ".omx/research/pr101_fec6_frontier_packetir_matrix_20260519_codex.json"
)
PR101_FRONTIER_PACKETIR_MATRIX_DEFAULT_MD = (
    ".omx/research/pr101_fec6_frontier_packetir_matrix_20260519_codex.md"
)

ContestAxis = Literal["contest_cpu", "contest_cuda"]
_REQUIRED_AXES: tuple[ContestAxis, ContestAxis] = ("contest_cpu", "contest_cuda")


@dataclass(frozen=True)
class PR101FEC6FrontierMatrixSpec:
    """Repo-local artifact paths for the PR101/FEC6 frontier matrix."""

    archive_path: str = (
        "experiments/results/"
        "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
        "archive.zip"
    )
    archive_manifest_path: str = (
        "experiments/results/"
        "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
        "archive_manifest.json"
    )
    packet_manifest_path: str = (
        "experiments/results/"
        "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
        "packet_manifest.json"
    )
    packetir_identity_proof_path: str = (
        ".omx/research/pr101_fec6_packetir_identity_proof_20260519_codex.json"
    )
    deterministic_compiler_manifest_path: str = (
        "experiments/results/"
        "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
        "deterministic_packet_compiler_manifest.json"
    )
    candidate_queue_path: str = (
        "experiments/results/"
        "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
        "packetir_candidate_queue.json"
    )
    exact_eval_paths: Mapping[ContestAxis, str] = field(
        default_factory=lambda: {
            "contest_cpu": (
                "experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/"
                "contest_auth_eval.json"
            ),
            "contest_cuda": (
                "experiments/results/modal_auth_eval/archive_6bae0201fb08/"
                "contest_auth_eval.json"
            ),
        }
    )
    parser_profile_paths: tuple[str, ...] = (
        "experiments/results/"
        "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
        "fec3_k16_fec6_k16_full_streaming_parity.json",
        "experiments/results/pr101_fec6_byte_escape_profile_20260515_codex/profile.json",
        "experiments/results/pr101_fec6_byte_escape_profile_20260515_codex/profile.md",
        ".omx/research/pr101_fec6_wrapper_profile_20260515_codex.md",
        ".omx/research/pr101_fec6_parser_wire_in_20260515_codex.md",
        ".omx/research/pr101_fec6_paired_cpu_cuda_axis_xray_20260515_codex.md",
        "experiments/results/xray_paired_cpu_cuda_axis_delta_pr101_fec6_20260515_codex/"
        "paired_axis_delta.json",
        "experiments/results/xray_entropy_pr101_fec6_vs_pr106_packetir_20260515_codex/"
        "heatmap.json",
        "experiments/results/fec6_selector_operator_space_20260517_codex/"
        "operator_space_manifest.json",
    )


DEFAULT_PR101_FEC6_FRONTIER_MATRIX_SPEC = PR101FEC6FrontierMatrixSpec()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _artifact_row(
    path_text: str | Path,
    *,
    repo_root: Path,
    artifact_kind: str,
) -> dict[str, Any]:
    path = _resolve(path_text, repo_root)
    row: dict[str, Any] = {
        "kind": artifact_kind,
        "path": repo_relative(path, repo_root),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return row
    row["bytes"] = path.stat().st_size
    row["sha256"] = sha256_file(path)
    return row


def _json_artifact_row(
    path_text: str | Path,
    *,
    repo_root: Path,
    artifact_kind: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    row = _artifact_row(path_text, repo_root=repo_root, artifact_kind=artifact_kind)
    if row["exists"] is not True:
        return row, {}
    path = _resolve(path_text, repo_root)
    try:
        payload = read_json(path)
    except Exception as exc:  # pragma: no cover - defensive malformed fixture
        row["json_valid"] = False
        row["json_error"] = type(exc).__name__
        return row, {}
    if not isinstance(payload, Mapping):
        row["json_valid"] = False
        row["json_error"] = "not_object"
        return row, {}
    row["json_valid"] = True
    row["schema"] = payload.get("schema") or payload.get("schema_version")
    return row, payload


def _provenance(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = payload.get("provenance")
    return provenance if isinstance(provenance, Mapping) else {}


def _runtime_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = _provenance(payload).get("inflate_runtime_manifest")
    return manifest if isinstance(manifest, Mapping) else {}


def _archive_sha_from_exact_eval(payload: Mapping[str, Any]) -> str:
    value = _provenance(payload).get("archive_sha256") or payload.get("archive_sha256")
    return str(value or "").strip().lower()


def _inflated_manifest_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    inflated = _provenance(payload).get("inflated_output_manifest")
    if not isinstance(inflated, Mapping):
        return {"present": False}
    nested = inflated.get("payload")
    nested_payload = nested if isinstance(nested, Mapping) else {}
    return {
        "present": True,
        "path": inflated.get("path", ""),
        "sha256": inflated.get("sha256", ""),
        "aggregate_sha256": nested_payload.get("aggregate_sha256", ""),
        "raw_file_count": nested_payload.get("raw_file_count"),
        "total_bytes": nested_payload.get("total_bytes"),
    }


def _exact_eval_row(
    *,
    axis: ContestAxis,
    path_text: str,
    repo_root: Path,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    row, payload = _json_artifact_row(
        path_text,
        repo_root=repo_root,
        artifact_kind=f"{axis}_exact_eval",
    )
    row["axis"] = axis
    if row["exists"] is not True or not payload:
        row["valid_axis_evidence"] = False
        row["blockers"] = ["exact_eval_artifact_missing_or_invalid"]
        return row

    provenance = _provenance(payload)
    runtime = _runtime_manifest(payload)
    archive_sha256 = _archive_sha_from_exact_eval(payload)
    observed_axis = str(payload.get("score_axis") or "").strip()
    device = str(provenance.get("device") or "").strip()
    platform_system = str(provenance.get("platform_system") or "").strip()
    platform_machine = str(provenance.get("platform_machine") or "").strip()
    gpu_model = str(provenance.get("gpu_model") or "").strip()
    blockers: list[str] = []
    if observed_axis != axis:
        blockers.append(f"score_axis_mismatch:{observed_axis or 'missing'}")
    if expected_archive_sha256 and archive_sha256 != expected_archive_sha256:
        blockers.append("archive_sha256_mismatch")
    if payload.get("n_samples") != 600:
        blockers.append("n_samples_not_600")
    if axis == "contest_cpu":
        if device != "cpu":
            blockers.append("device_not_cpu")
        if platform_system != "Linux" or platform_machine != "x86_64":
            blockers.append("hardware_not_linux_x86_64_cpu")
    if axis == "contest_cuda":
        if device != "cuda":
            blockers.append("device_not_cuda")
        if "t4" not in gpu_model.lower() and provenance.get("gpu_t4_match") is not True:
            blockers.append("hardware_not_t4_cuda")

    row.update(
        {
            "valid_axis_evidence": not blockers,
            "blockers": blockers,
            "score_axis": observed_axis,
            "evidence_grade": payload.get("evidence_grade"),
            "canonical_score": payload.get("canonical_score")
            or payload.get("score_recomputed_from_components"),
            "score_recomputed_from_components": payload.get(
                "score_recomputed_from_components"
            ),
            "archive_sha256": archive_sha256,
            "archive_size_bytes": payload.get("archive_size_bytes"),
            "avg_segnet_dist": payload.get("avg_segnet_dist"),
            "avg_posenet_dist": payload.get("avg_posenet_dist"),
            "n_samples": payload.get("n_samples"),
            "hardware": {
                "device": device,
                "platform_system": platform_system,
                "platform_machine": platform_machine,
                "gpu_model": gpu_model,
                "gpu_t4_match": provenance.get("gpu_t4_match"),
            },
            "runtime_tree_sha256": runtime.get("runtime_tree_sha256", ""),
            "runtime_content_tree_sha256": runtime.get(
                "runtime_content_tree_sha256", ""
            ),
            "inflated_output_manifest": _inflated_manifest_summary(payload),
            "score_claim_in_source_artifact": payload.get("score_claim") is True,
            "score_claim_valid_in_source_artifact": (
                payload.get("score_claim_valid") is True
            ),
            "promotion_eligible_in_source_artifact": (
                payload.get("promotion_eligible") is True
            ),
        }
    )
    return row


def _archive_summary(
    *,
    spec: PR101FEC6FrontierMatrixSpec,
    repo_root: Path,
) -> tuple[dict[str, Any], str]:
    row = _artifact_row(spec.archive_path, repo_root=repo_root, artifact_kind="archive")
    manifest_row, manifest = _json_artifact_row(
        spec.archive_manifest_path,
        repo_root=repo_root,
        artifact_kind="archive_manifest",
    )
    packet_row, packet_manifest = _json_artifact_row(
        spec.packet_manifest_path,
        repo_root=repo_root,
        artifact_kind="packet_manifest",
    )
    packet_archive = packet_manifest.get("archive")
    expected_sha = str(manifest.get("archive_sha256") or "").strip().lower()
    if not expected_sha and isinstance(packet_archive, Mapping):
        expected_sha = str(packet_archive.get("sha256") or "").strip().lower()
    observed_sha = str(row.get("sha256") or "").strip().lower()
    row.update(
        {
            "manifest": manifest_row,
            "packet_manifest": packet_row,
            "expected_sha256": expected_sha,
            "expected_bytes": manifest.get("archive_bytes")
            or (
                packet_archive.get("bytes")
                if isinstance(packet_archive, Mapping)
                else None
            ),
            "source_archive_sha256": manifest.get("source_archive_sha256")
            or (
                packet_archive.get("source")
                if isinstance(packet_archive, Mapping)
                else ""
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "sha256_matches_manifest": bool(expected_sha)
            and bool(observed_sha)
            and observed_sha == expected_sha,
        }
    )
    return row, expected_sha or observed_sha


def _parser_profile_artifacts(
    paths: Iterable[str],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows = [
        _artifact_row(path, repo_root=repo_root, artifact_kind="parser_profile")
        for path in paths
    ]
    return [row for row in rows if row.get("exists") is True]


def _packetir_primitives(repo_root: Path) -> list[dict[str, Any]]:
    primitive_paths = (
        "src/tac/packet_compiler/pr101_sidecar_grammar.py",
        "src/tac/packet_compiler/pr101_decoder_storage_order.py",
        "src/tac/packet_compiler/pr101_conv4_storage_perms.py",
        "src/tac/packet_compiler/pr101_decoder_byte_maps.py",
        "src/tac/packet_compiler/golden_vectors/pr101_decoder_storage_order_v1.json",
        "src/tac/packet_compiler/golden_vectors/pr101_conv4_storage_perms_v1.json",
        "src/tac/packet_compiler/golden_vectors/pr101_decoder_byte_maps_v1.json",
    )
    rows = [
        _artifact_row(path, repo_root=repo_root, artifact_kind="packetir_primitive")
        for path in primitive_paths
    ]
    for row in rows:
        row["general_pr101_packet_compiler_primitive"] = True
        row["score_claim"] = False
    return rows


_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "ready_for_operator_probe",
    "ready_for_provider_dispatch",
    "dispatch_attempted",
)


def _false_authority_blockers(
    payload: Mapping[str, Any],
    *,
    prefix: str,
    fields: Iterable[str] = _FALSE_AUTHORITY_FIELDS,
) -> list[str]:
    return [
        f"{prefix}_{field}_overclaimed"
        for field in fields
        if payload.get(field) is True
    ]


def _candidate_queue_row(
    path_text: str,
    *,
    repo_root: Path,
    expected_archive_sha256: str,
    expected_archive_bytes: int | None,
) -> dict[str, Any]:
    row, payload = _json_artifact_row(
        path_text,
        repo_root=repo_root,
        artifact_kind="pr101_fec6_packetir_candidate_queue",
    )
    blockers: list[str] = []
    candidate_queue_generated = False
    parser_byte_accounting_present = False
    runtime_consumption_proven = False
    if row["exists"] is not True or not payload:
        blockers.append("candidate_queue_missing_or_invalid_json")
    else:
        if payload.get("schema") != PR101_FEC6_CANDIDATE_QUEUE_SCHEMA:
            blockers.append("candidate_queue_schema_mismatch")
        archive_sha256 = str(payload.get("archive_sha256") or "").strip().lower()
        if expected_archive_sha256 and archive_sha256 != expected_archive_sha256:
            blockers.append("candidate_queue_archive_sha256_mismatch")
        archive_size = payload.get("archive_size_bytes")
        if (
            isinstance(expected_archive_bytes, int)
            and archive_size != expected_archive_bytes
        ):
            blockers.append("candidate_queue_archive_size_mismatch")
        if payload.get("expected_archive_sha256_matches") is False:
            blockers.append("candidate_queue_expected_archive_sha256_mismatch")
        queue_blockers = payload.get("blockers")
        if queue_blockers:
            blockers.append("candidate_queue_top_level_blockers_present")
        blockers.extend(_false_authority_blockers(payload, prefix="candidate_queue"))

        byte_accounting = payload.get("byte_accounting")
        if not isinstance(byte_accounting, Mapping):
            blockers.append("candidate_queue_byte_accounting_missing")
        else:
            if byte_accounting.get("schema") != PR101_FEC6_BYTE_ACCOUNTING_SCHEMA:
                blockers.append("candidate_queue_byte_accounting_schema_mismatch")
            if byte_accounting.get("all_payload_bytes_accounted") is not True:
                blockers.append("candidate_queue_payload_bytes_not_accounted")
            if (
                byte_accounting.get("runtime_consumed_byte_accounting_passed")
                is not True
            ):
                blockers.append("candidate_queue_runtime_consumption_accounting_failed")
            sections = byte_accounting.get("sections")
            if not isinstance(sections, list) or not sections:
                blockers.append("candidate_queue_byte_accounting_sections_missing")
            runtime_surfaces = byte_accounting.get("runtime_consumer_surfaces")
            queue_surfaces = byte_accounting.get("queue_consumer_surfaces")
            if not isinstance(runtime_surfaces, list) or not runtime_surfaces:
                blockers.append("candidate_queue_runtime_consumer_surfaces_missing")
            if not isinstance(queue_surfaces, list) or not queue_surfaces:
                blockers.append("candidate_queue_queue_consumer_surfaces_missing")
            blockers.extend(
                _false_authority_blockers(
                    byte_accounting,
                    prefix="candidate_queue_byte_accounting",
                    fields=(
                        "score_claim",
                        "promotion_eligible",
                        "ready_for_exact_eval_dispatch",
                    ),
                )
            )
            parser_byte_accounting_present = (
                byte_accounting.get("schema") == PR101_FEC6_BYTE_ACCOUNTING_SCHEMA
                and byte_accounting.get("all_payload_bytes_accounted") is True
                and byte_accounting.get("parser_byte_accounting_passed") is True
                and isinstance(sections, list)
                and bool(sections)
            )
            runtime_consumption_proven = (
                parser_byte_accounting_present
                and byte_accounting.get("runtime_consumption_proven") is True
                and byte_accounting.get("runtime_consumed_byte_accounting_passed")
                is True
            )

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            blockers.append("candidate_queue_candidates_missing")
        else:
            if payload.get("candidate_count") != len(candidates):
                blockers.append("candidate_queue_candidate_count_mismatch")
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, Mapping):
                    blockers.append(f"candidate_queue_candidate_{index}_not_object")
                    continue
                candidate_id = str(candidate.get("candidate_id") or index)
                blockers.extend(
                    _false_authority_blockers(
                        candidate,
                        prefix=f"candidate_queue_candidate_{candidate_id}",
                    )
                )
                surfaces = candidate.get("consumer_surfaces")
                if not isinstance(surfaces, list) or not surfaces:
                    blockers.append(
                        f"candidate_queue_candidate_{candidate_id}_consumer_surfaces_missing"
                    )
        candidate_queue_generated = (
            payload.get("schema") == PR101_FEC6_CANDIDATE_QUEUE_SCHEMA
            and str(payload.get("archive_sha256") or "").strip().lower()
            == expected_archive_sha256
            and (
                not isinstance(expected_archive_bytes, int)
                or payload.get("archive_size_bytes") == expected_archive_bytes
            )
            and payload.get("expected_archive_sha256_matches") is not False
            and isinstance(payload.get("candidates"), list)
            and bool(payload.get("candidates"))
            and payload.get("candidate_count") == len(payload.get("candidates", []))
        )
    row["pr106_style_packetir_candidate_queue"] = (
        row["exists"] is True and not blockers
    )
    row["candidate_byte_accounting_present"] = (
        row["pr106_style_packetir_candidate_queue"] is True
    )
    row["candidate_queue_generated"] = candidate_queue_generated
    row["parser_byte_accounting_present"] = parser_byte_accounting_present
    row["runtime_consumption_proven"] = runtime_consumption_proven
    row["blockers"] = blockers
    row["score_claim"] = False
    row["score_claim_valid"] = False
    row["promotion_eligible"] = False
    row["ready_for_exact_eval_dispatch"] = False
    if payload:
        row["archive_sha256"] = str(payload.get("archive_sha256") or "").strip().lower()
        row["archive_size_bytes"] = payload.get("archive_size_bytes")
        row["candidate_count"] = payload.get("candidate_count")
        row["operator_candidate_count"] = payload.get("operator_candidate_count")
    return row


def _deterministic_compiler_manifest_row(
    path_text: str,
    *,
    repo_root: Path,
    expected_archive_sha256: str,
    expected_archive_bytes: int | None,
) -> dict[str, Any]:
    row, payload = _json_artifact_row(
        path_text,
        repo_root=repo_root,
        artifact_kind="deterministic_compiler_identity_manifest",
    )
    blockers: list[str] = []
    if row["exists"] is not True or not payload:
        blockers.append("deterministic_compiler_manifest_missing_or_invalid_json")
    else:
        if payload.get("schema_version") != DETERMINISTIC_COMPILER_SCHEMA:
            blockers.append("deterministic_compiler_schema_mismatch")
        if payload.get("mode") != "identity":
            blockers.append("deterministic_compiler_mode_not_identity")
        if payload.get("target_profile") != "contest_one_video_replay":
            blockers.append("deterministic_compiler_target_profile_mismatch")
        archive_sha256 = str(payload.get("archive_sha256") or "").strip().lower()
        if expected_archive_sha256 and archive_sha256 != expected_archive_sha256:
            blockers.append("deterministic_compiler_archive_sha256_mismatch")
        archive_size = payload.get("archive_size_bytes")
        if (
            isinstance(expected_archive_bytes, int)
            and archive_size != expected_archive_bytes
        ):
            blockers.append("deterministic_compiler_archive_size_mismatch")
        if payload.get("blockers"):
            blockers.append("deterministic_compiler_blockers_present")
        blockers.extend(
            _false_authority_blockers(
                payload,
                prefix="deterministic_compiler",
                fields=(
                    "score_claim",
                    "promotion_eligible",
                    "ready_for_exact_eval_dispatch",
                ),
            )
        )
        no_op_proof = payload.get("no_op_proof")
        if not isinstance(no_op_proof, Mapping):
            blockers.append("deterministic_compiler_no_op_proof_missing")
        else:
            if no_op_proof.get("schema_version") != "deterministic_no_op_proof.v1":
                blockers.append("deterministic_compiler_no_op_schema_mismatch")
            if no_op_proof.get("mode") != "identity":
                blockers.append("deterministic_compiler_no_op_mode_not_identity")
            if no_op_proof.get("new_archive_sha256") != expected_archive_sha256:
                blockers.append("deterministic_compiler_no_op_archive_sha_mismatch")
            if no_op_proof.get("no_op_detector_passed") is not True:
                blockers.append("deterministic_compiler_no_op_detector_not_passed")
        # Per codex adversarial review 2026-05-19 F2: the v1 check
        # `isinstance(_, Mapping)` accepted `{}` as valid identity evidence.
        # v2 validates the full schema and binds archive_sha256 / runtime_tree_sha256
        # to the top-level manifest so a truncated or hand-authored manifest
        # cannot clear deterministic_compiler_identity.
        parser_manifest = payload.get("parser_section_manifest")
        parser_blockers = _validate_parser_section_manifest(parser_manifest)
        for blocker in parser_blockers:
            if blocker not in blockers:
                blockers.append(blocker)
        runtime_tree_sha = str(payload.get("runtime_tree_sha256") or "").strip().lower()
        golden_vectors = payload.get("golden_vectors")
        vector_blockers = _validate_golden_vectors(
            golden_vectors,
            expected_archive_sha256=str(
                payload.get("archive_sha256") or ""
            ).strip().lower(),
            expected_runtime_tree_sha256=(runtime_tree_sha or None),
        )
        for blocker in vector_blockers:
            if blocker not in blockers:
                blockers.append(blocker)
    row["deterministic_compiler_identity"] = row["exists"] is True and not blockers
    row["blockers"] = blockers
    row["score_claim"] = False
    row["promotion_eligible"] = False
    row["ready_for_exact_eval_dispatch"] = False
    if payload:
        row["mode"] = payload.get("mode")
        row["target_profile"] = payload.get("target_profile")
        row["archive_sha256"] = str(payload.get("archive_sha256") or "").strip().lower()
        row["archive_size_bytes"] = payload.get("archive_size_bytes")
        no_op_proof = payload.get("no_op_proof")
        if isinstance(no_op_proof, Mapping):
            row["no_op_detector_passed"] = no_op_proof.get("no_op_detector_passed")
    return row


def _packetir_identity_proof_row(
    path_text: str,
    *,
    repo_root: Path,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    row, payload = _json_artifact_row(
        path_text,
        repo_root=repo_root,
        artifact_kind="packetir_identity_proof",
    )
    row["score_claim"] = False
    row["promotion_eligible"] = False
    row["ready_for_exact_eval_dispatch"] = False
    if row["exists"] is not True or not payload:
        row["packetir_identity_passed"] = False
        row["blockers"] = ["packetir_identity_proof_missing_or_invalid"]
        return row
    blockers: list[str] = []
    if payload.get("packet_ir_identity_passed") is not True:
        blockers.append("packet_ir_identity_not_passed")
    archive_sha256 = str(payload.get("archive_sha256") or "").strip().lower()
    if expected_archive_sha256 and archive_sha256 != expected_archive_sha256:
        blockers.append("archive_sha256_mismatch")
    if payload.get("reemit_identity") is not True:
        blockers.append("identity_proof_reemit_identity_not_true")
    if payload.get("member_reemit_identity") is not True:
        blockers.append("identity_proof_member_reemit_identity_not_true")
    if payload.get("archive_reemit_identity") is not True:
        blockers.append("identity_proof_archive_reemit_identity_not_true")
    if payload.get("runtime_consumption_claim") is True:
        blockers.append("identity_proof_overclaims_runtime_consumption")
    if payload.get("full_frame_inflate_output_parity_claim") is True:
        blockers.append("identity_proof_overclaims_full_frame_parity")
    if payload.get("contest_axis_claim") is True:
        blockers.append("identity_proof_overclaims_contest_axis")
    if payload.get("score_claim") is True:
        blockers.append("identity_proof_makes_score_claim")
    if payload.get("promotion_eligible") is True:
        blockers.append("identity_proof_promotion_eligible")
    if payload.get("ready_for_exact_eval_dispatch") is True:
        blockers.append("identity_proof_ready_for_dispatch")
    row.update(
        {
            "packetir_identity_passed": not blockers,
            "blockers": blockers,
            "archive_sha256": archive_sha256,
            "member_name": payload.get("member_name", ""),
            "member_bytes": payload.get("member_bytes"),
            "member_sha256": payload.get("member_sha256", ""),
            "reemit_identity": payload.get("reemit_identity") is True,
            "member_reemit_identity": payload.get("member_reemit_identity") is True,
            "archive_reemit_identity": payload.get("archive_reemit_identity") is True,
            "runtime_consumption_claim": payload.get("runtime_consumption_claim")
            is True,
            "full_frame_inflate_output_parity_claim": payload.get(
                "full_frame_inflate_output_parity_claim"
            )
            is True,
            "contest_axis_claim": payload.get("contest_axis_claim") is True,
        }
    )
    return row


def _paired_exact_eval_authority(
    exact_eval_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [exact_eval_artifacts[axis] for axis in _REQUIRED_AXES]
    archives = {str(row.get("archive_sha256") or "") for row in rows}
    runtimes = {str(row.get("runtime_content_tree_sha256") or "") for row in rows}
    blockers: list[str] = []
    if any(row.get("valid_axis_evidence") is not True for row in rows):
        blockers.append("one_or_more_axis_evidence_rows_invalid")
    if len(archives) != 1 or "" in archives:
        blockers.append("paired_exact_archive_sha256_mismatch_or_missing")
    if len(runtimes) != 1 or "" in runtimes:
        blockers.append("paired_exact_runtime_content_sha256_mismatch_or_missing")
    return {
        "paired_exact_same_archive_runtime": not blockers,
        "blockers": blockers,
        "archive_sha256_values": sorted(archives),
        "runtime_content_tree_sha256_values": sorted(runtimes),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _next_actions(
    *,
    candidate_queue_present: bool,
    candidate_queue_generated: bool,
    candidate_byte_accounting_present: bool,
    parser_byte_accounting_present: bool,
    parser_profile_present: bool,
    runtime_consumption_proven: bool,
    deterministic_compiler_identity_present: bool,
) -> list[dict[str, Any]]:
    actions = [
        {
            "id": "run_compile_packet_identity_closure",
            "status": (
                "pending" if not deterministic_compiler_identity_present else "done"
            ),
            "description": (
                "Round-trip the exact PR101/FEC6 archive through the canonical "
                "deterministic compiler identity profile and persist the compiler "
                "manifest without dispatch or score promotion."
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        {
            "id": "generate_fec6_packetir_candidate_queue",
            "status": "done" if candidate_queue_generated else "pending",
            "description": (
                "Materialize a PR101/FEC6 PacketIR candidate queue from the FP11 "
                "wrapper sections and FEC6 selector stream."
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        {
            "id": "prove_parser_consumption_and_byte_accounting",
            "status": "done" if parser_byte_accounting_present else "pending",
            "description": (
                "For each candidate, record parser-consumption proof, member "
                "payload hashes, wrapper offsets, and all score-affecting bytes."
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        {
            "id": "prove_runtime_byte_consumption_noop_detector",
            "status": "done" if runtime_consumption_proven else "pending",
            "description": (
                "Prove candidate bytes are consumed by the runtime/inflate path, "
                "not only by the PacketIR parser; old identity no-op proofs do "
                "not satisfy this gate."
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        {
            "id": "local_identity_profile_smoke",
            "status": (
                "done"
                if candidate_byte_accounting_present and parser_profile_present
                else "pending"
                if candidate_byte_accounting_present
                else "blocked"
            ),
            "description": (
                "Run local parser/profile smoke on generated PacketIR candidates; "
                "keep CPU/CUDA exact-eval authority separate."
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        {
            "id": "paired_exact_eval_after_candidate_queue",
            "status": "blocked_until_candidate_queue_and_operator_authorization",
            "description": (
                "Only after candidate generation, local parser evidence, and an "
                "explicit lane claim should a future turn dispatch paired CPU/CUDA "
                "exact eval. This matrix emits no provider command."
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    ]
    return actions


def build_pr101_frontier_packetir_matrix(
    *,
    repo_root: str | Path | None = None,
    spec: PR101FEC6FrontierMatrixSpec = DEFAULT_PR101_FEC6_FRONTIER_MATRIX_SPEC,
) -> dict[str, Any]:
    """Build the PR101/FEC6 PacketIR authority matrix without dispatch."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    archive, expected_archive_sha = _archive_summary(spec=spec, repo_root=root)
    exact_eval_artifacts = {
        axis: _exact_eval_row(
            axis=axis,
            path_text=spec.exact_eval_paths.get(axis, ""),
            repo_root=root,
            expected_archive_sha256=expected_archive_sha,
        )
        for axis in _REQUIRED_AXES
    }
    paired_exact_eval = _paired_exact_eval_authority(exact_eval_artifacts)
    parser_profiles = _parser_profile_artifacts(spec.parser_profile_paths, repo_root=root)
    packetir_identity_proof = _packetir_identity_proof_row(
        spec.packetir_identity_proof_path,
        repo_root=root,
        expected_archive_sha256=expected_archive_sha,
    )
    deterministic_compiler_manifest = _deterministic_compiler_manifest_row(
        spec.deterministic_compiler_manifest_path,
        repo_root=root,
        expected_archive_sha256=expected_archive_sha,
        expected_archive_bytes=archive.get("bytes")
        if isinstance(archive.get("bytes"), int)
        else None,
    )
    candidate_queue = _candidate_queue_row(
        spec.candidate_queue_path,
        repo_root=root,
        expected_archive_sha256=expected_archive_sha,
        expected_archive_bytes=archive.get("bytes")
        if isinstance(archive.get("bytes"), int)
        else None,
    )
    has_cpu = exact_eval_artifacts["contest_cpu"].get("valid_axis_evidence") is True
    has_cuda = exact_eval_artifacts["contest_cuda"].get("valid_axis_evidence") is True
    packetir_identity_present = (
        packetir_identity_proof.get("packetir_identity_passed") is True
    )
    deterministic_compiler_identity_present = (
        deterministic_compiler_manifest.get("deterministic_compiler_identity") is True
    )
    candidate_queue_present = (
        candidate_queue.get("pr106_style_packetir_candidate_queue") is True
    )
    candidate_queue_generated = (
        candidate_queue.get("candidate_queue_generated") is True
    )
    candidate_byte_accounting_present = (
        candidate_queue.get("candidate_byte_accounting_present") is True
    )
    parser_byte_accounting_present = (
        candidate_queue.get("parser_byte_accounting_present") is True
    )
    runtime_consumption_proven = (
        candidate_queue.get("runtime_consumption_proven") is True
    )
    blockers: list[str] = []
    if archive.get("exists") is not True:
        blockers.append("fec6_archive_missing")
    if not has_cpu:
        blockers.append("contest_cpu_evidence_missing_or_blocked")
    if not has_cuda:
        blockers.append("contest_cuda_evidence_missing_or_blocked")
    if paired_exact_eval.get("paired_exact_same_archive_runtime") is not True:
        blockers.append("paired_exact_same_archive_runtime_missing_or_blocked")
    if not parser_profiles:
        blockers.append("parser_profile_artifacts_missing")
    if not packetir_identity_present:
        blockers.append("packetir_identity_proof_missing_or_blocked")
    if not deterministic_compiler_identity_present:
        blockers.append("deterministic_compiler_identity_manifest_missing")
    if not candidate_queue_present:
        blockers.append("pr106_style_packetir_candidate_queue_missing")
    if candidate_queue_generated and not parser_byte_accounting_present:
        blockers.append("packetir_candidate_parser_byte_accounting_missing_or_blocked")
    if parser_byte_accounting_present and not runtime_consumption_proven:
        blockers.append("packetir_candidate_runtime_consumption_missing_or_blocked")

    if (
        deterministic_compiler_identity_present
        and candidate_queue_present
        and candidate_byte_accounting_present
    ):
        current_authority = (
            "packetir_identity_deterministic_compiler_identity_and_candidate_queue_validated_no_score_claim"
        )
        status = "packetir_compiler_identity_and_candidate_queue_validated"
    elif deterministic_compiler_identity_present and candidate_queue_present:
        current_authority = (
            "packetir_identity_and_candidate_queue_present_needs_byte_accounting_review"
        )
        status = "packetir_candidate_queue_present_needs_review"
    elif (
        deterministic_compiler_identity_present
        and candidate_queue_generated
        and parser_byte_accounting_present
    ):
        current_authority = (
            "packetir_identity_compiler_identity_candidate_queue_parser_accounted_runtime_consumption_unproven_no_score_claim"
        )
        status = "packetir_candidate_queue_runtime_consumption_unproven"
    elif deterministic_compiler_identity_present:
        current_authority = (
            "parser_profile_exact_eval_and_compiler_identity_no_packetir_candidate_queue"
        )
        status = "parser_profile_no_packetir_candidate_queue"
    elif candidate_queue_present:
        current_authority = (
            "parser_profile_exact_eval_and_candidate_queue_no_compiler_identity"
        )
        status = "parser_profile_no_compiler_identity_candidate_queue_present_needs_review"
    else:
        current_authority = (
            "parser_profile_and_exact_eval_evidence_only_no_compiler_identity_no_packetir_candidate_queue"
        )
        status = "parser_profile_no_compiler_identity_no_packetir_candidate_queue"

    return {
        "schema": PR101_FRONTIER_PACKETIR_MATRIX_SCHEMA,
        "subject": "PR101/FEC6 frontier PacketIR authority matrix",
        "proof_scope": (
            "archive_exact_eval_parser_profile_join_no_candidate_generation_no_dispatch"
        ),
        "authority_summary": {
            "pr101_packet_compiler_packetir_primitives_are_general": True,
            "fec6_frontier_archive_present": archive.get("exists") is True,
            "fec6_has_contest_cpu_evidence": has_cpu,
            "fec6_has_contest_cuda_evidence": has_cuda,
            "fec6_has_paired_exact_same_archive_runtime": (
                paired_exact_eval.get("paired_exact_same_archive_runtime") is True
            ),
            "fec6_has_parser_profile_evidence": bool(parser_profiles),
            "fec6_has_packetir_identity_evidence": packetir_identity_present,
            "fec6_has_deterministic_compiler_identity_evidence": (
                deterministic_compiler_identity_present
            ),
            "fec6_has_pr106_style_packetir_candidate_queue": candidate_queue_present,
            "fec6_has_packetir_candidate_queue_artifact": candidate_queue_generated,
            "fec6_has_parser_byte_accounting_evidence": (
                parser_byte_accounting_present
            ),
            "fec6_has_runtime_consumption_evidence": runtime_consumption_proven,
            "fec6_has_candidate_byte_accounting_evidence": (
                candidate_byte_accounting_present
            ),
            "current_authority": current_authority,
        },
        "status": status,
        "blockers": blockers,
        "archive": archive,
        "exact_eval_artifacts": exact_eval_artifacts,
        "paired_exact_eval": paired_exact_eval,
        "parser_profile_artifacts": parser_profiles,
        "packetir_identity_proof": packetir_identity_proof,
        "deterministic_compiler_manifest": deterministic_compiler_manifest,
        "packetir_primitives": _packetir_primitives(root),
        "candidate_queue": candidate_queue,
        "next_actions": _next_actions(
            candidate_queue_present=candidate_queue_present,
            candidate_queue_generated=candidate_queue_generated,
            candidate_byte_accounting_present=candidate_byte_accounting_present,
            parser_byte_accounting_present=parser_byte_accounting_present,
            parser_profile_present=bool(parser_profiles),
            runtime_consumption_proven=runtime_consumption_proven,
            deterministic_compiler_identity_present=(
                deterministic_compiler_identity_present
            ),
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "dispatch_commands_emitted": False,
        "axis_semantics": {
            "contest_cpu": "kept separate from contest_cuda; no conversion",
            "contest_cuda": "kept separate from contest_cpu; no conversion",
        },
    }


def render_pr101_frontier_packetir_matrix_markdown(
    matrix: Mapping[str, Any],
) -> str:
    """Render a compact Markdown view of the PR101/FEC6 matrix."""

    summary = matrix.get("authority_summary")
    summary_map = summary if isinstance(summary, Mapping) else {}
    archive = matrix.get("archive")
    archive_map = archive if isinstance(archive, Mapping) else {}
    lines = [
        "# PR101/FEC6 frontier PacketIR authority matrix",
        "",
        f"Schema: `{matrix.get('schema')}`",
        "",
        "This is an audit artifact only: `score_claim=false`, "
        "`promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, "
        "and no dispatch commands are emitted.",
        "",
        "## Authority",
        "",
        f"- PR101 PacketIR primitives general: `{summary_map.get('pr101_packet_compiler_packetir_primitives_are_general')}`",
        f"- FEC6 frontier archive present: `{summary_map.get('fec6_frontier_archive_present')}`",
        f"- contest_cpu evidence: `{summary_map.get('fec6_has_contest_cpu_evidence')}`",
        f"- contest_cuda evidence: `{summary_map.get('fec6_has_contest_cuda_evidence')}`",
        f"- paired exact same archive/runtime: `{summary_map.get('fec6_has_paired_exact_same_archive_runtime')}`",
        f"- parser/profile evidence: `{summary_map.get('fec6_has_parser_profile_evidence')}`",
        f"- PacketIR identity evidence: `{summary_map.get('fec6_has_packetir_identity_evidence')}`",
        f"- deterministic compiler identity evidence: `{summary_map.get('fec6_has_deterministic_compiler_identity_evidence')}`",
        f"- PacketIR candidate queue artifact: `{summary_map.get('fec6_has_packetir_candidate_queue_artifact')}`",
        f"- PR106-style PacketIR candidate queue: `{summary_map.get('fec6_has_pr106_style_packetir_candidate_queue')}`",
        f"- parser byte accounting evidence: `{summary_map.get('fec6_has_parser_byte_accounting_evidence')}`",
        f"- runtime consumption evidence: `{summary_map.get('fec6_has_runtime_consumption_evidence')}`",
        f"- candidate byte accounting evidence: `{summary_map.get('fec6_has_candidate_byte_accounting_evidence')}`",
        "",
        "## Archive",
        "",
        "| path | bytes | sha256 | manifest match |",
        "|---|---:|---|---|",
        "| `{path}` | {bytes_} | `{sha}` | `{match}` |".format(
            path=archive_map.get("path", ""),
            bytes_=archive_map.get("bytes", ""),
            sha=archive_map.get("sha256", ""),
            match=archive_map.get("sha256_matches_manifest", ""),
        ),
        "",
        "## Exact Eval Evidence",
        "",
        "| axis | valid | score | archive bytes | artifact | artifact sha | runtime content sha | blockers |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    exact = matrix.get("exact_eval_artifacts")
    if isinstance(exact, Mapping):
        for axis in _REQUIRED_AXES:
            row = exact.get(axis)
            if not isinstance(row, Mapping):
                continue
            blockers = ", ".join(str(item) for item in row.get("blockers", [])) or "-"
            lines.append(
                "| `{axis}` | `{valid}` | {score} | {bytes_} | `{path}` | `{sha}` | `{runtime}` | {blockers} |".format(
                    axis=axis,
                    valid=row.get("valid_axis_evidence"),
                    score=row.get("canonical_score", ""),
                    bytes_=row.get("archive_size_bytes", ""),
                    path=row.get("path", ""),
                    sha=row.get("sha256", ""),
                    runtime=row.get("runtime_content_tree_sha256", ""),
                    blockers=blockers,
                )
            )
    lines.extend(
        [
            "",
            "## Parser/Profile Artifacts",
            "",
            "| artifact | bytes | sha256 |",
            "|---|---:|---|",
        ]
    )
    for row in matrix.get("parser_profile_artifacts", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| `{path}` | {bytes_} | `{sha}` |".format(
                path=row.get("path", ""),
                bytes_=row.get("bytes", ""),
                sha=row.get("sha256", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "| id | status | non-promotional flags |",
            "|---|---|---|",
        ]
    )
    for action in matrix.get("next_actions", []):
        if not isinstance(action, Mapping):
            continue
        flags = (
            f"score_claim={str(action.get('score_claim')).lower()}, "
            f"promotion_eligible={str(action.get('promotion_eligible')).lower()}, "
            "ready_for_exact_eval_dispatch="
            f"{str(action.get('ready_for_exact_eval_dispatch')).lower()}"
        )
        lines.append(
            "| `{id_}` | `{status}` | {flags} |".format(
                id_=action.get("id", ""),
                status=action.get("status", ""),
                flags=flags,
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_pr101_frontier_packetir_matrix(
    *,
    output_json: str | Path = PR101_FRONTIER_PACKETIR_MATRIX_DEFAULT_JSON,
    output_md: str | Path = PR101_FRONTIER_PACKETIR_MATRIX_DEFAULT_MD,
    repo_root: str | Path | None = None,
    spec: PR101FEC6FrontierMatrixSpec = DEFAULT_PR101_FEC6_FRONTIER_MATRIX_SPEC,
) -> dict[str, Any]:
    """Build and write the PR101/FEC6 PacketIR matrix JSON and Markdown."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    matrix = build_pr101_frontier_packetir_matrix(repo_root=root, spec=spec)
    json_path = _resolve(output_json, root)
    md_path = _resolve(output_md, root)
    write_json(json_path, matrix)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_pr101_frontier_packetir_matrix_markdown(matrix),
        encoding="utf-8",
    )
    matrix["artifact_paths"] = {
        "json": repo_relative(json_path, root),
        "markdown": repo_relative(md_path, root),
    }
    # A JSON file cannot honestly embed its own final SHA-256 without an
    # external manifest. Store only hashes that are stable inside this payload;
    # callers may hash the written JSON after this function returns.
    matrix["artifact_sha256"] = {
        "json": None,
        "json_omitted_reason": "self_referential_json_hash_requires_external_manifest",
        "markdown": sha256_file(md_path),
    }
    write_json(json_path, matrix)
    matrix["written_artifact_sha256"] = {
        "json": sha256_file(json_path),
        "markdown": sha256_file(md_path),
    }
    return matrix


def matrix_json_text(matrix: Mapping[str, Any]) -> str:
    """Return canonical JSON text for CLI/stdout callers."""

    return json_text(matrix)
