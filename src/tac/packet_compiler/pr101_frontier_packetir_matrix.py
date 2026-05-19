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

from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json

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


def _candidate_queue_row(
    path_text: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    row = _artifact_row(
        path_text,
        repo_root=repo_root,
        artifact_kind="pr101_fec6_packetir_candidate_queue",
    )
    row["pr106_style_packetir_candidate_queue"] = row["exists"] is True
    row["score_claim"] = False
    row["promotion_eligible"] = False
    row["ready_for_exact_eval_dispatch"] = False
    return row


def _deterministic_compiler_manifest_row(
    path_text: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    row = _artifact_row(
        path_text,
        repo_root=repo_root,
        artifact_kind="deterministic_compiler_identity_manifest",
    )
    row["deterministic_compiler_identity"] = row["exists"] is True
    row["score_claim"] = False
    row["promotion_eligible"] = False
    row["ready_for_exact_eval_dispatch"] = False
    return row


def _next_actions(
    *,
    candidate_queue_present: bool,
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
            "status": "pending" if not candidate_queue_present else "done",
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
            "status": "pending",
            "description": (
                "For each candidate, record parser-consumption proof, member "
                "payload hashes, wrapper offsets, and all score-affecting bytes."
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        {
            "id": "local_identity_profile_smoke",
            "status": "pending",
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
    parser_profiles = _parser_profile_artifacts(spec.parser_profile_paths, repo_root=root)
    deterministic_compiler_manifest = _deterministic_compiler_manifest_row(
        spec.deterministic_compiler_manifest_path,
        repo_root=root,
    )
    candidate_queue = _candidate_queue_row(spec.candidate_queue_path, repo_root=root)
    has_cpu = exact_eval_artifacts["contest_cpu"].get("valid_axis_evidence") is True
    has_cuda = exact_eval_artifacts["contest_cuda"].get("valid_axis_evidence") is True
    deterministic_compiler_identity_present = (
        deterministic_compiler_manifest.get("deterministic_compiler_identity") is True
    )
    candidate_queue_present = (
        candidate_queue.get("pr106_style_packetir_candidate_queue") is True
    )
    blockers: list[str] = []
    if archive.get("exists") is not True:
        blockers.append("fec6_archive_missing")
    if not has_cpu:
        blockers.append("contest_cpu_evidence_missing_or_blocked")
    if not has_cuda:
        blockers.append("contest_cuda_evidence_missing_or_blocked")
    if not parser_profiles:
        blockers.append("parser_profile_artifacts_missing")
    if not deterministic_compiler_identity_present:
        blockers.append("deterministic_compiler_identity_manifest_missing")
    if not candidate_queue_present:
        blockers.append("pr106_style_packetir_candidate_queue_missing")

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
            "fec6_has_parser_profile_evidence": bool(parser_profiles),
            "fec6_has_deterministic_compiler_identity_evidence": (
                deterministic_compiler_identity_present
            ),
            "fec6_has_pr106_style_packetir_candidate_queue": candidate_queue_present,
            "current_authority": (
                "parser_profile_and_exact_eval_evidence_only_no_compiler_identity_no_packetir_candidate_queue"
            ),
        },
        "status": (
            "parser_profile_no_compiler_identity_no_packetir_candidate_queue"
            if not deterministic_compiler_identity_present
            and not candidate_queue_present
            else "parser_profile_no_compiler_identity_candidate_queue_present_needs_review"
            if not deterministic_compiler_identity_present
            else "parser_profile_no_packetir_candidate_queue"
            if not candidate_queue_present
            else "packetir_candidate_queue_present_needs_review"
        ),
        "blockers": blockers,
        "archive": archive,
        "exact_eval_artifacts": exact_eval_artifacts,
        "parser_profile_artifacts": parser_profiles,
        "deterministic_compiler_manifest": deterministic_compiler_manifest,
        "packetir_primitives": _packetir_primitives(root),
        "candidate_queue": candidate_queue,
        "next_actions": _next_actions(
            candidate_queue_present=candidate_queue_present,
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
        f"- parser/profile evidence: `{summary_map.get('fec6_has_parser_profile_evidence')}`",
        f"- deterministic compiler identity evidence: `{summary_map.get('fec6_has_deterministic_compiler_identity_evidence')}`",
        f"- PR106-style PacketIR candidate queue: `{summary_map.get('fec6_has_pr106_style_packetir_candidate_queue')}`",
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
