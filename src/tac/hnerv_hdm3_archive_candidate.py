"""Deterministic HDM3 HNeRV decoder-section archive candidates.

HDM3 is a byte-level decoder-section fixture: fixed-schema q-stream Brotli plus
raw scale bytes. This module can build a deterministic archive that swaps only
the HNeRV decoder section to HDM3 and proves raw decoder equivalence. It does
not make the archive scorer-ready; the submission runtime must consume HDM3
before exact CUDA dispatch is valid.
"""

from __future__ import annotations

import importlib.util
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_decoder_recode import (
    decode_hdm3_q_brotli_split_fixture,
    decode_hdm4_q_brotli_split_fixture,
    decode_hdm6_q_brotli_tuned_fixture,
    decode_hdm7_q_brotli_len_elided_fixture,
    encode_hdm3_q_brotli_split_fixture,
    encode_hdm4_q_brotli_split_fixture,
    encode_hdm6_q_brotli_tuned_fixture,
    encode_hdm7_q_brotli_len_elided_fixture,
    parse_decoder_section_for_recode,
)
from tac.hnerv_lowlevel_packer import (
    PackedHnervPayload,
    read_packed_archive_view,
    sha256_bytes,
)
from tac.repo_io import read_json, repo_relative, sha256_file, write_json

SCHEMA_VERSION = 1
TOOL = "tac.hnerv_hdm3_archive_candidate.build_hdm3_archive_candidate"
HDM3_VARIANT_NAME = "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales"
HDM4_VARIANT_NAME = "hdm4_q_brotli_split_fixed_recipe_dp4_plus_raw_scales"
HDM6_VARIANT_NAME = "hdm6_q_brotli_split_fixed_recipe_tuned_lgwin_plus_raw_scales"
HDM7_VARIANT_NAME = "hdm7_q_brotli_split_fixed_recipe_tuned_lgwin_final_len_elided_plus_raw_scales"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
EXACT_PACKET_READINESS_FILENAME = "hdm3_exact_eval_packet_readiness.json"
STATIC_RELEASE_SURFACE_DIRNAME = "exact_eval_static_release_surface"
STATIC_COMPLIANCE_FILENAMES = (
    "pre_submission_compliance.static.json",
    "pre_submission_compliance.json",
)
RUNTIME_ADAPTER_PROOF_FILENAMES = (
    "runtime_adapter_proof.with_tool_run.json",
    "runtime_adapter_proof.json",
)
RUNTIME_TREE_CLOSURE_FILENAME = "hdm3_runtime_tree_closure.json"
HDM3_EXACT_EVAL_LANE_ID = "hnerv_hdm3_q_brotli_split_exact_eval"
HDM4_EXACT_EVAL_LANE_ID = "hnerv_hdm4_q_brotli_split_exact_eval"
HDM6_EXACT_EVAL_LANE_ID = "hnerv_hdm6_q_brotli_tuned_exact_eval"
HDM7_EXACT_EVAL_LANE_ID = "hnerv_hdm7_final_len_elided_exact_eval"
HDM3_RUNTIME_INFLATE = "submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh"
HDM3_UPSTREAM_DIR = "upstream"
CONTEST_AUTH_EVAL_RUNTIME_MANIFEST_SOURCE = (
    "experiments/contest_auth_eval.py::_runtime_dependency_manifest"
)


class HnervHdm3ArchiveCandidateError(ValueError):
    """Raised when an HDM3 archive candidate input is invalid."""


def _decoder_recode_variant(name: str) -> dict[str, Any]:
    key = name.strip().lower()
    if key == "hdm3":
        return {
            "key": "hdm3",
            "variant_name": HDM3_VARIANT_NAME,
            "candidate_id": "pr106x_hdm3_decoder_recode_14byte",
            "lane_id": HDM3_EXACT_EVAL_LANE_ID,
            "encode": encode_hdm3_q_brotli_split_fixture,
            "decode": decode_hdm3_q_brotli_split_fixture,
        }
    if key == "hdm4":
        return {
            "key": "hdm4",
            "variant_name": HDM4_VARIANT_NAME,
            "candidate_id": "pr106_r2_hdm4_decoder_recode_dp4_123byte",
            "lane_id": HDM4_EXACT_EVAL_LANE_ID,
            "encode": encode_hdm4_q_brotli_split_fixture,
            "decode": decode_hdm4_q_brotli_split_fixture,
        }
    if key == "hdm6":
        return {
            "key": "hdm6",
            "variant_name": HDM6_VARIANT_NAME,
            "candidate_id": "pr106_r2_hdm6_decoder_tuned_brotli_3byte",
            "lane_id": HDM6_EXACT_EVAL_LANE_ID,
            "encode": encode_hdm6_q_brotli_tuned_fixture,
            "decode": decode_hdm6_q_brotli_tuned_fixture,
        }
    if key == "hdm7":
        return {
            "key": "hdm7",
            "variant_name": HDM7_VARIANT_NAME,
            "candidate_id": "pr106_r2_hdm7_decoder_final_len_elided_3byte",
            "lane_id": HDM7_EXACT_EVAL_LANE_ID,
            "encode": encode_hdm7_q_brotli_len_elided_fixture,
            "decode": decode_hdm7_q_brotli_len_elided_fixture,
        }
    raise HnervHdm3ArchiveCandidateError(
        f"unsupported decoder recode variant {name!r}; expected hdm3, hdm4, hdm6, or hdm7"
    )


def build_hdm3_archive_candidate(
    *,
    source_archive: str | Path,
    output_dir: str | Path,
    source_label: str,
    decoder_recode_variant: str = "hdm3",
    allow_rate_regression: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic HDM3 decoder-section archive candidate.

    The returned manifest may be archive-byte-closed, but it always keeps
    ``ready_for_exact_eval_dispatch`` false until the runtime adapter and exact
    CUDA custody proofs exist.
    """

    source_path = Path(source_archive)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()

    view = read_packed_archive_view(source_path)
    source = view.archive
    packed = view.packed
    try:
        parsed_decoder, source_raw, source_decoder_codec = parse_decoder_section_for_recode(
            packed.decoder_packed_brotli
        )
    except (brotli.error, ValueError) as exc:
        raise HnervHdm3ArchiveCandidateError(
            f"source decoder parser does not support current section: {exc}"
        ) from exc
    if parsed_decoder.to_raw() != source_raw:
        raise HnervHdm3ArchiveCandidateError("source decoder parser does not match raw decoder")

    variant = _decoder_recode_variant(decoder_recode_variant)
    recoded_stream, recode_stats = variant["encode"](parsed_decoder)
    restored_decoder = variant["decode"](recoded_stream)
    raw_equal = restored_decoder.to_raw() == source_raw
    q_equal = restored_decoder.q_stream == parsed_decoder.q_stream
    scale_equal = restored_decoder.scale_stream == parsed_decoder.scale_stream
    section_byte_delta = len(recoded_stream) - len(packed.decoder_packed_brotli)
    rate_positive = section_byte_delta < 0
    archive_build_blockers: list[str] = []
    if not raw_equal:
        archive_build_blockers.append(f"{variant['key']}_raw_decoder_mismatch")
    if not q_equal:
        archive_build_blockers.append(f"{variant['key']}_q_stream_mismatch")
    if not scale_equal:
        archive_build_blockers.append(f"{variant['key']}_scale_stream_mismatch")
    if not rate_positive and not allow_rate_regression:
        archive_build_blockers.append(f"{variant['key']}_decoder_section_not_rate_positive")

    candidate_archive_path: Path | None = None
    candidate_payload: bytes | None = None
    candidate_archive_sha = ""
    candidate_archive_bytes: int | None = None
    candidate_payload_sha = ""
    ready_for_archive_preflight = False
    if not archive_build_blockers:
        candidate_packed = PackedHnervPayload(
            header=packed.header,
            decoder_packed_brotli=recoded_stream,
            latents_and_sidecar_brotli=packed.latents_and_sidecar_brotli,
            header_format=packed.header_format,
        )
        candidate_payload = view.emit_payload(candidate_packed)
        candidate_archive_path = (
            output_root / f"{_slug(source_label)}_{variant['key']}_archive_candidate.zip"
        )
        view.write_archive(candidate_archive_path, candidate_payload)
        candidate_archive_sha = sha256_file(candidate_archive_path)
        candidate_archive_bytes = candidate_archive_path.stat().st_size
        candidate_payload_sha = sha256_bytes(candidate_payload)
        checked_view = read_packed_archive_view(candidate_archive_path)
        checked = checked_view.packed
        if checked.decoder_packed_brotli != recoded_stream:
            archive_build_blockers.append(f"candidate_decoder_section_not_{variant['key']}_stream")
        if checked.latents_and_sidecar_brotli != packed.latents_and_sidecar_brotli:
            archive_build_blockers.append("candidate_latents_section_changed")
        if candidate_archive_sha == source.archive_sha256:
            archive_build_blockers.append("candidate_archive_sha256_unchanged")
        if candidate_payload_sha == sha256_bytes(source.payload):
            archive_build_blockers.append("candidate_payload_sha256_unchanged")
        ready_for_archive_preflight = not archive_build_blockers

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "contract": "hnerv_hdm3_archive_candidate_manifest_v1",
        "candidate_id": variant["candidate_id"],
        "lane_id": variant["lane_id"],
        "family": "hnerv_hdm_decoder_recode",
        "pareto_scope": "hnerv_rate_only_exact_archive",
        "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_archive_preflight": ready_for_archive_preflight,
        "archive_build_gate": ready_for_archive_preflight,
        "source_label": source_label,
        "source_archive_path": repo_relative(source_path, repo),
        "source_archive_sha256": source.archive_sha256,
        "source_archive_bytes": source.archive_bytes,
        "source_member_name": source.member_name,
        "source_member_bytes": source.member_bytes,
        "source_payload_kind": view.payload_kind,
        "source_payload_sha256": sha256_bytes(source.payload),
        "source_payload_bytes": len(source.payload),
        "source_hnerv_payload_sha256": sha256_bytes(view.hnerv_payload),
        "source_hnerv_payload_bytes": len(view.hnerv_payload),
        "source_decoder_section_sha256": sha256_bytes(packed.decoder_packed_brotli),
        "source_decoder_section_bytes": len(packed.decoder_packed_brotli),
        "source_decoder_section_codec": source_decoder_codec,
        "source_decoder_raw_sha256": sha256_bytes(source_raw),
        "source_decoder_raw_bytes": len(source_raw),
        "latents_and_sidecar_section_sha256": sha256_bytes(packed.latents_and_sidecar_brotli),
        "latents_and_sidecar_section_bytes": len(packed.latents_and_sidecar_brotli),
        "candidate_variant": variant["variant_name"],
        "candidate_decoder_recode_key": variant["key"],
        "candidate_decoder_section_sha256": sha256_bytes(recoded_stream),
        "candidate_decoder_section_bytes": len(recoded_stream),
        "candidate_decoder_section_byte_delta": section_byte_delta,
        "candidate_rate_positive": rate_positive,
        "candidate_rate_score_delta_if_runtime_supported_and_components_equal": round(
            section_byte_delta * (25 / 37_545_489),
            12,
        ),
        "candidate_payload_sha256": candidate_payload_sha,
        "candidate_payload_bytes": len(candidate_payload) if candidate_payload is not None else None,
        "candidate_hnerv_payload_sha256": (
            sha256_bytes(candidate_packed.to_bytes()) if candidate_payload is not None else ""
        ),
        "candidate_hnerv_payload_bytes": (
            len(candidate_packed.to_bytes()) if candidate_payload is not None else None
        ),
        "candidate_archive_path": (
            repo_relative(candidate_archive_path, repo) if candidate_archive_path is not None else ""
        ),
        "candidate_archive_sha256": candidate_archive_sha,
        "candidate_archive_bytes": candidate_archive_bytes,
        "candidate_archive_byte_delta": (
            candidate_archive_bytes - source.archive_bytes if candidate_archive_bytes is not None else None
        ),
        "candidate_member_name": source.member_name if candidate_archive_path is not None else "",
        "decoder_recode_stats": recode_stats,
        "hdm3_stats": recode_stats if variant["key"] == "hdm3" else {},
        "hdm4_stats": recode_stats if variant["key"] == "hdm4" else {},
        "hdm6_stats": recode_stats if variant["key"] == "hdm6" else {},
        "hdm7_stats": recode_stats if variant["key"] == "hdm7" else {},
        "decoder_raw_equivalence": {
            "contract": "hnerv_hdm_decoder_raw_equivalence_v1",
            "decoder_recode_key": variant["key"],
            "source_decoder_raw_sha256": sha256_bytes(source_raw),
            "restored_decoder_raw_sha256": sha256_bytes(restored_decoder.to_raw()),
            "raw_equal": raw_equal,
            "q_roundtrip_equal": q_equal,
            "scale_roundtrip_equal": scale_equal,
        },
        "section_replacement_proof": {
            "contract": "hnerv_hdm_single_section_replacement_v1",
            "decoder_recode_key": variant["key"],
            "replaced_section": "decoder_packed_brotli",
            "source_payload_kind": view.payload_kind,
            "header_rewritten": True,
            "latents_and_sidecar_preserved": True,
            "zip_member_preserved": candidate_archive_path is not None,
            "byte_different_archive": bool(candidate_archive_sha and candidate_archive_sha != source.archive_sha256),
            "byte_different_payload": bool(candidate_payload_sha and candidate_payload_sha != sha256_bytes(source.payload)),
        },
        "archive_build_blockers": archive_build_blockers,
    }
    if candidate_archive_path is not None and ready_for_archive_preflight:
        manifest["exact_eval_release_surface"] = materialize_hdm3_exact_eval_release_surface(
            manifest=manifest,
            output_dir=output_root,
            repo_root=repo,
        )
    readiness = build_hdm3_exact_eval_packet_readiness(
        manifest,
        output_dir=output_root,
        repo_root=repo,
        write=True,
    )
    manifest["runtime_adapter_proof"] = readiness["runtime_adapter_payload_identity"]
    manifest["runtime_tree_inflate_output_parity"] = readiness[
        "runtime_tree_inflate_output_parity"
    ]
    manifest["fixed_runtime_preflight"] = readiness["fixed_runtime_preflight"]
    manifest["exact_eval_packet_readiness"] = {
        "contract": "hnerv_hdm3_exact_eval_packet_readiness_link_v1",
        "path": repo_relative(output_root / EXACT_PACKET_READINESS_FILENAME, repo),
        "sha256": sha256_file(output_root / EXACT_PACKET_READINESS_FILENAME),
        "static_packet_ready": readiness["static_packet_ready"],
        "ready_for_exact_eval_dispatch": readiness["ready_for_exact_eval_dispatch"],
        "remaining_dispatch_blockers": readiness["dispatch_blockers"],
    }
    manifest["dispatch_blockers"] = list(readiness["dispatch_blockers"])
    write_json(output_root / "hdm3_archive_candidate_manifest.json", manifest)
    readiness = build_hdm3_exact_eval_packet_readiness(
        manifest,
        output_dir=output_root,
        repo_root=repo,
        write=True,
    )
    manifest["runtime_adapter_proof"] = readiness["runtime_adapter_payload_identity"]
    manifest["runtime_tree_inflate_output_parity"] = readiness[
        "runtime_tree_inflate_output_parity"
    ]
    manifest["fixed_runtime_preflight"] = readiness["fixed_runtime_preflight"]
    manifest["exact_eval_packet_readiness"]["sha256"] = sha256_file(
        output_root / EXACT_PACKET_READINESS_FILENAME
    )
    manifest["exact_eval_packet_readiness"]["static_packet_ready"] = readiness[
        "static_packet_ready"
    ]
    manifest["exact_eval_packet_readiness"]["ready_for_exact_eval_dispatch"] = readiness[
        "ready_for_exact_eval_dispatch"
    ]
    manifest["exact_eval_packet_readiness"]["remaining_dispatch_blockers"] = list(
        readiness["dispatch_blockers"]
    )
    manifest["dispatch_blockers"] = list(readiness["dispatch_blockers"])
    write_json(output_root / "hdm3_archive_candidate_manifest.json", manifest)
    return manifest


def materialize_hdm3_exact_eval_release_surface(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a static local release surface for strict packet checks.

    This is not a contest submission and does not include auth-eval output. It
    exists so the static file/ZIP/report surface can be checked before any GPU
    dispatch is considered.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    release_dir = Path(output_dir) / STATIC_RELEASE_SURFACE_DIRNAME
    release_dir.mkdir(parents=True, exist_ok=True)
    archive_src = _resolve_repo_path(manifest.get("candidate_archive_path"), repo)
    archive_dst = release_dir / "archive.zip"
    shutil.copyfile(archive_src, archive_dst)

    inflate_path = release_dir / "inflate.sh"
    inflate_path.write_text(_release_surface_inflate_wrapper_text(), encoding="utf-8")
    current_mode = stat.S_IMODE(inflate_path.stat().st_mode)
    inflate_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    archive_manifest_path = release_dir / "archive_manifest.json"
    archive_manifest = {
        "schema": "hnerv_hdm3_static_release_surface_archive_manifest_v1",
        "schema_version": SCHEMA_VERSION,
        "score_claim": False,
        "dispatch_attempted": False,
        "candidate_id": manifest.get("candidate_id"),
        "lane_id": manifest.get("lane_id"),
        "archive_sha256": manifest.get("candidate_archive_sha256"),
        "archive_bytes": manifest.get("candidate_archive_bytes"),
        "candidate_archive_sha256": manifest.get("candidate_archive_sha256"),
        "candidate_archive_bytes": manifest.get("candidate_archive_bytes"),
        "candidate_member_name": manifest.get("candidate_member_name"),
        "candidate_payload_sha256": manifest.get("candidate_payload_sha256"),
        "source_archive_sha256": manifest.get("source_archive_sha256"),
        "source_archive_bytes": manifest.get("source_archive_bytes"),
        "source_payload_sha256": manifest.get("source_payload_sha256"),
        "runtime_adapter": HDM3_RUNTIME_INFLATE,
        "runtime_adapter_payload_identity_proof_required": True,
        "exact_cuda_auth_eval_required_before_score_claim": True,
    }
    write_json(archive_manifest_path, archive_manifest)

    report_path = release_dir / "report.txt"
    report_path.write_text(
        _release_surface_report_text(manifest),
        encoding="utf-8",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_static_release_surface_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "path": repo_relative(release_dir, repo),
        "archive_path": repo_relative(archive_dst, repo),
        "archive_sha256": sha256_file(archive_dst),
        "archive_bytes": archive_dst.stat().st_size,
        "inflate_sh": repo_relative(inflate_path, repo),
        "report_txt": repo_relative(report_path, repo),
        "archive_manifest_json": repo_relative(archive_manifest_path, repo),
        "strict_pre_submission_compliance_command": [
            ".venv/bin/python",
            "scripts/pre_submission_compliance_check.py",
            "--submission-dir",
            repo_relative(release_dir, repo),
            "--archive",
            repo_relative(archive_dst, repo),
            "--archive-manifest-json",
            repo_relative(archive_manifest_path, repo),
            "--expect-single-member",
            str(manifest.get("candidate_member_name") or ""),
            "--expected-archive-sha256",
            str(manifest.get("candidate_archive_sha256") or ""),
            "--expected-archive-size-bytes",
            str(manifest.get("candidate_archive_bytes") or ""),
            "--public-scan-path",
            repo_relative(release_dir, repo),
            "--json-out",
            repo_relative(Path(output_dir) / STATIC_COMPLIANCE_FILENAMES[0], repo),
            "--strict",
        ],
    }


def build_hdm3_exact_eval_packet_readiness(
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    runtime_adapter_proof_path: str | Path | None = None,
    strict_pre_submission_compliance_path: str | Path | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Build the operator-facing HDM3 exact-eval packet readiness artifact.

    The artifact can mark the local static packet as ready only when the archive
    build, runtime payload-identity proof, and strict static compliance report
    are all closed. It never marks exact GPU dispatch ready; Level-2 lane claim
    and exact CUDA auth eval remain explicit blockers.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    output_root = Path(output_dir)
    runtime_identity = _runtime_adapter_payload_identity_evidence(
        manifest,
        output_dir=output_root,
        repo_root=repo,
        runtime_adapter_proof_path=runtime_adapter_proof_path,
    )
    static_compliance = _strict_static_compliance_evidence(
        manifest,
        output_dir=output_root,
        repo_root=repo,
        strict_pre_submission_compliance_path=strict_pre_submission_compliance_path,
    )
    release_surface = manifest.get("exact_eval_release_surface")
    if not isinstance(release_surface, Mapping):
        release_surface = {
            "contract": "hnerv_hdm3_static_release_surface_v1",
            "present": False,
            "blockers": ["exact_eval_static_release_surface_missing"],
        }

    archive_build_blockers = _string_list(manifest.get("archive_build_blockers"))
    runtime_tree_contract = build_hdm3_runtime_tree_inflate_output_parity_contract(
        manifest=manifest,
        runtime_adapter_payload_identity=runtime_identity,
        output_dir=output_root,
        repo_root=repo,
        write=write,
    )
    fixed_runtime_preflight = _fixed_runtime_preflight(runtime_tree_contract)
    static_blockers = _ordered_unique(
        [
            *archive_build_blockers,
            *runtime_identity["blockers"],
            *runtime_tree_contract["blockers"],
            *static_compliance["blockers"],
            *_string_list(release_surface.get("blockers")),
        ]
    )
    static_packet_ready = not static_blockers
    dispatch_blockers = _ordered_unique(
        [
            *static_blockers,
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    decoder_recode_key = str(manifest.get("candidate_decoder_recode_key") or "hdm3")
    decoder_label = decoder_recode_key.upper()
    byte_delta = _optional_int(manifest.get("candidate_archive_bytes")) - _optional_int(
        manifest.get("source_archive_bytes")
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_exact_eval_packet_readiness_v1",
        "packet_kind": "hnerv_hdm3_exact_eval_operator_packet_readiness",
        "tool": TOOL,
        "candidate_id": manifest.get("candidate_id"),
        "decoder_recode_key": manifest.get("candidate_decoder_recode_key") or "hdm3",
        "lane_id": manifest.get("lane_id"),
        "family": manifest.get("family"),
        "pareto_scope": manifest.get("pareto_scope"),
        "evidence_grade": (
            "empirical_archive_candidate_runtime_adapter_parity_static_packet_ready"
            if static_packet_ready
            else "empirical_archive_candidate_static_packet_blocked"
        ),
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "ready_for_exact_eval_packet": static_packet_ready,
        "static_packet_ready": static_packet_ready,
        "candidate_static_preflight_ready": static_packet_ready,
        "ready_for_exact_eval_dispatch": False,
        "readiness_scope": (
            "static local exact-eval packet only; no lane claim, no remote GPU "
            "dispatch, no CUDA auth eval, and no score claim"
        ),
        "dispatch_gate": (
            "blocked_until_level2_lane_claim_and_exact_cuda_auth_eval"
        ),
        "byte_delta": byte_delta,
        "expected_total_score_delta_rate_only_if_components_equal": (
            byte_delta * RATE_SCORE_PER_BYTE
        ),
        "expected_seg_dist_delta": 0.0,
        "expected_pose_dist_delta": 0.0,
        "source_archive_path": manifest.get("source_archive_path"),
        "source_archive_sha256": manifest.get("source_archive_sha256"),
        "source_archive_bytes": manifest.get("source_archive_bytes"),
        "candidate_archive_path": manifest.get("candidate_archive_path"),
        "candidate_archive_sha256": manifest.get("candidate_archive_sha256"),
        "candidate_archive_bytes": manifest.get("candidate_archive_bytes"),
        "candidate_member_name": manifest.get("candidate_member_name"),
        "candidate_payload_sha256": manifest.get("candidate_payload_sha256"),
        "candidate_payload_bytes": manifest.get("candidate_payload_bytes"),
        "runtime_adapter_payload_identity": runtime_identity,
        "runtime_tree_inflate_output_parity": runtime_tree_contract,
        "fixed_runtime_preflight": fixed_runtime_preflight,
        "static_release_surface": dict(release_surface),
        "strict_static_compliance": static_compliance,
        "static_blockers": static_blockers,
        "dispatch_blockers": dispatch_blockers,
        "score_blockers": [
            "exact_cuda_auth_eval_missing",
            "contest_auth_eval_adjudication_missing",
            "operator_score_claim_review_missing",
        ],
        "lane_dispatch_claim": {
            "required_before_gpu": True,
            "active_claim_present": False,
            "claims_path": ".omx/state/active_lane_dispatch_claims.md",
            "claim_command_template": [
                "tools/claim_lane_dispatch.py",
                "claim",
                "--lane-id",
                str(manifest.get("lane_id") or HDM3_EXACT_EVAL_LANE_ID),
                "--platform",
                "<lightning|vast|modal>",
                "--instance-job-id",
                "<job-id>",
                "--agent",
                "<agent>",
                "--status",
                "eval",
                "--notes",
                f"{decoder_label} exact CUDA auth eval for candidate archive "
                f"{manifest.get('candidate_archive_sha256')}",
            ],
        },
        "exact_cuda_auth_eval": {
            "required_before_score_claim": True,
            "present": False,
            "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
            "command_template": [
                ".venv/bin/python",
                "experiments/contest_auth_eval.py",
                "--archive",
                str(manifest.get("candidate_archive_path") or "<candidate-archive.zip>"),
                "--inflate-sh",
                HDM3_RUNTIME_INFLATE,
                "--upstream-dir",
                "upstream",
                "--device",
                "cuda",
            ],
        },
        "operator_next_steps": [
            "If pursuing exact eval, record a Level-2 lane claim before any GPU job.",
            "Run exact CUDA auth eval on the exact candidate archive bytes.",
            "Only after exact CUDA JSON exists, run contest-final compliance and adjudication.",
        ],
    }
    if write:
        write_json(output_root / EXACT_PACKET_READINESS_FILENAME, packet)
    return packet


def build_hdm3_runtime_tree_inflate_output_parity_contract(
    *,
    manifest: Mapping[str, Any],
    runtime_adapter_payload_identity: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Build the non-GPU HDM3 runtime-tree and inflate-parity contract.

    This closes only the fixed runtime tree and the adapter input payload
    identity proof. It does not execute CUDA auth eval and cannot establish a
    score or component parity claim.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    output_root = Path(output_dir)
    blockers: list[str] = []
    runtime_manifest: Mapping[str, Any] = {}
    try:
        runtime_manifest = _canonical_runtime_dependency_manifest(
            repo / HDM3_RUNTIME_INFLATE,
            repo / HDM3_UPSTREAM_DIR,
            repo_root=repo,
        )
    except HnervHdm3ArchiveCandidateError as exc:
        blockers.append(f"hdm3_runtime_tree_manifest_failed:{exc}")

    runtime_tree_sha256 = str(runtime_manifest.get("runtime_tree_sha256") or "")
    decoder_recode_key = str(manifest.get("candidate_decoder_recode_key") or "hdm3")
    decoder_label = decoder_recode_key.upper()
    if not _is_sha256(runtime_tree_sha256):
        blockers.append("hdm3_runtime_tree_sha256_missing_or_invalid")
    payload_identity_proven = (
        runtime_adapter_payload_identity.get("payload_identity_proven") is True
    )
    lossless_decoder_equivalence_proven = (
        runtime_adapter_payload_identity.get("lossless_decoder_equivalence_proven") is True
    )
    runtime_equivalence_proven = (
        runtime_adapter_payload_identity.get("runtime_equivalence_proven") is True
    )
    if runtime_adapter_payload_identity.get("runtime_adapter_parity_proven") is not True:
        blockers.append("hdm3_runtime_adapter_equivalence_not_proven")
    if not (payload_identity_proven or lossless_decoder_equivalence_proven):
        blockers.append("hdm3_runtime_adapter_decoder_equivalence_not_proven")
    if runtime_adapter_payload_identity.get("ready_for_public_runtime_inflate") is not True:
        blockers.append("hdm3_public_runtime_inflate_not_ready")

    closed = not blockers
    contract = {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_runtime_tree_inflate_output_parity_contract_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "candidate_archive_sha256": manifest.get("candidate_archive_sha256"),
        "candidate_archive_bytes": manifest.get("candidate_archive_bytes"),
        "source_archive_sha256": manifest.get("source_archive_sha256"),
        "source_archive_bytes": manifest.get("source_archive_bytes"),
        "runtime_inflate_sh": HDM3_RUNTIME_INFLATE,
        "upstream_dir": HDM3_UPSTREAM_DIR,
        "runtime_tree_manifest_source": CONTEST_AUTH_EVAL_RUNTIME_MANIFEST_SOURCE,
        "path": repo_relative(output_root / RUNTIME_TREE_CLOSURE_FILENAME, repo),
        "runtime_manifest_schema": runtime_manifest.get("schema") if runtime_manifest else "",
        "runtime_tree_sha256": runtime_tree_sha256,
        "runtime_file_count": runtime_manifest.get("runtime_file_count", 0)
        if runtime_manifest
        else 0,
        "external_dependency_root_count": len(
            runtime_manifest.get("external_dependency_roots") or []
        )
        if runtime_manifest
        else 0,
        "repo_local_tac_import_file_count": (
            runtime_manifest.get("repo_local_tac_import_manifest", {}).get("file_count", 0)
            if isinstance(runtime_manifest.get("repo_local_tac_import_manifest"), Mapping)
            else 0
        )
        if runtime_manifest
        else 0,
        "runtime_tree_closure_proven": closed,
        "inflate_output_parity_proof_kind": (
            "payload_identity"
            if payload_identity_proven
            else "decoder_raw_equivalence_with_submission_runtime_parse"
            if lossless_decoder_equivalence_proven and runtime_equivalence_proven
            else "missing"
        ),
        "inflate_output_parity_scope": (
            f"{decoder_label} candidate proof records exact payload identity when "
            "available, otherwise only decoder-raw equivalence plus actual "
            "submission-runtime parse support. This is not a CUDA frame or "
            "scorer parity result."
        ),
        "inflate_output_parity_proven_by_payload_identity": payload_identity_proven,
        "lossless_decoder_equivalence_proven": lossless_decoder_equivalence_proven,
        "runtime_equivalence_proven": runtime_equivalence_proven,
        "full_frame_inflate_output_parity_claim": False,
        "exact_frame_output_parity_run": False,
        "exact_cuda_auth_eval_required_before_score": True,
        "lane_dispatch_claim_required_before_gpu": True,
        "ready_for_fixed_runtime_exact_eval_readiness": closed,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_score_claim": False,
        "blockers": _ordered_unique(blockers),
        "remaining_blockers": _ordered_unique(blockers),
    }
    if runtime_manifest:
        contract["inflate_runtime_manifest"] = dict(runtime_manifest)
    if write:
        write_json(output_root / RUNTIME_TREE_CLOSURE_FILENAME, contract)
    return contract


def _fixed_runtime_preflight(contract: Mapping[str, Any]) -> dict[str, Any]:
    blockers = _string_list(contract.get("blockers"))
    closed = contract.get("runtime_tree_closure_proven") is True and not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_fixed_runtime_preflight_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "runtime_tree_sha256": str(contract.get("runtime_tree_sha256") or ""),
        "runtime_tree_source": str(contract.get("path") or RUNTIME_TREE_CLOSURE_FILENAME),
        "runtime_tree_manifest_source": contract.get("runtime_tree_manifest_source"),
        "ready_for_fixed_runtime_exact_eval_readiness": closed,
        "ready_for_exact_eval_dispatch": False,
        "exact_cuda_auth_eval_required_before_score": True,
        "blockers": blockers,
        "remaining_blockers": blockers,
    }


def _runtime_adapter_payload_identity_evidence(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    repo_root: Path,
    runtime_adapter_proof_path: str | Path | None,
) -> dict[str, Any]:
    proof_path = _first_existing_path(
        output_dir,
        RUNTIME_ADAPTER_PROOF_FILENAMES,
        repo_root=repo_root,
        explicit=runtime_adapter_proof_path,
    )
    decoder_recode_key = str(manifest.get("candidate_decoder_recode_key") or "hdm3")
    blockers: list[str] = []
    proof: Mapping[str, Any] = {}
    if proof_path is None:
        blockers.append("hdm3_runtime_adapter_archive_parity_proof_missing")
    else:
        loaded = _read_json_object(proof_path)
        if loaded is None:
            blockers.append("hdm3_runtime_adapter_archive_parity_proof_invalid")
        else:
            proof = loaded
    if proof:
        parity_by_payload_identity = (
            proof.get("inflate_output_parity_proven_by_payload_identity") is True
        )
        parity_by_lossless_decoder_equivalence = (
            proof.get("inflate_output_parity_proven_by_lossless_decoder_equivalence") is True
        )
        any_runtime_equivalence = bool(
            parity_by_payload_identity or parity_by_lossless_decoder_equivalence
        )
        submission_runtime_candidate_parse_claim = (
            proof.get("submission_runtime_candidate_parse_claim") is True
        )
        submission_runtime_equivalence_claim = (
            proof.get("submission_runtime_equivalence_claim") is True
        )
        if proof.get("candidate_archive_sha256") != manifest.get("candidate_archive_sha256"):
            blockers.append("hdm3_runtime_adapter_archive_sha256_mismatch")
        if proof.get("score_claim") is not False:
            blockers.append("hdm3_runtime_adapter_proof_score_claim_not_false")
        if proof.get("dispatch_attempted") is not False:
            blockers.append("hdm3_runtime_adapter_proof_dispatch_attempted_not_false")
        if proof.get("ready_for_public_runtime_inflate") is not True:
            blockers.append("hdm3_runtime_adapter_public_runtime_inflate_not_ready")
        if not any_runtime_equivalence:
            blockers.append("hdm3_decoder_equivalence_not_proven_by_runtime_adapter")
        if (
            parity_by_lossless_decoder_equivalence
            and not parity_by_payload_identity
            and not submission_runtime_candidate_parse_claim
        ):
            blockers.append("hdm3_submission_runtime_candidate_parse_not_proven")
        if (
            parity_by_lossless_decoder_equivalence
            and not parity_by_payload_identity
            and not submission_runtime_equivalence_claim
        ):
            blockers.append("hdm3_submission_runtime_equivalence_not_proven")
        if (
            proof.get("restored_payload_matches_source") is not True
            and not parity_by_lossless_decoder_equivalence
        ):
            blockers.append("hdm3_restored_payload_not_source_identical")
        if (
            proof.get("restored_decoder_section_matches_source") is not True
            and not parity_by_lossless_decoder_equivalence
        ):
            blockers.append("hdm3_restored_decoder_section_not_source_identical")
        if proof.get("latents_and_sidecar_match_source") is not True:
            blockers.append("hdm3_latents_sidecar_not_preserved")

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_runtime_adapter_equivalence_evidence_v1",
        "runtime_adapter_module": "tac.hnerv_hdm3_runtime_adapter",
        "runtime_decoder_support": f"direct_{decoder_recode_key}_decoder_section",
        "runtime_normalizer_path": "",
        "runtime_inflate_sh": HDM3_RUNTIME_INFLATE,
        "proof_path": repo_relative(proof_path, repo_root) if proof_path is not None else "",
        "proof_sha256": sha256_file(proof_path) if proof_path is not None and proof_path.is_file() else "",
        "proof_contract": proof.get("contract") if proof else "",
        "candidate_archive_sha256_matches": not proof
        or proof.get("candidate_archive_sha256") == manifest.get("candidate_archive_sha256"),
        "payload_identity_proven": (
            proof.get("inflate_output_parity_proven_by_payload_identity") is True
        )
        if proof
        else False,
        "lossless_decoder_equivalence_proven": (
            proof.get("inflate_output_parity_proven_by_lossless_decoder_equivalence") is True
        )
        if proof
        else False,
        "runtime_equivalence_proven": (
            proof.get("inflate_output_parity_proven_by_payload_identity") is True
            or (
                proof.get("inflate_output_parity_proven_by_lossless_decoder_equivalence") is True
                and proof.get("submission_runtime_candidate_parse_claim") is True
                and proof.get("submission_runtime_equivalence_claim") is True
            )
        )
        if proof
        else False,
        "submission_runtime_candidate_parse_claim": (
            proof.get("submission_runtime_candidate_parse_claim") is True
        )
        if proof
        else False,
        "submission_runtime_equivalence_claim": (
            proof.get("submission_runtime_equivalence_claim") is True
        )
        if proof
        else False,
        "restored_payload_matches_source": proof.get("restored_payload_matches_source") is True
        if proof
        else False,
        "restored_decoder_section_matches_source": proof.get("restored_decoder_section_matches_source") is True
        if proof
        else False,
        "latents_and_sidecar_match_source": proof.get("latents_and_sidecar_match_source") is True
        if proof
        else False,
        "ready_for_public_runtime_inflate": proof.get("ready_for_public_runtime_inflate") is True
        if proof
        else False,
        "runtime_adapter_parity_proven": not blockers,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def _strict_static_compliance_evidence(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    repo_root: Path,
    strict_pre_submission_compliance_path: str | Path | None,
) -> dict[str, Any]:
    compliance_path = _first_existing_path(
        output_dir,
        STATIC_COMPLIANCE_FILENAMES,
        repo_root=repo_root,
        explicit=strict_pre_submission_compliance_path,
    )
    blockers: list[str] = []
    failed_checks: list[str] = []
    payload: Mapping[str, Any] = {}
    if compliance_path is None:
        blockers.append("strict_pre_submission_compliance_json_missing")
    else:
        loaded = _read_json_object(compliance_path)
        if loaded is None:
            blockers.append("strict_pre_submission_compliance_json_invalid")
        else:
            payload = loaded
    if payload:
        if payload.get("schema") != "pre_submission_compliance_check_v1":
            blockers.append("strict_pre_submission_compliance_schema_unexpected")
        archive = payload.get("archive")
        archive_sha = archive.get("sha256") if isinstance(archive, Mapping) else None
        archive_bytes = archive.get("bytes") if isinstance(archive, Mapping) else None
        if archive_sha != manifest.get("candidate_archive_sha256"):
            blockers.append("strict_pre_submission_compliance_archive_sha_mismatch")
        if archive_bytes != manifest.get("candidate_archive_bytes"):
            blockers.append("strict_pre_submission_compliance_archive_bytes_mismatch")
        failed_checks = _failed_compliance_checks(payload)
        if failed_checks:
            blockers.append("strict_pre_submission_compliance_failed")
        if (
            payload.get("passed") is not True
            and "strict_pre_submission_compliance_failed" not in blockers
        ):
            blockers.append("strict_pre_submission_compliance_not_passed")

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_strict_static_compliance_evidence_v1",
        "path": (
            repo_relative(compliance_path, repo_root)
            if compliance_path is not None
            else repo_relative(output_dir / STATIC_COMPLIANCE_FILENAMES[0], repo_root)
        ),
        "sha256": (
            sha256_file(compliance_path)
            if compliance_path is not None and compliance_path.is_file()
            else ""
        ),
        "present": compliance_path is not None,
        "passed": payload.get("passed") is True if payload else False,
        "failed_error_checks": failed_checks,
        "blockers": blockers,
    }


def _failed_compliance_checks(payload: Mapping[str, Any]) -> list[str]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return []
    failed: list[str] = []
    for check in checks:
        if (
            isinstance(check, Mapping)
            and check.get("severity") == "error"
            and check.get("passed") is False
        ):
            failed.append(str(check.get("name")))
    return failed


def _release_surface_inflate_wrapper_text() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "# Static HDM exact-eval packet wrapper. Delegates to the reviewed\n"
        "# PR106-R2 PR101-grammar runtime; no score claim is made here.\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'if REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"; then\n'
        "  :\n"
        "else\n"
        '  REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"\n'
        "fi\n"
        f'ADAPTER_INFLATE="${{REPO_ROOT}}/{HDM3_RUNTIME_INFLATE}"\n'
        'if [ ! -x "$ADAPTER_INFLATE" ]; then\n'
        '  echo "FATAL: HDM3 adapter inflate.sh missing or not executable: $ADAPTER_INFLATE" >&2\n'
        "  exit 66\n"
        "fi\n"
        'exec "$ADAPTER_INFLATE" "$@"\n'
    )


def _release_surface_report_text(manifest: Mapping[str, Any]) -> str:
    decoder_label = str(manifest.get("candidate_decoder_recode_key") or "hdm").upper()
    return "\n".join(
        [
            f"{decoder_label} HNeRV static exact-eval packet surface",
            "",
            "This is a static packet artifact only. It makes no score claim and no GPU dispatch was attempted.",
            f"Candidate archive SHA-256: {manifest.get('candidate_archive_sha256')}",
            f"Candidate archive bytes: {manifest.get('candidate_archive_bytes')}",
            f"Candidate member: {manifest.get('candidate_member_name')}",
            f"Source archive SHA-256: {manifest.get('source_archive_sha256')}",
            "Remaining score/dispatch blockers: lane dispatch claim and exact CUDA auth eval.",
            "",
        ]
    )


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _first_existing_path(
    output_dir: Path,
    names: tuple[str, ...],
    *,
    repo_root: Path,
    explicit: str | Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(_resolve_repo_path(explicit, repo_root))
    candidates.extend(output_dir / name for name in names)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_repo_path(value: Any, repo_root: Path) -> Path:
    if value in (None, ""):
        raise HnervHdm3ArchiveCandidateError("missing required artifact path")
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _optional_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "hnerv"


def _canonical_runtime_dependency_manifest(
    inflate_sh: Path,
    upstream_dir: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    contest_auth_eval = repo_root / "experiments" / "contest_auth_eval.py"
    if not contest_auth_eval.is_file():
        raise HnervHdm3ArchiveCandidateError("contest_auth_eval.py missing")
    if not inflate_sh.is_file():
        raise HnervHdm3ArchiveCandidateError(f"runtime inflate.sh missing: {inflate_sh}")
    spec = importlib.util.spec_from_file_location(
        "_hdm3_contest_auth_eval_runtime_manifest",
        contest_auth_eval,
    )
    if spec is None or spec.loader is None:
        raise HnervHdm3ArchiveCandidateError("contest_auth_eval.py import spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest_func = getattr(module, "_runtime_dependency_manifest", None)
    if manifest_func is None:
        raise HnervHdm3ArchiveCandidateError(
            "contest_auth_eval._runtime_dependency_manifest unavailable"
        )
    manifest = manifest_func(inflate_sh, upstream_dir, repo_root=repo_root)
    if not isinstance(manifest, dict):
        raise HnervHdm3ArchiveCandidateError("runtime dependency manifest is not JSON object")
    return manifest


__all__ = [
    "HDM3_VARIANT_NAME",
    "HnervHdm3ArchiveCandidateError",
    "build_hdm3_archive_candidate",
    "build_hdm3_exact_eval_packet_readiness",
    "build_hdm3_runtime_tree_inflate_output_parity_contract",
    "materialize_hdm3_exact_eval_release_surface",
]
