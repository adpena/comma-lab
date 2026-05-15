#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile lossless PR106 latent-sidecar recode candidates.

This is a byte-closed planning tool. It decodes a PR106 latent-sidecar payload
into canonical ``(dim, delta_q)`` correction arrays, tries lossless alternative
byte grammars, proves each candidate decodes back to the same arrays, and emits
a no-score manifest. It does not build a candidate archive and it never claims
score movement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
    PR106SidecarPacket,
    StoredZipMember,
    decode_brotli_dim_delta_sidecar_payload,
    decode_pr106_sidecar_packet_dim_delta,
    emit_pr106_sidecar_recode_candidate_archive,
    lossless_pr106_sidecar_recode_candidates,
    parse_pr106_sidecar_packet,
    pr106_sidecar_recode_candidate_manifest,
    read_single_stored_member_archive,
    sha256_hex,
)

TOOL = "tools/profile_pr106_latent_sidecar_recode.py"
SCHEMA = "pr106_latent_sidecar_recode_profile_v1"
ARCHIVE_BYTES_DENOMINATOR = 37_545_489


class LoadedSidecarSource(NamedTuple):
    """Source sidecar bytes plus optional full PR106 PacketIR wrapper context."""

    sidecar_payload: bytes
    source: dict[str, Any]
    packet: PR106SidecarPacket | None = None
    member: StoredZipMember | None = None
    archive_bytes: bytes | None = None


def _load_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _index_runtime_consumption_proofs(paths: list[Path]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in paths:
        proof = _load_json_file(path)
        archive = proof.get("archive")
        if not isinstance(archive, dict):
            raise ValueError(f"runtime proof has no archive object: {path}")
        sha = archive.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(f"runtime proof has no archive.sha256: {path}")
        indexed[sha] = {"path": str(path), "proof": proof}
    return indexed


def _index_same_runtime_parity(paths: list[Path]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in paths:
        parity = _load_json_file(path)
        candidate = parity.get("candidate_archive")
        if not isinstance(candidate, dict):
            raise ValueError(f"same-runtime parity proof has no candidate_archive: {path}")
        sha = candidate.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(f"same-runtime parity proof has no candidate_archive.sha256: {path}")
        indexed[sha] = {"path": str(path), "proof": parity}
    return indexed


def _index_exact_result_reviews(paths: list[Path] | None = None) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    review_paths = (
        sorted((REPO_ROOT / ".omx" / "research").glob("*result_review*.json"))
        if paths is None
        else paths
    )
    for path in review_paths:
        try:
            review = _load_json_file(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if review.get("schema") != "tac_result_review_packet_v1":
            continue
        if review.get("exact_cuda_evidence") is not True:
            continue
        if review.get("score_claim_valid") is not True:
            continue
        custody = review.get("custody")
        if not isinstance(custody, dict):
            continue
        archive_sha = custody.get("archive_sha256")
        if not isinstance(archive_sha, str) or len(archive_sha) != 64:
            continue
        score_recomputation = review.get("score_recomputation")
        recomputed_score = None
        if isinstance(score_recomputation, dict):
            raw_score = score_recomputation.get("recomputed_score")
            if isinstance(raw_score, int | float):
                recomputed_score = float(raw_score)
        if recomputed_score is None and isinstance(review.get("canonical_score"), int | float):
            recomputed_score = float(review["canonical_score"])
        runtime_custody = review.get("runtime_custody")
        indexed.setdefault(archive_sha, []).append(
            {
                "path": str(path),
                "schema": review.get("schema"),
                "lane_id": review.get("lane_id"),
                "job_id": review.get("job_id"),
                "score_axis": review.get("score_axis"),
                "canonical_score": recomputed_score,
                "archive_sha256": archive_sha,
                "archive_bytes": custody.get("archive_bytes"),
                "runtime_tree_sha256": (
                    runtime_custody.get("runtime_tree_sha256")
                    if isinstance(runtime_custody, dict)
                    else None
                ),
                "runtime_content_tree_sha256": (
                    runtime_custody.get("runtime_content_tree_sha256")
                    if isinstance(runtime_custody, dict)
                    else None
                ),
                "measured_config_status": review.get("measured_config_status"),
                "promotion_eligible": review.get("promotion_eligible"),
                "ready_for_exact_eval_dispatch": review.get(
                    "ready_for_exact_eval_dispatch"
                ),
                "score_claim": review.get("score_claim"),
                "score_claim_valid": review.get("score_claim_valid"),
            }
        )
    return indexed


def _remove_blockers(blockers: Any, names: set[str]) -> list[str]:
    if not isinstance(blockers, list):
        return []
    return [str(item) for item in blockers if str(item) not in names]


def _attach_runtime_consumption_proof(
    target: dict[str, Any],
    *,
    archive_sha256: str,
    proofs_by_sha: dict[str, dict[str, Any]],
) -> bool:
    match = proofs_by_sha.get(archive_sha256)
    if match is None:
        return False
    proof = match["proof"]
    archive = proof.get("archive")
    blockers = proof.get("blockers")
    valid = (
        isinstance(archive, dict)
        and archive.get("sha256") == archive_sha256
        and blockers == []
        and proof.get("runtime_sidecar_decode_consumption_claim") is True
        and proof.get("runtime_sidecar_apply_consumption_claim") is True
        and proof.get("runtime_all_score_affecting_sections_consumed") is True
        and proof.get("score_claim") is False
        and proof.get("promotion_eligible") is False
    )
    summary = {
        "path": match["path"],
        "schema": proof.get("schema"),
        "proof_scope": proof.get("proof_scope"),
        "archive_sha256": archive_sha256,
        "runtime_dir": proof.get("runtime_dir"),
        "runtime_source_tree_sha256": (
            proof.get("runtime_source_manifest", {}).get("runtime_source_tree_sha256")
            if isinstance(proof.get("runtime_source_manifest"), dict)
            else None
        ),
        "runtime_sidecar_decode_consumption_claim": proof.get(
            "runtime_sidecar_decode_consumption_claim"
        ),
        "runtime_sidecar_apply_consumption_claim": proof.get(
            "runtime_sidecar_apply_consumption_claim"
        ),
        "runtime_all_score_affecting_sections_consumed": proof.get(
            "runtime_all_score_affecting_sections_consumed"
        ),
        "score_claim": proof.get("score_claim"),
        "valid_for_candidate_archive": valid,
    }
    target["runtime_consumption_proof"] = summary
    if not valid:
        return False
    target["runtime_consumption_claim"] = True
    target["runtime_decode_apply_proof_claim"] = True
    target["candidate_exact_eval_blockers"] = _remove_blockers(
        target.get("candidate_exact_eval_blockers") or target.get("exact_eval_blockers"),
        {"runtime_decode_apply_proof_required_for_new_candidate_archive"},
    )
    if "exact_eval_blockers" in target:
        target["exact_eval_blockers"] = _remove_blockers(
            target.get("exact_eval_blockers"),
            {"runtime_decode_apply_proof_required_for_new_candidate_archive"},
        )
    return True


def _attach_same_runtime_parity_proof(
    target: dict[str, Any],
    *,
    archive_sha256: str,
    parity_by_sha: dict[str, dict[str, Any]],
) -> bool:
    match = parity_by_sha.get(archive_sha256)
    if match is None:
        return False
    parity = match["proof"]
    candidate = parity.get("candidate_archive")
    valid = (
        isinstance(candidate, dict)
        and candidate.get("sha256") == archive_sha256
        and parity.get("full_frame_inflate_output_parity_claim") is True
        and parity.get("streaming_output_sha256_equal") is True
        and parity.get("streaming_output_total_bytes_equal") is True
        and parity.get("contest_axis_claim") is False
        and parity.get("score_claim") is False
        and parity.get("promotion_eligible") is False
    )
    summary = {
        "path": match["path"],
        "schema": parity.get("schema"),
        "proof_scope": parity.get("proof_scope"),
        "candidate_archive_sha256": archive_sha256,
        "device_axis_label": parity.get("device_axis_label"),
        "full_frame_inflate_output_parity_claim": parity.get(
            "full_frame_inflate_output_parity_claim"
        ),
        "streaming_output_sha256_equal": parity.get("streaming_output_sha256_equal"),
        "streaming_output_total_bytes_equal": parity.get(
            "streaming_output_total_bytes_equal"
        ),
        "score_claim": parity.get("score_claim"),
        "valid_for_candidate_archive": valid,
    }
    target["same_runtime_full_frame_parity_proof"] = summary
    if not valid:
        return False
    target["full_frame_inflate_output_parity_claim"] = True
    target["candidate_exact_eval_blockers"] = _remove_blockers(
        target.get("candidate_exact_eval_blockers") or target.get("exact_eval_blockers"),
        {"full_frame_same_runtime_parity_or_same_runtime_auth_eval_missing"},
    )
    if "exact_eval_blockers" in target:
        target["exact_eval_blockers"] = _remove_blockers(
            target.get("exact_eval_blockers"),
            {"full_frame_same_runtime_parity_or_same_runtime_auth_eval_missing"},
        )
    return True


def _attach_exact_result_review(
    target: dict[str, Any],
    *,
    archive_sha256: str,
    exact_reviews_by_sha: dict[str, list[dict[str, Any]]],
) -> bool:
    matches = exact_reviews_by_sha.get(archive_sha256)
    if not matches:
        return False
    target["exact_cuda_result_reviews"] = matches
    target["exact_cuda_auth_eval_claim"] = True
    blockers = _remove_blockers(
        target.get("candidate_exact_eval_blockers") or target.get("exact_eval_blockers"),
        {"exact_cuda_auth_eval_missing", "contest_auth_eval_adjudication_missing"},
    )
    if "exact_cuda_result_review_already_exists" not in blockers:
        blockers.append("exact_cuda_result_review_already_exists")
    target["candidate_exact_eval_blockers"] = blockers
    if "exact_eval_blockers" in target:
        target["exact_eval_blockers"] = blockers
    return True


def _safe_candidate_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "candidate"


def load_sidecar_source(args: argparse.Namespace) -> LoadedSidecarSource:
    if args.sidecar_bin is not None:
        payload = args.sidecar_bin.read_bytes()
        return LoadedSidecarSource(
            sidecar_payload=payload,
            source={
                "mode": "sidecar_bin",
                "path": str(args.sidecar_bin),
                "sidecar_payload_bytes": len(payload),
                "sidecar_payload_sha256": sha256_hex(payload),
                "sidecar_format_id": "0x01",
                "framing_meta_bytes": 0,
                "framing_meta_sha256": None,
            },
        )

    archive_bytes = args.sidecar_archive.read_bytes()
    member = read_single_stored_member_archive(
        archive_bytes,
        expected_member_name=args.member_name,
    )
    packet = parse_pr106_sidecar_packet(member.payload)
    source = {
        "mode": "sidecar_archive",
        "path": str(args.sidecar_archive),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": sha256_hex(archive_bytes),
        "member_name": member.name,
        "member_bytes": len(member.payload),
        "member_sha256": sha256_hex(member.payload),
        "sidecar_format_id": f"0x{packet.format_id:02X}",
        "sidecar_kind": packet.sidecar_kind,
        "pr106_inner_payload_bytes": len(packet.pr106_bytes),
        "pr106_inner_payload_sha256": sha256_hex(packet.pr106_bytes),
        "sidecar_payload_bytes": len(packet.sidecar_payload),
        "sidecar_payload_sha256": sha256_hex(packet.sidecar_payload),
        "framing_meta_bytes": 0 if packet.framing_meta is None else len(packet.framing_meta),
        "framing_meta_sha256": None
        if packet.framing_meta is None
        else sha256_hex(packet.framing_meta),
    }
    if packet.format_id == PR106_SIDECAR_FORMAT_BROTLI:
        return LoadedSidecarSource(
            sidecar_payload=packet.sidecar_payload,
            source=source,
            packet=packet,
            member=member,
            archive_bytes=archive_bytes,
        )
    if packet.format_id in {
        PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
        PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
    }:
        # Re-encode to the canonical 0x01 byte source for candidate comparison;
        # the decoded arrays remain the semantic source of truth.
        dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
        current = next(
            candidate
            for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
            if candidate.name == "current_pr100_dim_delta_brotli_q11"
        )
        source["semantic_source_format"] = f"{packet.sidecar_kind}_decoded_then_profiled"
        return LoadedSidecarSource(
            sidecar_payload=current.encoded_bytes,
            source=source,
            packet=packet,
            member=member,
            archive_bytes=archive_bytes,
        )
    raise ValueError(f"unsupported sidecar format_id=0x{packet.format_id:02X}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_sidecar_source(args)
    runtime_proofs_by_sha = _index_runtime_consumption_proofs(
        list(args.runtime_consumption_proof or [])
    )
    parity_by_sha = _index_same_runtime_parity(
        list(args.same_runtime_full_frame_parity or [])
    )
    explicit_review_paths = list(args.exact_result_review or [])
    if explicit_review_paths:
        exact_review_paths: list[Path] | None = explicit_review_paths
    elif args.scan_default_exact_result_reviews:
        exact_review_paths = None
    else:
        exact_review_paths = []
    exact_reviews_by_sha = _index_exact_result_reviews(exact_review_paths)
    sidecar_payload = loaded.sidecar_payload
    source = loaded.source
    dims, deltas = decode_brotli_dim_delta_sidecar_payload(sidecar_payload)
    current_charged_bytes = int(source["sidecar_payload_bytes"]) + int(
        source.get("framing_meta_bytes") or 0
    )
    emitted_candidate_manifests: list[dict[str, Any]] = []
    emit_dir = args.emit_runtime_candidates_dir
    if emit_dir is not None:
        emit_dir.mkdir(parents=True, exist_ok=True)
        if loaded.packet is None or loaded.member is None:
            raise ValueError(
                "--emit-runtime-candidates-dir requires --sidecar-archive, not --sidecar-bin"
            )
    candidates = lossless_pr106_sidecar_recode_candidates(dims, deltas)
    rows = []
    for candidate in candidates:
        applicable = bool(candidate.encoded_bytes)
        charged = candidate.charged_bytes if applicable else None
        delta_bytes = None if charged is None else charged - current_charged_bytes
        row: dict[str, Any] = {
            "name": candidate.name,
            "applicable": applicable,
            "charged_sidecar_bytes": charged,
            "delta_bytes_vs_current_charged_sidecar": delta_bytes,
            "rate_score_delta_if_runtime_consumed": None
            if delta_bytes is None
            else 25.0 * delta_bytes / ARCHIVE_BYTES_DENOMINATOR,
            "sidecar_format_id": None
            if candidate.sidecar_format_id is None
            else f"0x{candidate.sidecar_format_id:02X}",
            "encoded_payload_bytes": len(candidate.encoded_bytes),
            "encoded_payload_sha256": sha256_hex(candidate.encoded_bytes)
            if applicable
            else None,
            "framing_meta_bytes": len(candidate.framing_meta_bytes),
            "framing_meta_sha256": sha256_hex(candidate.framing_meta_bytes)
            if candidate.framing_meta_bytes
            else None,
            "runtime_decoder_implemented": candidate.runtime_decoder_implemented,
            "lossless_semantic_equivalence_proven": applicable,
            "notes": list(candidate.notes),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        if loaded.packet is not None and applicable:
            try:
                candidate_manifest = pr106_sidecar_recode_candidate_manifest(
                    loaded.packet,
                    candidate,
                    source_archive_sha256=source.get("archive_sha256"),
                )
            except ValueError as exc:
                row["candidate_packet_ir_manifest_error"] = str(exc)
                row["candidate_packet_ir_identity_passed"] = False
            else:
                proof = candidate_manifest.get("candidate_packet_ir_consumed_byte_proof")
                row["candidate_manifest_schema"] = candidate_manifest["schema"]
                row["candidate_packet_ir_identity_passed"] = candidate_manifest[
                    "candidate_packet_ir_identity_passed"
                ]
                row["candidate_packet_ir_all_payload_bytes_accounted"] = (
                    isinstance(proof, dict)
                    and proof.get("all_payload_bytes_accounted") is True
                )
                row["candidate_packet_ir_payload_sha256"] = (
                    None if not isinstance(proof, dict) else proof.get("emitted_payload_sha256")
                )
                row["candidate_exact_eval_blockers"] = candidate_manifest[
                    "exact_eval_blockers"
                ]
                if emit_dir is not None and candidate.runtime_decoder_implemented:
                    assert loaded.member is not None
                    candidate_member, candidate_archive = (
                        emit_pr106_sidecar_recode_candidate_archive(
                            loaded.member,
                            loaded.packet,
                            candidate,
                        )
                    )
                    candidate_archive_sha = sha256_hex(candidate_archive)
                    stem = _safe_candidate_stem(candidate.name)
                    archive_path = emit_dir / f"{stem}.archive.zip"
                    manifest_path = emit_dir / f"{stem}.manifest.json"
                    archive_path.write_bytes(candidate_archive)
                    emitted_manifest = pr106_sidecar_recode_candidate_manifest(
                        loaded.packet,
                        candidate,
                        source_archive_sha256=source.get("archive_sha256"),
                        candidate_archive_sha256=candidate_archive_sha,
                        candidate_archive_bytes=len(candidate_archive),
                        candidate_member_name=candidate_member.name,
                    )
                    runtime_proof_ok = _attach_runtime_consumption_proof(
                        row,
                        archive_sha256=candidate_archive_sha,
                        proofs_by_sha=runtime_proofs_by_sha,
                    )
                    parity_ok = _attach_same_runtime_parity_proof(
                        row,
                        archive_sha256=candidate_archive_sha,
                        parity_by_sha=parity_by_sha,
                    )
                    if runtime_proof_ok:
                        _attach_runtime_consumption_proof(
                            emitted_manifest,
                            archive_sha256=candidate_archive_sha,
                            proofs_by_sha=runtime_proofs_by_sha,
                        )
                    if parity_ok:
                        _attach_same_runtime_parity_proof(
                            emitted_manifest,
                            archive_sha256=candidate_archive_sha,
                            parity_by_sha=parity_by_sha,
                        )
                    exact_review_ok = _attach_exact_result_review(
                        row,
                        archive_sha256=candidate_archive_sha,
                        exact_reviews_by_sha=exact_reviews_by_sha,
                    )
                    if exact_review_ok:
                        _attach_exact_result_review(
                            emitted_manifest,
                            archive_sha256=candidate_archive_sha,
                            exact_reviews_by_sha=exact_reviews_by_sha,
                        )
                    manifest_path.write_text(
                        json.dumps(emitted_manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    row["emitted_candidate_archive_path"] = str(archive_path)
                    row["emitted_candidate_archive_bytes"] = len(candidate_archive)
                    row["emitted_candidate_archive_sha256"] = candidate_archive_sha
                    row["emitted_candidate_manifest_path"] = str(manifest_path)
                    source_archive_bytes = source.get("archive_bytes")
                    if isinstance(source_archive_bytes, int):
                        archive_delta = len(candidate_archive) - source_archive_bytes
                        row["emitted_candidate_archive_delta_vs_source"] = archive_delta
                        row["emitted_candidate_rate_score_delta_vs_source_archive"] = (
                            25.0 * archive_delta / ARCHIVE_BYTES_DENOMINATOR
                        )
                    emitted_candidate_manifests.append(
                        {
                            "candidate_name": candidate.name,
                            "archive_path": str(archive_path),
                            "archive_bytes": len(candidate_archive),
                            "archive_sha256": candidate_archive_sha,
                            "manifest_path": str(manifest_path),
                            "score_claim": False,
                            "ready_for_exact_eval_dispatch": False,
                            "runtime_consumption_claim": runtime_proof_ok,
                            "full_frame_inflate_output_parity_claim": parity_ok,
                            "exact_cuda_result_review_claim": exact_review_ok,
                        }
                    )
        rows.append(row)
    best = next((row for row in rows if row["applicable"]), None)
    best_runtime = next(
        (row for row in rows if row["applicable"] and row["runtime_decoder_implemented"]),
        None,
    )
    any_decoder_missing_for_lossless_candidate = any(
        row["applicable"] and not row["runtime_decoder_implemented"] for row in rows
    )
    any_runtime_consumed = any(
        row.get("runtime_consumption_claim") is True for row in rows
    )
    dispatch_blockers = []
    if any_decoder_missing_for_lossless_candidate:
        dispatch_blockers.append("candidate_runtime_decoder_missing_for_noncurrent_rows")
    if not any_runtime_consumed:
        dispatch_blockers.append("missing_no_op_runtime_consumption_proof_for_new_grammar")
    any_exact_review = any(
        row.get("exact_cuda_auth_eval_claim") is True for row in rows
    )
    if any_exact_review:
        dispatch_blockers.append("exact_cuda_result_review_already_exists_for_candidate")
    else:
        dispatch_blockers.append("missing_exact_contest_eval_for_any_candidate")
    if not emitted_candidate_manifests:
        dispatch_blockers.insert(0, "no_candidate_archive_emitted")

    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "created_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "source": source,
        "semantic_arrays": {
            "n_pairs": int(dims.size),
            "n_corrected_pairs": int(((dims != 255) & (deltas != 0)).sum()),
            "n_noop_pairs": int(((dims == 255) | (deltas == 0)).sum()),
            "dim_unique": [int(value) for value in sorted(set(dims.astype(int).tolist()))],
            "delta_q_unique": [
                int(value) for value in sorted(set(deltas.astype(int).tolist()))
            ],
            "dim_sha256": sha256_hex(dims.astype("uint8").tobytes()),
            "delta_q_sha256": sha256_hex(deltas.astype("int8").tobytes()),
        },
        "current_charged_sidecar_bytes": current_charged_bytes,
        "candidate_rows": rows,
        "best_lossless_candidate": best,
        "best_runtime_decoder_implemented_candidate": best_runtime,
        "best_runtime_consumed_candidate": best_runtime,
        "best_runtime_consumed_candidate_legacy_name": True,
        "emitted_runtime_candidate_manifests": emitted_candidate_manifests,
        "linked_proof_inputs": {
            "runtime_consumption_proofs": [str(path) for path in args.runtime_consumption_proof or []],
            "same_runtime_full_frame_parity": [
                str(path) for path in args.same_runtime_full_frame_parity or []
            ],
            "exact_result_reviews": [
                str(path) for path in args.exact_result_review or []
            ],
            "default_exact_result_review_scan": args.scan_default_exact_result_reviews,
        },
        "adversarial_claim_check": {
            "verdict": "planning_only_no_score_claim",
            "interpretation": (
                "Negative or positive byte deltas here are sidecar payload-rate "
                "signals only. A score claim requires a runtime that consumes the "
                "candidate grammar, a byte-closed archive, no-op proof, and exact "
                "contest eval on the emitted packet."
            ),
        },
        "dispatch_blockers": dispatch_blockers,
    }


def render_markdown(report: dict[str, Any]) -> str:
    source = report["source"]
    lines = [
        "# PR106 Latent Sidecar Recode Profile",
        "",
        f"- score_claim: `{str(report['score_claim']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(report['ready_for_exact_eval_dispatch']).lower()}`",
        f"- source_mode: `{source.get('mode')}`",
        f"- source_path: `{source.get('path')}`",
        f"- current_charged_sidecar_bytes: `{report['current_charged_sidecar_bytes']}`",
        f"- n_pairs: `{report['semantic_arrays']['n_pairs']}`",
        f"- delta_q_unique: `{report['semantic_arrays']['delta_q_unique']}`",
        "",
        "## Candidates",
        "",
        (
            "| candidate | charged bytes | delta bytes | rate delta if consumed | "
            "archive bytes | archive delta | archive rate delta | runtime decoder | equivalence |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["candidate_rows"]:
        lines.append(
            f"| `{row['name']}` | {row['charged_sidecar_bytes']} | "
            f"{row['delta_bytes_vs_current_charged_sidecar']} | "
            f"{row['rate_score_delta_if_runtime_consumed']} | "
            f"{row.get('emitted_candidate_archive_bytes')} | "
            f"{row.get('emitted_candidate_archive_delta_vs_source')} | "
            f"{row.get('emitted_candidate_rate_score_delta_vs_source_archive')} | "
            f"`{str(row['runtime_decoder_implemented']).lower()}` | "
            f"`{str(row['lossless_semantic_equivalence_proven']).lower()}` |"
        )
    claim = report["adversarial_claim_check"]
    lines.extend(
        [
            "",
            "## Adversarial Claim Check",
            "",
            f"- verdict: `{claim['verdict']}`",
            "",
            claim["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sidecar-bin", type=Path, help="Raw PR106 format-0x01 sidecar.bin")
    source.add_argument(
        "--sidecar-archive",
        type=Path,
        help="PR106 sidecar archive.zip containing one stored 0.bin/x member",
    )
    parser.add_argument("--member-name", default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--emit-runtime-candidates-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for byte-closed archive+manifest artifacts for "
            "candidates whose runtime decoder already exists. Emitted manifests "
            "remain score_claim=false and ready_for_exact_eval_dispatch=false."
        ),
    )
    parser.add_argument(
        "--runtime-consumption-proof",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional pr106_sidecar_runtime_decode_consumption_proof_v1 JSON. "
            "When its archive SHA matches an emitted candidate and the proof "
            "is blocker-free, the candidate row records runtime consumption "
            "without claiming score or promotion."
        ),
    )
    parser.add_argument(
        "--same-runtime-full-frame-parity",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional pr106_same_runtime_streaming_frame_parity_v1 JSON. "
            "When its candidate archive SHA matches an emitted candidate and "
            "full-frame parity is true, the candidate row records local "
            "same-runtime parity without claiming contest score."
        ),
    )
    parser.add_argument(
        "--exact-result-review",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional tac_result_review_packet_v1 JSON. When its archive SHA "
            "matches an emitted candidate and it carries exact-CUDA evidence, "
            "the profile records that exact eval already exists and removes "
            "the stale exact_cuda_auth_eval_missing blocker."
        ),
    )
    parser.add_argument(
        "--no-default-exact-result-review-scan",
        dest="scan_default_exact_result_reviews",
        action="store_false",
        default=True,
        help=(
            "Disable the default scan of .omx/research/*result_review*.json. "
            "Use only for isolated tests that need an empty review index."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.json_out}")
    if args.md_out is not None:
        print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
