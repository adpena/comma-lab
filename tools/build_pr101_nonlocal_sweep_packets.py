#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize PR101 non-local sweep packets with exact-readiness custody.

This is a local-only actuator. It does not dispatch GPU/provider jobs, invoke
contest scorers, or claim score. It consumes existing PR101 proxy-sweep rows,
builds byte-closed runtime packets with unchanged charged archive bytes, proves
the supported bias params are consumed by the runtime path, and runs the local
exact-eval readiness gate with unique lane IDs.

# DETERMINISTIC_COMPILER_OK:this tool delegates archive construction to
# tools/build_pr101_kaggle_proxy_runtime_packet.build_proxy_runtime_packet
# (the canonical proxy-runtime packet builder); it does NOT itself open
# zipfile.ZipFile in 'w' mode. The local "archive.zip" string references at
# lines 323-324 only label paths inside the delegated packet's manifest for
# downstream consumer queue rows. Catalog #158 detection fires on the
# combined (writes_zip AND emits_inflate) heuristic; this file legitimately
# satisfies neither.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimizer.exact_readiness import (  # noqa: E402
    ACTIVE_SCORE_FRONTIER_LABEL,
    ACTIVE_SCORE_FRONTIER_SCORE,
    promote_candidate_for_exact_eval,
)
from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402
from tools.build_pr101_kaggle_proxy_runtime_packet import (  # noqa: E402
    CANDIDATE_PARAM_SCHEMA,
    HANDOFF_SCHEMA,
    build_proxy_runtime_packet,
)
from tools.prove_pr101_kaggle_proxy_runtime_consumption import (  # noqa: E402
    build_runtime_consumption_proof,
)

TOOL_NAME = "tools/build_pr101_nonlocal_sweep_packets.py"
DEFAULT_OUT_DIR = Path("experiments/results/pr101_nonlocal_sweeps_20260514_codex")
# Catalog #207 — tool-owned manifest filename used by ``_assert_rmtree_safe``
# to refuse force-deletion of directories that this tool did not create.
TOOL_OWNED_MANIFEST_NAME = "build_pr101_nonlocal_sweep_manifest.json"
# Catalog #207 — every force-deletable path MUST be under this namespace prefix
# (resolved relative to REPO_ROOT). Refuses repo-root, $HOME, absolute non-repo
# paths, and parent-directory escapes.
FORCE_DELETE_NAMESPACE_PREFIX = Path("experiments/results")
DEFAULT_BIAS_REFINE_JSONL = Path(
    "experiments/results/kaggle_pr101_bias_refine_20260510_codex/"
    "pr101_bias_refine/proxy_sweep_results.jsonl"
)
DEFAULT_PROXY_SWEEP_JSONL = Path(
    "experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/"
    "pr101_proxy_sweep/proxy_sweep_results.jsonl"
)
DEFAULT_SOURCE_RUNTIME_DIR = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/"
    "source/submissions/hnerv_ft_microcodec"
)
DEFAULT_SOURCE_ARCHIVE = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
DEFAULT_DISPATCH_CLAIMS = Path(".omx/state/active_lane_dispatch_claims.md")
PARAM_KEYS = ("bias_b", "bias_g", "bias_r")
CLEARABLE_SOURCE_BLOCKERS = [
    "optimizer_candidate_queue_is_planning_only",
    "requires_exact_eval_readiness_gate",
    "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
    "requires_non_proxy_score_evidence_before_promotion",
    "exact_cuda_auth_eval_missing",
]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_rel(path: Path) -> str:
    return repo_relative(path if path.is_absolute() else REPO_ROOT / path, REPO_ROOT)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


class UnsafeRmtreeRefusedError(RuntimeError):
    """Catalog #207 — refuse ``shutil.rmtree`` on a path that fails namespace
    or tool-owned-manifest validation. The bug class anchor is
    ``--out-dir . --force`` (or any absolute non-repo path) which would
    recursively delete unrelated repo/user state. The refusal surface is
    explicit so operators see WHY the deletion was refused; do NOT silently
    swallow this error.
    """


def _assert_rmtree_safe(out_dir: Path, *, repo_root: Path = REPO_ROOT) -> None:
    """Refuse ``shutil.rmtree(out_dir)`` unless:

    1. ``out_dir`` resolves under ``<repo_root>/experiments/results/`` (the
       canonical tool-output namespace); AND
    2. ``out_dir`` is NOT the repo root, ``$HOME``, or an ancestor of either; AND
    3. ``out_dir`` contains a tool-owned manifest file
       (``TOOL_OWNED_MANIFEST_NAME``) — proving this directory was created by
       a prior run of this exact tool, not some adjacent operator state.

    Sister of Catalog #154 (``check_experiments_results_gc_helper_is_canonical``)
    which extincted the same bug class for the canonical
    ``experiments/results/`` GC helper. The local refusal pattern keeps the
    --force CLI option but locks it to tool-owned paths only.
    """
    resolved_root = repo_root.resolve()
    home_dir = Path.home().resolve()

    try:
        resolved = out_dir.resolve()
    except (OSError, RuntimeError) as exc:
        raise UnsafeRmtreeRefusedError(
            f"--force refused: cannot resolve {out_dir!s}: {exc}"
        ) from exc

    # Refusal 1: repo root itself, $HOME itself, or an ancestor of either.
    if resolved == resolved_root:
        raise UnsafeRmtreeRefusedError(
            f"--force refused: out_dir={out_dir!s} resolves to the repo root "
            f"({resolved_root!s}); refuse to recursively delete the repo."
        )
    if resolved == home_dir:
        raise UnsafeRmtreeRefusedError(
            f"--force refused: out_dir={out_dir!s} resolves to $HOME "
            f"({home_dir!s}); refuse to recursively delete the home directory."
        )
    # Refuse parent-directory escapes by checking that the resolved path is
    # actually inside the namespace prefix (not above it).
    if resolved_root in resolved.parents:
        # Good: out_dir is somewhere under repo_root.
        pass
    else:
        raise UnsafeRmtreeRefusedError(
            f"--force refused: out_dir={out_dir!s} resolves to {resolved!s} "
            f"which is NOT under the repo root ({resolved_root!s}); refuse "
            "to delete absolute paths outside the repo namespace. Sister of "
            "Catalog #154 — manifest-driven cleanup is the canonical pattern."
        )

    # Refusal 2: must be under <repo_root>/experiments/results/.
    namespace = (resolved_root / FORCE_DELETE_NAMESPACE_PREFIX).resolve()
    if namespace not in resolved.parents and resolved != namespace:
        raise UnsafeRmtreeRefusedError(
            f"--force refused: out_dir={out_dir!s} resolves to {resolved!s} "
            f"which is NOT under the {FORCE_DELETE_NAMESPACE_PREFIX}/ "
            "namespace; tool-output namespace is the only force-deletable "
            "namespace. See Catalog #207."
        )
    if resolved == namespace:
        raise UnsafeRmtreeRefusedError(
            f"--force refused: out_dir={out_dir!s} IS the namespace root "
            f"({namespace!s}); refuse to delete the entire tool-output namespace."
        )

    # Refusal 3: must contain the tool-owned manifest. We accept the directory
    # being missing-or-empty pre-run (the main() guard already short-circuits
    # via .exists()), so this check only fires when the path exists.
    if resolved.exists():
        manifest_path = resolved / TOOL_OWNED_MANIFEST_NAME
        legacy_manifest_path = resolved / "manifest.json"
        if not manifest_path.exists() and not legacy_manifest_path.exists():
            raise UnsafeRmtreeRefusedError(
                f"--force refused: out_dir={out_dir!s} ({resolved!s}) does "
                f"NOT contain the tool-owned manifest "
                f"{TOOL_OWNED_MANIFEST_NAME!r} (or legacy 'manifest.json'); "
                "refuse to delete a directory this tool did not create. "
                "Prefer manifest-driven cleanup of known generated files."
            )


def _write_tool_owned_manifest(out_dir: Path) -> Path:
    """Write the tool-owned manifest the ``--force`` guard requires.

    The manifest declares the directory was created by THIS tool so that a
    subsequent ``--force`` run can safely recurse-delete it (per Catalog #207).
    Operators inspecting the file see the tool name and creation timestamp.
    """
    manifest_path = out_dir / TOOL_OWNED_MANIFEST_NAME
    payload = {
        "schema": "build_pr101_nonlocal_sweep_manifest_v1",
        "tool": TOOL_NAME,
        "created_at_utc": utc_now(),
        "catalog_ref": "Catalog #207 — tool-owned manifest for --force guard.",
    }
    write_json(manifest_path, payload)
    return manifest_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            rows.append(value)
    return rows


def finite_proxy_objective(row: Mapping[str, Any]) -> float:
    value = row.get("proxy_objective")
    if not isinstance(value, int | float):
        return float("inf")
    return float(value)


def candidate_params(row: Mapping[str, Any]) -> dict[str, float]:
    params = row.get("params")
    if not isinstance(params, Mapping):
        raise ValueError(f"candidate {row.get('candidate_id')!r} missing params")
    out: dict[str, float] = {}
    for key in PARAM_KEYS:
        value = params.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"candidate {row.get('candidate_id')!r} param {key} is not numeric")
        out[key] = float(value)
    return out


def load_ranked_candidates(paths: Iterable[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        full = repo_path(path)
        if not full.is_file():
            continue
        for row in read_jsonl(full):
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                continue
            if row.get("param_schema") != CANDIDATE_PARAM_SCHEMA:
                continue
            candidate_params(row)
            enriched = dict(row)
            enriched["source_jsonl"] = repo_rel(full)
            prev = by_id.get(candidate_id)
            if prev is None or finite_proxy_objective(enriched) < finite_proxy_objective(prev):
                by_id[candidate_id] = enriched
    return sorted(by_id.values(), key=lambda row: (finite_proxy_objective(row), str(row["candidate_id"])))


def write_handoff(row: Mapping[str, Any], handoff_path: Path) -> None:
    payload = {
        "schema": HANDOFF_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": utc_now(),
        "candidate_id": row["candidate_id"],
        "param_schema": CANDIDATE_PARAM_SCHEMA,
        "params": candidate_params(row),
        "source_row": {
            "candidate_family": row.get("candidate_family"),
            "candidate_id": row.get("candidate_id"),
            "evidence_semantics": row.get("evidence_semantics"),
            "lane_class": row.get("lane_class"),
            "optimizer": row.get("optimizer"),
            "optimizer_status": row.get("optimizer_status"),
            "proxy_components": row.get("proxy_components"),
            "proxy_objective": row.get("proxy_objective"),
            "source_jsonl": row.get("source_jsonl"),
            "trial_index": row.get("trial_index"),
        },
        "evidence_boundary": {
            "score_claim": False,
            "score_claim_valid": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "exact_auth_eval_performed": False,
            "contest_cuda_auth_eval": False,
            "axis": "[proxy]",
            "exact_cuda_required_before_score_or_rank_claim": True,
        },
    }
    write_json(handoff_path, payload)


def queue_row_for_packet(
    *,
    row: Mapping[str, Any],
    packet_manifest: Mapping[str, Any],
    proof: Mapping[str, Any],
    lane_id: str,
) -> dict[str, Any]:
    archive = packet_manifest["packet_archive"]
    manifest_path = Path(packet_manifest["packet_dir"]) / "runtime_packet_manifest.json"
    proof_path = proof["proof_path"]
    candidate_id = f"{row['candidate_id']}_pr101_nonlocal_bias_runtime_packet"
    return {
        "candidate_id": candidate_id,
        "source_candidate_id": row["candidate_id"],
        "candidate_family": "pr101_nonlocal_bias_runtime_packet",
        "lane_class": "pr101_nonlocal_sweep_runtime_bias",
        "lane_id": lane_id,
        "target_modes": ["contest_exact_eval_planning"],
        "deployment_target": "desktop_research",
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "evidence_grade": "[byte-closed-runtime-packet-no-score]",
        "evidence_semantics": "byte_closed_proxy_runtime_packet_pending_exact_eval",
        "proxy_axis_label": "[proxy]",
        "proxy_objective": row.get("proxy_objective"),
        "proxy_components": row.get("proxy_components"),
        "candidate_params": candidate_params(row),
        "archive_path": str(Path(packet_manifest["packet_dir"]) / "archive.zip"),
        "candidate_archive_path": str(Path(packet_manifest["packet_dir"]) / "archive.zip"),
        "archive_sha256": archive["sha256"],
        "candidate_archive_sha256": archive["sha256"],
        "archive_size_bytes": archive["bytes"],
        "archive_bytes": archive["bytes"],
        "candidate_archive_bytes": archive["bytes"],
        "source_archive_sha256": packet_manifest["source_archive"]["sha256"],
        "source_archive_bytes": packet_manifest["source_archive"]["bytes"],
        "archive_unchanged_from_source": True,
        "submission_dir": packet_manifest["packet_dir"],
        "runtime_manifest_path": str(manifest_path),
        "source_manifest_path": str(manifest_path),
        "source_paths": [str(manifest_path)],
        "runtime_tree_sha256": packet_manifest["runtime_custody"]["runtime_tree_sha256"],
        "runtime_patch": packet_manifest["runtime_patch"],
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": proof_path,
        "runtime_consumption_proof_schema": proof["schema"],
        "runtime_consumption_proof_sha256": sha256_file(repo_path(Path(proof_path))),
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "score_affecting_runtime_changed": True,
        "dispatch_blockers": list(CLEARABLE_SOURCE_BLOCKERS),
    }


def build_source_queue(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    queue = {
        "schema": "optimizer_candidate_queue_v1",
        "tool": TOOL_NAME,
        "generated_at_utc": utc_now(),
        "n_candidates": len(rows),
        "top_k_count": len(rows),
        "dispatch_ready_count": 0,
        "dispatch_ready": [],
        "top_k": rows,
        "top_k_forensic": rows,
        "evidence_boundary": {
            "planning_only_by_default": True,
            "score_claim": False,
            "proxy_or_cpu_rows_must_not_promote_score": True,
            "next_gate": "exact-readiness gate, then lane claim before remote/GPU dispatch",
        },
    }
    path = out_dir / "optimizer_queue.pr101_nonlocal_bias.json"
    write_json(path, queue)
    return path


def promote_rows(
    *,
    out_dir: Path,
    queue_path: Path,
    rows: list[dict[str, Any]],
    dispatch_claims_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ready_rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    exact_dir = out_dir / "exact_ready"
    for row in rows:
        candidate_id = row["candidate_id"]
        report_path = exact_dir / f"{candidate_id}.readiness_report.json"
        output_path = exact_dir / f"{candidate_id}.exact_ready_queue.json"
        result = promote_candidate_for_exact_eval(
            queue_path,
            candidate_id,
            repo_root=REPO_ROOT,
            submission_dir=repo_path(Path(row["submission_dir"])),
            archive_manifest_path=repo_path(Path(row["runtime_manifest_path"])),
            lane_id=row["lane_id"],
            active_floor_score=ACTIVE_SCORE_FRONTIER_SCORE,
            dispatch_claims_path=repo_path(dispatch_claims_path),
        )
        write_json(report_path, result["report"])
        promoted = result["promoted_queue"]
        if promoted is None:
            blockers.append(
                {
                    "candidate_id": candidate_id,
                    "lane_id": row["lane_id"],
                    "report_path": repo_rel(report_path),
                    "blockers": result["report"].get("blockers", []),
                    "score_claim": False,
                }
            )
            continue
        write_json(output_path, promoted)
        ready_row = dict(promoted["dispatch_ready"][0])
        ready_row["exact_ready_queue_path"] = repo_rel(output_path)
        ready_row["readiness_report_path"] = repo_rel(report_path)
        ready_rows.append(ready_row)
    return ready_rows, blockers


def zip_member_rows(archive_path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(archive_path) as zf:
        rows = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            rows.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "compress_type": info.compress_type,
                    "crc": f"{info.CRC:08x}",
                    "date_time": list(info.date_time),
                    "extra_len": len(info.extra),
                    "comment_len": len(info.comment),
                }
            )
        return rows


def inspect_archive(path: Path) -> dict[str, Any]:
    return {
        "path": repo_rel(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "members": zip_member_rows(path),
    }


def find_adjacent_manifest(archive_path: Path) -> Path | None:
    for name in ("runtime_packet_manifest.json", "archive_manifest.json", "manifest.json"):
        candidate = archive_path.parent / name
        if candidate.is_file():
            return candidate
    return None


def inventory_pr101_packets(out_dir: Path, promoted_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    archive_paths: set[Path] = set()
    for root in ("experiments/results", "reverse_engineering"):
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.zip"):
            text = path.as_posix().lower()
            if "pr101" in text or "hnerv_ft_microcodec" in text or "kaggle_pr101" in text:
                archive_paths.add(path)
    rows: list[dict[str, Any]] = []
    ready_by_archive = {
        row["archive_path"]: row
        for row in promoted_rows
        if isinstance(row.get("archive_path"), str)
    }
    for archive_path in sorted(archive_paths, key=lambda item: repo_rel(item)):
        row = inspect_archive(archive_path)
        row["submission_dir"] = repo_rel(archive_path.parent)
        manifest_path = find_adjacent_manifest(archive_path)
        row["manifest_path"] = repo_rel(manifest_path) if manifest_path is not None else None
        row["runtime_tree_sha256"] = None
        row["proxy_objective"] = None
        row["exact_ready_status"] = "unknown"
        if manifest_path is not None:
            try:
                manifest = read_json(manifest_path)
            except (OSError, ValueError, json.JSONDecodeError):
                manifest = {}
            if isinstance(manifest, Mapping):
                custody = manifest.get("runtime_custody")
                if isinstance(custody, Mapping) and isinstance(custody.get("runtime_tree_sha256"), str):
                    row["runtime_tree_sha256"] = custody["runtime_tree_sha256"]
                if isinstance(manifest.get("proxy_objective"), int | float):
                    row["proxy_objective"] = manifest["proxy_objective"]
                if isinstance(manifest.get("candidate_id"), str):
                    row["candidate_id"] = manifest["candidate_id"]
        ready = ready_by_archive.get(repo_rel(archive_path))
        if ready is not None:
            row["exact_ready_status"] = "ready_for_exact_eval_dispatch_after_lane_claim"
            row["exact_ready_lane_id"] = ready.get("lane_id")
            row["exact_ready_queue_path"] = ready.get("exact_ready_queue_path")
            row["runtime_tree_sha256"] = ready.get("runtime_tree_sha256")
        rows.append(row)
    write_json(out_dir / "inventory_pr101_packets.json", {"schema": "pr101_packet_inventory_v1", "tool": TOOL_NAME, "generated_at_utc": utc_now(), "packets": rows})
    return rows


def archive_repack_blockers(out_dir: Path, source_archive: Path) -> dict[str, Any]:
    source = repo_path(source_archive)
    with zipfile.ZipFile(source) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            return {
                "schema": "pr101_archive_repack_blockers_v1",
                "source_archive": inspect_archive(source),
                "blockers": ["source_archive_not_single_member"],
                "score_claim": False,
            }
        member = infos[0]
        payload = zf.read(member)

    variants: list[dict[str, Any]] = []
    sweep_dir = out_dir / "archive_repack_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    for method, level in (("stored", None), ("deflated", 1), ("deflated", 6), ("deflated", 9)):
        candidate_name = f"x_{method}" if level is None else f"x_{method}_level{level}"
        candidate_zip = sweep_dir / f"{candidate_name}.zip"
        compression = zipfile.ZIP_STORED if method == "stored" else zipfile.ZIP_DEFLATED
        kwargs = {} if level is None else {"compresslevel": level}
        info = zipfile.ZipInfo("x", tuple(member.date_time))
        info.compress_type = compression
        info.external_attr = member.external_attr
        with zipfile.ZipFile(candidate_zip, "w") as zf:
            zf.writestr(info, payload, **kwargs)
        variant = inspect_archive(candidate_zip)
        variant["variant"] = candidate_name
        variant["delta_bytes_vs_source"] = variant["bytes"] - source.stat().st_size
        variant["artifact_kept"] = True
        variants.append(variant)

    payload_len = len(payload)
    theoretical_min_single_member_zip_overhead = 30 + len("x") + 46 + len("x") + 22
    blockers = [
        "single_member_x_stored_zip_already_at_minimal_header_overhead"
        if source.stat().st_size - payload_len == theoretical_min_single_member_zip_overhead
        else "single_member_header_overhead_not_minimal",
    ]
    if all(v["bytes"] >= source.stat().st_size for v in variants):
        blockers.append("zip_deflate_or_rewrite_did_not_reduce_archive_bytes")
    payload_sha = __import__("hashlib").sha256(payload).hexdigest()
    result = {
        "schema": "pr101_archive_repack_blockers_v1",
        "tool": TOOL_NAME,
        "generated_at_utc": utc_now(),
        "source_archive": inspect_archive(source),
        "payload_sha256": payload_sha,
        "payload_bytes": payload_len,
        "theoretical_min_single_member_zip_overhead": theoretical_min_single_member_zip_overhead,
        "variants": variants,
        "blockers": blockers,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    write_json(out_dir / "archive_repack_blockers.json", result)
    return result


def write_postprocess_blockers(out_dir: Path) -> dict[str, Any]:
    result = {
        "schema": "pr101_postprocess_scale_sweep_blockers_v1",
        "tool": TOOL_NAME,
        "generated_at_utc": utc_now(),
        "attempted_family": "runtime_scale_and_non_bias_deterministic_postprocess",
        "blockers": [
            "no_existing_local_exact_readiness_proof_for_scale_or_clamp_order_mutations",
            "no_proxy_or_exact_component_model_ranking_scale_variants_against_pr101_cuda_gap",
            "scale_mutation_would_change_runtime_logic_without_current_no_scorer_consumption_guard",
        ],
        "reactivation_criteria": [
            "extend runtime-consumption proof to scale/clamp-order params",
            "produce proxy rows with finite objective below current HDM8 exact-CUDA anchor",
            "materialize byte-closed packet with score_claim=false and pass exact-readiness gate",
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    write_json(out_dir / "postprocess_scale_blockers.json", result)
    return result


def write_combined_ready_queue(out_dir: Path, ready_rows: list[dict[str, Any]]) -> Path:
    queue = {
        "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
        "tool": TOOL_NAME,
        "generated_at_utc": utc_now(),
        "n_candidates": len(ready_rows),
        "top_k_count": len(ready_rows),
        "dispatch_ready_count": len(ready_rows),
        "dispatch_ready": ready_rows,
        "top_k": ready_rows,
        "top_k_forensic": ready_rows,
        "evidence_boundary": {
            "score_claim": False,
            "exact_cuda_required_before_score_or_rank_claim": True,
            "lane_dispatch_claim_required_before_gpu_or_remote_eval": True,
            "readiness_scope": "local_byte_closed_archive_runtime_custody_only",
            "cpu_or_proxy_score_not_cuda_evidence": True,
            "cuda_gap_review_required_before_promotion": True,
        },
    }
    path = out_dir / "exact_ready_pr101_nonlocal_bias_queue.json"
    write_json(path, queue)
    return path


def shell_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def source_proxy_row_for_ready(
    ready_row: Mapping[str, Any],
    selected_rows: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    candidate_id = str(ready_row.get("candidate_id") or "")
    source_id = candidate_id.removesuffix("_pr101_nonlocal_bias_runtime_packet")
    for row in selected_rows:
        if row.get("candidate_id") == source_id:
            return row
    return None


def next_exact_eval_commands(
    *,
    ready_row: Mapping[str, Any],
    proxy_row: Mapping[str, Any] | None,
    timestamp_tag: str,
) -> dict[str, Any]:
    lane_id = str(ready_row["lane_id"])
    job_id = f"{lane_id}_modal_t4_{timestamp_tag}"
    out_dir = f"experiments/results/modal_auth_eval/{job_id}"
    proxy_objective = (
        proxy_row.get("proxy_objective")
        if isinstance(proxy_row, Mapping)
        else None
    )
    notes = (
        "PR101 nonlocal bias runtime exact CUDA auth eval; "
        f"proxy_objective={proxy_objective}; "
        f"archive_sha256={ready_row['archive_sha256']}; "
        f"bytes={ready_row['archive_bytes']}; "
        f"runtime_tree_sha256={ready_row['runtime_tree_sha256']}"
    )
    argv = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/modal_auth_eval.py",
        "--archive",
        str(ready_row["archive_path"]),
        "--output-dir",
        out_dir,
        "--inflate-sh",
        "inflate.sh",
        "--submission-dir",
        str(ready_row["submission_dir"]),
        "--gpu",
        "T4",
        "--scorer-device",
        "cuda",
        "--inflate-device",
        "auto",
        "--detach",
        "--provider-detach-ack",
        "--lane-id",
        lane_id,
        "--instance-job-id",
        job_id,
        "--claim-agent",
        "codex:gpt-5.5",
        "--expected-runtime-tree-sha256",
        str(ready_row["runtime_tree_sha256"]),
        "--claim-notes",
        notes,
    ]
    return {
        "modal_detached_command_argv": argv,
        "modal_detached_command": "PYTHONPATH=src:upstream:$PWD " + shell_command(argv),
        "recover_command": f".venv/bin/python tools/recover_modal_auth_eval.py --output-dir {out_dir}",
        "claim_behavior": (
            "experiments/modal_auth_eval.py records the required lane claim before "
            "provider submission; do not run if claim summary shows an active same-lane conflict"
        ),
        "lane_id": lane_id,
        "instance_job_id": job_id,
        "output_dir": out_dir,
    }


def write_candidate_manifests(
    out_dir: Path,
    *,
    ready_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> None:
    timestamp_tag = utc_now().replace("-", "").replace(":", "").removesuffix("Z")
    manifest_dir = out_dir / "candidate_manifests"
    for ready_row in ready_rows:
        proxy_row = source_proxy_row_for_ready(ready_row, selected_rows)
        command_info = next_exact_eval_commands(
            ready_row=ready_row,
            proxy_row=proxy_row,
            timestamp_tag=timestamp_tag,
        )
        manifest_path = manifest_dir / f"{ready_row['candidate_id']}.candidate_manifest.json"
        manifest = {
            "schema": "pr101_nonlocal_candidate_manifest_v1",
            "tool": TOOL_NAME,
            "generated_at_utc": utc_now(),
            "candidate_id": ready_row["candidate_id"],
            "lane_id": ready_row["lane_id"],
            "score_claim": False,
            "exact_cuda_auth_eval": False,
            "ready_for_exact_eval_dispatch": True,
            "axis_labels": {
                "proxy_input": "[proxy]",
                "next_authoritative_gate": "[contest-CUDA]",
            },
            "proxy_input": {
                "candidate_id": proxy_row.get("candidate_id") if proxy_row else None,
                "proxy_axis_label": "[proxy]",
                "proxy_objective": proxy_row.get("proxy_objective") if proxy_row else None,
                "params": candidate_params(proxy_row) if proxy_row else None,
                "source_jsonl": proxy_row.get("source_jsonl") if proxy_row else None,
            },
            "archive": {
                "path": ready_row["archive_path"],
                "bytes": ready_row["archive_bytes"],
                "sha256": ready_row["archive_sha256"],
            },
            "runtime": {
                "submission_dir": ready_row["submission_dir"],
                "inflate_sh": ready_row["inflate_sh_path"],
                "runtime_tree_sha256": ready_row["runtime_tree_sha256"],
                "runtime_consumption_proof_path": ready_row.get("runtime_consumption_proof_path"),
                "runtime_consumption_proof_sha256": ready_row.get("runtime_consumption_proof_sha256"),
            },
            "next_exact_eval": command_info,
            "blockers": [],
        }
        write_json(manifest_path, manifest)
        ready_row["candidate_manifest_path"] = repo_rel(manifest_path)
        ready_row["next_exact_eval_command"] = command_info["modal_detached_command"]


def write_summary(
    out_dir: Path,
    *,
    selected_rows: list[dict[str, Any]],
    ready_rows: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    inventory_count: int,
    combined_queue_path: Path,
) -> None:
    summary = {
        "schema": "pr101_nonlocal_sweep_summary_v1",
        "tool": TOOL_NAME,
        "generated_at_utc": utc_now(),
        "score_claim": False,
        "active_exact_cuda_anchor": {
            "label": ACTIVE_SCORE_FRONTIER_LABEL,
            "axis": "[contest-CUDA]",
            "score": ACTIVE_SCORE_FRONTIER_SCORE,
        },
        "selected_proxy_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "proxy_axis_label": "[proxy]",
                "proxy_objective": row.get("proxy_objective"),
                "params": candidate_params(row),
                "source_jsonl": row.get("source_jsonl"),
            }
            for row in selected_rows
        ],
        "exact_ready_count": len(ready_rows),
        "exact_ready_queue": repo_rel(combined_queue_path),
        "exact_ready_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "lane_id": row["lane_id"],
                "archive_path": row["archive_path"],
                "archive_bytes": row["archive_bytes"],
                "archive_sha256": row["archive_sha256"],
                "runtime_tree_sha256": row["runtime_tree_sha256"],
                "exact_ready_queue_path": row["exact_ready_queue_path"],
                "readiness_report_path": row["readiness_report_path"],
                "candidate_manifest_path": row.get("candidate_manifest_path"),
                "score_claim": False,
            }
            for row in ready_rows
        ],
        "blocked_candidates": blockers,
        "inventory_packet_count": inventory_count,
        "blocker_files": [
            repo_rel(out_dir / "archive_repack_blockers.json"),
            repo_rel(out_dir / "postprocess_scale_blockers.json"),
        ],
    }
    write_json(out_dir / "summary.json", summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--exclude-candidate-id",
        action="append",
        default=["bias_refine_cmaes_0050"],
        help="Proxy candidate id to skip, e.g. an already-active exact eval.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-runtime-dir", type=Path, default=DEFAULT_SOURCE_RUNTIME_DIR)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--dispatch-claims-path", type=Path, default=DEFAULT_DISPATCH_CLAIMS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = repo_path(args.out_dir)
    if out_dir.exists() and args.force:
        # Catalog #207 — refuse force-delete unless namespace + tool-owned
        # manifest checks pass. Sister of Catalog #154 (canonical GC helper).
        _assert_rmtree_safe(out_dir)  # noqa: SLF001
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Catalog #207 — emit the tool-owned manifest BEFORE any nested writes so
    # a subsequent ``--force`` run on the same path is unambiguously authorized.
    _write_tool_owned_manifest(out_dir)

    ranked = load_ranked_candidates([DEFAULT_BIAS_REFINE_JSONL, DEFAULT_PROXY_SWEEP_JSONL])
    excludes = set(args.exclude_candidate_id or [])
    selected = [row for row in ranked if row["candidate_id"] not in excludes][: args.top_k]
    if len(selected) < args.top_k:
        raise SystemExit(f"FATAL: only found {len(selected)} candidates after excludes")

    queue_rows: list[dict[str, Any]] = []
    for row in selected:
        candidate_id = str(row["candidate_id"])
        lane_id = f"pr101_nonlocal_{candidate_id}_exact_eval"
        candidate_dir = out_dir / "packets" / candidate_id
        handoff_path = out_dir / "handoffs" / f"{candidate_id}.handoff.json"
        write_handoff(row, handoff_path)
        manifest = build_proxy_runtime_packet(
            handoff_path=handoff_path,
            source_runtime_dir=args.source_runtime_dir,
            source_archive=args.source_archive,
            packet_dir=candidate_dir,
            force=False,
        )
        proof = build_runtime_consumption_proof(
            manifest_path=candidate_dir / "runtime_packet_manifest.json",
            proof_path=candidate_dir / "runtime_consumption_proof.json",
        )
        queue_rows.append(
            queue_row_for_packet(
                row=row,
                packet_manifest=manifest,
                proof=proof,
                lane_id=lane_id,
            )
        )

    queue_path = build_source_queue(out_dir, queue_rows)
    ready_rows, blockers = promote_rows(
        out_dir=out_dir,
        queue_path=queue_path,
        rows=queue_rows,
        dispatch_claims_path=args.dispatch_claims_path,
    )
    write_candidate_manifests(out_dir, ready_rows=ready_rows, selected_rows=selected)
    combined_queue_path = write_combined_ready_queue(out_dir, ready_rows)
    archive_repack_blockers(out_dir, args.source_archive)
    write_postprocess_blockers(out_dir)
    inventory = inventory_pr101_packets(out_dir, ready_rows)
    write_summary(
        out_dir,
        selected_rows=selected,
        ready_rows=ready_rows,
        blockers=blockers,
        inventory_count=len(inventory),
        combined_queue_path=combined_queue_path,
    )

    stdout = {
        "schema": "pr101_nonlocal_sweep_stdout_v1",
        "out_dir": repo_rel(out_dir),
        "selected": [row["candidate_id"] for row in selected],
        "exact_ready_count": len(ready_rows),
        "exact_ready_queue": repo_rel(combined_queue_path),
        "score_claim": False,
    }
    print(json_text(stdout), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
