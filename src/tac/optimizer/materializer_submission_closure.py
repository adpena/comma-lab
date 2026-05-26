# SPDX-License-Identifier: MIT
"""Build byte-closed submission/runtime packets for materializer candidates.

Materializer harvest rows can already prove a byte transform and a receiver
contract, but exact-eval readiness needs the contest-shaped submission packet:
``archive.zip`` next to ``inflate.sh``, ``report.txt``, an archive manifest, and
the runtime proof. This module turns a selected harvested source-queue row into
that closure while preserving the planning-only authority boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)
from tac.optimizer.exact_readiness import runtime_dependency_manifest
from tac.repo_io import json_text, read_json, sha256_file, tree_sha256
from tac.zipwire_archive import inspect_zip_headers

SUBMISSION_CLOSURE_REPORT_SCHEMA = "materializer_submission_runtime_closure_report.v1"
SUBMISSION_CLOSURE_ARCHIVE_MANIFEST_SCHEMA = (
    "materializer_submission_runtime_closure_archive_manifest.v1"
)
SUBMISSION_CLOSURE_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotable": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "score_claim_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "field_selection_ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
}

RUNTIME_COPY_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".env",
        ".h",
        ".hpp",
        ".json",
        ".py",
        ".sh",
        ".toml",
        ".txt",
    }
)
RUNTIME_SKIP_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "eval_runs",
        "logs",
        "reports",
        "runs",
    }
)
SOURCE_RUNTIME_SKIP_FILENAMES = frozenset(
    {
        "compress.sh",
        "compress_archive.py",
        "compress_masks.py",
        "diagnose_scorer.py",
        "download_and_eval.sh",
        "eval.py",
        "runner.py",
    }
)
SOURCE_JSON_SKIP_PREFIXES = (
    "auth_eval",
    "contest_eval",
    "exact_readiness",
    "provenance",
    "report",
    "runtime_dependency_manifest",
)
SOURCE_JSON_SKIP_NAMES = {
    "archive_manifest.json",
    "manifest.json",
    "runtime_consumption_proof.json",
    "runtime_packet_manifest.json",
}


class MaterializerSubmissionClosureError(ValueError):
    """Raised when a materializer source row cannot be closed safely."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(
    value: Any,
    *,
    repo_root: Path,
    queue_dir: Path | None = None,
) -> Path | None:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    candidates = [repo_root / path]
    if queue_dir is not None:
        candidates.append(queue_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(strict=False)
    return candidates[0].resolve(strict=False)


def _safe_slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in out.split("_") if part) or "unknown"


def _source_queue_rows(queue_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in ("top_k", "top_k_forensic", "dispatch_ready"):
        for row in queue_payload.get(key) or []:
            if isinstance(row, Mapping):
                rows.append(row)
    return rows


def _find_candidate_row(
    queue_payload: Mapping[str, Any],
    *,
    candidate_id: str | None,
) -> Mapping[str, Any]:
    rows = _source_queue_rows(queue_payload)
    if candidate_id:
        for row in rows:
            if str(row.get("candidate_id") or "") == candidate_id:
                return row
        raise MaterializerSubmissionClosureError(
            f"candidate_id_missing_in_source_queue:{candidate_id}"
        )
    unique_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("candidate_id") or "")
        if row_id:
            unique_by_id.setdefault(row_id, row)
    if len(unique_by_id) == 1:
        return next(iter(unique_by_id.values()))
    raise MaterializerSubmissionClosureError(
        "candidate_id_required_for_multi_candidate_source_queue"
    )


def _selected_candidate_rows(
    queue_payload: Mapping[str, Any],
    *,
    candidate_ids: tuple[str, ...] = (),
) -> list[Mapping[str, Any]]:
    rows = _source_queue_rows(queue_payload)
    unique_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("candidate_id") or "")
        if row_id and row_id not in unique_by_id:
            unique_by_id[row_id] = row
    if not unique_by_id:
        raise MaterializerSubmissionClosureError("source_queue_has_no_candidate_rows")
    requested = tuple(str(candidate_id) for candidate_id in candidate_ids if candidate_id)
    if not requested:
        return list(unique_by_id.values())
    selected: list[Mapping[str, Any]] = []
    missing: list[str] = []
    for candidate_id in requested:
        row = unique_by_id.get(candidate_id)
        if row is None:
            missing.append(candidate_id)
        else:
            selected.append(row)
    if missing:
        raise MaterializerSubmissionClosureError(
            "candidate_id_missing_in_source_queue:" + ",".join(missing)
        )
    return selected


def _candidate_archive_path(row: Mapping[str, Any]) -> Any:
    return row.get("candidate_archive_path") or row.get("archive_path")


def _candidate_archive_sha(row: Mapping[str, Any]) -> str | None:
    value = row.get("candidate_archive_sha256") or row.get("archive_sha256")
    return str(value).lower() if isinstance(value, str) and len(value) == 64 else None


def _candidate_archive_bytes(row: Mapping[str, Any]) -> int | None:
    for key in ("candidate_archive_bytes", "archive_bytes", "archive_size_bytes"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _derive_source_runtime_dir(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path,
    source_runtime_dir: str | Path | None,
) -> Path:
    candidates: list[Path] = []
    if source_runtime_dir is not None:
        explicit = _resolve_path(source_runtime_dir, repo_root=repo_root)
        if explicit is not None:
            candidates.append(explicit)
    for key in (
        "source_runtime_dir",
        "source_submission_dir",
        "submission_dir",
        "candidate_runtime_dir",
    ):
        resolved = _resolve_path(row.get(key), repo_root=repo_root, queue_dir=queue_dir)
        if resolved is not None:
            candidates.append(resolved)
    for key in ("source_archive_path", "baseline_archive_path"):
        source_archive = _resolve_path(
            row.get(key),
            repo_root=repo_root,
            queue_dir=queue_dir,
        )
        if source_archive is not None:
            candidates.append(source_archive.parent)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "inflate.sh").is_file():
            return candidate
    raise MaterializerSubmissionClosureError(
        "source_runtime_dir_missing_or_inflate_sh_missing"
    )


def _mapping_value(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _proof_runtime_adapter_manifest(proof_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping_value(proof_payload, "runtime_adapter_manifest")


def _runtime_adapter_candidate_values(
    row: Mapping[str, Any],
    proof_payload: Mapping[str, Any],
) -> list[Any]:
    values: list[Any] = []
    proof_adapter = _proof_runtime_adapter_manifest(proof_payload)
    for payload in (
        proof_adapter,
        _mapping_value(proof_payload, "packet_member_merge_receiver_runtime"),
        _mapping_value(proof_payload, "tensor_factorize_receiver_runtime"),
        _mapping_value(proof_payload, "renderer_payload_dfl1_receiver_runtime"),
        _mapping_value(row, "packet_member_merge_receiver_runtime"),
        _mapping_value(row, "tensor_factorize_receiver_runtime"),
        _mapping_value(row, "renderer_payload_dfl1_receiver_runtime"),
        row,
    ):
        for key in (
            "runtime_dir",
            "candidate_runtime_dir",
            "packet_member_merge_runtime_dir",
            "tensor_factorize_runtime_dir",
            "renderer_payload_dfl1_candidate_runtime_dir",
        ):
            values.append(payload.get(key))
    return values


def _runtime_adapter_expected_shas(
    row: Mapping[str, Any],
    proof_payload: Mapping[str, Any],
) -> set[str]:
    shas: set[str] = set()
    for payload in (
        _proof_runtime_adapter_manifest(proof_payload),
        _mapping_value(proof_payload, "packet_member_merge_receiver_runtime"),
        _mapping_value(proof_payload, "tensor_factorize_receiver_runtime"),
        _mapping_value(proof_payload, "renderer_payload_dfl1_receiver_runtime"),
        _mapping_value(row, "packet_member_merge_receiver_runtime"),
        _mapping_value(row, "tensor_factorize_receiver_runtime"),
        _mapping_value(row, "renderer_payload_dfl1_receiver_runtime"),
        row,
    ):
        for key in (
            "runtime_tree_sha256",
            "candidate_runtime_tree_sha256",
            "packet_member_merge_receiver_runtime_tree_sha256",
            "tensor_factorize_receiver_runtime_tree_sha256",
            "renderer_payload_dfl1_runtime_tree_sha256",
        ):
            value = payload.get(key)
            if isinstance(value, str) and len(value) == 64:
                shas.add(value.lower())
    return shas


def _resolve_runtime_adapter_dir(
    row: Mapping[str, Any],
    *,
    proof_payload: Mapping[str, Any],
    repo_root: Path,
    queue_dir: Path,
) -> Path:
    candidates: list[Path] = []
    seen: set[str] = set()
    for value in _runtime_adapter_candidate_values(row, proof_payload):
        resolved = _resolve_path(value, repo_root=repo_root, queue_dir=queue_dir)
        if resolved is None:
            continue
        key = resolved.resolve(strict=False).as_posix()
        if key not in seen:
            seen.add(key)
            candidates.append(resolved)
    if not candidates:
        raise MaterializerSubmissionClosureError("runtime_adapter_dir_missing")

    expected_shas = _runtime_adapter_expected_shas(row, proof_payload)
    existing: list[Path] = []
    mismatches: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "inflate.sh").is_file():
            continue
        existing.append(candidate)
        runtime_sha = tree_sha256(candidate).lower()
        if not expected_shas or runtime_sha in expected_shas:
            return candidate
        mismatches.append(
            {
                "path": _repo_rel(candidate, repo_root),
                "runtime_tree_sha256": runtime_sha,
            }
        )
    if mismatches:
        raise MaterializerSubmissionClosureError(
            "runtime_adapter_tree_sha_mismatch:" + json.dumps(mismatches, sort_keys=True)
        )
    if existing:
        return existing[0]
    raise MaterializerSubmissionClosureError(
        "runtime_adapter_dir_missing_or_inflate_sh_missing"
    )


def _json_source_file_is_stale(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    if lower in SOURCE_JSON_SKIP_NAMES:
        return True
    return any(lower.startswith(prefix) for prefix in SOURCE_JSON_SKIP_PREFIXES)


def _runtime_file_should_copy(path: Path, runtime_root: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    rel_parts = path.relative_to(runtime_root).parts
    if any(part in RUNTIME_SKIP_DIRS for part in rel_parts):
        return False
    if path.name in SOURCE_RUNTIME_SKIP_FILENAMES:
        return False
    if path.name.startswith("._") or path.name in {".DS_Store", "Thumbs.db"}:
        return False
    suffix = path.suffix.lower()
    if suffix not in RUNTIME_COPY_SUFFIXES:
        return False
    return not (suffix == ".json" and _json_source_file_is_stale(path))


def _copy_runtime_tree(source_runtime_dir: Path, submission_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for source in sorted(
        source_runtime_dir.rglob("*"),
        key=lambda path: path.relative_to(source_runtime_dir).as_posix(),
    ):
        if not _runtime_file_should_copy(source, source_runtime_dir):
            continue
        rel = source.relative_to(source_runtime_dir)
        target = submission_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.suffix == ".sh":
            mode = target.stat().st_mode
            target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        copied.append(
            {
                "relative_path": rel.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    return copied


def _write_report_txt(
    *,
    submission_dir: Path,
    candidate_id: str,
    archive_sha256: str,
    archive_bytes: int,
) -> None:
    lines = [
        "Materializer submission runtime closure",
        "",
        f"candidate_id: {candidate_id}",
        f"archive_sha256: {archive_sha256}",
        f"archive_bytes: {archive_bytes}",
        "score_claim: false",
        "promotion_eligible: false",
        "rank_or_kill_eligible: false",
        "ready_for_exact_eval_dispatch: false",
        "",
        "This packet is a local exact-readiness closure artifact. It is not a score,",
        "rank, promotion, or dispatch authority until the canonical exact-eval gates pass.",
    ]
    (submission_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _member_payload_shas(archive_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.filename.endswith("/"):
                continue
            data = zf.read(info.filename)
            rows[info.filename] = {
                "name": info.filename,
                "member_name": info.filename,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": int(info.file_size),
                "compress_type": int(info.compress_type),
            }
    return rows


def _build_archive_manifest(
    *,
    archive_path: Path,
    source_queue_path: Path,
    source_row: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    zipwire = inspect_zip_headers(archive_path)
    member_payloads = _member_payload_shas(archive_path)
    members = []
    for member in zipwire.get("members") or []:
        if not isinstance(member, Mapping):
            continue
        name = str(member.get("name") or "")
        payload = dict(member_payloads.get(name, {"name": name, "member_name": name}))
        payload.update(
            {
                "name": name,
                "member_name": name,
                "zip_compressed_bytes": member.get("compressed_bytes"),
                "zip_uncompressed_bytes": member.get("uncompressed_bytes"),
                "zip_compression_method": member.get("compress_type"),
            }
        )
        members.append(payload)
    archive_sha = sha256_file(archive_path)
    archive_bytes = archive_path.stat().st_size
    return {
        "schema": SUBMISSION_CLOSURE_ARCHIVE_MANIFEST_SCHEMA,
        "generated_at_utc": _utc_now(),
        "candidate_id": source_row.get("candidate_id"),
        "target_kind": source_row.get("target_kind"),
        "materializer_id": source_row.get("materializer_id"),
        "receiver_contract_kind": source_row.get("receiver_contract_kind"),
        "source_queue_path": _repo_rel(source_queue_path, repo_root),
        "archive_path": _repo_rel(archive_path, repo_root),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "candidate_archive": {
            "path": _repo_rel(archive_path, repo_root),
            "sha256": archive_sha,
            "bytes": archive_bytes,
        },
        "source_archive": {
            "path": source_row.get("source_archive_path"),
            "sha256": source_row.get("source_archive_sha256"),
            "bytes": source_row.get("source_archive_bytes"),
        },
        "zipwire": {
            "zip_strict": zipwire.get("zip_strict"),
            "member_count": zipwire.get("member_count"),
            "duplicate_member_names": list(zipwire.get("duplicate_member_names") or []),
            "blockers": list(zipwire.get("blockers") or []),
        },
        "members": members,
        "realized_saved_bytes": source_row.get("realized_saved_bytes"),
        "serialized_archive_delta": source_row.get("serialized_archive_delta"),
        "allowed_use": "archive_manifest_for_exact_readiness_static_custody_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(payload), encoding="utf-8")


def _prepare_submission_dir(submission_dir: Path, *, overwrite: bool, repo_root: Path) -> None:
    resolved = submission_dir.resolve(strict=False)
    protected = {
        repo_root.resolve(strict=False),
        (repo_root / "submissions").resolve(strict=False),
        (repo_root / "src").resolve(strict=False),
        (repo_root / "tools").resolve(strict=False),
    }
    if resolved in protected:
        raise MaterializerSubmissionClosureError(
            f"refusing dangerous submission_dir_out:{submission_dir}"
        )
    if submission_dir.exists():
        if not overwrite:
            raise MaterializerSubmissionClosureError(
                f"submission_dir_out_exists:{submission_dir}"
            )
        if not submission_dir.is_dir():
            raise MaterializerSubmissionClosureError(
                f"submission_dir_out_not_directory:{submission_dir}"
            )
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)


def _closed_queue_payload(
    *,
    source_queue: Mapping[str, Any],
    source_row: Mapping[str, Any],
    closed_row: Mapping[str, Any],
) -> dict[str, Any]:
    source_schemas = list(source_queue.get("source_schemas") or [])
    payload = {
        key: value
        for key, value in source_queue.items()
        if key not in {"top_k", "top_k_forensic", "dispatch_ready"}
    }
    payload.update(
        {
            "schema": SUBMISSION_CLOSURE_QUEUE_SCHEMA,
            "tool": "tac.optimizer.materializer_submission_closure",
            "source_tool": source_queue.get("tool"),
            "generated_at_utc": _utc_now(),
            "n_candidates": 1,
            "top_k_count": 1,
            "dispatch_ready_count": 0,
            "top_k": [dict(closed_row)],
            "top_k_forensic": [dict(source_row), dict(closed_row)],
            "dispatch_ready": [],
            "source_schemas": source_schemas,
            "evidence_boundary": {
                "planning_only_by_default": True,
                "closure_packet_is_static_custody_only": True,
                "ready_for_exact_eval_dispatch_default": False,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
            },
            "allowed_use": "materializer_submission_runtime_closure_source_queue",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
            **FALSE_AUTHORITY,
        }
    )
    return payload


def _closed_queue_payload_many(
    *,
    source_queue: Mapping[str, Any],
    source_rows: list[Mapping[str, Any]],
    closed_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    source_schemas = list(source_queue.get("source_schemas") or [])
    payload = {
        key: value
        for key, value in source_queue.items()
        if key not in {"top_k", "top_k_forensic", "dispatch_ready"}
    }
    payload.update(
        {
            "schema": SUBMISSION_CLOSURE_QUEUE_SCHEMA,
            "tool": "tac.optimizer.materializer_submission_closure",
            "source_tool": source_queue.get("tool"),
            "generated_at_utc": _utc_now(),
            "n_candidates": len(closed_rows),
            "top_k_count": len(closed_rows),
            "dispatch_ready_count": 0,
            "top_k": [dict(row) for row in closed_rows],
            "top_k_forensic": [dict(row) for row in source_rows + closed_rows],
            "dispatch_ready": [],
            "source_schemas": source_schemas,
            "evidence_boundary": {
                "planning_only_by_default": True,
                "closure_packet_is_static_custody_only": True,
                "ready_for_exact_eval_dispatch_default": False,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
            },
            "allowed_use": "materializer_submission_runtime_closure_source_queue",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
            **FALSE_AUTHORITY,
        }
    )
    return payload


def build_materializer_submission_runtime_closure(
    *,
    repo_root: str | Path,
    source_queue_path: str | Path,
    submission_dir_out: str | Path,
    closed_source_queue_out: str | Path,
    closure_report_out: str | Path,
    candidate_id: str | None = None,
    source_runtime_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a contest-shaped submission closure for one materializer row."""

    repo = Path(repo_root).resolve(strict=False)
    source_queue = _resolve_path(source_queue_path, repo_root=repo)
    if source_queue is None or not source_queue.is_file():
        raise MaterializerSubmissionClosureError("source_queue_missing")
    queue_payload = read_json(source_queue)
    if not isinstance(queue_payload, Mapping):
        raise MaterializerSubmissionClosureError("source_queue_not_object")
    if queue_payload.get("schema") != SUBMISSION_CLOSURE_QUEUE_SCHEMA:
        raise MaterializerSubmissionClosureError(
            f"source_queue_schema_unsupported:{queue_payload.get('schema')!r}"
        )
    require_no_truthy_authority_fields(
        queue_payload,
        context=f"materializer_submission_closure_source_queue:{source_queue}",
    )
    queue_dir = source_queue.parent
    row = _find_candidate_row(queue_payload, candidate_id=candidate_id)
    require_no_truthy_authority_fields(
        row,
        context=f"materializer_submission_closure_source_row:{row.get('candidate_id')}",
    )
    if row.get("receiver_contract_satisfied") is not True:
        raise MaterializerSubmissionClosureError("receiver_contract_not_satisfied")
    candidate_archive = _resolve_path(
        _candidate_archive_path(row),
        repo_root=repo,
        queue_dir=queue_dir,
    )
    if candidate_archive is None or not candidate_archive.is_file():
        raise MaterializerSubmissionClosureError("candidate_archive_missing")
    archive_sha = sha256_file(candidate_archive)
    expected_sha = _candidate_archive_sha(row)
    if expected_sha is not None and archive_sha != expected_sha:
        raise MaterializerSubmissionClosureError("candidate_archive_sha_mismatch")
    archive_bytes = candidate_archive.stat().st_size
    expected_bytes = _candidate_archive_bytes(row)
    if expected_bytes is not None and archive_bytes != expected_bytes:
        raise MaterializerSubmissionClosureError("candidate_archive_bytes_mismatch")

    submission_dir = _resolve_path(submission_dir_out, repo_root=repo)
    closed_queue_path = _resolve_path(closed_source_queue_out, repo_root=repo)
    closure_report_path = _resolve_path(closure_report_out, repo_root=repo)
    if submission_dir is None or closed_queue_path is None or closure_report_path is None:
        raise MaterializerSubmissionClosureError("closure_output_path_missing")
    _prepare_submission_dir(submission_dir, overwrite=overwrite, repo_root=repo)

    proof_source = _resolve_path(
        row.get("runtime_consumption_proof_path"),
        repo_root=repo,
        queue_dir=queue_dir,
    )
    if proof_source is None or not proof_source.is_file():
        raise MaterializerSubmissionClosureError("runtime_consumption_proof_missing")
    proof_payload = read_json(proof_source)
    if not isinstance(proof_payload, Mapping):
        raise MaterializerSubmissionClosureError("runtime_consumption_proof_not_object")

    # Static submission closure copies the source contest runtime and swaps only
    # the candidate archive.  Some family-agnostic transforms, such as ZIP
    # header elision, can carry a broad runtime_adapter_ready marker from the
    # verifier even though no generated receiver adapter is needed or present.
    adapter_values = [
        value
        for value in _runtime_adapter_candidate_values(row, proof_payload)
        if isinstance(value, (str, os.PathLike)) and str(value).strip()
    ]
    runtime_adapter_ready = row.get("runtime_adapter_ready") is True and bool(
        adapter_values
    )

    if runtime_adapter_ready:
        runtime_source = _resolve_runtime_adapter_dir(
            row,
            proof_payload=proof_payload,
            repo_root=repo,
            queue_dir=queue_dir,
        )
        closure_kind = "runtime_adapter_closure_with_candidate_archive"
    else:
        runtime_source = _derive_source_runtime_dir(
            row,
            repo_root=repo,
            queue_dir=queue_dir,
            source_runtime_dir=source_runtime_dir,
        )
        closure_kind = "source_runtime_static_closure_with_candidate_archive"

    copied_runtime_files = _copy_runtime_tree(runtime_source, submission_dir)
    if not (submission_dir / "inflate.sh").is_file():
        raise MaterializerSubmissionClosureError("copied_runtime_inflate_sh_missing")

    archive_out = submission_dir / "archive.zip"
    shutil.copy2(candidate_archive, archive_out)
    proof_out = submission_dir / "runtime_consumption_proof.json"
    shutil.copy2(proof_source, proof_out)

    _write_report_txt(
        submission_dir=submission_dir,
        candidate_id=str(row.get("candidate_id") or ""),
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
    )
    archive_manifest = _build_archive_manifest(
        archive_path=archive_out,
        source_queue_path=source_queue,
        source_row=row,
        repo_root=repo,
    )
    archive_manifest_path = submission_dir / "archive_manifest.json"
    _write_json(archive_manifest_path, archive_manifest)
    runtime_manifest = runtime_dependency_manifest(submission_dir, repo)
    adapter_manifest = _proof_runtime_adapter_manifest(proof_payload)
    adapter_runtime_tree_sha = tree_sha256(runtime_source) if runtime_adapter_ready else None

    closed_row = dict(row)
    closed_row.update(
        {
            "archive_path": _repo_rel(archive_out, repo),
            "candidate_archive_path": _repo_rel(archive_out, repo),
            "archive_sha256": archive_sha,
            "candidate_archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "archive_size_bytes": archive_bytes,
            "candidate_archive_bytes": archive_bytes,
            "submission_dir": _repo_rel(submission_dir, repo),
            "archive_manifest_path": _repo_rel(archive_manifest_path, repo),
            "runtime_source_dir": _repo_rel(runtime_source, repo),
            "runtime_adapter_ready": runtime_adapter_ready,
            "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
            "submission_runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
            "runtime_content_tree_sha256": runtime_manifest[
                "runtime_content_tree_sha256"
            ],
            "submission_runtime_content_tree_sha256": runtime_manifest[
                "runtime_content_tree_sha256"
            ],
            "runtime_consumption_proof_path": _repo_rel(proof_out, repo),
            "runtime_consumption_proof_status": "present",
            "materializer_submission_closure_report_path": _repo_rel(
                closure_report_path,
                repo,
            ),
            "materializer_submission_closure_kind": closure_kind,
            "candidate_archive_path_unverified": False,
            "evidence_semantics": (
                "byte_closed_submission_runtime_closure_pending_exact_readiness"
            ),
            "allowed_use": "materializer_submission_closure_source_row_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            **FALSE_AUTHORITY,
        }
    )
    if runtime_adapter_ready:
        closed_row.update(
            {
                "candidate_runtime_dir": _repo_rel(runtime_source, repo),
                "candidate_runtime_tree_sha256": adapter_runtime_tree_sha,
                "adapter_runtime_tree_sha256": adapter_runtime_tree_sha,
                "runtime_adapter_manifest": dict(adapter_manifest),
            }
        )
        adapter_schema = adapter_manifest.get("schema")
        if adapter_schema == "packet_member_merge_receiver_runtime.v1":
            closed_row["packet_member_merge_receiver_runtime"] = dict(adapter_manifest)
        elif adapter_schema == "tensor_factorize_receiver_runtime.v1":
            closed_row["tensor_factorize_receiver_runtime"] = dict(adapter_manifest)
        elif adapter_schema == "renderer_payload_dfl1_receiver_runtime.v1":
            closed_row["renderer_payload_dfl1_receiver_runtime"] = dict(adapter_manifest)
    closed_queue = _closed_queue_payload(
        source_queue=queue_payload,
        source_row=row,
        closed_row=closed_row,
    )
    require_no_truthy_authority_fields(
        closed_queue,
        context="materializer_submission_closure_closed_queue",
    )
    _write_json(closed_queue_path, closed_queue)

    report = apply_proxy_evidence_boundary(
        {
            "schema": SUBMISSION_CLOSURE_REPORT_SCHEMA,
            "tool": "tac.optimizer.materializer_submission_closure",
            "generated_at_utc": _utc_now(),
            "source_queue_path": _repo_rel(source_queue, repo),
            "closed_source_queue_path": _repo_rel(closed_queue_path, repo),
            "candidate_id": row.get("candidate_id"),
            "target_kind": row.get("target_kind"),
            "submission_dir": _repo_rel(submission_dir, repo),
            "source_runtime_dir": _repo_rel(runtime_source, repo),
            "runtime_source_dir": _repo_rel(runtime_source, repo),
            "materializer_submission_closure_kind": closure_kind,
            "archive_path": _repo_rel(archive_out, repo),
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "archive_manifest_path": _repo_rel(archive_manifest_path, repo),
            "runtime_consumption_proof_path": _repo_rel(proof_out, repo),
            "copied_runtime_file_count": len(copied_runtime_files),
            "copied_runtime_files": copied_runtime_files,
            "runtime_manifest": runtime_manifest,
            "closure_queue_schema": closed_queue.get("schema"),
            "saved_bytes_at_risk": row.get("realized_saved_bytes"),
            "source_runtime_adapter_ready": runtime_adapter_ready,
            "targeted_correction_budget_signal": {
                "freed_bytes_require_receiver_and_exact_readiness_before_spend": True,
                "saved_bytes_at_risk": row.get("realized_saved_bytes"),
                **FALSE_AUTHORITY,
            },
            "allowed_use": "exact_readiness_static_submission_closure_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=[
            "submission_closure_report_is_not_dispatch_authority",
            "run_exact_readiness_bridge_on_closed_source_queue",
            "lane_claim_required_before_gpu_or_remote_eval",
            "contest_exact_auth_eval_required_before_score_claim",
        ],
    )
    require_no_truthy_authority_fields(
        report,
        context="materializer_submission_closure_report",
    )
    _write_json(closure_report_path, report)
    return report


def build_materializer_submission_runtime_closures(
    *,
    repo_root: str | Path,
    source_queue_path: str | Path,
    submission_dir_out: str | Path,
    closed_source_queue_out: str | Path,
    closure_report_out: str | Path,
    candidate_ids: tuple[str, ...] = (),
    source_runtime_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build submission/runtime closures for every selected source-queue row."""

    repo = Path(repo_root).resolve(strict=False)
    source_queue = _resolve_path(source_queue_path, repo_root=repo)
    if source_queue is None or not source_queue.is_file():
        raise MaterializerSubmissionClosureError("source_queue_missing")
    queue_payload = read_json(source_queue)
    if not isinstance(queue_payload, Mapping):
        raise MaterializerSubmissionClosureError("source_queue_not_object")
    if queue_payload.get("schema") != SUBMISSION_CLOSURE_QUEUE_SCHEMA:
        raise MaterializerSubmissionClosureError(
            f"source_queue_schema_unsupported:{queue_payload.get('schema')!r}"
        )
    require_no_truthy_authority_fields(
        queue_payload,
        context=f"materializer_submission_closure_source_queue:{source_queue}",
    )
    source_rows = _selected_candidate_rows(queue_payload, candidate_ids=candidate_ids)
    submission_root = _resolve_path(submission_dir_out, repo_root=repo)
    closed_queue_path = _resolve_path(closed_source_queue_out, repo_root=repo)
    closure_report_path = _resolve_path(closure_report_out, repo_root=repo)
    if submission_root is None or closed_queue_path is None or closure_report_path is None:
        raise MaterializerSubmissionClosureError("closure_output_path_missing")
    _prepare_submission_dir(submission_root, overwrite=overwrite, repo_root=repo)

    per_candidate_reports: list[dict[str, Any]] = []
    closed_rows: list[Mapping[str, Any]] = []
    candidate_sidecar_root = closure_report_path.parent / "candidate_closure_sidecars"
    for row in source_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise MaterializerSubmissionClosureError(
                "candidate_id_missing_for_submission_closure"
            )
        candidate_slug = _safe_slug(candidate_id)
        candidate_dir = submission_root / candidate_slug
        candidate_sidecar_dir = candidate_sidecar_root / candidate_slug
        candidate_closed_queue_path = candidate_sidecar_dir / "closed_source_queue.json"
        candidate_report = build_materializer_submission_runtime_closure(
            repo_root=repo,
            source_queue_path=source_queue,
            candidate_id=candidate_id,
            source_runtime_dir=source_runtime_dir,
            submission_dir_out=candidate_dir,
            closed_source_queue_out=candidate_closed_queue_path,
            closure_report_out=candidate_sidecar_dir / "submission_closure_report.json",
            overwrite=True,
        )
        per_candidate_reports.append(candidate_report)
        candidate_closed_queue = read_json(candidate_closed_queue_path)
        if not isinstance(candidate_closed_queue, Mapping):
            raise MaterializerSubmissionClosureError(
                "candidate_closed_source_queue_not_object"
            )
        candidate_top_k = candidate_closed_queue.get("top_k")
        if not isinstance(candidate_top_k, list) or len(candidate_top_k) != 1:
            raise MaterializerSubmissionClosureError(
                "candidate_closed_source_queue_expected_one_closed_row"
            )
        closed_row = candidate_top_k[0]
        if not isinstance(closed_row, Mapping):
            raise MaterializerSubmissionClosureError(
                "candidate_closed_source_queue_closed_row_not_object"
            )
        closed_rows.append(closed_row)

    closed_queue = _closed_queue_payload_many(
        source_queue=queue_payload,
        source_rows=[dict(row) for row in source_rows],
        closed_rows=[dict(row) for row in closed_rows],
    )
    require_no_truthy_authority_fields(
        closed_queue,
        context="materializer_submission_closure_closed_queue_many",
    )
    _write_json(closed_queue_path, closed_queue)

    report = apply_proxy_evidence_boundary(
        {
            "schema": SUBMISSION_CLOSURE_REPORT_SCHEMA,
            "tool": "tac.optimizer.materializer_submission_closure",
            "generated_at_utc": _utc_now(),
            "source_queue_path": _repo_rel(source_queue, repo),
            "closed_source_queue_path": _repo_rel(closed_queue_path, repo),
            "submission_dir": _repo_rel(submission_root, repo),
            "candidate_count": len(per_candidate_reports),
            "candidate_ids": [
                report.get("candidate_id") for report in per_candidate_reports
            ],
            "rows": per_candidate_reports,
            "allowed_use": "multi_candidate_materializer_submission_closure_static_custody",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            **FALSE_AUTHORITY,
        }
    )
    require_no_truthy_authority_fields(
        report,
        context="materializer_submission_closure_report_many",
    )
    _write_json(closure_report_path, report)
    return dict(report)


__all__ = [
    "FALSE_AUTHORITY",
    "SUBMISSION_CLOSURE_ARCHIVE_MANIFEST_SCHEMA",
    "SUBMISSION_CLOSURE_QUEUE_SCHEMA",
    "SUBMISSION_CLOSURE_REPORT_SCHEMA",
    "MaterializerSubmissionClosureError",
    "build_materializer_submission_runtime_closure",
    "build_materializer_submission_runtime_closures",
]
