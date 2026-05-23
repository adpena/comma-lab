# SPDX-License-Identifier: MIT
"""Fail-closed inflate-output parity proof for inverse-scorer IAS1 candidates."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.inverse_scorer_cell_materializer import (
    CANDIDATE_SCHEMA,
    FALSE_AUTHORITY,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import read_json, repo_relative, sha256_file

INFLATE_PARITY_PROBE_SCHEMA = "inverse_scorer_cell_inflate_parity_probe_v1"
INFLATE_PARITY_VERIFICATION_SCHEMA = "inverse_scorer_cell_inflate_parity_verification_v1"
DEFAULT_PARITY_SCOPE = "full_frame_inflate_output_tree"
PARITY_BLOCKER = "candidate_inflate_output_parity_missing"


class InverseScorerCellInflateParityError(ValueError):
    """Raised when IAS1 inflate parity inputs are malformed."""


def build_inverse_scorer_cell_inflate_parity_probe(
    *,
    candidate_manifest: str | Path | Mapping[str, Any],
    source_output_dir: str | Path | None = None,
    candidate_output_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    proof_scope: str = DEFAULT_PARITY_SCOPE,
    expect_output_byte_identical: bool = True,
) -> dict[str, Any]:
    """Compare source/candidate inflate output trees and emit a proxy proof.

    The proof is intentionally output-tree based. The caller must feed it
    directories produced by the actual inflate/runtime boundary they want to
    certify; this helper only canonicalizes the comparison and refuses to turn
    it into score, promotion, or dispatch authority.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    candidate, candidate_record = _load_candidate(candidate_manifest, repo=repo)
    source_archive = _archive_record(candidate.get("template_archive"))
    candidate_archive = _archive_record(candidate.get("candidate_archive"))
    descriptor = _descriptor_record(candidate.get("inverse_scorer_cell_descriptor"))
    blockers: list[str] = []
    if proof_scope != DEFAULT_PARITY_SCOPE:
        blockers.append("inflate_parity_proof_scope_not_full_frame")
    if not source_archive.get("sha256"):
        blockers.append("source_archive_sha256_missing")
    if not candidate_archive.get("sha256"):
        blockers.append("candidate_archive_sha256_missing")

    source_tree: dict[str, Any] = _missing_tree_record("source_inflate_output_dir_missing")
    candidate_tree: dict[str, Any] = _missing_tree_record("candidate_inflate_output_dir_missing")
    if source_output_dir is None or candidate_output_dir is None:
        blockers.append("inflate_output_parity_artifacts_missing")
        if source_output_dir is None:
            blockers.append("source_inflate_output_dir_missing")
        if candidate_output_dir is None:
            blockers.append("candidate_inflate_output_dir_missing")
    else:
        source_tree = _output_tree_record(
            _resolve_optional_dir(source_output_dir, repo),
            repo=repo,
            label="source",
        )
        candidate_tree = _output_tree_record(
            _resolve_optional_dir(candidate_output_dir, repo),
            repo=repo,
            label="candidate",
        )
        blockers.extend(source_tree["blockers"])
        blockers.extend(candidate_tree["blockers"])

    source_files = _file_map(source_tree)
    candidate_files = _file_map(candidate_tree)
    source_paths = set(source_files)
    candidate_paths = set(candidate_files)
    missing_from_candidate = sorted(source_paths - candidate_paths)
    extra_in_candidate = sorted(candidate_paths - source_paths)
    differing_paths = sorted(
        path
        for path in (source_paths | candidate_paths)
        if source_files.get(path) != candidate_files.get(path)
    )
    output_contract_paths_match = not missing_from_candidate and not extra_in_candidate
    output_contract_nonempty = bool(source_files) and bool(candidate_files)
    output_bytes_identical = not differing_paths and output_contract_paths_match
    if not output_contract_nonempty:
        blockers.append("inflate_output_contract_empty")
    if not output_contract_paths_match:
        blockers.append("inflate_output_paths_mismatch")
    if expect_output_byte_identical and not output_bytes_identical:
        blockers.append("inflate_output_bytes_not_identical")

    blockers = ordered_unique(blockers)
    full_frame_parity = (
        not blockers
        and proof_scope == DEFAULT_PARITY_SCOPE
        and output_contract_nonempty
        and output_contract_paths_match
        and output_bytes_identical
    )
    return apply_proxy_evidence_boundary(
        {
            "schema": INFLATE_PARITY_PROBE_SCHEMA,
            "proof_scope": proof_scope,
            "candidate_manifest": candidate_record,
            "source_archive": source_archive,
            "candidate_archive": candidate_archive,
            "inverse_scorer_cell_descriptor": descriptor,
            "source_output_tree": source_tree,
            "candidate_output_tree": candidate_tree,
            "output_contract_paths_match": output_contract_paths_match,
            "output_contract_nonempty": output_contract_nonempty,
            "expect_output_byte_identical": bool(expect_output_byte_identical),
            "output_bytes_identical": output_bytes_identical,
            "full_frame_inflate_output_parity_claim": full_frame_parity,
            "cleared_blockers": [PARITY_BLOCKER] if full_frame_parity else [],
            "differing_paths_sample": differing_paths[:100],
            "differing_path_count": len(differing_paths),
            "missing_from_candidate": missing_from_candidate[:100],
            "extra_in_candidate": extra_in_candidate[:100],
            "blockers": blockers,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=[
            *blockers,
            "inverse_scorer_cell_inflate_parity_is_not_score_authority",
            "exact_auth_eval_required_before_score_claim",
        ],
    )


def build_inverse_scorer_cell_inflate_parity_probe_from_archives(
    *,
    candidate_manifest: str | Path | Mapping[str, Any],
    inflate_runtime_dir: str | Path,
    source_archive: str | Path | None = None,
    candidate_archive: str | Path | None = None,
    repo_root: str | Path | None = None,
    proof_scope: str = DEFAULT_PARITY_SCOPE,
    expect_output_byte_identical: bool = True,
    timeout_seconds: int = 3600,
    file_list_entries: Sequence[str] = ("0.mkv",),
    work_dir: str | Path | None = None,
    keep_work_dir: bool = False,
) -> dict[str, Any]:
    """Run actual ``inflate.sh`` on source/candidate archives, then compare trees.

    This can clear only the IAS1 full-frame inflate-output parity blocker.
    It remains proxy evidence and cannot claim score, promotion, ranking, kill
    authority, or exact-eval dispatch eligibility.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    candidate, _candidate_record = _load_candidate(candidate_manifest, repo=repo)
    runtime = _resolve_existing_dir(inflate_runtime_dir, repo)
    inflate_sh = runtime / "inflate.sh"
    if not inflate_sh.is_file():
        raise InverseScorerCellInflateParityError(f"inflate.sh missing from runtime dir: {runtime}")
    source_archive_path = _resolve_archive_for_parity(
        source_archive,
        fallback=_archive_record(candidate.get("template_archive")),
        repo=repo,
        label="source archive",
    )
    candidate_archive_path = _resolve_archive_for_parity(
        candidate_archive,
        fallback=_archive_record(candidate.get("candidate_archive")),
        repo=repo,
        label="candidate archive",
    )
    archive_blockers: list[str] = []
    _match_text(
        archive_blockers,
        sha256_file(source_archive_path),
        _archive_record(candidate.get("template_archive")).get("sha256"),
        "source_inflate_archive_sha_mismatch",
    )
    _match_text(
        archive_blockers,
        sha256_file(candidate_archive_path),
        _archive_record(candidate.get("candidate_archive")).get("sha256"),
        "candidate_inflate_archive_sha_mismatch",
    )

    temp_root: Path | None = None
    if work_dir is None:
        temp_root = Path(tempfile.mkdtemp(prefix="ias1_inflate_parity_"))
        root = temp_root
    else:
        root = _repo_path(Path(work_dir), repo)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
    source_run = _run_inflate(
        inflate_sh=inflate_sh,
        archive=source_archive_path,
        run_dir=root / "source",
        timeout_seconds=timeout_seconds,
        file_list_entries=file_list_entries,
    )
    candidate_run = _run_inflate(
        inflate_sh=inflate_sh,
        archive=candidate_archive_path,
        run_dir=root / "candidate",
        timeout_seconds=timeout_seconds,
        file_list_entries=file_list_entries,
    )
    run_blockers = list(archive_blockers)
    if source_run["returncode"] != 0:
        run_blockers.append("source_inflate_runtime_failed")
    if candidate_run["returncode"] != 0:
        run_blockers.append("candidate_inflate_runtime_failed")

    if run_blockers:
        proof = _blocked_archive_parity_probe(
            candidate_manifest=candidate_manifest,
            candidate=candidate,
            repo=repo,
            proof_scope=proof_scope,
            runtime=runtime,
            source_archive_path=source_archive_path,
            candidate_archive_path=candidate_archive_path,
            source_run=source_run,
            candidate_run=candidate_run,
            blockers=run_blockers,
            expect_output_byte_identical=expect_output_byte_identical,
        )
    else:
        proof = build_inverse_scorer_cell_inflate_parity_probe(
            candidate_manifest=candidate_manifest,
            source_output_dir=source_run["output_dir"],
            candidate_output_dir=candidate_run["output_dir"],
            repo_root=repo,
            proof_scope=proof_scope,
            expect_output_byte_identical=expect_output_byte_identical,
        )
        proof["inflate_runtime"] = _inflate_runtime_record(
            runtime=runtime,
            inflate_sh=inflate_sh,
            repo=repo,
            timeout_seconds=timeout_seconds,
            file_list_entries=file_list_entries,
        )
        proof["source_inflate_run"] = source_run
        proof["candidate_inflate_run"] = candidate_run
        proof["source_archive_inflated"] = _archive_path_record(source_archive_path, repo=repo)
        proof["candidate_archive_inflated"] = _archive_path_record(candidate_archive_path, repo=repo)
    proof["work_dir"] = repo_relative(root, repo)
    proof["work_dir_retained"] = bool(keep_work_dir)
    if not keep_work_dir:
        shutil.rmtree(root, ignore_errors=True)
    if temp_root is not None and not keep_work_dir:
        shutil.rmtree(temp_root, ignore_errors=True)
    return proof


def verify_inverse_scorer_cell_inflate_parity_probe(
    *,
    candidate_manifest: str | Path | Mapping[str, Any],
    inflate_parity_probe: str | Path | Mapping[str, Any] | None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an IAS1 inflate parity proof against a candidate manifest."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    candidate, candidate_record = _load_candidate(candidate_manifest, repo=repo)
    candidate_archive = _archive_record(candidate.get("candidate_archive"))
    source_archive = _archive_record(candidate.get("template_archive"))
    descriptor = _descriptor_record(candidate.get("inverse_scorer_cell_descriptor"))
    blockers: list[str] = []
    proof = _load_optional_mapping(inflate_parity_probe, label="inflate parity probe")
    if proof is None:
        blockers.append("inflate_parity_probe_missing")
    else:
        if proof.get("schema") != INFLATE_PARITY_PROBE_SCHEMA:
            blockers.append("inflate_parity_probe_schema_mismatch")
        try:
            require_no_truthy_authority_fields(proof, context="inflate parity probe")
        except ValueError:
            blockers.append("inflate_parity_probe_has_truthy_authority_field")
        if proof.get("full_frame_inflate_output_parity_claim") is not True:
            blockers.append(PARITY_BLOCKER)
        if proof.get("output_contract_paths_match") is not True:
            blockers.append("inflate_parity_output_paths_not_matched")
        if proof.get("output_contract_nonempty") is not True:
            blockers.append("inflate_parity_output_contract_empty")
        if proof.get("output_bytes_identical") is not True:
            blockers.append("inflate_parity_output_bytes_not_identical")
        if proof.get("score_claim") is not False:
            blockers.append("inflate_parity_probe_must_not_claim_score")
        if proof.get("promotion_eligible") is not False:
            blockers.append("inflate_parity_probe_must_not_promote")
        if proof.get("rank_or_kill_eligible") is not False:
            blockers.append("inflate_parity_probe_must_not_rank_or_kill")
        _match_text(
            blockers,
            _archive_record(proof.get("candidate_archive")).get("sha256"),
            candidate_archive.get("sha256"),
            "inflate_parity_candidate_archive_sha_mismatch",
        )
        _match_text(
            blockers,
            _archive_record(proof.get("candidate_archive")).get("member_sha256"),
            candidate_archive.get("member_sha256"),
            "inflate_parity_candidate_member_sha_mismatch",
        )
        _match_text(
            blockers,
            _archive_record(proof.get("source_archive")).get("sha256"),
            source_archive.get("sha256"),
            "inflate_parity_source_archive_sha_mismatch",
        )
        proof_descriptor = _descriptor_record(proof.get("inverse_scorer_cell_descriptor"))
        _match_text(
            blockers,
            proof_descriptor.get("packet_sha256"),
            descriptor.get("packet_sha256"),
            "inflate_parity_descriptor_packet_sha_mismatch",
        )
        _match_text(
            blockers,
            proof_descriptor.get("packet_offset"),
            descriptor.get("packet_offset"),
            "inflate_parity_descriptor_packet_offset_mismatch",
        )
        _match_text(
            blockers,
            proof_descriptor.get("packet_bytes"),
            descriptor.get("packet_bytes"),
            "inflate_parity_descriptor_packet_bytes_mismatch",
        )
        proof_blockers = [str(item) for item in proof.get("blockers") or [] if str(item)]
        blockers.extend(proof_blockers)

    blockers = ordered_unique(blockers)
    satisfied = not blockers
    return apply_proxy_evidence_boundary(
        {
            "schema": INFLATE_PARITY_VERIFICATION_SCHEMA,
            "source_candidate_manifest": candidate_record,
            "proof_schema": None if proof is None else proof.get("schema"),
            "candidate_archive_sha256": candidate_archive.get("sha256", ""),
            "candidate_member_sha256": candidate_archive.get("member_sha256", ""),
            "source_archive_sha256": source_archive.get("sha256", ""),
            "descriptor_packet_sha256": descriptor.get("packet_sha256", ""),
            "inflate_parity_satisfied": satisfied,
            "cleared_blockers": [PARITY_BLOCKER] if satisfied else [],
            "blockers": blockers,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=blockers,
    )


def _output_tree_record(path: Path | None, *, repo: Path, label: str) -> dict[str, Any]:
    if path is None:
        return _missing_tree_record(f"{label}_inflate_output_dir_missing")
    records: list[dict[str, Any]] = []
    blockers: list[str] = []
    total_bytes = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        rel = child.relative_to(path).as_posix()
        if child.is_symlink():
            blockers.append(f"{label}_inflate_output_tree_contains_symlink")
            continue
        if not child.is_file():
            continue
        size = child.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": rel,
                "bytes": size,
                "sha256": sha256_file(child),
            }
        )
    return {
        "path": repo_relative(path, repo),
        "exists": True,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "tree_sha256": _canonical_tree_sha(records),
        "files": records,
        "blockers": ordered_unique(blockers),
    }


def _missing_tree_record(reason: str) -> dict[str, Any]:
    return {
        "path": "",
        "exists": False,
        "file_count": 0,
        "total_bytes": 0,
        "tree_sha256": "",
        "files": [],
        "blockers": [reason],
    }


def _file_map(tree: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    files = tree.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes, bytearray)):
        return {}
    result: dict[str, tuple[int, str]] = {}
    for item in files:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path") or "")
        digest = str(item.get("sha256") or "")
        try:
            size = int(item.get("bytes"))
        except (TypeError, ValueError):
            continue
        if path and digest:
            result[path] = (size, digest)
    return result


def _load_candidate(value: str | Path | Mapping[str, Any], *, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, Mapping):
        payload = dict(value)
        record = {"provided_inline": True, "path": "", "sha256": ""}
    else:
        path = _resolve_existing_path(value, repo)
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise InverseScorerCellInflateParityError(f"candidate manifest is not a JSON object: {path}")
        record = {
            "provided_inline": False,
            "path": repo_relative(path, repo),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if payload.get("schema") != CANDIDATE_SCHEMA:
        raise InverseScorerCellInflateParityError(f"candidate manifest must have schema {CANDIDATE_SCHEMA}")
    try:
        require_no_truthy_authority_fields(payload, context="candidate manifest")
    except ValueError as exc:
        raise InverseScorerCellInflateParityError(str(exc)) from exc
    return payload, record


def _load_optional_mapping(value: str | Path | Mapping[str, Any] | None, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise InverseScorerCellInflateParityError(f"{label} unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise InverseScorerCellInflateParityError(f"{label} is not a JSON object: {path}")
    return payload


def _archive_record(value: Any) -> dict[str, Any]:
    mapping = value if isinstance(value, Mapping) else {}
    return {
        "path": str(mapping.get("path") or ""),
        "bytes": mapping.get("bytes"),
        "sha256": str(mapping.get("sha256") or ""),
        "member_name": str(mapping.get("member_name") or ""),
        "member_bytes": mapping.get("member_bytes"),
        "member_sha256": str(mapping.get("member_sha256") or ""),
    }


def _descriptor_record(value: Any) -> dict[str, Any]:
    mapping = value if isinstance(value, Mapping) else {}
    return {
        "schema": str(mapping.get("schema") or ""),
        "packet_offset": mapping.get("packet_offset"),
        "packet_bytes": mapping.get("packet_bytes"),
        "packet_sha256": str(mapping.get("packet_sha256") or ""),
        "json_sha256": str(mapping.get("json_sha256") or ""),
        "selected_atom_ids": [
            str(item) for item in mapping.get("selected_atom_ids") or [] if str(item)
        ],
    }


def _resolve_optional_dir(value: str | Path, repo: Path) -> Path | None:
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    return path if path.is_dir() else None


def _resolve_existing_path(value: str | Path, repo: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    if not path.exists():
        raise InverseScorerCellInflateParityError(f"path does not exist: {path}")
    return path


def _match_text(
    blockers: list[str],
    actual: Any,
    expected: Any,
    blocker: str,
) -> None:
    expected_text = "" if expected is None else str(expected)
    actual_text = "" if actual is None else str(actual)
    if expected_text and actual_text != expected_text:
        blockers.append(blocker)


def _canonical_tree_sha(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        {"files": [dict(record) for record in records]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    from hashlib import sha256

    return sha256(payload).hexdigest()


def _blocked_archive_parity_probe(
    *,
    candidate_manifest: str | Path | Mapping[str, Any],
    candidate: Mapping[str, Any],
    repo: Path,
    proof_scope: str,
    runtime: Path,
    source_archive_path: Path,
    candidate_archive_path: Path,
    source_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    blockers: Sequence[str],
    expect_output_byte_identical: bool,
) -> dict[str, Any]:
    blocker_list = ordered_unique(blockers)
    return apply_proxy_evidence_boundary(
        {
            "schema": INFLATE_PARITY_PROBE_SCHEMA,
            "proof_scope": proof_scope,
            "candidate_manifest": _candidate_manifest_record(candidate_manifest, repo=repo),
            "source_archive": _archive_record(candidate.get("template_archive")),
            "candidate_archive": _archive_record(candidate.get("candidate_archive")),
            "inverse_scorer_cell_descriptor": _descriptor_record(
                candidate.get("inverse_scorer_cell_descriptor")
            ),
            "source_archive_inflated": _archive_path_record(source_archive_path, repo=repo),
            "candidate_archive_inflated": _archive_path_record(candidate_archive_path, repo=repo),
            "inflate_runtime": {
                "path": repo_relative(runtime, repo),
                "inflate_sh": repo_relative(runtime / "inflate.sh", repo),
            },
            "source_inflate_run": dict(source_run),
            "candidate_inflate_run": dict(candidate_run),
            "source_output_tree": _missing_tree_record("source_inflate_output_unavailable"),
            "candidate_output_tree": _missing_tree_record("candidate_inflate_output_unavailable"),
            "output_contract_paths_match": False,
            "output_contract_nonempty": False,
            "expect_output_byte_identical": bool(expect_output_byte_identical),
            "output_bytes_identical": False,
            "full_frame_inflate_output_parity_claim": False,
            "cleared_blockers": [],
            "differing_paths_sample": [],
            "differing_path_count": 0,
            "missing_from_candidate": [],
            "extra_in_candidate": [],
            "blockers": blocker_list,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=[
            *blocker_list,
            "inverse_scorer_cell_inflate_parity_is_not_score_authority",
            "exact_auth_eval_required_before_score_claim",
        ],
    )


def _run_inflate(
    *,
    inflate_sh: Path,
    archive: Path,
    run_dir: Path,
    timeout_seconds: int,
    file_list_entries: Sequence[str],
) -> dict[str, Any]:
    data_dir = run_dir / "data"
    output_dir = run_dir / "out"
    file_list = run_dir / "file_list.txt"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir()
    output_dir.mkdir()
    entries = [str(item) for item in file_list_entries if str(item)]
    try:
        if not entries:
            raise InverseScorerCellInflateParityError("file_list_entries must be non-empty")
        _extract_archive(archive, data_dir)
        file_list.write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")
        env = os.environ.copy()
        python_dir = str(Path(sys.executable).parent)
        env["PATH"] = f"{python_dir}{os.pathsep}{env.get('PATH', '')}"
        proc = subprocess.run(
            [
                "bash",
                str(inflate_sh.resolve()),
                str(data_dir),
                str(output_dir),
                str(file_list),
            ],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
        return {
            "returncode": proc.returncode,
            "timeout_seconds": timeout_seconds,
            "file_list_entries": entries,
            "full_frame_file_list_claim": True,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "output_dir": str(output_dir),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "timeout_seconds": timeout_seconds,
            "file_list_entries": entries,
            "full_frame_file_list_claim": True,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "output_dir": str(output_dir),
            "error": f"inflate.sh timed out after {timeout_seconds}s",
        }
    except Exception as exc:
        return {
            "returncode": 1,
            "timeout_seconds": timeout_seconds,
            "file_list_entries": entries,
            "full_frame_file_list_claim": True,
            "stdout_tail": "",
            "stderr_tail": "",
            "output_dir": str(output_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _extract_archive(archive: Path, data_dir: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or name.startswith("/") or ".." in Path(name).parts:
                raise InverseScorerCellInflateParityError(
                    f"unsafe archive member for inflate parity: {name!r}"
                )
            target = data_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.write_bytes(zf.read(info))


def _resolve_existing_dir(value: str | Path, repo: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    if not path.is_dir():
        raise InverseScorerCellInflateParityError(f"directory does not exist: {path}")
    return path


def _resolve_archive_for_parity(
    value: str | Path | None,
    *,
    fallback: Mapping[str, Any],
    repo: Path,
    label: str,
) -> Path:
    if value is not None:
        return _resolve_existing_path(value, repo)
    path_text = str(fallback.get("path") or "")
    if not path_text:
        raise InverseScorerCellInflateParityError(f"{label} path missing")
    return _resolve_existing_path(path_text, repo)


def _archive_path_record(path: Path, *, repo: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _candidate_manifest_record(
    candidate_manifest: str | Path | Mapping[str, Any],
    *,
    repo: Path,
) -> dict[str, Any]:
    if isinstance(candidate_manifest, Mapping):
        return {"provided_inline": True, "path": "", "sha256": ""}
    path = _resolve_existing_path(candidate_manifest, repo)
    return {
        "provided_inline": False,
        "path": repo_relative(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _inflate_runtime_record(
    *,
    runtime: Path,
    inflate_sh: Path,
    repo: Path,
    timeout_seconds: int,
    file_list_entries: Sequence[str],
) -> dict[str, Any]:
    return {
        "path": repo_relative(runtime, repo),
        "inflate_sh": repo_relative(inflate_sh, repo),
        "inflate_sh_sha256": sha256_file(inflate_sh),
        "timeout_seconds": timeout_seconds,
        "file_list_entries": [str(item) for item in file_list_entries],
        "full_frame_file_list_claim": True,
    }


def _repo_path(path: Path, repo: Path) -> Path:
    return path if path.is_absolute() else repo / path


__all__ = [
    "DEFAULT_PARITY_SCOPE",
    "INFLATE_PARITY_PROBE_SCHEMA",
    "INFLATE_PARITY_VERIFICATION_SCHEMA",
    "InverseScorerCellInflateParityError",
    "build_inverse_scorer_cell_inflate_parity_probe",
    "build_inverse_scorer_cell_inflate_parity_probe_from_archives",
    "verify_inverse_scorer_cell_inflate_parity_probe",
]
