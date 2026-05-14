#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a PR100-PR107 reproduction/deconstruction custody ledger.

This is a control-plane tool, not a score tool. It inspects local public-PR
intake directories, archive bytes, source entrypoints, exact-eval custody
artifacts, and existing research notes so every PR100-PR107 frontier atom has a
single auditable row before it is used for stacking or substitution.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
from typing import Any
from zipfile import BadZipFile, ZipFile

SCHEMA_VERSION = 1
TOOL_NAME = "tools/build_pr100_107_reproduction_ledger.py"
DEFAULT_PRS = tuple(range(100, 108))
PR_EXACT_EVAL_GLOB_ALIASES = {
    # PR107 was submitted as Apogee, but local score-bearing custody lives under
    # the earlier PR98/apogee adapter exact-eval name. Keep this explicit so the
    # ledger does not silently report "no eval" when the artifact is present.
    107: ("exact_eval_public_pr98_hnerv_adapter*",),
}


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: pathlib.Path, repo_root: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: pathlib.Path, repo_root: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": _rel(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize score fields across old/new contest-auth JSON schemas."""
    score = (
        payload.get("canonical_score")
        or payload.get("score_recomputed_from_components")
        or payload.get("record", {}).get("score")
        or payload.get("score")
        or payload.get("total_score")
    )
    provenance = payload.get("provenance") or {}
    runtime_manifest = provenance.get("inflate_runtime_manifest") or {}
    return {
        "score": float(score) if isinstance(score, (int, float)) else None,
        "score_basis": (
            "canonical_score"
            if payload.get("canonical_score") is not None
            else "score_recomputed_from_components"
            if payload.get("score_recomputed_from_components") is not None
            else "legacy_score_field"
            if score is not None
            else None
        ),
        "device": provenance.get("device") or payload.get("device"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "archive_size_bytes": payload.get("archive_size_bytes")
        or provenance.get("archive_size_bytes"),
        "archive_sha256": provenance.get("archive_sha256"),
        "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
    }


def _extract_result_json_from_log(
    path: pathlib.Path,
    repo_root: pathlib.Path,
) -> dict[str, Any] | None:
    """Parse the RESULT_JSON line from older auth_eval logs, if present."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "RESULT_JSON:" not in line:
            continue
        _, raw = line.split("RESULT_JSON:", 1)
        raw = raw.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        return {
            **_extract_score_payload(payload),
            "source": _rel(path, repo_root),
            "source_kind": "embedded_result_json_log",
        }
    return None


def _score_matches(score_a: Any, score_b: Any, *, tol: float = 5e-4) -> bool:
    if not isinstance(score_a, (int, float)) or not isinstance(score_b, (int, float)):
        return False
    return abs(float(score_a) - float(score_b)) <= tol


def _same_archive_exact_evals(
    *,
    archive: dict[str, Any],
    exact_evals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    archive_sha = archive.get("sha256")
    if not archive_sha:
        return []
    return [
        row
        for row in exact_evals
        if (row.get("archive") or {}).get("sha256") == archive_sha
    ]


def _same_archive_scored_exact_evals(
    *,
    archive: dict[str, Any],
    exact_evals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in _same_archive_exact_evals(archive=archive, exact_evals=exact_evals)
        if isinstance(row.get("score"), (int, float))
    ]


def _same_archive_structured_json_exact_evals(
    *,
    archive: dict[str, Any],
    exact_evals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in _same_archive_exact_evals(archive=archive, exact_evals=exact_evals)
        if row.get("json_files")
    ]


def _same_archive_embedded_result_json_exact_evals(
    *,
    archive: dict[str, Any],
    exact_evals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in _same_archive_exact_evals(archive=archive, exact_evals=exact_evals)
        if row.get("structured_result_kind") == "embedded_result_json_log"
    ]


def _same_archive_cuda_exact_evals(
    *,
    archive: dict[str, Any],
    exact_evals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in _same_archive_scored_exact_evals(archive=archive, exact_evals=exact_evals)
        if row.get("device") == "cuda"
    ]


def _replay_identity(row: dict[str, Any]) -> dict[str, Any]:
    archive = row.get("archive") or {}
    return {
        "archive_sha256": archive.get("sha256"),
        "archive_bytes": archive.get("bytes"),
        "device": row.get("device"),
        "runtime_tree_sha256": row.get("runtime_tree_sha256"),
        "structured_result_source": row.get("structured_result_source"),
        "score_basis": row.get("score_basis"),
    }


def _exact_eval_summary(
    *,
    archive: dict[str, Any],
    exact_evals: list[dict[str, Any]],
) -> dict[str, Any]:
    same_archive = _same_archive_exact_evals(archive=archive, exact_evals=exact_evals)
    same_archive_scored = [
        row for row in same_archive if isinstance(row.get("score"), (int, float))
    ]
    same_archive_cuda = [row for row in same_archive_scored if row.get("device") == "cuda"]
    same_archive_json = [
        row for row in same_archive if row.get("json_files")
    ]
    same_archive_embedded_result_json = [
        row
        for row in same_archive
        if row.get("structured_result_kind") == "embedded_result_json_log"
    ]
    identities = [_replay_identity(row) for row in same_archive_scored]
    unique_identity_keys = {
        (
            identity.get("archive_sha256"),
            identity.get("device"),
            identity.get("runtime_tree_sha256"),
        )
        for identity in identities
    }
    return {
        "total_eval_dir_count": len(exact_evals),
        "same_archive_eval_dir_count": len(same_archive),
        "same_archive_scored_eval_count": len(same_archive_scored),
        "same_archive_cuda_scored_eval_count": len(same_archive_cuda),
        "same_archive_structured_json_eval_count": len(same_archive_json),
        "same_archive_embedded_result_json_eval_count": len(same_archive_embedded_result_json),
        "identity_basis": ["archive_sha256", "device", "runtime_tree_sha256"],
        "same_archive_device_runtime_identity_count": len(unique_identity_keys),
        "same_archive_replay_identities": identities,
    }


def _score_drift_against_leaderboard(
    *,
    leaderboard_score: Any,
    archive: dict[str, Any],
    exact_evals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify local exact-replay scores against the public leaderboard row."""
    scored = _same_archive_scored_exact_evals(archive=archive, exact_evals=exact_evals)
    if not isinstance(leaderboard_score, (int, float)):
        return {
            "status": "leaderboard_score_missing",
            "leaderboard_score": leaderboard_score,
            "same_archive_scored_eval_count": len(scored),
            "identity_basis": ["archive_sha256", "device", "runtime_tree_sha256"],
        }
    if not scored:
        return {
            "status": "no_same_archive_score_to_compare",
            "leaderboard_score": leaderboard_score,
            "same_archive_scored_eval_count": 0,
            "identity_basis": ["archive_sha256", "device", "runtime_tree_sha256"],
        }
    by_device: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        device = str(row.get("device") or "unknown")
        by_device.setdefault(device, []).append(
            {
                "dir": row["dir"],
                "score": row["score"],
                "score_basis": row.get("score_basis"),
                "avg_posenet_dist": row.get("avg_posenet_dist"),
                "avg_segnet_dist": row.get("avg_segnet_dist"),
                "runtime_tree_sha256": row.get("runtime_tree_sha256"),
                "replay_identity": _replay_identity(row),
            }
        )
    matches = [
        {"device": device, **entry}
        for device, entries in by_device.items()
        for entry in entries
        if _score_matches(entry.get("score"), leaderboard_score)
    ]
    if matches:
        status = "leaderboard_matches_same_archive_local_replay"
    elif any(device == "cuda" for device in by_device):
        status = "leaderboard_mismatches_same_archive_cuda_replay"
    else:
        status = "leaderboard_mismatches_same_archive_local_replay"
    return {
        "status": status,
        "leaderboard_score": float(leaderboard_score),
        "same_archive_scored_eval_count": len(scored),
        "identity_basis": ["archive_sha256", "device", "runtime_tree_sha256"],
        "matches": matches,
        "local_scores_by_device": by_device,
        "note": (
            "A mismatch is not automatically a bad local replay. Public PR "
            "comments show some PR100+ rows were evaluated once on CUDA and "
            "again on CPU; the leaderboard may follow the CPU result while "
            "this repo's score claims still require CUDA for promoted internal lanes."
        ),
    }


def _zip_member_unsafe(name: str) -> bool:
    pure = pathlib.PurePosixPath(name)
    return (
        name.startswith("/")
        or ".." in pure.parts
        or "__MACOSX" in pure.parts
        or any(part.startswith(".") for part in pure.parts)
        or any(part == "" for part in pure.parts)
    )


def inspect_archive(path: pathlib.Path, repo_root: pathlib.Path) -> dict[str, Any]:
    record = _file_record(path, repo_root)
    if record is None:
        return {
            "present": False,
            "path": _rel(path, repo_root),
            "blockers": ["archive_missing"],
        }
    blockers: list[str] = []
    try:
        with ZipFile(path) as zf:
            members: list[dict[str, Any]] = []
            seen: set[str] = set()
            duplicate_members: list[str] = []
            unsafe_members: list[str] = []
            for order, info in enumerate(zf.infolist()):
                if info.is_dir():
                    continue
                name = info.filename
                if name in seen:
                    duplicate_members.append(name)
                seen.add(name)
                data = zf.read(name)
                unsafe = _zip_member_unsafe(name)
                if unsafe:
                    unsafe_members.append(name)
                prefix = data[:16]
                members.append(
                    {
                        "order": order,
                        "name": name,
                        "bytes": len(data),
                        "compressed_bytes": int(info.compress_size),
                        "crc32": f"{info.CRC:08x}",
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "prefix_hex": prefix.hex(),
                        "prefix_ascii": "".join(
                            chr(b) if 32 <= b <= 126 else "." for b in prefix
                        ),
                        "unsafe_name": unsafe,
                    }
                )
    except BadZipFile:
        return {
            **record,
            "present": True,
            "is_zip": False,
            "members": [],
            "blockers": ["archive_bad_zip"],
        }

    if not members:
        blockers.append("archive_has_no_file_members")
    if duplicate_members:
        blockers.append("archive_has_duplicate_members")
    if unsafe_members:
        blockers.append("archive_has_unsafe_members")
    return {
        **record,
        "present": True,
        "is_zip": True,
        "member_count": len(members),
        "members": members,
        "duplicate_members": sorted(duplicate_members),
        "unsafe_members": sorted(unsafe_members),
        "blockers": blockers,
    }


def _pr102_canonical_intake_dir(repo_root: pathlib.Path) -> pathlib.Path | None:
    intake = (
        repo_root
        / "experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/"
        / "public_pr102_intake_20260507_auto"
    )
    return intake if intake.is_dir() else None


def _pr102_canonical_manifest(repo_root: pathlib.Path) -> pathlib.Path | None:
    manifest = (
        repo_root
        / "experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/"
        / "CUSTODY_MANIFEST.json"
    )
    return manifest if manifest.is_file() else None


def _pr102_correct_archive(repo_root: pathlib.Path) -> pathlib.Path | None:
    canonical_manifest = _pr102_canonical_manifest(repo_root)
    if canonical_manifest is not None and canonical_manifest.is_file():
        data = _read_json(canonical_manifest)
        path = data.get("canonical_archive", {}).get("local_path")
        if path:
            candidate = repo_root / path
            if candidate.is_file():
                return candidate

    legacy_manifest = (
        repo_root
        / "experiments/results/pr102_zero_byte_tuning_custody_20260507_codex/custody_manifest.json"
    )
    if not legacy_manifest.is_file():
        return None
    data = _read_json(legacy_manifest)
    path = data.get("correct_pr102_archive", {}).get("path")
    if not path:
        return None
    candidate = repo_root / path
    return candidate if candidate.is_file() else None


def _pr103_lc_ac_schema_manifest(repo_root: pathlib.Path) -> pathlib.Path | None:
    manifest = (
        repo_root
        / "experiments/results/hnerv_pr103_lc_ac_schema_20260507_codex/manifest.json"
    )
    return manifest if manifest.is_file() else None


def _schema_manifest_for_pr(pr: int, repo_root: pathlib.Path) -> dict[str, Any] | None:
    if pr != 103:
        return None
    manifest_path = _pr103_lc_ac_schema_manifest(repo_root)
    if manifest_path is None:
        return None
    data = _read_json(manifest_path)
    data["_manifest_path"] = _rel(manifest_path, repo_root)
    return data


def _schema_manifest_matches_archive(
    manifest: dict[str, Any] | None,
    archive: dict[str, Any],
) -> bool:
    if not manifest or not archive.get("sha256"):
        return False
    source = manifest.get("source_archive") or {}
    return source.get("sha256") == archive.get("sha256")


def _schema_decode_parity_closed(
    manifest: dict[str, Any] | None,
    archive: dict[str, Any],
) -> bool:
    if not _schema_manifest_matches_archive(manifest, archive):
        return False
    stream = manifest.get("merged_arithmetic_stream") or {}
    return (
        manifest.get("ready_for_schema_review") is True
        and stream.get("reencoded_byte_identical") is True
    )


def _schema_binary_understanding(
    manifest: dict[str, Any] | None,
    archive: dict[str, Any],
) -> dict[str, Any]:
    if not _schema_manifest_matches_archive(manifest, archive):
        return {}
    stream = manifest.get("merged_arithmetic_stream") or {}
    return {
        "schema_manifest": manifest.get("_manifest_path"),
        "wire_grammar_status": "fixed_pr103_lc_ac_section_layout",
        "decode_reencode_parity_status": (
            "merged_arithmetic_stream_byte_identical"
            if stream.get("reencoded_byte_identical") is True
            else "schema_manifest_present_but_reencode_not_closed"
        ),
        "compress_reproduction_status": "missing_or_external",
        "merged_arithmetic_stream": {
            "source_bytes": stream.get("source_bytes"),
            "source_sha256": stream.get("source_sha256"),
            "reencoded_bytes": stream.get("reencoded_bytes"),
            "reencoded_sha256": stream.get("reencoded_sha256"),
            "decoded_symbol_count": stream.get("decoded_symbol_count"),
            "reencoded_byte_identical": stream.get("reencoded_byte_identical"),
        },
        "dispatch_blockers": list(manifest.get("dispatch_blockers") or []),
    }


def _normalize_pr102_metadata(metadata: dict[str, Any], repo_root: pathlib.Path) -> dict[str, Any]:
    """Fix known manual-intake metadata drift for PR102 corrected custody."""
    manifest_path = _pr102_canonical_manifest(repo_root)
    if manifest_path is None:
        return metadata
    data = _read_json(manifest_path)
    pr_info = data.get("pr", {})
    out = dict(metadata)
    out["leaderboard_name"] = "hnerv_lc_v2_scale095_rplus1"
    out["leaderboard_score"] = 0.195
    out["title"] = pr_info.get("title", out.get("title"))
    out["author"] = pr_info.get("author", out.get("author"))
    out["head_repo"] = pr_info.get("head_repo", out.get("head_repo"))
    out["head_sha"] = pr_info.get("head_sha", out.get("head_sha"))
    out["pr_number"] = pr_info.get("number", out.get("pr_number"))
    return out


def _archive_path_for_pr(pr: int, intake_dir: pathlib.Path, repo_root: pathlib.Path) -> tuple[pathlib.Path, str]:
    if pr == 102:
        corrected = _pr102_correct_archive(repo_root)
        if corrected is not None:
            return corrected, "pr102_zero_byte_tuning_corrected_archive"
    return intake_dir / "archive.zip", "public_pr_intake_full_archive"


def _source_key_files(submission_dir: pathlib.Path, repo_root: pathlib.Path) -> dict[str, Any]:
    names = {
        "inflate_sh": "inflate.sh",
        "inflate_py": "inflate.py",
        "compress_sh": "compress.sh",
        "compress_py": "compress.py",
        "readme": "README.md",
        "report": "report.md",
    }
    records = {
        key: _file_record(submission_dir / name, repo_root)
        for key, name in names.items()
    }
    source_files = []
    if submission_dir.is_dir():
        for path in sorted(submission_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                source_files.append(_file_record(path, repo_root))
    return {
        "submission_dir": _rel(submission_dir, repo_root),
        "submission_dir_present": submission_dir.is_dir(),
        "key_files": records,
        "source_file_count": len(source_files),
        "source_files_sha256": hashlib.sha256(
            "\n".join(
                f"{row['path']} {row['bytes']} {row['sha256']}"
                for row in source_files
                if row is not None
            ).encode("utf-8")
        ).hexdigest(),
    }


def _research_notes_for_pr(pr: int, repo_root: pathlib.Path) -> list[str]:
    roots = [repo_root / ".omx/research", repo_root / "reverse_engineering"]
    patterns = [
        f"*pr{pr}*.md",
        f"*PR{pr}*.md",
        f"*public_pr{pr}*.md",
    ]
    notes: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file():
                    notes.add(_rel(path, repo_root))
    return sorted(notes)


def _exact_eval_artifacts(pr: int, repo_root: pathlib.Path) -> list[dict[str, Any]]:
    root = repo_root / "experiments/results/lightning_batch"
    if not root.is_dir():
        return []
    artifacts: list[dict[str, Any]] = []
    patterns = [f"*pr{pr}*", *PR_EXACT_EVAL_GLOB_ALIASES.get(pr, ())]
    seen_dirs: set[pathlib.Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_dir():
                seen_dirs.add(path)
    for path in sorted(seen_dirs):
        if not path.is_dir():
            continue
        files = sorted(p for p in path.iterdir() if p.is_file())
        json_files = [p for p in files if p.name.startswith("contest_auth_eval") and p.suffix == ".json"]
        log_files = [p for p in files if p.suffix == ".log"]
        row: dict[str, Any] = {
            "dir": _rel(path, repo_root),
            "json_files": [_rel(p, repo_root) for p in json_files],
            "log_files": [_rel(p, repo_root) for p in log_files],
            "archive": _file_record(path / "archive.zip", repo_root),
            "score": None,
            "score_basis": None,
            "device": None,
            "avg_posenet_dist": None,
            "avg_segnet_dist": None,
            "runtime_tree_sha256": None,
            "structured_result_source": None,
            "structured_result_kind": None,
        }
        for json_file in json_files:
            try:
                payload = _read_json(json_file)
            except Exception:
                continue
            normalized = _extract_score_payload(payload)
            row.update({k: v for k, v in normalized.items() if v is not None})
            row["structured_result_source"] = _rel(json_file, repo_root)
            row["structured_result_kind"] = "contest_auth_eval_json_file"
            if row["score"] is not None:
                break
        if row["score"] is None:
            for log_file in log_files:
                normalized = _extract_result_json_from_log(log_file, repo_root)
                if normalized is None:
                    continue
                row.update({k: v for k, v in normalized.items() if v is not None})
                row["structured_result_source"] = normalized.get("source")
                row["structured_result_kind"] = normalized.get("source_kind")
                if row["score"] is not None:
                    break
        artifacts.append(row)
    return artifacts


def _missing_proofs(
    *,
    archive: dict[str, Any],
    source: dict[str, Any],
    exact_evals: list[dict[str, Any]],
    notes: list[str],
    schema_manifest: dict[str, Any] | None = None,
) -> list[str]:
    missing = []
    if not archive.get("present"):
        missing.append("archive_custody_missing")
    if archive.get("blockers"):
        missing.extend(f"archive_{b}" for b in archive["blockers"])
    if not source["submission_dir_present"]:
        missing.append("submission_source_dir_missing")
    if source["key_files"]["inflate_sh"] is None and source["key_files"]["inflate_py"] is None:
        missing.append("inflate_entrypoint_missing")
    if source["key_files"]["compress_sh"] is None and source["key_files"]["compress_py"] is None:
        missing.append("compress_entrypoint_missing")
    same_archive_cuda = _same_archive_cuda_exact_evals(
        archive=archive,
        exact_evals=exact_evals,
    )
    if not same_archive_cuda:
        missing.append("exact_cuda_replay_missing")
    if not _same_archive_structured_json_exact_evals(
        archive=archive,
        exact_evals=exact_evals,
    ):
        missing.append("structured_exact_eval_json_missing")
        missing.append("same_archive_structured_exact_eval_json_missing")
    if not notes:
        missing.append("research_note_missing")
    if not _schema_decode_parity_closed(schema_manifest, archive):
        missing.append("decode_reencode_parity_proof_required")
    missing.append("compress_to_archive_1to1_reproduction_required")
    return sorted(dict.fromkeys(missing))


def build_row(pr: int, repo_root: pathlib.Path) -> dict[str, Any]:
    if pr == 102:
        intake_dir = _pr102_canonical_intake_dir(repo_root) or (
            repo_root / f"experiments/results/public_pr_intake_full/public_pr{pr}_intake_20260505_auto"
        )
    else:
        intake_dir = repo_root / f"experiments/results/public_pr_intake_full/public_pr{pr}_intake_20260505_auto"
    metadata_path = intake_dir / "pr_metadata.json"
    provenance_path = intake_dir / "archive_provenance.json"
    metadata = _read_json(metadata_path) if metadata_path.is_file() else {"pr_number": pr}
    if pr == 102:
        metadata = _normalize_pr102_metadata(metadata, repo_root)
    provenance = _read_json(provenance_path) if provenance_path.is_file() else {}
    leaderboard_name = str(metadata.get("leaderboard_name") or "")
    source_dir = intake_dir / "source/submissions" / leaderboard_name
    archive_path, archive_basis = _archive_path_for_pr(pr, intake_dir, repo_root)
    archive = inspect_archive(archive_path, repo_root)
    source = _source_key_files(source_dir, repo_root)
    notes = _research_notes_for_pr(pr, repo_root)
    exact_evals = _exact_eval_artifacts(pr, repo_root)
    exact_eval_summary = _exact_eval_summary(archive=archive, exact_evals=exact_evals)
    schema_manifest = _schema_manifest_for_pr(pr, repo_root)
    missing = _missing_proofs(
        archive=archive,
        source=source,
        exact_evals=exact_evals,
        notes=notes,
        schema_manifest=schema_manifest,
    )
    binary_understanding = {
        "zip_member_manifest_present": bool(archive.get("members")),
        "payload_prefixes_recorded": bool(archive.get("members")),
        "wire_grammar_status": "member_prefix_only",
        "decode_reencode_parity_status": "missing_or_external",
        "compress_reproduction_status": "missing_or_external",
    }
    binary_understanding.update(_schema_binary_understanding(schema_manifest, archive))
    return {
        "pr": pr,
        "leaderboard_name": leaderboard_name,
        "title": metadata.get("title"),
        "author": metadata.get("author"),
        "head_repo": metadata.get("head_repo"),
        "head_sha": metadata.get("head_sha"),
        "created_at": metadata.get("created_at"),
        "closed_at": metadata.get("closed_at"),
        "merged_at": metadata.get("merged_at"),
        "leaderboard_score": metadata.get("leaderboard_score"),
        "diff_size": {
            "additions": metadata.get("additions"),
            "deletions": metadata.get("deletions"),
            "changed_files": metadata.get("changed_files"),
        },
        "intake": {
            "dir": _rel(intake_dir, repo_root),
            "metadata": _file_record(metadata_path, repo_root),
            "archive_provenance": _file_record(provenance_path, repo_root),
            "archive_provenance_status": provenance.get("status"),
        },
        "archive_basis": archive_basis,
        "archive": archive,
        "source": source,
        "exact_eval_artifacts": exact_evals,
        "exact_eval_summary": exact_eval_summary,
        "leaderboard_replay_drift": _score_drift_against_leaderboard(
            leaderboard_score=metadata.get("leaderboard_score"),
            archive=archive,
            exact_evals=exact_evals,
        ),
        "research_notes": notes,
        "binary_understanding": binary_understanding,
        "ready_for_stack_atom": not missing,
        "missing_proofs": missing,
    }


def build_ledger(
    repo_root: pathlib.Path,
    *,
    pr_numbers: tuple[int, ...] = DEFAULT_PRS,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    rows = [build_row(pr, repo_root) for pr in pr_numbers]
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "created_at_utc": created_at_utc or _utc_iso(),
        "score_claim": False,
        "evidence_grade": "[empirical:local-custody-inventory]",
        "repo_root": _rel(repo_root, repo_root),
        "pr_range": [min(pr_numbers), max(pr_numbers)],
        "summary": {
            "row_count": len(rows),
            "ready_for_stack_atom_count": sum(1 for row in rows if row["ready_for_stack_atom"]),
            "missing_exact_cuda_count": sum(
                1 for row in rows if "exact_cuda_replay_missing" in row["missing_proofs"]
            ),
            "missing_structured_json_count": sum(
                1 for row in rows if "structured_exact_eval_json_missing" in row["missing_proofs"]
            ),
            "missing_same_archive_structured_json_count": sum(
                1
                for row in rows
                if "same_archive_structured_exact_eval_json_missing" in row["missing_proofs"]
            ),
            "same_archive_embedded_result_json_count": sum(
                row.get("exact_eval_summary", {}).get(
                    "same_archive_embedded_result_json_eval_count",
                    0,
                )
                for row in rows
            ),
            "missing_decode_reencode_count": sum(
                1
                for row in rows
                if "decode_reencode_parity_proof_required" in row["missing_proofs"]
            ),
            "leaderboard_replay_drift_count": sum(
                1
                for row in rows
                if str(row.get("leaderboard_replay_drift", {}).get("status", "")).startswith(
                    "leaderboard_mismatches"
                )
            ),
        },
        "rows": rows,
    }


def render_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# PR100-PR107 reproduction and deconstruction ledger",
        "",
        f"Created: `{ledger['created_at_utc']}`",
        "",
        "Evidence grade: `[empirical:local-custody-inventory]`; no score claim.",
        "",
        "| PR | name | leaderboard | local replay | bytes | members | same-archive scored/total evals | missing proof count | first blockers |",
        "|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in ledger["rows"]:
        archive = row["archive"]
        missing = row["missing_proofs"]
        first = ", ".join(missing[:3])
        drift = row.get("leaderboard_replay_drift", {})
        local_scores = []
        for device, entries in (drift.get("local_scores_by_device") or {}).items():
            for entry in entries:
                score = entry.get("score")
                if isinstance(score, (int, float)):
                    local_scores.append(f"{device}:{score:.6f}")
        local_score_cell = "<br>".join(local_scores) if local_scores else drift.get("status")
        exact_summary = row.get("exact_eval_summary") or {}
        eval_count_cell = (
            f"{exact_summary.get('same_archive_scored_eval_count', 0)}/"
            f"{exact_summary.get('total_eval_dir_count', len(row['exact_eval_artifacts']))}"
        )
        lines.append(
            f"| {row['pr']} | {row['leaderboard_name']} | "
            f"{row['leaderboard_score']} | {local_score_cell} | {archive.get('bytes')} | "
            f"{archive.get('member_count', 0)} | {eval_count_cell} | "
            f"{len(missing)} | {first} |"
        )
    lines.extend(
        [
            "",
            "## Required next proof",
            "",
            "Every row must advance from member-prefix inventory to byte-level grammar, "
            "decode/re-encode parity, compress-to-archive reproduction, and exact CUDA "
            "structured JSON before it is used as a promoted stack atom.",
            "",
            "Public leaderboard drift is tracked per replay mode. For PR100+ public "
            "submissions, GitHub comments show CUDA and CPU evals can differ materially; "
            "do not compare a local CUDA replay to a CPU leaderboard row without labeling "
            "the device and runtime tree.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(ledger: dict[str, Any], output_json: pathlib.Path, output_md: pathlib.Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(ledger), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build PR100-PR107 reproduction/deconstruction ledger")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--created-at-utc", default=None)
    args = p.parse_args(argv)
    repo_root = pathlib.Path(args.repo_root).resolve()
    ledger = build_ledger(repo_root, created_at_utc=args.created_at_utc)
    write_outputs(ledger, pathlib.Path(args.output_json), pathlib.Path(args.output_md))
    summary = ledger["summary"]
    print(f"rows: {summary['row_count']}")
    print(f"ready_for_stack_atom_count: {summary['ready_for_stack_atom_count']}")
    print(f"missing_decode_reencode_count: {summary['missing_decode_reencode_count']}")
    print(f"json: {args.output_json}")
    print(f"md: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
