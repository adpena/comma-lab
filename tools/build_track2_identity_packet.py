#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a non-promotable Track 2 custom-decoder identity packet.

The scaffold proves byte-closed packet custody before any custom-decoder
optimization starts. It copies an existing contest packet byte-for-byte,
records archive/runtime identity, and fails closed on unsafe archive/runtime
surfaces. It does not run scorers, claim score, dispatch jobs, or mark the
packet promotable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import shutil
import stat
import struct
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, repo_relative, sha256_file  # noqa: E402

SCHEMA = "track2_custom_decoder_identity_packet_v1"
RUNTIME_MANIFEST_SCHEMA = "track2_identity_runtime_manifest_v1"
DEFAULT_CANDIDATE_ID = "track2_custom_decoder_identity"
DEFAULT_MANIFEST_NAME = "track2_identity_manifest.json"
PAYLOAD_CONTAINER_NAMES = (
    "p",
    "renderer_payload.bin",
    "renderer_payload.bin.br",
    "0.bin",
    "x",
)
FORBIDDEN_RUNTIME_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


class Track2IdentityPacketError(ValueError):
    """Raised when the identity scaffold cannot prove a closed packet."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return repo_relative(path, REPO_ROOT)
    except ValueError:
        return str(path)


def _format_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: str | None) -> dt.datetime:
    if value is None:
        return dt.datetime.now(tz=dt.UTC).replace(microsecond=0)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC).replace(microsecond=0)


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mode_string(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def _unsafe_archive_member_name(name: str | None) -> str | None:
    if not name:
        return "empty_member_name"
    if "\\" in name:
        return "backslash_member_name"
    if "\x00" in name or any(ord(ch) < 32 for ch in name):
        return "control_character_member_name"
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        return "windows_drive_member_name"
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        return "zip_slip_member_name"
    if "__MACOSX" in member.parts:
        return "macosx_resource_directory"
    if any(part.startswith("._") for part in member.parts):
        return "resource_fork_member_name"
    if member.name in {".DS_Store", "Thumbs.db"}:
        return "resource_sidecar_member_name"
    if any(part.startswith(".") for part in member.parts):
        return "hidden_sidecar_member_name"
    return None


def _decode_zip_name(raw: bytes, flag_bits: int) -> str | None:
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    try:
        return raw.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        return None


def _local_header_name(path: Path, info: zipfile.ZipInfo) -> str | None:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            return None
        flag_bits = struct.unpack_from("<H", header, 6)[0]
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        raw_name = handle.read(name_len)
        handle.read(extra_len)
    return _decode_zip_name(raw_name, flag_bits)


def _check_failed(checks: list[dict[str, Any]], name: str, details: str) -> None:
    checks.append({"name": name, "passed": False, "details": details})


def _check_passed(checks: list[dict[str, Any]], name: str, details: str) -> None:
    checks.append({"name": name, "passed": True, "details": details})


def inspect_archive(archive_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inspect ``archive.zip`` and return a manifest plus fail-closed checks."""

    checks: list[dict[str, Any]] = []
    if not archive_path.is_file():
        _check_failed(checks, "archive_exists", _display_path(archive_path))
        return {"path": _display_path(archive_path), "exists": False}, checks

    archive_record: dict[str, Any] = {
        "path": _display_path(archive_path),
        "exists": True,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "member_count": 0,
        "member_names": [],
        "members": [],
    }
    _check_passed(checks, "archive_exists", _display_path(archive_path))

    try:
        with zipfile.ZipFile(archive_path) as zf:
            bad_crc = zf.testzip()
            if bad_crc is None:
                _check_passed(checks, "archive_crc_ok", "all member CRCs read cleanly")
            else:
                _check_failed(checks, "archive_crc_ok", f"bad_crc_member={bad_crc}")

            infos = zf.infolist()
            names = [info.filename for info in infos]
            archive_record["member_count"] = len(infos)
            archive_record["member_names"] = list(names)
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            if duplicate_names:
                _check_failed(checks, "zip_no_duplicate_members", f"duplicates={duplicate_names}")
            else:
                _check_passed(checks, "zip_no_duplicate_members", f"members={len(names)}")

            payload_containers = [name for name in names if name in PAYLOAD_CONTAINER_NAMES]
            archive_record["payload_container_members"] = payload_containers
            if len(payload_containers) > 1:
                _check_failed(
                    checks,
                    "zip_at_most_one_payload_container",
                    f"payload_container_members={payload_containers}",
                )
            else:
                _check_passed(
                    checks,
                    "zip_at_most_one_payload_container",
                    f"payload_container_members={payload_containers}",
                )

            for info in infos:
                local_name = _local_header_name(archive_path, info)
                unsafe = _unsafe_archive_member_name(info.filename)
                local_unsafe = _unsafe_archive_member_name(local_name)
                if info.is_dir():
                    unsafe = unsafe or "directory_member_not_supported"
                if unsafe is None:
                    _check_passed(checks, f"zip_member_safe:{info.filename}", info.filename)
                else:
                    _check_failed(checks, f"zip_member_safe:{info.filename}", unsafe)
                if local_name == info.filename and local_unsafe is None:
                    _check_passed(
                        checks,
                        f"zip_local_header_matches:{info.filename}",
                        f"local={local_name!r}",
                    )
                else:
                    _check_failed(
                        checks,
                        f"zip_local_header_matches:{info.filename}",
                        f"central={info.filename!r} local={local_name!r} local_reason={local_unsafe}",
                    )

                member_payload = b"" if info.is_dir() else zf.read(info)
                archive_record["members"].append(
                    {
                        "name": info.filename,
                        "local_header_name": local_name,
                        "file_size": info.file_size,
                        "compress_size": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                        "sha256": None if info.is_dir() else _sha256_bytes(member_payload),
                        "compress_type": info.compress_type,
                        "flag_bits": info.flag_bits,
                        "header_offset": info.header_offset,
                    }
                )
    except zipfile.BadZipFile as exc:
        _check_failed(checks, "archive_zip_readable", repr(exc))
        return archive_record, checks

    _check_passed(checks, "archive_zip_readable", f"members={archive_record['member_count']}")
    return archive_record, checks


def _runtime_path_blocker(rel: Path) -> str | None:
    rel_posix = rel.as_posix()
    if not rel_posix:
        return "empty_runtime_path"
    if "\\" in rel_posix:
        return "runtime_path_uses_backslashes"
    if "\x00" in rel_posix or any(ord(ch) < 32 for ch in rel_posix):
        return "runtime_path_control_character"
    pure = PurePosixPath(rel_posix)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        return "runtime_path_not_relative_safe"
    if "__MACOSX" in pure.parts or any(part.startswith("._") for part in pure.parts):
        return "runtime_path_resource_fork_or_macosx"
    if pure.name in FORBIDDEN_RUNTIME_NAMES or any(part.startswith(".") for part in pure.parts):
        return "runtime_path_hidden_or_system"
    return None


def _runtime_file_record(root: Path, rel: Path) -> dict[str, Any]:
    path = root / rel
    return {
        "relative_path": rel.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mode": _mode_string(path),
        "executable": bool(path.stat().st_mode & stat.S_IXUSR),
    }


def _runtime_tree_sha256(files: list[dict[str, Any]]) -> str:
    basis = [
        {
            "relative_path": row["relative_path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "mode": row["mode"],
        }
        for row in sorted(files, key=lambda item: str(item["relative_path"]))
    ]
    return _canonical_json_sha256(basis)


def inspect_runtime_tree(packet_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    """Inspect runtime files selected for identity copy."""

    checks: list[dict[str, Any]] = []
    inflate_sh = packet_dir / "inflate.sh"
    if not inflate_sh.is_file():
        _check_failed(checks, "inflate_entrypoint_present", _display_path(inflate_sh))
        return {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "packet_dir": _display_path(packet_dir),
            "inflate_entrypoint": {"path": "inflate.sh", "exists": False},
            "files": [],
            "file_count": 0,
            "runtime_tree_sha256": _runtime_tree_sha256([]),
        }, checks, []
    _check_passed(checks, "inflate_entrypoint_present", _display_path(inflate_sh))
    if inflate_sh.stat().st_mode & stat.S_IXUSR:
        _check_passed(checks, "inflate_entrypoint_executable", _mode_string(inflate_sh))
    else:
        _check_failed(checks, "inflate_entrypoint_executable", _mode_string(inflate_sh))

    try:
        proc = subprocess.run(
            ["bash", "-n", str(inflate_sh)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode == 0:
            _check_passed(checks, "inflate_entrypoint_bash_syntax", "bash -n passed")
        else:
            _check_failed(
                checks,
                "inflate_entrypoint_bash_syntax",
                proc.stderr.strip() or f"returncode={proc.returncode}",
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _check_failed(checks, "inflate_entrypoint_bash_syntax", f"{exc.__class__.__name__}: {exc}")

    rels: list[Path] = []
    files: list[dict[str, Any]] = []
    for path in sorted(packet_dir.rglob("*"), key=lambda item: item.relative_to(packet_dir).as_posix()):
        rel = path.relative_to(packet_dir)
        if rel.as_posix() == "archive.zip":
            continue
        blocker = _runtime_path_blocker(rel)
        if blocker is not None:
            _check_failed(checks, f"runtime_path_safe:{rel.as_posix()}", blocker)
            continue
        _check_passed(checks, f"runtime_path_safe:{rel.as_posix()}", rel.as_posix())
        if path.is_symlink():
            _check_failed(checks, f"runtime_file_regular:{rel.as_posix()}", "symlink_not_supported")
            continue
        if not path.is_file():
            continue
        _check_passed(checks, f"runtime_file_regular:{rel.as_posix()}", rel.as_posix())
        rels.append(rel)
        files.append(_runtime_file_record(packet_dir, rel))

    manifest = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "packet_dir": _display_path(packet_dir),
        "inflate_entrypoint": {
            "path": "inflate.sh",
            "exists": True,
            "executable": bool(inflate_sh.stat().st_mode & stat.S_IXUSR),
            "bytes": inflate_sh.stat().st_size,
            "sha256": sha256_file(inflate_sh),
            "mode": _mode_string(inflate_sh),
        },
        "files": sorted(files, key=lambda item: str(item["relative_path"])),
        "file_count": len(files),
        "runtime_tree_sha256": _runtime_tree_sha256(files),
    }
    return manifest, checks, rels


def _raise_if_failed(checks: list[dict[str, Any]]) -> None:
    failures = [check for check in checks if not check["passed"]]
    if not failures:
        return
    first = failures[0]
    raise Track2IdentityPacketError(
        f"Track 2 identity packet validation failed: {first['name']}: {first['details']}"
    )


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise Track2IdentityPacketError(f"output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise Track2IdentityPacketError(
                f"output directory is not empty; pass --force to replace: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _copy_file_identity(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    os.chmod(target, stat.S_IMODE(source.stat().st_mode))


def _identity_record(source: Path, target: Path, relpath: str) -> dict[str, Any]:
    return {
        "relative_path": relpath,
        "source_bytes": source.stat().st_size,
        "output_bytes": target.stat().st_size,
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(target),
        "source_mode": _mode_string(source),
        "output_mode": _mode_string(target),
        "byte_identical": source.read_bytes() == target.read_bytes(),
        "mode_identical": stat.S_IMODE(source.stat().st_mode) == stat.S_IMODE(target.stat().st_mode),
    }


def build_track2_identity_packet(
    *,
    source_packet_dir: Path,
    output_packet_dir: Path,
    candidate_id: str = DEFAULT_CANDIDATE_ID,
    recorded_at_utc: dt.datetime | None = None,
    force: bool = False,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> dict[str, Any]:
    source_packet_dir = _repo_path(source_packet_dir).resolve()
    output_packet_dir = _repo_path(output_packet_dir).resolve()
    recorded_at_utc = recorded_at_utc or dt.datetime.now(tz=dt.UTC).replace(microsecond=0)

    if not source_packet_dir.is_dir():
        raise FileNotFoundError(f"source packet directory not found: {source_packet_dir}")
    if "/" in manifest_name or "\\" in manifest_name or not manifest_name.endswith(".json"):
        raise Track2IdentityPacketError(f"manifest_name must be a simple .json filename: {manifest_name!r}")
    if output_packet_dir == source_packet_dir:
        raise Track2IdentityPacketError("output packet directory must differ from source packet directory")
    try:
        output_packet_dir.relative_to(source_packet_dir)
    except ValueError:
        pass
    else:
        raise Track2IdentityPacketError("output packet directory must not be inside source packet directory")
    try:
        source_packet_dir.relative_to(output_packet_dir)
    except ValueError:
        pass
    else:
        raise Track2IdentityPacketError("output packet directory must not contain source packet directory")

    source_archive = source_packet_dir / "archive.zip"
    archive_record, archive_checks = inspect_archive(source_archive)
    runtime_manifest, runtime_checks, runtime_rels = inspect_runtime_tree(source_packet_dir)
    all_checks = archive_checks + runtime_checks
    _raise_if_failed(all_checks)

    _prepare_output_dir(output_packet_dir, force=force)
    output_archive = output_packet_dir / "archive.zip"
    _copy_file_identity(source_archive, output_archive)
    runtime_identity: list[dict[str, Any]] = []
    for rel in runtime_rels:
        source = source_packet_dir / rel
        target = output_packet_dir / rel
        _copy_file_identity(source, target)
        runtime_identity.append(_identity_record(source, target, rel.as_posix()))

    output_archive_record, output_archive_checks = inspect_archive(output_archive)
    output_runtime_manifest, output_runtime_checks, _ = inspect_runtime_tree(output_packet_dir)
    post_copy_checks = output_archive_checks + output_runtime_checks
    _raise_if_failed(post_copy_checks)

    archive_identity = _identity_record(source_archive, output_archive, "archive.zip")
    runtime_byte_identical = all(row["byte_identical"] and row["mode_identical"] for row in runtime_identity)
    runtime_tree_identical = (
        runtime_manifest["runtime_tree_sha256"] == output_runtime_manifest["runtime_tree_sha256"]
    )
    archive_byte_identical = bool(archive_identity["byte_identical"])
    identity_passed = archive_byte_identical and runtime_byte_identical and runtime_tree_identical
    if identity_passed:
        _check_passed(all_checks, "identity_copy_byte_closed", "archive and runtime copied byte-for-byte")
    else:
        _check_failed(
            all_checks,
            "identity_copy_byte_closed",
            (
                f"archive={archive_byte_identical} runtime={runtime_byte_identical} "
                f"runtime_tree={runtime_tree_identical}"
            ),
        )
        _raise_if_failed(all_checks)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": "tools.build_track2_identity_packet",
        "track": "track_2_custom_decoder",
        "mode": "identity",
        "candidate_id": candidate_id,
        "recorded_at_utc": _format_utc(recorded_at_utc),
        "source_packet_dir": _display_path(source_packet_dir),
        "output_packet_dir": _display_path(output_packet_dir),
        "status": {
            "optimization_applied": False,
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "score_claim": False,
            "ranking_claim": False,
            "promotion_eligible": False,
            "dispatchable": False,
            "ready_for_exact_eval_dispatch": False,
            "evidence_grade": "byte_custody_only",
            "classification": "non_optimized_identity_scaffold",
        },
        "archive": {
            "source": archive_record,
            "output": output_archive_record,
            "identity": archive_identity,
        },
        "runtime_manifest": {
            "source": runtime_manifest,
            "output": output_runtime_manifest,
            "identity": {
                "runtime_file_identity": sorted(
                    runtime_identity,
                    key=lambda item: str(item["relative_path"]),
                ),
                "source_runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
                "output_runtime_tree_sha256": output_runtime_manifest["runtime_tree_sha256"],
                "runtime_tree_identical": runtime_tree_identical,
                "runtime_files_byte_identical": runtime_byte_identical,
            },
        },
        "byte_closure": {
            "archive_copied_byte_for_byte": archive_byte_identical,
            "runtime_copied_byte_for_byte": runtime_byte_identical,
            "runtime_tree_sha256_identical": runtime_tree_identical,
            "identity_roundtrip_passed": identity_passed,
            "inflate_output_roundtrip_attempted": False,
            "inflate_output_roundtrip_reason": (
                "scorers and inflate execution are intentionally not run by this "
                "identity scaffold; exact CUDA auth eval is the promotion gate"
            ),
        },
        "fail_closed_checks": all_checks + post_copy_checks,
        "remaining_blockers": [
            "exact_cuda_auth_eval_missing",
            "contest_auth_eval_adjudication_missing",
            "pre_submission_compliance_check_strict_missing",
            "level2_dispatch_claim_missing_for_future_remote_eval",
            "custom_decoder_optimization_not_implemented",
        ],
        "next_required_gates": [
            "experiments/contest_auth_eval.py --device cuda on exact archive.zip when a candidate is ready",
            "scripts/pre_submission_compliance_check.py --strict before any release or promotion claim",
            "tools/claim_lane_dispatch.py claim before any future remote GPU dispatch",
        ],
    }
    payload["manifest_sha256_excluding_self"] = _canonical_json_sha256(payload)

    manifest_path = output_packet_dir / manifest_name
    manifest_path.write_text(json_text(payload), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-packet-dir", type=Path, required=True)
    parser.add_argument("--output-packet-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--now-utc", default=None)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        payload = build_track2_identity_packet(
            source_packet_dir=args.source_packet_dir,
            output_packet_dir=args.output_packet_dir,
            candidate_id=args.candidate_id,
            recorded_at_utc=_parse_utc(args.now_utc),
            force=args.force,
            manifest_name=args.manifest_name,
        )
    except (OSError, Track2IdentityPacketError, subprocess.SubprocessError) as exc:
        print(f"FATAL: {exc}")
        return 2

    print(f"[track2-identity] output_packet_dir={payload['output_packet_dir']}")
    print(
        "[track2-identity] archive_sha256="
        f"{payload['archive']['output']['sha256']} bytes={payload['archive']['output']['bytes']}"
    )
    print(
        "[track2-identity] runtime_tree_sha256="
        f"{payload['runtime_manifest']['output']['runtime_tree_sha256']}"
    )
    print("[track2-identity] score_claim=false promotion_eligible=false dispatchable=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
