#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ruff: noqa: I001
# pyc-recovery pass2: rehydrated from git blob bca3af5700822210f92d708931694a289f7b3b33 via `git fsck --lost-found`
# original path: experiments/preflight_public_replay_intake.py
# This is OUR source, dropped during commit 66c59aae filter-repo cleanup; the .pyc was the only
# orphan left behind. Original blob SHA verified intact.
# Recovered: 2026-05-05 by Sherlock pass2
"""Static public-PR replay intake preflight.

This tool validates a public archive/runtime pair before any exact-eval
dispatch.  It is intentionally local and static: it never inflates videos,
loads scorers, runs CUDA, submits remote jobs, or makes score claims.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
for _path in (SRC_ROOT, EXPERIMENTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from contest_auth_eval import (
    _runtime_dependency_manifest,
    _validate_archive_members,
    _validate_zip_container_integrity,
)
from tac.pr85_bundle import (
    Pr85BundleError,
    parse_pr85_bundle,
)
from tac.hnerv_lowlevel_packer import parse_ff_packed_brotli_hnerv
from tac.optimization.archive_bound_candidate_contract import (
    archive_bound_candidate_contract_fields_for_row,
)
from tac.submission_archive import validate_archive_member_name


SCHEMA = "public_replay_intake_preflight_v1"
TOOL = "experiments/preflight_public_replay_intake.py"
EVIDENCE_GRADE = "external/local_preflight_non_score_until_cuda"
PUBLIC_REPLAY_ARCHIVE_BOUND_FAMILY = "public_frontier_replay_intake"
PUBLIC_REPLAY_ARCHIVE_BOUND_TRANSFORM_KIND = "public_frontier_replay_archive_intake"
SOURCE_EMBEDDED_PAYLOAD_LITERAL_RE = re.compile(
    r"(?:b64decode|b85decode|a85decode|brotli\.decompress|lzma\.decompress|zlib\.decompress)"
    r"\s*\(\s*([rubfRUBF]*[\"'])(?P<payload>.{65536,}?)(?<!\\)\1",
    re.DOTALL,
)
RUNTIME_HYGIENE_FILE_NAMES = frozenset({".gitignore"})


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _blocker_codes(blockers: list[dict[str, str]]) -> list[str]:
    return [str(row.get("code") or row) for row in blockers]


def _magic_ascii(data: bytes, n: int = 8) -> str:
    return data[:n].decode("ascii", errors="replace")


def _try_brotli(data: bytes) -> tuple[bytes | None, dict[str, Any]]:
    try:
        import brotli
    except ImportError:
        return None, {"attempted": True, "ok": False, "error": "brotli_not_available"}
    try:
        decoded = brotli.decompress(data)
    except brotli.error as exc:
        return None, {"attempted": True, "ok": False, "error": str(exc)}
    return decoded, {
        "attempted": True,
        "ok": True,
        "decoded_bytes": len(decoded),
        "decoded_sha256": _sha256_bytes(decoded),
        "decoded_magic_ascii": _magic_ascii(decoded),
        "decoded_magic_hex": decoded[:8].hex(),
    }


def _try_gzip(data: bytes) -> tuple[bytes | None, dict[str, Any]]:
    try:
        decoded = gzip.decompress(data)
    except OSError as exc:
        return None, {"attempted": True, "ok": False, "error": str(exc)}
    return decoded, {
        "attempted": True,
        "ok": True,
        "decoded_bytes": len(decoded),
        "decoded_sha256": _sha256_bytes(decoded),
        "decoded_magic_ascii": _magic_ascii(decoded),
        "decoded_magic_hex": decoded[:8].hex(),
    }


def _contract_to_json(contract: Any) -> dict[str, Any]:
    return {
        "name": contract.name,
        "codec": contract.codec,
        "bytes": int(contract.bytes),
        "sha256": contract.sha256,
        "magic": contract.magic,
        "metadata": dict(contract.metadata),
    }


def _probe_pr85_family_x(raw: bytes) -> dict[str, Any]:
    bundle = parse_pr85_bundle(raw)
    segments: list[dict[str, Any]] = []
    for name, contract in bundle.segment_contracts.items():
        row = _contract_to_json(contract)
        segment = bytes(bundle.segments[name])
        if name != "mask":
            decoded, decompression = _try_brotli(segment)
            row["brotli_decode"] = decompression
            if decoded is not None:
                row["decoded_magic_ascii"] = _magic_ascii(decoded)
                row["decoded_magic_hex"] = decoded[:8].hex()
        segments.append(row)
    return {
        "recognized": True,
        "format": bundle.format,
        "header_bytes": int(bundle.header_bytes),
        "segment_lengths": bundle.segment_lengths,
        "fixed_length_segments": dict(bundle.fixed_length_segments),
        "segments": segments,
    }


def _probe_member(name: str, data: bytes) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "magic_ascii": _magic_ascii(data),
        "magic_hex": data[:8].hex(),
        "status": "passed",
    }
    lower = name.lower()
    try:
        if data.startswith(b"\xff"):
            row["format"] = _probe_hnerv_ff_payload(data)
        elif name == "x":
            row["format"] = _probe_pr85_family_x(data)
        elif lower.endswith(".br"):
            _decoded, row["decompression"] = _try_brotli(data)
            if not row["decompression"]["ok"]:
                row["status"] = "failed"
                row["blocker"] = "brotli_decode_failed"
        elif lower.endswith(".gz"):
            _decoded, row["decompression"] = _try_gzip(data)
            if not row["decompression"]["ok"]:
                row["status"] = "failed"
                row["blocker"] = "gzip_decode_failed"
        elif lower.endswith(".qma9"):
            if not data.startswith(b"QMA9"):
                row["status"] = "failed"
                row["blocker"] = "qma9_magic_mismatch"
        elif lower.endswith(".json"):
            json.loads(data.decode("utf-8"))
        elif lower.endswith(".txt"):
            data.decode("utf-8")
    except (Pr85BundleError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        row["status"] = "failed"
        row["blocker"] = f"{exc.__class__.__name__}: {exc}"
    return row


def _probe_hnerv_ff_payload(raw: bytes) -> dict[str, Any]:
    packed = parse_ff_packed_brotli_hnerv(raw)
    decoder_raw, decoder_decode = _try_brotli(packed.decoder_packed_brotli)
    latents_raw, latents_decode = _try_brotli(packed.latents_and_sidecar_brotli)
    if decoder_raw is None:
        raise ValueError("hnerv decoder brotli decode failed")
    if latents_raw is None:
        raise ValueError("hnerv latents brotli decode failed")
    return {
        "recognized": True,
        "format": "hnerv_ff_len24_brotli_sections",
        "header_bytes": len(packed.header),
        "decoder_packed_brotli": {
            "bytes": len(packed.decoder_packed_brotli),
            "sha256": _sha256_bytes(packed.decoder_packed_brotli),
            "brotli_decode": decoder_decode,
        },
        "latents_and_sidecar_brotli": {
            "bytes": len(packed.latents_and_sidecar_brotli),
            "sha256": _sha256_bytes(packed.latents_and_sidecar_brotli),
            "brotli_decode": latents_decode,
        },
    }


def _runtime_source_payload_scan(
    runtime_root: Path,
    archive_bytes: int,
    *,
    extra_roots: list[Path] | None = None,
) -> dict[str, Any]:
    source_bytes = 0
    violations: list[str] = []
    files: list[dict[str, Any]] = []
    roots = [runtime_root, *(extra_roots or [])]
    for root in roots:
        root = root.resolve()
        root_label = _repo_rel(root)
        if not root.exists():
            violations.append(f"declared runtime dependency root missing: {root_label}")
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if path.name in RUNTIME_HYGIENE_FILE_NAMES or "__pycache__" in Path(rel).parts:
                continue
            if any(part.startswith(".") or part.startswith("._") for part in Path(rel).parts):
                violations.append(f"hidden runtime file: {root_label}/{rel}")
                continue
            if path.suffix.lower() not in {".py", ".sh"}:
                continue
            raw = path.read_bytes()
            source_bytes += len(raw)
            file_row = {
                "root": root_label,
                "path": rel,
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
            }
            if path.suffix.lower() == ".py":
                text = raw.decode("utf-8", errors="ignore")
                if SOURCE_EMBEDDED_PAYLOAD_LITERAL_RE.search(text):
                    violations.append(
                        f"{root_label}/{rel} contains a >=64KiB encoded/decompressed literal"
                    )
                    file_row["large_encoded_payload_literal"] = True
            files.append(file_row)
    if archive_bytes <= 1024 and source_bytes > 64 * 1024:
        violations.append(
            f"archive is {archive_bytes} bytes but runtime .py/.sh source is {source_bytes} bytes"
        )
    return {
        "runtime_root": _repo_rel(runtime_root),
        "extra_roots": [_repo_rel(path) for path in (extra_roots or [])],
        "source_bytes_py_sh": source_bytes,
        "files": files,
        "violations": violations,
        "status": "passed" if not violations else "failed",
    }


def _archive_report(archive: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    if not archive.is_file():
        return {
            "path": _repo_rel(archive),
            "status": "failed",
            "error": "archive_missing",
        }, [{"code": "archive_missing", "detail": str(archive)}]

    report: dict[str, Any] = {
        "path": _repo_rel(archive),
        "bytes": archive.stat().st_size,
        "sha256": _sha256_file(archive),
        "status": "passed",
    }
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            report["member_names"] = names
            report["duplicate_member_names"] = duplicate_names
            if duplicate_names:
                blockers.append(
                    {
                        "code": "duplicate_member_names",
                        "detail": ", ".join(duplicate_names),
                    }
                )
            try:
                _validate_zip_container_integrity(archive, infos)
                report["zip_container_integrity"] = "passed"
            except RuntimeError as exc:
                report["zip_container_integrity"] = "failed"
                blockers.append({"code": "zip_container_integrity", "detail": str(exc)})

            member_rows = []
            for info in infos:
                row: dict[str, Any] = {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "crc32_hex": f"{info.CRC:08x}",
                }
                try:
                    validate_archive_member_name(info.filename)
                    row["name_grammar"] = "passed"
                except ValueError as exc:
                    row["name_grammar"] = "failed"
                    row["name_grammar_error"] = str(exc)
                    blockers.append({"code": "member_name_grammar", "detail": str(exc)})
                try:
                    data = zf.read(info)
                except Exception as exc:  # pragma: no cover - corrupt zips vary by platform
                    row["decode_smoke"] = {"status": "failed", "blocker": str(exc)}
                    blockers.append({"code": "member_decode_smoke", "detail": str(exc)})
                else:
                    probe = _probe_member(info.filename, data)
                    row["decode_smoke"] = probe
                    if probe["status"] != "passed":
                        blockers.append(
                            {
                                "code": "member_decode_smoke",
                                "detail": f"{info.filename}: {probe.get('blocker')}",
                            }
                        )
                member_rows.append(row)
            report["members"] = member_rows

            try:
                _validate_archive_members(names)
                report["charged_member_allowlist"] = "passed"
            except RuntimeError as exc:
                report["charged_member_allowlist"] = "failed"
                blockers.append({"code": "charged_member_allowlist", "detail": str(exc)})
    except zipfile.BadZipFile as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        blockers.append({"code": "bad_zip_file", "detail": str(exc)})

    if blockers:
        report["status"] = "failed"
    return report, blockers


def build_preflight(
    archive: Path,
    inflate_sh: Path,
    *,
    upstream_dir: Path = REPO_ROOT / "upstream",
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    expected_runtime_tree_sha256: str | None = None,
) -> dict[str, Any]:
    archive = Path(archive)
    inflate_sh = Path(inflate_sh)
    upstream_dir = Path(upstream_dir)
    archive_report, blockers = _archive_report(archive)

    if expected_archive_sha256:
        actual = archive_report.get("sha256")
        if actual != expected_archive_sha256:
            blockers.append(
                {
                    "code": "expected_archive_sha256_matches",
                    "detail": f"expected={expected_archive_sha256} actual={actual}",
                }
            )
    if expected_archive_size_bytes is not None:
        actual = archive_report.get("bytes")
        if actual != expected_archive_size_bytes:
            blockers.append(
                {
                    "code": "expected_archive_size_bytes_matches",
                    "detail": f"expected={expected_archive_size_bytes} actual={actual}",
                }
            )

    runtime_report: dict[str, Any]
    if not inflate_sh.is_file():
        runtime_report = {
            "inflate_sh": _repo_rel(inflate_sh),
            "status": "failed",
            "error": "inflate_sh_missing",
        }
        blockers.append({"code": "inflate_sh_missing", "detail": str(inflate_sh)})
    else:
        runtime_manifest = _runtime_dependency_manifest(inflate_sh, upstream_dir)
        extra_runtime_roots = [
            Path(row["root"])
            for row in runtime_manifest.get("external_dependency_roots", [])
            if isinstance(row, dict) and row.get("root")
        ]
        runtime_report = {
            "inflate_sh": _repo_rel(inflate_sh),
            "inflate_sh_sha256": _sha256_file(inflate_sh),
            "runtime_root": runtime_manifest.get("runtime_root"),
            "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
            "runtime_manifest": runtime_manifest,
            "source_payload_scan": _runtime_source_payload_scan(
                inflate_sh.parent,
                int(archive_report.get("bytes") or 0),
                extra_roots=extra_runtime_roots,
            ),
            "status": "passed",
        }
        if (
            expected_runtime_tree_sha256
            and runtime_manifest.get("runtime_tree_sha256") != expected_runtime_tree_sha256
        ):
            blockers.append(
                {
                    "code": "expected_runtime_tree_sha256_matches",
                    "detail": (
                        f"expected={expected_runtime_tree_sha256} "
                        f"actual={runtime_manifest.get('runtime_tree_sha256')}"
                    ),
                }
            )
        if runtime_report["source_payload_scan"]["status"] != "passed":
            runtime_report["status"] = "failed"
            for detail in runtime_report["source_payload_scan"]["violations"]:
                blockers.append({"code": "runtime_source_or_sidecar_payload", "detail": detail})

    public_replay_preclaim_ready = not blockers
    archive_status_passed = archive_report.get("status") == "passed"
    runtime_status_passed = runtime_report.get("status") == "passed"
    archive_sha = archive_report.get("sha256")
    candidate_id_suffix = archive_sha[:16] if isinstance(archive_sha, str) else "missing"
    contract_row = {
        "schema": "public_replay_intake_archive_bound_candidate_row.v1",
        "candidate_id": f"public_replay_intake_{candidate_id_suffix}",
        "candidate_family": PUBLIC_REPLAY_ARCHIVE_BOUND_FAMILY,
        "archive_native_transform_kind": PUBLIC_REPLAY_ARCHIVE_BOUND_TRANSFORM_KIND,
        "candidate_archive_path": archive_report.get("path"),
        "candidate_archive_sha256": archive_sha,
        "candidate_archive_bytes": archive_report.get("bytes"),
        "byte_closed_candidate_materialized": archive_status_passed,
        "candidate_archive_materialized": archive_status_passed,
        "runtime_adapter_ready": runtime_status_passed,
        "contest_runtime_decoder_adapter_ready": runtime_status_passed,
        "runtime_adapter_manifest": {
            "schema": "public_replay_intake_runtime_adapter_manifest.v1",
            "inflate_sh": runtime_report.get("inflate_sh"),
            "inflate_sh_sha256": runtime_report.get("inflate_sh_sha256"),
            "runtime_root": runtime_report.get("runtime_root"),
            "runtime_tree_sha256": runtime_report.get("runtime_tree_sha256"),
            "runtime_adapter_ready": runtime_status_passed,
            "contest_runtime_decoder_adapter_ready": runtime_status_passed,
        },
        "runtime_consumption_proof_ready": False,
        "receiver_contract_satisfied": False,
        "receiver_contract_kind": "public_replay_static_preflight_requires_runtime_consumption_proof",
        "semantic_payload_changed": False,
        "score_affecting_payload_changed": True,
        "exact_axis_score_affecting_adjudication_required": True,
        "charged_bits_changed": True,
        "blockers": _blocker_codes(blockers),
        "canonical_anti_pattern_ids": [
            "proxy_or_advisory_signal_masquerades_as_score_authority",
            "archive_static_preflight_without_receiver_consumption",
        ],
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    contract_fields = archive_bound_candidate_contract_fields_for_row(
        contract_row,
        repo_root=REPO_ROOT,
        family_id=PUBLIC_REPLAY_ARCHIVE_BOUND_FAMILY,
        candidate_chain_id=str(contract_row["candidate_id"]),
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "archive": archive_report,
        "runtime": runtime_report,
        "evidence_grade": EVIDENCE_GRADE,
        "public_replay_preclaim_ready": public_replay_preclaim_ready,
        "public_replay_preclaim_blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "cuda_required_for_score": True,
        "dispatch_performed": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        **contract_fields,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--inflate-sh", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--expected-archive-sha256", default=None)
    parser.add_argument("--expected-archive-size-bytes", type=int, default=None)
    parser.add_argument("--expected-runtime-tree-sha256", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_preflight(
        args.archive,
        args.inflate_sh,
        upstream_dir=args.upstream_dir,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_archive_size_bytes=args.expected_archive_size_bytes,
        expected_runtime_tree_sha256=args.expected_runtime_tree_sha256,
    )
    text = _json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 2 if args.fail_if_not_ready and not payload["public_replay_preclaim_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
