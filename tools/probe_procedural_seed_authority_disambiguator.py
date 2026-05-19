#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Procedural seed authority probe-disambiguator.

Compare two self-contained submission packet variants:

* ``archive_seeded``: score-affecting seed bytes are charged inside
  ``archive.zip``.
* ``runtime_constant`` / ``script_seed``: equivalent seed information is carried
  by ``inflate.sh`` or its runtime code path.

The probe runs each packet's ``inflate.sh`` against a tiny temporary
``file_list`` fixture, hashes inflated outputs, inspects the archive seed
member, and emits an evidence JSON. It does not load scorers, dispatch jobs, or
claim score authority.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.procedural_codebook_generator.authority import (  # noqa: E402
    build_procedural_seed_authority_packet,
)

SCHEMA = "procedural_seed_authority_disambiguator_v1"
DEFAULT_CANDIDATE_ID = "procedural_seed_authority_disambiguator"
FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_paid_dispatch": False,
    "research_only": True,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _is_hidden_or_resource_member(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return any(
        part.startswith(".") or part == "__MACOSX" or part.startswith("._")
        for part in parts
    )


def _validate_zip_member_name(name: str, *, allow_hidden: bool) -> None:
    if not name or name.endswith("/"):
        raise ValueError(f"unsafe archive member name: {name!r}")
    posix = PurePosixPath(name)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"zip-slip archive member rejected: {name!r}")
    if "\\" in name or ":" in name:
        raise ValueError(f"non-portable archive member rejected: {name!r}")
    if not allow_hidden and _is_hidden_or_resource_member(name):
        raise ValueError(f"hidden/resource archive member rejected: {name!r}")


def _packet_paths(packet_dir: Path) -> tuple[Path, Path]:
    packet_dir = packet_dir.resolve()
    archive_zip = packet_dir / "archive.zip"
    inflate_sh = packet_dir / "inflate.sh"
    if not packet_dir.is_dir():
        raise ValueError(f"packet directory not found: {packet_dir}")
    if not archive_zip.is_file():
        raise ValueError(f"packet archive.zip not found: {archive_zip}")
    if not inflate_sh.is_file():
        raise ValueError(f"packet inflate.sh not found: {inflate_sh}")
    return archive_zip, inflate_sh


def _inspect_seed_member(archive_zip: Path, seed_member: str) -> dict[str, object]:
    _validate_zip_member_name(seed_member, allow_hidden=False)
    with zipfile.ZipFile(archive_zip, "r") as zf:
        matches = [info for info in zf.infolist() if info.filename == seed_member]
        if not matches:
            raise ValueError(f"seed member missing from archive: {seed_member!r}")
        if len(matches) > 1:
            raise ValueError(f"duplicate seed member rejected: {seed_member!r}")
        info = matches[0]
        _validate_zip_member_name(info.filename, allow_hidden=False)
        payload = zf.read(info)
    return {
        "member": seed_member,
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "compressed_bytes": int(info.compress_size),
        "crc32": f"{int(info.CRC):08x}",
        "provenance_kind": "archive_member_seed",
    }


def _extract_zip_safely(archive_zip: Path, dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            _validate_zip_member_name(info.filename, allow_hidden=True)
            target = (dest / info.filename).resolve()
            if dest not in target.parents:
                raise ValueError(f"zip-slip archive member rejected: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def _hash_output_dir(output_dir: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(output_dir.rglob("*")):
        rel = path.relative_to(output_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"inflated output symlink rejected: {rel}")
        if not path.is_file():
            continue
        data = path.read_bytes()
        files.append(
            {
                "path": rel,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        )
    if not files:
        raise ValueError(f"inflate produced no output files under {output_dir}")
    return {
        "file_count": len(files),
        "files": files,
        "manifest_sha256": _json_digest(files),
        "provenance_kind": "inflated_output_manifest",
    }


def _run_inflate_variant(
    *,
    mode: str,
    packet_dir: Path,
    file_list_path: Path,
    work_root: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    archive_zip, inflate_sh = _packet_paths(packet_dir)
    archive_dir = work_root / mode / "archive"
    output_dir = work_root / mode / "out"
    _extract_zip_safely(archive_zip, archive_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(inflate_sh), str(archive_dir), str(output_dir), str(file_list_path)],
        cwd=str(work_root),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"{mode} inflate.sh failed with exit {proc.returncode}: "
            f"{proc.stderr[-1000:]}"
        )

    return {
        "mode": mode,
        "packet_dir": str(packet_dir.resolve()),
        "archive_zip": str(archive_zip),
        "archive_sha256": _sha256_file(archive_zip),
        "archive_bytes": archive_zip.stat().st_size,
        "inflate_sh": str(inflate_sh),
        "inflate_sh_sha256": _sha256_file(inflate_sh),
        "inflate_returncode": proc.returncode,
        "stdout_sha256": _sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(proc.stderr.encode("utf-8")),
        "output_manifest": _hash_output_dir(output_dir),
    }


def build_probe_payload(
    *,
    archive_seeded_packet: Path,
    runtime_constant_packet: Path,
    seed_member: str = "seed.bin",
    file_list_entries: tuple[str, ...] = ("0.mkv",),
    candidate_id: str = DEFAULT_CANDIDATE_ID,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    """Run the two packet variants and return the fail-closed probe payload."""

    if not file_list_entries:
        raise ValueError("at least one file-list entry is required")
    for entry in file_list_entries:
        _validate_zip_member_name(entry, allow_hidden=True)

    archive_zip, _ = _packet_paths(archive_seeded_packet)
    seed_info = _inspect_seed_member(archive_zip, seed_member)

    with tempfile.TemporaryDirectory(
        prefix="procedural-seed-authority-disambiguator-"
    ) as tmp:
        work_root = Path(tmp)
        file_list_path = work_root / "file_list.txt"
        file_list_path.write_text(
            "".join(f"{entry}\n" for entry in file_list_entries),
            encoding="utf-8",
        )
        archive_row = _run_inflate_variant(
            mode="archive_seeded",
            packet_dir=archive_seeded_packet,
            file_list_path=file_list_path,
            work_root=work_root,
            timeout_seconds=timeout_seconds,
        )
        runtime_row = _run_inflate_variant(
            mode="runtime_constant",
            packet_dir=runtime_constant_packet,
            file_list_path=file_list_path,
            work_root=work_root,
            timeout_seconds=timeout_seconds,
        )

    archive_manifest = archive_row["output_manifest"]
    runtime_manifest = runtime_row["output_manifest"]
    assert isinstance(archive_manifest, dict)
    assert isinstance(runtime_manifest, dict)
    same_outputs = (
        archive_manifest["manifest_sha256"] == runtime_manifest["manifest_sha256"]
    )

    authority_packet = build_procedural_seed_authority_packet(
        candidate_id,
        modes=("archive_seeded", "runtime_constant"),
        runtime_constant_kind="per_video_payload",
        score_affecting=True,
        runtime_consumption_proof=True,
        self_contained_archive_proof=True,
        scorer_free_inflate_proof=False,
        no_external_state_proof=False,
        packet_compiler_target_declared=False,
        exact_eval_validated=False,
    )

    archive_row["seed_member"] = seed_info
    archive_row["provenance_kind"] = "archive_seeded_submission_packet"
    runtime_row["seed_member"] = None
    runtime_row["provenance_kind"] = "runtime_constant_script_seed_submission_packet"

    return {
        "schema": SCHEMA,
        "candidate_id": str(candidate_id),
        "generated_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **FALSE_AUTHORITY_FLAGS,
        "file_list_fixture": {
            "entries": list(file_list_entries),
            "provenance_kind": "temporary_file_list_fixture",
        },
        "variants": {
            "archive_seeded": archive_row,
            "runtime_constant": runtime_row,
        },
        "comparison": {
            "same_inflated_output_manifest": same_outputs,
            "archive_seeded_output_manifest_sha256": archive_manifest["manifest_sha256"],
            "runtime_constant_output_manifest_sha256": runtime_manifest["manifest_sha256"],
            "provenance_kind": "paired_inflate_output_hash_comparison",
        },
        "authority_packet": authority_packet,
        "authority_disposition": {
            "archive_seeded": authority_packet["modes"]["archive_seeded"],
            "runtime_constant": authority_packet["modes"]["runtime_constant"],
        },
        "blockers": [
            "no_scorer_loaded",
            "no_exact_eval",
            "probe_only_authority",
            "runtime_constant_is_script_side_payload_until_compliance_ruling",
        ],
        "notes": (
            "Equal inflated outputs do not equal equal contest authority: charged "
            "archive seed bytes and script-side runtime constants carry different "
            "compliance dispositions."
        ),
    }


def write_probe_json(
    *,
    output: Path,
    archive_seeded_packet: Path,
    runtime_constant_packet: Path,
    seed_member: str = "seed.bin",
    file_list_entries: tuple[str, ...] = ("0.mkv",),
    candidate_id: str = DEFAULT_CANDIDATE_ID,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    payload = build_probe_payload(
        archive_seeded_packet=archive_seeded_packet,
        runtime_constant_packet=runtime_constant_packet,
        seed_member=seed_member,
        file_list_entries=file_list_entries,
        candidate_id=candidate_id,
        timeout_seconds=timeout_seconds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare archive-seeded vs runtime-constant procedural seed packet "
            "authority. Runs inflate.sh only; no scorer load, network, or dispatch."
        )
    )
    parser.add_argument("--archive-seeded-packet", type=Path, required=True)
    parser.add_argument(
        "--runtime-constant-packet",
        "--script-seed-packet",
        dest="runtime_constant_packet",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed-member", default="seed.bin")
    parser.add_argument(
        "--file-list-entry",
        action="append",
        default=None,
        help="Entry to place in the temporary file_list fixture. Repeatable.",
    )
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        payload = write_probe_json(
            output=args.output,
            archive_seeded_packet=args.archive_seeded_packet,
            runtime_constant_packet=args.runtime_constant_packet,
            seed_member=args.seed_member,
            file_list_entries=tuple(args.file_list_entry or ["0.mkv"]),
            candidate_id=args.candidate_id,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    print(
        "procedural-seed authority probe: "
        f"wrote {args.output} "
        f"(same_outputs={comparison['same_inflated_output_manifest']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
