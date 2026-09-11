#!/usr/bin/env python3
"""Relocate one large file between tiers, hashing the DESTINATION before retiring the source.

Third sr5 producer.  Its whole reason to exist is the vr7 lesson (memory
`moved_labels_are_not_custody_vr7_deleted_moved_payloads_behind_live_redirects_20260910`):
an apply step deleted eleven moved payloads behind live redirects because a MOVE
LABEL was treated as custody.  A label is not custody.  Bytes on the destination,
read back from disk and hashed, are custody.

Order of operations, and it is not negotiable:

  1. refuse unless the destination tier keeps its reserve after the copy;
  2. refuse unless the source hashes to the sha256 the caller declares;
  3. stream-copy, fsync, then re-read the DESTINATION from disk and hash it again
     (an independent read, not the streaming digest, so a bad write cannot pass);
  4. write the cert row, with both hashes, BEFORE anything is retired;
  5. repoint the redirect symlink, and verify it now resolves to the destination;
  6. only then unlink the source, and record that as its own ledger row.

Any failure stops before step 6, so the worst case is two copies of the bytes.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA = "ddm_sr5_certified_relocation.v1"


class RelocateError(RuntimeError):
    """Raised when the tool refuses to proceed."""


def utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, bufsize: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(bufsize)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def df_snapshot(mount: str) -> dict[str, object]:
    stat = os.statvfs(mount)
    out = subprocess.run(["df", "-k", mount], capture_output=True, text=True, check=True).stdout.strip()
    return {"mount": mount, "df_k": out, "free_bytes": int(stat.f_bavail) * int(stat.f_frsize)}


def mount_of(path: Path) -> str:
    parts = path.resolve().parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return f"/{parts[1]}/{parts[2]}"
    return "/"


def append_ledger(ledger: Path, row: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    with ledger.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def copy_with_fsync(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)
    fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-sha256", required=True, help="The sha256 the source must already have.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--update-symlink",
        default=None,
        help="A redirect that currently points at the source and must point at the destination after.",
    )
    parser.add_argument(
        "--dest-reserve-gib",
        type=float,
        default=40.0,
        help="Refuse if the destination tier would fall below this after the copy.",
    )
    parser.add_argument("--apply", action="store_true", help="Without this, plan and refusal checks only.")
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.destination)
    ledger = Path(args.ledger).resolve()
    row: dict[str, object] = {
        "schema": SCHEMA,
        "written_at_utc": utcnow(),
        "source": str(source),
        "destination": str(destination),
        "declared_sha256": args.expected_sha256,
        "reason": args.reason,
        "redirect_symlink": args.update_symlink,
        "tool": "tools/sr5_certify_relocate_file.py",
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "score_claim": False,
        "promotable": False,
        "phase": "REFUSED",
    }
    try:
        if not source.is_file() or source.is_symlink():
            raise RelocateError("source is not a regular file")
        if destination.exists():
            raise RelocateError("destination already exists; refusing to overwrite")
        size = source.stat().st_size
        row["bytes"] = size

        dest_mount = mount_of(destination.parent if destination.parent.exists() else destination.parent.parent)
        src_mount = mount_of(source)
        row["source_mount"] = src_mount
        row["destination_mount"] = dest_mount
        if src_mount == dest_mount:
            raise RelocateError("source and destination are on the same tier; a move would free nothing")
        before = {"source_tier": df_snapshot(src_mount), "destination_tier": df_snapshot(dest_mount)}
        row["df_before"] = before
        projected = int(before["destination_tier"]["free_bytes"]) - size
        row["projected_destination_free_bytes"] = projected
        if projected < int(args.dest_reserve_gib * (1024**3)):
            raise RelocateError(
                f"destination tier would fall to {projected / 1024**3:.3f} GiB, "
                f"below the {args.dest_reserve_gib} GiB reserve"
            )

        if args.update_symlink:
            link = Path(args.update_symlink)
            if not link.is_symlink():
                raise RelocateError("redirect path is not a symlink")
            current = os.readlink(link)
            row["redirect_before"] = current
            if Path(current).resolve() != source.resolve():
                raise RelocateError(f"redirect points at {current!r}, not at the source")

        source_sha = sha256_file(source)
        row["source_sha256"] = source_sha
        if source_sha != args.expected_sha256:
            raise RelocateError("source sha256 does not match the declared value")
    except (RelocateError, OSError, ValueError) as exc:
        row["refusal_reason"] = f"{type(exc).__name__}: {exc}"
        append_ledger(ledger, row)
        print(json.dumps(row, indent=1, sort_keys=True))
        return 2

    if not args.apply:
        row["phase"] = "PLAN_ONLY_NO_MUTATION"
        append_ledger(ledger, row)
        print(json.dumps(row, indent=1, sort_keys=True))
        return 0

    copy_with_fsync(source, destination)
    # Independent read-back. The streaming digest is not trusted as proof of what landed.
    destination_sha = sha256_file(destination)
    row["destination_sha256"] = destination_sha
    row["destination_bytes"] = destination.stat().st_size
    if destination_sha != source_sha or row["destination_bytes"] != row["bytes"]:
        row["phase"] = "REFUSED_DESTINATION_MISMATCH_SOURCE_RETAINED"
        row["refusal_reason"] = "destination read-back differs from the source; source left in place"
        append_ledger(ledger, row)
        print(json.dumps(row, indent=1, sort_keys=True), file=sys.stderr)
        return 3

    row["phase"] = "DESTINATION_VERIFIED_SOURCE_STILL_PRESENT"
    append_ledger(ledger, row)

    if args.update_symlink:
        link = Path(args.update_symlink)
        temporary = link.with_name(link.name + ".sr5.tmp")
        os.symlink(str(destination), temporary)
        os.replace(temporary, link)
        resolved = Path(os.readlink(link))
        if resolved != destination or not link.resolve().is_file():
            row["phase"] = "REFUSED_REDIRECT_NOT_REPOINTED_SOURCE_RETAINED"
            append_ledger(ledger, row)
            print(json.dumps(row, indent=1, sort_keys=True), file=sys.stderr)
            return 4
        row["redirect_after"] = str(resolved)

    source.unlink()
    final = dict(row)
    final["phase"] = "SOURCE_RETIRED"
    final["written_at_utc"] = utcnow()
    final["df_after"] = {
        "source_tier": df_snapshot(row["source_mount"]),
        "destination_tier": df_snapshot(row["destination_mount"]),
    }
    append_ledger(ledger, final)
    print(json.dumps(final, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
