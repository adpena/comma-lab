#!/usr/bin/env python3
"""Hardlink-dedup byte-identical files on one tier. Frees blocks without deleting anything.

Fourth sr5 producer, and the one the certify-or-block rule actually prefers: when two
paths hold the same bytes, collapsing them onto one inode releases the duplicate's
blocks while every original path keeps working and every byte stays readable. Nothing
is destroyed, so there is no rebuild to certify -- only an equality to prove.

The proof is a FULL sha256 of every candidate, never a sampled prefix. A prefilter may
choose which files to hash; it may never decide that two files are equal.

Per group, in order:

  1. every member is a regular file, not a symlink, with st_nlink == 1 (otherwise the
     link is already shared and collapsing it frees nothing, or something else owns it);
  2. every member has the same size and the same full sha256 as the keeper;
  3. the cert row lands BEFORE the link is installed;
  4. the link is installed through a temporary name and os.replace, so the member path
     is never absent, and the result is verified to share the keeper's inode and to
     still hash to the same value.

Caution carried from sr4 and recorded on every row: hardlinked files share future
writes. Any regeneration must write to a fresh path, never in place over a shared inode.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "ddm_sr5_certified_hardlink_dedup.v1"

PROTECTED_SUBSTRINGS = (
    "/ddm_pc3_",
    "/ddm_ntb2_",
    "/ddm_mxo2",
    "/ddm_mxo3",
    "/ddm_rlc5_cure_on_move43",
    "/ddm_sj1_pass6",
    "/ddm_rp1_round2",
    "/ddm_sr4_20260910/retained/ap_offload",
    "/public_datasets",
    "/upstream/",
)


class DedupError(RuntimeError):
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


def assert_not_protected(path: Path) -> None:
    text = str(path.resolve())
    for needle in PROTECTED_SUBSTRINGS:
        if needle in text:
            raise DedupError(f"protected path refused: {text} matches {needle!r}")


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


def inspect(path: Path) -> dict:
    assert_not_protected(path)
    if not path.is_file() or path.is_symlink():
        raise DedupError(f"not a regular file: {path}")
    stat = os.lstat(path)
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "st_nlink": int(stat.st_nlink),
        "st_ino": int(stat.st_ino),
        "st_dev": int(stat.st_dev),
        "sha256": sha256_file(path),
    }


def link_over(keeper: Path, member: Path) -> dict:
    """Replace member with a hardlink to keeper, without the path ever being absent."""
    temporary = member.with_name(member.name + ".sr5dedup.tmp")
    if temporary.exists():
        raise DedupError(f"temporary already exists: {temporary}")
    os.link(keeper, temporary)
    try:
        os.replace(temporary, member)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    after = os.lstat(member)
    keeper_stat = os.lstat(keeper)
    return {
        "member_ino_after": int(after.st_ino),
        "keeper_ino": int(keeper_stat.st_ino),
        "shares_inode": after.st_ino == keeper_stat.st_ino and after.st_dev == keeper_stat.st_dev,
        "st_nlink_after": int(after.st_nlink),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", required=True, help="Repeat once per candidate file.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--mount", required=True)
    parser.add_argument("--apply", action="store_true", help="Without this, hash and group only.")
    args = parser.parse_args()

    ledger = Path(args.ledger).resolve()
    tool_sha = sha256_file(Path(__file__).resolve())
    paths = [Path(p) for p in args.path]

    records: list[dict] = []
    for path in paths:
        try:
            records.append(inspect(path))
        except (DedupError, OSError) as exc:
            row = {
                "schema": SCHEMA,
                "written_at_utc": utcnow(),
                "path": str(path),
                "verdict": "REFUSED",
                "refusal_reason": f"{type(exc).__name__}: {exc}",
                "tool_sha256": tool_sha,
                "score_claim": False,
            }
            append_ledger(ledger, row)
            print(f"[sr5-dedup] REFUSED {path}: {exc}", file=sys.stderr)

    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["bytes"], record["sha256"])].append(record)

    before = df_snapshot(args.mount)
    linked = 0
    freed = 0
    refused = 0
    for (size, digest), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda r: r["path"])
        keeper, rest = members[0], members[1:]
        if len({m["st_ino"] for m in members}) == 1:
            print(f"[sr5-dedup] group {digest[:12]} already shares one inode; nothing to free")
            continue
        for member in rest:
            row = {
                "schema": SCHEMA,
                "written_at_utc": utcnow(),
                "group_sha256": digest,
                "group_bytes": size,
                "keeper": keeper,
                "member": member,
                "reason": args.reason,
                "caution": (
                    "Hardlinked files share future writes; any regeneration must target a fresh "
                    "output path, never an in-place write over a shared inode."
                ),
                "tool": "tools/sr5_certify_hardlink_dedup.py",
                "tool_sha256": tool_sha,
                "score_claim": False,
                "promotable": False,
                "verdict": "CERTIFIED_BYTE_IDENTICAL",
                "bytes_released": size,
            }
            if member["st_nlink"] != 1 or keeper["st_nlink"] != 1:
                row["verdict"] = "REFUSED"
                row["refusal_reason"] = "st_nlink != 1 on keeper or member; collapsing frees nothing"
                refused += 1
                append_ledger(ledger, row)
                continue
            if member["st_dev"] != keeper["st_dev"]:
                row["verdict"] = "REFUSED"
                row["refusal_reason"] = "different devices; a hardlink cannot span filesystems"
                refused += 1
                append_ledger(ledger, row)
                continue
            if not args.apply:
                row["action"] = "PLAN_ONLY_NO_MUTATION"
                append_ledger(ledger, row)
                print(f"[sr5-dedup] PLAN link {member['path']} -> {keeper['path']} ({size} B)")
                continue
            row["action"] = "LINK_PENDING"
            append_ledger(ledger, row)
            result = link_over(Path(keeper["path"]), Path(member["path"]))
            row.update(result)
            verified_sha = sha256_file(Path(member["path"]))
            row["member_sha256_after"] = verified_sha
            if not result["shares_inode"] or verified_sha != digest:
                row["action"] = "FAILED_VERIFY_AFTER_LINK"
                refused += 1
                append_ledger(ledger, row)
                print(f"[sr5-dedup] FAILED verify after link: {member['path']}", file=sys.stderr)
                continue
            row["action"] = "LINKED"
            append_ledger(ledger, row)
            linked += 1
            freed += size
            print(f"[sr5-dedup] LINKED {member['path']} -> {keeper['path']} ({size / 1024**3:.3f} GiB)")

    after = df_snapshot(args.mount)
    summary = {
        "schema": SCHEMA + ".summary",
        "written_at_utc": utcnow(),
        "candidates": len(paths),
        "hashed": len(records),
        "groups_with_duplicates": sum(1 for members in groups.values() if len(members) > 1),
        "linked": linked,
        "refused": refused,
        "bytes_released_accounted": freed,
        "reason": args.reason,
        "df_before": before,
        "df_after": after,
        "df_free_delta_bytes": int(after["free_bytes"]) - int(before["free_bytes"]),
        "apply": bool(args.apply),
        "tool_sha256": tool_sha,
        "note": "Nothing was deleted. Every original path still resolves to the same bytes.",
    }
    append_ledger(ledger, summary)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if refused == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
