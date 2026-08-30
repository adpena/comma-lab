#!/usr/bin/env python3
"""Certify-or-block cold move of a Vertigo subtree to a cold store.

The destination is whatever ``--dest-root`` names; every headroom number and
ledger ``df`` row is measured on the filesystem that will actually hold it. An
earlier revision hardcoded ``/Volumes/APDataStore`` for the headroom gate while
``--dest-root`` (required) drove the paths — so a caller naming any other tier
was gated against a volume it never asked about, and ``dest_df_before`` in the
ledger named the wrong filesystem. Per the CLAUDE.md storage waterfall
(Vertigo -> APDataStore -> local only by explicit opt-in), a destination off the
external ``/Volumes`` tiers now requires ``--allow-local-tier <rationale>``
before ``--apply``, and a destination on the SOURCE filesystem is refused
outright because such a "move" reclaims nothing while the cert would claim
freed bytes.

Implements the CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup, And Provenance"
certify-or-block rule and the ALWAYS-KEEP-THE-PAYLOAD non-negotiable: no source
byte is retired until a destination copy exists AND a full per-file SHA-256
manifest of the destination equals the source manifest row-for-row.

Deletion class permitted here is exactly one: removal of a source tree whose
destination copy has been hash-verified, with a cert row already durably
appended to the JSONL ledger and a symlink installed at the original path so
every historical citation still resolves.

Cross-filesystem note (established by ddm_ai1 / ddm_sr2): the destination is
ExFAT, which materialises AppleDouble ``._*`` sidecars. Those are metadata, not
data forks; they are excluded from the census, the manifest, and the equality
test. Comparing raw file counts across HFS+ -> ExFAT is a closed dead end.

Ledger rows are appended at every phase transition, so the ledger is valid and
truthful at any interruption point.

Usage:
    python tools/vertigo_certify_move.py \
        --source /Volumes/VertigoDataTier/pact/<tree> \
        --dest-root /Volumes/APDataStore/pact/vertigo_coldstore \
        --ledger .omx/research/ddm_vr1_move_cert_ledger.jsonl \
        --category rebuildable_inflate_output \
        --reason "deterministic inflate.sh output; archive.zip generator co-located" \
        --apply
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

APPLEDOUBLE_PREFIX = "._"
DS_STORE = ".DS_Store"
SCHEMA = "vertigo_certify_move_cert.v1"


class CensusError(RuntimeError):
    """A census could not observe every candidate data file."""


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def is_metadata_sidecar(name: str) -> bool:
    """AppleDouble / Finder sidecars are metadata, never data forks."""
    return name.startswith(APPLEDOUBLE_PREFIX) or name == DS_STORE


@dataclass
class Census:
    root: Path
    files: list[tuple[str, int]] = field(default_factory=list)
    n_dirs: int = 0
    n_symlinks: int = 0
    n_sidecars: int = 0
    logical_bytes: int = 0
    newest_mtime: float = 0.0
    newest_path: str = ""

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "n_data_files": len(self.files),
            "n_dirs": self.n_dirs,
            "n_symlinks": self.n_symlinks,
            "n_metadata_sidecars_excluded": self.n_sidecars,
            "logical_data_bytes": self.logical_bytes,
            "newest_descendant_mtime_utc": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.newest_mtime))
                if self.newest_mtime
                else None
            ),
            "newest_descendant_path": self.newest_path,
        }


def take_census(root: Path) -> Census:
    c = Census(root=root)
    def refuse_walk_error(exc: OSError) -> None:
        raise CensusError(f"unreadable directory during census: {exc}") from exc

    for dirpath, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=refuse_walk_error
    ):
        for dn in list(dirnames):
            p = Path(dirpath) / dn
            try:
                st = p.lstat()
            except OSError as exc:
                raise CensusError(f"unreadable during census: {p}: {exc}") from exc
            if stat.S_ISLNK(st.st_mode):
                c.n_symlinks += 1
                dirnames.remove(dn)
            else:
                c.n_dirs += 1
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.lstat()
            except OSError as exc:
                raise CensusError(f"unreadable during census: {p}: {exc}") from exc
            if stat.S_ISLNK(st.st_mode):
                c.n_symlinks += 1
                continue
            if is_metadata_sidecar(fn):
                c.n_sidecars += 1
                continue
            rel = str(p.relative_to(root))
            c.files.append((rel, st.st_size))
            c.logical_bytes += st.st_size
            if st.st_mtime > c.newest_mtime:
                c.newest_mtime = st.st_mtime
                c.newest_path = rel
    c.files.sort()
    return c


def sha256_file(path: Path, bufsize: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path, rels: list[str], out_path: Path, workers: int = 6) -> str:
    """Write '<sha256>  <size>  <relpath>' rows sorted by relpath; return digest of the manifest."""
    results: dict[str, tuple[str, int]] = {}
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(sha256_file, root / r): r for r in rels}
        for fut in concurrent.futures.as_completed(futs):
            rel = futs[fut]
            try:
                digest = fut.result()
            except OSError as exc:
                raise SystemExit(f"BLOCK: unreadable during hashing: {rel}: {exc}") from exc
            results[rel] = (digest, (root / rel).stat().st_size)
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                print(
                    f"  hashed {done}/{len(rels)} ({el:.0f}s)", file=sys.stderr, flush=True
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out_path, "w") as fh:
        for rel in sorted(results):
            digest, size = results[rel]
            line = f"{digest}  {size}  {rel}\n"
            fh.write(line)
            h.update(line.encode())
    return h.hexdigest()


def copy_and_hash_one(src_root: Path, dst_root: Path, rel: str) -> tuple[str, int]:
    """Read the source once: hash it and write the destination in the same pass.

    The rsync design read the source twice (once to hash, once to copy). On the
    measured Vertigo throughput (6.25 MiB/s single-stream, 15.7 MiB/s at six
    streams) that second full read costs hours, so the copy carries the hash.
    This does NOT weaken the proof: the destination is still re-read from disk
    afterwards and hashed independently.
    """
    s = src_root / rel
    d = dst_root / rel
    d.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    n = 0
    with open(s, "rb") as fi, open(d, "wb") as fo:
        while chunk := fi.read(8 * 1024 * 1024):
            h.update(chunk)
            fo.write(chunk)
            n += len(chunk)
        fo.flush()
        os.fsync(fo.fileno())
    st = s.stat()
    os.utime(d, (st.st_atime, st.st_mtime))
    return h.hexdigest(), n


def copy_and_hash(
    src_root: Path, dst_root: Path, rels: list[str], out_path: Path, workers: int
) -> str:
    """Stream-copy every relpath and return the digest of the source manifest."""
    results: dict[str, tuple[str, int]] = {}
    t0 = time.time()
    done = 0
    bytes_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(copy_and_hash_one, src_root, dst_root, r): r for r in rels}
        for fut in concurrent.futures.as_completed(futs):
            rel = futs[fut]
            try:
                digest, size = fut.result()
            except OSError as exc:
                raise SystemExit(f"BLOCK: copy failed for {rel}: {exc}") from exc
            results[rel] = (digest, size)
            done += 1
            bytes_done += size
            el = max(time.time() - t0, 1e-6)
            if done % 10 == 0 or done == len(rels):
                print(
                    f"  copied+hashed {done}/{len(rels)}  "
                    f"{bytes_done/2**30:.2f} GiB  {bytes_done/el/2**20:.1f} MiB/s",
                    file=sys.stderr,
                    flush=True,
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out_path, "w") as fh:
        for rel in sorted(results):
            digest, size = results[rel]
            line = f"{digest}  {size}  {rel}\n"
            fh.write(line)
            h.update(line.encode())
    return h.hexdigest()


def append_ledger(ledger: Path, row: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("schema", SCHEMA)
    row.setdefault("written_at_utc", utcnow())
    with open(ledger, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def df_kib(mount: str) -> dict:
    """``df -k`` for ``mount``, as a typed row.

    ``device`` (field 0) and the KiB counts (fields 2-3) are positional from the
    left and unambiguous. ``mounted_on`` is the LAST field and is DISPLAY-ONLY:
    a volume name containing spaces truncates it to the final word. Decide on
    ``device``; never on ``mounted_on``.
    """
    out = subprocess.run(
        ["df", "-k", mount], capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()[-1].split()
    return {
        "mount": mount,
        "device": out[0],
        "mounted_on": out[-1],
        "used_kib": int(out[2]),
        "avail_kib": int(out[3]),
    }


def existing_ancestor(path: Path) -> Path:
    """Nearest existing ancestor of ``path``; ``df`` needs a path that exists.

    The destination subtree is created by the copy, so it is absent at
    headroom-check time. What must be measured is the FILESYSTEM that will hold
    it, never a literal mount chosen at authoring time.
    """
    current = path.resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def df_kib_for_path(path: Path) -> dict:
    """``df`` for the filesystem that will actually hold ``path``."""
    return df_kib(str(existing_ancestor(path)))


def is_external_tier(path: Path) -> bool:
    """True when ``path`` resolves onto a mounted external volume (``/Volumes/...``).

    CLAUDE.md's storage waterfall makes local disk a destination only by explicit
    operator opt-in, so the tier is classified before any bulk copy is planned.
    """
    return Path("/Volumes") in path.resolve().parents


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--dest-root", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument(
        "--referenced-by",
        default="",
        help="Comma-separated manifests/receipts that cite this path (recorded, not ignored).",
    )
    ap.add_argument(
        "--no-known-references-rationale",
        default="",
        help="Substantive reason no citing manifest/receipt is known; required for --apply "
        "when --referenced-by is empty.",
    )
    ap.add_argument(
        "--allow-local-tier",
        default="",
        help="Substantive operator rationale for a --dest-root that is NOT on an "
        "external /Volumes tier. CLAUDE.md's storage waterfall makes local disk a "
        "destination only by explicit opt-in, so --apply refuses without it.",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--min-dest-avail-gib",
        type=float,
        default=25.0,
        help="Refuse if the destination volume would drop below this after the copy.",
    )
    ap.add_argument("--apply", action="store_true", help="Without this, census + plan only.")
    ap.add_argument(
        "--retire-source",
        action="store_true",
        help="After verified equality, remove source and install a symlink.",
    )
    args = ap.parse_args()

    referenced_by = [s.strip() for s in args.referenced_by.split(",") if s.strip()]
    no_refs_rationale = args.no_known_references_rationale.strip()
    if args.apply and not referenced_by and len(no_refs_rationale) < 12:
        return _block(
            "--apply requires --referenced-by, or a substantive "
            "--no-known-references-rationale (>=12 chars)"
        )

    src = Path(args.source).resolve()
    if not src.is_dir():
        return _block(f"source is not a directory: {src}")
    if src.is_symlink():
        return _block(f"source is already a symlink (already moved?): {src}")

    vertigo_root = Path("/Volumes/VertigoDataTier")
    try:
        rel_from_vol = src.relative_to(vertigo_root)
    except ValueError:
        return _block(f"source is not under {vertigo_root}: {src}")

    dest = Path(args.dest_root) / rel_from_vol
    ledger = Path(args.ledger)
    manifest_dir = Path(args.dest_root) / "_manifests" / str(rel_from_vol).replace("/", "__")

    print(f"[{utcnow()}] census: {src}", file=sys.stderr, flush=True)
    try:
        census = take_census(src)
    except CensusError as exc:
        return _block(str(exc))
    src_du = int(
        subprocess.run(["du", "-x", "-s", "-k", str(src)], capture_output=True, text=True, check=True)
        .stdout.split()[0]
    )
    print(
        f"  {len(census.files)} data files, {census.logical_bytes/2**30:.3f} GiB logical, "
        f"{src_du/2**20:.3f} GiB allocated, newest {census.as_dict()['newest_descendant_mtime_utc']}",
        file=sys.stderr,
        flush=True,
    )

    dest_root = Path(args.dest_root)
    dest_df = df_kib_for_path(dest_root)
    source_df = df_kib(str(vertigo_root))

    if dest_df["device"] == source_df["device"]:
        return _block(
            f"destination {dest_root} is on the SOURCE filesystem "
            f"({dest_df['device']} mounted at {dest_df['mounted_on']}): a move within "
            "one filesystem reclaims nothing, so the freed-bytes cert would be false"
        )

    dest_is_external = is_external_tier(dest_root)
    local_tier_rationale = args.allow_local_tier.strip()
    if not dest_is_external:
        if args.apply and len(local_tier_rationale) < 12:
            return _block(
                f"destination {dest_root} is NOT on an external /Volumes tier "
                f"(filesystem {dest_df['mounted_on']}); CLAUDE.md's storage waterfall "
                "makes local disk a destination only by explicit opt-in — pass "
                "--allow-local-tier '<substantive operator rationale>' (>=12 chars)"
            )
        print(
            f"[{utcnow()}] NOTICE: destination tier is LOCAL ({dest_df['mounted_on']}), "
            "not an external /Volumes volume; --apply requires --allow-local-tier",
            file=sys.stderr,
            flush=True,
        )

    projected_avail_gib = (dest_df["avail_kib"] - src_du) / 2**20
    plan = {
        "phase": "PLAN",
        "source": str(src),
        "destination": str(dest),
        "category": args.category,
        "reason": args.reason,
        "referenced_by": referenced_by,
        "no_known_references_rationale": no_refs_rationale or None,
        "census": census.as_dict(),
        "source_allocated_kib": src_du,
        "dest_tier": "external" if dest_is_external else "local",
        "local_tier_rationale": local_tier_rationale or None,
        "dest_df_before": dest_df,
        "projected_dest_avail_gib_after": round(projected_avail_gib, 3),
        "vertigo_df_before": source_df,
    }

    if projected_avail_gib < args.min_dest_avail_gib:
        plan["phase"] = "BLOCKED_DEST_HEADROOM"
        append_ledger(ledger, plan)
        return _block(
            f"destination headroom on {dest_df['mounted_on']} "
            f"({dest_df['device']}, {dest_df['avail_kib']/2**20:.1f} GiB free): "
            f"{projected_avail_gib:.1f} GiB after copy "
            f"< floor {args.min_dest_avail_gib} GiB"
        )

    if not args.apply:
        print(json.dumps(plan, indent=2))
        return 0

    append_ledger(ledger, plan)

    rels = [r for r, _ in census.files]

    print(
        f"[{utcnow()}] single-pass copy+hash ({len(rels)} files) -> {dest}",
        file=sys.stderr,
        flush=True,
    )
    dest.mkdir(parents=True, exist_ok=True)
    src_manifest = manifest_dir / "source.sha256"
    t_copy = time.time()
    src_digest = copy_and_hash(src, dest, rels, src_manifest, workers=args.workers)
    append_ledger(
        ledger,
        {
            "phase": "COPIED",
            "source": str(src),
            "destination": str(dest),
            "manifest_path": str(src_manifest),
            "source_manifest_sha256": src_digest,
            "n_rows": len(rels),
            "copy_seconds": round(time.time() - t_copy, 1),
            "note": "source hashed during the copy; destination is re-read and hashed independently below",
        },
    )

    print(f"[{utcnow()}] destination manifest", file=sys.stderr, flush=True)
    try:
        dest_census = take_census(dest)
    except CensusError as exc:
        append_ledger(
            ledger,
            {
                "phase": "BLOCKED_DEST_CENSUS_UNREADABLE",
                "source": str(src),
                "destination": str(dest),
                "error": str(exc),
            },
        )
        return _block(f"destination census incomplete; source untouched: {exc}")
    dest_rels = [r for r, _ in dest_census.files]
    if dest_rels != rels:
        missing = sorted(set(rels) - set(dest_rels))[:10]
        extra = sorted(set(dest_rels) - set(rels))[:10]
        append_ledger(
            ledger,
            {
                "phase": "BLOCKED_PATHSET_MISMATCH",
                "source": str(src),
                "n_source": len(rels),
                "n_dest": len(dest_rels),
                "missing_sample": missing,
                "extra_sample": extra,
            },
        )
        return _block("destination path set != source path set; source untouched")

    dest_manifest = manifest_dir / "destination.sha256"
    dest_digest = build_manifest(dest, dest_rels, dest_manifest, workers=args.workers)

    verified = dest_digest == src_digest
    append_ledger(
        ledger,
        {
            "phase": "VERIFIED" if verified else "BLOCKED_HASH_MISMATCH",
            "source": str(src),
            "destination": str(dest),
            "source_manifest_sha256": src_digest,
            "destination_manifest_sha256": dest_digest,
            "n_rows": len(rels),
            "logical_data_bytes": census.logical_bytes,
            "manifest_dir": str(manifest_dir),
        },
    )
    if not verified:
        return _block("manifest digests differ; source untouched")

    print(f"[{utcnow()}] VERIFIED equal ({len(rels)} rows, {src_digest[:16]}...)", file=sys.stderr)

    if not args.retire_source:
        print("source retained (no --retire-source); nothing freed", file=sys.stderr)
        return 0

    largest = max(census.files, key=lambda t: t[1])[0] if census.files else None
    tmp_old = src.parent / (src.name + ".RETIRING")
    src.rename(tmp_old)
    try:
        src.symlink_to(dest)
    except OSError as exc:
        tmp_old.rename(src)
        append_ledger(ledger, {"phase": "BLOCKED_SYMLINK", "source": str(src), "error": str(exc)})
        return _block(f"symlink install failed, source restored: {exc}")

    probe_ok = bool(largest) and (src / largest).is_file()
    if not probe_ok:
        src.unlink()
        tmp_old.rename(src)
        append_ledger(ledger, {"phase": "BLOCKED_SYMLINK_PROBE", "source": str(src)})
        return _block("symlink did not resolve to a known file; source restored")

    shutil.rmtree(tmp_old)
    after = {
        "vertigo_df_after": df_kib(str(vertigo_root)),
        "dest_df_after": df_kib_for_path(dest),
    }
    append_ledger(
        ledger,
        {
            "phase": "MOVED_SYMLINKED",
            "source": str(src),
            "destination": str(dest),
            "symlink_installed": True,
            "symlink_probe_relpath": largest,
            "freed_allocated_kib": src_du,
            "manifest_sha256": src_digest,
            **after,
        },
    )
    print(f"[{utcnow()}] MOVED + symlinked; freed ~{src_du/2**20:.2f} GiB", file=sys.stderr)
    return 0


def _block(msg: str) -> int:
    print(f"BLOCK: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
