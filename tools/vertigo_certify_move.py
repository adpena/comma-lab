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
byte is retired until a destination representation exists and content, exact
modes, and symlink targets all survive an independent verification pass.

Deletion class permitted here is exactly one: removal of a source tree whose
destination has been fidelity-verified, with a cert row already durably
appended to the JSONL ledger. A direct-tree move leaves a transparent source
symlink. A tar-wrapped move leaves a symlink to the retained archive and records
that callers must restore it onto a metadata-capable filesystem before use.

Cross-filesystem note (established by ddm_ai1 / ddm_sr2): some destinations
materialise AppleDouble ``._*`` sidecars. Those are metadata, not data forks;
they are excluded from the content census. Destination capability is measured
by try-create/stat/readlink probes, never inferred from a filesystem label.

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
import tarfile
import tempfile
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
    file_modes: dict[str, int] = field(default_factory=dict)
    dir_modes: dict[str, int] = field(default_factory=dict)
    symlinks: list[tuple[str, str]] = field(default_factory=list)
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
            "file_modes": {path: oct(mode) for path, mode in sorted(self.file_modes.items())},
            "dir_modes": {path: oct(mode) for path, mode in sorted(self.dir_modes.items())},
            "symlink_targets": dict(sorted(self.symlinks)),
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
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise CensusError(f"unreadable census root: {root}: {exc}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise CensusError(f"census root is not a directory: {root}")
    c.dir_modes["."] = stat.S_IMODE(root_stat.st_mode)

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
                try:
                    c.symlinks.append((str(p.relative_to(root)), os.readlink(p)))
                except OSError as exc:
                    raise CensusError(f"unreadable symlink target: {p}: {exc}") from exc
                dirnames.remove(dn)
            elif stat.S_ISDIR(st.st_mode):
                c.dir_modes[str(p.relative_to(root))] = stat.S_IMODE(st.st_mode)
            else:
                raise CensusError(f"non-directory entry in directory walk: {p}")
            if not stat.S_ISLNK(st.st_mode):
                c.n_dirs += 1
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.lstat()
            except OSError as exc:
                raise CensusError(f"unreadable during census: {p}: {exc}") from exc
            if stat.S_ISLNK(st.st_mode):
                c.n_symlinks += 1
                try:
                    c.symlinks.append((str(p.relative_to(root)), os.readlink(p)))
                except OSError as exc:
                    raise CensusError(f"unreadable symlink target: {p}: {exc}") from exc
                continue
            if is_metadata_sidecar(fn):
                c.n_sidecars += 1
                continue
            if not stat.S_ISREG(st.st_mode):
                raise CensusError(f"unsupported special filesystem node: {p}")
            rel = str(p.relative_to(root))
            c.files.append((rel, st.st_size))
            c.file_modes[rel] = stat.S_IMODE(st.st_mode)
            c.logical_bytes += st.st_size
            if st.st_mtime > c.newest_mtime:
                c.newest_mtime = st.st_mtime
                c.newest_path = rel
    c.files.sort()
    c.symlinks.sort()
    return c


def metadata_manifest_rows(census: Census) -> list[dict[str, object]]:
    """Canonical POSIX metadata rows paired with the existing content manifest.

    File SHA-256 values remain in ``source.sha256`` / ``destination.sha256``.  This
    companion manifest keys exact modes and symlink targets by the same relative
    paths, so equality means content *and* metadata survived; neither digest alone
    is deletion authority.
    """

    rows: list[dict[str, object]] = []
    for path, mode in sorted(census.dir_modes.items()):
        rows.append({"path": path, "type": "directory", "mode": mode})
    for path, _size in census.files:
        rows.append(
            {
                "path": path,
                "type": "regular_file",
                "mode": census.file_modes[path],
            }
        )
    for path, target in census.symlinks:
        rows.append(
            {
                "path": path,
                "type": "symlink",
                "mode": None,
                "symlink_target": target,
            }
        )
    return sorted(rows, key=lambda row: (str(row["path"]), str(row["type"])))


def write_metadata_manifest(census: Census, out_path: Path) -> str:
    """Write canonical JSONL metadata rows and return their SHA-256 digest."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with out_path.open("w", encoding="utf-8") as stream:
        for row in metadata_manifest_rows(census):
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest()


@dataclass(frozen=True)
class DestinationMetadataCapabilities:
    probe_root: str
    symlink_supported: bool
    file_mode_results: dict[int, int | None]
    dir_mode_results: dict[int, int | None]
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "vertigo_destination_metadata_capabilities.v1",
            "probe_root": self.probe_root,
            "symlink_supported": self.symlink_supported,
            "file_mode_results": {
                oct(requested): None if observed is None else oct(observed)
                for requested, observed in sorted(self.file_mode_results.items())
            },
            "dir_mode_results": {
                oct(requested): None if observed is None else oct(observed)
                for requested, observed in sorted(self.dir_mode_results.items())
            },
            "errors": list(self.errors),
        }


def probe_destination_metadata_capabilities(
    destination_root: Path,
    *,
    file_modes: set[int],
    dir_modes: set[int],
    require_symlink: bool,
) -> DestinationMetadataCapabilities:
    """Probe the destination filesystem itself; mount labels are not authority.

    ExFAT implementations differ.  The active APDataStore driver can create a
    symlink but maps every tested POSIX mode to 0700, so a filesystem-name check
    would be both over-broad and under-precise.  Exact try/create/stat/readlink
    probes make representability a measured property of the selected destination.
    """

    probe_root = existing_ancestor(destination_root)
    file_results: dict[int, int | None] = {}
    dir_results: dict[int, int | None] = {}
    errors: list[str] = []
    symlink_supported = not require_symlink
    try:
        with tempfile.TemporaryDirectory(
            prefix=".vertigo_metadata_probe_", dir=probe_root
        ) as temp_dir:
            base = Path(temp_dir)
            for index, requested in enumerate(sorted(file_modes)):
                path = base / f"file_{index}"
                try:
                    path.write_bytes(b"metadata-probe\n")
                    os.chmod(path, requested)
                    file_results[requested] = stat.S_IMODE(path.stat().st_mode)
                except OSError as exc:
                    file_results[requested] = None
                    errors.append(f"file_mode_{oct(requested)}:{type(exc).__name__}:{exc}")
            for index, requested in enumerate(sorted(dir_modes)):
                path = base / f"dir_{index}"
                try:
                    path.mkdir()
                    os.chmod(path, requested)
                    dir_results[requested] = stat.S_IMODE(path.stat().st_mode)
                except OSError as exc:
                    dir_results[requested] = None
                    errors.append(f"dir_mode_{oct(requested)}:{type(exc).__name__}:{exc}")
            if require_symlink:
                target = base / "symlink_target"
                target.write_bytes(b"target\n")
                link = base / "symlink_probe"
                try:
                    link.symlink_to("symlink_target")
                    symlink_supported = (
                        stat.S_ISLNK(link.lstat().st_mode)
                        and os.readlink(link) == "symlink_target"
                    )
                except OSError as exc:
                    symlink_supported = False
                    errors.append(f"symlink:{type(exc).__name__}:{exc}")
    except OSError as exc:
        errors.append(f"probe_root:{type(exc).__name__}:{exc}")
        for requested in file_modes:
            file_results.setdefault(requested, None)
        for requested in dir_modes:
            dir_results.setdefault(requested, None)
        symlink_supported = False if require_symlink else symlink_supported
    return DestinationMetadataCapabilities(
        probe_root=str(probe_root),
        symlink_supported=symlink_supported,
        file_mode_results=file_results,
        dir_mode_results=dir_results,
        errors=tuple(errors),
    )


def direct_move_metadata_blockers(
    census: Census, capabilities: DestinationMetadataCapabilities
) -> list[str]:
    """Reasons a direct tree copy cannot preserve the source metadata exactly."""

    blockers: list[str] = []
    if census.symlinks and not capabilities.symlink_supported:
        blockers.append("symlink_targets_not_representable")
    for mode in sorted(set(census.file_modes.values())):
        observed = capabilities.file_mode_results.get(mode)
        if observed != mode:
            blockers.append(
                f"regular_file_mode_not_representable:{oct(mode)}->"
                f"{('unavailable' if observed is None else oct(observed))}"
            )
    for mode in sorted(set(census.dir_modes.values())):
        observed = capabilities.dir_mode_results.get(mode)
        if observed != mode:
            blockers.append(
                f"directory_mode_not_representable:{oct(mode)}->"
                f"{('unavailable' if observed is None else oct(observed))}"
            )
    return blockers


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
                raise CensusError(f"unreadable during hashing: {rel}: {exc}") from exc
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
    shutil.copystat(s, d, follow_symlinks=False)
    return h.hexdigest(), n


def copy_and_hash(
    src_root: Path,
    dst_root: Path,
    rels: list[str],
    out_path: Path,
    workers: int,
    *,
    census: Census,
) -> str:
    """Stream-copy content plus representable POSIX metadata.

    This function is reached only when the destination probe proved every source
    mode and symlink target representable.  Fidelity is nevertheless rechecked
    from a fresh destination census before source retirement.
    """

    for rel, _mode in sorted(census.dir_modes.items(), key=lambda item: item[0].count("/")):
        if rel == ".":
            dst_root.mkdir(parents=True, exist_ok=True)
        else:
            (dst_root / rel).mkdir(parents=True, exist_ok=True)
    for rel, target in census.symlinks:
        link = dst_root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
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
                raise CensusError(f"copy failed for {rel}: {exc}") from exc
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
    for rel, mode in sorted(
        census.dir_modes.items(), key=lambda item: item[0].count("/"), reverse=True
    ):
        os.chmod(dst_root if rel == "." else dst_root / rel, mode)
    return h.hexdigest()


def create_metadata_preserving_tar(source: Path, archive_path: Path) -> dict[str, object]:
    """Atomically create the tar-wrap used when direct metadata is unrepresentable."""

    if archive_path.exists():
        raise CensusError(f"tar destination already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(f"{archive_path.name}.partial-{os.getpid()}")
    if partial.exists():
        raise CensusError(f"tar partial already exists: {partial}")

    def filter_sidecars(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if is_metadata_sidecar(Path(info.name).name) else info

    try:
        with tarfile.open(partial, mode="w", format=tarfile.PAX_FORMAT, dereference=False) as tar:
            tar.add(source, arcname=source.name, recursive=True, filter=filter_sidecars)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        partial.replace(archive_path)
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }


def verify_tar_roundtrip(
    *,
    source: Path,
    source_census: Census,
    source_content_manifest_sha256: str,
    archive_path: Path,
    workers: int,
) -> dict[str, object]:
    """Extract on the source filesystem and verify content, mode, and links."""

    expected_metadata_rows = metadata_manifest_rows(source_census)
    with tempfile.TemporaryDirectory(
        prefix=f".{source.name}.tar-restore-verify-", dir=source.parent
    ) as temp_dir:
        verify_root = Path(temp_dir)
        with tarfile.open(archive_path, mode="r") as tar:
            tar.extractall(verify_root, filter="fully_trusted")
        restored = verify_root / source.name
        restored_census = take_census(restored)
        restored_manifest = verify_root / "restored.sha256"
        restored_content_digest = build_manifest(
            restored,
            [rel for rel, _size in restored_census.files],
            restored_manifest,
            workers=workers,
        )
        restored_metadata_rows = metadata_manifest_rows(restored_census)
        if restored_content_digest != source_content_manifest_sha256:
            raise CensusError("tar restore content manifest differs from source")
        if restored_metadata_rows != expected_metadata_rows:
            raise CensusError("tar restore metadata manifest differs from source")
    return {
        "content_manifest_sha256": source_content_manifest_sha256,
        "metadata_rows": len(expected_metadata_rows),
        "metadata_manifest_equal": True,
        "content_manifest_equal": True,
        "restore_filesystem": str(source.parent),
    }


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

    raw_src = Path(args.source)
    if raw_src.is_symlink():
        return _block(f"source is already a symlink (already moved?): {raw_src}")
    src = raw_src.resolve()
    if not src.is_dir():
        return _block(f"source is not a directory: {src}")

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

    capabilities = probe_destination_metadata_capabilities(
        dest_root,
        file_modes=set(census.file_modes.values()),
        dir_modes=set(census.dir_modes.values()),
        require_symlink=bool(census.symlinks),
    )
    metadata_blockers = direct_move_metadata_blockers(census, capabilities)
    transfer_mode = "tar_wrap" if metadata_blockers else "direct_tree"
    transfer_destination = (
        dest.with_name(f"{dest.name}.tar") if transfer_mode == "tar_wrap" else dest
    )
    projected_avail_gib = (dest_df["avail_kib"] - src_du) / 2**20
    plan = {
        "phase": "PLAN",
        "source": str(src),
        "destination": str(transfer_destination),
        "transfer_mode": transfer_mode,
        "direct_move_refused": bool(metadata_blockers),
        "direct_move_metadata_blockers": metadata_blockers,
        "destination_metadata_capabilities": capabilities.as_dict(),
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

    rels = [r for r, _ in census.files]
    src_manifest = manifest_dir / "source.sha256"
    src_metadata_manifest = manifest_dir / "source.metadata.jsonl"
    src_metadata_digest = write_metadata_manifest(census, src_metadata_manifest)
    append_ledger(ledger, plan)

    if transfer_mode == "tar_wrap":
        print(
            f"[{utcnow()}] direct move REFUSED ({'; '.join(metadata_blockers)})",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[{utcnow()}] metadata-preserving tar-wrap -> {transfer_destination}",
            file=sys.stderr,
            flush=True,
        )
        if transfer_destination.exists():
            return _block(f"tar destination exists; source untouched: {transfer_destination}")
        src_digest = build_manifest(src, rels, src_manifest, workers=args.workers)
        t_copy = time.time()
        try:
            tar_receipt = create_metadata_preserving_tar(src, transfer_destination)
            post_source_census = take_census(src)
            post_source_rels = [rel for rel, _size in post_source_census.files]
            post_source_manifest = manifest_dir / "source.post_tar.sha256"
            post_source_digest = build_manifest(
                src,
                post_source_rels,
                post_source_manifest,
                workers=args.workers,
            )
            post_source_metadata_manifest = manifest_dir / "source.post_tar.metadata.jsonl"
            post_source_metadata_digest = write_metadata_manifest(
                post_source_census, post_source_metadata_manifest
            )
            if post_source_rels != rels or post_source_digest != src_digest:
                raise CensusError("source content changed while tar-wrap was materialized")
            if post_source_metadata_digest != src_metadata_digest:
                raise CensusError("source metadata changed while tar-wrap was materialized")
            restore_verification = verify_tar_roundtrip(
                source=src,
                source_census=post_source_census,
                source_content_manifest_sha256=post_source_digest,
                archive_path=transfer_destination,
                workers=args.workers,
            )
        except (CensusError, OSError, tarfile.TarError) as exc:
            append_ledger(
                ledger,
                {
                    "phase": "BLOCKED_TAR_WRAP_FIDELITY",
                    "source": str(src),
                    "destination": str(transfer_destination),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return _block(f"tar-wrap fidelity failed; source untouched: {exc}")
        append_ledger(
            ledger,
            {
                "phase": "TAR_WRAPPED_VERIFIED",
                "source": str(src),
                "destination": str(transfer_destination),
                "tar": tar_receipt,
                "source_content_manifest_path": str(src_manifest),
                "source_content_manifest_sha256": src_digest,
                "source_metadata_manifest_path": str(src_metadata_manifest),
                "source_metadata_manifest_sha256": src_metadata_digest,
                "post_tar_source_content_manifest_path": str(post_source_manifest),
                "post_tar_source_content_manifest_sha256": post_source_digest,
                "post_tar_source_metadata_manifest_path": str(
                    post_source_metadata_manifest
                ),
                "post_tar_source_metadata_manifest_sha256": post_source_metadata_digest,
                "source_stable_during_tar_materialization": True,
                "restore_verification": restore_verification,
                "copy_seconds": round(time.time() - t_copy, 1),
                "restore_command": [
                    "tar",
                    "-xf",
                    str(transfer_destination),
                    "-C",
                    "<metadata-capable-parent>",
                ],
            },
        )
        verified = True
        verification_phase = "TAR_WRAPPED_VERIFIED"
        tar_sha256 = str(tar_receipt["sha256"])
    else:
        if transfer_destination.exists():
            return _block(f"direct destination exists; source untouched: {transfer_destination}")

        print(
            f"[{utcnow()}] single-pass copy+hash ({len(rels)} files) -> {dest}",
            file=sys.stderr,
            flush=True,
        )
        t_copy = time.time()
        try:
            src_digest = copy_and_hash(
                src,
                dest,
                rels,
                src_manifest,
                workers=args.workers,
                census=census,
            )
        except (CensusError, OSError) as exc:
            append_ledger(
                ledger,
                {
                    "phase": "BLOCKED_DIRECT_COPY",
                    "source": str(src),
                    "destination": str(dest),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return _block(f"direct copy failed; source untouched: {exc}")
        append_ledger(
            ledger,
            {
                "phase": "COPIED",
                "source": str(src),
                "destination": str(dest),
                "manifest_path": str(src_manifest),
                "source_manifest_sha256": src_digest,
                "source_metadata_manifest_path": str(src_metadata_manifest),
                "source_metadata_manifest_sha256": src_metadata_digest,
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
        dest_metadata_manifest = manifest_dir / "destination.metadata.jsonl"
        dest_digest = build_manifest(dest, dest_rels, dest_manifest, workers=args.workers)
        dest_metadata_digest = write_metadata_manifest(dest_census, dest_metadata_manifest)
        try:
            post_source_census = take_census(src)
            post_source_rels = [rel for rel, _size in post_source_census.files]
            post_source_manifest = manifest_dir / "source.post_copy.sha256"
            post_source_digest = build_manifest(
                src,
                post_source_rels,
                post_source_manifest,
                workers=args.workers,
            )
            post_source_metadata_manifest = manifest_dir / "source.post_copy.metadata.jsonl"
            post_source_metadata_digest = write_metadata_manifest(
                post_source_census, post_source_metadata_manifest
            )
        except (CensusError, OSError) as exc:
            append_ledger(
                ledger,
                {
                    "phase": "BLOCKED_POST_COPY_SOURCE_REVERIFY",
                    "source": str(src),
                    "destination": str(dest),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return _block(f"post-copy source reverify failed; source untouched: {exc}")
        content_equal = (
            post_source_rels == rels
            and src_digest == post_source_digest
            and dest_digest == post_source_digest
        )
        metadata_equal = (
            src_metadata_digest == post_source_metadata_digest
            and dest_metadata_digest == post_source_metadata_digest
        )
        verified = content_equal and metadata_equal
        verification_phase = "VERIFIED" if verified else "BLOCKED_FIDELITY_MISMATCH"
        append_ledger(
            ledger,
            {
                "phase": verification_phase,
                "source": str(src),
                "destination": str(dest),
                "source_manifest_sha256": src_digest,
                "post_copy_source_manifest_sha256": post_source_digest,
                "destination_manifest_sha256": dest_digest,
                "source_metadata_manifest_sha256": src_metadata_digest,
                "post_copy_source_metadata_manifest_sha256": post_source_metadata_digest,
                "destination_metadata_manifest_sha256": dest_metadata_digest,
                "content_manifest_equal": content_equal,
                "metadata_manifest_equal": metadata_equal,
                "source_stable_during_copy": (
                    src_digest == post_source_digest
                    and src_metadata_digest == post_source_metadata_digest
                ),
                "n_rows": len(rels),
                "logical_data_bytes": census.logical_bytes,
                "manifest_dir": str(manifest_dir),
            },
        )
        if not verified:
            return _block("content/mode/symlink fidelity differs; source untouched")
        tar_sha256 = ""

    print(
        f"[{utcnow()}] {verification_phase} ({len(rels)} content rows, {src_digest[:16]}...)",
        file=sys.stderr,
    )

    if not args.retire_source:
        print("source retained (no --retire-source); nothing freed", file=sys.stderr)
        return 0

    largest = max(census.files, key=lambda t: t[1])[0] if census.files else None
    tmp_old = src.parent / (src.name + ".RETIRING")
    if tmp_old.exists() or tmp_old.is_symlink():
        return _block(f"retirement recovery path already exists: {tmp_old}")
    src.rename(tmp_old)
    try:
        src.symlink_to(transfer_destination)
    except OSError as exc:
        tmp_old.rename(src)
        append_ledger(ledger, {"phase": "BLOCKED_SYMLINK", "source": str(src), "error": str(exc)})
        return _block(f"symlink install failed, source restored: {exc}")

    if transfer_mode == "tar_wrap":
        probe_ok = src.is_file() and sha256_file(src) == tar_sha256
    else:
        probe_ok = src.is_dir() and (largest is None or (src / largest).is_file())
    if not probe_ok:
        src.unlink()
        tmp_old.rename(src)
        append_ledger(ledger, {"phase": "BLOCKED_SYMLINK_PROBE", "source": str(src)})
        return _block("symlink did not resolve to a known file; source restored")

    shutil.rmtree(tmp_old)
    after = {
        "vertigo_df_after": df_kib(str(vertigo_root)),
        "dest_df_after": df_kib_for_path(transfer_destination),
    }
    append_ledger(
        ledger,
        {
            "phase": (
                "MOVED_TAR_WRAPPED_RESTORE_REQUIRED"
                if transfer_mode == "tar_wrap"
                else "MOVED_SYMLINKED"
            ),
            "source": str(src),
            "destination": str(transfer_destination),
            "symlink_installed": True,
            "symlink_probe_relpath": largest,
            "original_path_compatibility": (
                "archive_pointer_requires_restore"
                if transfer_mode == "tar_wrap"
                else "transparent_directory_symlink"
            ),
            "freed_allocated_kib": src_du,
            "manifest_sha256": src_digest,
            "metadata_manifest_sha256": src_metadata_digest,
            **after,
        },
    )
    print(
        f"[{utcnow()}] MOVED ({transfer_mode}) + symlinked; "
        f"freed ~{src_du/2**20:.2f} GiB",
        file=sys.stderr,
    )
    return 0


def _block(msg: str) -> int:
    print(f"BLOCK: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
