#!/usr/bin/env python3
"""Assemble a candidate runtime tree whose receiver pin is DERIVED from its archive.

WHY THIS EXISTS
---------------
A candidate runtime carries a fail-closed receiver pin -- ``inflate.py`` holds
``ARCHIVE_SHA256`` / ``ARCHIVE_BYTES`` and refuses to decode anything else.  That
guard is correct and load-bearing: it is what stops a tree from silently decoding
an archive it was not built for.

But every arm that produces a new archive has, until now, assembled its runtime by
copying a predecessor's tree and dropping the new archive in beside a pin that
still names the PREDECESSOR's bytes.  The guard then fires at t≈10 s, which is the
guard working -- and the fix people reach for is to hand-type the new digest into
``inflate.py``, which is exactly the thing that must never happen: a hand-typed
pin is a claim, not a measurement, and the seal stage's pin-consistency gate
(#1123) re-derives it and refuses a value it cannot reproduce.

So this module makes the honest move the only convenient one: the pin is
**COMPUTED from the archive file** and written, and then **RE-READ from the file it
just wrote and checked against the archive again**.  There is no argument through
which a caller can supply a digest.

CONTROLS (all fail-closed)
--------------------------
1. the pin lines must be FOUND in the base ``inflate.py`` -- a tree whose guard has
   moved or been renamed is refused rather than silently left unpinned;
2. exactly one substitution per constant -- a file with two ``ARCHIVE_SHA256``
   assignments is ambiguous and is refused;
3. after writing, ``inflate.py`` is re-parsed and its constants must equal the
   digest and size measured from the archive on disk;
4. the archive is copied into the tree and re-hashed AT ITS DESTINATION, so a
   partial copy cannot pass;
5. every other file is byte-copied and the output tree is content-hashed, so the
   manifest can prove nothing else moved.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

ARCHIVE_MEMBER = "archive.zip"
INFLATE = "inflate.py"
SHA_CONST = "ARCHIVE_SHA256"
BYTES_CONST = "ARCHIVE_BYTES"


class AssemblyError(RuntimeError):
    """Fail-closed error."""


def sha256_of(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def tree_sha256(root: Path) -> str:
    """Content hash over (relative path, bytes) for every file, sorted."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name.startswith("._"):
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        file_digest, _ = sha256_of(path)
        digest.update(file_digest.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def read_pin(inflate_path: Path) -> tuple[str | None, int | None]:
    """Re-read the pin from source via AST -- never by trusting what we wrote."""
    tree = ast.parse(inflate_path.read_text())
    sha: str | None = None
    size: int | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == SHA_CONST and isinstance(node.value, ast.Constant):
                sha = str(node.value.value)
            elif target.id == BYTES_CONST and isinstance(node.value, ast.Constant):
                size = int(node.value.value)
    return sha, size


def repin(inflate_path: Path, archive_sha: str, archive_bytes: int) -> dict[str, Any]:
    source = inflate_path.read_text()
    # ``[ \t]*`` and NOT ``\s*`` at the line end: under re.M, ``\s`` matches
    # newlines, so ``\s*$`` swallows the blank lines AFTER the constant and the
    # re-pin silently reformats the file. Measured: the first version of this tool
    # ate two blank lines, which ddm_fs3's own same-instrument comparator then
    # (correctly) refused as "the receiver changed". A re-pin must touch the pin
    # and nothing else.
    sha_pat = re.compile(rf'^{SHA_CONST}[ \t]*=[ \t]*"[0-9a-fA-F]*"[ \t]*$', re.M)
    bytes_pat = re.compile(rf"^{BYTES_CONST}[ \t]*=[ \t]*[0-9_]+[ \t]*$", re.M)

    n_sha = len(sha_pat.findall(source))
    n_bytes = len(bytes_pat.findall(source))
    if n_sha == 0 or n_bytes == 0:
        raise AssemblyError(
            f"{inflate_path} has no recognisable receiver pin "
            f"({SHA_CONST}={n_sha} sites, {BYTES_CONST}={n_bytes} sites). Refusing "
            "to assemble a tree whose guard this module cannot see -- an unpinned "
            "receiver is worse than a stale one."
        )
    if n_sha > 1 or n_bytes > 1:
        raise AssemblyError(
            f"{inflate_path} has {n_sha}/{n_bytes} pin sites; ambiguous, refusing"
        )

    before_sha, before_bytes = read_pin(inflate_path)
    source = sha_pat.sub(f'{SHA_CONST} = "{archive_sha}"', source)
    source = bytes_pat.sub(f"{BYTES_CONST} = {archive_bytes}", source)
    inflate_path.write_text(source)

    after_sha, after_bytes = read_pin(inflate_path)
    if after_sha != archive_sha or after_bytes != archive_bytes:
        raise AssemblyError(
            f"re-pin did not take: file now reads {after_sha}/{after_bytes}, "
            f"expected {archive_sha}/{archive_bytes}"
        )
    return {
        "inherited_pin_sha256": before_sha,
        "inherited_pin_bytes": before_bytes,
        "derived_pin_sha256": after_sha,
        "derived_pin_bytes": after_bytes,
        # BOTH legs, not just the sha. A pin can be HALF-updated: the sha leg
        # re-pinned to the new archive while ARCHIVE_BYTES keeps the inherited
        # value. The two constants then describe DIFFERENT archives, no archive
        # satisfies both, and inflate.py raises "unexpected size" -- which reads
        # as a payload defect rather than a packaging one. Measured 2026-08-24
        # on both ddm_dg2 candidate runtimes (sha 59428f07/31d99f0b correct,
        # ARCHIVE_BYTES both 180_368 = the dx2 pointer's size); the first
        # advisory fire died rc=1 at t=10s on exactly that guard. Reporting
        # staleness from the sha alone would have called that pin "unchanged".
        "pin_was_stale": before_sha != after_sha or before_bytes != after_bytes,
    }


def run(args: argparse.Namespace) -> int:
    base = Path(args.base_runtime)
    archive = Path(args.archive)
    out = Path(args.out)
    if not base.is_dir():
        raise AssemblyError(f"base runtime not a directory: {base}")
    if not archive.is_file():
        raise AssemblyError(f"archive not a file: {archive}")

    base_tree_sha = tree_sha256(base)
    if out.exists():
        if not args.force:
            raise AssemblyError(f"{out} exists; pass --force to replace")
        shutil.rmtree(out)
    shutil.copytree(base, out)

    # Copy the archive in, then hash it AT ITS DESTINATION.
    dest_archive = out / ARCHIVE_MEMBER
    shutil.copyfile(archive, dest_archive)
    archive_sha, archive_bytes = sha256_of(dest_archive)
    source_sha, source_bytes = sha256_of(archive)
    if (archive_sha, archive_bytes) != (source_sha, source_bytes):
        raise AssemblyError("archive copy does not match its source")
    if args.expect_archive_sha256 and archive_sha != args.expect_archive_sha256:
        # A mismatch guard only -- it can REFUSE, it can never SUPPLY the pin.
        raise AssemblyError(
            f"archive sha256 {archive_sha} != expected {args.expect_archive_sha256}"
        )

    inflate_path = out / INFLATE
    if not inflate_path.is_file():
        raise AssemblyError(f"no {INFLATE} in {out}")
    pin = repin(inflate_path, archive_sha, archive_bytes)

    out_tree_sha = tree_sha256(out)
    manifest = {
        "schema": "candidate_runtime_assembly.v1",
        "base_tree": str(base),
        "base_tree_sha256": base_tree_sha,
        "output_tree": str(out),
        "output_tree_sha256": out_tree_sha,
        "archive": {
            "source": str(archive),
            "bytes": archive_bytes,
            "sha256": archive_sha,
            "hashed_at_destination": True,
        },
        "receiver_pin": pin,
        "pin_provenance": (
            "DERIVED by hashing the archive at its destination inside the tree, "
            "written, then RE-READ from the written file via AST and re-checked. "
            "No caller-supplied digest path exists."
        ),
        "files": sorted(
            str(p.relative_to(out))
            for p in out.rglob("*")
            if p.is_file() and not p.name.startswith("._")
        ),
    }
    manifest_path = out.parent / f"{out.name}_assembly_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"assembled {out}")
    print(f"  base tree   {base_tree_sha[:16]}  ->  output tree {out_tree_sha[:16]}")
    print(f"  archive     {archive_bytes:,} B  sha {archive_sha[:16]}")
    if pin["pin_was_stale"]:
        # ``inherited_pin_bytes`` is None when the AST could not read the
        # constant (regex matched, ast.walk did not -- e.g. a nested scope), so
        # format it defensively: the sha leg one line up already does. A receipt
        # that CRASHES while reporting a repaired pin is the same class of
        # damage as one that under-reports it.
        inherited_bytes = pin["inherited_pin_bytes"]
        inherited_bytes_str = (
            f"{inherited_bytes:,}" if isinstance(inherited_bytes, int) else str(inherited_bytes)
        )
        print(
            f"  pin RE-DERIVED: {str(pin['inherited_pin_sha256'])[:16]} "
            f"({inherited_bytes_str} B, INHERITED/STALE) -> "
            f"{pin['derived_pin_sha256'][:16]} ({pin['derived_pin_bytes']:,} B)"
        )
    else:
        print("  pin already matched the archive; unchanged")
    print(f"  manifest    {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-runtime", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expect-archive-sha256",
        default=None,
        help="MISMATCH GUARD ONLY -- refuses on disagreement; never supplies the pin",
    )
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(func=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
