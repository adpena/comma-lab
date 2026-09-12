#!/usr/bin/env python3
"""Build a candidate runtime tree from a staged body plus a closed archive.

``ddm_sj1_joint_admission.cmd_close`` writes ``candidate_archive.zip``; the parse-back, the
public smoke and the seal all want a TREE.  Pass 7 assembled that tree by hand and shipped a
listing that named the POINTER's ``inflate.py`` against its own re-pinned bytes -- one
mismatch, caught by the seal rather than by inspection, and its memo flagged it as a CLASS
("any arm that calls ``patch_inflate_pins`` and then seals will hit it").  This module is
that class closed: it copies the pointer tree, installs the archive, re-pins the receiver's
two archive constants, REWRITES every ``MANIFEST.sha256`` row from the bytes now on disk, and
then re-hashes the whole listing from OUTSIDE the tree and refuses on a single mismatch.

The tree it emits must differ from the pointer in exactly
``['MANIFEST.sha256', 'archive.zip', 'inflate.py']`` -- checked, not asserted.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

EXPECTED_DIFFERING = ["MANIFEST.sha256", "archive.zip", "inflate.py"]


def is_tree_file(path: Path) -> bool:
    """Exclude bytecode caches and macOS AppleDouble stubs from every tree comparison.

    APDataStore is ExFAT, where ``shutil.copytree``'s metadata copy makes a ``._<name>``
    sidecar next to every file that carries an extended attribute.  They are not part of the
    receiver and no digest counts them, so a census that included them would report 56
    phantom "extra" files ([[landing_an_arm_bundle_guard_on_file_count_and_skip_exfat_dot_underscore_stubs_20260910]]).
    The build avoids creating them at all by copying bytes without metadata; this predicate
    is the belt to that braces.
    """
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and not path.name.startswith("._")
    )


class Pd1TreeError(RuntimeError):
    """The candidate tree is not the pointer body plus this archive."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_manifest(tree: Path) -> dict[str, Any]:
    """Rewrite MANIFEST.sha256 from the bytes on disk, preserving its exact line shape."""
    listing = tree / "MANIFEST.sha256"
    if not listing.exists():
        raise Pd1TreeError(f"{listing} does not exist; this is not a receiver tree")
    original = listing.read_text(encoding="utf-8")
    lines_out: list[str] = []
    changed: list[str] = []
    for line in original.splitlines():
        if not line.strip():
            lines_out.append(line)
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise Pd1TreeError(f"MANIFEST.sha256 row is not '<sha>  <path>': {line!r}")
        old_sha, rest = parts[0], parts[1]
        relative = rest.strip().lstrip("*")
        target = tree / relative
        if not target.is_file():
            raise Pd1TreeError(f"MANIFEST.sha256 names {relative}, which is not a file")
        new_sha = sha256_file(target)
        if new_sha != old_sha:
            changed.append(relative)
        lines_out.append(line.replace(old_sha, new_sha, 1))
    text = "\n".join(lines_out)
    if original.endswith("\n"):
        text += "\n"
    listing.write_text(text, encoding="utf-8")
    return {"rows": len(lines_out), "rows_changed": changed}


def validate_manifest(tree: Path) -> dict[str, Any]:
    """Re-hash every row from OUTSIDE the listing and refuse on any mismatch."""
    problems: list[str] = []
    rows = 0
    for line in (tree / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        sha, rest = line.split(None, 1)
        relative = rest.strip().lstrip("*")
        observed = sha256_file(tree / relative)
        if observed != sha:
            problems.append(f"{relative}: listing {sha[:16]} vs bytes {observed[:16]}")
    if problems:
        raise Pd1TreeError(f"MANIFEST.sha256 does not describe the tree: {problems}")
    return {"rows": rows, "problems": 0}


def cmd_build(args) -> int:
    import ddm_sj1_joint_admission as joint

    pointer = Path(args.pointer_runtime)
    archive = Path(args.archive)
    if not archive.is_file():
        raise Pd1TreeError(f"{archive} is not a file")
    out = Path(args.out_dir) / "candidate_runtime"
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pointer, out, copy_function=shutil.copyfile)
    for cache in out.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    for stub in out.rglob("._*"):
        stub.unlink(missing_ok=True)
    shutil.copyfile(archive, out / "archive.zip")
    facts = {"sha256": sha256_file(out / "archive.zip"), "bytes": (out / "archive.zip").stat().st_size}
    pins = joint.patch_inflate_pins(out, facts["sha256"], facts["bytes"])
    manifest = rewrite_manifest(out)
    validation = validate_manifest(out)

    differing = sorted(
        str(f.relative_to(out))
        for f in out.rglob("*")
        if is_tree_file(f)
        and (pointer / f.relative_to(out)).is_file()
        and f.read_bytes() != (pointer / f.relative_to(out)).read_bytes()
    )
    # IDENTITY CONTROL: handed the pointer's OWN archive this builder must reproduce the
    # pointer tree byte for byte -- which also proves the shipped tree's MANIFEST.sha256
    # already describes its own bytes.  Without that control a builder that quietly
    # rewrites a row nobody checks would look identical to one that does not.
    if args.expect_identity:
        if differing:
            raise Pd1TreeError(
                f"IDENTITY CONTROL FAILED: rebuilding the pointer from its own archive "
                f"changed {differing}"
            )
    elif differing != EXPECTED_DIFFERING:
        raise Pd1TreeError(
            f"candidate tree differs from the pointer in {differing}, expected "
            f"{EXPECTED_DIFFERING}"
        )
    extra = sorted(
        str(f.relative_to(out)) for f in out.rglob("*")
        if is_tree_file(f) and not (pointer / f.relative_to(out)).exists()
    )
    missing = sorted(
        str(f.relative_to(pointer)) for f in pointer.rglob("*")
        if is_tree_file(f) and not (out / f.relative_to(pointer)).exists()
    )
    if extra or missing:
        raise Pd1TreeError(f"candidate tree has extra {extra} / missing {missing} files")

    report = {
        "schema": "ddm_pd1_candidate_tree.v1",
        "pointer_runtime": str(pointer),
        "candidate_runtime": str(out),
        "archive_source": str(archive),
        "archive": facts,
        "inflate_pins": pins,
        "manifest_rewrite": manifest,
        "manifest_validation": validation,
        "files_differing_from_pointer": differing,
        "identity_control": bool(args.expect_identity),
        "axis": "[scorer-free EXACT byte custody]",
        "score_claim": False,
    }
    (Path(args.out_dir) / "CANDIDATE_TREE.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    build = sub.add_parser("build", help="pointer tree + closed archive -> candidate tree")
    build.add_argument("--pointer-runtime", type=Path, required=True)
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--expect-identity", action="store_true")
    build.set_defaults(func=cmd_build)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
