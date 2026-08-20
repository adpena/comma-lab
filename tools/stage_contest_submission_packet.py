# SPDX-License-Identifier: MIT
"""Stage a contest submission packet from a sealed candidate runtime tree.

WHY THIS IS A COMMITTED TOOL AND NOT A THROWAWAY SCRIPT
--------------------------------------------------------
Packet generations 2, 3 and 4 were each staged by a fresh ad-hoc script, and
generation 5 was to be the fourth. The generation-4 arm recorded its own defect
in the landing message: its census "filtered AppleDouble out of BOTH sides of
its comparison and reported 33 files 0 undeclared while 38 sidecars sat in the
tree -- a check that excludes a file class cannot certify it". That is three
repetitions of the same staging work and two contamination incidents
(generation 3 shipped ``.pyc``).
Per CLAUDE.md "SUBMISSION CHAIN = CANONICAL never probe scripts" and the
least-hand-typing law (>=3x repetition => canonical surface), the staging step
becomes one reviewed tool with the invariants wired in.

THE AUTHORITY MODEL
-------------------
The runtime manifest inside the exact-eval receipt is the ONLY authority for
what executable content a packet may contain. It is produced by the same
evaluation that produced the score, so a packet that matches it byte for byte
is provably the tree the score was measured on. Consequences wired in here:

1. Files are selected BY THE MANIFEST, never by globbing the source directory.
   A glob would sweep up ``__pycache__``, AppleDouble sidecars and stale
   receipts; a manifest-driven copy cannot, by construction.
2. Every copied file is re-hashed AFTER the copy. Content is identity, never
   filename.
3. ``runtime_tree_sha256`` is re-derived from those FRESHLY MEASURED digests --
   not from the manifest's own claimed digests -- and compared to the receipt's.
   That distinction is the entire proof. Deriving from the manifest rows would
   re-hash the tool's own input and could never fail: a tautology wearing a
   proof's name. Deriving from measured bytes makes the value a function of what
   is on disk, so content drift moves it and the comparison catches it. The
   per-file digest diffs are the diagnostic that explains a failure.
4. The source tree is censused for undeclared files and the result is REPORTED
   with its denominator -- including the file classes we exclude from the copy.
   A count that silently drops a class is the generation-4 defect.

Every failure is fail-closed: the output directory is removed and a non-zero
exit code is returned. A half-staged packet is worse than none, because the
next step in the chain would read it as complete.

Usage::

    .venv/bin/python tools/stage_contest_submission_packet.py \\
        --auth-eval-json <receipt.json> \\
        --source-runtime-dir <sealed candidate runtime tree> \\
        --out-dir <staged submission dir> \\
        --expected-archive-sha256 <sha> \\
        --expected-archive-size-bytes <n> \\
        --json-out <staging receipt>

The receipt may be a normal JSON file or a Python ``bytes`` repr (some harvest
paths write ``repr(payload)`` rather than the payload); both are accepted and
the decode is proved lossless before use.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# Non-runtime files a staged packet is allowed to carry beside the runtime
# manifest rows. Kept identical to tools/packet_census_guard.py so the two
# guards cannot disagree about what "declared" means.
DECLARED_NON_RUNTIME: frozenset[str] = frozenset(
    {
        "README.md",
        "report.txt",
        "archive.zip",
        "archive_manifest.json",
        "GENERATION_RECEIPT.json",
        "RECEIVER_PARSEBACK.json",
        "BORROWED_SUBSTRATE_ACCOUNTING.md",
        # Merge-eligibility and licensing surface. Every merged neural submission
        # in this contest ships a licence and a compression script; we vendor
        # modified third-party code, which makes the notices obligation heavier
        # for us, not lighter. None of these suffixes is in
        # ``_RUNTIME_DEPENDENCY_SUFFIXES`` except ``compress.py`` -- see
        # ``RUNTIME_TREE_SHA256_SCOPE`` below for why that is safe and what it
        # obliges a re-validator to do.
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "MANIFEST.sha256",
        "compress.py",
        "COMPRESS.md",
    }
)

# Why the pin is over ENUMERATED ROWS and not over a directory walk.
#
# ``experiments/contest_auth_eval.py::_runtime_root_file_manifest`` walks
# ``root.rglob("*")`` and keeps every file whose suffix is in
# ``_RUNTIME_DEPENDENCY_SUFFIXES`` (``.py``, ``.sh``, ``.txt``, ``.json``,
# ``.c``). The manifest is therefore NOT a closed allowlist: it is whatever
# matched on the evaluated host. Two consequences we must state rather than
# discover:
#
# 1. ``report.txt`` and ``archive_manifest.json`` are manifest-ELIGIBLE by
#    suffix yet absent from the pinned rows, because the evaluated
#    ``submission_dir`` never contained them. So a staged packet ALREADY
#    differs from a fresh walk of the evaluated tree, before we add anything.
# 2. Staging ``compress.py`` widens that difference by one more file.
#
# Neither touches the score: ``upstream/evaluate.py`` sizes ``archive.zip``
# only. But a re-validator who re-walks the directory will compute a different
# tree hash and read it as corruption. The pin is over the enumerated rows, and
# this string is written into every staging receipt so that rule travels with
# the packet instead of living in one arm's memo.
RUNTIME_TREE_SHA256_SCOPE: str = (
    "runtime_tree_sha256 is pinned over the ENUMERATED runtime-manifest rows "
    "recorded in this receipt, NOT over a fresh recursive walk of the staged "
    "directory. Re-validation MUST hash the enumerated rows. The staged packet "
    "intentionally carries additional non-runtime documents (licence, notices, "
    "per-file manifest, compression script and its README) that a fresh walk "
    "would pick up by suffix; they are declared here, they carry no runtime "
    "role, and they cannot change the score because evaluate.py sizes "
    "archive.zip only."
)

# File classes that must never enter a staged packet. These are reported with
# exact paths rather than silently skipped: a staging run that removes a class
# without naming it is the check-that-excludes-a-class defect.
EXCLUDED_CLASSES: dict[str, str] = {
    "applesingle_sidecar": "AppleDouble ``._`` sidecar written by macOS on ExFAT volumes",
    "python_bytecode": "compiled ``.pyc`` -- embeds absolute local build paths",
    "pycache_dir": "``__pycache__`` directory content",
}


class StagingError(RuntimeError):
    """Raised when the staged tree cannot be proved identical to the evaluated tree."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Hash a payload the way the auth-eval runtime manifest hashes its tree.

    Mirrors ``tac.deploy.modal.auth_eval._canonical_json_sha256``. Kept as a
    local mirror rather than an import so this tool can stage a packet from a
    receipt produced by an older snapshot of that module; the value is checked
    against the receipt, so a drift surfaces as a refusal, never as a silent pass.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_possibly_repr_json(path: Path) -> dict[str, Any]:
    """Load JSON that may have been persisted as a Python ``bytes`` repr.

    Some harvest paths write ``repr(payload_bytes)`` to disk, so the file starts
    with ``b'`` and carries ``\\n`` as two literal characters. That encoding is
    lossless, so it is decoded rather than rejected -- but the decode is PROVED
    to round-trip before the result is trusted.
    """
    raw = path.read_bytes()
    if raw[:2] in (b"b'", b'b"'):
        text = raw.decode("utf-8")
        decoded = ast.literal_eval(text)
        if not isinstance(decoded, bytes):
            raise StagingError(f"{path}: bytes-repr decoded to {type(decoded).__name__}, not bytes")
        if repr(decoded).encode("utf-8") != raw.rstrip(b"\n"):
            raise StagingError(f"{path}: bytes-repr decode is not round-trip exact; refusing")
        raw = decoded
    return json.loads(raw.decode("utf-8"))


def extract_runtime_manifest(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the runtime dependency manifest from an exact-eval receipt."""
    manifest = receipt.get("provenance", {}).get("inflate_runtime_manifest")
    if not isinstance(manifest, Mapping):
        manifest = receipt.get("inflate_runtime_manifest")
    if not isinstance(manifest, Mapping):
        raise StagingError(
            "receipt carries no provenance.inflate_runtime_manifest; "
            "this tool cannot stage a packet without the evaluated file list"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise StagingError("runtime manifest carries no files[]; refusing to stage an empty tree")
    return dict(manifest)


def rederive_tree_sha256(manifest: Mapping[str, Any], rows: list[dict[str, Any]]) -> str:
    """Derive the runtime tree hash from ``rows``.

    Callers MUST pass rows carrying the FRESHLY MEASURED sha/bytes of the staged
    copies. Passing the manifest's own rows makes this a tautology: it would
    re-hash its own input and could never fail, which is exactly the defect this
    docstring exists to stop being re-introduced.
    """
    repo_local_tac = dict(manifest.get("repo_local_tac_import_manifest") or {})
    tree_payload = {
        "runtime_root_name": repo_local_tac.get("runtime_root_name", "submission_dir"),
        "files": rows,
        "external_dependency_roots": manifest.get("external_dependency_roots", []),
        "repo_local_tac_import_manifest": repo_local_tac,
        "upstream_evaluate_py": manifest.get("upstream_evaluate_py"),
    }
    return canonical_json_sha256(tree_payload)


def classify_excluded(relpath: str, name: str) -> str | None:
    if name.startswith("._"):
        return "applesingle_sidecar"
    if name.endswith(".pyc"):
        return "python_bytecode"
    if "__pycache__" in Path(relpath).parts:
        return "pycache_dir"
    return None


def census_source(source: Path, declared: Iterable[str]) -> dict[str, Any]:
    """Census the source tree, reporting excluded classes with their denominator."""
    declared_set = set(declared)
    real: list[str] = []
    excluded: dict[str, list[str]] = {key: [] for key in EXCLUDED_CLASSES}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(source))
        cls = classify_excluded(rel, path.name)
        if cls is not None:
            excluded[cls].append(rel)
        else:
            real.append(rel)
    undeclared = sorted(set(real) - declared_set)
    return {
        "source_real_file_count": len(real),
        "declared_count": len(declared_set),
        "undeclared_real_files": undeclared,
        "undeclared_real_file_count": len(undeclared),
        "excluded_by_class": {k: sorted(v) for k, v in excluded.items()},
        "excluded_total": sum(len(v) for v in excluded.values()),
        "excluded_class_descriptions": dict(EXCLUDED_CLASSES),
    }


def parse_doc_spec(spec: str) -> tuple[Path, str]:
    """Parse a ``SRC=DESTREL`` document staging spec.

    Split on the LAST ``=`` so a source path containing one still parses. The
    destination is validated against ``DECLARED_NON_RUNTIME`` by the caller;
    this function only splits and rejects the empty halves.
    """
    if "=" not in spec:
        raise StagingError(
            f"--doc expects SRC=DESTREL (destination relative to the packet root); got {spec!r}"
        )
    src_text, dest = spec.rsplit("=", 1)
    src_text, dest = src_text.strip(), dest.strip()
    if not src_text or not dest:
        raise StagingError(f"--doc has an empty source or destination: {spec!r}")
    if Path(dest).is_absolute() or ".." in Path(dest).parts:
        raise StagingError(f"--doc destination must be a relative path inside the packet: {dest!r}")
    return Path(src_text), dest


def stage(
    *,
    auth_eval_json: Path,
    source_runtime_dir: Path,
    out_dir: Path,
    archive_name: str = "archive.zip",
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    docs: Iterable[tuple[Path, str]] = (),
) -> dict[str, Any]:
    """Stage the packet and return a receipt proving the tree identity.

    ``docs`` are non-runtime packet documents (``SRC``, ``DESTREL``) pairs. They
    are copied through this tool rather than by hand for the same reason the
    runtime rows are: a packet assembled by three different mechanisms is a
    packet whose contents nobody can prove. Every destination must already be
    declared in ``DECLARED_NON_RUNTIME`` -- otherwise the census guard, which
    shares that constant, would report the file as undeclared. That coupling is
    enforced here rather than documented, so the two tools cannot drift.
    """
    receipt = load_possibly_repr_json(auth_eval_json)
    manifest = extract_runtime_manifest(receipt)
    rows = [dict(row) for row in manifest["files"]]
    declared_rel = [str(row["relative_path"]) for row in rows]

    if not source_runtime_dir.is_dir():
        raise StagingError(f"source runtime dir does not exist: {source_runtime_dir}")

    # Validate every doc BEFORE the output directory is created, so a bad spec
    # costs nothing and cannot leave a half-staged tree behind.
    doc_pairs = [(Path(src), str(dest)) for src, dest in docs]
    seen_dests: set[str] = set()
    for src, dest in doc_pairs:
        if dest not in DECLARED_NON_RUNTIME:
            raise StagingError(
                f"--doc destination {dest!r} is not in DECLARED_NON_RUNTIME. "
                "Add it to that constant in BOTH this tool and "
                "tools/packet_census_guard.py, or the census will report the "
                "staged file as undeclared."
            )
        if dest in declared_rel:
            raise StagingError(
                f"--doc destination {dest!r} collides with a runtime manifest row; "
                "refusing to overwrite proven runtime content with a document"
            )
        if dest in seen_dests:
            raise StagingError(f"--doc destination {dest!r} given twice")
        seen_dests.add(dest)
        if not src.is_file():
            raise StagingError(f"--doc source does not exist: {src}")

    # The census's "declared" set is the runtime manifest UNION the non-runtime
    # allowlist, so a source tree that already carries packet docs (README.md,
    # report.txt, ...) is not reported as contaminated. Using the shared constant
    # here rather than only documenting it is what keeps this tool and
    # packet_census_guard.py from disagreeing about what "declared" means.
    census = census_source(source_runtime_dir, set(declared_rel) | {archive_name} | set(DECLARED_NON_RUNTIME))

    if out_dir.exists():
        raise StagingError(
            f"output dir already exists: {out_dir}. "
            "Staging never overwrites: remove it deliberately or pick a new generation dir."
        )

    # ``staged_rows`` carries the FRESHLY MEASURED sha/bytes of each staged copy,
    # never the manifest's own claim about them. That distinction is the whole
    # proof: deriving the tree hash from the manifest's rows would re-hash the
    # input and could not fail, which is a tautology wearing a proof's name.
    # Substituting measured values makes the derived hash a function of the bytes
    # actually on disk, so content drift moves it.
    staged_rows: list[dict[str, Any]] = []
    measured_diffs: list[str] = []
    verified = 0
    try:
        out_dir.mkdir(parents=True)
        for row in rows:
            rel = str(row["relative_path"])
            src = source_runtime_dir / rel
            if not src.is_file():
                raise StagingError(f"manifest declares {rel} but it is absent from the source tree")
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            # Measure the COPY, not the source: this is what ties the derived tree
            # hash to the byte sequence a judge would actually run.
            payload = dst.read_bytes()
            measured_row = dict(row)
            measured_row["sha256"] = sha256_bytes(payload)
            measured_row["bytes"] = len(payload)
            if measured_row["sha256"] != row["sha256"] or measured_row["bytes"] != row["bytes"]:
                measured_diffs.append(
                    f"{rel}: sha {measured_row['sha256']} vs {row['sha256']}, "
                    f"{measured_row['bytes']} B vs {row['bytes']} B"
                )
            else:
                verified += 1
            staged_rows.append(measured_row)

        # The archive is not a runtime manifest row; it is the scored payload.
        archive_src = source_runtime_dir / archive_name
        if not archive_src.is_file():
            raise StagingError(f"archive missing from source tree: {archive_src}")
        archive_dst = out_dir / archive_name
        shutil.copy2(archive_src, archive_dst)
        archive_payload = archive_dst.read_bytes()
        archive_sha = sha256_bytes(archive_payload)
        if expected_archive_sha256 and archive_sha != expected_archive_sha256:
            raise StagingError(f"archive sha mismatch: staged {archive_sha} vs expected {expected_archive_sha256}")
        if expected_archive_size_bytes and len(archive_payload) != expected_archive_size_bytes:
            raise StagingError(
                f"archive size mismatch: staged {len(archive_payload)} B vs expected {expected_archive_size_bytes} B"
            )

        # THE identity proof: derive the tree hash from the MEASURED rows above.
        # Because those rows carry the digests of the bytes on disk, any content
        # drift changes this value, so the comparison can genuinely fail. The
        # per-file diffs are the DIAGNOSTIC that explains a failure, not the proof.
        rederived = rederive_tree_sha256(manifest, staged_rows)
        declared_tree = str(manifest.get("runtime_tree_sha256"))
        if rederived != declared_tree:
            detail = "; ".join(measured_diffs) if measured_diffs else "no per-file digest differs"
            raise StagingError(
                "runtime_tree_sha256 derived from the MEASURED staged bytes does not "
                f"match the receipt ({rederived} vs {declared_tree}); the staged tree "
                f"is NOT the evaluated tree. Per-file diffs: {detail}"
            )
        if measured_diffs:
            # Unreachable through the manifest's own derivation, kept because a
            # future change to the tree payload could decouple the two.
            raise StagingError(
                "staged content differs from the evaluated manifest while the tree "
                f"hash still matched: {'; '.join(measured_diffs)}"
            )

        # Documents are copied AFTER the tree identity is proved, so a document
        # can never be implicated in a failed identity proof.
        staged_docs: list[dict[str, Any]] = []
        for src, dest in doc_pairs:
            dst = out_dir / dest
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            payload = dst.read_bytes()
            src_payload = src.read_bytes()
            if payload != src_payload:
                raise StagingError(f"{dest}: staged document differs from its source after copy")
            staged_docs.append(
                {
                    "relative_path": dest,
                    "source": str(src),
                    "sha256": sha256_bytes(payload),
                    "bytes": len(payload),
                }
            )
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    return {
        "schema": "pact.contest_packet_staging.v1",
        "verdict": "STAGED_TREE_PROVED_IDENTICAL_TO_EVALUATED_TREE",
        "auth_eval_json": str(auth_eval_json),
        "source_runtime_dir": str(source_runtime_dir),
        "out_dir": str(out_dir),
        "runtime_files_verified": verified,
        "runtime_files_declared": len(rows),
        "runtime_tree_sha256": declared_tree,
        "runtime_tree_sha256_rederived_from_measured_staged_bytes": rederived,
        "runtime_tree_sha256_rederivation_input": (
            "freshly measured sha256/bytes of every staged copy, NOT the manifest's "
            "own claimed digests; deriving from the manifest rows would be a tautology"
        ),
        "runtime_tree_sha256_scope": RUNTIME_TREE_SHA256_SCOPE,
        "runtime_content_tree_sha256": manifest.get("runtime_content_tree_sha256"),
        "staged_documents": staged_docs,
        "staged_document_count": len(staged_docs),
        "archive": {
            "name": archive_name,
            "sha256": archive_sha,
            "bytes": len(archive_payload),
        },
        "source_census": census,
        "excluded_from_staging_note": (
            "Excluded classes are reported with exact paths above. They are absent "
            "from the staged tree by construction: files are selected by the runtime "
            "manifest, never by globbing the source directory."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--auth-eval-json",
        type=Path,
        required=True,
        help="exact-eval receipt carrying provenance.inflate_runtime_manifest (the authority)",
    )
    parser.add_argument("--source-runtime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--archive-name", default="archive.zip")
    parser.add_argument("--expected-archive-sha256", default=None)
    parser.add_argument("--expected-archive-size-bytes", type=int, default=None)
    parser.add_argument(
        "--doc",
        action="append",
        default=None,
        metavar="SRC=DESTREL",
        help=(
            "stage a non-runtime packet document, e.g. "
            "--doc README_PUBLIC.md=README.md . Repeatable. The destination must "
            "already appear in DECLARED_NON_RUNTIME (shared with "
            "tools/packet_census_guard.py) or staging refuses."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        docs = [parse_doc_spec(spec) for spec in (args.doc or [])]
        record = stage(
            auth_eval_json=args.auth_eval_json,
            source_runtime_dir=args.source_runtime_dir,
            out_dir=args.out_dir,
            archive_name=args.archive_name,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_archive_size_bytes=args.expected_archive_size_bytes,
            docs=docs,
        )
    except StagingError as exc:
        print(f"STAGING REFUSED: {exc}", file=sys.stderr)
        return 1

    census = record["source_census"]
    print(f"{record['verdict']}")
    print(f"  runtime files verified : {record['runtime_files_verified']}/{record['runtime_files_declared']}")
    print(f"  runtime_tree_sha256    : {record['runtime_tree_sha256']}")
    print(f"  re-derived (measured)  : {record['runtime_tree_sha256_rederived_from_measured_staged_bytes']}")
    print(f"  archive                : {record['archive']['sha256']} ({record['archive']['bytes']} B)")
    for doc in record["staged_documents"]:
        print(f"  document               : {doc['relative_path']} ({doc['bytes']} B, {doc['sha256'][:12]})")
    print(
        f"  source census          : {census['source_real_file_count']} real files, "
        f"{census['undeclared_real_file_count']} undeclared, "
        f"{census['excluded_total']} excluded by class"
    )
    for cls, paths in census["excluded_by_class"].items():
        if paths:
            print(f"    excluded[{cls}] = {len(paths)}: {paths[0]}{' ...' if len(paths) > 1 else ''}")
    for rel in census["undeclared_real_files"]:
        print(f"    UNDECLARED (not staged): {rel}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"  receipt                : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
