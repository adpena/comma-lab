#!/usr/bin/env python3
"""Register a pre-registration (prereg / falsifier) so its BIRTH STATE lives in git.

WHY THIS EXISTS
---------------
A prereg's entire value is immutability.  On 2026-08-20 the fs3 arm's
``FS3_DROP_FALSIFIER.json`` was rewritten in place 25 minutes after it was born
(rv17 wave-3 finding W3-F14): the file still claims
``"registered_before_any_build": true`` while citing a receipt created 24m47s
AFTER its own birthtime.  The added material was a good-faith recording of
wave-3 dispositions taken through the wrong mechanism -- but the effect is that
its threshold, ``row_dies_if_token_stream_shrinks_by_less_than_bytes``, is now
**unauditable**: nobody can show that it moved, and nobody can show that it did
not.  That is precisely the question append-only registration exists to answer.

A receipt written beside the file cures that ONE instance.  This tool cures the
CLASS: registration copies the prereg into ``.omx/research/preregs/<name>.json``
and commits it through the canonical serializer **in the same call**, so the
birth bytes are in git history from that moment.  Any later in-place rewrite of
the live file is then diffable forever, and ``verify`` reports it mechanically.

The guarantee is deliberately narrow and worth stating plainly: this proves what
the file said **at registration time**.  It cannot prove that registration
happened before the build -- only that the content has not changed since.  A
prereg registered late is still a late prereg; this tool makes that visible
instead of arguable.

WRITE-ONCE
----------
``register`` refuses if a birth copy already exists in the working tree OR in
git history.  There is no ``--force``.  Superseding is done by registering a new
name (``<name>.v2``) that declares ``supersedes`` -- never by overwriting.

SUBCOMMANDS
-----------
``register``  copy a prereg to the birth-copy path and commit it (write-once).
``verify``    compare the live prereg against its COMMITTED birth copy.
``census``    list prereg-like files that have no committed birth copy.

Sister of the append-only receipt discipline and of
``tools/subagent_commit_serializer.py`` (which this tool calls rather than
re-implementing).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_DIR_REL = Path(".omx/research/preregs")
SERIALIZER_REL = Path("tools/subagent_commit_serializer.py")

#: Filename tokens that mark a file as a pre-registration for census purposes.
PREREG_NAME_TOKENS = ("FALSIFIER", "PREREG", "PREREGISTERED", "PREDICTION")

#: Roots the census walks when ``--root`` is not given.  Missing roots are
#: skipped silently -- an unplugged SSD is not a finding.
DEFAULT_CENSUS_ROOTS = (
    Path("/Volumes/APDataStore/pact"),
    Path("/Volumes/VertigoDataTier/pact"),
    REPO_ROOT / ".omx/research",
)

INTACT = "INTACT"
MUTATED = "MUTATED"
MISSING_LIVE = "MISSING_LIVE"
UNREGISTERED = "UNREGISTERED"


class PreregError(RuntimeError):
    """Refusal raised by this tool.  Never a silent skip."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, root: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(root or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def blob_exists_in_history(rel: Path, root: Path | None = None) -> bool:
    """True if ``rel`` exists at HEAD.

    Checked separately from the working tree: a birth copy that was committed
    and later deleted from disk is STILL registered, and re-registering it would
    silently mint a second, conflicting birth state.
    """
    probe = _git("cat-file", "-e", f"HEAD:{rel.as_posix()}", root=root)
    return probe.returncode == 0


def read_committed_blob(rel: Path, root: Path | None = None) -> bytes | None:
    """The bytes of ``rel`` as committed at HEAD, or None if absent."""
    probe = subprocess.run(
        ["git", "show", f"HEAD:{rel.as_posix()}"],
        cwd=str(root or REPO_ROOT),
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    return probe.stdout


def birth_copy_rel(name: str) -> Path:
    return PREREG_DIR_REL / f"{name}.json"


def provenance_rel(name: str) -> Path:
    return PREREG_DIR_REL / f"{name}.provenance.json"


def _validate_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise PreregError(
            f"REFUSING: invalid prereg name {name!r}; must be a bare filename "
            "stem with no path separators"
        )
    return name


# --------------------------------------------------------------------------
# register
# --------------------------------------------------------------------------


def register(
    source: Path,
    name: str | None,
    supersedes: str | None,
    message: str | None,
    dry_run: bool = False,
) -> int:
    source = source.resolve()
    if not source.is_file():
        raise PreregError(f"REFUSING: source prereg not found: {source}")

    name = _validate_name(name or source.stem)
    birth_rel, prov_rel = birth_copy_rel(name), provenance_rel(name)
    birth_abs, prov_abs = REPO_ROOT / birth_rel, REPO_ROOT / prov_rel

    # WRITE-ONCE, both surfaces.  No --force exists on purpose.
    for rel, abs_ in ((birth_rel, birth_abs), (prov_rel, prov_abs)):
        if abs_.exists():
            raise PreregError(
                f"REFUSING: birth copy already exists in the working tree: {rel}. "
                "Preregs are write-once. Register a new name (e.g. "
                f"'{name}.v2') declaring --supersedes {name}; never overwrite."
            )
        if blob_exists_in_history(rel):
            raise PreregError(
                f"REFUSING: birth copy already exists in git history at HEAD:{rel} "
                "(even though it is absent from the working tree). Preregs are "
                "write-once. Register a new name declaring --supersedes."
            )

    payload = source.read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()

    st = source.stat()
    birthtime = getattr(st, "st_birthtime", None)
    provenance = {
        "schema": "pact_prereg_registration.v1",
        "name": name,
        "birth_copy": birth_rel.as_posix(),
        "source_path_at_registration": str(source),
        "source_sha256": source_sha,
        "source_size_bytes": st.st_size,
        "source_birthtime_utc": (
            _dt.datetime.fromtimestamp(int(birthtime), _dt.UTC).isoformat()
            if birthtime is not None
            else None
        ),
        "source_mtime_utc": _dt.datetime.fromtimestamp(
            int(st.st_mtime), _dt.UTC
        ).isoformat(),
        "registered_at_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "supersedes": supersedes,
        "what_this_proves": (
            "the byte-exact content of the prereg AT REGISTRATION TIME, committed "
            "to git history. Any later in-place edit of the live file is diffable "
            "against this copy via `register_prereg.py verify`."
        ),
        "what_this_does_NOT_prove": (
            "that registration happened before the build. A prereg registered late "
            "is still a late prereg; this records WHEN it was registered so that "
            "question is answerable rather than arguable."
        ),
        "already_mutated_at_registration": (
            birthtime is not None and int(st.st_mtime) > int(birthtime)
        ),
    }

    if dry_run:
        print(f"[dry-run] would write {birth_rel} ({len(payload)} B, {source_sha[:12]})")
        print(f"[dry-run] would write {prov_rel}")
        return 0

    birth_abs.parent.mkdir(parents=True, exist_ok=True)
    birth_abs.write_bytes(payload)
    prov_abs.write_text(json.dumps(provenance, indent=2) + "\n")

    # Post-edit working-tree shas, per the serializer's --expected-content-sha256
    # contract: declare what the content SHOULD be at lock-acquire time.
    birth_sha = sha256_file(birth_abs)
    prov_sha = sha256_file(prov_abs)
    if birth_sha != source_sha:
        raise PreregError(
            f"REFUSING: birth copy sha {birth_sha} != source sha {source_sha}; "
            "the copy is not byte-identical and would not be a valid birth state"
        )

    msg = message or (
        f"prereg register {name}: birth copy committed so later in-place edits "
        f"are diffable (src sha {source_sha[:12]}) [no-triality] [p0-ledger-ok]"
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / SERIALIZER_REL),
        "--message",
        msg,
        "--no-co-author",
        "--files",
        birth_rel.as_posix(),
        prov_rel.as_posix(),
        "--expected-content-sha256",
        f"{birth_rel.as_posix()}={birth_sha}",
        "--expected-content-sha256",
        f"{prov_rel.as_posix()}={prov_sha}",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise PreregError(
            f"REFUSING: serializer commit failed (rc={proc.returncode}). The birth "
            f"copy is on disk at {birth_rel} but NOT in git history, so it is not "
            "yet a registration. Resolve the commit failure and re-run; the "
            "write-once guard will refuse until the stale files are removed."
        )

    print(f"registered {name}")
    print(f"  birth copy : {birth_rel}  sha256 {birth_sha}")
    print(f"  provenance : {prov_rel}")
    print(f"  source     : {source}")
    if provenance["already_mutated_at_registration"]:
        print(
            "  NOTE: mtime > birthtime at registration -- this file was ALREADY "
            "edited before it was registered. The birth copy records its state "
            "as of NOW, not as of its creation."
        )
    return 0


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def _json_key_diff(a: bytes, b: bytes) -> dict[str, list[str]] | None:
    """Top-level key delta between two JSON blobs, when both parse."""
    try:
        da, db = json.loads(a), json.loads(b)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(da, dict) or not isinstance(db, dict):
        return None
    # JSON object keys are always strings; coerce so the sort is well-typed.
    keys_a: set[str] = {str(k) for k in da}
    keys_b: set[str] = {str(k) for k in db}
    return {
        "keys_added": sorted(keys_b - keys_a),
        "keys_removed": sorted(keys_a - keys_b),
        "keys_changed": sorted(k for k in keys_a & keys_b if da[k] != db[k]),
    }


def verify(name: str, live: Path | None, as_json: bool = False) -> int:
    name = _validate_name(name)
    birth_rel, prov_rel = birth_copy_rel(name), provenance_rel(name)

    committed = read_committed_blob(birth_rel)
    if committed is None:
        raise PreregError(
            f"REFUSING: no committed birth copy at HEAD:{birth_rel}. This prereg "
            "was never registered, so there is nothing to verify against. Run "
            "`register_prereg.py register --source <path>` first."
        )

    if live is None:
        prov_blob = read_committed_blob(prov_rel)
        if prov_blob is None:
            raise PreregError(
                f"REFUSING: no committed provenance at HEAD:{prov_rel} and no "
                "--live given, so the live prereg path is unknown."
            )
        live = Path(json.loads(prov_blob)["source_path_at_registration"])

    birth_sha = hashlib.sha256(committed).hexdigest()
    if not live.is_file():
        missing: dict[str, object] = {
            "name": name,
            "verdict": MISSING_LIVE,
            "live_path": str(live),
            "birth_sha256": birth_sha,
            "note": "the registered live file is gone; the birth copy in git is "
            "now the only surviving record of its content",
        }
        _emit(missing, as_json)
        return 1

    live_bytes = live.read_bytes()
    live_sha = hashlib.sha256(live_bytes).hexdigest()
    intact = live_sha == birth_sha
    result: dict[str, object] = {
        "name": name,
        "verdict": INTACT if intact else MUTATED,
        "live_path": str(live),
        "birth_copy": birth_rel.as_posix(),
        "birth_sha256": birth_sha,
        "live_sha256": live_sha,
    }
    if not intact:
        result["size_delta_bytes"] = len(live_bytes) - len(committed)
        diff = _json_key_diff(committed, live_bytes)
        if diff is not None:
            result["json_key_diff"] = diff
        result["note"] = (
            "the live prereg differs from its committed birth state. Whatever it "
            "now claims about itself, its content at registration is the blob in "
            "git -- diff them to see exactly what moved."
        )
    _emit(result, as_json)
    return 0 if intact else 1


def _emit(result: Mapping[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f"{result['verdict']}  {result['name']}")
    for key, value in result.items():
        if key in ("verdict", "name"):
            continue
        print(f"  {key}: {value}")


# --------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------


def _looks_like_prereg(path: Path) -> bool:
    upper = path.name.upper()
    return path.suffix == ".json" and any(t in upper for t in PREREG_NAME_TOKENS)


def _walk_bounded(root: Path, max_depth: int) -> Iterator[Path]:
    """Yield files under ``root`` no deeper than ``max_depth``, PRUNING as we go.

    ``Path.rglob`` descends the whole tree and filters afterwards, which on the
    SSD artifact tiers means walking millions of retained payload files to find a
    handful of preregs.  Pruning at the directory level is what makes the census
    cheap enough to run routinely -- and a census nobody runs is not a guard.

    Depth is counted on the FILE, not on its directory: a file sitting directly
    in ``root`` is depth 1.  So a directory at depth ``max_depth`` holds only
    files at ``max_depth + 1`` and is skipped outright.
    """
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        depth = len(here.parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []  # prune: nothing below this can be in range
            continue  # and this directory's own files are already too deep
        dirnames[:] = [d for d in dirnames if not d.startswith("._")]
        for filename in filenames:
            if filename.endswith(".json"):
                yield here / filename


def census(roots: list[Path], max_depth: int, as_json: bool = False) -> int:
    registered: dict[str, dict[str, object]] = {}
    listing = _git("ls-tree", "-r", "--name-only", "HEAD", PREREG_DIR_REL.as_posix())
    if listing.returncode == 0:
        for rel in listing.stdout.split():
            if not rel.endswith(".provenance.json"):
                continue
            blob = read_committed_blob(Path(rel))
            if blob is None:
                continue
            try:
                prov = json.loads(blob)
            except ValueError:
                continue
            src = prov.get("source_path_at_registration")
            if src:
                registered[str(Path(src))] = prov

    rows: list[dict[str, object]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in _walk_bounded(root.resolve(), max_depth):
            if path.name.startswith("._") or not _looks_like_prereg(path):
                continue
            key = str(path)
            prov = registered.get(key)
            if prov is None:
                verdict = UNREGISTERED
            else:
                birth = read_committed_blob(Path(str(prov["birth_copy"])))
                verdict = (
                    INTACT
                    if birth is not None
                    and hashlib.sha256(birth).hexdigest() == sha256_file(path)
                    else MUTATED
                )
            st = path.stat()
            birthtime = getattr(st, "st_birthtime", None)
            rows.append(
                {
                    "path": key,
                    "verdict": verdict,
                    "edited_after_birth": (
                        birthtime is not None and int(st.st_mtime) > int(birthtime)
                    ),
                }
            )

    rows.sort(key=lambda r: str(r["path"]))
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["verdict"])] = counts.get(str(row["verdict"]), 0) + 1
    edited = sum(1 for r in rows if r["edited_after_birth"])
    summary = {
        "roots": [str(r) for r in roots if r.is_dir()],
        "total_prereg_like_files": len(rows),
        "counts": counts,
        "unregistered": counts.get(UNREGISTERED, 0),
        "edited_after_birth_by_mtime": edited,
        "mtime_caveat": "edited_after_birth is a SINGLE instrument (APFS "
        "birthtime vs mtime) and is indicative, not probative. Only files with "
        "a committed birth copy get a real INTACT/MUTATED verdict.",
        "rows": rows,
    }
    if as_json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"prereg-like files: {len(rows)}  ({counts})")
    print(f"  no committed birth copy : {counts.get(UNREGISTERED, 0)}")
    print(f"  mtime > birthtime       : {edited}  ({summary['mtime_caveat']})")
    for row in rows:
        flag = " *edited" if row["edited_after_birth"] else ""
        print(f"  {row['verdict']!s:<12} {row['path']}{flag}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="register_prereg.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="copy a prereg to git and commit (write-once)")
    reg.add_argument("--source", required=True, type=Path, help="live prereg path")
    reg.add_argument("--name", default=None, help="birth-copy stem (default: source stem)")
    reg.add_argument("--supersedes", default=None, help="prior prereg name this replaces")
    reg.add_argument("--message", default=None, help="override the commit message")
    reg.add_argument("--dry-run", action="store_true", help="do not write or commit")

    ver = sub.add_parser("verify", help="compare live prereg to its committed birth copy")
    ver.add_argument("--name", required=True)
    ver.add_argument("--live", default=None, type=Path, help="override the live path")
    ver.add_argument("--json", action="store_true", dest="as_json")

    cen = sub.add_parser("census", help="prereg-like files with no committed birth copy")
    cen.add_argument("--root", action="append", default=None, type=Path)
    cen.add_argument("--max-depth", type=int, default=4)
    cen.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "register":
            return register(
                args.source, args.name, args.supersedes, args.message, args.dry_run
            )
        if args.command == "verify":
            return verify(args.name, args.live, args.as_json)
        if args.command == "census":
            roots = args.root or list(DEFAULT_CENSUS_ROOTS)
            return census(roots, args.max_depth, args.as_json)
    except PreregError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    # Deliberately NO os.chdir(REPO_ROOT): every git call already passes
    # cwd=REPO_ROOT explicitly, and chdir'ing would silently re-root a relative
    # --source against the repo instead of the caller's directory.
    raise SystemExit(main())
