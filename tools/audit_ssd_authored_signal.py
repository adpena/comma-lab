#!/usr/bin/env python3
"""audit_ssd_authored_signal.py — standing guard against AUTHORED signal living on the SSD alone.

WHY. The SSD tier (`/Volumes/APDataStore/pact`, `/Volumes/VertigoDataTier/pact`) is for ARTIFACTS:
bulky, rebuildable run output. Authored CODE and MARKDOWN belong in git. When an arm writes a
solver, an encoder, or a verdict-producing script into its SSD workspace and never commits it, the
signal exists on exactly one un-backed disk. `ddm_oc2` (2026-08-20) measured that gap for the first
time and found 1,071 distinct authored blobs with no counterpart anywhere in git history.

That was a one-shot sweep. This module is the same instrument made STANDING, so the gap is measured
on a cadence instead of rediscovered. It is READ-ONLY observability: it never copies, moves,
commits, or deletes anything. Per the CLAUDE.md default-off rule, score-neutral telemetry defaults
ON; the only gate here is compute cost, which is why the cadence monitor reads a cached summary
rather than re-scanning.

METHOD (byte identity, not path or name matching).
  1. Walk the SSD roots for code-like files, pruning caches/venvs/AppleDouble sidecars.
  2. Compute each file's git blob sha1 (`sha1(b"blob %d\\0" + content)`) — the exact identity
     `git hash-object` uses.
  3. Load every blob REACHABLE FROM A REF (`git rev-list --objects --all`). Reachable means a
     commit somewhere points at it, so it survives `git gc` and a fresh clone. This is ALL
     HISTORY, not just the working tree: a file deleted from HEAD but present in an old commit
     is still preserved and correctly does not count as absent.
  4. A file whose blob is not reachable exists ONLY on the SSD.

BEING IN THE OBJECT DATABASE IS NOT BEING PRESERVED. `ddm_oc2`'s original sweep used
`git cat-file --batch-all-objects`, which also returns blobs that were `git add`ed and never
committed, plus orphans from deleted branches. Measured 2026-08-20 on this repo: 129,656 blobs in
the object database, 104,425 reachable — a 25,231-blob gap that `git gc --prune` deletes and a
`git clone` never receives. Counting those as "safe in git history" UNDER-reports the debt, so
this module reports them as their own bucket E rather than as present.

HONEST DENOMINATOR. Absent files are bucketed, and every bucket is reported — the guard never
reports a filtered number as if it were the whole:
  A  third-party / clone / upstream mirror — excluded by operator policy (they stay put).
  B  run output / cold store — bulky rebuildable artifacts, untracked by repo convention.
  C  candidate AUTHORED sources — the real debt. Default disposition is COMMIT.
  D  certified-in-place — bucket-C blobs an owner explicitly certified with a rationale,
     recorded append-only in `.omx/state/ssd_authored_signal_certified.jsonl`.
  E  gc-eligible — the blob IS in the local object database but no ref reaches it (staged and
     never committed, or orphaned by a deleted branch). Recoverable today, deleted by the next
     `git gc --prune`, and absent from every clone. Owed like C, but cheaper to close.
Instance counts overstate authorship roughly 3x because arm workspaces copy a runtime tree per
run, so the headline is DISTINCT BLOBS. A file MOVED between SSD tiers is the same blob at a new
path and is therefore counted once, not twice.

USAGE
  .venv/bin/python tools/audit_ssd_authored_signal.py                  # scan + human report
  .venv/bin/python tools/audit_ssd_authored_signal.py --json           # machine-readable
  .venv/bin/python tools/audit_ssd_authored_signal.py --write-cache    # refresh the cadence cache
  .venv/bin/python tools/audit_ssd_authored_signal.py --manifest P.json  # full distinct-blob list
  .venv/bin/python tools/audit_ssd_authored_signal.py --summary-only   # read cache, no scan
  .venv/bin/python tools/audit_ssd_authored_signal.py --certify SHA --rationale "..." --owner ddm_x

Exit code 0 always (warn-only observability) unless `--strict`, which returns 2 when bucket C is
non-empty. The cadence consumer is `tools/consolidation_debt.py`, which reads the cache written by
`--write-cache` and reports staleness rather than paying for a full scan on every session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / ".omx" / "state" / "ssd_authored_signal_debt.json"
CERTIFIED = REPO / ".omx" / "state" / "ssd_authored_signal_certified.jsonl"

DEFAULT_ROOTS = (
    Path("/Volumes/APDataStore/pact"),
    Path("/Volumes/VertigoDataTier/pact"),
)

# Code-like extensions. Deliberately the same set ddm_oc2 measured, so the two denominators are
# comparable. Data extensions (.json/.npz/.bin) are NOT here: those are artifacts by definition.
CODE_EXT = frozenset({".py", ".sh", ".c", ".h", ".rs", ".md", ".toml", ".yaml", ".yml"})

# Directory names pruned entirely — build/cache/dependency trees carry no authored signal.
PRUNE_DIRS = frozenset({
    "__pycache__", ".venv", "venv", "site-packages", "node_modules", ".git",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "target", "build", "dist",
    ".cargo", ".rustup", ".conda", "conda-meta", ".tox", "wheels",
})

# Bucket A markers — a path segment equal to one of these, or matching one of the prefixes below,
# means third-party / harvested / mirrored content that the operator's policy leaves in place.
BUCKET_A_SEGMENTS = frozenset({
    "upstream", "third_party", "thirdparty", "vendor", "vendored",
    "public_datasets", "research_intake", "openpilot", "comma10k",
    "external", "downloads", "hf_cache", "huggingface", "torch_hub",
})
BUCKET_A_PREFIXES = ("public_pr", "comma-lab_latest", "codex_venvs")
BUCKET_A_SUBSTRINGS = ("_intake_", "/intake/")
# Vendored upstream compression libraries, unpacked into an arm workspace for a repro. Measured
# 2026-08-20: 105 blobs / 2.1 MB of bucket C were `brotli110_source/c/**` — genuinely somebody
# else's code, and committing it would put a third-party tree in our history for no signal.
BUCKET_A_VENDORED_RE = re.compile(
    r"/(brotli|zstd|zlib|lzma|xz|libdeflate|zopfli|lz4|snappy)[\w.\-]*_source/", re.IGNORECASE
)

# Bucket B markers — bulky rebuildable run output, untracked by long-standing repo convention.
# A bare `results` segment is deliberately NOT here. Arm workspaces routinely put authored builder
# scripts under `<arm>/results/`, and bucketing those as run output would hide real debt. This
# instrument exists to FIND debt, so an ambiguous path is flagged (bucket C) rather than excused.
BUCKET_B_SEGMENTS = frozenset({"cold_store", "coldstore", "harvested_artifacts"})
BUCKET_B_SUBSTRINGS = ("/experiments/results/", "/cold_store", "/coldstore")
# A cold store whose directory name merely CONTAINS the word, e.g. `vertigo_coldstore_20260811`.
BUCKET_B_SEGMENT_SUBSTRINGS = ("coldstore", "cold_store")
# A packet's `inflate.py` is a BUILD PRODUCT, not an authored source: the archive builder emits it
# with the payload embedded, which is why the largest "owed" rows were 4-6 MB single files. Its
# generator is committed, so it is rebuildable by definition. Measured 2026-08-20: 71 such blobs
# accounted for 15.1 MB — 64% of the entire owed byte volume — while carrying no authored signal.
BUCKET_B_GENERATED_RE = re.compile(r"/(packet|submission|runtime_tree|archive_dir)/inflate\.py$")


class AuditError(RuntimeError):
    """Raised when the audit cannot produce an honest answer (never for a merely dirty result)."""


# --------------------------------------------------------------------------------------- identity


def blob_sha1(path: Path) -> str | None:
    """Git blob sha1 of a file, or None if it cannot be read.

    A file we could not read is NOT silently dropped — the caller records it as unreadable so the
    denominator stays honest. Reporting a smaller scanned count as a cleaner result is exactly the
    vacuity==pass failure this repo has been bitten by.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def _git(args: list[str], repo: Path, timeout_s: int) -> str:
    try:
        proc = subprocess.run(args, cwd=repo, capture_output=True, text=True,
                              timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AuditError(f"{' '.join(args[:3])} failed: {type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        raise AuditError(f"{' '.join(args[:3])} rc={proc.returncode}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def odb_git_blobs(repo: Path = REPO, timeout_s: int = 300) -> set[str]:
    """Every blob sha PRESENT in the local object database, reachable or not.

    Used only to separate "gc-eligible" from "absent entirely" — never as the preservation test.
    """
    out = _git(["git", "cat-file", "--batch-all-objects",
                "--batch-check=%(objecttype) %(objectname)"], repo, timeout_s)
    blobs = {name.strip() for kind, _, name in (ln.partition(" ") for ln in out.splitlines())
             if kind == "blob" and name}
    if not blobs:
        raise AuditError("git cat-file returned zero blobs — refusing to report a phantom gap")
    return blobs


def reachable_git_blobs(repo: Path = REPO, timeout_s: int = 600) -> set[str]:
    """Every blob REACHABLE from a ref — the real preservation test.

    A reachable blob survives `git gc` and arrives in a fresh clone. Raises rather than returning
    a partial set: a truncated set would make tracked files look absent and manufacture phantom
    debt, which is the mirror-image failure of the under-count this function exists to fix.
    """
    try:
        out = _git(["git", "rev-list", "--objects", "--all",
                    "--filter=object:type=blob", "--no-object-names"], repo, timeout_s)
        blobs = {ln.strip() for ln in out.splitlines() if len(ln.strip()) == 40}
    except AuditError:
        # Older git without `--filter=object:type=blob`: enumerate then type-check in one batch.
        names = _git(["git", "rev-list", "--objects", "--all"], repo, timeout_s)
        shas = {ln.split(" ", 1)[0] for ln in names.splitlines() if ln.strip()}
        proc = subprocess.run(
            ["git", "cat-file", "--batch-check=%(objecttype) %(objectname)"],
            cwd=repo, input="\n".join(shas), capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )
        blobs = {name.strip() for kind, _, name in
                 (ln.partition(" ") for ln in proc.stdout.splitlines())
                 if kind == "blob" and name}
    if not blobs:
        raise AuditError("git rev-list returned zero reachable blobs — refusing to report a phantom gap")
    return blobs


# --------------------------------------------------------------------------------------- bucketing


def _is_clone_dir(directory: Path, cache: dict[Path, bool], root: Path) -> bool:
    """True when `directory` or an ancestor (up to `root`) contains its own `.git`.

    A nested `.git` is the strongest available evidence that a subtree is somebody else's repo
    rather than an arm's authored workspace. Memoized per directory so the walk stays linear.
    """
    if directory in cache:
        return cache[directory]
    try:
        here = (directory / ".git").exists()
    except OSError:
        here = False
    if here:
        cache[directory] = True
        return True
    parent = directory.parent
    if directory == root or parent == directory or root not in directory.parents:
        cache[directory] = False
        return False
    result = _is_clone_dir(parent, cache, root)
    cache[directory] = result
    return result


def classify(path: Path, root: Path, clone_cache: dict[Path, bool]) -> str:
    """Return bucket 'A', 'B', or 'C' for an absent file. Order matters: A wins over B wins over C."""
    text = str(path)
    parts = set(path.parts)
    if parts & BUCKET_A_SEGMENTS:
        return "A"
    if any(seg.startswith(BUCKET_A_PREFIXES) for seg in path.parts):
        return "A"
    if any(sub in text for sub in BUCKET_A_SUBSTRINGS):
        return "A"
    if BUCKET_A_VENDORED_RE.search(text):
        return "A"
    if _is_clone_dir(path.parent, clone_cache, root):
        return "A"
    if parts & BUCKET_B_SEGMENTS:
        return "B"
    if any(sub in text for sub in BUCKET_B_SUBSTRINGS):
        return "B"
    if any(sub in seg for seg in path.parts for sub in BUCKET_B_SEGMENT_SUBSTRINGS):
        return "B"
    if BUCKET_B_GENERATED_RE.search(text):
        return "B"
    return "C"


# ------------------------------------------------------------------------------------- certified


def load_certified(path: Path | None = None) -> dict[str, dict]:
    """Blob shas an owner explicitly certified in place, keyed by sha (latest row wins).

    The ledger location is resolved at CALL time, not bound as a default argument: an early-bound
    default silently ignores any later reassignment of `CERTIFIED`, so a redirected ledger would be
    read as empty and every certified blob would re-appear as debt. (A test caught exactly that.)
    """
    path = path or CERTIFIED
    rows: dict[str, dict] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return rows
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sha = row.get("blob_sha1")
        if isinstance(sha, str) and sha:
            rows[sha] = row
    return rows


def append_certification(sha: str, rationale: str, owner: str, path: str = "",
                         ledger: Path | None = None) -> dict:
    """Append one certify-in-place row. Narrow by design: scratch, third-party, or a rebuildable
    whose GENERATOR is committed. A placeholder rationale is refused — an uncertified blob is a
    more honest state than a fake certificate."""
    ledger = ledger or CERTIFIED
    rationale = (rationale or "").strip()
    if len(rationale) < 12 or rationale.lower() in {"<rationale>", "<reason>", "tbd", "placeholder"}:
        raise AuditError("certification needs a substantive rationale (>=12 chars, not a placeholder)")
    if not owner.strip():
        raise AuditError("certification needs an owner")
    if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha.lower()):
        raise AuditError(f"not a blob sha1: {sha!r}")
    row = {
        "blob_sha1": sha.lower(),
        "rationale": rationale,
        "owner": owner.strip(),
        "path": path,
        "certified_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


# ------------------------------------------------------------------------------------------ scan


def scan(roots=DEFAULT_ROOTS, reachable: set[str] | None = None,
         odb: set[str] | None = None) -> dict:
    """Full byte-identity sweep. Returns the report dict; writes nothing."""
    started = time.time()
    if reachable is None:
        reachable = reachable_git_blobs()
    if odb is None:
        odb = odb_git_blobs()
    certified = load_certified()

    scanned = 0
    unreadable: list[str] = []
    missing_roots: list[str] = []
    # blob sha -> {blob_sha1, ext, size_bytes, bucket, durability, representative_path,
    #              instance_count}. Keyed by blob so N copies of one source collapse to one debt.
    absent: dict[str, dict] = {}

    for root in roots:
        root = Path(root)
        if not root.is_dir():
            missing_roots.append(str(root))
            continue
        clone_cache: dict[Path, bool] = {}
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS and not d.endswith(".egg-info")]
            here = Path(dirpath)
            for name in filenames:
                # AppleDouble sidecars are filesystem metadata, never authored content.
                if name.startswith("._") or name == ".DS_Store":
                    continue
                if Path(name).suffix.lower() not in CODE_EXT:
                    continue
                fpath = here / name
                scanned += 1
                sha = blob_sha1(fpath)
                if sha is None:
                    unreadable.append(str(fpath))
                    continue
                if sha in reachable:
                    continue  # a ref points at these bytes: preserved, clone-safe, gc-safe.
                rec = absent.get(sha)
                if rec is None:
                    try:
                        size = fpath.stat().st_size
                    except OSError:
                        size = -1
                    rec = absent[sha] = {
                        "blob_sha1": sha,
                        "ext": Path(name).suffix.lower().lstrip("."),
                        "size_bytes": size,
                        "bucket": classify(fpath, root, clone_cache),
                        # "gc_eligible": the bytes are recoverable from the local ODB right now,
                        # but no ref holds them, so `git gc --prune` deletes them and no clone
                        # ever sees them. "absent": git has never held these bytes at all.
                        "durability": "gc_eligible" if sha in odb else "absent",
                        "representative_path": str(fpath),
                        "instance_count": 0,
                    }
                else:
                    # A blob seen in two buckets is authored if ANY instance is authored: the
                    # strictest (most-owed) reading wins, so a copy into cold_store cannot launder
                    # an authored source into bucket B.
                    b = classify(fpath, root, clone_cache)
                    if b == "C" and rec["bucket"] != "C":
                        rec["bucket"] = "C"
                        rec["representative_path"] = str(fpath)
                rec["instance_count"] += 1

    buckets = {"A": [], "B": [], "C": [], "D": []}
    for sha, rec in absent.items():
        b = rec["bucket"]
        if b == "C" and sha in certified:
            rec = dict(rec, certification=certified[sha])
            b = "D"
        buckets[b].append(rec)

    owed = sorted(buckets["C"], key=lambda r: (-r["size_bytes"], r["representative_path"]))
    gc_owed = [r for r in owed if r["durability"] == "gc_eligible"]
    return {
        "tool": "tools/audit_ssd_authored_signal.py",
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - started, 1),
        "method": {
            "identity": "git blob sha1 vs blobs REACHABLE from a ref (`git rev-list --objects --all`)",
            "why_not_odb": "`--batch-all-objects` also returns staged-never-committed and orphaned "
                           "blobs that `git gc --prune` deletes and no clone receives; counting "
                           "those as preserved under-reports the debt.",
            "roots": [str(r) for r in roots],
            "extensions": sorted(CODE_EXT),
            "pruned_dirs": sorted(PRUNE_DIRS),
        },
        "denominator": {
            "ssd_code_like_files_scanned": scanned,
            "files_unreadable": len(unreadable),
            "roots_absent_from_this_machine": missing_roots,
            "reachable_blobs_in_repo": len(reachable),
            "odb_blobs_in_repo": len(odb),
            "gc_eligible_blobs_in_repo": len(odb - reachable),
            "absent_file_instances": sum(r["instance_count"] for r in absent.values()),
            "absent_DISTINCT_blobs": len(absent),
            "bucket_A_third_party_or_clone": len(buckets["A"]),
            "bucket_B_run_output_or_coldstore": len(buckets["B"]),
            "bucket_C_authored_OWED": len(buckets["C"]),
            "bucket_C_of_which_gc_eligible": len(gc_owed),
            "bucket_D_certified_in_place": len(buckets["D"]),
        },
        "bucket_definitions": {
            "A": "third-party intake / OSS harvest / upstream mirror / nested clone — operator policy leaves these in place.",
            "B": "experiments/results/** and cold_store/** — bulky rebuildable run output, untracked by repo convention.",
            "C": "candidate AUTHORED sources in arm workspaces. Default disposition is COMMIT. THIS IS THE DEBT.",
            "D": "bucket-C blobs certified in place by an owner with a substantive rationale (append-only ledger).",
        },
        "unreadable_sample": unreadable[:20],
        "owed": owed,
        "certified_in_place": buckets["D"],
        "bucket_A_sample": [r["representative_path"] for r in buckets["A"][:20]],
        "bucket_B_sample": [r["representative_path"] for r in buckets["B"][:20]],
    }


# ---------------------------------------------------------------------------------------- output


def summarize(report: dict) -> dict:
    """The small dict the cadence monitor reads — no per-blob rows."""
    d = report["denominator"]
    return {
        "date_utc": report["date_utc"],
        "elapsed_s": report["elapsed_s"],
        "scanned": d["ssd_code_like_files_scanned"],
        "roots_absent": d["roots_absent_from_this_machine"],
        "unreadable": d["files_unreadable"],
        "absent_distinct_blobs": d["absent_DISTINCT_blobs"],
        "bucket_A": d["bucket_A_third_party_or_clone"],
        "bucket_B": d["bucket_B_run_output_or_coldstore"],
        "owed_authored_blobs": d["bucket_C_authored_OWED"],
        "owed_of_which_gc_eligible": d["bucket_C_of_which_gc_eligible"],
        "certified_in_place": d["bucket_D_certified_in_place"],
        "top_owed": [
            {"path": r["representative_path"], "ext": r["ext"], "size_bytes": r["size_bytes"]}
            for r in report["owed"][:10]
        ],
    }


def read_cache(path: Path = CACHE) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def fmt(report: dict) -> str:
    d = report["denominator"]
    lines = [
        f"[ssd-authored-signal] scanned {d['ssd_code_like_files_scanned']:,} code-like SSD files "
        f"vs {d['reachable_blobs_in_repo']:,} REACHABLE git blobs  ({report['elapsed_s']}s)",
        f"  (object database holds {d['odb_blobs_in_repo']:,}; "
        f"{d['gc_eligible_blobs_in_repo']:,} of those are gc-eligible, i.e. not preserved)",
        f"  unreachable from git     : {d['absent_file_instances']:,} instances "
        f"= {d['absent_DISTINCT_blobs']:,} DISTINCT blobs",
        f"    A third-party/clone    : {d['bucket_A_third_party_or_clone']:,}   (policy: stay put)",
        f"    B run-output/coldstore : {d['bucket_B_run_output_or_coldstore']:,}   (convention: untracked)",
        f"    D certified in place   : {d['bucket_D_certified_in_place']:,}   (owner + rationale on file)",
        f"    C AUTHORED — OWED      : {d['bucket_C_authored_OWED']:,}   <-- the debt"
        f"   ({d['bucket_C_of_which_gc_eligible']:,} of them gc-eligible: recoverable now, "
        f"gone after `git gc --prune`)",
    ]
    if d["roots_absent_from_this_machine"]:
        lines.append(f"  ! ROOT NOT MOUNTED (this scan is PARTIAL): "
                     f"{', '.join(d['roots_absent_from_this_machine'])}")
    if d["files_unreadable"]:
        lines.append(f"  ! {d['files_unreadable']} file(s) unreadable — excluded from the denominator, "
                     f"e.g. {report['unreadable_sample'][:1]}")
    if report["owed"]:
        lines.append("  largest owed:")
        for r in report["owed"][:8]:
            lines.append(f"    {r['size_bytes']:>9,} B  x{r['instance_count']:<3} {r['representative_path']}")
        lines.append("  -> COMMIT them into a repo home, or certify with:")
        lines.append("     tools/audit_ssd_authored_signal.py --certify <sha> --owner <arm> --rationale '...'")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument("--summary-only", action="store_true",
                    help="print the cached summary without scanning (cadence use)")
    ap.add_argument("--write-cache", action="store_true",
                    help=f"write the summary to {CACHE.relative_to(REPO)}")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="write the full distinct-blob report to this path")
    ap.add_argument("--root", action="append", default=None,
                    help="override scan root (repeatable)")
    ap.add_argument("--strict", action="store_true", help="rc=2 when bucket C is non-empty")
    ap.add_argument("--certify", default=None, metavar="SHA", help="certify one blob in place")
    ap.add_argument("--rationale", default="", help="substantive reason for --certify")
    ap.add_argument("--owner", default="", help="owning arm/operator for --certify")
    ap.add_argument("--path", default="", help="representative path for --certify")
    args = ap.parse_args(argv)

    if args.certify:
        try:
            row = append_certification(args.certify, args.rationale, args.owner, args.path)
        except AuditError as exc:
            print(f"[ssd-authored-signal] REFUSED: {exc}", file=sys.stderr)
            return 1
        print(f"[ssd-authored-signal] certified {row['blob_sha1'][:12]} by {row['owner']}")
        return 0

    if args.summary_only:
        cached = read_cache()
        if cached is None:
            print("[ssd-authored-signal] no cache — run without --summary-only to scan")
            return 0
        print(json.dumps(cached, indent=1) if args.json else
              f"[ssd-authored-signal] cached {cached['date_utc']}: "
              f"{cached['owed_authored_blobs']} authored blobs owed "
              f"({cached['scanned']:,} scanned, {cached['certified_in_place']} certified)")
        return 0

    roots = [Path(r) for r in args.root] if args.root else list(DEFAULT_ROOTS)
    try:
        report = scan(roots)
    except AuditError as exc:
        print(f"[ssd-authored-signal] DEGRADED — cannot answer honestly: {exc}", file=sys.stderr)
        return 1

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.write_cache:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(summarize(report), indent=1), encoding="utf-8")

    print(json.dumps(report, indent=1) if args.json else fmt(report))
    if args.strict and report["denominator"]["bucket_C_authored_OWED"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
