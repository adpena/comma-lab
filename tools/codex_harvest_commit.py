#!/usr/bin/env python3
"""codex_harvest_commit.py — MAIN-side harvest + review + commit of a codex arm's diff.

WHY (the drift-accrual root cause, failure_id codex_workspace_write_sandbox_blocks_git_objects
_20260712, status RESOLVED-AT-SOURCE 2026-07-20 — codex_delegate now promotes `--sandbox
workspace-write` -> `danger-full-access`, so no commit-capable arm is FS-starved; this harvest
path is RETAINED as defense-in-depth + the MAIN-side review gate): a codex arm launched with
`--sandbox workspace-write` historically COULD NOT write `.git/objects`, so its `git add/commit`
failed rc=128 — the arm finished green but its work was UNCOMMITTED. With N concurrent arms sharing ONE working tree, their diffs intermingle into an
un-attributable pile (293 files by 2026-07-14) and the non-negotiable landing-review is defeated
(nothing is committed to review; `held_entangled` churn replaces the real follow-up).

The ledger's designed-but-never-built mitigation, now ENFORCED here: the arm writes a MANIFEST of
the files it touched; MAIN (unsandboxed) reads the manifest, REVIEWS, and serializer-commits exactly
those files — that harvest IS the non-negotiable follow-up review, and it lands the arm's signal
atomically per-arm so the pile never accrues.

REVIEW DISCIPLINE (preserved, not bypassed): safe artifacts (.md/.json/.jsonl/.txt/docs/research/
DAG) are committed directly; CODE (.py/.sh/…) is NOT auto-committed — it is surfaced as owed-review
and only committed with `--code-reviewed` after MAIN has actually reviewed it (the review gate stays
intact; that is the whole point).

USAGE:
  # harvest one done arm from its manifest (safe artifacts commit; code surfaced for review):
  .venv/bin/python tools/codex_harvest_commit.py --label X --stamp 20260714T170416Z
  # after reviewing the arm's code, commit the code too + disposition reviewed_committed:
  .venv/bin/python tools/codex_harvest_commit.py --label X --stamp ... --code-reviewed
  # legacy/pile drain (no manifest): pass the file list explicitly
  .venv/bin/python tools/codex_harvest_commit.py --label X --stamp ... --files a.md b.py ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO / ".omx" / "tmp" / "codex_manifests"
SERIALIZER = REPO / "tools" / "subagent_commit_serializer.py"
LANDING_GATE = REPO / "tools" / "codex_landing_review_gate.py"
REVIEW_TRACKER = REPO / "tools" / "review_tracker.py"
VENV_PY = REPO / ".venv" / "bin" / "python"

# CODE = needs human/main review before commit (the non-negotiable). Everything else = safe artifact.
_CODE_SUFFIXES = {".py", ".sh", ".rs", ".c", ".h", ".zig", ".mojo", ".js", ".ts", ".metal"}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _changed_files() -> set[str]:
    """Repo-relative paths with uncommitted working-tree changes (staged or not)."""
    r = subprocess.run(["git", "-C", str(REPO), "status", "--short"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # A failed status must not read as "nothing changed" — the harvest
        # would silently see an empty tree and mis-dispose the landing.
        raise SystemExit(
            f"git status failed (rc={r.returncode}): {r.stderr.strip()} — "
            "cannot reason about the working tree"
        )
    out: set[str] = set()
    for line in r.stdout.splitlines():
        # format: XY <path>  (XY = status codes); path starts at col 3
        path = line[3:].strip()
        if " -> " in path:  # rename
            path = path.split(" -> ", 1)[1]
        if path:
            out.add(path)
    return out


def _manifest_files(label: str, stamp: str) -> tuple[list[str], dict]:
    mf = MANIFEST_DIR / f"{label}_{stamp}.json"
    if not mf.is_file():
        return [], {}
    try:
        d = json.loads(mf.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], {}
    files = d.get("files") or []
    return [str(f) for f in files], d


def harvest(
    label: str,
    stamp: str,
    explicit_files: list[str] | None,
    code_reviewed: bool,
    consumed_by: str | None,
) -> int:
    manifest_files, manifest = _manifest_files(label, stamp)
    declared = list(explicit_files) if explicit_files else manifest_files
    if not declared:
        print(f"REFUSED: no manifest at {MANIFEST_DIR / f'{label}_{stamp}.json'} and no --files. "
              f"Cannot attribute {label}'s diff in the shared tree — pass --files explicitly "
              f"(from the arm's final message) to drain it safely.")
        return 2

    changed = _changed_files()
    present = [f for f in declared if f in changed]
    absent = [f for f in declared if f not in changed]
    if absent:
        print(f"[note] {len(absent)} declared file(s) have no working-tree change (already "
              f"committed or untouched): {absent[:5]}{'…' if len(absent) > 5 else ''}")
    if not present:
        print(f"[ok] nothing to harvest for {label} — all declared files already committed/clean.")
        return 0

    safe = [f for f in present if Path(f).suffix not in _CODE_SUFFIXES]
    code = [f for f in present if Path(f).suffix in _CODE_SUFFIXES]
    will_disposition = not code or code_reviewed
    if will_disposition and not consumed_by:
        print(
            "REFUSED: this harvest would create a terminal reviewed_committed "
            "disposition, so --consumed-by must name the task, DAG FEED, spec, "
            "memo, lever, or commit that consumes the findings."
        )
        return 2

    committed_any = False
    # 1) commit safe artifacts directly (no code review needed for .md/.json/research/DAG)
    if safe:
        rc = _serializer_commit(
            label, stamp, safe,
            f"harvest[{label}] safe artifacts ({len(safe)}) — main-side commit of "
            f"sandbox-stranded codex diff [no-triality]")
        committed_any = committed_any or (rc == 0)

    # 2) code: NOT auto-committed unless --code-reviewed (preserve the non-negotiable review gate)
    if code and not code_reviewed:
        print(f"\n⚠ {len(code)} CODE file(s) OWED REVIEW before commit (review gate — NOT bypassed):")
        for f in code:
            print(f"    {f}")
        print("  → review them, then re-run with --code-reviewed to mark + commit + disposition "
              "reviewed_committed.")
        # partial: safe landed, code owed. Disposition stays NEEDS_REVIEW (code) — do NOT claim done.
        return 0 if not safe or committed_any else 1
    if code and code_reviewed:
        for f in code:
            subprocess.run([str(VENV_PY), str(REVIEW_TRACKER), "mark-file", f,  # subprocess-no-check-OK: best-effort mark; the serializer review gate downstream fail-closes on unmarked files
                            "--status", "reviewed"], cwd=REPO)
        rc = _serializer_commit(
            label, stamp, code,
            f"harvest[{label}] reviewed code ({len(code)}) — main reviewed + committed "
            f"sandbox-stranded codex diff")
        committed_any = committed_any or (rc == 0)

    # 3) disposition the landing as reviewed_committed IFF everything present is now committed
    if committed_any and not (code and not code_reviewed):
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],  # subprocess-no-check-OK: provenance capture; empty-on-failure visible in the disposition row
                              capture_output=True, text=True).stdout.strip()
        disposition = subprocess.run(
            [
                str(VENV_PY),
                str(LANDING_GATE),
                "disposition",
                "--label",
                label,
                "--stamp",
                stamp,
                "--status",
                "reviewed_committed",
                "--commit",
                head or "HEAD",
                "--reason",
                f"harvested main-side: {len(safe)} safe + {len(code)} code file(s) "
                "reviewed+committed (sandbox-stranded diff landed via "
                "codex_harvest_commit)",
                "--consumed-by",
                consumed_by,
            ],
            cwd=REPO,
        )
        if disposition.returncode != 0:
            print(
                f"REFUSED: {label} committed at {head}, but the required landing "
                "disposition failed; inspect codex_landing_review_gate output."
            )
            return disposition.returncode
        print(f"[ok] {label} harvested + dispositioned reviewed_committed @ {head}")
    return 0


def _serializer_commit(label: str, stamp: str, files: list[str], message: str) -> int:
    args = [str(VENV_PY), str(SERIALIZER), "--message", message, "--files", *files]
    for f in files:
        p = REPO / f
        if p.is_file():
            args += ["--expected-content-sha256", f"{f}={_sha256(p)}"]
    r = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    print(r.stdout.strip()[-400:] if r.stdout else "", r.stderr.strip()[-200:] if r.stderr else "")
    return r.returncode


def merge_worktree(
    label: str,
    stamp: str,
    branch: str,
    worktree: str,
    reviewed: bool,
    consumed_by: str | None,
) -> int:
    """Isolated-worktree harvest: MAIN reviews the arm's branch diff and MERGES it to main (the single
    coherent, Courant-serialized integration point). No trample (disjoint domain); review at the merge
    boundary preserves the non-negotiable follow-up. --reviewed asserts MAIN reviewed the branch diff."""
    import fcntl
    wt = Path(worktree)
    merge_lock = REPO / ".omx" / "state" / ".commit-lock"  # reuse the serializer lock -> coherent ref updates

    def _run(a, cwd=REPO):
        return subprocess.run([str(x) for x in a], cwd=str(cwd), capture_output=True, text=True)

    # 1) Refuse leftover edits. MAIN cannot honestly assert it reviewed a branch
    # while silently sweeping additional worktree bytes into an unreviewed commit.
    if wt.is_dir() and _run(["git", "-C", wt, "status", "--short"]).stdout.strip():
        print(
            f"REFUSED: isolated worktree {wt} has uncommitted edits. The arm must "
            "serialize them first so MAIN reviews an immutable branch diff."
        )
        return 2
    # 2) branch changes vs main
    names = _run(["git", "-C", REPO, "diff", "--name-only", f"main...{branch}"]).stdout.split()
    if not names:
        print(f"[ok] {label}: branch {branch} has no changes vs main — nothing to merge.")
        return 0
    code = [n for n in names if Path(n).suffix in _CODE_SUFFIXES]
    if not reviewed:
        print(f"\n=== {label} branch {branch} OWED REVIEW before merge "
              f"({len(names)} files, {len(code)} code) ===")
        print(_run(["git", "-C", REPO, "diff", "--stat", f"main...{branch}"]).stdout.strip()[:1500])
        print(f"\n  → review `git diff main...{branch}`, then re-run with --reviewed to merge to main.")
        return 1
    if not consumed_by:
        print(
            "REFUSED: --reviewed merge requires --consumed-by naming the task, "
            "DAG FEED, spec, memo, lever, or commit that consumes the findings."
        )
        return 2
    # 3) reviewed → merge under the serializer lock (serialized ref update = Courant<=1)
    with open(merge_lock, "a") as lk:  # BARE_WRITE_OK: lockfile descriptor only; immediately flocked below
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            m = _run(["git", "-C", REPO, "merge", "--no-ff", "-m",
                      f"merge codex worktree {branch} (reviewed main-side harvest)", branch])
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    print((m.stdout + m.stderr).strip()[-500:])
    if m.returncode != 0:
        print(f"REFUSED: merge of {branch} failed (main tree not clean, or conflict). "
              f"Clean/resolve main first, then re-run.")
        return 2
    # 4) cleanup + disposition
    remove = _run(["git", "-C", REPO, "worktree", "remove", str(wt)])
    if remove.returncode != 0:
        print(f"WARN: merged {branch}, but clean worktree removal failed: {remove.stderr.strip()}")
    delete = _run(["git", "-C", REPO, "branch", "-d", branch])
    if delete.returncode != 0:
        print(f"WARN: merged {branch}, but safe branch deletion failed: {delete.stderr.strip()}")
    head = _run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"]).stdout.strip()
    disposition = subprocess.run(
        [
            str(VENV_PY),
            str(LANDING_GATE),
            "disposition",
            "--label",
            label,
            "--stamp",
            stamp,
            "--status",
            "reviewed_committed",
            "--commit",
            head or "HEAD",
            "--reason",
            f"worktree-isolated arm: reviewed branch {branch} + merged to main "
            f"({len(names)} files); clean worktree removal attempted",
            "--consumed-by",
            consumed_by,
        ],
        cwd=REPO,
    )
    if disposition.returncode != 0:
        print(
            f"REFUSED: {label} merged at {head}, but its required landing "
            "disposition failed; inspect codex_landing_review_gate output."
        )
        return disposition.returncode
    print(f"[ok] {label} merged + dispositioned reviewed_committed @ {head}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Harvest + review + commit a codex arm's diff (main-side).")
    ap.add_argument("--label", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--files", nargs="*", help="explicit file list (legacy/no-manifest shared-tree drain)")
    ap.add_argument("--code-reviewed", action="store_true",
                    help="assert MAIN reviewed the arm's code files → mark reviewed + commit them")
    ap.add_argument("--merge-worktree", help="ISOLATED mode: path to the arm's git worktree to merge")
    ap.add_argument("--branch", help="the arm's branch (codexwt/<label>_<stamp>) to merge to main")
    ap.add_argument("--reviewed", action="store_true", help="assert MAIN reviewed the branch diff → merge")
    ap.add_argument(
        "--consumed-by",
        help=(
            "required for a terminal reviewed_committed disposition: task/DAG-FEED/"
            "spec/memo/lever/commit that consumes the landing"
        ),
    )
    args = ap.parse_args(argv)
    if args.merge_worktree:
        if not args.branch:
            ap.error("--merge-worktree requires --branch")
        return merge_worktree(
            args.label,
            args.stamp,
            args.branch,
            args.merge_worktree,
            args.reviewed,
            args.consumed_by,
        )
    return harvest(args.label, args.stamp, args.files, args.code_reviewed, args.consumed_by)


if __name__ == "__main__":
    raise SystemExit(main())
