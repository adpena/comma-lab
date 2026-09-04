#!/usr/bin/env python3
"""Certify-or-block reclaim of BOOT-VOLUME bulk (CLAUDE.md local-disk hygiene).

Scope, and why this is not a second mover
-----------------------------------------
``tools/vertigo_certify_move.py`` already implements the certified MOVE:
census -> hash -> copy -> independent destination re-read -> symlink -> retire.
This module deliberately does not repeat one byte of that. It adds the one
class that mover cannot express: a tree whose bytes are **already durably held
elsewhere in the repository's own object database**, so the honest reclaim is a
certified DELETE with no copy at all.

A git worktree or clone whose ``git status --porcelain`` is empty and whose
every ref resolves to a commit present in the main repository is fully
reconstructible by ``git worktree add`` / ``git checkout``. Copying 33 GiB of
such trees to the cold store would burn destination headroom to store bytes
git already stores. Deleting them without that proof would be signal loss.
The proof is the certificate.

Everything else routes OUT of here:

* ``certify_move_required`` -- dirty tree, unreachable ref, or plain bulk. The
  plan names ``vertigo_certify_move.py`` as the executor; this tool never
  deletes such a path.
* ``blocked_never_touch`` -- the hard boundary list (retained trees, GT caches,
  receipts, seals, live-claim and live-process referents, upstream/submissions).

Fail-closed: a candidate that does not *prove* itself into
``git_reconstructible`` is never deleted, whatever ``--apply`` says.

Usage:
    python tools/local_disk_reclaim.py --plan  --roots .omx/tmp/codex_worktrees
    python tools/local_disk_reclaim.py --apply --roots .omx/tmp/codex_worktrees \\
        --ledger .omx/state/disk_reclaim_certs_20260904.jsonl
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "local_disk_reclaim_cert.v1"

CLASS_GIT_RECONSTRUCTIBLE = "git_reconstructible"
CLASS_CERTIFY_MOVE_REQUIRED = "certify_move_required"
CLASS_BLOCKED_NEVER_TOUCH = "blocked_never_touch"

#: Path substrings that are never reclaimable by any class. Ordered by the
#: CLAUDE.md / charter boundary list; each entry is matched against the POSIX
#: path relative to the repo root AND against the absolute path.
NEVER_TOUCH_SUBSTRINGS: tuple[str, ...] = (
    "upstream/",
    "submissions/",
    "/retained/",
    "experiments/results/modal_auth_eval_mirror/",
    "experiments/results/ddm_fr2_final_review_20260903/",
    ".claude/worktrees/",
)

#: Path components that are never reclaimable when they name the leaf itself.
NEVER_TOUCH_LEAF_NAMES: tuple[str, ...] = ("retained", "seals", "receipts")

#: File suffixes that are never reclaimable (payload / GT caches / receipts).
NEVER_TOUCH_SUFFIXES: tuple[str, ...] = (
    ".done",
    ".last.txt",
    ".npz",
    ".pt",
)

#: Claim rows whose status begins with one of these are terminal: the lane is
#: closed, so the paths it names no longer pin bytes.
TERMINAL_CLAIM_PREFIXES: tuple[str, ...] = (
    "completed",
    "failed",
    "stopped",
    "refused_dispatch",
    "stale_superseded",
)

_PATH_TOKEN = re.compile(r"(?:/[\w.@+-]+)+/?|(?:\.omx|experiments|tools|src)(?:/[\w.@+-]+)+")


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ReclaimError(RuntimeError):
    """A candidate could not be classified with certainty."""


# ---------------------------------------------------------------------------
# Boundary predicates (pure; unit-testable without a filesystem)
# ---------------------------------------------------------------------------


def is_never_touch(path: str) -> bool:
    """True when ``path`` falls inside the hard never-touch boundary.

    ``path`` may be absolute or repo-relative. Matching is on the POSIX form so
    the same predicate serves both the planner and the tests.
    """
    p = str(path).replace(os.sep, "/")
    padded = p if p.endswith("/") else p + "/"
    for sub in NEVER_TOUCH_SUBSTRINGS:
        if sub in padded:
            return True
    leaf = padded.rstrip("/").rsplit("/", 1)[-1]
    if leaf in NEVER_TOUCH_LEAF_NAMES:
        return True
    stripped = p.rstrip("/")
    return any(stripped.endswith(sfx) for sfx in NEVER_TOUCH_SUFFIXES)


def find_never_touch_descendant(root: Path, *, limit: int = 200_000) -> str | None:
    """Return the first protected path found INSIDE ``root``, else None.

    ``is_never_touch`` only judges the candidate's own path. A tree whose own
    name is innocuous can still CONTAIN a protected subtree -- the case that
    caught this out is ``arm_receipts_local/ddm_mst1_manufactured_stage_split``,
    whose entire 20.55 GiB lives under ``capture_r2_local/retained/``. Moving it
    as one unit would have relocated a never-touch tree while every top-level
    check passed. A container of protected bytes is itself protected.
    """
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + filenames:
            seen += 1
            if seen > limit:  # fail-closed: an unscannable tree is not clearable
                return f"{root} (scan exceeded {limit} entries; not provably clear)"
            candidate = os.path.join(dirpath, name)
            if is_never_touch(candidate):
                return candidate
    return None


def claim_is_terminal(status: str) -> bool:
    """A claim row is terminal when its status starts with a closing prefix."""
    s = status.strip().lower()
    return any(s.startswith(pfx) for pfx in TERMINAL_CLAIM_PREFIXES)


def extract_path_tokens(text: str) -> set[str]:
    """Pull path-shaped tokens out of free text (claim notes, ps output)."""
    return {m.group(0).rstrip("/") for m in _PATH_TOKEN.finditer(text or "")}


def live_process_paths(ps_output: str, *, self_pids: set[int] | None = None) -> set[str]:
    """Paths named by live processes, EXCLUDING this tool's own command line.

    ``ps -Ao pid=,command=`` output. Without the self-exclusion the planner's own
    ``--roots`` argument enters the pin set and blocks every candidate under it,
    which is exactly how the first revision of this tool reported a 100%-blocked
    census that looked like caution and was actually a bug.
    """
    skip = self_pids or set()
    out: set[str] = set()
    for line in ps_output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, _, command = line.partition(" ")
        try:
            if int(pid_str) in skip:
                continue
        except ValueError:
            command = line
        out |= extract_path_tokens(command)
    return out


def active_claim_paths(
    claims_markdown: str,
    *,
    now_epoch: float,
    ttl_hours: float = 24.0,
) -> set[str]:
    """Paths pinned by a NON-terminal claim row inside the TTL window.

    A terminal row (completed/failed/stopped/...) releases its paths; an open
    row inside the TTL pins them. Rows older than the TTL are released whatever
    their status, matching the ledger's own 24 h conflict window.
    """
    pinned: set[str] = set()
    for line in claims_markdown.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8 or cells[0] in {"timestamp_utc", "---"}:
            continue
        try:
            # timegm, not mktime: the claim stamps are UTC. mktime reads local
            # time, and the usual "- time.timezone" patch is wrong under DST.
            ts = calendar.timegm(time.strptime(cells[0], "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            continue
        if (now_epoch - ts) > ttl_hours * 3600.0:
            continue
        if claim_is_terminal(cells[6]):
            continue
        pinned |= extract_path_tokens(cells[7])
        pinned |= extract_path_tokens(cells[4])
    return pinned


def path_is_pinned(candidate: Path, pinned: set[str]) -> bool:
    """True when ``candidate`` IS a pinned path, or CONTAINS one.

    Deliberately not an ancestor rule. A live reference to a candidate's parent
    directory (``--roots .omx/tmp/codex_worktrees``, or the repo root, which
    appears in essentially every command line) says nothing about whether one
    particular child holds live bytes -- and treating it as a pin blocks the
    entire census, which is a vacuous PASS wearing a safety costume. What
    genuinely pins a tree is a reference to the tree itself or to something
    inside it, because deleting it would destroy that referent.
    """
    c = str(candidate).rstrip("/")
    for raw in pinned:
        p = raw.rstrip("/")
        if not p or p == "/":
            continue
        if c == p or p.startswith(c + "/"):
            return True
    return False


# ---------------------------------------------------------------------------
# Git reconstructibility proof
# ---------------------------------------------------------------------------


@dataclass
class GitProof:
    """Evidence that a tree's bytes are already held by the main repo."""

    is_git: bool = False
    kind: str = "none"  # "worktree" (.git file) | "clone" (.git dir) | "none"
    head: str = ""
    porcelain_lines: int = -1
    n_refs: int = 0
    unreachable_refs: list[str] = field(default_factory=list)
    registered_worktree: bool = False
    error: str = ""

    @property
    def reconstructible(self) -> bool:
        return (
            self.is_git
            and not self.error
            and self.porcelain_lines == 0
            and self.n_refs > 0
            and not self.unreachable_refs
            and bool(self.head)
        )

    def as_dict(self) -> dict:
        return {
            "is_git": self.is_git,
            "kind": self.kind,
            "head": self.head,
            "porcelain_lines": self.porcelain_lines,
            "n_refs": self.n_refs,
            "n_unreachable_refs": len(self.unreachable_refs),
            "unreachable_refs": self.unreachable_refs[:10],
            "registered_worktree": self.registered_worktree,
            "reconstructible": self.reconstructible,
            "error": self.error or None,
        }


def _git(cwd: Path, *args: str, timeout: float = 120.0) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


def probe_git_proof(tree: Path, main_repo: Path, registered: set[str]) -> GitProof:
    """Measure, never assume, whether ``tree`` is git-reconstructible."""
    proof = GitProof()
    dot = tree / ".git"
    if dot.is_file():
        proof.is_git, proof.kind = True, "worktree"
    elif dot.is_dir():
        proof.is_git, proof.kind = True, "clone"
    else:
        return proof
    proof.registered_worktree = str(tree.resolve()) in registered
    try:
        rc, head = _git(tree, "rev-parse", "HEAD")
        if rc != 0:
            proof.error = "rev-parse HEAD failed"
            return proof
        proof.head = head.strip()
        rc, porcelain = _git(tree, "status", "--porcelain")
        if rc != 0:
            proof.error = "status --porcelain failed"
            return proof
        proof.porcelain_lines = len([ln for ln in porcelain.splitlines() if ln.strip()])
        # HEAD must specifically be a commit the main repo holds -- that is what
        # the rebuild command checks out.
        rc, _ = _git(main_repo, "cat-file", "-e", f"{proof.head}^{{commit}}")
        if rc != 0:
            proof.unreachable_refs.append(proof.head)
        rc, refs = _git(tree, "for-each-ref", "--format=%(objectname)")
        if rc != 0:
            proof.error = "for-each-ref failed"
            return proof
        objs = sorted({ln.strip() for ln in refs.splitlines() if ln.strip()})
        proof.n_refs = len(objs) + 1  # + HEAD
        for obj in objs:
            # PRESENCE, not commit-ness. Codex writes refs/codex/turn-diffs/*
            # pointing straight at TREE objects; dereferencing those with
            # ``^{commit}`` fails even though the main repo holds every byte.
            # Reading that failure as "absent from the main repo" is a false
            # BLOCK -- it cost a 38 GiB spurious copy plan before it was caught.
            rc, _ = _git(main_repo, "cat-file", "-e", obj)
            if rc != 0:
                proof.unreachable_refs.append(obj)
    except (subprocess.TimeoutExpired, OSError) as exc:
        proof.error = f"{type(exc).__name__}: {exc}"
    return proof


def registered_worktrees(main_repo: Path) -> set[str]:
    rc, out = _git(main_repo, "worktree", "list", "--porcelain")
    if rc != 0:
        return set()
    return {
        str(Path(ln.split(" ", 1)[1]).resolve())
        for ln in out.splitlines()
        if ln.startswith("worktree ")
    }


# ---------------------------------------------------------------------------
# Classification + certs
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    path: Path
    allocated_kib: int
    klass: str
    reason: str
    git: GitProof | None = None

    def cert_row(self, *, repo: Path, executor: str) -> dict:
        rel = os.path.relpath(self.path, repo)
        row = {
            "schema": SCHEMA,
            "utc": utcnow(),
            "original_path": str(self.path),
            "original_path_relative": rel,
            "allocated_kib": self.allocated_kib,
            "allocated_gib": round(self.allocated_kib / 2**20, 3),
            "class": self.klass,
            "rebuildable_reason": self.reason,
            "executor": executor,
        }
        if self.git is not None:
            row["git_proof"] = self.git.as_dict()
            row["rebuild_command"] = (
                ["git", "worktree", "add", str(self.path), self.git.head]
                if self.git.kind == "worktree"
                else ["git", "clone", str(repo), str(self.path), "&&", "git", "checkout", self.git.head]
            )
        return row


def du_kib(path: Path) -> int:
    out = subprocess.run(
        ["du", "-x", "-s", "-k", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return int(out.split()[0])


def classify(
    tree: Path,
    *,
    repo: Path,
    registered: set[str],
    pinned: set[str],
) -> Candidate:
    """Classify one candidate directory. Fail-closed by construction."""
    if is_never_touch(str(tree)):
        return Candidate(tree, 0, CLASS_BLOCKED_NEVER_TOUCH, "matches never-touch boundary")
    if path_is_pinned(tree, pinned):
        return Candidate(
            tree, 0, CLASS_BLOCKED_NEVER_TOUCH, "referenced by an active claim or live process"
        )
    kib = du_kib(tree)
    proof = probe_git_proof(tree, repo, registered)
    if proof.reconstructible:
        # No descendant scan here, deliberately. Every checkout contains
        # `upstream/` and `submissions/` paths, but in a clean tree whose refs
        # the main repo holds, those bytes ARE git's -- nothing is at risk.
        return Candidate(
            tree,
            kib,
            CLASS_GIT_RECONSTRUCTIBLE,
            f"clean {proof.kind} at {proof.head[:12]}; all {proof.n_refs} refs present in "
            f"{repo}; tree regenerable from the main object database",
            proof,
        )
    # Bulk, on the other hand, is moved verbatim -- so a protected subtree
    # anywhere inside it makes the whole container protected.
    protected = find_never_touch_descendant(tree)
    if protected is not None:
        return Candidate(
            tree, kib, CLASS_BLOCKED_NEVER_TOUCH, f"contains a never-touch path: {protected}"
        )
    if not proof.is_git:
        why = "not a git tree; bulk must be certify-MOVED, never deleted here"
    elif proof.error:
        why = f"git probe failed ({proof.error}); fail-closed"
    elif proof.porcelain_lines != 0:
        why = f"{proof.porcelain_lines} uncommitted path(s); bytes exist only here"
    else:
        why = f"{len(proof.unreachable_refs)} ref(s) absent from the main repo"
    return Candidate(tree, kib, CLASS_CERTIFY_MOVE_REQUIRED, why, proof)


def append_ledger(ledger: Path, row: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_moved_marker(path: Path, row: dict) -> None:
    """Leave a machine-readable marker where a tool may still read the path."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "MOVED_TO.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=".", help="Main repository root (reachability authority).")
    ap.add_argument(
        "--roots",
        action="append",
        default=[],
        required=True,
        metavar="DIR",
        help="Directory whose immediate children are reclaim candidates. Repeatable.",
    )
    ap.add_argument("--ledger", default=".omx/state/disk_reclaim_certs_20260904.jsonl")
    ap.add_argument(
        "--claims",
        default=".omx/state/active_lane_dispatch_claims.md",
        help="Claim ledger consulted for the 24 h pin window.",
    )
    ap.add_argument("--claim-ttl-hours", type=float, default=24.0)
    ap.add_argument("--plan", action="store_true", help="Dry run: print the cert table only.")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Execute the git_reconstructible class ONLY (certified delete, no copy).",
    )
    args = ap.parse_args(argv)

    if args.apply == args.plan:
        print("BLOCK: pass exactly one of --plan or --apply", file=sys.stderr)
        return 2

    repo = Path(args.repo).resolve()
    ledger = Path(args.ledger)
    registered = registered_worktrees(repo)

    pinned: set[str] = set()
    claims = Path(args.claims)
    if claims.exists():
        pinned |= active_claim_paths(
            claims.read_text(encoding="utf-8", errors="replace"),
            now_epoch=time.time(),
            ttl_hours=args.claim_ttl_hours,
        )
    try:
        ps = subprocess.run(
            ["ps", "-Ao", "pid=,command="], capture_output=True, text=True, timeout=30
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        print("BLOCK: could not read the live process table; fail-closed", file=sys.stderr)
        return 2
    pinned |= live_process_paths(ps, self_pids={os.getpid(), os.getppid()})

    candidates: list[Candidate] = []
    for root in args.roots:
        rp = Path(root)
        if not rp.is_dir():
            print(f"BLOCK: root is not a directory: {rp}", file=sys.stderr)
            return 2
        for child in sorted(rp.iterdir()):
            if child.is_symlink() or not child.is_dir():
                continue
            candidates.append(classify(child, repo=repo, registered=registered, pinned=pinned))

    by_class: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_class.setdefault(c.klass, []).append(c)

    print(f"{'CLASS':<26} {'GiB':>8}  PATH")
    for klass in (CLASS_GIT_RECONSTRUCTIBLE, CLASS_CERTIFY_MOVE_REQUIRED, CLASS_BLOCKED_NEVER_TOUCH):
        for c in by_class.get(klass, []):
            print(f"{klass:<26} {c.allocated_kib/2**20:>8.2f}  {c.path}  # {c.reason}")
    recl = sum(c.allocated_kib for c in by_class.get(CLASS_GIT_RECONSTRUCTIBLE, []))
    move = sum(c.allocated_kib for c in by_class.get(CLASS_CERTIFY_MOVE_REQUIRED, []))
    print(
        f"\ncertified-deletable {recl/2**20:.2f} GiB in "
        f"{len(by_class.get(CLASS_GIT_RECONSTRUCTIBLE, []))} tree(s); "
        f"certify-MOVE owed {move/2**20:.2f} GiB in "
        f"{len(by_class.get(CLASS_CERTIFY_MOVE_REQUIRED, []))} tree(s) "
        f"(executor: tools/vertigo_certify_move.py)"
    )

    if args.plan:
        return 0

    freed = 0
    for c in by_class.get(CLASS_GIT_RECONSTRUCTIBLE, []):
        assert c.git is not None and c.git.reconstructible, "fail-closed invariant"
        row = c.cert_row(repo=repo, executor="tools/local_disk_reclaim.py --apply")
        row["phase"] = "CERTIFIED_DELETE_PENDING"
        append_ledger(ledger, row)
        # Re-probe immediately before the irreversible step: the plan/apply gap
        # is a real window in which a tree can acquire uncommitted bytes.
        recheck = probe_git_proof(c.path, repo, registered)
        if not recheck.reconstructible or recheck.head != c.git.head:
            append_ledger(
                ledger,
                {**row, "phase": "BLOCKED_REPROBE_CHANGED", "reprobe": recheck.as_dict()},
            )
            print(f"BLOCK: {c.path} changed since classification; retained", file=sys.stderr)
            continue
        if c.git.registered_worktree:
            # Deliberately NOT --force: git's own dirty check is a second,
            # independent reader of the same invariant this class asserts.
            rc, _ = _git(repo, "worktree", "remove", str(c.path))
            if rc != 0:
                append_ledger(ledger, {**row, "phase": "BLOCKED_WORKTREE_REMOVE_REFUSED"})
                print(f"BLOCK: git refused to remove {c.path}; retained", file=sys.stderr)
                continue
        else:
            shutil.rmtree(c.path)
        if c.path.exists():
            append_ledger(ledger, {**row, "phase": "BLOCKED_STILL_PRESENT"})
            continue
        write_moved_marker(
            c.path,
            {
                **row,
                "phase": "CERTIFIED_DELETED",
                "note": "bytes retained in the main repository object database; "
                "rebuild with rebuild_command AFTER removing this marker directory "
                "(git refuses to populate a non-empty path)",
            },
        )
        append_ledger(ledger, {**row, "phase": "CERTIFIED_DELETED"})
        freed += c.allocated_kib
        print(f"[{utcnow()}] CERTIFIED_DELETED {c.path} (~{c.allocated_kib/2**20:.2f} GiB)")

    _rc, _ = _git(repo, "worktree", "prune")
    print(f"[{utcnow()}] freed ~{freed/2**20:.2f} GiB; git worktree prune run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
