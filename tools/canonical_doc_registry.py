#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""canonical_doc_registry — naming-INDEPENDENT registry of canonical SoT docs.

THE ROOT CAUSE THIS EXTINCTS (operator 2026-07-18, verbatim): "You searched,
but you searched for what you would have named it. You didn't do an exhaustive
search."  An agent globbed for the v10 spec under its OWN naming convention
(``*optimal_cold_start_capstone*``) and MISSED the canonical doc that already
existed under a DIFFERENT name (``SPEC_v10_capstone_cold_start_seeded_20260717``)
on an UNMERGED branch (``claude/p0_521_spec_v10_capstone_20260717``), then
created a duplicate.  Name-anchored, main-scoped search is structurally blind to
same-content docs under other names / on other branches.  Memory:
``vehicle_naming_v9c_warm_lineage_v10_reserved_capstone_20260718.md``.
Sisters: config-orphan confound
(``[[config_orphan_confound_permanent_fix_lever_registry_20260706]]``) +
velocity-driven orphaning
(``[[velocity_driven_orphaning_the_deepest_signal_loss_meta_bug]]``).

The fix is structural, not remorse:

* ``lookup(concept)`` matches canonical docs by CONCEPT TAGS (naming-independent
  registry entries) AND by a real ``git grep`` of the concept tokens across the
  trees of ALL git refs (``git for-each-ref refs/heads refs/remotes`` deduped by
  commit) — so a same-concept doc is found even when it lives on an unmerged
  branch under someone else's naming convention.
* ``check_before_create(proposed)`` is the pre-create dedup: BEFORE creating any
  spec/design doc, ask the registry whether a same-concept doc already exists
  anywhere.  A non-empty answer means FOLD INTO the existing doc, do not create.

CLI::

    .venv/bin/python tools/canonical_doc_registry.py lookup "cold start seeded capstone"
    .venv/bin/python tools/canonical_doc_registry.py check "SPEC_v10_my_new_name_20260718.md"
    # `check` exits 2 when a same-concept canonical doc already exists.

Registry store: ``.omx/state/canonical_doc_registry.json`` (committed; the
``.gitignore`` carries an explicit negation).  Every entry's
``canonical_path``+``branch`` was VERIFIED against the real tree at registration
time (``git cat-file -e <branch>:<path>`` / filesystem) — see ``verify_entry``.

This module is the SoT for the search mechanics;
``tac.confound_gates.check_no_duplicate_canonical_spec_across_refs`` is the
preflight gate that refuses re-introduction of the duplicate-SoT bug class.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / ".omx" / "state" / "canonical_doc_registry.json"

_VALID_STATUS = ("active", "superseded", "draft")

# Tokens too generic to carry concept signal on their own.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "into", "that", "this", "our",
        "new", "doc", "docs", "file", "notes", "note", "final", "plan",
        "research", "omx", "md", "json", "20260101",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.]*")
_DATE_RE = re.compile(r"20\d{6}(?:t\d{4,6}z?)?")
_VEHICLE_TOKEN_RE = re.compile(r"^v\d")


def _tokenize(text: str) -> set[str]:
    """Lowercase concept tokens: alnum runs, dates stripped, stopwords dropped.

    Keeps short vehicle tokens like ``v8``/``v10`` (they are the highest-signal
    concept keys) while dropping other tokens shorter than 3 chars.
    """
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip(".")
        if not tok or _DATE_RE.fullmatch(tok):
            continue
        if _VEHICLE_TOKEN_RE.match(tok):
            # normalize v7.5 -> v75 so "v7.5" and "v75" collide.
            out.add("v" + re.sub(r"[^0-9a-z]", "", tok[1:]))
            continue
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        out.add(tok)
    return out


@dataclasses.dataclass(frozen=True)
class Entry:
    """One canonical SoT document, identified by CONCEPT, not by filename."""

    doc_id: str
    concept_tags: tuple[str, ...]
    canonical_path: str
    branch: str  # "main" or the branch that holds the canonical bytes
    status: str  # active | superseded | draft
    superseded_by: str | None
    one_line: str
    vehicle: str | None = None  # e.g. "v10" for vehicle-spec docs
    untracked_live_state: bool = False  # e.g. the frontier pointer (gitignored)

    def tag_tokens(self) -> set[str]:
        toks: set[str] = set()
        for tag in self.concept_tags:
            toks |= _tokenize(tag)
        if self.vehicle:
            toks |= _tokenize(self.vehicle)
        return toks


class RegistryError(ValueError):
    pass


def load_registry(registry_path: str | Path | None = None) -> list[Entry]:
    """Load + schema-validate the registry. Raises ``RegistryError`` on bad rows."""
    path = Path(registry_path or DEFAULT_REGISTRY_PATH)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("entries", data if isinstance(data, list) else [])
    entries: list[Entry] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RegistryError(f"entry[{i}]: not an object")
        status = row.get("status", "active")
        if status not in _VALID_STATUS:
            raise RegistryError(
                f"entry[{i}] ({row.get('doc_id')!r}): invalid status {status!r} "
                f"(must be one of {_VALID_STATUS})"
            )
        for req in ("doc_id", "concept_tags", "canonical_path", "branch", "one_line"):
            if not row.get(req):
                raise RegistryError(f"entry[{i}]: missing required field {req!r}")
        entries.append(
            Entry(
                doc_id=str(row["doc_id"]),
                concept_tags=tuple(str(t) for t in row["concept_tags"]),
                canonical_path=str(row["canonical_path"]),
                branch=str(row["branch"]),
                status=status,
                superseded_by=row.get("superseded_by"),
                one_line=str(row["one_line"]),
                vehicle=row.get("vehicle"),
                untracked_live_state=bool(row.get("untracked_live_state", False)),
            )
        )
    return entries


# ---------------------------------------------------------------------------
# git plumbing — the ALL-REFS content search (the anti-name-anchored core)
# ---------------------------------------------------------------------------


def _git(args: list[str], repo_root: Path) -> str:
    """Run git; '' on nonzero exit (git grep exits 1 on no-match)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def all_ref_commits(repo_root: Path, max_refs: int = 128) -> list[tuple[str, str]]:
    """(commit, short-ref) for ALL local+remote refs, deduped by commit object.

    This is the structural cure for main-scoped search: every branch's tree is
    a first-class search surface.
    """
    out = _git(
        [
            "for-each-ref",
            "--format=%(objectname) %(refname:short)",
            "refs/heads",
            "refs/remotes",
        ],
        repo_root,
    )
    seen: set[str] = set()
    commits: list[tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        sha, ref = parts
        if sha in seen:
            continue
        seen.add(sha)
        commits.append((sha, ref))
        if len(commits) >= max_refs:
            break
    return commits


def grep_ref_for_tokens(
    repo_root: Path,
    commit: str,
    tokens: list[str],
    pathspec: str = "*.md",
) -> list[str]:
    """Paths in ``commit``'s tree whose CONTENT contains all ``tokens``.

    Real ``git grep --all-match`` over the ref's tree — content-based, so a doc
    is found regardless of what its creator named it.  Falls back to the two
    longest tokens when the full conjunction finds nothing (docs rarely repeat
    every query word verbatim).
    """
    if not tokens:
        return []

    def _run(toks: list[str]) -> list[str]:
        args = ["grep", "-i", "-l", "--all-match"]
        for t in toks:
            args += ["-e", t]
        args += [commit, "--", pathspec]
        out = _git(args, repo_root)
        hits = []
        for line in out.splitlines():
            # format: <commit>:<path>
            _, _, path = line.partition(":")
            if path:
                hits.append(path)
        return hits

    hits = _run(tokens)
    if not hits and len(tokens) > 2:
        fallback = sorted(tokens, key=len, reverse=True)[:2]
        hits = _run(fallback)
    return hits


def grep_worktree_for_tokens(
    repo_root: Path, tokens: list[str], pathspec: str = "*.md"
) -> list[str]:
    """Same content search over the working tree INCLUDING untracked files."""
    if not tokens:
        return []
    args = ["grep", "-i", "-l", "--all-match", "--untracked"]
    for t in tokens:
        args += ["-e", t]
    args += ["--", pathspec]
    return [line for line in _git(args, repo_root).splitlines() if line]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup(
    concept: str,
    *,
    repo_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    git_search: bool = True,
    max_refs: int = 128,
) -> list[Entry]:
    """Canonical docs matching ``concept`` — by tags AND by all-refs content grep.

    Returns registry ``Entry`` rows first (sorted by tag-overlap score), then
    synthesized ``unregistered:*`` entries for content hits on ANY ref that are
    not already covered by a registry row.  The query is matched by CONCEPT
    (token overlap / content), never by the caller's guessed filename.
    """
    root = Path(repo_root or REPO_ROOT)
    tokens = _tokenize(concept)
    entries = load_registry(
        registry_path
        if registry_path is not None
        else (root / ".omx" / "state" / "canonical_doc_registry.json")
    )

    scored: list[tuple[int, Entry]] = []
    registered_paths: set[str] = set()
    for e in entries:
        registered_paths.add(e.canonical_path)
        overlap = tokens & e.tag_tokens()
        score = len(overlap)
        vehicle_hit = any(_VEHICLE_TOKEN_RE.match(t) for t in overlap)
        if score >= 2 or vehicle_hit:
            scored.append((score + (2 if vehicle_hit else 0), e))
    scored.sort(key=lambda se: -se[0])
    results = [e for _, e in scored]

    if git_search and tokens:
        tok_list = sorted(tokens)
        seen_paths: set[str] = set(registered_paths)
        # Working tree (incl. untracked) first, then every ref's tree.
        for path in grep_worktree_for_tokens(root, tok_list):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            results.append(
                Entry(
                    doc_id=f"unregistered:{path}",
                    concept_tags=tuple(tok_list),
                    canonical_path=path,
                    branch="worktree",
                    status="draft",
                    superseded_by=None,
                    one_line="content match in working tree (unregistered)",
                )
            )
        for commit, ref in all_ref_commits(root, max_refs=max_refs):
            for path in grep_ref_for_tokens(root, commit, tok_list):
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                results.append(
                    Entry(
                        doc_id=f"unregistered:{path}",
                        concept_tags=tuple(tok_list),
                        canonical_path=path,
                        branch=ref,
                        status="draft",
                        superseded_by=None,
                        one_line=f"content match on ref {ref} (unregistered)",
                    )
                )
    return results


def check_before_create(
    proposed_path_or_concept: str,
    *,
    repo_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    git_search: bool = True,
    max_refs: int = 128,
) -> list[Entry]:
    """Pre-create dedup: existing same-concept canonical docs for a proposal.

    Feed it the path/name YOU were about to use; it answers with the docs that
    already hold that concept (on any branch, under any name).  Non-empty ⇒
    fold into the existing doc instead of creating a duplicate.
    """
    stem = Path(proposed_path_or_concept).name
    stem = re.sub(r"\.[a-z0-9]+$", "", stem, flags=re.IGNORECASE)
    concept = re.sub(r"[_\-/]+", " ", stem)
    matches = lookup(
        concept,
        repo_root=repo_root,
        registry_path=registry_path,
        git_search=git_search,
        max_refs=max_refs,
    )
    # The proposal itself (if it already exists on disk) is not a "duplicate".
    proposed_norm = proposed_path_or_concept.strip()
    if proposed_norm.startswith("./"):
        proposed_norm = proposed_norm[2:]
    return [m for m in matches if m.canonical_path != proposed_norm]


def verify_entry(entry: Entry, repo_root: str | Path | None = None) -> str:
    """Verify an entry's canonical bytes are reachable. Returns a status string:
    'ok-worktree' | 'ok-ref' | 'ok-live-state' | 'missing'."""
    root = Path(repo_root or REPO_ROOT)
    if (root / entry.canonical_path).is_file():
        return "ok-worktree"
    for ref in (entry.branch, f"origin/{entry.branch}"):
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{ref}:{entry.canonical_path}"],
            cwd=str(root),
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return "ok-ref"
    if entry.untracked_live_state:
        # Live-state files (e.g. the frontier pointer) exist only on the
        # primary checkout; a linked worktree legitimately lacks them.
        return "ok-live-state"
    return "missing"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt(e: Entry) -> str:
    sup = f" superseded_by={e.superseded_by}" if e.superseded_by else ""
    return (
        f"{e.doc_id}\n    path={e.canonical_path} branch={e.branch} "
        f"status={e.status}{sup}\n    {e.one_line}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, hint in (
        ("lookup", "concept phrase, e.g. 'cold start seeded capstone'"),
        ("check", "proposed doc name/path or concept (pre-create dedup)"),
    ):
        p = sub.add_parser(name)
        p.add_argument("concept", help=hint)
        p.add_argument("--no-git", action="store_true", help="registry-only match")
        p.add_argument("--repo-root", default=None)
        p.add_argument("--registry", default=None)
    args = parser.parse_args(argv)

    kwargs = {
        "repo_root": args.repo_root,
        "registry_path": args.registry,
        "git_search": not args.no_git,
    }
    if args.cmd == "lookup":
        hits = lookup(args.concept, **kwargs)
        for e in hits:
            print(_fmt(e))
        if not hits:
            print("(no canonical doc matches — safe to treat as new concept)")
        return 0
    hits = check_before_create(args.concept, **kwargs)
    if hits:
        print(
            f"DUPLICATE-SoT RISK: {len(hits)} existing canonical/same-concept "
            f"doc(s) for {args.concept!r} — FOLD INTO the canonical doc, do not "
            f"create a new one (name-anchored-search bug class; memory: "
            f"vehicle_naming_v9c_warm_lineage_v10_reserved_capstone_20260718.md):"
        )
        for e in hits:
            print(_fmt(e))
        return 2
    print("(no existing same-concept doc found across registry + all refs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
