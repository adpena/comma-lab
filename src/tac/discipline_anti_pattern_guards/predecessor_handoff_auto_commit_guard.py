# SPDX-License-Identifier: MIT
"""PREDECESSOR-HANDOFF guard — sister of Catalog #117 at the handoff surface.

Anti-pattern #14 (``predecessor_working_tree_uncommitted_handoff_v1``)
canonical_unwind_path implementation. Returns a typed
:class:`HandoffGuardVerdict` for the parent agent / successor subagent
at session-entry; optionally auto-commits via canonical
``tools/subagent_commit_serializer.py`` if the predecessor left
uncommitted edits in the shared working tree.

The guard inspects ``git status --porcelain`` to detect predecessor-
owned modifications. Per CLAUDE.md "Subagent commits MUST use serializer"
non-negotiable: auto-commit MUST route through the canonical serializer
so Catalog #117 + #157 + #174 + #289 + #340 sister-checkpoint guard all
fire correctly.

Recommendation taxonomy
───────────────────────
``PROCEED``                              — clean working tree; no handoff residue
``PREDECESSOR_UNCOMMITTED_WORK_DETECTED`` — uncommitted edits detected;
                                            ``auto_commit=False`` mode returns
                                            this; caller must coordinate
                                            manually
``AUTO_COMMIT_LANDED``                   — auto-committed via canonical
                                            serializer; new HEAD sha in
                                            ``diagnostic``
``AUTO_COMMIT_FAILED``                   — auto-commit attempt failed; manual
                                            intervention needed (e.g. canonical
                                            serializer unavailable, or commit
                                            failed mid-flight)

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: this helper is
observability-only when ``auto_commit=False`` (default). When
``auto_commit=True``, it routes a single ``subagent_commit_serializer.py``
invocation. Per Catalog #117 + #174 + #340: the canonical serializer
enforces the full commit-machinery discipline.

Per CLAUDE.md "Beauty, simplicity, and developer experience": narrow
typed API; frozen dataclasses; explicit invariants in ``__post_init__``;
no hidden state.
"""
from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from pathlib import Path
from typing import Literal


# ── Constants ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
"""Repo root: src/tac/discipline_anti_pattern_guards/ lives three levels under repo root."""

CANONICAL_SERIALIZER_PATH = REPO_ROOT / "tools" / "subagent_commit_serializer.py"
"""Per CLAUDE.md "Subagent commits MUST use serializer" non-negotiable +
Catalog #117 + #157 + #174 + #289 + #340 sister enforcement."""

# Common ephemeral/state files that should NOT trigger auto-commit
# (they are commonly multi-subagent state per Catalog #340 exempt set;
# fcntl-locked helpers handle their writes).
EXEMPT_PATHS: frozenset[str] = frozenset({
    ".omx/state/modal_call_id_ledger.jsonl",
    ".omx/state/active_lane_dispatch_claims.md",
    ".omx/state/commit-serializer.log",
    ".omx/state/catalog-claim.log",
    ".omx/state/subagent_progress.jsonl",
    ".omx/state/lane_registry.json",
    ".omx/state/lane_maturity_audit.log",
    ".omx/state/continual_learning_posterior.jsonl",
    ".omx/state/cost_band_posterior.jsonl",
    ".omx/state/next_catalog_number.txt",
    ".omx/state/probe_outcomes.jsonl",
    ".omx/state/council_deliberation_posterior.jsonl",
    ".omx/state/canonical_equations_registry.jsonl",
    ".omx/state/canonical_anti_patterns_registry.jsonl",
    ".omx/state/canonical_task_status.jsonl",
    "MEMORY.md",
})


HandoffGuardRecommendation = Literal[
    "PROCEED",
    "PREDECESSOR_UNCOMMITTED_WORK_DETECTED",
    "AUTO_COMMIT_LANDED",
    "AUTO_COMMIT_FAILED",
]


# ── Typed verdict ────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class PredecessorHandoffContext:
    """Optional context for ``verify_predecessor_working_tree_committed_or_auto_commit``.

    Attributes
    ----------
    predecessor_subagent_id : str | None
        ID of the predecessor subagent whose handoff is being inspected.
        Used for the auto-commit message body if auto-commit fires.
    successor_subagent_id : str | None
        ID of the successor subagent calling this helper. Used for the
        canonical serializer's ``--current-subagent-id`` arg.
    auto_commit_message_subject : str | None
        Override the auto-commit subject line. Default:
        "handoff: predecessor uncommitted edits auto-committed".
    """

    predecessor_subagent_id: str | None = None
    successor_subagent_id: str | None = None
    auto_commit_message_subject: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "predecessor_subagent_id",
            "successor_subagent_id",
            "auto_commit_message_subject",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a str or None")


@dataclasses.dataclass(frozen=True)
class HandoffGuardVerdict:
    """Typed verdict for predecessor-handoff inspection.

    Attributes
    ----------
    recommendation : HandoffGuardRecommendation
        ``PROCEED`` | ``PREDECESSOR_UNCOMMITTED_WORK_DETECTED`` |
        ``AUTO_COMMIT_LANDED`` | ``AUTO_COMMIT_FAILED``
    uncommitted_paths : tuple[str, ...]
        Repo-relative path strings the predecessor left uncommitted
        (filtered to non-exempt paths). Empty if ``recommendation == "PROCEED"``.
    diagnostic : str
        Human-readable diagnostic explaining the verdict.
    auto_commit_attempted : bool
        True iff the caller requested auto-commit AND uncommitted edits
        were detected.
    new_head_sha : str | None
        HEAD sha after a successful auto-commit; None otherwise.
    canonical_serializer_invoked : bool
        True iff the canonical serializer was invoked (whether or not the
        commit succeeded).
    """

    recommendation: HandoffGuardRecommendation
    uncommitted_paths: tuple[str, ...]
    diagnostic: str
    auto_commit_attempted: bool
    new_head_sha: str | None
    canonical_serializer_invoked: bool

    def has_handoff_residue(self) -> bool:
        """True iff predecessor left uncommitted work (regardless of resolution)."""
        return bool(self.uncommitted_paths)


# ── Internal helpers ─────────────────────────────────────────────────────
def _git_status_porcelain(repo_root: Path) -> tuple[bool, list[str]]:
    """Return (success, list_of_porcelain_lines). Empty list = clean tree."""
    cmd = ["git", "-C", str(repo_root), "status", "--porcelain"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, []
    if proc.returncode != 0:
        return False, []
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return True, lines


def _parse_porcelain_lines(lines: list[str]) -> list[str]:
    """Extract repo-relative paths from ``git status --porcelain`` lines.

    Porcelain v1 format: ``XY <path>`` where X/Y are status chars and
    <path> may include a ``->`` for renames (``XY <old> -> <new>``).
    Returns paths in their git-canonical form (with backslash escapes
    decoded for shell-special chars). For renames, returns BOTH old + new.
    """
    out: list[str] = []
    for line in lines:
        if len(line) < 4:
            continue
        rest = line[3:].strip()
        # Strip surrounding quotes (git quotes paths with spaces / special chars)
        rest = rest.strip('"')
        if " -> " in rest:
            old, new = rest.split(" -> ", 1)
            out.append(old.strip().strip('"'))
            out.append(new.strip().strip('"'))
        else:
            out.append(rest)
    return out


def _filter_non_exempt_paths(paths: list[str]) -> list[str]:
    """Filter out canonical multi-subagent state paths."""
    return [p for p in paths if p and p not in EXEMPT_PATHS]


def _compute_post_edit_sha256(repo_root: Path, paths: list[str]) -> dict[str, str]:
    """Compute sha256 of each file's current working-tree content.

    Per Catalog #157 ``--expected-content-sha256`` discipline: callers
    must declare the working-tree content sha at lock-acquire time. We
    snapshot now so the canonical serializer's pre-pre-lock check has the
    correct value.
    """
    out: dict[str, str] = {}
    for rel in paths:
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            out[rel] = digest
        except OSError:
            continue
    return out


def _git_rev_parse_head(repo_root: Path) -> str | None:
    """Return HEAD sha or None on failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha if sha else None


def _invoke_canonical_serializer(
    repo_root: Path,
    serializer_path: Path,
    *,
    message: str,
    files: list[str],
    expected_shas: dict[str, str],
) -> tuple[bool, str]:
    """Invoke the canonical serializer; return (success, stdout_or_stderr)."""
    if not serializer_path.is_file():
        return False, f"canonical serializer not found at {serializer_path!r}"
    cmd = [
        ".venv/bin/python" if (repo_root / ".venv" / "bin" / "python").is_file() else "python",
        str(serializer_path),
        "--message",
        message,
        "--files",
        *files,
    ]
    for rel, sha in expected_shas.items():
        cmd.append("--expected-content-sha256")
        cmd.append(f"{rel}={sha}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"serializer invocation failed: {exc!s}"
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, combined


# ── Public API ───────────────────────────────────────────────────────────
def verify_predecessor_working_tree_committed_or_auto_commit(
    *,
    handoff_context: PredecessorHandoffContext | None = None,
    auto_commit: bool = False,
    repo_root: Path | None = None,
    serializer_path: Path | None = None,
) -> HandoffGuardVerdict:
    """Sister of Catalog #117 at the predecessor-handoff surface.

    Returns a :class:`HandoffGuardVerdict` whose ``recommendation`` is one of:

    * ``PROCEED`` — clean working tree; predecessor left no residue.
    * ``PREDECESSOR_UNCOMMITTED_WORK_DETECTED`` — uncommitted edits in
      the shared working tree (``auto_commit=False`` mode); successor
      must coordinate with predecessor or invoke this helper with
      ``auto_commit=True``.
    * ``AUTO_COMMIT_LANDED`` — uncommitted edits detected AND
      ``auto_commit=True``; auto-committed cleanly via canonical
      ``tools/subagent_commit_serializer.py``.
    * ``AUTO_COMMIT_FAILED`` — uncommitted edits detected AND
      ``auto_commit=True``; canonical serializer invocation failed
      (e.g. serializer missing, sister-checkpoint conflict per
      Catalog #340, commit-machinery error).

    Parameters
    ----------
    handoff_context
        Optional :class:`PredecessorHandoffContext` for diagnostic + auto-
        commit message body.
    auto_commit
        If True AND uncommitted edits detected, invoke the canonical
        serializer to auto-commit. Default False (observability-only mode).
    repo_root
        Override the repo root (useful for testing).
    serializer_path
        Override the canonical serializer path (useful for testing).
    """
    ctx = handoff_context if handoff_context is not None else PredecessorHandoffContext()
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    serializer_path = (
        Path(serializer_path) if serializer_path is not None else CANONICAL_SERIALIZER_PATH
    )

    success, lines = _git_status_porcelain(repo_root)
    if not success:
        # Per CLAUDE.md "Beauty, simplicity, and developer experience":
        # surface git-failure as PROCEED with diagnostic so the helper does
        # not silently block the successor; the canonical serializer will
        # surface git failure separately if/when invoked.
        return HandoffGuardVerdict(
            recommendation="PROCEED",
            uncommitted_paths=(),
            diagnostic=(
                "PROCEED (advisory): git status --porcelain failed; cannot "
                "determine handoff residue. The canonical serializer will "
                "surface git failure separately if/when invoked."
            ),
            auto_commit_attempted=False,
            new_head_sha=None,
            canonical_serializer_invoked=False,
        )

    all_paths = _parse_porcelain_lines(lines)
    non_exempt = _filter_non_exempt_paths(all_paths)

    if not non_exempt:
        diagnostic_tail = (
            ""
            if not all_paths
            else (
                f" ({len(all_paths)} exempt-state file(s) modified — these are "
                f"canonical multi-subagent state per Catalog #340 exempt set; "
                f"fcntl-locked helpers handle their writes)"
            )
        )
        return HandoffGuardVerdict(
            recommendation="PROCEED",
            uncommitted_paths=(),
            diagnostic=(
                "PROCEED: working tree has no non-exempt uncommitted "
                f"modifications{diagnostic_tail}."
            ),
            auto_commit_attempted=False,
            new_head_sha=None,
            canonical_serializer_invoked=False,
        )

    # Non-exempt uncommitted paths detected.
    if not auto_commit:
        path_lines = "\n".join(f"    - {p}" for p in non_exempt[:10])
        more = (
            ""
            if len(non_exempt) <= 10
            else f"\n    ... (+{len(non_exempt) - 10} more)"
        )
        diagnostic = (
            f"PREDECESSOR_UNCOMMITTED_WORK_DETECTED: {len(non_exempt)} non-"
            f"exempt path(s) modified in the working tree. Per CLAUDE.md "
            f"'Subagent commits MUST use serializer' + anti-pattern #14 "
            f"canonical_unwind_path: invoke this helper with "
            f"auto_commit=True to route the auto-commit through the "
            f"canonical serializer, OR coordinate with the predecessor "
            f"subagent to land its work via "
            f"tools/subagent_commit_serializer.py:\n"
            f"{path_lines}{more}"
        )
        return HandoffGuardVerdict(
            recommendation="PREDECESSOR_UNCOMMITTED_WORK_DETECTED",
            uncommitted_paths=tuple(non_exempt),
            diagnostic=diagnostic,
            auto_commit_attempted=False,
            new_head_sha=None,
            canonical_serializer_invoked=False,
        )

    # auto_commit=True path: snapshot post-edit shas per Catalog #157 +
    # invoke canonical serializer per Catalog #117 + #174 + #340.
    expected_shas = _compute_post_edit_sha256(repo_root, non_exempt)
    if not expected_shas:
        # All non-exempt paths failed to read — likely deleted files only.
        # Auto-commit via serializer without expected-shas (rc=4 may fire
        # if the serializer requires expected-sha for deleted files).
        # Per Catalog #174 + #289: declaring shas is mandatory for content
        # change; deleted-only commits may pass without.
        message = ctx.auto_commit_message_subject or (
            "handoff: predecessor uncommitted deletions auto-committed via Catalog #14 unwind"
        )
        ok, output = _invoke_canonical_serializer(
            repo_root,
            serializer_path,
            message=message,
            files=non_exempt,
            expected_shas={},
        )
    else:
        message = ctx.auto_commit_message_subject or (
            f"handoff: predecessor uncommitted edits auto-committed via "
            f"Catalog #14 unwind ({len(non_exempt)} file(s))"
        )
        ok, output = _invoke_canonical_serializer(
            repo_root,
            serializer_path,
            message=message,
            files=non_exempt,
            expected_shas=expected_shas,
        )

    if not ok:
        return HandoffGuardVerdict(
            recommendation="AUTO_COMMIT_FAILED",
            uncommitted_paths=tuple(non_exempt),
            diagnostic=(
                f"AUTO_COMMIT_FAILED: canonical serializer invocation failed. "
                f"Manual intervention needed. Output:\n{output[:2000]}"
            ),
            auto_commit_attempted=True,
            new_head_sha=None,
            canonical_serializer_invoked=True,
        )

    new_head = _git_rev_parse_head(repo_root)
    return HandoffGuardVerdict(
        recommendation="AUTO_COMMIT_LANDED",
        uncommitted_paths=tuple(non_exempt),
        diagnostic=(
            f"AUTO_COMMIT_LANDED: {len(non_exempt)} non-exempt path(s) auto-"
            f"committed via canonical serializer. New HEAD sha: "
            f"{new_head!r}. Per Catalog #117 + #174 + #340 sister-checkpoint "
            f"guard: the commit is canonical-machinery-compliant."
        ),
        auto_commit_attempted=True,
        new_head_sha=new_head,
        canonical_serializer_invoked=True,
    )
