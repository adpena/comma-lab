# SPDX-License-Identifier: MIT
"""Pre-WRITE sister-activity check — git-log surface canonical helper.

The canonical helper landed by WAVE-3-PRE-WRITE-SISTER-ACTIVITY-CHECK-HELPER
2026-05-20 per CLAUDE.md "Subagent coherence-by-default" + "Bugs must be
permanently fixed AND self-protected against" non-negotiables. Sister of
``tac.commit_safety.sister_checkpoint_guard`` (Catalog #340 helper) at the
PRE-WRITE / git-log surface — together the two close the multi-subagent
edit/commit collision class bidirectionally at the EDIT-time-checkpoint
surface (sister) + the EDIT-time-git-log surface (this helper).

What this guards against
────────────────────────
Bug class anchor (2026-05-20): NERV-FAMILY-L0-BUILD subagent dispatched at
~13:30 UTC. Sister BUILD-1 NeRV-trio commit ``18b0beed6`` had landed
equivalent task at 2026-05-20T08:05:36 (~4.5h earlier) — adding L0 SCAFFOLD
trainers for ego_nerv / e_nerv / nervdc. NERV-FAMILY-L0-BUILD's PV read
sister NeRV trainers (TCNeRV / BlockNeRV / etc) for the canonical pattern
but did NOT run ``git log --oneline -5 -- experiments/train_substrate_ego_nerv.py``
— so it missed that the file was just-landed by sister + proceeded to write
its own version. Stood down correctly post-Write when post-PV audit caught
the duplication, but ~30 min wasted on triplicate work.

Gap in existing protection
──────────────────────────
- Catalog #340 sister-checkpoint guard fires at COMMIT time (subagent never
  reached commit step; helper never invoked)
- Catalog #314 detects POST-COMMIT absorption pattern (post-hoc detect)
- No canonical PRE-WRITE sister-activity check existed today

This helper closes that gap by checking ``git log --since="<lookback_hours>
hours ago" -- <files>`` for sister commits touching the target files in the
recent past. Distinct from sister_checkpoint_guard which checks the
in-flight ``.omx/state/subagent_progress.jsonl`` ledger (still-running
sisters); this helper checks the LANDED-COMMIT ledger (sisters that have
already shipped equivalent work in the lookback window).

Recommendation taxonomy
───────────────────────
``PROCEED``                — no sister commit touched any target file in the
                             lookback window; caller may proceed with Writes
``STAND_DOWN_DUPLICATE``   — at least one sister commit landed equivalent
                             work on a target file within the lookback;
                             caller should stand down + clean up rather than
                             duplicate the sister's work (NERV-FAMILY-L0
                             empirical anchor)
``WAIT_AND_REASSESS``      — sister activity present but ambiguous (e.g.
                             only some of the target files were touched);
                             caller should re-read sister landing memos
                             first + reconsider scope before proceeding

Wire-in surfaces
────────────────
1. ``tools/check_sister_files_recently_landed.py`` — operator-runnable CLI
   helper. Invoke BEFORE first Write in any subagent's PV step. Mirrors the
   sister Catalog #340 ``check_sister_checkpoint_before_git_add.py`` exit
   codes (rc=0/8/9) so operator habits transfer.
2. Subagent prompt template recommendation (see
   ``docs/canonical_subagent_pre_flight_checklist.md``) — every subagent's
   PV step should run this helper for its declared target files. The
   canonical pattern is documented (NOT enforced by a new STRICT preflight
   gate per Catalog #299 quota brake) because subagent-prompt-template
   discipline is the right surface for this enforcement rather than another
   gate.

Why no new STRICT preflight gate
────────────────────────────────
Per CLAUDE.md "Gate consolidation discipline" non-negotiable + Catalog #299:
the right enforcement surface for PRE-WRITE sister-activity is the subagent
prompt template (operator-side), not another preflight gate. A new STRICT
gate would require scanning subagent PV transcripts which is structurally
hard + adds noise. The helper + CLI + canonical doc pattern is sufficient
when wired into subagent prompts.

Memory: ``feedback_wave_3_pre_write_sister_activity_check_helper_landed_20260520.md``.
Lane: ``lane_wave_3_pre_write_sister_activity_check_helper_20260520``.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import subprocess
from pathlib import Path
from typing import Literal, Sequence

# ── Constants ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
"""Repo root: src/tac/commit_safety/ lives three levels under repo root."""

DEFAULT_LOOKBACK_HOURS = 6
"""Default lookback window in hours.

Mirrors the NERV-FAMILY-L0-BUILD empirical anchor (sister commit landed
~4.5h before duplicate dispatch). A 6h window catches the typical
"sister landed during my session, before I started" case while staying
short enough to avoid false-positives on related-but-not-duplicate
work landed earlier in the day.
"""

STAND_DOWN_FILE_OVERLAP_THRESHOLD = 1
"""If a sister commit touched at least this many target files, recommend
STAND_DOWN_DUPLICATE. Single-file overlap is the canonical signal because
the NERV-FAMILY-L0-BUILD empirical anchor had ONE sister commit
(``18b0beed6``) touching THREE target trainer files; even single-file
overlap is enough signal to stand down.
"""


Recommendation = Literal[
    "PROCEED",
    "STAND_DOWN_DUPLICATE",
    "WAIT_AND_REASSESS",
]


# ── Typed verdict ────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class SisterRecentlyLandedVerdict:
    """Typed verdict returned by ``check_sister_files_recently_landed``.

    Attributes
    ----------
    recommendation : Recommendation
        One of ``PROCEED`` / ``STAND_DOWN_DUPLICATE`` / ``WAIT_AND_REASSESS``.
        Callers should translate to their exit-code or branching convention
        (CLI exit codes 0 / 8 / 9 mirror the sister Catalog #340 helper).
    sister_commits : tuple[tuple[str, str, str, str], ...]
        For each sister commit detected in the lookback window, a tuple of
        ``(short_sha, iso_timestamp, author, message_subject)``. Sorted by
        timestamp descending (most-recent first).
    file_to_sister_commits : tuple[tuple[str, tuple[str, ...]], ...]
        For each target file with at least one sister-commit hit, a tuple
        of ``(file_path, tuple_of_short_shas)``. Useful for granular
        reasoning about which files have sister activity vs which don't.
    rationale : str
        Human-readable diagnostic explaining the verdict (printed to stderr
        by the CLI wrapper).
    lookback_hours : int
        The window size used (echoes the input for downstream consumers).
    target_files : tuple[str, ...]
        The files that were checked (sorted for stable output).
    """

    recommendation: Recommendation
    sister_commits: tuple[tuple[str, str, str, str], ...]
    file_to_sister_commits: tuple[tuple[str, tuple[str, ...]], ...]
    rationale: str
    lookback_hours: int
    target_files: tuple[str, ...]

    def has_sister_activity(self) -> bool:
        """True iff at least one sister commit was detected in the window."""
        return bool(self.sister_commits)


# ── Internal helpers ─────────────────────────────────────────────────────
def _parse_files(files: Sequence[str | Path]) -> list[str]:
    """Normalize ``files`` to a sorted list of stripped string paths.

    Accepts str OR Path; rejects empty / whitespace-only strings.
    """
    out: list[str] = []
    for f in files:
        if isinstance(f, Path):
            s = str(f)
        elif isinstance(f, str):
            s = f
        else:
            raise TypeError(
                f"files must be Sequence[str | Path]; got element of type "
                f"{type(f).__name__}: {f!r}"
            )
        s = s.strip()
        if not s:
            continue
        out.append(s)
    return sorted(set(out))


def _git_log_for_file(
    file_path: str,
    *,
    since_iso: str,
    repo_root: Path,
) -> list[tuple[str, str, str, str]]:
    """Run ``git log --since=<iso> -- <file_path>`` and parse rows.

    Returns list of ``(short_sha, iso_timestamp, author, message_subject)``
    tuples. Empty list if no commits in the window OR file is untracked.

    Raises
    ------
    ValueError
        If ``git log`` exits non-zero (e.g. not a git repo). The error
        message includes the git stderr for diagnosis.
    """
    # Use a sentinel separator unlikely to appear in commit subjects.
    sep = "\x1f"  # ASCII Unit Separator
    fmt = f"%h{sep}%aI{sep}%an{sep}%s"
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                f"--since={since_iso}",
                f"--pretty=format:{fmt}",
                "--",
                file_path,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"git log timed out after 30s for file {file_path!r}: {exc!s}"
        ) from exc
    except FileNotFoundError as exc:
        raise ValueError(
            f"git executable not found; helper requires git on PATH: {exc!s}"
        ) from exc

    if result.returncode != 0:
        raise ValueError(
            f"git log failed (rc={result.returncode}) for file "
            f"{file_path!r}: {result.stderr.strip()}"
        )

    rows: list[tuple[str, str, str, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(sep, 3)
        if len(parts) != 4:
            # Malformed line; skip rather than crash. Real git output should
            # always produce exactly 4 fields with our format string.
            continue
        short_sha, ts, author, subject = parts
        rows.append((short_sha.strip(), ts.strip(), author.strip(), subject.strip()))
    return rows


def _filter_own_commits(
    rows: list[tuple[str, str, str, str]],
    *,
    own_subagent_id: str | None,
    repo_root: Path,
) -> list[tuple[str, str, str, str]]:
    """Filter out commits authored by ``own_subagent_id`` via Co-Authored-By.

    Catalog #119 ``check_subagent_commits_have_co_author_trailer`` requires
    every subagent commit to carry the canonical Co-Authored-By trailer. If
    ``own_subagent_id`` is provided, fetch each commit's full body and skip
    rows whose body mentions the subagent_id (typical pattern: the body
    references the lane_id or subagent label).

    Conservative filter: if we cannot fetch a commit body, KEEP the row
    (better false-positive than false-negative — the caller will then
    investigate that commit and either ratify or override).
    """
    if not own_subagent_id:
        return rows
    filtered: list[tuple[str, str, str, str]] = []
    for row in rows:
        sha = row[0]
        try:
            body_result = subprocess.run(
                ["git", "show", "--no-patch", "--format=%B", sha],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # Fail-open conservative: keep the row.
            filtered.append(row)
            continue
        if body_result.returncode != 0:
            filtered.append(row)
            continue
        body = body_result.stdout
        # Catalog #119 Co-Authored-By trailer doesn't contain subagent_id
        # directly; we filter on subagent_id appearing in the body's free-
        # text (subject or rationale section).
        if own_subagent_id in body:
            continue
        filtered.append(row)
    return filtered


# ── Public API ───────────────────────────────────────────────────────────
def check_sister_files_recently_landed(
    files: Sequence[str | Path],
    *,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    own_subagent_id: str | None = None,
    repo_root: str | Path | None = None,
    now_utc: _dt.datetime | None = None,
) -> SisterRecentlyLandedVerdict:
    """Check ``files`` against recently-landed sister commits via git log.

    Returns a ``SisterRecentlyLandedVerdict`` whose ``recommendation`` is one of:

    - ``PROCEED`` — no sister commit touched any of ``files`` within the
      ``lookback_hours`` window. Caller may proceed with Writes.
    - ``STAND_DOWN_DUPLICATE`` — at least one sister commit touched at
      least ``STAND_DOWN_FILE_OVERLAP_THRESHOLD`` (=1) target file within
      the window. Caller should stand down + clean up rather than duplicate
      the sister's work. The NERV-FAMILY-L0-BUILD empirical anchor anchors
      this verdict.
    - ``WAIT_AND_REASSESS`` — sister commits touched SOME of the target
      files but not enough to trigger STAND_DOWN. Caller should re-read
      sister landing memos first + reconsider scope (the sister's work
      may have superseded part of the caller's intended scope).

    Parameters
    ----------
    files
        Files the caller intends to Write. Accepts ``Sequence[str | Path]``.
    lookback_hours
        Window for "recent sister commit" classification. Commits older
        than this many hours are ignored. Default 6 hours per the
        NERV-FAMILY-L0-BUILD anchor (sister landed ~4.5h before duplicate
        dispatch).
    own_subagent_id
        The caller's own ``subagent_id``. If provided, commits whose body
        mentions this id are filtered out (so the caller does not flag
        its own checkpointed work). If ``None``, no own-commit filtering
        happens — every sister commit in the window counts.
    repo_root
        Override the canonical repo-root path (useful for testing). If
        ``None``, uses ``REPO_ROOT`` (this module's three-levels-up parent).
    now_utc
        Override the current UTC time (useful for testing). If ``None``,
        uses ``datetime.now(timezone.utc)``.

    Raises
    ------
    TypeError
        If ``files`` is not a ``Sequence[str | Path]``.
    ValueError
        If ``git log`` fails (e.g. not a git repo, git binary missing).
        Per Catalog #229 PV discipline + fail-closed pattern: missing git
        surfaces as an exception so the caller knows to investigate rather
        than silently proceeding.
    """
    if not isinstance(files, (list, tuple)) and not hasattr(files, "__iter__"):
        raise TypeError(
            f"files must be Sequence[str | Path]; got {type(files).__name__}"
        )

    normalized = _parse_files(files)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    now = now_utc if now_utc is not None else _dt.datetime.now(_dt.timezone.utc)

    if not normalized:
        return SisterRecentlyLandedVerdict(
            recommendation="PROCEED",
            sister_commits=(),
            file_to_sister_commits=(),
            rationale=(
                "PROCEED: caller declared no target files; sister-activity "
                "scan skipped (nothing to check)."
            ),
            lookback_hours=lookback_hours,
            target_files=(),
        )

    since_dt = now - _dt.timedelta(hours=lookback_hours)
    since_iso = since_dt.isoformat()

    # Per-file git log; dedupe shas across files for the aggregated list.
    file_to_shas: dict[str, list[str]] = {}
    all_rows_by_sha: dict[str, tuple[str, str, str, str]] = {}
    for f in normalized:
        rows = _git_log_for_file(f, since_iso=since_iso, repo_root=root)
        rows = _filter_own_commits(rows, own_subagent_id=own_subagent_id, repo_root=root)
        if rows:
            file_to_shas[f] = [r[0] for r in rows]
            for r in rows:
                all_rows_by_sha[r[0]] = r

    # Sort sister_commits by timestamp descending (most-recent first).
    sister_commits = tuple(
        sorted(all_rows_by_sha.values(), key=lambda r: r[1], reverse=True)
    )
    # Build file_to_sister_commits as sorted-by-file mapping.
    file_to_sister_commits = tuple(
        (f, tuple(sorted(set(shas)))) for f, shas in sorted(file_to_shas.items())
    )

    overlap_count = len(file_to_shas)
    if overlap_count == 0:
        return SisterRecentlyLandedVerdict(
            recommendation="PROCEED",
            sister_commits=(),
            file_to_sister_commits=(),
            rationale=(
                f"PROCEED: no sister commits touched any of "
                f"{len(normalized)} target file(s) within the "
                f"{lookback_hours}-hour lookback window. Safe to write."
            ),
            lookback_hours=lookback_hours,
            target_files=tuple(normalized),
        )

    # Classify: STAND_DOWN if overlap >= threshold; WAIT_AND_REASSESS
    # otherwise. Per the empirical anchor, ANY single-file overlap is
    # enough signal to stand down (NERV-FAMILY-L0-BUILD lost 30 min on
    # 3-file overlap; even 1-file overlap should pause).
    if overlap_count >= STAND_DOWN_FILE_OVERLAP_THRESHOLD:
        # Heuristic for WAIT_AND_REASSESS vs STAND_DOWN_DUPLICATE: if a
        # SINGLE sister commit accounts for ALL overlapping files AND
        # overlapping files == ALL target files, the sister landed
        # equivalent work → STAND_DOWN_DUPLICATE. Otherwise: sister
        # touched some-but-not-all of our targets → WAIT_AND_REASSESS.
        all_target_overlap = (overlap_count == len(normalized))
        unique_shas = {sha for shas in file_to_shas.values() for sha in shas}
        single_sister_landed_everything = (
            all_target_overlap and len(unique_shas) == 1
        )
        # Also STAND_DOWN if a single sister commit touched the majority
        # of target files (>= 50%) — that's the NERV-FAMILY-L0-BUILD anchor
        # pattern (one sister commit touched all 3 of 3 target files).
        sister_to_files: dict[str, set[str]] = {}
        for f, shas in file_to_shas.items():
            for sha in shas:
                sister_to_files.setdefault(sha, set()).add(f)
        max_sister_overlap = max(
            (len(s) for s in sister_to_files.values()), default=0
        )
        majority_overlap_by_single_sister = (
            len(normalized) > 0
            and max_sister_overlap >= max(1, len(normalized) // 2 + (len(normalized) % 2))
        )

        if single_sister_landed_everything or majority_overlap_by_single_sister:
            recommendation: Recommendation = "STAND_DOWN_DUPLICATE"
            verdict_label = "STAND_DOWN_DUPLICATE"
            action_phrase = (
                "Recommendation: STAND DOWN. Sister already landed equivalent "
                "work. Per CLAUDE.md \"Subagent coherence-by-default\" "
                "non-negotiable + Catalog #229 PV discipline: do not "
                "duplicate the sister's work. Clean up any pending Writes + "
                "stand down via TaskStop. Optionally read the sister landing "
                "memo to understand what was actually delivered."
            )
        else:
            recommendation = "WAIT_AND_REASSESS"
            verdict_label = "WAIT_AND_REASSESS"
            action_phrase = (
                "Recommendation: re-read sister landing memos first + "
                "reconsider scope. Sister activity is ambiguous (touched "
                "some but not all of your target files; or multiple sisters "
                "involved). The sister may have superseded part of your "
                "intended scope without fully duplicating it."
            )
    else:
        # Unreachable given STAND_DOWN_FILE_OVERLAP_THRESHOLD = 1, but
        # defensive in case the threshold is configured higher in future.
        recommendation = "PROCEED"
        verdict_label = "PROCEED"
        action_phrase = "Recommendation: proceed with writes."

    # Build human-readable rationale.
    sister_lines = []
    for sha, ts, author, subj in sister_commits[:10]:
        # Truncate subject for readability.
        subj_short = subj if len(subj) <= 80 else subj[:77] + "..."
        sister_lines.append(f"  {sha} @ {ts} ({author}): {subj_short}")
    if len(sister_commits) > 10:
        sister_lines.append(f"  ... (+{len(sister_commits) - 10} more)")

    file_lines = []
    for f, shas in file_to_sister_commits:
        shas_str = ", ".join(shas)
        file_lines.append(f"  {f} ← {{ {shas_str} }}")

    rationale = (
        f"{verdict_label}: {len(sister_commits)} sister commit(s) touched "
        f"{overlap_count} of {len(normalized)} target file(s) within "
        f"{lookback_hours}-hour lookback window.\n"
        f"\nSister commits (most-recent first):\n"
        + "\n".join(sister_lines)
        + "\n\nPer-file overlap:\n"
        + "\n".join(file_lines)
        + "\n\n"
        + action_phrase
    )

    return SisterRecentlyLandedVerdict(
        recommendation=recommendation,
        sister_commits=sister_commits,
        file_to_sister_commits=file_to_sister_commits,
        rationale=rationale,
        lookback_hours=lookback_hours,
        target_files=tuple(normalized),
    )
