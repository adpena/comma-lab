# SPDX-License-Identifier: MIT
"""Main-thread spawn-decision PV guard — sister of Catalog #376 at the PARENT-side surface.

Per Wave N+25 OPERATOR-CRITIQUE-DRIVEN AUDIT memo
``.omx/research/operator_critique_existing_work_audit_20260528T222243Z.md``
op-routable #1 + #2 + #6 (canonical 2-landing pattern). Catalog #376
covers the SUBAGENT-side first-checkpoint PV evidence (the subagent's
OWN ``subagent_progress.jsonl`` row carries PV tokens); this helper +
gate (Catalog #378) cover the PARENT-side spawn-decision PV BEFORE the
``Agent`` tool call returns. Together they extinct the cascade-of-
STAND_DOWNs bug class structurally at TWO orthogonal surfaces.

Empirical anchor (Wave N+25 audit memo §"empirical_falsifications"):
today (2026-05-28) alone produced 4 STAND_DOWN incidents — PR111 Slot 4
RESUME / Z4 Wave N+23 / Cascade A FEC10 Wave N+24 / paper review
Wave N+13.5 — ALL detected at STAGING via Catalog #340 sister-checkpoint
guard rather than at SPAWN via the canonical 4-layer PV check.

This helper implements the canonical 4-layer falling-rule PV decision:

1. **HEAD-state commit grep** — does the declared scope appear in recent
   commit subjects / bodies / paths? If so, the work is likely already
   landed at HEAD. Recommendation: ``ABORT_DUPLICATE_WORK_ON_DISK``.

2. **Existing-file Glob scan** — does the declared scope already exist
   as committed files on disk? E.g. proposing a "NEW substrate" whose
   ``src/tac/substrates/<id>/`` directory already exists. Recommendation:
   ``ABORT_DUPLICATE_WORK_ON_DISK``.

3. **Canonical equation registry query** — does the declared scope
   match an existing registered canonical equation (Catalog #344)? If
   so, the apparatus already formalized the same idea; pivot scope to
   COMPLEMENTARY-EXTEND or DEFER. Recommendation: ``ABORT_DUPLICATE_WORK_ON_DISK``.

4. **Sister-subagent checkpoint scan** — sister subagent's in-flight
   checkpoint declares overlapping files. The work is being done RIGHT
   NOW; spawning a parallel sister is the canonical Variant 1
   STAND_DOWN pattern. Recommendation: ``ABORT_SISTER_IN_FLIGHT`` OR
   ``WAIT_AND_RETRY`` (depending on sister's checkpoint age).

Per CLAUDE.md "Beauty, simplicity, and developer experience": narrow
typed API; frozen dataclasses; explicit invariants in ``__post_init__``;
no hidden state.

Per Catalog #341 canonical-routing-markers: this helper is observability-
only — it does NOT block ``Agent.spawn(...)``; the STRICT preflight gate
``check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state``
(Catalog #378) enforces structural enforcement that the canonical helper
is invoked at the SOURCE level.

Sister of:
    Catalog #229 (premise-verification-before-edit) — design-memo surface
    Catalog #340 (sister-checkpoint guard at STAGING) — STAGING-time surface
    Catalog #314 (bare-commit absorbs in-flight files) — POST-COMMIT surface
    Catalog #376 (subagent-side first-checkpoint PV evidence) — SUBAGENT-side SPAWN-time surface
    Catalog #378 (THIS gate's strict preflight) — PARENT-side SPAWN-time surface
    Catalog #344 (canonical equations registry) — formalization surface (Layer 3 source)
    Catalog #335 (cathedral consumer canonical contract) — auto-discovery downstream
    Catalog #287 (placeholder-rationale rejection) — waiver-discipline sister

Lane: lane_main_thread_spawn_pv_gap_extinction_20260528
Memory: feedback_canonical_2_landing_main_thread_spawn_pv_gap_extinction_landed_20260528.md
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import re
import subprocess
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[3]
"""Repo root computed at import."""

CHECKPOINT_JSONL_PATH = REPO_ROOT / ".omx" / "state" / "subagent_progress.jsonl"
"""Sister-subagent checkpoint store per Catalog #131 + #138 + #206."""

CANONICAL_EQUATIONS_REGISTRY_PATH = REPO_ROOT / ".omx" / "state" / "canonical_equations_registry.jsonl"
"""Canonical equations registry per Catalog #344."""

DEFAULT_LOOKBACK_MINUTES = 60
"""Mirrors Catalog #302 / #314 / #340 / #376 60-minute lookback so cross-
surface alerts agree on the same time horizon."""

DEFAULT_RECENT_COMMIT_LIMIT = 30
"""Number of recent commits to scan in Layer 1 (HEAD-state grep). Mirrors
CLAUDE.md ``git log --oneline -30`` canonical pattern."""

WAIT_AND_RETRY_THRESHOLD_MINUTES = 10
"""If a sister subagent's last checkpoint is more recent than this, the
helper recommends ``WAIT_AND_RETRY`` (sister is actively working).
Otherwise recommends ``ABORT_SISTER_IN_FLIGHT`` (sister may be stalled
and the spawn should defer + coordinate via Catalog #230 ownership map)."""


MainThreadSpawnRecommendation = Literal[
    "PROCEED",
    "ABORT_DUPLICATE_WORK_ON_DISK",
    "ABORT_SISTER_IN_FLIGHT",
    "WAIT_AND_RETRY",
]


@dataclasses.dataclass(frozen=True)
class MainThreadSpawnGuardVerdict:
    """Typed verdict returned by ``verify_head_state_before_main_thread_spawn``.

    Attributes
    ----------
    recommendation : MainThreadSpawnRecommendation
        One of ``PROCEED`` / ``ABORT_DUPLICATE_WORK_ON_DISK``
        / ``ABORT_SISTER_IN_FLIGHT`` / ``WAIT_AND_RETRY``.
    conflict_source : str
        ``none`` | ``head_recent_commit`` | ``existing_file`` |
        ``canonical_equation`` | ``sister_checkpoint``. Tells downstream
        consumers which falling-rule layer fired.
    overlapping_scope : tuple[str, ...]
        Paths from ``declared_scope`` that overlap the conflict source.
        Empty if ``recommendation == "PROCEED"``.
    diagnostic : str
        Human-readable diagnostic citing the conflict source + concrete
        evidence (commit shas / file paths / equation ids / sister
        subagent ids).
    cited_commits : tuple[str, ...]
        Layer 1 evidence — commit sha prefixes touching declared scope.
    cited_existing_paths : tuple[str, ...]
        Layer 2 evidence — committed paths matching declared scope.
    cited_equation_ids : tuple[str, ...]
        Layer 3 evidence — canonical equation ids matching declared scope.
    cited_sister_subagent_ids : tuple[str, ...]
        Layer 4 evidence — in-flight sister subagent ids with overlapping
        files_touched.
    """

    recommendation: MainThreadSpawnRecommendation
    conflict_source: str
    overlapping_scope: tuple[str, ...]
    diagnostic: str
    cited_commits: tuple[str, ...] = ()
    cited_existing_paths: tuple[str, ...] = ()
    cited_equation_ids: tuple[str, ...] = ()
    cited_sister_subagent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        valid_recs = {
            "PROCEED",
            "ABORT_DUPLICATE_WORK_ON_DISK",
            "ABORT_SISTER_IN_FLIGHT",
            "WAIT_AND_RETRY",
        }
        if self.recommendation not in valid_recs:
            raise ValueError(
                f"recommendation must be one of {valid_recs!r}, "
                f"got {self.recommendation!r}"
            )
        valid_sources = {
            "none",
            "head_recent_commit",
            "existing_file",
            "canonical_equation",
            "sister_checkpoint",
        }
        if self.conflict_source not in valid_sources:
            raise ValueError(
                f"conflict_source must be one of {valid_sources!r}, "
                f"got {self.conflict_source!r}"
            )
        for field_name in (
            "overlapping_scope",
            "cited_commits",
            "cited_existing_paths",
            "cited_equation_ids",
            "cited_sister_subagent_ids",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise TypeError(f"{field_name} must be a tuple (frozen)")
            for i, item in enumerate(value):
                if not isinstance(item, str):
                    raise TypeError(f"{field_name}[{i}] must be a str")
        if not isinstance(self.diagnostic, str):
            raise TypeError("diagnostic must be a str")

    @property
    def is_proceed(self) -> bool:
        """True iff recommendation is PROCEED."""
        return self.recommendation == "PROCEED"

    @property
    def is_abort(self) -> bool:
        """True iff recommendation is any ABORT variant (includes WAIT_AND_RETRY)."""
        return self.recommendation != "PROCEED"


# ── Helper utilities ────────────────────────────────────────────────────


def _parse_iso_utc(ts: str | None) -> _dt.datetime | None:
    """Parse ISO-8601 UTC timestamp, tolerating trailing Z."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return _dt.datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _normalize_scope_token(token: str) -> str:
    """Normalize a declared_scope entry for substring matching.

    Strips leading/trailing whitespace + trailing slash. Used in all
    4 layers' overlap detection."""
    return token.strip().rstrip("/")


def _path_overlaps(declared: str, candidate: str) -> bool:
    """True iff candidate path overlaps the declared scope token.

    Sister of subagent_spawn_head_pv_guard._path_overlaps. Two paths
    overlap if either is a path-prefix of the other (matching at the
    path-component boundary)."""
    d = _normalize_scope_token(declared)
    c = candidate.strip().rstrip("/")
    if not d or not c:
        return False
    if d == c:
        return True
    if c.startswith(d + "/"):
        return True
    if d.startswith(c + "/"):
        return True
    return False


# ── Layer 1: HEAD-state commit grep ─────────────────────────────────────


def _layer1_head_state_commit_overlap(
    declared_scope: list[str],
    *,
    repo_root: Path,
    recent_commit_limit: int,
) -> list[tuple[str, str, str]]:
    """Return list of (sha, declared_scope_token, commit_path_or_subject)
    where a recent commit touched a path that overlaps the declared scope.

    Per CLAUDE.md "Subagent coherence-by-default" Mandatory pre-flight:
    every subagent reads the canonical ``git log --oneline -30`` history
    before starting work. THIS helper layer is the structural enforcement.

    The helper scans both commit-touched paths (via ``git log --name-only``)
    AND commit subject lines (so a declared_scope token like
    "Cascade A FEC10" matches a recent commit "Cascade A FEC10 STAND_DOWN
    audit per Catalog #340"). The subject scan is the substring sister of
    the path scan."""
    overlaps: list[tuple[str, str, str]] = []
    try:
        # Combined log: subject + touched paths, separated by NUL.
        cp = subprocess.run(
            ["git", "log", f"-{recent_commit_limit}", "--name-only", "--format=%H%n%s"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
        if cp.returncode != 0:
            return overlaps
        lines = cp.stdout.split("\n")
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        return overlaps

    # Walk: sha line, subject line, path lines until blank, repeat.
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or len(line) < 40 or not all(c in "0123456789abcdef" for c in line):
            # Not a sha line; skip.
            continue
        sha = line
        # Next line is the subject.
        if i >= len(lines):
            break
        subject = lines[i].strip()
        i += 1
        # Path lines until blank.
        commit_paths: list[str] = []
        while i < len(lines) and lines[i].strip():
            commit_paths.append(lines[i].strip())
            i += 1
        # Check subject overlap.
        subject_folded = subject.casefold()
        for declared in declared_scope:
            d = _normalize_scope_token(declared)
            if not d:
                continue
            if d.casefold() in subject_folded:
                overlaps.append((sha, declared, f"<subject>: {subject}"))
                break  # one match per (sha, declared) is enough
        # Check path overlap.
        for declared in declared_scope:
            for commit_path in commit_paths:
                if _path_overlaps(declared, commit_path):
                    overlaps.append((sha, declared, commit_path))
                    break
    return overlaps


# ── Layer 2: Existing-file Glob scan ───────────────────────────────────


def _layer2_existing_file_overlap(
    declared_scope: list[str],
    *,
    repo_root: Path,
) -> list[tuple[str, str]]:
    """Return list of (declared_scope_token, existing_path) where the
    declared scope already exists as a committed file/dir on disk.

    Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline"
    + sister anti-patterns: proposing a "NEW substrate" whose
    ``src/tac/substrates/<id>/`` directory already exists is the
    canonical DUPLICATE_HEAD_STATE pattern. THIS helper layer extincts
    that class structurally at the file-system surface.

    The helper uses ``git ls-files`` to scope the scan to TRACKED files
    only — untracked working-tree noise (e.g. ``.omx/state/`` JSONL append
    rows) is excluded by design."""
    overlaps: list[tuple[str, str]] = []
    if not declared_scope:
        return overlaps
    try:
        cp = subprocess.run(
            ["git", "ls-files"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
        if cp.returncode != 0:
            return overlaps
        tracked = [line.strip() for line in cp.stdout.split("\n") if line.strip()]
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        return overlaps

    for declared in declared_scope:
        d = _normalize_scope_token(declared)
        if not d:
            continue
        for tracked_path in tracked:
            if _path_overlaps(declared, tracked_path):
                overlaps.append((declared, tracked_path))
                break  # one match per declared is enough for overlap signal
    return overlaps


# ── Layer 3: Canonical equation registry query ─────────────────────────


def _layer3_canonical_equation_overlap(
    declared_scope: list[str],
    *,
    registry_path: Path,
) -> list[tuple[str, str]]:
    """Return list of (declared_scope_token, equation_id) where the
    declared scope matches an existing registered canonical equation.

    Per Catalog #344: the canonical equations registry is the canonical
    apparatus surface for formalized findings. Proposing a NEW substrate
    / scaffold whose ``equation_id`` or ``one_line_summary`` already
    exists is the canonical DUPLICATE_HEAD_STATE pattern at the
    formalization layer.

    The helper queries the registry's APPEND-ONLY JSONL ledger (per
    Catalog #131 / #138) and matches declared_scope tokens against the
    latest payload of each equation (latest-wins per Catalog #344
    ``with_new_anchor`` semantics)."""
    overlaps: list[tuple[str, str]] = []
    if not declared_scope or not registry_path.exists():
        return overlaps
    # Build latest-payload-per-equation_id index.
    latest_payloads: dict[str, dict] = {}
    try:
        with open(registry_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                eq_id = rec.get("equation_id")
                if not isinstance(eq_id, str) or not eq_id:
                    continue
                # latest-wins
                latest_payloads[eq_id] = rec
    except OSError:
        return overlaps

    for eq_id, payload in latest_payloads.items():
        # Search fields: equation_id + name + one_line_summary.
        searchable_parts: list[str] = [eq_id]
        for field in ("name", "one_line_summary"):
            v = payload.get(field)
            if isinstance(v, str):
                searchable_parts.append(v)
        searchable = "\n".join(searchable_parts).casefold()
        for declared in declared_scope:
            d = _normalize_scope_token(declared)
            if not d:
                continue
            if d.casefold() in searchable:
                overlaps.append((declared, eq_id))
                break  # one match per equation is enough
    return overlaps


# ── Layer 4: Sister-subagent checkpoint scan ───────────────────────────


def _layer4_sister_checkpoint_overlap(
    declared_scope: list[str],
    *,
    checkpoint_path: Path,
    lookback_minutes: int,
    now_utc: _dt.datetime,
    current_subagent_id: str | None = None,
) -> list[tuple[str, _dt.datetime, tuple[str, ...]]]:
    """Return list of (sister_subagent_id, last_checkpoint_utc,
    overlapping_files) for in-flight sister subagents whose
    files_touched overlap the declared scope.

    Mirrors subagent_spawn_head_pv_guard._load_sister_subagent_checkpoints
    + Catalog #340 sister-checkpoint scan semantics. Excludes the caller's
    own subagent id so a parent agent calling THIS helper from an active
    subagent context doesn't flag itself."""
    if not checkpoint_path.exists():
        return []
    cutoff = now_utc - _dt.timedelta(minutes=lookback_minutes)
    # Build latest-checkpoint-per-subagent index.
    latest_by_subagent: dict[str, tuple[_dt.datetime, dict]] = {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                sid = rec.get("subagent_id")
                if not isinstance(sid, str) or not sid:
                    continue
                ts = _parse_iso_utc(rec.get("written_at_utc"))
                if ts is None:
                    continue
                prior = latest_by_subagent.get(sid)
                if prior is None or ts > prior[0]:
                    latest_by_subagent[sid] = (ts, rec)
    except OSError:
        return []

    overlaps: list[tuple[str, _dt.datetime, tuple[str, ...]]] = []
    for sid, (ts, rec) in latest_by_subagent.items():
        if current_subagent_id is not None and sid == current_subagent_id:
            continue
        if ts < cutoff:
            continue
        if rec.get("status") != "in_progress":
            continue
        files_touched = rec.get("files_touched", [])
        if not isinstance(files_touched, list):
            continue
        files_clean = [f for f in files_touched if isinstance(f, str)]
        overlapping = set()
        for sister_path in files_clean:
            for declared in declared_scope:
                if _path_overlaps(declared, sister_path):
                    overlapping.add(sister_path)
                    break
        if overlapping:
            overlaps.append((sid, ts, tuple(sorted(overlapping))))
    overlaps.sort(key=lambda x: x[0])
    return overlaps


# ── Public API ──────────────────────────────────────────────────────────


def verify_head_state_before_main_thread_spawn(
    declared_scope: list[str],
    *,
    current_subagent_id: str | None = None,
    lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
    recent_commit_limit: int = DEFAULT_RECENT_COMMIT_LIMIT,
    wait_and_retry_threshold_minutes: int = WAIT_AND_RETRY_THRESHOLD_MINUTES,
    repo_root: Path | None = None,
    checkpoint_path: Path | None = None,
    canonical_equations_registry_path: Path | None = None,
    now_utc: _dt.datetime | None = None,
) -> MainThreadSpawnGuardVerdict:
    """Canonical 4-layer PV decision for main-thread parent-agent Agent-spawn.

    Returns a :class:`MainThreadSpawnGuardVerdict` whose ``recommendation``
    is one of:

    * ``PROCEED`` — no conflict across any of the 4 layers; spawn is safe.
    * ``ABORT_DUPLICATE_WORK_ON_DISK`` — Layer 1/2/3 detected duplicate
      work at HEAD / on-disk / in canonical equations. Pivot scope to
      COMPLEMENTARY-EXTEND or DEFER.
    * ``ABORT_SISTER_IN_FLIGHT`` — Layer 4 detected sister subagent with
      overlapping files_touched, and sister's last checkpoint is OLDER
      than ``wait_and_retry_threshold_minutes`` ago (sister may be
      stalled; defer + coordinate via Catalog #230 ownership map).
    * ``WAIT_AND_RETRY`` — Layer 4 detected sister subagent with
      overlapping files_touched, and sister's last checkpoint is MORE
      recent than the threshold (sister is actively working; defer
      spawn until sister completes).

    Per CLAUDE.md "Beauty, simplicity, and developer experience": narrow
    typed API; falling-rule decision; no hidden state.

    Parameters
    ----------
    declared_scope
        Repo-relative path strings / scope tokens the proposed Agent
        spawn intends to touch. Must be a list (not a string — coercing
        a string to ``list('foo.py')`` would silently produce a list of
        characters).
    current_subagent_id
        Optional subagent id of the caller, used to exclude the caller's
        own checkpoint from Layer 4.
    lookback_minutes
        Window for "recent" classification in Layers 1 + 4. Default 60.
    recent_commit_limit
        Number of recent commits to scan in Layer 1. Default 30.
    wait_and_retry_threshold_minutes
        Threshold for distinguishing ``WAIT_AND_RETRY`` vs
        ``ABORT_SISTER_IN_FLIGHT`` in Layer 4. Sister checkpoints more
        recent than this threshold yield ``WAIT_AND_RETRY``. Default 10.
    repo_root
        Override the repo root (useful for testing).
    checkpoint_path
        Override the canonical subagent_progress.jsonl path.
    canonical_equations_registry_path
        Override the canonical canonical_equations_registry.jsonl path.
    now_utc
        Override the current UTC time (useful for testing).

    Raises
    ------
    TypeError
        If ``declared_scope`` is not a list of strings.
    """
    if not isinstance(declared_scope, list) or not all(
        isinstance(p, str) for p in declared_scope
    ):
        raise TypeError("declared_scope must be a list of strings")

    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    checkpoint_path_resolved = (
        Path(checkpoint_path) if checkpoint_path is not None else CHECKPOINT_JSONL_PATH
    )
    registry_path_resolved = (
        Path(canonical_equations_registry_path)
        if canonical_equations_registry_path is not None
        else CANONICAL_EQUATIONS_REGISTRY_PATH
    )
    now = now_utc if now_utc is not None else _dt.datetime.now(_dt.timezone.utc)

    # Normalize declared_scope to non-empty tokens.
    declared_clean = [s for s in (_normalize_scope_token(p) for p in declared_scope) if s]
    if not declared_clean:
        return MainThreadSpawnGuardVerdict(
            recommendation="PROCEED",
            conflict_source="none",
            overlapping_scope=(),
            diagnostic=(
                "PROCEED: declared_scope is empty; no overlap analysis "
                "possible. Spawn approved without PV scan (per Catalog "
                "#229 PV is operator-routable when declared_scope is "
                "unknown)."
            ),
        )

    # ── Layer 4: sister-subagent checkpoint scan (HIGHEST PRIORITY) ────
    # Sister-in-flight detection runs FIRST because it has the most
    # actionable disposition (WAIT_AND_RETRY vs ABORT_SISTER_IN_FLIGHT
    # vs PROCEED). If a sister is actively touching the same files,
    # spawning a parallel sister is the canonical Variant 1 STAND_DOWN
    # pattern; sibling-other-layer signals are downstream.
    sister_overlaps = _layer4_sister_checkpoint_overlap(
        declared_clean,
        checkpoint_path=checkpoint_path_resolved,
        lookback_minutes=lookback_minutes,
        now_utc=now,
        current_subagent_id=current_subagent_id,
    )
    if sister_overlaps:
        wait_threshold = now - _dt.timedelta(minutes=wait_and_retry_threshold_minutes)
        # WAIT_AND_RETRY if MOST-RECENT sister checkpoint is within threshold.
        most_recent_ts = max(ts for _, ts, _ in sister_overlaps)
        rec_class = (
            "WAIT_AND_RETRY"
            if most_recent_ts >= wait_threshold
            else "ABORT_SISTER_IN_FLIGHT"
        )
        all_overlap = sorted(
            {p for _, _, paths in sister_overlaps for p in paths}
        )
        sister_ids = tuple(sorted(sid for sid, *_ in sister_overlaps))
        lines = []
        for sid, ts, paths in sister_overlaps[:5]:
            paths_repr = ", ".join(paths[:5])
            more = "" if len(paths) <= 5 else f" (+{len(paths) - 5} more)"
            age_min = (now - ts).total_seconds() / 60.0
            lines.append(
                f"    sister={sid!r} (age {age_min:.1f} min) overlaps: "
                f"{{ {paths_repr} }}{more}"
            )
        diagnostic = (
            f"{rec_class}: {len(sister_overlaps)} sister subagent(s) "
            f"declare overlapping files within {lookback_minutes}-min "
            f"lookback. Per Catalog #340 sister-checkpoint guard + "
            f"Catalog #230 ownership map: coordinate BEFORE spawn OR "
            f"stand down per Catalog #229 PV.\n" + "\n".join(lines)
        )
        return MainThreadSpawnGuardVerdict(
            recommendation=rec_class,
            conflict_source="sister_checkpoint",
            overlapping_scope=tuple(all_overlap),
            diagnostic=diagnostic,
            cited_sister_subagent_ids=sister_ids,
        )

    # ── Layer 1: HEAD-state commit grep ────────────────────────────────
    head_overlaps = _layer1_head_state_commit_overlap(
        declared_clean,
        repo_root=repo_root,
        recent_commit_limit=recent_commit_limit,
    )
    if head_overlaps:
        cited_commits = tuple(sorted({sha[:9] for sha, *_ in head_overlaps}))
        all_overlap = tuple(sorted({declared for _, declared, _ in head_overlaps}))
        lines = []
        for sha, declared, commit_path in head_overlaps[:5]:
            lines.append(
                f"    commit {sha[:9]} touched {commit_path!r} "
                f"(overlaps declared={declared!r})"
            )
        diagnostic = (
            f"ABORT_DUPLICATE_WORK_ON_DISK: {len(head_overlaps)} recent "
            f"commit(s) at HEAD touched paths/subjects overlapping "
            f"declared_scope within last {recent_commit_limit} commits. "
            f"Per Wave N+25 audit + Catalog #229 PV + anti-pattern "
            f"main_thread_subagent_spawn_without_catalog_376_verify_"
            f"head_state_before_spawn_pv_v1: read the commit bodies + "
            f"sister landing memos BEFORE spawn to avoid STAND_DOWN.\n"
            + "\n".join(lines)
        )
        return MainThreadSpawnGuardVerdict(
            recommendation="ABORT_DUPLICATE_WORK_ON_DISK",
            conflict_source="head_recent_commit",
            overlapping_scope=all_overlap,
            diagnostic=diagnostic,
            cited_commits=cited_commits,
        )

    # ── Layer 2: Existing-file Glob scan ──────────────────────────────
    existing_overlaps = _layer2_existing_file_overlap(
        declared_clean,
        repo_root=repo_root,
    )
    if existing_overlaps:
        cited_paths = tuple(sorted({path for _, path in existing_overlaps}))
        all_overlap = tuple(sorted({declared for declared, _ in existing_overlaps}))
        lines = []
        for declared, existing in existing_overlaps[:5]:
            lines.append(
                f"    tracked path {existing!r} (overlaps declared={declared!r})"
            )
        diagnostic = (
            f"ABORT_DUPLICATE_WORK_ON_DISK: {len(existing_overlaps)} "
            f"tracked path(s) on disk already overlap declared_scope. "
            f"Per Wave N+25 audit + anti-pattern "
            f"main_thread_subagent_spawn_without_catalog_376_verify_"
            f"head_state_before_spawn_pv_v1: the work is already on "
            f"disk; pivot scope to COMPLEMENTARY-EXTEND or DEFER.\n"
            + "\n".join(lines)
        )
        return MainThreadSpawnGuardVerdict(
            recommendation="ABORT_DUPLICATE_WORK_ON_DISK",
            conflict_source="existing_file",
            overlapping_scope=all_overlap,
            diagnostic=diagnostic,
            cited_existing_paths=cited_paths,
        )

    # ── Layer 3: Canonical equation registry query ────────────────────
    equation_overlaps = _layer3_canonical_equation_overlap(
        declared_clean,
        registry_path=registry_path_resolved,
    )
    if equation_overlaps:
        cited_eqs = tuple(sorted({eq_id for _, eq_id in equation_overlaps}))
        all_overlap = tuple(sorted({declared for declared, _ in equation_overlaps}))
        lines = []
        for declared, eq_id in equation_overlaps[:5]:
            lines.append(
                f"    canonical equation {eq_id!r} (overlaps declared={declared!r})"
            )
        diagnostic = (
            f"ABORT_DUPLICATE_WORK_ON_DISK: {len(equation_overlaps)} "
            f"canonical equation(s) already registered in "
            f"`.omx/state/canonical_equations_registry.jsonl` overlap "
            f"declared_scope. Per Catalog #344 + Wave N+25 audit + "
            f"anti-pattern "
            f"main_thread_subagent_spawn_without_catalog_376_verify_"
            f"head_state_before_spawn_pv_v1: the apparatus already "
            f"formalized the idea; pivot scope to COMPLEMENTARY-EXTEND "
            f"or DEFER.\n" + "\n".join(lines)
        )
        return MainThreadSpawnGuardVerdict(
            recommendation="ABORT_DUPLICATE_WORK_ON_DISK",
            conflict_source="canonical_equation",
            overlapping_scope=all_overlap,
            diagnostic=diagnostic,
            cited_equation_ids=cited_eqs,
        )

    # No conflict across any of the 4 layers.
    return MainThreadSpawnGuardVerdict(
        recommendation="PROCEED",
        conflict_source="none",
        overlapping_scope=(),
        diagnostic=(
            f"PROCEED: no overlap detected across all 4 PV layers: "
            f"(1) HEAD-state commit grep (last {recent_commit_limit}); "
            f"(2) existing-file Glob scan; "
            f"(3) canonical equation registry query; "
            f"(4) sister-subagent checkpoint scan (last {lookback_minutes} min). "
            f"Per CLAUDE.md Catalog #229 PV non-negotiable: spawn approved."
        ),
    )
