# SPDX-License-Identifier: MIT
"""SPAWN-time PV guard — sister of Catalog #229 at the agent-spawn surface.

Anti-pattern #13 (``subagent_spawn_without_head_state_premise_verification_v1``)
canonical_unwind_path implementation. Returns a typed
:class:`SpawnGuardVerdict` so the parent agent can decide PROCEED vs
STAND_DOWN BEFORE Agent.spawn(...) cost is paid.

The guard consults THREE evidence surfaces, in order, and applies a
falling-rule decision:

1. **HEAD-state recency** — was the declared scope recently landed at HEAD?
   If a commit within the lookback window touches a path that overlaps the
   ``declared_scope`` AND no superseding work has landed since, then the
   spawn is at risk of producing STAND_DOWN. Recommendation:
   ``DUPLICATE_HEAD_STATE``.

2. **Sister landing memos** — was the declared scope recently landed via a
   landing memo at ``.omx/research/*landed_<YYYYMMDD>.md`` within the
   lookback window? If so, the parent agent should at minimum READ the
   memo before deciding spawn vs STAND_DOWN. Recommendation:
   ``DUPLICATE_HEAD_STATE`` (with a memo path in the diagnostic).

3. **Sister-subagent checkpoints** — sister subagent's in-flight
   checkpoint declares overlapping files. Even if HEAD is clean, the
   sister is mid-edit and the spawn would collide. Recommendation:
   ``SISTER_IN_FLIGHT``. This is the sister of Catalog #340 STAGING-time
   guard's checkpoint scan at the SPAWN-time surface.

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: this helper is
observability-only at runtime. It does NOT block ``Agent.spawn(...)``;
the STRICT preflight gate ``check_subagent_spawn_includes_head_state_pv_evidence`` (Catalog #376)
enforces source-text discipline that requires preceding PV evidence in
``subagent_progress.jsonl`` spawn-event rows.

Per CLAUDE.md "Beauty, simplicity, and developer experience": narrow
typed API; frozen dataclasses; explicit invariants in ``__post_init__``;
no hidden state.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import re
import subprocess
from pathlib import Path
from typing import Literal


# ── Constants ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
"""Repo root: src/tac/discipline_anti_pattern_guards/ lives three levels under repo root."""

CHECKPOINT_JSONL_PATH = REPO_ROOT / ".omx" / "state" / "subagent_progress.jsonl"
"""Canonical crash-resume checkpoint store; managed by
``tools/subagent_checkpoint.py`` per CLAUDE.md "Mandatory crash-resume
protocol" + Catalog #131 + #138 + #206."""

RESEARCH_DIR = REPO_ROOT / ".omx" / "research"
"""Sister-landing-memo location per CLAUDE.md "Required durable state"."""

DEFAULT_SISTER_LANDING_LOOKBACK_MINUTES = 60
"""Mirrors Catalog #302 + #314 + #340 60-minute lookback so cross-surface
alerts agree on the same time horizon."""

DEFAULT_SISTER_LANDING_MEMO_GLOB = "*landed_*.md"
"""Canonical landing-memo filename pattern per CLAUDE.md "Mandatory wire-in
for every landing"."""

# Canonical tokens that constitute PV evidence in a subagent_progress.jsonl
# row's notes / next_action / files_touched.
HEAD_STATE_PV_TOKENS = frozenset({
    # git log + git status invocations
    "git log",
    "git status",
    "git rev-parse",
    "git show",
    # PV-discipline tokens
    "Catalog #229",
    "catalog #229",
    "premise verification",
    "premise_verification",
    "premise-verification",
    "PV complete",
    "PV verified",
    "PV passed",
    # Sister-landing-memo checks
    ".omx/research/",
    "sister landing memo",
    "sister-landing-memo",
    "predecessor commit",
    "predecessor_commit",
    # Canonical helper invocation
    "verify_head_state_before_spawn",
})


SpawnGuardRecommendation = Literal[
    "PROCEED",
    "DUPLICATE_HEAD_STATE",
    "SISTER_IN_FLIGHT",
]


# ── Typed verdict ────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class SpawnPvEvidenceContext:
    """Optional context for ``verify_head_state_before_spawn``.

    Per Catalog #229 PV non-negotiable: the parent agent SHOULD record the
    HEAD sha + git status summary it observed BEFORE the spawn so the
    verdict's diagnostic can cite the evidence.

    Attributes
    ----------
    observed_head_sha : str | None
        Result of ``git rev-parse HEAD`` AT spawn-decision time (NOT now;
        the spawn may take seconds, and HEAD may advance during it).
    observed_dirty_paths : tuple[str, ...]
        Result of ``git status --porcelain`` filtered to non-empty rows.
        Empty tuple = clean working tree.
    observed_recent_commits : tuple[str, ...]
        Subject lines from ``git log --oneline -30`` so the diagnostic
        can cite which commits were considered.
    consulted_landing_memo_paths : tuple[str, ...]
        Paths to ``.omx/research/*landed_*.md`` files the parent agent
        consulted BEFORE the spawn decision. Used for the "PV evidence
        positive" branch of the verdict.
    """

    observed_head_sha: str | None = None
    observed_dirty_paths: tuple[str, ...] = ()
    observed_recent_commits: tuple[str, ...] = ()
    consulted_landing_memo_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.observed_head_sha is not None and not isinstance(self.observed_head_sha, str):
            raise TypeError("observed_head_sha must be a str or None")
        for field_name in (
            "observed_dirty_paths",
            "observed_recent_commits",
            "consulted_landing_memo_paths",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise TypeError(f"{field_name} must be a tuple (frozen)")
            for i, item in enumerate(value):
                if not isinstance(item, str):
                    raise TypeError(f"{field_name}[{i}] must be a str")


@dataclasses.dataclass(frozen=True)
class SpawnGuardVerdict:
    """Typed verdict returned by ``verify_head_state_before_spawn``.

    Attributes
    ----------
    recommendation : SpawnGuardRecommendation
        ``PROCEED`` | ``DUPLICATE_HEAD_STATE`` | ``SISTER_IN_FLIGHT``
    overlapping_scope : tuple[str, ...]
        Paths from ``declared_scope`` that overlap the conflict source.
        Empty if ``recommendation == "PROCEED"``.
    diagnostic : str
        Human-readable diagnostic explaining the verdict.
    conflict_source : str
        ``head_recent_commit`` | ``sister_landing_memo`` | ``sister_checkpoint``
        | ``none``. Used by downstream consumers to route the verdict.
    consulted_evidence : SpawnPvEvidenceContext
        Echo of the evidence context the caller provided (so downstream
        consumers can cite what was checked).
    sister_subagent_ids : tuple[str, ...]
        IDs of every in-flight sister subagent whose checkpoint was
        considered (may include sisters with no file overlap; surfaced
        for diagnostics).
    recent_landing_memo_matches : tuple[str, ...]
        Paths to ``.omx/research/*landed_*.md`` files within the lookback
        window that mention any path in ``declared_scope``.
    """

    recommendation: SpawnGuardRecommendation
    overlapping_scope: tuple[str, ...]
    diagnostic: str
    conflict_source: str
    consulted_evidence: SpawnPvEvidenceContext
    sister_subagent_ids: tuple[str, ...]
    recent_landing_memo_matches: tuple[str, ...]

    def has_conflict(self) -> bool:
        """True iff the recommendation is not ``PROCEED``."""
        return self.recommendation != "PROCEED"


# ── Internal helpers ─────────────────────────────────────────────────────
def _parse_iso_utc(ts: str | None) -> _dt.datetime | None:
    """Parse an ISO-UTC timestamp; return aware datetime or None on failure."""
    if not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        d = _dt.datetime.fromisoformat(ts)
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d
    except (TypeError, ValueError):
        return None


def _normalize_files_touched(raw) -> list[str]:
    """Coerce a checkpoint row's ``files_touched`` field to a flat str list.

    Mirrors ``tac.commit_safety.sister_checkpoint_guard._normalize_files_touched``
    so the two sister surfaces agree on legacy-row handling.
    """
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for f in raw:
        if not isinstance(f, str):
            continue
        f = f.strip()
        if not f:
            continue
        if " " in f and not f.startswith("/"):
            out.extend(p.strip() for p in f.split() if p.strip())
        elif "," in f:
            out.extend(p.strip() for p in f.split(",") if p.strip())
        else:
            out.append(f)
    return out


def _path_overlaps(declared: str, candidate: str) -> bool:
    """True iff ``candidate`` overlaps ``declared`` (prefix-match either way).

    A path overlap is when one path is a prefix of the other (treating
    paths as directory hierarchies). E.g.:
      - ``src/tac/foo/`` overlaps ``src/tac/foo/bar.py``
      - ``src/tac/foo/bar.py`` overlaps ``src/tac/foo/``
      - ``src/tac/foo/`` does NOT overlap ``src/tac/bar/``
    """
    if not declared or not candidate:
        return False
    d = declared.rstrip("/")
    c = candidate.rstrip("/")
    if d == c:
        return True
    # Treat directory paths as having a trailing slash for prefix semantics.
    if c.startswith(d + "/") or d.startswith(c + "/"):
        return True
    return False


def _load_sister_subagent_checkpoints(
    jsonl_path: Path,
    *,
    lookback_minutes: int,
    now_utc: _dt.datetime,
) -> list[tuple[str, _dt.datetime, set[str]]]:
    """Load in-flight sister subagent checkpoints; mirrors Catalog #340 sister."""
    if not jsonl_path.exists():
        return []
    rows: list[dict] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    # Lenient mode: skip malformed lines (we are observability-
                    # only at runtime; the STRICT gate Catalog #376 enforces
                    # strict-load via fail-closed path).
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)
    except OSError:
        return []

    # Latest checkpoint per subagent.
    latest_per_subagent: dict[str, dict] = {}
    for row in rows:
        sid = row.get("subagent_id")
        if not isinstance(sid, str) or not sid:
            continue
        latest_per_subagent[sid] = row

    lookback_seconds = lookback_minutes * 60
    in_flight: list[tuple[str, _dt.datetime, set[str]]] = []
    for sid, row in latest_per_subagent.items():
        status = row.get("status")
        if status != "in_progress":
            continue
        ts = _parse_iso_utc(row.get("written_at_utc"))
        if ts is None:
            continue
        try:
            delta = (now_utc - ts).total_seconds()
        except (TypeError, ValueError):
            continue
        if delta < 0 or delta > lookback_seconds:
            continue
        declared = _normalize_files_touched(row.get("files_touched"))
        if not declared:
            continue
        in_flight.append((sid, ts, set(declared)))
    return in_flight


def _list_recent_landing_memos(
    research_dir: Path,
    *,
    lookback_minutes: int,
    now_utc: _dt.datetime,
    memo_glob: str = DEFAULT_SISTER_LANDING_MEMO_GLOB,
) -> list[Path]:
    """Return landing-memo paths whose mtime is within the lookback window."""
    if not research_dir.exists() or not research_dir.is_dir():
        return []
    lookback_seconds = lookback_minutes * 60
    out: list[Path] = []
    for path in research_dir.glob(memo_glob):
        try:
            stat = path.stat()
        except OSError:
            continue
        mtime = _dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.timezone.utc)
        try:
            delta = (now_utc - mtime).total_seconds()
        except (TypeError, ValueError):
            continue
        if delta < 0 or delta > lookback_seconds:
            continue
        out.append(path)
    return sorted(out)


def _read_text_safely(path: Path, max_bytes: int = 200_000) -> str:
    """Read a text file with a size cap; return empty string on failure."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_bytes)
    except OSError:
        return ""


# Compiled regex for git log --oneline parsing: "<sha> <subject>".
_GIT_LOG_LINE_RE = re.compile(r"^[0-9a-f]{7,40}\s+(.*)$")


def _git_recent_commit_paths(
    repo_root: Path,
    *,
    lookback_minutes: int,
) -> dict[str, list[str]]:
    """Map recent commit-sha → list of paths modified.

    Uses ``git log --since=<lookback_minutes>m --name-only --pretty=format:%H``.
    Returns empty dict on git failure (the helper is observability-only).
    """
    cmd = [
        "git",
        "-C",
        str(repo_root),
        "log",
        f"--since={lookback_minutes} minutes ago",
        "--name-only",
        "--pretty=format:%H",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    out: dict[str, list[str]] = {}
    current_sha: str | None = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            current_sha = None
            continue
        if re.fullmatch(r"[0-9a-f]{40}", line):
            current_sha = line
            out[current_sha] = []
        elif current_sha is not None:
            out[current_sha].append(line)
    return out


# ── Public API ───────────────────────────────────────────────────────────
def verify_head_state_before_spawn(
    declared_scope: list[str],
    *,
    pv_evidence: SpawnPvEvidenceContext | None = None,
    lookback_minutes: int = DEFAULT_SISTER_LANDING_LOOKBACK_MINUTES,
    repo_root: Path | None = None,
    research_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    now_utc: _dt.datetime | None = None,
) -> SpawnGuardVerdict:
    """Sister of Catalog #229 PV at the agent-spawn surface.

    Returns a :class:`SpawnGuardVerdict` whose ``recommendation`` is one of:

    * ``PROCEED`` — no conflict; spawn may proceed safely.
    * ``DUPLICATE_HEAD_STATE`` — declared_scope overlaps a recent
      commit-at-HEAD OR a recent landing memo. Spawn likely produces
      STAND_DOWN.
    * ``SISTER_IN_FLIGHT`` — sister subagent's in-flight checkpoint
      declares overlapping files. Coordinate via Catalog #230 ownership
      map OR stand down per Catalog #229.

    Parameters
    ----------
    declared_scope
        Repo-relative path strings the proposed subagent intends to touch.
        Must be a list (not a string — coercing a string to
        ``list('foo.py')`` would silently produce a list of characters).
    pv_evidence
        Optional evidence context the parent agent observed BEFORE the
        spawn-decision. The verdict's diagnostic cites this evidence so
        the operator can audit what was checked.
    lookback_minutes
        Window for "recent" classification across both HEAD commits + sister
        landing memos + sister checkpoints. Default 60 min mirrors Catalog
        #302 + #314 + #340.
    repo_root
        Override the repo root (useful for testing).
    research_dir
        Override the canonical ``.omx/research/`` directory (useful for testing).
    checkpoint_path
        Override the canonical ``.omx/state/subagent_progress.jsonl`` path
        (useful for testing).
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
    research_dir = Path(research_dir) if research_dir is not None else RESEARCH_DIR
    checkpoint_path_resolved = (
        Path(checkpoint_path) if checkpoint_path is not None else CHECKPOINT_JSONL_PATH
    )
    now = now_utc if now_utc is not None else _dt.datetime.now(_dt.timezone.utc)
    evidence = pv_evidence if pv_evidence is not None else SpawnPvEvidenceContext()

    # Normalize declared_scope to non-empty paths.
    declared_clean = [p.strip() for p in declared_scope if p and p.strip()]
    if not declared_clean:
        return SpawnGuardVerdict(
            recommendation="PROCEED",
            overlapping_scope=(),
            diagnostic=(
                "PROCEED: declared_scope is empty; no overlap analysis possible. "
                "Spawn approved without PV scan (per Catalog #229 PV is "
                "operator-routable when declared_scope is unknown)."
            ),
            conflict_source="none",
            consulted_evidence=evidence,
            sister_subagent_ids=(),
            recent_landing_memo_matches=(),
        )

    # ── Surface 1: recent HEAD commits ──────────────────────────────────
    recent_commit_paths = _git_recent_commit_paths(repo_root, lookback_minutes=lookback_minutes)
    head_overlap: list[tuple[str, str, str]] = []  # (sha, declared, commit_path)
    for sha, paths in recent_commit_paths.items():
        for commit_path in paths:
            for declared in declared_clean:
                if _path_overlaps(declared, commit_path):
                    head_overlap.append((sha, declared, commit_path))

    # ── Surface 2: recent landing memos ─────────────────────────────────
    landing_memos = _list_recent_landing_memos(
        research_dir,
        lookback_minutes=lookback_minutes,
        now_utc=now,
    )
    memo_matches: list[Path] = []
    for memo in landing_memos:
        body = _read_text_safely(memo)
        if not body:
            continue
        for declared in declared_clean:
            # A memo "mentions" a path if the path or its basename appears
            # in the memo body. Trim trailing slash for prefix matching.
            d_clean = declared.rstrip("/")
            if d_clean and d_clean in body:
                memo_matches.append(memo)
                break

    # ── Surface 3: sister-subagent checkpoints ──────────────────────────
    in_flight_sisters = _load_sister_subagent_checkpoints(
        checkpoint_path_resolved,
        lookback_minutes=lookback_minutes,
        now_utc=now,
    )
    sister_overlap: list[tuple[str, tuple[str, ...]]] = []
    declared_set = set(declared_clean)
    for sid, _ts, sister_files in in_flight_sisters:
        overlap_paths: set[str] = set()
        for sister_path in sister_files:
            for declared in declared_clean:
                if _path_overlaps(declared, sister_path):
                    overlap_paths.add(declared)
                    break
        if overlap_paths:
            sister_overlap.append((sid, tuple(sorted(overlap_paths))))
    sister_overlap.sort(key=lambda x: x[0])
    sister_in_flight_ids = tuple(sorted(sid for sid, *_ in in_flight_sisters))

    # ── Falling-rule decision (priority order: sister > HEAD > memo) ────
    if sister_overlap:
        overlapping = tuple(sorted({p for _, paths in sister_overlap for p in paths}))
        sister_lines = []
        for sid, paths in sister_overlap[:5]:
            paths_repr = ", ".join(paths[:5])
            more = "" if len(paths) <= 5 else f" (+{len(paths) - 5} more)"
            sister_lines.append(f"    sister={sid!r} overlaps: {{ {paths_repr} }}{more}")
        diagnostic = (
            f"SISTER_IN_FLIGHT: {len(sister_overlap)} sister subagent(s) declare "
            f"overlapping files within the {lookback_minutes}-min lookback window. "
            f"Per Catalog #340 sister-checkpoint guard + Catalog #230 ownership map: "
            f"coordinate BEFORE spawn OR stand down per Catalog #229 PV.\n"
            + "\n".join(sister_lines)
        )
        return SpawnGuardVerdict(
            recommendation="SISTER_IN_FLIGHT",
            overlapping_scope=overlapping,
            diagnostic=diagnostic,
            conflict_source="sister_checkpoint",
            consulted_evidence=evidence,
            sister_subagent_ids=sister_in_flight_ids,
            recent_landing_memo_matches=tuple(
                str(m.relative_to(repo_root)) for m in memo_matches
            ),
        )

    if head_overlap:
        overlapping = tuple(sorted({d for _, d, _ in head_overlap}))
        head_lines = []
        for sha, declared, commit_path in head_overlap[:5]:
            head_lines.append(
                f"    HEAD commit {sha[:9]} touched {commit_path!r} "
                f"(overlaps declared={declared!r})"
            )
        diagnostic = (
            f"DUPLICATE_HEAD_STATE: {len(head_overlap)} recent commit(s) at HEAD "
            f"touched paths overlapping declared_scope within the "
            f"{lookback_minutes}-min lookback window. Per Catalog #229 PV + "
            f"anti-pattern #13 canonical_unwind_path: read the recent commit "
            f"bodies + sister landing memos BEFORE spawn to avoid STAND_DOWN.\n"
            + "\n".join(head_lines)
        )
        return SpawnGuardVerdict(
            recommendation="DUPLICATE_HEAD_STATE",
            overlapping_scope=overlapping,
            diagnostic=diagnostic,
            conflict_source="head_recent_commit",
            consulted_evidence=evidence,
            sister_subagent_ids=sister_in_flight_ids,
            recent_landing_memo_matches=tuple(
                str(m.relative_to(repo_root)) for m in memo_matches
            ),
        )

    if memo_matches:
        # Landing-memo overlap surfaces an advisory: the path was recently
        # landed but is NOT yet in the HEAD-commit lookback window (the
        # commit may have landed outside the window OR the path appears in
        # the memo body without a commit touching the file). Recommend
        # DUPLICATE_HEAD_STATE so the parent agent reads the memo before
        # spawning.
        overlapping = tuple(sorted(declared_set))
        memo_lines = [f"    landing memo: {m.relative_to(repo_root)!s}" for m in memo_matches[:5]]
        diagnostic = (
            f"DUPLICATE_HEAD_STATE: {len(memo_matches)} recent landing memo(s) "
            f"in .omx/research/ mention paths overlapping declared_scope within "
            f"the {lookback_minutes}-min lookback window. Per Catalog #229 PV + "
            f"anti-pattern #13 canonical_unwind_path: read the memo(s) BEFORE "
            f"spawn to avoid STAND_DOWN.\n"
            + "\n".join(memo_lines)
        )
        return SpawnGuardVerdict(
            recommendation="DUPLICATE_HEAD_STATE",
            overlapping_scope=overlapping,
            diagnostic=diagnostic,
            conflict_source="sister_landing_memo",
            consulted_evidence=evidence,
            sister_subagent_ids=sister_in_flight_ids,
            recent_landing_memo_matches=tuple(
                str(m.relative_to(repo_root)) for m in memo_matches
            ),
        )

    # No conflict across any surface.
    return SpawnGuardVerdict(
        recommendation="PROCEED",
        overlapping_scope=(),
        diagnostic=(
            f"PROCEED: no overlap detected across all 3 PV surfaces: "
            f"(a) {len(recent_commit_paths)} recent commit(s) at HEAD; "
            f"(b) {len(landing_memos)} recent landing memo(s); "
            f"(c) {len(in_flight_sisters)} in-flight sister subagent(s) "
            f"within the {lookback_minutes}-min lookback window. Per Catalog "
            f"#229 PV non-negotiable: spawn approved."
        ),
        conflict_source="none",
        consulted_evidence=evidence,
        sister_subagent_ids=sister_in_flight_ids,
        recent_landing_memo_matches=tuple(
            str(m.relative_to(repo_root)) for m in memo_matches
        ),
    )
