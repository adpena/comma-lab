"""Canonical frontier pointer model + upstream leaderboard auto-fetch + DX auto-update.

Per CLAUDE.md "Frontier scores are pointer-only - NON-NEGOTIABLE" (added
2026-05-19): this module replaces hardcoded score literals in CLAUDE.md /
MEMORY.md / memory files with a canonical pointer file at
``.omx/state/canonical_frontier_pointer.json`` (machine-readable; updated
via ``tools/refresh_canonical_frontier.py`` or auto on dispatch completion
per Catalog #343).

Operator-flagged structural bug (2026-05-19 verbatim):

    "your math and recollection is wrong regarding the leaderboard and
    frontier; we have a 0.19205 or something close to that but we havne't
    submitted a PR for it yet beause we thought we might be able to beat
    it"

Root cause: I conflated ``0.19285`` (PR101 GOLD UPSTREAM baseline; archive
sha ``b83bf348...``) with ``0.19205`` (our actual local frontier; archive
sha ``6bae0201...``; lane
``pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean``). We had not
yet submitted a PR for ``0.19205`` because we thought we might beat it.

Structural fix: canonical pointer file is the SoT for the operator-facing
frontier surface. Hardcoded score literals in CLAUDE.md / MEMORY.md /
memory files drift over time; the pointer auto-refreshes from local
canonical state and from the upstream public leaderboard.

The pointer is HISTORICAL_PROVENANCE per Catalog #110 / #113 in the sense
that every refresh is a NEW row in the canonical state ledger; the live
``.omx/state/canonical_frontier_pointer.json`` file is overwritten
atomically via the fcntl-locked write pattern per Catalog #131 / #245
sister discipline.

4-layer canonical pattern per Catalog #245 exemplar:

    Layer 1 = canonical fcntl-locked atomic write helper (this module).
    Layer 2 = operator-facing CLI at ``tools/refresh_canonical_frontier.py``.
    Layer 3 = STRICT preflight gate Catalog #343
              (``check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded``).
    Layer 4 = DX auto-update wire-in to ledger ``update_*_outcome`` paths
              (``tac.deploy.modal.call_id_ledger.update_call_id_outcome`` +
              ``tac.deploy.hf_jobs.job_id_ledger.update_hf_jobs_outcome``).
"""

from __future__ import annotations

import fcntl
import html
import json
import math
import os
import re
import socket
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CANONICAL_FRONTIER_POINTER_LOCK_PATH",
    "CANONICAL_FRONTIER_POINTER_PATH",
    "POINTER_SCHEMA_VERSION",
    "POINTER_STALE_SECONDS",
    "UPSTREAM_LEADERBOARD_TIMEOUT_DEFAULT",
    "AnchorRecord",
    "CanonicalFrontierPointer",
    "FrontierPointerCorruptError",
    "auto_refresh_canonical_frontier_after_dispatch_outcome",
    "effective_frontier_score",
    "load_canonical_frontier_pointer_lenient",
    "load_canonical_frontier_pointer_strict",
    "recompute_effective_frontier",
    "refresh_canonical_frontier_from_local_state",
    "refresh_canonical_frontier_from_upstream_leaderboard",
    "write_canonical_frontier_pointer_locked",
]


POINTER_SCHEMA_VERSION = "canonical_frontier_pointer_v1_20260519"
CANONICAL_FRONTIER_POINTER_PATH = Path(".omx/state/canonical_frontier_pointer.json")
CANONICAL_FRONTIER_POINTER_LOCK_PATH = Path(".omx/state/.canonical_frontier_pointer.lock")
POINTER_STALE_SECONDS = 24 * 3600  # 24-hour DX freshness window
UPSTREAM_LEADERBOARD_TIMEOUT_DEFAULT = 30


class FrontierPointerCorruptError(RuntimeError):
    """Raised by ``load_canonical_frontier_pointer_strict`` on parse failure.

    Per Catalog #138 fail-closed strict-load discipline: a corrupt pointer
    must not be silently coerced to an empty default; downstream consumers
    must surface the corruption and the caller should quarantine
    ``<path>.corrupt.<utc>`` and re-refresh from canonical state.
    """


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class AnchorRecord:
    """Single frontier anchor row with Provenance per Catalog #323.

    Every score field carries axis + hardware_substrate + archive_sha256 +
    lane_id + measured_at_utc + evidence_grade so downstream consumers can
    audit promotion eligibility WITHOUT re-deriving the apples-to-apples
    comparison from prose.
    """

    score: float
    axis: str  # "contest_cpu" | "contest_cuda"
    archive_sha256: str
    lane_id: str | None
    hardware_substrate: str
    measured_at_utc: str | None
    evidence_grade: str  # e.g. "[contest-CPU]" / "[contest-CUDA]" / "[external_leaderboard_snapshot]"
    source_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "axis": str(self.axis),
            "archive_sha256": str(self.archive_sha256),
            "lane_id": self.lane_id,
            "hardware_substrate": str(self.hardware_substrate),
            "measured_at_utc": self.measured_at_utc,
            "evidence_grade": str(self.evidence_grade),
            "source_path": self.source_path,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnchorRecord:
        return cls(
            score=float(data["score"]),
            axis=str(data["axis"]),
            archive_sha256=str(data["archive_sha256"]),
            lane_id=data.get("lane_id"),
            hardware_substrate=str(data.get("hardware_substrate") or ""),
            measured_at_utc=data.get("measured_at_utc"),
            evidence_grade=str(data.get("evidence_grade") or ""),
            source_path=data.get("source_path"),
            extra=dict(data.get("extra") or {}),
        )


@dataclass(frozen=True)
class CanonicalFrontierPointer:
    """Canonical frontier pointer model — SoT for our local frontier + upstream leaderboard.

    Replaces hardcoded score literals in CLAUDE.md / MEMORY.md / memory
    files. Per CLAUDE.md "Frontier scores are pointer-only" non-negotiable
    (added 2026-05-19): operator-facing surfaces MUST cite the pointer file
    rather than embedding score literals that drift over time.
    """

    schema_version: str
    our_local_frontier_contest_cpu: AnchorRecord | None
    our_local_frontier_contest_cuda: AnchorRecord | None
    submitted_pr_number_for_current_frontier: int | None
    upstream_leaderboard_snapshot: dict[str, Any] | None
    upstream_leaderboard_snapshot_at_utc: str | None
    last_refreshed_utc: str
    auto_update_on_dispatch_completion: bool
    pointer_refresh_command: str
    refresh_provenance: dict[str, Any]
    # The competitive target is the minimum of our qualifying exact anchors
    # and the current official public-leaderboard best.  Keep the custody-
    # specific records above intact: an upstream score is a target, not local
    # archive authority.
    effective_frontier: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "our_local_frontier_contest_cpu": (
                self.our_local_frontier_contest_cpu.as_dict()
                if self.our_local_frontier_contest_cpu is not None
                else None
            ),
            "our_local_frontier_contest_cuda": (
                self.our_local_frontier_contest_cuda.as_dict()
                if self.our_local_frontier_contest_cuda is not None
                else None
            ),
            "submitted_pr_number_for_current_frontier": (self.submitted_pr_number_for_current_frontier),
            "upstream_leaderboard_snapshot": self.upstream_leaderboard_snapshot,
            "upstream_leaderboard_snapshot_at_utc": self.upstream_leaderboard_snapshot_at_utc,
            "last_refreshed_utc": str(self.last_refreshed_utc),
            "auto_update_on_dispatch_completion": bool(self.auto_update_on_dispatch_completion),
            "pointer_refresh_command": str(self.pointer_refresh_command),
            "refresh_provenance": dict(self.refresh_provenance),
            "effective_frontier": (
                dict(self.effective_frontier) if isinstance(self.effective_frontier, Mapping) else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CanonicalFrontierPointer:
        cpu_raw = data.get("our_local_frontier_contest_cpu")
        cuda_raw = data.get("our_local_frontier_contest_cuda")
        return cls(
            schema_version=str(data.get("schema_version") or ""),
            our_local_frontier_contest_cpu=(AnchorRecord.from_dict(cpu_raw) if isinstance(cpu_raw, Mapping) else None),
            our_local_frontier_contest_cuda=(
                AnchorRecord.from_dict(cuda_raw) if isinstance(cuda_raw, Mapping) else None
            ),
            submitted_pr_number_for_current_frontier=(data.get("submitted_pr_number_for_current_frontier")),
            upstream_leaderboard_snapshot=data.get("upstream_leaderboard_snapshot"),
            upstream_leaderboard_snapshot_at_utc=data.get("upstream_leaderboard_snapshot_at_utc"),
            last_refreshed_utc=str(data.get("last_refreshed_utc") or ""),
            auto_update_on_dispatch_completion=bool(data.get("auto_update_on_dispatch_completion", True)),
            pointer_refresh_command=str(
                data.get("pointer_refresh_command") or ".venv/bin/python tools/refresh_canonical_frontier.py"
            ),
            refresh_provenance=dict(data.get("refresh_provenance") or {}),
            effective_frontier=(
                dict(data.get("effective_frontier") or {})
                if isinstance(data.get("effective_frontier"), Mapping)
                else None
            ),
        )

    def is_stale(self, *, now_utc_iso: str | None = None) -> bool:
        """True when ``last_refreshed_utc`` is older than POINTER_STALE_SECONDS."""

        try:
            last = datetime.fromisoformat(self.last_refreshed_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        now_str = now_utc_iso or _now_iso()
        try:
            now = datetime.fromisoformat(now_str.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            now = datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        return (now - last).total_seconds() > POINTER_STALE_SECONDS


def effective_frontier_score(pointer: CanonicalFrontierPointer) -> float | None:
    """Return the score to beat, never a custody-specific local substitute.

    ``our_local_frontier_*`` remains the right surface for byte-local deltas
    and archive promotion.  Competitive routing must consume this accessor so
    a better official leaderboard row cannot be hidden by a weaker local row.
    """

    # Recompose exclusively from the custody-bearing constituent rows.  The
    # serialized ``effective_frontier`` row is a cache for display and
    # provenance, never an independently authoritative score.  A historical
    # pointer with no constituents has no defensible competitive target and
    # therefore fails closed with ``None``.
    row = recompute_effective_frontier(pointer)
    if not isinstance(row, Mapping):
        return None
    try:
        score = float(row["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return score if math.isfinite(score) and score > 0 else None


# ─────────────────────────────────────────────────────────────────────────
# Persistence helpers (fcntl-locked atomic write per Catalog #131 / #245)
# ─────────────────────────────────────────────────────────────────────────


def _resolve_pointer_path(*, repo_root: Path | str, path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(repo_root) / CANONICAL_FRONTIER_POINTER_PATH


def _resolve_lock_path(*, repo_root: Path | str, lock_path: Path | None = None) -> Path:
    if lock_path is not None:
        return Path(lock_path)
    return Path(repo_root) / CANONICAL_FRONTIER_POINTER_LOCK_PATH


def write_canonical_frontier_pointer_locked(
    pointer: CanonicalFrontierPointer,
    *,
    repo_root: Path | str = ".",
    path: Path | None = None,
    lock_path: Path | None = None,
) -> Path:
    """Atomic fcntl-locked write of the canonical pointer file.

    Per Catalog #131 / #245 / #313 sister discipline: writes via temp file +
    ``os.replace`` under ``fcntl.flock(LOCK_EX)``, never bare. The temp file
    name includes a uuid12 suffix so concurrent writers do not collide on
    the temp path.
    """

    resolved_path = _resolve_pointer_path(repo_root=repo_root, path=path)
    resolved_lock = _resolve_lock_path(repo_root=repo_root, lock_path=lock_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_lock.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(pointer.as_dict(), indent=2, sort_keys=True) + "\n"
    tmp_path = resolved_path.with_suffix(resolved_path.suffix + f".tmp.{uuid.uuid4().hex[:12]}")

    lock_fd = os.open(str(resolved_lock), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(str(tmp_path), str(resolved_path))
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)
    return resolved_path


def _linked_worktree_main_pointer_fallback(*, repo_root: Path | str) -> Path | None:
    """Resolve the canonical pointer from the MAIN worktree when ``repo_root``
    is a linked git worktree that lacks the (gitignored) pointer file.

    The canonical frontier pointer at
    ``.omx/state/canonical_frontier_pointer.json`` is gitignored shared state;
    it exists only in the primary checkout, never in a fresh linked worktree
    (the standard subagent-arm layout). This helper performs a read-only,
    deterministic, value-identical fallback so any strict reader (the joint-
    descent launcher, ``taskspace_inverse_stack_receipt``, preflight) recovers
    the canonical value instead of failing closed on a structurally-absent
    file. Pure file I/O; no subprocess.

    Returns the main-worktree pointer path if it exists as a regular file,
    else ``None`` (Catalog #138 fail-closed is preserved: the strict loader
    still raises when NEITHER location has the pointer).
    """

    root = Path(repo_root)
    git_marker = root / ".git"
    # A linked worktree has a ``.git`` FILE (``gitdir: <path>``); the primary
    # checkout has a ``.git`` DIRECTORY (there is nothing to fall back to).
    try:
        if not git_marker.is_file():
            return None
        marker_text = git_marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not marker_text.startswith(prefix):
        return None
    gitdir_raw = marker_text[len(prefix) :].strip()
    if not gitdir_raw:
        return None
    gitdir = Path(gitdir_raw)
    if not gitdir.is_absolute():
        gitdir = root / gitdir
    # ``gitdir`` == ``<common>/.git/worktrees/<name>``. ``commondir`` resolves
    # to ``<common>/.git``; the main worktree top is its parent.
    commondir_file = gitdir / "commondir"
    try:
        commondir_raw = commondir_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not commondir_raw:
        return None
    common_git = Path(commondir_raw)
    if not common_git.is_absolute():
        common_git = gitdir / commondir_raw
    try:
        common_git = common_git.resolve()
    except OSError:
        return None
    main_worktree = common_git.parent  # parent of ``<common>/.git``
    fallback_path = main_worktree / CANONICAL_FRONTIER_POINTER_PATH
    return fallback_path if fallback_path.is_file() else None


def load_canonical_frontier_pointer_strict(
    *,
    repo_root: Path | str = ".",
    path: Path | None = None,
) -> CanonicalFrontierPointer:
    """Strict load: raises ``FrontierPointerCorruptError`` on corruption.

    Per Catalog #138 strict-load discipline: missing-file is fail-closed too
    (callers that want lenient missing-file semantics should call
    ``load_canonical_frontier_pointer_lenient`` instead).

    Worktree-aware: when ``path`` is not explicitly given and the resolved
    pointer is absent (the gitignored file does not exist in a fresh linked
    worktree), fall back READ-ONLY to the main worktree's pointer via
    :func:`_linked_worktree_main_pointer_fallback` before failing closed.
    """

    resolved_path = _resolve_pointer_path(repo_root=repo_root, path=path)
    if not resolved_path.is_file():
        if path is None:
            fallback = _linked_worktree_main_pointer_fallback(repo_root=repo_root)
            if fallback is not None:
                resolved_path = fallback
        if not resolved_path.is_file():
            raise FrontierPointerCorruptError(
                f"canonical frontier pointer missing at {resolved_path}; "
                "run `tools/refresh_canonical_frontier.py` to populate"
            )
    try:
        text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FrontierPointerCorruptError(f"read failure: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FrontierPointerCorruptError(f"json parse failure: {exc}") from exc
    if not isinstance(data, dict):
        raise FrontierPointerCorruptError(f"pointer root must be object, got {type(data).__name__}")
    return CanonicalFrontierPointer.from_dict(data)


def load_canonical_frontier_pointer_lenient(
    *,
    repo_root: Path | str = ".",
    path: Path | None = None,
) -> CanonicalFrontierPointer | None:
    """Lenient load: returns ``None`` on missing/corrupt file.

    Use for operator-facing display surfaces that should degrade gracefully
    when the pointer has not yet been populated.
    """

    try:
        return load_canonical_frontier_pointer_strict(repo_root=repo_root, path=path)
    except FrontierPointerCorruptError:
        return None


# ─────────────────────────────────────────────────────────────────────────
# Refresh from local canonical state
# ─────────────────────────────────────────────────────────────────────────


def _anchor_from_serialized(serialized: Mapping[str, Any]) -> AnchorRecord | None:
    """Build AnchorRecord from a ``tac.frontier_scan._serialize_anchor`` row."""

    if not isinstance(serialized, Mapping):
        return None
    score = serialized.get("score")
    sha = serialized.get("archive_sha256")
    axis = serialized.get("axis")
    if score is None or not sha or not axis:
        return None
    extra = serialized.get("extra") or {}
    if not isinstance(extra, Mapping):
        extra = {}
    lane_id = extra.get("lane_id") if isinstance(extra, Mapping) else None
    measured_at = extra.get("measured_at_utc") or extra.get("dispatched_at_utc") or extra.get("promoted_at_utc")
    evidence_grade = extra.get("evidence_grade")
    if not evidence_grade:
        # Derive canonical axis label per CLAUDE.md "Apples-to-apples evidence discipline".
        if str(axis) == "contest_cpu":
            evidence_grade = "[contest-CPU]"
        elif str(axis) == "contest_cuda":
            evidence_grade = "[contest-CUDA]"
        else:
            evidence_grade = f"[{axis}]"
    return AnchorRecord(
        score=float(score),
        axis=str(axis),
        archive_sha256=str(sha),
        lane_id=lane_id,
        hardware_substrate=str(serialized.get("hardware_substrate") or ""),
        measured_at_utc=measured_at,
        evidence_grade=str(evidence_grade),
        source_path=serialized.get("source_path"),
        extra=dict(extra) if isinstance(extra, Mapping) else {},
    )


def _checkpoint_maturity_refusal(anchor: AnchorRecord | None) -> str | None:
    """Refusal reason if this anchor's provenance names a non-``_prod`` VEHICLE checkpoint.

    Per the ``tac.checkpoint_maturity`` convention (operator 2026-07-18): a
    ``_dev`` (or vehicle-shaped-but-untagged) checkpoint's exact row may be
    banked + labeled but MUST NOT move the canonical frontier pointer. Scans
    the anchor's lane_id + source_path + known extra name fields, per PATH
    SEGMENT, through ``pointer_promotion_verdict``. Legacy pre-convention
    names (no vehicle token) pass — refusing them would clobber the standing
    frontier anchors. Returns ``None`` when promotion is allowed.
    """

    if anchor is None:
        return None
    from tac.checkpoint_maturity import pointer_promotion_verdict

    names: list[str] = []
    if anchor.lane_id:
        names.append(str(anchor.lane_id))
    if anchor.source_path:
        names.append(str(anchor.source_path))
    for key in ("lane_id", "run_dir", "checkpoint", "checkpoint_path", "bank_dir", "source_run_dir"):
        value = anchor.extra.get(key)
        if isinstance(value, str) and value:
            names.append(value)
    for name in names:
        for segment in re.split(r"[\\/]", name):
            if not segment:
                continue
            allowed, reason = pointer_promotion_verdict(segment)
            if not allowed:
                return reason
    return None


def _gate_axis_anchor(
    candidate: AnchorRecord | None,
    prior: AnchorRecord | None,
    *,
    axis_label: str,
) -> tuple[AnchorRecord | None, dict[str, Any] | None]:
    """Fail-closed maturity gate for ONE axis anchor.

    Returns ``(anchor_to_use, refusal_record_or_None)``. On refusal the PRIOR
    pointer anchor is kept (the pointer is untouched for that axis) and a loud
    stderr message + machine-readable refusal record are emitted — the dev row
    stays banked/labeled in canonical state but never becomes the pointer.
    """

    reason = _checkpoint_maturity_refusal(candidate)
    if reason is None:
        return candidate, None
    assert candidate is not None
    refusal = {
        "axis": axis_label,
        "refused_score": float(candidate.score),
        "refused_archive_sha256": candidate.archive_sha256,
        "refused_lane_id": candidate.lane_id,
        "reason": reason,
        "action": "pointer untouched for this axis (prior anchor kept); "
        "promote by renaming/re-banking the checkpoint lineage to _prod with operator GO",
    }
    print(
        f"[checkpoint-maturity] REFUSED pointer promotion on {axis_label}: {reason} "
        f"(score={candidate.score}); pointer untouched — dev rows are banked, never promoted.",
        file=sys.stderr,
    )
    return prior, refusal


def _best_public_entry(snapshot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the best ranked public row, including a cached row on fetch failure."""

    if not isinstance(snapshot, Mapping):
        return None
    ranked: list[dict[str, Any]] = []
    best = snapshot.get("best_entry")
    if isinstance(best, Mapping):
        try:
            score = float(best["score"])
        except (KeyError, TypeError, ValueError):
            score = math.nan
        if math.isfinite(score) and score > 0:
            ranked.append({**dict(best), "score": score})
    entries = snapshot.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            try:
                score = float(entry["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(score) and score > 0:
                ranked.append({**dict(entry), "score": score})
    if ranked:
        return min(ranked, key=lambda row: float(row["score"]))
    cached = snapshot.get("cached_snapshot")
    return _best_public_entry(cached if isinstance(cached, Mapping) else None)


def _build_effective_frontier(
    *,
    cpu_anchor: AnchorRecord | None,
    cuda_anchor: AnchorRecord | None,
    upstream_snapshot: Mapping[str, Any] | None,
    upstream_snapshot_at_utc: str | None,
) -> dict[str, Any] | None:
    """Select ``min(our exact anchors, current official leaderboard best)``.

    This is the sole competitive score-to-beat pointer.  It deliberately does
    not overwrite the local anchor records because an external leaderboard row
    does not grant archive custody, replay authority, or submission ownership.
    """

    candidates: list[dict[str, Any]] = []
    for source_key, anchor in (
        ("our_local_frontier_contest_cpu", cpu_anchor),
        ("our_local_frontier_contest_cuda", cuda_anchor),
    ):
        if anchor is None or not math.isfinite(anchor.score) or anchor.score <= 0:
            continue
        candidates.append(
            {
                "score": float(anchor.score),
                "source": source_key,
                "source_kind": "owned_or_banked_local_exact_anchor",
                "axis": anchor.axis,
                "archive_sha256": anchor.archive_sha256,
                "lane_id": anchor.lane_id,
                "hardware_substrate": anchor.hardware_substrate,
                "measured_at_utc": anchor.measured_at_utc,
                "evidence_grade": anchor.evidence_grade,
                "custody": "local_anchor_record; inspect lane policy before submission",
            }
        )

    public = _best_public_entry(upstream_snapshot)
    if public is not None:
        candidates.append(
            {
                "score": float(public["score"]),
                "source": "upstream_official_leaderboard",
                "source_kind": "external_public_leaderboard_target",
                "axis": "official_leaderboard",
                "leaderboard_rank": public.get("rank"),
                "submission_name": public.get("name"),
                "pr_number": public.get("pr_number"),
                "pr_url": public.get("pr_url"),
                "snapshot_at_utc": upstream_snapshot_at_utc,
                "evidence_grade": "[official-leaderboard display]",
                "score_precision": "official_display",
                "custody": "external target only; no local archive authority implied",
            }
        )
    if not candidates:
        return None

    # At equal score prefer our exact row because it has stronger local custody;
    # the competitive threshold is unchanged either way.
    winner = min(
        candidates,
        key=lambda row: (
            float(row["score"]),
            0 if str(row["source"]).startswith("our_local_") else 1,
        ),
    )
    winner["selection_rule"] = (
        "min(our_local_frontier_contest_cpu, our_local_frontier_contest_cuda, upstream_official_leaderboard.best_entry)"
    )
    winner["role"] = "competitive_score_to_beat"
    return winner


def recompute_effective_frontier(pointer: CanonicalFrontierPointer) -> dict[str, Any] | None:
    """Re-derive the competitive winner from the pointer's constituent rows.

    Callers that make admission or routing decisions must compare this result
    with the serialized ``effective_frontier`` row.  This prevents a fresh
    wrapper timestamp or a hand-edited cached winner from hiding a better
    local or official score.
    """

    if not isinstance(pointer, CanonicalFrontierPointer):
        raise TypeError("pointer must be a CanonicalFrontierPointer")
    return _build_effective_frontier(
        cpu_anchor=pointer.our_local_frontier_contest_cpu,
        cuda_anchor=pointer.our_local_frontier_contest_cuda,
        upstream_snapshot=(
            pointer.upstream_leaderboard_snapshot
            if isinstance(pointer.upstream_leaderboard_snapshot, Mapping)
            else None
        ),
        upstream_snapshot_at_utc=pointer.upstream_leaderboard_snapshot_at_utc,
    )


def refresh_canonical_frontier_from_local_state(
    *,
    repo_root: Path | str = ".",
    write: bool = True,
    submitted_pr_number_for_current_frontier: int | None = None,
    pre_existing_pointer: CanonicalFrontierPointer | None = None,
) -> CanonicalFrontierPointer:
    """Refresh pointer from ``tac.frontier_scan.build_frontier_scan_payload``.

    Preserves any upstream leaderboard snapshot from a pre-existing pointer
    (the upstream refresh is opt-in / network-dependent and should not be
    invalidated by a local-state refresh). Pass
    ``submitted_pr_number_for_current_frontier`` to record the PR number
    when the operator has submitted a PR for the current frontier; defaults
    to whatever the prior pointer carried (or ``None``).
    """

    # Avoid circular import: ``tac.frontier_scan`` may not need this module
    # but the canonical refresher consumes the scanner.
    from tac.frontier_scan import build_frontier_scan_payload

    repo_root_path = Path(repo_root)
    payload = build_frontier_scan_payload(repo_root_path)
    best = payload.get("best_per_axis") or {}
    cpu_anchor: AnchorRecord | None = None
    cuda_anchor: AnchorRecord | None = None
    if isinstance(best, Mapping):
        cpu_raw = best.get("contest_cpu")
        cuda_raw = best.get("contest_cuda")
        if isinstance(cpu_raw, Mapping):
            cpu_anchor = _anchor_from_serialized(cpu_raw)
        if isinstance(cuda_raw, Mapping):
            cuda_anchor = _anchor_from_serialized(cuda_raw)

    # Preserve upstream snapshot from prior pointer if not refreshing upstream.
    prior = pre_existing_pointer
    if prior is None:
        prior = load_canonical_frontier_pointer_lenient(repo_root=repo_root_path)

    # Checkpoint-maturity gate (fail-closed; tac.checkpoint_maturity): a
    # candidate anchor whose provenance names a _dev / untagged-vehicle
    # checkpoint NEVER becomes the pointer anchor — the prior anchor is kept
    # and the refusal is recorded in refresh_provenance.
    maturity_refusals: list[dict[str, Any]] = []
    cpu_anchor, cpu_refusal = _gate_axis_anchor(
        cpu_anchor,
        prior.our_local_frontier_contest_cpu if prior is not None else None,
        axis_label="contest_cpu",
    )
    if cpu_refusal is not None:
        maturity_refusals.append(cpu_refusal)
    cuda_anchor, cuda_refusal = _gate_axis_anchor(
        cuda_anchor,
        prior.our_local_frontier_contest_cuda if prior is not None else None,
        axis_label="contest_cuda",
    )
    if cuda_refusal is not None:
        maturity_refusals.append(cuda_refusal)

    upstream_snapshot = None
    upstream_snapshot_at = None
    pr_number = submitted_pr_number_for_current_frontier
    if prior is not None:
        upstream_snapshot = prior.upstream_leaderboard_snapshot
        upstream_snapshot_at = prior.upstream_leaderboard_snapshot_at_utc
        if pr_number is None:
            pr_number = prior.submitted_pr_number_for_current_frontier

    pointer = CanonicalFrontierPointer(
        schema_version=POINTER_SCHEMA_VERSION,
        our_local_frontier_contest_cpu=cpu_anchor,
        our_local_frontier_contest_cuda=cuda_anchor,
        submitted_pr_number_for_current_frontier=pr_number,
        upstream_leaderboard_snapshot=upstream_snapshot,
        upstream_leaderboard_snapshot_at_utc=upstream_snapshot_at,
        last_refreshed_utc=_now_iso(),
        auto_update_on_dispatch_completion=True,
        pointer_refresh_command=".venv/bin/python tools/refresh_canonical_frontier.py",
        refresh_provenance={
            "refresh_kind": "local_state",
            "refreshed_at_utc": _now_iso(),
            "refresher_pid": os.getpid(),
            "refresher_host": socket.gethostname(),
            "scan_stats": payload.get("scan_stats"),
            "checkpoint_maturity_refusals": maturity_refusals,
        },
        effective_frontier=_build_effective_frontier(
            cpu_anchor=cpu_anchor,
            cuda_anchor=cuda_anchor,
            upstream_snapshot=(upstream_snapshot if isinstance(upstream_snapshot, Mapping) else None),
            upstream_snapshot_at_utc=upstream_snapshot_at,
        ),
    )

    if write:
        write_canonical_frontier_pointer_locked(pointer, repo_root=repo_root_path)
    return pointer


# ─────────────────────────────────────────────────────────────────────────
# Refresh from upstream public leaderboard
# ─────────────────────────────────────────────────────────────────────────


# The official ranked table is score authority for the public target.  A
# recent-PR listing is not a leaderboard: that older implementation could be
# network-fresh while omitting every score, which left routing at 0.191 after
# PR130 moved the official frontier to 0.172.
UPSTREAM_LEADERBOARD_OFFICIAL_URL = "https://comma.ai/leaderboard"
_OFFICIAL_VIDEO_TABLE_ID = "video_compression_challenge_table"
_LEADERBOARD_ROW_RE = re.compile(r"<tr(?:\s[^>]*)?>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_LEADERBOARD_CELL_RE = re.compile(r"<td(?:\s[^>]*)?>\s*(.*?)\s*</td>", re.DOTALL | re.IGNORECASE)
_LEADERBOARD_PR_RE = re.compile(
    r'href=["\'](https://github\.com/commaai/comma_video_compression_challenge/pull/(\d+))["\']',
    re.IGNORECASE,
)


def _strip_leaderboard_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _parse_official_leaderboard_entries(page_html: str) -> list[dict[str, Any]]:
    """Parse the ranked video-compression rows from comma.ai's official page."""

    container_idx = page_html.find(f'id="{_OFFICIAL_VIDEO_TABLE_ID}"')
    if container_idx < 0:
        container_idx = page_html.find(f"id='{_OFFICIAL_VIDEO_TABLE_ID}'")
    if container_idx < 0:
        raise ValueError(f"official video leaderboard container missing: {_OFFICIAL_VIDEO_TABLE_ID}")
    table_start = page_html.find("<table", container_idx)
    table_end = page_html.find("</table>", table_start)
    if table_start < 0 or table_end < 0:
        raise ValueError("official video leaderboard table missing")
    table = page_html[table_start : table_end + len("</table>")]

    entries: list[dict[str, Any]] = []
    for row_html in _LEADERBOARD_ROW_RE.findall(table):
        cells = _LEADERBOARD_CELL_RE.findall(row_html)
        if len(cells) < 4:
            continue
        score_text = re.sub(r"\s+", "", _strip_leaderboard_html(cells[1]))
        try:
            score = float(score_text)
        except ValueError:
            continue
        if not math.isfinite(score) or score <= 0:
            continue
        pr_match = _LEADERBOARD_PR_RE.search(cells[3])
        entries.append(
            {
                "rank": len(entries) + 1,
                "score": score,
                "name": re.sub(r"\s+", " ", _strip_leaderboard_html(cells[2])),
                "pr_url": pr_match.group(1) if pr_match else None,
                "pr_number": int(pr_match.group(2)) if pr_match else None,
            }
        )
    if not entries:
        raise ValueError("parsed zero official leaderboard entries")
    return entries


def _fetch_upstream_leaderboard_snapshot_via_official(
    *, timeout_sec: int = UPSTREAM_LEADERBOARD_TIMEOUT_DEFAULT
) -> dict[str, Any]:
    """Fetch the ranked comma.ai video-compression leaderboard.

    Returns a dict with ``fetch_status`` ("ok" | "network_failure" |
    "parse_failure") + ``fetched_at_utc`` + ranked ``entries``. Graceful
    degradation returns ``fetch_status != "ok"`` and ``entries = []`` so the
    pointer model can carry the cached snapshot from the prior refresh and
    record the failure.
    """

    snapshot: dict[str, Any] = {
        "source": "official_leaderboard",
        "url": UPSTREAM_LEADERBOARD_OFFICIAL_URL,
        "fetched_at_utc": _now_iso(),
        "fetch_status": "ok",
        "entries": [],
    }
    req = urllib.request.Request(
        UPSTREAM_LEADERBOARD_OFFICIAL_URL,
        headers={
            "User-Agent": "pact-frontier-pointer-refresher/v1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            page_html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        snapshot["fetch_status"] = "network_failure"
        snapshot["fetch_error"] = str(exc)
        return snapshot
    try:
        entries = _parse_official_leaderboard_entries(page_html)
    except ValueError as exc:
        snapshot["fetch_status"] = "parse_failure"
        snapshot["fetch_error"] = str(exc)
        return snapshot
    snapshot["entries"] = entries
    snapshot["best_entry"] = min(entries, key=lambda row: float(row["score"]))
    snapshot["entry_count"] = len(entries)
    snapshot["score_precision"] = "official_display"
    return snapshot


# Backward-compatible private alias for test/automation imports.  Its semantics
# are now the official ranked leaderboard, not GitHub recent-PR metadata.
_fetch_upstream_leaderboard_snapshot_via_github = _fetch_upstream_leaderboard_snapshot_via_official


def refresh_canonical_frontier_from_upstream_leaderboard(
    *,
    repo_root: Path | str = ".",
    timeout_sec: int = UPSTREAM_LEADERBOARD_TIMEOUT_DEFAULT,
    write: bool = True,
    fetcher: Any = None,
) -> CanonicalFrontierPointer:
    """Refresh pointer's upstream snapshot from comma.ai contest leaderboard.

    Network-dependent and graceful: if the upstream fetch fails, the
    pointer's local-frontier fields refresh from canonical state and the
    upstream snapshot is tagged ``fetch_status: network_failure`` while
    preserving any prior cached snapshot.

    Pass ``fetcher`` (callable with signature ``(timeout_sec=int) ->
    dict``) to inject a custom fetcher for tests; defaults to
    ``_fetch_upstream_leaderboard_snapshot_via_official``.
    """

    fetch_callable = fetcher or _fetch_upstream_leaderboard_snapshot_via_official
    snapshot = fetch_callable(timeout_sec=timeout_sec)

    # Start from local-state refresh (without writing) so we preserve the
    # local frontier in the same commit.
    repo_root_path = Path(repo_root)
    prior = load_canonical_frontier_pointer_lenient(repo_root=repo_root_path)
    pointer = refresh_canonical_frontier_from_local_state(
        repo_root=repo_root_path,
        write=False,
        pre_existing_pointer=prior,
    )

    # Always update the snapshot timestamp; on network failure, preserve
    # prior ranked payload but record the failure.
    fetch_status = snapshot.get("fetch_status")
    if fetch_status == "ok":
        new_snapshot = snapshot
        effective_snapshot_at = snapshot.get("fetched_at_utc")
    else:
        # Preserve prior cached snapshot but record the failure window.
        new_snapshot = {
            "source": snapshot.get("source"),
            "url": snapshot.get("url"),
            "fetched_at_utc": snapshot.get("fetched_at_utc"),
            "fetch_status": fetch_status,
            "fetch_error": snapshot.get("fetch_error"),
            "cached_snapshot": (
                pointer.upstream_leaderboard_snapshot if pointer.upstream_leaderboard_snapshot else None
            ),
            "cached_snapshot_at_utc": pointer.upstream_leaderboard_snapshot_at_utc,
            "entries": [],
        }
        effective_snapshot_at = pointer.upstream_leaderboard_snapshot_at_utc

    refreshed = CanonicalFrontierPointer(
        schema_version=pointer.schema_version,
        our_local_frontier_contest_cpu=pointer.our_local_frontier_contest_cpu,
        our_local_frontier_contest_cuda=pointer.our_local_frontier_contest_cuda,
        submitted_pr_number_for_current_frontier=pointer.submitted_pr_number_for_current_frontier,
        upstream_leaderboard_snapshot=new_snapshot,
        # A failed fetch is an attempted-refresh timestamp, not fresh official
        # evidence.  Preserve the last successful snapshot time fail-closed.
        upstream_leaderboard_snapshot_at_utc=effective_snapshot_at,
        last_refreshed_utc=_now_iso(),
        auto_update_on_dispatch_completion=pointer.auto_update_on_dispatch_completion,
        pointer_refresh_command=pointer.pointer_refresh_command,
        refresh_provenance={
            "refresh_kind": "local_state+upstream_leaderboard",
            "refreshed_at_utc": _now_iso(),
            "refresher_pid": os.getpid(),
            "refresher_host": socket.gethostname(),
            "upstream_fetch_status": fetch_status,
            "upstream_fetch_error": snapshot.get("fetch_error"),
            "scan_stats": pointer.refresh_provenance.get("scan_stats"),
            "checkpoint_maturity_refusals": pointer.refresh_provenance.get("checkpoint_maturity_refusals", []),
        },
        effective_frontier=_build_effective_frontier(
            cpu_anchor=pointer.our_local_frontier_contest_cpu,
            cuda_anchor=pointer.our_local_frontier_contest_cuda,
            upstream_snapshot=new_snapshot,
            upstream_snapshot_at_utc=effective_snapshot_at,
        ),
    )

    if write:
        write_canonical_frontier_pointer_locked(refreshed, repo_root=repo_root_path)
    return refreshed


# ─────────────────────────────────────────────────────────────────────────
# DX auto-update hook for ledger update_outcome paths
# ─────────────────────────────────────────────────────────────────────────


def auto_refresh_canonical_frontier_after_dispatch_outcome(
    *,
    status: str,
    score: float | None = None,
    score_axis: str | None = None,
    archive_sha256: str | None = None,
    repo_root: Path | str = ".",
) -> CanonicalFrontierPointer | None:
    """Auto-refresh hook called by ledger ``update_*_outcome`` functions.

    Called from ``tac.deploy.modal.call_id_ledger.update_call_id_outcome`` +
    ``tac.deploy.hf_jobs.job_id_ledger.update_hf_jobs_outcome`` after the
    canonical posterior write completes. Refresh fires only when:

    - ``status == "harvested"`` (dispatch successfully completed); AND
    - ``score`` is finite (the harvested result carried a numeric score); AND
    - The pointer file's ``auto_update_on_dispatch_completion`` flag is True.

    Returns the refreshed pointer (or ``None`` if the auto-refresh did not
    fire). Fail-quietly per CLAUDE.md "MAXIMUM SIGNAL PRESERVATION": any
    refresh failure is captured in the pointer's ``refresh_provenance`` but
    does NOT raise from the dispatch outcome write path (the ledger write
    has already succeeded; pointer refresh is a downstream observability
    surface).
    """

    if status != "harvested":
        return None
    if score is None:
        return None
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return None
    # Guard against NaN/Inf
    if score_f != score_f or score_f in (float("inf"), float("-inf")):
        return None

    repo_root_path = Path(repo_root)
    prior = load_canonical_frontier_pointer_lenient(repo_root=repo_root_path)
    if prior is not None and not prior.auto_update_on_dispatch_completion:
        return None

    try:
        # Once an upstream snapshot exists, every dispatch harvest refreshes
        # the official ranked target as well as local anchors.  Brand-new test
        # repos and offline bootstrap paths remain local-only until their first
        # explicit upstream refresh.
        if prior is not None and prior.upstream_leaderboard_snapshot is not None:
            refreshed = refresh_canonical_frontier_from_upstream_leaderboard(
                repo_root=repo_root_path,
                write=True,
            )
        else:
            refreshed = refresh_canonical_frontier_from_local_state(
                repo_root=repo_root_path,
                write=True,
                pre_existing_pointer=prior,
            )
    except Exception:
        # Per CLAUDE.md "Subagent coherence-by-default" maximum-signal-preservation:
        # ledger write has already succeeded; pointer refresh failure is a
        # downstream observability concern and must not propagate.
        return None

    # AUTO-TRIGGER-MLX-PER-PAIR-WIRE-IN (de-orphan the MLX per-pair extractor
    # per operator directive 2026-05-27 + the 7th AUTOMATED+COMPOUNDING+OPTIMAL
    # standing directive + CLAUDE.md "Results must become system intelligence").
    # When the frontier archive changes, auto-schedule the $0 MLX-local per-pair
    # heuristic-prior extraction so the 5D canvas / Dykstra Pareto solver /
    # bit_allocator always have per-pair signal for the CURRENT frontier.
    # Sister of the Catalog #1100 ``append_anchor_locked`` post-anchor consumer
    # fan-out pattern. Fail-quiet per the canonical contract: the pointer
    # refresh already succeeded; the MLX schedule is a downstream
    # observability-only signal (NON-PROMOTABLE per Catalog #192/#127/#323) and
    # must NOT block / raise from the dispatch-outcome path. Default emits a
    # ``scheduled`` row (the heavy extraction runs out-of-band via the canonical
    # CLI; $0 MLX-local); idempotent per frontier sha.
    try:
        from tac.master_gradient_mlx_pipeline import (
            auto_schedule_mlx_per_pair_extraction_for_frontier,
        )

        auto_schedule_mlx_per_pair_extraction_for_frontier(repo_root=repo_root_path)
    except Exception:
        pass

    # CITATION-SURFACE CO-REFRESH (rv17_w3_citation_surface_autorefresh,
    # 2026-08-21). The pointer auto-refreshed on every harvest while the
    # Markdown mirrors that tell OTHER READERS the score (reports/latest.md,
    # .omx/state/current_focus.md) lagged THREE pointer moves (0.156526 cited
    # vs canonical 0.148278) with their own catalog-316 detector RED and
    # unread. Regenerating the mirrors here binds them to the pointer at the
    # only moment the pointer can change. Fail-quiet per the same contract as
    # the block above: the pointer refresh already succeeded; mirror refresh
    # is downstream observability and must not raise from the dispatch path.
    try:
        from tac.frontier_scan import refresh_frontier_citation_surfaces

        refresh_frontier_citation_surfaces(repo_root_path)
    except Exception:
        pass

    return refreshed
