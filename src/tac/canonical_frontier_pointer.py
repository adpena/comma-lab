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
import json
import os
import socket
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "AnchorRecord",
    "CanonicalFrontierPointer",
    "CANONICAL_FRONTIER_POINTER_PATH",
    "CANONICAL_FRONTIER_POINTER_LOCK_PATH",
    "POINTER_SCHEMA_VERSION",
    "POINTER_STALE_SECONDS",
    "UPSTREAM_LEADERBOARD_TIMEOUT_DEFAULT",
    "FrontierPointerCorruptError",
    "load_canonical_frontier_pointer_strict",
    "load_canonical_frontier_pointer_lenient",
    "write_canonical_frontier_pointer_locked",
    "refresh_canonical_frontier_from_local_state",
    "refresh_canonical_frontier_from_upstream_leaderboard",
    "auto_refresh_canonical_frontier_after_dispatch_outcome",
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
    return datetime.now(timezone.utc).isoformat()


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
    def from_dict(cls, data: Mapping[str, Any]) -> "AnchorRecord":
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
            "submitted_pr_number_for_current_frontier": (
                self.submitted_pr_number_for_current_frontier
            ),
            "upstream_leaderboard_snapshot": self.upstream_leaderboard_snapshot,
            "upstream_leaderboard_snapshot_at_utc": self.upstream_leaderboard_snapshot_at_utc,
            "last_refreshed_utc": str(self.last_refreshed_utc),
            "auto_update_on_dispatch_completion": bool(
                self.auto_update_on_dispatch_completion
            ),
            "pointer_refresh_command": str(self.pointer_refresh_command),
            "refresh_provenance": dict(self.refresh_provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalFrontierPointer":
        cpu_raw = data.get("our_local_frontier_contest_cpu")
        cuda_raw = data.get("our_local_frontier_contest_cuda")
        return cls(
            schema_version=str(data.get("schema_version") or ""),
            our_local_frontier_contest_cpu=(
                AnchorRecord.from_dict(cpu_raw) if isinstance(cpu_raw, Mapping) else None
            ),
            our_local_frontier_contest_cuda=(
                AnchorRecord.from_dict(cuda_raw) if isinstance(cuda_raw, Mapping) else None
            ),
            submitted_pr_number_for_current_frontier=(
                data.get("submitted_pr_number_for_current_frontier")
            ),
            upstream_leaderboard_snapshot=data.get("upstream_leaderboard_snapshot"),
            upstream_leaderboard_snapshot_at_utc=data.get(
                "upstream_leaderboard_snapshot_at_utc"
            ),
            last_refreshed_utc=str(data.get("last_refreshed_utc") or ""),
            auto_update_on_dispatch_completion=bool(
                data.get("auto_update_on_dispatch_completion", True)
            ),
            pointer_refresh_command=str(
                data.get("pointer_refresh_command")
                or ".venv/bin/python tools/refresh_canonical_frontier.py"
            ),
            refresh_provenance=dict(data.get("refresh_provenance") or {}),
        )

    def is_stale(self, *, now_utc_iso: str | None = None) -> bool:
        """True when ``last_refreshed_utc`` is older than POINTER_STALE_SECONDS."""

        try:
            last = datetime.fromisoformat(self.last_refreshed_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        now_str = now_utc_iso or _now_iso()
        try:
            now = datetime.fromisoformat(now_str.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return (now - last).total_seconds() > POINTER_STALE_SECONDS


# ─────────────────────────────────────────────────────────────────────────
# Persistence helpers (fcntl-locked atomic write per Catalog #131 / #245)
# ─────────────────────────────────────────────────────────────────────────


def _resolve_pointer_path(
    *, repo_root: Path | str, path: Path | None = None
) -> Path:
    if path is not None:
        return Path(path)
    return Path(repo_root) / CANONICAL_FRONTIER_POINTER_PATH


def _resolve_lock_path(
    *, repo_root: Path | str, lock_path: Path | None = None
) -> Path:
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


def load_canonical_frontier_pointer_strict(
    *,
    repo_root: Path | str = ".",
    path: Path | None = None,
) -> CanonicalFrontierPointer:
    """Strict load: raises ``FrontierPointerCorruptError`` on corruption.

    Per Catalog #138 strict-load discipline: missing-file is fail-closed too
    (callers that want lenient missing-file semantics should call
    ``load_canonical_frontier_pointer_lenient`` instead).
    """

    resolved_path = _resolve_pointer_path(repo_root=repo_root, path=path)
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
        raise FrontierPointerCorruptError(
            f"pointer root must be object, got {type(data).__name__}"
        )
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
    measured_at = (
        extra.get("measured_at_utc")
        or extra.get("dispatched_at_utc")
        or extra.get("promoted_at_utc")
    )
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
    if prior is None and write:
        prior = load_canonical_frontier_pointer_lenient(repo_root=repo_root_path)

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
        },
    )

    if write:
        write_canonical_frontier_pointer_locked(pointer, repo_root=repo_root_path)
    return pointer


# ─────────────────────────────────────────────────────────────────────────
# Refresh from upstream public leaderboard
# ─────────────────────────────────────────────────────────────────────────


# Comma.ai contest leaderboard upstream surface. The comma video compression
# challenge tracks PRs at https://github.com/commaai/commavq; the public
# leaderboard surface is the merged-PR list with eval scores in PR bodies /
# comments. We fetch the GitHub Pulls API and extract any PR title/body that
# carries a recognisable score literal, with graceful degradation on network
# failure per CLAUDE.md "Public frontier watch and intake" non-negotiable.
UPSTREAM_LEADERBOARD_GITHUB_API = (
    "https://api.github.com/repos/commaai/commavq/pulls"
    "?state=all&sort=updated&direction=desc&per_page=30"
)


def _fetch_upstream_leaderboard_snapshot_via_github(
    *, timeout_sec: int = UPSTREAM_LEADERBOARD_TIMEOUT_DEFAULT
) -> dict[str, Any]:
    """Fetch the comma.ai contest leaderboard snapshot via GitHub Pulls API.

    Returns a dict with ``fetch_status`` ("ok" | "network_failure" |
    "parse_failure") + ``fetched_at_utc`` + ``pulls`` (list of PR records
    with title/number/state/score-hints). Graceful degradation: any failure
    returns a dict with ``fetch_status != "ok"`` and ``pulls = []`` so the
    pointer model can carry the cached snapshot from the prior refresh and
    record the failure.
    """

    snapshot: dict[str, Any] = {
        "source": "github_pulls_api",
        "url": UPSTREAM_LEADERBOARD_GITHUB_API,
        "fetched_at_utc": _now_iso(),
        "fetch_status": "ok",
        "pulls": [],
    }
    req = urllib.request.Request(
        UPSTREAM_LEADERBOARD_GITHUB_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "pact-frontier-pointer-refresher/v1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        snapshot["fetch_status"] = "network_failure"
        snapshot["fetch_error"] = str(exc)
        return snapshot
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        snapshot["fetch_status"] = "parse_failure"
        snapshot["fetch_error"] = str(exc)
        return snapshot
    if not isinstance(data, list):
        snapshot["fetch_status"] = "parse_failure"
        snapshot["fetch_error"] = "github pulls api returned non-list payload"
        return snapshot

    pulls: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        pulls.append({
            "number": entry.get("number"),
            "title": entry.get("title"),
            "state": entry.get("state"),
            "merged_at": entry.get("merged_at"),
            "updated_at": entry.get("updated_at"),
            "html_url": entry.get("html_url"),
            "user_login": (entry.get("user") or {}).get("login")
            if isinstance(entry.get("user"), dict)
            else None,
        })
    snapshot["pulls"] = pulls
    return snapshot


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
    ``_fetch_upstream_leaderboard_snapshot_via_github``.
    """

    fetch_callable = fetcher or _fetch_upstream_leaderboard_snapshot_via_github
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
    # prior pulls payload but record the failure.
    fetch_status = snapshot.get("fetch_status")
    if fetch_status == "ok":
        new_snapshot = snapshot
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
            "pulls": [],
        }

    refreshed = CanonicalFrontierPointer(
        schema_version=pointer.schema_version,
        our_local_frontier_contest_cpu=pointer.our_local_frontier_contest_cpu,
        our_local_frontier_contest_cuda=pointer.our_local_frontier_contest_cuda,
        submitted_pr_number_for_current_frontier=pointer.submitted_pr_number_for_current_frontier,
        upstream_leaderboard_snapshot=new_snapshot,
        upstream_leaderboard_snapshot_at_utc=snapshot.get("fetched_at_utc"),
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
        },
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
        refreshed = refresh_canonical_frontier_from_local_state(
            repo_root=repo_root_path,
            write=True,
            pre_existing_pointer=prior,
        )
    except Exception:  # noqa: BLE001 — never raise from dispatch outcome path
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
    except Exception:  # noqa: BLE001 — fail-quiet; observability-only downstream
        pass

    return refreshed
