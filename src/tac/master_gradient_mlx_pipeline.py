# SPDX-License-Identifier: MIT
"""Auto-trigger pipeline for the MLX per-pair master-gradient extractor.

De-orphans `tac.master_gradient_mlx_extractor` / `tools/extract_master_gradient_mlx.py`
per the operator directive 2026-05-27 *"Shouldn't that be automated and wired
and integrated into a pipeline rather than orphaned tool?"* + the 7th
AUTOMATED+COMPOUNDING+OPTIMAL standing directive + CLAUDE.md "Results must
become system intelligence" non-negotiable.

The canonical seam: when a NEW frontier archive lands (the canonical frontier
pointer's ``archive_sha256`` changes per Catalog #343), the MLX per-pair
extraction is auto-scheduled so the 5D canvas / Dykstra Pareto solver /
bit_allocator always have per-pair heuristic-prior signal for the *current*
frontier rather than a stale one.

Design (matches the Catalog #343 +
``tac.master_gradient.append_anchor_locked`` #1100 auto-trigger patterns):

- ``auto_schedule_mlx_per_pair_extraction_for_frontier`` reads the current
  frontier pointer, derives the frontier archive sha (CPU axis preferred,
  CUDA fallback), and consults the fcntl-locked extraction-state ledger
  (``.omx/state/mlx_per_pair_extraction_state.jsonl``). If the frontier sha
  already has a ``completed`` extraction row, it is a no-op (idempotent). If
  the frontier sha changed, it emits a ``scheduled`` row (default) OR, when
  ``run_now=True`` and an artifact path is resolvable, runs the extraction
  inline and emits a ``completed`` row.

- Deterministic + idempotent: re-running with the same frontier sha never
  re-extracts (unless ``force=True``). The state ledger is APPEND-ONLY per
  Catalog #110/#113 HISTORICAL_PROVENANCE; latest-row-wins per sha.

- Fail-quiet on the frontier-refresh hot path: any failure is captured in the
  returned verdict's ``error`` field but NEVER raises from the dispatch-outcome
  / frontier-refresh seam (the canonical write has already succeeded; the MLX
  extraction is a downstream observability-only signal).

NON-PROMOTABLE: the MLX per-pair signal is a ``macOS-MLX research-signal``
heuristic gradient prior (Catalog #192/#127/#323). The pipeline NEVER promotes
it to authority; the artifacts it schedules REFUSE
``master_gradient_anchors.jsonl`` rows by construction (per
``tac.master_gradient_mlx_extractor.mlx_master_gradient_anchor_blockers``).

RIGOR-GATING: consumption trust level is gated by the rigor-review verdict
(``master_gradient_analysis_rigor_signal_review_*``). If that review finds the
per-tensor-FD-via-MLX-oracle output is a uniform-mantissa-projection ARTIFACT
rather than a genuine heuristic prior, the pipeline consumer remains Tier-A
observability-only and a PyTorch-autograd authority cross-check
(``tools/extract_master_gradient.py``) gates any promotion. The pipeline is
built signal-AGNOSTIC: it is SAFE regardless of the rigor verdict because it
NEVER promotes the signal to a score adjustment.

Canonical equation: ``mlx_per_pair_master_gradient_per_byte_fd_v1`` (Catalog
#344, ``RECALIBRATE_ON_NEW_ANCHORS``). This module is registered as a
canonical PRODUCER of that equation.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "MLX_EXTRACTION_STATE_PATH",
    "MLX_PER_PAIR_ARTIFACT_GLOB",
    "MLXExtractionScheduleVerdict",
    "auto_schedule_mlx_per_pair_extraction_for_frontier",
    "derive_frontier_archive_sha",
    "latest_extraction_state_for_sha",
    "load_extraction_state_lenient",
    "resolve_latest_mlx_artifact_for_sha",
]

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]

# Canonical fcntl-locked extraction-state ledger (APPEND-ONLY per Catalog
# #110/#113; latest-row-wins per (archive_sha256, n_pairs) key).
MLX_EXTRACTION_STATE_PATH = (
    REPO_ROOT_DEFAULT / ".omx" / "state" / "mlx_per_pair_extraction_state.jsonl"
)
_MLX_EXTRACTION_STATE_LOCK_PATH = (
    REPO_ROOT_DEFAULT / ".omx" / "state" / ".mlx_per_pair_extraction_state.lock"
)

# Canonical MLX research-signal manifest (the extractor CLI appends here).
MLX_RESEARCH_SIGNAL_MANIFEST_PATH = (
    REPO_ROOT_DEFAULT / ".omx" / "state" / "mlx_research_signal_manifest.jsonl"
)

# Glob for the per-pair artifacts the extractor lands.
MLX_PER_PAIR_ARTIFACT_GLOB = "master_gradient_*_mlx_per_pair_*.npy"

# Frontier fec6/PR101 archive — the canonical CPU-frontier dominant component
# the MLX extractor targets. Resolved by sha at trigger time; this default is
# the well-known fec6 archive used at the landing.
_DEFAULT_FRONTIER_ARCHIVE_RELPATH = (
    "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "archive.zip"
)

SCHEMA_VERSION = "mlx_per_pair_extraction_state_v1_20260527"

# Default fast tier; the deep tier is full-600.
DEFAULT_FAST_TIER_PAIRS = 64
DEFAULT_DEEP_TIER_PAIRS = 600

# State row event types.
EVENT_SCHEDULED = "scheduled"
EVENT_COMPLETED = "completed"
EVENT_NO_OP = "no_op"
EVENT_ERROR = "error"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MLXExtractionScheduleVerdict:
    """Typed verdict from the auto-trigger seam (observability-only)."""

    fired: bool
    event_type: str  # scheduled | completed | no_op | error
    frontier_archive_sha256: str | None
    frontier_axis: str | None
    n_pairs: int
    artifact_path: str | None
    state_row_written: bool
    rationale: str
    error: str | None = None
    # Per Catalog #341 canonical non-promotable markers: the pipeline NEVER
    # promotes the MLX signal to a score adjustment.
    promotable: bool = False
    score_claim: bool = False
    evidence_grade: str = "macOS-MLX research-signal"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fired": bool(self.fired),
            "event_type": str(self.event_type),
            "frontier_archive_sha256": self.frontier_archive_sha256,
            "frontier_axis": self.frontier_axis,
            "n_pairs": int(self.n_pairs),
            "artifact_path": self.artifact_path,
            "state_row_written": bool(self.state_row_written),
            "rationale": str(self.rationale),
            "error": self.error,
            "promotable": bool(self.promotable),
            "score_claim": bool(self.score_claim),
            "evidence_grade": str(self.evidence_grade),
            "extra": dict(self.extra),
        }


def load_extraction_state_lenient(path: Path | None = None) -> list[dict]:
    """Skip malformed rows; return parsed list (newest last)."""
    target = path or MLX_EXTRACTION_STATE_PATH
    if not target.exists():
        return []
    rows: list[dict] = []
    for line in target.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def latest_extraction_state_for_sha(
    archive_sha256: str,
    *,
    n_pairs: int | None = None,
    path: Path | None = None,
) -> dict | None:
    """Return the latest extraction-state row for an archive sha (latest-wins).

    When ``n_pairs`` is given, only rows with the matching tier are considered.
    """
    if not archive_sha256:
        return None
    rows = load_extraction_state_lenient(path)
    latest: dict | None = None
    for row in rows:
        if row.get("frontier_archive_sha256") != archive_sha256:
            continue
        if n_pairs is not None and int(row.get("n_pairs", -1)) != int(n_pairs):
            continue
        latest = row
    return latest


def _append_state_row_locked(row: dict, *, path: Path | None = None) -> None:
    """fcntl-locked append per Catalog #131/#138/#245."""
    target = path or MLX_EXTRACTION_STATE_PATH
    _ensure_parent(target)
    _ensure_parent(_MLX_EXTRACTION_STATE_LOCK_PATH)
    payload = dict(row)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload["written_at_utc"] = _utc_now()
    payload["written_pid"] = os.getpid()
    payload["written_host"] = socket.gethostname()
    line = json.dumps(payload, sort_keys=True)
    with open(_MLX_EXTRACTION_STATE_LOCK_PATH, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
            existing = target.read_text() if target.exists() else ""
            tmp.write_text(existing + line + "\n")
            os.replace(tmp, target)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def derive_frontier_archive_sha(
    *, repo_root: Path | str = REPO_ROOT_DEFAULT
) -> tuple[str | None, str | None]:
    """Derive the current frontier archive sha + axis from the canonical pointer.

    CPU axis preferred (the public leaderboard ranks by CPU), CUDA fallback.
    Returns ``(archive_sha256, axis)`` or ``(None, None)`` if the pointer is
    absent / unparseable.
    """
    try:
        from tac.canonical_frontier_pointer import (
            load_canonical_frontier_pointer_lenient,
        )
    except (ImportError, ModuleNotFoundError):
        return None, None
    try:
        pointer = load_canonical_frontier_pointer_lenient(repo_root=Path(repo_root))
    except Exception:  # noqa: BLE001 — fail-quiet; pointer is downstream signal
        return None, None
    if pointer is None:
        return None, None
    cpu = pointer.our_local_frontier_contest_cpu
    if cpu is not None and getattr(cpu, "archive_sha256", None):
        return str(cpu.archive_sha256), "contest_cpu"
    cuda = pointer.our_local_frontier_contest_cuda
    if cuda is not None and getattr(cuda, "archive_sha256", None):
        return str(cuda.archive_sha256), "contest_cuda"
    return None, None


def resolve_latest_mlx_artifact_for_sha(
    archive_sha256: str,
    *,
    state_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> str | None:
    """Resolve the latest landed MLX per-pair artifact .npy for an archive sha.

    Reads the canonical MLX research-signal manifest (newest matching row) so
    a consumer can locate the artifact WITHOUT re-deriving the filename
    convention. Falls back to ``None`` (artifact not yet landed).
    """
    if not archive_sha256:
        return None
    manifest = manifest_path or MLX_RESEARCH_SIGNAL_MANIFEST_PATH
    if not manifest.exists():
        return None
    latest: str | None = None
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("archive_sha256") != archive_sha256:
            continue
        npy = obj.get("npy_path")
        if isinstance(npy, str) and npy:
            latest = npy
    return latest


def _run_extraction_inline(
    *,
    archive_path: Path,
    out_path: Path,
    n_pairs: int,
    upstream_dir: Path | None,
    video_path: Path | None,
    manifest_path: Path,
) -> str | None:
    """Run the MLX extraction inline (deep path; heavy — opt-in via run_now).

    Returns the artifact path on success or raises. Used only when the caller
    explicitly opts in via ``run_now=True``; the default seam emits a
    ``scheduled`` row instead so the frontier-refresh hot path stays fast.
    """
    from tac.master_gradient_mlx_extractor import (
        extract_mlx_per_pair_master_gradient,
    )
    import numpy as np

    repo_root = REPO_ROOT_DEFAULT
    upstream = upstream_dir or (repo_root / "upstream")
    video = video_path or (upstream / "videos" / "0.mkv")

    result = extract_mlx_per_pair_master_gradient(
        archive_path,
        upstream_dir=upstream,
        video_path=video,
        n_pairs_used=n_pairs,
        n_pairs_total=DEFAULT_DEEP_TIER_PAIRS,
    )
    _ensure_parent(out_path)
    np.save(out_path, result.per_pair_per_byte)
    # Sidecar meta + manifest are written by the canonical CLI; for the inline
    # path we write a minimal NON-PROMOTABLE manifest row so consumers can find
    # the artifact (the heavy CLI remains the canonical full-provenance path).
    _ensure_parent(manifest_path)
    row = {
        "schema_version": "mlx_research_signal_manifest_v1",
        "archive_sha256": result.archive_sha256,
        "npy_path": str(out_path),
        "n_pairs": result.n_pairs_used,
        "evidence_grade": "macOS-MLX research-signal",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "operating_point": dict(result.operating_point),
        "captured_at_utc": _utc_now(),
        "source": "tac.master_gradient_mlx_pipeline.auto_schedule(run_now=True)",
    }
    with open(_MLX_EXTRACTION_STATE_LOCK_PATH, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            tmp = manifest_path.with_suffix(manifest_path.suffix + f".tmp.{os.getpid()}")
            existing = manifest_path.read_text() if manifest_path.exists() else ""
            tmp.write_text(existing + json.dumps(row, sort_keys=True) + "\n")
            os.replace(tmp, manifest_path)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    return str(out_path)


def auto_schedule_mlx_per_pair_extraction_for_frontier(
    *,
    repo_root: Path | str = REPO_ROOT_DEFAULT,
    n_pairs: int = DEFAULT_FAST_TIER_PAIRS,
    frontier_archive_sha256: str | None = None,
    frontier_axis: str | None = None,
    archive_path: Path | str | None = None,
    run_now: bool = False,
    force: bool = False,
    upstream_dir: Path | str | None = None,
    video_path: Path | str | None = None,
    state_path: Path | None = None,
    manifest_path: Path | None = None,
) -> MLXExtractionScheduleVerdict:
    """Auto-trigger seam: schedule MLX per-pair extraction when frontier changed.

    Canonical seam called from the Catalog #343 frontier-pointer refresh path
    (``auto_refresh_canonical_frontier_after_dispatch_outcome``). Deterministic
    + idempotent + fail-quiet.

    Behavior:

    - Derive the frontier archive sha (CPU preferred, CUDA fallback) unless one
      is explicitly supplied.
    - If the sha has a ``completed`` extraction-state row for the requested tier
      AND ``force`` is False, return a ``no_op`` verdict (idempotent).
    - If an MLX artifact already exists for the sha (per the manifest), record a
      ``completed`` state row pointing at it (de-duplicates re-discovery) and
      return ``no_op`` for re-extraction.
    - Otherwise: when ``run_now`` is False (default), append a ``scheduled`` row
      (the deep extraction is heavy and runs out-of-band via the canonical CLI);
      when ``run_now`` is True, run the extraction inline and append a
      ``completed`` row.

    NEVER raises from the seam: any failure is captured in the verdict's
    ``error`` field and an ``error`` state row is appended.

    Per Catalog #341 the verdict carries ``promotable=False`` +
    ``score_claim=False``: the pipeline is observability-only and NEVER promotes
    the MLX heuristic prior to a score signal.
    """
    root = Path(repo_root)
    state_target = state_path or MLX_EXTRACTION_STATE_PATH
    manifest_target = manifest_path or MLX_RESEARCH_SIGNAL_MANIFEST_PATH

    try:
        sha = frontier_archive_sha256
        axis = frontier_axis
        if sha is None:
            sha, axis = derive_frontier_archive_sha(repo_root=root)

        if not sha:
            return MLXExtractionScheduleVerdict(
                fired=False,
                event_type=EVENT_NO_OP,
                frontier_archive_sha256=None,
                frontier_axis=None,
                n_pairs=n_pairs,
                artifact_path=None,
                state_row_written=False,
                rationale=(
                    "no frontier archive sha resolvable from canonical pointer; "
                    "MLX per-pair extraction not scheduled [predicted]"
                ),
            )

        # Idempotency: already completed OR already scheduled for this sha+tier?
        # A prior ``completed`` row -> the artifact is landed; a prior
        # ``scheduled`` row -> the extraction is pending out-of-band. Either is
        # a no-op (deterministic per the directive); ``force`` overrides.
        prior = latest_extraction_state_for_sha(
            sha, n_pairs=n_pairs, path=state_target
        )
        if (
            prior is not None
            and prior.get("event_type") in (EVENT_COMPLETED, EVENT_SCHEDULED)
            and not force
        ):
            return MLXExtractionScheduleVerdict(
                fired=False,
                event_type=EVENT_NO_OP,
                frontier_archive_sha256=sha,
                frontier_axis=axis,
                n_pairs=n_pairs,
                artifact_path=prior.get("artifact_path"),
                state_row_written=False,
                rationale=(
                    f"frontier archive {sha[:12]} already has a "
                    f"{prior.get('event_type')} MLX per-pair {n_pairs}-pair "
                    f"extraction; idempotent no-op [predicted]"
                ),
            )

        # Already-landed artifact (e.g. landed by the canonical CLI out-of-band)?
        existing_artifact = resolve_latest_mlx_artifact_for_sha(
            sha, manifest_path=manifest_target
        )
        if existing_artifact is not None and not force:
            row = {
                "event_type": EVENT_COMPLETED,
                "frontier_archive_sha256": sha,
                "frontier_axis": axis,
                "n_pairs": n_pairs,
                "artifact_path": existing_artifact,
                "evidence_grade": "macOS-MLX research-signal",
                "promotable": False,
                "score_claim": False,
                "rationale": (
                    "MLX per-pair artifact already landed for frontier sha "
                    "(found via manifest); recording completed state row"
                ),
            }
            _append_state_row_locked(row, path=state_target)
            return MLXExtractionScheduleVerdict(
                fired=True,
                event_type=EVENT_COMPLETED,
                frontier_archive_sha256=sha,
                frontier_axis=axis,
                n_pairs=n_pairs,
                artifact_path=existing_artifact,
                state_row_written=True,
                rationale=(
                    f"MLX per-pair artifact already landed for frontier "
                    f"{sha[:12]}; recorded completed state row [predicted]"
                ),
            )

        if not run_now:
            # Default path: emit a scheduled row. The heavy extraction runs
            # out-of-band via the canonical CLI (idempotent re-run).
            out_rel = (
                f".omx/state/master_gradient_frontier_mlx_per_pair_"
                f"{n_pairs}pair_{sha[:12]}.npy"
            )
            cli_cmd = (
                f".venv/bin/python tools/extract_master_gradient_mlx.py "
                f"--archive {archive_path or _DEFAULT_FRONTIER_ARCHIVE_RELPATH} "
                f"--n-pairs {n_pairs} --out {out_rel} "
                f"--axes seg,pose,rate "
                f"--manifest-jsonl .omx/state/mlx_research_signal_manifest.jsonl"
            )
            row = {
                "event_type": EVENT_SCHEDULED,
                "frontier_archive_sha256": sha,
                "frontier_axis": axis,
                "n_pairs": n_pairs,
                "artifact_path": None,
                "scheduled_out_path": out_rel,
                "scheduled_cli_command": cli_cmd,
                "evidence_grade": "macOS-MLX research-signal",
                "promotable": False,
                "score_claim": False,
                "rationale": (
                    "frontier archive changed; MLX per-pair extraction "
                    "scheduled (run out-of-band via canonical CLI; $0 MLX-local)"
                ),
            }
            _append_state_row_locked(row, path=state_target)
            return MLXExtractionScheduleVerdict(
                fired=True,
                event_type=EVENT_SCHEDULED,
                frontier_archive_sha256=sha,
                frontier_axis=axis,
                n_pairs=n_pairs,
                artifact_path=None,
                state_row_written=True,
                rationale=(
                    f"frontier archive {sha[:12]} changed; scheduled MLX "
                    f"per-pair {n_pairs}-pair extraction (run out-of-band via "
                    f"canonical CLI) [predicted]"
                ),
                extra={"scheduled_cli_command": cli_cmd, "scheduled_out_path": out_rel},
            )

        # run_now=True: inline extraction (heavy; opt-in).
        if archive_path is None:
            archive_path = root / _DEFAULT_FRONTIER_ARCHIVE_RELPATH
        out_rel = (
            root
            / ".omx"
            / "state"
            / f"master_gradient_frontier_mlx_per_pair_{n_pairs}pair_{sha[:12]}.npy"
        )
        artifact = _run_extraction_inline(
            archive_path=Path(archive_path),
            out_path=out_rel,
            n_pairs=n_pairs,
            upstream_dir=Path(upstream_dir) if upstream_dir else None,
            video_path=Path(video_path) if video_path else None,
            manifest_path=manifest_target,
        )
        row = {
            "event_type": EVENT_COMPLETED,
            "frontier_archive_sha256": sha,
            "frontier_axis": axis,
            "n_pairs": n_pairs,
            "artifact_path": artifact,
            "evidence_grade": "macOS-MLX research-signal",
            "promotable": False,
            "score_claim": False,
            "rationale": "MLX per-pair extraction completed inline (run_now=True)",
        }
        _append_state_row_locked(row, path=state_target)
        return MLXExtractionScheduleVerdict(
            fired=True,
            event_type=EVENT_COMPLETED,
            frontier_archive_sha256=sha,
            frontier_axis=axis,
            n_pairs=n_pairs,
            artifact_path=artifact,
            state_row_written=True,
            rationale=(
                f"MLX per-pair {n_pairs}-pair extraction completed inline for "
                f"frontier {sha[:12]} [predicted]"
            ),
        )

    except Exception as exc:  # noqa: BLE001 — NEVER raise from the seam
        try:
            _append_state_row_locked(
                {
                    "event_type": EVENT_ERROR,
                    "frontier_archive_sha256": frontier_archive_sha256,
                    "frontier_axis": frontier_axis,
                    "n_pairs": n_pairs,
                    "artifact_path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "evidence_grade": "macOS-MLX research-signal",
                    "promotable": False,
                    "score_claim": False,
                    "rationale": "MLX per-pair extraction auto-trigger raised",
                },
                path=state_target,
            )
            state_written = True
        except Exception:  # noqa: BLE001 — even the error row write is fail-quiet
            state_written = False
        return MLXExtractionScheduleVerdict(
            fired=False,
            event_type=EVENT_ERROR,
            frontier_archive_sha256=frontier_archive_sha256,
            frontier_axis=frontier_axis,
            n_pairs=n_pairs,
            artifact_path=None,
            state_row_written=state_written,
            rationale="MLX per-pair extraction auto-trigger raised (fail-quiet)",
            error=f"{type(exc).__name__}: {exc}",
        )
