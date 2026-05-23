"""Canonical frontier-anchor scan helpers.

Per CLAUDE.md "tac stays clean; comma-lab owns research state": this module
is the reusable library; ``tools/scan_best_anchor_per_axis.py`` is the
thin CLI. Other operator surfaces can consume the same module instead of
reimplementing anchor parsing.

This module exists to extinct the **frontier-signal-loss bug class**: the
2026-05-15 anchor at 0.19205 [contest-CPU] (archive ``6bae0201...``) and
0.20533 [contest-CUDA] (archive ``9cb989cef519...``) sat in
``.omx/state/continual_learning_posterior.json`` and
``.omx/state/active_lane_dispatch_claims.md`` for 2 days while
``reports/latest.md``, ``MEMORY.md`` index, and conversation memory kept
citing the stale PR101 GOLD 0.193 baseline. Per CLAUDE.md "Apples-to-apples
evidence discipline" non-negotiable: every promotion / planning / dispatch
decision must apples-to-apples against the actual best anchor in state,
not against a citation that drifted.

The committed surface is intentionally small: canonical parser/payload/render
helpers plus one CLI. Preflight/autopilot/operator-authorize consumers should
wire this module directly when those surfaces become the next bottleneck.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE" non-negotiable: only anchors whose
``hardware_substrate`` is in ``QUALIFYING_HARDWARE`` (Linux x86_64 +
recognised GPU class) qualify for the frontier. macOS-CPU advisory rows
+ MPS-noise rows are explicitly excluded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "AXIS_LABELS",
    "FRONTIER_CITATION_SURFACE_PATHS",
    "QUALIFYING_AXES",
    "QUALIFYING_HARDWARE",
    "Anchor",
    "DriftRow",
    "best_per_axis",
    "build_cpu_axis_optimal_payload",
    "build_frontier_scan_payload",
    "collect_all_anchors",
    "cpu_axis_family_for_anchor",
    "detect_drift",
    "load_active_lane_dispatch_claims_anchors",
    "load_continual_learning_anchors",
    "load_experiments_results_anchors",
    "load_modal_call_id_ledger_anchors",
    "render_frontier_scan_json",
    "render_frontier_scan_text",
    "scan_frontier_citation_surface",
    "scan_reports_latest_md",
    "select_cpu_optimal_per_family",
]

# Per CLAUDE.md "Submission auth eval" non-negotiable: only 1:1
# contest-compliant hardware qualifies for frontier promotion.
QUALIFYING_HARDWARE: tuple[str, ...] = (
    "linux_x86_64_cpu",
    "linux_x86_64_gha_cpu",
    "linux_x86_64_t4",
    "linux_x86_64_a10g",
    "linux_x86_64_a100",
    "linux_x86_64_4090",
    "linux_x86_64_h100",
    "linux_x86_64_l40s",
)
QUALIFYING_AXES: tuple[str, ...] = ("contest_cpu", "contest_cuda", "cpu", "cuda")

AXIS_LABELS = {
    "contest_cpu": "[contest-CPU GHA Linux x86_64]",
    "contest_cuda": "[contest-CUDA T4]",
}
FRONTIER_CITATION_SURFACE_PATHS: tuple[str, ...] = (
    "reports/latest.md",
    ".omx/state/current_focus.md",
    ".omx/state/next_experiments.md",
)
_AXIS_TOKENS: tuple[tuple[str, str], ...] = (
    ("contest-cpu", "contest_cpu"),
    ("contest_cpu", "contest_cpu"),
    ("contest-cuda", "contest_cuda"),
    ("contest_cuda", "contest_cuda"),
)


@dataclass
class Anchor:
    """One frontier-qualifying score anchor.

    ``axis`` is normalized lowercase; call ``canonical_axis()`` for the
    ``contest_cpu`` / ``contest_cuda`` namespace key used in downstream
    dicts. ``is_qualifying()`` mirrors CLAUDE.md "Submission auth eval"
    1:1 contest-hardware filter (Linux x86_64 + recognised GPU class).
    """

    score: float
    axis: str
    archive_sha256: str
    hardware_substrate: str
    source_path: str
    extra: dict = field(default_factory=dict)

    def canonical_axis(self) -> str:
        if self.axis in ("contest_cpu", "cpu"):
            return "contest_cpu"
        if self.axis in ("contest_cuda", "cuda"):
            return "contest_cuda"
        return self.axis

    def is_qualifying(self) -> bool:
        return (
            self.axis in QUALIFYING_AXES
            and self.hardware_substrate.lower() in QUALIFYING_HARDWARE
            and self.score > 0
        )


def load_continual_learning_anchors(repo_root: Path | str) -> list[Anchor]:
    repo_root = Path(repo_root)
    path = repo_root / ".omx/state/continual_learning_posterior.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    rows: list = []
    if isinstance(data, dict):
        for key in ("anchors", "accepted_anchor_history"):
            value = data.get(key)
            if isinstance(value, list):
                rows.extend(value)
    elif isinstance(data, list):
        rows = data
    out: list[Anchor] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        score = row.get("score_value") or row.get("score")
        axis = row.get("axis") or row.get("score_axis")
        sha = row.get("archive_sha256")
        hw = row.get("hardware_substrate") or row.get("hw_substrate") or ""
        if score is None or axis is None or not sha:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        out.append(
            Anchor(
                score=score_f,
                axis=str(axis).lower(),
                archive_sha256=str(sha),
                hardware_substrate=str(hw).lower(),
                source_path=str(path.relative_to(repo_root)),
                extra={
                    "evidence_grade": row.get("evidence_grade"),
                    "evidence_tag": row.get("evidence_tag"),
                    "measured_at_utc": (
                        row.get("measured_at_utc")
                        or row.get("promoted_at_utc")
                        or row.get("observed_at_utc")
                    ),
                    "lane_id": row.get("lane_id"),
                    "architecture_class": row.get("architecture_class"),
                    "archive_id": row.get("archive_id"),
                    "label": row.get("label"),
                    "archive_bytes": row.get("archive_bytes"),
                },
            )
        )
    return out


def load_modal_call_id_ledger_anchors(repo_root: Path | str) -> list[Anchor]:
    repo_root = Path(repo_root)
    path = repo_root / ".omx/state/modal_call_id_ledger.jsonl"
    if not path.is_file():
        return []
    out: list[Anchor] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        score = row.get("score")
        axis = row.get("score_axis")
        sha = row.get("archive_sha256")
        if score is None or axis is None or not sha:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        out.append(
            Anchor(
                score=score_f,
                axis=str(axis).lower(),
                archive_sha256=str(sha),
                hardware_substrate=str(row.get("hardware_substrate") or "").lower(),
                source_path=str(path.relative_to(repo_root)),
                extra={
                    "evidence_grade": row.get("evidence_grade"),
                    "dispatched_at_utc": row.get("dispatched_at_utc"),
                    "lane_id": row.get("lane_id"),
                    "call_id": row.get("call_id"),
                },
            )
        )
    return out


_CLAIM_ROW_RE = re.compile(
    r"score_recomputed=([0-9.e\-+]+);\s*axis=(\w+);\s*"
    r"hardware_substrate=(\w+);\s*posterior_update=accepted",
    re.IGNORECASE,
)
_CLAIM_SHA_RE = re.compile(r"archive_sha=([0-9a-f]{64})", re.IGNORECASE)
_CLAIM_SHA_ALT_RE = re.compile(r"archive_sha256=([0-9a-f]{64})", re.IGNORECASE)
_CLAIM_LANE_RE = re.compile(r"\|\s*(lane_[a-z0-9_]+?)\s*\|", re.IGNORECASE)


def load_active_lane_dispatch_claims_anchors(repo_root: Path | str) -> list[Anchor]:
    repo_root = Path(repo_root)
    path = repo_root / ".omx/state/active_lane_dispatch_claims.md"
    if not path.is_file():
        return []
    out: list[Anchor] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        m = _CLAIM_ROW_RE.search(line)
        if not m:
            continue
        try:
            score_f = float(m.group(1))
        except ValueError:
            continue
        axis = m.group(2).lower()
        hw = m.group(3).lower()
        sha_m = _CLAIM_SHA_RE.search(line) or _CLAIM_SHA_ALT_RE.search(line)
        if not sha_m:
            continue
        lane_m = _CLAIM_LANE_RE.search(line)
        lane_id = lane_m.group(1) if lane_m else None
        out.append(
            Anchor(
                score=score_f,
                axis=axis,
                archive_sha256=sha_m.group(1),
                hardware_substrate=hw,
                source_path=str(path.relative_to(repo_root)),
                extra={"posterior_update": "accepted", "lane_id": lane_id},
            )
        )
    return out


def load_experiments_results_anchors(repo_root: Path | str) -> list[Anchor]:
    repo_root = Path(repo_root)
    base = repo_root / "experiments/results"
    if not base.is_dir():
        return []
    out: list[Anchor] = []
    patterns = (
        "modal_auth_eval/archive_*/modal_*_auth_eval_result.json",
        "modal_auth_eval_cpu/archive_*/modal_*_auth_eval_result.json",
        "*/contest_auth_eval*.json",
        "*/auth_eval*.json",
    )
    for pattern in patterns:
        for path in base.glob(pattern):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            score = (
                data.get("score")
                or data.get("final_score")
                or data.get("contest_score")
                or data.get("auth_eval_score")
                or data.get("canonical_score")
            )
            axis = (
                data.get("score_axis") or data.get("axis") or data.get("evidence_grade")
            )
            sha = data.get("archive_sha256") or data.get("archive_sha")
            hw = data.get("hardware_substrate") or data.get("hw_substrate") or ""
            if score is None or axis is None or not sha:
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            axis_str = str(axis).lower()
            if "cpu" in axis_str:
                axis_str = "contest_cpu"
            elif "cuda" in axis_str:
                axis_str = "contest_cuda"
            out.append(
                Anchor(
                    score=score_f,
                    axis=axis_str,
                    archive_sha256=str(sha),
                    hardware_substrate=str(hw).lower(),
                    source_path=str(path.relative_to(repo_root)),
                    extra={
                        "evidence_grade": data.get("evidence_grade"),
                        "promotion_eligible": data.get("promotion_eligible"),
                    },
                )
            )
    return out


def collect_all_anchors(repo_root: Path | str) -> list[Anchor]:
    """Scan every canonical anchor source and return Anchor records.

    Accepts ``repo_root`` as either ``Path`` or ``str``; normalizes via
    ``Path(repo_root)`` for downstream loader functions. Sister normalization
    in each loader keeps the API symmetric so callers can pass either type
    interchangeably without producing the historical ``TypeError`` when a
    bare string was concatenated with the ``/`` operator.
    """
    repo_root = Path(repo_root)
    anchors: list[Anchor] = []
    anchors.extend(load_continual_learning_anchors(repo_root))
    anchors.extend(load_modal_call_id_ledger_anchors(repo_root))
    anchors.extend(load_active_lane_dispatch_claims_anchors(repo_root))
    anchors.extend(load_experiments_results_anchors(repo_root))
    return anchors


def best_per_axis(anchors: list[Anchor]) -> dict[str, list[Anchor]]:
    """Group qualifying anchors by canonical axis, sorted lowest-first.

    Output: ``{"contest_cpu": [Anchor, ...], "contest_cuda": [Anchor, ...]}``
    where each list is sorted ascending (best score first).
    """
    qualifying = [a for a in anchors if a.is_qualifying()]
    by_axis: dict[str, list[Anchor]] = {}
    for a in qualifying:
        by_axis.setdefault(a.canonical_axis(), []).append(a)
    for lst in by_axis.values():
        lst.sort(key=lambda a: a.score)
    return by_axis


_CPU_AXIS_FAMILY_PREFIXES: tuple[str, ...] = (
    "pr101",
    "pr102",
    "pr103",
    "pr106",
    "pr107",
    "a1",
)


def cpu_axis_family_for_anchor(anchor: Anchor) -> str:
    """Return the metadata bucket used by G1 CPU-axis reranking.

    The bucket is derived only from custody metadata that already exists on
    frontier anchors. Unknown/null lane ids intentionally fall into ``other``
    rather than guessing from archive SHA.
    """

    candidates = (
        anchor.extra.get("lane_id"),
        anchor.extra.get("architecture_class"),
        anchor.extra.get("archive_id"),
        anchor.extra.get("label"),
        anchor.source_path,
    )
    for raw in candidates:
        if not raw:
            continue
        text = str(raw).lower()
        for token in re.split(r"[^a-z0-9]+", text):
            if token in _CPU_AXIS_FAMILY_PREFIXES:
                return token
    return "other"


def select_cpu_optimal_per_family(anchors: list[Anchor]) -> dict[str, Anchor]:
    """Select the lowest qualifying contest-CPU anchor for each family."""

    grouped: dict[str, list[Anchor]] = {}
    for anchor in anchors:
        if anchor.canonical_axis() != "contest_cpu" or not anchor.is_qualifying():
            continue
        grouped.setdefault(cpu_axis_family_for_anchor(anchor), []).append(anchor)
    return {
        family: sorted(family_anchors, key=lambda anchor: anchor.score)[0]
        for family, family_anchors in sorted(grouped.items())
        if family_anchors
    }


def build_cpu_axis_optimal_payload(
    anchors: list[Anchor],
    *,
    current_frontier_cpu: float | None = None,
) -> dict[str, object]:
    """Build the G1 CPU-axis reranking payload.

    This is evidence routing, not a promotion claim. ``improvement_found`` is
    true only when an existing qualifying CPU anchor beats the current CPU
    frontier by more than numerical tolerance.
    """

    qualifying_cpu_anchors = [
        anchor
        for anchor in anchors
        if anchor.canonical_axis() == "contest_cpu" and anchor.is_qualifying()
    ]
    per_family = select_cpu_optimal_per_family(anchors)
    overall = None
    if per_family:
        overall = sorted(per_family.values(), key=lambda anchor: anchor.score)[0]
    if current_frontier_cpu is None:
        best = best_per_axis(anchors).get("contest_cpu") or []
        current_frontier_cpu = best[0].score if best else None
    delta = None
    improvement_found = False
    if overall is not None and current_frontier_cpu is not None:
        delta = overall.score - current_frontier_cpu
        improvement_found = delta < -1e-9
    blockers = []
    if not qualifying_cpu_anchors:
        blockers.append("no_qualifying_contest_cpu_anchors")
    if current_frontier_cpu is None:
        blockers.append("current_frontier_cpu_missing")
    return {
        "schema": "g1_cpu_axis_optimal_archive_v1",
        "axis": "contest_cpu",
        "evidence_grade": "[contest-CPU]",
        "input_anchor_count": len(anchors),
        "qualifying_cpu_anchor_count": len(qualifying_cpu_anchors),
        "excluded_anchor_count": len(anchors) - len(qualifying_cpu_anchors),
        "metadata_bucket_method": "metadata_token_match_no_archive_sha_guess",
        "family_method": "metadata_token_match_no_archive_sha_guess",
        "current_frontier_cpu": current_frontier_cpu,
        "overall_cpu_optimal": None if overall is None else _serialize_anchor(overall),
        "delta_vs_current_frontier": delta,
        "improvement_found": improvement_found,
        "existing_anchor_selection_valid": overall is not None,
        "new_score_claim_valid": False,
        "score_claim_kind": "existing_anchor_rerank_no_new_score_claim",
        "blockers": blockers,
        "per_family_optimal": {
            family: _serialize_anchor(anchor)
            for family, anchor in sorted(per_family.items())
        },
        "per_metadata_bucket_optimal": {
            family: _serialize_anchor(anchor)
            for family, anchor in sorted(per_family.items())
        },
    }


def _record_cited_score(cited: dict[str, float], axis_key: str, score: float) -> None:
    prev = cited.get(axis_key)
    if prev is None or score < prev:
        cited[axis_key] = score


def _score_before_axis_citations(text: str, cited: dict[str, float]) -> None:
    for axis_token, axis_key in _AXIS_TOKENS:
        pattern = re.compile(
            r"\b(0\.[0-9]+)[\s`*_]*\[?\s*" + re.escape(axis_token),
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            try:
                _record_cited_score(cited, axis_key, float(m.group(1)))
            except ValueError:
                continue


def _markdown_table_citations(text: str, cited: dict[str, float]) -> None:
    for line in text.splitlines():
        _markdown_table_line_citations(line, cited)


def _markdown_table_line_citations(line: str, cited: dict[str, float]) -> None:
    if not line.lstrip().startswith("|"):
        return
    line_lower = line.lower()
    for axis_token, axis_key in _AXIS_TOKENS:
        if axis_token not in line_lower:
            continue
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        score_cells = [
            cell
            for index, cell in enumerate(cells)
            if axis_token in cell.lower()
            for cell in cells[index + 1 : index + 2]
        ]
        for cell in score_cells:
            m = re.search(r"\b(0\.[0-9]+)\b", cell)
            if not m:
                continue
            try:
                _record_cited_score(cited, axis_key, float(m.group(1)))
            except ValueError:
                continue


_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_CURRENT_BEST_HEADING_RE = re.compile(r"^\s{0,3}#{2,6}\s+current\s+best\b", re.I)


def _reports_latest_current_best_table_citations(text: str) -> dict[str, float] | None:
    """Parse the authoritative reports/latest.md current-frontier table.

    ``reports/latest.md`` is a long historical control surface. It deliberately
    carries advisory deltas and old comparisons that are axis-tagged prose, so
    the best-score citation for Catalog #316 must come from the generated
    ``### Current best`` table when that table exists.
    """

    cited: dict[str, float] = {}
    in_current_best = False
    saw_table = False
    for line in text.splitlines():
        if not in_current_best:
            if _CURRENT_BEST_HEADING_RE.match(line):
                in_current_best = True
            continue
        if _MARKDOWN_HEADING_RE.match(line):
            break
        if not line.lstrip().startswith("|"):
            if saw_table and line.strip():
                break
            continue
        saw_table = True
        _markdown_table_line_citations(line, cited)
    return cited if in_current_best else None


def _previous_line_axis_citations(text: str, cited: dict[str, float]) -> None:
    """Parse state-doc bullets where score line is followed by axis line.

    This intentionally inspects only the immediately previous line when the
    axis line itself has no decimal score. It catches:

        - Best anchor: `0.192...`
          `[contest-CPU; ...]`

    without misreading comparison deltas such as "we beat by 0.00095" later in
    a prose line that also names `[contest-CPU]`.
    """

    lines = text.splitlines()
    for index, line in enumerate(lines):
        line_lower = line.lower()
        if re.search(r"\b0\.[0-9]+\b", line):
            continue
        for axis_token, axis_key in _AXIS_TOKENS:
            if axis_token not in line_lower or index == 0:
                continue
            scores = re.findall(r"\b0\.[0-9]+\b", lines[index - 1])
            if len(scores) != 1:
                continue
            try:
                _record_cited_score(cited, axis_key, float(scores[0]))
            except ValueError:
                continue


def scan_frontier_citation_surface(repo_root: Path | str, relative_path: str) -> dict[str, float]:
    """Return best CPU/CUDA scores cited in a Markdown control surface."""

    repo_root = Path(repo_root)
    path = repo_root / relative_path
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    cited: dict[str, float] = {}
    _score_before_axis_citations(text, cited)
    _markdown_table_citations(text, cited)
    _previous_line_axis_citations(text, cited)
    return cited


def scan_reports_latest_md(repo_root: Path | str) -> dict[str, float]:
    """Return the best CPU + CUDA scores currently cited in reports/latest.md.

    Output: ``{"contest_cpu": 0.197..., "contest_cuda": 0.231...}`` (only
    keys for axes that have a recognised citation in the file). Used by
    Catalog #316 to detect drift between citation surface and canonical
    state.
    """
    repo_root = Path(repo_root)
    path = repo_root / "reports/latest.md"
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    current_best = _reports_latest_current_best_table_citations(text)
    if current_best is not None:
        return current_best
    return scan_frontier_citation_surface(repo_root, "reports/latest.md")


@dataclass(frozen=True)
class DriftRow:
    axis: str
    cited_score: float
    best_score: float
    best_anchor: Anchor

    @property
    def delta(self) -> float:
        """Positive number = state has a better score than what is cited."""
        return self.cited_score - self.best_score


def detect_drift(
    best: dict[str, list[Anchor]],
    cited: dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> list[DriftRow]:
    """Detect axes where the canonical best beats the citation by > tolerance."""
    out: list[DriftRow] = []
    for axis, anchors in best.items():
        if not anchors:
            continue
        best_anchor = anchors[0]
        cited_score = cited.get(axis)
        if cited_score is None:
            continue
        if cited_score - best_anchor.score > tolerance:
            out.append(
                DriftRow(
                    axis=axis,
                    cited_score=cited_score,
                    best_score=best_anchor.score,
                    best_anchor=best_anchor,
                )
            )
    return out


def _serialize_anchor(anchor: Anchor) -> dict[str, object]:
    return {
        "score": anchor.score,
        "axis": anchor.canonical_axis(),
        "archive_sha256": anchor.archive_sha256,
        "hardware_substrate": anchor.hardware_substrate,
        "source_path": anchor.source_path,
        "extra": anchor.extra,
    }


def build_frontier_scan_payload(repo_root: Path | str) -> dict[str, object]:
    """Return the canonical best-anchor scan payload for operator surfaces.

    Accepts ``repo_root`` as either ``Path`` or ``str``; normalizes via
    ``Path(repo_root)`` so the canonical pointer model + DX auto-update
    surfaces can pass either type interchangeably.
    """

    repo_root = Path(repo_root)
    all_anchors = collect_all_anchors(repo_root)
    best = best_per_axis(all_anchors)
    citation_surfaces = {}
    for path in FRONTIER_CITATION_SURFACE_PATHS:
        if path == "reports/latest.md":
            citation_surfaces[path] = scan_reports_latest_md(repo_root)
        else:
            citation_surfaces[path] = scan_frontier_citation_surface(repo_root, path)
    cited = citation_surfaces.get("reports/latest.md", {})
    surface_drift = {
        path: [
            {
                "axis": row.axis,
                "cited_score": row.cited_score,
                "best_score": row.best_score,
                "delta_better": row.delta,
                "best_anchor": _serialize_anchor(row.best_anchor),
            }
            for row in detect_drift(best, surface_cited)
        ]
        for path, surface_cited in citation_surfaces.items()
    }
    drift = detect_drift(best, cited)
    return {
        "schema": "pact_frontier_scan_v1",
        "best_per_axis": {
            axis: _serialize_anchor(anchors[0])
            for axis, anchors in best.items()
            if anchors
        },
        "top_5_per_axis": {
            axis: [_serialize_anchor(anchor) for anchor in anchors[:5]]
            for axis, anchors in best.items()
        },
        "frontier_citation_surfaces": citation_surfaces,
        "frontier_citation_surface_drift": surface_drift,
        "reports_latest_md_cited": cited,
        "g1_cpu_axis_optimization": build_cpu_axis_optimal_payload(all_anchors),
        "drift": [
            {
                "axis": row.axis,
                "cited_score": row.cited_score,
                "best_score": row.best_score,
                "delta_better": row.delta,
                "best_anchor": _serialize_anchor(row.best_anchor),
            }
            for row in drift
        ],
        "scan_stats": {
            "total_anchors": len(all_anchors),
            "qualifying": sum(1 for anchor in all_anchors if anchor.is_qualifying()),
            "excluded": sum(1 for anchor in all_anchors if not anchor.is_qualifying()),
        },
    }


def render_frontier_scan_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def render_frontier_scan_text(payload: Mapping[str, object]) -> str:
    lines: list[str] = [
        "=" * 72,
        "CURRENT BEST ANCHOR PER AXIS (qualifying = 1:1 contest hardware)",
        "=" * 72,
    ]
    best = payload.get("best_per_axis")
    top5 = payload.get("top_5_per_axis")
    cited = payload.get("reports_latest_md_cited")
    surfaces = payload.get("frontier_citation_surfaces")
    drift = payload.get("drift")
    surface_drift = payload.get("frontier_citation_surface_drift")
    stats = payload.get("scan_stats")
    best_map = best if isinstance(best, dict) else {}
    top5_map = top5 if isinstance(top5, dict) else {}
    for axis in ("contest_cpu", "contest_cuda"):
        top = best_map.get(axis)
        if not isinstance(top, dict):
            lines.append(f"\n  {axis}: <no qualifying anchor>")
            continue
        score = float(top.get("score") or 0.0)
        archive_sha256 = str(top.get("archive_sha256") or "")
        hardware = str(top.get("hardware_substrate") or "")
        source = str(top.get("source_path") or "")
        lines.append(f"\n  {axis}: BEST = {score:.10f}")
        lines.append(f"    archive_sha256 = {archive_sha256}")
        lines.append(f"    hardware       = {hardware}")
        lines.append(f"    source         = {source}")
        extra = top.get("extra")
        if isinstance(extra, dict) and extra.get("lane_id"):
            lines.append(f"    lane_id        = {extra['lane_id']}")
        rows = top5_map.get(axis)
        if isinstance(rows, list) and len(rows) > 1:
            lines.append("    top-5 (best to worst):")
            for index, row in enumerate(rows[:5], 1):
                if not isinstance(row, dict):
                    continue
                row_score = float(row.get("score") or 0.0)
                row_sha = str(row.get("archive_sha256") or "")
                row_hw = str(row.get("hardware_substrate") or "")
                lines.append(
                    f"      {index}. {row_score:.10f}  {row_sha[:12]}  {row_hw}"
                )
    if isinstance(cited, dict) and cited:
        lines.extend(["", "=" * 72, "REPORTS/LATEST.MD CITED FRONTIER", "=" * 72])
        for axis, score in sorted(cited.items()):
            lines.append(f"  {axis}: {float(score):.6f}")
    if isinstance(surfaces, dict) and surfaces:
        lines.extend(["", "=" * 72, "FRONTIER CITATION SURFACES", "=" * 72])
        for path, row in sorted(surfaces.items()):
            if not isinstance(row, dict):
                continue
            rendered = ", ".join(
                f"{axis}={float(score):.6f}" for axis, score in sorted(row.items())
            )
            lines.append(f"  {path}: {rendered or '<no recognized citations>'}")
    if isinstance(drift, list) and drift:
        lines.extend(["", "=" * 72, "FRONTIER DRIFT", "=" * 72])
        for row in drift:
            if not isinstance(row, dict):
                continue
            axis = str(row.get("axis") or "")
            cited_score = float(row.get("cited_score") or 0.0)
            best_score = float(row.get("best_score") or 0.0)
            delta = float(row.get("delta_better") or 0.0)
            best_anchor = row.get("best_anchor")
            anchor_sha = (
                str(best_anchor.get("archive_sha256") or "")[:16]
                if isinstance(best_anchor, dict)
                else ""
            )
            lines.append(
                f"  {axis}: cited={cited_score:.6f}; "
                f"actual={best_score:.6f}; delta={delta:+.6f} better in state"
            )
            lines.append(f"    promote: archive {anchor_sha}")
    surface_drift_rows: list[dict[str, object]] = []
    if isinstance(surface_drift, dict):
        for path, rows in sorted(surface_drift.items()):
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        surface_drift_rows.append({"path": path, **row})
    if surface_drift_rows:
        lines.extend(["", "=" * 72, "FRONTIER CITATION SURFACE DRIFT", "=" * 72])
        for row in surface_drift_rows:
            axis = str(row.get("axis") or "")
            path = str(row.get("path") or "")
            cited_score = float(row.get("cited_score") or 0.0)
            best_score = float(row.get("best_score") or 0.0)
            delta = float(row.get("delta_better") or 0.0)
            lines.append(
                f"  {path}: {axis} cited={cited_score:.6f}; "
                f"actual={best_score:.6f}; delta={delta:+.6f}"
            )
    if isinstance(stats, dict):
        lines.append("")
        lines.append(
            "Scanned: "
            f"{int(stats.get('total_anchors') or 0)} anchors total; "
            f"{int(stats.get('qualifying') or 0)} qualifying; "
            f"{int(stats.get('excluded') or 0)} excluded"
        )
    g1 = payload.get("g1_cpu_axis_optimization")
    if isinstance(g1, dict):
        lines.extend(["", "=" * 72, "G1 CPU-AXIS OPTIMIZATION", "=" * 72])
        overall = g1.get("overall_cpu_optimal")
        if isinstance(overall, dict):
            score = float(overall.get("score") or 0.0)
            delta = g1.get("delta_vs_current_frontier")
            delta_s = "n/a" if delta is None else f"{float(delta):+.10f}"
            sha = str(overall.get("archive_sha256") or "")
            extra = overall.get("extra")
            lane_id = extra.get("lane_id") if isinstance(extra, dict) else None
            lines.append(f"  best existing CPU-axis archive = {score:.10f}")
            lines.append(f"  delta vs current frontier      = {delta_s}")
            lines.append(f"  archive_sha256                 = {sha}")
            if lane_id:
                lines.append(f"  lane_id                        = {lane_id}")
            lines.append(
                "  verdict                        = "
                + (
                    "improvement_found"
                    if bool(g1.get("improvement_found"))
                    else "current_frontier_remains_cpu_optimal"
                )
            )
            lines.append(
                "  score_claim_kind               = "
                + str(g1.get("score_claim_kind") or "unknown")
            )
            lines.append(
                "  qualifying_cpu_anchor_count    = "
                + str(g1.get("qualifying_cpu_anchor_count") or 0)
            )
        per_family = g1.get("per_family_optimal")
        if isinstance(per_family, dict) and per_family:
            lines.append("  per-metadata-bucket CPU optima:")
            for family, row in sorted(per_family.items()):
                if not isinstance(row, dict):
                    continue
                score = float(row.get("score") or 0.0)
                sha = str(row.get("archive_sha256") or "")[:12]
                extra = row.get("extra")
                lane_id = ""
                if isinstance(extra, dict) and extra.get("lane_id"):
                    lane_id = f"  {extra['lane_id']}"
                lines.append(f"    {family}: {score:.10f}  {sha}{lane_id}")
    return "\n".join(lines)
