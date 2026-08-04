# SPDX-License-Identifier: MIT
"""Cross-run telemetry harvest and trajectory mining for TM1.

The harvester is intentionally scorer-free: it reads existing JSON/JSONL/Markdown
artifacts and emits normalized frames that downstream schedule/equation work can
consume without re-opening raw run directories.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "crossrun_frame_v1_20260804"

DEFAULT_FILE_HINTS = (
    "telemetry",
    "trajectory",
    "loss_terms",
    "mem_probe",
    "memory",
    "gradient",
    "gradshare",
    "verdict",
    "confound",
    "plateau",
    "receipt",
    "decision",
    "manifest",
    "training_artifact",
    "costate",
)

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "site-packages",
}

NUMERIC_METRIC_HINT_RE = re.compile(
    r"(?:^|_)(?:dseg|d_seg|dpose|d_pose|score|loss|pose_mse|rate|bytes|flips|eta|"
    r"lambda|margin|accepted_frac|epoch|steps|wall|global_step|intervals)(?:$|_)"
)
MEMORY_HINT_RE = re.compile(
    r"(?:^|_)(?:rss|rss_gib|peak_rss|resident|vram|ram|memory|mem|"
    r"mlx_peak_memory|measured_free_memory|maximum_measured_peak_rss)(?:$|_)"
)
GRADIENT_HINT_RE = re.compile(r"(?:grad|gnorm|gradient)")
PER_CLASS_KEYS = {"topology_per_class", "per_class", "perclass", "class_metrics"}


@dataclass(frozen=True)
class CrossRunFrame:
    """One normalized metric/event/config frame from a persisted run artifact."""

    schema_version: str
    frame_type: str
    root: str
    run_id: str
    source_path: str
    source_sha256: str | None
    source_size: int
    record_index: int
    line_number: int | None
    metric: str | None
    value: float | None
    axis: str | None
    event: str | None
    stage: str | None
    epoch: float | None
    t_wall: float | None
    global_step: float | None
    payload: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FileHarvestStats:
    path: str
    parsed_records: int
    emitted_frames: int
    bad_records: int
    skipped_reason: str | None = None


@dataclass(frozen=True)
class HarvestResult:
    frames: list[CrossRunFrame]
    file_stats: list[FileHarvestStats]

    def to_summary(self) -> dict[str, Any]:
        parsed = [s for s in self.file_stats if s.skipped_reason is None]
        skipped = [s for s in self.file_stats if s.skipped_reason is not None]
        return {
            "schema_version": SCHEMA_VERSION,
            "files_considered": len(self.file_stats),
            "files_parsed": len(parsed),
            "files_skipped": len(skipped),
            "records_parsed": sum(s.parsed_records for s in parsed),
            "bad_records": sum(s.bad_records for s in parsed),
            "frames_emitted": len(self.frames),
            "runs": len({f.run_id for f in self.frames}),
            "frame_types": dict(Counter(f.frame_type for f in self.frames)),
            "metrics_top20": dict(Counter(f.metric for f in self.frames if f.metric).most_common(20)),
            "skipped_reasons": dict(Counter(s.skipped_reason for s in skipped if s.skipped_reason)),
        }


@dataclass(frozen=True)
class AnalysisIndex:
    frames: list[CrossRunFrame]
    run_ids: tuple[str, ...]
    by_run_metric: dict[tuple[str, str], list[tuple[float, float, CrossRunFrame]]]
    event_frames: list[CrossRunFrame]
    by_run_events: dict[str, list[CrossRunFrame]]


def candidate_files(
    roots: Iterable[Path],
    *,
    file_hints: Iterable[str] = DEFAULT_FILE_HINTS,
    max_files: int | None = None,
) -> list[Path]:
    """Return artifact files whose names indicate run telemetry/receipts."""

    hints = tuple(h.lower() for h in file_hints)
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            out.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                p = Path(dirpath) / name
                suffix = p.suffix.lower()
                if suffix not in {".json", ".jsonl", ".md"}:
                    continue
                low = str(p).lower()
                if any(h in low for h in hints):
                    out.append(p)
                    if max_files is not None and len(out) >= max_files:
                        return out
    return out


def harvest_roots(
    roots: Iterable[Path],
    *,
    max_files: int | None = None,
    max_bytes_per_file: int = 8_000_000,
    max_records_per_file: int = 50_000,
    hash_files: bool = True,
) -> HarvestResult:
    files = candidate_files(roots, max_files=max_files)
    frames: list[CrossRunFrame] = []
    stats: list[FileHarvestStats] = []
    for p in files:
        file_frames, file_stats = harvest_file(
            p,
            roots=tuple(roots),
            max_bytes_per_file=max_bytes_per_file,
            max_records_per_file=max_records_per_file,
            hash_file=hash_files,
        )
        frames.extend(file_frames)
        stats.append(file_stats)
    return HarvestResult(frames=frames, file_stats=stats)


def harvest_file(
    path: Path,
    *,
    roots: tuple[Path, ...],
    max_bytes_per_file: int,
    max_records_per_file: int,
    hash_file: bool,
) -> tuple[list[CrossRunFrame], FileHarvestStats]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        return [], FileHarvestStats(str(path), 0, 0, 0, f"stat_failed:{exc}")
    if size > max_bytes_per_file:
        return [], FileHarvestStats(str(path), 0, 0, 0, "too_large")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], FileHarvestStats(str(path), 0, 0, 0, f"read_failed:{exc}")

    source_hash = _sha256_text(text) if hash_file else None
    records, bad = _load_records(text, path.suffix.lower(), max_records=max_records_per_file)
    root = _matching_root(path, roots)
    run_id = _run_id(path, root)
    frames: list[CrossRunFrame] = []
    for idx, (line_number, record) in enumerate(records):
        frames.extend(
            _frames_for_record(
                record,
                root=root,
                run_id=run_id,
                source_path=path,
                source_sha256=source_hash,
                source_size=size,
                record_index=idx,
                line_number=line_number,
            )
        )
    return frames, FileHarvestStats(str(path), len(records), len(frames), bad)


def write_frames_jsonl(frames: Iterable[CrossRunFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for frame in frames:
            f.write(json.dumps(frame.to_json_dict(), sort_keys=True, separators=(",", ":")) + "\n")


def load_frames_jsonl(path: Path) -> list[CrossRunFrame]:
    out: list[CrossRunFrame] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(CrossRunFrame(**row))
    return out


def analyze_frames(frames: Iterable[CrossRunFrame]) -> dict[str, Any]:
    frame_list = list(frames)
    index = _build_analysis_index(frame_list)
    base = {
        "schema_version": "crossrun_analysis_v1_20260804",
        "frame_count": len(frame_list),
        "run_count": len(index.run_ids),
        "frame_types": dict(Counter(f.frame_type for f in frame_list)),
        "metrics": dict(Counter(f.metric for f in frame_list if f.metric).most_common(40)),
    }
    base["seg_constancy"] = _seg_constancy(index)
    base["per_class_topology"] = _per_class_topology(index)
    base["plateau_census"] = _plateau_census(index)
    base["warm_resume"] = _warm_resume(index)
    base["lever_impulses"] = _lever_impulses(index)
    base["memory_envelope"] = _memory_envelope(index)
    base["event_timing"] = _event_timing(index)
    base["stopping_policy"] = _stopping_policy(index)
    return base


def _build_analysis_index(frames: list[CrossRunFrame]) -> AnalysisIndex:
    by_run_metric: dict[tuple[str, str], list[tuple[float, float, CrossRunFrame]]] = defaultdict(list)
    by_run_events: dict[str, list[CrossRunFrame]] = defaultdict(list)
    event_frames: list[CrossRunFrame] = []
    run_ids = set()
    for f in frames:
        run_ids.add(f.run_id)
        if f.metric is not None and f.value is not None:
            by_run_metric[(f.run_id, f.metric)].append((_order_key(f), float(f.value), f))
        if f.metric is None and f.event:
            event_frames.append(f)
            by_run_events[f.run_id].append(f)
    for seq in by_run_metric.values():
        seq.sort(key=lambda x: (x[0], x[2].record_index))
    for seq in by_run_events.values():
        seq.sort(key=lambda f: (_order_key(f), f.record_index))
    event_frames.sort(key=lambda f: (f.run_id, _order_key(f), f.record_index))
    return AnalysisIndex(
        frames=frames,
        run_ids=tuple(sorted(run_ids)),
        by_run_metric=dict(by_run_metric),
        event_frames=event_frames,
        by_run_events=dict(by_run_events),
    )


def _frames_for_record(
    record: dict[str, Any],
    *,
    root: Path,
    run_id: str,
    source_path: Path,
    source_sha256: str | None,
    source_size: int,
    record_index: int,
    line_number: int | None,
) -> list[CrossRunFrame]:
    event = _string_or_none(record.get("event"))
    stage = _string_or_none(record.get("stage") or record.get("stage_name") or record.get("tr1_stage"))
    axis = _string_or_none(record.get("evidence_axis") or record.get("axis_tag") or record.get("authority_tag"))
    epoch = _float_or_none(record.get("epoch") or record.get("ep") or record.get("global_epoch"))
    t_wall = _float_or_none(record.get("t_wall") or record.get("wall_clock_s"))
    global_step = _float_or_none(record.get("global_step"))
    common = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "run_id": run_id,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_size": source_size,
        "record_index": record_index,
        "line_number": line_number,
        "axis": axis,
        "event": event,
        "stage": stage,
        "epoch": epoch,
        "t_wall": t_wall,
        "global_step": global_step,
    }

    frames: list[CrossRunFrame] = []
    dense_metrics = _path_is_dense_metric_source(source_path, event)
    if (event and event != "epoch") or (_path_event_hint(source_path) and not dense_metrics):
        frames.append(
            CrossRunFrame(
                **common,
                frame_type=_event_frame_type(event, source_path),
                metric=None,
                value=None,
                payload=_compact_payload(record),
            )
        )

    cfg = record.get("cfg")
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            num = _float_or_none(value)
            if num is None:
                continue
            frames.append(
                CrossRunFrame(
                    **common,
                    frame_type="config",
                    metric=f"cfg.{key}",
                    value=num,
                    payload={"config_key": key},
                )
            )

    terms = record.get("terms")
    if isinstance(terms, dict):
        for key, value in terms.items():
            num = _float_or_none(value)
            if num is None:
                continue
            frames.append(
                CrossRunFrame(
                    **common,
                    frame_type="loss_term",
                    metric=f"loss_term.{key}",
                    value=num,
                    payload={"term": key},
                )
            )

    if dense_metrics:
        for key, value in record.items():
            if key in {"cfg", "terms"} or isinstance(value, (dict, list)):
                continue
            num = _float_or_none(value)
            if num is None:
                continue
            metric = str(key)
            if metric in {"epoch", "ep", "global_epoch", "t_wall", "wall_clock_s", "global_step"}:
                continue
            if not _metric_name_is_relevant(metric):
                continue
            frames.append(
                CrossRunFrame(
                    **common,
                    frame_type=_numeric_frame_type(metric),
                    metric=metric,
                    value=num,
                    payload={},
                )
            )

        for metric, value, payload in _nested_metric_frames(record):
            frames.append(
                CrossRunFrame(
                    **common,
                    frame_type=_numeric_frame_type(metric),
                    metric=metric,
                    value=value,
                    payload=payload,
                )
            )
    return frames


def _load_records(
    text: str,
    suffix: str,
    *,
    max_records: int,
) -> tuple[list[tuple[int | None, dict[str, Any]]], int]:
    if suffix == ".jsonl":
        return _load_jsonl_records(text, max_records=max_records)
    if suffix == ".md":
        return _load_markdown_json_records(text, max_records=max_records)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _load_jsonl_records(text, max_records=max_records)
    return _records_from_json_obj(parsed, max_records=max_records), 0


def _load_jsonl_records(text: str, *, max_records: int) -> tuple[list[tuple[int | None, dict[str, Any]]], int]:
    rows: list[tuple[int | None, dict[str, Any]]] = []
    bad = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if len(rows) >= max_records:
            break
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append((lineno, obj))
        else:
            bad += 1
    return rows, bad


def _load_markdown_json_records(text: str, *, max_records: int) -> tuple[list[tuple[int | None, dict[str, Any]]], int]:
    rows: list[tuple[int | None, dict[str, Any]]] = []
    bad = 0
    fence_re = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    for match in fence_re.finditer(text):
        if len(rows) >= max_records:
            break
        line_number = text.count("\n", 0, match.start()) + 1
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            bad += 1
            continue
        for rec_line, rec in _records_from_json_obj(obj, max_records=max_records - len(rows)):
            rows.append((line_number if rec_line is None else line_number + rec_line - 1, rec))
    return rows, bad


def _records_from_json_obj(obj: Any, *, max_records: int) -> list[tuple[int | None, dict[str, Any]]]:
    if isinstance(obj, dict):
        return [(None, obj)]
    if isinstance(obj, list):
        return [(None, x) for x in obj[:max_records] if isinstance(x, dict)]
    return []


def _nested_metric_frames(record: dict[str, Any]) -> list[tuple[str, float, dict[str, Any]]]:
    frames: list[tuple[str, float, dict[str, Any]]] = []
    for container_key in PER_CLASS_KEYS:
        obj = record.get(container_key)
        if not isinstance(obj, dict):
            continue
        for metric, value in obj.items():
            if isinstance(value, list):
                for class_idx, item in enumerate(value):
                    num = _float_or_none(item)
                    if num is not None:
                        frames.append(
                            (
                                f"{container_key}.{metric}.class_{class_idx}",
                                num,
                                {"container": container_key, "class_index": class_idx},
                            )
                        )
            elif isinstance(value, dict):
                for class_key, item in value.items():
                    num = _float_or_none(item)
                    if num is not None:
                        frames.append(
                            (
                                f"{container_key}.{metric}.{class_key}",
                                num,
                                {"container": container_key, "class_key": class_key},
                            )
                        )
    return frames


def _seg_constancy(index: AnalysisIndex) -> dict[str, Any]:
    priority = ("realized_gate_dseg_mean", "d_seg")
    rows: list[dict[str, Any]] = []
    for run_id in index.run_ids:
        for metric in priority:
            series = _metric_series(index, run_id, metric)
            if len(series) < 2:
                continue
            vals = [v for _, v, _ in series]
            first, last = vals[0], vals[-1]
            mean_abs = sum(abs(v) for v in vals) / len(vals)
            rel_delta = (last - first) / first if first else None
            rel_span = (max(vals) - min(vals)) / mean_abs if mean_abs else None
            rows.append(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "n": len(vals),
                    "first": first,
                    "last": last,
                    "min": min(vals),
                    "max": max(vals),
                    "relative_delta": rel_delta,
                    "relative_span": rel_span,
                    "smooth_fit": _fit_exp_floor(series),
                }
            )
            break
    constant = [r for r in rows if r.get("relative_span") is not None and abs(r["relative_span"]) <= 0.02]
    return {
        "runs_with_seg_series": len(rows),
        "runs_constant_within_2pct": len(constant),
        "rows": rows[:30],
    }


def _per_class_topology(index: AnalysisIndex) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (run_id, metric), seq in index.by_run_metric.items():
        if not metric.startswith("topology_per_class."):
            continue
        if len(seq) < 2:
            continue
        vals = [v for _, v, _ in seq]
        rows.append(
            {
                "run_id": run_id,
                "metric": metric,
                "n": len(vals),
                "first": vals[0],
                "last": vals[-1],
                "delta": vals[-1] - vals[0],
                "min": min(vals),
                "max": max(vals),
            }
        )
    return {"series_count": len(rows), "rows": rows[:40]}


def _plateau_census(index: AnalysisIndex) -> dict[str, Any]:
    classifications = Counter()
    confounds = Counter()
    confound_runs = set()
    for f in index.event_frames:
        if f.event == "a1_gate" and f.metric is None:
            c = f.payload.get("a1_classification")
            if isinstance(c, str):
                classifications[c] += 1
        if f.event == "confound_alarm" and f.metric is None:
            confound_runs.add(f.run_id)
            kind = f.payload.get("kind", "unknown")
            term = f.payload.get("term", "")
            confounds[f"{kind}:{term}"] += 1

    plateau_segments = []
    for run_id in index.run_ids:
        series = _metric_series(index, run_id, "realized_gate_dseg_mean")
        if len(series) < 2:
            continue
        for (x0, v0, f0), (x1, v1, f1) in zip(series, series[1:]):
            denom = max(abs(v0), 1e-12)
            rel = (v1 - v0) / denom
            if abs(rel) <= 0.005:
                label = "unclassified"
                cls = f1.payload.get("a1_classification") if f1.payload else None
                if isinstance(cls, str) and "ALARM" in cls:
                    label = "spike_guard_or_realization_gap"
                elif run_id in confound_runs:
                    label = "confound_alarm_present"
                plateau_segments.append(
                    {
                        "run_id": run_id,
                        "x0": x0,
                        "x1": x1,
                        "v0": v0,
                        "v1": v1,
                        "relative_delta": rel,
                        "classification": label,
                    }
                )
    return {
        "a1_classifications": dict(classifications),
        "confound_alarms": dict(confounds),
        "plateau_segments_threshold_abs_rel_lte_0p005": len(plateau_segments),
        "plateau_segment_classes": dict(Counter(s["classification"] for s in plateau_segments)),
        "examples": plateau_segments[:20],
    }


def _warm_resume(index: AnalysisIndex) -> dict[str, Any]:
    resumes = [f for f in index.event_frames if f.event == "resume" and f.metric is None]
    rows = []
    for resume in resumes:
        series = _best_loss_series(index, resume.run_id)
        after = [(x, v, f) for x, v, f in series if x >= _order_key(resume)]
        if len(after) < 3:
            rows.append(
                {
                    "run_id": resume.run_id,
                    "resume_x": _order_key(resume),
                    "status": "DATA_INSUFFICIENT",
                    "missing": ">=3 post-resume loss samples",
                }
            )
            continue
        vals = [v for _, v, _ in after]
        floor = min(vals)
        start = vals[0]
        target = floor + 0.5 * (start - floor)
        hit_x = next((x for x, v, _ in after if v <= target), None)
        rows.append(
            {
                "run_id": resume.run_id,
                "loss_metric": after[0][2].metric,
                "n_post_resume": len(after),
                "start": start,
                "floor_observed": floor,
                "half_descent_target": target,
                "half_descent_x": hit_x,
                "half_descent_delta_x": None if hit_x is None else hit_x - _order_key(resume),
                "smooth_fit": _fit_exp_floor(after),
                "status": "MEASURED" if hit_x is not None else "NO_HALF_DESCENT_IN_OBSERVED_WINDOW",
            }
        )
    return {"resume_events": len(resumes), "rows": rows}


def _lever_impulses(index: AnalysisIndex) -> dict[str, Any]:
    lever_events = [
        f
        for f in index.event_frames
        if f.event
        and f.metric is None
        and (
            "guard" in f.event
            or "lever" in f.event
            or f.event.endswith("_init")
            or f.event in {"resume_form_reanchor", "telemetry_v9_port"}
        )
    ]
    counts = Counter(f.event for f in lever_events)
    impulse_rows = []
    for ev in lever_events:
        series = _best_loss_series(index, ev.run_id)
        before = [(x, v, f) for x, v, f in series if x < _order_key(ev)]
        after = [(x, v, f) for x, v, f in series if x > _order_key(ev)]
        if len(before) < 2 or len(after) < 2:
            continue
        impulse_rows.append(
            {
                "run_id": ev.run_id,
                "event": ev.event,
                "x": _order_key(ev),
                "metric": after[0][2].metric,
                "slope_before": _slope(before[-3:]),
                "slope_after": _slope(after[:3]),
            }
        )
    return {
        "lever_event_counts": dict(counts),
        "impulses_with_before_after": len(impulse_rows),
        "rows": impulse_rows[:40],
        "data_insufficiency": (
            "Most lever events occur at run start or without enough comparable pre/post "
            "samples for a population transfer function."
        )
        if len(impulse_rows) < len(lever_events)
        else None,
    }


def _memory_envelope(index: AnalysisIndex) -> dict[str, Any]:
    memory = [
        f
        for f in index.frames
        if f.metric is not None and _metric_name_is_memory(f.metric) and f.value is not None
    ]
    by_metric = Counter(f.metric for f in memory)
    rows = []
    for run_id in sorted({f.run_id for f in memory}):
        vals = [f for f in memory if f.run_id == run_id and f.value is not None]
        if not vals:
            continue
        peak = max(vals, key=lambda f: f.value if f.value is not None else -math.inf)
        rows.append({"run_id": run_id, "metric": peak.metric, "peak_value": peak.value})
    status = "DATA_INSUFFICIENT"
    if len(rows) >= 2:
        status = "MEASURED_NO_CONFIG_FIT"
    if len(rows) >= 2 and any(metric.startswith("cfg.") for _, metric in index.by_run_metric):
        status = "MEASURED_CONFIG_JOIN_AVAILABLE"
    return {
        "status": status,
        "memory_frame_count": len(memory),
        "memory_metrics": dict(by_metric),
        "rows": rows[:30],
    }


def _event_timing(index: AnalysisIndex) -> dict[str, Any]:
    rows = []
    for run_id in index.run_ids:
        gate_every = _first_metric_value(index, run_id, "cfg.gate_every")
        gates = [
            f
            for f in index.by_run_events.get(run_id, [])
            if f.run_id == run_id and f.event == "a1_gate" and f.metric is None and f.epoch is not None
        ]
        gates.sort(key=lambda f: f.epoch or 0.0)
        if gate_every is None or len(gates) < 2:
            continue
        gaps = [(b.epoch or 0.0) - (a.epoch or 0.0) for a, b in zip(gates, gates[1:])]
        errors = [g - gate_every for g in gaps]
        rows.append(
            {
                "run_id": run_id,
                "gate_every": gate_every,
                "n_gates": len(gates),
                "gaps": gaps[:20],
                "max_abs_epoch_error": max(abs(e) for e in errors) if errors else 0.0,
                "all_gaps_match_cfg": all(abs(e) <= 1e-9 for e in errors),
            }
        )
    return {
        "a1_gate_cadence_runs": len(rows),
        "all_cadence_matched_runs": sum(1 for r in rows if r["all_gaps_match_cfg"]),
        "rows": rows[:30],
    }


def _stopping_policy(index: AnalysisIndex) -> dict[str, Any]:
    decision_events = [
        f
        for f in index.event_frames
        if f.metric is None
        and ("decision" in Path(f.source_path).name.lower()
        or (f.event and any(token in f.event for token in ("stop", "cap", "abort", "terminal")))
        )
    ]
    by_run = Counter(f.run_id for f in decision_events)
    return {
        "decision_or_stop_event_count": len(decision_events),
        "runs_with_decision_or_stop": len(by_run),
        "top_runs": dict(by_run.most_common(20)),
        "status": "DATA_INSUFFICIENT_FOR_COUNTERFACTUAL_DESCENT",
        "missing": (
            "A stopping-policy validation needs an explicit stop point plus post-stop "
            "counterfactual continuation on the same objective; those pairs are not "
            "present in the normalized frames."
        ),
    }


def _metric_series(
    index: AnalysisIndex,
    run_id: str,
    metric: str,
) -> list[tuple[float, float, CrossRunFrame]]:
    return list(index.by_run_metric.get((run_id, metric), ()))


def _best_loss_series(index: AnalysisIndex, run_id: str) -> list[tuple[float, float, CrossRunFrame]]:
    for metric in ("ep_loss", "loss", "loss_term.seg", "pose_mse"):
        seq = _metric_series(index, run_id, metric)
        if len(seq) >= 2:
            return seq
    return []


def _fit_exp_floor(series: list[tuple[float, float, CrossRunFrame]]) -> dict[str, Any]:
    """Fit y = floor + amplitude * exp(-k * (x - x0)) with closed-form log slope."""

    if len(series) < 3:
        return {"status": "DATA_INSUFFICIENT", "family": "exponential_floor"}
    xs = [x for x, _, _ in series]
    ys = [y for _, y, _ in series]
    floor = min(ys)
    shifted = [max(y - floor, 1e-12) for y in ys]
    x0 = xs[0]
    logs = [math.log(v) for v in shifted]
    denom = sum((x - x0) ** 2 for x in xs)
    if denom <= 0:
        return {"status": "DATA_INSUFFICIENT", "family": "exponential_floor"}
    slope = sum((x - x0) * (l - logs[0]) for x, l in zip(xs, logs)) / denom
    k = -slope
    pred = [floor + shifted[0] * math.exp(-k * (x - x0)) for x in xs]
    rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(pred, ys)) / len(ys))
    span = max(ys) - min(ys)
    return {
        "status": "MEASURED_FIT",
        "family": "exponential_floor",
        "floor_observed": floor,
        "amplitude": shifted[0],
        "k": k,
        "rmse": rmse,
        "rmse_over_span": None if span == 0 else rmse / span,
        "n": len(series),
    }


def _slope(series: list[tuple[float, float, CrossRunFrame]]) -> float | None:
    if len(series) < 2:
        return None
    x0, y0, _ = series[0]
    x1, y1, _ = series[-1]
    if x1 == x0:
        return None
    return (y1 - y0) / (x1 - x0)


def _first_metric_value(index: AnalysisIndex, run_id: str, metric: str) -> float | None:
    for _, v, _ in _metric_series(index, run_id, metric):
        return v
    return None


def _order_key(frame: CrossRunFrame) -> float:
    if frame.epoch is not None:
        return float(frame.epoch)
    if frame.global_step is not None:
        return float(frame.global_step)
    if frame.t_wall is not None:
        return float(frame.t_wall)
    return float(frame.record_index)


def _matching_root(path: Path, roots: tuple[Path, ...]) -> Path:
    resolved = path.resolve()
    best: Path | None = None
    for root in roots:
        try:
            root_resolved = root.resolve()
            resolved.relative_to(root_resolved)
        except ValueError:
            continue
        if best is None or len(str(root_resolved)) > len(str(best.resolve())):
            best = root
    return best if best is not None else path.parent


def _run_id(path: Path, root: Path) -> str:
    try:
        rel_parent = path.parent.resolve().relative_to(root.resolve())
    except ValueError:
        rel_parent = path.parent
    parts = rel_parent.parts
    if not parts:
        return path.parent.name
    if path.name == "telemetry.jsonl" and len(parts) >= 2:
        return "::".join(parts[:2])
    return "::".join(parts[: min(len(parts), 3)])


def _event_frame_type(event: str | None, path: Path) -> str:
    low_path = str(path).lower()
    low_event = (event or "").lower()
    if "checkpoint" in low_path:
        return "checkpoint"
    if "verdict" in low_path or "verdict" in low_event or "gate" in low_event:
        return "verdict"
    if "confound" in low_path or "confound" in low_event:
        return "confound"
    return "event"


def _path_event_hint(path: Path) -> bool:
    low = str(path).lower()
    return any(token in low for token in ("decision", "receipt", "manifest", "verdict", "checkpoint"))


def _path_is_dense_metric_source(path: Path, event: str | None) -> bool:
    low = str(path).lower()
    if event in {"a1_gate", "lane_guard", "loss_terms", "confound_alarm", "positive_control"}:
        return True
    return any(
        token in low
        for token in (
            "telemetry",
            "trajectory",
            "loss_terms",
            "progress",
            ".partial.jsonl",
            "verdict",
            "confound",
            "costate",
            "mem_probe",
            "gradient",
        )
    )


def _metric_name_is_relevant(metric: str) -> bool:
    low = metric.lower()
    return (
        NUMERIC_METRIC_HINT_RE.search(low) is not None
        or _metric_name_is_memory(metric)
        or GRADIENT_HINT_RE.search(low) is not None
        or low
        in {
            "d_seg",
            "d_pose",
            "score",
            "rate",
            "archive_bytes",
            "ep_loss",
            "pose_mse",
            "n_intervals",
            "realized_gate_dseg_mean",
        }
    )


def _numeric_frame_type(metric: str) -> str:
    low = metric.lower()
    if _metric_name_is_memory(metric):
        return "memory"
    if GRADIENT_HINT_RE.search(low):
        return "gradient"
    if low in {"d_seg", "d_pose", "score", "rate", "archive_bytes"}:
        return "score_metric"
    if NUMERIC_METRIC_HINT_RE.search(low):
        return "trajectory_metric"
    return "numeric_metric"


def _metric_name_is_memory(metric: str) -> bool:
    return MEMORY_HINT_RE.search(metric.lower()) is not None


def _compact_payload(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "<truncated>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= 80:
                out["<truncated_keys>"] = len(value) - idx
                break
            out[str(k)] = _compact_payload(v, depth=depth + 1)
        return out
    if isinstance(value, list):
        out = [_compact_payload(v, depth=depth + 1) for v in value[:16]]
        if len(value) > 16:
            out.append({"<truncated_items>": len(value) - 16})
        return out
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "...<truncated>"
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
