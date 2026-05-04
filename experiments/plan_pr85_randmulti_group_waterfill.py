#!/usr/bin/env python3
"""Plan PR85 randmulti group-level water-fill policies.

This is a planning-only tool. It decomposes the public PR85 randmulti
side-channel into replay groups and sparse row atoms, allocates existing exact
negative component traces across those groups, and emits deterministic JSON
policy candidates. It does not build archives, load scorers, claim scores,
write dispatch state, or launch remote jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli

REPO_ROOT = Path(__file__).resolve().parents[1]
RECODE_BUILDER_PATH = REPO_ROOT / "experiments" / "build_pr85_sidechannel_recode_candidates.py"
TOOL = "experiments/plan_pr85_randmulti_group_waterfill.py"
SCHEMA = "pr85_randmulti_group_waterfill_plan_v1"
GROUP_LEDGER_SCHEMA = "pr85_randmulti_group_exact_negative_ledger_v1"
POLICY_SCHEMA = "pr85_randmulti_group_policy_candidates_v1"
EVIDENCE_GRADE = "planning_only_from_exact_negative_cuda_traces"
EXPECTED_CONTEST_SAMPLES = 600
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
PAIR_COUNT = 600

DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_BASELINE_EVAL_DIR = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z"
)
DEFAULT_MINUS_RANDMULTI_EVAL_DIR = REPO_ROOT / (
    "experiments/results/lightning_batch/exact_eval_pr85_minus_randmulti_t4_20260504T0002Z"
)
DEFAULT_MINUS_POST_EVAL_DIR = REPO_ROOT / (
    "experiments/results/lightning_batch/exact_eval_pr85_minus_post_t4_20260504T0002Z"
)
DEFAULT_MINUS_MOTION_EVAL_DIR = REPO_ROOT / (
    "experiments/results/lightning_batch/exact_eval_pr85_minus_motion_stack_t4_20260504T0002Z"
)
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_randmulti_group_waterfill_20260504_codex"
DEFAULT_TOPKS = (1, 2, 4, 8, 16, 32, 48, 64, 72)
DEFAULT_BUDGET_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


class PlannerError(ValueError):
    """Raised when a planning input is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class AuthMetrics:
    """Score terms extracted from exact auth eval JSON."""

    label: str
    path: Path
    archive_bytes: int
    score: float
    pose_dist: float
    seg_dist: float
    n_samples: int
    rate_score: float
    pose_score: float
    seg_score: float

    @property
    def component_score(self) -> float:
        return self.pose_score + self.seg_score


@dataclass(frozen=True)
class TraceSample:
    """Per-pair score contribution terms from component_trace.json."""

    pair_index: int
    seg_score: float
    pose_score: float

    @property
    def component_score(self) -> float:
        return self.seg_score + self.pose_score


@dataclass(frozen=True)
class TraceProfile:
    """Validated exact component trace."""

    label: str
    path: Path
    archive_bytes: int
    n_samples: int
    archive_sha256: str | None
    samples: tuple[TraceSample, ...]


@dataclass(frozen=True)
class RandmultiAtom:
    """One sparse selector row inside a PR85 randmulti group."""

    group_index: int
    row_index: int
    raw_payload_bytes: int
    nonzero_choice_count: int
    nonzero_value_sum: int
    active_pair_indices: tuple[int, ...]


@dataclass(frozen=True)
class RandmultiGroup:
    """Decoded PR85 randmulti replay group."""

    group_index: int
    height: int
    width: int
    amplitude: int
    scount: int
    rows: tuple[bytes, ...]
    raw_payload_bytes: int
    atoms: tuple[RandmultiAtom, ...]

    @property
    def nonzero_choice_total(self) -> int:
        return sum(atom.nonzero_choice_count for atom in self.atoms)

    @property
    def active_pair_count(self) -> int:
        active = set()
        for atom in self.atoms:
            active.update(atom.active_pair_indices)
        return len(active)


def _load_recode_module() -> Any:
    spec = importlib.util.spec_from_file_location("pr85_recode_for_randmulti_waterfill", RECODE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise PlannerError(f"could not load PR85 recode helper from {_rel(RECODE_BUILDER_PATH)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recode = _load_recode_module()
HEADERLESS_RANDMULTI_SPECS: tuple[tuple[int, int, int, int], ...] = tuple(
    tuple(int(value) for value in row) for row in recode.HEADERLESS_RANDMULTI_SPECS
)


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise PlannerError(f"required JSON input is missing: {_rel(path)}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlannerError(f"required JSON input is invalid: {_rel(path)}") from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        raise PlannerError(f"required custody file is missing: {_rel(path)}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannerError(f"{_rel(path)} missing numeric field {field}")
    out = float(value)
    if not math.isfinite(out):
        raise PlannerError(f"{_rel(path)} has non-finite field {field}")
    return out


def _int_field(payload: dict[str, Any], field: str, *, path: Path) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerError(f"{_rel(path)} missing integer field {field}")
    return int(value)


def _score_field(payload: dict[str, Any], *, path: Path) -> float:
    for field in ("score_recomputed_from_components", "canonical_score", "final_score"):
        if field in payload:
            return _finite_float(payload[field], field=field, path=path)
    raise PlannerError(f"{_rel(path)} missing recomputed score")


def _round(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _load_auth_metrics(path: Path, *, label: str) -> AuthMetrics:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{_rel(path)} must contain a JSON object")
    metrics = AuthMetrics(
        label=label,
        path=path,
        archive_bytes=_int_field(payload, "archive_size_bytes", path=path),
        score=_score_field(payload, path=path),
        pose_dist=_finite_float(payload.get("avg_posenet_dist"), field="avg_posenet_dist", path=path),
        seg_dist=_finite_float(payload.get("avg_segnet_dist"), field="avg_segnet_dist", path=path),
        n_samples=_int_field(payload, "n_samples", path=path),
        rate_score=_finite_float(payload.get("score_rate_contribution"), field="score_rate_contribution", path=path),
        pose_score=_finite_float(payload.get("score_pose_contribution"), field="score_pose_contribution", path=path),
        seg_score=_finite_float(payload.get("score_seg_contribution"), field="score_seg_contribution", path=path),
    )
    if metrics.n_samples != EXPECTED_CONTEST_SAMPLES:
        raise PlannerError(
            f"{_rel(path)} has n_samples={metrics.n_samples}, expected {EXPECTED_CONTEST_SAMPLES}"
        )
    return metrics


def _load_trace(path: Path, *, label: str, auth: AuthMetrics) -> TraceProfile:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{_rel(path)} must contain a JSON object")
    archive_bytes = _int_field(payload, "archive_size_bytes", path=path)
    if archive_bytes != auth.archive_bytes:
        raise PlannerError(
            f"{_rel(path)} archive_size_bytes={archive_bytes} does not match {label} auth bytes={auth.archive_bytes}"
        )
    n_samples = _int_field(payload, "n_samples", path=path)
    if n_samples != auth.n_samples:
        raise PlannerError(f"{_rel(path)} n_samples does not match {label} auth JSON")
    samples_raw = payload.get("samples")
    if not isinstance(samples_raw, list) or len(samples_raw) != n_samples:
        raise PlannerError(f"{_rel(path)} samples must contain {n_samples} rows")
    samples_by_pair: dict[int, TraceSample] = {}
    for row in samples_raw:
        if not isinstance(row, dict):
            raise PlannerError(f"{_rel(path)} samples must be objects")
        pair_index = _int_field(row, "pair_index", path=path)
        if not 0 <= pair_index < n_samples:
            raise PlannerError(f"{_rel(path)} pair_index out of range: {pair_index}")
        if pair_index in samples_by_pair:
            raise PlannerError(f"{_rel(path)} duplicate pair_index: {pair_index}")
        samples_by_pair[pair_index] = TraceSample(
            pair_index=pair_index,
            seg_score=_finite_float(
                row.get("score_seg_contribution_exact"),
                field="score_seg_contribution_exact",
                path=path,
            ),
            pose_score=_finite_float(
                row.get("score_pose_contribution_first_order"),
                field="score_pose_contribution_first_order",
                path=path,
            ),
        )
    missing = sorted(set(range(n_samples)) - set(samples_by_pair))
    if missing:
        raise PlannerError(f"{_rel(path)} missing pair indices, first missing={missing[:8]}")
    trace_inputs = payload.get("trace_inputs")
    archive_sha = None
    if isinstance(trace_inputs, dict) and isinstance(trace_inputs.get("archive_sha256"), str):
        archive_sha = trace_inputs["archive_sha256"]
    return TraceProfile(
        label=label,
        path=path,
        archive_bytes=archive_bytes,
        n_samples=n_samples,
        archive_sha256=archive_sha,
        samples=tuple(samples_by_pair[index] for index in range(n_samples)),
    )


def _auth_summary(metrics: AuthMetrics) -> dict[str, Any]:
    return {
        "archive_bytes": metrics.archive_bytes,
        "auth_json": _rel(metrics.path),
        "avg_posenet_dist": _round(metrics.pose_dist),
        "avg_segnet_dist": _round(metrics.seg_dist),
        "canonical_score": _round(metrics.score),
        "component_score": _round(metrics.component_score),
        "n_samples": metrics.n_samples,
        "rate_score": _round(metrics.rate_score),
        "score_pose_contribution": _round(metrics.pose_score),
        "score_seg_contribution": _round(metrics.seg_score),
    }


def _segment_negative_summary(baseline: AuthMetrics, variant: AuthMetrics) -> dict[str, Any]:
    component_delta = variant.component_score - baseline.component_score
    rate_delta = variant.rate_score - baseline.rate_score
    score_delta = variant.score - baseline.score
    byte_delta = variant.archive_bytes - baseline.archive_bytes
    return {
        "archive_byte_delta_vs_baseline": int(byte_delta),
        "component_score_delta_minus_vs_baseline": _round(component_delta),
        "exact_negative_component_value_of_removed_segment": _round(component_delta),
        "rate_score_delta_minus_vs_baseline": _round(rate_delta),
        "score_delta_minus_vs_baseline": _round(score_delta),
        "score_delta_recomposition_residual": _round(score_delta - component_delta - rate_delta),
        "score_direction": "worse_when_removed" if score_delta > 0 else "better_when_removed_or_noop",
    }


def _pair_component_deltas(
    baseline: TraceProfile,
    variant: TraceProfile,
    *,
    exact_component_delta: float,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    if baseline.n_samples != variant.n_samples:
        raise PlannerError(f"{variant.label} trace sample count does not match baseline")
    raw = tuple(
        variant.samples[index].component_score - baseline.samples[index].component_score
        for index in range(baseline.n_samples)
    )
    raw_sum = sum(raw)
    if abs(raw_sum) <= 1e-15:
        scale = 0.0
        calibrated = tuple(0.0 for _ in raw)
    else:
        scale = exact_component_delta / raw_sum
        calibrated = tuple(value * scale for value in raw)
    residual = exact_component_delta - sum(calibrated)
    positive = sum(value for value in calibrated if value > 0.0)
    negative = sum(value for value in calibrated if value < 0.0)
    return calibrated, {
        "exact_component_delta": _round(exact_component_delta),
        "raw_trace_component_delta_sum": _round(raw_sum),
        "calibration_scale": _round(scale),
        "calibrated_component_delta_sum": _round(sum(calibrated)),
        "calibration_residual": _round(residual),
        "positive_pair_delta_sum": _round(positive),
        "negative_pair_delta_sum": _round(negative),
    }


def _read_varints(raw: bytes, pos: int, count: int) -> tuple[list[int], int]:
    values: list[int] = []
    for _ in range(count):
        acc = 0
        shift = 0
        while True:
            if pos >= len(raw):
                raise PlannerError("truncated randmulti varint stream")
            byte = raw[pos]
            pos += 1
            acc |= (byte & 0x7F) << shift
            if byte & 0x80:
                shift += 7
                if shift > 63:
                    raise PlannerError("overlong randmulti varint stream")
            else:
                values.append(acc)
                break
    return values, pos


def _write_varint(value: int) -> bytes:
    if value < 0:
        raise PlannerError(f"cannot encode negative varint: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_randmulti_row(raw: bytes, pos: int) -> tuple[bytes, int, int]:
    start = pos
    if pos >= len(raw):
        raise PlannerError("randmulti stream ended before count byte")
    count = int(raw[pos])
    pos += 1
    if count == 255:
        if pos + 2 > len(raw):
            raise PlannerError("truncated extended randmulti count")
        count = int.from_bytes(raw[pos : pos + 2], "little")
        pos += 2
    gaps, pos = _read_varints(raw, pos, count)
    values = raw[pos : pos + count]
    pos += count
    if len(values) != count:
        raise PlannerError("truncated randmulti values")
    row = bytearray(PAIR_COUNT)
    idx = -1
    for gap, value in zip(gaps, values, strict=True):
        idx += gap + 1
        if not 0 <= idx < PAIR_COUNT:
            raise PlannerError(f"randmulti sparse index out of range: {idx}")
        row[idx] = value
    return bytes(row), pos - start, pos


def _row_atom(group_index: int, row_index: int, row: bytes, raw_payload_bytes: int) -> RandmultiAtom:
    active = tuple(index for index, value in enumerate(row) if value)
    return RandmultiAtom(
        group_index=group_index,
        row_index=row_index,
        raw_payload_bytes=raw_payload_bytes,
        nonzero_choice_count=len(active),
        nonzero_value_sum=sum(int(row[index]) for index in active),
        active_pair_indices=active,
    )


def _decode_pr85_randmulti_groups(decoded_raw: bytes) -> tuple[RandmultiGroup, ...]:
    pos = 0
    groups: list[RandmultiGroup] = []
    for group_index, (height, width, amplitude, scount) in enumerate(HEADERLESS_RANDMULTI_SPECS):
        group_start = pos
        rows: list[bytes] = []
        atoms: list[RandmultiAtom] = []
        for row_index in range(int(scount)):
            row, payload_bytes, pos = _decode_randmulti_row(decoded_raw, pos)
            rows.append(row)
            atoms.append(_row_atom(group_index, row_index, row, payload_bytes))
        groups.append(
            RandmultiGroup(
                group_index=group_index,
                height=int(height),
                width=int(width),
                amplitude=int(amplitude),
                scount=int(scount),
                rows=tuple(rows),
                raw_payload_bytes=pos - group_start,
                atoms=tuple(atoms),
            )
        )
    if pos != len(decoded_raw):
        raise PlannerError("randmulti stream has trailing bytes after PR85 schedule")
    return tuple(groups)


def _encode_sparse_row(row: bytes) -> bytes:
    indices = [index for index, value in enumerate(row) if value]
    out = bytearray()
    if len(indices) >= 255:
        out.append(255)
        out += len(indices).to_bytes(2, "little")
    else:
        out.append(len(indices))
    previous = -1
    for index in indices:
        out += _write_varint(index - previous - 1)
        previous = index
    out += bytes(row[index] for index in indices)
    return bytes(out)


def _encode_selected_headerless_raw(groups: Sequence[RandmultiGroup], selected_group_ids: set[int]) -> bytes:
    by_id = {group.group_index: group for group in groups}
    out = bytearray()
    for group_index, (_height, _width, _amplitude, scount) in enumerate(HEADERLESS_RANDMULTI_SPECS):
        group = by_id[group_index]
        rows = group.rows if group_index in selected_group_ids else tuple(bytes(PAIR_COUNT) for _ in range(scount))
        for row in rows:
            out += _encode_sparse_row(row)
    return bytes(out)


def _brotli_best(raw: bytes) -> tuple[bytes, dict[str, int | str]]:
    return recode._brotli_best(raw)


def _compressed_len_for_groups(groups: Sequence[RandmultiGroup], selected_group_ids: set[int]) -> tuple[int, str]:
    encoded, _params = _brotli_best(_encode_selected_headerless_raw(groups, selected_group_ids))
    return len(encoded), _sha256_bytes(encoded)


def _load_source_randmulti(archive: Path) -> tuple[dict[str, Any], dict[str, Any], tuple[RandmultiGroup, ...]]:
    source_archive, raw = recode._read_pr85_archive(archive)
    source_bundle, source_segments = recode._parse_bundle(raw)
    randmulti_segment = source_segments["randmulti"]
    decoded = brotli.decompress(randmulti_segment)
    groups = _decode_pr85_randmulti_groups(decoded)
    recode_groups = recode._decode_randmulti_groups(decoded)
    if len(recode_groups) != len(groups):
        raise PlannerError("local randmulti decoder group count does not match PR85 recode helper")
    return (
        source_archive,
        {
            "bundle_format": source_bundle["format"],
            "header_bytes": int(source_bundle["header_bytes"]),
            "segment_lengths": source_bundle["segment_lengths"],
            "randmulti_segment_bytes": len(randmulti_segment),
            "randmulti_segment_sha256": _sha256_bytes(randmulti_segment),
            "randmulti_decoded_bytes": len(decoded),
            "randmulti_decoded_sha256": _sha256_bytes(decoded),
        },
        groups,
    )


def _pair_group_weights(groups: Sequence[RandmultiGroup]) -> list[dict[int, int]]:
    per_pair: list[dict[int, int]] = [{} for _ in range(PAIR_COUNT)]
    for group in groups:
        for atom in group.atoms:
            for pair_index in atom.active_pair_indices:
                per_pair[pair_index][group.group_index] = per_pair[pair_index].get(group.group_index, 0) + 1
    return per_pair


def _group_peer_overlap(pair_values: Sequence[float], group: RandmultiGroup) -> float:
    weighted_sum = 0.0
    weight_total = 0
    for atom in group.atoms:
        for pair_index in atom.active_pair_indices:
            weighted_sum += pair_values[pair_index]
            weight_total += 1
    if weight_total == 0:
        return 0.0
    return weighted_sum / weight_total


def _allocate_group_values(
    groups: Sequence[RandmultiGroup],
    randmulti_pair_values: Sequence[float],
    post_pair_values: Sequence[float],
    motion_pair_values: Sequence[float],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    weights_by_pair = _pair_group_weights(groups)
    group_values = {group.group_index: 0.0 for group in groups}
    unallocated = 0.0
    allocated_pairs = 0
    for pair_index, pair_value in enumerate(randmulti_pair_values):
        weights = weights_by_pair[pair_index]
        total_weight = sum(weights.values())
        if total_weight <= 0:
            unallocated += pair_value
            continue
        allocated_pairs += 1
        for group_index, weight in weights.items():
            group_values[group_index] += pair_value * (weight / total_weight)

    profiles: dict[int, dict[str, Any]] = {}
    for group in groups:
        profiles[group.group_index] = {
            "estimated_component_score_rescue": group_values[group.group_index],
            "post_trace_overlap_component_delta_mean": _group_peer_overlap(post_pair_values, group),
            "motion_trace_overlap_component_delta_mean": _group_peer_overlap(motion_pair_values, group),
        }
    return profiles, {
        "allocated_pair_count": allocated_pairs,
        "unallocated_pair_component_score_delta": _round(unallocated),
        "allocated_component_score_delta": _round(sum(group_values.values())),
    }


def _group_id(groups: Sequence[RandmultiGroup], selected: Iterable[int]) -> str:
    raw = json.dumps(sorted(int(value) for value in selected), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _group_summary(
    group: RandmultiGroup,
    value_profile: dict[str, Any],
    *,
    zero_brotli_bytes: int,
    isolated_brotli_bytes: int,
    isolated_sha256: str,
) -> dict[str, Any]:
    component_value = float(value_profile["estimated_component_score_rescue"])
    isolated_delta_bytes = isolated_brotli_bytes - zero_brotli_bytes
    isolated_rate_cost = isolated_delta_bytes * RATE_SCORE_PER_BYTE
    net = component_value - isolated_rate_cost
    return {
        "active_pair_count": group.active_pair_count,
        "amplitude": group.amplitude,
        "group_index": group.group_index,
        "height": group.height,
        "isolated_brotli_bytes": isolated_brotli_bytes,
        "isolated_brotli_delta_bytes_vs_zero": isolated_delta_bytes,
        "isolated_brotli_sha256": isolated_sha256,
        "isolated_rate_score_cost_vs_zero": _round(isolated_rate_cost),
        "estimated_component_score_rescue": _round(component_value),
        "estimated_net_score_rescue_after_isolated_rate": _round(net),
        "motion_trace_overlap_component_delta_mean": _round(
            float(value_profile["motion_trace_overlap_component_delta_mean"])
        ),
        "nonzero_choice_total": group.nonzero_choice_total,
        "post_trace_overlap_component_delta_mean": _round(
            float(value_profile["post_trace_overlap_component_delta_mean"])
        ),
        "raw_payload_bytes": group.raw_payload_bytes,
        "rate_value_density_component_per_isolated_byte": _round(
            component_value / max(1, isolated_delta_bytes)
        ),
        "scount": group.scount,
        "width": group.width,
    }


def _atom_summary(atom: RandmultiAtom, group_component_value: float, group_nonzero_total: int) -> dict[str, Any]:
    share = 0.0
    if group_nonzero_total > 0:
        share = group_component_value * (atom.nonzero_choice_count / group_nonzero_total)
    return {
        "active_pair_count": len(atom.active_pair_indices),
        "atom_id": f"pr85_randmulti_g{atom.group_index:03d}_r{atom.row_index:02d}",
        "estimated_component_score_rescue_share": _round(share),
        "group_index": atom.group_index,
        "nonzero_choice_count": atom.nonzero_choice_count,
        "nonzero_value_sum": atom.nonzero_value_sum,
        "raw_payload_bytes": atom.raw_payload_bytes,
        "row_index": atom.row_index,
    }


def _policy_record(
    policy_id: str,
    selected_group_ids: Sequence[int],
    *,
    groups: Sequence[RandmultiGroup],
    group_profiles: dict[int, dict[str, Any]],
    zero_brotli_bytes: int,
    source_randmulti_bytes: int,
) -> dict[str, Any]:
    selected = tuple(sorted({int(value) for value in selected_group_ids}))
    compressed_bytes, compressed_sha = _compressed_len_for_groups(groups, set(selected))
    byte_cost_vs_zero = compressed_bytes - zero_brotli_bytes
    byte_delta_vs_source = compressed_bytes - source_randmulti_bytes
    component_value = sum(float(group_profiles[group_id]["estimated_component_score_rescue"]) for group_id in selected)
    rate_cost = byte_cost_vs_zero * RATE_SCORE_PER_BYTE
    net_rescue = component_value - rate_cost
    selected_group_set = set(selected)
    selected_atoms = sum(
        len(group.atoms)
        for group in groups
        if group.group_index in selected_group_set
    )
    selected_nonzero = sum(
        group.nonzero_choice_total
        for group in groups
        if group.group_index in selected_group_set
    )
    return {
        "candidate_policy_id": policy_id,
        "compressed_randmulti_brotli_bytes": int(compressed_bytes),
        "compressed_randmulti_brotli_sha256": compressed_sha,
        "dispatch_gate": "planning_only/no_remote_dispatch",
        "estimated_component_score_rescue": _round(component_value),
        "estimated_net_score_rescue_after_rate": _round(net_rescue),
        "estimated_rate_score_cost_vs_zero_randmulti": _round(rate_cost),
        "no_remote_dispatch": True,
        "planning_only": True,
        "policy_hash": _group_id(groups, selected),
        "score_claim": False,
        "selected_atom_count": int(selected_atoms),
        "selected_group_count": len(selected),
        "selected_group_ids": list(selected),
        "selected_nonzero_choice_total": int(selected_nonzero),
        "value_model": "exact-negative PR85 minus-randmulti component delta allocated by active sparse row overlap",
        "byte_model": "Brotli-best PR85 headerless sparse schedule with unselected groups zeroed",
        "randmulti_brotli_byte_cost_vs_zero": int(byte_cost_vs_zero),
        "randmulti_brotli_byte_delta_vs_source_segment": int(byte_delta_vs_source),
    }


def _rank_group_ids(group_rows: Sequence[dict[str, Any]]) -> list[int]:
    return [
        int(row["group_index"])
        for row in sorted(
            group_rows,
            key=lambda row: (
                -float(row["estimated_net_score_rescue_after_isolated_rate"]),
                -float(row["rate_value_density_component_per_isolated_byte"]),
                -float(row["estimated_component_score_rescue"]),
                int(row["isolated_brotli_delta_bytes_vs_zero"]),
                int(row["group_index"]),
            ),
        )
    ]


def _policy_rows(
    groups: Sequence[RandmultiGroup],
    group_rows: Sequence[dict[str, Any]],
    group_profiles: dict[int, dict[str, Any]],
    *,
    zero_brotli_bytes: int,
    source_randmulti_bytes: int,
    topks: Sequence[int],
    budget_fractions: Sequence[float],
) -> list[dict[str, Any]]:
    ranked = _rank_group_ids(group_rows)
    rows: list[dict[str, Any]] = []
    all_groups = [group.group_index for group in groups]
    rows.append(
        _policy_record(
            "source_all_groups_recompressed",
            all_groups,
            groups=groups,
            group_profiles=group_profiles,
            zero_brotli_bytes=zero_brotli_bytes,
            source_randmulti_bytes=source_randmulti_bytes,
        )
    )
    positive = [
        int(row["group_index"])
        for row in group_rows
        if float(row["estimated_component_score_rescue"]) > 0.0
    ]
    rows.append(
        _policy_record(
            "component_positive_groups",
            positive,
            groups=groups,
            group_profiles=group_profiles,
            zero_brotli_bytes=zero_brotli_bytes,
            source_randmulti_bytes=source_randmulti_bytes,
        )
    )
    net_positive = [
        int(row["group_index"])
        for row in group_rows
        if float(row["estimated_net_score_rescue_after_isolated_rate"]) > 0.0
    ]
    rows.append(
        _policy_record(
            "isolated_net_positive_groups",
            net_positive,
            groups=groups,
            group_profiles=group_profiles,
            zero_brotli_bytes=zero_brotli_bytes,
            source_randmulti_bytes=source_randmulti_bytes,
        )
    )
    best_prefix: list[int] = []
    best_net = -float("inf")
    current: list[int] = []
    for group_id in ranked:
        current.append(group_id)
        record = _policy_record(
            "waterfill_best_prefix_by_net",
            current,
            groups=groups,
            group_profiles=group_profiles,
            zero_brotli_bytes=zero_brotli_bytes,
            source_randmulti_bytes=source_randmulti_bytes,
        )
        net = float(record["estimated_net_score_rescue_after_rate"])
        if net > best_net:
            best_net = net
            best_prefix = list(current)
    rows.append(
        _policy_record(
            "waterfill_best_prefix_by_net",
            best_prefix,
            groups=groups,
            group_profiles=group_profiles,
            zero_brotli_bytes=zero_brotli_bytes,
            source_randmulti_bytes=source_randmulti_bytes,
        )
    )
    for topk in sorted({int(value) for value in topks if int(value) > 0}):
        rows.append(
            _policy_record(
                f"waterfill_top{topk:03d}",
                ranked[:topk],
                groups=groups,
                group_profiles=group_profiles,
                zero_brotli_bytes=zero_brotli_bytes,
                source_randmulti_bytes=source_randmulti_bytes,
            )
        )
    source_budget_delta = max(0, source_randmulti_bytes - zero_brotli_bytes)
    for fraction in sorted({float(value) for value in budget_fractions if float(value) >= 0.0}):
        limit = zero_brotli_bytes + math.floor(source_budget_delta * fraction)
        selected: list[int] = []
        for group_id in ranked:
            trial = [*selected, group_id]
            compressed_bytes, _sha = _compressed_len_for_groups(groups, set(trial))
            if compressed_bytes <= limit:
                selected = trial
        rows.append(
            _policy_record(
                f"waterfill_budget_{round(fraction * 100):03d}pct_source_randmulti",
                selected,
                groups=groups,
                group_profiles=group_profiles,
                zero_brotli_bytes=zero_brotli_bytes,
                source_randmulti_bytes=source_randmulti_bytes,
            )
        )
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row["candidate_policy_id"]] = row
    return [deduped[key] for key in sorted(deduped)]


def _source_archive_zip_info(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
    return {
        "path": _rel(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "member_names": names,
    }


def build_plan(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    out_dir: Path = DEFAULT_OUT_DIR,
    baseline_auth_json: Path = DEFAULT_BASELINE_EVAL_DIR / "contest_auth_eval.adjudicated.json",
    baseline_trace_json: Path = DEFAULT_BASELINE_EVAL_DIR / "component_trace.json",
    minus_randmulti_auth_json: Path = DEFAULT_MINUS_RANDMULTI_EVAL_DIR / "contest_auth_eval.adjudicated.json",
    minus_randmulti_trace_json: Path = DEFAULT_MINUS_RANDMULTI_EVAL_DIR / "component_trace.json",
    minus_post_auth_json: Path = DEFAULT_MINUS_POST_EVAL_DIR / "contest_auth_eval.adjudicated.json",
    minus_post_trace_json: Path = DEFAULT_MINUS_POST_EVAL_DIR / "component_trace.json",
    minus_motion_auth_json: Path = DEFAULT_MINUS_MOTION_EVAL_DIR / "contest_auth_eval.adjudicated.json",
    minus_motion_trace_json: Path = DEFAULT_MINUS_MOTION_EVAL_DIR / "component_trace.json",
    topks: Sequence[int] = DEFAULT_TOPKS,
    budget_fractions: Sequence[float] = DEFAULT_BUDGET_FRACTIONS,
) -> dict[str, Any]:
    source_archive, source_bundle, groups = _load_source_randmulti(archive)
    baseline_auth = _load_auth_metrics(baseline_auth_json, label="baseline_pr85")
    minus_randmulti_auth = _load_auth_metrics(minus_randmulti_auth_json, label="minus_randmulti")
    minus_post_auth = _load_auth_metrics(minus_post_auth_json, label="minus_post")
    minus_motion_auth = _load_auth_metrics(minus_motion_auth_json, label="minus_motion")
    baseline_trace = _load_trace(baseline_trace_json, label="baseline_pr85", auth=baseline_auth)
    minus_randmulti_trace = _load_trace(
        minus_randmulti_trace_json,
        label="minus_randmulti",
        auth=minus_randmulti_auth,
    )
    minus_post_trace = _load_trace(minus_post_trace_json, label="minus_post", auth=minus_post_auth)
    minus_motion_trace = _load_trace(minus_motion_trace_json, label="minus_motion", auth=minus_motion_auth)

    exact_negatives = {
        "minus_motion": _segment_negative_summary(baseline_auth, minus_motion_auth),
        "minus_post": _segment_negative_summary(baseline_auth, minus_post_auth),
        "minus_randmulti": _segment_negative_summary(baseline_auth, minus_randmulti_auth),
    }
    randmulti_pair_values, randmulti_calibration = _pair_component_deltas(
        baseline_trace,
        minus_randmulti_trace,
        exact_component_delta=minus_randmulti_auth.component_score - baseline_auth.component_score,
    )
    post_pair_values, post_calibration = _pair_component_deltas(
        baseline_trace,
        minus_post_trace,
        exact_component_delta=minus_post_auth.component_score - baseline_auth.component_score,
    )
    motion_pair_values, motion_calibration = _pair_component_deltas(
        baseline_trace,
        minus_motion_trace,
        exact_component_delta=minus_motion_auth.component_score - baseline_auth.component_score,
    )
    group_profiles, allocation_summary = _allocate_group_values(
        groups,
        randmulti_pair_values,
        post_pair_values,
        motion_pair_values,
    )

    zero_brotli_bytes, zero_brotli_sha = _compressed_len_for_groups(groups, set())
    all_brotli_bytes, all_brotli_sha = _compressed_len_for_groups(
        groups,
        {group.group_index for group in groups},
    )
    group_rows = []
    atom_rows = []
    for group in groups:
        isolated_bytes, isolated_sha = _compressed_len_for_groups(groups, {group.group_index})
        group_row = _group_summary(
            group,
            group_profiles[group.group_index],
            zero_brotli_bytes=zero_brotli_bytes,
            isolated_brotli_bytes=isolated_bytes,
            isolated_sha256=isolated_sha,
        )
        group_rows.append(group_row)
        group_component_value = float(group_profiles[group.group_index]["estimated_component_score_rescue"])
        for atom in group.atoms:
            atom_rows.append(_atom_summary(atom, group_component_value, group.nonzero_choice_total))
    group_rows = sorted(group_rows, key=lambda row: int(row["group_index"]))
    atom_rows = sorted(
        atom_rows,
        key=lambda row: (
            -float(row["estimated_component_score_rescue_share"]),
            int(row["group_index"]),
            int(row["row_index"]),
        ),
    )
    policies = _policy_rows(
        groups,
        group_rows,
        group_profiles,
        zero_brotli_bytes=zero_brotli_bytes,
        source_randmulti_bytes=int(source_bundle["randmulti_segment_bytes"]),
        topks=topks,
        budget_fractions=budget_fractions,
    )
    evidence_inputs = {
        "baseline_pr85": {
            "auth": _auth_summary(baseline_auth),
            "component_trace_json": _rel(baseline_trace.path),
            "trace_archive_sha256": baseline_trace.archive_sha256,
        },
        "minus_motion": {
            "auth": _auth_summary(minus_motion_auth),
            "component_trace_json": _rel(minus_motion_trace.path),
            "trace_archive_sha256": minus_motion_trace.archive_sha256,
        },
        "minus_post": {
            "auth": _auth_summary(minus_post_auth),
            "component_trace_json": _rel(minus_post_trace.path),
            "trace_archive_sha256": minus_post_trace.archive_sha256,
        },
        "minus_randmulti": {
            "auth": _auth_summary(minus_randmulti_auth),
            "component_trace_json": _rel(minus_randmulti_trace.path),
            "trace_archive_sha256": minus_randmulti_trace.archive_sha256,
        },
    }
    ledger = {
        "schema": GROUP_LEDGER_SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "planning_only": True,
        "remote_jobs_dispatched": False,
        "evidence_grade": EVIDENCE_GRADE,
        "source_archive": source_archive,
        "source_archive_zip_info": _source_archive_zip_info(archive),
        "source_bundle": source_bundle,
        "evidence_inputs": evidence_inputs,
        "exact_negative_segment_values": exact_negatives,
        "pair_delta_calibration": {
            "minus_motion": motion_calibration,
            "minus_post": post_calibration,
            "minus_randmulti": randmulti_calibration,
        },
        "allocation_summary": allocation_summary,
        "randmulti_byte_model": {
            "zero_group_brotli_bytes": zero_brotli_bytes,
            "zero_group_brotli_sha256": zero_brotli_sha,
            "all_groups_recompressed_brotli_bytes": all_brotli_bytes,
            "all_groups_recompressed_brotli_sha256": all_brotli_sha,
            "source_randmulti_brotli_bytes": int(source_bundle["randmulti_segment_bytes"]),
            "all_groups_recompressed_delta_vs_source": all_brotli_bytes - int(source_bundle["randmulti_segment_bytes"]),
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "model": "Brotli-best PR85 headerless sparse schedule with unselected groups zeroed",
        },
        "group_count": len(groups),
        "atom_count": sum(len(group.atoms) for group in groups),
        "groups": group_rows,
        "atoms": atom_rows,
    }
    policy_payload = {
        "schema": POLICY_SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "planning_only": True,
        "remote_jobs_dispatched": False,
        "evidence_grade": EVIDENCE_GRADE,
        "policy_count": len(policies),
        "group_ledger_json": _rel(out_dir / "randmulti_group_ledger.json"),
        "source_archive": source_archive,
        "randmulti_byte_model": ledger["randmulti_byte_model"],
        "exact_negative_segment_values": exact_negatives,
        "ranking": {
            "sort_order": [
                "estimated_net_score_rescue_after_isolated_rate desc",
                "rate_value_density_component_per_isolated_byte desc",
                "estimated_component_score_rescue desc",
                "isolated_brotli_delta_bytes_vs_zero asc",
                "group_index asc",
            ],
            "value_model": "planning-only allocation from exact-negative PR85 minus-randmulti trace",
        },
        "policies": policies,
    }
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "planning_only": True,
        "remote_jobs_dispatched": False,
        "evidence_grade": EVIDENCE_GRADE,
        "source_archive": source_archive,
        "group_ledger_json": _rel(out_dir / "randmulti_group_ledger.json"),
        "candidate_policies_json": _rel(out_dir / "candidate_policies.json"),
        "candidate_count": len(policies),
        "top_group_ids_by_waterfill": _rank_group_ids(group_rows)[:16],
        "exact_negative_segment_values": exact_negatives,
        "notes": [
            "Group values are estimates, not score claims.",
            "No remote jobs were dispatched.",
            "Future exact eval requires lane claim and archive construction from a selected policy.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "randmulti_group_ledger.json", ledger)
    _write_json(out_dir / "candidate_policies.json", policy_payload)
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def _parse_int_list(values: Sequence[str] | None, default: Sequence[int]) -> tuple[int, ...]:
    if not values:
        return tuple(default)
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return tuple(out)


def _parse_float_list(values: Sequence[str] | None, default: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return tuple(default)
    out: list[float] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline-auth-json", type=Path, default=DEFAULT_BASELINE_EVAL_DIR / "contest_auth_eval.adjudicated.json")
    parser.add_argument("--baseline-trace-json", type=Path, default=DEFAULT_BASELINE_EVAL_DIR / "component_trace.json")
    parser.add_argument("--minus-randmulti-auth-json", type=Path, default=DEFAULT_MINUS_RANDMULTI_EVAL_DIR / "contest_auth_eval.adjudicated.json")
    parser.add_argument("--minus-randmulti-trace-json", type=Path, default=DEFAULT_MINUS_RANDMULTI_EVAL_DIR / "component_trace.json")
    parser.add_argument("--minus-post-auth-json", type=Path, default=DEFAULT_MINUS_POST_EVAL_DIR / "contest_auth_eval.adjudicated.json")
    parser.add_argument("--minus-post-trace-json", type=Path, default=DEFAULT_MINUS_POST_EVAL_DIR / "component_trace.json")
    parser.add_argument("--minus-motion-auth-json", type=Path, default=DEFAULT_MINUS_MOTION_EVAL_DIR / "contest_auth_eval.adjudicated.json")
    parser.add_argument("--minus-motion-trace-json", type=Path, default=DEFAULT_MINUS_MOTION_EVAL_DIR / "component_trace.json")
    parser.add_argument("--topk", action="append", dest="topks", help="Top-k group counts; comma-separated values accepted")
    parser.add_argument(
        "--budget-fraction",
        action="append",
        dest="budget_fractions",
        help="Source randmulti byte-budget fractions; comma-separated values accepted",
    )
    args = parser.parse_args(argv)

    payload = build_plan(
        archive=args.archive,
        out_dir=args.out_dir,
        baseline_auth_json=args.baseline_auth_json,
        baseline_trace_json=args.baseline_trace_json,
        minus_randmulti_auth_json=args.minus_randmulti_auth_json,
        minus_randmulti_trace_json=args.minus_randmulti_trace_json,
        minus_post_auth_json=args.minus_post_auth_json,
        minus_post_trace_json=args.minus_post_trace_json,
        minus_motion_auth_json=args.minus_motion_auth_json,
        minus_motion_trace_json=args.minus_motion_trace_json,
        topks=_parse_int_list(args.topks, DEFAULT_TOPKS),
        budget_fractions=_parse_float_list(args.budget_fractions, DEFAULT_BUDGET_FRACTIONS),
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
