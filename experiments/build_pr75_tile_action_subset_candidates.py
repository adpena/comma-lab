#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build C067 + PR75 tile-action subset candidates.

The builder is empirical planning only: it uses component traces to rank the
charged PR75 SegNet tile-action records, emits deterministic P3 single-member
archives, and records enough provenance for later exact CUDA auth eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import brotli


C067_MASK_BR_LEN = 219_472
C067_MODEL_BR_LEN = 55_965
PR75_MASK_BR_LEN = 219_472
PR75_MODEL_BR_LEN = 56_034
PR75_ACTION_BR_LEN = 236
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RATE_DENOM = 37_545_489
SEG_TILE_ACTION_DICT_MAGIC = b"TAD1"
SEG_TILE_ACTION_DICT_HEADER_STRUCT = "<4sHH"
P5_MAX_DICT_ACTIONS = 64


@dataclass(frozen=True)
class ActionRecord:
    index: int
    pair_index: int
    tile_id: int
    action_id: int
    delta_combined: float
    delta_seg: float
    delta_pose: float
    order_key: int | None = None
    source_index: int | None = None
    source_action_id: int | None = None
    transform: str = "identity"
    custom_delta_rgb: tuple[float, float, float] | None = None
    calibration_source: str | None = None
    calibration_rank: int | None = None

    def encode4(self) -> bytes:
        return (
            int(self.pair_index).to_bytes(2, "little")
            + bytes([int(self.tile_id), int(self.action_id)])
        )


@dataclass(frozen=True)
class CalibrationTrace:
    transform: str
    max_rank: int
    path: Path
    by_pair: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class EncodedActions:
    wire_format: str
    raw_runtime_records: bytes
    encoded_action_stream: bytes
    encoded_action_codec: str
    action_dict_raw: bytes
    action_dict_br: bytes
    dictionary_entries: list[dict[str, Any]]
    remap: dict[str, int]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_sort_key(record: ActionRecord) -> int:
    return int(record.order_key if record.order_key is not None else record.index * 10)


def _record_source_index(record: ActionRecord) -> int:
    return int(record.source_index if record.source_index is not None else record.index)


def _record_source_action_id(record: ActionRecord) -> int:
    return int(
        record.source_action_id
        if record.source_action_id is not None
        else record.action_id
    )


def _records_runtime_signature(records: list[ActionRecord]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(rec.pair_index), int(rec.tile_id), int(rec.action_id))
        for rec in sorted(records, key=_record_sort_key)
    )


def _records_effective_signature(
    records: list[ActionRecord],
) -> tuple[tuple[int, int, int, tuple[float, float, float] | None], ...]:
    return tuple(
        (
            int(rec.pair_index),
            int(rec.tile_id),
            int(rec.action_id),
            tuple(float(x) for x in rec.custom_delta_rgb)
            if rec.custom_delta_rgb is not None
            else None,
        )
        for rec in sorted(records, key=_record_sort_key)
    )


def _read_single_member(path: Path, *, member: str = "p") -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if names != [member]:
            raise ValueError(f"expected single member {member!r} in {path}, got {names!r}")
        return zf.read(member)


def _read_c067_slices(path: Path) -> dict[str, bytes]:
    payload = _read_single_member(path)
    if len(payload) <= C067_MASK_BR_LEN + C067_MODEL_BR_LEN:
        raise ValueError(f"C067 payload too short: {len(payload)}")
    return {
        "mask_br": payload[:C067_MASK_BR_LEN],
        "model_br": payload[C067_MASK_BR_LEN:C067_MASK_BR_LEN + C067_MODEL_BR_LEN],
        "pose_br": payload[C067_MASK_BR_LEN + C067_MODEL_BR_LEN:],
    }


def _read_pr75_action_records(path: Path) -> bytes:
    payload = _read_single_member(path)
    start = PR75_MASK_BR_LEN + PR75_MODEL_BR_LEN
    end = start + PR75_ACTION_BR_LEN
    if len(payload) <= end:
        raise ValueError(f"PR75 payload too short for action slice: {len(payload)}")
    raw = brotli.decompress(payload[start:end])
    if len(raw) % 4:
        raise ValueError(f"PR75 action payload is not compact 4-byte records: {len(raw)}")
    return raw


def _trace_by_pair(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return _samples_by_pair(payload, path=path)


def _samples_by_pair(payload: dict[str, Any], *, path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for sample in payload.get("samples", []):
        if not sample:
            continue
        out[int(sample["pair_index"])] = sample
    if not out:
        raise ValueError(f"no component-trace samples found: {path}")
    return out


def _component_trace_score(path: Path) -> float | None:
    payload = json.loads(path.read_text())
    value = payload.get("score_recomputed_from_components")
    return float(value) if value is not None else None


def _parse_calibration_component_traces(values: list[str]) -> list[CalibrationTrace]:
    traces: list[CalibrationTrace] = []
    for raw in values:
        try:
            transform, max_rank_raw, path_raw = raw.split(":", 2)
        except ValueError as exc:
            raise ValueError(
                "--calibration-component-trace must be TRANSFORM:MAX_RANK:PATH"
            ) from exc
        if transform not in {"identity", "ampminus1", "ampminus2", "ampminus3", "ampplus1", "ampplus2"}:
            raise ValueError(f"unsupported calibration transform: {transform}")
        max_rank = int(max_rank_raw)
        if max_rank <= 0:
            raise ValueError(f"calibration max rank must be positive: {raw}")
        path = Path(path_raw)
        payload = json.loads(path.read_text())
        traces.append(
            CalibrationTrace(
                transform=transform,
                max_rank=max_rank,
                path=path,
                by_pair=_samples_by_pair(payload, path=path),
            )
        )
    return traces


def _rank_records(
    raw_records: bytes,
    *,
    base_trace: dict[int, dict[str, Any]],
    candidate_trace: dict[int, dict[str, Any]],
) -> list[ActionRecord]:
    records: list[ActionRecord] = []
    for offset in range(0, len(raw_records), 4):
        pair_index = int.from_bytes(raw_records[offset:offset + 2], "little")
        tile_id = raw_records[offset + 2]
        action_id = raw_records[offset + 3]
        try:
            base = base_trace[pair_index]
            candidate = candidate_trace[pair_index]
        except KeyError as exc:
            raise ValueError(f"action pair {pair_index} missing from traces") from exc
        records.append(
            ActionRecord(
                index=offset // 4,
                pair_index=pair_index,
                tile_id=tile_id,
                action_id=action_id,
                delta_combined=(
                    float(base["score_combined_contribution_first_order"])
                    - float(candidate["score_combined_contribution_first_order"])
                ),
                delta_seg=(
                    float(base["score_seg_contribution_exact"])
                    - float(candidate["score_seg_contribution_exact"])
                ),
                delta_pose=(
                    float(base["score_pose_contribution_first_order"])
                    - float(candidate["score_pose_contribution_first_order"])
                ),
                order_key=(offset // 4) * 10,
                source_index=offset // 4,
                source_action_id=action_id,
            )
        )
    return sorted(records, key=lambda rec: rec.delta_combined, reverse=True)


def _select_policy(
    records: list[ActionRecord],
    policy: str,
    *,
    base_trace: dict[int, dict[str, Any]] | None = None,
    calibration_traces: Iterable[CalibrationTrace] | dict[str, CalibrationTrace] | None = None,
) -> list[ActionRecord]:
    if policy == "all_ampminus1":
        selected = [_action_amp_shift(rec, -1) for rec in records]
        return _validated_selected_policy(policy, records, selected)
    if policy == "poseharm_ampminus1":
        selected = [
            _action_amp_shift(rec, -1) if rec.delta_pose < 0.0 else rec
            for rec in records
        ]
        return _validated_selected_policy(policy, records, selected)
    if policy == "positive_poseharm_ampminus1":
        base = [rec for rec in records if rec.delta_combined > 0.0]
        selected = [
            _action_amp_shift(rec, -1) if rec.delta_pose < 0.0 else rec
            for rec in base
        ]
        return _validated_selected_policy(policy, base, selected)

    base_policy, transform = _split_policy_transform(policy)
    base_policy, amp_shift = _split_policy_amp_shift(base_policy)
    base_selected = _select_base_policy(
        records,
        base_policy,
        base_trace=base_trace,
        calibration_traces=calibration_traces,
    )
    selected = base_selected
    if amp_shift:
        selected = [_action_amp_shift(rec, amp_shift) for rec in selected]
    if transform:
        selected = _apply_policy_transform(selected, transform)
    return _validated_selected_policy(policy, base_selected, selected)


def _select_base_policy(
    records: list[ActionRecord],
    policy: str,
    *,
    base_trace: dict[int, dict[str, Any]] | None = None,
    calibration_traces: Iterable[CalibrationTrace] | dict[str, CalibrationTrace] | None = None,
) -> list[ActionRecord]:
    if policy == "all":
        return list(records)
    if policy == "positive":
        return [rec for rec in records if rec.delta_combined > 0.0]
    if policy == "pose_safe_positive":
        return [
            rec
            for rec in records
            if rec.delta_combined > 0.0 and rec.delta_pose >= 0.0
        ]
    match = re.fullmatch(r"top(\d+)_drop([0-9_]+)(?:_add([0-9_]+))?", policy)
    if match:
        n = int(match.group(1))
        drop_ranks = _parse_policy_rank_list(match.group(2), policy=policy)
        add_ranks = (
            _parse_policy_rank_list(match.group(3), policy=policy)
            if match.group(3)
            else []
        )
        if any(rank > n for rank in drop_ranks):
            raise ValueError(f"{policy!r} cannot drop rank outside top{n}: {drop_ranks}")
        selected = [
            rec
            for rank, rec in enumerate(records[:n], start=1)
            if rank not in set(drop_ranks)
        ]
        for rank in add_ranks:
            if rank <= 0 or rank > len(records):
                raise ValueError(f"{policy!r} add rank out of range: {rank}")
            selected.append(records[rank - 1])
        return selected
    match = re.fullmatch(r"top(\d+)", policy)
    if match:
        return list(records[: int(match.group(1))])
    match = re.fullmatch(r"segtop(\d+)", policy)
    if match:
        return sorted(records, key=lambda rec: rec.delta_seg, reverse=True)[: int(match.group(1))]
    match = re.fullmatch(r"posetop(\d+)", policy)
    if match:
        return sorted(records, key=lambda rec: rec.delta_pose, reverse=True)[: int(match.group(1))]
    match = re.fullmatch(r"beam_rate_top(\d+)", policy)
    if match:
        return _beam_select_records(records[: int(match.group(1))], pose_weight=1.0)
    match = re.fullmatch(r"beam_pose([0-9]+(?:p[0-9]+)?)_top(\d+)", policy)
    if match:
        pose_weight = float(match.group(1).replace("p", "."))
        return _beam_select_records(records[: int(match.group(2))], pose_weight=pose_weight)
    match = re.fullmatch(r"lag_eval(?:_pose([0-9]+(?:p[0-9]+)?))?_top(\d+)", policy)
    if match:
        if base_trace is None or not calibration_traces:
            raise ValueError(f"policy {policy!r} requires calibration component traces")
        pose_weight = float(match.group(1).replace("p", ".")) if match.group(1) else 1.0
        return _calibrated_lagrangian_select_records(
            records[: int(match.group(2))],
            base_trace=base_trace,
            calibration_traces=calibration_traces,
            pose_weight=pose_weight,
        )
    raise ValueError(f"unsupported policy: {policy}")


def _parse_policy_rank_list(raw: str, *, policy: str) -> list[int]:
    ranks = [int(part) for part in raw.split("_") if part]
    if not ranks:
        raise ValueError(f"{policy!r} has an empty rank list")
    if any(rank <= 0 for rank in ranks):
        raise ValueError(f"{policy!r} ranks are 1-based positive integers: {ranks}")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{policy!r} repeats rank values: {ranks}")
    return ranks


def _split_policy_transform(policy: str) -> tuple[str, str]:
    for suffix, transform in (
        ("_wilddirmean", "wilddirmean"),
        ("_wilddiramp8", "wilddiramp8"),
        ("_wilddiramp6", "wilddiramp6"),
        ("_wilddiramp4", "wilddiramp4"),
        ("_signedposemix1", "signedposemix1"),
        ("_signedboost1", "signedboost1"),
        ("_signedshrink1", "signedshrink1"),
        ("_custompose150", "custompose150"),
        ("_custompose125", "custompose125"),
        ("_customboost125", "customboost125"),
        ("_ampfit_pose", "ampfit_pose"),
        ("_ampfit", "ampfit"),
    ):
        if policy.endswith(suffix):
            return policy[: -len(suffix)], transform
    return policy, ""


def _split_policy_amp_shift(policy: str) -> tuple[str, int]:
    for suffix, shift in (
        ("_ampminus3", -3),
        ("_ampminus2", -2),
        ("_ampminus1", -1),
        ("_ampplus1", 1),
        ("_ampplus2", 2),
    ):
        if policy.endswith(suffix):
            return policy[: -len(suffix)], shift
    return policy, 0


def _action_amp_shift(record: ActionRecord, shift: int) -> ActionRecord:
    """Move within PR75's direction/amplitude/sign dictionary."""
    direction = record.action_id // 12
    within = record.action_id % 12
    amp_index = within // 2
    sign_index = within % 2
    shifted_amp = min(5, max(0, amp_index + shift))
    return replace(
        record,
        action_id=direction * 12 + shifted_amp * 2 + sign_index,
        source_index=_record_source_index(record),
        source_action_id=_record_source_action_id(record),
        transform=f"amp_shift_{shift:+d}",
        custom_delta_rgb=None,
    )


def _apply_calibration_transform(record: ActionRecord, transform: str) -> ActionRecord:
    if transform == "identity":
        return record
    match = re.fullmatch(r"amp(minus|plus)(\d+)", transform)
    if not match:
        raise ValueError(f"unsupported calibration transform: {transform}")
    shift = int(match.group(2))
    if match.group(1) == "minus":
        shift = -shift
    return _action_amp_shift(record, shift)


def _action_sign_residual(
    record: ActionRecord,
    *,
    same_sign: bool,
    amp_index: int = 0,
    transform: str,
) -> ActionRecord | None:
    source_action_id = _record_source_action_id(record)
    direction = int(record.action_id) // 12
    within = int(record.action_id) % 12
    source_amp = within // 2
    source_sign = within % 2
    if not same_sign and source_amp == amp_index:
        return None
    sign_index = source_sign if same_sign else 1 - source_sign
    action_id = direction * 12 + amp_index * 2 + sign_index
    if same_sign and action_id == int(record.action_id):
        return None
    return replace(
        record,
        index=int(record.index),
        action_id=action_id,
        order_key=_record_sort_key(record) + 1,
        source_index=_record_source_index(record),
        source_action_id=source_action_id,
        transform=transform,
        custom_delta_rgb=None,
    )


def _apply_policy_transform(
    records: list[ActionRecord],
    transform: str,
) -> list[ActionRecord]:
    if transform in {"ampfit", "ampfit_pose"}:
        return [_action_amp_fit(rec, pose_only=(transform == "ampfit_pose")) for rec in records]
    if transform in {"signedboost1", "signedshrink1", "signedposemix1"}:
        return _apply_signed_combo(records, transform)
    if transform in {"custompose125", "custompose150", "customboost125"}:
        return [_action_custom_scale(rec, transform) for rec in records]
    if transform in {"wilddiramp4", "wilddiramp6", "wilddiramp8"}:
        amplitude = float(transform.removeprefix("wilddiramp"))
        return [_action_wild_direction_amp(rec, amplitude) for rec in records]
    if transform == "wilddirmean":
        return _apply_wild_direction_mean(records)
    raise ValueError(f"unsupported policy transform: {transform}")


def _action_amp_fit(record: ActionRecord, *, pose_only: bool) -> ActionRecord:
    if record.delta_pose < 0.0:
        return _action_amp_shift(record, -1)
    if not pose_only and record.delta_pose > 0.0 and record.delta_seg >= 0.0:
        return _action_amp_shift(record, 1)
    return record


def _apply_signed_combo(records: list[ActionRecord], transform: str) -> list[ActionRecord]:
    out: list[ActionRecord] = []
    for rec in records:
        out.append(rec)
        residual: ActionRecord | None = None
        if transform == "signedboost1":
            residual = _action_sign_residual(
                rec,
                same_sign=True,
                transform="signed_same_amp0_residual",
            )
        elif transform == "signedshrink1":
            residual = _action_sign_residual(
                rec,
                same_sign=False,
                transform="signed_opposite_amp0_residual",
            )
        elif rec.delta_pose < 0.0:
            residual = _action_sign_residual(
                rec,
                same_sign=False,
                transform="signed_poseharm_shrink_amp0",
            )
        elif rec.delta_pose > 0.0 and rec.delta_seg >= 0.0:
            residual = _action_sign_residual(
                rec,
                same_sign=True,
                transform="signed_posehelp_boost_amp0",
            )
        if residual is not None:
            out.append(residual)
    return out


def _action_custom_scale(record: ActionRecord, transform: str) -> ActionRecord:
    scale = 1.0
    if transform == "customboost125":
        scale = 1.25
    elif transform == "custompose125":
        if record.delta_pose < 0.0:
            scale = 0.75
        elif record.delta_pose > 0.0 and record.delta_seg >= 0.0:
            scale = 1.25
    elif transform == "custompose150":
        if record.delta_pose < 0.0:
            scale = 0.75
        elif record.delta_pose > 0.0 and record.delta_seg >= 0.0:
            scale = 1.50
    else:
        raise ValueError(f"unsupported custom scale transform: {transform}")
    if scale == 1.0:
        return record
    rgb = _fixed_action_specs()[int(record.action_id)]
    return replace(
        record,
        source_action_id=_record_source_action_id(record),
        transform=f"{transform}_scale_{scale:g}",
        custom_delta_rgb=tuple(round(float(value) * scale, 6) for value in rgb),
    )


def _action_direction_sign(record: ActionRecord) -> tuple[int, int]:
    action_id = int(record.action_id)
    return action_id // 12, action_id % 2


def _action_amplitude(record: ActionRecord) -> float:
    amp_values = (2.0, 4.0, 6.0, 8.0, 12.0, 16.0)
    return amp_values[(int(record.action_id) % 12) // 2]


def _action_unit_direction(direction: int) -> tuple[float, float, float]:
    directions = [
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (-0.35, 0.15, 0.45),
        (0.25, 0.15, -0.20),
    ]
    try:
        raw = directions[direction]
    except IndexError as exc:
        raise ValueError(f"action direction out of range: {direction}") from exc
    denom = max(abs(value) for value in raw)
    return tuple(float(value) / denom for value in raw)


def _custom_delta_for_direction(
    direction: int,
    sign_index: int,
    amplitude: float,
) -> tuple[float, float, float]:
    sign = 1.0 if sign_index == 0 else -1.0
    unit = _action_unit_direction(direction)
    return tuple(round(sign * amplitude * value, 6) for value in unit)


def _action_wild_direction_amp(
    record: ActionRecord,
    amplitude: float,
) -> ActionRecord:
    direction, sign_index = _action_direction_sign(record)
    return replace(
        record,
        source_action_id=_record_source_action_id(record),
        transform=f"wilddiramp{amplitude:g}",
        custom_delta_rgb=_custom_delta_for_direction(direction, sign_index, amplitude),
    )


def _apply_wild_direction_mean(records: list[ActionRecord]) -> list[ActionRecord]:
    amplitudes: dict[tuple[int, int], list[float]] = {}
    for rec in records:
        key = _action_direction_sign(rec)
        amplitudes.setdefault(key, []).append(_action_amplitude(rec))
    group_amplitudes = {
        key: round(sum(values) / len(values), 6)
        for key, values in amplitudes.items()
    }
    return [
        replace(
            rec,
            source_action_id=_record_source_action_id(rec),
            transform=f"wilddirmean_amp_{group_amplitudes[_action_direction_sign(rec)]:g}",
            custom_delta_rgb=_custom_delta_for_direction(
                *_action_direction_sign(rec),
                group_amplitudes[_action_direction_sign(rec)],
            ),
        )
        for rec in records
    ]


def _beam_select_records(
    records: list[ActionRecord],
    *,
    pose_weight: float,
    beam_width: int = 128,
    rate_weight: float = 1.0,
) -> list[ActionRecord]:
    if not records:
        return []
    beams: list[tuple[ActionRecord, ...]] = [()]
    score_cache: dict[tuple[tuple[int, int], ...], tuple[float, float, int, int]] = {}

    def key_for(selection: tuple[ActionRecord, ...]) -> tuple[tuple[int, int], ...]:
        return tuple(sorted((int(rec.index), int(rec.action_id)) for rec in selection))

    def rank_key(selection: tuple[ActionRecord, ...]) -> tuple[float, float, int, int]:
        key = key_for(selection)
        cached = score_cache.get(key)
        if cached is not None:
            return cached
        sorted_selection = sorted(selection, key=_record_sort_key)
        raw = b"".join(rec.encode4() for rec in sorted_selection)
        action_br_bytes = len(brotli.compress(raw, quality=11)) if raw else 0
        trace_value = sum(rec.delta_seg + pose_weight * rec.delta_pose for rec in selection)
        combined = sum(rec.delta_combined for rec in selection)
        objective = trace_value - rate_weight * 25.0 * action_br_bytes / RATE_DENOM
        cached = (objective, combined, -action_br_bytes, -len(selection))
        score_cache[key] = cached
        return cached

    for rec in records:
        candidates = beams + [selection + (rec,) for selection in beams]
        unique = {key_for(selection): selection for selection in candidates}
        beams = sorted(unique.values(), key=rank_key, reverse=True)[:beam_width]
    nonempty = [selection for selection in beams if selection]
    if not nonempty:
        return [max(records, key=lambda rec: rec.delta_combined)]
    return list(max(nonempty, key=rank_key))


def _calibrated_lagrangian_select_records(
    records: list[ActionRecord],
    *,
    base_trace: dict[int, dict[str, Any]],
    calibration_traces: Iterable[CalibrationTrace] | dict[str, CalibrationTrace],
    pose_weight: float,
    beam_width: int = 256,
    rate_weight: float = 1.0,
) -> list[ActionRecord]:
    if not records:
        return []

    option_groups: list[list[ActionRecord]] = []
    for rank, rec in enumerate(records, start=1):
        options: list[ActionRecord] = []
        base_sample = base_trace[int(rec.pair_index)]
        for calibration in _iter_calibration_traces(calibration_traces):
            if rank > calibration.max_rank:
                continue
            candidate_sample = calibration.by_pair.get(int(rec.pair_index))
            if candidate_sample is None:
                continue
            calibrated = _apply_calibration_transform(rec, calibration.transform)
            delta_seg = (
                float(base_sample["score_seg_contribution_exact"])
                - float(candidate_sample["score_seg_contribution_exact"])
            )
            delta_pose = (
                float(base_sample["score_pose_contribution_first_order"])
                - float(candidate_sample["score_pose_contribution_first_order"])
            )
            options.append(
                replace(
                    calibrated,
                    delta_seg=delta_seg,
                    delta_pose=delta_pose,
                    delta_combined=delta_seg + delta_pose,
                    source_index=_record_source_index(rec),
                    source_action_id=_record_source_action_id(rec),
                    calibration_source=(
                        f"{calibration.transform}:top{calibration.max_rank}:"
                        f"{calibration.path}"
                    ),
                    calibration_rank=rank,
                )
            )
        if options:
            option_groups.append(options)

    if not option_groups:
        return [max(records, key=lambda rec: rec.delta_combined)]

    beams: list[tuple[ActionRecord, ...]] = [()]
    score_cache: dict[
        tuple[tuple[int, int, int, str], ...],
        tuple[float, float, int, int],
    ] = {}

    def key_for(selection: tuple[ActionRecord, ...]) -> tuple[tuple[int, int, int, str], ...]:
        return tuple(
            sorted(
                (
                    _record_source_index(rec),
                    int(rec.action_id),
                    _record_sort_key(rec),
                    rec.calibration_source or "",
                )
                for rec in selection
            )
        )

    def rank_key(selection: tuple[ActionRecord, ...]) -> tuple[float, float, int, int]:
        key = key_for(selection)
        cached = score_cache.get(key)
        if cached is not None:
            return cached
        sorted_selection = sorted(selection, key=_record_sort_key)
        raw = b"".join(rec.encode4() for rec in sorted_selection)
        action_br_bytes = len(brotli.compress(raw, quality=11)) if raw else 0
        trace_value = sum(rec.delta_seg + pose_weight * rec.delta_pose for rec in selection)
        combined = sum(rec.delta_combined for rec in selection)
        objective = trace_value - rate_weight * 25.0 * action_br_bytes / RATE_DENOM
        cached = (objective, combined, -action_br_bytes, -len(selection))
        score_cache[key] = cached
        return cached

    for options in option_groups:
        candidates = list(beams)
        for selection in beams:
            candidates.extend(selection + (option,) for option in options)
        unique = {key_for(selection): selection for selection in candidates}
        beams = sorted(unique.values(), key=rank_key, reverse=True)[:beam_width]

    nonempty = [selection for selection in beams if selection]
    if not nonempty:
        return [max(records, key=lambda rec: rec.delta_combined)]
    return list(max(nonempty, key=rank_key))


def _iter_calibration_traces(
    calibration_traces: Iterable[CalibrationTrace] | dict[str, CalibrationTrace],
) -> Iterable[CalibrationTrace]:
    if isinstance(calibration_traces, dict):
        return calibration_traces.values()
    return calibration_traces


def _validated_selected_policy(
    policy: str,
    base_records: list[ActionRecord],
    selected: list[ActionRecord],
) -> list[ActionRecord]:
    selected = sorted(selected, key=_record_sort_key)
    if not selected:
        raise ValueError(f"policy {policy!r} selected no action records")
    summary = _selection_guard_summary(selected)
    if summary["exact_duplicate_record_count"]:
        raise ValueError(
            f"policy {policy!r} emitted "
            f"{summary['exact_duplicate_record_count']} exact duplicate action records"
        )
    base_signature = _records_effective_signature(base_records)
    selected_signature = _records_effective_signature(selected)
    if _policy_requires_non_noop(policy) and selected_signature == base_signature:
        raise ValueError(f"policy {policy!r} did not change the selected action payload")
    return selected


def _policy_requires_non_noop(policy: str) -> bool:
    return any(
        marker in policy
        for marker in (
            "_ampminus",
            "_ampplus",
            "_ampfit",
            "_signed",
            "_custom",
            "poseharm_ampminus",
        )
    )


def _fixed_action_specs() -> list[tuple[float, float, float]]:
    directions = [
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (-0.35, 0.15, 0.45),
        (0.25, 0.15, -0.20),
    ]
    specs: list[tuple[float, float, float]] = []
    for direction in directions:
        denom = max(abs(value) for value in direction)
        unit = tuple(float(value) / denom for value in direction)
        for amp in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
            specs.append(tuple(value * amp for value in unit))
            specs.append(tuple(-value * amp for value in unit))
    return specs


def _action_dict_key(record: ActionRecord) -> tuple[str, int | tuple[float, float, float]]:
    if record.custom_delta_rgb is not None:
        return ("custom", tuple(round(float(value), 6) for value in record.custom_delta_rgb))
    return ("fixed", int(record.action_id))


def _action_dict_key_string(key: tuple[str, int | tuple[float, float, float]]) -> str:
    tag, value = key
    if tag == "fixed":
        return f"fixed:{int(value)}"
    rgb = ",".join(f"{float(component):.6g}" for component in value)  # type: ignore[arg-type]
    return f"custom:{rgb}"


def _build_action_dict(
    records: list[ActionRecord],
) -> tuple[bytes, list[dict[str, Any]], dict[str, int]]:
    keys = sorted(
        {_action_dict_key(record) for record in records},
        key=_action_dict_sort_key,
    )
    fixed = _fixed_action_specs()
    if not keys:
        return b"", [], {}
    fixed_ids = [int(value) for tag, value in keys if tag == "fixed"]
    if fixed_ids and (max(fixed_ids) >= len(fixed) or min(fixed_ids) < 0):
        raise ValueError(f"action id outside fixed dictionary: {fixed_ids}")
    header = struct.pack(SEG_TILE_ACTION_DICT_HEADER_STRUCT, SEG_TILE_ACTION_DICT_MAGIC, 1, len(keys))
    body = bytearray()
    entries: list[dict[str, Any]] = []
    remap: dict[str, int] = {}
    for new_id, key in enumerate(keys):
        tag, value = key
        if tag == "fixed":
            old_id = int(value)
            rgb = fixed[old_id]
            source_ids = [old_id]
            custom = False
        else:
            old_id = None
            rgb = tuple(float(component) for component in value)  # type: ignore[arg-type]
            source_ids = sorted(
                {
                    _record_source_action_id(record)
                    for record in records
                    if _action_dict_key(record) == key
                }
            )
            custom = True
        body.extend(struct.pack("<fff", *rgb))
        entries.append(
            {
                "dictionary_action_id": new_id,
                "source_action_id": old_id,
                "source_action_ids": source_ids,
                "delta_rgb": list(rgb),
                "custom": custom,
            }
        )
        remap[_action_dict_key_string(key)] = new_id
    return header + bytes(body), entries, remap


def _action_dict_sort_key(
    key: tuple[str, int | tuple[float, float, float]],
) -> tuple[int, int | tuple[float, float, float]]:
    tag, value = key
    if tag == "fixed":
        return (0, int(value))
    return (1, tuple(float(component) for component in value))  # type: ignore[arg-type]


def _remap_records(records: list[ActionRecord], remap: dict[str, int]) -> list[ActionRecord]:
    return [
        replace(
            rec,
            action_id=remap[_action_dict_key_string(_action_dict_key(rec))],
            custom_delta_rgb=None,
        )
        for rec in records
    ]


def _pack_p5_records(records: list[ActionRecord]) -> bytes:
    out = bytearray(len(records) * 3)
    offset = 0
    for rec in records:
        pair_index = int(rec.pair_index)
        tile_id = int(rec.tile_id)
        action_id = int(rec.action_id)
        if pair_index < 0 or pair_index >= 1024:
            raise ValueError(f"P5 pair index out of range: {pair_index}")
        if tile_id < 0 or tile_id >= 256:
            raise ValueError(f"P5 tile id out of range: {tile_id}")
        if action_id < 0 or action_id >= P5_MAX_DICT_ACTIONS:
            raise ValueError(f"P5 action id out of range: {action_id}")
        word = pair_index | (tile_id << 10) | (action_id << 18)
        out[offset] = word & 0xFF
        out[offset + 1] = (word >> 8) & 0xFF
        out[offset + 2] = (word >> 16) & 0xFF
        offset += 3
    return bytes(out)


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"cannot varint-encode negative value {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_delta_varint_records(records: list[ActionRecord]) -> bytes:
    out = bytearray()
    previous_pair = 0
    for index, rec in enumerate(records):
        pair_index = int(rec.pair_index)
        tile_id = int(rec.tile_id)
        action_id = int(rec.action_id)
        if pair_index < 0:
            raise ValueError(f"P6 pair index out of range: {pair_index}")
        if tile_id < 0 or tile_id >= 256:
            raise ValueError(f"P6 tile id out of range: {tile_id}")
        if action_id < 0 or action_id >= 256:
            raise ValueError(f"P6 action id out of range: {action_id}")
        delta = pair_index if index == 0 else pair_index - previous_pair
        if delta < 0:
            raise ValueError("P6 delta-varint action encoding requires nondecreasing pairs")
        out.extend(_uleb128(delta))
        out.append(tile_id)
        out.append(action_id)
        previous_pair = pair_index
    return bytes(out)


def _encode_actions(records: list[ActionRecord], wire_format: str) -> EncodedActions:
    if wire_format == "p3":
        if any(rec.custom_delta_rgb is not None for rec in records):
            raise ValueError("custom action deltas require p4 or p5 wire format")
        raw = b"".join(rec.encode4() for rec in records)
        return EncodedActions(
            wire_format=wire_format,
            raw_runtime_records=raw,
            encoded_action_stream=raw,
            encoded_action_codec="brotli_runtime_u16_pair_u8_tile_u8_action",
            action_dict_raw=b"",
            action_dict_br=b"",
            dictionary_entries=[],
            remap={},
        )
    if wire_format == "p6":
        if any(rec.custom_delta_rgb is not None for rec in records):
            raise ValueError("custom action deltas require p4 or p5 wire format")
        raw = b"".join(rec.encode4() for rec in records)
        return EncodedActions(
            wire_format=wire_format,
            raw_runtime_records=raw,
            encoded_action_stream=_encode_delta_varint_records(records),
            encoded_action_codec="brotli_delta_varint_pair_tile_action",
            action_dict_raw=b"",
            action_dict_br=b"",
            dictionary_entries=[],
            remap={},
        )
    if wire_format not in {"p4", "p5"}:
        raise ValueError(f"unsupported wire format: {wire_format}")
    action_dict_raw, entries, remap = _build_action_dict(records)
    remapped = _remap_records(records, remap)
    raw_runtime_records = b"".join(rec.encode4() for rec in remapped)
    if wire_format == "p4":
        encoded_action_stream = raw_runtime_records
        codec = "brotli_custom_dict_u16_pair_u8_tile_u8_action"
    else:
        if len(entries) > P5_MAX_DICT_ACTIONS:
            raise ValueError(
                f"P5 supports at most {P5_MAX_DICT_ACTIONS} dictionary actions, got {len(entries)}"
            )
        encoded_action_stream = _pack_p5_records(remapped)
        codec = "brotli_custom_dict_packed24_pair10_tile8_action6"
    return EncodedActions(
        wire_format=wire_format,
        raw_runtime_records=raw_runtime_records,
        encoded_action_stream=encoded_action_stream,
        encoded_action_codec=codec,
        action_dict_raw=action_dict_raw,
        action_dict_br=brotli.compress(action_dict_raw, quality=11),
        dictionary_entries=entries,
        remap=remap,
    )


def _build_p3_payload(c067: dict[str, bytes], action_raw: bytes) -> bytes:
    action_br = brotli.compress(action_raw, quality=11)
    return (
        b"P3"
        + struct.pack("<IHH", len(c067["mask_br"]), len(c067["model_br"]), len(action_br))
        + c067["mask_br"]
        + c067["model_br"]
        + action_br
        + c067["pose_br"]
    )


def _build_p4_payload(c067: dict[str, bytes], encoded: EncodedActions) -> bytes:
    action_br = brotli.compress(encoded.encoded_action_stream, quality=11)
    return (
        b"P4"
        + struct.pack(
            "<IHHH",
            len(c067["mask_br"]),
            len(c067["model_br"]),
            len(encoded.action_dict_br),
            len(action_br),
        )
        + c067["mask_br"]
        + c067["model_br"]
        + encoded.action_dict_br
        + action_br
        + c067["pose_br"]
    )


def _build_p5_payload(c067: dict[str, bytes], encoded: EncodedActions, record_count: int) -> bytes:
    action_br = brotli.compress(encoded.encoded_action_stream, quality=11)
    return (
        b"P5"
        + struct.pack(
            "<IHHHH",
            len(c067["mask_br"]),
            len(c067["model_br"]),
            len(encoded.action_dict_br),
            len(action_br),
            record_count,
        )
        + c067["mask_br"]
        + c067["model_br"]
        + encoded.action_dict_br
        + action_br
        + c067["pose_br"]
    )


def _build_p6_payload(c067: dict[str, bytes], encoded: EncodedActions, record_count: int) -> bytes:
    action_br = brotli.compress(encoded.encoded_action_stream, quality=11)
    return (
        b"P6"
        + struct.pack(
            "<IHHH",
            len(c067["mask_br"]),
            len(c067["model_br"]),
            len(action_br),
            record_count,
        )
        + c067["mask_br"]
        + c067["model_br"]
        + action_br
        + c067["pose_br"]
    )


def _build_payload(c067: dict[str, bytes], encoded: EncodedActions, record_count: int) -> bytes:
    if encoded.wire_format == "p3":
        return _build_p3_payload(c067, encoded.encoded_action_stream)
    if encoded.wire_format == "p4":
        return _build_p4_payload(c067, encoded)
    if encoded.wire_format == "p5":
        return _build_p5_payload(c067, encoded, record_count)
    if encoded.wire_format == "p6":
        return _build_p6_payload(c067, encoded, record_count)
    raise ValueError(f"unsupported wire format: {encoded.wire_format}")


def _selection_guard_summary(records: list[ActionRecord]) -> dict[str, Any]:
    exact_counts: dict[tuple[int, int, int, tuple[float, float, float] | None], int] = {}
    pair_tile_counts: dict[tuple[int, int], int] = {}
    for rec in records:
        exact_key = (
            int(rec.pair_index),
            int(rec.tile_id),
            int(rec.action_id),
            tuple(float(x) for x in rec.custom_delta_rgb)
            if rec.custom_delta_rgb is not None
            else None,
        )
        exact_counts[exact_key] = exact_counts.get(exact_key, 0) + 1
        pair_tile_key = (int(rec.pair_index), int(rec.tile_id))
        pair_tile_counts[pair_tile_key] = pair_tile_counts.get(pair_tile_key, 0) + 1
    exact_duplicate_record_count = sum(count - 1 for count in exact_counts.values() if count > 1)
    pair_tile_duplicate_groups = [key for key, count in pair_tile_counts.items() if count > 1]
    transformed_records = [
        rec for rec in records if rec.transform != "identity" or rec.custom_delta_rgb is not None
    ]
    changed_action_records = [
        rec for rec in records if int(rec.action_id) != _record_source_action_id(rec)
    ]
    no_op_transformed_records = [
        rec
        for rec in transformed_records
        if int(rec.action_id) == _record_source_action_id(rec)
        and rec.custom_delta_rgb is None
    ]
    return {
        "record_count": len(records),
        "unique_source_record_count": len({_record_source_index(rec) for rec in records}),
        "exact_duplicate_record_count": exact_duplicate_record_count,
        "pair_tile_duplicate_group_count": len(pair_tile_duplicate_groups),
        "pair_tile_duplicate_record_count": sum(
            pair_tile_counts[key] - 1 for key in pair_tile_duplicate_groups
        ),
        "transformed_record_count": len(transformed_records),
        "changed_action_id_record_count": len(changed_action_records),
        "custom_delta_record_count": sum(1 for rec in records if rec.custom_delta_rgb is not None),
        "no_op_transformed_record_count": len(no_op_transformed_records),
        "runtime_signature_sha256": _sha256(
            json.dumps(_records_runtime_signature(records), sort_keys=True).encode("utf-8")
        ),
    }


def _unique_source_records(records: list[ActionRecord]) -> list[ActionRecord]:
    by_source: dict[int, ActionRecord] = {}
    for rec in sorted(records, key=_record_sort_key):
        by_source.setdefault(_record_source_index(rec), rec)
    return list(by_source.values())


def _source_preservation_summary(
    *,
    selected: list[ActionRecord],
    source_records: list[ActionRecord],
) -> dict[str, Any]:
    source_signature = _records_effective_signature(source_records)
    selected_signature = _records_effective_signature(selected)
    full_pr75_actions_preserved = selected_signature == source_signature
    selected_action_ids_preserved = all(
        int(rec.action_id) == _record_source_action_id(rec)
        and rec.custom_delta_rgb is None
        for rec in selected
    )
    return {
        "noop": False,
        "noop_status": "not_noop_action_payload_nonempty",
        "source_preserving": full_pr75_actions_preserved,
        "source_preservation": {
            "status": (
                "full_pr75_action_records_preserved"
                if full_pr75_actions_preserved
                else "subset_or_transformed_pr75_action_records"
            ),
            "full_pr75_action_records_preserved": full_pr75_actions_preserved,
            "selected_action_ids_preserved": selected_action_ids_preserved,
            "source_record_count": len(source_records),
            "selected_record_count": len(selected),
            "selected_effective_signature_sha256": _sha256(
                json.dumps(selected_signature, sort_keys=True).encode("utf-8")
            ),
            "source_effective_signature_sha256": _sha256(
                json.dumps(source_signature, sort_keys=True).encode("utf-8")
            ),
        },
    }


def _record_manifest(record: ActionRecord) -> dict[str, Any]:
    return {
        "index": int(record.index),
        "order_key": _record_sort_key(record),
        "source_index": _record_source_index(record),
        "pair_index": int(record.pair_index),
        "tile_id": int(record.tile_id),
        "action_id": int(record.action_id),
        "source_action_id": _record_source_action_id(record),
        "transform": record.transform,
        "custom_delta_rgb": (
            list(record.custom_delta_rgb)
            if record.custom_delta_rgb is not None
            else None
        ),
        "calibration_source": record.calibration_source,
        "calibration_rank": record.calibration_rank,
        "delta_combined": float(record.delta_combined),
        "delta_seg": float(record.delta_seg),
        "delta_pose": float(record.delta_pose),
    }


def _write_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("p", FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _candidate_plan(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.candidate:
        plan: list[tuple[str, str]] = []
        for raw in args.candidate:
            try:
                policy, wire_format = raw.rsplit(":", 1)
            except ValueError as exc:
                raise ValueError("--candidate must be POLICY:WIRE_FORMAT") from exc
            if wire_format not in {"p3", "p4", "p5", "p6"}:
                raise ValueError(f"unsupported candidate wire format: {wire_format}")
            plan.append((policy, wire_format))
        return plan
    return [(policy, wire_format) for policy in args.policy for wire_format in args.wire_format]


def build_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    c067 = _read_c067_slices(args.c067_archive)
    raw_records = _read_pr75_action_records(args.pr75_archive)
    base_trace = _trace_by_pair(args.base_component_trace)
    calibration_traces = _parse_calibration_component_traces(
        args.calibration_component_trace
    )
    ranked = _rank_records(
        raw_records,
        base_trace=base_trace,
        candidate_trace=_trace_by_pair(args.actions_component_trace),
    )
    baseline_component_score = _component_trace_score(args.base_component_trace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix: list[dict[str, Any]] = []
    for policy, wire_format in _candidate_plan(args):
        selected = _select_policy(
            ranked,
            policy,
            base_trace=base_trace,
            calibration_traces=calibration_traces,
        )
        encoded = _encode_actions(selected, wire_format)
        payload = _build_payload(c067, encoded, len(selected))
        name = f"c067_pr75_actions_{policy}_{wire_format}"
        out_dir = args.output_dir / name
        archive = out_dir / "archive.zip"
        _write_zip(archive, payload)
        action_br = brotli.compress(encoded.encoded_action_stream, quality=11)
        guard_summary = _selection_guard_summary(selected)
        source_preservation = _source_preservation_summary(
            selected=selected,
            source_records=ranked,
        )
        source_selected = _unique_source_records(selected)
        selected_delta_combined = sum(rec.delta_combined for rec in source_selected)
        selected_delta_seg = sum(rec.delta_seg for rec in source_selected)
        selected_delta_pose = sum(rec.delta_pose for rec in source_selected)
        formula_rate_delta = (
            25.0 * (archive.stat().st_size - args.c067_archive_bytes) / RATE_DENOM
        )
        manifest = {
            "schema_version": 2,
            "tool": "experiments/build_pr75_tile_action_subset_candidates.py",
            "name": name,
            "policy": policy,
            "wire_format": wire_format,
            "score_claim": False,
            **source_preservation,
            "evidence": "byte_and_trace_planning_only_until_exact_cuda",
            "source_c067_archive": str(args.c067_archive),
            "source_c067_archive_sha256": _file_sha256(args.c067_archive),
            "source_pr75_archive": str(args.pr75_archive),
            "source_pr75_archive_sha256": _file_sha256(args.pr75_archive),
            "base_component_trace": str(args.base_component_trace),
            "actions_component_trace": str(args.actions_component_trace),
            "calibration_component_traces": [
                {
                    "transform": trace.transform,
                    "max_rank": trace.max_rank,
                    "path": str(trace.path),
                }
                for trace in calibration_traces
            ],
            "selected_record_count": len(selected),
            "selected_unique_source_record_count": guard_summary[
                "unique_source_record_count"
            ],
            "source_record_count": len(raw_records) // 4,
            "action_selection_guard": guard_summary,
            "runtime_action_raw_bytes": len(encoded.raw_runtime_records),
            "runtime_action_raw_sha256": _sha256(encoded.raw_runtime_records),
            "encoded_action_stream_bytes": len(encoded.encoded_action_stream),
            "encoded_action_stream_sha256": _sha256(encoded.encoded_action_stream),
            "encoded_action_codec": encoded.encoded_action_codec,
            "encoded_action_brotli_bytes": len(action_br),
            "encoded_action_brotli_sha256": _sha256(action_br),
            "custom_action_dictionary_raw_bytes": len(encoded.action_dict_raw),
            "custom_action_dictionary_raw_sha256": (
                _sha256(encoded.action_dict_raw) if encoded.action_dict_raw else None
            ),
            "custom_action_dictionary_brotli_bytes": len(encoded.action_dict_br),
            "custom_action_dictionary_brotli_sha256": (
                _sha256(encoded.action_dict_br) if encoded.action_dict_br else None
            ),
            "custom_action_dictionary_entries": encoded.dictionary_entries,
            "selected_delta_combined_trace_sum": selected_delta_combined,
            "selected_delta_seg_trace_sum": selected_delta_seg,
            "selected_delta_pose_trace_sum": selected_delta_pose,
            "trace_delta_note": (
                "Trace deltas are counted once per source PR75 record. "
                "Amplitude-shifted, custom-dictionary, and signed-combo "
                "policies require exact CUDA eval for measured effect."
            ),
            "archive": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": _file_sha256(archive),
            "delta_bytes_vs_c067": archive.stat().st_size - args.c067_archive_bytes,
            "formula_rate_delta_vs_c067": formula_rate_delta,
            "estimated_score_delta_vs_c067_from_trace_and_bytes": (
                selected_delta_combined - formula_rate_delta
            ),
            "estimated_score_recomputed_from_c067_trace": (
                baseline_component_score - selected_delta_combined + formula_rate_delta
                if baseline_component_score is not None
                else None
            ),
            "selected_records": [_record_manifest(rec) for rec in selected],
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        matrix.append(manifest)
    ranked_matrix = sorted(
        matrix,
        key=lambda row: (
            row["estimated_score_recomputed_from_c067_trace"]
            if row["estimated_score_recomputed_from_c067_trace"] is not None
            else float("inf"),
            row["archive_size_bytes"],
            row["name"],
        ),
    )
    for priority, row in enumerate(ranked_matrix, start=1):
        row["exact_eval_priority_order"] = priority
    (args.output_dir / "candidate_matrix.json").write_text(
        json.dumps(ranked_matrix, indent=2, sort_keys=True) + "\n"
    )
    return ranked_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c067-archive", type=Path, required=True)
    parser.add_argument("--pr75-archive", type=Path, required=True)
    parser.add_argument("--base-component-trace", type=Path, required=True)
    parser.add_argument("--actions-component-trace", type=Path, required=True)
    parser.add_argument(
        "--calibration-component-trace",
        action="append",
        default=[],
        help=(
            "Exact component trace for calibrated Lagrangian policies as "
            "TRANSFORM:MAX_RANK:PATH, e.g. identity:40:.../component_trace.json "
            "or ampminus1:25:.../component_trace.json."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help=(
            "Selection policy: positive, pose_safe_positive, top<N>, "
            "top<N>_drop<R[_R...]>[_add<R[_R...]>], "
            "segtop<N>, posetop<N>, beam_rate_top<N>, "
            "beam_pose<W>_top<N> where W may use p as decimal, "
            "all_ampminus1, poseharm_ampminus1, or "
            "positive_poseharm_ampminus1. The positive, pose_safe_positive, "
            "and top<N> policies may also use _ampminus{1,2,3} or "
            "_ampplus{1,2} suffixes. Non-P3/custom variants may use "
            "_ampfit, _ampfit_pose, _signedboost1, _signedshrink1, "
            "_signedposemix1, _custompose125, _custompose150, "
            "_customboost125, _wilddiramp{4,6,8}, or _wilddirmean."
        ),
    )
    parser.add_argument(
        "--wire-format",
        action="append",
        choices=("p3", "p4", "p5", "p6"),
        default=[],
        help=(
            "Archive payload format. p3 uses the runtime fixed dictionary; "
            "p4 carries a charged custom subset dictionary and 4-byte records; "
            "p5 carries the same dictionary with 3-byte packed records; "
            "p6 carries fixed-dictionary action records as pair-delta varints."
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Exact candidate to emit as POLICY:WIRE_FORMAT; bypasses policy/wire cross-product.",
    )
    parser.add_argument("--c067-archive-bytes", type=int, default=276_214)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.policy:
        args.policy = [
            "positive",
            "pose_safe_positive",
            "top40",
            "top25",
            "top49",
            "beam_rate_top67",
            "beam_pose2_top55",
            "beam_pose4_top55",
            "top40_ampminus1",
            "top25_ampminus1",
            "top40_ampminus2",
            "top40_ampfit",
            "top49_ampfit",
            "top40_signedposemix1",
            "all_ampminus1",
            "poseharm_ampminus1",
            "positive_poseharm_ampminus1",
        ]
    if not args.wire_format:
        args.wire_format = ["p3"]
    matrix = build_candidates(args)
    for row in matrix:
        print(
            row["name"],
            row["archive_size_bytes"],
            row["delta_bytes_vs_c067"],
            row["selected_record_count"],
            row["archive_sha256"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
