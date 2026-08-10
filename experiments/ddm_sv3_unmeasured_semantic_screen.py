#!/usr/bin/env python3
"""Screen retained semantic re-representations before scorer admission.

The tool has two deliberately separate instruments:

* exact reconstructed-weight differences for every tensor in an ``SM3STATE``
  payload; and
* streaming RGB24 differences by frame parity for retained decoded RAW files.

It does not decode candidates and it does not run either scorer.  Every input is
an already-retained payload.  The JSON output records the byte identity of each
input and the sensitivity limits of both instruments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SM3STATE_MAGIC = b"SM3STATE\x01"
DEFAULT_FRAME_BYTES = 874 * 1164 * 3
DEFAULT_FRAME_COUNT = 1200


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_named_path(text: str) -> tuple[str, Path]:
    candidate_id, separator, raw_path = text.partition("=")
    if not separator or not candidate_id or not raw_path:
        raise argparse.ArgumentTypeError("expected CANDIDATE_ID=PATH")
    return candidate_id, Path(raw_path)


def parse_sm3state(path: Path) -> dict[str, np.ndarray]:
    payload = path.read_bytes()
    view = memoryview(payload)
    offset = len(SM3STATE_MAGIC)
    if view[:offset].tobytes() != SM3STATE_MAGIC:
        raise ValueError(f"not an SM3STATE v1 payload: {path}")
    if len(view) < offset + 4:
        raise ValueError(f"truncated SM3STATE tensor count: {path}")
    tensor_count = struct.unpack_from("<I", view, offset)[0]
    offset += 4
    state: dict[str, np.ndarray] = {}
    for _ in range(tensor_count):
        if len(view) < offset + 2:
            raise ValueError(f"truncated SM3STATE name length: {path}")
        name_bytes = struct.unpack_from("<H", view, offset)[0]
        offset += 2
        if len(view) < offset + name_bytes + 1:
            raise ValueError(f"truncated SM3STATE name or rank: {path}")
        name = view[offset : offset + name_bytes].tobytes().decode()
        offset += name_bytes
        rank = int(view[offset])
        offset += 1
        shape_bytes = rank * 4
        if len(view) < offset + shape_bytes + 8:
            raise ValueError(f"truncated SM3STATE shape: {path}")
        shape = struct.unpack_from(f"<{rank}I", view, offset) if rank else ()
        offset += shape_bytes
        data_bytes = struct.unpack_from("<Q", view, offset)[0]
        offset += 8
        expected_bytes = math.prod(shape) * np.dtype("<f4").itemsize
        if data_bytes != expected_bytes or len(view) < offset + data_bytes:
            raise ValueError(f"invalid SM3STATE tensor payload for {name}: {path}")
        if name in state:
            raise ValueError(f"duplicate SM3STATE tensor {name}: {path}")
        state[name] = np.frombuffer(
            view[offset : offset + data_bytes], dtype="<f4"
        ).reshape(shape).copy()
        offset += data_bytes
    if offset != len(view):
        raise ValueError(f"SM3STATE has {len(view) - offset} trailing bytes: {path}")
    return state


def compare_states(
    base: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    relative_l2_alert: float,
) -> dict[str, Any]:
    if list(base) != list(candidate):
        raise ValueError("candidate tensor names or order differ from base")
    tensor_rows: list[dict[str, Any]] = []
    base_sq_total = 0.0
    error_sq_total = 0.0
    changed_total = 0
    value_total = 0
    absolute_total = 0.0
    max_absolute = 0.0
    for name, base_value in base.items():
        candidate_value = candidate[name]
        if base_value.shape != candidate_value.shape:
            raise ValueError(f"candidate tensor shape differs for {name}")
        base64 = base_value.astype(np.float64, copy=False)
        delta = candidate_value.astype(np.float64, copy=False) - base64
        base_sq = float(np.square(base64).sum(dtype=np.float64))
        error_sq = float(np.square(delta).sum(dtype=np.float64))
        absolute = np.abs(delta)
        count = int(delta.size)
        changed = int(np.count_nonzero(delta))
        rel_l2 = math.sqrt(error_sq / base_sq) if base_sq else (math.inf if error_sq else 0.0)
        row_max = float(absolute.max(initial=0.0))
        tensor_rows.append(
            {
                "name": name,
                "shape": list(base_value.shape),
                "values": count,
                "changed_values": changed,
                "changed_fraction": changed / count if count else 0.0,
                "mean_absolute_error": float(absolute.sum(dtype=np.float64) / count) if count else 0.0,
                "max_absolute_error": row_max,
                "error_l2": math.sqrt(error_sq),
                "base_l2": math.sqrt(base_sq),
                "relative_l2": rel_l2,
                "relative_l2_alert": rel_l2 >= relative_l2_alert,
            }
        )
        base_sq_total += base_sq
        error_sq_total += error_sq
        changed_total += changed
        value_total += count
        absolute_total += float(absolute.sum(dtype=np.float64))
        max_absolute = max(max_absolute, row_max)
    global_relative_l2 = (
        math.sqrt(error_sq_total / base_sq_total)
        if base_sq_total
        else (math.inf if error_sq_total else 0.0)
    )
    alerting_tensors = [row["name"] for row in tensor_rows if row["relative_l2_alert"]]
    return {
        "tensor_denominator": len(tensor_rows),
        "value_denominator": value_total,
        "changed_values": changed_total,
        "changed_fraction": changed_total / value_total if value_total else 0.0,
        "mean_absolute_error": absolute_total / value_total if value_total else 0.0,
        "max_absolute_error": max_absolute,
        "error_l2": math.sqrt(error_sq_total),
        "base_l2": math.sqrt(base_sq_total),
        "relative_l2": global_relative_l2,
        "relative_l2_alert_threshold": relative_l2_alert,
        "alerting_tensors": alerting_tensors,
        "alert_fired": global_relative_l2 >= relative_l2_alert or bool(alerting_tensors),
        "tensors": tensor_rows,
    }


def _empty_parity_accumulator() -> dict[str, int | float]:
    return {
        "frames": 0,
        "byte_denominator": 0,
        "rgb_pixel_denominator": 0,
        "absolute_sum": 0,
        "squared_sum": 0,
        "max_absolute": 0,
        "changed_bytes": 0,
        "changed_rgb_pixels": 0,
        "worst_frame_mean_absolute": 0.0,
    }


def compare_rgb24_raws(
    base_path: Path,
    candidate_path: Path,
    *,
    frame_bytes: int,
    frame_count: int,
    catastrophic_mean_absolute: float,
    catastrophic_changed_rgb_fraction: float,
) -> dict[str, Any]:
    expected_bytes = frame_bytes * frame_count
    if frame_bytes <= 0 or frame_bytes % 3:
        raise ValueError("frame bytes must be positive RGB24 geometry")
    if base_path.stat().st_size != expected_bytes:
        raise ValueError(f"base RAW size differs from expected {expected_bytes}: {base_path}")
    if candidate_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"candidate RAW size differs from expected {expected_bytes}: {candidate_path}"
        )
    parity = {"even_pose_carrier": _empty_parity_accumulator(), "odd_semantic": _empty_parity_accumulator()}
    with base_path.open("rb") as base_handle, candidate_path.open("rb") as candidate_handle:
        for frame_index in range(frame_count):
            base_payload = base_handle.read(frame_bytes)
            candidate_payload = candidate_handle.read(frame_bytes)
            if len(base_payload) != frame_bytes or len(candidate_payload) != frame_bytes:
                raise ValueError(f"short RAW read at frame {frame_index}")
            base = np.frombuffer(base_payload, dtype=np.uint8)
            candidate = np.frombuffer(candidate_payload, dtype=np.uint8)
            signed_delta = candidate.astype(np.int16) - base.astype(np.int16)
            absolute = np.abs(signed_delta).astype(np.uint16, copy=False)
            changed_bytes = absolute != 0
            changed_pixels = np.any(changed_bytes.reshape(-1, 3), axis=1)
            row = parity["even_pose_carrier" if frame_index % 2 == 0 else "odd_semantic"]
            row["frames"] = int(row["frames"]) + 1
            row["byte_denominator"] = int(row["byte_denominator"]) + frame_bytes
            row["rgb_pixel_denominator"] = int(row["rgb_pixel_denominator"]) + frame_bytes // 3
            frame_absolute_sum = int(absolute.sum(dtype=np.int64))
            row["absolute_sum"] = int(row["absolute_sum"]) + frame_absolute_sum
            row["squared_sum"] = int(row["squared_sum"]) + int(
                np.square(signed_delta.astype(np.int32), dtype=np.int64).sum(dtype=np.int64)
            )
            row["max_absolute"] = max(int(row["max_absolute"]), int(absolute.max(initial=0)))
            row["changed_bytes"] = int(row["changed_bytes"]) + int(changed_bytes.sum(dtype=np.int64))
            row["changed_rgb_pixels"] = int(row["changed_rgb_pixels"]) + int(
                changed_pixels.sum(dtype=np.int64)
            )
            row["worst_frame_mean_absolute"] = max(
                float(row["worst_frame_mean_absolute"]), frame_absolute_sum / frame_bytes
            )
    finalized: dict[str, Any] = {}
    catastrophic_parities: list[str] = []
    for parity_name, raw in parity.items():
        byte_denominator = int(raw["byte_denominator"])
        pixel_denominator = int(raw["rgb_pixel_denominator"])
        mean_absolute = int(raw["absolute_sum"]) / byte_denominator
        root_mean_square = math.sqrt(int(raw["squared_sum"]) / byte_denominator)
        changed_rgb_fraction = int(raw["changed_rgb_pixels"]) / pixel_denominator
        row = {
            "frames": int(raw["frames"]),
            "byte_denominator": byte_denominator,
            "rgb_pixel_denominator": pixel_denominator,
            "mean_absolute_delta": mean_absolute,
            "root_mean_square_delta": root_mean_square,
            "max_absolute_delta": int(raw["max_absolute"]),
            "changed_bytes": int(raw["changed_bytes"]),
            "changed_byte_fraction": int(raw["changed_bytes"]) / byte_denominator,
            "changed_rgb_pixels": int(raw["changed_rgb_pixels"]),
            "changed_rgb_pixel_fraction": changed_rgb_fraction,
            "worst_frame_mean_absolute_delta": float(raw["worst_frame_mean_absolute"]),
            "byte_identical": int(raw["changed_bytes"]) == 0,
        }
        row["catastrophic_alert"] = (
            mean_absolute >= catastrophic_mean_absolute
            or (
                changed_rgb_fraction >= catastrophic_changed_rgb_fraction
                and root_mean_square >= catastrophic_mean_absolute
            )
        )
        if row["catastrophic_alert"]:
            catastrophic_parities.append(parity_name)
        finalized[parity_name] = row
    return {
        "frame_bytes": frame_bytes,
        "frame_count": frame_count,
        "expected_raw_bytes": expected_bytes,
        "catastrophic_mean_absolute_threshold": catastrophic_mean_absolute,
        "catastrophic_changed_rgb_fraction_threshold": catastrophic_changed_rgb_fraction,
        "catastrophic_alert_rule": (
            "mean_absolute >= mean_threshold OR "
            "(changed_rgb_fraction >= fraction_threshold AND rms >= mean_threshold)"
        ),
        "catastrophic_parities": catastrophic_parities,
        "catastrophic_alert_fired": bool(catastrophic_parities),
        "parity": finalized,
    }


def screen(
    *,
    base_state_path: Path,
    candidate_states: Mapping[str, Path],
    base_raw_path: Path | None,
    candidate_raws: Mapping[str, Path],
    positive_control: str,
    frame_bytes: int,
    frame_count: int,
    relative_l2_alert: float,
    catastrophic_mean_absolute: float,
    catastrophic_changed_rgb_fraction: float,
    base_state_record: Mapping[str, Any] | None = None,
    base_raw_record: Mapping[str, Any] | None = None,
    completed_candidates: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoint: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if positive_control not in candidate_states:
        raise ValueError("positive control is absent from candidate states")
    unknown_raws = sorted(set(candidate_raws) - set(candidate_states))
    if unknown_raws:
        raise ValueError(f"candidate RAW has no matching state: {unknown_raws}")
    if candidate_raws and base_raw_path is None:
        raise ValueError("candidate RAW inputs require --base-raw")
    base_state = parse_sm3state(base_state_path)
    actual_base_state_record = dict(base_state_record or file_record(base_state_path))
    actual_base_raw_record = (
        dict(base_raw_record or file_record(base_raw_path))
        if base_raw_path is not None
        else None
    )
    completed_candidates = completed_candidates or {}
    candidates: dict[str, Any] = {}
    for candidate_id, state_path in candidate_states.items():
        if candidate_id in completed_candidates:
            completed = dict(completed_candidates[candidate_id])
            if completed.get("state") != file_record(state_path):
                raise ValueError(f"resumed state payload changed for {candidate_id}")
            expected_raw_record = (
                file_record(candidate_raws[candidate_id])
                if candidate_id in candidate_raws
                else None
            )
            if completed.get("raw") != expected_raw_record:
                raise ValueError(f"resumed RAW payload changed for {candidate_id}")
            candidates[candidate_id] = completed
            continue
        state_record = file_record(state_path)
        weight_result = compare_states(
            base_state,
            parse_sm3state(state_path),
            relative_l2_alert=relative_l2_alert,
        )
        raw_result = None
        raw_record = None
        if candidate_id in candidate_raws:
            raw_path = candidate_raws[candidate_id]
            raw_record = file_record(raw_path)
            raw_result = compare_rgb24_raws(
                base_raw_path,
                raw_path,
                frame_bytes=frame_bytes,
                frame_count=frame_count,
                catastrophic_mean_absolute=catastrophic_mean_absolute,
                catastrophic_changed_rgb_fraction=catastrophic_changed_rgb_fraction,
            )
        candidate_result = {
            "state": state_record,
            "raw": raw_record,
            "weight_screen": weight_result,
            "frame_parity_screen": raw_result,
        }
        candidates[candidate_id] = candidate_result
        if checkpoint is not None:
            checkpoint(candidate_id, candidate_result)
    control = candidates[positive_control]
    if control["weight_screen"]["alert_fired"] is not True:
        raise RuntimeError("positive control did not fire the reconstructed-weight alert")
    if positive_control in candidate_raws:
        if control["frame_parity_screen"]["catastrophic_alert_fired"] is not True:
            raise RuntimeError("positive control did not fire the frame-parity catastrophe alert")
        positive_control_status = "PASS_WEIGHT_AND_RAW_ALERTS_FIRED"
    else:
        positive_control_status = "PASS_WEIGHT_ALERT_FIRED_RAW_NOT_PROVIDED"
    return {
        "schema": "ddm_sv3_semantic_screen.v1",
        "written_at_utc": utc_now(),
        "axis": "[scorer-free exact reconstructed weights and retained RGB24 RAW bytes]",
        "score_claim": False,
        "base": {"state": actual_base_state_record, "raw": actual_base_raw_record},
        "positive_control": positive_control,
        "positive_control_status": positive_control_status,
        "thresholds": {
            "relative_l2_alert": relative_l2_alert,
            "catastrophic_mean_absolute": catastrophic_mean_absolute,
            "catastrophic_changed_rgb_fraction": catastrophic_changed_rgb_fraction,
        },
        "resolution": {
            "weight_screen_can_see": (
                "exact fp32 differences in every reconstructed tensor, down to any nonzero stored value"
            ),
            "weight_screen_cannot_see": (
                "activation sensitivity, rendered-pixel effects, SegNet argmax changes, or PoseNet output changes"
            ),
            "weight_alert_semantics": (
                "triage alert calibrated to fire on the known low-rank catastrophe; it is not a safety or score verdict"
            ),
            "raw_screen_can_see": (
                "every changed decoded RGB24 byte and RGB pixel, exact parity attribution, and delta magnitude"
            ),
            "raw_screen_cannot_see": (
                "SegNet or PoseNet sensitivity to small pixel changes; a non-catastrophic raw result still requires the scorer"
            ),
        },
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--candidate-state", action="append", type=parse_named_path, required=True)
    parser.add_argument("--base-raw", type=Path)
    parser.add_argument("--candidate-raw", action="append", type=parse_named_path, default=[])
    parser.add_argument("--positive-control", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--frame-bytes", type=int, default=DEFAULT_FRAME_BYTES)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--relative-l2-alert", type=float, default=0.05)
    parser.add_argument("--catastrophic-mean-absolute", type=float, default=5.0)
    parser.add_argument("--catastrophic-changed-rgb-fraction", type=float, default=0.5)
    return parser.parse_args()


def unique_named_paths(rows: list[tuple[str, Path]], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for candidate_id, path in rows:
        if candidate_id in result:
            raise ValueError(f"duplicate {label} candidate: {candidate_id}")
        result[candidate_id] = path
    return result


def main() -> None:
    args = parse_args()
    candidate_states = unique_named_paths(args.candidate_state, "state")
    candidate_raws = unique_named_paths(args.candidate_raw, "RAW")
    measurement_source = file_record(Path(__file__))
    base_state_record = file_record(args.base_state)
    base_raw_record = file_record(args.base_raw) if args.base_raw is not None else None
    binding = {
        "measurement_source": measurement_source,
        "base_state": base_state_record,
        "base_raw": base_raw_record,
        "candidate_states": {name: str(path.resolve()) for name, path in candidate_states.items()},
        "candidate_raws": {name: str(path.resolve()) for name, path in candidate_raws.items()},
        "positive_control": args.positive_control,
        "frame_bytes": args.frame_bytes,
        "frame_count": args.frame_count,
        "relative_l2_alert": args.relative_l2_alert,
        "catastrophic_mean_absolute": args.catastrophic_mean_absolute,
        "catastrophic_changed_rgb_fraction": args.catastrophic_changed_rgb_fraction,
    }
    invocation: dict[str, Any] = {
        "schema": "ddm_sv3_semantic_screen_resume.v1",
        "updated_at_utc": utc_now(),
        "complete": False,
        "argv": list(os.sys.argv),
        "binding": binding,
        "candidate_results": {},
    }
    if args.resume_from.is_file():
        prior = json.loads(args.resume_from.read_text())
        if prior.get("schema") != invocation["schema"] or prior.get("binding") != binding:
            raise ValueError("resume binding differs from this screen invocation")
        invocation = prior
        invocation["complete"] = False
        invocation["updated_at_utc"] = utc_now()
    atomic_json(args.resume_from, invocation)

    def save_candidate(candidate_id: str, candidate_result: Mapping[str, Any]) -> None:
        invocation["candidate_results"][candidate_id] = dict(candidate_result)
        invocation["completed_candidates"] = list(invocation["candidate_results"])
        invocation["updated_at_utc"] = utc_now()
        atomic_json(args.resume_from, invocation)

    result = screen(
        base_state_path=args.base_state,
        candidate_states=candidate_states,
        base_raw_path=args.base_raw,
        candidate_raws=candidate_raws,
        positive_control=args.positive_control,
        frame_bytes=args.frame_bytes,
        frame_count=args.frame_count,
        relative_l2_alert=args.relative_l2_alert,
        catastrophic_mean_absolute=args.catastrophic_mean_absolute,
        catastrophic_changed_rgb_fraction=args.catastrophic_changed_rgb_fraction,
        base_state_record=base_state_record,
        base_raw_record=base_raw_record,
        completed_candidates=invocation["candidate_results"],
        checkpoint=save_candidate,
    )
    result["provenance"] = {
        "argv": list(os.sys.argv),
        "measurement_source": measurement_source,
    }
    atomic_json(args.out, result)
    invocation["complete"] = True
    invocation["updated_at_utc"] = utc_now()
    invocation["result"] = file_record(args.out)
    atomic_json(args.resume_from, invocation)
    print(json.dumps({"complete": True, "result": str(args.out.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
