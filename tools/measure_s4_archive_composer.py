#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure Task #578 S4 native/standalone parity and local advisory score.

The parity path intentionally replays the payload with repository-native
parsers and realization operators instead of importing the standalone
receiver.  The advisory path materializes its only bulky scratch on the SSD,
checkpoints each stage, and deletes the raw file after a successful evaluator
run unless ``--keep-inflated`` is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import shutil
import struct
import subprocess
import time
import zlib
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

from tac.boundary_math.analytic_lane_render_band import (
    deserialize_lane_band_any,
    render_config_from_header,
)
from tac.optimization.predict_project_receiver import (
    _catmull_rom,
    _interpolate_track_knot,
    _nearest_shift,
)
from tac.optimization.predict_project_schema import parse_constraint_seed
from tac.optimization.predictor_r3_causal import (
    decode_component_event_alphabet_raw,
    parse_static_chart_quotient,
)
from tac.optimization.predictor_upgrade_xi_chart import parse_static_charts, render_lane_mask
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.s4_archive_composer import section_map

PAIR_COUNT: Final = 600
SCORER_H: Final = 384
SCORER_W: Final = 512
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
LZMA_FILTERS: Final = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}
]
DEFAULT_ARTIFACT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721/canonical_s4_20260721"
)
DEFAULT_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721/measurement_s4_20260721"
)
UPSTREAM: Final = Path("/Users/adpena/Projects/pact/upstream")
HOST_PYTHON: Final = Path("/Users/adpena/Projects/pact/.venv/bin/python")


class MeasureError(RuntimeError):
    """A parity, custody, storage, or evaluation gate failed closed."""


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _storage_preflight(path: Path, required: int) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < required:
        raise MeasureError(f"storage preflight failed: {free} < {required} at {path}")
    return {"path": str(path), "free_bytes": free, "required_free_bytes": required, "passed": True}


def _parse_component_packets(payload: bytes) -> list[list[tuple[int, np.ndarray]]]:
    frames: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(PAIR_COUNT)]
    offset = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise MeasureError("component packet length is truncated")
        size = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        if not size or offset + size > len(payload):
            raise MeasureError("component packet size is invalid")
        raw = zlib.decompress(payload[offset : offset + size])
        offset += size
        if len(raw) < 12:
            raise MeasureError("component packet header is truncated")
        frame, class_id, _stratum, count, first = struct.unpack_from("<HBBII", raw)
        if frame >= PAIR_COUNT or class_id >= 5 or count == 0:
            raise MeasureError("component packet metadata is invalid")
        cursor, values = 12, [first]
        for _ in range(count - 1):
            value, shift = 0, 0
            while True:
                if cursor >= len(raw) or shift > 63:
                    raise MeasureError("component delta varint is truncated or overlong")
                byte = raw[cursor]
                cursor += 1
                value |= (byte & 127) << shift
                if not byte & 128:
                    break
                shift += 7
            if value <= 0:
                raise MeasureError("component deltas must be positive")
            values.append(values[-1] + value)
        if cursor != len(raw) or values[-1] >= SCORER_H * SCORER_W:
            raise MeasureError("component packet has trailing or out-of-grid data")
        frames[frame].append((class_id, np.asarray(values, dtype=np.int64)))
    return frames


def _trajectory_at_validated(seed: dict[str, Any], time_index: int) -> tuple[float, float, float]:
    """Use the native spline on the seed already validated by its binary parser."""

    controls = seed["trajectory"]["controls"]
    times = [row["time"] for row in controls]
    values = [[row[key] for row in controls] for key in ("tx_q", "ty_q", "yaw_q")]
    residual = next(
        (row for row in seed["trajectory"]["ar_residuals"] if row["time"] == time_index), None
    )
    deltas = (0, 0, 0) if residual is None else (
        residual["dtx_q"],
        residual["dty_q"],
        residual["dyaw_q"],
    )
    quanta = seed["units"]
    translation = quanta["trajectory_translation"]["numerator"] / quanta[
        "trajectory_translation"
    ]["denominator"]
    rotation = quanta["trajectory_rotation"]["numerator"] / quanta["trajectory_rotation"][
        "denominator"
    ]
    return (
        (_catmull_rom(values[0], times, time_index) + deltas[0]) * translation,
        (_catmull_rom(values[1], times, time_index) + deltas[1]) * translation,
        (_catmull_rom(values[2], times, time_index) + deltas[2]) * rotation,
    )


def _native_prepare(payload: bytes) -> tuple[dict[str, Any], dict[str, float]]:
    started = time.perf_counter()
    sections = section_map(payload)
    manifest = json.loads(sections["manifest.json"].payload.decode("ascii"))
    seed = parse_constraint_seed(sections["seed.ppcs"].payload)
    palette = np.asarray(
        manifest["weight_derived_constants"]["R2_max_margin_palette"]["value_u8"], dtype=np.uint8
    )
    if palette.shape != (5, 3):
        raise MeasureError("counted palette shape mismatch")
    base = sections["base.pbase3"].payload
    if len(base) < 8:
        raise MeasureError("PBASE3 is truncated")
    static_len, lane_len = struct.unpack_from("<II", base)
    if len(base) != 8 + static_len + lane_len:
        raise MeasureError("PBASE3 length mismatch")
    pxq = brotli.decompress(base[8 : 8 + static_len])
    charts = parse_static_charts(parse_static_chart_quotient(pxq))
    lane_raw = lzma.decompress(
        base[8 + static_len :], format=lzma.FORMAT_RAW, filters=LZMA_FILTERS
    )
    lanes, lane_header = deserialize_lane_band_any(lane_raw)
    lane_config = render_config_from_header(lane_header)
    event_section = sections["events.pce3"]
    event_raw = lzma.decompress(
        event_section.payload, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS
    )
    if len(event_raw) != event_section.decoded_bytes:
        raise MeasureError("event decoded length mismatch")
    events = decode_component_event_alphabet_raw(event_raw)
    components = _parse_component_packets(sections["components.pcomp3"].payload)
    causal = sections["causal.pcr3"]
    if causal.payload or causal.decoded_bytes:
        raise MeasureError("selected receiver expects zero-parameter causal policy")
    constraints: list[list[dict[str, Any]]] = [[] for _ in range(PAIR_COUNT)]
    for row in seed["constraint_seeds"]:
        constraints[row["time"]].append(row)
    trajectories = [_trajectory_at_validated(seed, pair) for pair in range(PAIR_COUNT)]
    state = {
        "seed": seed,
        "palette": palette,
        "charts": charts,
        "lanes": lanes,
        "lane_config": lane_config,
        "events": events,
        "components": components,
        "constraints": constraints,
        "trajectories": trajectories,
        "kernel": FullResizeKernel.build(
            camera_h=CAMERA_H, camera_w=CAMERA_W, scorer_h=SCORER_H, scorer_w=SCORER_W
        ),
    }
    return state, {"parse_and_state_seconds": time.perf_counter() - started}


def _apply_tracks(field: np.ndarray, tracks: list[dict[str, Any]], pair: int) -> None:
    for track in tracks:
        knots = track["knots"]
        if pair < knots[0]["time"] or pair > knots[-1]["time"]:
            continue
        cy = _interpolate_track_knot(knots, pair, "y_q") / 256.0
        cx = _interpolate_track_knot(knots, pair, "x_q") / 256.0
        height = max(1, round(_interpolate_track_knot(knots, pair, "height_q") / 256.0))
        width = max(1, round(_interpolate_track_knot(knots, pair, "width_q") / 256.0))
        y0, x0 = max(0, round(cy - height / 2)), max(0, round(cx - width / 2))
        field[y0 : min(SCORER_H, y0 + height), x0 : min(SCORER_W, x0 + width)] = track[
            "cell_id"
        ]


def native_decode(payload: bytes, pairs: int) -> dict[str, Any]:
    if not 1 <= pairs <= PAIR_COUNT:
        raise MeasureError("pairs must be in [1,600]")
    state, timing = _native_prepare(payload)
    digest, pair_hashes = hashlib.sha256(), []
    previous: np.ndarray | None = None
    previous_pose = (0.0, 0.0, 0.0)
    realization_started = time.perf_counter()
    for pair in range(pairs):
        pose = state["trajectories"][pair]
        if previous is None:
            field = np.zeros((SCORER_H, SCORER_W), dtype=np.uint8)
            field[state["charts"].road_undrivable == 2] = 2
        else:
            field = _nearest_shift(
                previous,
                pose[0] - previous_pose[0],
                pose[1] - previous_pose[1],
                pose[2] - previous_pose[2],
            )
        static = np.isin(field, (0, 2)) & (state["charts"].road_undrivable != 0)
        field[static] = np.where(state["charts"].road_undrivable[static] == 1, 0, 2).astype(
            np.uint8
        )
        field[
            render_lane_mask(
                state["lanes"][pair], state["lane_config"], h=SCORER_H, w=SCORER_W
            )
        ] = 1
        field[state["charts"].hood] = 4
        _apply_tracks(field, state["seed"]["movable_tracks"], pair)
        flat = field.reshape(-1)
        for class_id, class_components in enumerate(state["events"][pair]):
            for sites in class_components:
                flat[sites] = class_id
        for class_id, sites in state["components"][pair]:
            flat[sites] = class_id
        for row in state["constraints"][pair]:
            field[row["y"], row["x"]] = row["cell_id"]
        frame = state["kernel"].operator.realize_factor2_uint8(state["palette"][field])
        raw = frame.tobytes(order="C")
        pair_hashes.append(_sha(raw + raw))
        digest.update(raw)
        digest.update(raw)
        previous, previous_pose = field.copy(), pose
    timing["realization_seconds"] = time.perf_counter() - realization_started
    timing["total_seconds"] = timing["parse_and_state_seconds"] + timing["realization_seconds"]
    return {
        "schema": "s4_repo_native_decode_receipt.v1",
        "pairs": pairs,
        "output_bytes": pairs * 2 * CAMERA_H * CAMERA_W * 3,
        "stream_sha256": digest.hexdigest(),
        "first_pair_sha256": pair_hashes[0],
        "last_pair_sha256": pair_hashes[-1],
        "timing": timing,
        "scorer_invocations": 0,
    }


def _standalone_decode(runtime: Path, payload: Path, pairs: int, receipt: Path) -> dict[str, Any]:
    command = [
        str(HOST_PYTHON),
        str(runtime),
        str(payload),
        "--max-pairs",
        str(pairs),
        "--hash-only",
        "--receipt",
        str(receipt),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    value = json.loads(completed.stdout)
    value["wall_seconds"] = time.perf_counter() - started
    value["command"] = command
    return value


def measure_parity(artifact: Path, output: Path) -> dict[str, Any]:
    payload_path = artifact / "0.bin"
    runtime = artifact / "runtime" / "inflate.py"
    payload = payload_path.read_bytes()
    checkpoint_path = output / "parity_checkpoint.json"
    checkpoint: dict[str, Any] = {
        "schema": "s4_parity_checkpoint.v1",
        "archive_sha256": _sha_file(artifact / "archive.zip"),
        "payload_sha256": _sha(payload),
        "runtime_sha256": _sha_file(runtime),
        "rows": [],
    }
    for pairs in (16, 64, 600):
        native = native_decode(payload, pairs)
        standalone_a = _standalone_decode(runtime, payload_path, pairs, output / f"standalone_n{pairs}_a.json")
        standalone_b = _standalone_decode(runtime, payload_path, pairs, output / f"standalone_n{pairs}_b.json")
        keys = ("pairs", "output_bytes", "stream_sha256", "first_pair_sha256", "last_pair_sha256")
        parity = all(native[key] == standalone_a[key] == standalone_b[key] for key in keys)
        deterministic = all(standalone_a[key] == standalone_b[key] for key in keys)
        row = {
            "pairs": pairs,
            "repo_native": native,
            "standalone_first": standalone_a,
            "standalone_second": standalone_b,
            "byte_exact_parity": parity,
            "standalone_double_decode_deterministic": deterministic,
        }
        checkpoint["rows"].append(row)
        _atomic_json(checkpoint_path, checkpoint)
        if not parity or not deterministic:
            raise MeasureError(f"n{pairs} native/standalone parity failed")
    checkpoint["passed"] = True
    _atomic_json(checkpoint_path, checkpoint)
    return checkpoint


def _parse_report(report: str) -> dict[str, float | int]:
    patterns = {
        "d_pose": r"Average PoseNet Distortion: ([0-9.eE+-]+)",
        "d_seg": r"Average SegNet Distortion: ([0-9.eE+-]+)",
        "archive_bytes": r"Submission file size: ([0-9,]+) bytes",
        "rate": r"Compression Rate: ([0-9.eE+-]+)",
        "score": r"= ([0-9.eE+-]+)\s*$",
    }
    values: dict[str, float | int] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, report, flags=re.MULTILINE)
        if match is None:
            raise MeasureError(f"upstream report lacks {name}")
        values[name] = int(match.group(1).replace(",", "")) if name == "archive_bytes" else float(match.group(1))
    return values


def advisory_eval(artifact: Path, output: Path, *, keep_inflated: bool) -> dict[str, Any]:
    preflight = _storage_preflight(output, 8 << 30)
    submission = output / "submission"
    inflated = submission / "inflated"
    inflated.mkdir(parents=True, exist_ok=True)
    raw_path = inflated / "0.raw"
    inflate_receipt = output / "standalone_n600_materialized.json"
    shutil.copy2(artifact / "archive.zip", submission / "archive.zip")
    names = output / "video_names.txt"
    names.write_text("0.mkv\n", encoding="ascii")
    checkpoint_path = output / "advisory_eval_checkpoint.json"
    checkpoint: dict[str, Any] = {
        "schema": "s4_advisory_eval_checkpoint.v1",
        "axis": "macOS-CPU advisory",
        "storage_preflight": preflight,
        "archive_sha256": _sha_file(artifact / "archive.zip"),
        "archive_bytes": (artifact / "archive.zip").stat().st_size,
        "stages": {},
    }
    expected_bytes = PAIR_COUNT * 2 * CAMERA_H * CAMERA_W * 3
    reusable = False
    if raw_path.is_file() and inflate_receipt.is_file():
        prior = json.loads(inflate_receipt.read_text(encoding="utf-8"))
        reusable = prior.get("output_bytes") == expected_bytes and raw_path.stat().st_size == expected_bytes
    if not reusable:
        command = [
            str(HOST_PYTHON),
            str(artifact / "runtime" / "inflate.py"),
            str(artifact / "0.bin"),
            str(raw_path),
            "--max-pairs",
            str(PAIR_COUNT),
            "--receipt",
            str(inflate_receipt),
        ]
        started = time.perf_counter()
        subprocess.run(command, check=True)
        checkpoint["stages"]["inflate"] = {
            "status": "complete",
            "command": command,
            "wall_seconds": time.perf_counter() - started,
            "receipt": str(inflate_receipt),
            "output_bytes": raw_path.stat().st_size,
        }
        _atomic_json(checkpoint_path, checkpoint)
    else:
        checkpoint["stages"]["inflate"] = {"status": "reused_validated", "receipt": str(inflate_receipt)}
    report_path = output / "upstream_evaluate_macos_cpu_advisory.txt"
    command = [
        str(HOST_PYTHON),
        str(UPSTREAM / "evaluate.py"),
        "--batch-size",
        "32",
        "--num-threads",
        "2",
        "--prefetch-queue-depth",
        "4",
        "--submission-dir",
        str(submission),
        "--uncompressed-dir",
        str(UPSTREAM / "videos"),
        "--seed",
        "1234",
        "--device",
        "cpu",
        "--report",
        str(report_path),
        "--video-names-file",
        str(names),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, cwd=UPSTREAM, capture_output=True, text=True)
    wall = time.perf_counter() - started
    report = report_path.read_text(encoding="utf-8")
    measured = _parse_report(report)
    if measured["archive_bytes"] != (artifact / "archive.zip").stat().st_size:
        raise MeasureError("upstream evaluator charged the wrong archive bytes")
    checkpoint["stages"]["evaluate"] = {
        "status": "complete",
        "command": command,
        "wall_seconds": wall,
        "stdout_tail": completed.stdout[-4000:],
        "report": str(report_path),
        "measured": measured,
    }
    if not keep_inflated:
        receipt = json.loads(inflate_receipt.read_text(encoding="utf-8"))
        cleanup = {
            "schema": "s4_rebuildable_bulk_cleanup.v1",
            "original_path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": receipt["stream_sha256"],
            "archive_sha256": checkpoint["archive_sha256"],
            "rebuild_command": checkpoint["stages"]["inflate"].get("command"),
            "reason": "deterministic evaluator scratch; report and decode receipt preserved",
            "rebuildable": True,
            "deleted_after_success": True,
        }
        _atomic_json(output / "cleanup_manifest.json", cleanup)
        raw_path.unlink()
        checkpoint["stages"]["cleanup"] = cleanup
    else:
        checkpoint["stages"]["cleanup"] = {"deleted_after_success": False, "operator_opt_in_keep": True}
    checkpoint["passed"] = True
    _atomic_json(checkpoint_path, checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--advisory-eval", action="store_true")
    parser.add_argument("--keep-inflated", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "s4_archive_composer_measurement.v1",
        "artifact": str(args.artifact),
        "research_only": True,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "parity": None if args.skip_parity else measure_parity(args.artifact, args.output),
        "advisory_eval": advisory_eval(args.artifact, args.output, keep_inflated=args.keep_inflated)
        if args.advisory_eval
        else None,
    }
    _atomic_json(args.output / "measurement_receipt.json", result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
