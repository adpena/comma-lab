#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure full-SE(3) advection and chart-coefficient residual pricing.

This is an advisory compress-time measurement.  It composes the deterministic
#549 target reconstruction and native CPU-Torch oracle from the predecessor,
but replaces planar xi and literal pixel exceptions with the complete stored
six-coordinate motion plus a strict scene-chart coefficient packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
from measure_predict_project_receiver import (
    ADVECTED_TARGET_MEMO,
    CAMERA_HW,
    PAIR_COUNT,
    POINTER,
    SCORER_HW,
    MeasurementError,
    _exact_solved_target_plane,
    _load_native_distortion_net,
    _metric_without_rate,
    _realize_advected_plane,
    _score_variants,
    _sha256_file,
    _stored_npy_memmap,
)

from tac.optimization.predict_project_receiver import (
    GLOBAL_WATERFILL_LAMBDA_STAR,
    ChartRGBCoefficientPacket,
    advect_motion_base,
    apply_chart_rgb_coefficients,
    counted_full_screw_xi_series,
    decode_chart_rgb_coefficients,
    encode_chart_rgb_coefficients,
    fit_chart_rgb_coefficients,
    predict_cell_field,
    projected_plane_array_sha256,
)
from tac.optimization.predict_project_schema import (
    canonical_json_bytes,
    parse_constraint_seed,
    serialize_constraint_seed,
)
from tac.optimization.predictor_upgrade_xi_chart import load_g1_worldsheet_motion
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.seed_compose_b2 import GT_CACHE_SHA256

SCHEMA: Final = "advected_screw6_chartlevel_measurement.v1"
STAGE_SCHEMA: Final = "advected_screw6_chartlevel_pair.v1"
PREFIXES: Final = (16, 64, 600)
FULL_SCREW_LAWREF_VALUES: Final = {
    "translation_scale": {
        "value": 0.16,
        "equation_id": "ego_motion_cumulative_se3_bspline_v1",
        "ladder": "MEASURED W7 pose-carry calibration",
        "anchor_path": ".omx/research/canonical_research_index_vehicle_warp_20260629.md",
        "anchor_locator": "W7",
    },
    "rotation_scale": {
        "value": 1.0,
        "equation_id": "ego_motion_cumulative_se3_bspline_v1",
        "ladder": "DERIVED identity conversion from stored rotation coordinates to tac.lie radians",
        "anchor_path": "src/tac/lie/_se3_numpy.py",
        "anchor_locator": "exp_so3/log_se3 convention",
    },
}
PLANAR_PIXEL_BASELINE_N64: Final = 19_739_340
PLANAR_RECEIPT_PATH: Final = ".omx/research/advected_motion_base_20260721.json"
AXIS: Final = "[macOS-CPU advisory]"
IMPLEMENTATION_SOURCE_PATHS: Final = (
    "tools/measure_advected_screw6_chartlevel.py",
    "tools/measure_predict_project_receiver.py",
    "src/tac/optimization/predict_project_receiver.py",
    "src/tac/optimization/predict_project_schema.py",
    "src/tac/optimization/predictor_upgrade_xi_chart.py",
    "src/tac/optimization/resize_full_kernel.py",
    "src/tac/boundary_math/warp_real_luma_frame0.py",
    "src/tac/lie/__init__.py",
    "src/tac/lie/_se3_numpy.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def _load_cache(path: Path) -> dict[str, np.memmap]:
    if _sha256_file(path) != GT_CACHE_SHA256:
        raise MeasurementError("full-screw GT-cache SHA-256 mismatch")
    fields = {key: _stored_npy_memmap(path, key) for key in ("n_pairs", "gt_f0", "gt_f1", "gt_poses")}
    if int(np.asarray(fields["n_pairs"]).reshape(())) != PAIR_COUNT:
        raise MeasurementError("full-screw measurement requires exact n600 cache")
    if fields["gt_f0"].shape != (PAIR_COUNT, *CAMERA_HW, 3) or fields["gt_f1"].shape != fields["gt_f0"].shape:
        raise MeasurementError("full-screw GT RGB geometry mismatch")
    if fields["gt_poses"].shape != (PAIR_COUNT, 6):
        raise MeasurementError("full-screw stored pose geometry mismatch")
    return fields


def _bucket_edges(xi: np.ndarray) -> list[float]:
    magnitudes = np.linalg.norm(np.asarray(xi, dtype=np.float64), axis=1)
    return [float(value) for value in np.quantile(magnitudes, (0.0, 0.25, 0.5, 0.75, 1.0))]


def _bucket(value: float, edges: Sequence[float]) -> str:
    for index in range(4):
        if value <= edges[index + 1] or index == 3:
            return f"q{index + 1}"
    raise AssertionError("unreachable magnitude bucket")


def _packet_record(coefficients: np.ndarray, scales: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    packet = ChartRGBCoefficientPacket(coefficients=coefficients, scales=scales)
    raw = encode_chart_rgb_coefficients(packet)
    if encode_chart_rgb_coefficients(decode_chart_rgb_coefficients(raw)) != raw:
        raise MeasurementError("chart coefficient raw parse/re-encode changed bytes")
    terminal = brotli.compress(raw, quality=11)
    restored = brotli.decompress(terminal)
    if restored != raw or encode_chart_rgb_coefficients(decode_chart_rgb_coefficients(restored)) != raw:
        raise MeasurementError("chart coefficient Brotli terminal parse-back changed bytes")
    return terminal, {
        "schema": "advected_screw6_chartlevel_packet.v1",
        "raw_packet_bytes": len(raw),
        "raw_packet_sha256": _sha256(raw),
        "terminal_coder": "brotli-11(strict_chart_rgb_coefficient_packet)",
        "terminal_bytes": len(terminal),
        "terminal_sha256": _sha256(terminal),
        "pair_count": packet.pair_count,
        "coefficient_shape": list(packet.coefficients.shape),
        "receiver_basis": "decoded_scene_chart_class_indicator",
        "target_exact": False,
        "receiver_closed": True,
        "parseback_byte_identical": True,
    }


def _load_stages(stage_dir: Path, config_sha256: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not stage_dir.exists():
        return rows
    for path in sorted(stage_dir.glob("pair_*.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MeasurementError(f"corrupt full-screw stage: {path}") from exc
        pair = row.get("pair_index")
        if (
            row.get("schema") != STAGE_SCHEMA
            or row.get("config_sha256") != config_sha256
            or isinstance(pair, bool)
            or not isinstance(pair, int)
            or pair in rows
        ):
            raise MeasurementError(f"full-screw stage custody mismatch: {path}")
        rows[pair] = row
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]], edges: Sequence[float]) -> dict[str, Any]:
    variants = ("static", "full_screw", "static_chart", "full_screw_chart", "solved_target")

    def one(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"pair_count": 0}
        count = len(selected)
        result: dict[str, Any] = {
            "pair_count": count,
            "xi_l2_min": min(float(row["xi_l2"]) for row in selected),
            "xi_l2_max": max(float(row["xi_l2"]) for row in selected),
        }
        for variant in variants:
            result[variant] = {
                "d_seg_mean": sum(float(row["hard_oracle"][variant]["d_seg"]) for row in selected) / count,
                "d_pose_mean": sum(float(row["hard_oracle"][variant]["d_pose"]) for row in selected) / count,
            }
        return result

    return {
        "aggregate": one(rows),
        "by_xi_magnitude_bucket": [
            {
                "bucket": f"q{index + 1}",
                "interval": [float(edges[index]), float(edges[index + 1])],
                **one([row for row in rows if row["xi_bucket"] == f"q{index + 1}"]),
            }
            for index in range(4)
        ],
    }


def _coefficient_streams(rows: Sequence[Mapping[str, Any]], edges: Sequence[float]) -> dict[str, Any]:
    def packet(selected: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
        if not selected:
            return {
                "schema": "advected_screw6_chartlevel_packet.v1",
                "pair_count": 0,
                "raw_packet_bytes": 0,
                "terminal_bytes": 0,
                "status": "EMPTY_PREFIX_BUCKET",
                "receiver_closed": True,
                "parseback_byte_identical": True,
            }
        coefficients = np.asarray([row[arm]["coefficients"] for row in selected], dtype=np.int8)
        scales = np.asarray([row[arm]["scale"] for row in selected], dtype="<f2")
        _, record = _packet_record(coefficients, scales)
        return record

    aggregate = {arm: packet(rows, arm) for arm in ("static_chart", "full_screw_chart")}
    aggregate["full_screw_terminal_byte_delta_vs_static"] = (
        aggregate["full_screw_chart"]["terminal_bytes"] - aggregate["static_chart"]["terminal_bytes"]
    )
    by_bucket = []
    for index in range(4):
        bucket_id = f"q{index + 1}"
        selected = [row for row in rows if row["xi_bucket"] == bucket_id]
        static = packet(selected, "static_chart")
        screw = packet(selected, "full_screw_chart")
        by_bucket.append(
            {
                "bucket": bucket_id,
                "interval": [float(edges[index]), float(edges[index + 1])],
                "static_chart": static,
                "full_screw_chart": screw,
                "full_screw_terminal_byte_delta_vs_static": screw["terminal_bytes"] - static["terminal_bytes"],
            }
        )
    return {"aggregate": aggregate, "by_xi_magnitude_bucket": by_bucket}


def run(
    *,
    seed_path: Path,
    gt_cache_path: Path,
    upstream: Path,
    output_root: Path,
    pair_end: int,
    chunk_size: int,
    threads: int,
) -> dict[str, Any]:
    if pair_end not in PREFIXES or chunk_size <= 0 or threads <= 0:
        raise MeasurementError("full-screw prefix must be n16/n64/n600 with positive chunk/threads")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < 1 << 30:
        raise MeasurementError("full-screw storage preflight requires at least 1 GiB free")
    repository_root = Path(__file__).resolve().parents[1]
    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    if serialize_constraint_seed(seed) != seed_bytes or seed["receiver"]["seed"] != 1234:
        raise MeasurementError("full-screw measurement requires canonical seed 1234")
    cache = _load_cache(gt_cache_path)
    _, _, pitch_rad, g1_custody = load_g1_worldsheet_motion(repository_root)
    translation_scale = float(FULL_SCREW_LAWREF_VALUES["translation_scale"]["value"])
    rotation_scale = float(FULL_SCREW_LAWREF_VALUES["rotation_scale"]["value"])
    full_xi, xi_custody = counted_full_screw_xi_series(
        np.asarray(cache["gt_poses"], dtype=np.float64),
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
        pitch_rad=pitch_rad,
        source_sha256=GT_CACHE_SHA256,
    )
    edges = _bucket_edges(full_xi)
    target_memo = repository_root / ADVECTED_TARGET_MEMO
    planar_receipt = repository_root / PLANAR_RECEIPT_PATH
    if not target_memo.is_file() or not planar_receipt.is_file():
        raise MeasurementError("full-screw predecessor/target custody is missing")
    predecessor = json.loads(planar_receipt.read_text())
    observed_planar = int(predecessor["aggregate"]["advected"]["exception_bytes_total"])
    if observed_planar != PLANAR_PIXEL_BASELINE_N64:
        raise MeasurementError("planar pixel baseline drifted from delegated authority")
    implementation = {
        relative_path: _sha256_file(repository_root / relative_path) for relative_path in IMPLEMENTATION_SOURCE_PATHS
    }
    lawrefs = {
        name: {
            **binding,
            "anchor_sha256": _sha256_file(repository_root / str(binding["anchor_path"])),
        }
        for name, binding in FULL_SCREW_LAWREF_VALUES.items()
    }
    lawrefs.update(
        {
            "ground_pitch": g1_custody["pitch_custody"],
            "rate_exchange": {
                "value": GLOBAL_WATERFILL_LAMBDA_STAR,
                "equation_id": "realization_breakeven_bytes_v1",
            },
        }
    )
    config = {
        "schema": SCHEMA,
        "seed": {"path": str(seed_path), "sha256": _sha256(seed_bytes), "bytes": len(seed_bytes)},
        "gt_cache": {"path": str(gt_cache_path), "sha256": GT_CACHE_SHA256},
        "target": {
            "definition": "#549/C1 exact rational source scorer planes plus canonical factor2 realization",
            "memo_path": ADVECTED_TARGET_MEMO,
            "memo_sha256": _sha256_file(target_memo),
            "old_archive_bytes_consumed": False,
        },
        "xi_custody": xi_custody,
        "lawrefs": lawrefs,
        "g1_geometry_custody": g1_custody,
        "xi_bucket_edges_full_n600": edges,
        "chunk_size": chunk_size,
        "threads": threads,
        "seed_value": 1234,
        "implementation_sources": implementation,
        "planar_pixel_baseline": {
            "n64_bytes": PLANAR_PIXEL_BASELINE_N64,
            "receipt_path": PLANAR_RECEIPT_PATH,
            "receipt_sha256": _sha256_file(planar_receipt),
        },
    }
    planar_xi_l2_max = float(predecessor["aggregate"]["xi_l2_max"])
    config["planar_pixel_baseline"]["xi_l2_max"] = planar_xi_l2_max
    config_sha256 = _sha256(canonical_json_bytes(config))
    lane_root = output_root / "advected_screw6_chartlevel"
    prior_receipt_path = lane_root / "receipt.json"
    if pair_end in (64, 600):
        if not prior_receipt_path.is_file():
            raise MeasurementError("larger prefix is blocked until the previous governed receipt exists")
        prior = json.loads(prior_receipt_path.read_text())
        expected_prefix = 16 if pair_end == 64 else 64
        expected_gate = "PASS_N64_AUTHORIZED" if pair_end == 64 else "PASS_N600_AUTHORIZED"
        if prior.get("prefix") != expected_prefix or prior.get("next_gate") != expected_gate:
            raise MeasurementError("larger prefix is blocked because the prior two-axis gate did not pass")
    net, torch, scorer_custody = _load_native_distortion_net(upstream, threads)
    kernel = FullResizeKernel.build()
    from tac.boundary_math import warp_real_luma_frame0 as g1_warp

    geom = g1_warp.GroundHomographyGeom.eon(native_hw=SCORER_HW, pitch=pitch_rad)
    stage_dir = lane_root / "stages"
    rows = _load_stages(stage_dir, config_sha256)
    resumed = len([pair for pair in rows if pair < pair_end])
    for chunk_begin in range(0, pair_end, chunk_size):
        chunk_end = min(pair_end, chunk_begin + chunk_size)
        for pair in range(chunk_begin, chunk_end):
            if pair in rows:
                continue
            started = time.perf_counter()
            source0 = np.asarray(cache["gt_f0"][pair], dtype=np.uint8).copy()
            source1 = np.asarray(cache["gt_f1"][pair], dtype=np.uint8).copy()
            target0 = _exact_solved_target_plane(kernel.operator, source0)
            target1 = _exact_solved_target_plane(kernel.operator, source1)
            chart0 = predict_cell_field(seed, pair)
            screw = advect_motion_base(target0, chart0, full_xi[pair], geom)
            static_coeff, static_scale = fit_chart_rgb_coefficients(target0, target1, chart0)
            screw_coeff, screw_scale = fit_chart_rgb_coefficients(screw["frame1_base"], target1, screw["frame1_cells"])
            static_packet = ChartRGBCoefficientPacket(
                coefficients=static_coeff[None], scales=np.asarray([static_scale], dtype="<f2")
            )
            screw_packet = ChartRGBCoefficientPacket(
                coefficients=screw_coeff[None], scales=np.asarray([screw_scale], dtype="<f2")
            )
            static_chart_plane = apply_chart_rgb_coefficients(target0, chart0, static_packet, 0)
            screw_chart_plane = apply_chart_rgb_coefficients(
                screw["frame1_base"], screw["frame1_cells"], screw_packet, 0
            )
            frame0 = _realize_advected_plane(
                target0,
                chart0,
                seed_sha256=config["seed"]["sha256"],
                generator_id="task549_solved_target_frame0_reconstruction",
                additional_seed_bytes=int(target0.nbytes),
                kernel=kernel,
            )
            variants = {
                "static": _realize_advected_plane(
                    target0,
                    chart0,
                    seed_sha256=config["seed"]["sha256"],
                    generator_id="static_frame1_base",
                    additional_seed_bytes=0,
                    kernel=kernel,
                ),
                "full_screw": _realize_advected_plane(
                    screw["frame1_base"],
                    screw["frame1_cells"],
                    seed_sha256=config["seed"]["sha256"],
                    generator_id="counted_full_screw_advected_motion_base",
                    additional_seed_bytes=0,
                    kernel=kernel,
                ),
                "static_chart": _realize_advected_plane(
                    static_chart_plane,
                    chart0,
                    seed_sha256=config["seed"]["sha256"],
                    generator_id="static_scene_chart_rgb_coefficients",
                    additional_seed_bytes=int(static_coeff.nbytes + np.dtype("<f2").itemsize),
                    kernel=kernel,
                ),
                "full_screw_chart": _realize_advected_plane(
                    screw_chart_plane,
                    screw["frame1_cells"],
                    seed_sha256=config["seed"]["sha256"],
                    generator_id="full_screw_scene_chart_rgb_coefficients",
                    additional_seed_bytes=int(screw_coeff.nbytes + np.dtype("<f2").itemsize),
                    kernel=kernel,
                ),
                "solved_target": _realize_advected_plane(
                    target1,
                    screw["frame1_cells"],
                    seed_sha256=config["seed"]["sha256"],
                    generator_id="task549_solved_target_frame1_reconstruction",
                    additional_seed_bytes=int(target1.nbytes),
                    kernel=kernel,
                ),
            }
            hard = _score_variants(net, torch, source0, source1, frame0, variants)
            xi_l2 = float(np.linalg.norm(full_xi[pair]))
            row = {
                "schema": STAGE_SCHEMA,
                "config_sha256": config_sha256,
                "pair_index": pair,
                "xi": full_xi[pair].tolist(),
                "xi_l2": xi_l2,
                "xi_bucket": _bucket(xi_l2, edges),
                "hard_oracle": hard,
                "static_chart": {
                    "coefficients": static_coeff.astype(int).tolist(),
                    "scale": static_scale,
                    "plane_sha256": projected_plane_array_sha256(static_chart_plane),
                },
                "full_screw_chart": {
                    "coefficients": screw_coeff.astype(int).tolist(),
                    "scale": screw_scale,
                    "plane_sha256": projected_plane_array_sha256(screw_chart_plane),
                },
                "transport": {
                    "ground_pixels": screw["ground_pixels"],
                    "offground_pixels": screw["offground_pixels"],
                    "base_sha256": screw["frame1_base_sha256"],
                    "cells_sha256": screw["frame1_cells_sha256"],
                    "additional_video_derived_motion_bytes": 0,
                },
                "timing_seconds": time.perf_counter() - started,
                "authority": f"MEASURED {AXIS}",
                "score_claim": False,
                "promotion_eligible": False,
            }
            _atomic_json(stage_dir / f"pair_{pair:04d}.json", row)
            rows[pair] = row
        _atomic_json(
            lane_root / "checkpoints" / f"chunk_{chunk_begin:04d}_{chunk_end:04d}.json",
            {
                "schema": "advected_screw6_chartlevel_checkpoint.v1",
                "config_sha256": config_sha256,
                "completed_through_exclusive": chunk_end,
                "completed_pairs": len([pair for pair in rows if pair < pair_end]),
                "all_pair_stages_preserved": True,
            },
        )
    ordered = [rows[pair] for pair in range(pair_end)]
    distortion = _aggregate(ordered, edges)
    streams = _coefficient_streams(ordered, edges)
    aggregate = distortion["aggregate"]
    static_bytes = int(streams["aggregate"]["static_chart"]["terminal_bytes"])
    screw_bytes = int(streams["aggregate"]["full_screw_chart"]["terminal_bytes"])
    dpose_gate = aggregate["full_screw"]["d_pose_mean"] < aggregate["static"]["d_pose_mean"]
    rate_gate = screw_bytes <= static_bytes
    gate_pass = dpose_gate and rate_gate
    next_gate = (
        "PASS_N64_AUTHORIZED"
        if pair_end == 16 and gate_pass
        else "PASS_N600_AUTHORIZED"
        if pair_end == 64 and gate_pass
        else "STOP_TWO_AXIS_GATE_FAILED"
        if not gate_pass
        else "N600_COMPLETE"
    )
    points = {}
    for arm, byte_count in (("static_chart", static_bytes), ("full_screw_chart", screw_bytes)):
        metric = _metric_without_rate(
            {
                "d_seg": aggregate[arm]["d_seg_mean"],
                "d_pose": aggregate[arm]["d_pose_mean"],
            }
        )
        points[arm] = {
            "bytes": byte_count,
            "d_seg": aggregate[arm]["d_seg_mean"],
            "d_pose": aggregate[arm]["d_pose_mean"],
            "advisory_action_with_lambda": metric + GLOBAL_WATERFILL_LAMBDA_STAR * byte_count,
        }
    magnitudes = np.linalg.norm(full_xi, axis=1)
    receipt = {
        "schema": SCHEMA,
        "prefix": pair_end,
        "config": config,
        "config_sha256": config_sha256,
        "scorer_custody": scorer_custody,
        "resumed_pair_stages": resumed,
        "D1_full_screw": {
            "status": "BUILT_AND_EXECUTED",
            "xi_custody": xi_custody,
            "xi_l2_full_n600_quantiles": {
                str(quantile): float(np.quantile(magnitudes, quantile)) for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
            "planar_predecessor_xi_l2_max": planar_xi_l2_max,
            "ground_homography": True,
            "scene_chart_full_screw": True,
            "additional_video_derived_motion_bytes": 0,
            "decoder_pose_blind": True,
        },
        "D2_distortion": distortion,
        "D3_chart_coefficient_rate": {
            **streams,
            "lambda_star_s_per_byte": GLOBAL_WATERFILL_LAMBDA_STAR,
            "composed_points": points,
            "planar_pixel_baseline_n64_bytes": PLANAR_PIXEL_BASELINE_N64,
            "superseded_surface": "literal exact RGB pixel exception",
            "current_surface": "receiver-closed lossy decoded-scene-chart RGB coefficient packet",
        },
        "D4_route": {
            "einstein_kolmogorov_ultra": "U1_R_D_frontier",
            "p0_register": "G-pose #603",
            "route_status": "ROUTE_COEFFICIENT_IF_TWO_AXIS_GATE_PASS_ELSE_REFRAME_COEFFICIENT_BASIS",
            "pointer": POINTER,
            "pointer_moved": False,
            "main_landing_review_required": True,
        },
        "gate": {"dpose_full_screw_lt_static": dpose_gate, "chart_bytes_full_screw_le_static": rate_gate},
        "next_gate": next_gate,
        "authority": {
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
        },
        "verdict_scope": (
            "stored-PoseNet full-screw calibration plus five-class RGB-offset chart coefficient family, "
            f"prefix n{pair_end}, this clip, macOS CPU advisory only"
        ),
        "storage": {
            "root": str(output_root),
            "automatic_disk_hygiene": "ZIP_STORED GT memmaps; one-pair tensors only; immutable JSON stages/checkpoints",
            "old_archive_bytes_consumed": False,
        },
    }
    _atomic_json(prior_receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/advected_screw6_20260721"),
    )
    parser.add_argument("--pair-end", type=int, choices=PREFIXES, default=16)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run(
        seed_path=args.seed,
        gt_cache_path=args.gt_cache,
        upstream=args.upstream,
        output_root=args.output_dir,
        pair_end=args.pair_end,
        chunk_size=args.chunk_size,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "receipt": str(args.output_dir / "advected_screw6_chartlevel" / "receipt.json"),
                "prefix": receipt["prefix"],
                "next_gate": receipt["next_gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
