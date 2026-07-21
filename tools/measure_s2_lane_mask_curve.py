#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the real-n600 coherent Lane chart and genuine-frame residual inverse.

This is a representation-fidelity measurement, not a score.  It measures the
actual finite LBND2 stream after decode and reports pixel confusion, macro-F1,
and boundary/interior strata.  The residual stage fits finite genuine literal
polar curvelet and compact shearlet atoms, codes the selected qints with the
#557 repository context-arithmetic stack, parses the exact sidecar back, and
compares generic image coordinates with an arc-length/dash-phase treatment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    deserialize_lane_band_rd,
    rasterize_lane_coverage_range_dependent,
    roundtrip_lines_through_rd_tracked,
)
from tac.canonical_equations.day_consolidation_laws_20260720 import (  # noqa: E402
    breakeven_bytes,
)
from tac.optimization.boundary_inverse_custody import (  # noqa: E402
    RATE_PRICE_S_PER_BYTE,
    SparseInverseProgram,
    apply_sparse_program,
    authority_labels,
    chart_correction_domain,
    decode_program,
    deterministic_training_indices,
    dictionary_metadata,
    encode_program,
    fit_sparse_program_sweep,
    flip_accounting,
)
from tac.optimization.s2_partition_seed import detect_partition_semantics  # noqa: E402

SCHEMA: Final = "s2_lane_true_mask_curve.v2"
GT_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
PARENT_RECEIPT_SHA256: Final = "273d7ef28b9312973831403c57552274a8fef57f53a1eb4517c1e3551d76ef94"
BASELINE_CHART_BYTES: Final = 41_303
RATE_PRICE_PER_BYTE: Final = RATE_PRICE_S_PER_BYTE


def _validate_args(args: argparse.Namespace) -> None:
    prefixes = tuple(int(value) for value in args.atom_prefixes)
    if not prefixes or any(value < 2 for value in prefixes):
        raise ValueError("atom prefixes must all be at least two")
    if args.phase_bins <= 0:
        raise ValueError("phase bins must be positive")
    if not math.isfinite(args.qstep) or args.qstep <= 0.0:
        raise ValueError("qstep must be finite and positive")
    if not math.isfinite(args.correction_threshold) or args.correction_threshold <= 0.0:
        raise ValueError("correction threshold must be finite and positive")
    if args.max_per_residual_sign < 0 or args.max_zero_samples < 0:
        raise ValueError("training sample caps must be non-negative")
    if args.chunk_pairs <= 0:
        raise ValueError("chunk pairs must be positive")
    if args.minimum_stage_free_bytes < 0:
        raise ValueError("minimum stage free bytes must be non-negative")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def npz_member_memmap(path: Path, key: str) -> np.memmap:
    member_name = f"{key}.npy"
    with zipfile.ZipFile(path, "r") as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
            raise ValueError(f"{member_name} must be ZIP_STORED")
        with path.open("rb") as handle:
            handle.seek(info.header_offset)
            local = handle.read(30)
            if len(local) != 30:
                raise ValueError(f"truncated ZIP local header for {member_name}")
            fields = struct.unpack("<IHHHHHIIIHH", local)
            if fields[0] != 0x04034B50:
                raise ValueError(f"invalid ZIP local header for {member_name}")
            name_length, extra_length = fields[-2:]
            handle.seek(info.header_offset + 30 + name_length + extra_length)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
            offset = handle.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def _boundary(mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    result[1:] |= mask[1:] != mask[:-1]
    result[:-1] |= mask[:-1] != mask[1:]
    result[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    result[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return result


def _confusion(predicted: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    tn = int(predicted.size - tp - fp - fn)
    return tp, fp, fn, tn


def _metrics(counts: tuple[int, int, int, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _stratum_rate(predicted: np.ndarray, truth: np.ndarray, selector: np.ndarray) -> dict[str, Any]:
    selected = int(np.count_nonzero(selector))
    positives = int(np.count_nonzero(truth & selector))
    predicted_positive = int(np.count_nonzero(predicted & selector))
    true_positive = int(np.count_nonzero(predicted & truth & selector))
    return {
        "pixels": selected,
        "truth_positive": positives,
        "predicted_positive": predicted_positive,
        "true_positive": true_positive,
        "recall_if_positive": true_positive / positives if positives else None,
        "false_positive_rate_if_negative": (
            (predicted_positive - true_positive) / (selected - positives) if selected > positives else None
        ),
    }


def _render_mask(lines: list[Any], label_shape: tuple[int, int], cfg: LaneBandRenderConfig) -> np.ndarray:
    return (
        rasterize_lane_coverage_range_dependent(
            lines,
            h=label_shape[0],
            w=label_shape[1],
            softness=cfg.softness,
            dash_gate=cfg.dash_gate,
            dash_forward_max_m=cfg.dash_forward_max_m,
            v_h=cfg.v_h,
            cx=cfg.cx,
        )
        >= 0.5
    )


def _selectors(label: np.ndarray, truth: np.ndarray, road_class: int) -> dict[str, np.ndarray]:
    boundary = _boundary(truth)
    rows = np.indices(label.shape)[0]
    return {
        "lane_boundary": truth & boundary,
        "lane_interior": truth & ~boundary,
        "road_negative": label == road_class,
        "other_negative": (label != road_class) & ~truth,
        "upper_half": rows < label.shape[0] // 2,
        "lower_half": rows >= label.shape[0] // 2,
    }


def _empty_strata() -> dict[str, dict[str, int]]:
    return {
        key: {"pixels": 0, "truth_positive": 0, "predicted_positive": 0, "true_positive": 0}
        for key in (
            "lane_boundary",
            "lane_interior",
            "road_negative",
            "other_negative",
            "upper_half",
            "lower_half",
        )
    }


def _finalize_strata(strata: dict[str, dict[str, int]]) -> dict[str, Any]:
    finalized: dict[str, Any] = {}
    for key, row in strata.items():
        positives = row["truth_positive"]
        negatives = row["pixels"] - positives
        finalized[key] = {
            **row,
            "recall_if_positive": row["true_positive"] / positives if positives else None,
            "false_positive_rate_if_negative": (
                (row["predicted_positive"] - row["true_positive"]) / negatives if negatives else None
            ),
        }
    return finalized


def _collect_training_samples(
    *,
    labels: np.ndarray,
    decoded_lines: list[list[Any]],
    cfg: LaneBandRenderConfig,
    phase_bin_count: int,
    max_per_residual_sign: int,
    max_zero: int,
) -> dict[str, dict[str, np.ndarray]]:
    chunks: dict[str, dict[str, list[np.ndarray]]] = {
        mode: {"coords": [], "phase_bins": [], "residual": []} for mode in ("generic_2d", "dash_arc_phase")
    }
    for pair_index, lines in enumerate(decoded_lines):
        label = np.asarray(labels[pair_index])
        truth = label == cfg.lane_cls
        baseline = _render_mask(lines, label.shape, cfg)
        for mode, bins in (("generic_2d", 1), ("dash_arc_phase", phase_bin_count)):
            domain = chart_correction_domain(
                lines,
                height=label.shape[0],
                width=label.shape[1],
                coordinate_mode=mode,
                phase_bin_count=bins,
            )
            if domain.flat_indices.size == 0:
                continue
            truth_domain = truth.reshape(-1)[domain.flat_indices]
            baseline_domain = baseline.reshape(-1)[domain.flat_indices]
            residual = truth_domain.astype(np.int8) - baseline_domain.astype(np.int8)
            selected = deterministic_training_indices(
                residual,
                max_per_residual_sign=max_per_residual_sign,
                max_zero=max_zero,
            )
            chunks[mode]["coords"].append(domain.coords[selected])
            chunks[mode]["phase_bins"].append(domain.phase_bins[selected])
            chunks[mode]["residual"].append(residual[selected])
    result: dict[str, dict[str, np.ndarray]] = {}
    for mode, fields in chunks.items():
        result[mode] = {
            "coords": (
                np.concatenate(fields["coords"]).astype(np.float32, copy=False)
                if fields["coords"]
                else np.zeros((0, 2), dtype=np.float32)
            ),
            "phase_bins": (
                np.concatenate(fields["phase_bins"]).astype(np.int16, copy=False)
                if fields["phase_bins"]
                else np.zeros(0, dtype=np.int16)
            ),
            "residual": (
                np.concatenate(fields["residual"]).astype(np.int8, copy=False)
                if fields["residual"]
                else np.zeros(0, dtype=np.int8)
            ),
        }
    return result


def _measure_inverse_row(
    *,
    variant: str,
    labels: np.ndarray,
    decoded_lines: list[list[Any]],
    program: SparseInverseProgram,
    sidecar: bytes,
    sidecar_path: Path,
    cfg: LaneBandRenderConfig,
    road_class: int,
    cache_sha256: str,
    renderer_source_sha256: str,
    state_path: Path,
    resume: bool,
    chunk_pairs: int,
) -> dict[str, Any]:
    sidecar_sha = hashlib.sha256(sidecar).hexdigest()
    state: dict[str, Any]
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        if (
            state.get("variant") != variant
            or state.get("sidecar_sha256") != sidecar_sha
            or state.get("cache_sha256") != cache_sha256
            or state.get("renderer_source_sha256") != renderer_source_sha256
        ):
            raise ValueError(f"resume-state custody mismatch for {variant}")
        next_pair = state.get("next_pair")
        if (
            isinstance(next_pair, bool)
            or not isinstance(next_pair, int)
            or not 0 <= next_pair <= len(labels)
            or len(state.get("pair_f1", ())) != next_pair
            or len(state.get("total", ())) != 4
            or set(state.get("strata", {})) != set(_empty_strata())
            or set(state.get("flip", {})) != {"all", *_empty_strata()}
            or set(state.get("render", {})) != {"support_pixels", "added_pixels", "removed_pixels"}
        ):
            raise ValueError(f"resume-state shape/progress mismatch for {variant}")
    else:
        state = {
            "variant": variant,
            "sidecar_sha256": sidecar_sha,
            "cache_sha256": cache_sha256,
            "renderer_source_sha256": renderer_source_sha256,
            "next_pair": 0,
            "total": [0, 0, 0, 0],
            "pair_f1": [],
            "strata": _empty_strata(),
            "flip": {
                key: {
                    "pixels": 0,
                    "changed": 0,
                    "beneficial": 0,
                    "harmful": 0,
                    "remaining_false_negative": 0,
                    "remaining_false_positive": 0,
                }
                for key in ("all", *_empty_strata().keys())
            },
            "render": {"support_pixels": 0, "added_pixels": 0, "removed_pixels": 0},
        }
    total = np.asarray(state["total"], dtype=np.int64)
    start = int(state["next_pair"])
    for pair_index in range(start, len(labels)):
        label = np.asarray(labels[pair_index])
        truth = label == cfg.lane_cls
        baseline = _render_mask(decoded_lines[pair_index], label.shape, cfg)
        corrected, render = apply_sparse_program(baseline, decoded_lines[pair_index], program)
        counts = _confusion(corrected, truth)
        total += np.asarray(counts, dtype=np.int64)
        state["pair_f1"].append(float(_metrics(counts)["f1"]))
        selectors = _selectors(label, truth, road_class)
        for key, selector in selectors.items():
            values = _stratum_rate(corrected, truth, selector)
            for field in state["strata"][key]:
                state["strata"][key][field] += int(values[field])
            flips = flip_accounting(baseline, corrected, truth, selector)
            for field, value in flips.items():
                state["flip"][key][field] += int(value)
        flips = flip_accounting(baseline, corrected, truth)
        for field, value in flips.items():
            state["flip"]["all"][field] += int(value)
        for field, value in render.items():
            state["render"][field] += int(value)
        state["next_pair"] = pair_index + 1
        state["total"] = [int(value) for value in total]
        if state["next_pair"] % chunk_pairs == 0 or state["next_pair"] == len(labels):
            atomic_json(state_path, state)
    aggregate = _metrics(tuple(int(value) for value in total))
    aggregate["macro_pair_f1_mean"] = float(np.mean(state["pair_f1"]))
    aggregate["macro_pair_f1_p10"] = float(np.quantile(state["pair_f1"], 0.1))
    aggregate["macro_pair_f1_p90"] = float(np.quantile(state["pair_f1"], 0.9))
    required_recovery_s = len(sidecar) * RATE_PRICE_PER_BYTE
    callable_roundtrip = breakeven_bytes(required_recovery_s)
    if not math.isclose(callable_roundtrip, len(sidecar), rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError("realization_breakeven_bytes_v1 roundtrip drift")
    try:
        receipt_sidecar_path = str(sidecar_path.resolve().relative_to(REPO))
    except ValueError:
        receipt_sidecar_path = str(sidecar_path.resolve())
    return {
        "variant": variant,
        "finite_sidecar_bytes": len(sidecar),
        "sidecar_sha256": sidecar_sha,
        "sidecar_path": receipt_sidecar_path,
        "context_arithmetic_parseback_exact": np.array_equal(decode_program(sidecar).qcoeff, program.qcoeff),
        "atom_count": program.atom_count,
        "selected_curvelet_count": program.selected_curvelet_count,
        "selected_shearlet_count": program.selected_shearlet_count,
        "coordinate_mode": program.coordinate_mode,
        "phase_bin_count": program.phase_bin_count,
        "quantization_step": program.qstep,
        "correction_threshold": program.threshold,
        "composed_chart_bytes": BASELINE_CHART_BYTES,
        "composed_total_bytes": BASELINE_CHART_BYTES + len(sidecar),
        "incremental_rate_term": required_recovery_s,
        "mask_fidelity": aggregate,
        "per_stratum": _finalize_strata(state["strata"]),
        "eat_the_flip_remainder": state["flip"],
        "render_totals": state["render"],
        "waterfill": {
            "equation_id": "realization_breakeven_bytes_v1",
            "realized_through_r_recovery_s": None,
            "required_realized_recovery_s": required_recovery_s,
            "callable_roundtrip_bytes": callable_roundtrip,
            "status": "FORMALIZATION_PENDING",
            "reason": "mask F1 is not through-R d_seg and cannot supply the law input",
        },
    }


def _measure_row(
    *,
    name: str,
    labels: np.ndarray,
    decoded_lines: list[list[Any]],
    packet: bytes,
    cfg: LaneBandRenderConfig,
    road_class: int,
) -> dict[str, Any]:
    if len(decoded_lines) != len(labels):
        raise ValueError("decoded Lane chart does not cover the complete label cache")
    total = np.zeros(4, dtype=np.int64)
    pair_f1: list[float] = []
    strata_names = (
        "lane_boundary",
        "lane_interior",
        "road_negative",
        "other_negative",
        "upper_half",
        "lower_half",
    )
    strata: dict[str, dict[str, int]] = {
        key: {"pixels": 0, "truth_positive": 0, "predicted_positive": 0, "true_positive": 0} for key in strata_names
    }
    for pair_index, lines in enumerate(decoded_lines):
        label = np.asarray(labels[pair_index])
        truth = label == cfg.lane_cls
        predicted = (
            rasterize_lane_coverage_range_dependent(
                lines,
                h=label.shape[0],
                w=label.shape[1],
                softness=cfg.softness,
                dash_gate=cfg.dash_gate,
                dash_forward_max_m=cfg.dash_forward_max_m,
                v_h=cfg.v_h,
                cx=cfg.cx,
            )
            >= 0.5
        )
        counts = _confusion(predicted, truth)
        total += np.asarray(counts, dtype=np.int64)
        pair_f1.append(float(_metrics(counts)["f1"]))
        boundary = _boundary(truth)
        selectors = {
            "lane_boundary": truth & boundary,
            "lane_interior": truth & ~boundary,
            "road_negative": label == road_class,
            "other_negative": (label != road_class) & ~truth,
            "upper_half": np.indices(label.shape)[0] < label.shape[0] // 2,
            "lower_half": np.indices(label.shape)[0] >= label.shape[0] // 2,
        }
        for key, selector in selectors.items():
            values = _stratum_rate(predicted, truth, selector)
            for field in strata[key]:
                strata[key][field] += int(values[field])
    aggregate = _metrics(tuple(int(value) for value in total))
    aggregate["macro_pair_f1_mean"] = float(np.mean(pair_f1))
    aggregate["macro_pair_f1_p10"] = float(np.quantile(pair_f1, 0.1))
    aggregate["macro_pair_f1_p90"] = float(np.quantile(pair_f1, 0.9))
    finalized_strata: dict[str, Any] = {}
    for key, row in strata.items():
        positives = row["truth_positive"]
        negatives = row["pixels"] - positives
        finalized_strata[key] = {
            **row,
            "recall_if_positive": row["true_positive"] / positives if positives else None,
            "false_positive_rate_if_negative": (
                (row["predicted_positive"] - row["true_positive"]) / negatives if negatives else None
            ),
        }
    counted_bytes = len(brotli.compress(packet, quality=11))
    return {
        "variant": name,
        "finite_brotli_bytes": counted_bytes,
        "rate_term": counted_bytes * RATE_PRICE_PER_BYTE,
        "decoder_dash_gate": cfg.dash_gate,
        "mask_fidelity": aggregate,
        "per_stratum": finalized_strata,
    }


def measure(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    _validate_args(args)
    cache_sha = sha256_file(args.gt_cache)
    if cache_sha != args.expected_gt_cache_sha256:
        raise ValueError(f"GT cache SHA drift: {cache_sha}")
    labels = npz_member_memmap(args.gt_cache, "lstars")
    if labels.shape != (600, 384, 512):
        raise ValueError(f"unexpected n600 label geometry: {labels.shape}")

    parent_sha = sha256_file(args.parent_receipt)
    if parent_sha != args.expected_parent_receipt_sha256:
        raise ValueError(f"parent Lane receipt SHA drift: {parent_sha}")
    parent_receipt = json.loads(args.parent_receipt.read_text())
    parent_dash = next(row for row in parent_receipt["curve"] if row["variant"] == "coherent_slot_none_dash")
    if int(parent_dash["finite_brotli_bytes"]) != BASELINE_CHART_BYTES:
        raise ValueError("settled parent Lane chart byte count drift")

    args.stage_dir.mkdir(parents=True, exist_ok=True)
    storage = shutil.disk_usage(args.stage_dir)
    if storage.free < args.minimum_stage_free_bytes:
        raise OSError(f"stage storage preflight refused: {storage.free} < {args.minimum_stage_free_bytes}")

    semantics = detect_partition_semantics(labels)
    road_class, lane_class = semantics.semantic_class_ids[:2]
    fit_cfg = LaneBandRenderConfig(dash_gate=True, lane_cls=lane_class)
    continuous_cfg = LaneBandRenderConfig(dash_gate=False, lane_cls=lane_class)
    chart_manifest_path = args.stage_dir / "chart_manifest.json"
    if args.resume and chart_manifest_path.exists():
        chart_manifest = json.loads(chart_manifest_path.read_text())
        if (
            chart_manifest.get("schema") != "boundary_inverse_chart_stage.v1"
            or chart_manifest.get("cache_sha256") != cache_sha
            or chart_manifest.get("parent_receipt_sha256") != parent_sha
            or set(chart_manifest.get("variants", {}))
            != {"coherent_slot_none_dash", "coherent_slot_none_continuous"}
        ):
            raise ValueError("chart resume manifest custody mismatch")
        packets: dict[str, bytes] = {}
        decoded_by_name: dict[str, list[list[Any]]] = {}
        for name, row in chart_manifest["variants"].items():
            if row.get("packet_file") != f"{name}.lbnd2":
                raise ValueError(f"chart packet path drift for {name}")
            path = args.stage_dir / row["packet_file"]
            packet = path.read_bytes()
            if (
                hashlib.sha256(packet).hexdigest() != row["packet_sha256"]
                or len(packet) != int(row["packet_bytes"])
                or len(brotli.compress(packet, quality=11)) != int(row["brotli_bytes"])
            ):
                raise ValueError(f"chart packet SHA drift for {name}")
            packets[name] = packet
            decoded_by_name[name], _ = deserialize_lane_band_rd(packet)
        fit_stats = chart_manifest["fit_stats"]
    else:
        pairs, fit_stats = build_lane_band_pairs_from_lstars(labels, fit_cfg)
        packets = {}
        decoded_by_name = {}
        variant_metadata: dict[str, Any] = {}
        for name, cfg in (
            ("coherent_slot_none_dash", fit_cfg),
            ("coherent_slot_none_continuous", continuous_cfg),
        ):
            decoded, packet, metadata = roundtrip_lines_through_rd_tracked(
                pairs, cfg, pack_mode="coherent_slot", smooth="none"
            )
            packet_path = args.stage_dir / f"{name}.lbnd2"
            atomic_bytes(packet_path, packet)
            packets[name] = packet
            decoded_by_name[name] = decoded
            variant_metadata[name] = {
                "packet_file": packet_path.name,
                "packet_bytes": len(packet),
                "packet_sha256": hashlib.sha256(packet).hexdigest(),
                "brotli_bytes": len(brotli.compress(packet, quality=11)),
                "codec_metadata": metadata,
            }
        chart_manifest = {
            "schema": "boundary_inverse_chart_stage.v1",
            "cache_sha256": cache_sha,
            "parent_receipt_sha256": parent_sha,
            "fit_stats": fit_stats,
            "variants": variant_metadata,
        }
        atomic_json(chart_manifest_path, chart_manifest)
    if len(brotli.compress(packets["coherent_slot_none_dash"], quality=11)) != BASELINE_CHART_BYTES:
        raise ValueError("reconstructed coherent Lane chart does not match settled 41,303 bytes")
    dash_lines = decoded_by_name["coherent_slot_none_dash"]

    sample_manifest_path = args.stage_dir / "sample_manifest.json"
    sample_paths = {mode: args.stage_dir / f"training_{mode}.npz" for mode in ("generic_2d", "dash_arc_phase")}
    if args.resume and sample_manifest_path.exists():
        sample_manifest = json.loads(sample_manifest_path.read_text())
        if (
            sample_manifest.get("schema") != "boundary_inverse_training_samples.v1"
            or sample_manifest.get("chart_packet_sha256")
            != hashlib.sha256(packets["coherent_slot_none_dash"]).hexdigest()
            or int(sample_manifest.get("phase_bins", -1)) != args.phase_bins
            or int(sample_manifest.get("max_per_residual_sign", -1)) != args.max_per_residual_sign
            or int(sample_manifest.get("max_zero_samples", -1)) != args.max_zero_samples
            or set(sample_manifest.get("modes", {})) != {"generic_2d", "dash_arc_phase"}
        ):
            raise ValueError("training-sample resume custody mismatch")
        samples: dict[str, dict[str, np.ndarray]] = {}
        for mode, path in sample_paths.items():
            if sha256_file(path) != sample_manifest["modes"][mode]["sha256"]:
                raise ValueError(f"training sample SHA drift for {mode}")
            with np.load(path, allow_pickle=False) as data:
                samples[mode] = {key: np.asarray(data[key]) for key in data.files}
    else:
        samples = _collect_training_samples(
            labels=labels,
            decoded_lines=dash_lines,
            cfg=fit_cfg,
            phase_bin_count=args.phase_bins,
            max_per_residual_sign=args.max_per_residual_sign,
            max_zero=args.max_zero_samples,
        )
        sample_rows: dict[str, Any] = {}
        for mode, path in sample_paths.items():
            atomic_npz(path, **samples[mode])
            sample_rows[mode] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "samples": int(samples[mode]["residual"].size),
                "nonzero_residual": int(np.count_nonzero(samples[mode]["residual"])),
            }
        sample_manifest = {
            "schema": "boundary_inverse_training_samples.v1",
            "chart_packet_sha256": hashlib.sha256(packets["coherent_slot_none_dash"]).hexdigest(),
            "phase_bins": args.phase_bins,
            "max_per_residual_sign": args.max_per_residual_sign,
            "max_zero_samples": args.max_zero_samples,
            "modes": sample_rows,
        }
        atomic_json(sample_manifest_path, sample_manifest)

    prefixes = tuple(sorted(set(args.atom_prefixes)))
    program_manifest_path = args.stage_dir / "program_manifest.json"
    sidecar_dir = args.stage_dir / "sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    programs: dict[str, SparseInverseProgram] = {}
    sidecars: dict[str, bytes] = {}
    if args.resume and program_manifest_path.exists():
        program_manifest = json.loads(program_manifest_path.read_text())
        expected = {f"{mode}_k{prefix}" for mode in ("generic_2d", "dash_arc_phase") for prefix in prefixes}
        if (
            program_manifest.get("schema") != "boundary_inverse_program_stage.v1"
            or program_manifest.get("dictionary") != dictionary_metadata()
            or set(program_manifest.get("programs", {})) != expected
            or tuple(program_manifest.get("prefixes", ())) != prefixes
            or int(program_manifest.get("phase_bins", -1)) != args.phase_bins
            or float(program_manifest.get("qstep", float("nan"))) != args.qstep
            or float(program_manifest.get("correction_threshold", float("nan"))) != args.correction_threshold
        ):
            raise ValueError("program resume prefix/mode/quantization mismatch")
        for variant, row in program_manifest["programs"].items():
            if row.get("file") != f"{variant}.bic1":
                raise ValueError(f"program sidecar path drift for {variant}")
            path = sidecar_dir / row["file"]
            sidecar = path.read_bytes()
            if hashlib.sha256(sidecar).hexdigest() != row["sha256"] or len(sidecar) != int(row["bytes"]):
                raise ValueError(f"program sidecar SHA drift for {variant}")
            sidecars[variant] = sidecar
            program = decode_program(sidecar)
            expected_mode, _, prefix_text = variant.rpartition("_k")
            expected_bins = 1 if expected_mode == "generic_2d" else args.phase_bins
            if (
                not prefix_text.isdigit()
                or program.coordinate_mode != expected_mode
                or program.phase_bin_count != expected_bins
                or program.qstep != args.qstep
                or program.threshold != args.correction_threshold
                or program.atom_count != int(row["atom_count"])
            ):
                raise ValueError(f"program sidecar metadata drift for {variant}")
            programs[variant] = program
    else:
        program_rows: dict[str, Any] = {}
        for mode, phase_bin_count in (("generic_2d", 1), ("dash_arc_phase", args.phase_bins)):
            fit = fit_sparse_program_sweep(
                coords=samples[mode]["coords"],
                phase_bins=samples[mode]["phase_bins"],
                residual=samples[mode]["residual"],
                coordinate_mode=mode,
                phase_bin_count=phase_bin_count,
                atoms_per_bin_values=prefixes,
                qstep=args.qstep,
                threshold=args.correction_threshold,
            )
            for prefix, result in fit.items():
                variant = f"{mode}_k{prefix}"
                sidecar = encode_program(result.program)
                path = sidecar_dir / f"{variant}.bic1"
                atomic_bytes(path, sidecar)
                sidecars[variant] = sidecar
                programs[variant] = result.program
                program_rows[variant] = {
                    "file": path.name,
                    "bytes": len(sidecar),
                    "sha256": hashlib.sha256(sidecar).hexdigest(),
                    "atom_count": result.program.atom_count,
                    "selected_atom_ids_by_bin": [list(values) for values in result.selected_atom_ids_by_bin],
                    "sample_count_by_bin": list(result.sample_count_by_bin),
                    "residual_count_by_bin": list(result.residual_count_by_bin),
                }
        program_manifest = {
            "schema": "boundary_inverse_program_stage.v1",
            "dictionary": dictionary_metadata(),
            "prefixes": list(prefixes),
            "phase_bins": args.phase_bins,
            "qstep": args.qstep,
            "correction_threshold": args.correction_threshold,
            "programs": program_rows,
        }
        atomic_json(program_manifest_path, program_manifest)

    evaluation_dir = args.stage_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    inverse_rows: list[dict[str, Any]] = []
    renderer_source_sha256 = sha256_file(REPO / "src/tac/optimization/boundary_inverse_custody.py")
    for variant in sorted(programs):
        inverse_rows.append(
            _measure_inverse_row(
                variant=variant,
                labels=labels,
                decoded_lines=dash_lines,
                program=programs[variant],
                sidecar=sidecars[variant],
                sidecar_path=sidecar_dir / f"{variant}.bic1",
                cfg=fit_cfg,
                road_class=road_class,
                cache_sha256=cache_sha,
                renderer_source_sha256=renderer_source_sha256,
                state_path=evaluation_dir / f"{variant}.json",
                resume=args.resume,
                chunk_pairs=args.chunk_pairs,
            )
        )
    best_generic = max(
        (row for row in inverse_rows if row["coordinate_mode"] == "generic_2d"),
        key=lambda row: (row["mask_fidelity"]["f1"], -row["finite_sidecar_bytes"]),
    )
    best_phase = max(
        (row for row in inverse_rows if row["coordinate_mode"] == "dash_arc_phase"),
        key=lambda row: (row["mask_fidelity"]["f1"], -row["finite_sidecar_bytes"]),
    )

    proof = args.genuine_frame_proof
    try:
        proof_receipt_path = str(proof.resolve().relative_to(REPO))
    except ValueError:
        proof_receipt_path = str(proof.resolve())
    receipt = {
        "schema": SCHEMA,
        **authority_labels(),
        "gt_cache": {
            "path": str(args.gt_cache),
            "bytes": args.gt_cache.stat().st_size,
            "sha256": cache_sha,
            "access": "ZIP_STORED read-only memmap; one 384x512 label plane per render",
        },
        "parent_lane_receipt": {
            "path": str(args.parent_receipt),
            "bytes": args.parent_receipt.stat().st_size,
            "sha256": parent_sha,
            "settled_curve_consumed_without_relabelling": True,
        },
        "storage_preflight": {
            "stage_dir": str(args.stage_dir.resolve()),
            "free_bytes_before": storage.free,
            "minimum_required_bytes": args.minimum_stage_free_bytes,
            "passed": True,
            "source_cache_mutated": False,
        },
        "resumability": {
            "resume_enabled": bool(args.resume),
            "chart_manifest": str(chart_manifest_path.resolve()),
            "sample_manifest": str(sample_manifest_path.resolve()),
            "program_manifest": str(program_manifest_path.resolve()),
            "evaluation_chunk_pairs": args.chunk_pairs,
            "all_stage_outputs_preserved": True,
        },
        "fit": fit_stats,
        "semantic_detection": semantics.to_dict(),
        "curve": parent_receipt["curve"],
        "chart_reconstruction": chart_manifest,
        "curvelet_shearlet_residual": {
            "status": "TARGET_BOUNDARY_INVERSE_CUSTODY_CLOSED_MASK_ONLY",
            "genuine_structural_proof_path": proof_receipt_path,
            "genuine_structural_proof_sha256": sha256_file(proof),
            "dictionary": dictionary_metadata(),
            "solver": "deterministic correlation screen plus ridge finite-column solve",
            "coefficient_codec": "#557 repository left/up sign-magnitude context arithmetic",
            "sweep": inverse_rows,
            "phase_conditioning_treatment": {
                "generic_control_variant": best_generic["variant"],
                "phase_treatment_variant": best_phase["variant"],
                "generic_f1": best_generic["mask_fidelity"]["f1"],
                "phase_f1": best_phase["mask_fidelity"]["f1"],
                "phase_minus_generic_f1": (best_phase["mask_fidelity"]["f1"] - best_generic["mask_fidelity"]["f1"]),
                "treatment_is_structural": True,
                "coordinate": "decoded polynomial centerline arc length modulo decoded dash period",
            },
            "remaining_gate": "FORMALIZATION_PENDING_THROUGH_R_REALIZED_SCORE_RECOVERY",
        },
        "waterfill_contract": {
            "equation_id": "realization_breakeven_bytes_v1",
            "rate_price_s_per_byte": RATE_PRICE_PER_BYTE,
            "mask_f1_substituted_for_score": False,
            "status": "FORMALIZATION_PENDING",
        },
        "rule118_split": {
            "counted": [
                "coordinate mode and phase-bin count",
                "qstep and correction threshold",
                "dense int8 coefficient tensor whose nonzeros select finite atoms",
                "context arithmetic model/header/payload bytes",
            ],
            "free_generic_interpreter": [
                "literal polar curvelet frame construction",
                "compact shearlet frame construction",
                "decoded-chart corridor and arc-length phase construction",
                "sparse solve replay and correction renderer",
            ],
            "another_video_decode_tested": True,
            "truth_required_at_decode": False,
        },
        "content_lineage": {
            "from_scratch_our_solve": True,
            "inherited_bytes_in_candidate": 0,
            "genuine_frame_artifact_role": "generic interpreter law; zero learned frame bytes",
        },
        "argv": sys.argv,
        "measured_seconds": time.monotonic() - started,
        "verdict": "GENUINE_RESIDUAL_INVERSE_MEASURED_MASK_ONLY_RATE_ADOPTION_PENDING",
        "verdict_scope": (
            "real n600 coherent-slot polynomial Lane chart after finite LBND2 decode, true-mask "
            "fidelity plus exact finite genuine-frame sidecars only; not through-R, not d_seg, "
            "not a score, and not rate-admissible until realized score recovery supplies the "
            "canonical break-even law input."
        ),
    }
    atomic_json(args.receipt, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--expected-gt-cache-sha256", default=GT_CACHE_SHA256)
    parser.add_argument(
        "--parent-receipt",
        type=Path,
        default=REPO / ".omx/research/s2_lane_true_mask_curve_20260721T042500Z.json",
    )
    parser.add_argument("--expected-parent-receipt-sha256", default=PARENT_RECEIPT_SHA256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--atom-prefixes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--phase-bins", type=int, default=8)
    parser.add_argument("--qstep", type=float, default=1.0 / 64.0)
    parser.add_argument("--correction-threshold", type=float, default=0.25)
    parser.add_argument("--max-per-residual-sign", type=int, default=32)
    parser.add_argument("--max-zero-samples", type=int, default=32)
    parser.add_argument("--chunk-pairs", type=int, default=25)
    parser.add_argument("--minimum-stage-free-bytes", type=int, default=512 << 20)
    parser.add_argument(
        "--genuine-frame-proof",
        type=Path,
        default=REPO
        / ".omx/research/genuine_curvelet_shearlet_structural_proof_v2_polar_frequency_wedge_20260714.json",
    )
    return parser.parse_args()


def main() -> int:
    receipt = measure(parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
