#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the real-n600 coherent Lane chart against the true Lane mask.

This is a representation-fidelity measurement, not a score.  It measures the
actual finite LBND2 stream after decode and reports pixel confusion, macro-F1,
and boundary/interior strata.  It deliberately does not synthesize a fake
curvelet residual: the genuine #502 frame is structural custody, while the
target-boundary inverse needed to fit the chart residual is still absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    rasterize_lane_coverage_range_dependent,
    roundtrip_lines_through_rd_tracked,
)
from tac.optimization.s2_partition_seed import detect_partition_semantics  # noqa: E402

SCHEMA: Final = "s2_lane_true_mask_curve.v1"
GT_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
RATE_PRICE_PER_BYTE: Final = 25.0 / 37_545_489.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
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
            (predicted_positive - true_positive) / (selected - positives)
            if selected > positives
            else None
        ),
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
        key: {"pixels": 0, "truth_positive": 0, "predicted_positive": 0, "true_positive": 0}
        for key in strata_names
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
                (row["predicted_positive"] - row["true_positive"]) / negatives
                if negatives
                else None
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
    cache_sha = sha256_file(args.gt_cache)
    if cache_sha != args.expected_gt_cache_sha256:
        raise ValueError(f"GT cache SHA drift: {cache_sha}")
    labels = npz_member_memmap(args.gt_cache, "lstars")
    if labels.shape != (600, 384, 512):
        raise ValueError(f"unexpected n600 label geometry: {labels.shape}")

    semantics = detect_partition_semantics(labels)
    road_class, lane_class = semantics.semantic_class_ids[:2]
    fit_cfg = LaneBandRenderConfig(dash_gate=True, lane_cls=lane_class)
    pairs, fit_stats = build_lane_band_pairs_from_lstars(labels, fit_cfg)
    rows: list[dict[str, Any]] = []
    for name, cfg in (
        ("coherent_slot_none_dash", fit_cfg),
        (
            "coherent_slot_none_continuous",
            LaneBandRenderConfig(dash_gate=False, lane_cls=lane_class),
        ),
    ):
        decoded, packet, metadata = roundtrip_lines_through_rd_tracked(
            pairs,
            cfg,
            pack_mode="coherent_slot",
            smooth="none",
        )
        row = _measure_row(
            name=name,
            labels=labels,
            decoded_lines=decoded,
            packet=packet,
            cfg=cfg,
            road_class=road_class,
        )
        row["codec_metadata"] = metadata
        rows.append(row)

    proof = args.genuine_frame_proof
    try:
        proof_receipt_path = str(proof.resolve().relative_to(REPO))
    except ValueError:
        proof_receipt_path = str(proof.resolve())
    receipt = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_unmoved": "0.19108 [contest-CPU]",
        "gt_cache": {
            "path": str(args.gt_cache),
            "bytes": args.gt_cache.stat().st_size,
            "sha256": cache_sha,
            "access": "ZIP_STORED read-only memmap; one 384x512 label plane per render",
        },
        "fit": fit_stats,
        "semantic_detection": semantics.to_dict(),
        "curve": rows,
        "curvelet_shearlet_residual": {
            "status": "BLOCKED_TARGET_BOUNDARY_INVERSE_CUSTODY",
            "genuine_structural_proof_path": proof_receipt_path,
            "genuine_structural_proof_sha256": sha256_file(proof),
            "allowed_families": ["finite_polar_curvelet", "compact_shearlet"],
            "forbidden_substitute": "Fourier",
            "reason": (
                "#502 proves genuine deterministic frames, but no custodied inverse maps the "
                "coherent Lane true-mask residual to finite selected atoms. A polynomial-only "
                "control curve cannot be relabeled as a curvelet/shearlet residual closure."
            ),
        },
        "content_lineage": {
            "from_scratch_our_solve": True,
            "inherited_bytes_in_candidate": 0,
            "genuine_frame_artifact_role": "structural law fixture only; zero bytes consumed",
        },
        "measured_seconds": time.monotonic() - started,
        "verdict": "LANE_TRUE_MASK_CURVE_MEASURED_GENUINE_RESIDUAL_INVERSE_OPEN",
        "verdict_scope": (
            "real n600 coherent-slot polynomial Lane chart after finite LBND2 decode, true-mask "
            "fidelity only; not through-R, not a score, and not gap-1 closure because the "
            "genuine curvelet/shearlet residual inverse is not composed."
        ),
    }
    atomic_json(args.receipt, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-gt-cache-sha256", default=GT_CACHE_SHA256)
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
