#!/usr/bin/env python3
"""D3A: source-local analytic Lane carrier over the receiver-closed D3 quotient.

This experiment measures whether a counted analytic chart can restore Lane after
the D3 Lane->Road alphabet quotient.  The video-derived LaneLine coefficients,
their quantized LBND2 stream, the scalar rendering configuration, and the
coverage threshold are all counted in the candidate archive.  Only the generic
LBND2 decoder and analytic AA-SDF rasterizer are rule-118-free.

Stages are explicit, crash-resumable, and retain every materialized payload:

``build``
    Fit the source-local chart once, retain the raw LBND1 chart, then create four
    LBND2 quantization rungs.  Independently parse each counted carrier and retain
    its regenerated coverage field plus decoded and Road-gated masks.
``render``
    Parse one exact counted carrier again, regenerate its Lane paint, compose it
    with the receiver-closed D3 four-symbol stream, pack an archive, and retain
    the GB1-rendered n600 raw with chunk checkpoints.
``score``
    Score that exact raw against the pinned DALI GT tables with the frozen CPU
    SegNet/PoseNet, retaining every chunk output and the n600 arrays.
``summarize``
    Join the four byte-closed rows without silently filling missing measurements.

AXIS: [macOS-CPU advisory / DALI-GT pinned n600].  Nothing produced here is an
exact contest score or authority to move the canonical frontier pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_d3_alphabet_merge as d3
from experiments import ddm_jg2_tail_reencode as jg2
from tac.boundary_math.analytic_lane_render_band import (
    LaneBandRDTolerance,
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    deserialize_lane_band_any,
    rasterize_lane_coverage_range_dependent,
    serialize_lane_band,
    serialize_lane_band_rd_tracked,
)

STORE = Path("/Volumes/APDataStore/pact/ddm_d3a_analytic_lane_carrier")
D3_STORE = Path("/Volumes/APDataStore/pact/ddm_d3_alphabet_merge")
D3_STREAM = D3_STORE / "retained/encode/token_stream_alphabet4_n600.bin"
D3_STREAM_SHA256 = "84fa2f499fb6c052cf6a43f8cae98c227ac32412ce1495cc715aa5af94b8692d"
D3_STREAM_BYTES = 49_696
D3_MERGED = D3_STORE / "retained/fields/tokens_lane_to_road_canonical.u8"
D3_MERGED_SHA256 = "deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07"

N, H, W = 600, 384, 512
PLANE = H * W
FIELD_BYTES = N * PLANE
LANE = 1
ROAD = 0
AXIS = "[macOS-CPU advisory / DALI-GT pinned n600]"
MAGIC = b"D3A1"
MIN_BUILD_FREE_BYTES = 3 << 30
MIN_RENDER_FREE_BYTES = 5 << 30
RATE_DENOMINATOR = 37_545_489
GB1_REFERENCE = {
    "score": 0.14811799921260607,
    "archive_bytes": d3.BASE_ARCHIVE_BYTES,
    "d_seg_report_8dp": 0.00020139,
    "d_pose_report_8dp": 0.00000637,
    "axis": "[contest-CUDA T4 n600]",
}

RUNG_CONFIGS: dict[str, dict[str, float]] = {
    "q8": {"scale": 8.0, "coverage_threshold": 0.95},
    "q4": {"scale": 4.0, "coverage_threshold": 0.95},
    "q2": {"scale": 2.0, "coverage_threshold": 0.95},
    "q1": {"scale": 1.0, "coverage_threshold": 0.95},
}


class D3AError(RuntimeError):
    """A custody, receiver, retention, or authority gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def progress(**record: Any) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def require_fact(path: Path, *, size: int, sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != sha256:
        raise D3AError(f"{label} custody drifted: {path}")
    return file_fact(path)


def verify_inputs(store: Path, required_free_bytes: int) -> dict[str, Any]:
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < required_free_bytes:
        raise D3AError(
            f"storage preflight failed at {store}: {free} B free < {required_free_bytes} B"
        )
    if not (D3_STORE / "DECODE_RESULT.json").is_file():
        raise D3AError("D3 receiver proof is absent")
    proof = json.loads((D3_STORE / "DECODE_RESULT.json").read_text(encoding="utf-8"))
    if not (proof.get("complete") and proof.get("receiver_closed") and proof.get("byte_identical")):
        raise D3AError("D3 receiver proof is not complete and byte-identical")
    return {
        "storage": {
            "path": str(store),
            "required_free_bytes": required_free_bytes,
            "observed_free_bytes": free,
            "status": "PASS",
        },
        "base_archive": require_fact(
            d3.BASE_ARCHIVE,
            size=d3.BASE_ARCHIVE_BYTES,
            sha256=d3.BASE_ARCHIVE_SHA256,
            label="GB1 archive",
        ),
        "source_field": require_fact(
            d3.SOURCE_FIELD,
            size=FIELD_BYTES,
            sha256=d3.SOURCE_FIELD_SHA256,
            label="source token field",
        ),
        "d3_stream": require_fact(
            D3_STREAM, size=D3_STREAM_BYTES, sha256=D3_STREAM_SHA256, label="D3 stream"
        ),
        "d3_merged": require_fact(
            D3_MERGED, size=FIELD_BYTES, sha256=D3_MERGED_SHA256, label="D3 merged field"
        ),
        "d3_receiver_proof": file_fact(D3_STORE / "DECODE_RESULT.json"),
    }


def render_config() -> LaneBandRenderConfig:
    return LaneBandRenderConfig(
        softness=1.0,
        dash_gate=True,
        dash_forward_max_m=55.0,
        weight=1.0,
        lane_cls=LANE,
        u_mask_enabled=False,
    )


def tolerance(scale: float) -> LaneBandRDTolerance:
    return LaneBandRDTolerance(
        lat_tol_m=0.02 * scale,
        f_ref_m=30.0,
        hw_tol_px=0.1 * scale,
        v_ref_rows=200.0,
        dash_period_tol_m=0.1 * scale,
        dash_phase_tol_m=0.1 * scale,
        dash_duty_tol=0.02 * scale,
        forward_range_tol_m=0.5 * scale,
    )


def brotli_compress(source: Path, destination: Path) -> list[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/opt/homebrew/bin/brotli",
        "-f",
        "-q",
        "11",
        "-o",
        str(destination),
        str(source),
    ]
    subprocess.run(command, check=True)
    restored = subprocess.run(
        ["/opt/homebrew/bin/brotli", "-d", "-c", str(destination)],
        check=True,
        capture_output=True,
    ).stdout
    if restored != source.read_bytes():
        raise D3AError(f"Brotli parse-back changed {source}")
    return command


def encode_counted_carrier(
    *, candidate_id: str, compressed_body: bytes, tolerance_config: dict[str, Any],
    coverage_threshold: float, body_fact: dict[str, Any], source_chart_fact: dict[str, Any],
) -> bytes:
    header = {
        "schema": "ddm_d3a_counted_lane_carrier.v1",
        "candidate_id": candidate_id,
        "shape": [N, H, W],
        "chart_codec": "LBND2 coherent_slot smooth=none, Brotli q11",
        "tolerance": tolerance_config,
        "coverage_threshold": float(coverage_threshold),
        "road_gate": "decoded analytic Lane coverage AND receiver token == Road",
        "counted_video_derived": [
            "all quantized LaneLine coefficients and presence bits in the LBND2 body",
            "the scalar tolerance and coverage-threshold header",
        ],
        "rule118_free_generic": [
            "LBND2 parser and coefficient dequantizer",
            "AA-SDF range-dependent-dash rasterizer",
            "Road-gate and Lane token paint algorithm",
        ],
        "forbidden_absent": [
            "GT mask", "per-pixel lookup table", "scorer weights", "hidden donor chart"
        ],
        "source_chart": source_chart_fact,
        "uncompressed_lbnd2": body_fact,
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return (
        MAGIC
        + struct.pack("<I", len(header_bytes))
        + header_bytes
        + struct.pack("<I", len(compressed_body))
        + compressed_body
    )


def decode_counted_carrier(payload: bytes) -> tuple[list[list[Any]], dict[str, Any], bytes]:
    if payload[: len(MAGIC)] != MAGIC:
        raise D3AError("counted carrier magic differs")
    offset = len(MAGIC)
    if len(payload) < offset + 8:
        raise D3AError("counted carrier is truncated")
    (header_length,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    header = json.loads(payload[offset : offset + header_length].decode("utf-8"))
    offset += header_length
    (body_length,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    compressed = payload[offset : offset + body_length]
    if offset + body_length != len(payload):
        raise D3AError("counted carrier has trailing or truncated bytes")
    restored = subprocess.run(
        ["/opt/homebrew/bin/brotli", "-d", "-c"],
        input=compressed,
        check=True,
        capture_output=True,
    ).stdout
    pairs_lines, lbnd_header = deserialize_lane_band_any(restored)
    if len(pairs_lines) != N:
        raise D3AError(f"counted carrier decoded {len(pairs_lines)} pairs, expected {N}")
    return pairs_lines, {"carrier": header, "lbnd2": lbnd_header}, restored


def source_chart(store: Path) -> tuple[list[list[Any]], dict[str, Any], dict[str, Any]]:
    root = store / "retained/source_chart"
    raw = root / "source_local_lane_chart.lbnd1"
    compressed = root / "source_local_lane_chart.lbnd1.br"
    receipt_path = root / "SOURCE_CHART_RESULT.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if file_fact(raw) != receipt["raw_lbnd1"] or file_fact(compressed) != receipt["brotli_q11"]:
            raise D3AError("retained source chart drifted")
        if receipt.get("source_field") != file_fact(d3.SOURCE_FIELD):
            raise D3AError("retained source chart is bound to a different source field")
        if receipt.get("fit_config", {}).get("render_config") != vars(render_config()):
            raise D3AError("retained source chart is bound to a different render config")
        pairs_lines, header = deserialize_lane_band_any(raw.read_bytes())
        return pairs_lines, header, receipt

    cfg = render_config()
    source = np.memmap(d3.SOURCE_FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    started = time.perf_counter()
    pairs_lines, fit_stats = build_lane_band_pairs_from_lstars(source, cfg, centerline_deg=3)
    raw_payload = serialize_lane_band(pairs_lines, cfg)
    atomic_bytes(raw, raw_payload)
    command = brotli_compress(raw, compressed)
    parsed, header = deserialize_lane_band_any(raw.read_bytes())
    if len(parsed) != N:
        raise D3AError("source LBND1 parse-back lost pairs")
    receipt = {
        "schema": "ddm_d3a_source_chart.v1",
        "complete": True,
        "source_field": file_fact(d3.SOURCE_FIELD),
        "fit_config": {
            "centerline_degree": 3,
            "render_config": vars(cfg),
            "source_semantics": "GB1 source-local canonical token field; Lane class == 1",
        },
        "fit_stats": fit_stats,
        "raw_lbnd1": file_fact(raw),
        "brotli_q11": file_fact(compressed),
        "coder_argv": command,
        "parse_back_pairs": len(parsed),
        "parse_back_complete": True,
        "elapsed_seconds": time.perf_counter() - started,
        "retention": "FULL_SOURCE_LOCAL_CHART_BYTES",
        "axis": "[scorer-free source-local fit]",
        "score_claim": False,
    }
    atomic_json(receipt_path, receipt)
    return pairs_lines, header, receipt


def mask_statistics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, bool)
    pred = np.asarray(candidate, bool)
    tp = int(np.logical_and(ref, pred).sum())
    fp = int(np.logical_and(~ref, pred).sum())
    fn = int(np.logical_and(ref, ~pred).sum())
    return {
        "reference_positive": int(ref.sum()),
        "candidate_positive": int(pred.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
    }


def build_rung(
    store: Path, candidate_id: str, pairs_lines: list[list[Any]], source_receipt: dict[str, Any],
) -> dict[str, Any]:
    spec = RUNG_CONFIGS[candidate_id]
    root = store / "retained/rungs" / candidate_id
    receipt_path = root / "BUILD_RESULT.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_tol = vars(tolerance(RUNG_CONFIGS[candidate_id]["scale"]))
        if (
            receipt.get("candidate_id") != candidate_id
            or receipt.get("tolerance") != expected_tol
            or receipt.get("coverage_threshold")
            != RUNG_CONFIGS[candidate_id]["coverage_threshold"]
        ):
            raise D3AError(f"{candidate_id}: retained rung binding differs")
        for key in (
            "lbnd2_raw", "lbnd2_brotli_q11", "counted_carrier", "coverage_n600",
            "decoded_mask", "road_gated_mask",
        ):
            if file_fact(Path(receipt[key]["path"])) != receipt[key]:
                raise D3AError(f"{candidate_id}: retained {key} drifted")
        if "source_lane_mask" not in receipt:
            source_lane_mask = d3.retained_paths(D3_STORE)["class1_mask"]
            if not source_lane_mask.is_file():
                raise D3AError("D3 retained exact source-Lane mask is absent")
            receipt["source_lane_mask"] = file_fact(source_lane_mask)
            atomic_json(receipt_path, receipt)
        elif file_fact(Path(receipt["source_lane_mask"]["path"])) != receipt["source_lane_mask"]:
            raise D3AError(f"{candidate_id}: retained source-Lane mask drifted")
        return receipt

    root.mkdir(parents=True, exist_ok=True)
    cfg = render_config()
    tol = tolerance(spec["scale"])
    raw = root / "lane_chart.lbnd2"
    compressed = root / "lane_chart.lbnd2.br"
    carrier_path = root / "counted_lane_carrier.d3a"
    coverage_path = root / "coverage_n600.float32.npy"
    coverage_partial = root / "coverage_n600.float32.npy.partial"
    decoded_mask_path = root / "decoded_lane_mask.packbits"
    road_mask_path = root / "road_gated_lane_mask.packbits"
    started = time.perf_counter()

    lbnd2, tracking = serialize_lane_band_rd_tracked(
        pairs_lines,
        cfg,
        tol=tol,
        pack_mode="coherent_slot",
        smooth="none",
    )
    atomic_bytes(raw, lbnd2)
    command = brotli_compress(raw, compressed)
    tolerance_config = vars(tol)
    carrier = encode_counted_carrier(
        candidate_id=candidate_id,
        compressed_body=compressed.read_bytes(),
        tolerance_config=tolerance_config,
        coverage_threshold=spec["coverage_threshold"],
        body_fact=file_fact(raw),
        source_chart_fact=source_receipt["raw_lbnd1"],
    )
    atomic_bytes(carrier_path, carrier)

    decoded_lines, parsed_headers, restored = decode_counted_carrier(carrier_path.read_bytes())
    if restored != raw.read_bytes():
        raise D3AError(f"{candidate_id}: counted carrier parse-back changed LBND2 bytes")

    checkpoints = root / "coverage_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    if coverage_path.is_file():
        coverage = np.load(coverage_path, mmap_mode="r", allow_pickle=False)
        if coverage.shape != (N, H, W) or coverage.dtype != np.float32:
            raise D3AError(f"{candidate_id}: retained coverage geometry drifted")
    else:
        completed = sorted(checkpoints.glob("frame_*.json"))
        start = 0
        if completed:
            last = json.loads(completed[-1].read_text(encoding="utf-8"))
            start = int(last["frame_stop_exclusive"])
        if coverage_partial.is_file():
            coverage = np.lib.format.open_memmap(coverage_partial, mode="r+")
            if coverage.shape != (N, H, W) or coverage.dtype != np.float32:
                raise D3AError(f"{candidate_id}: partial coverage geometry drifted")
        else:
            if start:
                raise D3AError(f"{candidate_id}: coverage checkpoint lacks its partial payload")
            coverage = np.lib.format.open_memmap(
                coverage_partial, mode="w+", dtype=np.float32, shape=(N, H, W)
            )
        for frame in range(start, N):
            coverage[frame] = rasterize_lane_coverage_range_dependent(
                decoded_lines[frame],
                h=H,
                w=W,
                softness=cfg.softness,
                dash_gate=cfg.dash_gate,
                dash_forward_max_m=cfg.dash_forward_max_m,
                v_h=cfg.v_h,
                cx=cfg.cx,
            )
            if (frame + 1) % 20 == 0 or frame + 1 == N:
                coverage.flush()
                checkpoint_path = checkpoints / f"frame_{frame + 1:04d}.json"
                atomic_json(
                    checkpoint_path,
                    {
                        "schema": "ddm_d3a_coverage_checkpoint.v1",
                        "candidate_id": candidate_id,
                        "frame_stop_exclusive": frame + 1,
                        "partial_payload": {
                            "path": str(coverage_partial),
                            "bytes": coverage_partial.stat().st_size,
                        },
                        "checkpoint_complete": True,
                    },
                )
                progress(stage="build", candidate_id=candidate_id, frame=frame + 1)
        del coverage
        os.replace(coverage_partial, coverage_path)
        coverage = np.load(coverage_path, mmap_mode="r", allow_pickle=False)

    merged = np.memmap(D3_MERGED, dtype=np.uint8, mode="r", shape=(N, H, W))
    raw_mask_tmp = decoded_mask_path.with_suffix(decoded_mask_path.suffix + ".partial")
    road_mask_tmp = road_mask_path.with_suffix(road_mask_path.suffix + ".partial")
    if not (decoded_mask_path.is_file() and road_mask_path.is_file()):
        with raw_mask_tmp.open("wb") as raw_out, road_mask_tmp.open("wb") as road_out:
            for frame in range(N):
                decoded = np.asarray(coverage[frame]) >= float(spec["coverage_threshold"])
                road_gated = np.logical_and(decoded, np.asarray(merged[frame]) == ROAD)
                raw_out.write(np.packbits(decoded.reshape(-1), bitorder="little").tobytes())
                road_out.write(np.packbits(road_gated.reshape(-1), bitorder="little").tobytes())
        os.replace(raw_mask_tmp, decoded_mask_path)
        os.replace(road_mask_tmp, road_mask_path)

    packed_decoded = np.frombuffer(decoded_mask_path.read_bytes(), dtype=np.uint8)
    packed_road = np.frombuffer(road_mask_path.read_bytes(), dtype=np.uint8)
    decoded = np.unpackbits(packed_decoded, bitorder="little")[:FIELD_BYTES].reshape(N, H, W)
    road_gated = np.unpackbits(packed_road, bitorder="little")[:FIELD_BYTES].reshape(N, H, W)
    source_lane_mask = d3.retained_paths(D3_STORE)["class1_mask"]
    if not source_lane_mask.is_file():
        raise D3AError("D3 retained exact source-Lane mask is absent")
    reference = d3.exact_lane_mask(D3_STORE)
    archive_projection = 116_287 + carrier_path.stat().st_size
    result = {
        "schema": "ddm_d3a_rung_build.v1",
        "complete": True,
        "candidate_id": candidate_id,
        "source_chart": source_receipt["raw_lbnd1"],
        "source_lane_mask": file_fact(source_lane_mask),
        "tolerance": tolerance_config,
        "coverage_threshold": spec["coverage_threshold"],
        "tracking": tracking,
        "lbnd2_raw": file_fact(raw),
        "lbnd2_brotli_q11": file_fact(compressed),
        "brotli_argv": command,
        "counted_carrier": file_fact(carrier_path),
        "coverage_n600": file_fact(coverage_path),
        "decoded_mask": file_fact(decoded_mask_path),
        "road_gated_mask": file_fact(road_mask_path),
        "metrics_vs_source_lane": mask_statistics(reference, decoded),
        "metrics_after_receiver_road_gate": mask_statistics(reference, road_gated),
        "parse_back": {
            "carrier_bytes_consumed": len(carrier),
            "lbnd2_bytes_recovered": len(restored),
            "pairs": len(decoded_lines),
            "headers": parsed_headers,
            "byte_identical": True,
        },
        "projected_archive_bytes": archive_projection,
        "projected_archive_delta_vs_gb1_bytes": archive_projection - d3.BASE_ARCHIVE_BYTES,
        "projection_warning": "pre-pack arithmetic only; actual archive and scorer rows are separate gates",
        "retention": "FULL_CHART_COMPRESSED_CARRIER_COVERAGE_AND_MASK_BYTES",
        "elapsed_seconds": time.perf_counter() - started,
        "axis": "[scorer-free source-local chart build]",
        "score_claim": False,
    }
    atomic_json(receipt_path, result)
    return result


def stage_build(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.resume_from)
    custody = verify_inputs(store, MIN_BUILD_FREE_BYTES)
    pairs_lines, _header, source_receipt = source_chart(store)
    rows = []
    candidate_ids = list(RUNG_CONFIGS) if args.candidate == "all" else [args.candidate]
    for candidate_id in candidate_ids:
        rows.append(build_rung(store, candidate_id, pairs_lines, source_receipt))
    result = {
        "schema": "ddm_d3a_build.v1",
        "complete": len(rows) == len(candidate_ids),
        "custody": custody,
        "source_chart": source_receipt,
        "rows": rows,
        "axis": "[scorer-free source-local chart build]",
        "score_claim": False,
    }
    atomic_json(store / "BUILD_RESULT.json", result)
    return result


def load_mask_from_carrier(carrier_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    pairs_lines, headers, restored = decode_counted_carrier(carrier_path.read_bytes())
    carrier_header = headers["carrier"]
    cfg_header = headers["lbnd2"]
    threshold = float(carrier_header["coverage_threshold"])
    cfg = render_config()
    mask = np.zeros((N, H, W), dtype=bool)
    for frame in range(N):
        coverage = rasterize_lane_coverage_range_dependent(
            pairs_lines[frame], h=H, w=W, softness=cfg.softness,
            dash_gate=cfg.dash_gate, dash_forward_max_m=cfg.dash_forward_max_m,
            v_h=cfg.v_h, cx=cfg.cx,
        )
        mask[frame] = coverage >= threshold
    return mask, {
        "carrier": file_fact(carrier_path),
        "restored_lbnd2_bytes": len(restored),
        "pairs": len(pairs_lines),
        "coverage_threshold": threshold,
        "lbnd2_format": cfg_header.get("format"),
        "independent_parse_back": True,
    }


def load_renderer_environment() -> dict[str, Any]:
    os.environ.setdefault("CP135_BROTLI_CLI", "/opt/homebrew/bin/brotli")
    residual, renderer, renderer_dir = jg2.load_runtime(d3.RUNTIME)
    parts = residual.read_residual_archive(d3.BASE_ARCHIVE)
    sections = jg2.split_member(jg2.read_archive_member(d3.BASE_ARCHIVE))
    if sections["tail"][jg2.RESIDUAL_COMPACT_BYTES :] != parts.token_stream:
        raise D3AError("GB1 member tail disagrees with its runtime parser")
    return {
        "residual": residual,
        "renderer": renderer,
        "renderer_dir": renderer_dir,
        "parts": parts,
        "sections": sections,
    }


def prepare_renderer(env: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any] | None]:
    import torch
    from runtime.carrier_repack import materialize_cpr1, split_frame0_selector_carrier
    from runtime.compensation_overlay import apply_compensation_overlay
    from runtime.entropy.renderer_weight_codec import decode_wans1

    renderer = env["renderer"]
    parts = env["parts"]
    carrier_blob, selector_blob = split_frame0_selector_carrier(parts.carrier_blob)
    canonical_carrier = materialize_cpr1(carrier_blob, renderer)
    semantic_pose = struct.pack("<II", 40_252, len(canonical_carrier)) + bytes(40_252) + canonical_carrier
    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
    compensation = None
    if parts.compensation_blob is not None:
        basis_count = renderer.CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W
        _, _, coefficient_scales, encoded = renderer.decode_compact_carrier(
            canonical_carrier,
            basis_count=basis_count,
            frames=renderer.N,
            dimensions=renderer.CARRIER_DIM,
        )
        delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
        base_codes = np.cumsum(delta, axis=0) & 0xFFF
        base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes).astype(np.int32)
        candidate_codes = apply_compensation_overlay(base_codes, parts.compensation_blob)
        coefficients = torch.from_numpy(candidate_codes).float() * torch.from_numpy(coefficient_scales)[None]
        compensation = {
            "payload_bytes": len(parts.compensation_blob),
            "payload_sha256": hashlib.sha256(parts.compensation_blob).hexdigest(),
        }
    semantic = renderer.SemanticTokenRenderer(96)
    tagged = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, semantic.state_dict())
    if tagged is None:
        tagged = {
            record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
            for record in decode_wans1(parts.semantic_blob)
        }
    semantic.load_state_dict(tagged, strict=True)
    return semantic, basis, coefficients, {"selector_blob": selector_blob, "compensation": compensation}


def stage_render(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.resume_from)
    custody = verify_inputs(store, MIN_RENDER_FREE_BYTES)
    candidate_id = args.candidate
    build_path = store / "retained/rungs" / candidate_id / "BUILD_RESULT.json"
    if not build_path.is_file():
        raise D3AError(f"{candidate_id}: build stage is absent")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    carrier_path = Path(build["counted_carrier"]["path"])
    if file_fact(carrier_path) != build["counted_carrier"]:
        raise D3AError(f"{candidate_id}: counted carrier drifted before render")
    root = store / "retained/candidates" / candidate_id
    result_path = root / "RENDER_RESULT.json"
    raw = root / "submission/inflated/0.raw"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for key in ("tokens", "archive", "raw"):
            if file_fact(Path(result[key]["path"])) != result[key]:
                raise D3AError(f"{candidate_id}: retained render {key} drifted")
        if result.get("carrier_receiver", {}).get("carrier") != file_fact(carrier_path):
            raise D3AError(f"{candidate_id}: retained render is bound to a different carrier")
        return result

    analytic_mask, parse_back = load_mask_from_carrier(carrier_path)
    retained_road_mask = Path(build["road_gated_mask"]["path"])
    retained_bits = np.unpackbits(
        np.frombuffer(retained_road_mask.read_bytes(), dtype=np.uint8), bitorder="little"
    )[:FIELD_BYTES].reshape(N, H, W).astype(bool, copy=False)
    merged = np.memmap(D3_MERGED, dtype=np.uint8, mode="r", shape=(N, H, W))
    road_gated = np.logical_and(analytic_mask, merged == ROAD)
    if not np.array_equal(road_gated, retained_bits):
        raise D3AError(f"{candidate_id}: independent carrier parse-back differs from retained mask")

    token_path = root / "payloads/tokens_road_gated.u8"
    token_partial = token_path.with_suffix(token_path.suffix + ".partial")
    token_checkpoints = root / "token_checkpoints"
    token_checkpoints.mkdir(parents=True, exist_ok=True)
    if not token_path.is_file():
        completed = sorted(token_checkpoints.glob("frame_*.json"))
        start = int(json.loads(completed[-1].read_text())["frame_stop_exclusive"]) if completed else 0
        if token_partial.is_file():
            if token_partial.stat().st_size != start * PLANE:
                raise D3AError(f"{candidate_id}: token checkpoint and partial payload disagree")
            mode = "ab"
        else:
            if start:
                raise D3AError(f"{candidate_id}: token checkpoint lacks its partial payload")
            token_partial.parent.mkdir(parents=True, exist_ok=True)
            mode = "wb"
        with token_partial.open(mode) as output:
            for frame in range(start, N):
                painted = np.asarray(merged[frame], dtype=np.uint8).copy()
                painted[road_gated[frame]] = LANE
                output.write(painted.tobytes(order="C"))
                if (frame + 1) % 20 == 0 or frame + 1 == N:
                    output.flush()
                    os.fsync(output.fileno())
                    atomic_json(
                        token_checkpoints / f"frame_{frame + 1:04d}.json",
                        {
                            "schema": "ddm_d3a_token_checkpoint.v1",
                            "candidate_id": candidate_id,
                            "frame_stop_exclusive": frame + 1,
                            "partial_bytes": token_partial.stat().st_size,
                            "checkpoint_complete": True,
                        },
                    )
        os.replace(token_partial, token_path)
    if token_path.stat().st_size != FIELD_BYTES:
        raise D3AError(f"{candidate_id}: token payload has the wrong size")

    env = load_renderer_environment()
    sections = dict(env["sections"])
    carrier = carrier_path.read_bytes()
    stream = D3_STREAM.read_bytes()
    sections["tail"] = (
        sections["tail"][: jg2.RESIDUAL_COMPACT_BYTES]
        + b"D3Q1"
        + struct.pack("<I", len(carrier))
        + carrier
        + stream
    )
    archive_path = root / "submission/archive.zip"
    jg2.pack_archive(jg2.join_member(sections), archive_path)

    from runtime import ddm_wc1_advisory_runtime as wc1

    semantic, basis, coefficients, renderer_meta = prepare_renderer(env)
    render_stage = root / "render_stage.raw"
    parallel = wc1.render_video_parallel(
        semantic=semantic,
        basis=basis,
        coefficients=coefficients,
        token_path=token_path,
        token_sha256=sha256_file(token_path),
        renderer_dir=d3.RUNTIME / "cpr1",
        output_path=render_stage,
        progress_dir=root / "render_checkpoints",
        pair_count=N,
        camera_height=int(env["renderer"].CAMERA_H),
        camera_width=int(env["renderer"].CAMERA_W),
        requested_workers="2",
        per_process_threads=4,
        measured_worker_rss_bytes=None,
    )
    selector = None
    if renderer_meta and renderer_meta["selector_blob"] is not None:
        from runtime.f26_inflate import _apply_frame0_selector

        selector = _apply_frame0_selector(
            render_stage, env["renderer"], renderer_meta["selector_blob"], pair_count=N
        )
    raw.parent.mkdir(parents=True, exist_ok=True)
    # A detached terminal can leave the original parent alive while a human
    # mistakenly starts a second resume parent.  Both parents validate the same
    # chunk receipts; tolerate the winner having already performed the final
    # atomic rename, but never accept an absent payload.
    if render_stage.is_file():
        os.replace(render_stage, raw)
    elif not raw.is_file():
        raise D3AError(f"{candidate_id}: completed renderer retained neither stage nor final raw")
    result = {
        "schema": "ddm_d3a_render.v1",
        "complete": True,
        "candidate_id": candidate_id,
        "custody": custody,
        "carrier_receiver": parse_back,
        "tokens": file_fact(token_path),
        "archive": file_fact(archive_path),
        "archive_delta_vs_gb1_bytes": archive_path.stat().st_size - d3.BASE_ARCHIVE_BYTES,
        "archive_delta_vs_d3_rate_only_bytes": archive_path.stat().st_size - 116_287,
        "container_schema": "residual96 || D3Q1 || uint32 carrier_bytes || D3A carrier || RC64 alphabet4 stream",
        "road_gated_lane_pixels": int(road_gated.sum()),
        "raw": file_fact(raw),
        "parallel_render": parallel,
        "selector": selector,
        "compensation": renderer_meta["compensation"] if renderer_meta else None,
        "runtime_status": "research renderer closed; public contest inflate runtime not emitted",
        "retention": "FULL_TOKENS_ARCHIVE_RAW_AND_RENDER_CHECKPOINTS",
        "axis": "[macOS-CPU advisory / unscored real renderer output]",
        "score_claim": False,
    }
    atomic_json(result_path, result)
    return result


def stage_score(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from experiments import ddm_ap1_residue_purchase_scorer as ap1

    store = Path(args.resume_from)
    verify_inputs(store, 2 << 30)
    candidate_id = args.candidate
    render_path = store / "retained/candidates" / candidate_id / "RENDER_RESULT.json"
    if not render_path.is_file():
        raise D3AError(f"{candidate_id}: render stage is absent")
    render = json.loads(render_path.read_text(encoding="utf-8"))
    raw = Path(render["raw"]["path"])
    archive = Path(render["archive"]["path"])
    if file_fact(raw) != render["raw"] or file_fact(archive) != render["archive"]:
        raise D3AError(f"{candidate_id}: render payload drifted before scoring")
    out = store / "retained/scorer" / candidate_id
    result_path = out / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            file_fact(raw) != result["candidate_raw"]
            or file_fact(archive) != result["candidate_archive"]
        ):
            raise D3AError(f"{candidate_id}: scorer result points at changed candidate bytes")
        return result

    if sha256_file(ap1.GT_SEG) != ap1.GT_SEG_SHA256 or sha256_file(ap1.GT_POSE) != ap1.GT_POSE_SHA256:
        raise D3AError("pinned DALI GT scorer tables drifted")
    if str(ap1.ADVISORY_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(ap1.ADVISORY_UPSTREAM))
    from frame_utils import TensorVideoDataset
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.manual_seed(12_341)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    distortion_net = DistortionNet().eval().to("cpu")
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    names = [line.strip() for line in ap1.VIDEO_NAMES.read_text().splitlines() if line.strip()]
    if len(names) != 1:
        raise D3AError("advisory scorer expected one public video")
    dataset = TensorVideoDataset(
        names, data_dir=raw.parent, batch_size=16, device=torch.device("cpu"),
        num_threads=2, seed=1234,
    )
    dataset.prepare_data()
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
    gt_seg = np.load(ap1.GT_SEG, allow_pickle=False, mmap_mode="r")
    gt_pose = np.load(ap1.GT_POSE, allow_pickle=False, mmap_mode="r")
    receipts = []
    start = 0
    started = time.perf_counter()
    for chunk_index, (_, _, batch_candidate) in enumerate(loader):
        stop = start + int(batch_candidate.shape[0])
        receipts.append(
            ap1._score_chunk(
                chunk_index=chunk_index, start=start, stop=stop,
                batch_candidate=batch_candidate, distortion_net=distortion_net,
                gt_seg=gt_seg, gt_pose=gt_pose, out_dir=out,
            )
        )
        progress(stage="score", candidate_id=candidate_id, stop=stop)
        start = stop
    if start != N:
        raise D3AError(f"{candidate_id}: scorer covered {start} pairs instead of n600")
    summary = ap1._aggregate_chunks(receipts)
    full_seg = np.concatenate(
        [ap1.verify_array_fact(receipt["candidate_argmax"]) for receipt in receipts], axis=0
    )
    full_pose = np.concatenate(
        [ap1.verify_array_fact(receipt["candidate_pose6"]) for receipt in receipts], axis=0
    )
    full_seg_record = ap1.atomic_npy(out / "candidate_argmax_n600.uint8.npy", full_seg)
    full_pose_record = ap1.atomic_npy(out / "candidate_pose6_n600.float32.npy", full_pose)
    score = (
        100.0 * summary["d_seg"]
        + math.sqrt(10.0 * summary["d_pose"])
        + 25.0 * archive.stat().st_size / RATE_DENOMINATOR
    )
    derivative = {
        "rate_dS_per_byte": 25.0 / RATE_DENOMINATOR,
        "seg_dS_per_unit_dseg": 100.0,
        "pose_dS_per_unit_dpose_at_gb1": 5.0 / math.sqrt(10.0 * GB1_REFERENCE["d_pose_report_8dp"]),
    }
    delta = {
        "archive_bytes": archive.stat().st_size - GB1_REFERENCE["archive_bytes"],
        "d_seg": summary["d_seg"] - GB1_REFERENCE["d_seg_report_8dp"],
        "d_pose": summary["d_pose"] - GB1_REFERENCE["d_pose_report_8dp"],
        "S": score - GB1_REFERENCE["score"],
    }
    first_order = {
        "rate": derivative["rate_dS_per_byte"] * delta["archive_bytes"],
        "seg": derivative["seg_dS_per_unit_dseg"] * delta["d_seg"],
        "pose": derivative["pose_dS_per_unit_dpose_at_gb1"] * delta["d_pose"],
    }
    result = {
        "schema": "ddm_d3a_dali_score.v1",
        "complete": True,
        "candidate_id": candidate_id,
        "axis": AXIS,
        "promotable": False,
        "score_claim": False,
        "candidate_raw": file_fact(raw),
        "candidate_archive": file_fact(archive),
        "gt_seg": file_fact(ap1.GT_SEG),
        "gt_pose": file_fact(ap1.GT_POSE),
        "segnet_weights": file_fact(Path(segnet_sd_path)),
        "posenet_weights": file_fact(Path(posenet_sd_path)),
        "chunk_receipts": [
            file_fact(
                out / "chunks" /
                f"{int(receipt['pair_start']):04d}_{int(receipt['pair_stop_exclusive']) - 1:04d}.json"
            )
            for receipt in receipts
        ],
        "candidate_argmax_n600": full_seg_record,
        "candidate_pose6_n600": full_pose_record,
        "summary": summary,
        "advisory_composed_S": score,
        "gb1_reference": GB1_REFERENCE,
        "mixed_axis_delta_warning": (
            "candidate is a macOS-CPU model forward while GB1 is contest-CUDA; "
            "the delta is route triage, not a promotable comparison"
        ),
        "delta_vs_gb1_reference": delta,
        "first_order_derivative_at_gb1": derivative,
        "first_order_delta_components": first_order,
        "exact_nonlinear_delta_components": {
            "rate": 25.0 * delta["archive_bytes"] / RATE_DENOMINATOR,
            "seg": 100.0 * delta["d_seg"],
            "pose": math.sqrt(10.0 * summary["d_pose"])
            - math.sqrt(10.0 * GB1_REFERENCE["d_pose_report_8dp"]),
        },
        "retention": "FULL_PER_CHUNK_AND_N600_SCORER_OUTPUT_BYTES",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    return result


def stage_summarize(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.resume_from)
    rows = []
    for candidate_id in RUNG_CONFIGS:
        build_path = store / "retained/rungs" / candidate_id / "BUILD_RESULT.json"
        render_path = store / "retained/candidates" / candidate_id / "RENDER_RESULT.json"
        score_path = store / "retained/scorer" / candidate_id / "RESULT.json"
        row: dict[str, Any] = {"candidate_id": candidate_id}
        if build_path.is_file():
            build = json.loads(build_path.read_text(encoding="utf-8"))
            row.update(
                {
                    "counted_carrier_bytes": build["counted_carrier"]["bytes"],
                    "source_lane_metrics": build["metrics_after_receiver_road_gate"],
                }
            )
        if render_path.is_file():
            render = json.loads(render_path.read_text(encoding="utf-8"))
            row.update(
                {
                    "archive_bytes": render["archive"]["bytes"],
                    "archive_sha256": render["archive"]["sha256"],
                    "raw_sha256": render["raw"]["sha256"],
                }
            )
        if score_path.is_file():
            score = json.loads(score_path.read_text(encoding="utf-8"))
            if not score.get("complete"):
                raise D3AError(f"{candidate_id}: scorer result is not complete")
            row.update(
                {
                    "d_seg": score["summary"]["d_seg"],
                    "d_pose": score["summary"]["d_pose"],
                    "advisory_composed_S": score["advisory_composed_S"],
                    "delta_vs_gb1_reference": score["delta_vs_gb1_reference"],
                }
            )
        row["typed_outcome"] = (
            "MEASURED_N600_ADVISORY" if score_path.is_file() else
            "BYTE_CLOSED_UNSCORED" if render_path.is_file() else
            "CARRIER_BUILT" if build_path.is_file() else "ABSENT"
        )
        rows.append(row)
    complete = all(row["typed_outcome"] == "MEASURED_N600_ADVISORY" for row in rows)
    result = {
        "schema": "ddm_d3a_summary.v1",
        "complete": complete,
        "denominator": {"candidates_planned": len(RUNG_CONFIGS), "candidates_scored_n600": sum(
            row["typed_outcome"] == "MEASURED_N600_ADVISORY" for row in rows
        )},
        "rows": rows,
        "axis": AXIS,
        "promotable": False,
        "score_claim": False,
    }
    atomic_json(store / "SUMMARY.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("build", "render", "score", "summarize"))
    parser.add_argument("--resume-from", required=True)
    parser.add_argument("--candidate", choices=(*RUNG_CONFIGS, "all"), default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage in {"render", "score"} and args.candidate == "all":
        raise SystemExit(f"--candidate must name one rung for stage {args.stage}")
    result = {
        "build": stage_build,
        "render": stage_render,
        "score": stage_score,
        "summarize": stage_summarize,
    }[args.stage](args)
    progress(stage=args.stage, event="done", complete=result.get("complete"), axis=AXIS)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
