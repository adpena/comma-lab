#!/usr/bin/env python3
"""D1 semantic-renderer harness for ddm_tk2.

This runner is scorer-free by default except for the charter-permitted
``--harness-smoke`` path (n<=4) and an explicit queued full-D1 mode. It consumes
tk1's verified semantic-label source array, renders candidate RGB witnesses,
passes those RGB frames through the canonical R operator, and evaluates with the
frozen CPU SegNet argmax path.

Important artifact boundary: tk1 did not persist a standalone 142,001-byte
semantic payload. The byte receipt is a closed-form KT/context-arithmetic result
over the persisted tq1c label array plus subset real-coder round-trip proof.
This runner therefore verifies that source-label artifact and records subset
round-trip evidence before rendering; it never pretends a missing stream file was
decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
for _path in (str(REPO_ROOT), str(SRC_ROOT), str(EXPERIMENTS_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


DEFAULT_RECEIPT_DIR = REPO_ROOT / ".omx" / "research" / "ddm_tk2_20260806"
DEFAULT_SSD_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_tk2_20260806")
DEFAULT_TQ1C_LABELS = Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy")
DEFAULT_GT_LABELS = Path("/Volumes/VertigoDataTier/pact/ddm_ph1_lstars_u8.npy")
DEFAULT_TK1_RECEIPT = REPO_ROOT / ".omx" / "research" / "ddm_tk1_20260806" / "RECEIPT.md"
DEFAULT_TR1_CHECKPOINT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_tb1_20260728/t2_n600_lotto/checkpoints/stage_seg_trunk_tau_final.npz"
)

EXPECTED_TQ1C_FILE_SHA256 = "764a244c4890b22a67c4dbe95a959e970c29328778d41ffe4deb85f5b650eee6"
EXPECTED_TQ1C_RAW_SHA256 = "a7dd6f4271eedfa877f6499348de5f9dae2d97311f9e98f4f534908eb66e044e"
EXPECTED_GT_FILE_SHA256 = "b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d"
EXPECTED_GT_RAW_SHA256 = "f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557"
TK1_SELECTED_KT_BYTES = 142_001

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
N_CLASSES = len(CLASS_NAMES)
SEG_H = 384
SEG_W = 512
CAMERA_H = 874
CAMERA_W = 1164

FP1_PALETTE_RGB_U8 = np.asarray(
    [
        [30, 39, 72],
        [77, 87, 119],
        [157, 66, 56],
        [76, 108, 141],
        [129, 154, 153],
    ],
    dtype=np.float32,
)
V15_LANE_BOUNDARY_RGB_U8 = np.asarray([51, 255, 204], dtype=np.float32)
V14_MOVABLE_RGB_U8 = np.asarray([107, 0, 114], dtype=np.float32)


class CandidateUnavailable(RuntimeError):
    """Raised when a named candidate would otherwise be a fake implementation."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    title: str
    provenance: str
    renderer: Callable[[np.ndarray], np.ndarray]


def _now_utc() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"cannot JSON encode {type(obj).__name__}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _ssd_preflight(ssd_dir: Path) -> dict[str, Any]:
    ssd_dir.mkdir(parents=True, exist_ok=True)
    probe = ssd_dir / "PREFLIGHT_WRITE_PROBE.json"
    payload = {
        "schema": "ddm_tk2_ssd_preflight_v1",
        "written_at_utc": _now_utc(),
        "path": str(ssd_dir),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    _atomic_write_json(probe, payload)
    stat = os.statvfs(ssd_dir)
    payload.update(
        {
            "probe_sha256": _sha256_file(probe),
            "free_bytes": int(stat.f_bavail * stat.f_frsize),
            "total_bytes": int(stat.f_blocks * stat.f_frsize),
        }
    )
    _atomic_write_json(probe, payload)
    return payload


def _config_identity(args: argparse.Namespace) -> str:
    identity = {
        "candidate_ids": args.candidates,
        "pair_start": args.pair_start,
        "pair_count": args.pair_count,
        "chunk_size": args.chunk_size,
        "semantic_labels": str(args.semantic_labels),
        "gt_labels": str(args.gt_labels),
        "runner": str(Path(__file__).relative_to(REPO_ROOT)),
        "tk1_selected_kt_bytes": TK1_SELECTED_KT_BYTES,
    }
    blob = json.dumps(identity, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_labels_window(path: Path, start: int, stop: int) -> np.ndarray:
    if start < 0 or stop <= start:
        raise ValueError(f"invalid label window start={start} stop={stop}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3 or arr.shape[1:] != (SEG_H, SEG_W):
        raise ValueError(f"{path} has shape {arr.shape}; expected (N,{SEG_H},{SEG_W})")
    if stop > arr.shape[0]:
        raise ValueError(f"requested stop={stop} beyond {path} length {arr.shape[0]}")
    out = np.asarray(arr[start:stop], dtype=np.int64)
    if out.size and (int(out.min()) < 0 or int(out.max()) >= N_CLASSES):
        raise ValueError(f"{path} contains class ids outside [0,{N_CLASSES - 1}] in requested window")
    return out


def _verify_source_artifacts(args: argparse.Namespace, stop: int) -> dict[str, Any]:
    import ddm_pp1_direct_partition_coder as pp1

    semantic_file_sha = _sha256_file(args.semantic_labels)
    gt_file_sha = _sha256_file(args.gt_labels)
    semantic = _load_labels_window(args.semantic_labels, args.pair_start, stop)
    gt = _load_labels_window(args.gt_labels, args.pair_start, stop)
    template = pp1._PREV5 + pp1._INTRA_O8  # noqa: SLF001 - tk1 selected this exact public template.
    subset_nf = min(int(semantic.shape[0]), 6)
    subset_roundtrip = pp1.roundtrip_proof(semantic, template, nf=subset_nf)
    subset_kt_bytes, subset_contexts = pp1.adaptive_code_bytes(semantic, template, alpha=0.5)
    source_ok = bool(semantic_file_sha == EXPECTED_TQ1C_FILE_SHA256)
    gt_ok = bool(gt_file_sha == EXPECTED_GT_FILE_SHA256)
    return {
        "schema": "ddm_tk2_source_verification_v1",
        "semantic_labels_path": str(args.semantic_labels),
        "semantic_labels_file_sha256": semantic_file_sha,
        "semantic_labels_file_sha256_expected": EXPECTED_TQ1C_FILE_SHA256,
        "semantic_labels_file_sha256_match": source_ok,
        "semantic_labels_raw_sha256_expected_full_n600": EXPECTED_TQ1C_RAW_SHA256,
        "gt_labels_path": str(args.gt_labels),
        "gt_labels_file_sha256": gt_file_sha,
        "gt_labels_file_sha256_expected": EXPECTED_GT_FILE_SHA256,
        "gt_labels_file_sha256_match": gt_ok,
        "gt_labels_raw_sha256_expected_full_n600": EXPECTED_GT_RAW_SHA256,
        "window_start": int(args.pair_start),
        "window_stop": int(stop),
        "window_count": int(semantic.shape[0]),
        "semantic_window_raw_sha256": _sha256_array(semantic),
        "gt_window_raw_sha256": _sha256_array(gt),
        "tk1_selected_context_arith_bytes_from_receipt": TK1_SELECTED_KT_BYTES,
        "tk1_receipt_path": str(DEFAULT_TK1_RECEIPT),
        "artifact_boundary": (
            "tk1 did not persist a standalone 142001-byte full stream; this harness verifies "
            "the persisted source-label array plus subset real-coder roundtrip before rendering"
        ),
        "subset_context_arith_bytes_kt_alpha_0_5": float(subset_kt_bytes),
        "subset_context_count": int(subset_contexts),
        "subset_real_coder_roundtrip": subset_roundtrip,
        "source_file_matches_expected": bool(source_ok and gt_ok),
    }


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels)
    mask = np.zeros(lab.shape, dtype=bool)
    mask[:, :, 1:] |= lab[:, :, 1:] != lab[:, :, :-1]
    mask[:, :, :-1] |= lab[:, :, 1:] != lab[:, :, :-1]
    mask[:, 1:, :] |= lab[:, 1:, :] != lab[:, :-1, :]
    mask[:, :-1, :] |= lab[:, 1:, :] != lab[:, :-1, :]
    return mask


def _dilate_4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    return out


def _neighbor_mean_rgb(rgb: np.ndarray) -> np.ndarray:
    padded = np.pad(rgb, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    acc = (
        padded[:, 1:-1, 1:-1, :]
        + padded[:, :-2, 1:-1, :]
        + padded[:, 2:, 1:-1, :]
        + padded[:, 1:-1, :-2, :]
        + padded[:, 1:-1, 2:, :]
    )
    return acc / 5.0


def render_c0_flat_paint(labels: np.ndarray) -> np.ndarray:
    return FP1_PALETTE_RGB_U8[np.asarray(labels, dtype=np.int64)]


def render_c1_v15_template_paint(labels: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64)
    rgb = render_c0_flat_paint(lab)
    bmask = _boundary_mask(lab)
    rgb[(lab == 1) & bmask] = V15_LANE_BOUNDARY_RGB_U8
    rgb[lab == 3] = V14_MOVABLE_RGB_U8
    return rgb


def render_c2_boundary_aa(labels: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64)
    rgb = render_c1_v15_template_paint(lab)
    band = _dilate_4(_boundary_mask(lab))
    coverage_rgb = _neighbor_mean_rgb(rgb)
    rgb = rgb.copy()
    rgb[band] = np.clip(0.55 * rgb[band] + 0.45 * coverage_rgb[band], 0.0, 255.0)
    return rgb


def render_c3_tr1_onehot_retarget(labels: np.ndarray) -> np.ndarray:
    del labels
    raise CandidateUnavailable(
        "C3 refused: no compatible TR1 one-hot input adapter is declared in the banked LOTTO "
        "checkpoints. TR1 consumes its own token/code surface; retargeting it to one-hot class "
        "planes without a real adapter or training cell would be fake."
    )


def _candidate_specs() -> dict[str, CandidateSpec]:
    return {
        "c0_flat_paint": CandidateSpec(
            candidate_id="c0_flat_paint",
            title="C0 flat-paint prototype colors",
            provenance=(
                "fp1 measured palette; positive control target is the tk1-cited flat-paint floor "
                "near d_seg 0.008305 on n600 when driven from GT argmax"
            ),
            renderer=render_c0_flat_paint,
        ),
        "c1_v15_template_paint": CandidateSpec(
            candidate_id="c1_v15_template_paint",
            title="C1 v14/v15 analytic margin paint",
            provenance=(
                "fp1 palette plus v15 lane-boundary scorer-solved template color "
                "(51,255,204) and v14 movable prototype (107,0,114); no training"
            ),
            renderer=render_c1_v15_template_paint,
        ),
        "c2_boundary_aa": CandidateSpec(
            candidate_id="c2_boundary_aa",
            title="C2 C1 plus boundary coverage treatment",
            provenance=(
                "#149 survival-wall lesson: all-class boundary band dominates through-R errors; "
                "this applies a deterministic 4-neighbor coverage blend before R"
            ),
            renderer=render_c2_boundary_aa,
        ),
        "c3_tr1_onehot_retarget": CandidateSpec(
            candidate_id="c3_tr1_onehot_retarget",
            title="C3 TR1 one-hot retarget probe",
            provenance=(
                f"banked LOTTO checkpoint candidate {DEFAULT_TR1_CHECKPOINT}; fail-closed until "
                "a real one-hot adapter/checkpoint shape is declared"
            ),
            renderer=render_c3_tr1_onehot_retarget,
        ),
    }


def _apply_actual_r(rgb: np.ndarray) -> np.ndarray:
    """Torch-CPU contest R: render-res RGB -> bicubic camera -> uint8.

    This mirrors ``_torch_R_to_camera_uint8`` in
    ``experiments/train_witness_realized_through_R_mlx.py``. The final official
    bilinear downsample is intentionally left to ``SegNet.preprocess_input`` in
    the frozen CPU scorer path.
    """

    import torch

    rgb_f = np.asarray(rgb, dtype=np.float32)
    x = torch.from_numpy(np.ascontiguousarray(rgb_f)).permute(0, 3, 1, 2).float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up.permute(0, 2, 3, 1).contiguous().numpy().astype(np.uint8)


def _load_cpu_segnet() -> Any:
    from tac.boundary_math.seg_core import load_real_segnet

    return load_real_segnet(device="cpu")


def _class_error_rows(realized: np.ndarray, gt: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    err = realized != gt
    for cid, cname in enumerate(CLASS_NAMES):
        mask = gt == cid
        sites = int(mask.sum())
        errors = int((err & mask).sum())
        rows.append(
            {
                "class_id": cid,
                "class_name": cname,
                "sites": sites,
                "errors": errors,
                "d_seg_class": float(errors / sites) if sites else 0.0,
                "share_of_total_errors": float(errors / max(1, int(err.sum()))),
            }
        )
    return rows


def _measure_chunk(segnet_cpu: Any, spec: CandidateSpec, semantic: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    from train_witness_realized_through_R_mlx import cpu_verdict_d_seg_argmax_batch

    t0 = time.perf_counter()
    rgb = spec.renderer(semantic)
    render_seconds = time.perf_counter() - t0
    if rgb.shape != semantic.shape + (3,):
        raise ValueError(f"{spec.candidate_id} rendered shape {rgb.shape}; expected {semantic.shape + (3,)}")
    if not np.isfinite(rgb).all():
        raise ValueError(f"{spec.candidate_id} rendered non-finite RGB")
    t1 = time.perf_counter()
    post_r = _apply_actual_r(rgb)
    r_seconds = time.perf_counter() - t1
    t2 = time.perf_counter()
    d_seg_list, realized = cpu_verdict_d_seg_argmax_batch(segnet_cpu, list(post_r), list(gt))
    scorer_seconds = time.perf_counter() - t2
    total_sites = int(gt.size)
    total_errors = int(np.count_nonzero(realized != gt))
    total = {
        "sites": total_sites,
        "errors": total_errors,
        "d_seg": float(total_errors / total_sites),
    }
    return {
        "candidate_id": spec.candidate_id,
        "status": "measured_harness_smoke",
        "pair_count": int(semantic.shape[0]),
        "rgb_input_sha256": _sha256_array(np.clip(np.rint(rgb), 0, 255).astype(np.uint8)),
        "r_path": "torch_cpu_bicubic_to_camera_uint8_then_segnet_preprocess_bilinear",
        "post_r_camera_rgb_sha256": _sha256_array(post_r),
        "realized_argmax_sha256": _sha256_array(realized.astype(np.uint8)),
        "d_seg_by_pair": [float(x) for x in d_seg_list],
        "total": total,
        "per_class": _class_error_rows(realized, gt),
        "wall_seconds": {
            "render": float(render_seconds),
            "actual_R": float(r_seconds),
            "frozen_cpu_segnet": float(scorer_seconds),
            "total": float(render_seconds + r_seconds + scorer_seconds),
        },
    }


def _checkpoint_path(receipt_dir: Path, candidate_id: str, start: int, stop: int) -> Path:
    return receipt_dir / "stage_checkpoints" / candidate_id / f"chunk_{start:04d}_{stop:04d}.json"


def _load_checkpoint(path: Path, config_sha256: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("config_sha256") != config_sha256:
        return None
    return payload


def run_harness(args: argparse.Namespace) -> dict[str, Any]:
    if args.pair_count > 4 and not args.claim_scorer_slot:
        raise SystemExit(
            "Refusing pair_count>4 without --claim-scorer-slot. tk2 is scorer-free except the n<=4 "
            "harness smoke; full D1 must be queued into the scorer slot."
        )
    if args.harness_smoke and args.pair_count > 4:
        raise SystemExit("--harness-smoke enforces pair_count<=4")
    if args.chunk_size <= 0 or args.chunk_size > 120:
        raise SystemExit("--chunk-size must be in [1,120]")

    receipt_dir = args.receipt_dir
    receipt_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = _config_identity(args)
    stop = args.pair_start + args.pair_count
    preflight = _ssd_preflight(args.ssd_dir)
    source = _verify_source_artifacts(args, stop)
    if not source["source_file_matches_expected"]:
        raise SystemExit("Source/GT file sha mismatch; refusing to run scorer smoke on ambiguous labels.")

    candidate_specs = _candidate_specs()
    selected_ids = [cid.strip() for cid in args.candidates.split(",") if cid.strip()]
    unknown = sorted(set(selected_ids) - set(candidate_specs))
    if unknown:
        raise SystemExit(f"unknown candidates: {unknown}; valid={sorted(candidate_specs)}")

    segnet_cpu = _load_cpu_segnet()
    all_results: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []

    for cid in selected_ids:
        spec = candidate_specs[cid]
        cand_chunks: list[dict[str, Any]] = []
        cand_errors = 0
        cand_sites = 0
        cand_pair_dseg: list[float] = []
        cand_seconds = 0.0
        cand_class_errors = np.zeros(N_CLASSES, dtype=np.int64)
        cand_class_sites = np.zeros(N_CLASSES, dtype=np.int64)
        try:
            for chunk_start in range(args.pair_start, stop, args.chunk_size):
                chunk_stop = min(chunk_start + args.chunk_size, stop)
                ckpt = _checkpoint_path(receipt_dir, cid, chunk_start, chunk_stop)
                if args.resume:
                    existing = _load_checkpoint(ckpt, config_sha256)
                    if existing is not None:
                        result = existing["result"]
                    else:
                        semantic = _load_labels_window(args.semantic_labels, chunk_start, chunk_stop)
                        gt = _load_labels_window(args.gt_labels, chunk_start, chunk_stop)
                        result = _measure_chunk(segnet_cpu, spec, semantic, gt)
                        _atomic_write_json(
                            ckpt,
                            {
                                "schema": "ddm_tk2_chunk_checkpoint_v1",
                                "candidate_id": cid,
                                "config_sha256": config_sha256,
                                "chunk_start": chunk_start,
                                "chunk_stop": chunk_stop,
                                "written_at_utc": _now_utc(),
                                "score_claim": False,
                                "axis": "[macOS-CPU frozen-SegNet harness-smoke]",
                                "result": result,
                            },
                        )
                else:
                    semantic = _load_labels_window(args.semantic_labels, chunk_start, chunk_stop)
                    gt = _load_labels_window(args.gt_labels, chunk_start, chunk_stop)
                    result = _measure_chunk(segnet_cpu, spec, semantic, gt)
                    _atomic_write_json(
                        ckpt,
                        {
                            "schema": "ddm_tk2_chunk_checkpoint_v1",
                            "candidate_id": cid,
                            "config_sha256": config_sha256,
                            "chunk_start": chunk_start,
                            "chunk_stop": chunk_stop,
                            "written_at_utc": _now_utc(),
                            "score_claim": False,
                            "axis": "[macOS-CPU frozen-SegNet harness-smoke]",
                            "result": result,
                        },
                    )
                cand_chunks.append({"chunk_start": chunk_start, "chunk_stop": chunk_stop, "checkpoint": str(ckpt)})
                cand_errors += int(result["total"]["errors"])
                cand_sites += int(result["total"]["sites"])
                cand_pair_dseg.extend(float(x) for x in result["d_seg_by_pair"])
                cand_seconds += float(result["wall_seconds"]["total"])
                for row in result["per_class"]:
                    idx = int(row["class_id"])
                    cand_class_errors[idx] += int(row["errors"])
                    cand_class_sites[idx] += int(row["sites"])
        except CandidateUnavailable as exc:
            unavailable.append(
                {
                    "candidate_id": cid,
                    "status": "blocked_fail_closed",
                    "reason": str(exc),
                    "checkpoint_candidate": str(DEFAULT_TR1_CHECKPOINT),
                    "score_claim": False,
                }
            )
            continue

        class_rows = []
        for idx, cname in enumerate(CLASS_NAMES):
            sites = int(cand_class_sites[idx])
            errors = int(cand_class_errors[idx])
            class_rows.append(
                {
                    "class_id": idx,
                    "class_name": cname,
                    "sites": sites,
                    "errors": errors,
                    "d_seg_class": float(errors / sites) if sites else 0.0,
                }
            )
        all_results.append(
            {
                "candidate_id": cid,
                "title": spec.title,
                "provenance": spec.provenance,
                "status": "measured_harness_smoke" if args.harness_smoke else "measured_claimed_slot",
                "chunks": cand_chunks,
                "pair_start": int(args.pair_start),
                "pair_count": int(args.pair_count),
                "d_seg": float(cand_errors / cand_sites) if cand_sites else None,
                "errors": int(cand_errors),
                "sites": int(cand_sites),
                "d_seg_by_pair": cand_pair_dseg,
                "per_class": class_rows,
                "wall_seconds_total": float(cand_seconds),
                "score_claim": False,
            }
        )

    payload = {
        "schema": "ddm_tk2_harness_smoke_v1",
        "written_at_utc": _now_utc(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "axis": "[macOS-CPU frozen-SegNet harness-smoke]" if args.harness_smoke else "[macOS-CPU frozen-SegNet claimed-slot]",
        "r_path": "torch_cpu_bicubic_to_camera_uint8_then_segnet_preprocess_bilinear",
        "score_claim": False,
        "claim_scorer_slot": bool(args.claim_scorer_slot),
        "config_sha256": config_sha256,
        "pair_start": int(args.pair_start),
        "pair_count": int(args.pair_count),
        "chunk_size": int(args.chunk_size),
        "ssd_preflight": preflight,
        "source_verification": source,
        "candidate_results": all_results,
        "candidate_unavailable": unavailable,
    }
    out_path = receipt_dir / ("harness_smoke.json" if args.harness_smoke else "claimed_slot_results.json")
    _atomic_write_json(out_path, payload)
    return payload


def _format_result_table(payload: dict[str, Any]) -> str:
    rows = ["| candidate | n | d_seg | errors/sites | wall_s | status |", "|---|---:|---:|---:|---:|---|"]
    for res in payload.get("candidate_results", []):
        rows.append(
            "| {candidate_id} | {pair_count} | {d_seg:.9f} | {errors}/{sites} | {wall_seconds_total:.2f} | {status} |".format(
                **res
            )
        )
    for res in payload.get("candidate_unavailable", []):
        rows.append(f"| {res['candidate_id']} | 0 | NA | NA | NA | {res['status']} |")
    return "\n".join(rows)


def write_receipts(payload: dict[str, Any], args: argparse.Namespace) -> None:
    receipt_dir = args.receipt_dir
    table = _format_result_table(payload)
    candidate_lines = []
    for spec in _candidate_specs().values():
        candidate_lines.append(f"- `{spec.candidate_id}`: {spec.title}. Provenance: {spec.provenance}")
    fire_base = (
        ".venv/bin/python experiments/ddm_tk2_d1_runner.py "
        "--pair-start 0 --pair-count 600 --chunk-size 120 --resume --claim-scorer-slot "
        "--candidates "
    )
    fire_cmds = [
        fire_base + "c0_flat_paint",
        fire_base + "c1_v15_template_paint",
        fire_base + "c2_boundary_aa",
        "# C3 is intentionally not fireable until a real one-hot TR1 adapter/checkpoint shape is declared.",
    ]
    source = payload["source_verification"]
    receipt = f"""# ddm_tk2 D1 Harness Receipt

## Summary

Axis: `{payload['axis']}`

`score_claim=false`. This is a D1 harness/candidate smoke, not a contest score and not a frontier move.

R path: `{payload['r_path']}`.

{table}

## Artifact Boundary

tk1 selected a `{TK1_SELECTED_KT_BYTES}` byte KT/context-arith semantic stream by receipt, but did not persist a standalone full stream file. This runner verified the persisted tq1c source-label `.npy` file and GT `.npy` file by SHA-256, then ran a real subset coder round-trip before rendering.

- tq1c labels: `{source['semantic_labels_path']}`
- tq1c file sha256: `{source['semantic_labels_file_sha256']}` (expected match: `{source['semantic_labels_file_sha256_match']}`)
- GT labels: `{source['gt_labels_path']}`
- GT file sha256: `{source['gt_labels_file_sha256']}` (expected match: `{source['gt_labels_file_sha256_match']}`)
- subset coder proof: `{json.dumps(source['subset_real_coder_roundtrip'], sort_keys=True)}`

## Candidate Ladder

{chr(10).join(candidate_lines)}

`c3_tr1_onehot_retarget` is fail-closed here. The banked TR1 LOTTO checkpoint consumes its own token/code surface; no compatible one-hot class-plane adapter was found or trained in this arm.

## D1 Fire Order Packet

Claim the scorer slot first. Run one candidate at a time, chunked at 120 with `--resume`; do not run this while et/rw scorer work owns the slot.

```bash
{chr(10).join(fire_cmds)}
```

Expected wall clock from smoke scales roughly linearly from the observed per-candidate smoke wall seconds in `harness_smoke.json`; the exact n600 fire must record actual wall clock per chunk.

## RECALL EVIDENCE

- Read `.omx/tmp/codex_runs/tk2_prompt.md` and `_common_contract.md`.
- Read `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md`.
- Read tk1 receipts under `.omx/research/ddm_tk1_20260806/` and confirmed the selected 142,001-byte stream is a receipt-backed closed-form coder row, not a persisted full stream file.
- Searched/read through-R and frozen SegNet harness code in `experiments/train_witness_realized_through_R_mlx.py` and `src/tac/boundary_math/seg_core.py`.
- Read fp1/v14/v15/#149/TR1 provenance surfaces for the candidate ladder; C3 remained scoped negative because no real one-hot TR1 adapter surface was found in this arm.

## Frontier Honesty

Own-vehicle frontier remains the current pointer from `.omx/state/main_hot_state.md`: `S=0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest borrowed pointer remains unmoved. This tk2 artifact is means, not goal progress.
"""
    _atomic_write_text(receipt_dir / "RECEIPT.md", receipt)

    next_text = f"""# NEXT_IF_RESUMED

1. If the scorer slot is free, run C0, C1, then C2 with the D1 fire-order commands in `RECEIPT.md`.
2. Preserve `--chunk-size 120 --resume`; checkpoints are under `stage_checkpoints/<candidate>/`.
3. Do not fire C3 until a real one-hot TR1 adapter/checkpoint shape is declared. A zero/identity placeholder is not a C3 measurement.
4. After any n600 D1 run, update `RECEIPT.md` with the measured per-class/total d_seg and keep `score_claim=false` unless an exact archive is evaluated by `upstream/evaluate.py`.

Current smoke output: `{receipt_dir / 'harness_smoke.json'}`
"""
    _atomic_write_text(receipt_dir / "NEXT_IF_RESUMED.md", next_text)

    checkpoints = f"""# CHECKPOINTS

- Runner: `experiments/ddm_tk2_d1_runner.py`
- Smoke JSON: `.omx/research/ddm_tk2_20260806/harness_smoke.json`
- Chunk checkpoints: `.omx/research/ddm_tk2_20260806/stage_checkpoints/`
- SSD preflight receipt: `{args.ssd_dir / 'PREFLIGHT_WRITE_PROBE.json'}`
- Config sha256: `{payload['config_sha256']}`
- Written at UTC: `{payload['written_at_utc']}`
- Score claim: `false`
- Harness boundary: n<=4 smoke unless `--claim-scorer-slot` is explicitly supplied.
"""
    _atomic_write_text(receipt_dir / "CHECKPOINTS.md", checkpoints)


def _self_test() -> None:
    labels = np.zeros((2, SEG_H, SEG_W), dtype=np.int64)
    labels[:, 100:140, 100:160] = 1
    labels[:, 160:220, 220:300] = 3
    labels[:, 40:60, 40:60] = 2
    labels[:, 340:, :] = 4
    for cid in ("c0_flat_paint", "c1_v15_template_paint", "c2_boundary_aa"):
        spec = _candidate_specs()[cid]
        rgb = spec.renderer(labels)
        if rgb.shape != labels.shape + (3,):
            raise AssertionError(f"{cid} bad shape {rgb.shape}")
        if not np.isfinite(rgb).all():
            raise AssertionError(f"{cid} non-finite")
        if float(rgb.min()) < 0.0 or float(rgb.max()) > 255.0:
            raise AssertionError(f"{cid} out-of-range")
    try:
        _candidate_specs()["c3_tr1_onehot_retarget"].renderer(labels)
    except CandidateUnavailable:
        pass
    else:
        raise AssertionError("C3 must fail closed without a real adapter")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    p.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    p.add_argument("--semantic-labels", type=Path, default=DEFAULT_TQ1C_LABELS)
    p.add_argument("--gt-labels", type=Path, default=DEFAULT_GT_LABELS)
    p.add_argument("--pair-start", type=int, default=0)
    p.add_argument("--pair-count", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument(
        "--candidates",
        default="c0_flat_paint,c1_v15_template_paint,c2_boundary_aa,c3_tr1_onehot_retarget",
        help="comma-separated candidate ids",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--harness-smoke", action="store_true", help="label as n<=4 harness smoke")
    p.add_argument("--claim-scorer-slot", action="store_true", help="operator/lane has claimed the full scorer slot")
    p.add_argument("--write-receipts", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        print("self-test ok")
        return 0
    if not args.harness_smoke and not args.claim_scorer_slot:
        raise SystemExit("Use --harness-smoke for n<=4 smoke, or --claim-scorer-slot for queued D1.")
    payload = run_harness(args)
    if args.write_receipts:
        write_receipts(payload, args)
    print(json.dumps({"ok": True, "output_schema": payload["schema"], "candidate_results": payload["candidate_results"], "unavailable": payload["candidate_unavailable"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
