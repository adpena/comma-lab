#!/usr/bin/env python3
"""ddm_et2 -- projected static Q3 compose of the block16 phase field.

This is the fire-order-1 runner from .omx/tmp/codex_runs/et2_prompt.md.
It measures the projected rank-6 pose-null correction on the live tq1c parent,
using the canonical inflated parent raw as input.  It does not claim a contest
score and does not promote any pointer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "experiments", REPO / "upstream"):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from ddm_et1_ph1_block16_on_our_vehicle import price, solve_blocks, translate_blocks  # noqa: E402
from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_pose_null_constrained_paint import (  # noqa: E402
    project_null,
    snap_band_to_blocks,
    yuv6_shift,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (  # noqa: E402
    confusion,
    realize_scorer_paint_to_camera,
    resize_to_scorer,
    solve_margin_optimal_paint,
)
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402


AXIS = "[macOS-CPU frozen-scorer advisory]"
SCORE_CLAIM = False
DEN = 37_545_489
BASELINE_S = 0.7534578126155775
BASELINE_BYTES = 357_837
BASELINE_D_SEG = 0.004305419922
BASELINE_D_POSE = 0.000716508925
BASELINE_ARCHIVE_SHA256 = "b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06"
ET1_BASE_ARCHIVE_SHA256 = "c72ef357416b66e716b2863c4c49360306b80cc0fafd094e02394c8a4dd37209"
S_PER_FLIP = 100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)
RATE_PER_BYTE = 25.0 / DEN
POSE_KY = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as exc:  # pragma: no cover - provenance fallback only
        return f"UNKNOWN:{type(exc).__name__}:{exc}"


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=1, default=jsonable, allow_nan=False)
        fh.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=jsonable, allow_nan=False))
        fh.write("\n")


def score_from_components(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * float(d_seg) + math.sqrt(10.0 * float(d_pose)) + RATE_PER_BYTE * int(archive_bytes)


def pose_constraint_matrix_np() -> np.ndarray:
    """Return the 6x12 frame_1 yuv6 constraint matrix for one 2x2 scorer block."""
    a = np.zeros((6, 12), dtype=np.float64)
    for p in range(4):
        a[p, 3 * p : 3 * p + 3] = POSE_KY
    for p in range(4):
        a[4, 3 * p + 0] = 0.25
        a[5, 3 * p + 2] = 0.25
    return a


POSE_A = pose_constraint_matrix_np()


def pose_null_projector_np() -> np.ndarray:
    p = np.eye(12, dtype=np.float64) - np.linalg.pinv(POSE_A) @ POSE_A
    if not np.allclose(p @ p, p) or float(np.abs(POSE_A @ p).max()) >= 1e-10:
        raise RuntimeError("Euclidean pose projector failed algebraic checks")
    return p


def load_models(upstream_root: Path, *, threads: int) -> tuple[Any, Any, dict[str, Any]]:
    import modules as upstream_modules
    from safetensors.torch import load_file

    modules_path = upstream_root / "modules.py"
    if Path(upstream_modules.__file__).resolve() != modules_path.resolve():
        raise RuntimeError(f"imported wrong upstream modules.py: {upstream_modules.__file__}")
    torch.set_num_threads(int(threads))
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    segnet = upstream_modules.SegNet().eval().to("cpu")
    posenet = upstream_modules.PoseNet().eval().to("cpu")
    seg_path = Path(upstream_modules.segnet_sd_path).resolve()
    pose_path = Path(upstream_modules.posenet_sd_path).resolve()
    segnet.load_state_dict(load_file(str(seg_path), device="cpu"))
    posenet.load_state_dict(load_file(str(pose_path), device="cpu"))
    for model in (segnet, posenet):
        for param in model.parameters():
            param.requires_grad_(False)
    custody = {
        "modules_path": str(modules_path),
        "modules_sha256": sha256_file(modules_path),
        "segnet_weights_path": str(seg_path),
        "segnet_weights_sha256": sha256_file(seg_path),
        "posenet_weights_path": str(pose_path),
        "posenet_weights_sha256": sha256_file(pose_path),
        "device": "cpu",
        "threads": int(threads),
        "deterministic_algorithms": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
    }
    return segnet, posenet, custody


def forward(segnet: Any, posenet: Any, camera_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(camera_pairs)
    if value.dtype != np.uint8 or value.ndim != 5 or value.shape[1:] != (2, CAM_H, CAM_W, 3):
        raise RuntimeError(f"bad camera batch shape: dtype={value.dtype} shape={value.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        cells = segnet(segnet.preprocess_input(tensor)).argmax(dim=1).cpu().numpy().astype(np.uint8)
        pose_out = posenet(posenet.preprocess_input(tensor))
        pose = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        pose6 = pose[:, :6].cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(cells), np.ascontiguousarray(pose6)


def raw_memmap(path: Path) -> np.memmap:
    expected = N_PAIRS_TOTAL * seq_len * CAM_H * CAM_W * 3
    got = path.stat().st_size
    if got != expected:
        raise RuntimeError(f"parent raw size drift: {got} != {expected}")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))


def score_parent_and_cache(
    *,
    args: argparse.Namespace,
    raw: np.memmap,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    scorer_custody: dict[str, Any],
) -> dict[str, Any]:
    stage = args.bulk_dir / "parent_score"
    summary_path = stage / "aggregate.json"
    lstar_path = stage / "parent_tq1c_argmax_n600.npy"
    pose_path = stage / "parent_tq1c_pose6_n600.npy"
    if summary_path.exists() and lstar_path.exists() and pose_path.exists():
        return json.loads(summary_path.read_text())

    stage.mkdir(parents=True, exist_ok=True)
    lstars = np.lib.format.open_memmap(lstar_path, mode="w+", dtype=np.uint8, shape=(N_PAIRS_TOTAL, SEG_H, SEG_W))
    pose6_all = np.lib.format.open_memmap(pose_path, mode="w+", dtype=np.float64, shape=(N_PAIRS_TOTAL, 6))
    batch_rows: list[dict[str, Any]] = []
    started = time.time()
    for start in range(0, N_PAIRS_TOTAL, args.scorer_batch_size):
        stop = min(start + args.scorer_batch_size, N_PAIRS_TOTAL)
        checkpoint = stage / f"batch_{start:04d}_{stop:04d}.json"
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text())
            batch_rows.append(row)
            continue
        camera = np.asarray(raw[start * seq_len : stop * seq_len]).reshape(stop - start, seq_len, CAM_H, CAM_W, 3)
        cells, pose6 = forward(segnet, posenet, camera)
        lstars[start:stop] = cells
        pose6_all[start:stop] = pose6
        target = np.asarray(labels[start:stop], dtype=np.uint8)
        target_pose = np.asarray(poses[start:stop], dtype=np.float64)
        errors = cells != target
        row = {
            "schema": "ddm_et2_parent_score_batch.v1",
            "pair_range": [start, stop],
            "errors": int(np.count_nonzero(errors)),
            "sites": int(errors.size),
            "pose_squared_error_sum": f"{float(np.square(pose6 - target_pose).sum(dtype=np.float64)):.12f}",
            "pose_coordinates": int(pose6.size),
            "cells_sha256": hashlib.sha256(np.ascontiguousarray(cells).tobytes()).hexdigest(),
            "pose6_sha256": hashlib.sha256(np.ascontiguousarray(pose6).tobytes()).hexdigest(),
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
        }
        write_json_atomic(checkpoint, row)
        batch_rows.append(row)
        print(f"[et2] parent score batch {start:04d}:{stop:04d} errors={row['errors']}", flush=True)
    lstars.flush()
    pose6_all.flush()
    errors = sum(int(r["errors"]) for r in batch_rows)
    sites = sum(int(r["sites"]) for r in batch_rows)
    pose_sse = sum(float(r["pose_squared_error_sum"]) for r in batch_rows)
    pose_coords = sum(int(r["pose_coordinates"]) for r in batch_rows)
    d_seg = errors / sites
    d_pose = pose_sse / pose_coords
    aggregate = {
        "schema": "ddm_et2_parent_tq1c_score.v1",
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "archive_path": str(args.parent_archive),
        "archive_sha256": sha256_file(args.parent_archive),
        "archive_bytes": int(args.parent_archive.stat().st_size),
        "parent_raw_path": str(args.parent_raw),
        "parent_raw_bytes": int(args.parent_raw.stat().st_size),
        "parent_raw_sha256": sha256_file(args.parent_raw),
        "lstars_path": str(lstar_path),
        "lstars_sha256": sha256_file(lstar_path),
        "pose6_path": str(pose_path),
        "pose6_sha256": sha256_file(pose_path),
        "d_seg": f"{d_seg:.12f}",
        "d_pose": f"{d_pose:.12f}",
        "score": f"{score_from_components(d_seg, d_pose, int(args.parent_archive.stat().st_size)):.12f}",
        "expected_neighborhood": {
            "S": BASELINE_S,
            "d_seg": BASELINE_D_SEG,
            "d_pose": BASELINE_D_POSE,
            "archive_bytes": BASELINE_BYTES,
        },
        "delta_vs_expected_neighborhood": {
            "delta_S": score_from_components(d_seg, d_pose, int(args.parent_archive.stat().st_size)) - BASELINE_S,
            "delta_d_seg": d_seg - BASELINE_D_SEG,
            "delta_d_pose": d_pose - BASELINE_D_POSE,
            "delta_bytes": int(args.parent_archive.stat().st_size) - BASELINE_BYTES,
        },
        "errors": errors,
        "sites": sites,
        "pose_squared_error_sum": f"{pose_sse:.12f}",
        "pose_coordinates": pose_coords,
        "batch_count": len(batch_rows),
        "batch_size": int(args.scorer_batch_size),
        "seconds": time.time() - started,
        "scorer_custody": scorer_custody,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
    }
    write_json_atomic(summary_path, aggregate)
    return aggregate


def field_summary_for_lstars(
    *,
    name: str,
    lstars: np.ndarray,
    gt_labels: np.ndarray,
    offsets_path: Path,
    block: int,
    rmax: int,
) -> dict[str, Any]:
    offsets_path.parent.mkdir(parents=True, exist_ok=True)
    if offsets_path.exists():
        offsets = np.load(offsets_path, mmap_mode="r")
    else:
        offsets = np.lib.format.open_memmap(
            offsets_path,
            mode="w+",
            dtype=np.int8,
            shape=(N_PAIRS_TOTAL, (SEG_H // block) * (SEG_W // block), 2),
        )
        for pair in range(N_PAIRS_TOTAL):
            off = solve_blocks(np.asarray(lstars[pair]), np.asarray(gt_labels[pair]), block, rmax)
            offsets[pair] = off.reshape(-1, 2)
            if (pair + 1) % 50 == 0:
                print(f"[et2] solved {name} block offsets {pair + 1}/600", flush=True)
        offsets.flush()

    base_flips = 0
    left = 0
    fixed = 0
    broken = 0
    band_px = 0
    for pair in range(N_PAIRS_TOTAL):
        cur = np.asarray(lstars[pair])
        gt = np.asarray(gt_labels[pair])
        target = translate_blocks(cur, np.asarray(offsets[pair]), block)
        flips0 = cur != gt
        after = target != gt
        base_flips += int(flips0.sum())
        left += int(after.sum())
        fixed += int((flips0 & ~after).sum())
        broken += int(((~flips0) & after).sum())
        band_px += int((target != cur).sum())

    pr = price(np.asarray(offsets))
    gross = (base_flips - left) * S_PER_FLIP
    projected_bytes = int(pr["smevr_projected_bytes"])
    rate = projected_bytes * RATE_PER_BYTE
    return {
        "schema": "ddm_et2_block16_phase_field_summary.v1",
        "name": name,
        "block": block,
        "rmax": rmax,
        "offsets_path": str(offsets_path),
        "offsets_sha256": sha256_file(offsets_path),
        "offset_shape": list(np.asarray(offsets).shape),
        "base_flips": base_flips,
        "flips_left_if_exact": left,
        "label_ceiling_net_fixed": base_flips - left,
        "label_ceiling_fixed": fixed,
        "label_ceiling_broken": broken,
        "reach": (base_flips - left) / base_flips if base_flips else None,
        "band_px": band_px,
        "band_frac": band_px / float(N_PAIRS_TOTAL * SEG_H * SEG_W),
        "gross_S": gross,
        "price": pr,
        "rate_S_at_projected_bytes": rate,
        "breakeven_eta": rate / gross if gross else None,
        "axis": "[macOS-CPU cache-derived advisory]",
        "score_claim": SCORE_CLAIM,
    }


def compare_fields(
    *,
    current_lstars: np.ndarray,
    old_lstars: np.ndarray,
    current_offsets: np.ndarray,
    old_offsets: np.ndarray,
    gt_labels: np.ndarray,
    block: int,
) -> dict[str, Any]:
    argmax_changed = 0
    target_changed = 0
    current_target_fixed_not_old = 0
    old_target_fixed_not_current = 0
    for pair in range(N_PAIRS_TOTAL):
        cur = np.asarray(current_lstars[pair])
        old = np.asarray(old_lstars[pair])
        gt = np.asarray(gt_labels[pair])
        ct = translate_blocks(cur, np.asarray(current_offsets[pair]), block)
        ot = translate_blocks(old, np.asarray(old_offsets[pair]), block)
        argmax_changed += int((cur != old).sum())
        target_changed += int((ct != ot).sum())
        current_good = ct == gt
        old_good = ot == gt
        current_target_fixed_not_old += int((current_good & ~old_good).sum())
        old_target_fixed_not_current += int((old_good & ~current_good).sum())
    block_changed = int(np.count_nonzero(np.any(np.asarray(current_offsets) != np.asarray(old_offsets), axis=2)))
    total_blocks = int(np.asarray(current_offsets).shape[0] * np.asarray(current_offsets).shape[1])
    return {
        "schema": "ddm_et2_field_delta_vs_20260803.v1",
        "old_base_archive_sha256": ET1_BASE_ARCHIVE_SHA256,
        "current_base_archive_sha256": BASELINE_ARCHIVE_SHA256,
        "argmax_changed_cells": argmax_changed,
        "argmax_changed_frac": argmax_changed / float(N_PAIRS_TOTAL * SEG_H * SEG_W),
        "offset_blocks_changed": block_changed,
        "offset_blocks_total": total_blocks,
        "offset_blocks_changed_frac": block_changed / float(total_blocks),
        "translated_target_changed_cells": target_changed,
        "translated_target_changed_frac": target_changed / float(N_PAIRS_TOTAL * SEG_H * SEG_W),
        "current_target_correct_old_not": current_target_fixed_not_old,
        "old_target_correct_current_not": old_target_fixed_not_current,
        "net_target_correct_delta_vs_old": current_target_fixed_not_old - old_target_fixed_not_current,
    }


def project_paint_to_q3(base: torch.Tensor, paint_hwc: np.ndarray, band: np.ndarray, projector: torch.Tensor) -> np.ndarray:
    paint = torch.from_numpy(np.ascontiguousarray(paint_hwc)).permute(2, 0, 1)[None].float()
    mask = torch.from_numpy(band.astype(bool))[None, None].float()
    delta = (paint - base) * mask
    projected = project_null(delta, projector)
    cur = torch.clamp(base + projected, 0.0, 255.0)
    return torch.round(cur)[0].permute(1, 2, 0).numpy().astype(np.uint8)


def seg_target_saliency_hwc(
    segnet: Any,
    base: torch.Tensor,
    target: np.ndarray,
    band: np.ndarray,
) -> np.ndarray:
    """Current-pair diagonal seg metric source for Arm M.

    The charter asked for banked margin-saliency producers when available. The
    only located per-pixel cache is a 50-pair prefix, so Arm M uses the measured
    current-pair CE gradient on the same phase-field target/band instead of
    importing a partial-population map.
    """
    mask_np = snap_band_to_blocks(band).astype(np.float32)
    if not bool(mask_np.any()):
        return np.zeros((SEG_H, SEG_W, 3), dtype=np.float64)
    tgt = torch.from_numpy(np.asarray(target, dtype=np.int64))[None]
    mask = torch.from_numpy(mask_np)[None]
    with torch.enable_grad():
        x = base.detach().clone().requires_grad_(True)
        logits = segnet(x)
        loss_map = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
        loss = (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
        segnet.zero_grad(set_to_none=True)
        loss.backward()
        grad = x.grad.detach().abs()[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(grad)


def metric_project_block(
    delta12: np.ndarray,
    saliency12: np.ndarray,
    *,
    lambda_frac: float,
    lambda_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    spectrum = np.square(np.asarray(saliency12, dtype=np.float64))
    positive = spectrum[spectrum > 0.0]
    anchor = float(np.median(positive)) if positive.size else float(spectrum.max(initial=0.0))
    lam = max(float(lambda_floor), float(lambda_frac) * max(anchor, float(spectrum.max(initial=0.0)), 0.0))
    m_diag = spectrum + lam
    inv_m = 1.0 / m_diag
    if not np.isfinite(inv_m).all():
        raise RuntimeError("non-finite metric inverse diagonal in Arm M projector")
    amat = (POSE_A * inv_m[None, :]) @ POSE_A.T
    bridge = np.linalg.solve(amat, POSE_A)
    left = np.multiply(POSE_A.T, inv_m[:, None])
    p = np.eye(12, dtype=np.float64) - np.einsum("ij,jk->ik", left, bridge)
    if not np.isfinite(p).all():
        raise RuntimeError("non-finite Arm M projector matrix")
    projected = p @ np.asarray(delta12, dtype=np.float64)
    return projected, {
        "lambda": lam,
        "spectrum_min": float(spectrum.min(initial=0.0)),
        "spectrum_median": float(np.median(spectrum)),
        "spectrum_max": float(spectrum.max(initial=0.0)),
        "max_abs_A_P": float(np.abs(POSE_A @ p).max()),
        "max_abs_P2_minus_P": float(np.abs(p @ p - p).max()),
    }


def project_paint_to_q3_metric(
    base: torch.Tensor,
    paint_hwc: np.ndarray,
    band: np.ndarray,
    saliency_hwc: np.ndarray,
    *,
    lambda_frac: float,
    lambda_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_hwc = base.detach()[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    paint = np.asarray(paint_hwc, dtype=np.float64)
    mask = band.astype(bool)
    delta = (paint - base_hwc) * mask[..., None]
    out_delta = np.zeros_like(delta, dtype=np.float64)
    block_mask = mask.reshape(SEG_H // 2, 2, SEG_W // 2, 2).any(axis=(1, 3))
    stats: list[dict[str, float]] = []
    for by, bx in np.argwhere(block_mask):
        y = int(by) * 2
        x = int(bx) * 2
        d12 = delta[y : y + 2, x : x + 2].reshape(12)
        s12 = np.asarray(saliency_hwc[y : y + 2, x : x + 2], dtype=np.float64).reshape(12)
        projected, diag = metric_project_block(
            d12,
            s12,
            lambda_frac=lambda_frac,
            lambda_floor=lambda_floor,
        )
        out_delta[y : y + 2, x : x + 2] = projected.reshape(2, 2, 3)
        stats.append(diag)
    cur = np.clip(base_hwc + out_delta, 0.0, 255.0)
    lambdas = np.asarray([s["lambda"] for s in stats], dtype=np.float64)
    spectrum_max = np.asarray([s["spectrum_max"] for s in stats], dtype=np.float64)
    diag = {
        "metric_source": "current_pair_segnet_ce_gradient_on_phase_field_target",
        "metric_form": "diagonal_M_equals_grad_abs_squared_plus_lambda",
        "lambda_frac": float(lambda_frac),
        "lambda_floor": float(lambda_floor),
        "blocks_projected": int(len(stats)),
        "lambda_min": float(lambdas.min()) if lambdas.size else None,
        "lambda_median": float(np.median(lambdas)) if lambdas.size else None,
        "lambda_max": float(lambdas.max()) if lambdas.size else None,
        "spectrum_max_median": float(np.median(spectrum_max)) if spectrum_max.size else None,
        "max_abs_A_P": float(max((s["max_abs_A_P"] for s in stats), default=0.0)),
        "max_abs_P2_minus_P": float(max((s["max_abs_P2_minus_P"] for s in stats), default=0.0)),
    }
    return np.round(cur).astype(np.uint8), diag


def solve_margin_wrapper(*args: Any, **kwargs: Any) -> tuple[int, np.ndarray, str, dict[str, Any] | None]:
    result = solve_margin_optimal_paint(*args, **kwargs)
    if len(result) == 3:
        nbad, paint, tag = result
        return int(nbad), paint, str(tag), None
    if len(result) == 4:
        nbad, paint, tag, diag = result
        return int(nbad), paint, str(tag), diag
    raise RuntimeError(f"unexpected solve_margin_optimal_paint return arity {len(result)}")


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_fire_order_1(
    rows: list[dict[str, Any]],
    *,
    arm_id: str,
    projector_mode: str,
    bar: float,
    parent_d_pose: float,
    parent_errors: int,
    pose_leakage_ratio: float,
) -> dict[str, Any]:
    before = int(sum(r["flips_before"] for r in rows))
    after = int(sum(r["projected"]["flips_after"] for r in rows))
    denom = int(sum(r["label_ceiling_net_fixed"] for r in rows))
    eta = (before - after) / denom if denom else None
    ratios = np.array([r["projected"]["d_pose_ratio"] for r in rows], dtype=np.float64)
    pose_sse_delta = float(sum(r["projected"]["pose_sse_delta"] for r in rows))
    seg_delta = ((after - before) * S_PER_FLIP) if rows else 0.0
    pose_delta = math.sqrt(10.0 * (parent_d_pose + pose_sse_delta / (N_PAIRS_TOTAL * 6))) - math.sqrt(10.0 * parent_d_pose)
    pose_pass = bool(ratios.size and float(ratios.max()) <= pose_leakage_ratio)
    eta_pass = bool(eta is not None and eta > bar)
    if not eta_pass:
        verdict = "FOLDED_FORMULATION_PROJECTED_STATIC_ETA_NOT_ABOVE_BAR"
    elif not pose_pass:
        verdict = "POSE_NEUTRALITY_FAILED_FALLBACK_ONLY_DO_NOT_FIRE_FULL_N600"
    else:
        verdict = "GREEN_SURVIVES_FIRE_ORDER_1_FIRE_ORDER_2_ALLOWED"
    accepted_fallback = [
        r["pair"]
        for r in rows
        if r["projected"]["joint_delta_S_no_rate_against_parent_pose"] < 0.0
    ]
    return {
        "schema": "ddm_et2_fire_order_1_aggregate.v1",
        "arm_id": arm_id,
        "projector_mode": projector_mode,
        "n_rows": len(rows),
        "flips_before_subset": before,
        "projected_flips_after_subset": after,
        "projected_net_flip_reduction_subset": before - after,
        "label_ceiling_net_fixed_subset": denom,
        "projected_eta_realized": eta,
        "breakeven_eta_bar": bar,
        "eta_over_bar": (eta / bar) if eta is not None and bar else None,
        "projected_eta_clears_bar": eta_pass,
        "pose_ratio_min": float(ratios.min()) if ratios.size else None,
        "pose_ratio_p25": float(np.quantile(ratios, 0.25)) if ratios.size else None,
        "pose_ratio_median": float(np.median(ratios)) if ratios.size else None,
        "pose_ratio_p75": float(np.quantile(ratios, 0.75)) if ratios.size else None,
        "pose_ratio_max": float(ratios.max()) if ratios.size else None,
        "pose_ratio_mean": float(ratios.mean()) if ratios.size else None,
        "pose_leakage_ratio_threshold": pose_leakage_ratio,
        "pose_neutrality_pass": pose_pass,
        "subset_projected_seg_delta_S_no_rate": seg_delta,
        "subset_projected_pose_delta_S_against_parent": pose_delta,
        "subset_projected_joint_delta_S_no_rate_against_parent": seg_delta + pose_delta,
        "parent_errors": parent_errors,
        "fallback_per_pair_joint_acceptance_no_rate": {
            "mode": "SEPARATE_LABELLED_FALLBACK_ONLY_NOT_BLENDED",
            "accepted_pairs": accepted_fallback,
            "accepted_count": len(accepted_fallback),
            "rejected_count": len(rows) - len(accepted_fallback),
        },
        "verdict": verdict,
        "verdict_scope": "FORMULATION: projected-static rank-6 Q3 correction on re-solved block16 phase field, n32 advisory gate",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
    }


def run_fire_order_1(
    *,
    args: argparse.Namespace,
    arm_id: str,
    projector_mode: str,
    raw: np.memmap,
    gt_labels: np.ndarray,
    parent_lstars: np.ndarray,
    current_offsets: np.ndarray,
    segnet: Any,
    posenet: Any,
    parent_score: dict[str, Any],
    field_current: dict[str, Any],
) -> dict[str, Any]:
    if arm_id == "E":
        rows_path = args.bulk_dir / "fire_order_1_projected_rows.jsonl"
        summary_path = args.bulk_dir / "fire_order_1_projected_summary.json"
    else:
        rows_path = args.bulk_dir / f"fire_order_1_{arm_id.lower()}_projected_rows.jsonl"
        summary_path = args.bulk_dir / f"fire_order_1_{arm_id.lower()}_projected_summary.json"
    pairs_all = json.loads(args.et1_n32_json.read_text())["pairs"]
    pairs = [int(p) for p in pairs_all[: args.limit]] if args.limit else [int(p) for p in pairs_all]
    existing = load_existing_rows(rows_path) if args.resume else []
    done = {int(row["pair"]) for row in existing}
    todo = [p for p in pairs if p not in done]
    wanted = set()
    for pair in todo:
        wanted.update({seq_len * pair, seq_len * pair + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted) if wanted else {}
    projector = torch.from_numpy(pose_null_projector_np()).float()
    rows = list(existing)
    parent_d_pose = float(parent_score["d_pose"])
    parent_errors = int(parent_score["errors"])

    print(f"[et2] fire-order-1 arm={arm_id} rows={len(rows)} remaining={len(todo)}", flush=True)
    for idx, pair in enumerate(todo, start=1):
        started = time.time()
        dec = np.stack([raw[seq_len * pair], raw[seq_len * pair + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * pair], gt_frames[seq_len * pair + 1]]).astype(np.uint8)
        cells, pose_base = forward(segnet, posenet, dec[None])
        lstar = cells[0]
        pose_gt_cells, pose_gt = forward(segnet, posenet, gt[None])
        lgt = pose_gt_cells[0]
        cached_parent = np.asarray(parent_lstars[pair])
        cached_gt = np.asarray(gt_labels[pair], dtype=np.uint8)
        if not np.array_equal(lstar, cached_parent):
            raise RuntimeError(f"C2 failed for pair {pair}: decoded parent argmax != cached parent")
        if not np.array_equal(lgt, cached_gt):
            raise RuntimeError(f"C3 failed for pair {pair}: canonical GT decode argmax != gt cache")

        off = np.asarray(current_offsets[pair])
        target = translate_blocks(lstar, off, args.block)
        band = target != lstar
        flips0_map = lstar != lgt
        flips0 = int(flips0_map.sum())
        label_after = target != lgt
        label_ceiling_net_fixed = flips0 - int(label_after.sum())
        d_pose_before = float(np.square(pose_base[0] - pose_gt[0]).sum() / 6.0)

        nbad, paint, tag, solve_diag = solve_margin_wrapper(
            segnet,
            dec[1],
            gt[1],
            band,
            target,
            steps=args.steps,
            lr=args.lr,
            eval_every=args.eval_every,
            convergence_patience_evals=args.convergence_patience_evals,
            convergence_min_improvement=args.convergence_min_improvement,
        )
        base = resize_to_scorer(dec[1])
        band_snapped = snap_band_to_blocks(band)
        metric_diag = None
        if projector_mode == "euclidean":
            projected_paint = project_paint_to_q3(base, paint, band, projector)
        elif projector_mode == "seg_metric_diagonal":
            saliency = seg_target_saliency_hwc(segnet, base, target, band)
            projected_paint, metric_diag = project_paint_to_q3_metric(
                base,
                paint,
                band,
                saliency,
                lambda_frac=args.metric_lambda_frac,
                lambda_floor=args.metric_lambda_floor,
            )
        else:
            raise RuntimeError(f"unknown projector mode: {projector_mode}")
        cam_projected = realize_scorer_paint_to_camera(dec[1], band_snapped, projected_paint)
        pair_projected = np.stack([dec[0], cam_projected]).astype(np.uint8)
        proj_cells, proj_pose = forward(segnet, posenet, pair_projected[None])
        lam = proj_cells[0]
        flips_after = int((lam != lgt).sum())
        d_pose_after = float(np.square(proj_pose[0] - pose_gt[0]).sum() / 6.0)
        pose_sse_delta = float(np.square(proj_pose[0] - pose_gt[0]).sum() - np.square(pose_base[0] - pose_gt[0]).sum())
        seg_delta_pair = (flips_after - flips0) * S_PER_FLIP
        pose_delta_pair = math.sqrt(10.0 * (parent_d_pose + pose_sse_delta / (N_PAIRS_TOTAL * 6))) - math.sqrt(10.0 * parent_d_pose)
        base_sc = torch.round(base)[0].permute(1, 2, 0).numpy().astype(np.uint8)
        rec = {
            "schema": "ddm_et2_projected_phase_field_pair.v1",
            "arm_id": arm_id,
            "projector_mode": projector_mode,
            "pair": int(pair),
            "flips_before": flips0,
            "label_ceiling_flips_left": int(label_after.sum()),
            "label_ceiling_net_fixed": label_ceiling_net_fixed,
            "label_ceiling_fixed": int((flips0_map & ~label_after).sum()),
            "label_ceiling_broken": int(((~flips0_map) & label_after).sum()),
            "band_px": int(band.sum()),
            "band_snapped_px": int(band_snapped.sum()),
            "band_snap_tax": float(band_snapped.sum() / max(1, band.sum())),
            "controls": {
                "C2_parent_argmax_matches_cache": True,
                "C3_gt_argmax_matches_cache": True,
                "offset_shape": list(off.shape),
            },
            "unprojected_solve": {
                "proxy_flips_scorer_lattice": int(nbad),
                "tag": tag,
                "diagnostics": solve_diag,
            },
            "projected": {
                "flips_after": flips_after,
                "eta_realized": ((flips0 - flips_after) / label_ceiling_net_fixed) if label_ceiling_net_fixed else None,
                "fixed_global": int((flips0_map & (lam == lgt)).sum()),
                "introduced_global": int(((~flips0_map) & (lam != lgt)).sum()),
                "C_after": confusion(lgt, lam).tolist(),
                "d_pose_before": d_pose_before,
                "d_pose_after": d_pose_after,
                "d_pose_ratio": d_pose_after / d_pose_before if d_pose_before else None,
                "pose_sse_delta": pose_sse_delta,
                "seg_delta_S_no_rate": seg_delta_pair,
                "pose_delta_S_against_parent": pose_delta_pair,
                "joint_delta_S_no_rate_against_parent_pose": seg_delta_pair + pose_delta_pair,
                "changed_scorer_pixels": int((projected_paint != base_sc).any(axis=2).sum()),
                "changed_scorer_channel_values": int((projected_paint != base_sc).sum()),
                "yuv6_residual": yuv6_shift(base_sc, projected_paint),
            },
            "metric_projector": metric_diag,
            "elapsed_s": time.time() - started,
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
        }
        rows.append(rec)
        append_jsonl(rows_path, rec)
        aggregate = aggregate_fire_order_1(
            rows,
            arm_id=arm_id,
            projector_mode=projector_mode,
            bar=float(field_current["breakeven_eta"]),
            parent_d_pose=parent_d_pose,
            parent_errors=parent_errors,
            pose_leakage_ratio=args.pose_leakage_ratio,
        )
        payload = {
            "schema": "ddm_et2_fire_order_1_summary.v1",
            "captured_at_utc": utc_now(),
            "git": git_head(),
            "pairs": pairs,
            "parent_score": parent_score,
            "field_current": field_current,
            "solver": {
                "steps": args.steps,
                "lr": args.lr,
                "eval_every": args.eval_every,
                "convergence_patience_evals": args.convergence_patience_evals,
                "convergence_min_improvement": args.convergence_min_improvement,
                "arm_id": arm_id,
                "projector": "rank-6 frame_1 yuv6 null per 2x2 scorer block",
                "projection_mode": projector_mode,
                "metric_lambda_frac": args.metric_lambda_frac,
                "metric_lambda_floor": args.metric_lambda_floor,
            },
            "aggregate": aggregate,
            "rows": rows,
        }
        write_json_atomic(summary_path, payload)
        print(
            f"[et2] pair {pair:3d} ({len(rows)}/{len(pairs)}) "
            f"arm={arm_id} "
            f"eta={rec['projected']['eta_realized']:+.4f} "
            f"pose={rec['projected']['d_pose_ratio']:.4f}x "
            f"agg_eta={aggregate['projected_eta_realized']} verdict={aggregate['verdict']} "
            f"[{time.time()-started:.1f}s]",
            flush=True,
        )

    if summary_path.exists():
        return json.loads(summary_path.read_text())
    aggregate = aggregate_fire_order_1(
        rows,
        arm_id=arm_id,
        projector_mode=projector_mode,
        bar=float(field_current["breakeven_eta"]),
        parent_d_pose=parent_d_pose,
        parent_errors=parent_errors,
        pose_leakage_ratio=args.pose_leakage_ratio,
    )
    payload = {
        "schema": "ddm_et2_fire_order_1_summary.v1",
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "pairs": pairs,
        "parent_score": parent_score,
        "field_current": field_current,
        "solver": {
            "steps": args.steps,
            "lr": args.lr,
            "eval_every": args.eval_every,
            "convergence_patience_evals": args.convergence_patience_evals,
            "convergence_min_improvement": args.convergence_min_improvement,
            "arm_id": arm_id,
            "projector": "rank-6 frame_1 yuv6 null per 2x2 scorer block",
            "projection_mode": projector_mode,
            "metric_lambda_frac": args.metric_lambda_frac,
            "metric_lambda_floor": args.metric_lambda_floor,
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    write_json_atomic(summary_path, payload)
    return payload


def choose_fire_order_1_winner(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aggregates = {arm: payload["aggregate"] for arm, payload in arms.items()}
    pose_ok = {
        arm: agg
        for arm, agg in aggregates.items()
        if bool(agg.get("pose_neutrality_pass")) and agg.get("projected_eta_realized") is not None
    }
    candidates = pose_ok or {
        arm: agg
        for arm, agg in aggregates.items()
        if agg.get("projected_eta_realized") is not None
    }
    if not candidates:
        return {
            "schema": "ddm_et2_fire_order_1_ab_verdict.v1",
            "winner_arm": None,
            "winner_projector_mode": None,
            "winner_verdict": "NO_MEASURABLE_ARM_ROWS",
            "fire_order_2_allowed": False,
            "arms": aggregates,
        }
    winner_arm, winner = max(
        candidates.items(),
        key=lambda kv: float(kv[1].get("projected_eta_realized") or -math.inf),
    )
    winner_projector_mode = winner.get("projector_mode")
    if winner_projector_mode is None and winner_arm == "E":
        winner_projector_mode = "euclidean"
    return {
        "schema": "ddm_et2_fire_order_1_ab_verdict.v1",
        "winner_arm": winner_arm,
        "winner_projector_mode": winner_projector_mode,
        "winner_verdict": winner.get("verdict"),
        "fire_order_2_allowed": winner.get("verdict") == "GREEN_SURVIVES_FIRE_ORDER_1_FIRE_ORDER_2_ALLOWED",
        "selection_rule": "highest projected_eta_realized among pose-neutral arms; fallback to highest eta if none pose-neutral",
        "arms": aggregates,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
    }


def execute(args: argparse.Namespace) -> int:
    if args.scorer_batch_size > args.max_chunk_pairs:
        raise RuntimeError("scorer batch size exceeds common-contract chunk cap")
    args.bulk_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_memmap(args.parent_raw)
    labels = open_stored_npy_memmap(args.gt_cache, "lstars")
    poses = open_stored_npy_memmap(args.gt_cache, "gt_poses")
    if labels.shape != (N_PAIRS_TOTAL, SEG_H, SEG_W) or poses.shape != (N_PAIRS_TOTAL, 6):
        raise RuntimeError(f"GT cache shape drift: labels={labels.shape} poses={poses.shape}")
    segnet, posenet, scorer_custody = load_models(args.upstream_root, threads=args.threads)
    parent_score = score_parent_and_cache(
        args=args,
        raw=raw,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        scorer_custody=scorer_custody,
    )
    parent_lstars = np.load(args.bulk_dir / "parent_score" / "parent_tq1c_argmax_n600.npy", mmap_mode="r")
    old_lstars = np.load(args.old_argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gt_labels = np.asarray(labels)

    current_field = field_summary_for_lstars(
        name="tq1c_parent_b35e7568",
        lstars=parent_lstars,
        gt_labels=gt_labels,
        offsets_path=args.bulk_dir / "phase_field" / "tq1c_block16_offsets.npy",
        block=args.block,
        rmax=args.rmax,
    )
    old_field = field_summary_for_lstars(
        name="et1_20260803_base_cx1",
        lstars=old_lstars,
        gt_labels=gt_labels,
        offsets_path=args.bulk_dir / "phase_field" / "et1_base_block16_offsets.npy",
        block=args.block,
        rmax=args.rmax,
    )
    current_offsets = np.load(args.bulk_dir / "phase_field" / "tq1c_block16_offsets.npy", mmap_mode="r")
    old_offsets = np.load(args.bulk_dir / "phase_field" / "et1_base_block16_offsets.npy", mmap_mode="r")
    field_delta = compare_fields(
        current_lstars=parent_lstars,
        old_lstars=old_lstars,
        current_offsets=current_offsets,
        old_offsets=old_offsets,
        gt_labels=gt_labels,
        block=args.block,
    )
    field_payload = {
        "schema": "ddm_et2_phase_field_rederive.v1",
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "current": current_field,
        "old_20260803": old_field,
        "delta": field_delta,
        "axis": "[macOS-CPU cache-derived advisory]",
        "score_claim": SCORE_CLAIM,
    }
    write_json_atomic(args.bulk_dir / "phase_field" / "phase_field_rederive_summary.json", field_payload)
    fire_e = run_fire_order_1(
        args=args,
        arm_id="E",
        projector_mode="euclidean",
        raw=raw,
        gt_labels=gt_labels,
        parent_lstars=parent_lstars,
        current_offsets=current_offsets,
        segnet=segnet,
        posenet=posenet,
        parent_score=parent_score,
        field_current=current_field,
    )
    fire_m = run_fire_order_1(
        args=args,
        arm_id="M",
        projector_mode="seg_metric_diagonal",
        raw=raw,
        gt_labels=gt_labels,
        parent_lstars=parent_lstars,
        current_offsets=current_offsets,
        segnet=segnet,
        posenet=posenet,
        parent_score=parent_score,
        field_current=current_field,
    )
    fire_ab = choose_fire_order_1_winner({"E": fire_e, "M": fire_m})
    final = {
        "schema": "ddm_et2_projected_phase_field_final.v1",
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "parent_score": parent_score,
        "phase_field": field_payload,
        "fire_order_1": fire_ab,
        "fire_order_1_arms": {
            "E": {
                "summary_path": str(args.bulk_dir / "fire_order_1_projected_summary.json"),
                "rows_path": str(args.bulk_dir / "fire_order_1_projected_rows.jsonl"),
            },
            "M": {
                "summary_path": str(args.bulk_dir / "fire_order_1_m_projected_summary.json"),
                "rows_path": str(args.bulk_dir / "fire_order_1_m_projected_rows.jsonl"),
            },
        },
        "fire_order_2": {
            "status": (
                "QUEUED_BY_FIRE_ORDER_1_GREEN"
                if fire_ab["fire_order_2_allowed"]
                else "NOT_FIRED_BECAUSE_FIRE_ORDER_1_NOT_GREEN"
            ),
            "winner_arm": fire_ab["winner_arm"],
            "note": "Full n600 byte-close is gated by the fire-order-1 A/B verdict and is not performed by this runner.",
        },
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
    }
    write_json_atomic(args.bulk_dir / "et2_projected_phase_field_final.json", final)
    write_json_atomic(args.receipt_dir / "et2_projected_phase_field_final.json", final)
    print(json.dumps(final["fire_order_1"], indent=1, default=jsonable), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-raw", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/submission/inflated/0.raw"))
    ap.add_argument("--parent-archive", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes"))
    ap.add_argument("--gt-cache", type=Path, default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--old-argmax-cache", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"))
    ap.add_argument("--gt-mkv", type=Path, default=REPO / "upstream/videos/0.mkv")
    ap.add_argument("--upstream-root", type=Path, default=REPO / "upstream")
    ap.add_argument("--et1-n32-json", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et1_20260803/et1_b16_realization_n32.json"))
    ap.add_argument("--bulk-dir", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806"))
    ap.add_argument("--receipt-dir", type=Path, default=REPO / ".omx/research/ddm_et2_20260806")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--scorer-batch-size", type=int, default=16)
    ap.add_argument("--max-chunk-pairs", type=int, default=120)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--convergence-patience-evals", type=int, default=0)
    ap.add_argument("--convergence-min-improvement", type=int, default=1)
    ap.add_argument("--pose-leakage-ratio", type=float, default=1.04)
    ap.add_argument("--metric-lambda-frac", type=float, default=1.0e-3)
    ap.add_argument("--metric-lambda-floor", type=float, default=1.0e-12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
