#!/usr/bin/env python3
"""ddm_sw1 -- solve the block16 phase-field paint inside Q3 null coordinates.

Axis: [macOS-CPU frozen-scorer advisory]. score_claim=false.

This runner is deliberately bounded: it uses the live tq1c parent custody from
ddm_et2, runs only the small n<=8 validation requested by the charter by
default, and writes durable receipts under ddm_sw1 paths.  The solve-within arm
parameterizes each included 2x2 scorer block as delta = N @ c, where N is an
orthonormal basis for ker(A).  No post-solve projection exists in that arm.
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
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from ddm_et1_ph1_block16_on_our_vehicle import translate_blocks  # noqa: E402
from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_pose_null_constrained_paint import snap_band_to_blocks, yuv6_shift  # noqa: E402
from ddm_sq1_stage_decomposition_and_solved_paint import (  # noqa: E402
    confusion,
    realize_scorer_paint_to_camera,
    resize_to_scorer,
)
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.margin_saliency_map import compute_margin_saliency_map  # noqa: E402


AXIS = "[macOS-CPU frozen-scorer advisory]"
SCORE_CLAIM = False
BASELINE_ARCHIVE_SHA256 = "b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06"
BASELINE_S = 0.7534578126155775
BASELINE_BYTES = 357_837
BASELINE_D_SEG = 0.004305419922
BASELINE_D_POSE = 0.000716508925
DEN = 37_545_489
RATE_PER_BYTE = 25.0 / DEN
S_PER_FLIP = 100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)
KY = (0.299, 0.587, 0.114)
MODES = (
    "unconstrained",
    "project_after_euclidean",
    "project_after_metric",
    "solve_within_null_basis",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as exc:  # pragma: no cover - provenance fallback only
        return f"UNKNOWN:{type(exc).__name__}:{exc}"


def sha256_file(path: Path, chunk_size: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def score_from_components(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * float(d_seg) + math.sqrt(10.0 * float(d_pose)) + RATE_PER_BYTE * int(archive_bytes)


def raw_memmap(path: Path) -> np.memmap:
    expected = N_PAIRS_TOTAL * seq_len * CAM_H * CAM_W * 3
    got = path.stat().st_size
    if got != expected:
        raise RuntimeError(f"parent raw size drift: {got} != {expected}")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))


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


def pose_constraint_matrix() -> np.ndarray:
    a = np.zeros((6, 12), dtype=np.float64)
    for p in range(4):
        a[p, 3 * p : 3 * p + 3] = KY
        a[4, 3 * p + 0] = 0.25
        a[5, 3 * p + 2] = 0.25
    return a


def null_coordinate_basis() -> tuple[np.ndarray, dict[str, Any]]:
    a = pose_constraint_matrix()
    _u, s, vh = np.linalg.svd(a, full_matrices=True)
    rank = int(np.sum(s > 1e-12))
    n = vh[rank:].T.copy()
    cert = {
        "schema": "ddm_sw1_null_coordinate_basis.v1",
        "basis_shape": list(n.shape),
        "constraint_shape": list(a.shape),
        "constraint_rank": rank,
        "singular_values": s.tolist(),
        "max_abs_A_times_N": float(np.abs(a @ n).max()),
        "max_abs_NtN_minus_I": float(np.abs(n.T @ n - np.eye(n.shape[1])).max()),
        "construction": "numpy.linalg.svd(A), N = Vh[rank:].T; vector order is 2x2 row-major RGB",
    }
    if n.shape != (12, 6):
        raise RuntimeError(f"null basis shape drift: {n.shape}")
    if cert["max_abs_A_times_N"] > 1e-10:
        raise RuntimeError(f"null basis is not in ker(A): {cert['max_abs_A_times_N']}")
    if cert["max_abs_NtN_minus_I"] > 1e-10:
        raise RuntimeError(f"null basis not orthonormal: {cert['max_abs_NtN_minus_I']}")
    return n, cert


def block_mask_from_band(band: np.ndarray) -> np.ndarray:
    if band.shape != (SEG_H, SEG_W):
        raise RuntimeError(f"bad scorer-grid band shape: {band.shape}")
    return band.reshape(SEG_H // 2, 2, SEG_W // 2, 2).any(axis=(1, 3))


def chw_to_block_vectors(delta: torch.Tensor) -> torch.Tensor:
    b, c, h, w = delta.shape
    if c != 3 or h % 2 or w % 2:
        raise RuntimeError(f"expected even BCHW RGB tensor, got {tuple(delta.shape)}")
    x = delta.reshape(b, c, h // 2, 2, w // 2, 2)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, h // 2, w // 2, 12)


def block_vectors_to_chw(blocks: torch.Tensor) -> torch.Tensor:
    b, h2, w2, d = blocks.shape
    if d != 12:
        raise RuntimeError(f"expected block vectors with 12 channels, got {tuple(blocks.shape)}")
    x = blocks.reshape(b, h2, w2, 2, 2, 3)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(b, 3, h2 * 2, w2 * 2)


def coeffs_to_delta_chw(coeffs: torch.Tensor, basis_t: torch.Tensor) -> torch.Tensor:
    blocks = coeffs @ basis_t.T
    return block_vectors_to_chw(blocks)


def coeffs_from_delta_euclidean(delta: torch.Tensor, basis_t: torch.Tensor) -> torch.Tensor:
    return chw_to_block_vectors(delta) @ basis_t


def project_delta_euclidean_chw(
    delta: torch.Tensor,
    block_mask_t: torch.Tensor,
    basis_t: torch.Tensor,
) -> torch.Tensor:
    coeffs = coeffs_from_delta_euclidean(delta, basis_t) * block_mask_t
    return coeffs_to_delta_chw(coeffs, basis_t)


def _hwc_to_block_vectors_np(delta_hwc: np.ndarray) -> np.ndarray:
    x = np.asarray(delta_hwc, dtype=np.float64)
    return x.reshape(SEG_H // 2, 2, SEG_W // 2, 2, 3).transpose(0, 2, 1, 3, 4).reshape(
        SEG_H // 2, SEG_W // 2, 12
    )


def _block_vectors_to_hwc_np(blocks: np.ndarray) -> np.ndarray:
    return blocks.reshape(SEG_H // 2, SEG_W // 2, 2, 2, 3).transpose(0, 2, 1, 3, 4).reshape(
        SEG_H, SEG_W, 3
    )


def project_delta_metric_hwc(
    delta_hwc: np.ndarray,
    block_mask: np.ndarray,
    metric_weights_hw: np.ndarray,
    basis_np: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    blocks = _hwc_to_block_vectors_np(delta_hwc)
    sal_blocks = metric_weights_hw.reshape(SEG_H // 2, 2, SEG_W // 2, 2).transpose(0, 2, 1, 3)
    weights = np.repeat(sal_blocks.reshape(SEG_H // 2, SEG_W // 2, 4), 3, axis=2)
    out = np.zeros_like(blocks)
    eye = np.eye(basis_np.shape[1], dtype=np.float64)
    coords: list[tuple[int, int]] = list(zip(*np.nonzero(block_mask), strict=False))
    for by, bx in coords:
        w = np.clip(weights[by, bx], 1e-6, None)
        n_w = basis_np * w[:, None]
        lhs = basis_np.T @ n_w + ridge * eye
        rhs = basis_np.T @ (w * blocks[by, bx])
        out[by, bx] = basis_np @ np.linalg.solve(lhs, rhs)
    return _block_vectors_to_hwc_np(out)


def metric_weights_from_saliency(
    saliency_hw: np.ndarray,
    snapped_band: np.ndarray,
    *,
    lambda_saliency: float,
    outside_weight: float,
    clip: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    sal = np.asarray(saliency_hw, dtype=np.float64)
    if sal.shape != (SEG_H, SEG_W):
        raise RuntimeError(f"bad saliency shape: {sal.shape}")
    inside = sal[snapped_band.astype(bool)]
    denom = float(np.median(inside)) if inside.size else float(np.median(sal))
    denom = max(denom, 1e-12)
    normalized = np.clip(sal / denom, 0.0, float(clip))
    weights = np.full((SEG_H, SEG_W), float(outside_weight), dtype=np.float32)
    weights[snapped_band.astype(bool)] = (1.0 + float(lambda_saliency) * normalized[snapped_band.astype(bool)]).astype(
        np.float32
    )
    stats = {
        "producer": "tac.margin_saliency_map.compute_margin_saliency_map",
        "metric": "diagonal margin-saliency G = diag(weight); weight=1+lambda*saliency/median_on_snapped_band",
        "lambda_saliency": float(lambda_saliency),
        "outside_weight": float(outside_weight),
        "normalization": "median saliency on snapped phase-target band",
        "clip": float(clip),
        "saliency_min": float(sal.min()) if sal.size else 0.0,
        "saliency_median_band": denom,
        "saliency_max": float(sal.max()) if sal.size else 0.0,
        "weight_min": float(weights.min()),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()),
    }
    return weights, stats


def evaluate_paint(
    *,
    segnet: Any,
    posenet: Any,
    dec_f0: np.ndarray,
    dec_f1: np.ndarray,
    pose_gt: np.ndarray,
    lgt: np.ndarray,
    flips0_map: np.ndarray,
    label_ceiling_net_fixed: int,
    band_snapped: np.ndarray,
    paint_hwc: np.ndarray,
    base_sc_u8: np.ndarray,
    mode: str,
    null_float_max_abs_A_delta: float | None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cam = realize_scorer_paint_to_camera(dec_f1, band_snapped, paint_hwc)
    pair = np.stack([dec_f0, cam]).astype(np.uint8)
    cells, pose = forward(segnet, posenet, pair[None])
    lam = cells[0]
    after = lam != lgt
    before = int(flips0_map.sum())
    after_count = int(after.sum())
    d_pose_after = float(np.square(pose[0] - pose_gt).sum() / 6.0)
    changed = paint_hwc != base_sc_u8
    row = {
        "mode": mode,
        "flips_after": after_count,
        "net_flip_reduction": before - after_count,
        "eta_realized": (before - after_count) / label_ceiling_net_fixed if label_ceiling_net_fixed else None,
        "fixed_global": int((flips0_map & ~after).sum()),
        "introduced_global": int((~flips0_map & after).sum()),
        "C_after": confusion(lgt, lam).tolist(),
        "d_pose_after": d_pose_after,
        "changed_scorer_pixels": int(changed.any(axis=2).sum()),
        "changed_scorer_channel_values": int(changed.sum()),
        "yuv6_residual_after_naive_rounding": yuv6_shift(base_sc_u8, paint_hwc),
        "float_null_certificate_max_abs_A_delta_before_rounding": null_float_max_abs_A_delta,
        "naive_rounding": True,
        "dk1_realizer_interface": "NullRealizer.apply_coefficients(pair, coeffs, basis_id, rounding_policy)",
    }
    if diagnostics is not None:
        row["diagnostics"] = diagnostics
    return row


def solve_unconstrained_weighted(
    segnet: Any,
    dec_f1: np.ndarray,
    gt_f1: np.ndarray,
    target_labels: np.ndarray,
    snapped_band: np.ndarray,
    objective_weights: np.ndarray,
    *,
    steps: int,
    lr: float,
    eval_every: int,
    convergence_patience_evals: int,
    convergence_min_improvement: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = resize_to_scorer(dec_f1)
    truth = resize_to_scorer(gt_f1)
    target = torch.from_numpy(target_labels.astype(np.int64))[None]
    mask = torch.from_numpy(snapped_band.astype(np.float32))[None, None]
    weights = torch.from_numpy(objective_weights.astype(np.float32))[None]
    best: tuple[int, int, str, np.ndarray] | None = None
    diagnostics = {"starts": [], "loss": "weighted cross_entropy against translated phase target"}
    with torch.enable_grad():
        for start_name, start in (("dec", base), ("truth", truth)):
            raw = torch.zeros_like(base, requires_grad=True)
            opt = torch.optim.Adam([raw], lr=lr)
            start_diag = {"start": start_name, "curve": [], "stop_reason": "iteration_cap"}
            start_best: tuple[int, int] | None = None
            evals_since_best = 0
            for step in range(steps + 1):
                preclip = base * (1.0 - mask) + (start + raw) * mask
                cur = torch.clamp(preclip, 0.0, 255.0)
                if step % eval_every == 0 or step == steps:
                    q = torch.round(cur).detach()
                    with torch.no_grad():
                        lam = segnet(q).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                    bad = int((lam != target_labels).sum())
                    clipped = int(((preclip < 0.0) | (preclip > 255.0)).sum().item())
                    start_diag["curve"].append(
                        {"step": int(step), "proxy_phase_target_flips": bad, "clipped_channel_values": clipped}
                    )
                    improved = (
                        start_best is None
                        or int(start_best[0]) - bad >= max(1, int(convergence_min_improvement))
                    )
                    if improved:
                        start_best = (bad, int(step))
                        evals_since_best = 0
                    else:
                        evals_since_best += 1
                    if best is None or bad < best[0]:
                        paint = q[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        best = (bad, int(step), start_name, paint)
                    if (
                        convergence_patience_evals > 0
                        and evals_since_best >= convergence_patience_evals
                        and step < steps
                    ):
                        start_diag["stop_reason"] = "plateau_no_proxy_improvement"
                        break
                if step == steps:
                    start_diag["stop_reason"] = (
                        "iteration_cap_best_at_cap"
                        if start_best is not None and start_best[1] == step
                        else "iteration_cap_before_plateau"
                    )
                    break
                logits = segnet(cur)
                per_pixel = torch.nn.functional.cross_entropy(logits, target, reduction="none")
                loss = (per_pixel * weights).sum() / weights.sum().clamp_min(1e-9)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            start_diag["best_proxy_phase_target_flips"] = start_best[0] if start_best else None
            start_diag["best_step"] = start_best[1] if start_best else None
            diagnostics["starts"].append(start_diag)
    if best is None:
        raise RuntimeError("unconstrained weighted solve produced no iterate")
    diagnostics["selected"] = {
        "start": best[2],
        "best_step": best[1],
        "best_proxy_phase_target_flips": best[0],
    }
    return best[3], diagnostics


def solve_within_null_basis(
    segnet: Any,
    dec_f1: np.ndarray,
    gt_f1: np.ndarray,
    target_labels: np.ndarray,
    block_mask: np.ndarray,
    objective_weights: np.ndarray,
    basis_t: torch.Tensor,
    constraint_t: torch.Tensor,
    *,
    steps: int,
    lr: float,
    eval_every: int,
    convergence_patience_evals: int,
    convergence_min_improvement: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = resize_to_scorer(dec_f1)
    truth = resize_to_scorer(gt_f1)
    target = torch.from_numpy(target_labels.astype(np.int64))[None]
    weights = torch.from_numpy(objective_weights.astype(np.float32))[None]
    block_mask_t = torch.from_numpy(block_mask.astype(np.float32))[None, :, :, None]
    truth_delta = truth - base
    truth_init = coeffs_from_delta_euclidean(truth_delta, basis_t) * block_mask_t
    starts = (("dec", torch.zeros_like(truth_init)), ("truth_null", truth_init.detach()))
    best: tuple[int, int, str, np.ndarray, float] | None = None
    diagnostics = {
        "starts": [],
        "parameterization": "per-2x2-block c in R^6, delta=N@c, no projection step",
        "loss": "diagonal margin-saliency weighted cross_entropy against translated phase target",
    }
    with torch.enable_grad():
        for start_name, init in starts:
            coeffs = init.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([coeffs], lr=lr)
            start_diag = {"start": start_name, "curve": [], "stop_reason": "iteration_cap"}
            start_best: tuple[int, int] | None = None
            evals_since_best = 0
            for step in range(steps + 1):
                live_coeffs = coeffs * block_mask_t
                delta = coeffs_to_delta_chw(live_coeffs, basis_t)
                preclip = base + delta
                cur = torch.clamp(preclip, 0.0, 255.0)
                if step % eval_every == 0 or step == steps:
                    q = torch.round(cur).detach()
                    with torch.no_grad():
                        lam = segnet(q).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                    bad = int((lam != target_labels).sum())
                    blocks = chw_to_block_vectors(delta.detach())[0].cpu().numpy()
                    a_delta = np.einsum("ad,hwd->hwa", constraint_t.cpu().numpy(), blocks)
                    max_a = float(np.abs(a_delta).max())
                    clipped = int(((preclip < 0.0) | (preclip > 255.0)).sum().item())
                    start_diag["curve"].append(
                        {
                            "step": int(step),
                            "proxy_phase_target_flips": bad,
                            "max_abs_A_delta_float": max_a,
                            "clipped_channel_values": clipped,
                        }
                    )
                    improved = (
                        start_best is None
                        or int(start_best[0]) - bad >= max(1, int(convergence_min_improvement))
                    )
                    if improved:
                        start_best = (bad, int(step))
                        evals_since_best = 0
                    else:
                        evals_since_best += 1
                    if best is None or bad < best[0]:
                        paint = q[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        best = (bad, int(step), start_name, paint, max_a)
                    if (
                        convergence_patience_evals > 0
                        and evals_since_best >= convergence_patience_evals
                        and step < steps
                    ):
                        start_diag["stop_reason"] = "plateau_no_proxy_improvement"
                        break
                if step == steps:
                    start_diag["stop_reason"] = (
                        "iteration_cap_best_at_cap"
                        if start_best is not None and start_best[1] == step
                        else "iteration_cap_before_plateau"
                    )
                    break
                logits = segnet(cur)
                per_pixel = torch.nn.functional.cross_entropy(logits, target, reduction="none")
                loss = (per_pixel * weights).sum() / weights.sum().clamp_min(1e-9)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            start_diag["best_proxy_phase_target_flips"] = start_best[0] if start_best else None
            start_diag["best_step"] = start_best[1] if start_best else None
            diagnostics["starts"].append(start_diag)
    if best is None:
        raise RuntimeError("null-basis solve produced no iterate")
    diagnostics["selected"] = {
        "start": best[2],
        "best_step": best[1],
        "best_proxy_phase_target_flips": best[0],
        "max_abs_A_delta_float": best[4],
    }
    return best[3], diagnostics


def aggregate(rows: list[dict[str, Any]], *, bar: float, parent_d_pose: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "ddm_sw1_null_basis_phase_solve_aggregate.v1",
        "n_rows": len(rows),
        "breakeven_eta_bar": float(bar),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
    }
    if not rows:
        return out
    denom = int(sum(r["label_ceiling_net_fixed"] for r in rows))
    before = int(sum(r["flips_before"] for r in rows))
    out["flips_before_subset"] = before
    out["label_ceiling_net_fixed_subset"] = denom
    for mode in MODES:
        mode_rows = [r["modes"][mode] for r in rows]
        after = int(sum(m["flips_after"] for m in mode_rows))
        eta = (before - after) / denom if denom else None
        ratios = np.array([m["d_pose_after"] / r["d_pose_before"] for r, m in zip(rows, mode_rows, strict=True)])
        pose_sse_delta = sum(
            (m["d_pose_after"] - r["d_pose_before"]) * 6.0 for r, m in zip(rows, mode_rows, strict=True)
        )
        pose_delta_s = math.sqrt(10.0 * (parent_d_pose + pose_sse_delta / (N_PAIRS_TOTAL * 6))) - math.sqrt(
            10.0 * parent_d_pose
        )
        seg_delta_s = (after - before) * S_PER_FLIP
        out[mode] = {
            "flips_after_subset": after,
            "net_flip_reduction_subset": before - after,
            "eta_realized": eta,
            "eta_over_bar": eta / bar if eta is not None and bar else None,
            "clears_bar": bool(eta is not None and eta >= bar),
            "seg_delta_S_no_rate": seg_delta_s,
            "pose_delta_S_against_parent": pose_delta_s,
            "joint_delta_S_no_rate_against_parent": seg_delta_s + pose_delta_s,
            "pose_ratio_min": float(ratios.min()),
            "pose_ratio_mean": float(ratios.mean()),
            "pose_ratio_max": float(ratios.max()),
            "selected_start_census": dict(
                sorted(Counter(m.get("diagnostics", {}).get("selected", {}).get("start", "unknown") for m in mode_rows).items())
            ),
        }
    e = out["project_after_euclidean"]["eta_realized"]
    m = out["project_after_metric"]["eta_realized"]
    sw = out["solve_within_null_basis"]["eta_realized"]
    u = out["unconstrained"]["eta_realized"]
    out["tax_recovery"] = {
        "solve_within_minus_project_after_euclidean_eta": (sw - e) if sw is not None and e is not None else None,
        "solve_within_minus_project_after_metric_eta": (sw - m) if sw is not None and m is not None else None,
        "project_after_metric_minus_euclidean_eta": (m - e) if m is not None and e is not None else None,
        "solve_within_recovered_fraction_of_unconstrained_minus_euclidean": (
            (sw - e) / (u - e) if None not in (sw, e, u) and abs(u - e) > 1e-12 else None
        ),
        "headline_scope": "small-n advisory, same parent/protocol; not n600 and not a score claim",
    }
    verdict = "READY_TO_FIRE_N32_IF_MAIN_ACCEPTS_FORM" if sw is not None and sw > max(bar, e or -1e9, m or -1e9) else "READY_TO_FIRE_CODEPATH_ONLY_MAIN_ADJUDICATION_REQUIRED"
    out["verdict"] = verdict
    out["verdict_scope"] = "FORMULATION: null-coordinate block16 phase paint on tq1c parent, bounded small-n advisory"
    return out


def payload(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    parent_archive_sha: str,
    parent_score: dict[str, Any],
    field_current: dict[str, Any],
    basis_cert: dict[str, Any],
    scorer_custody: dict[str, Any],
) -> dict[str, Any]:
    parent_d_pose = float(parent_score.get("d_pose", BASELINE_D_POSE))
    pairs_all = json.loads(args.et1_n32_json.read_text())["pairs"]
    pairs = [int(p) for p in pairs_all[: args.limit]]
    return {
        "schema": "ddm_sw1_null_basis_phase_solve_summary.v1",
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": False,
        "n600_run": False,
        "parent": {
            "archive": str(args.parent_archive),
            "archive_sha256": parent_archive_sha,
            "archive_bytes": int(args.parent_archive.stat().st_size),
            "expected_own_vehicle_frontier": {
                "S": BASELINE_S,
                "d_seg": BASELINE_D_SEG,
                "d_pose": BASELINE_D_POSE,
                "bytes": BASELINE_BYTES,
            },
            "parent_score": parent_score,
        },
        "phase_field": field_current,
        "selection": {
            "mode": "et1/et2 fixed n32 order prefix, bounded by sw1 --limit",
            "source": str(args.et1_n32_json),
            "pairs": pairs,
            "n_pairs": len(pairs),
        },
        "solver": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "eval_every": int(args.eval_every),
            "convergence_patience_evals": int(args.convergence_patience_evals),
            "convergence_min_improvement": int(args.convergence_min_improvement),
            "s_metric": {
                "producer": "tac.margin_saliency_map.compute_margin_saliency_map",
                "lambda_saliency": float(args.lambda_saliency),
                "outside_weight": float(args.outside_weight),
                "saliency_clip": float(args.saliency_clip),
                "metric_scope": "diagonal scorer-grid margin-saliency weights; full MS4D margin-Fisher substitution is ledger-queued",
            },
            "realizer": {
                "current": "naive uint8 rounding after float null-coordinate solve",
                "dk1_interface": "NullRealizer.apply_coefficients(pair, coeffs, basis_id, rounding_policy)",
            },
        },
        "basis": basis_cert,
        "scorer_custody": scorer_custody,
        "aggregate": aggregate(rows, bar=float(field_current["breakeven_eta"]), parent_d_pose=parent_d_pose),
        "rows": rows,
        "boundaries": [
            "small-n advisory only",
            "no n600 scorer slot consumed",
            "no archive build",
            "naive rounding used; lattice-native realization belongs to dk1",
            "full rank-4 MS4D margin-Fisher row-Gram not consumed in this build",
        ],
    }


def execute(args: argparse.Namespace) -> int:
    if args.limit < 1 or args.limit > 8:
        raise RuntimeError("sw1 charter bound requires 1 <= --limit <= 8")
    parent_archive_sha = sha256_file(args.parent_archive)
    if parent_archive_sha != BASELINE_ARCHIVE_SHA256:
        raise RuntimeError(f"parent archive SHA drifted: {parent_archive_sha}")
    args.bulk_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.bulk_dir / "sw1_null_basis_rows.jsonl"
    summary_path = args.bulk_dir / "sw1_null_basis_summary.json"
    receipt_summary_path = args.receipt_dir / "sw1_null_basis_summary.json"

    raw = raw_memmap(args.parent_raw)
    parent_lstars = np.load(args.parent_argmax, mmap_mode="r")
    current_offsets = np.load(args.current_offsets, mmap_mode="r")
    gt_labels = open_stored_npy_memmap(args.gt_cache, "lstars")
    parent_score = json.loads(args.parent_score.read_text())
    field_payload = json.loads(args.phase_field_summary.read_text())
    field_current = field_payload["current"]
    basis_np, basis_cert = null_coordinate_basis()
    basis_t = torch.from_numpy(basis_np.astype(np.float32))
    constraint_t = torch.from_numpy(pose_constraint_matrix().astype(np.float32))
    segnet, posenet, scorer_custody = load_models(args.upstream_root, threads=args.threads)
    pairs_all = json.loads(args.et1_n32_json.read_text())["pairs"]
    pairs = [int(p) for p in pairs_all[: args.limit]]
    existing = load_jsonl_rows(rows_path) if args.resume else []
    done = {int(row["pair"]) for row in existing}
    todo = [p for p in pairs if p not in done]
    wanted: set[int] = set()
    for pair in todo:
        wanted.update({seq_len * pair, seq_len * pair + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted) if wanted else {}
    rows = list(existing)

    write_json_atomic(
        summary_path,
        payload(
            args,
            rows,
            parent_archive_sha=parent_archive_sha,
            parent_score=parent_score,
            field_current=field_current,
            basis_cert=basis_cert,
            scorer_custody=scorer_custody,
        ),
    )
    print(f"[sw1] ready rows={len(rows)} remaining={len(todo)} steps={args.steps} pairs={pairs}", flush=True)

    for pair in todo:
        started = time.time()
        dec = np.stack([raw[seq_len * pair], raw[seq_len * pair + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * pair], gt_frames[seq_len * pair + 1]]).astype(np.uint8)
        cells, pose_base = forward(segnet, posenet, dec[None])
        gt_cells, pose_gt_all = forward(segnet, posenet, gt[None])
        lstar = cells[0]
        lgt = gt_cells[0]
        pose_gt = pose_gt_all[0]
        cached_parent = np.asarray(parent_lstars[pair])
        cached_gt = np.asarray(gt_labels[pair], dtype=np.uint8)
        if not np.array_equal(lstar, cached_parent):
            raise RuntimeError(f"C2 failed for pair {pair}: decoded parent argmax != cached parent")
        if not np.array_equal(lgt, cached_gt):
            raise RuntimeError(f"C3 failed for pair {pair}: canonical GT decode argmax != GT cache")

        target = translate_blocks(lstar, np.asarray(current_offsets[pair]), args.block)
        band = target != lstar
        snapped = snap_band_to_blocks(band)
        block_mask = block_mask_from_band(snapped)
        block_mask_t = torch.from_numpy(block_mask.astype(np.float32))[None, :, :, None]
        flips0_map = lstar != lgt
        flips0 = int(flips0_map.sum())
        label_after = target != lgt
        label_ceiling_net_fixed = flips0 - int(label_after.sum())
        d_pose_before = float(np.square(pose_base[0] - pose_gt).sum() / 6.0)
        base = resize_to_scorer(dec[1])
        base_sc_u8 = torch.round(base)[0].permute(1, 2, 0).numpy().astype(np.uint8)

        with torch.enable_grad():
            sal = compute_margin_saliency_map(
                segnet,
                torch.from_numpy(np.ascontiguousarray(dec[1])).permute(2, 0, 1).float(),
                flip_pixel_mask=torch.from_numpy(snapped.astype(bool)),
            )
        weights, weight_stats = metric_weights_from_saliency(
            sal.saliency.cpu().numpy(),
            snapped,
            lambda_saliency=args.lambda_saliency,
            outside_weight=args.outside_weight,
            clip=args.saliency_clip,
        )

        unconstrained_paint, unconstrained_diag = solve_unconstrained_weighted(
            segnet,
            dec[1],
            gt[1],
            target,
            snapped,
            weights,
            steps=args.steps,
            lr=args.lr,
            eval_every=args.eval_every,
            convergence_patience_evals=args.convergence_patience_evals,
            convergence_min_improvement=args.convergence_min_improvement,
        )
        unconstrained_delta = (
            torch.from_numpy(np.ascontiguousarray(unconstrained_paint)).permute(2, 0, 1)[None].float()
            - base
        )
        e_delta = project_delta_euclidean_chw(unconstrained_delta, block_mask_t, basis_t)
        e_paint = torch.round(torch.clamp(base + e_delta, 0.0, 255.0))[0].permute(1, 2, 0).numpy().astype(np.uint8)
        e_blocks = chw_to_block_vectors(e_delta.detach())[0].cpu().numpy()
        e_a = np.einsum("ad,hwd->hwa", pose_constraint_matrix(), e_blocks)
        e_max_a = float(np.abs(e_a).max())

        base_sc_float = base[0].permute(1, 2, 0).detach().numpy().astype(np.float64)
        unconstrained_delta_hwc = unconstrained_paint.astype(np.float64) - base_sc_float
        m_delta_hwc = project_delta_metric_hwc(
            unconstrained_delta_hwc,
            block_mask,
            weights,
            basis_np,
            ridge=args.metric_ridge,
        )
        m_paint = np.round(np.clip(base_sc_float + m_delta_hwc, 0.0, 255.0)).astype(np.uint8)
        m_blocks = _hwc_to_block_vectors_np(m_delta_hwc)
        m_a = np.einsum("ad,hwd->hwa", pose_constraint_matrix(), m_blocks)
        m_max_a = float(np.abs(m_a).max())

        sw_paint, sw_diag = solve_within_null_basis(
            segnet,
            dec[1],
            gt[1],
            target,
            block_mask,
            weights,
            basis_t,
            constraint_t,
            steps=args.steps,
            lr=args.lr,
            eval_every=args.eval_every,
            convergence_patience_evals=args.convergence_patience_evals,
            convergence_min_improvement=args.convergence_min_improvement,
        )

        modes = {
            "unconstrained": evaluate_paint(
                segnet=segnet,
                posenet=posenet,
                dec_f0=dec[0],
                dec_f1=dec[1],
                pose_gt=pose_gt,
                lgt=lgt,
                flips0_map=flips0_map,
                label_ceiling_net_fixed=label_ceiling_net_fixed,
                band_snapped=snapped,
                paint_hwc=unconstrained_paint,
                base_sc_u8=base_sc_u8,
                mode="unconstrained",
                null_float_max_abs_A_delta=None,
                diagnostics=unconstrained_diag,
            ),
            "project_after_euclidean": evaluate_paint(
                segnet=segnet,
                posenet=posenet,
                dec_f0=dec[0],
                dec_f1=dec[1],
                pose_gt=pose_gt,
                lgt=lgt,
                flips0_map=flips0_map,
                label_ceiling_net_fixed=label_ceiling_net_fixed,
                band_snapped=snapped,
                paint_hwc=e_paint,
                base_sc_u8=base_sc_u8,
                mode="project_after_euclidean",
                null_float_max_abs_A_delta=e_max_a,
                diagnostics={"arm": "Arm E", "source": "unconstrained weighted paint projected by Euclidean N N^T"},
            ),
            "project_after_metric": evaluate_paint(
                segnet=segnet,
                posenet=posenet,
                dec_f0=dec[0],
                dec_f1=dec[1],
                pose_gt=pose_gt,
                lgt=lgt,
                flips0_map=flips0_map,
                label_ceiling_net_fixed=label_ceiling_net_fixed,
                band_snapped=snapped,
                paint_hwc=m_paint,
                base_sc_u8=base_sc_u8,
                mode="project_after_metric",
                null_float_max_abs_A_delta=m_max_a,
                diagnostics={
                    "arm": "Arm M",
                    "source": "unconstrained weighted paint projected by diagonal margin-saliency metric",
                    "ridge": float(args.metric_ridge),
                },
            ),
            "solve_within_null_basis": evaluate_paint(
                segnet=segnet,
                posenet=posenet,
                dec_f0=dec[0],
                dec_f1=dec[1],
                pose_gt=pose_gt,
                lgt=lgt,
                flips0_map=flips0_map,
                label_ceiling_net_fixed=label_ceiling_net_fixed,
                band_snapped=snapped,
                paint_hwc=sw_paint,
                base_sc_u8=base_sc_u8,
                mode="solve_within_null_basis",
                null_float_max_abs_A_delta=sw_diag["selected"]["max_abs_A_delta_float"],
                diagnostics=sw_diag,
            ),
        }
        row = {
            "schema": "ddm_sw1_null_basis_phase_solve_pair.v1",
            "pair": int(pair),
            "flips_before": flips0,
            "label_ceiling_flips_left": int(label_after.sum()),
            "label_ceiling_net_fixed": label_ceiling_net_fixed,
            "label_ceiling_fixed": int((flips0_map & ~label_after).sum()),
            "label_ceiling_broken": int(((~flips0_map) & label_after).sum()),
            "band_px": int(band.sum()),
            "band_snapped_px": int(snapped.sum()),
            "band_snap_tax": float(snapped.sum() / max(1, band.sum())),
            "d_pose_before": d_pose_before,
            "metric_weights": weight_stats,
            "controls": {
                "C2_parent_argmax_matches_cache": True,
                "C3_gt_argmax_matches_cache": True,
                "offset_shape": list(np.asarray(current_offsets[pair]).shape),
                "null_basis_max_abs_A_times_N": basis_cert["max_abs_A_times_N"],
            },
            "modes": modes,
            "elapsed_s": time.time() - started,
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
        }
        rows.append(row)
        append_jsonl(rows_path, row)
        current = payload(
            args,
            rows,
            parent_archive_sha=parent_archive_sha,
            parent_score=parent_score,
            field_current=field_current,
            basis_cert=basis_cert,
            scorer_custody=scorer_custody,
        )
        write_json_atomic(summary_path, current)
        write_json_atomic(receipt_summary_path, current)
        agg = current["aggregate"]
        print(
            f"[sw1] pair {pair:3d} ({len(rows)}/{len(pairs)}) "
            f"E={modes['project_after_euclidean']['eta_realized']:+.4f} "
            f"M={modes['project_after_metric']['eta_realized']:+.4f} "
            f"SW={modes['solve_within_null_basis']['eta_realized']:+.4f} "
            f"tax={agg.get('tax_recovery', {}).get('solve_within_minus_project_after_euclidean_eta')} "
            f"[{time.time()-started:.1f}s]",
            flush=True,
        )

    final = payload(
        args,
        rows,
        parent_archive_sha=parent_archive_sha,
        parent_score=parent_score,
        field_current=field_current,
        basis_cert=basis_cert,
        scorer_custody=scorer_custody,
    )
    write_json_atomic(summary_path, final)
    write_json_atomic(receipt_summary_path, final)
    print(json.dumps(final["aggregate"], indent=1, default=jsonable), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-raw", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/submission/inflated/0.raw"))
    ap.add_argument("--parent-archive", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes"))
    ap.add_argument("--parent-argmax", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy"))
    ap.add_argument("--parent-score", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/aggregate.json"))
    ap.add_argument("--current-offsets", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/tq1c_block16_offsets.npy"))
    ap.add_argument("--phase-field-summary", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/phase_field_rederive_summary.json"))
    ap.add_argument("--gt-cache", type=Path, default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--gt-mkv", type=Path, default=REPO / "upstream/videos/0.mkv")
    ap.add_argument("--upstream-root", type=Path, default=REPO / "upstream")
    ap.add_argument("--et1-n32-json", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et1_20260803/et1_b16_realization_n32.json"))
    ap.add_argument("--bulk-dir", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_sw1_20260806"))
    ap.add_argument("--receipt-dir", type=Path, default=REPO / ".omx/research/ddm_sw1_20260806")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--convergence-patience-evals", type=int, default=0)
    ap.add_argument("--convergence-min-improvement", type=int, default=1)
    ap.add_argument("--lambda-saliency", type=float, default=1.0)
    ap.add_argument("--outside-weight", type=float, default=0.02)
    ap.add_argument("--saliency-clip", type=float, default=20.0)
    ap.add_argument("--metric-ridge", type=float, default=1e-5)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
