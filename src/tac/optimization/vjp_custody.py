# SPDX-License-Identifier: MIT
"""Frozen CPU-Torch VJP custody for joint SegNet/PoseNet inverse solves.

The sidecars produced by this module are measurement inputs, never decoder
payloads or score evidence.  They bind an exact cached winner/native-rival
arrangement to real scorer-plane derivatives and retain the camera pullbacks
needed to audit the shared resize adjoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_SCHEMA = "vjp_custody_manifest.v1"
PAIR_SCHEMA = "vjp_custody_pair.v1"
RECEIVER_ARITHMETIC = "native_float32_cpu_torch"
ACTIVE_ARRANGEMENT = "cached_winner_native_rival"
REPRESENTATION = "solver_scorer_plane_y_with_camera_adjoint_x"
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
N_CLASSES = 5
EXPECTED_HASHES = {
    "cache_sha256": "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6",
    "modules_sha256": "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa",
    "frame_utils_sha256": "d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90",
    "segnet_weights_sha256": "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6",
    "posenet_weights_sha256": "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576",
}


class VJPCustodyError(ValueError):
    """A malformed derivative, source, arrangement, or immutable artifact."""


@dataclass(frozen=True)
class CustodiedPair:
    """Validated arrays consumed by the positive-band measurement path."""

    pair_id: int
    winner: np.ndarray
    rival: np.ndarray
    cached_margin: np.ndarray
    native_margin: np.ndarray
    pair_norms: np.ndarray
    seg_g_y: np.ndarray
    seg_g_x: np.ndarray
    seg_q: np.ndarray
    seg_local_lipschitz: np.ndarray
    pose_j_y: np.ndarray
    pose_j_x: np.ndarray
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def source_hashes(cache: Path, upstream: Path) -> dict[str, str]:
    paths = {
        "cache_sha256": cache,
        "modules_sha256": upstream / "modules.py",
        "frame_utils_sha256": upstream / "frame_utils.py",
        "segnet_weights_sha256": upstream / "models/segnet.safetensors",
        "posenet_weights_sha256": upstream / "models/posenet.safetensors",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise VJPCustodyError(f"frozen custody sources are missing: {missing}")
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_HASHES:
        differences = {
            name: {"expected": EXPECTED_HASHES[name], "actual": actual[name]}
            for name in actual
            if actual[name] != EXPECTED_HASHES[name]
        }
        raise VJPCustodyError(f"frozen source hash mismatch: {differences}")
    return actual


def factor_vjp(g_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact pointwise ``g_y = Lip_local * q`` decomposition."""

    gradient = np.asarray(g_y)
    if gradient.dtype != np.float32 or gradient.ndim != 3 or gradient.shape[-1] != 3:
        raise VJPCustodyError("Seg scorer-plane VJP must be fp32 HxWx3")
    if not np.isfinite(gradient).all():
        raise VJPCustodyError("Seg scorer-plane VJP contains nonfinite values")
    local_lipschitz = np.linalg.norm(gradient, axis=-1).astype(np.float32)
    q = np.zeros_like(gradient)
    np.divide(
        gradient,
        local_lipschitz[..., None],
        out=q,
        where=local_lipschitz[..., None] > 0,
    )
    residual = np.max(
        np.abs(gradient - local_lipschitz[..., None] * q), initial=0.0
    )
    if float(residual) > 8 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(gradient), initial=0.0))):
        raise VJPCustodyError(f"Seg VJP factorization residual is too large: {residual}")
    return q, local_lipschitz


def largest_feasible_pose_step(
    pose_j_y: np.ndarray,
    frame1_delta: np.ndarray,
    frame0_direction: np.ndarray,
    source_pose_delta6: np.ndarray,
    tau_pose: float,
    *,
    max_step: float = 1.0,
) -> dict[str, Any]:
    """Largest deterministic predictor step whose linearized pose MSE fits.

    ``source_pose_delta6`` is the native source Pose6 minus the cached target
    Pose6.  ``frame1_delta`` is already fixed by the Seg solve, while
    ``frame0_direction`` points from the source scorer plane toward its
    generated predictor.
    """

    jac = np.asarray(pose_j_y, dtype=np.float64)
    d1 = np.asarray(frame1_delta, dtype=np.float64)
    d0 = np.asarray(frame0_direction, dtype=np.float64)
    base = np.asarray(source_pose_delta6, dtype=np.float64)
    if jac.shape[0:2] != (6, 2) or jac.shape[2:] != (d0.shape[0], d0.shape[1], 3):
        raise VJPCustodyError("Pose J_y must have shape (6,2,H,W,3)")
    if d0.shape != d1.shape or d0.ndim != 3 or d0.shape[-1] != 3:
        raise VJPCustodyError("Pose deltas must be same-shape HxWx3 fields")
    if base.shape != (6,):
        raise VJPCustodyError("source Pose6 base delta must have shape (6,)")
    values = (tau_pose, max_step)
    if not all(math.isfinite(float(value)) for value in values) or tau_pose < 0 or not (0 <= max_step <= 1):
        raise VJPCustodyError("tau_pose must be nonnegative and max_step must lie in [0,1]")
    if (
        not np.isfinite(jac).all()
        or not np.isfinite(d0).all()
        or not np.isfinite(d1).all()
        or not np.isfinite(base).all()
    ):
        raise VJPCustodyError("Pose proposal inputs must be finite")

    frame1_linear = np.einsum("khwc,hwc->k", jac[:, 1], d1, optimize=True)
    fixed = base + frame1_linear
    direction = np.einsum("khwc,hwc->k", jac[:, 0], d0, optimize=True)
    a = float(np.mean(direction * direction))
    b = float(2.0 * np.mean(fixed * direction))
    c = float(np.mean(fixed * fixed) - tau_pose)
    candidates = [0.0, float(max_step)]
    if a > 0:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0:
            root = math.sqrt(max(0.0, discriminant))
            candidates.extend(((-b - root) / (2 * a), (-b + root) / (2 * a)))
    elif b != 0:
        candidates.append(-c / b)
    feasible = [
        min(float(max_step), max(0.0, value))
        for value in candidates
        if math.isfinite(value)
        and -1e-12 <= value <= float(max_step) + 1e-12
        and a * value * value + b * value + c <= 1e-10
    ]
    if not feasible:
        return {
            "feasible": False,
            "selected_step": None,
            "source_pose_base_delta6": base.tolist(),
            "frame1_linear_pose_delta6": frame1_linear.tolist(),
            "fixed_pose_delta6": fixed.tolist(),
            "direction_pose_delta6": direction.tolist(),
            "planned_predictor_step_pose_delta6": None,
            "planned_predictor_step_pose_mse": None,
        }
    step = max(feasible)
    predicted = fixed + step * direction
    return {
        "feasible": True,
        "selected_step": step,
        "source_pose_base_delta6": base.tolist(),
        "frame1_linear_pose_delta6": frame1_linear.tolist(),
        "fixed_pose_delta6": fixed.tolist(),
        "direction_pose_delta6": direction.tolist(),
        "planned_predictor_step_pose_delta6": predicted.tolist(),
        "planned_predictor_step_pose_mse": float(np.mean(predicted * predicted)),
    }


def linearized_pose_delta6(
    pose_j_y: np.ndarray,
    frame0_delta: np.ndarray,
    frame1_delta: np.ndarray,
    source_pose_delta6: np.ndarray,
) -> np.ndarray:
    """Evaluate the first-order Pose6 debt for actual scorer-plane deltas."""

    jac = np.asarray(pose_j_y, dtype=np.float64)
    d0 = np.asarray(frame0_delta, dtype=np.float64)
    d1 = np.asarray(frame1_delta, dtype=np.float64)
    base = np.asarray(source_pose_delta6, dtype=np.float64)
    if jac.shape[0:2] != (6, 2) or jac.shape[2:] != d0.shape:
        raise VJPCustodyError("Pose J_y must have shape (6,2,H,W,3)")
    if d0.shape != d1.shape or d0.ndim != 3 or d0.shape[-1] != 3:
        raise VJPCustodyError("Pose deltas must be same-shape HxWx3 fields")
    if base.shape != (6,):
        raise VJPCustodyError("source Pose6 base delta must have shape (6,)")
    if not all(np.isfinite(value).all() for value in (jac, d0, d1, base)):
        raise VJPCustodyError("Pose linearization inputs must be finite")
    return base + np.einsum("kthwc,thwc->k", jac, np.stack((d0, d1)), optimize=True)


def _torch_hwc(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().permute(1, 2, 0).contiguous().numpy().astype(np.float32)


def _pose6(output: Any) -> Any:
    pose = output["pose"] if isinstance(output, dict) else output
    if pose.ndim != 2 or pose.shape[1] < 6:
        raise VJPCustodyError("PoseNet output lacks the first six scored coordinates")
    return pose[0, :6]


def _resize_adjoint(torch: Any, scorer_hwc: np.ndarray, camera_hw: tuple[int, int]) -> np.ndarray:
    import torch.nn.functional as functional

    probe = torch.zeros((1, 3, *camera_hw), dtype=torch.float32, requires_grad=True)
    scorer = functional.interpolate(probe, size=scorer_hwc.shape[:2], mode="bilinear", align_corners=False)
    cotangent = torch.from_numpy(scorer_hwc).permute(2, 0, 1).unsqueeze(0)
    (adjoint,) = torch.autograd.grad((scorer * cotangent).sum(), probe)
    return _torch_hwc(adjoint[0])


def compute_pair_derivatives(
    *,
    pair_id: int,
    frame0: np.ndarray,
    frame1: np.ndarray,
    cached_winner: np.ndarray,
    cached_margin: np.ndarray,
    segnet: Any,
    posenet: Any,
    torch: Any,
) -> dict[str, Any]:
    """Compute real frozen scorer VJPs for one camera-frame pair."""

    import torch.nn.functional as functional

    from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

    f0, f1 = np.asarray(frame0), np.asarray(frame1)
    winner = np.asarray(cached_winner, dtype=np.int64)
    margin = np.asarray(cached_margin, dtype=np.float32)
    camera_hw, scorer_hw = f0.shape[:2], winner.shape
    if f0.shape != (*camera_hw, 3) or f1.shape != f0.shape or f0.dtype != np.uint8 or f1.dtype != np.uint8:
        raise VJPCustodyError("producer frames must be same-shape camera uint8 HWC")
    if scorer_hw != SCORER_HW or camera_hw != CAMERA_HW:
        raise VJPCustodyError("producer only accepts the frozen contest camera/scorer geometry")
    if margin.shape != scorer_hw or not np.isfinite(margin).all() or np.any(margin < 0):
        raise VJPCustodyError("cached margin geometry or values are invalid")

    x0 = torch.from_numpy(f0).permute(2, 0, 1).unsqueeze(0).float().requires_grad_(True)
    x1 = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).float().requires_grad_(True)
    y0 = functional.interpolate(x0, size=scorer_hw, mode="bilinear", align_corners=False)
    y1 = functional.interpolate(x1, size=scorer_hw, mode="bilinear", align_corners=False)
    logits = segnet(y1)
    if tuple(logits.shape) != (1, N_CLASSES, *scorer_hw):
        raise VJPCustodyError(f"unexpected SegNet logits shape: {tuple(logits.shape)}")
    native_winner_t = logits[0].argmax(dim=0)
    native_winner = native_winner_t.detach().cpu().numpy().astype(np.int64)
    if not np.array_equal(native_winner, winner):
        mismatch = int(np.count_nonzero(native_winner != winner))
        raise VJPCustodyError(
            f"pair {pair_id} active arrangement incompatible: cached/native winner mismatch pixels={mismatch}"
        )
    masked = logits[0].detach().clone()
    masked.scatter_(0, native_winner_t.unsqueeze(0), -torch.inf)
    rival_t = masked.argmax(dim=0)
    rival = rival_t.cpu().numpy().astype(np.int64)

    try:
        head_weight = segnet.segmentation_head[0].weight
    except (AttributeError, IndexError, TypeError) as exc:
        raise VJPCustodyError("SegNet spatial 3x3 head weight was not found") from exc
    if head_weight.ndim != 4 or tuple(head_weight.shape[:1] + head_weight.shape[2:]) != (N_CLASSES, 3, 3):
        raise VJPCustodyError(f"unexpected SegNet head kernel shape: {tuple(head_weight.shape)}")
    flat_weight = head_weight.detach().reshape(N_CLASSES, -1)
    norm_table = torch.linalg.vector_norm(flat_weight[:, None] - flat_weight[None, :], dim=-1)
    norms_t = norm_table[native_winner_t, rival_t]
    if bool(torch.any(~torch.isfinite(norms_t))) or bool(torch.any(norms_t <= 0)):
        raise VJPCustodyError("nonpositive or nonfinite active head-pair norm")
    active_pairs = {(int(w), int(r)) for w, r in zip(winner.flat, rival.flat, strict=True)}
    unit_errors = [
        abs(float(torch.linalg.vector_norm((flat_weight[w] - flat_weight[r]) / norm_table[w, r])) - 1.0)
        for w, r in active_pairs
    ]
    unit_normal_max_error = max(unit_errors, default=0.0)
    if unit_normal_max_error > 8 * np.finfo(np.float32).eps:
        raise VJPCustodyError(f"unit-head-normal review failed: {unit_normal_max_error}")

    winner_logits = logits[0].gather(0, native_winner_t.unsqueeze(0)).squeeze(0)
    rival_logits = logits[0].gather(0, rival_t.unsqueeze(0)).squeeze(0)
    native_margin_t = winner_logits - rival_logits
    native_margin = native_margin_t.detach().cpu().numpy().astype(np.float32)
    margin_abs = np.abs(native_margin - margin)
    margin_agreement = {
        "max_abs": float(np.max(margin_abs, initial=0.0)),
        "mean_abs": float(np.mean(margin_abs)),
        "allclose_rtol_1e-4_atol_1e-5": bool(np.allclose(native_margin, margin, rtol=1e-4, atol=1e-5)),
    }
    if not margin_agreement["allclose_rtol_1e-4_atol_1e-5"]:
        raise VJPCustodyError(f"pair {pair_id} cached/native margin agreement failed: {margin_agreement}")

    scalar = (native_margin_t / norms_t).sum(dtype=torch.float64)
    seg_g_y_t, seg_g_x_t = torch.autograd.grad(scalar, (y1, x1), retain_graph=True)
    seg_g_y = _torch_hwc(seg_g_y_t[0])
    seg_g_x = _torch_hwc(seg_g_x_t[0])
    if not np.isfinite(seg_g_x).all() or float(np.linalg.norm(seg_g_x.astype(np.float64))) == 0:
        raise VJPCustodyError("Seg camera pullback is nonfinite or all zero")
    seg_q, seg_lipschitz = factor_vjp(seg_g_y)
    seg_g_x_adjoint = _resize_adjoint(torch, seg_g_y, camera_hw)
    seg_adjoint_residual = float(np.max(np.abs(seg_g_x - seg_g_x_adjoint), initial=0.0))
    seg_adjoint_scale = max(1.0, float(np.max(np.abs(seg_g_x), initial=0.0)))
    if not math.isfinite(seg_adjoint_residual) or seg_adjoint_residual > 5e-6 * seg_adjoint_scale:
        raise VJPCustodyError(f"Seg A^T relation review failed: {seg_adjoint_residual}")

    grad_norm = float(np.linalg.norm(seg_g_y.astype(np.float64)))
    if not math.isfinite(grad_norm) or grad_norm == 0:
        raise VJPCustodyError("Seg VJP is nonfinite or all zero")
    direction = torch.from_numpy(seg_g_y / grad_norm).permute(2, 0, 1).unsqueeze(0)

    def seg_scalar(input_y: Any) -> Any:
        value = segnet(input_y)[0]
        win_value = value.gather(0, native_winner_t.unsqueeze(0)).squeeze(0)
        riv_value = value.gather(0, rival_t.unsqueeze(0)).squeeze(0)
        return ((win_value - riv_value) / norms_t).sum(dtype=torch.float64)

    epsilon = 0.25
    with torch.no_grad():
        plus = float(seg_scalar(y1.detach() + epsilon * direction))
        minus = float(seg_scalar(y1.detach() - epsilon * direction))
    finite_difference = (plus - minus) / (2 * epsilon)
    analytic_directional = float((seg_g_y_t * direction).sum())
    fd_relative_error = abs(finite_difference - analytic_directional) / max(1.0, abs(analytic_directional))
    if not math.isfinite(fd_relative_error) or fd_relative_error > 0.05:
        raise VJPCustodyError(f"Seg directional finite-difference review failed: relative error {fd_relative_error}")

    pose_input = torch.cat((differentiable_rgb_to_yuv6(y0), differentiable_rgb_to_yuv6(y1)), dim=1)
    pose6 = _pose6(posenet(pose_input))
    with torch.inference_mode():
        camera_pair = torch.stack((x0.detach()[0], x1.detach()[0]), dim=0).unsqueeze(0)
        upstream_pose6 = _pose6(posenet(posenet.preprocess_input(camera_pair)))
    pose_forward_max_abs = float(torch.max(torch.abs(pose6.detach() - upstream_pose6)).cpu())
    if not math.isfinite(pose_forward_max_abs) or pose_forward_max_abs > 1e-5:
        raise VJPCustodyError(f"Pose YUV6 forward parity failed: max abs {pose_forward_max_abs}")

    j_y_rows: list[np.ndarray] = []
    j_x_rows: list[np.ndarray] = []
    pose_adjoint_residual = 0.0
    for row in range(6):
        gy0_t, gy1_t, gx0_t, gx1_t = torch.autograd.grad(
            pose6[row], (y0, y1, x0, x1), retain_graph=row < 5
        )
        gy = np.stack((_torch_hwc(gy0_t[0]), _torch_hwc(gy1_t[0])))
        gx = np.stack((_torch_hwc(gx0_t[0]), _torch_hwc(gx1_t[0])))
        if not np.isfinite(gy).all() or float(np.linalg.norm(gy.astype(np.float64))) == 0:
            raise VJPCustodyError(f"Pose J_y row {row} is nonfinite or all zero")
        for frame_index in range(2):
            rebuilt = _resize_adjoint(torch, gy[frame_index], camera_hw)
            pose_adjoint_residual = max(
                pose_adjoint_residual,
                float(np.max(np.abs(gx[frame_index] - rebuilt), initial=0.0)),
            )
        j_y_rows.append(gy)
        j_x_rows.append(gx)
    pose_j_y = np.stack(j_y_rows).astype(np.float32, copy=False)
    pose_j_x = np.stack(j_x_rows).astype(np.float32, copy=False)
    if not np.isfinite(pose_j_x).all() or any(
        float(np.linalg.norm(pose_j_x[row].astype(np.float64))) == 0 for row in range(6)
    ):
        raise VJPCustodyError("Pose J_x contains a nonfinite or all-zero row")
    pose_adjoint_scale = max(1.0, float(np.max(np.abs(pose_j_x), initial=0.0)))
    if not math.isfinite(pose_adjoint_residual) or pose_adjoint_residual > 5e-6 * pose_adjoint_scale:
        raise VJPCustodyError(f"Pose A^T relation review failed: {pose_adjoint_residual}")

    seed_points = ((0, 0), (scorer_hw[0] // 2, scorer_hw[1] // 2), (scorer_hw[0] - 1, scorer_hw[1] - 1))
    seed_sign_review = [
        {
            "y": y,
            "x": x,
            "winner": int(winner[y, x]),
            "rival": int(rival[y, x]),
            "winner_cotangent_sign": 1,
            "rival_cotangent_sign": -1,
            "all_other_class_cotangent": 0,
        }
        for y, x in seed_points
    ]

    return {
        "pair_id": int(pair_id),
        "winner": winner.astype(np.int8),
        "rival": rival.astype(np.int8),
        "cached_margin": margin,
        "native_margin": native_margin,
        "head_pair_norms": norms_t.detach().cpu().numpy().astype(np.float32),
        "seg_g_y": seg_g_y,
        "seg_g_x": seg_g_x,
        "seg_q": seg_q,
        "seg_local_lipschitz": seg_lipschitz,
        "pose_j_y": pose_j_y,
        "pose_j_x": pose_j_x,
        "checks": {
            "head_kernel_shape": list(head_weight.shape),
            "unit_head_normal_max_error": unit_normal_max_error,
            "seed_class_and_sign_review": seed_sign_review,
            "cached_native_winner_agreement": True,
            "cached_native_margin_agreement": margin_agreement,
            "seg_directional_fd": {
                "epsilon": epsilon,
                "finite_difference": finite_difference,
                "analytic": analytic_directional,
                "relative_error": fd_relative_error,
            },
            "seg_A_transpose_max_abs_residual": seg_adjoint_residual,
            "pose_yuv6_forward_max_abs": pose_forward_max_abs,
            "pose_A_transpose_max_abs_residual": pose_adjoint_residual,
        },
    }


def pair_metadata(arrays: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    tensor_keys = (
        "winner", "rival", "cached_margin", "native_margin", "head_pair_norms",
        "seg_g_y", "seg_g_x", "seg_q", "seg_local_lipschitz", "pose_j_y", "pose_j_x",
    )
    return {
        "schema": PAIR_SCHEMA,
        "pair_id": int(arrays["pair_id"]),
        "receiver_arithmetic": RECEIVER_ARITHMETIC,
        "active_arrangement": ACTIVE_ARRANGEMENT,
        "winner_source": "cached_lstars_verified_against_fresh_native_fp32_logits",
        "rival_source": "fresh_native_fp32_logits_highest_nonwinner_not_cached",
        "representation": REPRESENTATION,
        "source_hashes": hashes,
        "checks": arrays["checks"],
        "tensors": {
            key: {
                "dtype": str(np.asarray(arrays[key]).dtype),
                "shape": list(np.asarray(arrays[key]).shape),
                "sha256": sha256_array(np.asarray(arrays[key])),
            }
            for key in tensor_keys
        },
        "reconstruction": {
            "rebuildable": True,
            "command_contract": "tools/produce_vjp_custody.py with manifest config and frozen source hashes",
            "cold_store": "SSD evidence tier; final sidecars are never auto-deleted",
            "score_claim": False,
        },
    }


def write_pair_sidecar(path: Path, arrays: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    """Write one immutable same-directory NPZ and return its manifest row."""

    if path.exists():
        raise VJPCustodyError(f"immutable VJP pair sidecar already exists: {path}")
    metadata = pair_metadata(arrays, hashes)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {key: np.asarray(value) for key, value in arrays.items() if isinstance(value, np.ndarray)}
    payload["pair_id"] = np.asarray(int(arrays["pair_id"]), dtype=np.int64)
    payload["custody_json"] = np.asarray(canonical_json(metadata).decode())
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {
        "pair_id": int(arrays["pair_id"]),
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_hashes": {key: value["sha256"] for key, value in metadata["tensors"].items()},
    }


def _load_pair(
    row: dict[str, Any],
    manifest: dict[str, Any],
    *,
    scorer_hw: tuple[int, int],
    camera_hw: tuple[int, int],
) -> CustodiedPair:
    path = Path(row["path"])
    if not path.is_file() or sha256_file(path) != row.get("sha256"):
        raise VJPCustodyError(f"pair {row.get('pair_id')} sidecar hash/path custody failed")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "pair_id", "custody_json", "winner", "rival", "cached_margin", "native_margin",
            "head_pair_norms", "seg_g_y", "seg_g_x", "seg_q", "seg_local_lipschitz",
            "pose_j_y", "pose_j_x",
        }
        missing = required.difference(data.files)
        if missing:
            raise VJPCustodyError(f"pair sidecar lacks keys: {sorted(missing)}")
        metadata = json.loads(str(np.asarray(data["custody_json"]).reshape(())))
        values = {key: np.asarray(data[key]).copy() for key in required - {"custody_json"}}
    pair_id = int(np.asarray(values.pop("pair_id")).reshape(()))
    if pair_id != int(row["pair_id"]) or metadata.get("pair_id") != pair_id:
        raise VJPCustodyError("pair id differs across manifest, NPZ, and custody metadata")
    if metadata.get("schema") != PAIR_SCHEMA:
        raise VJPCustodyError("pair custody schema mismatch")
    if metadata.get("winner_source") != "cached_lstars_verified_against_fresh_native_fp32_logits":
        raise VJPCustodyError("pair winner source declaration mismatch")
    if metadata.get("rival_source") != "fresh_native_fp32_logits_highest_nonwinner_not_cached":
        raise VJPCustodyError("pair rival source declaration mismatch")
    for key in ("receiver_arithmetic", "active_arrangement", "representation", "source_hashes"):
        if metadata.get(key) != manifest.get(key):
            raise VJPCustodyError(f"pair/manifest {key} mismatch")

    expected = {
        "winner": (np.dtype("int8"), scorer_hw),
        "rival": (np.dtype("int8"), scorer_hw),
        "cached_margin": (np.dtype("float32"), scorer_hw),
        "native_margin": (np.dtype("float32"), scorer_hw),
        "head_pair_norms": (np.dtype("float32"), scorer_hw),
        "seg_g_y": (np.dtype("float32"), (*scorer_hw, 3)),
        "seg_g_x": (np.dtype("float32"), (*camera_hw, 3)),
        "seg_q": (np.dtype("float32"), (*scorer_hw, 3)),
        "seg_local_lipschitz": (np.dtype("float32"), scorer_hw),
        "pose_j_y": (np.dtype("float32"), (6, 2, *scorer_hw, 3)),
        "pose_j_x": (np.dtype("float32"), (6, 2, *camera_hw, 3)),
    }
    for key, (dtype, shape) in expected.items():
        value = values[key]
        declared = metadata.get("tensors", {}).get(key, {})
        actual_hash = sha256_array(value)
        if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
            raise VJPCustodyError(f"pair {pair_id} tensor {key} dtype/shape/finiteness failed")
        if declared != {"dtype": str(dtype), "shape": list(shape), "sha256": actual_hash}:
            raise VJPCustodyError(f"pair {pair_id} tensor {key} metadata/hash failed")
        if row.get("tensor_hashes", {}).get(key) != actual_hash:
            raise VJPCustodyError(f"pair {pair_id} manifest tensor hash failed for {key}")
    if (
        np.any(values["winner"] < 0)
        or np.any(values["winner"] >= N_CLASSES)
        or np.any(values["rival"] < 0)
        or np.any(values["rival"] >= N_CLASSES)
        or np.any(values["winner"] == values["rival"])
        or np.any(values["head_pair_norms"] <= 0)
    ):
        raise VJPCustodyError(f"pair {pair_id} active arrangement is invalid")
    if not np.allclose(values["cached_margin"], values["native_margin"], rtol=1e-4, atol=1e-5):
        raise VJPCustodyError(f"pair {pair_id} cached/native margin agreement failed")
    rebuilt = values["seg_local_lipschitz"][..., None] * values["seg_q"]
    if not np.allclose(rebuilt, values["seg_g_y"], rtol=2e-6, atol=2e-7):
        raise VJPCustodyError(f"pair {pair_id} Seg VJP factorization failed")
    if (
        float(np.linalg.norm(values["seg_g_y"].astype(np.float64))) == 0
        or float(np.linalg.norm(values["seg_g_x"].astype(np.float64))) == 0
    ):
        raise VJPCustodyError(f"pair {pair_id} Seg VJP has an all-zero plane")
    for key in ("pose_j_y", "pose_j_x"):
        if any(float(np.linalg.norm(values[key][row].astype(np.float64))) == 0 for row in range(6)):
            raise VJPCustodyError(f"pair {pair_id} {key} has an all-zero row")
    return CustodiedPair(
        pair_id=pair_id,
        winner=values["winner"], rival=values["rival"],
        cached_margin=values["cached_margin"], native_margin=values["native_margin"],
        pair_norms=values["head_pair_norms"], seg_g_y=values["seg_g_y"],
        seg_g_x=values["seg_g_x"], seg_q=values["seg_q"],
        seg_local_lipschitz=values["seg_local_lipschitz"],
        pose_j_y=values["pose_j_y"], pose_j_x=values["pose_j_x"], metadata=metadata,
    )


def load_vjp_manifest(
    path: Path,
    pair_ids: list[int],
    *,
    scorer_hw: tuple[int, int] = SCORER_HW,
    camera_hw: tuple[int, int] = CAMERA_HW,
    require_frozen_hashes: bool = True,
) -> dict[int, CustodiedPair]:
    """Load a manifest/per-pair directory with full hash and shape custody."""

    manifest_path = path / "manifest.json" if path.is_dir() else path
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VJPCustodyError(f"cannot read VJP custody manifest: {manifest_path}") from exc
    expected_common = {
        "schema": MANIFEST_SCHEMA,
        "receiver_arithmetic": RECEIVER_ARITHMETIC,
        "active_arrangement": ACTIVE_ARRANGEMENT,
        "representation": REPRESENTATION,
    }
    for key, expected in expected_common.items():
        if manifest.get(key) != expected:
            raise VJPCustodyError(f"VJP manifest {key} mismatch")
    claimed_manifest_hash = manifest.get("manifest_content_sha256")
    actual_manifest_hash = hashlib.sha256(
        canonical_json({key: value for key, value in manifest.items() if key != "manifest_content_sha256"})
    ).hexdigest()
    if claimed_manifest_hash != actual_manifest_hash:
        raise VJPCustodyError("VJP manifest content hash mismatch")
    if require_frozen_hashes and manifest.get("source_hashes") != EXPECTED_HASHES:
        raise VJPCustodyError("VJP manifest frozen source hashes mismatch")
    if manifest.get("pair_ids") != pair_ids:
        raise VJPCustodyError("VJP manifest pair ids/order differ from invocation")
    rows = manifest.get("sidecars")
    if not isinstance(rows, list) or [row.get("pair_id") for row in rows] != pair_ids:
        raise VJPCustodyError("VJP manifest sidecar order/coverage mismatch")
    return {
        int(row["pair_id"]): _load_pair(row, manifest, scorer_hw=scorer_hw, camera_hw=camera_hw)
        for row in rows
    }


def load_vjp_pair_row(
    row: dict[str, Any],
    manifest: dict[str, Any],
    *,
    scorer_hw: tuple[int, int] = SCORER_HW,
    camera_hw: tuple[int, int] = CAMERA_HW,
) -> CustodiedPair:
    """Validate one completed row while a producer manifest is still partial."""

    return _load_pair(row, manifest, scorer_hw=scorer_hw, camera_hw=camera_hw)


def recover_pair_sidecar_row(
    path: Path,
    pair_id: int,
    manifest: dict[str, Any],
    *,
    scorer_hw: tuple[int, int] = SCORER_HW,
    camera_hw: tuple[int, int] = CAMERA_HW,
) -> dict[str, Any]:
    """Recover a valid immutable sidecar written before its manifest append.

    The row is reconstructed only from the already-renamed NPZ, then passed
    through the same source, tensor, shape, arithmetic, and arrangement checks
    used by the consumer.  The sidecar itself is never rewritten.
    """

    if not path.is_file():
        raise VJPCustodyError(f"orphan VJP sidecar is absent: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            if "pair_id" not in data.files or "custody_json" not in data.files:
                raise VJPCustodyError("orphan VJP sidecar lacks custody metadata")
            stored_pair_id = int(np.asarray(data["pair_id"]).reshape(()))
            metadata = json.loads(str(np.asarray(data["custody_json"]).reshape(())))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VJPCustodyError(f"cannot inspect orphan VJP sidecar: {path}") from exc
    if stored_pair_id != int(pair_id) or metadata.get("pair_id") != int(pair_id):
        raise VJPCustodyError("orphan VJP sidecar pair id mismatch")
    tensor_metadata = metadata.get("tensors")
    if not isinstance(tensor_metadata, dict):
        raise VJPCustodyError("orphan VJP sidecar tensor custody is missing")
    row = {
        "pair_id": int(pair_id),
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_hashes": {
            key: value.get("sha256")
            for key, value in tensor_metadata.items()
            if isinstance(value, dict)
        },
    }
    _load_pair(row, manifest, scorer_hw=scorer_hw, camera_hw=camera_hw)
    return row


__all__ = [
    "ACTIVE_ARRANGEMENT",
    "CAMERA_HW",
    "EXPECTED_HASHES",
    "MANIFEST_SCHEMA",
    "PAIR_SCHEMA",
    "RECEIVER_ARITHMETIC",
    "REPRESENTATION",
    "SCORER_HW",
    "CustodiedPair",
    "VJPCustodyError",
    "atomic_json",
    "canonical_json",
    "compute_pair_derivatives",
    "factor_vjp",
    "largest_feasible_pose_step",
    "linearized_pose_delta6",
    "load_vjp_manifest",
    "load_vjp_pair_row",
    "pair_metadata",
    "recover_pair_sidecar_row",
    "sha256_array",
    "sha256_file",
    "source_hashes",
    "write_pair_sidecar",
]
