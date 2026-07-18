# SPDX-License-Identifier: MIT
"""Measure local SegNet/PoseNet coupling through their exact shared resize A.

This is a read-only, local-CPU advisory tool.  It requires a real n600-trained
EMA NPZ, the real ZIP_STORED gt_n600 cache, and frozen upstream scorer weights.
It never trains, dispatches, updates a pointer, or writes anywhere except the
explicit ``--output`` JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tools", REPO_ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEMA = "shared_resize_joint_coupling_measurement.v2"
AXIS = "[macOS-CPU advisory]"
N_PAIRS_TOTAL = 600
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
SCORER_BATCH_SIZE = 32


class MeasurementError(RuntimeError):
    """Fail-closed input, custody, or scorer-geometry error."""


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint_payload_keys(keys: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Refuse any known pose-carrier state before invoking the base-INR decoder."""

    normalized = tuple(str(key) for key in keys)
    carrier_keys = sorted(
        key
        for key in normalized
        if "pose_carrier." in key
        or key.startswith("__cfg_pose_carrier")
        or key.startswith("__pose_carrier")
    )
    if carrier_keys:
        raise MeasurementError(
            "checkpoint contains pose-carrier payload/config keys and cannot be measured as "
            f"base-INR-only: {carrier_keys}"
        )
    key_digest = hashlib.sha256(
        json.dumps(sorted(normalized), separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "carrier_absent": True,
        "base_inr_only": True,
        "detected_carrier_keys": [],
        "checkpoint_key_count": len(normalized),
        "checkpoint_key_manifest_sha256": key_digest,
    }


def deterministic_stride_sample(n_total: int, n_sample: int, seed: int) -> tuple[int, ...]:
    """Even cyclic stride sample with a deterministic seed-derived phase."""

    if n_total <= 0 or not (1 <= n_sample <= n_total):
        raise ValueError("require 1 <= n_sample <= n_total")
    centers = np.floor((np.arange(n_sample, dtype=np.float64) + 0.5) * n_total / n_sample).astype(int)
    selected = np.sort((centers + int(seed) % n_total) % n_total)
    if len(np.unique(selected)) != n_sample:
        raise AssertionError("deterministic stride sampler produced duplicate pair IDs")
    return tuple(int(i) for i in selected)


def mmap_stored_npy_member(npz_path: str | Path, member: str) -> np.memmap:
    """Memory-map one uncompressed NPY member directly inside a ZIP_STORED NPZ."""

    archive = Path(npz_path)
    with zipfile.ZipFile(archive) as zf:
        try:
            info = zf.getinfo(member)
        except KeyError as exc:
            raise MeasurementError(f"gt cache lacks required member {member!r}") from exc
        if info.compress_type != zipfile.ZIP_STORED:
            raise MeasurementError(f"{member} is not ZIP_STORED; direct mmap would be false custody")
        if info.flag_bits & 0x1:
            raise MeasurementError(f"{member} is encrypted")
        local_header_offset = int(info.header_offset)
    with archive.open("rb") as handle:
        handle.seek(local_header_offset)
        header = handle.read(30)
        if len(header) != 30:
            raise MeasurementError(f"{member} has a truncated ZIP local header")
        signature, *_rest, filename_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50:
            raise MeasurementError(f"{member} has an invalid ZIP local header signature")
        npy_offset = local_header_offset + 30 + filename_len + extra_len
        handle.seek(npy_offset)
        try:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in {(2, 0), (3, 0)}:
                # NumPy exposes no public v3 wrapper.  The common parser is
                # version-aware and preserves v3's UTF-8 header decoding.
                shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
            else:
                raise MeasurementError(f"{member} uses unsupported NPY version {version}")
        except (ValueError, EOFError) as exc:
            raise MeasurementError(f"{member} has an invalid NPY header") from exc
        data_offset = handle.tell()
    order = "F" if fortran_order else "C"
    return np.memmap(archive, dtype=dtype, mode="r", offset=data_offset, shape=shape, order=order)


def topk_sign_direction(gradient: np.ndarray, support_fraction: float) -> np.ndarray:
    """Return the deterministic one-LSB *descent* direction on top-|gradient| entries."""

    values = np.asarray(gradient)
    fraction = float(support_fraction)
    if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
        raise ValueError("support_fraction must be in (0,1]")
    flat = values.reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError("gradient contains non-finite values")
    k = max(1, math.ceil(fraction * flat.size))
    # Stable index tie-break: lexsort by descending magnitude then ascending index.
    indices = np.lexsort((np.arange(flat.size), -np.abs(flat)))[:k]
    out = np.zeros(flat.shape, dtype=np.int8)
    out[indices] = -np.sign(flat[indices]).astype(np.int8)
    return out.reshape(values.shape)


def sample_mean_product_gram(
    sum_seg_seg: float,
    sum_seg_pose: float,
    sum_pose_pose: float,
    *,
    n_sample: int,
) -> list[list[float]]:
    """Convert block inner-product sums to the Gram of two sample-mean losses."""

    values = (float(sum_seg_seg), float(sum_seg_pose), float(sum_pose_pose))
    if not isinstance(n_sample, int) or isinstance(n_sample, bool) or n_sample < 1:
        raise ValueError("n_sample must be a positive integer")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Gram block sums must be finite")
    scale = float(n_sample**2)
    ss, sp, pp = (value / scale for value in values)
    return [[ss, sp], [sp, pp]]


def aggregate_response_metrics(
    baseline: dict[str, float],
    seg_plus: dict[str, float],
    seg_minus: dict[str, float],
    pose_plus: dict[str, float],
    pose_minus: dict[str, float],
    joint_plus: dict[str, float],
) -> dict[str, Any]:
    """Aggregate exact one-sided/central two-output secants from scorer summaries."""

    keys = ("d_seg", "d_pose")
    for payload in (baseline, seg_plus, seg_minus, pose_plus, pose_minus, joint_plus):
        if any(key not in payload or not math.isfinite(float(payload[key])) for key in keys):
            raise ValueError("response payload lacks finite d_seg/d_pose")
    one_sided = [
        [seg_plus[row] - baseline[row], pose_plus[row] - baseline[row]]
        for row in keys
    ]
    central = [
        [(seg_plus[row] - seg_minus[row]) / 2.0, (pose_plus[row] - pose_minus[row]) / 2.0]
        for row in keys
    ]
    joint = {key: joint_plus[key] - baseline[key] for key in keys}
    seg_target_delta = one_sided[0][0]
    pose_target_delta = one_sided[1][1]
    baseline_score = 100.0 * baseline["d_seg"] + math.sqrt(10.0 * baseline["d_pose"])
    joint_score = 100.0 * joint_plus["d_seg"] + math.sqrt(10.0 * joint_plus["d_pose"])

    def _cross_effect(delta: float) -> str:
        if delta < 0.0:
            return "MEASURED_HELP"
        if delta > 0.0:
            return "MEASURED_HARM"
        return "MEASURED_NEUTRAL"

    seg_informative = seg_target_delta < 0.0
    pose_informative = pose_target_delta < 0.0
    joint_informative = joint_score < baseline_score
    return {
        "one_sided_response_2x2": one_sided,
        "central_secant_response_2x2": central,
        "joint_plus_delta": joint,
        "measured_direction_classification": {
            "seg_direction": {
                "target_metric": "d_seg",
                "target_delta": seg_target_delta,
                "quality": "MEASURED_HELP" if seg_informative else "UNINFORMATIVE_DIRECTION",
                "cross_d_pose_effect": (
                    _cross_effect(one_sided[1][0]) if seg_informative else "UNINFORMATIVE_DIRECTION"
                ),
            },
            "pose_direction": {
                "target_metric": "d_pose",
                "target_delta": pose_target_delta,
                "quality": "MEASURED_HELP" if pose_informative else "UNINFORMATIVE_DIRECTION",
                "cross_d_seg_effect": (
                    _cross_effect(one_sided[0][1]) if pose_informative else "UNINFORMATIVE_DIRECTION"
                ),
            },
            "joint_direction": {
                "target_metric": "100*d_seg+sqrt(10*d_pose)",
                "target_delta": joint_score - baseline_score,
                "quality": "MEASURED_HELP" if joint_informative else "UNINFORMATIVE_DIRECTION",
            },
        },
        "cross_response_ratios": {
            "pose_change_per_abs_seg_change_under_seg_direction": (
                one_sided[1][0] / max(abs(one_sided[0][0]), 1e-30)
            ),
            "seg_change_per_abs_pose_change_under_pose_direction": (
                one_sided[0][1] / max(abs(one_sided[1][1]), 1e-30)
            ),
        },
    }


def realized_lsb_counts(
    pair: tuple[np.ndarray, np.ndarray], direction: np.ndarray, sign: int
) -> dict[str, int]:
    """Count nonzero requests, realized changes, and boundary clipping in camera layout."""

    base = np.stack(pair, axis=0).astype(np.int16)
    move = int(sign) * np.asarray(direction, dtype=np.int16)
    if base.shape != move.shape:
        raise ValueError(f"pair/direction shape mismatch: {base.shape} != {move.shape}")
    requested = move != 0
    realized = np.clip(base + move, 0, 255)
    changed = realized != base
    return {
        "nonzero_requested": int(np.count_nonzero(requested)),
        "realized_changed": int(np.count_nonzero(changed)),
        "boundary_clipped": int(np.count_nonzero(requested & ~changed)),
    }


def target_cache_mismatch_metrics(
    cached_labels: np.ndarray,
    rederived_labels: np.ndarray,
    cached_pose: np.ndarray,
    rederived_pose: np.ndarray,
) -> dict[str, Any]:
    """Compare stored targets with the exact targets rederived for this scorer invocation."""

    cached_l = np.asarray(cached_labels, dtype=np.int64)
    derived_l = np.asarray(rederived_labels, dtype=np.int64)
    cached_p = np.asarray(cached_pose, dtype=np.float64)
    derived_p = np.asarray(rederived_pose, dtype=np.float64)
    if cached_l.shape != derived_l.shape or cached_p.shape != derived_p.shape:
        raise ValueError("cached/rederived target shapes differ")
    seg_mismatch = cached_l != derived_l
    pose_delta = cached_p - derived_p
    return {
        "seg_mismatched_pixels": int(np.count_nonzero(seg_mismatch)),
        "seg_total_pixels": int(seg_mismatch.size),
        "seg_mismatch_fraction": float(np.mean(seg_mismatch)),
        "pose_mismatched_elements": int(np.count_nonzero(pose_delta)),
        "pose_total_elements": int(pose_delta.size),
        "pose_max_abs": float(np.max(np.abs(pose_delta), initial=0.0)),
        "pose_mse": float(np.mean(pose_delta**2)),
    }


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _as_pair_b2chw(frames: tuple[np.ndarray, np.ndarray]):
    import torch

    array = np.stack(frames, axis=0)
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().float()


def _shared_resize(pair_b2chw):
    import torch.nn.functional as functional

    return functional.interpolate(
        pair_b2chw,
        size=SCORER_HW,
        mode="bilinear",
        align_corners=False,
    )


def _pose_first_six(output: Any):
    pose = output["pose"] if isinstance(output, dict) else output
    if pose.ndim != 2 or pose.shape[1] < 6:
        raise MeasurementError(f"PoseNet output does not contain six outputs: {tuple(pose.shape)}")
    return pose[:, :6]


def _forward_shared(pair_b2chw, *, segnet, posenet):
    from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

    shared = _shared_resize(pair_b2chw)
    seg_logits = segnet(shared[1:2])
    yuv = differentiable_rgb_to_yuv6(shared)
    pose_input = yuv.reshape(1, 12, yuv.shape[-2], yuv.shape[-1])
    pose = _pose_first_six(posenet(pose_input))
    return shared, seg_logits, pose


def _winner_rival_zero_margin_hinge(logits, labels):
    import torch

    target = logits.gather(1, labels[:, None]).squeeze(1)
    mask = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1]).permute(0, 3, 1, 2).bool()
    rival = logits.masked_fill(mask, -torch.inf).amax(dim=1)
    return torch.relu(rival - target).mean()


def _pair_vjps(pair_uint8, labels_np, pose_target_np, *, segnet, posenet):
    import torch

    pair = _as_pair_b2chw(pair_uint8).requires_grad_(True)
    labels = torch.from_numpy(np.asarray(labels_np, dtype=np.int64))[None]
    pose_target = torch.from_numpy(np.asarray(pose_target_np, dtype=np.float32))[None]
    _shared, logits, pose = _forward_shared(pair, segnet=segnet, posenet=posenet)
    seg_loss = _winner_rival_zero_margin_hinge(logits, labels)
    pose_loss = torch.mean((pose - pose_target) ** 2)
    g_seg = torch.autograd.grad(seg_loss, pair, retain_graph=True)[0]
    g_pose = torch.autograd.grad(pose_loss, pair)[0]
    if float(g_seg[0].abs().max()) != 0.0:
        raise MeasurementError("SegNet frame-0 gradient is nonzero; exact factorization violated")
    return (
        float(seg_loss.detach()),
        float(pose_loss.detach()),
        g_seg.detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.float32),
        g_pose.detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.float32),
    )


def _apply_lsb(pair: tuple[np.ndarray, np.ndarray], direction: np.ndarray, sign: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.stack(pair, axis=0).astype(np.int16)
    moved = np.clip(base + int(sign) * direction.astype(np.int16), 0, 255).astype(np.uint8)
    return moved[0], moved[1]


def _score_pairs_padded32(pairs, labels, pose_targets, *, segnet, posenet) -> dict[str, float]:
    import torch

    seg_errors: list[float] = []
    pose_errors: list[float] = []
    for start in range(0, len(pairs), SCORER_BATCH_SIZE):
        chunk_pairs = pairs[start : start + SCORER_BATCH_SIZE]
        chunk_labels = labels[start : start + SCORER_BATCH_SIZE]
        chunk_pose = pose_targets[start : start + SCORER_BATCH_SIZE]
        n_real = len(chunk_pairs)
        if n_real == 0:
            continue
        while len(chunk_pairs) < SCORER_BATCH_SIZE:
            chunk_pairs.append(chunk_pairs[-1])
            chunk_labels.append(chunk_labels[-1])
            chunk_pose.append(chunk_pose[-1])
        array = np.stack([np.stack(pair, axis=0) for pair in chunk_pairs], axis=0)
        tensor = torch.from_numpy(array).permute(0, 1, 4, 2, 3).contiguous().float()
        flat = tensor.reshape(-1, 3, *CAMERA_HW)
        shared = _shared_resize(flat).reshape(SCORER_BATCH_SIZE, 2, 3, *SCORER_HW)
        with torch.inference_mode():
            logits = segnet(shared[:, 1])
            from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

            yuv = differentiable_rgb_to_yuv6(shared.reshape(-1, 3, *SCORER_HW))
            pose = _pose_first_six(posenet(yuv.reshape(SCORER_BATCH_SIZE, 12, 192, 256)))
        target_labels = torch.from_numpy(np.stack(chunk_labels).astype(np.int64))
        target_pose = torch.from_numpy(np.stack(chunk_pose).astype(np.float32))
        seg = (logits.argmax(dim=1) != target_labels).float().mean(dim=(1, 2))
        pose_dist = ((pose - target_pose) ** 2).mean(dim=1)
        seg_errors.extend(float(v) for v in seg[:n_real])
        pose_errors.extend(float(v) for v in pose_dist[:n_real])
    return {"d_seg": float(np.mean(seg_errors)), "d_pose": float(np.mean(pose_errors))}


def _derive_gt_targets_padded32(
    pairs,
    cached_labels,
    cached_pose,
    *,
    segnet,
    posenet,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Rederive selected GT targets with exact B32 duplicate-last scorer geometry."""

    import torch

    if not (1 <= len(pairs) <= SCORER_BATCH_SIZE):
        raise MeasurementError("GT target rederivation requires a 1..32-pair subset")
    padded = list(pairs)
    n_real = len(padded)
    while len(padded) < SCORER_BATCH_SIZE:
        padded.append(padded[-1])
    array = np.stack([np.stack(pair, axis=0) for pair in padded], axis=0)
    tensor = torch.from_numpy(array).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        labels = segnet(segnet.preprocess_input(tensor)).argmax(dim=1)[:n_real]
        poses = _pose_first_six(posenet(posenet.preprocess_input(tensor)))[:n_real]
    labels_np = labels.cpu().numpy().astype(np.int64)
    poses_np = poses.cpu().numpy().astype(np.float32)
    metrics = target_cache_mismatch_metrics(
        np.stack(cached_labels), labels_np, np.stack(cached_pose), poses_np
    )
    return list(labels_np), list(poses_np), metrics


def _load_checkpoint_and_decode(checkpoint: Path, pair_ids: tuple[int, ...], args):
    from tac.local_acceleration.torch_levelset_inflate import decode_levelset_torch
    from tools import levelset_byte_close_and_eval as byte_close

    with np.load(checkpoint, allow_pickle=False) as raw:
        payload_custody = inspect_checkpoint_payload_keys(tuple(raw.files))
        params, cfg = byte_close._load_levelset_ckpt(checkpoint.parent, checkpoint.name)
        if int(cfg["n_pairs"]) != N_PAIRS_TOTAL:
            raise MeasurementError("checkpoint is not a n600 renderer")
        basis = str(cfg.get("basis", byte_close.LEGACY_FOURIER_AB_CONTROL))
        if basis != byte_close.LEGACY_FOURIER_AB_CONTROL:
            raise MeasurementError(
                "canonical torch decoder in this tool currently supports the checkpoint's "
                "legacy Fourier A/B control only; refusing to misdecode another basis"
            )
        initial_so = byte_close.detect_self_orient(
            cfg,
            {"freq_across": 0.0, "freq_along": 0.0, "tau": 0.0, "iters": 0},
        )
        if initial_so["self_orient"]:
            required = ("__cfg_freq_across", "__cfg_freq_along")
            if (
                any(key not in raw for key in required)
                or args.so_tau is None
                or args.so_iters is None
            ):
                raise MeasurementError(
                    "self-orient checkpoint requires persisted freq fields and explicit "
                    "--so-tau/--so-iters"
                )
            overrides = {
                "freq_across": float(raw["__cfg_freq_across"]),
                "freq_along": float(raw["__cfg_freq_along"]),
                "tau": float(args.so_tau),
                "iters": int(args.so_iters),
            }
            so = byte_close.detect_self_orient(cfg, overrides)
        else:
            so = initial_so
    manifest = {
        "n_pairs": len(pair_ids),
        "n_hidden": int(cfg["n_hidden"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "activation": str(cfg["activation"]),
        "softmax_temp": float(cfg["softmax_temp"]),
        "chroma": bool(cfg["chroma"]),
        "wire_w0": float(cfg["wire_w0"]),
        "wire_s0": float(cfg["wire_s0"]),
        "hosc_beta": float(cfg["hosc_beta"]),
        "hosc_omega": float(cfg["hosc_omega"]),
        "bank_n_scales": int(cfg["bank_n_scales"]),
        "bank_n_orient0": int(cfg["bank_n_orient0"]),
        "bank_f0": float(cfg["bank_f0"]),
        "bank_base": float(cfg["bank_base"]),
        "bank_n_iso": int(cfg["bank_n_iso"]),
        "max_bank_freq": cfg["max_bank_freq"],
        "render_h": int(cfg["render_h"]),
        "render_w": int(cfg["render_w"]),
        "camera_h": CAMERA_HW[0],
        "camera_w": CAMERA_HW[1],
        "self_orient": bool(so["self_orient"]),
        "n_dir_freqs": int(so.get("n_dir_freqs", 0)),
        "so_freq_across": float(so.get("freq_across", 0.0)),
        "so_freq_along": float(so.get("freq_along", 0.0)),
        "so_tau": float(so.get("tau", 0.0)),
        "so_iters": int(so.get("iters", 0)),
    }
    code = np.concatenate([params["code"][2 * i : 2 * i + 2] for i in pair_ids], axis=0)
    decoded = decode_levelset_torch(
        manifest,
        {key: value for key, value in params.items() if key != "code"},
        code,
        device="cpu",
        dtype="fp32",
        return_frames=True,
    )
    return decoded["frames"], cfg, manifest, payload_custody


@dataclass
class PairWork:
    pair: tuple[np.ndarray, np.ndarray]
    labels: np.ndarray
    pose_target: np.ndarray
    g_seg: np.ndarray
    g_pose: np.ndarray


def _parse_support_fractions(text: str) -> tuple[float, ...]:
    values = tuple(float(token) for token in text.split(",") if token.strip())
    if not values or any(not math.isfinite(v) or not (0.0 < v <= 1.0) for v in values):
        raise argparse.ArgumentTypeError("support fractions must be comma-separated values in (0,1]")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("support fractions must be unique")
    return values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=538)
    parser.add_argument("--support-fractions", type=_parse_support_fractions, default=(1e-4, 1e-3))
    parser.add_argument("--so-tau", type=float)
    parser.add_argument("--so-iters", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    for path, label in ((args.checkpoint, "checkpoint"), (args.gt_cache, "gt cache"), (args.upstream, "upstream")):
        if not path.exists():
            raise MeasurementError(f"{label} does not exist: {path}")
    if not (1 <= args.n_sample <= 32):
        raise MeasurementError("n-sample must be 1..32; this is a labeled stride subset, not n600 evidence")
    if 1 < args.n_sample < 8:
        raise MeasurementError("n-sample 2..7 is neither liveness-only nor an advisory verdict")
    output = args.output.resolve()
    source_roots = {
        args.checkpoint.resolve().parent,
        args.gt_cache.resolve().parent,
        args.upstream.resolve(),
    }
    if any(output == root or output.is_relative_to(root) for root in source_roots):
        raise MeasurementError("output must not be written within checkpoint/cache/upstream inputs")

    import torch

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    pair_ids = deterministic_stride_sample(N_PAIRS_TOTAL, args.n_sample, args.seed)
    cache_maps = {
        name: mmap_stored_npy_member(args.gt_cache, f"{name}.npy")
        for name in ("n_pairs", "gt_f0", "gt_f1", "lstars", "margins", "gt_poses")
    }
    if int(np.asarray(cache_maps["n_pairs"]).reshape(-1)[0]) != N_PAIRS_TOTAL:
        raise MeasurementError("gt cache n_pairs is not 600")
    expected_shapes = {
        "gt_f0": (600, *CAMERA_HW, 3),
        "gt_f1": (600, *CAMERA_HW, 3),
        "lstars": (600, *SCORER_HW),
        "margins": (600, *SCORER_HW),
        "gt_poses": (600, 6),
    }
    for name, shape in expected_shapes.items():
        if cache_maps[name].shape != shape:
            raise MeasurementError(f"gt cache {name} shape {cache_maps[name].shape} != {shape}")

    upstream = args.upstream.resolve()
    model_paths = {
        "segnet": upstream / "models" / "segnet.safetensors",
        "posenet": upstream / "models" / "posenet.safetensors",
    }
    upstream_sources = {
        "modules_py": upstream / "modules.py",
        "frame_utils_py": upstream / "frame_utils.py",
        "evaluate_py": upstream / "evaluate.py",
    }
    for label, path in {**model_paths, **upstream_sources}.items():
        if not path.is_file():
            raise MeasurementError(f"required frozen authority input {label} is missing: {path}")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from tac.scorer import load_default_scorers

    posenet, segnet = load_default_scorers(upstream, device="cpu")
    gt_pairs = [
        (
            np.asarray(cache_maps["gt_f0"][pair_id], dtype=np.uint8),
            np.asarray(cache_maps["gt_f1"][pair_id], dtype=np.uint8),
        )
        for pair_id in pair_ids
    ]
    cached_labels = [
        np.asarray(cache_maps["lstars"][pair_id], dtype=np.int64) for pair_id in pair_ids
    ]
    cached_poses = [
        np.asarray(cache_maps["gt_poses"][pair_id], dtype=np.float32) for pair_id in pair_ids
    ]
    rederived_labels, rederived_poses, target_mismatch = _derive_gt_targets_padded32(
        gt_pairs,
        cached_labels,
        cached_poses,
        segnet=segnet,
        posenet=posenet,
    )
    frames, checkpoint_cfg, decode_manifest, checkpoint_payload = _load_checkpoint_and_decode(
        args.checkpoint, pair_ids, args
    )
    if len(frames) != len(pair_ids):
        raise MeasurementError("canonical decoder returned the wrong number of pairs")

    # Exact source-path parity of shared A and the differentiable YUV6 clone.
    from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

    first_pair = _as_pair_b2chw(frames[0])[None]
    flat = first_pair.reshape(-1, 3, *CAMERA_HW)
    shared = _shared_resize(flat).reshape(1, 2, 3, *SCORER_HW)
    with torch.inference_mode():
        seg_pre = segnet.preprocess_input(first_pair)
        pose_pre = posenet.preprocess_input(first_pair)
        clone_pose_pre = differentiable_rgb_to_yuv6(shared.reshape(-1, 3, *SCORER_HW)).reshape(
            1, 12, 192, 256
        )
    if not torch.equal(seg_pre, shared[:, 1]):
        raise MeasurementError("SegNet preprocess is not the sealed shared A")
    yuv6_max_abs = float((pose_pre - clone_pose_pre).abs().max())
    if yuv6_max_abs != 0.0:
        raise MeasurementError(f"differentiable YUV6 clone parity failed: max_abs={yuv6_max_abs}")

    works: list[PairWork] = []
    raw_ss = raw_sp = raw_pp = 0.0
    shared_sp = shared_ss = shared_pp = 0.0
    pose_frame0_norm2 = pose_total_norm2 = 0.0
    smooth_seg_losses: list[float] = []
    smooth_pose_losses: list[float] = []
    for pair, labels, pose_target in zip(
        frames, rederived_labels, rederived_poses, strict=True
    ):
        seg_loss, pose_loss, g_seg, g_pose = _pair_vjps(
            pair, labels, pose_target, segnet=segnet, posenet=posenet
        )
        smooth_seg_losses.append(seg_loss)
        smooth_pose_losses.append(pose_loss)
        raw_ss += float(np.vdot(g_seg, g_seg))
        raw_sp += float(np.vdot(g_seg, g_pose))
        raw_pp += float(np.vdot(g_pose, g_pose))
        shared_ss += float(np.vdot(g_seg[1], g_seg[1]))
        shared_sp += float(np.vdot(g_seg[1], g_pose[1]))
        shared_pp += float(np.vdot(g_pose[1], g_pose[1]))
        pose_frame0_norm2 += float(np.vdot(g_pose[0], g_pose[0]))
        pose_total_norm2 += float(np.vdot(g_pose, g_pose))
        works.append(PairWork(pair, labels, pose_target, g_seg, g_pose))

    # The reported objectives are sample means.  On the product render their
    # block gradients are g_i/n, hence every Gram entry is sum_i <g_i,g'_i>/n^2.
    raw_gram = sample_mean_product_gram(raw_ss, raw_sp, raw_pp, n_sample=len(works))
    shared_gram = sample_mean_product_gram(
        shared_ss, shared_sp, shared_pp, n_sample=len(works)
    )
    raw_ss, raw_sp, raw_pp = raw_gram[0][0], raw_gram[0][1], raw_gram[1][1]
    shared_ss, shared_sp, shared_pp = (
        shared_gram[0][0],
        shared_gram[0][1],
        shared_gram[1][1],
    )

    labels_all = [work.labels for work in works]
    pose_all = [work.pose_target for work in works]
    baseline_pairs = [work.pair for work in works]
    baseline = _score_pairs_padded32(
        baseline_pairs.copy(), labels_all.copy(), pose_all.copy(), segnet=segnet, posenet=posenet
    )
    from tac.canonical_equations.shared_resize_joint_coupling_20260718 import (
        joint_costate_coefficients,
        smooth_coupling_summary,
    )

    coeff = joint_costate_coefficients(baseline["d_pose"])
    response_rows: list[dict[str, Any]] = []
    for fraction in args.support_fractions:
        arms = {name: [] for name in ("seg_plus", "seg_minus", "pose_plus", "pose_minus", "joint_plus")}
        counts = {
            name: {"nonzero_requested": 0, "realized_changed": 0, "boundary_clipped": 0}
            for name in arms
        }
        for work in works:
            seg_dir = topk_sign_direction(work.g_seg, fraction)
            shared_pose_gradient = work.g_pose.copy()
            shared_pose_gradient[0] = 0.0
            pose_dir = topk_sign_direction(shared_pose_gradient, fraction)
            joint_gradient = (
                coeff["lambda_seg"] * work.g_seg
                + coeff["lambda_pose"] * shared_pose_gradient
            )
            joint_dir = topk_sign_direction(joint_gradient, fraction)
            directions = {
                "seg_plus": (seg_dir, +1),
                "seg_minus": (seg_dir, -1),
                "pose_plus": (pose_dir, +1),
                "pose_minus": (pose_dir, -1),
                "joint_plus": (joint_dir, +1),
            }
            for name, (direction, sign) in directions.items():
                arms[name].append(_apply_lsb(work.pair, direction, sign))
                row_counts = realized_lsb_counts(work.pair, direction, sign)
                for count_name, value in row_counts.items():
                    counts[name][count_name] += value
        scored = {
            name: _score_pairs_padded32(
                pairs.copy(), labels_all.copy(), pose_all.copy(), segnet=segnet, posenet=posenet
            )
            for name, pairs in arms.items()
        }
        aggregated = aggregate_response_metrics(baseline, **scored)
        response_rows.append(
            {
                "support_fraction": fraction,
                "camera_lsb": 1,
                "baseline": baseline,
                "arm_scores": scored,
                "realized_lsb_counts": counts,
                **aggregated,
            }
        )

    shared_priced = smooth_coupling_summary(
        shared_gram, d_pose_baseline=baseline["d_pose"]
    )
    full_pair_priced = smooth_coupling_summary(
        raw_gram, d_pose_baseline=baseline["d_pose"]
    )
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS,
        "evidence_status": (
            "LIVENESS_ONLY_NOT_A_MEASUREMENT_VERDICT" if args.n_sample == 1 else "MEASURED_ADVISORY_SUBSET"
        ),
        "n_pairs_total": N_PAIRS_TOTAL,
        "sample": {
            "method": "deterministic_cyclic_stride",
            "n_of_600": args.n_sample,
            "pair_ids": list(pair_ids),
            "seed": args.seed,
        },
        "input_custody": {
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "gt_cache_path": str(args.gt_cache.resolve()),
            "gt_cache_sha256": sha256_file(args.gt_cache),
            "segnet_path": str(model_paths["segnet"].resolve()),
            "segnet_sha256": sha256_file(model_paths["segnet"]),
            "posenet_path": str(model_paths["posenet"].resolve()),
            "posenet_sha256": sha256_file(model_paths["posenet"]),
            "checkpoint_n_pairs": int(checkpoint_cfg["n_pairs"]),
            "checkpoint_payload": checkpoint_payload,
        },
        "execution_custody": {
            "git_head": git_head,
            "argv": (
                list(sys.argv)
                if argv is None
                else [str(Path(__file__).resolve()), *(str(token) for token in argv)]
            ),
            "config": {
                "n_sample": args.n_sample,
                "seed": args.seed,
                "support_fractions": list(args.support_fractions),
                "so_tau": args.so_tau,
                "so_iters": args.so_iters,
                "torch_num_threads": 1,
                "torch_deterministic_algorithms": True,
            },
            "input_bytes": {
                "checkpoint": args.checkpoint.stat().st_size,
                "gt_cache": args.gt_cache.stat().st_size,
                "segnet": model_paths["segnet"].stat().st_size,
                "posenet": model_paths["posenet"].stat().st_size,
                **{name: path.stat().st_size for name, path in upstream_sources.items()},
            },
            "upstream_source_sha256": {
                name: sha256_file(path) for name, path in upstream_sources.items()
            },
            "upstream_source_paths": {
                name: str(path.resolve()) for name, path in upstream_sources.items()
            },
        },
        "evidence_labels": {
            "smooth_gram": "B1_LOCAL_DERIVED",
            "finite_response": "B32_DUPLICATE_LAST_SUBSET_ADVISORY",
            "contest_score": "NOT_MEASURED",
        },
        "gt_target_custody": {
            "targets_used": "rederived_from_gt_frames",
            "source_members": ["gt_f0.npy", "gt_f1.npy"],
            "scorer_batch_size": SCORER_BATCH_SIZE,
            "last_batch_padding": "duplicate-last then discard padded outputs",
            "cache_vs_rederived": target_mismatch,
        },
        "shared_A": {
            "seg_pose_operator_identical": True,
            "operator": "torch.nn.functional.interpolate(mode=bilinear,align_corners=False)",
            "camera_hw": list(CAMERA_HW),
            "scorer_hw": list(SCORER_HW),
            "seg_preprocess_tensor_equal": True,
            "pose_yuv6_clone_max_abs": yuv6_max_abs,
        },
        "decode": {
            "path": "tac.local_acceleration.torch_levelset_inflate.decode_levelset_torch",
            "device": "cpu",
            "dtype": "fp32",
            "manifest": decode_manifest,
        },
        "smooth_coupling": {
            "evidence_label": "B1_LOCAL_DERIVED",
            "aggregation": "product-render sample-mean Gram: sum_i inner_product(g_i,g'_i)/n^2",
            "primary_surface": "shared_frame1",
            "shared_frame1": shared_priced,
            "full_pair_context": full_pair_priced,
            "smooth_seg_loss_mean": float(np.mean(smooth_seg_losses)),
            "smooth_pose_loss_mean": float(np.mean(smooth_pose_losses)),
            "pose_gradient_frame0_share": pose_frame0_norm2 / max(pose_total_norm2, 1e-30),
            "seg_gradient_frame0_norm2": 0.0,
            "seg_row_label": "B1_LOCAL_DERIVED winner-rival zero-margin hinge; NOT exact d_seg",
        },
        "actual_response": {
            "evidence_label": "B32_DUPLICATE_LAST_SUBSET_ADVISORY",
            "native_or_full_n600_comparable": False,
            "scorer_batch_size": SCORER_BATCH_SIZE,
            "last_batch_padding": "duplicate-last then discard padded outputs",
            "perturbation_surface": "shared_frame1_only",
            "perturbation": (
                "top-absolute-gradient one-camera-LSB uint8 sign direction; frame0 held fixed"
            ),
            "by_support_fraction": response_rows,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "timings_seconds": {"total": time.monotonic() - started},
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "paid_dispatch": False,
        "trainer_activation": False,
        "research_only": True,
        "sacred_c2_mutated": False,
        "verdict_scope": (
            "INSTANCE x real n600-trained EMA checkpoint x labeled deterministic subset x "
            "macOS CPU frozen-scorer formulation; structural shared-A coupling plus subset finite "
            "secants only, not a family/paradigm or contest-authority verdict"
        ),
    }
    from tac.witness_dsl.shared_resize_joint_coupling_policy import validate_measurement_receipt

    validate_measurement_receipt(receipt)
    _atomic_json_write(args.output.resolve(), receipt)
    print(json.dumps({"status": "written", "output": str(args.output.resolve()), "n": args.n_sample}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MeasurementError",
    "aggregate_response_metrics",
    "deterministic_stride_sample",
    "inspect_checkpoint_payload_keys",
    "mmap_stored_npy_member",
    "realized_lsb_counts",
    "sample_mean_product_gram",
    "sha256_file",
    "target_cache_mismatch_metrics",
    "topk_sign_direction",
]
