#!/usr/bin/env python3
"""Lane 12 — NeRV mask codec trainer.

Trains a tiny coordinate-MLP (``tac.nerv_mask_codec.NeRVMaskCodec``) to overfit
the SegNet argmax mask sequence for a single 1200-frame video at 384×512. The
trained codec is encoded as an NRV2 self-describing payload and written
alongside provenance + metrics for the dispatch script to bundle into the
archive.

CLAUDE.md non-negotiables enforced:
    - CUDA-required default (``--device cuda``); MPS rejected at trainer
      construction. Explicit ``--device cpu`` opt-in for the unit-test path.
    - EMA decay 0.997 (canonical ``tac.training.EMA``); shadow is what gets
      shipped via ``trainer.encode()`` — NOT the live weights.
    - eval_roundtrip-aware: mask-CODEC layer uses cross-entropy on raw logits
      (no ``.round()`` zero-gradient bug). The eval-roundtrip 384→874→uint8→384
      simulation lives in the auth-eval stage that consumes the produced
      archive (delegated to ``experiments/contest_auth_eval.py``).
    - No score claim: this trainer writes empirical artifacts only. Exact CUDA
      auth eval remains a separate, explicitly gated archive path.
    - Provenance JSON written: git hash, GPU info, profile dict, byte counts.
    - No silent defaults: ``--profile`` required; CLI overrides are explicit.

Usage (dispatch via ``scripts/remote_lane_nerv.sh``):

    python experiments/train_nerv_mask.py \\
        --profile nerv_mask_lane_g_v3 \\
        --device cuda \\
        --gt-masks-source decoded-baseline \\
        --decoded-baseline-path experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \\
        --alpha-primitive-contract experiments/results/alpha_geo_contract.json \\
        --output-dir results/lane_12_nerv

Outputs (under ``--output-dir``):
    masks.nrv          — NRV2 self-describing payload (~12-23 KB)
    provenance.json    — git hash, GPU, profile, training metrics
    train_metrics.csv  — per-eval-step disagreement rate
"""

# CLAUDE.md non-negotiable: CUDA-required default. NEVER fall back to MPS.
# An explicit `--device cpu` exists ONLY to keep the unit-test path
# deterministic-bytes acceptable; CPU is NOT used for production dispatch.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import torch

# Add src/ to path BEFORE any tac imports so this works in detached/Vast.ai
# bootstraps where the package is editable-installed under src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.nerv_mask_codec import (  # noqa: E402
    NeRVMaskCodec,
    NeRVSamplingComponent,
    NeRVSamplingPool,
    NeRVMaskTrainer,
    encode_nerv_codec,
    nerv_codec_bytes,
)
from tac.profiles import PROFILES  # noqa: E402


def _load_segnet_argmax_masks(
    upstream_dir: Path,
    device: str,
    num_frames: int = 1200,
) -> torch.Tensor:
    """Run SegNet over upstream/videos/0.mkv and return (T, H, W) argmax masks.

    This is the contest-compliant compress-time path: SegNet is loaded ONCE
    here at compress time to extract ground-truth argmax labels for the
    NeRV trainer to overfit. Per CLAUDE.md "Strict scorer rule" the loaded
    SegNet is NOT shipped in archive.zip — it stays on the compress-time
    machine.

    Args:
        upstream_dir: directory containing videos/0.mkv + scorers.
        device: "cuda" required for production. CPU possible for tests.
        num_frames: total frames to extract (default 1200).

    Returns:
        (T, H, W) long tensor on CPU with class IDs in [0, NUM_CLASSES).
    """
    # Lazy import — SegNet load is heavy.
    from tac.scorer import load_differentiable_scorers

    print(f"[lane-12] loading SegNet from {upstream_dir} on {device} ...", flush=True)
    _posenet, segnet = load_differentiable_scorers(
        upstream_dir=str(upstream_dir),
        device=device,
    )
    segnet.eval()
    # Decode video frames at scorer resolution. Use ffmpeg-cpu pipe for
    # robustness; NVDEC is for batch training, not for one-shot extraction.
    import subprocess

    video = upstream_dir / "videos" / "0.mkv"
    if not video.exists():
        raise FileNotFoundError(f"upstream video not found: {video}")
    # Probe original size
    probe = subprocess.run(  # subprocess-no-check-OK: check=True is set on the same call (multi-line — scanner's regex doesn't span lines)
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    orig_w, orig_h = (int(x) for x in probe.stdout.strip().split(","))
    # Decode raw rgb24 then resize to scorer resolution per upstream evaluate.
    cmd = [
        "ffmpeg",
        "-i",
        str(video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-v",
        "error",
        "pipe:1",
    ]
    decode = subprocess.run(cmd, capture_output=True, timeout=600, check=True)
    raw = np.frombuffer(decode.stdout, dtype=np.uint8)
    frame_size = orig_h * orig_w * 3
    n_decoded = len(raw) // frame_size
    if n_decoded < num_frames:
        raise RuntimeError(
            f"video has {n_decoded} frames; expected at least {num_frames}"
        )
    frames = raw[: num_frames * frame_size].reshape(num_frames, orig_h, orig_w, 3)
    # Resize to scorer (384, 512) and run SegNet in batches.
    SCORER_H, SCORER_W = 384, 512
    masks_THW = torch.empty(num_frames, SCORER_H, SCORER_W, dtype=torch.long)
    BATCH = 16
    import torch.nn.functional as F  # noqa: N812

    print(
        f"[lane-12] extracting argmax masks for {num_frames} frames ...",
        flush=True,
    )
    with torch.no_grad():
        for start in range(0, num_frames, BATCH):
            end = min(start + BATCH, num_frames)
            chunk = (
                torch.from_numpy(frames[start:end].copy())
                .permute(0, 3, 1, 2)
                .float()
                .to(device)
            )  # (B, 3, H, W)
            chunk = F.interpolate(
                chunk, size=(SCORER_H, SCORER_W), mode="bilinear", align_corners=False
            )
            # SegNet's scorer API expects (B, T=1, C, H, W) at
            # preprocess_input, then its 2D encoder consumes (B, C, H, W).
            seg_in = segnet.preprocess_input(chunk.unsqueeze(1))
            seg_logits = segnet(seg_in)  # (B, 5, H, W)
            seg_argmax = seg_logits.argmax(dim=1).cpu().long()
            masks_THW[start:end] = seg_argmax
    return masks_THW


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validated_zip_infos(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for info in zf.infolist():
        member_path = PurePosixPath(info.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"unsafe archive member path: {info.filename!r}")
        if info.filename in infos:
            raise ValueError(f"duplicate archive member: {info.filename!r}")
        infos[info.filename] = info
    return infos


def _normalize_mask_tensor(masks: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(masks, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(masks).__name__}")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"{name} must have shape (T,H,W); got {tuple(masks.shape)}")
    if masks.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if torch.is_floating_point(masks):
        if not torch.equal(masks, masks.round()):
            raise ValueError(f"{name} floating tensor contains non-integer class IDs")
        masks = masks.round()
    masks = masks.detach().cpu().to(torch.long).contiguous()
    if int(masks.min().item()) < 0:
        raise ValueError(f"{name} contains negative class IDs")
    return masks


def _mask_tensor_sha256(masks: torch.Tensor) -> str:
    normalized = _normalize_mask_tensor(masks, name="mask_sha256")
    digest = hashlib.sha256()
    digest.update(str(tuple(normalized.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(b"\0")
    arr = normalized.contiguous().numpy()
    digest.update(memoryview(arr))
    return digest.hexdigest()


def _load_tensor_mask_file(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    if path.suffix == ".npz":
        data = np.load(path)
        for key in ("masks", "mask_classes", "class_ids"):
            if key in data:
                return torch.from_numpy(data[key])
        raise KeyError(f"{path} has no masks/mask_classes/class_ids array")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("masks", "mask_classes", "class_ids"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"{path} did not contain a mask tensor")


def _decode_baseline_mask_file(path: Path, *, expected_frames: int | None) -> torch.Tensor:
    if path.suffix in {".pt", ".pth", ".npy", ".npz"}:
        return _load_tensor_mask_file(path)
    if path.suffix.lower() in {".mkv", ".mp4", ".webm"}:
        from tac.mask_codec import decode_masks

        return decode_masks(path, expected_frames=expected_frames)
    raise ValueError(f"unsupported decoded-baseline mask path suffix for {path}")


def _load_decoded_baseline_masks(
    path: Path,
    *,
    archive_member: str | None = None,
    expected_frames: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load the exact decoded baseline mask stream for Alpha-Geo-1 training."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    metadata: dict[str, Any] = {
        "path": str(path),
        "source_size_bytes": int(path.stat().st_size),
        "source_sha256": _sha256_file(path),
    }
    if path.suffix == ".zip":
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(path, "r") as zf:
                infos = _validated_zip_infos(zf)
                resolved_member = archive_member or "masks.mkv"
                member_path = PurePosixPath(resolved_member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"unsafe archive member path: {resolved_member!r}")
                if resolved_member not in infos:
                    raise FileNotFoundError(f"{path} missing archive member {resolved_member!r}")
                data = zf.read(resolved_member)
                info = infos[resolved_member]
            local_member = Path(td) / member_path.name
            local_member.write_bytes(data)
            masks = _decode_baseline_mask_file(local_member, expected_frames=expected_frames)
        metadata.update(
            {
                "archive_member_requested": archive_member,
                "archive_member_resolved": resolved_member,
                "archive_member_size_bytes": int(info.file_size),
                "archive_member_compressed_bytes": int(info.compress_size),
                "archive_member_sha256": _sha256_bytes(data),
            }
        )
    else:
        masks = _decode_baseline_mask_file(path, expected_frames=expected_frames)

    normalized = _normalize_mask_tensor(masks, name="decoded_baseline_masks")
    metadata.update(
        {
            "decoded_mask_sha256": _mask_tensor_sha256(normalized),
            "decoded_mask_sha256_algo": "sha256(shape,dtype,contiguous-raw-bytes)",
            "decoded_mask_shape": [int(v) for v in normalized.shape],
            "decoded_mask_dtype": str(normalized.dtype),
        }
    )
    return normalized, metadata


def _validate_decoded_baseline_target_shape(
    masks: torch.Tensor,
    *,
    expected_frames: int,
    expected_height: int,
    expected_width: int,
) -> None:
    expected = (int(expected_frames), int(expected_height), int(expected_width))
    if tuple(masks.shape) != expected:
        raise ValueError(
            "decoded-baseline masks must match the requested scorer geometry; "
            f"got {tuple(masks.shape)}, expected {expected}"
        )


ALPHA_PRIMITIVE_CONTRACT_DIAGNOSTIC = "alpha_geo_primitive_contract_v1"
ALPHA_SAMPLING_POOL_SCHEMA = "alpha_geo_weighted_sampling_pool_scaffold_v1"
ALPHA_SAMPLING_DEFAULTS = {
    "uniform_base_weight": 1.0,
    "critical_boxes_weight": 3.0,
    "boundary_bands_weight": 2.0,
    "transition_endpoints_weight": 1.5,
    "max_critical_box_indices": 50_000,
    "max_boundary_band_indices": 50_000,
    "max_transition_endpoint_indices": 25_000,
    "max_critical_boxes": 64,
    "max_transition_pairs": 64,
    "max_boundary_frames": 256,
}


def _load_alpha_primitive_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    contract = json.loads(raw)
    if not isinstance(contract, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return contract, {
        "path": str(path),
        "size_bytes": int(len(raw)),
        "sha256": _sha256_bytes(raw),
    }


def _contract_baseline_source(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract.get("source", {})
    if not isinstance(source, dict):
        return {}
    baseline = source.get("baseline", {})
    return baseline if isinstance(baseline, dict) else {}


def _contract_decoded_shape(contract: dict[str, Any]) -> list[int] | None:
    baseline = _contract_baseline_source(contract)
    shape = baseline.get("decoded_mask_shape")
    if shape is None:
        top_shape = contract.get("shape", {})
        if isinstance(top_shape, dict) and {"frames", "height", "width"}.issubset(top_shape):
            shape = [top_shape["frames"], top_shape["height"], top_shape["width"]]
    if shape is None:
        return None
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError(f"contract decoded_mask_shape must be length 3; got {shape!r}")
    return [int(v) for v in shape]


def _record_contract_gate(
    gates: dict[str, Any],
    blockers: list[str],
    name: str,
    *,
    passed: bool,
    expected: Any = None,
    observed: Any = None,
    required: bool = True,
    skipped: bool = False,
) -> None:
    gates[name] = {
        "passed": bool(passed),
        "required": bool(required),
        "skipped": bool(skipped),
        "expected": expected,
        "observed": observed,
    }
    if required and not skipped and not passed:
        blockers.append(name)


def _validate_alpha_primitive_contract(
    contract: dict[str, Any],
    *,
    contract_metadata: dict[str, Any],
    masks_THW: torch.Tensor,
    decoded_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Fail-closed custody validation for Alpha-Geo decoded-baseline training."""
    normalized = _normalize_mask_tensor(masks_THW, name="alpha_contract_target_masks")
    target_sha = _mask_tensor_sha256(normalized)
    target_shape = [int(v) for v in normalized.shape]
    baseline = _contract_baseline_source(contract)
    gates: dict[str, Any] = {}
    blockers: list[str] = []

    _record_contract_gate(
        gates,
        blockers,
        "schema_diagnostic",
        passed=contract.get("diagnostic") == ALPHA_PRIMITIVE_CONTRACT_DIAGNOSTIC,
        expected=ALPHA_PRIMITIVE_CONTRACT_DIAGNOSTIC,
        observed=contract.get("diagnostic"),
    )
    _record_contract_gate(
        gates,
        blockers,
        "schema_version",
        passed=int(contract.get("schema_version", -1)) == 1,
        expected=1,
        observed=contract.get("schema_version"),
    )
    for key in ("promotion_eligible", "score_claim_eligible", "exact_eval_claim"):
        _record_contract_gate(
            gates,
            blockers,
            f"{key}_false",
            passed=contract.get(key) is False,
            expected=False,
            observed=contract.get(key),
        )

    contract_shape = _contract_decoded_shape(contract)
    _record_contract_gate(
        gates,
        blockers,
        "decoded_mask_shape_match",
        passed=contract_shape == target_shape,
        expected=target_shape,
        observed=contract_shape,
    )
    contract_sha = baseline.get("decoded_mask_sha256")
    _record_contract_gate(
        gates,
        blockers,
        "decoded_mask_sha256_match",
        passed=contract_sha == target_sha,
        expected=target_sha,
        observed=contract_sha,
    )

    optional_pairs = (
        ("source_archive_sha256_match", "archive_sha256", "source_sha256"),
        ("source_archive_member_sha256_match", "archive_member_sha256", "archive_member_sha256"),
        ("source_archive_member_match", "archive_member", "archive_member_resolved"),
    )
    for gate_name, contract_key, decoded_key in optional_pairs:
        expected = baseline.get(contract_key)
        observed = decoded_metadata.get(decoded_key)
        skipped = expected is None or observed is None
        _record_contract_gate(
            gates,
            blockers,
            gate_name,
            passed=True if skipped else expected == observed,
            expected=expected,
            observed=observed,
            required=False,
            skipped=skipped,
        )

    result = {
        "schema": "alpha_primitive_contract_consumption_gates_v1",
        "contract_path": contract_metadata["path"],
        "contract_sha256": contract_metadata["sha256"],
        "target_decoded_mask_sha256": target_sha,
        "target_decoded_mask_shape": target_shape,
        "overall_passed": not blockers,
        "blockers": blockers,
        "gates": gates,
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "exact_eval_claim": False,
    }
    if blockers:
        raise ValueError(
            "alpha primitive contract validation failed: " + ", ".join(blockers)
        )
    return result


def _deterministic_positions(total: int, budget: int) -> torch.Tensor:
    total = int(total)
    budget = int(budget)
    if total <= 0 or budget <= 0:
        return torch.empty(0, dtype=torch.long)
    budget = min(total, budget)
    if budget == total:
        return torch.arange(total, dtype=torch.long)
    if budget == 1:
        return torch.zeros(1, dtype=torch.long)
    return torch.linspace(0, total - 1, budget).round().long().unique(sorted=True)


def _contract_frames(row: dict[str, Any], total_frames: int, *, max_frames: int) -> list[int]:
    raw_frames = row.get("frames")
    if raw_frames is None and row.get("frame") is not None:
        raw_frames = [row["frame"]]
    if raw_frames is None:
        return list(range(min(total_frames, max_frames)))
    if not isinstance(raw_frames, (list, tuple)):
        raw_frames = [raw_frames]
    frames = [int(v) for v in raw_frames]
    if len(frames) == 2 and frames[1] > frames[0] + 1:
        span = frames[1] - frames[0] + 1
        offsets = _deterministic_positions(span, min(span, max_frames))
        frames = [frames[0] + int(v) for v in offsets.tolist()]
    clipped = sorted({f for f in frames if 0 <= f < total_frames})
    return clipped[:max_frames]


def _contract_box_xyxy(row: dict[str, Any], *, height: int, width: int) -> tuple[int, int, int, int] | None:
    box = row.get("box_xyxy") or row.get("candidate_box_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x0 = max(0, min(width - 1, int(float(box[0]))))
    y0 = max(0, min(height - 1, int(float(box[1]))))
    x1 = max(x0 + 1, min(width, int(float(box[2]))))
    y1 = max(y0 + 1, min(height, int(float(box[3]))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _box_flat_indices(
    *,
    frames: list[int],
    box_xyxy: tuple[int, int, int, int],
    height: int,
    width: int,
    max_indices: int,
) -> torch.Tensor:
    x0, y0, x1, y1 = box_xyxy
    box_h = y1 - y0
    box_w = x1 - x0
    per_frame = box_h * box_w
    total = len(frames) * per_frame
    positions = _deterministic_positions(total, max_indices)
    if positions.numel() == 0:
        return positions
    frame_lookup = torch.tensor(frames, dtype=torch.long)
    frame_ids = frame_lookup[positions // per_frame]
    rem = positions % per_frame
    y = y0 + (rem // box_w)
    x = x0 + (rem % box_w)
    return frame_ids * (height * width) + y * width + x


def _critical_box_pool_indices(
    contract: dict[str, Any],
    *,
    height: int,
    width: int,
    frames: int,
    max_indices: int,
) -> torch.Tensor:
    rows = contract.get("ranked_critical_boxes", [])
    if not isinstance(rows, list) or not rows:
        return torch.empty(0, dtype=torch.long)
    rows = rows[: int(ALPHA_SAMPLING_DEFAULTS["max_critical_boxes"])]
    per_row_budget = max(1, int(max_indices) // max(1, len(rows)))
    parts: list[torch.Tensor] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        box = _contract_box_xyxy(row, height=height, width=width)
        if box is None:
            continue
        row_frames = _contract_frames(row, frames, max_frames=frames)
        if not row_frames:
            continue
        parts.append(
            _box_flat_indices(
                frames=row_frames,
                box_xyxy=box,
                height=height,
                width=width,
                max_indices=per_row_budget,
            )
        )
    if not parts:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(parts).unique(sorted=True)[: int(max_indices)]


def _contract_protected_classes(contract: dict[str, Any], recipe: dict[str, Any] | None = None) -> list[int]:
    if recipe is not None and isinstance(recipe.get("protected_classes"), list):
        return [int(v) for v in recipe["protected_classes"]]
    protected = []
    for row in contract.get("protected_classes", []):
        if isinstance(row, dict) and row.get("class_id") is not None:
            protected.append(int(row["class_id"]))
    return protected or [1, 2]


def _frame_boundary_mask(mask_HW: torch.Tensor, protected_classes: list[int]) -> torch.Tensor:
    boundary = torch.zeros_like(mask_HW, dtype=torch.bool)
    diff_x = mask_HW[:, 1:] != mask_HW[:, :-1]
    boundary[:, 1:] |= diff_x
    boundary[:, :-1] |= diff_x
    diff_y = mask_HW[1:, :] != mask_HW[:-1, :]
    boundary[1:, :] |= diff_y
    boundary[:-1, :] |= diff_y
    protected = torch.zeros_like(boundary)
    for class_id in protected_classes:
        protected |= mask_HW == int(class_id)
    return boundary & protected


def _dilate_bool_mask(mask_HW: torch.Tensor, radius: int) -> torch.Tensor:
    if int(radius) <= 0:
        return mask_HW
    import torch.nn.functional as F  # noqa: N812

    return (
        F.max_pool2d(
            mask_HW.float().view(1, 1, *mask_HW.shape),
            kernel_size=2 * int(radius) + 1,
            stride=1,
            padding=int(radius),
        )[0, 0]
        > 0
    )


def _boundary_band_pool_indices(
    contract: dict[str, Any],
    masks_THW: torch.Tensor,
    *,
    max_indices: int,
) -> torch.Tensor:
    recipes = contract.get("decoded_baseline_boundary_recipes", [])
    if not isinstance(recipes, list) or not recipes:
        return torch.empty(0, dtype=torch.long)
    T, H, W = (int(v) for v in masks_THW.shape)
    max_frames = min(T, int(ALPHA_SAMPLING_DEFAULTS["max_boundary_frames"]))
    frame_ids = _deterministic_positions(T, max_frames).tolist()
    radii = []
    protected_by_radius: dict[int, list[int]] = {}
    for recipe in recipes:
        if not isinstance(recipe, dict) or recipe.get("radius_px") is None:
            continue
        radius = int(recipe["radius_px"])
        radii.append(radius)
        protected_by_radius[radius] = _contract_protected_classes(contract, recipe)
    radii = sorted(set(radii))
    if not radii or not frame_ids:
        return torch.empty(0, dtype=torch.long)
    per_radius_budget = max(1, int(max_indices) // len(radii))
    per_frame_budget = max(1, per_radius_budget // len(frame_ids))
    parts: list[torch.Tensor] = []
    for radius in radii:
        radius_parts: list[torch.Tensor] = []
        protected_classes = protected_by_radius.get(radius, [1, 2])
        for frame_id in frame_ids:
            base = _frame_boundary_mask(masks_THW[int(frame_id)], protected_classes)
            band = _dilate_bool_mask(base, radius)
            local = band.reshape(-1).nonzero(as_tuple=False).reshape(-1)
            if local.numel() == 0:
                continue
            selected = local[_deterministic_positions(int(local.numel()), per_frame_budget)]
            radius_parts.append(int(frame_id) * (H * W) + selected)
        if radius_parts:
            parts.append(torch.cat(radius_parts).unique(sorted=True)[:per_radius_budget])
    if not parts:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(parts).unique(sorted=True)[: int(max_indices)]


def _transition_endpoint_pool_indices(
    contract: dict[str, Any],
    *,
    height: int,
    width: int,
    frames: int,
    max_indices: int,
) -> torch.Tensor:
    rows = contract.get("worst_transition_pairs", [])
    if not isinstance(rows, list) or not rows:
        return torch.empty(0, dtype=torch.long)
    endpoint_frames: list[int] = []
    for row in rows[: int(ALPHA_SAMPLING_DEFAULTS["max_transition_pairs"])]:
        if isinstance(row, dict):
            endpoint_frames.extend(_contract_frames(row, frames, max_frames=2))
    endpoint_frames = sorted({f for f in endpoint_frames if 0 <= f < frames})
    if not endpoint_frames:
        return torch.empty(0, dtype=torch.long)
    per_frame_budget = max(1, int(max_indices) // len(endpoint_frames))
    parts = []
    for frame_id in endpoint_frames:
        positions = _deterministic_positions(height * width, per_frame_budget)
        parts.append(int(frame_id) * (height * width) + positions)
    return torch.cat(parts).unique(sorted=True)[: int(max_indices)]


def _uniform_sampling_provenance(*, seed: int) -> dict[str, Any]:
    return {
        "schema": ALPHA_SAMPLING_POOL_SCHEMA,
        "mode": "uniform_default",
        "deterministic": True,
        "seed": int(seed),
        "applied_to_training": False,
        "components": [
            {
                "name": "uniform_base",
                "weight": 1.0,
                "active": True,
                "flat_index_count": None,
                "source": "implicit_full_mask_uniform",
            }
        ],
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "exact_eval_claim": False,
    }


def _build_alpha_primitive_sampling_pool(
    contract: dict[str, Any],
    masks_THW: torch.Tensor,
    *,
    seed: int,
    contract_sha256: str,
) -> tuple[NeRVSamplingPool, dict[str, Any]]:
    T, H, W = (int(v) for v in masks_THW.shape)
    critical = _critical_box_pool_indices(
        contract,
        height=H,
        width=W,
        frames=T,
        max_indices=int(ALPHA_SAMPLING_DEFAULTS["max_critical_box_indices"]),
    )
    boundary = _boundary_band_pool_indices(
        contract,
        masks_THW,
        max_indices=int(ALPHA_SAMPLING_DEFAULTS["max_boundary_band_indices"]),
    )
    transitions = _transition_endpoint_pool_indices(
        contract,
        height=H,
        width=W,
        frames=T,
        max_indices=int(ALPHA_SAMPLING_DEFAULTS["max_transition_endpoint_indices"]),
    )
    pool_components = [
        NeRVSamplingComponent(
            name="uniform_base",
            weight=float(ALPHA_SAMPLING_DEFAULTS["uniform_base_weight"]),
            flat_indices=None,
        )
    ]
    component_records: list[dict[str, Any]] = [
        {
            "name": "uniform_base",
            "weight": float(ALPHA_SAMPLING_DEFAULTS["uniform_base_weight"]),
            "active": True,
            "flat_index_count": None,
            "source": "implicit_full_mask_uniform",
        }
    ]
    weighted_sources = (
        (
            "critical_boxes",
            float(ALPHA_SAMPLING_DEFAULTS["critical_boxes_weight"]),
            critical,
            "contract.ranked_critical_boxes",
        ),
        (
            "boundary_bands",
            float(ALPHA_SAMPLING_DEFAULTS["boundary_bands_weight"]),
            boundary,
            "contract.decoded_baseline_boundary_recipes",
        ),
        (
            "transition_endpoints",
            float(ALPHA_SAMPLING_DEFAULTS["transition_endpoints_weight"]),
            transitions,
            "contract.worst_transition_pairs",
        ),
    )
    for name, weight, indices, source in weighted_sources:
        active = int(indices.numel()) > 0
        if active:
            pool_components.append(
                NeRVSamplingComponent(name=name, weight=weight, flat_indices=indices)
            )
        component_records.append(
            {
                "name": name,
                "weight": weight,
                "active": active,
                "flat_index_count": int(indices.numel()),
                "source": source,
            }
        )
    pool = NeRVSamplingPool(components=tuple(pool_components))
    return pool, {
        "schema": ALPHA_SAMPLING_POOL_SCHEMA,
        "mode": "alpha_primitive_contract_weighted",
        "contract_sha256": contract_sha256,
        "deterministic": True,
        "seed": int(seed),
        "applied_to_training": True,
        "config": dict(ALPHA_SAMPLING_DEFAULTS),
        "components": component_records,
        "promotion_eligible": False,
        "score_claim_eligible": False,
        "exact_eval_claim": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Lane 12 NeRV mask codec trainer")
    p.add_argument("--profile", required=True, help="Profile name from tac.profiles.PROFILES")
    p.add_argument(
        "--device",
        required=True,
        choices=["cuda", "cpu"],
        help="cuda for production; cpu allowed for unit tests only",
    )
    p.add_argument(
        "--upstream",
        type=Path,
        default=_REPO_ROOT / "upstream",
        help="upstream directory containing videos/0.mkv + scorers",
    )
    p.add_argument(
        "--gt-masks-source",
        choices=["segnet", "amrc", "decoded-baseline", "synthetic"],
        default="segnet",
        help=(
            "segnet (production: extract from upstream/videos/0.mkv via SegNet),"
            " amrc (load from existing AMRC payload),"
            " decoded-baseline (decode baseline archive masks.mkv for Alpha-Geo-1),"
            " synthetic (unit-test stripes pattern)"
        ),
    )
    p.add_argument(
        "--amrc-path",
        type=Path,
        default=None,
        help="Path to existing masks.amrc (when --gt-masks-source=amrc)",
    )
    p.add_argument(
        "--decoded-baseline-path",
        type=Path,
        default=None,
        help=(
            "Path to baseline masks.mkv or an archive containing masks.mkv "
            "(required when --gt-masks-source=decoded-baseline)."
        ),
    )
    p.add_argument(
        "--decoded-baseline-member",
        type=str,
        default=None,
        help="Archive member to decode for --gt-masks-source=decoded-baseline; default masks.mkv.",
    )
    p.add_argument(
        "--alpha-primitive-contract",
        type=Path,
        default=None,
        help=(
            "Path to alpha_geo_primitive_contract_v1 JSON for decoded-baseline "
            "Alpha-Geo retraining. The decoded-mask SHA/shape must match the "
            "loaded decoded-baseline masks."
        ),
    )
    p.add_argument(
        "--allow-forensic-segnet-target",
        action="store_true",
        help=(
            "Forensic/debug override only: allow --gt-masks-source=segnet. "
            "Production dispatch must use decoded-baseline targets."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where masks.nrv + provenance.json + train_metrics.csv land",
    )
    p.add_argument(
        "--num-frames", type=int, default=1200, help="Total frame count (default 1200)"
    )
    p.add_argument(
        "--mask-height", type=int, default=384, help="Scorer-resolution H (default 384)"
    )
    p.add_argument(
        "--mask-width", type=int, default=512, help="Scorer-resolution W (default 512)"
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="SGD steps; if None, uses profile['nerv_steps']",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Evaluate disagreement every N steps; default = profile['nerv_eval_every']",
    )
    p.add_argument(
        "--weight-dtype",
        choices=["fp16", "int8"],
        default=None,
        help="Quantization for shipping; default = profile['nerv_weight_dtype']",
    )
    args = p.parse_args()

    if args.profile not in PROFILES:
        raise SystemExit(
            f"profile {args.profile!r} not in PROFILES; available: "
            f"{sorted(PROFILES.keys())}"
        )
    profile = PROFILES[args.profile]
    # Required NeRV knobs:
    nerv_keys = (
        "nerv_num_freqs",
        "nerv_hidden_dim",
        "nerv_depth",
        "nerv_num_classes",
        "nerv_learning_rate",
        "nerv_ema_decay",
        "nerv_batch_coords",
        "nerv_steps",
        "nerv_eval_every",
        "nerv_weight_dtype",
    )
    missing = [k for k in nerv_keys if k not in profile]
    if missing:
        raise SystemExit(
            f"profile {args.profile!r} missing NeRV keys: {missing}. "
            f"Use a profile registered for Lane 12 (e.g. nerv_mask_lane_g_v3)."
        )

    steps = int(args.steps if args.steps is not None else profile["nerv_steps"])
    eval_every = int(
        args.eval_every if args.eval_every is not None else profile["nerv_eval_every"]
    )
    weight_dtype = (
        args.weight_dtype if args.weight_dtype is not None else profile["nerv_weight_dtype"]
    )
    production_shape_requested = (
        int(args.num_frames) == 1200
        and int(args.mask_height) == 384
        and int(args.mask_width) == 512
    )
    production_training_requested = (
        args.device == "cuda"
        and production_shape_requested
        and steps == int(profile["nerv_steps"])
    )

    if args.gt_masks_source == "segnet" and not args.allow_forensic_segnet_target:
        raise SystemExit(
            "FATAL: --gt-masks-source=segnet is fail-closed for production "
            "Lane 12 retraining because it repeats the retired SegNet-target "
            "path. Use --gt-masks-source=decoded-baseline with "
            "--alpha-primitive-contract, or set --allow-forensic-segnet-target "
            "only for a documented forensic/debug rerun."
        )
    if args.alpha_primitive_contract is not None and args.gt_masks_source != "decoded-baseline":
        raise SystemExit(
            "--alpha-primitive-contract requires --gt-masks-source=decoded-baseline"
        )
    if (
        args.gt_masks_source == "decoded-baseline"
        and production_training_requested
        and args.alpha_primitive_contract is None
    ):
        raise SystemExit(
            "FATAL: decoded-baseline production retraining requires "
            "--alpha-primitive-contract so contract custody, sampling gates, "
            "and non-score provenance are recorded."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determinism (CLAUDE.md canonical pipeline standard)
    seed = int(profile.get("seed", 12))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "FATAL: --device cuda requested but torch.cuda.is_available() is False. "
                "CLAUDE.md FORBIDDEN PATTERN: NEVER fall back to MPS/CPU silently."
            )
        torch.cuda.manual_seed_all(seed)

    # ── Stage 1: load ground-truth argmax masks ──
    target_mask_metadata: dict[str, Any] = {"source": args.gt_masks_source}
    decoded_baseline_metadata: dict[str, Any] = {}
    if args.gt_masks_source == "segnet":
        masks_THW = _load_segnet_argmax_masks(
            args.upstream, device=args.device, num_frames=args.num_frames
        )
        target_mask_metadata.update(
            {
                "upstream": str(args.upstream),
                "video": str(args.upstream / "videos" / "0.mkv"),
                "forensic_debug_override": bool(args.allow_forensic_segnet_target),
                "promotion_eligible": False,
                "score_claim_eligible": False,
            }
        )
    elif args.gt_masks_source == "amrc":
        if args.amrc_path is None or not args.amrc_path.exists():
            raise SystemExit(
                "--gt-masks-source=amrc requires --amrc-path pointing to an "
                "existing masks.amrc file"
            )
        from tac.lossless.argmax_codec import decode_argmax_masks  # type: ignore

        masks_THW = decode_argmax_masks(args.amrc_path.read_bytes()).long()
        target_mask_metadata.update(
            {
                "path": str(args.amrc_path),
                "source_size_bytes": int(args.amrc_path.stat().st_size),
                "source_sha256": _sha256_file(args.amrc_path),
            }
        )
        if masks_THW.shape[0] != args.num_frames:
            print(
                f"[lane-12] amrc has {masks_THW.shape[0]} frames; expected "
                f"{args.num_frames}. Continuing with the smaller count.",
                flush=True,
            )
    elif args.gt_masks_source == "decoded-baseline":
        if args.decoded_baseline_path is None:
            raise SystemExit(
                "--gt-masks-source=decoded-baseline requires --decoded-baseline-path "
                "pointing to baseline masks.mkv or an archive containing masks.mkv"
            )
        masks_THW, decoded_metadata = _load_decoded_baseline_masks(
            args.decoded_baseline_path,
            archive_member=args.decoded_baseline_member,
            expected_frames=args.num_frames,
        )
        decoded_baseline_metadata = decoded_metadata
        _validate_decoded_baseline_target_shape(
            masks_THW,
            expected_frames=args.num_frames,
            expected_height=args.mask_height,
            expected_width=args.mask_width,
        )
        target_mask_metadata.update(decoded_metadata)
    else:  # synthetic
        # 4×8×8 stripes for unit-test path (matches tests in test_nerv_mask_codec.py)
        T = min(args.num_frames, 4)
        H = min(args.mask_height, 8)
        W = min(args.mask_width, 8)
        masks_THW = torch.zeros(T, H, W, dtype=torch.long)
        for t in range(T):
            cols = (torch.arange(W) + t) % int(profile["nerv_num_classes"])
            masks_THW[t] = cols.unsqueeze(0).expand(H, W)
        target_mask_metadata.update(
            {
                "pattern": "rolling_column_stripes",
                "requested_shape": [int(args.num_frames), int(args.mask_height), int(args.mask_width)],
            }
        )

    masks_THW = _normalize_mask_tensor(masks_THW, name="target_masks")
    max_class_id = int(masks_THW.max().item())
    if max_class_id >= int(profile["nerv_num_classes"]):
        raise SystemExit(
            f"target masks contain class ID {max_class_id}, but profile "
            f"nerv_num_classes={int(profile['nerv_num_classes'])}"
        )
    target_mask_sha256 = _mask_tensor_sha256(masks_THW)
    target_mask_metadata.update(
        {
            "target_mask_sha256": target_mask_sha256,
            "target_mask_sha256_algo": "sha256(shape,dtype,contiguous-raw-bytes)",
            "target_mask_shape": [int(v) for v in masks_THW.shape],
            "target_mask_dtype": str(masks_THW.dtype),
        }
    )
    alpha_contract_provenance: dict[str, Any] | None = None
    alpha_sampling_pool: NeRVSamplingPool | None = None
    alpha_sampling_provenance = _uniform_sampling_provenance(seed=seed)
    if args.alpha_primitive_contract is not None:
        contract, contract_metadata = _load_alpha_primitive_contract(
            args.alpha_primitive_contract
        )
        contract_gates = _validate_alpha_primitive_contract(
            contract,
            contract_metadata=contract_metadata,
            masks_THW=masks_THW,
            decoded_metadata=decoded_baseline_metadata,
        )
        alpha_sampling_pool, alpha_sampling_provenance = _build_alpha_primitive_sampling_pool(
            contract,
            masks_THW,
            seed=seed,
            contract_sha256=contract_metadata["sha256"],
        )
        alpha_contract_provenance = {
            **contract_metadata,
            "diagnostic": contract.get("diagnostic"),
            "schema_version": contract.get("schema_version"),
            "score_evidence_grade": contract.get("score_evidence_grade"),
            "threshold_gates": contract.get("threshold_gates", {}),
            "consumption_gates": contract_gates,
            "promotion_eligible": False,
            "score_claim_eligible": False,
            "exact_eval_claim": False,
        }

    print(
        f"[lane-12] masks shape: {tuple(masks_THW.shape)}, dtype={masks_THW.dtype}, "
        f"unique={sorted(masks_THW.unique().tolist())}, sha256={target_mask_sha256}",
        flush=True,
    )

    # ── Stage 2: build codec + trainer ──
    codec = NeRVMaskCodec(
        num_freqs=int(profile["nerv_num_freqs"]),
        hidden_dim=int(profile["nerv_hidden_dim"]),
        num_classes=int(profile["nerv_num_classes"]),
        depth=int(profile["nerv_depth"]),
        seed=seed,
    )
    trainer = NeRVMaskTrainer(
        codec=codec,
        device=args.device,
        learning_rate=float(profile["nerv_learning_rate"]),
        ema_decay=float(profile["nerv_ema_decay"]),
        seed=seed,
    )
    fp16_bytes = nerv_codec_bytes(codec, weight_dtype="fp16")
    int8_bytes = nerv_codec_bytes(codec, weight_dtype="int8")
    print(
        f"[lane-12] codec: params={codec.num_params()}, "
        f"fp16={fp16_bytes}B, int8={int8_bytes}B (excluding header+scale)",
        flush=True,
    )

    # ── Stage 3: training loop with periodic eval ──
    metrics_path = args.output_dir / "train_metrics.csv"
    metrics_writer = csv.writer(metrics_path.open("w", newline=""))
    metrics_writer.writerow(["step", "loss", "acc", "eval_disagreement_rate"])
    t0 = time.monotonic()
    best_disagreement = 1.0
    last_loss = float("nan")
    last_acc = float("nan")
    for step in range(1, steps + 1):
        m = trainer.step(
            masks_THW,
            batch_size=int(profile["nerv_batch_coords"]),
            sampling_pool=alpha_sampling_pool,
        )
        last_loss, last_acc = m["loss"], m["acc"]
        if step % eval_every == 0 or step == steps:
            ev = trainer.evaluate_argmax_disagreement(masks_THW)
            dr = ev["disagreement_rate"]
            best_disagreement = min(best_disagreement, dr)
            metrics_writer.writerow([step, last_loss, last_acc, dr])
            print(
                f"[lane-12] step {step}/{steps} loss={last_loss:.4f} "
                f"train_acc={last_acc:.3f} eval_disagree={dr:.4f} "
                f"(best={best_disagreement:.4f}) elapsed={time.monotonic() - t0:.1f}s",
                flush=True,
            )
        else:
            metrics_writer.writerow([step, last_loss, last_acc, ""])
    metrics_path.open("a").close()  # ensure flushed

    # ── Stage 4: encode EMA shadow + write artifacts ──
    blob = trainer.encode(weight_dtype=weight_dtype)
    out_payload = args.output_dir / "masks.nrv"
    out_payload.write_bytes(blob)
    elapsed = time.monotonic() - t0

    final_eval = trainer.evaluate_argmax_disagreement(masks_THW)

    # Round-trip sanity: decode the just-written payload + render argmax + compare
    from tac.nerv_mask_codec import decode_nerv_codec, render_mask_argmax  # noqa: E402

    decoded = decode_nerv_codec(blob)
    rendered = render_mask_argmax(
        decoded,
        num_frames=int(masks_THW.shape[0]),
        height=int(masks_THW.shape[1]),
        width=int(masks_THW.shape[2]),
        device=args.device,
    )
    rt_disagree = float((rendered.long() != masks_THW).float().mean().item())
    requested_shape = (int(args.num_frames), int(args.mask_height), int(args.mask_width))
    actual_shape = tuple(int(v) for v in masks_THW.shape)
    trainer_smoke_run = (
        args.device != "cuda"
        or actual_shape != (1200, 384, 512)
        or steps != int(profile["nerv_steps"])
    )
    trainer_non_promotable_reasons = [
        "trainer output is not score evidence; exact contest_auth_eval is required for score claims"
    ]
    if args.device != "cuda":
        trainer_non_promotable_reasons.append(f"training device is {args.device}, not cuda")
    if actual_shape != (1200, 384, 512):
        trainer_non_promotable_reasons.append(
            f"mask shape {actual_shape} is not the full contest scorer geometry"
        )
    if steps != int(profile["nerv_steps"]):
        trainer_non_promotable_reasons.append(
            f"steps={steps} differs from profile nerv_steps={int(profile['nerv_steps'])}"
        )
    if args.gt_masks_source == "segnet":
        trainer_non_promotable_reasons.append(
            "gt_masks_source=segnet is allowed only for forensic/debug reruns"
        )
    if alpha_contract_provenance is None:
        trainer_non_promotable_reasons.append(
            "alpha_geo_primitive_contract_v1 was not provided; weighted sampling "
            "contract gates were not consumed"
        )
    else:
        trainer_non_promotable_reasons.append(
            "alpha_geo_primitive_contract_v1 and weighted sampling are empirical "
            "training inputs only; they are not score evidence"
        )

    provenance = {
        "lane_name": "lane_12_nerv_mask_codec",
        "lane_script": "experiments/train_nerv_mask.py",
        "trainer_artifact_evidence_grade": "empirical",
        "trainer_score_claim_eligible": False,
        "trainer_exact_eval_requested": False,
        "trainer_score_claim_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "trainer_smoke_run": trainer_smoke_run,
        "trainer_non_promotable_reasons": trainer_non_promotable_reasons,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - elapsed)),
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "git_hash": os.environ.get("GIT_HASH", "no-git"),
        "gpu_name": os.environ.get("GPU_NAME", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "device": args.device,
        "profile": args.profile,
        "profile_nerv_keys": {k: profile[k] for k in nerv_keys},
        "seed": seed,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "num_frames": int(masks_THW.shape[0]),
        "mask_height": int(masks_THW.shape[1]),
        "mask_width": int(masks_THW.shape[2]),
        "requested_mask_shape": [int(v) for v in requested_shape],
        "gt_masks_source": args.gt_masks_source,
        "allow_forensic_segnet_target": bool(args.allow_forensic_segnet_target),
        "target_mask_sha256": target_mask_sha256,
        "target_mask_sha256_algo": "sha256(shape,dtype,contiguous-raw-bytes)",
        "target_mask_shape": [int(v) for v in masks_THW.shape],
        "target_mask_dtype": str(masks_THW.dtype),
        "target_mask_metadata": target_mask_metadata,
        "alpha_primitive_contract": alpha_contract_provenance,
        "alpha_sampling_pool": alpha_sampling_provenance,
        "weight_dtype": weight_dtype,
        "nrv_payload_bytes": len(blob),
        "nrv_payload_path": str(out_payload),
        "codec_params": codec.num_params(),
        "fp16_weights_bytes": fp16_bytes,
        "int8_weights_bytes": int8_bytes,
        "final_loss": last_loss,
        "final_acc": last_acc,
        "final_eval_disagreement_rate": final_eval["disagreement_rate"],
        "best_eval_disagreement_rate": best_disagreement,
        "roundtrip_disagreement_rate": rt_disagree,
        "predicted_band_bytes": [23000, 80000],  # KB
        "kill_criterion_bytes": 100000,
        "kill_criterion_segnet_delta": 0.25,  # +25% vs Lane G v3
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print(
        f"[lane-12] DONE: nrv={out_payload} ({len(blob)}B), "
        f"final_disagreement={final_eval['disagreement_rate']:.4f}, "
        f"roundtrip={rt_disagree:.4f}, "
        f"best={best_disagreement:.4f}, elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"RESULT_JSON {json.dumps({'lane': 'lane_12_nerv', 'bytes': len(blob), 'disagreement': final_eval['disagreement_rate']})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
