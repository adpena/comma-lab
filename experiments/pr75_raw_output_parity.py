#!/usr/bin/env python3
"""Local raw-output parity harness for public PR75 versus robust_current.

This tool is forensic only.  It does not run the scorer, does not dispatch
remote jobs, and records ``score_claim=false`` in its output.  The core check is
to inflate the same public PR75 payload through:

1. the public ``qpose14_r55_segactions_minp`` runtime contract; and
2. this repo's ``submissions/robust_current`` unpack/render path.

It compares decoded streams, pose precision, selected renderer-native pair
tensors, and selected camera-resolution raw bytes.  A third counterfactual path
feeds the public QP1 float32 pose values into the robust renderer so the report
can distinguish pose materialization drift from model or action semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEEP_DIR = REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_deep_codex"
DEFAULT_PUBLIC_ARCHIVE = DEEP_DIR / "downloads/pr75_pr67_qpose14_r55_segactions_minp_archive.zip"
DEFAULT_PUBLIC_INFLATE = (
    DEEP_DIR
    / "sources/pr75_head/submissions/qpose14_r55_segactions_minp/inflate.py"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr75_raw_output_parity_20260503_codex"
ROBUST_INFLATE_RENDERER = REPO_ROOT / "submissions/robust_current/inflate_renderer.py"
ROBUST_UNPACKER = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
PUBLIC_PR75_MASK_LEN = 219_472
PUBLIC_PR75_MODEL_LEN = 56_034
PUBLIC_PR75_ACTIONS_LEN = 236
PUBLIC_PR75_PAYLOAD_LEN = 276_641
PUBLIC_PR75_FIXED_SLICES = {
    # payload_bytes: (mask_br, renderer_br, actions_br)
    276_641: (PUBLIC_PR75_MASK_LEN, 56_034, 236),
    276_520: (PUBLIC_PR75_MASK_LEN, 55_914, 236),
    276_381: (PUBLIC_PR75_MASK_LEN, 55_756, 255),
    276_379: (PUBLIC_PR75_MASK_LEN, 55_756, 253),
    276_362: (PUBLIC_PR75_MASK_LEN, 55_756, 236),
    277_288: (PUBLIC_PR75_MASK_LEN, 55_756, 1_162),
}
OUT_W = 1164
OUT_H = 874


class NumericAccumulator:
    """Streaming numeric comparison over deterministic chunks."""

    def __init__(self, lhs_shape: list[int], rhs_shape: list[int]) -> None:
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape
        self.compared_values = 0
        self.changed_values = 0
        self.first_diff_flat_index: int | None = None
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.lhs_sha256 = hashlib.sha256()
        self.rhs_sha256 = hashlib.sha256()

    def update(self, lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
        lhs_arr = np.ascontiguousarray(lhs)
        rhs_arr = np.ascontiguousarray(rhs)
        self.lhs_sha256.update(lhs_arr.view(np.uint8).tobytes())
        self.rhs_sha256.update(rhs_arr.view(np.uint8).tobytes())
        metrics = numeric_metrics(lhs_arr, rhs_arr)
        if metrics["compared_values"] == 0:
            return metrics
        if metrics["first_diff_flat_index"] is not None and self.first_diff_flat_index is None:
            self.first_diff_flat_index = (
                self.compared_values + int(metrics["first_diff_flat_index"])
            )
        self.compared_values += int(metrics["compared_values"])
        self.changed_values += int(metrics["changed_values"])
        self.max_abs = max(self.max_abs, float(metrics["max_abs"]))
        self.sum_abs += float(metrics["mean_abs"]) * int(metrics["compared_values"])
        self.sum_sq += float(metrics["rms"]) ** 2 * int(metrics["compared_values"])
        return metrics

    def finish(self) -> dict[str, Any]:
        if self.compared_values == 0:
            mean_abs = 0.0
            rms = 0.0
        else:
            mean_abs = self.sum_abs / self.compared_values
            rms = float(np.sqrt(self.sum_sq / self.compared_values))
        lhs_values = int(np.prod(self.lhs_shape)) if self.lhs_shape else 0
        rhs_values = int(np.prod(self.rhs_shape)) if self.rhs_shape else 0
        return {
            "lhs_shape": self.lhs_shape,
            "rhs_shape": self.rhs_shape,
            "compared_values": self.compared_values,
            "exact_equal": bool(
                self.lhs_shape == self.rhs_shape
                and self.changed_values == 0
                and self.compared_values == lhs_values == rhs_values
            ),
            "changed_values": self.changed_values,
            "first_diff_flat_index": self.first_diff_flat_index,
            "max_abs": self.max_abs,
            "mean_abs": mean_abs,
            "rms": rms,
            "lhs_sha256": self.lhs_sha256.hexdigest(),
            "rhs_sha256": self.rhs_sha256.hexdigest(),
        }


class ByteAccumulator:
    """Streaming byte comparison with full-stream hashes."""

    def __init__(self) -> None:
        self.lhs_bytes = 0
        self.rhs_bytes = 0
        self.compared_bytes = 0
        self.changed_prefix_bytes = 0
        self.first_diff_offset: int | None = None
        self.max_abs_byte_delta = 0
        self.lhs_sha256 = hashlib.sha256()
        self.rhs_sha256 = hashlib.sha256()

    def update(self, lhs: bytes, rhs: bytes) -> dict[str, Any]:
        self.lhs_sha256.update(lhs)
        self.rhs_sha256.update(rhs)
        lhs_arr = np.frombuffer(lhs, dtype=np.uint8)
        rhs_arr = np.frombuffer(rhs, dtype=np.uint8)
        n = min(lhs_arr.size, rhs_arr.size)
        changed = 0
        first = None
        max_abs = 0
        if n:
            l = lhs_arr[:n]
            r = rhs_arr[:n]
            neq = l != r
            changed = int(np.count_nonzero(neq))
            if changed:
                first = int(np.argmax(neq))
            max_abs = int(np.max(np.abs(l.astype(np.int16) - r.astype(np.int16))))
        if first is not None and self.first_diff_offset is None:
            self.first_diff_offset = self.compared_bytes + first
        self.lhs_bytes += len(lhs)
        self.rhs_bytes += len(rhs)
        self.compared_bytes += n
        self.changed_prefix_bytes += changed
        self.max_abs_byte_delta = max(self.max_abs_byte_delta, max_abs)
        return {
            "lhs_bytes": len(lhs),
            "rhs_bytes": len(rhs),
            "compared_bytes": n,
            "exact_equal": len(lhs) == len(rhs) and changed == 0,
            "changed_prefix_bytes": changed,
            "first_diff_offset": first,
            "max_abs_byte_delta": max_abs,
            "lhs_sha256": sha256_bytes(lhs),
            "rhs_sha256": sha256_bytes(rhs),
        }

    def finish(self) -> dict[str, Any]:
        return {
            "lhs_bytes": self.lhs_bytes,
            "rhs_bytes": self.rhs_bytes,
            "compared_bytes": self.compared_bytes,
            "exact_equal": bool(
                self.lhs_bytes == self.rhs_bytes
                and self.compared_bytes == self.lhs_bytes
                and self.changed_prefix_bytes == 0
            ),
            "changed_prefix_bytes": self.changed_prefix_bytes,
            "first_diff_offset": self.first_diff_offset,
            "max_abs_byte_delta": self.max_abs_byte_delta,
            "lhs_sha256": self.lhs_sha256.hexdigest(),
            "rhs_sha256": self.rhs_sha256.hexdigest(),
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_single_payload_zip(path: Path, *, expected_member: str = "p") -> bytes:
    """Read a strict single-member public payload ZIP."""
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [expected_member]:
            raise ValueError(f"{path} must contain single member {expected_member!r}; got {names!r}")
        info = infos[0]
        member = info.filename
        if member.startswith("/") or "\\" in member or ".." in Path(member).parts:
            raise ValueError(f"unsafe ZIP member name: {member!r}")
        return zf.read(info)


def split_public_pr75_payload(payload: bytes) -> dict[str, bytes]:
    """Return charged Brotli slices for fixed public PR75 or public P3 payloads."""
    fixed_slices = PUBLIC_PR75_FIXED_SLICES.get(len(payload))
    if fixed_slices is not None:
        mask_len, model_len, actions_len = fixed_slices
        cursor = 0
        mask_br = payload[cursor : cursor + mask_len]
        cursor += mask_len
        model_br = payload[cursor : cursor + model_len]
        cursor += model_len
        actions_br = payload[cursor : cursor + actions_len]
        cursor += actions_len
        pose_br = payload[cursor:]
        return {
            "payload_format": b"public_pr75_fixed",
            "masks.mkv.br": mask_br,
            "renderer.bin.br": model_br,
            "seg_tile_actions.br": actions_br,
            "optimized_poses.bin.br": pose_br,
        }
    if payload.startswith(b"P3"):
        header_size = 2 + struct.calcsize("<IHH")
        if len(payload) <= header_size:
            raise ValueError("P3 payload too short")
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        cursor = header_size
        if min(mask_len, model_len, actions_len) <= 0:
            raise ValueError("P3 payload has an empty required slice")
        if cursor + mask_len + model_len + actions_len >= len(payload):
            raise ValueError("P3 payload leaves no pose slice")
        mask_br = payload[cursor : cursor + mask_len]
        cursor += mask_len
        model_br = payload[cursor : cursor + model_len]
        cursor += model_len
        actions_br = payload[cursor : cursor + actions_len]
        cursor += actions_len
        return {
            "payload_format": b"public_pr75_p3",
            "masks.mkv.br": mask_br,
            "renderer.bin.br": model_br,
            "seg_tile_actions.br": actions_br,
            "optimized_poses.bin.br": payload[cursor:],
        }
    raise ValueError(
        f"unsupported PR75 parity payload: len={len(payload)} prefix={payload[:8]!r}"
    )


def decode_public_qp1_pose_float32(
    data: bytes,
    *,
    pose_dim: int = 6,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
    pose_scale: float = 2048.0,
) -> np.ndarray:
    """Decode public runtime QP1/qpose uint16 poses exactly as PR75 does."""
    if data.startswith(b"QP1"):
        if len(data) < 5:
            raise ValueError("QP1 payload too short")
        first = np.frombuffer(data[3:5], dtype=np.uint16, count=1)[0]
        vals = [int(first)]
        cursor = 5
        while cursor < len(data):
            shift = 0
            acc = 0
            while True:
                if cursor >= len(data):
                    raise ValueError("truncated QP1 VLQ payload")
                byte = data[cursor]
                cursor += 1
                acc |= (byte & 0x7F) << shift
                if byte < 0x80:
                    break
                shift += 7
            delta = (acc >> 1) ^ -(acc & 1)
            vals.append(vals[-1] + delta)
        q_pose = np.zeros((len(vals), pose_dim), dtype=np.uint16)
        q_pose[:, 0] = np.asarray(vals, dtype=np.uint16)
    else:
        if len(data) % (pose_dim * 2) != 0:
            raise ValueError(
                f"raw qpose payload length {len(data)} is not divisible by pose_dim*2"
            )
        q_pose = np.frombuffer(data, dtype=np.uint16).reshape(-1, pose_dim)

    pose_np = np.empty(q_pose.shape, dtype=np.float32)
    pose_np[:, 0] = q_pose[:, 0].astype(np.float32) / velocity_scale + velocity_offset
    pose_np[:, 1:] = q_pose[:, 1:].view(np.int16).astype(np.float32) / pose_scale
    return pose_np


def fp16_materialize_pose(public_pose: np.ndarray) -> np.ndarray:
    """Mimic robust_current's decoded optimized_poses.bin precision boundary."""
    return public_pose.astype(np.float16).astype(np.float32)


def parse_pair_indices(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return []
    out: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        out.append(int(value))
    return out


def resolve_pair_indices(
    *,
    total_pairs: int,
    pair_indices: list[int],
    max_pairs: int | None,
) -> list[int]:
    if max_pairs is not None:
        if max_pairs < 0:
            raise ValueError("--max-pairs must be nonnegative")
        pair_indices = list(range(min(max_pairs, total_pairs)))
    if not pair_indices:
        pair_indices = [0, 33, 36, 587]
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in pair_indices:
        if idx < 0 or idx >= total_pairs:
            raise ValueError(f"pair index {idx} outside [0, {total_pairs})")
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def array_bytes(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).view(np.uint8).tobytes()


def array_sha256(arr: np.ndarray) -> str:
    return sha256_bytes(array_bytes(arr))


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def tensor_sha256(tensor: Any) -> str:
    return array_sha256(tensor_to_numpy(tensor))


def numeric_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    lhs_flat = np.ravel(lhs)
    rhs_flat = np.ravel(rhs)
    n = min(lhs_flat.size, rhs_flat.size)
    if n == 0:
        return {
            "lhs_shape": list(lhs.shape),
            "rhs_shape": list(rhs.shape),
            "compared_values": 0,
            "exact_equal": lhs.shape == rhs.shape,
        }
    l = lhs_flat[:n]
    r = rhs_flat[:n]
    changed_mask = l != r
    changed = int(np.count_nonzero(changed_mask))
    first = int(np.argmax(changed_mask)) if changed else None
    diff = l.astype(np.float64) - r.astype(np.float64)
    return {
        "lhs_shape": list(lhs.shape),
        "rhs_shape": list(rhs.shape),
        "compared_values": int(n),
        "exact_equal": bool(lhs.shape == rhs.shape and changed == 0),
        "changed_values": changed,
        "first_diff_flat_index": first,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff * diff))),
    }


def byte_metrics(lhs: bytes, rhs: bytes) -> dict[str, Any]:
    n = min(len(lhs), len(rhs))
    changed = sum(1 for i in range(n) if lhs[i] != rhs[i])
    first = next((i for i in range(n) if lhs[i] != rhs[i]), None)
    return {
        "lhs_bytes": len(lhs),
        "rhs_bytes": len(rhs),
        "compared_bytes": n,
        "exact_equal": len(lhs) == len(rhs) and changed == 0,
        "changed_prefix_bytes": changed,
        "first_diff_offset": first,
        "lhs_sha256": sha256_bytes(lhs),
        "rhs_sha256": sha256_bytes(rhs),
    }


def stream_summary(name: str, data: bytes) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "magic_hex": data[:8].hex(),
    }


def pose_precision_report(public_pose: np.ndarray, robust_pose: np.ndarray) -> dict[str, Any]:
    fp16_public = fp16_materialize_pose(public_pose)
    return {
        "public_qp1_float32": {
            "shape": list(public_pose.shape),
            "dtype": str(public_pose.dtype),
            "sha256": array_sha256(public_pose),
        },
        "robust_current_fp16_materialized": {
            "shape": list(robust_pose.shape),
            "dtype": str(robust_pose.dtype),
            "sha256": array_sha256(robust_pose),
        },
        "public_float32_vs_robust_fp16": numeric_metrics(public_pose, robust_pose),
        "robust_matches_public_fp16_roundtrip": bool(np.array_equal(fp16_public, robust_pose)),
        "public_float32_vs_public_fp16_roundtrip": numeric_metrics(public_pose, fp16_public),
    }


def load_public_masks(public_mod: Any, mask_obu: bytes, tmp_dir: Path) -> Any:
    mask_path = tmp_dir / "public_masks.obu"
    mask_path.write_bytes(mask_obu)
    masks = public_mod.load_encoded_mask_video(str(mask_path))
    if int(masks.shape[0]) < 600:
        repeat = int(np.ceil(600 / int(masks.shape[0])))
        masks = masks.repeat_interleave(repeat, dim=0)[:600].contiguous()
    return masks


def load_robust_masks(robust_mod: Any, mask_obu: bytes, tmp_dir: Path) -> tuple[Any, Any]:
    mask_path = tmp_dir / "masks.mkv"
    mask_path.write_bytes(mask_obu)
    masks = robust_mod._load_masks_from_archive(mask_path, expected_frames=1200)  # noqa: SLF001
    if getattr(masks, "_half_frame_only", False):
        expanded = masks.repeat_interleave(2, dim=0)
    else:
        expanded = masks
    return masks, expanded


def write_decoded_members(members: dict[str, bytes], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in members.items():
        rel = Path(name)
        target = out_dir / rel
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe decoded member name: {name!r}")
        target.write_bytes(data)


def load_public_renderer(public_mod: Any, renderer_qzs3: bytes, device: str) -> Any:
    import torch

    generator = public_mod.JointFrameGenerator().to(torch.device(device))
    state = public_mod.get_decoded_state_dict(renderer_qzs3, torch.device(device))
    generator.load_state_dict(state, strict=True)
    return generator.eval()


def render_public_pairs(
    *,
    public_mod: Any,
    generator: Any,
    masks: Any,
    poses_float32: np.ndarray,
    action_state: dict[str, Any] | None,
    pair_indices: list[int],
    device: str,
) -> tuple[Any, Any]:
    import torch

    before_rows = []
    after_rows = []
    tile_size = 32
    grid_w = 512 // tile_size
    with torch.inference_mode():
        for pair_idx in pair_indices:
            mask = masks[pair_idx : pair_idx + 1].to(device).long()
            pose = torch.from_numpy(poses_float32[pair_idx : pair_idx + 1]).to(device).float()
            fake1, fake2 = generator(mask, pose)
            before = torch.stack([fake1, fake2], dim=1)
            before_hwc = before.permute(0, 1, 3, 4, 2).contiguous()
            before_rows.append(before_hwc.cpu())
            if action_state is not None:
                for tile_id, action_id in action_state["by_frame"].get(pair_idx, []):
                    y0 = (tile_id // grid_w) * tile_size
                    x0 = (tile_id % grid_w) * tile_size
                    fake2[:, :, y0 : y0 + tile_size, x0 : x0 + tile_size] = (
                        fake2[:, :, y0 : y0 + tile_size, x0 : x0 + tile_size]
                        + action_state["deltas"][action_id]
                    ).clamp(0, 255)
            after = torch.stack([fake1, fake2], dim=1)
            after_hwc = after.permute(0, 1, 3, 4, 2).contiguous()
            after_rows.append(after_hwc.cpu())
    return torch.cat(before_rows, dim=0), torch.cat(after_rows, dim=0)


def render_robust_pairs(
    *,
    robust_mod: Any,
    renderer: Any,
    masks_expanded: Any,
    poses_float32: np.ndarray,
    action_state: dict[str, Any] | None,
    pair_indices: list[int],
    device: str,
) -> tuple[Any, Any]:
    import torch

    before_rows = []
    after_rows = []
    with torch.inference_mode():
        for pair_idx in pair_indices:
            mask_t = masks_expanded[2 * pair_idx : 2 * pair_idx + 1].to(device=device, dtype=torch.long)
            mask_t1 = masks_expanded[2 * pair_idx + 1 : 2 * pair_idx + 2].to(device=device, dtype=torch.long)
            pose = torch.from_numpy(poses_float32[pair_idx : pair_idx + 1]).to(device=device).float()
            pairs = renderer(mask_t, mask_t1, pose=pose)
            before_rows.append(pairs.detach().cpu())
            after = pairs.clone()
            robust_mod._apply_seg_tile_actions_to_pairs(  # noqa: SLF001
                after,
                action_state,
                pair_start=pair_idx,
            )
            after_rows.append(after.detach().cpu())
    return torch.cat(before_rows, dim=0), torch.cat(after_rows, dim=0)


def render_public_chunk(
    *,
    public_mod: Any,
    generator: Any,
    masks: Any,
    poses_float32: np.ndarray,
    action_state: dict[str, Any] | None,
    start_pair: int,
    end_pair: int,
    device: str,
) -> tuple[Any, Any]:
    import torch

    tile_size = 32
    grid_w = 512 // tile_size
    with torch.inference_mode():
        mask = masks[start_pair:end_pair].to(device).long()
        pose = torch.from_numpy(poses_float32[start_pair:end_pair]).to(device).float()
        fake1, fake2 = generator(mask, pose)
        before = torch.stack([fake1, fake2], dim=1)
        before_hwc = before.permute(0, 1, 3, 4, 2).contiguous().cpu()
        if action_state is not None:
            for batch_j, pair_idx in enumerate(range(start_pair, end_pair)):
                for tile_id, action_id in action_state["by_frame"].get(pair_idx, []):
                    y0 = (tile_id // grid_w) * tile_size
                    x0 = (tile_id % grid_w) * tile_size
                    fake2[batch_j, :, y0 : y0 + tile_size, x0 : x0 + tile_size] = (
                        fake2[batch_j, :, y0 : y0 + tile_size, x0 : x0 + tile_size]
                        + action_state["deltas"][action_id]
                    ).clamp(0, 255)
        after = torch.stack([fake1, fake2], dim=1)
        after_hwc = after.permute(0, 1, 3, 4, 2).contiguous().cpu()
    return before_hwc, after_hwc


def render_robust_chunk(
    *,
    robust_mod: Any,
    renderer: Any,
    masks_expanded: Any,
    poses_float32: np.ndarray,
    action_state: dict[str, Any] | None,
    start_pair: int,
    end_pair: int,
    device: str,
) -> tuple[Any, Any]:
    import torch

    with torch.inference_mode():
        mask_t = masks_expanded[2 * start_pair : 2 * end_pair : 2].to(
            device=device, dtype=torch.long
        )
        mask_t1 = masks_expanded[2 * start_pair + 1 : 2 * end_pair + 1 : 2].to(
            device=device, dtype=torch.long
        )
        pose = torch.from_numpy(poses_float32[start_pair:end_pair]).to(device=device).float()
        pairs = renderer(mask_t, mask_t1, pose=pose)
        before = pairs.detach().cpu()
        after = pairs.clone()
        robust_mod._apply_seg_tile_actions_to_pairs(  # noqa: SLF001
            after,
            action_state,
            pair_start=start_pair,
        )
        after_cpu = after.detach().cpu()
    return before, after_cpu


def raw_bytes_from_hwc_pairs(pairs_hwc: Any, *, out_h: int = OUT_H, out_w: int = OUT_W) -> bytes:
    import torch
    import torch.nn.functional as F

    n_pairs = int(pairs_hwc.shape[0])
    chw = pairs_hwc.permute(0, 1, 4, 2, 3).reshape(n_pairs * 2, 3, pairs_hwc.shape[2], pairs_hwc.shape[3])
    up = F.interpolate(chw.float(), size=(out_h, out_w), mode="bilinear", align_corners=False)
    out = up.round().clamp(0, 255).to(dtype=torch.uint8)
    hwc = out.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return hwc.tobytes()


def compare_tensor_pair(lhs: Any, rhs: Any) -> dict[str, Any]:
    lhs_np = tensor_to_numpy(lhs)
    rhs_np = tensor_to_numpy(rhs)
    return {
        "lhs_sha256": array_sha256(lhs_np),
        "rhs_sha256": array_sha256(rhs_np),
        "metrics": numeric_metrics(lhs_np, rhs_np),
    }


def per_pair_render_metrics(
    pair_indices: list[int],
    public_after: Any,
    robust_after: Any,
    robust_public_pose_after: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_idx, pair_idx in enumerate(pair_indices):
        public_pair = public_after[local_idx : local_idx + 1]
        robust_pair = robust_after[local_idx : local_idx + 1]
        override_pair = robust_public_pose_after[local_idx : local_idx + 1]
        public_raw = raw_bytes_from_hwc_pairs(public_pair)
        robust_raw = raw_bytes_from_hwc_pairs(robust_pair)
        override_raw = raw_bytes_from_hwc_pairs(override_pair)
        rows.append(
            {
                "pair_index": pair_idx,
                "public_vs_robust_current_native_after_actions": compare_tensor_pair(
                    public_pair, robust_pair
                )["metrics"],
                "public_vs_robust_current_raw_after_actions": byte_metrics(
                    public_raw, robust_raw
                ),
                "public_vs_robust_with_public_qp1_float32_pose_native_after_actions": compare_tensor_pair(
                    public_pair, override_pair
                )["metrics"],
                "public_vs_robust_with_public_qp1_float32_pose_raw_after_actions": byte_metrics(
                    public_raw, override_raw
                ),
            }
        )
    return rows


def compare_all_pairs_chunked(
    *,
    public_mod: Any,
    robust_mod: Any,
    public_generator: Any,
    robust_renderer: Any,
    public_masks: Any,
    robust_masks_expanded: Any,
    public_pose: np.ndarray,
    robust_pose: np.ndarray,
    public_actions: dict[str, Any] | None,
    robust_actions_current: dict[str, Any] | None,
    chunk_size: int,
    fast_fail: bool,
    device: str,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    total_pairs = int(public_pose.shape[0])
    public_mask_shape = [total_pairs] + list(tensor_to_numpy(public_masks[:1]).shape[1:])
    robust_pair_mask_shape = [
        total_pairs
    ] + list(tensor_to_numpy(robust_masks_expanded[1:2]).shape[1:])
    mask_acc = NumericAccumulator(public_mask_shape, robust_pair_mask_shape)
    native_before_acc: NumericAccumulator | None = None
    native_after_acc: NumericAccumulator | None = None
    raw_after_acc = ByteAccumulator()
    chunks: list[dict[str, Any]] = []
    failed_at_chunk: dict[str, Any] | None = None
    t0 = time.monotonic()

    for start in range(0, total_pairs, chunk_size):
        end = min(start + chunk_size, total_pairs)
        chunk_t0 = time.monotonic()
        public_mask_np = tensor_to_numpy(public_masks[start:end]).astype(np.uint8, copy=False)
        robust_mask_np = tensor_to_numpy(
            robust_masks_expanded[2 * start + 1 : 2 * end + 1 : 2]
        ).astype(np.uint8, copy=False)
        mask_metrics = mask_acc.update(public_mask_np, robust_mask_np)

        public_before, public_after = render_public_chunk(
            public_mod=public_mod,
            generator=public_generator,
            masks=public_masks,
            poses_float32=public_pose,
            action_state=public_actions,
            start_pair=start,
            end_pair=end,
            device=device,
        )
        robust_before, robust_after = render_robust_chunk(
            robust_mod=robust_mod,
            renderer=robust_renderer,
            masks_expanded=robust_masks_expanded,
            poses_float32=robust_pose,
            action_state=robust_actions_current,
            start_pair=start,
            end_pair=end,
            device=device,
        )

        public_before_np = tensor_to_numpy(public_before)
        robust_before_np = tensor_to_numpy(robust_before)
        public_after_np = tensor_to_numpy(public_after)
        robust_after_np = tensor_to_numpy(robust_after)
        if native_before_acc is None:
            native_before_acc = NumericAccumulator(
                [total_pairs] + list(public_before_np.shape[1:]),
                [total_pairs] + list(robust_before_np.shape[1:]),
            )
        if native_after_acc is None:
            native_after_acc = NumericAccumulator(
                [total_pairs] + list(public_after_np.shape[1:]),
                [total_pairs] + list(robust_after_np.shape[1:]),
            )
        native_before_metrics = native_before_acc.update(public_before_np, robust_before_np)
        native_after_metrics = native_after_acc.update(public_after_np, robust_after_np)
        public_raw = raw_bytes_from_hwc_pairs(public_after)
        robust_raw = raw_bytes_from_hwc_pairs(robust_after)
        raw_metrics = raw_after_acc.update(public_raw, robust_raw)

        chunk = {
            "start_pair": start,
            "end_pair_exclusive": end,
            "pair_count": end - start,
            "elapsed_seconds": time.monotonic() - chunk_t0,
            "mask_metrics": mask_metrics,
            "native_before_actions": native_before_metrics,
            "native_after_actions": native_after_metrics,
            "raw_after_actions": raw_metrics,
        }
        chunks.append(chunk)
        if fast_fail and not (
            mask_metrics["exact_equal"]
            and native_before_metrics["exact_equal"]
            and native_after_metrics["exact_equal"]
            and raw_metrics["exact_equal"]
        ):
            failed_at_chunk = chunk
            break

    native_before_final = (
        native_before_acc.finish()
        if native_before_acc is not None
        else {"exact_equal": total_pairs == 0, "compared_values": 0}
    )
    native_after_final = (
        native_after_acc.finish()
        if native_after_acc is not None
        else {"exact_equal": total_pairs == 0, "compared_values": 0}
    )
    raw_after_final = raw_after_acc.finish()
    mask_final = mask_acc.finish()
    completed = failed_at_chunk is None and (
        not chunks or int(chunks[-1]["end_pair_exclusive"]) == total_pairs
    )
    pose_metrics = numeric_metrics(public_pose, robust_pose)
    pose_metrics["lhs_sha256"] = array_sha256(public_pose)
    pose_metrics["rhs_sha256"] = array_sha256(robust_pose)
    controlled = bool(
        completed
        and mask_final["exact_equal"]
        and native_before_final["exact_equal"]
        and native_after_final["exact_equal"]
        and raw_after_final["exact_equal"]
        and pose_metrics["exact_equal"]
    )

    return {
        "enabled": True,
        "schema": "pr75_all_pairs_chunked_parity_v1",
        "pair_count": total_pairs,
        "chunk_size": chunk_size,
        "fast_fail": fast_fail,
        "completed": completed,
        "failed_at_chunk": failed_at_chunk,
        "elapsed_seconds": time.monotonic() - t0,
        "pose_public_qp1_float32_vs_robust_current": pose_metrics,
        "mask_public_vs_robust_pair_mask": mask_final,
        "render_public_vs_robust_current": {
            "native_before_actions": native_before_final,
            "native_after_actions": native_after_final,
            "raw_after_actions": raw_after_final,
        },
        "chunks": chunks,
        "current_runtime_pr75_comparisons_controlled": controlled,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    import brotli
    import torch

    t0 = time.monotonic()
    archive = Path(args.public_archive)
    public_inflate = Path(args.public_inflate_py)
    output_dir = Path(args.output_dir)
    payload = read_single_payload_zip(archive)
    charged = split_public_pr75_payload(payload)
    mask_obu = brotli.decompress(charged["masks.mkv.br"])
    renderer_qzs3 = brotli.decompress(charged["renderer.bin.br"])
    actions_raw = brotli.decompress(charged["seg_tile_actions.br"])
    pose_qp1 = brotli.decompress(charged["optimized_poses.bin.br"])

    robust_unpacker = load_module(ROBUST_UNPACKER, "pr75_parity_robust_unpacker")
    robust_header, robust_members = robust_unpacker._parse_payload(payload)  # noqa: SLF001
    public_pose = decode_public_qp1_pose_float32(pose_qp1)
    if "optimized_poses.qp1" in robust_members:
        robust_pose = decode_public_qp1_pose_float32(robust_members["optimized_poses.qp1"])
        robust_pose_runtime = "QP1 decoded directly to float32 before JointFrameGenerator"
    else:
        robust_pose = np.frombuffer(
            robust_members["optimized_poses.bin"], dtype=np.float16
        ).reshape(-1, public_pose.shape[1]).astype(np.float32)
        robust_pose_runtime = "QP1 materialized as fp16 optimized_poses.bin, then loaded as float32"

    pair_indices = resolve_pair_indices(
        total_pairs=int(public_pose.shape[0]),
        pair_indices=parse_pair_indices(args.pair_indices),
        max_pairs=args.max_pairs,
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    report: dict[str, Any] = {
        "schema": "pr75_raw_output_parity_v1",
        "tool": str(Path(__file__).relative_to(REPO_ROOT)),
        "score_claim": False,
        "evidence_grade": "empirical_local_raw_parity",
        "remote_gpu_dispatch": False,
        "device": device,
        "inputs": {
            "public_archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            },
            "public_inflate_py": {
                "path": str(public_inflate),
                "sha256": sha256_file(public_inflate),
            },
            "robust_inflate_renderer_py": {
                "path": str(ROBUST_INFLATE_RENDERER),
                "sha256": sha256_file(ROBUST_INFLATE_RENDERER),
            },
            "robust_unpacker_py": {
                "path": str(ROBUST_UNPACKER),
                "sha256": sha256_file(ROBUST_UNPACKER),
            },
        },
        "charged_slices": {
            name: stream_summary(name, data)
            for name, data in charged.items()
            if isinstance(data, bytes) and name.endswith(".br")
        },
        "decoded_streams": {
            "public_runtime": {
                "masks.mkv": stream_summary("masks.mkv", mask_obu),
                "renderer.bin": stream_summary("renderer.bin", renderer_qzs3),
                "seg_tile_actions.bin": stream_summary("seg_tile_actions.bin", actions_raw),
                "optimized_poses.qp1": stream_summary("optimized_poses.qp1", pose_qp1),
            },
            "robust_unpacker": {
                "payload_format": robust_header.get("payload_format"),
                "members": {
                    name: stream_summary(name, data)
                    for name, data in sorted(robust_members.items())
                },
            },
        },
        "pose_precision": pose_precision_report(public_pose, robust_pose),
        "pair_indices": pair_indices,
        "runtime_contract": {
            "public_pr75": {
                "pose_runtime": "QP1 decoded directly to float32 before JointFrameGenerator",
                "mask_runtime": "600 pair masks; one mask feeds each generated pair",
                "actions_runtime": "Brotli 4-byte records grouped by pair; CHW fake2 tile deltas",
            },
            "robust_current": {
                "pose_runtime": robust_pose_runtime,
                "mask_runtime": "600-frame masks.mkv detected as half-frame and duplicated to 1200 without zoom_warp",
                "actions_runtime": "decoded seg_tile_actions.bin grouped by pair; HWC fake2 tile deltas",
            },
        },
    }

    if args.skip_render:
        report["render_parity"] = {"skipped": True}
        report["all_pairs_parity"] = {"enabled": False, "skipped": True}
        report["elapsed_seconds"] = time.monotonic() - t0
        return report

    public_mod = load_module(public_inflate, "pr75_public_inflate_for_parity")
    robust_mod = load_module(ROBUST_INFLATE_RENDERER, "pr75_robust_inflate_for_parity")
    with tempfile.TemporaryDirectory(prefix="pr75_parity_") as tmp_raw:
        tmp_dir = Path(tmp_raw)
        public_masks = load_public_masks(public_mod, mask_obu, tmp_dir)
        robust_masks, robust_masks_expanded = load_robust_masks(robust_mod, mask_obu, tmp_dir)
        member_dir = tmp_dir / "robust_members"
        write_decoded_members(robust_members, member_dir)

        public_generator = load_public_renderer(public_mod, renderer_qzs3, device)
        robust_renderer = robust_mod._load_renderer(str(member_dir / "renderer.bin"), device)  # noqa: SLF001
        public_actions = public_mod.load_seg_tile_actions_data(charged["seg_tile_actions.br"], torch.device(device))
        robust_actions_current = robust_mod._load_seg_tile_actions_from_archive_dir(member_dir, device)  # noqa: SLF001
        robust_actions_override = robust_mod._load_seg_tile_actions_from_archive_dir(member_dir, device)  # noqa: SLF001

        public_pair_masks_np = tensor_to_numpy(public_masks[pair_indices])
        robust_pair_masks_np = tensor_to_numpy(robust_masks_expanded[1::2][pair_indices])
        mask_metrics = numeric_metrics(public_pair_masks_np, robust_pair_masks_np)

        if args.all_pairs:
            report["mask_parity"] = {
                "public_mask_shape": list(public_masks.shape),
                "robust_decoded_mask_shape": list(robust_masks.shape),
                "robust_expanded_mask_shape": list(robust_masks_expanded.shape),
                "robust_half_frame_only": bool(getattr(robust_masks, "_half_frame_only", False)),
                "selected_public_pair_mask_vs_robust_pair_mask": mask_metrics,
                "selected_public_pair_mask_sha256": array_sha256(public_pair_masks_np),
                "selected_robust_pair_mask_sha256": array_sha256(robust_pair_masks_np),
            }
            report["render_parity"] = {"skipped": True, "reason": "--all-pairs uses chunked streaming parity"}
            report["all_pairs_parity"] = compare_all_pairs_chunked(
                public_mod=public_mod,
                robust_mod=robust_mod,
                public_generator=public_generator,
                robust_renderer=robust_renderer,
                public_masks=public_masks,
                robust_masks_expanded=robust_masks_expanded,
                public_pose=public_pose,
                robust_pose=robust_pose,
                public_actions=public_actions,
                robust_actions_current=robust_actions_current,
                chunk_size=args.chunk_size,
                fast_fail=args.fast_fail,
                device=device,
            )
            controlled = report["all_pairs_parity"]["current_runtime_pr75_comparisons_controlled"]
            if controlled:
                diagnosis = (
                    "all-600 public PR75 and robust_current raw outputs match under the "
                    "current QP1-preserving runtime"
                )
                fix = (
                    "current-runtime PR75 comparisons are controlled for local raw parity; "
                    "exact CUDA auth eval remains the only score evidence"
                )
            else:
                diagnosis = (
                    "all-600 chunked parity did not complete cleanly or found a public "
                    "versus robust_current runtime delta"
                )
                fix = "inspect all_pairs_parity.failed_at_chunk and aggregate hashes before exact CUDA replay"
            report["diagnosis"] = {
                "summary": diagnosis,
                "highest_ev_next_fix_or_experiment": fix,
            }
            report["elapsed_seconds"] = time.monotonic() - t0
            return report

        public_before, public_after = render_public_pairs(
            public_mod=public_mod,
            generator=public_generator,
            masks=public_masks,
            poses_float32=public_pose,
            action_state=public_actions,
            pair_indices=pair_indices,
            device=device,
        )
        robust_before, robust_after = render_robust_pairs(
            robust_mod=robust_mod,
            renderer=robust_renderer,
            masks_expanded=robust_masks_expanded,
            poses_float32=robust_pose,
            action_state=robust_actions_current,
            pair_indices=pair_indices,
            device=device,
        )
        robust_public_pose_before, robust_public_pose_after = render_robust_pairs(
            robust_mod=robust_mod,
            renderer=robust_renderer,
            masks_expanded=robust_masks_expanded,
            poses_float32=public_pose,
            action_state=robust_actions_override,
            pair_indices=pair_indices,
            device=device,
        )

        public_raw = raw_bytes_from_hwc_pairs(public_after)
        robust_raw = raw_bytes_from_hwc_pairs(robust_after)
        robust_public_pose_raw = raw_bytes_from_hwc_pairs(robust_public_pose_after)
        if args.write_selected_raw:
            raw_dir = output_dir / "selected_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "public_pr75_selected.raw").write_bytes(public_raw)
            (raw_dir / "robust_current_selected.raw").write_bytes(robust_raw)
            (raw_dir / "robust_public_qp1_pose_selected.raw").write_bytes(robust_public_pose_raw)

        report["mask_parity"] = {
            "public_mask_shape": list(public_masks.shape),
            "robust_decoded_mask_shape": list(robust_masks.shape),
            "robust_expanded_mask_shape": list(robust_masks_expanded.shape),
            "robust_half_frame_only": bool(getattr(robust_masks, "_half_frame_only", False)),
            "selected_public_pair_mask_vs_robust_pair_mask": mask_metrics,
            "selected_public_pair_mask_sha256": array_sha256(public_pair_masks_np),
            "selected_robust_pair_mask_sha256": array_sha256(robust_pair_masks_np),
        }
        report["render_parity"] = {
            "skipped": False,
            "selected_pair_count": len(pair_indices),
            "per_pair": per_pair_render_metrics(
                pair_indices,
                public_after,
                robust_after,
                robust_public_pose_after,
            ),
            "public_vs_robust_current": {
                "native_before_actions": compare_tensor_pair(public_before, robust_before),
                "native_after_actions": compare_tensor_pair(public_after, robust_after),
                "selected_raw_after_actions": byte_metrics(public_raw, robust_raw),
            },
            "public_vs_robust_with_public_qp1_float32_pose": {
                "native_before_actions": compare_tensor_pair(public_before, robust_public_pose_before),
                "native_after_actions": compare_tensor_pair(public_after, robust_public_pose_after),
                "selected_raw_after_actions": byte_metrics(public_raw, robust_public_pose_raw),
            },
        }
        report["all_pairs_parity"] = {"enabled": False, "skipped": True}

        current_equal = report["render_parity"]["public_vs_robust_current"]["selected_raw_after_actions"]["exact_equal"]
        override_equal = report["render_parity"]["public_vs_robust_with_public_qp1_float32_pose"]["selected_raw_after_actions"]["exact_equal"]
        if not current_equal and override_equal:
            diagnosis = (
                "robust_current diverges because it materializes public QP1 poses "
                "through fp16 optimized_poses.bin before rendering"
            )
            fix = (
                "preserve PR75 QP1 pose precision at inflate time, e.g. carry the "
                "QP1 stream as a logical pose member or add a metadata-marked "
                "loader path that decodes QP1 directly to float32 before QZS3"
            )
        elif not current_equal:
            diagnosis = (
                "selected raw output still diverges after public float32 pose override; "
                "inspect renderer loader, mask decoder, and action application parity"
            )
            fix = "diff native tensors per pair and port the first remaining runtime delta"
        else:
            diagnosis = "selected public and robust raw outputs match"
            fix = "scale the harness to all 600 pairs before any exact CUDA replay"
        report["diagnosis"] = {
            "summary": diagnosis,
            "highest_ev_next_fix_or_experiment": fix,
        }

    report["elapsed_seconds"] = time.monotonic() - t0
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare public PR75 raw output against robust_current locally."
    )
    parser.add_argument("--public-archive", type=Path, default=DEFAULT_PUBLIC_ARCHIVE)
    parser.add_argument("--public-inflate-py", type=Path, default=DEFAULT_PUBLIC_INFLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pair-indices",
        default="0,33,36,587",
        help="Comma-separated absolute pair indices to render; ignored with --max-pairs.",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Run chunked deterministic parity over all 600 PR75 pairs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Pair chunk size for --all-pairs streaming render parity.",
    )
    parser.add_argument(
        "--fast-fail",
        action="store_true",
        help="Stop --all-pairs at the first non-identical chunk.",
    )
    parser.add_argument(
        "--write-selected-raw",
        action="store_true",
        help="Write selected-pair raw snippets under output-dir/selected_raw.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    out_path = output_dir / "pr75_raw_output_parity.json"
    write_json(out_path, report)
    print(json.dumps({"report": str(out_path), "diagnosis": report.get("diagnosis")}, indent=2))


if __name__ == "__main__":
    main()
