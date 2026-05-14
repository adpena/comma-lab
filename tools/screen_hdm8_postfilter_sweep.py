#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Screen deterministic inflate-time postfilters on the HDM8 HNeRV frontier.

This is a diagnostic/proxy tool. It does not make score claims. It runs the
same HNeRV decode path used by ``submissions/pr106_latent_sidecar_r2_pr101_grammar``
and evaluates a bounded prefix of non-overlapping frame pairs through the
scorers so we can decide whether a deterministic postfilter is worth exact
CUDA dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

RUNTIME_ROOT = REPO_ROOT / "submissions" / "pr106_latent_sidecar_r2_pr101_grammar"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))
if str(RUNTIME_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from inflate import (  # type: ignore[import-not-found]
    CAMERA_H,
    CAMERA_W,
    NO_OP_DIM,
    SIDECAR_FORMAT_BROTLI,
    decode_brotli_sidecar,
    decode_pr101_grammar_sidecar,
    parse_sidecar_archive,
)
from codec import parse_packed_archive  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]
from tac.data import decode_video
from tac.scorer import comma_score, load_default_scorers


UNSHARP_ROW = torch.tensor([1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0])
RATE_DENOMINATOR_BYTES = 37_545_489
TOOL_VERSION = "hdm8_postfilter_proxy_sweep_v2_batched_segnet_reuse"


def _sha256_jsonable(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:
        return "unknown"


def _read_single_member_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise SystemExit(f"expected a single archive payload member, found {len(infos)}")
        return zf.read(infos[0].filename)


def _decode_payload(payload: bytes, *, device: torch.device) -> tuple[HNeRVDecoder, torch.Tensor, dict[str, Any]]:
    format_id, pr106_bytes, sidecar_blob, framing_meta = parse_sidecar_archive(payload)
    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)
    if format_id == SIDECAR_FORMAT_BROTLI:
        if sidecar_blob:
            dim_arr, delta_q_arr = decode_brotli_sidecar(sidecar_blob)
        else:
            dim_arr = torch.full((int(meta["n_pairs"]),), NO_OP_DIM, dtype=torch.uint8).numpy()
            delta_q_arr = torch.zeros((int(meta["n_pairs"]),), dtype=torch.int8).numpy()
    else:
        if framing_meta is None:
            raise ValueError("framing_meta missing for PR101 grammar payload")
        dim_arr, delta_q_arr = decode_pr101_grammar_sidecar(sidecar_blob, framing_meta)

    for p in range(latents.shape[0]):
        dim = int(dim_arr[p])
        if dim != NO_OP_DIM:
            latents[p, dim] = latents[p, dim] + float(delta_q_arr[p]) * 0.01

    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    return decoder, latents.to(device), meta


def _kernel(device: torch.device) -> torch.Tensor:
    row = UNSHARP_ROW.to(device)
    kernel_2d = torch.outer(row, row) / (row.sum() ** 2)
    return kernel_2d.expand(3, 1, 9, 9).contiguous()


def _parse_mode(mode: str) -> tuple[str, float]:
    if ":" not in mode:
        return mode, 0.0
    name, raw = mode.split(":", 1)
    return name.strip(), float(raw)


def _parse_triplet(raw: str) -> tuple[float, float, float]:
    parts = raw.split(",")
    if len(parts) != 3:
        raise ValueError(f"expected three comma-separated values, got {raw!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _translate_keep_borders(frames_bchw: torch.Tensor, *, dy: int, dx: int) -> torch.Tensor:
    if dy == 0 and dx == 0:
        return frames_bchw
    out = frames_bchw.clone()
    _, _, height, width = frames_bchw.shape
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = frames_bchw[
            :, :, src_y0:src_y1, src_x0:src_x1
        ]
    return out


def _coordinate_noise(
    frames_bchw: torch.Tensor,
    *,
    frame_start: int,
    channel_independent: bool,
) -> torch.Tensor:
    batch, channels, height, width = frames_bchw.shape
    yy = torch.arange(height, device=frames_bchw.device, dtype=torch.float32).view(1, 1, height, 1)
    xx = torch.arange(width, device=frames_bchw.device, dtype=torch.float32).view(1, 1, 1, width)
    ff = (
        torch.arange(frame_start, frame_start + batch, device=frames_bchw.device, dtype=torch.float32)
        .view(batch, 1, 1, 1)
    )
    cc = torch.arange(channels, device=frames_bchw.device, dtype=torch.float32).view(1, channels, 1, 1)
    if not channel_independent:
        cc = torch.zeros_like(cc)
    phase = xx * 12.9898 + yy * 78.233 + ff * 37.719 + cc * 19.191
    return (torch.frac(torch.sin(phase) * 43758.5453) - 0.5) * 2.0


def _apply_grain(frames_bchw: torch.Tensor, mode: str, *, frame_start: int) -> torch.Tensor:
    name, value = _parse_mode(mode)
    amp = float(value)
    if name == "checker":
        batch, channels, height, width = frames_bchw.shape
        yy = torch.arange(height, device=frames_bchw.device).view(1, 1, height, 1)
        xx = torch.arange(width, device=frames_bchw.device).view(1, 1, 1, width)
        ff = torch.arange(frame_start, frame_start + batch, device=frames_bchw.device).view(batch, 1, 1, 1)
        pattern = (((xx + yy + ff) & 1).float() * 2.0 - 1.0).expand(batch, channels, height, width)
        return frames_bchw + amp * pattern
    if name == "tile_chroma":
        batch, _channels, height, width = frames_bchw.shape
        base = torch.tensor(
            [
                [-1, 1, -1, 1, 1, -1, 1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1],
                [-1, 1, 1, -1, 1, -1, -1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1],
                [-1, -1, 1, 1, -1, -1, 1, 1],
                [1, -1, -1, 1, 1, -1, -1, 1],
                [-1, 1, 1, -1, -1, 1, 1, -1],
            ],
            dtype=frames_bchw.dtype,
            device=frames_bchw.device,
        )
        reps_h = (height + 7) // 8
        reps_w = (width + 7) // 8
        pattern = base.repeat(reps_h, reps_w)[:height, :width].view(1, 1, height, width)
        pattern = pattern.expand(batch, 1, height, width)
        out = frames_bchw.clone()
        out[:, 0:1].add_(amp * pattern)
        out[:, 2:3].sub_(amp * pattern)
        return out

    if name == "grain":
        return frames_bchw + amp * _coordinate_noise(
            frames_bchw,
            frame_start=frame_start,
            channel_independent=True,
        )
    if name == "grain_luma":
        noise = _coordinate_noise(
            frames_bchw[:, :1],
            frame_start=frame_start,
            channel_independent=False,
        )
        return frames_bchw + amp * noise.expand_as(frames_bchw)
    if name == "grain_chroma":
        noise = _coordinate_noise(
            frames_bchw[:, :1],
            frame_start=frame_start,
            channel_independent=False,
        )
        weights = torch.tensor([1.0, -0.5, -0.5], device=frames_bchw.device).view(1, 3, 1, 1)
        return frames_bchw + amp * noise.expand_as(frames_bchw) * weights
    if name == "blue":
        noise = _coordinate_noise(
            frames_bchw,
            frame_start=frame_start,
            channel_independent=True,
        )
        low = F.avg_pool2d(F.pad(noise, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        high = noise - low
        high = high / high.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        return frames_bchw + amp * high.clamp(-2.0, 2.0)
    if name == "grain_var":
        noise = _coordinate_noise(
            frames_bchw,
            frame_start=frame_start,
            channel_independent=True,
        )
        luma = (
            0.299 * frames_bchw[:, 0:1]
            + 0.587 * frames_bchw[:, 1:2]
            + 0.114 * frames_bchw[:, 2:3]
        )
        local_mean = F.avg_pool2d(F.pad(luma, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        local_sq = F.avg_pool2d(F.pad(luma.square(), (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        local_var = (local_sq - local_mean.square()).clamp_min(0)
        scale = (local_var / (local_var + 64.0)).expand_as(frames_bchw)
        return frames_bchw + amp * noise * scale
    raise ValueError(f"unsupported grain mode {mode!r}")


def _apply_filter(frames_bchw: torch.Tensor, mode: str, *, frame_start: int) -> torch.Tensor:
    if "+" in mode:
        out = frames_bchw
        for part in mode.split("+"):
            out = _apply_filter(out, part, frame_start=frame_start)
        return out
    if mode.startswith("even_") or mode.startswith("odd_"):
        want_even = mode.startswith("even_")
        inner_mode = mode.split("_", 1)[1]
        filtered = _apply_filter(frames_bchw, inner_mode, frame_start=frame_start)
        frame_ids = torch.arange(
            frame_start,
            frame_start + frames_bchw.shape[0],
            device=frames_bchw.device,
        )
        parity_mask = ((frame_ids % 2) == 0) if want_even else ((frame_ids % 2) == 1)
        mask = parity_mask.view(-1, 1, 1, 1)
        return torch.where(mask, filtered, frames_bchw)
    if mode.startswith("translate:"):
        _, raw = mode.split(":", 1)
        dy_raw, dx_raw = raw.split(",", 1)
        return _translate_keep_borders(frames_bchw, dy=int(dy_raw), dx=int(dx_raw))
    if mode.startswith("rgb_bias:"):
        _, raw = mode.split(":", 1)
        dr, dg, db = _parse_triplet(raw)
        delta = torch.tensor([dr, dg, db], device=frames_bchw.device).view(1, 3, 1, 1)
        return frames_bchw + delta
    if mode.startswith("rgb_scale:"):
        _, raw = mode.split(":", 1)
        sr, sg, sb = _parse_triplet(raw)
        scale = torch.tensor([sr, sg, sb], device=frames_bchw.device).view(1, 3, 1, 1)
        return frames_bchw * scale
    name, value = _parse_mode(mode)
    if name == "none":
        return frames_bchw
    if name == "bias":
        return frames_bchw + value
    if name == "contrast":
        return (frames_bchw - 127.5) * (1.0 + value) + 127.5
    if name == "gamma":
        return (frames_bchw / 255.0).clamp(0.0, 1.0).pow(value) * 255.0
    if name in {"grain", "grain_luma", "grain_chroma", "grain_var", "blue", "checker", "tile_chroma"}:
        return _apply_grain(frames_bchw, mode, frame_start=frame_start)

    kernel = _kernel(frames_bchw.device)
    padded = F.pad(frames_bchw, (4, 4, 4, 4), mode="reflect")
    blur = F.conv2d(padded, kernel, padding=0, groups=3)
    detail = frames_bchw - blur

    if name == "unsharp":
        return frames_bchw + value * detail
    if name == "soften":
        return frames_bchw - value * detail
    if name == "adaptive":
        luma = (
            0.299 * frames_bchw[:, 0:1]
            + 0.587 * frames_bchw[:, 1:2]
            + 0.114 * frames_bchw[:, 2:3]
        )
        local_mean = F.avg_pool2d(F.pad(luma, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        local_sq = F.avg_pool2d(F.pad(luma.square(), (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        local_var = (local_sq - local_mean.square()).clamp_min(0)
        alpha = value * (local_var / (local_var + 100.0))
        return frames_bchw + alpha * detail
    raise ValueError(f"unsupported postfilter mode {mode!r}")


def _mode_preserves_second_frame(mode: str) -> bool:
    """Return true when SegNet's scored frame is byte-identical to no-filter.

    The contest SegNet contract scores the second frame of each pair. HDM8
    emits pair frames as even then odd indices, so ``even_*`` transforms are a
    SegNet-null search surface while still moving PoseNet through frame zero.
    """

    text = str(mode).strip()
    if text == "none":
        return True
    if "+" in text:
        return all(_mode_preserves_second_frame(part) for part in text.split("+"))
    return text.startswith("even_")


def _decode_candidate_frames(
    decoder: HNeRVDecoder,
    latents: torch.Tensor,
    *,
    n_pairs: int,
    batch_pairs: int,
    mode: str,
) -> torch.Tensor:
    eval_h, eval_w = tuple(decoder.eval_size)
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, n_pairs, batch_pairs):
            end = min(start + batch_pairs, n_pairs)
            decoded = decoder(latents[start:end])
            flat = decoded.reshape((end - start) * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            filtered = _apply_filter(up, mode, frame_start=start * 2)
            frames = filtered.clamp(0, 255).round().to(torch.uint8)
            chunks.append(frames.permute(0, 2, 3, 1).cpu())
    return torch.cat(chunks, dim=0)


def _score_frames(
    *,
    frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    archive_bytes: int,
    batch_pairs: int,
    include_per_pair: bool = False,
) -> dict[str, float | int]:
    total_pose = 0.0
    total_seg = 0.0
    pair_pose: list[float] = []
    pair_seg: list[float] = []
    n_pairs = frames.shape[0] // 2
    for start in range(0, n_pairs, batch_pairs):
        end = min(start + batch_pairs, n_pairs)
        cand = frames[start * 2 : end * 2].to(device).float()
        gt = torch.stack(gt_frames[start * 2 : end * 2]).to(device).float()
        cand_btchw = cand.reshape(end - start, 2, CAMERA_H, CAMERA_W, 3).permute(0, 1, 4, 2, 3)
        gt_btchw = gt.reshape(end - start, 2, CAMERA_H, CAMERA_W, 3).permute(0, 1, 4, 2, 3)
        with torch.inference_mode():
            pose_pred = posenet(posenet.preprocess_input(cand_btchw))["pose"][..., :6]
            pose_gt = posenet(posenet.preprocess_input(gt_btchw))["pose"][..., :6]
            pose_values = (pose_pred - pose_gt).square().mean(dim=-1)
            total_pose += pose_values.sum().item()

            seg_pred = segnet(segnet.preprocess_input(cand_btchw))
            seg_gt = segnet(segnet.preprocess_input(gt_btchw))
            diff = (seg_pred.argmax(dim=1) != seg_gt.argmax(dim=1)).float()
            seg_values = diff.mean(dim=tuple(range(1, diff.ndim)))
            total_seg += seg_values.sum().item()
            if include_per_pair:
                pair_pose.extend(float(x) for x in pose_values.detach().cpu().tolist())
                pair_seg.extend(float(x) for x in seg_values.detach().cpu().tolist())

    pose = total_pose / max(1, n_pairs)
    seg = total_seg / max(1, n_pairs)
    rate = archive_bytes / RATE_DENOMINATOR_BYTES
    out: dict[str, float | int | list[float]] = {
        "n_pairs": n_pairs,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "score_proxy": comma_score(pose, seg, rate),
        "pose_contribution": math.sqrt(10.0 * pose),
        "seg_contribution": 100.0 * seg,
        "rate_contribution": 25.0 * rate,
    }
    if include_per_pair:
        out["pair_posenet_dist"] = pair_pose
        out["pair_segnet_dist"] = pair_seg
    return out


def _score_modes_batched(
    *,
    decoder: HNeRVDecoder,
    latents: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    archive_bytes: int,
    n_pairs: int,
    modes: list[str],
    decode_batch_pairs: int,
    score_batch_pairs: int,
    mode_batch_size: int,
    include_per_pair: bool,
) -> list[dict[str, float | int | str | list[float]]]:
    """Decode each HNeRV batch once, then score all postfilter modes.

    The previous screen loop decoded the same HNeRV frames once per mode and
    ran separate scorer forwards for every mode. That made broad local CPU/MPS
    sweeps and Modal CUDA confirmation unnecessarily expensive. This routine
    keeps the exact scorer contract while reusing decoded frames and batching
    mode chunks through the scorer.
    """

    eval_h, eval_w = tuple(decoder.eval_size)
    score_batch_pairs = max(1, int(score_batch_pairs))
    mode_batch_size = max(1, int(mode_batch_size))
    totals = {
        mode: {
            "pose": 0.0,
            "seg": 0.0,
            "pair_pose": [],
            "pair_seg": [],
        }
        for mode in modes
    }
    with torch.inference_mode():
        for start in range(0, n_pairs, decode_batch_pairs):
            end = min(start + decode_batch_pairs, n_pairs)
            pair_count = end - start
            decoded = decoder(latents[start:end])
            flat = decoded.reshape(pair_count * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)

            for rel_start in range(0, pair_count, score_batch_pairs):
                rel_end = min(rel_start + score_batch_pairs, pair_count)
                local_pair_count = rel_end - rel_start
                frame_start = (start + rel_start) * 2
                up_slice = up[rel_start * 2 : rel_end * 2]
                gt = torch.stack(gt_frames[frame_start : frame_start + local_pair_count * 2]).to(
                    device
                ).float()
                gt_btchw = gt.reshape(local_pair_count, 2, CAMERA_H, CAMERA_W, 3).permute(
                    0, 1, 4, 2, 3
                )
                pose_gt = posenet(posenet.preprocess_input(gt_btchw))["pose"][..., :6]
                seg_gt_argmax = segnet(segnet.preprocess_input(gt_btchw)).argmax(dim=1)
                base_frames = up_slice.clamp(0, 255).round().to(torch.uint8)
                base_cand_btchw = base_frames.reshape(
                    local_pair_count, 2, 3, CAMERA_H, CAMERA_W
                ).float()
                base_seg_pred = segnet(segnet.preprocess_input(base_cand_btchw))
                base_seg_diff = (
                    base_seg_pred.argmax(dim=1) != seg_gt_argmax
                ).float()
                base_seg_values = base_seg_diff.mean(dim=tuple(range(1, base_seg_diff.ndim)))

                for mode_start in range(0, len(modes), mode_batch_size):
                    mode_chunk = modes[mode_start : mode_start + mode_batch_size]
                    filtered_chunk = [
                        _apply_filter(up_slice, mode, frame_start=frame_start)
                        .clamp(0, 255)
                        .round()
                        .to(torch.uint8)
                        for mode in mode_chunk
                    ]
                    stacked = torch.stack(filtered_chunk, dim=0)
                    cand_btchw = stacked.reshape(
                        len(mode_chunk) * local_pair_count,
                        2,
                        3,
                        CAMERA_H,
                        CAMERA_W,
                    ).float()
                    pose_pred = posenet(posenet.preprocess_input(cand_btchw))["pose"][..., :6]
                    pose_values = (
                        (pose_pred.reshape(len(mode_chunk), local_pair_count, 6) - pose_gt.unsqueeze(0))
                        .square()
                        .mean(dim=-1)
                    )
                    if all(_mode_preserves_second_frame(mode) for mode in mode_chunk):
                        seg_values = base_seg_values.unsqueeze(0).expand(
                            len(mode_chunk), local_pair_count
                        )
                    else:
                        seg_pred = segnet(segnet.preprocess_input(cand_btchw))
                        seg_argmax = seg_pred.argmax(dim=1).reshape(
                            len(mode_chunk), local_pair_count, *seg_gt_argmax.shape[1:]
                        )
                        diff = (seg_argmax != seg_gt_argmax.unsqueeze(0)).float()
                        seg_values = diff.mean(dim=tuple(range(2, diff.ndim)))

                    for chunk_idx, mode in enumerate(mode_chunk):
                        mode_pose = pose_values[chunk_idx]
                        mode_seg = seg_values[chunk_idx]
                        totals[mode]["pose"] = float(totals[mode]["pose"]) + mode_pose.sum().item()
                        totals[mode]["seg"] = float(totals[mode]["seg"]) + mode_seg.sum().item()
                        if include_per_pair:
                            totals[mode]["pair_pose"].extend(
                                float(x) for x in mode_pose.detach().cpu().tolist()
                            )
                            totals[mode]["pair_seg"].extend(
                                float(x) for x in mode_seg.detach().cpu().tolist()
                            )

    rate = archive_bytes / RATE_DENOMINATOR_BYTES
    results: list[dict[str, float | int | str | list[float]]] = []
    for mode in modes:
        pose = float(totals[mode]["pose"]) / max(1, n_pairs)
        seg = float(totals[mode]["seg"]) / max(1, n_pairs)
        row: dict[str, float | int | str | list[float]] = {
            "mode": mode,
            "n_pairs": n_pairs,
            "avg_posenet_dist": pose,
            "avg_segnet_dist": seg,
            "score_proxy": comma_score(pose, seg, rate),
            "pose_contribution": math.sqrt(10.0 * pose),
            "seg_contribution": 100.0 * seg,
            "rate_contribution": 25.0 * rate,
        }
        if include_per_pair:
            row["pair_posenet_dist"] = list(totals[mode]["pair_pose"])
            row["pair_segnet_dist"] = list(totals[mode]["pair_seg"])
        results.append(row)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--n-pairs", type=int, default=24)
    parser.add_argument("--decode-batch-pairs", type=int, default=4)
    parser.add_argument("--score-batch-pairs", type=int, default=2)
    parser.add_argument(
        "--mode-batch-size",
        type=int,
        default=4,
        help="Number of postfilter modes to batch into one scorer forward.",
    )
    parser.add_argument("--include-per-pair", action="store_true")
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Filter mode. Examples: none, unsharp:0.15, adaptive:0.85, soften:0.1",
    )
    args = parser.parse_args(argv)

    modes = args.mode or [
        "none",
        "unsharp:0.10",
        "unsharp:0.20",
        "unsharp:0.35",
        "adaptive:0.50",
        "adaptive:0.85",
        "soften:0.05",
    ]
    if args.n_pairs <= 0:
        raise SystemExit("--n-pairs must be positive")

    device = torch.device(args.device)
    started = time.time()
    payload = _read_single_member_payload(args.archive)
    decoder, latents, meta = _decode_payload(payload, device=device)
    n_pairs = min(args.n_pairs, int(meta["n_pairs"]))

    gt_frames = decode_video(
        args.upstream_dir / "videos" / "0.mkv",
        target_h=CAMERA_H,
        target_w=CAMERA_W,
        max_frames=n_pairs * 2,
    )
    posenet, segnet = load_default_scorers(args.upstream_dir, device=device)

    results = _score_modes_batched(
        decoder=decoder,
        latents=latents,
        gt_frames=gt_frames,
        posenet=posenet,
        segnet=segnet,
        device=device,
        archive_bytes=args.archive.stat().st_size,
        n_pairs=n_pairs,
        modes=modes,
        decode_batch_pairs=args.decode_batch_pairs,
        score_batch_pairs=args.score_batch_pairs,
        mode_batch_size=args.mode_batch_size,
        include_per_pair=args.include_per_pair,
    )
    for scored in results:
        print(
            f"{str(scored['mode']):>14} score={scored['score_proxy']:.9f} "
            f"pose={scored['avg_posenet_dist']:.8g} seg={scored['avg_segnet_dist']:.8g}",
            flush=True,
        )

    baseline = next(item for item in results if item["mode"] == "none")
    for item in results:
        item["delta_vs_none"] = float(item["score_proxy"]) - float(baseline["score_proxy"])

    payload_out = {
        "schema": "hdm8_postfilter_proxy_sweep_v1",
        "tool_version": TOOL_VERSION,
        "score_claim": False,
        "promotion_eligible": False,
        "axis": f"local-{args.device}-proxy-prefix",
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "n_pairs": n_pairs,
        "requested_n_pairs": int(args.n_pairs),
        "mode_count": len(modes),
        "mode_list_sha256": _sha256_jsonable(modes),
        "decode_batch_pairs": int(args.decode_batch_pairs),
        "score_batch_pairs": int(args.score_batch_pairs),
        "mode_batch_size": int(args.mode_batch_size),
        "elapsed_seconds": time.time() - started,
        "source_repo_commit": _source_commit(),
        "deterministic_contract": {
            "randomness": "none",
            "postfilter_modes": "deterministic_coordinate_functions_or_scalar_biases",
            "segnet_reuse_for_first_frame_safe_modes": True,
            "first_frame_safe_modes_preserve_second_frame": all(
                _mode_preserves_second_frame(mode) for mode in modes
            ),
            "score_claim": False,
            "promotion_eligible": False,
        },
        "modes": results,
        "best": min(results, key=lambda item: float(item["score_proxy"])),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload_out, indent=2) + "\n")
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
