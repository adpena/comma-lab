#!/usr/bin/env python3
"""Build a scorer-backed PR106 yshift candidate table.

This is a compress-time profiler for
``experiments/build_pr106_yshift_sidechannel.py --search-mode score_table``.
It does not modify inflate behavior and it does not produce a score claim.

The real CUDA mode computes a local marginal objective for each frame and
candidate [y_off, dy, dx]:

  1. Decode the PR106 HNeRV source archive to frame pairs.
  2. Load the official DistortionNet and ground-truth video through upstream
     dataset code.
  3. For each candidate, perturb exactly one frame inside its two-frame pair.
  4. Score the full two-frame pair objective, because PoseNet sees both frames
     and SegNet sees the second frame.

The resulting table has shape ``(n_frames, n_candidates)`` and is only a
compression-time profile artifact. Promotion still requires building a charged
archive from the table and running exact CUDA auth eval on that archive.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    prepend_paths = _tool_bootstrap.prepend_paths
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

prepend_paths(
    REPO_ROOT / "submissions" / "apogee_intN" / "src",
    REPO_ROOT / "submissions" / "pr106_yshift_sidechannel",
)

from build_pr106_yshift_sidechannel import (  # type: ignore[import-not-found]
    build_yshift_candidate_grid,
)
from codec import parse_packed_archive  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]

from tac.repo_io import json_text, sha256_file
from tac.sidechannel_score_table import (
    atomic_save_npy as _atomic_save_npy,
    atomic_write_text as _atomic_write_text,
    clear_cuda_retry_state,
    completed_prefix_rows,
    is_cuda_oom,
    load_distortion_net,
    load_gt_dataloader,
    resume_safe_prefix_pairs,
    score_without_rate,
    verify_active_lane_claim,
)

CAMERA_H = 874
CAMERA_W = 1164
EVAL_SIZE = (384, 512)
SCORE_TABLE_NPY = "score_table.npy"
SCORE_TABLE_MANIFEST = "score_table_manifest.json"
CHECKPOINT_TABLE_NPY = "score_table.partial.npy"
CHECKPOINT_MANIFEST = "score_table_checkpoint.json"


def _read_pr106_bytes(pr106_archive: Path) -> bytes:
    import zipfile

    with zipfile.ZipFile(pr106_archive) as z:
        return z.read("0.bin")


completed_prefix_frames = completed_prefix_rows


def resume_safe_prefix_frames(table: np.ndarray) -> int:
    """Return a pair-aligned prefix so resume never trusts a half-scored pair."""
    return resume_safe_prefix_pairs(table)


def _score_table_contract(
    args: argparse.Namespace,
    *,
    candidates_np: np.ndarray,
    candidates_path: Path,
    n_frames: int,
) -> dict[str, Any]:
    return {
        "source_archive_sha256": sha256_file(args.pr106_archive),
        "candidate_grid_sha256": sha256_file(candidates_path),
        "candidate_radius": int(args.candidate_radius),
        "candidate_count": int(candidates_np.shape[0]),
        "n_pairs": int(args.n_pairs),
        "n_frames": int(n_frames),
        "score_table_shape": [int(n_frames), int(candidates_np.shape[0])],
        "score_step": float(args.score_step),
        "max_frames": None if args.max_frames is None else int(args.max_frames),
    }


def _validate_checkpoint_contract(manifest: dict[str, Any], contract: dict[str, Any]) -> None:
    for key, expected in contract.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"score-table checkpoint contract mismatch for {key}: "
                f"expected {expected!r}, got {manifest.get(key)!r}"
            )


def _checkpoint_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / CHECKPOINT_TABLE_NPY, out_dir / CHECKPOINT_MANIFEST


def _load_score_table_checkpoint(
    args: argparse.Namespace,
    *,
    contract: dict[str, Any],
) -> tuple[np.ndarray, int] | None:
    table_path, manifest_path = _checkpoint_paths(args.out_dir)
    if not table_path.is_file() and not manifest_path.is_file():
        return None
    if not table_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            "incomplete score-table checkpoint; expected both "
            f"{table_path.name} and {manifest_path.name}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema") != "pr106_yshift_score_table_checkpoint_v1":
        raise ValueError(f"unsupported score-table checkpoint schema: {manifest.get('manifest_schema')!r}")
    _validate_checkpoint_contract(manifest, contract)
    table = np.load(table_path, allow_pickle=False)
    expected_shape = tuple(contract["score_table_shape"])
    if tuple(table.shape) != expected_shape:
        raise ValueError(f"score-table checkpoint shape mismatch: expected {expected_shape}, got {table.shape}")
    complete_frames = resume_safe_prefix_frames(table)
    if complete_frames < completed_prefix_frames(table):
        table[complete_frames:] = np.nan
    print(
        "[yshift-score-table] resumed checkpoint "
        f"{table_path} with {complete_frames}/{table.shape[0]} complete frames",
        flush=True,
    )
    return table.astype(np.float32, copy=False), complete_frames


def _write_score_table_checkpoint(
    args: argparse.Namespace,
    *,
    table: np.ndarray,
    contract: dict[str, Any],
    claim_row: dict[str, str],
    started_at: float,
    terminal: bool,
) -> None:
    table_path, manifest_path = _checkpoint_paths(args.out_dir)
    complete_frames = resume_safe_prefix_frames(table)
    table_to_write = table
    if complete_frames < completed_prefix_frames(table):
        table_to_write = table.copy()
        table_to_write[complete_frames:] = np.nan
    _atomic_save_npy(table_path, table_to_write)
    manifest = {
        "manifest_schema": "pr106_yshift_score_table_checkpoint_v1",
        "producer": "experiments/build_pr106_yshift_score_table.py",
        "score_claim": False,
        "terminal": bool(terminal),
        "completed_frames": int(complete_frames),
        "completed_pairs_floor": int(complete_frames // 2),
        "elapsed_seconds_this_invocation": float(time.time() - started_at),
        "lane_claim": claim_row,
        **contract,
        "checkpoint_table_npy_path": str(table_path),
        "checkpoint_table_npy_bytes": int(table_path.stat().st_size),
        "checkpoint_table_npy_sha256": sha256_file(table_path),
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_write_text(manifest_path, json_text(manifest))


def _reuse_completed_score_table_if_valid(
    args: argparse.Namespace,
    *,
    contract: dict[str, Any],
) -> bool:
    table_path = args.out_dir / SCORE_TABLE_NPY
    manifest_path = args.out_dir / SCORE_TABLE_MANIFEST
    if not table_path.is_file() and not manifest_path.is_file():
        return False
    if not table_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            "incomplete completed score table; expected both "
            f"{table_path.name} and {manifest_path.name}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_checkpoint_contract(manifest, contract)
    if manifest.get("ready_for_builder") is not True or manifest.get("score_claim") is not False:
        raise ValueError("existing score-table manifest is not a reusable non-claim builder artifact")
    table = np.load(table_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(contract["score_table_shape"])
    if tuple(table.shape) != expected_shape:
        raise ValueError(f"completed score-table shape mismatch: expected {expected_shape}, got {table.shape}")
    if completed_prefix_frames(np.asarray(table)) != int(table.shape[0]):
        raise ValueError("completed score table contains non-finite rows")
    print(f"[yshift-score-table] reusing completed table {table_path}", flush=True)
    return True


def apply_yshift_candidates_torch(
    frame_batch: torch.Tensor,
    candidates: torch.Tensor,
    *,
    step: float,
) -> torch.Tensor:
    """Apply per-row [y_off, dy, dx] to a batch of HWC frames on one device."""
    if frame_batch.ndim != 4 or frame_batch.shape[-1] != 3:
        raise ValueError(f"frame_batch must be (B,H,W,3), got {tuple(frame_batch.shape)}")
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError(f"candidates must be (B,3), got {tuple(candidates.shape)}")
    if frame_batch.shape[0] != candidates.shape[0]:
        raise ValueError("frame_batch and candidates batch sizes differ")

    out = frame_batch.float()
    y_off = candidates[:, 0].float().view(-1, 1, 1, 1) * float(step)
    out = torch.clamp(torch.round(out + y_off), 0.0, 255.0)
    shifted = out.clone()
    h, w = int(out.shape[1]), int(out.shape[2])
    for idx in range(out.shape[0]):
        dy = int(candidates[idx, 1].item())
        dx = int(candidates[idx, 2].item())
        if dy == 0 and dx == 0:
            continue
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(w, w + dx)
        if src_y1 > src_y0 and src_x1 > src_x0:
            shifted[idx, dst_y0:dst_y1, dst_x0:dst_x1] = out[idx, src_y0:src_y1, src_x0:src_x1]
    return shifted.to(torch.uint8)


def _load_distortion_net(device: torch.device):
    return load_distortion_net(device, repo_root=REPO_ROOT)


def _load_gt_dataloader(
    *,
    device: torch.device,
    video_names_file: Path,
    uncompressed_dir: Path,
    batch_pairs: int,
    seed: int,
    num_threads: int,
    prefetch_queue_depth: int,
):
    return load_gt_dataloader(
        device=device,
        video_names_file=video_names_file,
        uncompressed_dir=uncompressed_dir,
        batch_pairs=batch_pairs,
        seed=seed,
        num_threads=num_threads,
        prefetch_queue_depth=prefetch_queue_depth,
        repo_root=REPO_ROOT,
    )


def _load_pr106_decoder(pr106_archive: Path, device: torch.device):
    pr106_bytes = _read_pr106_bytes(pr106_archive)
    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    return decoder, latents.to(device), meta, pr106_bytes


@torch.inference_mode()
def _decode_pr106_pairs(
    decoder,
    latents: torch.Tensor,
    *,
    start_pair: int,
    end_pair: int,
) -> torch.Tensor:
    decoded = decoder(latents[start_pair:end_pair])
    bsz = end_pair - start_pair
    flat = decoded.reshape(bsz * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
    up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
    frames = up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8)
    return frames.reshape(bsz, 2, CAMERA_H, CAMERA_W, 3)


@torch.inference_mode()
def score_frame_candidate_table(
    distortion_net,
    *,
    gt_pair: torch.Tensor,
    comp_pair: torch.Tensor,
    candidates: torch.Tensor,
    candidate_batch_size: int,
    score_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score both frame slots for one pair; returns two candidate rows."""
    rows: list[np.ndarray] = []
    for frame_slot in (0, 1):
        scores = []
        for start in range(0, int(candidates.shape[0]), candidate_batch_size):
            cand = candidates[start:start + candidate_batch_size]
            batch_size = int(cand.shape[0])
            cand_pairs = comp_pair.repeat(batch_size, 1, 1, 1, 1)
            shifted = apply_yshift_candidates_torch(
                cand_pairs[:, frame_slot],
                cand,
                step=score_step,
            )
            cand_pairs[:, frame_slot] = shifted
            gt_batch = gt_pair.repeat(batch_size, 1, 1, 1, 1)
            pose_dist, seg_dist = distortion_net.compute_distortion(gt_batch, cand_pairs)
            scores.append(score_without_rate(pose_dist, seg_dist).detach().cpu().numpy().astype(np.float32))
        rows.append(np.concatenate(scores, axis=0))
    return rows[0], rows[1]


@torch.inference_mode()
def score_pair_batch_candidate_table(
    distortion_net,
    *,
    gt_pairs: torch.Tensor,
    comp_pairs: torch.Tensor,
    candidates: torch.Tensor,
    candidate_batch_size: int,
    score_step: float,
) -> np.ndarray:
    """Score yshift candidates for a batch of pairs.

    Output rows are ordered by frame index: pair0 frame0, pair0 frame1,
    pair1 frame0, pair1 frame1, ...
    """
    if gt_pairs.ndim != 5 or gt_pairs.shape[1] != 2 or gt_pairs.shape[-1] != 3:
        raise ValueError(f"gt_pairs must be (P,2,H,W,3), got {tuple(gt_pairs.shape)}")
    if comp_pairs.shape != gt_pairs.shape:
        raise ValueError(
            f"comp_pairs shape must match gt_pairs: {tuple(comp_pairs.shape)} != "
            f"{tuple(gt_pairs.shape)}"
        )
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError(f"candidates must be (C,3), got {tuple(candidates.shape)}")
    pair_count = int(gt_pairs.shape[0])
    candidate_count = int(candidates.shape[0])
    rows = np.empty((pair_count * 2, candidate_count), dtype=np.float32)
    for frame_slot in (0, 1):
        for start in range(0, candidate_count, candidate_batch_size):
            cand = candidates[start:start + candidate_batch_size]
            cand_count = int(cand.shape[0])
            cand_pairs = (
                comp_pairs.unsqueeze(1)
                .repeat(1, cand_count, 1, 1, 1, 1)
                .reshape(pair_count * cand_count, 2, *comp_pairs.shape[2:])
            )
            cand_rows = cand.repeat(pair_count, 1)
            shifted = apply_yshift_candidates_torch(
                cand_pairs[:, frame_slot],
                cand_rows,
                step=score_step,
            )
            cand_pairs[:, frame_slot] = shifted
            gt_batch = (
                gt_pairs.unsqueeze(1)
                .repeat(1, cand_count, 1, 1, 1, 1)
                .reshape(pair_count * cand_count, 2, *gt_pairs.shape[2:])
            )
            pose_dist, seg_dist = distortion_net.compute_distortion(gt_batch, cand_pairs)
            scores = score_without_rate(pose_dist, seg_dist).detach().cpu().numpy().astype(np.float32)
            rows[frame_slot::2, start:start + cand_count] = scores.reshape(pair_count, cand_count)
    return rows


@torch.inference_mode()
def score_pair_batch_candidate_table_adaptive(
    distortion_net,
    *,
    gt_pairs: torch.Tensor,
    comp_pairs: torch.Tensor,
    candidates: torch.Tensor,
    pair_chunk_size: int,
    candidate_batch_size: int,
    score_step: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, int]]:
    """Score a pair batch with deterministic CUDA-OOM backoff.

    The mathematical objective is identical to :func:`score_pair_batch_candidate_table`.
    Only the execution tiling changes: on CUDA OOM, halve the candidate tile first,
    then the pair tile, until the same rows can be scored or a single-pair,
    single-candidate tile still fails.
    """
    if pair_chunk_size <= 0:
        raise ValueError(f"pair_chunk_size must be positive, got {pair_chunk_size}")
    if candidate_batch_size <= 0:
        raise ValueError(f"candidate_batch_size must be positive, got {candidate_batch_size}")
    if gt_pairs.ndim != 5 or gt_pairs.shape[1] != 2 or gt_pairs.shape[-1] != 3:
        raise ValueError(f"gt_pairs must be (P,2,H,W,3), got {tuple(gt_pairs.shape)}")
    if comp_pairs.shape != gt_pairs.shape:
        raise ValueError(
            f"comp_pairs shape must match gt_pairs: {tuple(comp_pairs.shape)} != "
            f"{tuple(gt_pairs.shape)}"
        )

    pair_count = int(gt_pairs.shape[0])
    rows = np.empty((pair_count * 2, int(candidates.shape[0])), dtype=np.float32)
    current_pair_chunk = min(int(pair_chunk_size), max(1, pair_count))
    current_candidate_batch = int(candidate_batch_size)
    telemetry = {
        "initial_pair_chunk_size": int(pair_chunk_size),
        "initial_candidate_batch_size": int(candidate_batch_size),
        "min_pair_chunk_size_used": current_pair_chunk,
        "min_candidate_batch_size_used": current_candidate_batch,
        "oom_retry_count": 0,
    }

    pair_start = 0
    while pair_start < pair_count:
        pair_chunk = min(current_pair_chunk, pair_count - pair_start)
        while True:
            try:
                chunk_rows = score_pair_batch_candidate_table(
                    distortion_net,
                    gt_pairs=gt_pairs[pair_start:pair_start + pair_chunk],
                    comp_pairs=comp_pairs[pair_start:pair_start + pair_chunk],
                    candidates=candidates,
                    candidate_batch_size=current_candidate_batch,
                    score_step=score_step,
                )
                break
            except Exception as exc:
                if not is_cuda_oom(exc):
                    raise
                telemetry["oom_retry_count"] += 1
                clear_cuda_retry_state(device)
                if current_candidate_batch > 1:
                    current_candidate_batch = max(1, current_candidate_batch // 2)
                    telemetry["min_candidate_batch_size_used"] = min(
                        telemetry["min_candidate_batch_size_used"],
                        current_candidate_batch,
                    )
                    print(
                        "[yshift-score-table] CUDA OOM retry: "
                        f"pair_chunk_size={pair_chunk} "
                        f"candidate_batch_size={current_candidate_batch}",
                        flush=True,
                    )
                    continue
                if pair_chunk > 1:
                    current_pair_chunk = max(1, pair_chunk // 2)
                    telemetry["min_pair_chunk_size_used"] = min(
                        telemetry["min_pair_chunk_size_used"],
                        current_pair_chunk,
                    )
                    pair_chunk = min(current_pair_chunk, pair_count - pair_start)
                    print(
                        "[yshift-score-table] CUDA OOM retry: "
                        f"pair_chunk_size={pair_chunk} candidate_batch_size=1",
                        flush=True,
                    )
                    continue
                raise RuntimeError(
                    "CUDA OOM while scoring PR106 yshift table even at "
                    "pair_chunk_size=1 and candidate_batch_size=1"
                ) from exc

        rows[pair_start * 2:(pair_start + pair_chunk) * 2] = chunk_rows
        telemetry["min_pair_chunk_size_used"] = min(
            telemetry["min_pair_chunk_size_used"],
            pair_chunk,
        )
        telemetry["min_candidate_batch_size_used"] = min(
            telemetry["min_candidate_batch_size_used"],
            current_candidate_batch,
        )
        pair_start += pair_chunk

    return rows, telemetry


def build_dry_run_plan(args: argparse.Namespace, candidates: np.ndarray) -> dict[str, Any]:
    n_frames = int(args.n_pairs) * 2
    if args.max_frames is not None:
        n_frames = min(n_frames, int(args.max_frames))
    return {
        "manifest_schema": "pr106_yshift_score_table_plan_v1",
        "producer": "experiments/build_pr106_yshift_score_table.py",
        "score_claim": False,
        "dry_run_plan": True,
        "ready_for_builder": False,
        "ready_for_exact_eval_dispatch": False,
        "source_archive_path": str(args.pr106_archive),
        "source_archive_sha256": sha256_file(args.pr106_archive) if args.pr106_archive.is_file() else None,
        "candidate_radius": int(args.candidate_radius),
        "candidate_count": int(candidates.shape[0]),
        "n_pairs": int(args.n_pairs),
        "n_frames": n_frames,
        "expected_score_table_shape": [n_frames, int(candidates.shape[0])],
        "score_step": float(args.score_step),
        "max_frames": None if args.max_frames is None else int(args.max_frames),
        "objective": "100*seg_dist + sqrt(10*pose_dist), without rate constant",
        "pair_marginal_semantics": "one frame perturbed at a time inside the official two-frame scorer pair",
        "dispatch_blockers": [
            "dry_run_plan_only",
            "requires_active_lane_claim_for_cuda_scoring",
            "requires_real_cuda_score_table",
            "requires_archive_build_from_table",
            "requires_exact_cuda_auth_eval_on_built_archive",
        ],
    }


def build_score_table(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates_np = build_yshift_candidate_grid(radius=args.candidate_radius)
    candidates_path = args.out_dir / "candidate_grid.npy"
    np.save(candidates_path, candidates_np, allow_pickle=False)
    n_frames = int(args.n_pairs) * 2
    if args.max_frames is not None:
        n_frames = min(n_frames, int(args.max_frames))

    if args.dry_run_plan:
        plan = build_dry_run_plan(args, candidates_np)
        manifest_path = args.out_dir / "score_table_manifest.json"
        manifest_path.write_text(json_text(plan), encoding="utf-8")
        print(f"[yshift-score-table] wrote dry-run plan {manifest_path}")
        print(f"[yshift-score-table] wrote candidate grid {candidates_path}")
        return 0

    if args.device != "cuda":
        raise SystemExit("real score-table generation requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("real score-table generation requires torch.cuda.is_available()")
    if not args.instance_job_id:
        raise SystemExit("--instance-job-id is required for real CUDA score-table generation")
    claim_row = verify_active_lane_claim(
        args.claims_path,
        lane_id=args.lane_id,
        instance_job_id=args.instance_job_id,
    )
    contract = _score_table_contract(
        args,
        candidates_np=candidates_np,
        candidates_path=candidates_path,
        n_frames=n_frames,
    )
    if args.resume_checkpoint and _reuse_completed_score_table_if_valid(args, contract=contract):
        return 0

    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    started = time.time()

    decoder, latents, meta, pr106_bytes = _load_pr106_decoder(args.pr106_archive, device)
    if int(meta["n_pairs"]) < int(args.n_pairs):
        raise SystemExit(f"requested n_pairs={args.n_pairs}, but PR106 archive has {meta['n_pairs']}")
    distortion_net = _load_distortion_net(device)
    gt_loader = _load_gt_dataloader(
        device=device,
        video_names_file=args.video_names_file,
        uncompressed_dir=args.uncompressed_dir,
        batch_pairs=args.batch_pairs,
        seed=args.seed,
        num_threads=args.num_threads,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )

    n_pairs_to_score = math.ceil(n_frames / 2)
    checkpoint = _load_score_table_checkpoint(args, contract=contract) if args.resume_checkpoint else None
    if checkpoint is None:
        table = np.full((n_frames, candidates_np.shape[0]), np.nan, dtype=np.float32)
        resume_pair_cursor = 0
    else:
        table, complete_frames = checkpoint
        resume_pair_cursor = min(math.ceil(complete_frames / 2), n_pairs_to_score)
    candidates = torch.from_numpy(candidates_np.astype(np.int8)).to(device)
    adaptive_batching = {
        "initial_pair_chunk_size": int(args.batch_pairs),
        "initial_candidate_batch_size": int(args.candidate_batch_size),
        "min_pair_chunk_size_used": int(args.batch_pairs),
        "min_candidate_batch_size_used": int(args.candidate_batch_size),
        "oom_retry_count": 0,
    }

    pair_cursor = 0
    for _, _, gt_batch in gt_loader:
        if pair_cursor >= n_pairs_to_score:
            break
        gt_rows = int(gt_batch.shape[0])
        batch_pairs = min(gt_rows, n_pairs_to_score - pair_cursor)
        if pair_cursor + batch_pairs <= resume_pair_cursor:
            pair_cursor += batch_pairs
            print(
                f"[yshift-score-table] skipped checkpointed {min(pair_cursor * 2, n_frames)}/{n_frames} frames",
                flush=True,
            )
            continue
        local_start = max(0, resume_pair_cursor - pair_cursor)
        if local_start:
            gt_batch = gt_batch[local_start:]
            pair_cursor += local_start
            batch_pairs -= local_start
        gt_batch = gt_batch.to(device)
        comp_batch = _decode_pr106_pairs(
            decoder,
            latents,
            start_pair=pair_cursor,
            end_pair=pair_cursor + batch_pairs,
        )
        scored_rows, batch_telemetry = score_pair_batch_candidate_table_adaptive(
            distortion_net,
            gt_pairs=gt_batch[:batch_pairs],
            comp_pairs=comp_batch[:batch_pairs],
            candidates=candidates,
            pair_chunk_size=batch_pairs,
            candidate_batch_size=args.candidate_batch_size,
            score_step=args.score_step,
            device=device,
        )
        adaptive_batching["min_pair_chunk_size_used"] = min(
            adaptive_batching["min_pair_chunk_size_used"],
            batch_telemetry["min_pair_chunk_size_used"],
        )
        adaptive_batching["min_candidate_batch_size_used"] = min(
            adaptive_batching["min_candidate_batch_size_used"],
            batch_telemetry["min_candidate_batch_size_used"],
        )
        adaptive_batching["oom_retry_count"] += batch_telemetry["oom_retry_count"]
        for local_frame, row in enumerate(scored_rows):
            frame_index = pair_cursor * 2 + local_frame
            if frame_index < n_frames:
                table[frame_index] = row
        pair_cursor += batch_pairs
        del comp_batch, gt_batch, scored_rows
        clear_cuda_retry_state(device)
        print(f"[yshift-score-table] scored {min(pair_cursor * 2, n_frames)}/{n_frames} frames", flush=True)
        if args.resume_checkpoint:
            _write_score_table_checkpoint(
                args,
                table=table,
                contract=contract,
                claim_row=claim_row,
                started_at=started,
                terminal=False,
            )

    if not np.isfinite(table).all():
        raise SystemExit("score table contains unfilled or non-finite entries")

    table_path = args.out_dir / SCORE_TABLE_NPY
    _atomic_save_npy(table_path, table)
    elapsed = time.time() - started
    zero_idx = int(np.flatnonzero((candidates_np == 0).all(axis=1))[0])
    best_idx = table.argmin(axis=1)
    best_scores = table[np.arange(table.shape[0]), best_idx]
    zero_scores = table[:, zero_idx]
    strict_improvements = best_scores < zero_scores
    manifest = {
        "manifest_schema": "pr106_yshift_score_table_manifest_v1",
        "producer": "experiments/build_pr106_yshift_score_table.py",
        "score_claim": False,
        "dry_run_plan": False,
        "ready_for_builder": True,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "remote_jobs_dispatched": False,
        "lane_claim_verified": True,
        "lane_claim": claim_row,
        "source_archive_path": str(args.pr106_archive),
        "source_archive_bytes": int(args.pr106_archive.stat().st_size),
        "source_archive_sha256": sha256_file(args.pr106_archive),
        "source_zero_bin_sha256": hashlib.sha256(pr106_bytes).hexdigest(),
        "candidate_grid_path": str(candidates_path),
        "candidate_grid_sha256": sha256_file(candidates_path),
        "score_table_npy_path": str(table_path),
        "score_table_npy_bytes": int(table_path.stat().st_size),
        "score_table_npy_sha256": sha256_file(table_path),
        "candidate_radius": int(args.candidate_radius),
        "candidate_count": int(candidates_np.shape[0]),
        "n_pairs": int(args.n_pairs),
        "n_frames": int(n_frames),
        "score_table_shape": [int(table.shape[0]), int(table.shape[1])],
        "score_step": float(args.score_step),
        "max_frames": None if args.max_frames is None else int(args.max_frames),
        "objective": "100*seg_dist + sqrt(10*pose_dist), without rate constant",
        "pair_marginal_semantics": "one frame perturbed at a time inside the official two-frame scorer pair",
        "zero_candidate_index": int(zero_idx),
        "strict_improvement_frame_count": int(strict_improvements.sum()),
        "best_improvement_min": float((zero_scores - best_scores).min()),
        "best_improvement_mean": float((zero_scores - best_scores).mean()),
        "best_improvement_max": float((zero_scores - best_scores).max()),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "adaptive_batching": adaptive_batching,
        "elapsed_seconds": float(elapsed),
        "resume_checkpoint_enabled": bool(args.resume_checkpoint),
        "resume_pair_cursor_at_start": int(resume_pair_cursor),
        "checkpoint_table_npy_path": str(args.out_dir / CHECKPOINT_TABLE_NPY) if args.resume_checkpoint else None,
        "checkpoint_manifest_path": str(args.out_dir / CHECKPOINT_MANIFEST) if args.resume_checkpoint else None,
        "dispatch_blockers": [
            "requires_archive_build_from_table",
            "requires_exact_cuda_auth_eval_on_built_archive",
        ],
    }
    manifest_path = args.out_dir / SCORE_TABLE_MANIFEST
    _atomic_write_text(manifest_path, json_text(manifest))
    if args.resume_checkpoint:
        _write_score_table_checkpoint(
            args,
            table=table,
            contract=contract,
            claim_row=claim_row,
            started_at=started,
            terminal=True,
        )
    print(f"[yshift-score-table] wrote table {table_path}")
    print(f"[yshift-score-table] wrote manifest {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pr106-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-radius", type=int, default=3)
    parser.add_argument("--score-step", type=float, default=1.0)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional partial scorer-table cap for debugging. Non-promotable unless full.")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--prefetch-queue-depth", type=int, default=4)
    parser.add_argument("--video-names-file", type=Path,
                        default=REPO_ROOT / "upstream" / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path,
                        default=REPO_ROOT / "upstream" / "videos")
    parser.add_argument("--claims-path", type=Path,
                        default=REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md")
    parser.add_argument("--lane-id", default="lane_pr106_yshift_score_table")
    parser.add_argument("--instance-job-id", default="",
                        help="Required for real CUDA scoring; must match an active lane claim row.")
    parser.add_argument("--dry-run-plan", action="store_true",
                        help="Write candidate-grid and plan metadata only; no CUDA/scorer/claim required.")
    parser.add_argument("--resume-checkpoint", action="store_true",
                        help="Resume and update score_table.partial.npy in --out-dir after each DALI batch.")
    args = parser.parse_args()
    return build_score_table(args)


if __name__ == "__main__":
    raise SystemExit(main())
