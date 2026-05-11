#!/usr/bin/env python3
"""Build a scorer-backed PR106 latent-sidecar candidate table.

This is the compress-time profiler for
``experiments/build_pr106_latent_sidecar.py --search-mode score_table``.
It does not modify inflate behavior and it does not produce a score claim.

For each frame pair and candidate ``[dim_idx, delta_q]``:

  1. Decode the PR106 HNeRV source archive to its latent tensor.
  2. Load the official DistortionNet and ground-truth video on CUDA.
  3. Perturb exactly one latent dimension for that pair.
  4. Decode the pair and score the official two-frame objective.

The resulting table has shape ``(n_pairs, n_candidates)``. Promotion still
requires building charged sidecar bytes from the table and then running exact
CUDA auth eval on the emitted archive.
"""
from __future__ import annotations

import argparse
import json
import importlib.util
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
    REPO_ROOT / "submissions" / "pr106_latent_sidecar" / "src",
    REPO_ROOT / "submissions" / "pr106_latent_sidecar",
    REPO_ROOT / "experiments",
)

from build_pr106_latent_sidecar import (  # type: ignore[import-not-found]
    DELTA_SCALE,
    NO_OP_DIM,
    build_latent_candidate_grid,
    latent_candidate_grid_npy_sha256,
)
from codec import parse_packed_archive  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]
from tac.repo_io import json_text, sha256_bytes, sha256_file

CAMERA_H = 874
CAMERA_W = 1164
EVAL_SIZE = (384, 512)
SCORE_TABLE_NPY = "score_table.npy"
SCORE_TABLE_MANIFEST = "score_table_manifest.json"
CHECKPOINT_TABLE_NPY = "score_table.partial.npy"
CHECKPOINT_MANIFEST = "score_table_checkpoint.json"
TERMINAL_PREFIXES = (
    "completed_",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def completed_prefix_frames(table: np.ndarray) -> int:
    """Return the finite scored-row prefix, rejecting non-prefix checkpoints."""
    if table.ndim != 2:
        raise ValueError(f"score table checkpoint must be 2-D, got shape {table.shape}")
    finite_rows = np.isfinite(table).all(axis=1)
    incomplete = np.flatnonzero(~finite_rows)
    if incomplete.size == 0:
        return int(table.shape[0])
    first_incomplete = int(incomplete[0])
    if finite_rows[first_incomplete:].any():
        raise ValueError("score table checkpoint has non-prefix finite rows")
    return first_incomplete


def _is_terminal_status(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_PREFIXES)


def verify_active_lane_claim(
    claims_path: Path,
    *,
    lane_id: str,
    instance_job_id: str,
) -> dict[str, str]:
    """Return the newest matching active claim row or raise ValueError."""
    if not claims_path.is_file():
        raise ValueError(f"missing lane-claim ledger: {claims_path}")
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "lane_id" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        row = {
            "timestamp_utc": cells[0],
            "agent": cells[1],
            "lane_id": cells[2],
            "platform": cells[3],
            "instance_job_id": cells[4],
            "predicted_eta_utc": cells[5],
            "status": cells[6],
            "notes": cells[7],
        }
        if row["lane_id"] != lane_id or row["instance_job_id"] != instance_job_id:
            continue
        if _is_terminal_status(row["status"]):
            raise ValueError(
                "newest matching claim is terminal: "
                f"lane_id={lane_id} instance_job_id={instance_job_id} status={row['status']}"
            )
        return row
    raise ValueError(
        "no active lane claim found for "
        f"lane_id={lane_id} instance_job_id={instance_job_id}"
    )


def score_without_rate(pose_dist: torch.Tensor, seg_dist: torch.Tensor) -> torch.Tensor:
    """Contest objective without the archive-rate constant."""
    return 100.0 * seg_dist + torch.sqrt(torch.clamp(10.0 * pose_dist, min=0.0))


def _load_distortion_net(device: torch.device):
    upstream_dir = REPO_ROOT / "upstream"
    prepend_paths(upstream_dir)
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # type: ignore[import-not-found]

    net = DistortionNet().eval().to(device=device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    return net


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
    upstream_dir = REPO_ROOT / "upstream"
    prepend_paths(upstream_dir)
    from frame_utils import DaliVideoDataset  # type: ignore[import-not-found]

    names = [line.strip() for line in video_names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    ds_gt = DaliVideoDataset(
        names,
        data_dir=uncompressed_dir,
        batch_size=batch_pairs,
        device=device,
        num_threads=num_threads,
        seed=seed,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    ds_gt.prepare_data()
    return torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)


def _read_pr106_bytes(pr106_archive: Path) -> bytes:
    import zipfile

    with zipfile.ZipFile(pr106_archive) as z:
        return z.read("0.bin")


def _score_table_contract(
    args: argparse.Namespace,
    *,
    candidates_np: np.ndarray,
    candidates_path: Path,
    n_pairs: int,
) -> dict[str, Any]:
    return {
        "source_archive_sha256": sha256_file(args.pr106_archive),
        "candidate_grid_sha256": sha256_file(candidates_path),
        "candidate_grid_npy_sha256": latent_candidate_grid_npy_sha256(candidates_np),
        "delta_radius": int(args.delta_radius),
        "candidate_count": int(candidates_np.shape[0]),
        "latent_dim": int(args.latent_dim),
        "n_pairs": int(n_pairs),
        "score_table_shape": [int(n_pairs), int(candidates_np.shape[0])],
        "max_pairs": None if args.max_pairs is None else int(args.max_pairs),
    }


def _validate_checkpoint_contract(manifest: dict[str, Any], contract: dict[str, Any]) -> None:
    for key, expected in contract.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"latent score-table checkpoint contract mismatch for {key}: "
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
            "incomplete latent score-table checkpoint; expected both "
            f"{table_path.name} and {manifest_path.name}"
        )
    manifest = json_text_to_dict(manifest_path)
    if manifest.get("manifest_schema") != "pr106_latent_score_table_checkpoint_v1":
        raise ValueError(f"unsupported latent score-table checkpoint schema: {manifest.get('manifest_schema')!r}")
    _validate_checkpoint_contract(manifest, contract)
    table = np.load(table_path, allow_pickle=False)
    expected_shape = tuple(contract["score_table_shape"])
    if tuple(table.shape) != expected_shape:
        raise ValueError(f"latent checkpoint shape mismatch: expected {expected_shape}, got {table.shape}")
    complete_pairs = completed_prefix_frames(table)
    print(
        "[latent-score-table] resumed checkpoint "
        f"{table_path} with {complete_pairs}/{table.shape[0]} complete pairs",
        flush=True,
    )
    return table.astype(np.float32, copy=False), int(complete_pairs)


def json_text_to_dict(path: Path) -> dict[str, Any]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


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
    _atomic_save_npy(table_path, table)
    complete_pairs = completed_prefix_frames(table)
    manifest = {
        "manifest_schema": "pr106_latent_score_table_checkpoint_v1",
        "producer": "experiments/build_pr106_latent_score_table.py",
        "score_claim": False,
        "terminal": bool(terminal),
        "completed_pairs": int(complete_pairs),
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
            "incomplete completed latent score table; expected both "
            f"{table_path.name} and {manifest_path.name}"
        )
    manifest = json_text_to_dict(manifest_path)
    _validate_checkpoint_contract(manifest, contract)
    if manifest.get("ready_for_builder") is not True or manifest.get("score_claim") is not False:
        raise ValueError("existing latent score-table manifest is not reusable builder evidence")
    table = np.load(table_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(contract["score_table_shape"])
    if tuple(table.shape) != expected_shape:
        raise ValueError(f"completed latent score-table shape mismatch: expected {expected_shape}, got {table.shape}")
    if completed_prefix_frames(np.asarray(table)) != int(table.shape[0]):
        raise ValueError("completed latent score table contains non-finite rows")
    print(f"[latent-score-table] reusing completed table {table_path}", flush=True)
    return True


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


def apply_latent_candidates_torch(
    latents: torch.Tensor,
    candidates: torch.Tensor,
    *,
    scale: float = DELTA_SCALE,
) -> torch.Tensor:
    """Return latents expanded across candidate rows with latent deltas applied."""
    if latents.ndim != 2:
        raise ValueError(f"latents must be (pairs, latent_dim), got {tuple(latents.shape)}")
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError(f"candidates must be (n_candidates, 2), got {tuple(candidates.shape)}")
    pair_count, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    candidate_count = int(candidates.shape[0])
    expanded = (
        latents.unsqueeze(1)
        .repeat(1, candidate_count, 1)
        .reshape(pair_count * candidate_count, latent_dim)
        .clone()
    )
    repeated_candidates = candidates.repeat(pair_count, 1)
    dims = repeated_candidates[:, 0].long()
    deltas = repeated_candidates[:, 1].float()
    nonzero = (dims != NO_OP_DIM) & (deltas != 0)
    if bool(nonzero.any()):
        rows = torch.arange(expanded.shape[0], device=expanded.device)[nonzero]
        if int(dims[nonzero].max().item()) >= latent_dim:
            raise ValueError("candidate dimension exceeds latent_dim")
        expanded[rows, dims[nonzero]] = expanded[rows, dims[nonzero]] + deltas[nonzero] * float(scale)
    return expanded


@torch.inference_mode()
def score_pair_batch_candidate_table(
    distortion_net,
    decoder,
    *,
    gt_pairs: torch.Tensor,
    latents_batch: torch.Tensor,
    candidates: torch.Tensor,
    candidate_batch_size: int,
) -> np.ndarray:
    """Score latent-sidecar candidates for a batch of frame pairs."""
    if gt_pairs.ndim != 5 or gt_pairs.shape[1] != 2 or gt_pairs.shape[-1] != 3:
        raise ValueError(f"gt_pairs must be (P,2,H,W,3), got {tuple(gt_pairs.shape)}")
    if latents_batch.ndim != 2:
        raise ValueError(f"latents_batch must be (P,D), got {tuple(latents_batch.shape)}")
    if int(gt_pairs.shape[0]) != int(latents_batch.shape[0]):
        raise ValueError("gt_pairs and latents_batch pair counts differ")
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError(f"candidates must be (C,2), got {tuple(candidates.shape)}")
    pair_count = int(gt_pairs.shape[0])
    candidate_count = int(candidates.shape[0])
    rows = np.empty((pair_count, candidate_count), dtype=np.float32)
    for start in range(0, candidate_count, candidate_batch_size):
        cand = candidates[start:start + candidate_batch_size]
        cand_count = int(cand.shape[0])
        cand_latents = apply_latent_candidates_torch(latents_batch, cand)
        decoded = decoder(cand_latents)
        flat = decoded.reshape(pair_count * cand_count * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
        up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
        cand_pairs = (
            up.clamp(0, 255)
            .permute(0, 2, 3, 1)
            .round()
            .to(torch.uint8)
            .reshape(pair_count * cand_count, 2, CAMERA_H, CAMERA_W, 3)
        )
        gt_batch = (
            gt_pairs.unsqueeze(1)
            .repeat(1, cand_count, 1, 1, 1, 1)
            .reshape(pair_count * cand_count, 2, CAMERA_H, CAMERA_W, 3)
        )
        pose_dist, seg_dist = distortion_net.compute_distortion(gt_batch, cand_pairs)
        scores = score_without_rate(pose_dist, seg_dist).detach().cpu().numpy().astype(np.float32)
        rows[:, start:start + cand_count] = scores.reshape(pair_count, cand_count)
    return rows


def build_dry_run_plan(args: argparse.Namespace, candidates: np.ndarray) -> dict[str, Any]:
    n_pairs = int(args.n_pairs)
    if args.max_pairs is not None:
        n_pairs = min(n_pairs, int(args.max_pairs))
    return {
        "manifest_schema": "pr106_latent_score_table_plan_v1",
        "producer": "experiments/build_pr106_latent_score_table.py",
        "score_claim": False,
        "dry_run_plan": True,
        "ready_for_builder": False,
        "ready_for_exact_eval_dispatch": False,
        "source_archive_path": str(args.pr106_archive),
        "source_archive_sha256": sha256_file(args.pr106_archive) if args.pr106_archive.is_file() else None,
        "delta_radius": int(args.delta_radius),
        "latent_dim": int(args.latent_dim),
        "candidate_count": int(candidates.shape[0]),
        "n_pairs": n_pairs,
        "expected_score_table_shape": [n_pairs, int(candidates.shape[0])],
        "max_pairs": None if args.max_pairs is None else int(args.max_pairs),
        "objective": "100*seg_dist + sqrt(10*pose_dist), without rate constant",
        "pair_marginal_semantics": "one latent perturbation scored against the official two-frame pair",
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
    candidates_np = build_latent_candidate_grid(
        latent_dim=args.latent_dim,
        delta_radius=args.delta_radius,
    )
    candidates_path = args.out_dir / "candidate_grid.npy"
    np.save(candidates_path, candidates_np, allow_pickle=False)
    n_pairs = int(args.n_pairs)
    if args.max_pairs is not None:
        n_pairs = min(n_pairs, int(args.max_pairs))

    if args.dry_run_plan:
        plan = build_dry_run_plan(args, candidates_np)
        manifest_path = args.out_dir / SCORE_TABLE_MANIFEST
        manifest_path.write_text(json_text(plan), encoding="utf-8")
        print(f"[latent-score-table] wrote dry-run plan {manifest_path}")
        print(f"[latent-score-table] wrote candidate grid {candidates_path}")
        return 0

    if args.device != "cuda":
        raise SystemExit("real latent score-table generation requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("real latent score-table generation requires torch.cuda.is_available()")
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
        n_pairs=n_pairs,
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
    if int(meta["latent_dim"]) != int(args.latent_dim):
        raise SystemExit(f"requested latent_dim={args.latent_dim}, but PR106 archive has {meta['latent_dim']}")
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

    checkpoint = _load_score_table_checkpoint(args, contract=contract) if args.resume_checkpoint else None
    if checkpoint is None:
        table = np.full((n_pairs, candidates_np.shape[0]), np.nan, dtype=np.float32)
        resume_pair_cursor = 0
    else:
        table, resume_pair_cursor = checkpoint
    candidates = torch.from_numpy(candidates_np.astype(np.int16)).to(device)

    pair_cursor = 0
    for _, _, gt_batch in gt_loader:
        if pair_cursor >= n_pairs:
            break
        gt_rows = int(gt_batch.shape[0])
        batch_pairs = min(gt_rows, n_pairs - pair_cursor)
        if pair_cursor + batch_pairs <= resume_pair_cursor:
            pair_cursor += batch_pairs
            print(f"[latent-score-table] skipped checkpointed {pair_cursor}/{n_pairs} pairs", flush=True)
            continue
        local_start = max(0, resume_pair_cursor - pair_cursor)
        if local_start:
            gt_batch = gt_batch[local_start:]
            pair_cursor += local_start
            batch_pairs -= local_start
        gt_batch = gt_batch[:batch_pairs].to(device)
        scored_rows = score_pair_batch_candidate_table(
            distortion_net,
            decoder,
            gt_pairs=gt_batch,
            latents_batch=latents[pair_cursor:pair_cursor + batch_pairs],
            candidates=candidates,
            candidate_batch_size=args.candidate_batch_size,
        )
        table[pair_cursor:pair_cursor + batch_pairs] = scored_rows
        pair_cursor += batch_pairs
        print(f"[latent-score-table] scored {pair_cursor}/{n_pairs} pairs", flush=True)
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
        raise SystemExit("latent score table contains unfilled or non-finite entries")

    table_path = args.out_dir / SCORE_TABLE_NPY
    _atomic_save_npy(table_path, table)
    elapsed = time.time() - started
    noop_idx = int(np.flatnonzero((candidates_np[:, 0] == NO_OP_DIM) & (candidates_np[:, 1] == 0))[0])
    best_idx = table.argmin(axis=1)
    best_scores = table[np.arange(table.shape[0]), best_idx]
    noop_scores = table[:, noop_idx]
    strict_improvements = best_scores < noop_scores
    manifest = {
        "manifest_schema": "pr106_latent_score_table_manifest_v1",
        "producer": "experiments/build_pr106_latent_score_table.py",
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
        "source_zero_bin_sha256": sha256_bytes(pr106_bytes),
        "candidate_grid_path": str(candidates_path),
        "candidate_grid_sha256": sha256_file(candidates_path),
        "candidate_grid_npy_sha256": latent_candidate_grid_npy_sha256(candidates_np),
        "score_table_npy_path": str(table_path),
        "score_table_npy_bytes": int(table_path.stat().st_size),
        "score_table_npy_sha256": sha256_file(table_path),
        "delta_radius": int(args.delta_radius),
        "latent_dim": int(args.latent_dim),
        "candidate_count": int(candidates_np.shape[0]),
        "n_pairs": int(n_pairs),
        "score_table_shape": [int(table.shape[0]), int(table.shape[1])],
        "max_pairs": None if args.max_pairs is None else int(args.max_pairs),
        "objective": "100*seg_dist + sqrt(10*pose_dist), without rate constant",
        "pair_marginal_semantics": "one latent perturbation scored against the official two-frame pair",
        "noop_candidate_index": int(noop_idx),
        "strict_improvement_pair_count": int(strict_improvements.sum()),
        "best_improvement_min": float((noop_scores - best_scores).min()),
        "best_improvement_mean": float((noop_scores - best_scores).mean()),
        "best_improvement_max": float((noop_scores - best_scores).max()),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
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
    print(f"[latent-score-table] wrote table {table_path}")
    print(f"[latent-score-table] wrote manifest {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pr106-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--delta-radius", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=28)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Optional partial scorer-table cap for debugging. Non-promotable unless full.")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-pairs", type=int, default=2)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--prefetch-queue-depth", type=int, default=4)
    parser.add_argument("--video-names-file", type=Path,
                        default=REPO_ROOT / "upstream" / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path,
                        default=REPO_ROOT / "upstream" / "videos")
    parser.add_argument("--claims-path", type=Path,
                        default=REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md")
    parser.add_argument("--lane-id", default="lane_pr106_latent_score_table")
    parser.add_argument("--instance-job-id", default="",
                        help="Required for real CUDA scoring; must match an active lane claim row.")
    parser.add_argument("--dry-run-plan", action="store_true",
                        help="Write candidate-grid and plan metadata only; no CUDA/scorer/claim required.")
    parser.add_argument("--resume-checkpoint", action="store_true",
                        help="Resume and update score_table.partial.npy in --out-dir after each batch.")
    args = parser.parse_args()
    return build_score_table(args)


if __name__ == "__main__":
    raise SystemExit(main())
