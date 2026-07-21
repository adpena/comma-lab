#!/usr/bin/env python3
"""Decompose one exact n600 witness row without changing the frozen scorer.

The aggregate row is computed with the same ``DistortionNet.compute_distortion``
call and float32 accumulation geometry as
``tools/measure_r1b_boundary_generator_n600.py``.  A forward hook only observes
the two SegNet outputs already produced by that call; it does not replace or
modify any scorer operation.  The observed argmax stacks are decomposed with
the canonical v2 residual-target helper.

This is a diagnostic adapter, not a score authority.  Every emitted row is
``[macOS-CPU advisory]`` and the contest pointer remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_ROOT: Final = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tac.v2_compose.residual_target import compute_residual_target  # noqa: E402

SCHEMA: Final = "r1b_n600_hard_oracle_decomposition.v1"
CLASS_NAMES: Final = ("Road", "Lane", "Undriv", "Movable", "MyCar")
PAIR_COUNT: Final = 600
FRAME_COUNT: Final = 1200
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CHANNELS: Final = 3
EXPECTED_RAW_BYTES: Final = FRAME_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS
SCORER_FILES: Final = (
    ("modules.py", "modules.py"),
    ("frame_utils.py", "frame_utils.py"),
    ("posenet.safetensors", "models/posenet.safetensors"),
    ("segnet.safetensors", "models/segnet.safetensors"),
)


class DecompositionError(RuntimeError):
    """Fail-closed input, scorer, or output-custody error."""


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise DecompositionError(f"output overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if partial.exists():
            partial.unlink()


def _append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_label_chunk(
    path: Path,
    *,
    pair_index: np.ndarray,
    site_index: np.ndarray,
    gt_class: np.ndarray,
    candidate_class: np.ndarray,
) -> dict[str, Any]:
    """Persist one write-once, receiver-closed mismatch chunk atomically.

    A cut may occur after the atomic chunk rename but before the JSONL checkpoint
    append.  In that narrow window the next run re-derives the same arrays and
    accepts the orphan only when every member is byte-value identical.
    """
    arrays = {
        "pair_index": np.asarray(pair_index, dtype=np.uint16),
        "site_index": np.asarray(site_index, dtype=np.uint32),
        "gt_class": np.asarray(gt_class, dtype=np.uint8),
        "candidate_class": np.asarray(candidate_class, dtype=np.uint8),
    }
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(
                    not np.array_equal(saved[name], value) for name, value in arrays.items()
                ):
                    raise DecompositionError(f"label-cache orphan conflicts with replay: {path}")
        except (OSError, ValueError) as exc:
            raise DecompositionError(f"label-cache orphan is unreadable: {path}") from exc
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "mismatch_sites": len(arrays["site_index"]),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mismatch_sites": len(site_index),
    }


def _validate_label_chunks(
    rows: list[dict[str, Any]], *, expected_count: int, expected_mismatch_sites: int
) -> None:
    if len(rows) != expected_count:
        raise DecompositionError(
            f"label-cache checkpoint has {len(rows)} chunks for {expected_count} completed batches"
        )
    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_file():
            raise DecompositionError(f"label-cache chunk missing on resume: {path}")
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise DecompositionError(f"label-cache chunk custody drift on resume: {path}")
    observed_sites = sum(int(row["mismatch_sites"]) for row in rows)
    if observed_sites != expected_mismatch_sites:
        raise DecompositionError(
            "label-cache mismatch count drift on resume: "
            f"chunks={observed_sites}, checkpoint={expected_mismatch_sites}"
        )


def _load_checkpoint(path: Path, *, contract_sha256: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    state = None
    for row in reversed(rows):
        try:
            state = json.loads(row)
            break
        except json.JSONDecodeError:
            # A process cut may leave only the newest append partial. Older
            # fsync-complete batch rows remain valid resume authorities.
            continue
    if state is None:
        raise DecompositionError("checkpoint contains no valid JSON row")
    if state.get("schema") != f"{SCHEMA}.checkpoint.v1":
        raise DecompositionError("checkpoint schema mismatch")
    if state.get("contract_sha256") != contract_sha256:
        raise DecompositionError("checkpoint input contract mismatch")
    return state


def summarize_label_batch(
    gt_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> dict[str, Any]:
    """Return canonical mismatch and per-GT-class integer counts for one batch."""
    residual = compute_residual_target(
        candidate_labels,
        gt_labels,
        class_names=CLASS_NAMES,
    )
    mismatch = residual.residual_mask
    per_class: dict[str, dict[str, int]] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        gt_mask = residual.gt_lstars == class_id
        per_class[name] = {
            "gt_pixels": int(np.count_nonzero(gt_mask)),
            "mismatch_pixels": int(np.count_nonzero(mismatch & gt_mask)),
        }
    pair_mismatch = np.count_nonzero(mismatch, axis=(1, 2)).astype(np.int64)
    return {
        "total_pixels": int(mismatch.size),
        "mismatch_pixels": int(np.count_nonzero(mismatch)),
        "pair_mismatch_pixels": pair_mismatch.tolist(),
        "per_class": per_class,
    }


def top_pair_rows(values: list[float], *, top_k: int) -> list[dict[str, Any]]:
    """Stable descending top-k with pair index as the tie breaker."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))[:top_k]
    return [{"pair_index": index, "value": values[index]} for index in order]


def _source_hashes(upstream: Path) -> dict[str, str]:
    return {
        label: sha256_file((upstream / relative).resolve(strict=True))
        for label, relative in SCORER_FILES
    }


def _git_custody() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {"head": head, "dirty_paths": status}


def _measure(
    *,
    raw_path: Path,
    upstream: Path,
    batch_size: int,
    cpu_threads: int,
    seed: int,
    top_k: int,
    checkpoint_path: Path,
    contract_sha256: str,
    label_cache_dir: Path | None,
) -> dict[str, Any]:
    if raw_path.name != "0.raw":
        raise DecompositionError("TensorVideoDataset custody requires a path named 0.raw")
    if raw_path.stat().st_size != EXPECTED_RAW_BYTES:
        raise DecompositionError(
            f"raw size {raw_path.stat().st_size} != expected n600 size {EXPECTED_RAW_BYTES}"
        )
    if batch_size not in (16, 32):
        raise DecompositionError("hard CPU decomposition supports evaluator batch_size in {16,32}")
    if cpu_threads <= 0:
        raise DecompositionError("cpu_threads must be positive")

    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    upstream_string = str(upstream)
    if upstream_string not in sys.path:
        sys.path.insert(0, upstream_string)
    try:
        from frame_utils import AVVideoDataset, TensorVideoDataset
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path
    except ImportError as exc:
        raise DecompositionError("failed to import the explicit upstream scorer") from exc

    device = torch.device("cpu")
    scorer = DistortionNet().eval().to(device=device)
    scorer.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    ground_truth = AVVideoDataset(
        ["0.mkv"],
        data_dir=upstream / "videos",
        batch_size=batch_size,
        device=device,
        num_threads=1,
        seed=seed,
        prefetch_queue_depth=1,
    )
    candidate = TensorVideoDataset(
        ["0.mkv"],
        data_dir=raw_path.parent,
        batch_size=batch_size,
        device=device,
        num_threads=1,
        seed=seed,
        prefetch_queue_depth=1,
    )
    ground_truth.prepare_data()
    candidate.prepare_data()
    gt_loader = torch.utils.data.DataLoader(ground_truth, batch_size=None, num_workers=0)
    candidate_loader = torch.utils.data.DataLoader(candidate, batch_size=None, num_workers=0)

    state = _load_checkpoint(checkpoint_path, contract_sha256=contract_sha256)
    pose_sum = torch.tensor(
        float(state["pose_sum"]) if state else 0.0,
        device=device,
        dtype=torch.float32,
    )
    seg_sum = torch.tensor(
        float(state["seg_sum"]) if state else 0.0,
        device=device,
        dtype=torch.float32,
    )
    pair_pose: list[float] = []
    pair_seg: list[float] = []
    pair_mismatch: list[int] = []
    per_class_counts = {
        name: {"gt_pixels": 0, "mismatch_pixels": 0} for name in CLASS_NAMES
    }
    total_pixels = 0
    mismatch_pixels = 0
    sample_count = 0
    batch_count = 0
    label_chunks: list[dict[str, Any]] = []
    if state:
        pair_pose = [float(value) for value in state["pair_pose"]]
        pair_seg = [float(value) for value in state["pair_seg"]]
        pair_mismatch = [int(value) for value in state["pair_mismatch"]]
        per_class_counts = {
            name: {
                "gt_pixels": int(state["per_class_counts"][name]["gt_pixels"]),
                "mismatch_pixels": int(state["per_class_counts"][name]["mismatch_pixels"]),
            }
            for name in CLASS_NAMES
        }
        total_pixels = int(state["total_pixels"])
        mismatch_pixels = int(state["mismatch_pixels"])
        sample_count = int(state["sample_count"])
        batch_count = int(state["batch_count"])
        label_chunks = [dict(row) for row in state.get("label_chunks", [])]
        if label_cache_dir is not None:
            _validate_label_chunks(
                label_chunks,
                expected_count=batch_count,
                expected_mismatch_sites=mismatch_pixels,
            )
    resume_batch_count = batch_count
    captured_segnet: list[Any] = []

    def capture_segnet(_module: Any, _inputs: Any, output: Any) -> None:
        captured_segnet.append(output.detach())

    hook = scorer.segnet.register_forward_hook(capture_segnet)
    try:
        with torch.inference_mode():
            for loader_batch_index, (gt_row, candidate_row) in enumerate(
                zip(gt_loader, candidate_loader, strict=True)
            ):
                if loader_batch_index < resume_batch_count:
                    continue
                batch_gt = gt_row[2].to(device)
                batch_candidate = candidate_row[2].to(device)
                if batch_gt.shape != batch_candidate.shape:
                    raise DecompositionError("ground-truth/candidate batch geometry mismatch")
                captured_segnet.clear()
                pose_dist, seg_dist = scorer.compute_distortion(batch_gt, batch_candidate)
                if len(captured_segnet) != 2:
                    raise DecompositionError(
                        f"expected two SegNet observations inside compute_distortion, got {len(captured_segnet)}"
                    )
                if pose_dist.shape != (batch_gt.shape[0],) or seg_dist.shape != (
                    batch_gt.shape[0],
                ):
                    raise DecompositionError("scorer distortion output geometry mismatch")
                gt_labels = captured_segnet[0].argmax(dim=1).cpu().numpy()
                candidate_labels = captured_segnet[1].argmax(dim=1).cpu().numpy()
                label_row = summarize_label_batch(gt_labels, candidate_labels)
                official_pair_seg = seg_dist.cpu().tolist()
                exact_pair_seg = [
                    count / (gt_labels.shape[1] * gt_labels.shape[2])
                    for count in label_row["pair_mismatch_pixels"]
                ]
                if not np.allclose(official_pair_seg, exact_pair_seg, rtol=0.0, atol=2e-8):
                    raise DecompositionError("observed argmax decomposition drifted from official SegNet distortion")

                if label_cache_dir is not None:
                    mismatch_batch, mismatch_y, mismatch_x = np.nonzero(
                        candidate_labels != gt_labels
                    )
                    pair_indices = mismatch_batch.astype(np.int64) + sample_count
                    site_indices = (
                        mismatch_y.astype(np.int64) * gt_labels.shape[2]
                        + mismatch_x.astype(np.int64)
                    )
                    chunk_path = label_cache_dir / (
                        f"mismatch_{sample_count:04d}_{sample_count + batch_gt.shape[0]:04d}.npz"
                    )
                    label_chunks.append(
                        _write_label_chunk(
                            chunk_path,
                            pair_index=pair_indices,
                            site_index=site_indices,
                            gt_class=gt_labels[mismatch_batch, mismatch_y, mismatch_x],
                            candidate_class=candidate_labels[mismatch_batch, mismatch_y, mismatch_x],
                        )
                    )

                pose_sum += pose_dist.sum()
                seg_sum += seg_dist.sum()
                pair_pose.extend(float(value) for value in pose_dist.cpu().tolist())
                pair_seg.extend(float(value) for value in official_pair_seg)
                pair_mismatch.extend(int(value) for value in label_row["pair_mismatch_pixels"])
                total_pixels += int(label_row["total_pixels"])
                mismatch_pixels += int(label_row["mismatch_pixels"])
                for name in CLASS_NAMES:
                    per_class_counts[name]["gt_pixels"] += label_row["per_class"][name]["gt_pixels"]
                    per_class_counts[name]["mismatch_pixels"] += label_row["per_class"][name][
                        "mismatch_pixels"
                    ]
                sample_count += int(batch_gt.shape[0])
                batch_count += 1
                _append_checkpoint(
                    checkpoint_path,
                    {
                        "schema": f"{SCHEMA}.checkpoint.v1",
                        "contract_sha256": contract_sha256,
                        "batch_count": batch_count,
                        "sample_count": sample_count,
                        "pose_sum": float(pose_sum.item()),
                        "seg_sum": float(seg_sum.item()),
                        "pair_pose": pair_pose,
                        "pair_seg": pair_seg,
                        "pair_mismatch": pair_mismatch,
                        "per_class_counts": per_class_counts,
                        "total_pixels": total_pixels,
                        "mismatch_pixels": mismatch_pixels,
                        "label_chunks": label_chunks,
                    },
                )
                print(
                    json.dumps(
                        {
                            "batch_count": batch_count,
                            "checkpoint": str(checkpoint_path),
                            "pair_count": sample_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        hook.remove()

    if sample_count != PAIR_COUNT or len(pair_pose) != PAIR_COUNT or len(pair_seg) != PAIR_COUNT:
        raise DecompositionError(f"hard scorer covered {sample_count} pairs, expected {PAIR_COUNT}")
    if sum(pair_mismatch) != mismatch_pixels:
        raise DecompositionError("pair mismatch accounting is not closed")

    official_d_seg = float((seg_sum / sample_count).item())
    official_d_pose = float((pose_sum / sample_count).item())
    exact_d_seg = mismatch_pixels / total_pixels
    per_class: dict[str, dict[str, float | int]] = {}
    for name in CLASS_NAMES:
        gt_pixels = per_class_counts[name]["gt_pixels"]
        mismatches = per_class_counts[name]["mismatch_pixels"]
        per_class[name] = {
            "gt_pixels": gt_pixels,
            "mismatch_pixels": mismatches,
            "area_fraction": gt_pixels / total_pixels,
            "d_seg_contribution_all_pixels": mismatches / total_pixels,
            "conditional_error_rate": mismatches / gt_pixels if gt_pixels else 0.0,
        }
    if sum(int(row["mismatch_pixels"]) for row in per_class.values()) != mismatch_pixels:
        raise DecompositionError("per-class mismatch accounting is not closed")
    if not math.isclose(official_d_seg, exact_d_seg, rel_tol=0.0, abs_tol=2e-8):
        raise DecompositionError("aggregate exact label comparison drifted from official d_seg")

    top_seg = top_pair_rows(pair_seg, top_k=top_k)
    for row in top_seg:
        row["d_seg"] = row.pop("value")
        row["mismatch_pixels"] = pair_mismatch[int(row["pair_index"])]
    top_pose = top_pair_rows(pair_pose, top_k=top_k)
    for row in top_pose:
        row["d_pose"] = row.pop("value")

    return {
        "aggregate": {
            "d_seg_official_float32": official_d_seg,
            "d_seg_exact_argmax_rational": exact_d_seg,
            "d_seg_crosscheck_abs_delta": abs(official_d_seg - exact_d_seg),
            "d_pose_official_float32": official_d_pose,
            "pair_count": sample_count,
            "batch_count": batch_count,
            "batch_size": batch_size,
        },
        "per_class": per_class,
        "top_pairs_by_d_seg": top_seg,
        "top_pairs_by_d_pose": top_pose,
        "integer_accounting": {
            "total_segnet_pixels": total_pixels,
            "mismatch_pixels": mismatch_pixels,
            "per_pair_mismatch_sum": sum(pair_mismatch),
        },
        "runtime": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cpu_threads": torch.get_num_threads(),
            "seed": seed,
            "resumed_from_batch_count": resume_batch_count,
            "checkpoint_jsonl": str(checkpoint_path),
        },
        "label_cache": {
            "status": "WRITE_ONCE_MISMATCH_SITES" if label_cache_dir is not None else "NOT_REQUESTED",
            "directory": None if label_cache_dir is None else str(label_cache_dir),
            "chunks": label_chunks,
            "mismatch_sites": sum(int(row["mismatch_sites"]) for row in label_chunks),
            "coordinate_contract": "pair_index plus row-major site_index on SegNet 384x512 output",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-jsonl", type=Path, required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--expected-d-seg", type=float)
    parser.add_argument("--expected-d-pose", type=float)
    parser.add_argument("--label-cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.expected_d_seg is None) != (args.expected_d_pose is None):
        raise DecompositionError("--expected-d-seg and --expected-d-pose must be provided together")
    started = time.time()
    raw_path = args.raw.expanduser().resolve(strict=True)
    upstream = args.upstream.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    checkpoint_path = args.checkpoint_jsonl.expanduser().resolve()
    label_cache_dir = (
        None if args.label_cache_dir is None else args.label_cache_dir.expanduser().resolve()
    )
    tool_path = Path(__file__).resolve(strict=True)
    source_video = (upstream / "videos" / "0.mkv").resolve(strict=True)
    raw_sha256 = sha256_file(raw_path)
    if raw_sha256 != args.expected_raw_sha256:
        raise DecompositionError(
            f"raw sha256 {raw_sha256} != expected {args.expected_raw_sha256}"
        )
    scorer_hashes = _source_hashes(upstream)
    contract_payload = {
        "raw_sha256": raw_sha256,
        "upstream": str(upstream),
        "scorer_hashes": scorer_hashes,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "seed": args.seed,
        "top_k": args.top_k,
        "expected_d_seg": args.expected_d_seg,
        "expected_d_pose": args.expected_d_pose,
        "label_cache_dir": None if label_cache_dir is None else str(label_cache_dir),
    }
    contract_sha256 = sha256(
        json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    measurement = _measure(
        raw_path=raw_path,
        upstream=upstream,
        batch_size=args.batch_size,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
        top_k=args.top_k,
        checkpoint_path=checkpoint_path,
        contract_sha256=contract_sha256,
        label_cache_dir=label_cache_dir,
    )
    aggregate = measurement["aggregate"]
    d_seg_delta = (
        None
        if args.expected_d_seg is None
        else aggregate["d_seg_official_float32"] - args.expected_d_seg
    )
    d_pose_delta = (
        None
        if args.expected_d_pose is None
        else aggregate["d_pose_official_float32"] - args.expected_d_pose
    )
    if args.expected_d_seg is not None and d_seg_delta != 0.0:
        raise DecompositionError(f"official d_seg anchor drift: delta={d_seg_delta}")
    if args.expected_d_pose is not None and d_pose_delta != 0.0:
        raise DecompositionError(f"official d_pose anchor drift: delta={d_pose_delta}")

    payload = {
        "schema": SCHEMA,
        "verdict": (
            "MEASURED_N600_HARD_ORACLE_DECOMPOSITION_MATCHES_SETTLED_CONTROL"
            if args.expected_d_seg is not None and args.expected_d_pose is not None
            else "MEASURED_N600_HARD_ORACLE_DECOMPOSITION"
        ),
        "verdict_scope": (
            "one exact receiver-output instance; diagnostic decomposition only; "
            "no contest score claim, no family negative"
        ),
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "hard_cpu_torch": True,
            "through_r": True,
            "contest_score_claim": False,
            "pointer_mutation": False,
        },
        "measurement": measurement,
        "settled_control_crosscheck": {
            "expected_d_seg": args.expected_d_seg,
            "measured_d_seg": aggregate["d_seg_official_float32"],
            "d_seg_delta": d_seg_delta,
            "expected_d_pose": args.expected_d_pose,
            "measured_d_pose": aggregate["d_pose_official_float32"],
            "d_pose_delta": d_pose_delta,
        },
        "custody": {
            "raw": {
                "path": str(raw_path),
                "bytes": raw_path.stat().st_size,
                "sha256": raw_sha256,
            },
            "tool": {
                "path": str(tool_path),
                "sha256": sha256_file(tool_path),
            },
            "upstream": str(upstream),
            "scorer_hashes": scorer_hashes,
            "source_video": {
                "path": str(source_video),
                "bytes": source_video.stat().st_size,
                "sha256": sha256_file(source_video),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
                "contract": contract_payload,
                "contract_sha256": contract_sha256,
                "preserved_batch_rows": sum(1 for line in checkpoint_path.read_text().splitlines() if line),
            },
            "git": _git_custody(),
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "runtime_seconds": time.time() - started,
        "pointer": "0.19108 [contest-CPU] UNMOVED",
    }
    _atomic_json(output, payload)
    print(json.dumps({"output": str(output), "verdict": payload["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
