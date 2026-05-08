#!/usr/bin/env python3
"""Probe evaluator loader drift between CUDA DALI/NVDEC and CPU PyAV.

The public scorer changes two things between ``--device cuda`` and
``--device cpu``:

* CUDA ground-truth video path: ``DaliVideoDataset`` with GPU/NVDEC decode.
* CPU ground-truth video path: ``AVVideoDataset`` with PyAV/FFmpeg decode.

This tool compares those decoded RGB uint8 tensors before PoseNet or SegNet.
It is diagnostic only and never a score claim. If CUDA/DALI is unavailable it
still writes a non-promotable plan artifact so the exact remote command is
auditable.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().to(device="cpu", dtype=torch.float64)
    if t.numel() == 0:
        return {"shape": list(t.shape), "numel": 0}
    return {
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "rms": float(torch.sqrt(torch.mean(t * t)).item()),
    }


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    aa = a.detach().to(device="cpu", dtype=torch.float64)
    bb = b.detach().to(device="cpu", dtype=torch.float64)
    if aa.shape != bb.shape:
        return {
            "shape_match": False,
            "shape_a": list(aa.shape),
            "shape_b": list(bb.shape),
        }
    diff = aa - bb
    abs_diff = diff.abs()
    if diff.numel() == 0:
        return {"shape_match": True, "numel": 0}
    return {
        "shape_match": True,
        "shape": list(diff.shape),
        "numel": int(diff.numel()),
        "max_abs_lsb": float(abs_diff.max().item()),
        "mean_abs_lsb": float(abs_diff.mean().item()),
        "rms_abs_lsb": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "nonzero_fraction": float((abs_diff > 0).to(torch.float64).mean().item()),
    }


def per_channel_compare(a: torch.Tensor, b: torch.Tensor) -> list[dict[str, Any]]:
    # Raw evaluator batches are (B, T, H, W, C).
    if a.shape != b.shape or a.ndim < 1:
        return []
    channel_dim = a.ndim - 1
    if a.shape[channel_dim] not in {1, 3, 6, 12}:
        return []
    rows = []
    for idx in range(a.shape[channel_dim]):
        rows.append({"channel": idx, **compare_tensors(a.select(channel_dim, idx), b.select(channel_dim, idx))})
    return rows


def _device_metadata() -> dict[str, Any]:
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "matmul_allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }
    if torch.cuda.is_available():
        cuda.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name_0": torch.cuda.get_device_name(0),
                "capability_0": list(torch.cuda.get_device_capability(0)),
            }
        )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": cuda,
        "mps_available": bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available(),
    }


def _load_video_names(path: Path, limit: int | None) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        return names[:limit]
    return names


def _cuda_dali_available() -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false"
    try:
        import nvidia.dali  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on CUDA environment
        return False, f"nvidia.dali import failed: {exc}"
    return True, None


def _batch_iterator(
    dataset: torch.utils.data.IterableDataset,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)


def _next_batch(iterator: Any) -> tuple[str, int, torch.Tensor]:
    path, idx, batch = next(iterator)
    return str(path), int(idx), batch


def _next_batch_or_none(iterator: Any) -> tuple[str, int, torch.Tensor] | None:
    try:
        return _next_batch(iterator)
    except StopIteration:
        return None


def build_probe_report(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "eval_loader_device_drift_probe.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "evidence_grade": "diagnostic_loader_drift_probe",
        "repo": str(REPO),
        "upstream_evaluate": str(UPSTREAM / "evaluate.py"),
        "upstream_frame_utils": str(UPSTREAM / "frame_utils.py"),
        "video_names_file": str(args.video_names_file),
        "data_dir": str(args.data_dir),
        "batch_size": args.batch_size,
        "num_threads": args.num_threads,
        "prefetch_queue_depth": args.prefetch_queue_depth,
        "video_limit": args.video_limit,
        "max_batches": args.max_batches,
        "environment": _device_metadata(),
        "interpretation_guardrails": [
            "This probe compares decoded evaluator input tensors before PoseNet/SegNet.",
            "It does not run inflate.sh, evaluate.py scoring, or any contest promotion path.",
            "A nonzero DALI-vs-PyAV raw RGB diff proves loader drift exists, but score impact still requires paired exact eval.",
        ],
    }
    cuda_ok, cuda_reason = _cuda_dali_available()
    report["comparison_available"] = cuda_ok
    report["comparison_unavailable_reason"] = cuda_reason
    if not cuda_ok:
        return report

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from frame_utils import AVVideoDataset, DaliVideoDataset, camera_size, seq_len  # type: ignore

    video_names = _load_video_names(args.video_names_file, args.video_limit)
    cuda_device = torch.device("cuda", 0)
    dali = DaliVideoDataset(
        video_names,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=cuda_device,
        num_threads=args.num_threads,
        seed=args.seed,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    av = AVVideoDataset(
        video_names,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=torch.device("cpu"),
        num_threads=args.num_threads,
        seed=args.seed,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    dali.prepare_data()
    av.prepare_data()

    rows = []
    dali_iter = iter(_batch_iterator(dali))
    av_iter = iter(_batch_iterator(av))
    incomplete_reason: str | None = None
    for batch_idx in range(args.max_batches):
        dali_next = _next_batch_or_none(dali_iter)
        av_next = _next_batch_or_none(av_iter)
        if dali_next is None or av_next is None:
            incomplete_reason = (
                "dataset_iterator_exhausted_before_requested_max_batches"
            )
            break
        dali_path, dali_seq_idx, dali_batch = dali_next
        av_path, av_seq_idx, av_batch = av_next
        dali_cpu = dali_batch.detach().to(device="cpu")
        av_cpu = av_batch.detach().to(device="cpu")
        if list(dali_cpu.shape)[1:] != [seq_len, camera_size[1], camera_size[0], 3]:
            raise RuntimeError(f"unexpected DALI batch shape: {list(dali_cpu.shape)}")
        if list(av_cpu.shape)[1:] != [seq_len, camera_size[1], camera_size[0], 3]:
            raise RuntimeError(f"unexpected AV batch shape: {list(av_cpu.shape)}")
        rows.append(
            {
                "batch_order": batch_idx,
                "dali_path": dali_path,
                "av_path": av_path,
                "dali_sequence_index": dali_seq_idx,
                "av_sequence_index": av_seq_idx,
                "path_match": dali_path == av_path,
                "sequence_index_match": dali_seq_idx == av_seq_idx,
                "dali_stats": tensor_stats(dali_cpu),
                "av_stats": tensor_stats(av_cpu),
                "comparison": compare_tensors(dali_cpu, av_cpu),
                "per_rgb_channel": per_channel_compare(dali_cpu, av_cpu),
            }
        )
    report["comparison_rows"] = rows
    report["comparison_available"] = bool(rows)
    report["comparison_incomplete"] = incomplete_reason is not None
    report["comparison_incomplete_reason"] = incomplete_reason
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-names-file", type=Path, default=UPSTREAM / "public_test_video_names.txt")
    parser.add_argument("--data-dir", type=Path, default=UPSTREAM / "videos")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--prefetch-queue-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--video-limit", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_probe_report(args)
    text = json.dumps(_jsonable(report), indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report.get("comparison_available") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
