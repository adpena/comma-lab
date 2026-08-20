#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preflight, run, resume, or verify an exact batch-16 candidate replay.

``run`` is a full CPU frozen-scorer pass and must be invoked through
``tools/safe_run.py``.  The default object is the original full-lattice C1
archive plus its already-realized raw output; no decode or input mutation is
performed.  The result is macOS-CPU advisory evidence, never a pointer update.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.witness_control.taskspace_candidate_batch_replay_v1 import (  # noqa: E402
    BatchDistancesV1,
    CandidateBatchReplayError,
    build_candidate_replay_preflight,
    load_and_reverify_candidate_replay_receipt,
    replay_candidate_batches,
    reverify_candidate_replay_preflight,
)
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (  # noqa: E402
    CAMERA_HW_PUBLIC,
    PAIR_COUNT_PUBLIC,
    UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
    atomic_write_json,
    load_json_mapping,
)

C1_ROOT = Path("/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719")
C1_RAW_SHA256 = "31d77be9ab9f00e9f814542368396a35ffa119a32571e701636d4747540e255b"
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/c1_batch16_exact_replay_20260726")
PACKAGE_DISTRIBUTIONS = (
    "av",
    "einops",
    "numpy",
    "safetensors",
    "segmentation-models-pytorch",
    "timm",
    "torch",
)
IMPLEMENTATION_FILES = (
    Path(__file__),
    REPO_ROOT / "src/tac/admission_guard.py",
    REPO_ROOT / "src/tac/contest_score.py",
    REPO_ROOT / "src/tac/witness_control/taskspace_candidate_batch_replay_v1.py",
    REPO_ROOT / "src/tac/witness_control/taskspace_fresh_teacher_materializer_v1.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "status"), required=True)
    parser.add_argument("--source-video", type=Path, default=REPO_ROOT / "upstream/videos/0.mkv")
    parser.add_argument("--upstream-root", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--candidate-archive", type=Path, default=C1_ROOT / "capstone_submission/archive.zip")
    parser.add_argument("--candidate-raw", type=Path, default=C1_ROOT / "capstone_submission/inflated/0.raw")
    parser.add_argument("--expected-candidate-raw-sha256", default=C1_RAW_SHA256)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--replay-receipt", type=Path)
    parser.add_argument("--pair-count", type=int, default=PAIR_COUNT_PUBLIC)
    parser.add_argument("--batch-size", type=int, default=UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--safety-reserve-gib", type=float, default=8.0)
    return parser


def _preflight_path(args: argparse.Namespace) -> Path:
    return (
        args.preflight_receipt
        if args.preflight_receipt is not None
        else args.output_root / "00_custody_storage_preflight.json"
    ).resolve()


def _receipt_path(args: argparse.Namespace) -> Path:
    return (
        args.replay_receipt
        if args.replay_receipt is not None
        else args.output_root / "11_batch_replay_receipt.json"
    ).resolve()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise CandidateBatchReplayError(f"required package is absent: {distribution}") from exc
    return versions


def _run_argv(preflight_path: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--mode",
        "run",
        "--preflight-receipt",
        str(preflight_path),
    ]


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    path = _preflight_path(args)
    expected = args.output_root.resolve() / "00_custody_storage_preflight.json"
    if path != expected:
        raise CandidateBatchReplayError(f"preflight receipt must be exact stage-00 path {expected}")
    if path.exists():
        value = load_json_mapping(path)
        reverify_candidate_replay_preflight(value)
        return value
    value = build_candidate_replay_preflight(
        source_video=args.source_video,
        candidate_archive=args.candidate_archive,
        candidate_raw=args.candidate_raw,
        expected_candidate_raw_sha256=args.expected_candidate_raw_sha256,
        upstream_root=args.upstream_root,
        output_root=args.output_root,
        pair_count=args.pair_count,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        seed=args.seed,
        package_versions=_package_versions(),
        implementation_files=IMPLEMENTATION_FILES,
        run_argv=_run_argv(path),
        camera_hw=CAMERA_HW_PUBLIC,
        safety_reserve_bytes=int(args.safety_reserve_gib * (1 << 30)),
    )
    atomic_write_json(path, value)
    reopened = load_json_mapping(path)
    reverify_candidate_replay_preflight(reopened)
    if reopened != value:
        raise CandidateBatchReplayError("preflight changed across parse-back")
    return reopened


def _configure_determinism(*, seed: int, num_threads: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


def _paired_batches(preflight: dict[str, Any]):
    import torch

    upstream_root = Path(str(preflight["upstream_closure"]["root"]))
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    from frame_utils import AVVideoDataset

    source_path = Path(str(preflight["source_video"]["path"]))
    dataset = AVVideoDataset(
        [source_path.name],
        data_dir=source_path.parent,
        batch_size=int(preflight["batch_size"]),
        device=torch.device("cpu"),
        num_threads=int(preflight["num_threads"]),
        seed=int(preflight["seed"]),
        prefetch_queue_depth=1,
    )
    dataset.prepare_data()
    pair_count = int(preflight["pair_count"])
    height, width = (int(value) for value in preflight["camera_hw"])
    raw_path = Path(str(preflight["candidate_raw"]["path"]))
    candidate = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(pair_count, 2, height, width, 3))
    offset = 0
    for _path, _batch_index, batch in dataset:
        if offset >= pair_count:
            break
        source = np.ascontiguousarray(batch.cpu().numpy(), dtype=np.uint8)
        count = min(int(source.shape[0]), pair_count - offset)
        yield source[:count], np.array(candidate[offset : offset + count], copy=True, order="C")
        offset += count
    del candidate
    if offset != pair_count:
        raise CandidateBatchReplayError(f"source stream ended at pair {offset}, expected {pair_count}")


def _load_distortion_net(preflight: dict[str, Any]):
    import torch

    upstream_root = Path(str(preflight["upstream_closure"]["root"]))
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    net = DistortionNet().eval().to(device=torch.device("cpu"))
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def _score_callback(net):
    import torch

    def score(source: np.ndarray, candidate: np.ndarray) -> BatchDistancesV1:
        source_tensor = torch.from_numpy(np.ascontiguousarray(source))
        candidate_tensor = torch.from_numpy(np.ascontiguousarray(candidate))
        with torch.inference_mode():
            d_pose, d_seg = net.compute_distortion(source_tensor, candidate_tensor)
            pose_sum = d_pose.sum().item()
            seg_sum = d_seg.sum().item()
        return BatchDistancesV1(
            d_pose=np.ascontiguousarray(d_pose.cpu().numpy()),
            d_seg=np.ascontiguousarray(d_seg.cpu().numpy()),
            d_pose_batch_sum_f32=float(pose_sum),
            d_seg_batch_sum_f32=float(seg_sum),
        )

    return score


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    assert_governed_admission("taskspace_candidate_batch16_replay_v1", on_refuse="raise")
    preflight_path = _preflight_path(args)
    if not preflight_path.is_file():
        raise CandidateBatchReplayError(f"run requires stage-00 preflight: {preflight_path}")
    preflight = load_json_mapping(preflight_path)
    reverify_candidate_replay_preflight(preflight)
    if preflight.get("package_versions") != _package_versions():
        raise CandidateBatchReplayError("runtime package versions drifted from stage-00")
    if not preflight.get("batch_geometry_matches_upstream_default"):
        raise CandidateBatchReplayError("production replay refuses non-default upstream batch geometry")
    _configure_determinism(seed=int(preflight["seed"]), num_threads=int(preflight["num_threads"]))
    net = _load_distortion_net(preflight)
    return replay_candidate_batches(
        preflight=preflight,
        paired_batches=_paired_batches(preflight),
        score_batch=_score_callback(net),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "preflight":
            value = run_preflight(args)
        elif args.mode == "run":
            value = run_replay(args)
        else:
            value = load_and_reverify_candidate_replay_receipt(_receipt_path(args))
    except (CandidateBatchReplayError, OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
