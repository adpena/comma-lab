#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preflight, materialize, resume and reverify fresh task-space teacher labels.

The ``preflight`` mode is read-only with respect to source/scorer inputs and
writes only the small stage-00 custody receipt.  The ``run`` mode is a heavy
CPU scorer pass and therefore asserts governed admission; invoke it through
``tools/safe_run.py`` with an explicit RSS/timeout budget.

No output from this tool is candidate payload.  Target labels, scorer inputs,
weights and checkpoints are encoder-only evidence and are forbidden from
``archive.zip`` and ``inflate.py``.
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
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (  # noqa: E402
    CAMERA_HW_PUBLIC,
    PAIR_COUNT_PUBLIC,
    SEG_HW_PUBLIC,
    UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
    FreshTeacherMaterializationError,
    PreparedTeacherBatchV1,
    atomic_write_json,
    build_fresh_teacher_preflight,
    load_and_reverify_materialization_receipt,
    load_json_mapping,
    materialize_fresh_teacher_from_batches,
    projected_materialization_bytes,
    reverify_preflight,
    sha256_file,
)

PACKAGE_DISTRIBUTIONS = (
    "av",
    "einops",
    "numpy",
    "safetensors",
    "segmentation-models-pytorch",
    "timm",
    "torch",
)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise FreshTeacherMaterializationError(
                f"required package distribution is unavailable: {distribution}"
            ) from exc
    return versions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "status"), required=True)
    parser.add_argument("--source-video", type=Path, default=REPO_ROOT / "upstream/videos/0.mkv")
    parser.add_argument("--upstream-root", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_20260726"),
    )
    parser.add_argument("--preflight-receipt", type=Path, default=None)
    parser.add_argument("--materialization-receipt", type=Path, default=None)
    parser.add_argument("--pair-count", type=int, default=PAIR_COUNT_PUBLIC)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
        help="pair batch geometry; default exactly matches frozen upstream evaluate.py",
    )
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


def _materialization_path(args: argparse.Namespace) -> Path:
    return (
        args.materialization_receipt
        if args.materialization_receipt is not None
        else args.output_root / "12_encoder_only_receipt.json"
    ).resolve()


def _run_argv(preflight_path: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--mode",
        "run",
        "--preflight-receipt",
        str(preflight_path.resolve()),
    ]


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    preflight_path = _preflight_path(args)
    output_root = args.output_root.resolve()
    expected_path = output_root / "00_custody_storage_preflight.json"
    if preflight_path != expected_path:
        raise FreshTeacherMaterializationError(
            f"preflight receipt must be the stage-00 path {expected_path}, got {preflight_path}"
        )
    safety_reserve = int(float(args.safety_reserve_gib) * (1 << 30))
    if preflight_path.exists():
        existing = load_json_mapping(preflight_path)
        reverify_preflight(existing)
        expected_static = {
            "output_root": str(output_root),
            "pair_count": args.pair_count,
            "batch_size": args.batch_size,
            "num_threads": args.num_threads,
            "seed": args.seed,
            "camera_hw": list(CAMERA_HW_PUBLIC),
            "seg_hw": list(SEG_HW_PUBLIC),
            "package_versions": package_versions(),
            "run_argv": _run_argv(preflight_path),
        }
        for field, expected in expected_static.items():
            if existing.get(field) != expected:
                raise FreshTeacherMaterializationError(
                    f"existing stage-00 preflight {field} differs; choose a new output root"
                )
        if Path(str(existing["source_video"]["path"])) != args.source_video.resolve():
            raise FreshTeacherMaterializationError("existing stage-00 preflight names another source video")
        if Path(str(existing["upstream_closure"]["root"])) != args.upstream_root.resolve():
            raise FreshTeacherMaterializationError("existing stage-00 preflight names another upstream closure")
        required = projected_materialization_bytes(
            pair_count=args.pair_count,
            seg_hw=SEG_HW_PUBLIC,
            safety_reserve_bytes=safety_reserve,
        )
        if existing["storage_preflight"]["required_free_bytes"] != required:
            raise FreshTeacherMaterializationError(
                "existing stage-00 preflight uses another storage reserve; choose a new output root"
            )
        return existing
    preflight = build_fresh_teacher_preflight(
        source_video=args.source_video,
        segnet_weights=args.upstream_root / "models/segnet.safetensors",
        upstream_root=args.upstream_root,
        output_root=output_root,
        pair_count=args.pair_count,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        seed=args.seed,
        package_versions=package_versions(),
        run_argv=_run_argv(preflight_path),
        camera_hw=CAMERA_HW_PUBLIC,
        seg_hw=SEG_HW_PUBLIC,
        safety_reserve_bytes=safety_reserve,
    )
    atomic_write_json(preflight_path, preflight)
    reopened = load_json_mapping(preflight_path)
    reverify_preflight(reopened)
    if reopened != preflight:
        raise FreshTeacherMaterializationError("stage-00 preflight changed across parse-back")
    return reopened


def _reverify_runtime_packages(preflight: dict[str, Any]) -> None:
    actual = package_versions()
    expected = preflight.get("package_versions")
    if expected != actual:
        raise FreshTeacherMaterializationError(
            f"runtime package versions drifted: expected {expected!r}, got {actual!r}"
        )


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


def _source_batches(preflight: dict[str, Any]):
    import torch

    upstream_root = Path(str(preflight["upstream_closure"]["root"]))
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    from frame_utils import AVVideoDataset

    source = Path(str(preflight["source_video"]["path"]))
    dataset = AVVideoDataset(
        [source.name],
        data_dir=source.parent,
        batch_size=int(preflight["batch_size"]),
        device=torch.device("cpu"),
        num_threads=int(preflight["num_threads"]),
        seed=int(preflight["seed"]),
        prefetch_queue_depth=1,
    )
    dataset.prepare_data()
    remaining = int(preflight["pair_count"])
    for _path, _batch_index, batch in dataset:
        if remaining <= 0:
            break
        array = np.ascontiguousarray(batch.cpu().numpy(), dtype=np.uint8)
        emitted = min(remaining, int(array.shape[0]))
        yield array[:emitted]
        remaining -= emitted
    if remaining != 0:
        raise FreshTeacherMaterializationError(
            f"source AVVideoDataset ended with {remaining} requested pairs still missing"
        )


def _load_frozen_segnet(preflight: dict[str, Any]):
    import torch
    from safetensors.torch import load_file

    upstream_root = Path(str(preflight["upstream_closure"]["root"]))
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    from modules import SegNet

    weights = Path(str(preflight["segnet_weights"]["path"]))
    if sha256_file(weights) != preflight["segnet_weights"]["sha256"]:
        raise FreshTeacherMaterializationError("SegNet weights drifted before model load")
    model = SegNet().eval().to(device=torch.device("cpu"))
    model.load_state_dict(load_file(weights, device="cpu"))
    return model


def _batch_preparer(segnet):
    import torch

    def prepare(source_batch: np.ndarray) -> PreparedTeacherBatchV1:
        # Matches DistortionNet.preprocess_input exactly: B,T,H,W,C ->
        # B,T,C,H,W float32, then SegNet's frame-1 bilinear resize.
        tensor = torch.from_numpy(np.ascontiguousarray(source_batch))
        tensor = tensor.permute(0, 1, 4, 2, 3).float().contiguous()
        with torch.inference_mode():
            scorer_input = segnet.preprocess_input(tensor).contiguous()
        scorer_hashes = tuple(
            sha256_file_like_array(scorer_input[index].cpu().numpy()) for index in range(int(scorer_input.shape[0]))
        )

        def infer_missing(local_indices: tuple[int, ...]):
            if not local_indices:
                return {}
            # SegNet contains batch-sensitive numerical kernels: a subset
            # forward can change rare argmax cells.  Forward the exact original
            # upstream-sized batch whenever any member is missing, then retain
            # only the missing rows.  Fully committed batches still do no
            # forward at all.
            with torch.inference_mode():
                labels = segnet(scorer_input).argmax(dim=1).to(torch.uint8).cpu().numpy()
            return {
                local_index: np.ascontiguousarray(labels[local_index], dtype=np.uint8)
                for local_index in local_indices
            }

        return PreparedTeacherBatchV1(
            scorer_input_sha256=scorer_hashes,
            infer_missing=infer_missing,
        )

    return prepare


def sha256_file_like_array(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def run_materialization(args: argparse.Namespace) -> dict[str, Any]:
    assert_governed_admission("taskspace_fresh_teacher_materializer_v1", on_refuse="raise")
    preflight_path = _preflight_path(args)
    if not preflight_path.is_file():
        raise FreshTeacherMaterializationError(f"run requires an existing stage-00 preflight receipt: {preflight_path}")
    preflight = load_json_mapping(preflight_path)
    reverify_preflight(preflight)
    if Path(str(preflight["output_root"])) / "00_custody_storage_preflight.json" != preflight_path:
        raise FreshTeacherMaterializationError("preflight path is not inside its exact output root")
    _reverify_runtime_packages(preflight)
    _configure_determinism(
        seed=int(preflight["seed"]),
        num_threads=int(preflight["num_threads"]),
    )
    segnet = _load_frozen_segnet(preflight)
    return materialize_fresh_teacher_from_batches(
        preflight=preflight,
        source_batches=_source_batches(preflight),
        prepare_batch=_batch_preparer(segnet),
    )


def run_status(args: argparse.Namespace) -> dict[str, Any]:
    return load_and_reverify_materialization_receipt(_materialization_path(args))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "preflight":
            payload = run_preflight(args)
        elif args.mode == "run":
            payload = run_materialization(args)
        else:
            payload = run_status(args)
    except (FreshTeacherMaterializationError, PermissionError, OSError, RuntimeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
