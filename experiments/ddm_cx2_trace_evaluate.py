#!/usr/bin/env python3
"""Run the immutable upstream evaluator and retain its full-precision locals.

``upstream/evaluate.py`` prints distortions to eight decimals and the final
score to two.  This wrapper executes that exact file through ``runpy`` and uses
a line trace restricted to the evaluator's own ``main`` frame to preserve the
already-computed Python values before formatting.  It does not copy, patch, or
reimplement evaluator logic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import runpy
import sys
from pathlib import Path
from types import FrameType
from typing import Any

EXPECTED_EVALUATE_SHA256 = (
    "7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b"
)
EXPECTED_VIDEO_NAMES_SHA256 = (
    "7ff99d08c8351dd8167ec09213b758da5bbb705dedabe361ba881217374029a8"
)
EXPECTED_ORIGINAL_SHA256 = (
    "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9"
)
EXPECTED_EVALUATOR_DEPENDENCIES = {
    "frame_utils.py": "d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90",
    "modules.py": "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa",
    "models/posenet.safetensors": (
        "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576"
    ),
    "models/segnet.safetensors": (
        "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
    ),
}
EXPECTED_ORIGINAL_BYTES = 37_545_489
EXPECTED_SAMPLE_COUNT = 600
EXPECTED_RAW_BYTES = 1_200 * 874 * 1_164 * 3
AXIS = "[macOS-CPU advisory; upstream AV GT; immutable evaluate.py; n600]"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-py", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = list(args.evaluator_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    return args, forwarded


def unique_option(argv: list[str], name: str) -> str:
    values: list[str] = []
    for index, value in enumerate(argv):
        if value == name:
            if index + 1 >= len(argv):
                raise RuntimeError(f"forwarded evaluator argv truncates {name}")
            values.append(argv[index + 1])
        elif value.startswith(f"{name}="):
            values.append(value.split("=", 1)[1])
    if len(values) != 1:
        raise RuntimeError(
            f"forwarded evaluator argv must contain exactly one {name}; "
            f"observed {len(values)}"
        )
    return values[0]


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    args, forwarded = parse_args()
    evaluate_py = args.evaluate_py.resolve()
    evaluate_filename = str(evaluate_py)
    evaluate_pre_sha256 = sha256_file(evaluate_py)
    if evaluate_pre_sha256 != EXPECTED_EVALUATE_SHA256:
        raise RuntimeError("evaluate.py SHA-256 differs from the immutable CX2 pin")

    submission_dir = Path(unique_option(forwarded, "--submission-dir")).resolve()
    uncompressed_dir = Path(unique_option(forwarded, "--uncompressed-dir")).resolve()
    video_names = Path(unique_option(forwarded, "--video-names-file")).resolve()
    device_arg = unique_option(forwarded, "--device")
    batch_size = int(unique_option(forwarded, "--batch-size"))
    seed = int(unique_option(forwarded, "--seed"))
    num_threads = int(unique_option(forwarded, "--num-threads"))
    prefetch_depth = int(unique_option(forwarded, "--prefetch-queue-depth"))
    unique_option(forwarded, "--report")
    if device_arg != "cpu":
        raise RuntimeError("CX2 local evaluator receipt requires explicit --device cpu")
    if batch_size != 16:
        raise RuntimeError("CX2 evaluator batch size must be exactly 16")
    if seed != 1234 or num_threads != 2 or prefetch_depth != 4:
        raise RuntimeError("CX2 evaluator seed/worker/prefetch policy differs")
    if sha256_file(video_names) != EXPECTED_VIDEO_NAMES_SHA256:
        raise RuntimeError("video-names file differs from the pinned public n600 list")
    if directory_bytes(uncompressed_dir) != EXPECTED_ORIGINAL_BYTES:
        raise RuntimeError("uncompressed corpus byte count differs from the CX2 pin")

    archive = submission_dir / "archive.zip"
    raw = submission_dir / "inflated" / "0.raw"
    raw_files = sorted((submission_dir / "inflated").glob("*.raw"))
    if raw_files != [raw]:
        raise RuntimeError("submission must contain exactly evaluator-facing inflated/0.raw")
    if raw.stat().st_size != EXPECTED_RAW_BYTES:
        raise RuntimeError("inflated/0.raw byte count differs from the public tensor shape")
    archive_pre_sha256 = sha256_file(archive)
    raw_pre_sha256 = sha256_file(raw)
    original = uncompressed_dir / "0.mkv"
    original_pre_sha256 = sha256_file(original)
    if original_pre_sha256 != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError("uncompressed 0.mkv SHA-256 differs from the CX2 pin")
    video_names_pre_sha256 = sha256_file(video_names)
    evaluator_dependencies_pre = {
        relative: sha256_file(evaluate_py.parent / relative)
        for relative in EXPECTED_EVALUATOR_DEPENDENCIES
    }
    if evaluator_dependencies_pre != EXPECTED_EVALUATOR_DEPENDENCIES:
        raise RuntimeError("evaluator dependency closure differs from the CX2 pins")
    captured: dict[str, Any] = {}

    def trace(frame: FrameType, event: str, arg: Any):
        if frame.f_code.co_filename != evaluate_filename:
            return None
        if frame.f_code.co_name != "main":
            return trace
        if event == "line" and {
            "posenet_dist",
            "segnet_dist",
            "score",
            "compressed_size",
            "uncompressed_size",
            "rate",
            "batch_sizes",
        }.issubset(frame.f_locals):
            captured.update(
                {
                    "average_posenet_distortion": float(
                        frame.f_locals["posenet_dist"]
                    ),
                    "average_segnet_distortion": float(
                        frame.f_locals["segnet_dist"]
                    ),
                    "archive_bytes": int(frame.f_locals["compressed_size"]),
                    "uncompressed_bytes": int(frame.f_locals["uncompressed_size"]),
                    "compression_rate": float(frame.f_locals["rate"]),
                    "score": float(frame.f_locals["score"]),
                    "sample_count": int(frame.f_locals["batch_sizes"].item()),
                    "resolved_device": frame.f_locals["device"].type,
                    "ground_truth_dataset": frame.f_locals[
                        "DefaultDatasetClass"
                    ].__name__,
                    "capture_line": frame.f_lineno,
                }
            )
        return trace

    prior_argv = sys.argv
    prior_sys_path = list(sys.path)
    sys.argv = [str(evaluate_py), *forwarded]
    sys.path.insert(0, str(evaluate_py.parent))
    sys.settrace(trace)
    try:
        runpy.run_path(str(evaluate_py), run_name="__main__")
    finally:
        sys.settrace(None)
        sys.argv = prior_argv
        sys.path[:] = prior_sys_path
    if not captured:
        raise RuntimeError("upstream evaluator finished without a precision capture")

    evaluate_post_sha256 = sha256_file(evaluate_py)
    archive_post_sha256 = sha256_file(archive)
    raw_post_sha256 = sha256_file(raw)
    original_post_sha256 = sha256_file(original)
    video_names_post_sha256 = sha256_file(video_names)
    evaluator_dependencies_post = {
        relative: sha256_file(evaluate_py.parent / relative)
        for relative in EXPECTED_EVALUATOR_DEPENDENCIES
    }
    if evaluate_post_sha256 != evaluate_pre_sha256:
        raise RuntimeError("evaluate.py changed during the evaluator run")
    if archive_post_sha256 != archive_pre_sha256:
        raise RuntimeError("archive.zip changed during the evaluator run")
    if raw_post_sha256 != raw_pre_sha256:
        raise RuntimeError("inflated/0.raw changed during the evaluator run")
    if original_post_sha256 != original_pre_sha256:
        raise RuntimeError("uncompressed 0.mkv changed during the evaluator run")
    if video_names_post_sha256 != video_names_pre_sha256:
        raise RuntimeError("video-names file changed during the evaluator run")
    if evaluator_dependencies_post != evaluator_dependencies_pre:
        raise RuntimeError("evaluator dependency closure changed during the run")
    if captured["resolved_device"] != "cpu":
        raise RuntimeError("upstream evaluator did not resolve the requested CPU rail")
    if captured["ground_truth_dataset"] != "AVVideoDataset":
        raise RuntimeError("CPU evaluator did not select the AV ground-truth backend")
    if captured["sample_count"] != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("upstream evaluator sample denominator is not n600")
    if captured["uncompressed_bytes"] != EXPECTED_ORIGINAL_BYTES:
        raise RuntimeError("evaluator uncompressed denominator differs from the pin")
    if captured["archive_bytes"] != archive.stat().st_size:
        raise RuntimeError("evaluator archive byte count differs from the scored file")
    recomputed_score = (
        100 * captured["average_segnet_distortion"]
        + math.sqrt(captured["average_posenet_distortion"] * 10)
        + 25 * captured["archive_bytes"] / captured["uncompressed_bytes"]
    )
    if recomputed_score != captured["score"]:
        raise RuntimeError("captured score differs from the immutable equation")
    captured.update(
        {
            "schema": "ddm_cx2_exact_evaluator_trace.v1",
            "written_at_utc": dt.datetime.now(dt.UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
            "evaluate_py": str(evaluate_py),
            "evaluate_py_sha256": evaluate_post_sha256,
            "evaluator_argv": [str(evaluate_py), *forwarded],
            "archive_path": str(archive),
            "archive_sha256": archive_post_sha256,
            "raw_path": str(raw),
            "raw_bytes": raw.stat().st_size,
            "raw_sha256": raw_post_sha256,
            "original_path": str(original),
            "original_sha256": original_post_sha256,
            "video_names_path": str(video_names),
            "video_names_sha256": video_names_post_sha256,
            "evaluator_dependency_sha256": evaluator_dependencies_post,
            "recomputed_score": recomputed_score,
            "pre_post_identity": {
                "evaluate_py": True,
                "archive": True,
                "raw": True,
                "original": True,
                "video_names": True,
                "evaluator_dependencies": True,
            },
            "capture_mechanism": (
                "line trace of immutable evaluate.py main locals after exact score "
                "computation and before display formatting"
            ),
        }
    )
    atomic_json(args.receipt, captured)
    print(json.dumps(captured, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
