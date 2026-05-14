#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build per-pair SegNet boundary and margin marginals for A5 allocation."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.segnet_boundary_marginals import (  # noqa: E402
    BoundaryFeatureSummary,
    merge_feature_summaries,
    summarize_boundary_features,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

SCHEMA = "pr101_segnet_boundary_marginals.v1"
DEFAULT_UPSTREAM_DIR = Path("upstream")
DEFAULT_VIDEO_NAMES_FILE = Path("upstream/public_test_video_names.txt")
DEFAULT_UNCOMPRESSED_DIR = Path("upstream/videos")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--video-names-file", type=Path, default=DEFAULT_VIDEO_NAMES_FILE)
    parser.add_argument("--uncompressed-dir", type=Path, default=DEFAULT_UNCOMPRESSED_DIR)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--boundary-dilation", type=int, default=5)
    parser.add_argument("--low-margin-threshold", type=float, default=1.0)
    parser.add_argument("--candidate-id", default="pr101_segnet_boundary_marginals")
    return parser.parse_args(argv)


def build_manifest(
    *,
    upstream_dir: Path,
    video_names_file: Path,
    uncompressed_dir: Path,
    json_out: Path,
    device_name: str = "cpu",
    batch_size: int = 16,
    max_pairs: int | None = None,
    boundary_dilation: int = 5,
    low_margin_threshold: float = 1.0,
    candidate_id: str = "pr101_segnet_boundary_marginals",
) -> dict[str, Any]:
    upstream_dir = _resolve(upstream_dir)
    video_names_file = _resolve(video_names_file)
    uncompressed_dir = _resolve(uncompressed_dir)
    json_out = _resolve(json_out)
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested --device mps, but torch.backends.mps is unavailable")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive when provided")

    prepend_paths(upstream_dir)
    from frame_utils import AVVideoDataset  # type: ignore  # noqa: PLC0415
    from modules import SegNet, segnet_sd_path  # type: ignore  # noqa: PLC0415
    from safetensors.torch import load_file  # type: ignore  # noqa: PLC0415

    device = torch.device(device_name)
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))

    video_names = [
        line.strip()
        for line in video_names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = AVVideoDataset(
        video_names,
        data_dir=uncompressed_dir,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    dataset.prepare_data()

    summaries: list[BoundaryFeatureSummary] = []
    pair_count = 0
    with torch.inference_mode():
        for _path, _idx, batch_hwc in dataset:
            if max_pairs is not None:
                remaining = max_pairs - pair_count
                if remaining <= 0:
                    break
                batch_hwc = batch_hwc[:remaining]
            batch_btchw = (
                batch_hwc.to(device=device)
                .float()
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
            seg_in = segnet.preprocess_input(batch_btchw)
            logits_t = segnet(seg_in)
            labels = logits_t.argmax(dim=1).detach().cpu().numpy()
            logits = logits_t.detach().cpu().numpy()
            summaries.append(
                summarize_boundary_features(
                    labels=labels,
                    logits=logits,
                    dilation=boundary_dilation,
                    low_margin_threshold=low_margin_threshold,
                )
            )
            pair_count += int(labels.shape[0])
            if max_pairs is not None and pair_count >= max_pairs:
                break

    summary = merge_feature_summaries(summaries)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_id": candidate_id,
        "created_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "score_claim": False,
        "dispatch_attempted": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[analysis; SegNet GT boundary marginals; no eval]",
        "evidence_semantics": (
            "frozen SegNet boundary/margin features for allocator design only; "
            "build a packet and run exact eval before any score claim"
        ),
        "device": device_name,
        "loader": "AVVideoDataset",
        "batch_size": batch_size,
        "max_pairs": max_pairs,
        "n_pairs": int(summary.boundary_mass.size),
        "boundary_dilation": boundary_dilation,
        "low_margin_threshold": float(low_margin_threshold),
        "source_artifacts": {
            "upstream_dir": repo_relative(upstream_dir, REPO_ROOT),
            "video_names_file": {
                "path": repo_relative(video_names_file, REPO_ROOT),
                "sha256": sha256_file(video_names_file),
            },
            "segnet_safetensors": {
                "path": repo_relative(Path(segnet_sd_path), REPO_ROOT),
                "sha256": sha256_file(Path(segnet_sd_path)),
            },
        },
        "feature_summary": _feature_summary(summary),
        **summary.as_jsonable(),
    }
    payload["manifest_sha256_excluding_self"] = sha256_bytes(
        json_text(payload).encode("utf-8")
    )
    payload = attach_tool_run_manifest(
        payload,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=sys.argv[1:],
        input_paths=[
            video_names_file,
            Path(segnet_sd_path),
        ],
        repo_root=REPO_ROOT,
        output_path=json_out,
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json_text(payload), encoding="utf-8")
    return payload


def _feature_summary(summary: BoundaryFeatureSummary) -> dict[str, Any]:
    return {
        "boundary_mass": _stats(summary.boundary_mass),
        "low_margin_mass": _stats(summary.low_margin_mass),
        "mean_logit_margin": _stats(summary.mean_logit_margin),
        "p10_logit_margin": _stats(summary.p10_logit_margin),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_manifest(
        upstream_dir=args.upstream_dir,
        video_names_file=args.video_names_file,
        uncompressed_dir=args.uncompressed_dir,
        json_out=args.json_out,
        device_name=args.device,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        boundary_dilation=args.boundary_dilation,
        low_margin_threshold=args.low_margin_threshold,
        candidate_id=args.candidate_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
