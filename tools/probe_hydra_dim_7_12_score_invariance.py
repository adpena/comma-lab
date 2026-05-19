#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compatibility wrapper for the corrected F1-as-A2 RGB invariance probe.

The original routed filename said "hydra dim 7-12 score invariance".  That
direct dim-channel framing is stale: dims 7-12 are internal PoseNet outputs,
not archive-visible bytes.  This wrapper now emits
``f1_framing_version=corrected_A2`` and probes the physical RGB-output channel.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.contest_exploits.f1_as_a2_rgb_invariance import (  # noqa: E402
    build_f1_as_a2_report_from_outputs,
    build_f1_as_a2_rgb_invariance_probe,
    write_json_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "upstream/videos/0.mkv")
    parser.add_argument("--pair-count", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pixel-stride", type=int, default=128)
    parser.add_argument("--amplitude", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pose-0-5-rmse-threshold", type=float, default=1e-4)
    parser.add_argument("--seg-delta-threshold", type=float, default=0.0)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic scorer outputs for fast CLI/tests instead of loading scorers.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--register-probe-outcome",
        action="store_true",
        help="Append the summarized result to .omx/state/probe_outcomes.jsonl.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    report = (
        _synthetic_report(args, generated_at)
        if args.synthetic
        else build_f1_as_a2_rgb_invariance_probe(
            upstream_dir=args.upstream_dir,
            video_path=args.video,
            pair_count=args.pair_count,
            start_frame=args.start_frame,
            device=args.device,
            pixel_stride=args.pixel_stride,
            amplitude=args.amplitude,
            seed=args.seed,
            pose_0_5_rmse_threshold=args.pose_0_5_rmse_threshold,
            seg_delta_threshold=args.seg_delta_threshold,
            generated_at_utc=generated_at,
        )
    )
    report["compatibility_wrapper"] = "probe_hydra_dim_7_12_score_invariance.py"
    _attach_cli_metadata(report, argv)
    if args.output_json:
        report["evidence_path"] = str(write_json_report(report, args.output_json))
    if args.register_probe_outcome:
        _register_probe_outcome(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _synthetic_report(args: argparse.Namespace, generated_at: str) -> dict:
    import numpy as np

    baseline_pose = np.zeros((args.pair_count, 12), dtype=np.float64)
    perturbed_pose = baseline_pose.copy()
    perturbed_pose[:, 6:12] = 0.25
    baseline_seg = np.zeros((args.pair_count, 4, 4), dtype=np.int64)
    perturbed_seg = baseline_seg.copy()
    perturbation = {
        "pixel_stride": args.pixel_stride,
        "amplitude": args.amplitude,
        "seed": args.seed,
        "attempted_payload_bits_total": max(args.pair_count, 1) * 12,
        "changed_rgb_values_total": max(args.pair_count, 1) * 12,
        "physical_rgb_changed": True,
    }
    return build_f1_as_a2_report_from_outputs(
        baseline_pose=baseline_pose,
        perturbed_pose=perturbed_pose,
        baseline_seg_argmax=baseline_seg,
        perturbed_seg_argmax=perturbed_seg,
        perturbation=perturbation,
        pose_0_5_rmse_threshold=args.pose_0_5_rmse_threshold,
        seg_delta_threshold=args.seg_delta_threshold,
        generated_at_utc=generated_at,
        evidence_label="synthetic_scorer_outputs",
        evidence_sha256=None,
        device=args.device,
    )


def _register_probe_outcome(report: dict) -> None:
    from tac.probe_outcomes_ledger import register_probe_outcome

    common = {
        "evidence_path": report.get("evidence_path"),
        "agent": "codex",
        "subagent_id": "codex_session_019de465_f1_phase1",
        "session_id": "019de465",
    }
    direct_kwargs = dict(report["direct_hydra_dim_probe_outcome_kwargs"])
    direct_kwargs.update(common)
    direct_kwargs["probe_id"] = f"f1_direct_hydra_dim_channel_blocked_{_stamp()}"
    direct_kwargs["notes"] = (
        "Compatibility wrapper registered the direct Hydra dim 7:12 channel as "
        "blocking DEFER: it is internal scorer-output invariance only, not an "
        "archive-visible byte channel."
    )
    register_probe_outcome(**direct_kwargs)

    corrected_kwargs = dict(report["probe_outcome_kwargs"])
    corrected_kwargs.update(common)
    corrected_kwargs["probe_id"] = f"f1_corrected_a2_rgb_invariance_{_stamp()}"
    corrected_kwargs["notes"] = (
        "Compatibility wrapper for corrected F1-as-A2 physical RGB perturbation "
        "local invariance probe. Changed RGB values are not recovered payload bits."
    )
    register_probe_outcome(**corrected_kwargs)


def _attach_cli_metadata(report: dict, argv: Sequence[str] | None) -> None:
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    command = [sys.executable, str(Path(__file__).resolve()), *effective_argv]
    report["probe_command"] = shlex.join(command)
    report["probe_argv"] = effective_argv


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
