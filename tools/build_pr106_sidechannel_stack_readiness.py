#!/usr/bin/env python3
"""Build a local PR106 sidechannel-stack readiness atom ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.pr106_sidechannel_stack_readiness import (  # noqa: E402
    build_pr106_sidechannel_stack_readiness_from_paths,
)
from tac.repo_io import json_text  # noqa: E402

DEFAULT_BASELINE_JSON = Path(
    "experiments/results/lightning_batch/"
    "exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_PR106_ANCHOR = Path(
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/"
    "archive.zip"
)
DEFAULT_LATENT_METADATA = Path(
    "experiments/results/lane_pr106_latent_sidecar_cpu_smoke_20260505/"
    "build_metadata.json"
)
DEFAULT_YSHIFT_METADATA = Path(
    "experiments/results/lane_pr106_yshift_cpu_smoke_20260505T140325Z/"
    "build_metadata.json"
)
DEFAULT_LRL1_METADATA = Path(
    "experiments/results/lane_pr106_lrl1_cpu_smoke_20260505T140325Z/"
    "build_metadata.json"
)
DEFAULT_THREE_SISTER_STACKED_METADATA = Path(
    "experiments/results/lane_pr106_stacked_3sister_cpu_smoke_20260505T140325Z/"
    "build_metadata.json"
)
DEFAULT_WAVELET_SIDECHANNEL_MANIFEST = Path(
    "experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/"
    "manifest.json"
)
DEFAULT_WAVELET_STACKED_METADATA = Path(
    "experiments/results/pr106_stacked_wavelet_wr01_noop_20260506_codex/"
    "build_metadata.json"
)
DEFAULT_WAVELET_APPLY_GATE = Path(
    "experiments/results/hnerv_wavelet_apply_gate_pr106x_20260506_codex.json"
)
DEFAULT_WR01_PACKET = Path(
    "experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/"
    "wr01_exact_eval_packet.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--pr106-anchor-archive", type=Path, default=DEFAULT_PR106_ANCHOR)
    parser.add_argument("--latent-metadata", type=Path, default=DEFAULT_LATENT_METADATA)
    parser.add_argument("--yshift-metadata", type=Path, default=DEFAULT_YSHIFT_METADATA)
    parser.add_argument("--lrl1-metadata", type=Path, default=DEFAULT_LRL1_METADATA)
    parser.add_argument(
        "--three-sister-stacked-metadata",
        type=Path,
        default=DEFAULT_THREE_SISTER_STACKED_METADATA,
    )
    parser.add_argument(
        "--wavelet-sidechannel-manifest",
        type=Path,
        default=DEFAULT_WAVELET_SIDECHANNEL_MANIFEST,
    )
    parser.add_argument(
        "--wavelet-stacked-metadata",
        type=Path,
        default=DEFAULT_WAVELET_STACKED_METADATA,
    )
    parser.add_argument("--wavelet-apply-gate", type=Path, default=DEFAULT_WAVELET_APPLY_GATE)
    parser.add_argument("--wr01-exact-eval-packet", type=Path, default=DEFAULT_WR01_PACKET)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-if-dispatch-ready",
        action="store_true",
        help="Return 2 if the generated local artifact unexpectedly becomes dispatch-ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_pr106_sidechannel_stack_readiness_from_paths(
        repo_root=REPO_ROOT,
        baseline_json_path=args.baseline_json,
        pr106_anchor_archive=args.pr106_anchor_archive,
        latent_metadata_path=args.latent_metadata,
        yshift_metadata_path=args.yshift_metadata,
        lrl1_metadata_path=args.lrl1_metadata,
        three_sister_stacked_metadata_path=args.three_sister_stacked_metadata,
        wavelet_sidechannel_manifest_path=args.wavelet_sidechannel_manifest,
        wavelet_stacked_metadata_path=args.wavelet_stacked_metadata,
        wavelet_apply_gate_path=args.wavelet_apply_gate,
        wr01_exact_eval_packet_path=args.wr01_exact_eval_packet,
    )
    text = json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_if_dispatch_ready and payload["ready_for_exact_eval_dispatch"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
