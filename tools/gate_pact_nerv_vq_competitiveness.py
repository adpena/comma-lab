#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority competitiveness gate for PACT-NeRV-VQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import write_json  # noqa: E402
from tac.substrates.pact_nerv_vq.competitiveness_gate import (  # noqa: E402
    PACT_VQ_COMPETITIVENESS_GATE_SCHEMA,
    build_pact_vq_competitiveness_gate_from_paths,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-sweep-report", required=True, type=Path)
    parser.add_argument("--source-replay-profile", required=True, type=Path)
    parser.add_argument("--best-codec-replay-profile", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--exact-spend-score-threshold", type=float, default=1.0)
    parser.add_argument("--max-segnet-dist-for-exact-spend", type=float, default=0.02)
    parser.add_argument("--max-posenet-dist-for-exact-spend", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_pact_vq_competitiveness_gate_from_paths(
        codec_sweep_report_path=args.codec_sweep_report,
        source_replay_profile_path=args.source_replay_profile,
        best_codec_replay_profile_path=args.best_codec_replay_profile,
        exact_spend_score_threshold=float(args.exact_spend_score_threshold),
        max_segnet_dist_for_exact_spend=float(args.max_segnet_dist_for_exact_spend),
        max_posenet_dist_for_exact_spend=float(args.max_posenet_dist_for_exact_spend),
    )
    output_json = args.output_json.expanduser().resolve(strict=False)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output_json.as_posix()
    write_json(output_json, report)
    print(
        json.dumps(
            {
                "schema": PACT_VQ_COMPETITIVENESS_GATE_SCHEMA,
                "report_path": output_json.as_posix(),
                "verdict": report["verdict"],
                "best_decoder_codec": report["best_decoder_codec"],
                "exact_spend_candidate": report["exact_spend_candidate"],
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

