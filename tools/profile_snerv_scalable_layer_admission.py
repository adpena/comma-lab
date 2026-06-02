#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile SNeRV base/enhancement layer admission from real SNAR bytes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_scalable_layer_admission import (  # noqa: E402
    render_snerv_scalable_layer_admission_markdown,
    write_snerv_scalable_layer_admission_report,
)
from tac.repo_io import write_text_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--layer-nonrate-deltas-json", type=Path)
    parser.add_argument("--full-video-coverage", action="store_true")
    parser.add_argument("--frontier-bytes", type=int, default=178_493)
    args = parser.parse_args(argv)

    report = write_snerv_scalable_layer_admission_report(
        input_path=args.input,
        output_path=args.output_json,
        layer_nonrate_deltas=_load_deltas(args.layer_nonrate_deltas_json),
        full_video_coverage=bool(args.full_video_coverage),
        frontier_bytes=int(args.frontier_bytes),
    )
    if args.output_md is not None:
        write_text_artifact(
            args.output_md,
            render_snerv_scalable_layer_admission_markdown(report),
        )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "verdict": report["verdict"],
                "deserves_separate_scalable_layer_lane": report[
                    "deserves_separate_scalable_layer_lane"
                ],
                "output_json": args.output_json.as_posix(),
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def _load_deltas(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object")
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
