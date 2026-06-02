#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local Apple MLX/Metal/Accelerate readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.apple_local_compute_readiness import (  # noqa: E402
    APPLE_LOCAL_COMPUTE_READINESS_SCHEMA,
    build_apple_local_compute_readiness,
    render_apple_local_compute_readiness_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument("--substrate-id", action="append", default=None)
    args = parser.parse_args(argv)

    report = build_apple_local_compute_readiness(
        substrate_ids=args.substrate_id or ("hi_nerv", "snerv"),
    )
    output = args.output_json.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    write_json(output, report)
    if args.output_md is not None:
        md_output = args.output_md.expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
        md_output.write_text(
            render_apple_local_compute_readiness_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": APPLE_LOCAL_COMPUTE_READINESS_SCHEMA,
        "report_path": report.get("report_path"),
        "recommended_dev_velocity_backend": report[
            "recommended_dev_velocity_backend"
        ],
        "mlx_metal_ready": report["mlx_metal_ready"],
        "torch_mps_ready": report["torch_mps_ready"],
        "numpy_accelerate_ready": report["numpy_accelerate_ready"],
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
        "blockers": report["blockers"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
