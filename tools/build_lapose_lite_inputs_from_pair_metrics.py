#!/usr/bin/env python3
"""Build deterministic LA-POSE-lite inputs from CUDA pair metric metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.lapose_lite_inputs import inputs_from_pair_metric_payload  # noqa: E402
from tac.repo_io import json_text, read_json, sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-metrics-json", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = inputs_from_pair_metric_payload(
        read_json(args.pair_metrics_json),
        source_path=args.pair_metrics_json.as_posix(),
        source_sha256=sha256_file(args.pair_metrics_json),
        max_pairs=args.max_pairs,
    )
    text = json_text(manifest)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
