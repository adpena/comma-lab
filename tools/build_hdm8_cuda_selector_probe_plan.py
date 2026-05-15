#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build sparse exact-CUDA probe configs for HDM8 selector calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.hdm8_cuda_selector_probe_plan import (  # noqa: E402
    DEFAULT_PREFIX_SIZES,
    build_hdm8_cuda_selector_probe_plan,
)
from tac.repo_io import read_json, repo_relative, write_json  # noqa: E402

DEFAULT_ARCHIVE = (
    "experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/"
    "exact_eval_static_release_surface/archive.zip"
)
DEFAULT_RUNTIME_TEMPLATE = "submissions/hdm8_film_grain_sidecar"


def _parse_prefix_sizes(value: str) -> list[int]:
    sizes = [int(item) for item in value.split(",") if item.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one prefix size is required")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("prefix sizes must be positive")
    return sizes


def _with_commands(
    plan: dict[str, Any],
    *,
    archive: str,
    runtime_template: str,
    output_dir: Path,
    selector_codec: str,
) -> dict[str, Any]:
    for row in plan["probe_configs"]:
        config_path = output_dir / f"{row['name']}.selector_config.json"
        packet_dir = output_dir / row["name"]
        row["selector_config_path"] = repo_relative(config_path, REPO_ROOT)
        row["packet_output_dir"] = repo_relative(packet_dir, REPO_ROOT)
        row["packet_build_command"] = [
            ".venv/bin/python",
            "tools/build_hdm8_film_grain_sidecar_packet.py",
            "--archive",
            archive,
            "--runtime-template",
            runtime_template,
            "--output-dir",
            repo_relative(packet_dir, REPO_ROOT),
            "--selector-config-json",
            repo_relative(config_path, REPO_ROOT),
            "--pack-selector-into-archive",
            "--selector-codec",
            selector_codec,
            "--require-positive-proxy",
        ]
    return plan


def _render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# HDM8 CUDA Selector Probe Plan",
        "",
        f"- score_claim: `{str(plan['score_claim']).lower()}`",
        f"- axis: `{plan['axis']}`",
        f"- candidate_atom_count: `{plan['candidate_atom_count']}`",
        f"- probe_config_count: `{len(plan['probe_configs'])}`",
        "",
        "## Probe Configs",
        "",
        "| name | selected pairs | proxy delta | config bytes | command |",
        "|---|---:|---:|---:|---|",
    ]
    for row in plan["probe_configs"]:
        lines.append(
            f"| `{row['name']}` | {row['selected_pair_count']} | "
            f"{row['proxy_delta_vs_none']} | {row['selector_config_json_bytes']} | "
            f"`{' '.join(row['packet_build_command'])}` |"
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in plan["dispatch_blockers"]:
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--runtime-template", default=DEFAULT_RUNTIME_TEMPLATE)
    parser.add_argument("--selector-codec", choices=["brotli", "json"], default="brotli")
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument(
        "--prefix-sizes",
        type=_parse_prefix_sizes,
        default=list(DEFAULT_PREFIX_SIZES),
        help="Comma-separated sparse prefix sizes, for example 1,2,4,8,16.",
    )
    parser.add_argument("--min-pair-gain", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep = read_json(args.sweep_json)
    plan = build_hdm8_cuda_selector_probe_plan(
        sweep,
        evidence_source_path=repo_relative(args.sweep_json, REPO_ROOT),
        max_atoms=args.max_atoms,
        prefix_sizes=args.prefix_sizes,
        min_pair_gain=args.min_pair_gain,
    )
    plan = _with_commands(
        plan,
        archive=args.archive,
        runtime_template=args.runtime_template,
        output_dir=args.output_dir,
        selector_codec=args.selector_codec,
    )
    for row in plan["probe_configs"]:
        config_path = REPO_ROOT / row["selector_config_path"]
        write_json(config_path, row["config"])
    output_json = args.output_json or (args.output_dir / "probe_plan.json")
    output_md = args.output_md or (args.output_dir / "probe_plan.md")
    write_json(output_json, plan)
    output_md.write_text(_render_markdown(plan), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
