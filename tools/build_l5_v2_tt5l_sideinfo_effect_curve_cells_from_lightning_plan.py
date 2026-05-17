#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build TT5L side-info effect-curve cells from the Lightning paired-axis plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_harvest import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_REPORT_PATH,
    build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan,
    l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_json,
    render_l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_markdown,
)

DEFAULT_OUTPUT_JSON = Path(L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH)
DEFAULT_OUTPUT_MD = Path(L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_REPORT_PATH)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(
            "refusing to write L5 v2 TT5L side-info harvest cells to tmp: "
            f"{text!r}"
        )


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Lightning paired-axis plan must be a JSON object: {path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lightning-plan-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH),
        help="Input L5 v2 TT5L side-info effect-curve Lightning paired-axis plan JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output builder-ready harvest cell JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help=(
            "Output Markdown status report for the `.omx` control plane. "
            "Defaults to the canonical report path when --output-json is canonical, "
            "otherwise to a sibling .md beside --output-json."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used to resolve plan, artifacts, and manifests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_md = args.output_md
        if output_md is None:
            output_md = (
                DEFAULT_OUTPUT_MD
                if args.output_json == DEFAULT_OUTPUT_JSON
                else args.output_json.with_suffix(".md")
            )
        plan = _read_json_object(args.lightning_plan_json)
        payload = build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
            plan=plan,
            plan_path=args.lightning_plan_json,
            repo_root=args.repo_root,
        )
        _refuse_tmp(args.output_json)
        _refuse_tmp(output_md)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_json(payload),
            encoding="utf-8",
        )
        output_md.write_text(
            render_l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_markdown(payload),
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-tt5l-sideinfo-harvest-cells] FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        "[l5-v2-tt5l-sideinfo-harvest-cells] "
        f"cell_count={payload['cell_count']} "
        f"harvested_exact_eval_artifact_count={payload['harvested_exact_eval_artifact_count']} "
        f"missing_exact_eval_artifact_count={payload['missing_exact_eval_artifact_count']} "
        f"ready_for_effect_curve_build={str(payload['ready_for_effect_curve_build']).lower()} "
        f"blockers={len(payload['blockers'])} "
        f"output={args.output_json} "
        f"report={output_md} "
        "score_claim=false promotion_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
