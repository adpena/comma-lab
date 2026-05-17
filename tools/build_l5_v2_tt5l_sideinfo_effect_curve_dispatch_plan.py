#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a byte-closed TT5L side-info effect-curve dispatch plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_measurement_schedule import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_REPORT_PATH,
    build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan,
    l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_json,
    render_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_markdown,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(f"refusing to write L5 v2 TT5L dispatch plan to tmp: {text!r}")


def _write(path: Path, text: str) -> None:
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"variant manifest JSON must be an object: {path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant-manifest",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_VARIANT_PACKET_ARTIFACT_PATH),
        help="Input TT5L side-info variant packet manifest JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH),
        help="Output dispatch-plan JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_REPORT_PATH),
        help="Output dispatch-plan markdown path.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--modal-bin", default=".venv/bin/modal")
    parser.add_argument("--gpu", default="T4")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = _read_json(args.variant_manifest)
        plan = build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan(
            manifest=manifest,
            manifest_path=args.variant_manifest,
            repo_root=args.repo_root,
            modal_bin=args.modal_bin,
            gpu=args.gpu,
        )
        _write(args.output_json, l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_json(plan))
        _write(
            args.output_md,
            render_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_markdown(plan),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-tt5l-sideinfo-dispatch] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-tt5l-sideinfo-dispatch] "
        f"plan_id={plan['plan_id']} "
        f"work_unit_count={plan['work_unit_count']} "
        f"ready_work_unit_count={plan['ready_work_unit_count']} "
        "score_claim=false dispatch_attempted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
