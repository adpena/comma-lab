#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the TT5L side-info Lightning execution preflight artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_preflight import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_REPORT_PATH,
    build_l5_v2_tt5l_sideinfo_lightning_execution_preflight,
    l5_v2_tt5l_sideinfo_lightning_execution_preflight_json,
    render_l5_v2_tt5l_sideinfo_lightning_execution_preflight_markdown,
)

DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(
            "refusing to write L5 v2 TT5L execution preflight to tmp: "
            f"{text!r}"
        )


def _write(path: Path, text: str) -> None:
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Lightning paired-axis plan must be a JSON object: {path}")
    return payload


def _git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lightning-plan-json",
        type=Path,
        default=Path(
            L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
        ),
        help="Input L5 v2 TT5L Lightning paired-axis dry-run plan JSON.",
    )
    parser.add_argument(
        "--claims-path",
        type=Path,
        default=DEFAULT_CLAIMS_PATH,
        help="Active lane claims ledger used for conflict checks.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH),
        help="Output execution-preflight JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_REPORT_PATH),
        help="Output execution-preflight markdown path.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--current-head-commit",
        default="",
        help="Current git commit recorded in the memo; defaults to git HEAD.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = _read_json_object(args.lightning_plan_json)
        claims_text = (
            args.claims_path.read_text(encoding="utf-8")
            if args.claims_path.is_file()
            else ""
        )
        payload = build_l5_v2_tt5l_sideinfo_lightning_execution_preflight(
            plan=plan,
            plan_path=args.lightning_plan_json,
            repo_root=args.repo_root,
            claims_text=claims_text,
            current_head_commit=args.current_head_commit or _git_head(args.repo_root),
        )
        _write(
            args.output_json,
            l5_v2_tt5l_sideinfo_lightning_execution_preflight_json(payload),
        )
        _write(
            args.output_md,
            render_l5_v2_tt5l_sideinfo_lightning_execution_preflight_markdown(
                payload
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-tt5l-lightning-exec-preflight] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-tt5l-lightning-exec-preflight] "
        f"cell_count={payload['cell_count']} "
        f"ready_cell_count={payload['ready_cell_count']} "
        f"ready_for_operator_claiming={payload['ready_for_operator_claiming']} "
        "score_claim=false promotion_eligible=false dispatch_attempted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
