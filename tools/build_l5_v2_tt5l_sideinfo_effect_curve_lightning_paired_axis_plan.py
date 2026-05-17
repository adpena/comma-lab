#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the TT5L side-info effect-curve Lightning paired-axis plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_measurement_schedule import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (  # noqa: E402
    DEFAULT_LIGHTNING_REMOTE_REPO_DIR,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_ARTIFACT_ROOT,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_REPORT_PATH,
    build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan,
    l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_json,
    render_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_markdown,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(
            "refusing to write L5 v2 TT5L Lightning paired-axis plan to tmp: "
            f"{text!r}"
        )


def _write(path: Path, text: str) -> None:
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"variant manifest JSON must be an object: {path}")
    return payload


def _git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


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
        default=Path(
            L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
        ),
        help="Output Lightning paired-axis plan JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_REPORT_PATH
        ),
        help="Output Lightning paired-axis plan markdown path.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--remote-repo-dir",
        default=DEFAULT_LIGHTNING_REMOTE_REPO_DIR,
        help="Remote pact checkout path used inside Lightning Batch commands.",
    )
    parser.add_argument(
        "--artifact-root",
        default=L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_ARTIFACT_ROOT,
        help="Repo-relative dry-run artifact root for per-cell state/stdout/stderr.",
    )
    parser.add_argument("--machine", default="T4")
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument(
        "--source-commit",
        default="",
        help="Source commit recorded in the operator memo; defaults to git HEAD.",
    )
    parser.add_argument(
        "--no-materialize-dry-runs",
        action="store_true",
        help="Build only the plan object; do not write per-cell dry-run state files.",
    )
    parser.add_argument(
        "--preserve-state",
        action="store_true",
        help="Append to existing per-cell dry-run state instead of resetting it first.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = _read_json(args.variant_manifest)
        plan = build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan(
            manifest=manifest,
            manifest_path=args.variant_manifest,
            repo_root=args.repo_root,
            remote_repo_dir=args.remote_repo_dir,
            artifact_root=args.artifact_root,
            machine=args.machine,
            python_bin=args.python_bin,
            source_commit=args.source_commit or _git_head(args.repo_root),
            materialize_dry_runs=not args.no_materialize_dry_runs,
            reset_state=not args.preserve_state,
        )
        _write(
            args.output_json,
            l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_json(plan),
        )
        _write(
            args.output_md,
            render_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_markdown(
                plan
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-tt5l-lightning-paired-axis] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-tt5l-lightning-paired-axis] "
        f"cell_count={plan['cell_count']} "
        f"all_cells_dry_run_ready={plan['all_cells_dry_run_ready']} "
        "score_claim=false dispatch_attempted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
