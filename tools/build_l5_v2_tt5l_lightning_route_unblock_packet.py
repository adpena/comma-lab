#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the generated TT5L Lightning route-unblock packet."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_measurement_schedule import (  # noqa: E402
    L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_lightning_route_unblock_packet import (  # noqa: E402
    L5V2_PROVIDER_READINESS_REFRESH_ARTIFACT_PATH,
    L5V2_TT5L_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH,
    L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_ARTIFACT_PATH,
    L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_REPORT_PATH,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH,
    build_l5_v2_tt5l_lightning_route_unblock_packet,
    l5_v2_tt5l_lightning_route_unblock_packet_json,
    render_l5_v2_tt5l_lightning_route_unblock_packet_markdown,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_ARTIFACT_PATH,
)

_TT5L_LIGHTNING_PAIRED_AXIS_STATIC_SOURCE_PATHS = (
    "tools/build_l5_v2_architecture_lock_packet.py",
    "tools/build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py",
    "tools/build_l5_v2_tt5l_sideinfo_lightning_execution_preflight.py",
    "tools/build_tt5l_sideinfo_variant_packets.py",
    "src/tac/exact_eval_custody.py",
    "src/tac/optimization/l5_staircase_v2.py",
    "src/tac/optimization/l5_v2_measurement_schedule.py",
    "src/tac/optimization/l5_v2_sideinfo_effect_curve.py",
    "src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py",
    "src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_preflight.py",
    "src/tac/optimization/tt5l_sideinfo_variant_packets.py",
    "src/tac/deploy/lightning/batch_jobs.py",
    "scripts/launch_lightning_batch_job.py",
    "scripts/adjudicate_contest_auth_eval.py",
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(
            "refusing to write TT5L route-unblock packet to tmp: "
            f"{text!r}"
        )


def _write(path: Path, text: str) -> None:
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
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


def _source_relevant_diff_paths(
    *,
    repo_root: Path,
    source_commit: str,
    current_head_commit: str,
    extra_source_paths: tuple[str, ...] = (),
) -> list[str]:
    source_paths = {
        path
        for path in (
            *_TT5L_LIGHTNING_PAIRED_AXIS_STATIC_SOURCE_PATHS,
            *extra_source_paths,
        )
        if path
    }
    raw_paths: list[str] = []
    if source_commit and current_head_commit and source_commit != current_head_commit:
        proc = subprocess.run(
            ["git", "diff", "--name-only", source_commit, current_head_commit],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return ["<git-diff-failed>"]
        raw_paths.extend(line.strip() for line in proc.stdout.splitlines())

    for command in (
        ["git", "diff", "--name-only"],
        ["git", "diff", "--name-only", "--cached"],
    ):
        proc = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return ["<git-diff-failed>"]
        raw_paths.extend(line.strip() for line in proc.stdout.splitlines())

    out: list[str] = []
    for path in raw_paths:
        if path and path in source_paths and path not in out:
            out.append(path)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--provider-readiness-json",
        type=Path,
        default=Path(L5V2_PROVIDER_READINESS_REFRESH_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--execution-preflight-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--execution-bundle-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--dry-run-verification-json",
        type=Path,
        default=Path(
            L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_ARTIFACT_PATH
        ),
    )
    parser.add_argument(
        "--paired-axis-plan-json",
        type=Path,
        default=Path(
            L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
        ),
    )
    parser.add_argument(
        "--harvest-cells-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--sideinfo-effect-curve-json",
        type=Path,
        default=Path(L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--legacy-alt-provider-plan-json",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_REPORT_PATH),
    )
    parser.add_argument(
        "--current-head-commit",
        default="",
        help="Current git commit recorded in the packet; defaults to git HEAD.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        current_head = args.current_head_commit or _git_head(args.repo_root)
        paired_axis_plan = _read_json_object(args.paired_axis_plan_json)
        source_commit = str(paired_axis_plan.get("source_commit") or "")
        source_diff_paths = _source_relevant_diff_paths(
            repo_root=args.repo_root,
            source_commit=source_commit,
            current_head_commit=current_head,
            extra_source_paths=(
                str(paired_axis_plan.get("source_variant_manifest") or ""),
                str(paired_axis_plan.get("source_dispatch_plan") or ""),
            ),
        )
        payload = build_l5_v2_tt5l_lightning_route_unblock_packet(
            repo_root=args.repo_root,
            current_head_commit=current_head,
            source_relevant_diff_paths=source_diff_paths,
            provider_readiness_path=str(args.provider_readiness_json),
            execution_preflight_path=str(args.execution_preflight_json),
            execution_bundle_path=str(args.execution_bundle_json),
            dry_run_verification_path=str(args.dry_run_verification_json),
            paired_axis_plan_path=str(args.paired_axis_plan_json),
            harvest_cells_path=str(args.harvest_cells_json),
            sideinfo_effect_curve_path=str(args.sideinfo_effect_curve_json),
            legacy_alt_provider_plan_path=str(args.legacy_alt_provider_plan_json),
        )
        _write(args.output_json, l5_v2_tt5l_lightning_route_unblock_packet_json(payload))
        _write(
            args.output_md,
            render_l5_v2_tt5l_lightning_route_unblock_packet_markdown(payload),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-tt5l-route-unblock] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-tt5l-route-unblock] "
        f"artifact_blocker_count={len(payload['blockers'])} "
        f"remaining_route_blocker_count={len(payload['remaining_blockers'])} "
        "score_claim=false promotion_eligible=false provider_spend_attempted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
