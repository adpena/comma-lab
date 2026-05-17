#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the TT5L side-info Lightning execution bundle artifact."""

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
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_REPORT_PATH,
    build_l5_v2_tt5l_sideinfo_lightning_execution_bundle,
    l5_v2_tt5l_sideinfo_lightning_execution_bundle_json,
    render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_markdown,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_preflight import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(
            "refusing to write L5 v2 TT5L execution bundle to tmp: "
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


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH),
        help="Input TT5L Lightning execution preflight JSON.",
    )
    parser.add_argument(
        "--lightning-plan-json",
        type=Path,
        default=Path(
            L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
        ),
        help="Input L5 v2 TT5L Lightning paired-axis dry-run plan JSON.",
    )
    parser.add_argument(
        "--variant-manifest-json",
        type=Path,
        default=None,
        help=(
            "Input TT5L side-info variant manifest. Defaults to the "
            "source_variant_manifest field in --lightning-plan-json."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH),
        help="Output execution-bundle JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_REPORT_PATH),
        help="Output execution-bundle markdown path.",
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
        preflight = _read_json_object(args.preflight_json)
        lightning_plan = _read_json_object(args.lightning_plan_json)
        variant_manifest_path = args.variant_manifest_json
        if variant_manifest_path is None:
            raw = str(lightning_plan.get("source_variant_manifest") or "").strip()
            if not raw:
                raise ValueError(
                    "missing --variant-manifest-json and lightning plan "
                    "source_variant_manifest"
                )
            variant_manifest_path = _resolve_repo_path(raw, args.repo_root)
        variant_manifest = _read_json_object(variant_manifest_path)
        payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle(
            preflight=preflight,
            preflight_path=args.preflight_json,
            lightning_plan=lightning_plan,
            lightning_plan_path=args.lightning_plan_json,
            variant_manifest=variant_manifest,
            variant_manifest_path=variant_manifest_path,
            repo_root=args.repo_root,
            current_head_commit=args.current_head_commit or _git_head(args.repo_root),
        )
        _write(
            args.output_json,
            l5_v2_tt5l_sideinfo_lightning_execution_bundle_json(payload),
        )
        _write(
            args.output_md,
            render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_markdown(payload),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-tt5l-lightning-exec-bundle] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-tt5l-lightning-exec-bundle] "
        f"cell_count={payload['cell_count']} "
        f"ready_dry_run_cell_count={payload['ready_dry_run_cell_count']} "
        f"ready_for_dry_run_submit={payload['ready_for_dry_run_submit']} "
        "score_claim=false promotion_eligible=false dispatch_attempted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
