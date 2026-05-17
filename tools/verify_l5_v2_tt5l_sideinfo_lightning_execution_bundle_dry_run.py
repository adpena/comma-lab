#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify all TT5L side-info Lightning execution-bundle dry-run commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_REPORT_PATH,
    build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification,
    l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_json,
    render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_markdown,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(
            "refusing to write TT5L dry-run verification to tmp: "
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH),
        help="Input TT5L Lightning execution bundle JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_ARTIFACT_PATH
        ),
        help="Output dry-run verification JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_REPORT_PATH
        ),
        help="Output dry-run verification markdown path.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Per-cell dry-run command timeout.",
    )
    parser.add_argument(
        "--current-head-commit",
        default="",
        help="Accepted for operator symmetry; the artifact derives source identity from the bundle SHA.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bundle = _read_json_object(args.bundle_json)
        payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
            bundle=bundle,
            bundle_path=args.bundle_json,
            repo_root=args.repo_root,
            timeout_seconds=args.timeout_seconds,
        )
        payload["current_head_commit"] = args.current_head_commit or _git_head(args.repo_root)
        _write(
            args.output_json,
            l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_json(payload),
        )
        _write(
            args.output_md,
            render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_markdown(
                payload
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-tt5l-lightning-dry-run-verify] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-tt5l-lightning-dry-run-verify] "
        f"passed_cell_count={payload['passed_cell_count']}/"
        f"{payload['cell_count']} "
        f"all_dry_runs_passed={payload['all_dry_runs_passed']} "
        "score_claim=false promotion_eligible=false dispatch_attempted=false"
    )
    return 0 if payload["all_dry_runs_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
