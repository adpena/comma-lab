#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the TT5L Lightning non-dry-run fail-closed gate artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_tt5l_lightning_doctor_plan import (  # noqa: E402
    L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_ARTIFACT_PATH,
    L5V2_TT5L_LIGHTNING_REQUIRED_DOCTOR_OUTPUT_PATH,
)
from tac.optimization.l5_v2_tt5l_lightning_non_dry_run_gate import (  # noqa: E402
    L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_ARTIFACT_PATH,
    L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_REPORT_PATH,
    build_l5_v2_tt5l_lightning_non_dry_run_gate,
    l5_v2_tt5l_lightning_non_dry_run_gate_json,
    render_l5_v2_tt5l_lightning_non_dry_run_gate_markdown,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_CLAIMS_PATH,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(
            "refusing to write TT5L non-dry-run gate to tmp: "
            f"{text!r}"
        )


def _write(path: Path, text: str) -> None:
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json_object(path: Path, *, missing_ok: bool = False) -> dict[str, object]:
    if missing_ok and not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_text(path: Path, *, missing_ok: bool = False) -> str:
    if missing_ok and not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


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
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--bundle-json",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--doctor-plan-json",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--doctor-output-json",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_REQUIRED_DOCTOR_OUTPUT_PATH),
    )
    parser.add_argument(
        "--claims-md",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_CLAIMS_PATH),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_REPORT_PATH),
    )
    parser.add_argument(
        "--current-head-commit",
        default="",
        help="Current git commit recorded and enforced against staged manifests.",
    )
    parser.add_argument(
        "--strict-ready",
        action="store_true",
        help="Exit 1 when the gate is not ready instead of only writing artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bundle = _read_json_object(args.bundle_json)
        doctor_plan = _read_json_object(args.doctor_plan_json)
        doctor_output = _read_json_object(args.doctor_output_json, missing_ok=True)
        claims_text = _read_text(args.claims_md, missing_ok=True)
        payload = build_l5_v2_tt5l_lightning_non_dry_run_gate(
            bundle=bundle,
            bundle_path=args.bundle_json,
            doctor_plan=doctor_plan,
            doctor_plan_path=args.doctor_plan_json,
            doctor_output=doctor_output,
            doctor_output_path=args.doctor_output_json,
            claims_text=claims_text,
            claims_path=args.claims_md,
            repo_root=args.repo_root,
            current_head_commit=args.current_head_commit or _git_head(args.repo_root),
        )
        _write(args.output_json, l5_v2_tt5l_lightning_non_dry_run_gate_json(payload))
        _write(args.output_md, render_l5_v2_tt5l_lightning_non_dry_run_gate_markdown(payload))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-tt5l-non-dry-run-gate] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-tt5l-non-dry-run-gate] "
        f"ready_for_non_dry_run_submit={payload['ready_for_non_dry_run_submit']} "
        f"ready_cells={payload['ready_cell_count']}/{payload['cell_count']} "
        f"blocker_count={len(payload['blockers'])} "
        "score_claim=false promotion_eligible=false dispatch_attempted=false"
    )
    if args.strict_ready and payload["ready_for_non_dry_run_submit"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
