#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the TT5L Lightning required-doctor plan artifact."""

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
    L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_REPORT_PATH,
    build_l5_v2_tt5l_lightning_doctor_plan,
    l5_v2_tt5l_lightning_doctor_plan_json,
    render_l5_v2_tt5l_lightning_doctor_plan_markdown,
)
from tac.optimization.l5_v2_tt5l_lightning_route_unblock_packet import (  # noqa: E402
    L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_ARTIFACT_PATH,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(f"refusing to write TT5L doctor plan to tmp: {text!r}")


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
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--route-packet-json",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_ARTIFACT_PATH),
        help="Input TT5L Lightning route-unblock packet JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_REPORT_PATH),
    )
    parser.add_argument(
        "--current-head-commit",
        default="",
        help="Current git commit recorded in the artifact; defaults to git HEAD.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        route_packet = _read_json_object(args.route_packet_json)
        payload = build_l5_v2_tt5l_lightning_doctor_plan(
            route_packet=route_packet,
            route_packet_path=args.route_packet_json,
            repo_root=args.repo_root,
            current_head_commit=args.current_head_commit or _git_head(args.repo_root),
        )
        _write(args.output_json, l5_v2_tt5l_lightning_doctor_plan_json(payload))
        _write(args.output_md, render_l5_v2_tt5l_lightning_doctor_plan_markdown(payload))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-tt5l-lightning-doctor-plan] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-tt5l-lightning-doctor-plan] "
        f"ready_for_operator_doctor={payload['ready_for_operator_doctor']} "
        f"blocker_count={len(payload['blockers'])} "
        "score_claim=false promotion_eligible=false provider_spend_attempted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
