#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the minimal evaluator-action lowering race for one action id."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_effect import ActionEffect  # noqa: E402
from tac.analysis.evaluator_action_lowering_race import (  # noqa: E402
    LOWERING_RACE_SCHEMA,
    build_lowering_race_report,
    write_lowering_race_report,
)

DEFAULT_OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/experiments/results")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL object row expected: {path}")
        rows.append(payload)
    return rows


def _default_output_dir() -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"evaluator_action_lowering_race_{stamp}_codex"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-id", required=True)
    parser.add_argument("--support-codec-report", type=Path, default=None)
    parser.add_argument("--action-effect-rows", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    support_codec_report = None
    if args.support_codec_report is not None:
        support_codec_report = _read_json(args.support_codec_report.expanduser().resolve(strict=True))
    effects = []
    for path in args.action_effect_rows:
        effects.extend(ActionEffect.from_dict(row) for row in _read_jsonl(path.expanduser().resolve(strict=True)))
    report = build_lowering_race_report(
        action_id=str(args.action_id),
        action_effects=effects,
        support_codec_report=support_codec_report,
    )
    out_dir = (args.output_dir or _default_output_dir()).expanduser().resolve(strict=False)
    written = write_lowering_race_report(report, out_dir)
    verdict = report["verdict"]
    summary = {
        "schema": LOWERING_RACE_SCHEMA,
        "output_dir": out_dir.as_posix(),
        **written,
        "action_id": args.action_id,
        "best_lowering": verdict.get("best_lowering"),
        "first_failing_surface": verdict.get("first_failing_surface"),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
