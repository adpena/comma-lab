#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Route PathActionProducer supports through the cheapest valid codec."""

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
from tac.analysis.support_codec_router import (  # noqa: E402
    SUPPORT_CODEC_ROUTER_SCHEMA,
    route_support_codecs_for_path_candidates,
    write_support_codec_router_report,
)

DEFAULT_OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/experiments/results")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row is not an object: {path}")
        rows.append(payload)
    return rows


def _default_output_dir() -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"support_codec_router_{stamp}_codex"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-action-candidates", type=Path, required=True)
    parser.add_argument("--action-effect-rows", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    candidates_path = args.path_action_candidates.expanduser().resolve(strict=True)
    candidates = _read_jsonl(candidates_path)
    effects = []
    if args.action_effect_rows is not None:
        effects_path = args.action_effect_rows.expanduser().resolve(strict=True)
        effects = [ActionEffect.from_dict(row) for row in _read_jsonl(effects_path)]
    out_dir = (args.output_dir or _default_output_dir()).expanduser().resolve(strict=False)
    report = route_support_codecs_for_path_candidates(candidates, source_effects=effects)
    written = write_support_codec_router_report(report, out_dir.as_posix())
    selected = [
        sub.get("selected_support_encoding")
        for sub in report.get("reports", [])
        if isinstance(sub, dict)
    ]
    summary = {
        "schema": SUPPORT_CODEC_ROUTER_SCHEMA,
        "path_action_candidates": candidates_path.as_posix(),
        "action_effect_rows": args.action_effect_rows.as_posix() if args.action_effect_rows is not None else None,
        "output_dir": out_dir.as_posix(),
        **written,
        "selected_support_encodings": selected,
        "candidate_queue_contains_selected_only": True,
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
