#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the L5 v2 asymptotic candidate surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization.l5_staircase_v2 import (  # noqa: E402
    L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_ARTIFACT_PATH,
    L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_REPORT_PATH,
    asymptotic_candidate_surface_json,
    l5_v2_asymptotic_pursuit_candidates,
    render_l5_v2_asymptotic_candidate_surface_markdown,
)


def _refuse_transient_output(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(
            f"refusing to write L5 v2 asymptotic candidate surface to tmp: {text!r}"
        )


def _resolve_output(path: Path, *, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.expanduser().resolve()
    resolved.relative_to(repo_root)
    _refuse_transient_output(resolved)
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_REPORT_PATH),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    try:
        output_json = _resolve_output(args.output_json, repo_root=repo_root)
        output_md = _resolve_output(args.output_md, repo_root=repo_root)
        payload = l5_v2_asymptotic_pursuit_candidates(repo_root=repo_root)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            asymptotic_candidate_surface_json(payload),
            encoding="utf-8",
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(
            render_l5_v2_asymptotic_candidate_surface_markdown(payload),
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"[l5-v2-asymptotic-surface] FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        "[l5-v2-asymptotic-surface] "
        f"candidate_count={payload['candidate_count']} "
        f"ready_for_paid_dispatch={str(payload['ready_for_paid_dispatch']).lower()} "
        "score_claim=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
