#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Attach non-authoritative structural priors to a scorer-response dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.scorer_response_dataset import (  # noqa: E402
    ScorerResponseDatasetError,
    render_markdown,
)
from tac.optimization.scorer_response_structural_features import (  # noqa: E402
    attach_structural_features,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--frame-axis-l1-npy", type=Path)
    parser.add_argument("--frame-decomposition-json", type=Path)
    parser.add_argument("--decoder-q-mutation-manifest", type=Path)
    parser.add_argument("--decoder-q-family", default="mlx_decoder_q")
    return parser.parse_args(argv)


def _load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScorerResponseDatasetError(f"{path}: expected JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = _load_json(args.dataset)
        assert dataset is not None
        frame_axis_l1 = (
            None
            if args.frame_axis_l1_npy is None
            else np.load(args.frame_axis_l1_npy)
        )
        frame_decomposition = _load_json(args.frame_decomposition_json)
        decoder_q_mutation_manifest = _load_json(args.decoder_q_mutation_manifest)
        enriched = attach_structural_features(
            dataset,
            frame_axis_l1=frame_axis_l1,
            frame_axis_l1_source=(
                None if args.frame_axis_l1_npy is None else str(args.frame_axis_l1_npy)
            ),
            frame_decomposition=frame_decomposition,
            frame_decomposition_source=(
                None
                if args.frame_decomposition_json is None
                else str(args.frame_decomposition_json)
            ),
            decoder_q_mutation_manifest=decoder_q_mutation_manifest,
            decoder_q_mutation_manifest_source=(
                None
                if args.decoder_q_mutation_manifest is None
                else str(args.decoder_q_mutation_manifest)
            ),
            decoder_q_family=args.decoder_q_family,
        )
    except (OSError, ValueError, ScorerResponseDatasetError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(enriched), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "structural_features": enriched["structural_features"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
