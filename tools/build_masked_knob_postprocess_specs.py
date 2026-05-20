#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build exact-frame raw postprocess specs from a masked knob curriculum."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


def _spec(
    *,
    spec_id: str,
    kind: str,
    frame_indices: list[int],
    channel_deltas: tuple[int, int, int] = (0, 0, 0),
    alpha_num: int = 0,
    alpha_den: int = 1,
    notes: str,
) -> dict[str, Any]:
    return {
        "spec_id": spec_id,
        "kind": kind,
        "frame_selector": "all",
        "frame_indices": frame_indices,
        "channel_deltas": list(channel_deltas),
        "alpha_num": alpha_num,
        "alpha_den": alpha_den,
        "notes": notes,
    }


def build_specs(curriculum: Mapping[str, Any], *, max_specs: int) -> dict[str, Any]:
    layers = curriculum.get("adjustment_layers")
    if not isinstance(layers, list):
        raise ValueError("curriculum missing adjustment_layers[]")
    specs: list[dict[str, Any]] = []
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        target = layer.get("target")
        if not isinstance(target, Mapping):
            continue
        frames = [int(value) for value in target.get("frames", [])]
        if not frames:
            continue
        layer_id = str(layer.get("layer_id", "layer"))
        primary_axis = str(layer.get("primary_axis", "unknown"))
        if primary_axis == "seg":
            for delta, suffix in ((-1, "m1"), (1, "p1")):
                specs.append(
                    _spec(
                        spec_id=f"{layer_id}_rgb_bias_{suffix}",
                        kind="channel_bias",
                        frame_indices=frames,
                        channel_deltas=(delta, delta, delta),
                        notes=f"Exact-frame RGB bias probe from {layer_id}; advisory only.",
                    )
                )
        elif primary_axis == "pose":
            specs.append(
                _spec(
                    spec_id=f"{layer_id}_temporal_blend_a1_8",
                    kind="temporal_blend",
                    frame_indices=frames,
                    alpha_num=1,
                    alpha_den=8,
                    notes=f"Exact-frame temporal blend probe from {layer_id}; advisory only.",
                )
            )
        if len(specs) >= max_specs:
            specs = specs[:max_specs]
            break
    return {
        "schema": "masked_knob_postprocess_specs.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "source_curriculum_schema": curriculum.get("schema"),
        "spec_count": len(specs),
        "specs": specs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--max-specs", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        curriculum = json.loads(args.curriculum_json.read_text(encoding="utf-8"))
        if not isinstance(curriculum, dict):
            raise ValueError("curriculum must be a JSON object")
        payload = build_specs(curriculum, max_specs=args.max_specs)
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "spec_count": payload["spec_count"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
