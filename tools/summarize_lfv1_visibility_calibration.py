#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize LFV1 sparse visibility sweeps into per-pair calibration rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _pair_changed(visibility: dict[str, Any], pair: int) -> bool:
    changed = visibility.get("changed_frame_indices")
    if not isinstance(changed, list):
        return False
    changed_set = {int(frame) for frame in changed if isinstance(frame, int)}
    return (2 * int(pair) in changed_set) or (2 * int(pair) + 1 in changed_set)


def _load_candidate_row(result: dict[str, Any]) -> dict[str, Any] | None:
    if result.get("returncode") != 0:
        return None
    manifest_path = result.get("manifest_path")
    visibility_path = result.get("visibility_path")
    if not isinstance(manifest_path, str) or not isinstance(visibility_path, str):
        return None
    manifest = _read_json(Path(manifest_path))
    visibility = _read_json(Path(visibility_path))
    selection = manifest.get("selection") if isinstance(manifest.get("selection"), dict) else {}
    sidecar = manifest.get("sidecar") if isinstance(manifest.get("sidecar"), dict) else {}
    active = sidecar.get("active_row_float_source") if isinstance(sidecar.get("active_row_float_source"), dict) else {}
    archive = manifest.get("archive") if isinstance(manifest.get("archive"), dict) else {}
    alpha = _float(active.get("alpha"))
    selected_pairs = selection.get("selected_pairs")
    if alpha is None or not isinstance(selected_pairs, list):
        return None
    return {
        "candidate_id": result.get("candidate_id") or manifest.get("candidate_id"),
        "alpha": alpha,
        "radius": _float(active.get("radius")),
        "power": _float(active.get("power")),
        "origin_x": _float(active.get("origin_x")),
        "origin_y": _float(active.get("origin_y")),
        "selected_pairs": [int(pair) for pair in selected_pairs if isinstance(pair, int)],
        "changed_frame_indices": visibility.get("changed_frame_indices"),
        "uint8_visible": bool(visibility.get("uint8_visible")),
        "archive_delta_bytes": archive.get("delta_bytes_vs_source_archive"),
        "manifest_path": manifest_path,
        "visibility_path": visibility_path,
        "visibility": visibility,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    batch = _read_json(args.sparse_batch_manifest.resolve())
    rows = [
        row
        for result in batch.get("results", [])
        if isinstance(result, dict)
        for row in [_load_candidate_row(result)]
        if row is not None
    ]
    by_pair: dict[int, dict[str, Any]] = {}
    for row in rows:
        alpha = float(row["alpha"])
        geometry_key = (
            _float(row.get("radius")),
            _float(row.get("power")),
            _float(row.get("origin_x")),
            _float(row.get("origin_y")),
        )
        for pair in row["selected_pairs"]:
            pair_row = by_pair.setdefault(
                int(pair),
                {
                    "pair_index": int(pair),
                    "geometry": {},
                    "candidate_ids": [],
                },
            )
            pair_row["candidate_ids"].append(row["candidate_id"])
            geometry_row = pair_row["geometry"].setdefault(
                geometry_key,
                {
                    "radius": geometry_key[0],
                    "power": geometry_key[1],
                    "origin_x": geometry_key[2],
                    "origin_y": geometry_key[3],
                    "tested_alpha_values": set(),
                    "visible_alpha_values": set(),
                    "noop_alpha_values": set(),
                    "candidate_ids": [],
                },
            )
            geometry_row["tested_alpha_values"].add(alpha)
            geometry_row["candidate_ids"].append(row["candidate_id"])
            if _pair_changed(row["visibility"], int(pair)):
                geometry_row["visible_alpha_values"].add(alpha)
            else:
                geometry_row["noop_alpha_values"].add(alpha)

    pair_rows: list[dict[str, Any]] = []
    for pair, raw in sorted(by_pair.items()):
        geometry_rows: list[dict[str, Any]] = []
        all_visible: set[float] = set()
        all_noop: set[float] = set()
        all_tested: set[float] = set()
        for geometry_key, geometry in sorted(raw["geometry"].items()):
            visible = sorted(geometry["visible_alpha_values"])
            noop = sorted(geometry["noop_alpha_values"])
            tested = sorted(geometry["tested_alpha_values"])
            all_visible.update(visible)
            all_noop.update(noop)
            all_tested.update(tested)
            geometry_rows.append(
                {
                    "radius": geometry["radius"],
                    "power": geometry["power"],
                    "origin_x": geometry["origin_x"],
                    "origin_y": geometry["origin_y"],
                    "tested_alpha_values": tested,
                    "noop_alpha_values": noop,
                    "visible_alpha_values": visible,
                    "min_visible_alpha": visible[0] if visible else None,
                    "max_noop_alpha": noop[-1] if noop else None,
                    "candidate_ids": sorted({str(value) for value in geometry["candidate_ids"]}),
                }
            )
        visible = sorted(all_visible)
        noop = sorted(all_noop)
        tested = sorted(all_tested)
        scalar_alpha_threshold_valid = not (
            visible and noop and max(noop) >= min(visible)
        )
        pair_rows.append(
            {
                "pair_index": pair,
                "tested_alpha_values": tested,
                "noop_alpha_values": noop,
                "visible_alpha_values": visible,
                "global_min_visible_alpha": visible[0] if visible else None,
                "global_max_noop_alpha": noop[-1] if noop else None,
                "scalar_alpha_threshold_valid": scalar_alpha_threshold_valid,
                "geometry_dependent_visibility": not scalar_alpha_threshold_valid,
                "geometry_rows": geometry_rows,
                "candidate_ids": sorted({str(value) for value in raw["candidate_ids"]}),
                "score_claim": False,
                "promotion_eligible": False,
            }
        )

    return {
        "schema": "lfv1_alpha_visibility_calibration_v1",
        "sparse_batch_manifest": str(args.sparse_batch_manifest),
        "source_queue": batch.get("queue"),
        "candidate_count": len(rows),
        "pair_count": len(pair_rows),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "next_queue_rule": (
            "Use min_visible_alpha as a lower bound only; scorer eval still must "
            "separately prove per-axis PoseNet/SegNet benefit exceeds rate."
        ),
        "pair_rows": pair_rows,
        "candidate_rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"visibility"}
            }
            for row in rows
        ],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LFV1 Alpha Visibility Calibration",
        "",
        f"- sparse_batch_manifest: `{payload['sparse_batch_manifest']}`",
        f"- candidates: `{payload['candidate_count']}`",
        f"- pairs: `{payload['pair_count']}`",
        "- score_claim: `false`",
        "",
        "| pair | geometry_dependent | global_max_noop_alpha | global_min_visible_alpha | tested_alpha_values |",
        "|---:|---:|---:|---|",
    ]
    for row in payload["pair_rows"]:
        lines.append(
            "| {pair_index} | {geometry_dependent} | {global_max_noop_alpha} | {global_min_visible_alpha} | `{tested}` |".format(
                pair_index=row["pair_index"],
                geometry_dependent=row["geometry_dependent_visibility"],
                global_max_noop_alpha=row["global_max_noop_alpha"],
                global_min_visible_alpha=row["global_min_visible_alpha"],
                tested=",".join(str(value) for value in row["tested_alpha_values"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-batch-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = summarize(args)
    _write_json(args.output_json, payload)
    if args.output_md is not None:
        write_markdown(args.output_md, payload)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md) if args.output_md else None,
                "candidate_count": payload["candidate_count"],
                "pair_count": payload["pair_count"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
