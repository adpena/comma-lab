#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rank PR106 component-moving candidate cells from scorer-table rows.

This is a local planning artifact for PR106 latent/y-shift score tables.  It
does not build archives, dispatch jobs, or claim scores.  A ranked cell is only
the local marginal:

    candidate_score_without_rate - baseline_score_without_rate
    + 25 * estimated_cell_byte_delta / 37_545_489

Negative values are promising planning rows; exact CPU/CUDA auth eval on a
byte-closed archive is still mandatory before any frontier or promotion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tac.packet_compiler.pr106_latent_sidecar_selection import (  # noqa: E402
    PR106_NO_OP_DIM,
    build_latent_candidate_grid,
)
from tac.repo_io import repo_relative, sha256_file, write_json  # noqa: E402

SCHEMA = "pr106_component_moving_cell_plan_v1"
TOOL = "tools/plan_pr106_component_moving_cells.py"
ORIGINAL_UNCOMPRESSED_SIZE_BYTES = 37_545_489
FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "dispatch_attempted": False,
    "remote_jobs_dispatched": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, REPO_ROOT),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _build_yshift_candidate_grid(radius: int) -> np.ndarray:
    if radius < 0 or radius > 127:
        raise ValueError(f"candidate_radius must be in 0..127, got {radius}")
    values = range(-radius, radius + 1)
    return np.asarray(
        [(y_off, dy, dx) for y_off in values for dy in values for dx in values],
        dtype=np.int8,
    )


def _table_kind(manifest: Mapping[str, Any], candidates: np.ndarray | None) -> str:
    schema = str(manifest.get("manifest_schema", ""))
    if schema == "pr106_latent_score_table_manifest_v1":
        return "latent_sidecar"
    if schema == "pr106_yshift_score_table_manifest_v1":
        return "yshift"
    if candidates is not None:
        if candidates.ndim == 2 and candidates.shape[1] == 2:
            return "latent_sidecar"
        if candidates.ndim == 2 and candidates.shape[1] == 3:
            return "yshift"
    raise ValueError(
        "cannot infer score-table kind; expected PR106 latent/yshift manifest or a 2-column/3-column candidate grid"
    )


def _candidate_grid(
    *,
    manifest: Mapping[str, Any],
    candidate_grid_npy: Path | None,
) -> tuple[np.ndarray, str]:
    if candidate_grid_npy is not None:
        grid = np.load(candidate_grid_npy, allow_pickle=False)
        if not isinstance(grid, np.ndarray):
            raise TypeError(f"candidate grid must be ndarray, got {type(grid).__name__}")
        return grid, "candidate_grid_npy"

    schema = str(manifest.get("manifest_schema", ""))
    if schema == "pr106_latent_score_table_manifest_v1":
        latent_dim = int(manifest.get("latent_dim", 28))
        delta_radius = int(manifest.get("delta_radius", 1))
        return (
            build_latent_candidate_grid(latent_dim=latent_dim, delta_radius=delta_radius),
            "latent_manifest_reconstruction",
        )
    if schema == "pr106_yshift_score_table_manifest_v1":
        candidate_radius = int(manifest.get("candidate_radius", 3))
        return _build_yshift_candidate_grid(candidate_radius), "yshift_manifest_reconstruction"

    raise ValueError("--candidate-grid-npy is required for unknown score-table manifest schema")


def _baseline_candidate_index(
    *,
    kind: str,
    manifest: Mapping[str, Any],
    candidates: np.ndarray,
) -> int:
    key = "noop_candidate_index" if kind == "latent_sidecar" else "zero_candidate_index"
    if key in manifest:
        idx = int(manifest[key])
        if 0 <= idx < int(candidates.shape[0]):
            return idx
        raise ValueError(f"{key}={idx} is outside candidate grid")

    if kind == "latent_sidecar":
        matches = np.flatnonzero((candidates[:, 0] == PR106_NO_OP_DIM) & (candidates[:, 1] == 0))
    else:
        matches = np.flatnonzero((candidates == 0).all(axis=1))
    if len(matches) != 1:
        raise ValueError(f"candidate grid must contain exactly one {kind} baseline row")
    return int(matches[0])


def _axis_label(manifest: Mapping[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    device = str(manifest.get("device", "")).lower()
    if "cuda" in device:
        return "[compress-time CUDA scorer table]"
    if "cpu" in device:
        return "[CPU advisory scorer table]"
    return "[unknown proxy scorer table]"


def _default_cell_byte_delta(kind: str) -> tuple[float, str]:
    if kind == "latent_sidecar":
        return 2.0, "default_raw_dim_delta_cell_estimate_not_archive_delta"
    return 3.0, "default_raw_yoff_dy_dx_cell_estimate_not_archive_delta"


def _candidate_payload(kind: str, row: np.ndarray) -> dict[str, int]:
    values = [int(v) for v in row.tolist()]
    if kind == "latent_sidecar":
        return {"dim": values[0], "delta_q": values[1]}
    return {"y_off": values[0], "dy": values[1], "dx": values[2]}


def _row_location(kind: str, row_idx: int) -> dict[str, int | None]:
    if kind == "yshift":
        return {
            "row_idx": int(row_idx),
            "pair_idx": int(row_idx // 2),
            "frame_slot": int(row_idx % 2),
        }
    return {"row_idx": int(row_idx), "pair_idx": int(row_idx), "frame_slot": None}


def _load_xray_rows(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json(path)
    if payload.get("schema") != "pair_component_error_xray_v1":
        raise ValueError(f"{path} is not a pair_component_error_xray_v1 JSON")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and "pair_idx" in row:
            out[int(row["pair_idx"])] = row
    return out


def _finite_table(path: Path) -> np.ndarray:
    table = np.load(path, allow_pickle=False)
    if not isinstance(table, np.ndarray):
        raise TypeError(f"score table must be ndarray, got {type(table).__name__}")
    if table.ndim != 2:
        raise ValueError(f"score table must be 2D, got shape {table.shape}")
    table = np.asarray(table, dtype=np.float64)
    if not np.isfinite(table).all():
        raise ValueError("score table contains NaN/Inf")
    return table


def _top_cell_indices(
    net_delta: np.ndarray,
    *,
    baseline_idx: int,
    top_k: int,
) -> list[tuple[int, int]]:
    if top_k <= 0:
        return []
    ranked = net_delta.copy()
    ranked[:, baseline_idx] = math.inf
    flat = ranked.reshape(-1)
    take = min(int(top_k), int(np.isfinite(flat).sum()))
    if take == 0:
        return []
    best = np.argsort(flat, kind="stable")[:take]
    cols = int(ranked.shape[1])
    return [(int(idx // cols), int(idx % cols)) for idx in best.tolist()]


def build_plan(
    *,
    score_table_npy: Path,
    score_table_manifest: Path,
    candidate_grid_npy: Path | None = None,
    xray_json: Path | None = None,
    top_k: int = 32,
    cell_byte_delta: float | None = None,
    score_axis_label: str | None = None,
    label: str = "pr106_component_moving_cells",
) -> dict[str, Any]:
    manifest = _load_json(score_table_manifest)
    candidates, candidate_grid_source = _candidate_grid(
        manifest=manifest,
        candidate_grid_npy=candidate_grid_npy,
    )
    kind = _table_kind(manifest, candidates)
    table = _finite_table(score_table_npy)
    if int(table.shape[1]) != int(candidates.shape[0]):
        raise ValueError(f"score table candidate dimension {table.shape[1]} != candidate grid {candidates.shape[0]}")

    expected_shape = manifest.get("score_table_shape")
    if (
        isinstance(expected_shape, list)
        and len(expected_shape) == 2
        and [int(table.shape[0]), int(table.shape[1])] != [int(expected_shape[0]), int(expected_shape[1])]
    ):
        raise ValueError(f"score table shape {list(table.shape)} does not match manifest {expected_shape}")

    baseline_idx = _baseline_candidate_index(
        kind=kind,
        manifest=manifest,
        candidates=candidates,
    )
    if cell_byte_delta is None:
        byte_delta, byte_delta_source = _default_cell_byte_delta(kind)
    else:
        byte_delta = float(cell_byte_delta)
        byte_delta_source = "cli_cell_byte_delta"
    rate_delta = 25.0 * byte_delta / ORIGINAL_UNCOMPRESSED_SIZE_BYTES
    baseline = table[:, baseline_idx][:, None]
    component_delta = table - baseline
    net_delta = component_delta + rate_delta
    xray_rows = _load_xray_rows(xray_json)
    source_axis = _axis_label(manifest, score_axis_label)
    top_cells: list[dict[str, Any]] = []
    for rank, (row_idx, cand_idx) in enumerate(
        _top_cell_indices(net_delta, baseline_idx=baseline_idx, top_k=top_k),
        start=1,
    ):
        location = _row_location(kind, row_idx)
        xray = xray_rows.get(int(location["pair_idx"])) if location["pair_idx"] is not None else None
        cell = {
            "rank": rank,
            "cell_id": f"{kind}:row{row_idx}:candidate{cand_idx}",
            **location,
            "candidate_index": cand_idx,
            "candidate": _candidate_payload(kind, candidates[cand_idx]),
            "baseline_candidate_index": int(baseline_idx),
            "baseline_candidate": _candidate_payload(kind, candidates[baseline_idx]),
            "baseline_score_without_rate": float(table[row_idx, baseline_idx]),
            "candidate_score_without_rate": float(table[row_idx, cand_idx]),
            "component_score_delta_no_rate": float(component_delta[row_idx, cand_idx]),
            "byte_delta": float(byte_delta),
            "byte_delta_source": byte_delta_source,
            "rate_score_delta": float(rate_delta),
            "net_score_delta_charged": float(net_delta[row_idx, cand_idx]),
            "negative_is_better": True,
            "source_score_axis_label": source_axis,
            "required_promotion_axes": ["[contest-CUDA]", "[contest-CPU]"],
            "false_authority": {
                **FALSE_AUTHORITY_FLAGS,
                "reason": (
                    "score-table cells are compress-time marginal evidence only; "
                    "interactions, byte packing, and CPU/CUDA scorer axes require "
                    "byte-closed exact eval"
                ),
            },
        }
        if xray is not None:
            cell["xray_pair_context"] = {
                key: xray.get(key)
                for key in (
                    "component_score_no_rate",
                    "pose_score_contribution",
                    "seg_score_contribution",
                    "frame0_l1",
                    "frame1_l1",
                )
                if key in xray
            }
        top_cells.append(cell)

    improving_component = component_delta < 0.0
    improving_net = net_delta < 0.0
    improving_component[:, baseline_idx] = False
    improving_net[:, baseline_idx] = False
    state_hash = json.dumps(
        {
            "score_table": sha256_file(score_table_npy),
            "manifest": sha256_file(score_table_manifest),
            "candidate_grid_source": candidate_grid_source,
            "top_k": int(top_k),
            "byte_delta": byte_delta,
            "label": label,
        },
        sort_keys=True,
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "from_state_hash": hashlib.sha256(state_hash.encode("utf-8")).hexdigest()[:16],
        "label": label,
        **FALSE_AUTHORITY_FLAGS,
        "kind": kind,
        "score_delta_semantics": (
            "net_score_delta_charged = candidate_score_without_rate - "
            "baseline_score_without_rate + 25*byte_delta/37545489; negative is better"
        ),
        "axis_labels": {
            "source_score_table": source_axis,
            "contest_cuda": "[contest-CUDA] required before promotion or frontier language",
            "contest_cpu": "[contest-CPU] paired public-axis check required before CPU-axis language",
        },
        "inputs": {
            "score_table_npy": _artifact(score_table_npy),
            "score_table_manifest": _artifact(score_table_manifest),
            "candidate_grid_npy": _artifact(candidate_grid_npy) if candidate_grid_npy else None,
            "candidate_grid_source": candidate_grid_source,
            "xray_json": _artifact(xray_json) if xray_json else None,
        },
        "manifest_authority_flags": {
            key: manifest.get(key)
            for key in (
                "score_claim",
                "ready_for_builder",
                "ready_for_exact_eval_dispatch",
                "dispatch_attempted",
                "remote_jobs_dispatched",
                "promotion_eligible",
                "rank_or_kill_eligible",
            )
            if key in manifest
        },
        "table_summary": {
            "row_count": int(table.shape[0]),
            "candidate_count": int(table.shape[1]),
            "baseline_candidate_index": int(baseline_idx),
            "component_improving_cell_count": int(improving_component.sum()),
            "net_improving_cell_count": int(improving_net.sum()),
            "best_component_score_delta_no_rate": float(
                np.min(np.where(np.arange(table.shape[1]) == baseline_idx, math.inf, component_delta))
            ),
            "best_net_score_delta_charged": float(
                np.min(np.where(np.arange(table.shape[1]) == baseline_idx, math.inf, net_delta))
            ),
            "cell_byte_delta": float(byte_delta),
            "cell_byte_delta_source": byte_delta_source,
            "rate_score_delta_per_cell": float(rate_delta),
        },
        "top_cells": top_cells,
        "solver_hooks": {
            "sensitivity_map": "top_cells expose pair/frame candidate component deltas for tac.sensitivity_map ingestion",
            "pareto_constraint": "all rows carry score_claim=false and exact-eval dispatch blockers",
            "bit_allocator_hook": "rank by net_score_delta_charged per estimated cell byte",
            "cathedral_autopilot_dispatch_hook": "materialize selected cells first, then claim exact CPU/CUDA eval lanes",
            "continual_learning_posterior": "N/A until exact auth-eval empirical anchor exists",
            "probe_disambiguator": "N/A single scorer-table reduction rule; alternate pack grammars must be probed separately",
        },
        "dispatch_blockers": [
            "planning_artifact_only",
            "requires_byte_closed_archive_materialization",
            "requires_lane_dispatch_claim_before_exact_eval",
            "requires_paired_contest_cuda_auth_eval",
            "requires_paired_contest_cpu_auth_eval",
            "requires_adjudicated_component_recompute_before_score_claim",
        ],
        "recommended_next_step": (
            "Materialize a small top-k cell packet using the matching PR106 score-table "
            "builder, then run paired exact CPU/CUDA auth eval under claimed lanes."
        ),
    }


def render_markdown(plan: Mapping[str, Any]) -> str:
    summary = plan["table_summary"]
    lines = [
        "# PR106 Component-Moving Cell Plan",
        "",
        f"- schema: `{plan['schema']}`",
        f"- label: `{plan['label']}`",
        f"- kind: `{plan['kind']}`",
        f"- source_score_table_axis: `{plan['axis_labels']['source_score_table']}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "",
        "## Summary",
        "",
        f"- rows: `{summary['row_count']}`",
        f"- candidates: `{summary['candidate_count']}`",
        f"- component_improving_cells: `{summary['component_improving_cell_count']}`",
        f"- net_improving_cells: `{summary['net_improving_cell_count']}`",
        f"- cell_byte_delta: `{summary['cell_byte_delta']}` ({summary['cell_byte_delta_source']})",
        f"- best_net_score_delta_charged: `{summary['best_net_score_delta_charged']}`",
        "",
        "## Top Cells",
        "",
        "| rank | row | pair | frame | candidate | component_delta | byte_delta | net_delta |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for cell in plan["top_cells"]:
        candidate = ",".join(f"{k}={v}" for k, v in cell["candidate"].items())
        frame = "" if cell["frame_slot"] is None else str(cell["frame_slot"])
        lines.append(
            f"| {cell['rank']} | {cell['row_idx']} | {cell['pair_idx']} | {frame} | "
            f"`{candidate}` | {cell['component_score_delta_no_rate']:.9f} | "
            f"{cell['byte_delta']:.3f} | {cell['net_score_delta_charged']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            *[f"- `{blocker}`" for blocker in plan["dispatch_blockers"]],
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table-npy", type=Path, required=True)
    parser.add_argument("--score-table-manifest", type=Path, required=True)
    parser.add_argument("--candidate-grid-npy", type=Path)
    parser.add_argument("--xray-json", type=Path)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--cell-byte-delta", type=float)
    parser.add_argument("--score-axis-label")
    parser.add_argument("--label", default="pr106_component_moving_cells")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(
        score_table_npy=args.score_table_npy,
        score_table_manifest=args.score_table_manifest,
        candidate_grid_npy=args.candidate_grid_npy,
        xray_json=args.xray_json,
        top_k=args.top_k,
        cell_byte_delta=args.cell_byte_delta,
        score_axis_label=args.score_axis_label,
        label=args.label,
    )
    write_json(args.output_json, plan)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(plan), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
