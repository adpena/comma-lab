#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Append score-axis-dominance metadata corrections for master-gradient anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.master_gradient import (  # noqa: E402
    AGGREGATE_GRADIENT_TENSOR_KIND,
    MASTER_GRADIENT_LEDGER_PATH,
    append_score_axis_dominance_backfill,
    build_score_axis_dominance_backfill_anchor,
    effective_anchor_sort_key,
    is_usable_planning_anchor,
    load_anchors_lenient,
)
from tac.repo_io import write_json  # noqa: E402

SCHEMA = "master_gradient_score_axis_dominance_backfill_manifest_v1"


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _select_anchor(
    rows: list[dict[str, Any]],
    *,
    archive_sha256: str,
    tensor_kind: str,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("archive_sha256") == archive_sha256
        and row.get("gradient_tensor_kind", AGGREGATE_GRADIENT_TENSOR_KIND) == tensor_kind
        and is_usable_planning_anchor(row)
    ]
    if not candidates:
        raise SystemExit(
            f"no usable master-gradient anchor for archive={archive_sha256} tensor_kind={tensor_kind}"
        )
    return max(candidates, key=effective_anchor_sort_key)


def build_backfill_manifest(
    *,
    archive_sha256: str,
    anchor_path: Path,
    manifest_path: Path,
    tensor_kind: str = AGGREGATE_GRADIENT_TENSOR_KIND,
    axis_dominance_threshold: float = 0.7,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    rows = load_anchors_lenient(anchor_path)
    selected = _select_anchor(rows, archive_sha256=archive_sha256, tensor_kind=tensor_kind)
    ledger_sha_before = _sha256_file(anchor_path)
    already_present = isinstance(selected.get("score_axis_dominance"), Mapping)

    corrected = build_score_axis_dominance_backfill_anchor(
        selected,
        axis_dominance_threshold=axis_dominance_threshold,
    )
    append_attempted = False
    append_performed = False
    skip_reason = None
    if already_present and not force:
        skip_reason = "score_axis_dominance_already_present"
    elif dry_run:
        append_attempted = True
        skip_reason = "dry_run"
    else:
        append_attempted = True
        append_score_axis_dominance_backfill(
            selected,
            path=anchor_path,
            axis_dominance_threshold=axis_dominance_threshold,
        )
        append_performed = True

    ledger_sha_after = _sha256_file(anchor_path)
    dominance = corrected.get("score_axis_dominance")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_sha256": archive_sha256,
        "tensor_kind": tensor_kind,
        "anchor_path": _repo_rel(anchor_path),
        "anchor_sha256_before": ledger_sha_before,
        "anchor_sha256_after": ledger_sha_after,
        "append_attempted": append_attempted,
        "append_performed": append_performed,
        "skip_reason": skip_reason,
        "axis_dominance_threshold": float(axis_dominance_threshold),
        "selected_anchor": {
            "archive_sha256": selected.get("archive_sha256"),
            "measurement_axis": selected.get("measurement_axis"),
            "measurement_hardware": selected.get("measurement_hardware"),
            "measurement_method": selected.get("measurement_method"),
            "measurement_utc": selected.get("measurement_utc"),
            "written_at_utc": selected.get("written_at_utc"),
            "gradient_array_path": selected.get("gradient_array_path"),
            "gradient_tensor_kind": selected.get("gradient_tensor_kind"),
            "scored_archive_sha256": selected.get("scored_archive_sha256"),
            "scored_archive_bytes": selected.get("scored_archive_bytes"),
            "score_axis_dominance_already_present": already_present,
        },
        "corrected_score_axis_dominance": dominance,
        "manifest_path": _repo_rel(manifest_path),
    }
    write_json(manifest_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--anchor-path", type=Path, default=MASTER_GRADIENT_LEDGER_PATH)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument(
        "--tensor-kind",
        default=AGGREGATE_GRADIENT_TENSOR_KIND,
        choices=[AGGREGATE_GRADIENT_TENSOR_KIND],
    )
    parser.add_argument("--axis-dominance-threshold", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_backfill_manifest(
        archive_sha256=args.archive_sha256,
        anchor_path=args.anchor_path,
        manifest_path=args.manifest_path,
        tensor_kind=args.tensor_kind,
        axis_dominance_threshold=args.axis_dominance_threshold,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
