#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a contest-granularity atom lattice from pair/xray/gradient signals.

The output is planning-only.  It canonicalizes the signals that drive local
CPU scorer-oracle search: pair distortions, frame/pixel xray evidence,
sidecar-selected pairs, and master-gradient byte saliency.  The report also
exports meta-Lagrangian and cathedral-autopilot projections so downstream
rankers can consume the same lattice without re-parsing source artifacts.
"""

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

from tac.atom.contest_granularity import (  # noqa: E402
    ContestAtom,
    ContestSignal,
    build_lattice_report,
    byte_atoms_from_master_gradient,
    frame_and_pixel_atoms_from_xray_row,
    load_jsonl,
    merge_atoms_by_id,
    pair_atom_from_component_row,
    select_latest_master_gradient_anchor,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


MappingLike = dict[str, Any]


def _pair_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if limit is not None:
        return rows[:limit]
    return rows


def _xray_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise SystemExit(f"xray JSON missing list field 'rows': {path}")
    out = [row for row in rows if isinstance(row, dict)]
    if limit is not None:
        return out[:limit]
    return out


def _selected_pairs_from_manifest(path: Path | None) -> set[int]:
    if path is None:
        return set()
    payload = _read_json(path)
    selected = (
        payload.get("selection", {}).get("selected_pairs")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(selected, list):
        raise SystemExit(f"candidate manifest missing selection.selected_pairs: {path}")
    return {int(value) for value in selected}


def _load_master_gradient_anchor(
    *,
    anchors_jsonl: Path | None,
    archive_sha256: str | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if anchors_jsonl is None:
        return None, None
    rows = load_jsonl(anchors_jsonl)
    anchor = select_latest_master_gradient_anchor(rows, archive_sha256=archive_sha256)
    if anchor is None:
        return None, None
    gradient_path = Path(str(anchor["gradient_array_path"]))
    if not gradient_path.is_absolute():
        gradient_path = REPO_ROOT / gradient_path
    if not gradient_path.is_file():
        raise SystemExit(f"master-gradient array not found: {gradient_path}")
    return dict(anchor), gradient_path


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    atoms: list[ContestAtom] = []
    selected_pairs = _selected_pairs_from_manifest(args.sidecar_manifest)
    pair_ref = _repo_relative(args.pair_rows)
    for row in _pair_rows(args.pair_rows, limit=args.max_pair_rows):
        pair = int(row.get("pair", row.get("pair_idx", -1)))
        atoms.append(
            pair_atom_from_component_row(
                row,
                evidence_axis=args.evidence_axis,
                evidence_ref=pair_ref,
                selected=pair in selected_pairs,
            )
        )

    if args.xray_json is not None:
        xray_ref = _repo_relative(args.xray_json)
        for row in _xray_rows(args.xray_json, limit=args.max_xray_rows):
            atoms.extend(
                frame_and_pixel_atoms_from_xray_row(
                    row,
                    evidence_axis=args.evidence_axis,
                    evidence_ref=xray_ref,
                )
            )

    anchor, gradient_path = _load_master_gradient_anchor(
        anchors_jsonl=args.master_gradient_anchors,
        archive_sha256=args.archive_sha256,
    )
    if anchor is not None and gradient_path is not None and args.top_gradient_bytes > 0:
        import numpy as np

        atoms.extend(
            byte_atoms_from_master_gradient(
                np.load(gradient_path),
                anchor=anchor,
                top_k=args.top_gradient_bytes,
                evidence_ref=_repo_relative(gradient_path),
            )
        )

    merged = merge_atoms_by_id(atoms)
    report = build_lattice_report(
        merged,
        source="tools/build_contest_atom_lattice.py",
        generated_at_utc=datetime.now(UTC).isoformat(),
    )
    report["input_artifacts"] = {
        "pair_rows": _repo_relative(args.pair_rows),
        "xray_json": _repo_relative(args.xray_json) if args.xray_json else None,
        "sidecar_manifest": (
            _repo_relative(args.sidecar_manifest) if args.sidecar_manifest else None
        ),
        "master_gradient_anchors": (
            _repo_relative(args.master_gradient_anchors)
            if args.master_gradient_anchors
            else None
        ),
        "selected_pairs": sorted(selected_pairs),
        "master_gradient_anchor": anchor,
    }
    report["cathedral_contract"] = {
        "canonical_signal_surface": True,
        "represents": ["bytes", "pixels", "frames", "pairs"],
        "wired_outputs": [
            "meta_lagrangian_rows",
            "cathedral_candidate_rows",
            "top_waterfill_atoms",
        ],
        "next_candidate_builder_contract": (
            "consume top_waterfill_atoms, materialize byte-closed candidates, "
            "measure with CPU scorer oracle, then promote only after exact CUDA auth eval"
        ),
    }
    return report


def _repo_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-rows", type=Path, required=True)
    parser.add_argument("--xray-json", type=Path)
    parser.add_argument("--sidecar-manifest", type=Path)
    parser.add_argument("--master-gradient-anchors", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--top-gradient-bytes", type=int, default=64)
    parser.add_argument("--max-pair-rows", type=int)
    parser.add_argument("--max-xray-rows", type=int)
    parser.add_argument("--evidence-axis", default="[macOS-CPU advisory]")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "atom_count": report["atom_count"],
                "scope_counts": report["scope_counts"],
                "venn_counts": report["venn_counts"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
