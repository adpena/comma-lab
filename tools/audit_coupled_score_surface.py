#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit candidate points and transitions on the exact coupled score surface.

The target defaults to ``effective_frontier.score`` in the canonical frontier
pointer.  Inputs are dated research manifests; this tool contains no transient
score, component threshold, archive budget, or lane literal.

This is planning apparatus only.  It does not run scorers and cannot turn a
predicted or advisory component triple into an exact score claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.score_geometry import score_sublevel_audit, score_transition_audit  # noqa: E402

DEFAULT_POINTER = Path(".omx/state/canonical_frontier_pointer.json")


class CoupledScoreSurfaceError(ValueError):
    """Raised when a score-surface manifest lacks fail-closed custody."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoupledScoreSurfaceError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CoupledScoreSurfaceError(f"JSON root must be an object: {path}")
    return payload


def _point_coordinates(point: Mapping[str, Any]) -> tuple[float, float, int]:
    try:
        return (
            float(point["d_seg"]),
            float(point["d_pose"]),
            int(point["archive_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CoupledScoreSurfaceError(
            "every point requires numeric d_seg, d_pose, and archive_bytes"
        ) from exc


def _target_from_pointer(pointer: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    effective = pointer.get("effective_frontier")
    if not isinstance(effective, Mapping):
        raise CoupledScoreSurfaceError("pointer lacks effective_frontier object")
    try:
        score = float(effective["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CoupledScoreSurfaceError("pointer effective_frontier lacks score") from exc
    if score < 0.0:
        raise CoupledScoreSurfaceError("pointer effective score must be non-negative")
    return score, {
        "axis": effective.get("axis"),
        "custody": effective.get("custody"),
        "evidence_grade": effective.get("evidence_grade"),
        "score_precision": effective.get("score_precision"),
        "source": effective.get("source"),
        "source_kind": effective.get("source_kind"),
    }


def build_report(
    *,
    manifest_path: Path,
    pointer_path: Path,
) -> dict[str, Any]:
    """Build a JSON-safe exact score-surface audit from two immutable inputs."""
    manifest = _load_json(manifest_path)
    pointer = _load_json(pointer_path)
    target_score, target_custody = _target_from_pointer(pointer)

    points = manifest.get("points", [])
    transitions = manifest.get("transitions", [])
    if not isinstance(points, list) or not isinstance(transitions, list):
        raise CoupledScoreSurfaceError("points and transitions must be arrays")

    point_rows: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, Mapping) or not str(point.get("id", "")).strip():
            raise CoupledScoreSurfaceError("every point requires a non-empty id")
        d_seg, d_pose, archive_bytes = _point_coordinates(point)
        audit = score_sublevel_audit(
            target_score=target_score,
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=archive_bytes,
        )
        point_rows.append(
            {
                "id": str(point["id"]),
                "input_custody": point.get("custody"),
                "input_evidence_axis": point.get("evidence_axis"),
                "audit": asdict(audit),
            }
        )

    transition_rows: list[dict[str, Any]] = []
    for transition in transitions:
        if not isinstance(transition, Mapping) or not str(transition.get("id", "")).strip():
            raise CoupledScoreSurfaceError("every transition requires a non-empty id")
        before = transition.get("before")
        after = transition.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise CoupledScoreSurfaceError("every transition requires before and after points")
        before_d_seg, before_d_pose, before_bytes = _point_coordinates(before)
        after_d_seg, after_d_pose, after_bytes = _point_coordinates(after)
        audit = score_transition_audit(
            target_score=target_score,
            before_d_seg=before_d_seg,
            before_d_pose=before_d_pose,
            before_archive_bytes=before_bytes,
            after_d_seg=after_d_seg,
            after_d_pose=after_d_pose,
            after_archive_bytes=after_bytes,
        )
        transition_rows.append(
            {
                "id": str(transition["id"]),
                "input_custody": transition.get("custody"),
                "input_evidence_axis": transition.get("evidence_axis"),
                "audit": asdict(audit),
            }
        )

    return {
        "schema": "coupled_score_surface_audit.v1",
        "target": {
            "score": target_score,
            "pointer_path": str(pointer_path),
            **target_custody,
        },
        "admission_rule": (
            "accept a candidate transition iff its exact finite joint score delta is negative; "
            "the competitive terminal set is the strict coupled score sublevel"
        ),
        "independent_component_thresholds_are_admission_rules": False,
        "points": point_rows,
        "transitions": transition_rows,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "source_manifest": str(manifest_path),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    report = build_report(
        manifest_path=_resolve(args.repo_root, args.manifest),
        pointer_path=_resolve(args.repo_root, args.pointer),
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _atomic_write_json(_resolve(args.repo_root, args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
