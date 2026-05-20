#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inspect the decoder-q observable lattice for planning/debugging."""

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

from tac.optimization.decoder_q_observable import DecoderQObservableLattice  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    feasibility = _read_json(args.feasibility.resolve())
    advisory = _read_json(args.advisory_summary.resolve()) if args.advisory_summary else None
    lattice = DecoderQObservableLattice.from_feasibility_and_advisory(
        feasibility,
        advisory,
        baseline_score=args.baseline_score,
    )
    tensors = lattice.tensorize(device=args.device)
    measured_delta = tensors["measured_delta_score"]
    measured_mask = tensors["measured_mask"]
    return {
        "schema": "decoder_q_observable_lattice_inspection_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/inspect_decoder_q_observable_lattice.py",
        "inputs": {
            "feasibility": str(args.feasibility.resolve()),
            "advisory_summary": str(args.advisory_summary.resolve()) if args.advisory_summary else None,
            "baseline_score": args.baseline_score,
            "device": args.device,
        },
        "summary": lattice.summary(),
        "tensor_stats": {
            "target_mass_sum": float(tensors["target_mass"].sum().item()),
            "seg_mass_sum": float(tensors["axis_mass"][:, 0].sum().item()),
            "pose_mass_sum": float(tensors["axis_mass"][:, 1].sum().item()),
            "measured_delta_min": (
                float(measured_delta[measured_mask].min().item()) if bool(measured_mask.any()) else None
            ),
            "measured_delta_max": (
                float(measured_delta[measured_mask].max().item()) if bool(measured_mask.any()) else None
            ),
        },
        "signed_slope_rows": lattice.signed_slope_rows(),
        "top_atoms": lattice.top_atoms(limit=args.limit),
        "top_unmeasured_atoms": lattice.top_atoms(limit=args.limit, require_unmeasured=True),
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "notes": "Observable planning surface only; byte rebuild and scorer outputs remain authoritative.",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--advisory-summary", type=Path)
    parser.add_argument("--baseline-score", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": payload["summary"],
                "measured_delta_min": payload["tensor_stats"]["measured_delta_min"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
