#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register complete Task #494 receipts as canonical empirical anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.canonical_equations.registry import (  # noqa: E402
    get_equation_by_id,
    update_equation_with_empirical_anchor,
)
from tac.canonical_equations.throughput_authority_anchors_20260714 import (  # noqa: E402
    ARGMAX_CERTIFICATE_EQUATION_ID,
    EXACT_REDUCTION_EQUATION_ID,
    build_full_r_anchor,
    build_integer_r_backend_anchor,
    build_metal_segnet_anchor,
    build_qdq_anchor,
)

DEFAULTS = {
    "qdq_fixed": REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "fixedpoint_scorer_forward_n600_v2.json",
    "qdq_dynamic": REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "dynamic_fixedpoint_scorer_forward_n600.json",
    "full_r": REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "full_r_adjoint_n600.json",
    "metal": REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "metal_dynamic_fixedpoint_segnet_n600.json",
    "integer_r": REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "integer_r_backend_n600.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdq-fixed", type=Path, default=DEFAULTS["qdq_fixed"])
    parser.add_argument("--qdq-dynamic", type=Path, default=DEFAULTS["qdq_dynamic"])
    parser.add_argument("--full-r", type=Path, default=DEFAULTS["full_r"])
    parser.add_argument("--metal", type=Path, default=DEFAULTS["metal"])
    parser.add_argument("--integer-r", type=Path, default=DEFAULTS["integer_r"])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    specs = (
        (
            "qdq_fixed",
            args.qdq_fixed,
            ARGMAX_CERTIFICATE_EQUATION_ID,
            build_qdq_anchor,
        ),
        (
            "qdq_dynamic",
            args.qdq_dynamic,
            ARGMAX_CERTIFICATE_EQUATION_ID,
            build_qdq_anchor,
        ),
        ("full_r", args.full_r, EXACT_REDUCTION_EQUATION_ID, build_full_r_anchor),
        ("metal", args.metal, ARGMAX_CERTIFICATE_EQUATION_ID, build_metal_segnet_anchor),
        (
            "integer_r",
            args.integer_r,
            EXACT_REDUCTION_EQUATION_ID,
            build_integer_r_backend_anchor,
        ),
    )
    rows: list[dict[str, object]] = []
    for name, path, equation_id, builder in specs:
        if not path.is_file():
            rows.append({"name": name, "status": "OWED", "path": str(path)})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            anchor = builder(path, payload, repo=REPO)
        except (ValueError, KeyError, TypeError) as exc:
            rows.append(
                {
                    "name": name,
                    "status": "INCOMPLETE_OR_INVALID",
                    "path": str(path),
                    "error": str(exc),
                }
            )
            continue
        equation = get_equation_by_id(equation_id)
        if equation is None:
            rows.append(
                {
                    "name": name,
                    "status": "EQUATION_OWED",
                    "equation_id": equation_id,
                }
            )
            continue
        duplicate = any(row.anchor_id == anchor.anchor_id for row in equation.empirical_anchors)
        status = "ALREADY_REGISTERED" if duplicate else "READY"
        if args.write and not duplicate:
            update_equation_with_empirical_anchor(
                equation_id,
                anchor,
                agent="codex",
                subagent_id="throughput_authority_ladder",
                notes="Task #494 authority-ladder receipt; MEANS; pointer unmoved",
            )
            status = "REGISTERED"
        rows.append(
            {
                "name": name,
                "status": status,
                "equation_id": equation_id,
                "anchor_id": anchor.anchor_id,
                "source_artifact": anchor.source_artifact,
            }
        )
    print(json.dumps({"write": args.write, "rows": rows}, indent=2, sort_keys=True))
    bad = {"OWED", "INCOMPLETE_OR_INVALID", "EQUATION_OWED"}
    return 2 if args.require_all and any(row["status"] in bad for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
