#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reweight a fec6 selector-discovery artifact by sensitivity-map axis weights.

This is the operator-facing CLI for Ext 3 of the fec6 stacking wave
(``lane_fec6_stacking_wave_5_grammar_extensions_20260517``). It reads an
existing ``selector_policy_sample.json`` + per-pair candidate-mode table,
applies per-pair axis-weight reweighting via
``tac.fec6_selector_discovery_sensitivity_weighted``, and emits a NEW
artifact suitable for consumption by
``tools/build_pr101_frame_exploit_selector_packet.py``.

This tool does NOT modify the existing fec6 builder. It writes a sister
artifact and the operator can point ``--artifact-dir`` at the new
artifact to build a sensitivity-weighted fec6 archive variant.

Design memo: ``.omx/research/fec6_plus_sensitivity_weighted_discovery_design_20260517.md``

Per CLAUDE.md "Apples-to-apples evidence discipline": the output
artifact carries the ``axis_weights.evidence_tag()`` in its metadata so
downstream tooling can trace the operating-point assumption.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": the
output artifact has ``score_claim=false`` and
``evidence_tag="[predicted, theoretical]"``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.fec6_selector_discovery_sensitivity_weighted import (  # noqa: E402
    reweight_per_pair_candidate_table,
)
from tac.sensitivity_map.axis_weights import (  # noqa: E402
    PR106_R2_FRONTIER_AXIS_WEIGHTS,
    axis_weights_for_named_operating_point,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reweight a fec6 selector-discovery artifact by sensitivity-map "
            "axis weights (Ext 3 of fec6 stacking wave)."
        )
    )
    parser.add_argument(
        "--candidate-modes-json",
        type=Path,
        required=True,
        help=(
            "Path to JSON file with per-pair candidate mode tables. "
            "Format: list of N_PAIRS entries, each a list of K mode dicts "
            "with keys 'delta_d_pose' and 'delta_d_seg'."
        ),
    )
    parser.add_argument(
        "--operating-point",
        type=str,
        default="pr106_r2_frontier",
        help=(
            "Named operating point for axis weights (default: 'pr106_r2_frontier'). "
            "Resolved via tac.sensitivity_map.axis_weights.axis_weights_for_named_operating_point."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write the reweighted artifact JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print divergence count summary to stdout.",
    )
    return parser.parse_args(argv)


def reweight_candidate_modes_file(
    *,
    candidate_modes_json: Path,
    operating_point: str,
    output_json: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Reweight a per-pair candidate-modes JSON file and write the result.

    Returns the result dict (also written to ``output_json``).
    """
    if not candidate_modes_json.is_file():
        raise FileNotFoundError(f"--candidate-modes-json not found: {candidate_modes_json}")

    raw = json.loads(candidate_modes_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"{candidate_modes_json}: expected top-level list of N_PAIRS entries; got {type(raw)}"
        )

    if operating_point == "pr106_r2_frontier":
        # Use the canonical PR106 r2 frontier axis weights directly per
        # CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent".
        axis_weights = PR106_R2_FRONTIER_AXIS_WEIGHTS
    else:
        axis_weights = axis_weights_for_named_operating_point(operating_point)

    result = reweight_per_pair_candidate_table(
        per_pair_candidate_modes=raw,
        axis_weights=axis_weights,
    )

    # Augment with provenance fields per Catalog #305 observability
    # surface contract: cite-able + diff-able across runs.
    result["source_artifact"] = str(candidate_modes_json)
    result["operating_point"] = operating_point
    result["promotion_eligible"] = False  # diagnostic-only per Catalog #192
    result["ready_for_exact_eval_dispatch"] = False
    result["lane_id"] = "lane_fec6_stacking_wave_5_grammar_extensions_20260517"
    result["design_memo"] = (
        ".omx/research/fec6_plus_sensitivity_weighted_discovery_design_20260517.md"
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    if verbose:
        sys.stdout.write(
            f"[reweight-fec6-selector-discovery] wrote {output_json}\n"
            f"  n_pairs={result['n_pairs']}\n"
            f"  per_pair_divergence_count={result['per_pair_divergence_count']}\n"
            f"  axis_weights_evidence_tag={result['axis_weights_evidence_tag']}\n"
        )

    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reweight_candidate_modes_file(
        candidate_modes_json=args.candidate_modes_json,
        operating_point=args.operating_point,
        output_json=args.output_json,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
