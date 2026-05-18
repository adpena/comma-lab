# SPDX-License-Identifier: MIT
"""Probe-disambiguator for WZ pipeline-stage codec primitive composition alpha.

Per CLAUDE.md "Subagent coherence-by-default" Hook #6 (probe-disambiguator) +
Catalog #125 wire-in #6 + lane
``lane_wyner_ziv_pipeline_stage_codec_primitive_20260517``.

Status: DEFERRED-PENDING-EMPIRICAL-EVIDENCE

The two defensible interpretations of composition_alpha between a WZ stage
and an adjacent pipeline stage are:

A. **Quantizer-shifts-correlation-toward-side-info**: when the upstream
   stage is a quantizer, it concentrates the byte distribution so the WZ
   side-info Y has MORE structural overlap with the residual. Predicts
   higher alpha (more additive composition).

B. **Transform-shifts-correlation-away-from-side-info**: when the upstream
   stage is a transform coder (DCT / wavelet), it decorrelates the byte
   distribution so Y has LESS structural overlap with the residual.
   Predicts lower alpha (more saturating composition).

Per CLAUDE.md "Subagent coherence-by-default" wire-in hook #6 + Catalog
#296 (predicted-band Dykstra feasibility): a probe-disambiguator is built
when 2+ defensible interpretations exist. Both interpretations above are
PRIOR-LEVEL hypotheses; they require a paired empirical anchor to
disambiguate. This stub exists as the canonical entry point for that
future empirical adjudication.

The canonical empirical disambiguation when funded:

1. For each upstream stage class S ∈ {quantizer, transform, predictor,
   identity}:
   a. Build a synthetic pipeline ``S → WZ(side="torch_defaults") → entropy``
   b. Measure observed alpha = max(0, 1 - savings_with_WZ / savings_without_WZ)
   c. Compare against the heuristic prior from
      ``tac.codec.wyner_ziv_layer.estimate_composition_alpha``.

2. If observed alpha agrees with hypothesis A → quantizer-stages SHIFT-TOWARD;
   if it agrees with B → transform-stages SHIFT-AWAY; if neither agrees →
   the prior is mis-specified and the composition matrix at Catalog #227
   needs a per-stage-class refinement.

Until the empirical anchor lands, this stub returns a structured DEFERRED
verdict that downstream consumers (autopilot ranker, Lagrangian planner)
must propagate as ``measured_config_status=composition_alpha_prior_only``
per CLAUDE.md "Apples-to-apples evidence discipline".
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


DISAMBIGUATOR_SCHEMA_VERSION = "wyner_ziv_composition_alpha_disambiguator_v1"


def emit_deferred_verdict() -> dict[str, Any]:
    return {
        "schema_version": DISAMBIGUATOR_SCHEMA_VERSION,
        "verdict": "DEFERRED-PENDING-EMPIRICAL-EVIDENCE",
        "hypothesis_a_quantizer_shifts_toward_side_info": (
            "WZ + upstream quantizer → alpha closer to 1.0 (additive)"
        ),
        "hypothesis_b_transform_shifts_away_from_side_info": (
            "WZ + upstream transform coder → alpha closer to 0.3 (saturating)"
        ),
        "reactivation_criteria": (
            "Run paired empirical smoke per upstream-stage class "
            "(quantizer/transform/predictor/identity) on fp16 state_dict "
            "fixture; compare observed alpha to the prior from "
            "tac.codec.wyner_ziv_layer.estimate_composition_alpha"
        ),
        "lane_id": "lane_wyner_ziv_pipeline_stage_codec_primitive_20260517",
        "downstream_consumer_obligation": (
            "Propagate measured_config_status=composition_alpha_prior_only "
            "until the paired anchor lands."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WZ composition_alpha probe-disambiguator (deferred stub).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON verdict to stdout."
    )
    args = parser.parse_args(argv)
    verdict = emit_deferred_verdict()
    if args.json:
        json.dump(verdict, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write("[wyner_ziv_composition_alpha_disambiguator]\n")
        for k, v in verdict.items():
            sys.stdout.write(f"  {k}: {v}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
