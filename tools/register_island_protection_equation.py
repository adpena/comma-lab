# SPDX-License-Identifier: MIT
"""Register the canonical equation for the islands-protection stack (triality: EQUATIONS leg).

Reads the latest island-survival smoke JSON (frozen CPU-torch SegNet, real n600 GT)
and registers ``island_finest_scale_protection_survival_v1`` with an EmpiricalAnchor
(predicted DESIGN gains vs the measured n600 recall gains). Idempotent (append-only
'registered' event keyed by equation_id).

Usage:
    .venv/bin/python tools/register_island_protection_equation.py \
        --result experiments/results/island_protection_survival_n600_*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.equation import (  # noqa: E402
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402
from tac.provenance.builders import (  # noqa: E402
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

# DESIGN prediction (the qualitative law the equation codifies, pre-measurement):
# a sparse GT-appearance seed on the self-detected island band RECOVERS a substantial
# fraction of the finest-scale erasure recall loss through the frozen scorer, and
# freezing the protected pixels RETAINS recall through the bulk wash. Lower bounds:
PRED_SEED_BIRTH_GAIN = 0.30      # design lower bound on (seeded - erased) island recall
PRED_CONTAINMENT_GAIN = 0.05     # design lower bound on (contained - uncontained) island recall


def _latest_result(pattern: str) -> Path:
    hits = sorted(glob.glob(str(REPO / pattern)) + glob.glob(pattern))
    # prefer n600
    n600 = [h for h in hits if "_n600_" in h]
    chosen = (n600 or hits)[-1]
    return Path(chosen)


def build_equation(result: dict, result_path: str) -> CanonicalEquation:
    summ = result["summary"]
    n = int(summ.get("n", 0))
    # Anchor on the BINDING island = the MOST-ERASED one (lowest erased recall). A thick,
    # barely-erased island (movable erased ~0.91) has a ceiling on its possible gain and is
    # NOT the design test case; the finest-scale erasure tail (lane erased ~0.56) is. Report
    # per-island so the ceiling effect is visible; the headline residual is the binding island.
    per_island = {}
    binding_kind, binding_erased = None, 2.0
    for kind in ("lane", "movable"):
        if kind in summ:
            s = summ[kind]
            per_island[kind] = {"erased": s["erased"], "seeded": s["seeded"],
                                "seed_birth_gain": s["seed_birth_gain"],
                                "wash_uncontained": s["wash_uncontained"],
                                "wash_contained": s["wash_contained"],
                                "containment_gain": s["containment_gain"]}
            if s["erased"] < binding_erased:
                binding_erased = s["erased"]
                binding_kind = kind
    b = summ.get(binding_kind, {}) if binding_kind else {}
    emp_seed = float(b.get("seed_birth_gain", 0.0))            # binding-island (most-erased) gain
    emp_contain = float(b.get("containment_gain", 0.0))
    # residual = normalized how far the DESIGN lower bound was from the measured (0 if measured >= bound)
    res_seed = max(0.0, (PRED_SEED_BIRTH_GAIN - emp_seed) / max(PRED_SEED_BIRTH_GAIN, 1e-6))
    res_contain = max(0.0, (PRED_CONTAINMENT_GAIN - emp_contain) / max(PRED_CONTAINMENT_GAIN, 1e-6))
    residual = float(0.5 * (res_seed + res_contain))

    anchor = EmpiricalAnchor(
        anchor_id=f"island_survival_n{n}_frozen_cpu_torch_segnet",
        measurement_utc=result.get("utc", "2026-07-02T00:00:00Z"),
        inputs={"n_pairs": n, "erase_factor": summ.get("erase_factor"),
                "wash_k": summ.get("wash_k"), "dilate_px": summ.get("dilate_px"),
                "lane_cls": result["island_detect"]["lane_cls"],
                "movable_cls": result["island_detect"]["movable_cls"]},
        predicted_output={"seed_birth_gain_lower_bound": PRED_SEED_BIRTH_GAIN,
                          "containment_gain_lower_bound": PRED_CONTAINMENT_GAIN},
        empirical_output={"binding_island": binding_kind, "binding_island_erased_recall": binding_erased,
                          "binding_seed_birth_gain": emp_seed,
                          "binding_containment_gain": emp_contain,
                          "per_island": per_island},
        residual=residual,
        source_artifact=result_path,
        measurement_method="frozen_cpu_torch_segnet_argmax_island_recall_erasure_recovery",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=result_path,
            reactivation_criteria="byte-closed upstream/evaluate.py exact row through the wired witness",
            measurement_axis="[contest-CPU advisory]",
            hardware_substrate="m5_max_cpu_torch",
        ),
    )
    return CanonicalEquation(
        equation_id="island_finest_scale_protection_survival_v1",
        name="Finest-scale island protection (early-seed + containment) survival through the frozen scorer",
        one_line_summary=(
            "A sparse GT-appearance seed on the self-detected island band births the erased island "
            "class through the frozen SegNet; freezing the protected pixels retains it under bulk wash."
        ),
        latex_form=(
            r"\mathrm{recall}_{\mathrm{seeded}} \ge \mathrm{recall}_{\mathrm{erased}} + \Delta_{\mathrm{birth}}"
            r";\; \mathrm{recall}_{\mathrm{contained}} \ge \mathrm{recall}_{\mathrm{washed}} + \Delta_{\mathrm{contain}}"
        ),
        python_callable_module_path="tac.boundary_math.island_protection:island_birth_term_np",
        domain_of_validity={"scorer": ["frozen_cpu_torch_segnet"], "islands": ["lane", "movable"],
                            "erasure_model": ["blockmean_lowpass_surrogate"], "n_pairs_range": [1, 600]},
        units_in={"seg_logits": "logit", "island_mask": "bool", "margin_target": "logit"},
        units_out={"birth_penalty": "logit_deficit", "island_recall_gain": "fraction"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"island_recall_gain": residual},
        last_calibration_utc=result.get("utc", "2026-07-02T00:00:00Z"),
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.boundary_math.island_protection",
            "experiments.train_witness_realized_through_R_mlx",
        ),
        canonical_producers=(
            "experiments/island_protection_survival_smoke.py",
            "tools/register_island_protection_equation.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="island_finest_scale_protection_survival.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        ),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default="experiments/results/island_protection_survival_n*.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    rp = _latest_result(args.result)
    result = json.loads(rp.read_text())
    eq = build_equation(result, str(rp))
    eo = eq.empirical_anchors[0].empirical_output
    print(json.dumps({"equation_id": eq.equation_id, "anchor": eq.empirical_anchors[0].anchor_id,
                      "residual": eq.empirical_anchors[0].residual,
                      "binding_island": eo.get("binding_island"),
                      "binding_seed_birth_gain": eo.get("binding_seed_birth_gain"),
                      "binding_containment_gain": eo.get("binding_containment_gain"),
                      "result": str(rp)}, indent=2))
    if not args.dry_run:
        register_canonical_equation(eq, subagent_id="early-seed-containment-amplification",
                                    notes="FEED-lz islands-protection stack; n600 frozen-CPU-torch survival anchor")
        print(json.dumps({"stage": "registered", "equation_id": eq.equation_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
