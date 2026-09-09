"""Append ddm_ane3's fixed-output anchor to the scorer fp16 drift equation.

Run after the receipt memo exists so its post-edit SHA-256 can be embedded:

    .venv/bin/python tools/ddm_ane3_register_equation_anchor.py \
      --memo-sha256 <sha256> --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.canonical_equations import EmpiricalAnchor  # noqa: E402
from tac.canonical_equations.registry import (  # noqa: E402
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar  # noqa: E402

EQUATION_ID = "scorer_fp16_drift_by_axis_v1"
MEMO = ".omx/research/ddm_ane3_residue_fixed_conversion_remeasure_20260908.md"
ITEM1 = Path("/Volumes/APDataStore/pact/ddm_ane3_residue/item1")
PREDECESSOR = Path("/Volumes/VertigoDataTier/pact/ddm_ane2_precision/stage8")
ANCHOR_ID = "ane3_segnet_fixed_fp32_output_cpu_fallback_n600_20260909"


def build_anchor(memo_sha256: str) -> EmpiricalAnchor:
    fixed_report = json.loads((ITEM1 / "g13stem_n600.json").read_text())
    fixed = fixed_report["rows"][0]
    provisional = json.loads(
        (PREDECESSOR / "targeted_segnet_generated_n600.json").read_text()
    )
    previous = next(row for row in provisional["rows"] if row["label"] == "g13stem")
    old_rate = float(previous["fidelity"]["flip_rate"])
    new_rate = float(fixed["fidelity"]["flip_rate"])
    placement = fixed["placement"]["CPU_AND_NE"]
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "repeat on a native host where E5RT can populate its cache and MLComputePlan "
            f"proves at least 75% ANE placement (memo sha256: {memo_sha256})"
        ),
        measurement_axis="[macOS-CPU/CoreML advisory; requested CPU_AND_NE, measured CPU fallback]",
        hardware_substrate="m5_max_macos_26_4_coremltools_9_0_managed_shell",
    )
    return EmpiricalAnchor(
        anchor_id=ANCHOR_ID,
        measurement_utc="2026-09-09T13:51:00Z",
        inputs={
            "architecture_class": "segnet_smp_unet_efficientnet_b2",
            "backend": "coreml_mixed_fp16_cpu_fallback",
            "route": "g13stem",
            "pairs": int(fixed["fidelity"]["pairs"]),
            "reading_axis": "argmax_disagreement_rate",
            "model_output_dtype": "float32 (Core ML id 65568)",
            "compute_units_requested": "CPU_AND_NE",
            "ane_op_fraction": float(placement["ane_op_fraction"]),
            "axis_bar": float(fixed_report["seg_bar"]),
        },
        predicted_output={
            "flip_rate_upper": float(fixed_report["seg_bar"]),
            "flip_rate_lower": 0.5 * old_rate,
            "prior": "fixed output removes rather than adds argmax flips",
        },
        empirical_output={
            "flip_rate": new_rate,
            "flips": int(fixed["fidelity"]["flips"]),
            "total_px": int(fixed["fidelity"]["total_px"]),
            "prior_rate_with_fp16_output": old_rate,
            "rate_ratio_fixed_over_prior": new_rate / old_rate,
            "ane_op_fraction": float(placement["ane_op_fraction"]),
            "placement_devices": placement["ops_by_device"],
            "registration_gate_passed": False,
        },
        residual=abs(new_rate - old_rate) / old_rate,
        source_artifact=str(ITEM1 / "g13stem_n600.json"),
        measurement_method=(
            "fixed-fp32-output Core ML graph on generated decode n600 vs retained "
            "cpu_torch fp32 argmax; per-pair ledger retained; MLComputePlan measured CPU fallback"
        ),
        provenance=provenance,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memo-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    anchor = build_anchor(args.memo_sha256)
    print(json.dumps(anchor.to_dict(), indent=2, sort_keys=True))
    if args.apply:
        update_equation_with_empirical_anchor(
            EQUATION_ID,
            anchor,
            agent="codex",
            subagent_id="ddm_ane3",
            notes="ddm_ane3 fixed-output n600 row; CPU fallback blocks ANE registration",
        )
        print(f"appended {ANCHOR_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
