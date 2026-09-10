"""Register the retained TC2 fixed-family codelength result, never a byte price."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_tc2_lane_context_map")
EQUATION = "lane_boundary_context_map_bound_v1"


def lane_boundary_context_map_bound(base_bits, loss_nats, gradient, weights):
    """Numerical convex tangent bound for the specified continuous five-weight box.

    All-symbol base_bits and retained-symbol loss give omitted symbols their full
    baseline credit. This neither bounds arbitrary geometry nor integer bytes.
    """
    gradient = np.asarray(gradient, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if gradient.shape != (5,) or weights.shape != (5,):
        raise ValueError("exactly five fixed-base coefficients required")
    if not np.isfinite([base_bits, loss_nats]).all() or base_bits < 0 or loss_nats < 0:
        raise ValueError("finite nonnegative losses required")
    if not np.isfinite(gradient).all() or not np.isfinite(weights).all():
        raise ValueError("finite gradient and weights required")
    if np.any(weights < -4) or np.any(weights > 127 / 32):
        raise ValueError("weights outside declared int8-scale continuous box")
    gap = float(np.sum(gradient * np.where(gradient >= 0, weights + 4, weights - 127 / 32)))
    return {
        "optimistic_gain_bytes": (base_bits - (loss_nats - gap) / math.log(2)) / 8,
        "gap_bits": gap / math.log(2),
        "universal_upper_bound": False,
        "integer_coder_bound": False,
        "score_claim": False,
    }


def register_locked():
    from experiments import ddm_jg2_tail_reencode as io
    from tac.canonical_equations import CanonicalEquation, EmpiricalAnchor, query_equations, register_canonical_equation
    from tac.provenance import build_provenance_for_macos_cpu_advisory

    pointer = (REPO / ".omx/state/canonical_frontier_pointer.json").read_bytes()
    io.persist_immutable_bytes(
        ROOT / "retained" / ("pointer_register_" + hashlib.sha256(pointer).hexdigest() + ".json"),
        pointer,
        label="live pointer at equation registration",
    )

    result_path = ROOT / "RESULT.json"
    result = json.loads(result_path.read_text())
    fit = json.loads((ROOT / "fit/RESULT.json").read_text())
    prep = json.loads((ROOT / "prepare/RESULT.json").read_text())
    if not result["full_n600"] or result["positions"] != 117964800 or result["score_claim"]:
        raise ValueError("registration needs actual full-n600 non-score result")
    for row in fit["rows"]:
        last = sorted((ROOT / "fit" / row["name"]).glob("ITER_*.json"))[-1]
        iteration = json.loads(last.read_text())
        bound = lane_boundary_context_map_bound(
            prep["totals"][0], iteration["loss_nats"], iteration["gradient"], iteration["weights"]
        )
        if abs(bound["optimistic_gain_bytes"] - row["optimistic_gain_bytes"]) > 1e-7:
            raise ValueError("registered evaluator disagrees with actual fit certificate")
    for eq in query_equations():
        if eq.equation_id == EQUATION:
            old = eq.to_dict()
            if not any(
                a.get("empirical_output", {}).get("codelength_result") == result for a in old["empirical_anchors"]
            ):
                raise ValueError("existing equation has different empirical custody")
            io.atomic_json(ROOT / "EQUATION_REGISTRATION.json", old)
            return
    stamp = datetime.now(UTC).isoformat()
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        source_path=str(result_path),
        captured_at_utc=stamp,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_tc2_move37_full_n600_20260910",
        measurement_utc=stamp,
        inputs={"binding": result["binding"], "symbols": 117964800, "baseline_bits": prep["totals"][0]},
        predicted_output={"charter_prior_oracle_gain_bytes": 15000, "prior_causal_encoded_gain_bytes": [3000, 8000]},
        empirical_output={
            "signed_difference_from_15KB_prior_bytes": result["reference_gain_bytes"] - 15000,
            "codelength_result": result,
            "fit_certificates": fit["rows"],
            "conditional_entropy": prep["conditional_entropy"],
        },
        residual=abs(result["reference_gain_bytes"] - 15000) / 15000,
        source_artifact=str(result_path),
        measurement_method="All n600 shipped-frequency rows; fixed-base online-ratio features; numerical convex certificate and full-symbol codelength totals. No encoded-byte or score claim.",
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    equation = CanonicalEquation(
        equation_id=EQUATION,
        name="Lane-context fixed-family numerical codelength bound",
        one_line_summary="Price a concrete extra context on pinned TC1 probabilities without treating GT geometry as a universal ceiling or ideal bits as charged bytes.",
        latex_form=r"\Delta L/8\leq [L_0-(F(w)-\sum_j g_j(w_j-e_j))/\ln2]/8",
        python_callable_module_path="experiments.ddm_tc2_result:lane_boundary_context_map_bound",
        domain_of_validity={
            "included": [
                "fixed shipped base probabilities",
                "fixed features",
                "five-weight box",
                "continuous numerical log-loss certificate",
            ],
            "excluded": [
                "joint 40-weight refit",
                "arbitrary lane geometry",
                "integer coder or archive byte bounds",
                "successor-field transfer",
            ],
        },
        units_in={"base_bits": "bits", "loss_nats": "nats", "gradient": "nats per weight", "weights": "dimensionless"},
        units_out={"optimistic_gain_bytes": "ideal bits divided by eight; not a price", "gap_bits": "bits"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={},
        last_calibration_utc=stamp,
        next_recalibration_trigger="when_operator_invokes_recalibrate_equation",
        canonical_producers=("experiments.ddm_tc2_lane_context.fit",),
        canonical_consumers=("experiments.ddm_tc2_result.register",),
        provenance=provenance,
    )
    register_canonical_equation(
        equation,
        agent="codex",
        subagent_id="ddm_tc2",
        notes="Named move37 field only; noncausal reference is not a universal upper bound.",
    )
    receipt = ROOT / "EQUATION_REGISTRATION.json"
    io.atomic_json(receipt, equation.to_dict())
    print(str(receipt))


def register():
    from tac.canonical_equations.registry import _registry_lock, load_equation_registry_strict

    with _registry_lock():
        load_equation_registry_strict()
        return register_locked()


if __name__ == "__main__":
    register()
