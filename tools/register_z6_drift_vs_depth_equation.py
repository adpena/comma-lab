# SPDX-License-Identifier: MIT
"""One-shot registration of canonical equation `mlx_pytorch_drift_vs_training_depth_z6_v1`.

Per CLAUDE.md "Canonical equations + models registry" non-negotiable + Catalog #344
(check_empirical_finding_memo_references_canonical_equation) + DRIFT-VS-DEPTH-CHAR-D-Z6
charter 2026-05-26: this script formalizes the empirical drift-vs-training-depth
trajectory measured on D=Z6 (substrate-specific FIRST anchor; future MLX-first doctrine
extension lifts this to substrate-agnostic when sister substrates land more anchors).

Empirical anchors (all measured on Z6PCWM1-grammar archives via Sister #1265 gate
``tools/gate_mlx_candidate_contest_equivalence_z6.py``; sigmoid [0,1] decoder output space;
M5 Max Apple Silicon; 50 pairs @ 48×64 resolution):

  epochs  drift_max_abs  ratio_vs_pr95
     300       0.000253          23.0×
     500       0.000358          32.6×
    1000       0.000458          41.6×
    2000       0.000721          65.5×
    3000       0.000725          65.9×

Power-law fit: drift = 1.81e-5 * epochs^0.47 (R² = 0.971)
Extrapolation: drift crosses 0.001 gate threshold at ~4973 epochs.

Per CLAUDE.md "MLX portable-local-substrate authority": every anchor remains
`[macOS-MLX research-signal]` and carries non-promotable markers until paired
contest CPU/CUDA evidence exists. The equation IS planner/observability-only;
the gate threshold parameterization recommendation surfaces as the operator-routable
verdict in the landing memo (not in this script).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
    register_canonical_equation,
    update_equation_with_empirical_anchor,
    get_equation_by_id,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "mlx_pytorch_drift_vs_training_depth_z6_v1"
GATE_THRESHOLD = 0.001  # Sister #1265 decoder-parity threshold (sigmoid [0,1] space)

# Power-law fit coefficients from 5-anchor regression (300/500/1000/2000/3000 epochs).
POWER_LAW_A = 1.8105e-05
POWER_LAW_B = 0.4713
POWER_LAW_R_SQUARED = 0.9713

# 5 anchors per ground-truth measurement
ANCHORS = [
    {
        "epochs": 300,
        "drift": 0.00025318562984466553,
        "archive_sha": "dabdcf94c44092c1ce075639c53391973c67abbad8e59fa12c4b4ebe8fc447ad",
        "archive_path": "experiments/results/z6_l2_canonical_LONGTRAIN_20260526T123600Z/0.bin",
        "gate_path": "experiments/results/z6_l2_canonical_LONGTRAIN_20260526T123600Z/gate_1265_verdict.json",
        "wall_seconds": 3.79,
        "ema_drift_l2": 10.120,
        "loss_initial": 0.3382,
        "loss_final": 0.1144,
    },
    {
        "epochs": 500,
        "drift": 0.000358,
        "archive_sha": "8eef1dff289d7704cb0ba01001bee096f503a0c82220b05cd4d9a7567a021d0f",
        "archive_path": "experiments/results/z6_drift_vs_depth_500ep_20260526T124730Z/0.bin",
        "gate_path": "experiments/results/z6_drift_vs_depth_500ep_20260526T124730Z/gate_1265_verdict.json",
        "wall_seconds": 5.78,
        "ema_drift_l2": 9.249,
        "loss_initial": 0.3382,
        "loss_final": 0.1020,
    },
    {
        "epochs": 1000,
        "drift": 0.000458,
        "archive_sha": "6442f96301a4b2cb",
        "archive_path": "experiments/results/z6_drift_vs_depth_1000ep_20260526T124750Z/0.bin",
        "gate_path": "experiments/results/z6_drift_vs_depth_1000ep_20260526T124750Z/gate_1265_verdict.json",
        "wall_seconds": 11.4,
        "ema_drift_l2": 5.611,
        "loss_initial": 0.3382,
        "loss_final": 0.0958,
    },
    {
        "epochs": 2000,
        "drift": 0.000721,
        "archive_sha": "822f0a1ecb2b8dc9",
        "archive_path": "experiments/results/z6_drift_vs_depth_2000ep_20260526T124753Z/0.bin",
        "gate_path": "experiments/results/z6_drift_vs_depth_2000ep_20260526T124753Z/gate_1265_verdict.json",
        "wall_seconds": 22.0,
        "ema_drift_l2": 3.840,
        "loss_initial": 0.3382,
        "loss_final": 0.0789,
    },
    {
        "epochs": 3000,
        "drift": 0.000725,
        "archive_sha": "fbe405e08e651743",
        "archive_path": "experiments/results/z6_drift_vs_depth_3000ep_20260526T124756Z/0.bin",
        "gate_path": "experiments/results/z6_drift_vs_depth_3000ep_20260526T124756Z/gate_1265_verdict.json",
        "wall_seconds": 34.2,
        "ema_drift_l2": 2.703,
        "loss_initial": 0.3382,
        "loss_final": 0.0793,
    },
]


def predict_drift_for_epochs(epochs: int) -> float:
    """Predict MLX↔PyTorch decoder drift for a given Z6 training epoch count.

    Power-law model fit empirically from 5 anchors (300/500/1000/2000/3000):
        drift = A * epochs^B with A = 1.81e-5, B = 0.471 (R² = 0.971).

    Returns predicted max_abs drift in sigmoid [0,1] decoder output space.

    Per CLAUDE.md "MLX portable-local-substrate authority": this prediction is
    `[macOS-MLX research-signal]` non-promotable; the contest-CUDA score
    propagation requires paired Linux x86_64 + NVIDIA evidence.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    return POWER_LAW_A * (epochs ** POWER_LAW_B)


def predict_threshold_crossing_epoch(threshold: float = GATE_THRESHOLD) -> int:
    """Predict the training-depth at which drift crosses the given gate threshold.

    Inverts drift = A * epochs^B to solve for epochs:
        epochs = (threshold / A) ** (1/B)
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    return int(math.ceil((threshold / POWER_LAW_A) ** (1.0 / POWER_LAW_B)))


def _sha256_json(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    captured_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build canonical Provenance for the equation itself (PREDICTED model).
    eq_inputs_sha = _sha256_json(
        {
            "substrate_id": "time_traveler_l5_z6",
            "grammar": "Z6PCWM1",
            "n_anchors": len(ANCHORS),
            "anchor_epoch_set": [a["epochs"] for a in ANCHORS],
            "power_law_A": POWER_LAW_A,
            "power_law_B": POWER_LAW_B,
        }
    )
    equation_provenance = build_provenance_for_predicted(
        model_id=EQUATION_ID,
        inputs_sha256=eq_inputs_sha,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="darwin_arm64_apple_silicon_m5_max",
        captured_at_utc=captured_at,
    )

    # Build first anchor (will register equation with it; remaining anchors appended).
    def _build_anchor(payload: dict, idx: int) -> EmpiricalAnchor:
        epochs = payload["epochs"]
        empirical_drift = payload["drift"]
        predicted_drift = predict_drift_for_epochs(epochs)
        residual_normalized = abs(empirical_drift - predicted_drift) / max(
            abs(empirical_drift), abs(predicted_drift)
        )
        # Provenance for each anchor's evidence file (gate verdict JSON)
        anchor_prov = build_provenance_for_research_sidecar(
            sidecar_path=payload["gate_path"],
            reactivation_criteria=(
                "rerun on paired contest CPU/CUDA (T4/A10G) for D=Z6 substrate "
                "before promotion or rank/kill use; canonical authority requires "
                "1:1 contest-compliant hardware per CLAUDE.md 'Submission auth "
                "eval — BOTH CPU AND CUDA'"
            ),
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="darwin_arm64_apple_silicon_m5_max",
            captured_at_utc=captured_at,
        )
        return EmpiricalAnchor(
            anchor_id=f"z6_l2_drift_vs_depth_{epochs}ep_{captured_at.replace('-','').replace(':','').replace('Z','')}",
            measurement_utc=captured_at,
            inputs={
                "epochs": epochs,
                "n_pairs": 50,
                "resolution": "48x64",
                "substrate_id": "time_traveler_l5_z6",
                "grammar": "Z6PCWM1",
                "archive_sha256": payload["archive_sha"],
                "archive_path": payload["archive_path"],
                "wall_seconds": payload["wall_seconds"],
                "ema_drift_l2_final": payload["ema_drift_l2"],
                "loss_initial": payload["loss_initial"],
                "loss_final": payload["loss_final"],
            },
            predicted_output={
                "drift_max_abs_sigmoid_space": predicted_drift,
                "verdict_below_gate_threshold_0_001": predicted_drift < GATE_THRESHOLD,
            },
            empirical_output={
                "drift_max_abs_sigmoid_space": empirical_drift,
                "verdict": "PASS",
                "ratio_vs_pr95_anchor_0_000011": empirical_drift / 0.000011,
                "gate_threshold": GATE_THRESHOLD,
            },
            residual=residual_normalized,
            source_artifact=payload["gate_path"],
            measurement_method="sister_1265_z6pcwm1_grammar_mlx_pytorch_decoder_parity_gate",
            provenance=anchor_prov,
        )

    first_anchor = _build_anchor(ANCHORS[0], 0)

    equation = CanonicalEquation(
        equation_id=EQUATION_ID,
        name="MLX/PyTorch decoder parity drift vs training-depth for D=Z6",
        one_line_summary=(
            "For D=Z6 substrate (Z6PCWM1 grammar), MLX↔PyTorch decoder-parity "
            "drift grows as drift = 1.81e-5 * epochs^0.47 (sigmoid [0,1] space); "
            "Sister #1265 gate threshold 0.001 crosses at ~4973 epochs."
        ),
        latex_form=(
            r"\delta_{\mathrm{MLX}\leftrightarrow\mathrm{PyTorch}}(E) = "
            r"A \cdot E^{B} \;\;\text{where}\;\; A = 1.81 \times 10^{-5}, "
            r"\; B = 0.471"
        ),
        python_callable_module_path=(
            "tools.register_z6_drift_vs_depth_equation:predict_drift_for_epochs"
        ),
        domain_of_validity={
            "substrate_family": "D=Z6 predictive-coding-world-model",
            "grammar": "Z6PCWM1",
            "framework_pair": "MLX vs PyTorch",
            "measurement_axis": "[macOS-MLX research-signal]",
            "hardware_substrate": "darwin_arm64_apple_silicon_m5_max",
            "training_depth_range_anchored": [300, 3000],
            "resolution": "48x64",
            "n_pairs": 50,
            "promotion_authority": False,
            "requires_paired_contest_cpu_cuda_for_promotion": True,
            "scope_note": (
                "Substrate-specific FIRST anchor; sister substrates need their own "
                "drift-vs-depth canonical equations until cross-substrate trend "
                "evidence permits lifting to substrate-agnostic v2."
            ),
        },
        units_in={
            "epochs": "training_epoch_count",
        },
        units_out={
            "drift_max_abs_sigmoid_space": "float_sigmoid_0_to_1_units",
            "predicted_threshold_crossing_epochs": "training_epoch_count",
        },
        empirical_anchors=(first_anchor,),
        predicted_vs_empirical_residual={
            f"epochs_{ANCHORS[0]['epochs']}": first_anchor.residual,
        },
        last_calibration_utc=captured_at,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.gate_mlx_candidate_contest_equivalence_z6",
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
        ),
        canonical_producers=(
            "experiments.train_substrate_z6_predictive_coding_mlx_l2",
            "tools.register_z6_drift_vs_depth_equation",
        ),
        provenance=equation_provenance,
    )

    # Check if equation already registered
    existing = get_equation_by_id(EQUATION_ID)
    if existing is not None:
        print(f"[register-z6-drift-vs-depth] Equation {EQUATION_ID} already registered; skipping initial register, only appending remaining anchors")
    else:
        print(f"[register-z6-drift-vs-depth] Registering NEW equation {EQUATION_ID}")
        register_canonical_equation(
            equation,
            subagent_id="drift_vs_depth_char_d_z6",
            notes=(
                f"DRIFT-VS-DEPTH-CHAR-D-Z6 charter landing; 5 anchors "
                f"(300/500/1000/2000/3000 epochs); power-law fit A={POWER_LAW_A:.4e} "
                f"B={POWER_LAW_B:.4f} R²={POWER_LAW_R_SQUARED:.4f}; "
                f"extrapolated 0.001 gate threshold crossing at ~"
                f"{predict_threshold_crossing_epoch():d} epochs."
            ),
        )
        print(f"[register-z6-drift-vs-depth] Registered with anchor 1 (300ep)")

    # Append remaining 4 anchors (500, 1000, 2000, 3000)
    for idx in range(1, len(ANCHORS)):
        anchor = _build_anchor(ANCHORS[idx], idx)
        update_equation_with_empirical_anchor(
            EQUATION_ID,
            anchor,
            subagent_id="drift_vs_depth_char_d_z6",
            notes=f"DRIFT-VS-DEPTH anchor {idx+1}/5: {ANCHORS[idx]['epochs']}ep",
        )
        print(
            f"[register-z6-drift-vs-depth] Appended anchor {idx+1}/{len(ANCHORS)}: "
            f"epochs={ANCHORS[idx]['epochs']} drift={ANCHORS[idx]['drift']:.6f} "
            f"residual_vs_powerlaw={anchor.residual:.4f}"
        )

    print()
    print(
        f"[register-z6-drift-vs-depth] DONE registration: {EQUATION_ID} with "
        f"{len(ANCHORS)} empirical anchors"
    )
    print(
        f"[register-z6-drift-vs-depth] Power-law fit: drift = {POWER_LAW_A:.4e} * "
        f"epochs^{POWER_LAW_B:.4f} (R²={POWER_LAW_R_SQUARED:.4f})"
    )
    print(
        f"[register-z6-drift-vs-depth] Extrapolated threshold-crossing: "
        f"~{predict_threshold_crossing_epoch():d} epochs (gate threshold {GATE_THRESHOLD})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
