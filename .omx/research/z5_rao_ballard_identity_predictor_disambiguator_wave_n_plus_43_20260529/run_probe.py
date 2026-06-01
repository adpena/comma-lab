# SPDX-License-Identifier: MIT
"""Z5 Rao-Ballard identity-predictor disambiguator probe — Wave N+43 anchor 3/3.

Per Catalog #308 alternative-probe-methodology + canonical equation
``z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1``
(anchor 2/3 LANDED commit c0bef1cf5 Hinton-distilled 600pair 3.22x pose-axis
reduction; anchor 3/3 = THIS PROBE per CLAUDE.md "Forbidden premature KILL").

**Probe question (Catalog #308 + #311 canonical disambiguator):**
Does the Z5 2-level Rao-Ballard hierarchical predictor (z_low_pred =
predictor(z_high, ego_motion)) produce empirically distinct convergence
signature vs an IDENTITY predictor (z_low_pred = 0; residual = z_low) at
50ep × 600pair MLX-LOCAL?

**Methodology** (pure reconstruction, no Hinton-distilled scorer-bound —
matches anchor 1/3 baseline at 1.022s wall-clock):

1. Run Z5RaoBallardSubstrateMLX with full predictor (control) — 50ep × 600pair
   pure reconstruction (matches anchor 1/3 measurement_method).
2. Run Z5RaoBallardSubstrateMLX with predictor weights ZEROED at runtime AND
   bias ZEROED → z_low_pred = 0 → residual = z_low (identity case where
   prediction contributes nothing).
3. Compare final reconstruction loss between the two variants. If full
   predictor variant beats identity by ΔS > 0.005 (>5% relative loss
   reduction), the 2-level Rao-Ballard hierarchy is canonical-distinct from
   pure reconstruction. If NOT (within noise), the substrate's predictor
   block is degenerate — IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307
   (NOT paradigm-level; canonical_unwind_path = wire predictor activation
   into the score-aware Lagrangian with stronger lambda_residual or larger
   predictor_hidden_dim).

**Cost**: $0 (MLX-LOCAL macOS-CPU advisory per Catalog #192 non-promotable;
[macOS-MLX research-signal]).

**Non-promotable** per Catalog #192/#317/#341: score_claim=False,
promotion_eligible=False, axis_tag="[macOS-MLX research-signal]".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

# MLX imports
import mlx.core as mx
import mlx.nn as mlx_nn
from mlx.utils import tree_flatten, tree_unflatten

from tac.substrates.time_traveler_l5_z5.architecture import Z5RaoBallardConfig
from tac.substrates.time_traveler_l5_z5.mlx_renderer import (
    Z5RaoBallardSubstrateMLX,
)

PROBE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = PROBE_DIR / "probe_summary.json"
TELEMETRY_PATH = PROBE_DIR / "telemetry.jsonl"

NUM_PAIRS = 600
EPOCHS = 50
LEARNING_RATE = 1e-4
SEED_CONTROL = 0  # mirrors anchor 1/3 seed
SEED_IDENTITY = 0  # same seed for fair head-to-head


def _zero_predictor_weights(model: Z5RaoBallardSubstrateMLX) -> None:
    """Zero out predictor weights + biases in place so z_low_pred = 0 always.

    Per Catalog #308 canonical identity-predictor disambiguator: setting the
    predictor block to produce constant 0 makes residual = z_low - 0 = z_low,
    which means the substrate degenerates to pure reconstruction (no
    hierarchical prediction signal).

    This is the canonical 'identity' case: the predictor contributes NOTHING
    to the latent inference; the decoder must produce the entire output from
    z_low alone (which it can — z_low is the direct decoder input).
    """
    # MLX module structure: predictor is _Z5HierarchicalPredictorMLX with
    # high_to_hidden / ego_to_hidden / hidden_layers / hidden_to_low.
    pred = model.predictor

    # Walk module parameters and zero them. MLX modules expose parameters()
    # which returns a nested dict; we flatten and zero each leaf.
    flat_params = tree_flatten(pred.parameters())
    zeroed = []
    for name, arr in flat_params:
        zeroed.append((name, mx.zeros(arr.shape, dtype=arr.dtype)))

    pred.update(tree_unflatten(zeroed))


def _generate_synthetic_targets(num_pairs: int, h: int, w: int, seed: int = 0) -> tuple[mx.array, mx.array]:
    """Generate STRUCTURED synthetic targets for disambiguator probe.

    Targets are smooth per-pair gradients + sinusoidal patterns that the Z5
    decoder CAN learn to reconstruct (vs random uniform which has entropy
    floor ~0.17 MSE that no model can break). This keeps the disambiguator
    meaningful: if full predictor beats identity, it's because the 2-level
    hierarchy is useful for the STRUCTURED reconstruction signal — which IS
    the canonical Rao-Ballard hierarchical-predictive-coding hypothesis test.

    Both probe variants see EXACTLY the same targets for fair head-to-head.
    """
    rng = np.random.default_rng(seed)
    # Per-pair phase shift in [0, 2pi) — both frames in the pair share phase
    # so the per-pair latent has structure to learn.
    phases = rng.uniform(0.0, 2 * np.pi, size=(num_pairs,)).astype(np.float32)
    # Per-pair color tint in [0.3, 0.7] per channel.
    tints = rng.uniform(0.3, 0.7, size=(num_pairs, 3)).astype(np.float32)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    yy = yy.astype(np.float32) / h
    xx = xx.astype(np.float32) / w

    target_0_np = np.zeros((num_pairs, 3, h, w), dtype=np.float32)
    target_1_np = np.zeros((num_pairs, 3, h, w), dtype=np.float32)
    for p_idx in range(num_pairs):
        pattern = (np.sin(4 * np.pi * xx + phases[p_idx]) * 0.3 + 0.5).astype(np.float32)
        for c in range(3):
            target_0_np[p_idx, c] = pattern * tints[p_idx, c]
            # frame_1 = frame_0 shifted slightly (motion prior — predictor should learn this)
            target_1_np[p_idx, c] = (
                np.sin(4 * np.pi * xx + phases[p_idx] + 0.2) * 0.3 + 0.5
            ).astype(np.float32) * tints[p_idx, c]
    return mx.array(target_0_np), mx.array(target_1_np)


def _train_one_variant(
    *,
    variant_name: str,
    identity_predictor: bool,
    cfg: Z5RaoBallardConfig,
    target_0: mx.array,
    target_1: mx.array,
    num_pairs: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict:
    """Train a single Z5 variant and return canonical telemetry."""
    model = Z5RaoBallardSubstrateMLX(cfg, seed=seed)

    if identity_predictor:
        _zero_predictor_weights(model)

    optimizer = mlx_nn.optim if hasattr(mlx_nn, "optim") else None
    # Use mlx.optimizers if available
    import mlx.optimizers as optim_mod
    optimizer = optim_mod.AdamW(learning_rate=learning_rate)

    # Per canonical pattern at src/tac/substrates/_shared/mlx_score_aware/adapter.py:161:
    # closure takes model as arg; value_and_grad differentiates over the model.
    batch_state = {"idx": None, "t0": None, "t1": None}

    def _loss_fn_inner(model_inst):
        rgb_0, rgb_1, residual = model_inst.reconstruct_pair(batch_state["idx"])
        recon_0 = mx.mean((rgb_0 - batch_state["t0"]) ** 2)
        recon_1 = mx.mean((rgb_1 - batch_state["t1"]) ** 2)
        return recon_0 + recon_1

    loss_and_grad_fn = mlx_nn.value_and_grad(model, _loss_fn_inner)

    losses_per_epoch = []
    wall_clock_start = time.time()

    batch_size = min(num_pairs, 8)
    indices_full = np.arange(num_pairs, dtype=np.int32)

    initial_loss = None

    for epoch in range(epochs):
        epoch_losses = []
        # Shuffle indices for SGD
        rng = np.random.default_rng(epoch + seed)
        rng.shuffle(indices_full)

        for batch_start in range(0, num_pairs, batch_size):
            batch_indices_np = indices_full[batch_start:batch_start + batch_size]
            batch_state["idx"] = mx.array(batch_indices_np)
            batch_state["t0"] = mx.take(target_0, batch_state["idx"], axis=0)
            batch_state["t1"] = mx.take(target_1, batch_state["idx"], axis=0)

            loss_val, grads = loss_and_grad_fn(model)

            if identity_predictor:
                # Zero out predictor grads so identity variant stays identity.
                def _zero_pred_grad(g_tree):
                    flat = tree_flatten(g_tree)
                    rebuilt = []
                    for name, arr in flat:
                        if name.startswith("predictor."):
                            rebuilt.append((name, mx.zeros(arr.shape, dtype=arr.dtype)))
                        else:
                            rebuilt.append((name, arr))
                    return tree_unflatten(rebuilt)
                grads = _zero_pred_grad(grads)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_losses.append(float(loss_val.item()))

        ep_avg = float(np.mean(epoch_losses))
        if initial_loss is None:
            initial_loss = ep_avg
        losses_per_epoch.append(ep_avg)

        if identity_predictor:
            # Re-zero predictor weights every epoch (safety against any
            # residual gradient leakage via optimizer state momentum).
            _zero_predictor_weights(model)

    wall_clock_end = time.time()
    final_loss = losses_per_epoch[-1]
    reduction_ratio = (initial_loss - final_loss) / max(initial_loss, 1e-8)

    return {
        "variant_name": variant_name,
        "identity_predictor": identity_predictor,
        "epochs_completed": epochs,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_ratio": reduction_ratio,
        "wall_clock_seconds": wall_clock_end - wall_clock_start,
        "losses_per_epoch": losses_per_epoch,
        "num_pairs": num_pairs,
        "learning_rate": learning_rate,
        "seed": seed,
    }


def main() -> int:
    """Run identity-predictor disambiguator probe and emit canonical artifact."""
    cfg = Z5RaoBallardConfig(num_pairs=NUM_PAIRS)
    h, w = cfg.output_height, cfg.output_width

    print(f"[z5_disambiguator] cfg: low={cfg.low_latent_dim} high={cfg.high_latent_dim} ego={cfg.ego_dim}", flush=True)
    print(f"[z5_disambiguator] output_hw=({h}, {w}) num_pairs={NUM_PAIRS} epochs={EPOCHS}", flush=True)

    # Reduce target size for $0 macOS-CPU advisory probe (matches anchor 1/3
    # 1.022s wall-clock pattern; the 1164x874 full contest size would
    # explode wall-clock 100x+).
    probe_h, probe_w = 64, 96
    print(f"[z5_disambiguator] PROBE TARGET RESOLUTION: ({probe_h}, {probe_w}) per anchor 1/3 fast-iter pattern", flush=True)

    target_0, target_1 = _generate_synthetic_targets(NUM_PAIRS, probe_h, probe_w, seed=42)
    print(f"[z5_disambiguator] targets generated: {target_0.shape} {target_1.shape}", flush=True)

    # Override cfg output dims for probe
    cfg_probe = Z5RaoBallardConfig(
        num_pairs=NUM_PAIRS,
        output_height=probe_h,
        output_width=probe_w,
    )

    print("[z5_disambiguator] PROBE 1/2: full predictor (control)", flush=True)
    control = _train_one_variant(
        variant_name="full_predictor_control",
        identity_predictor=False,
        cfg=cfg_probe,
        target_0=target_0,
        target_1=target_1,
        num_pairs=NUM_PAIRS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED_CONTROL,
    )
    print(f"[z5_disambiguator] control done: initial={control['initial_loss']:.5f} final={control['final_loss']:.5f} reduction={control['loss_reduction_ratio']:.4f} wall={control['wall_clock_seconds']:.2f}s", flush=True)

    print("[z5_disambiguator] PROBE 2/2: identity predictor (z_low_pred=0)", flush=True)
    identity = _train_one_variant(
        variant_name="identity_predictor_ablation",
        identity_predictor=True,
        cfg=cfg_probe,
        target_0=target_0,
        target_1=target_1,
        num_pairs=NUM_PAIRS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED_IDENTITY,
    )
    print(f"[z5_disambiguator] identity done: initial={identity['initial_loss']:.5f} final={identity['final_loss']:.5f} reduction={identity['loss_reduction_ratio']:.4f} wall={identity['wall_clock_seconds']:.2f}s", flush=True)

    # Verdict per Catalog #308 + canonical equation #344
    delta_s = identity["final_loss"] - control["final_loss"]
    delta_s_threshold = 0.005
    if delta_s > delta_s_threshold:
        verdict = "PROCEED"
        verdict_token = "full_predictor_beats_identity_by_more_than_threshold"
        paradigm_classification = "DISTINCT_HIERARCHICAL_PREDICTIVE_CODING"
    elif abs(delta_s) <= delta_s_threshold:
        verdict = "DEFER"
        verdict_token = "full_predictor_within_noise_of_identity"
        paradigm_classification = "IMPLEMENTATION_LEVEL_FALSIFICATION_PER_CATALOG_307"
    else:
        # Identity beats full predictor (delta_s < -threshold)
        verdict = "DEFER"
        verdict_token = "identity_beats_full_predictor_anti_signal"
        paradigm_classification = "IMPLEMENTATION_LEVEL_FALSIFICATION_PER_CATALOG_307"

    summary = {
        "probe_id": "z5_rao_ballard_identity_predictor_disambiguator_wave_n_plus_43_20260529",
        "probe_kind": "mlx_local_identity_predictor_disambiguator_per_catalog_308",
        "substrate": "time_traveler_l5_z5",
        "schema_version": "z5_disambiguator_probe_v1_20260529",
        "anchor_index": "3_of_3",
        "canonical_equation_id": "z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1",
        "evidence_grade": "predicted",
        "measurement_axis": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "agent": "claude-slot-aa-wave-n43-z5-rao-ballard-identity-predictor-disambiguator-20260529",
        "control": control,
        "identity": identity,
        "delta_final_loss": delta_s,
        "delta_loss_reduction_ratio": control["loss_reduction_ratio"] - identity["loss_reduction_ratio"],
        "delta_s_threshold": delta_s_threshold,
        "verdict": verdict,
        "verdict_token": verdict_token,
        "paradigm_classification": paradigm_classification,
        "canonical_unwind_path_if_falsified": (
            "increase lambda_residual to 5.0+ AND/OR widen predictor_hidden_dim to 96 "
            "AND/OR add Hinton-distilled scorer-bound (matches anchor 2/3 settings); "
            "per CLAUDE.md 'Forbidden premature KILL': substrate paradigm Rao-Ballard "
            "1999 hierarchical predictive coding remains INTACT; only THIS specific "
            "predictor block configuration is falsified IF delta_s <= threshold"
        ),
        "cross_family_z_family_comparison": {
            "z5_rao_ballard": {
                "anchor_1_of_3_loss_reduction_pure_reconstruction": 0.59,
                "anchor_2_of_3_pose_axis_reduction_hinton_distilled": 3.2218,
                "anchor_3_of_3_disambiguator_full_vs_identity_delta_s_final_loss": delta_s,
            },
            "z6_v2_film_ego_motion": {
                "sister_anchor_pose_axis_reduction_baseline": 3.74,
                "disambiguator_status_per_slot_w": "DEFER_INFRASTRUCTURE_GAP_MLX_RAISES_NOTIMPLEMENTEDERROR",
            },
            "z7_mamba_2_state_space": {
                "anchor_4_of_4_pose_axis_convergence_parity_per_slot_t_wave_n_plus_32": -19.478,
                "convergence_band_per_wave_n_plus_28": "WITHIN_2.7PP_NOISE_BAND_OF_Z5",
                "disambiguator_status_per_slot_t": "PROBE_LANDED_CONVERGENCE_PARITY_VERDICT",
            },
            "z8_hierarchical_predictive_coding_canonical_quadruple": {
                "disambiguator_status_per_slot_w": "DEFER_INFRASTRUCTURE_GAP_NO_IDENTITY_PREDICTOR_FLAG",
            },
        },
        "non_promotable_per_catalog_192_317_341": True,
        "macos_cpu_advisory_per_catalog_192": True,
        "next_action": (
            "Per anchor 3/3 verdict + Wave N+28 Z5≈Z6-v2 parity + Wave N+32 Z7-Mamba-2 "
            "4/4 anchor: cross-family Z-family disambiguator anchor extension landed; "
            "canonical equation #344 anchor advances; Catalog #371 auto-recalibrator "
            "trigger check (3+ anchors in domain triggers refit); Catalog #313 probe "
            "outcome row LANDED; per Catalog #307 paradigm-vs-implementation: if "
            "DEFER verdict the paradigm REMAINS INTACT — reactivation criteria "
            "pinned in canonical_unwind_path_if_falsified field."
        ),
    }

    ARTIFACT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[z5_disambiguator] artifact: {ARTIFACT_PATH}", flush=True)
    print(f"[z5_disambiguator] VERDICT: {verdict} (delta_s={delta_s:+.5f} threshold={delta_s_threshold})", flush=True)
    print(f"[z5_disambiguator] paradigm_classification: {paradigm_classification}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
