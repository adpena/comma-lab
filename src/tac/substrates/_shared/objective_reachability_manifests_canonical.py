# SPDX-License-Identifier: MIT
"""Seed objective-reachability manifests for the score-aware carrier vehicles.

Populated 2026-06-09 (P1 of the operator V6 ObjectiveReachability packet) from:

* ``.omx/research/snerv_b_first_scorer_probe_verdict_20260609.md`` (the link-5
  case study + the post-``f5c66f43c`` telemetry fields:
  ``segnet_direct_live=7.24`` / ``pose_direct_live=7.0`` / CE=3.78 / recon
  annealed 0.2; the uncrossed objective CONFIRMED ACTIVE in
  ``loss_components.active_loss_weight__*``).
* ``src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py``
  (the trainer whose ``segnet_distillation_weight: float = 0.0`` /
  ``pose_distillation_weight: float = 0.0`` defaults were the ep22399 starvation;
  post-uncrossing the dispatched run sets them 7.24 / 7.0).
* ``src/tac/substrates/hi_nerv/short_scorer_readiness.py`` (the HiNeRV scorer
  readiness probe whose distillation weights are caller-supplied and default to
  recon-only on the shared MLX harness — the honest CURRENT state).
* ``src/tac/substrates/pact_nerv_vq/`` (genuine VQ + correct PyTorch
  score_aware_loss shape; the MLX long-run scorer-objective wiring is the
  in-flight audit -> AUDIT_PENDING grad norms).

Each manifest is a frozen :class:`ObjectiveReachabilityManifest`. The module's
``emit_all(...)`` writes them as durable JSON under
``.omx/state/objective_reachability/`` (the Catalog #386 gate + the Vehicle-OS
dashboard consume them).

The 3 seed verdicts (operator-requested):

* ``snerv``       — REACHES. Post-``f5c66f43c`` both VJPs reach the renderer and
  the loss weights are nonzero (segnet 7.24 / pose 7.0). ``verify()`` PASSES.
  This is the FAITHFUL score-aware carrier on the live surface (the residual
  chasm is EXPORT/RATE per the probe verdict, NOT reachability).
* ``hi_nerv``     — FAILS. The shared MLX harness leaves the SegNet/PoseNet
  distillation weights at their recon-only default (0.0), so the score-aware
  claim has no nonzero weight and no surrogate gradient. ``verify()`` RAISES
  (the honest current state; fix is config-only — set nonzero weights + wire
  the surrogate).
* ``pact_nerv_vq`` — PARTIAL/PENDING. PyTorch score_aware_loss has the right
  shape (frozen-scorer, surrogate CE), but the MLX long-run lane's objective
  wiring is AUDIT_PENDING, so the carrier is recorded recon-only-by-default for
  the MLX route (conservative honest absence) and ``verify()`` RAISES on the
  active-claim-without-nonzero-weight condition for that route until the audit
  closes.
"""

from __future__ import annotations

from pathlib import Path

from tac.substrates._shared.objective_reachability_manifest import (
    AUDIT_PENDING,
    ObjectiveReachabilityManifest,
    emit_objective_reachability_manifest,
)

__all__ = [
    "CANONICAL_OBJECTIVE_REACHABILITY_MANIFESTS",
    "HI_NERV_REACHABILITY",
    "PACT_NERV_VQ_REACHABILITY",
    "SNERV_REACHABILITY",
    "emit_all",
]

_PROBE_MEMO = ".omx/research/snerv_b_first_scorer_probe_verdict_20260609.md"
_FLEET_MEMO = (
    ".omx/research/snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md"
)


# ---------------------------------------------------------------------------
# 1. snerv — post-f5c66f43c the objective REACHES (both VJPs, weights 7.24/7.0)
# ---------------------------------------------------------------------------
SNERV_REACHABILITY = ObjectiveReachabilityManifest(
    vehicle="snerv",
    segnet_objective_active=True,
    posenet_objective_active=True,
    # J_seg reaches the renderer through the CE surrogate (the probe verdict
    # cites CE=3.78 as the active seg surrogate term); the official argmax d_seg
    # (live 0.0023) is the VERIFICATION metric only, not a training row.
    segnet_surrogate_rows=("ce",),
    # Post-f5c66f43c (the pose-VJP uncrossing): both objective gradients reach
    # the renderer. Confirmed live in-gradient per the commit + the probe memo
    # ("the uncrossed objective CONFIRMED ACTIVE in loss_components.
    # active_loss_weight__*"); the pose VJP severance was at 3 layers BEFORE
    # f5c66f43c and is now repaired.
    segnet_vjp_reaches_renderer=True,
    posenet_vjp_reaches_renderer=True,
    # The dispatched run sets nonzero weights (segnet_direct_live=7.24,
    # pose_direct_live=7.0 per the probe telemetry) — NOT the trainer's 0.0
    # source defaults that produced ep22399 starvation.
    loss_weights_nonzero=True,
    # SNeRV's optimizable surface is the small HF/LF residual (the LF is stored
    # from source). The decoder blocks + HFR head + latents carry the objective
    # gradient; norms not yet harvested into this manifest -> AUDIT_PENDING
    # (honest not-measured, NOT a severance).
    gradient_norm_by_mechanism={
        "latents": AUDIT_PENDING,
        "decoder_blocks": AUDIT_PENDING,
        "hf_residual": AUDIT_PENDING,
        "mfu": AUDIT_PENDING,
        "hfr": AUDIT_PENDING,
        "tub": AUDIT_PENDING,
    },
    severed_layers=(),
    first_failed_surface="",
    dseg_is_verification_metric_only=True,
    summary=(
        "REACHES: post-f5c66f43c both SegNet+PoseNet VJPs reach the renderer; "
        "weights nonzero (segnet 7.24 / pose 7.0); CE surrogate carries J_seg; "
        "argmax d_seg (live 0.0023) is verification-only. The SNeRV chasm is "
        "EXPORT/RATE (G1b + LF bytes), NOT reachability [macOS-MLX research-signal]."
    ),
    source_artifacts=(_PROBE_MEMO, "commit f5c66f43c"),
)


# ---------------------------------------------------------------------------
# 2. hi_nerv — distill weights default 0.0 on the shared harness -> FAILS
# ---------------------------------------------------------------------------
HI_NERV_REACHABILITY = ObjectiveReachabilityManifest(
    vehicle="hi_nerv",
    # The carrier's short_scorer_readiness probe + design CLAIM a score-aware
    # path (both objectives are nominally part of the readiness DAG), so it
    # makes a score-aware claim that the reachability surface must honor.
    segnet_objective_active=True,
    posenet_objective_active=True,
    # No surrogate row is wired into the default MLX harness route -> the only
    # seg signal would be gradient-zero argmax d_seg (no seg learning). Empty
    # surrogate rows is itself a SURROGATE-ABSENCE finding.
    segnet_surrogate_rows=(),
    # With the weights at their recon-only default, no objective gradient is
    # produced, so neither VJP reaches the renderer in the default trained path.
    segnet_vjp_reaches_renderer=False,
    posenet_vjp_reaches_renderer=False,
    # THE failing surface: the shared MLX harness leaves
    # segnet/pose distillation weights at 0.0 (recon-MSE-only), so the
    # score-aware claim has no nonzero weight. (Catalog #384 sister condition.)
    loss_weights_nonzero=False,
    gradient_norm_by_mechanism={
        # Measured-ZERO under the (absent) scorer objective: the objective is
        # not wired so the latent + decoder gradient under it is 0.0 — recorded
        # as severance, the honest current state.
        "latents": 0.0,
        "decoder_blocks": 0.0,
    },
    severed_layers=("scorer_objective_to_renderer (weights default 0.0)",),
    first_failed_surface="weight",
    dseg_is_verification_metric_only=True,
    summary=(
        "FAILS: hi_nerv claims score-aware but the shared MLX harness leaves "
        "SegNet/PoseNet distillation weights at the recon-only 0.0 default, so no "
        "surrogate gradient reaches the renderer (first_failed_surface=weight). "
        "Honest current state; fix is config-only (set nonzero weights + wire a "
        "CE/margin surrogate). verify() RAISES."
    ),
    source_artifacts=(
        "src/tac/substrates/hi_nerv/short_scorer_readiness.py",
        _FLEET_MEMO,
    ),
)


# ---------------------------------------------------------------------------
# 3. pact_nerv_vq — PyTorch loss correct; MLX route objective wiring PENDING
# ---------------------------------------------------------------------------
PACT_NERV_VQ_REACHABILITY = ObjectiveReachabilityManifest(
    vehicle="pact_nerv_vq",
    # The carrier claims a score-aware path (the PyTorch score_aware_loss is the
    # right shape). The default MLX long-run route's objective wiring is the
    # in-flight audit.
    segnet_objective_active=True,
    posenet_objective_active=True,
    # The PyTorch score_aware_loss uses a CE surrogate on the frozen SegNet
    # logits (the right shape). Recorded as the carrier's seg surrogate.
    segnet_surrogate_rows=("ce",),
    # Conservative honest absence for the DEFAULT MLX route: the long-run lane
    # routes through the shared recon-MSE harness whose objective wiring to the
    # renderer is AUDIT_PENDING -> recorded as not-yet-reaching for the trained
    # MLX path. (The PyTorch path's reachability is correct but is not the
    # dispatched MLX route.)
    segnet_vjp_reaches_renderer=False,
    posenet_vjp_reaches_renderer=False,
    loss_weights_nonzero=False,
    gradient_norm_by_mechanism={
        # The VQ codebook is genuinely used (STE + EMA), but the scorer
        # objective's gradient norm into it on the MLX route is not yet measured.
        "codebook": AUDIT_PENDING,
        "decoder_blocks": AUDIT_PENDING,
        "latents": AUDIT_PENDING,
    },
    # The MLX route's recon-only default severs the scorer objective from the
    # renderer (same shared-harness class as hi_nerv) until the audit closes.
    severed_layers=("mlx_route_scorer_objective (shared recon-MSE harness; AUDIT_PENDING)",),
    first_failed_surface="weight",
    dseg_is_verification_metric_only=True,
    summary=(
        "PARTIAL/PENDING: pact_nerv_vq has a correct PyTorch score_aware_loss "
        "(CE surrogate, frozen scorer) but the default MLX long-run route's "
        "objective wiring is AUDIT_PENDING and routes through the shared recon-MSE "
        "harness -> recorded recon-only-by-default for the MLX route. verify() "
        "RAISES on the active-claim-without-nonzero-weight condition until the "
        "MLX-route audit closes [macOS-MLX research-signal]."
    ),
    source_artifacts=(
        "src/tac/substrates/pact_nerv_vq/architecture.py:141-166",
        _FLEET_MEMO,
    ),
)


CANONICAL_OBJECTIVE_REACHABILITY_MANIFESTS: tuple[ObjectiveReachabilityManifest, ...] = (
    SNERV_REACHABILITY,
    HI_NERV_REACHABILITY,
    PACT_NERV_VQ_REACHABILITY,
)


def emit_all(repo_root: str | Path | None = None) -> list[Path]:
    """Emit all 3 seed manifests as durable JSON; return the paths.

    Manifests are emitted with ``verify=False`` so the deliberately-FAILING
    ``hi_nerv`` + ``pact_nerv_vq`` manifests ARE written (the Catalog #386 gate +
    operator review consume them to SURFACE the severance / starvation).
    Verification is the gate's job, not a precondition for recording the honest
    reachability truth.
    """
    return [
        emit_objective_reachability_manifest(m, repo_root=repo_root, verify=False)
        for m in CANONICAL_OBJECTIVE_REACHABILITY_MANIFESTS
    ]
