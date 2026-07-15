# SPDX-License-Identifier: MIT
"""Canonical equation: AdamC weight-decay x lr-schedule equilibrium (Defazio, arXiv:2506.02285).

THE LAW (Van Laarhoven 2017 recursion, Defazio 2025 Eq. 1-2): for a weight tensor whose update
magnitude is weight-independent (paper: normalized layers, ``<g,x>=0``; Chou arXiv:2512.08217:
the weaker independence assumption ``E<theta,u> ~= 0`` — near-exact for per-param-normalized
updates and for Muon's NS-orthonormalized updates), decoupled weight decay drives the
gradient-to-weight ratio to the steady state

    ||g_t|| / ||x_t|| = sqrt(2*lambda / gamma_t)          (norm ODE: d||x||^2/dt =
                                                            -2*gamma*lambda*||x||^2 + gamma^2*E||u||^2)

so a DECAYING lr schedule raises the target like 1/sqrt(gamma_t) — the measured tail
gradient-norm blow-up + weight-norm collapse. The AdamC correction (paper Alg. 1)

    lambda_hat_t = lambda * gamma_t / gamma_max     =>     ||g||/||x|| -> sqrt(2*lambda/gamma_max)

is schedule-independent. Time constant of the equilibrium chase: ~ 1/(2*lambda*gamma_t) steps.
The config-computable MECHANISM-STRENGTH scalar is ``lambda * sum_t(gamma_t)`` (the total
multiplicative weight-norm pressure): >~ O(1) in the paper's LLM runs (lambda=0.05, weight norms
moved ~70%); ~2e-4 (n24 ep1-75) .. ~1e-2 (n600 x 3000 ep) on the live v9_cgauge config
(lambda=1e-4) => the mechanism is PREDICTED-NULL on our runs (memo P1 — the cargo-cult guard).

Assumption fork + the three reconciliation verdicts (maglaw reversal DOES-NOT-APPLY-as-mechanism /
C0 saturation DOES-NOT-APPLY / Muon finisher CONSISTENT-WITH-A-TWIST: MLX Muon wd is
coupled-through-NS ~= inert): ``.omx/research/adamc_muonc_optimizer_research_20260715.md``.

DSL leg: ``curriculum_dsl.CorrectedWeightDecay`` (``--weight-decay-corrected``, default OFF).
Mechanism: the levelset trainer's per-epoch ``opt.weight_decay = _corrected_weight_decay(...)``.
Muon leg: ticketed only (``adaptivization_tickets_20260715``, ``--muon-weight-decay``) — "MuonC"
is NOT a named optimizer in the literature (2026-07 sweep); nearest relative is ScionC.

means != ends: a derived LAW + a predicted-null lever; NO score claim. Pointer UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    INFERRED_FROM_DOMAIN_LITERATURE,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "adamc_wd_lr_equilibrium_v1"

_UTC = "2026-07-15T00:00:00Z"
_PREDICTED = "[predicted]"
_MEMO = ".omx/research/adamc_muonc_optimizer_research_20260715.md"

# --- live-config mechanism-strength inputs (SOURCE-INSPECTED from the sealed argv) ------------
LIVE_LAMBDA = 1e-4            # --weight-decay (trunk + finisher default)
LIVE_LR_MAX = 1e-3            # --lr (cosine peak)
LIVE_LR_END = 1e-4            # --lr-end
PAPER_LAMBDA_LLM = 0.05       # Defazio Fig.4 run (120M Llama3, 200B tok)
PAPER_LOSS_ADAMW = 2.461      # Fig. 4
PAPER_LOSS_ADAMC = 2.457


def steady_state_grad_to_weight_ratio(lam: float, gamma: float) -> float:
    """||g||/||x|| = sqrt(2*lambda/gamma) (Defazio Eq. 2; requires lam>0, gamma>0)."""
    if lam <= 0.0 or gamma <= 0.0:
        raise ValueError(f"lam and gamma must be positive, got {lam!r}, {gamma!r}")
    return float((2.0 * lam / gamma) ** 0.5)


def corrected_weight_decay(lam: float, gamma_t: float, gamma_max: float) -> float:
    """AdamC Alg. 1: lambda_hat_t = lambda * gamma_t/gamma_max (degenerate inputs pass through)."""
    if lam <= 0.0 or gamma_max <= 0.0:
        return float(lam)
    return float(lam) * (float(gamma_t) / float(gamma_max))


def mechanism_strength(lam: float, gamma_sum: float) -> float:
    """lambda * sum_t(gamma_t): total multiplicative weight-norm pressure of decoupled wd over a
    run (weight-norm shrink factor ~ exp(-mechanism_strength) absent gradient replenishment).
    << 1 => the AdamC wd x schedule mechanism cannot materially move norms within the run."""
    return float(lam) * float(gamma_sum)


def build_adamc_wd_lr_equilibrium_v1() -> CanonicalEquation:
    """Build the AdamC equilibrium law with its literature + config-derived anchors."""

    anchor_paper = EmpiricalAnchor(
        anchor_id="defazio_adamc_llm_200b_tail_blowup_20250614",
        measurement_utc="2025-06-14T00:00:00Z",
        inputs={
            "source": "arXiv:2506.02285v2 (read in full 2026-07-15)",
            "setting": ("120M-param Llama3-arch LLM, FineWeb-Edu 200B tokens, cosine schedule, "
                        "wd 0.05, lr swept power-of-2 in [0.001, 0.02]; + ResNet-50/ImageNet SGDM"),
        },
        predicted_output={
            "claim": ("||g||/||x|| tracks sqrt(2*lambda/gamma_t); cosine tail => gradient-norm "
                      "blow-up + weight-norm collapse; lambda_hat=lambda*gamma_t/gamma_max "
                      "flattens both"),
        },
        empirical_output={
            "llm_final_loss": {"AdamW": PAPER_LOSS_ADAMW, "AdamC": PAPER_LOSS_ADAMC},
            "llm_weight_norm": "AdamW ~6500 -> ~2000 over the tail; AdamC ~flat (Fig. 4)",
            "imagenet_top1": {"SGDC": "77.07±0.10", "SGDM": "76.95±0.14"},
            "residual_drift": ("a separate slow linear gradient-norm drift REMAINS on ImageNet "
                               "(their theory covers only the wd x schedule component)"),
            "verdict_scope": ("LITERATURE anchor — external measurements at 500x our lambda, with "
                              "normalization layers we do not have; NON-PROMOTABLE; transfers as "
                              "a LAW + a frame, never as our numbers (L18 ancestor rule sister)"),
        },
        residual=0.0,
        source_artifact="https://arxiv.org/abs/2506.02285",
        measurement_method="published paper figures/tables (Defazio, FAIR at Meta)",
        empirical_verification_status=INFERRED_FROM_DOMAIN_LITERATURE,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=("re-read on any follow-up correcting the law (ScionC "
                                   "arXiv:2512.08217 already folded); supersede if a Muon-native "
                                   "corrected-wd paper lands"),
            measurement_axis=_PREDICTED,
            hardware_substrate="external_literature",
        ),
    )
    anchor_ours = EmpiricalAnchor(
        anchor_id="v9_cgauge_mechanism_strength_derived_20260715",
        measurement_utc=_UTC,
        inputs={
            "lambda": LIVE_LAMBDA,
            "lr": [LIVE_LR_MAX, LIVE_LR_END],
            "argv_source": "experiments/results/levelset_n24_maglaw_arm*/launch.sh (sealed v9 config)",
            "muon_wd_mechanism": ("mlx.optimizers.Muon.apply_single: gradient += wd*parameter "
                                  "BEFORE momentum+NS (COUPLED, source-inspected)"),
        },
        predicted_output={
            "claim": ("mechanism_strength = lambda*sum(gamma_t) ~ 1.8e-4 (n24 ep1-75) .. ~1.1e-2 "
                      "(n600 x 3000ep @ ~75 opt-steps/ep) << 1 => AdamC correction PREDICTED-NULL "
                      "on our runs; MLX Muon coupled wd ~= INERT (3-4 orders below C0 gradient "
                      "band 5.9-17.5, then NS re-normalizes)"),
        },
        empirical_output={
            "n24_maglaw_window": ("reconciliation: the armB post-ep25 reversal CANNOT be the wd x "
                                  "schedule mechanism (lr ~flat in-window, no norm layers, "
                                  "strength 1.8e-4); C0 frac_clipped=1.0 is a level mismatch from "
                                  "ep1, not tail growth — both DOES-NOT-APPLY (memo §4)"),
            "owed_local_anchor": ("adamc_null_effect_n24_ab_owed_20260715 — the staged "
                                  "CorrectedWeightDecay ON/OFF n24 >=150ep A/B (memo P1); "
                                  "P3 ($0): Muon-group ||W|| growth across finisher checkpoints"),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method=("derivation from source-inspected sealed config constants + installed "
                            "mlx.optimizers source; no run was consumed"),
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=("recompute mechanism_strength whenever --weight-decay / the lr "
                                   "schedule / run length changes by >2x; the owed n24 A/B "
                                   "replaces the derived null with a measured one"),
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("AdamC weight-decay x lr-schedule equilibrium: ||g||/||x|| -> sqrt(2*lambda/gamma_t); "
              "corrected lambda_hat_t = lambda*gamma_t/gamma_max pins sqrt(2*lambda/gamma_max) "
              "(schedule-independent); mechanism strength = lambda*sum(gamma_t)"),
        one_line_summary=(
            "Decoupled wd + decaying lr raises steady-state ||g||/||x|| like 1/sqrt(gamma_t); "
            "lambda_hat=lambda*gamma_t/gamma_max corrects it; predicted-null at our lambda=1e-4."
        ),
        latex_form=(
            r"\frac{\|g_t\|}{\|x_t\|}=\sqrt{\frac{2\lambda}{\gamma_t}},\qquad "
            r"\hat\lambda_t=\lambda\frac{\gamma_t}{\gamma_{\max}}\ \Rightarrow\ "
            r"\frac{\|g\|}{\|x\|}=\sqrt{\frac{2\lambda}{\gamma_{\max}}},\qquad "
            r"S_{\mathrm{mech}}=\lambda\sum_t\gamma_t"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.adamc_wd_lr_equilibrium_20260715:steady_state_grad_to_weight_ratio"
        ),
        domain_of_validity={
            "assumption": ("update magnitude weight-independent: normalized layers (Defazio, "
                           "exact) OR Chou independence form (per-param-normalized / "
                           "NS-orthonormalized updates, near-exact); DECOUPLED wd only — MLX "
                           "Muon's coupled-through-NS wd is OUTSIDE the law (inert, ticketed)"),
            "vehicle": ["softmax_of_sdf_levelset_witness", "v9_cgauge_*"],
            "lever": ("tac.witness_dsl.curriculum_dsl.CorrectedWeightDecay "
                      "(--weight-decay-corrected, trunk-only, default OFF)"),
            "measurement_axis": ["predicted", "external_literature"],
            "note": ("training-path lever under the 2026-07-15 relaxed-identity directive; "
                     "decode/verdict/byte-close untouched; PREDICTED-NULL at live lambda=1e-4 — "
                     "the A/B is the cargo-cult guard, not a hoped win"),
        },
        units_in={"lam": "1/step_scaled_decay", "gamma": "learning_rate", "gamma_sum": "sum_lr"},
        units_out={"ratio": "grad_norm_over_weight_norm", "lambda_hat": "weight_decay",
                   "strength": "dimensionless"},
        empirical_anchors=(anchor_paper, anchor_ours),
        # NOTE: the local null-prediction residual is OWED (anchor
        # adamc_null_effect_n24_ab_owed_20260715, staged n24 A/B) — schema requires numeric
        # values, so the owed axis is carried in anchor_ours.empirical_output instead.
        predicted_vs_empirical_residual={
            "paper_literature_anchor": 0.0,
            "config_derived_mechanism_strength": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",  # CorrectedWeightDecay factory
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # _corrected_weight_decay
            "tac.witness_dsl.adaptivization_tickets_20260715",  # the Muon decoupled+corrected ticket
        ),
        canonical_producers=(
            ".omx/research/adamc_muonc_optimizer_research_20260715.md",
        ),
        provenance=build_provenance_for_predicted(
            model_id="adamc_wd_lr_equilibrium.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_adamc_wd_lr_equilibrium_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the AdamC equilibrium law (latest-row-wins).

    EQUATIONS leg of the adamc/muonc research unit; DSL leg = ``CorrectedWeightDecay`` +
    the Muon adaptivization ticket; DAG leg = FEED row in the sub015 DAG."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_adamc_wd_lr_equilibrium_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="adamc_wd_lr_equilibrium_20260715 (equations leg of the adamc/muonc research; "
              "literature + config-derived anchors; local n24 A/B anchor OWED "
              "adamc_null_effect_n24_ab_owed_20260715)",
    )
    return eq
