# SPDX-License-Identifier: MIT
"""Canonical equation: the INR radial weight-norm ODE + the chosen-invariant law (2026-07-15).

THE LAW (derived + MEASURED on our own stage checkpoints — the follow-up to
``adamc_wd_lr_equilibrium_v1``): for a coordinate-INR with sin/hosc activations, FiLM, NO
normalization layers (our witness), the per-tensor norm obeys

    d||W||^2/dt = -2*gamma*lambda*||W||^2  -  2*gamma*<grad L, W>  +  gamma^2 * E||u||^2
                       (wd)                     (radial force R)          (diffusion)

Defazio/Chou's setting has R == 0 (scale invariance); OURS DOES NOT. Measured channel magnitudes
(film.weight, mod32cap n600 Muon finisher ep726-1000, 20,550 steps): radial -464 vs diffusion +63
vs wd -3.3 in ||W||^2 units — **the radial gradient force dominates 7:1:0.05**, so weight norms
equilibrate where the LANDSCAPE's radial slope matches half the diffusion (<grad L, W> ~=
gamma*E||u||^2/2), not at the wd-lr equilibrium (measured ||g||/||W|| sits 4-50x BELOW
sqrt(2*lambda/gamma); chase time-constant 1/(2*lambda*gamma) ~ 5e6 steps >> any run).

Consequences (each MEASURED, see anchors):
  1. Muon finisher (flat lr, NS => weight-independent update norm u): eta_rel = u/||W(t)|| is a
     STATE variable. Measured: Muon-group norms SHRANK 0.9-28.7% during the finisher =>
     a hidden per-layer LR INCREASE x1.01-1.40 nobody chose (P3's growth prediction FALSIFIED —
     the drift is gradient-driven, wd is 70x too small, and it is inward not a random walk).
  2. The softmax head is a GAUGE direction: (tau, W_head, b_head) -> (c*tau, c*W_head, c*b_head)
     is loss-invariant; only kappa = ||W_head||/tau is physical. Whatever sharpening the tau
     schedule does not deliver, the CE radial force delivers via head-norm growth. Measured:
     l7-lineage head x6.07 (+2.6 UNSCHEDULED octaves) vs mod32cap x0.96 (explicit temp schedule).
  3. ||W|| IS spectral content for sin/hosc rows (frequency omega*||w|| along w-hat): norm drift
     = NTK band shift; and the quantization grammar couples it to rate/fidelity (measured slope
     0.17-0.23 bits/octave at fixed absolute step, r~0.52; FIRST-order coupling is the fidelity
     channel: per-tensor scale delta = linf/127 grows with the norm => phase jitter on spectral
     tensors).

THE CHOSEN INVARIANT (prescriptive, operator full-pipeline co-design authority 2026-07-15):
    ||w||* = min(k_need, k_R)/omega   (row-norm profile: band edge at the R-operator cutoff —
                                       any norm above is strictly dominated: pays bits AND aliases)
    delta* = eps_phase/(omega*||x||)  (grid step from the phase budget)
    kappa* = ||W_head||/tau           (head: schedule kappa, not tau alone)
Hold-it candidates (staged, default-OFF, ticketed not armed): row-norm projection at ckpt
cadence / Muon eta_rel pin (gamma proportional to ||W(t)||) / restoring decay
-lambda*(||W||-||W||*)*W-hat / grid-native training on the shipping lattice (FakeQuantSTE).

Memo (all numbers + scripts): ``.omx/research/optimizer_dynamics_followup_20260715.md``.
means != ends: derived law + measured anchors; NO score claim. Pointer 0.19108 UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "inr_weight_norm_radial_ode_v1"

_UTC = "2026-07-15T00:00:00Z"
_PREDICTED = "[predicted]"
_MEMO = ".omx/research/optimizer_dynamics_followup_20260715.md"
_ART = "experiments/results/optdyn_followup_20260715"


def muon_update_norm(n: int, m: int, lr: float) -> float:
    """MLX Muon applied update Frobenius norm (weight-independent by NS construction):
    ||u|| = lr * sqrt(max(1, n/m)) * sqrt(min(n, m))  for an (n, m) tensor."""
    if n <= 0 or m <= 0 or lr <= 0.0:
        raise ValueError(f"n, m, lr must be positive, got {n!r}, {m!r}, {lr!r}")
    return float(lr) * (max(1.0, n / m) ** 0.5) * (min(n, m) ** 0.5)


def muon_relative_step(n: int, m: int, lr: float, weight_norm: float) -> float:
    """eta_rel = ||u||/||W|| — the Muon-group effective per-step relative LR (a STATE variable
    under flat lr: measured x1.40 hidden increase on film.weight across the mod32cap finisher)."""
    if weight_norm <= 0.0:
        raise ValueError(f"weight_norm must be positive, got {weight_norm!r}")
    return muon_update_norm(n, m, lr) / float(weight_norm)


def effective_inverse_temperature(head_norm: float, tau: float) -> float:
    """kappa = ||W_head||/tau — the ONLY physical sharpness (softmax head gauge invariant).
    Reading tau alone mis-states the anneal by the head-norm drift factor (measured x6.07 l7)."""
    if head_norm <= 0.0 or tau <= 0.0:
        raise ValueError(f"head_norm and tau must be positive, got {head_norm!r}, {tau!r}")
    return float(head_norm) / float(tau)


def radial_channel_decomposition(
    delta_w2: float, sum_u2: float, lam: float, gamma_sum: float, w2_mean: float
) -> dict:
    """Decompose an observed Delta(||W||^2) into wd / radial / diffusion channels:
    radial_work = -(delta_w2 - sum_u2 + 2*lam*gamma_sum*w2_mean)/2 (positive => inward force)."""
    wd_term = -2.0 * lam * gamma_sum * w2_mean
    radial = -(float(delta_w2) - float(sum_u2) - wd_term) / 2.0
    return {"wd": wd_term, "diffusion": float(sum_u2), "radial_work": radial}


def build_inr_weight_norm_radial_ode_v1() -> CanonicalEquation:
    """Build the INR radial-norm law with its three measured anchors + the T4 pre-registration."""

    anchor_muon = EmpiricalAnchor(
        anchor_id="muon_finisher_norm_shrink_hidden_lr_increase_measured_20260715",
        measurement_utc=_UTC,
        inputs={
            "run": "levelset_n600_witness_mod32cap_20260706T115554Z (Muon ep726-1000, flat lr 2e-3)",
            "checkpoints": "resume_stageCE_ep299 / stageMuonStart_ep726 / stageTau_muon_ep1000",
            "script": f"{_ART}/t1b_effective_step.py",
        },
        predicted_output={
            "predecessor_P3": ("undamped random-walk ||W|| GROWTH in the finisher (inert coupled "
                               "wd => no damping) — the prediction under R==0"),
        },
        empirical_output={
            "measured": ("Muon-group norms SHRANK: film 28.58->20.38 (-28.7%), hidden.3 -18.2%, "
                         "hidden.2 -15.3%, hidden.1 -10.7%, hidden.0 -8.6%, in_proj -0.9%; "
                         "group aggregate 54.27->45.93 (-15.4%)"),
            "decomposition_film": {"delta_w2": -401.4, "diffusion_sum_u2": 63.1,
                                   "radial_work": 232.3, "mean_inward_cosine": 0.0085,
                                   "wd_channel_pct": -0.41},
            "hidden_lr_increase": {"film": 1.402, "hidden3": 1.222, "hidden2": 1.180,
                                   "hidden1": 1.120, "hidden0": 1.094, "in_proj": 1.009},
            "adam_v_equilibrium_check": ("||g||/||W|| measured 0.008-0.10 (out_tex 0.82 sole "
                                         "outlier) vs Defazio target 0.447 — 4-50x below; "
                                         "equilibrium never approached"),
            "verdict": "P3 FALSIFIED (sign inverted): gradient-driven inward radial force, not wd, "
                       "not a random walk; the finisher ran a hidden per-layer LR INCREASE",
            "verdict_scope": "FORMULATION (mod32cap n600 lineage, MLX advisory); NON-PROMOTABLE",
        },
        residual=0.0,
        source_artifact=f"{_ART}/t1_norm_trajectories.json",
        measurement_method="numpy over existing stage-checkpoint npz (live weights + Adam v "
                           "buffers); $0, no run consumed",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=("re-measure on any finisher config change (muon lr schedule, "
                                   "NS steps, group routing); the eta_rel-pin ticket A/B replaces "
                                   "the drift with a chosen invariant"),
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_head = EmpiricalAnchor(
        anchor_id="head_norm_temperature_gauge_drift_measured_20260715",
        measurement_utc=_UTC,
        inputs={
            "runs": ("l7 lineage n200 (20260628, AdamW-only, CE->Tau->L7) vs mod32cap n600 "
                     "(explicit softmax-temp schedule 1.0->0.05)"),
            "script": f"{_ART}/t1b_effective_step.py",
        },
        predicted_output={
            "law": ("kappa = ||W_head||/tau is the gauge invariant; the CE radial force delivers "
                    "whatever sharpening the schedule does not"),
        },
        empirical_output={
            "l7_head_growth": {"ep299": 16.24, "ep899": 90.30, "ep1500": 98.58, "factor": 6.07,
                               "unscheduled_octaves": 2.6},
            "mod32cap_head_growth": {"ep299": 17.32, "ep1000": 16.64, "factor": 0.96,
                                     "scheduled_tau_sharpening": 3.74},
            "dual_consequence": ("Adam per-param step ~lr regardless of ||W|| => the l7 head's "
                                 "RELATIVE step fell x6 during Tau — a hidden per-layer LR DECAY "
                                 "on the head exactly during the sharpness-tuning stage"),
            "verdict": ("tau-anneal history is formulation-dependently confounded by head-norm "
                        "drift; only kappa is physical — gauge-fix it (schedule kappa or project "
                        "the head norm)"),
            "verdict_scope": "FORMULATION (l7 n200 + mod32cap n600 lineages); NON-PROMOTABLE",
        },
        residual=0.0,
        source_artifact=f"{_ART}/t1_norm_trajectories.json",
        measurement_method="numpy over existing stage-checkpoint npz (EMA + live)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=("owed ONE logging change: per-tensor ||W|| telemetry row at "
                                   "verdict cadence (score-neutral, defaults ON) — the stream the "
                                   "kappa gauge-fix reads"),
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_rate = EmpiricalAnchor(
        anchor_id="weight_norm_quant_rate_coupling_measured_20260715",
        measurement_utc=_UTC,
        inputs={
            "method": ("EMA-shadow tensors; (i) per-tensor-scale int8 (L29 grammar) vs (ii) fixed "
                       "absolute step frozen from ep299; entropy + zlib per stage"),
            "script": f"{_ART}/t3_norm_rate_cost.py",
        },
        predicted_output={
            "naive_theory": "1.0 bits/weight per octave of norm growth at fixed absolute fidelity",
        },
        empirical_output={
            "measured_slope_bits_per_octave": {"mod32cap": 0.228, "l7": 0.172},
            "cross_tensor_corr_r": {"mod32cap": 0.535, "l7": 0.517},
            "l7_fixed_delta_bits_per_w": {"ep299": 7.054, "ep1500": 7.298, "pct": +3.5},
            "shape_effect": ("linf grows via tails while the bulk stays put — per-tensor-scale "
                             "entropy FELL on l7 (7.05->6.85) as kurtosis rose"),
            "verdict": ("norm->BYTES is real but SECOND-order at measured drifts; FIRST-order is "
                        "the FIDELITY channel: delta = linf/127 grows with the norm => x2.45 "
                        "coarser phase resolution on hidden.3 (+1.29 octaves) = quantization "
                        "spectral jitter, a d_seg risk not a rate line-item. Norm control enters "
                        "the config as a QUANT-FIDELITY argument; grid-native co-design "
                        "(GridNativeWitness, memo T3') kills the channel by construction"),
            "verdict_scope": "FORMULATION (mod32cap + l7 lineages, EMA shadows); NON-PROMOTABLE",
        },
        residual=0.0,
        source_artifact=f"{_ART}/t3_norm_rate_cost.json",
        measurement_method="numpy int8/int16 quantization + Shannon entropy + zlib-9 over existing "
                           "EMA checkpoints",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=("re-measure at larger drifts or on the GridNativeWitness A/B "
                                   "(grid-native vs post-hoc int8 at equal bytes through the real "
                                   "byte-close)"),
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("INR radial weight-norm ODE: d||W||^2/dt = -2*gamma*lambda*||W||^2 "
              "- 2*gamma*<gradL,W> + gamma^2*E||u||^2 — radial force dominates (7:1:0.05); "
              "gauge kappa=||W_head||/tau; Muon eta_rel=||u||/||W|| is a state variable; "
              "chosen invariant ||w||*=min(k_need,k_R)/omega"),
        one_line_summary=(
            "Radial gradient force (not wd) sets INR weight norms: Muon finisher shrank 0.9-28.7% "
            "(hidden LR x1.4), l7 head grew x6.07 (unscheduled anneal); hold the CHOSEN invariant."
        ),
        latex_form=(
            r"\frac{d\|W\|^2}{dt}=-2\gamma\lambda\|W\|^2-2\gamma\langle\nabla L,W\rangle"
            r"+\gamma^2\mathbb{E}\|u\|^2,\quad \kappa=\frac{\|W_{head}\|}{\tau},\quad "
            r"\eta_{rel}=\frac{\|u\|}{\|W\|},\quad \|w\|^*=\frac{\min(k_{need},k_R)}{\omega}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.inr_weight_norm_radial_ode_20260715:muon_relative_step"
        ),
        domain_of_validity={
            "assumption": ("coordinate-INR with non-homogeneous activations (sin/hosc/step), NO "
                           "normalization layers, softmax head; AdamW trunk + MLX Muon finisher; "
                           "wd mechanism-strength lambda*sum(gamma) << 1 (else the "
                           "adamc_wd_lr_equilibrium_v1 regime takes over)"),
            "vehicle": ["softmax_of_sdf_levelset_witness", "mod32cap", "l7_lineage", "v9_cgauge_*"],
            "measurement_axis": ["predicted", "measured_existing_checkpoints"],
            "t4_preregistration": ("AutoClip reversal mechanism = lagged norm-target with "
                                   "window-memory floor (spike-guard + tau-handoff FALSIFIED from "
                                   "telemetry); discriminators S1 window-sweep / S2 percentile / "
                                   "S3 EoS wobble / S4 causal rebase of armB@ep75 — memo T4"),
            "note": ("prescriptive half (chosen invariant + hold-it engineering) is DESIGN, "
                     "ticketed default-OFF; nothing armed; verdicts through the real decode"),
        },
        units_in={"gamma": "learning_rate", "lam": "weight_decay", "n_m": "tensor_dims",
                  "weight_norm": "l2", "tau": "softmax_temperature"},
        units_out={"eta_rel": "relative_step_per_iter", "kappa": "inverse_temperature",
                   "channels": "weight_norm_squared_units"},
        empirical_anchors=(anchor_muon, anchor_head, anchor_rate),
        predicted_vs_empirical_residual={
            "muon_norm_shrink_measured": 0.0,
            "head_gauge_drift_measured": 0.0,
            "rate_coupling_slope_vs_naive_theory": 0.79,  # 1.0 - ~0.21 measured => shape shortfall
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.adaptivization_tickets_20260715",  # eta_rel pin + autoclip-window tickets
            ".omx/research/optimizer_dynamics_followup_20260715.md",
        ),
        canonical_producers=(
            ".omx/research/optimizer_dynamics_followup_20260715.md",
            ".omx/research/adamc_muonc_optimizer_research_20260715.md",
        ),
        provenance=build_provenance_for_predicted(
            model_id="inr_weight_norm_radial_ode.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_inr_weight_norm_radial_ode_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins).

    EQUATIONS leg of the optimizer-dynamics follow-up; DSL leg = the eta_rel-pin +
    autoclip-window adaptivization tickets; DAG leg = FEED row in the sub015 DAG."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_inr_weight_norm_radial_ode_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="inr_weight_norm_radial_ode_20260715 (optimizer-dynamics follow-up: T1 measured "
              "norm trajectories falsify P3 sign; T2 derived law + chosen invariant; T3 measured "
              "rate coupling; T4 mechanism resolution carried in domain_of_validity)",
    )
    return eq
