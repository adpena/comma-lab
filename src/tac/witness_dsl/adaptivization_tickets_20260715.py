# SPDX-License-Identifier: MIT
"""Typed ADAPTIVIZATION-QUEUED tickets for the V9-CGauge constants-are-poison audit (2026-07-15).

Operator directive (verbatim): "Constants are poison unless truly optimal across all
surfaces and all curriculum we can do much better" + "This entire system can be modeled
and measured and optimized using our system of equations and differential equations and
all deep math and geometry."

This module is the DSL-native carrier for the constants audit's ADAPTIVIZATION-QUEUED
class (memory ``constants_are_poison_unless_optimal_across_surfaces_curriculum_20260715``):
every sealed scalar in the live ``v9_cgauge_ideal_mod19`` / ``c1_optimal_form`` argv that
is NOT proven cross-stage-optimal and NOT yet compiled from a state-dependent law gets a
typed ticket here — with the derived LAW it should become, the built implementation (when
one exists), and the named unlock. Tickets, never hand flags: no ticket in this module
emits argv; folding one into a launch config requires a Lever factory + the cited unlock.

Audit memo: .omx/research/v9_missing_signal_constants_audit_20260715.md
CONTAINMENT: pure data ($0); research_only; no dispatch, no run mutation, no argv.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class AdaptivizationTicket:
    """One sealed constant queued to become a state-dependent law."""

    constant: str                  # the launch scalar (flag or manifest key)
    current_value: str             # the sealed value in the live config (as emitted)
    poison_evidence: str           # MEASURED/DERIVED evidence the scalar is not cross-stage optimal
    law: str                       # the derived state-dependent law (formula or named controller)
    law_source: str                # equation id / module / memo that holds the derivation
    built_implementation: str      # existing code path if the law is already built ("" if none)
    unlock: str                    # what is owed before the law can replace the scalar
    joint_wallclock_axis: str      # which factor it moves: "epochs_to_target" | "sec_per_ep" | "both" | "neither(score)"


# Ordered by expected |d(wall-clock-to-target)| under the joint objective
# (feedback_wallclock_to_target_joint_objective_drift_ok_if_gradient_good_20260715).
ADAPTIVIZATION_TICKETS: tuple[AdaptivizationTicket, ...] = (
    AdaptivizationTicket(
        constant="--grad-clip",
        current_value="0.5",
        poison_evidence=(
            "MEASURED (C0 run levelset_n600_witness_20260715T095030Z, grad_clip_activation rows "
            "ep1-39): global frac_clipped=1.0 at EVERY accum step, norm_mean~5.9-6.2, norm_max up to "
            "17.5 vs threshold 0.5 — the clip is SATURATED 100% of CE-stage steps, so the effective "
            "step is lr*0.5/gnorm (~12x below the scheduled LR) and the LR cosine no longer controls "
            "the actual step size; the constant silently re-parametrizes the whole descent clock. "
            "Never re-validated at l7/Muon/finisher stages."
        ),
        law=(
            "clip_t = percentile_p(gnorm history, window w) (AutoClip, arXiv:2007.14469; z-score "
            "spike variant ZClip arXiv:2504.02507) as the cheap static-free approximation; the "
            "principled form is the #500 Fisher trust region (step bounded in G_dec, "
            "D_KL <= delta per class) which subsumes clipping entirely."
        ),
        law_source=(
            "categorical_fisher_natural_trust_region_20260715 (src/tac/canonical_equations/) + "
            "AutoClip/ZClip (online, corpus-gap confirmed 2026-07-15)"
        ),
        built_implementation="",
        unlock=(
            "n24 A/B (clip 0.5 control vs percentile law) on gradient-quality + flicker telemetry "
            "per the relaxed-identity directive; then a Lever factory (never a hand flag)"
        ),
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--lr-anneal-epochs / --lr-hold-frac",
        current_value="1000 / 1.0",
        poison_evidence=(
            "DERIVED-AT-CONFIG as a control-reproduction pin (v6.4 LR-pin: reproduce the mod32cap "
            "control's LR(ep) bit-identically on [1,726]) — a provenance choice, not an optimum. "
            "The live config advances tau by EVENTS (--tau-advance-mode event) while LR anneals on a "
            "fixed 1000-epoch clock: when a rung fires early/late the LR is wrong for the rung."
        ),
        law=(
            "eta(k) proportional to the tau-octave fraction k/N (LR follows the continuation "
            "parameter, not the wall clock) — Ch.6 critical-slowing: a clock cannot slow itself."
        ),
        law_source="TauAdvanceController.lr_anneal_fraction (src/tac/witness_control/tau_advance.py)",
        built_implementation="src/tac/witness_control/tau_advance.py:lr_anneal_fraction",
        unlock="wire --lr-anneal-mode event (Lever factory) + n24 byte-identity-at-OFF proof",
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--eval-every (verdict cadence)",
        current_value="25",
        poison_evidence=(
            "MEASURED: one n600 CPU verdict costs 2555.7s (C0 ep25 witness_component_wallclock row) "
            "and inflates the hosting epoch 250s->1153s (+~36s/ep amortized, +14%). A fixed cadence "
            "spends the same 42.6 CPU-minutes whether or not d_seg plausibly moved since the last "
            "verdict; early-CE verdicts are near-zero-information, late plateau verdicts under-sample."
        ),
        law=(
            "verdict when predicted information gain >= cost: fire when the #344 NCDE-predicted "
            "|d_seg_now - d_seg_last_verdict| exceeds the verdict detection floor (or on stage "
            "events); cadence-25 retained as the fail-safe backstop cap, same EventBackstopGate "
            "pattern as the #315 start-events."
        ),
        law_source="src/tac/witness_control/ncde_trajectory.py (hit->solve detector) + event_wirings.EventBackstopGate",
        built_implementation="NCDE shadow sensor built (advisory); no verdict-cadence consumer yet",
        unlock="a --verdict-event trainer hook consuming the NCDE advisory (Lever factory)",
        joint_wallclock_axis="sec_per_ep",
    ),
    AdaptivizationTicket(
        constant="--seg-phase-advect-start-epoch",
        current_value="726",
        poison_evidence=(
            "The constants manifest itself declares it a STATIC APPROXIMATION of the label_floor "
            "detector event (N7 BUILD-OWED: --seg-phase-advect-start-event); compiled from the "
            "flicker-floor law placement, not from run state."
        ),
        law="engage T1 phase-advection when the label_floor detector fires (flicker-floor Law-5 placement)",
        law_source="gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1 + costate label_floor_detector",
        built_implementation="label_floor detector exists as costate sensor; trainer start-event hook OWED",
        unlock="N7: --seg-phase-advect-start-event trainer hook (mirror of the three live #315 events)",
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--hosc-beta-end / --hosc-beta-anneal",
        current_value="3.177 / linear (clock)",
        poison_evidence=(
            "DERIVED-AT-CONFIG as a slope-preserving rephase of the control's beta(ep) on [1,726] "
            "(v6.3 MAJOR-2(i)) — control-preservation, not optimality; the StepNativeActivation ISO "
            "arm (34.2% duty) tests endpoint 8.0, so the endpoint is explicitly contested. Beta on a "
            "wall-clock line while tau advances by events de-couples the two continuations that the "
            "Ch.4 Deriv-3 co-anneal law says share ONE Gamma-limit."
        ),
        law="beta interpolated on the tau-octave fraction k/N (co-annealed with the rung ladder)",
        law_source="TauAdvanceController.hosc_beta_for_epoch (src/tac/witness_control/tau_advance.py)",
        built_implementation="src/tac/witness_control/tau_advance.py:hosc_beta_for_epoch",
        unlock="event-mode beta wire-in Lever + the step_iso arm verdict (endpoint contest)",
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--eikonal-weight / eikonal viscosity eps",
        current_value="0.01->0.05 (rung-coupled) / fixed eps, adaptive OFF",
        poison_evidence=(
            "The weight already follows the rung law (eikonal_retention_couples_to_tau_rung_v1) but "
            "the viscosity eps that keeps the flow WELL-POSED is a fixed floor: the #318 von-Neumann "
            "analysis proves the flat-margin annulus mode is a backward heat equation unless "
            "eps >= |c_a|*sqrt(eta*lambda_eik/8); the ep110 runaway (|grad m|~2070) is the measured "
            "bifurcation. Adaptive-eps is BUILT + byte-identical-at-OFF but in NO live config."
        ),
        law="eps(t) = clamp(|c_a(t)|*sqrt(eta(t)*lambda_eik(t)/8)*(1+margin), eps_floor, eps_upper)",
        law_source="adaptive_eps_cfl_edge_tracking_v1 (src/tac/canonical_equations/adaptive_eps_cfl_edge_tracking_20260705.py)",
        built_implementation="trainer _adaptive_visco_eps + --eikonal-viscosity-adaptive (default OFF)",
        unlock="bounded n24 stability A/B (the '8' is FORMALIZATION_PENDING; floor-clamp inertness flagged)",
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--w-pose",
        current_value="1.0",
        poison_evidence=(
            "The costate law gives the EXACT state-dependent multiplier lambda_pose = 5/sqrt(10*d_pose) "
            "(costate_lambda_marginal_ds_v1); a scalar 1.0 is only correct at one d_pose. Low risk today "
            "(pose is terminal-gated / PoseBlindComputeGate) but the terminal joint finish inherits it."
        ),
        law="lambda_pose(t) = 5/sqrt(10*d_pose(t)) applied at the pose-finish engage",
        law_source="src/tac/canonical_equations/costate_lambda_marginal_ds_20260705.py",
        built_implementation="equation registered; no trainer consumer",
        unlock="pose-finish engage-time Lever computing the multiplier from the live verdict d_pose",
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--ema-decay",
        current_value="0.997",
        poison_evidence=(
            "ANCESTOR-VEHICLE anchor (Quantizr 0.997, HNeRV/PR95 lineage) — per the L18 ancestor rule "
            "numbers do not transfer as optima; never re-measured on the witness vehicle. The Polyak "
            "finisher law already exists precisely because short-horizon EMA loses to a uniform tail "
            "mean at the constant-tau* turnpike (muon_finisher_schedule_warmstart_and_lr_anneal_v1)."
        ),
        law=(
            "stage-dependent averaging: EMA(0.997-class) during descent; uniform Polyak-Ruppert tail "
            "mean over ~0.2*stage_window at the turnpike (already compiled: polyak start 2546)"
        ),
        law_source="src/tac/witness_control/polyak_finisher.py + muon_finisher_schedule_warmstart_and_lr_anneal_v1",
        built_implementation="polyak finisher BUILT default-off (--polyak-finisher-arm)",
        unlock="arm the finisher (already a compiled constant) + a decay cross-stage sweep ticket",
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--accum-pairs / --micro-batch-pairs",
        current_value="8 / 1",
        poison_evidence=(
            "micro-batch-pairs is PINNED to 1 by the S_R fail-close (batched LEVER-4 twin does not "
            "consume S_R — the named fallen-crack), not by measurement; D15 records a 2-4x sec/ep "
            "lever. Under the 2026-07-15 relaxed-identity directive B>1 is admissible in principle "
            "(batched-twin functional tolerances ARE the gradient bar). accum=8 fixes 75 optimizer "
            "steps/epoch at n600 across ALL stages — never swept jointly with clip/LR."
        ),
        law=(
            "B(V) chosen by the throughput authority under the joint objective: maximize measured "
            "(epochs_to_target x sec_per_ep)^-1 subject to gradient-quality + no-flicker telemetry"
        ),
        law_source="throughput_authority_policy_20260714 + D15 deferral row",
        built_implementation="batched micro-batch path exists; S_R consumer missing",
        unlock="#447-adjacent: batched LEVER-4 S_R consumer, then the D15 A/B",
        joint_wallclock_axis="both",
    ),
    AdaptivizationTicket(
        constant="--ladder-{movable,lane}-{r0,birth,hold,anneal}",
        current_value="movable r0=0.2252 birth=60 anneal=200; lane r0=2.0 birth=80 anneal=260",
        poison_evidence=(
            "Hand-placed island-birth schedule; the continuation reframe says island birth is a "
            "saddle-node bifurcation in the class-occupancy order parameter whose critical dilation-"
            "lambda is COMPUTABLE — the birth lever should sit AT the computed critical value, not at "
            "a swept epoch/r0 pair (curriculum_is_continuation_instabilities_are_bifurcations_20260714)."
        ),
        law=(
            "derive the low-dim class-occupancy order-parameter ODE (from #318 stability + #344 NCDE "
            "+ #180 Morse-Smale), continue in lambda, place birth at the computed fold point"
        ),
        law_source="curriculum_is_continuation memo + #318/#344/#180 reduced-order pieces",
        built_implementation="",
        unlock="the reduced-order model derivation (continuator eats low-dim only)",
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--muon-lr / --muon-momentum / --muon-ns-steps",
        current_value="0.002 / 0.95 / 5",
        poison_evidence=(
            "Muon internals are PR95-lineage constants, unmapped in the lever registry (no LawRef "
            "owner); the finishing-schedule treatment (#217/#270: anneal LR + warm-start momentum) is "
            "the measured direction, with --muon-warm-start-momentum present in argv surface but the "
            "leap-residual law still design-only."
        ),
        law="Muon finishing schedule: warm-started momentum + lr-final-frac anneal (#270 restart law)",
        law_source=".omx/research/muonh_manifold_muon_dig_20260713.md + #270 GO row",
        built_implementation="--muon-warm-start-momentum / --muon-lr-final-frac flags exist",
        unlock="registry LawRef ownership + cross-stage sweep at the Muon stage boundary",
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--muon-weight-decay",
        current_value="None => --weight-decay = 1e-4",
        poison_evidence=(
            "SOURCE-VERIFIED (adamc/muonc research 2026-07-15, memo "
            ".omx/research/adamc_muonc_optimizer_research_20260715.md §2): mlx.optimizers.Muon "
            "applies weight decay COUPLED — added to the raw gradient BEFORE momentum + "
            "Newton-Schulz orthogonalization (gradient += wd*parameter) — NOT decoupled. At "
            "wd=1e-4 the term enters 3-4 orders below the measured raw gradient scale (C0 band "
            "5.9-17.5) and NS re-normalizes the update anyway => the Muon group's wd is "
            "effectively INERT: the finisher stage that polishes the shipped EMA shadow has NO "
            "weight-norm control (undamped ||W||^2 random-walk growth ~ lr_t^2 per step; the "
            "lr-final-frac 0.1 anneal only shrinks the increments). Our "
            "build_muon_finisher_optimizer docstring mis-labeled it 'Decoupled' (fixed same "
            "landing). Defazio arXiv:2506.02285 §4.1 is the exact failure class: coupled decay "
            "through a normalizing preconditioner loses the norm-damping role."
        ),
        law=(
            "decoupled Muon-group wd (Moonlight form, arXiv:2502.16982: W <- W*(1-lr_t*wd) - "
            "lr_t*NS(m)) with the AdamC/Chou schedule correction lambda_hat_t = "
            "lambda*lr_t/lr_max_muon (arXiv:2506.02285 Alg.1 / arXiv:2512.08217 ScionC) => "
            "steady-state ||W|| = sqrt(lr_max/(2*lambda))*||u|| schedule-independent; for Muon "
            "the derivation is near-exact (NS-orthonormalized update norm is weight-independent "
            "by construction). 'MuonC' is NOT a named optimizer in the literature (2026-07 "
            "sweep); this ticket IS the honest referent."
        ),
        law_source=(
            "adamc_wd_lr_equilibrium_v1 (src/tac/canonical_equations/"
            "adamc_wd_lr_equilibrium_20260715.py) + arXiv:2506.02285 + 2512.08217 + 2502.16982"
        ),
        built_implementation=(
            "TRUNK half only: --weight-decay-corrected (levelset trainer, per-epoch AdamW "
            "opt.weight_decay = lambda*lr_t/lr_max; DSL curriculum_dsl.CorrectedWeightDecay). "
            "Muon half: NONE — scaling the existing coupled wd would arm a no-op (#417 "
            "counted-but-inert fake), so it is deliberately NOT wired."
        ),
        unlock=(
            "P3 RESOLVED 2026-07-15 ($0, memo .omx/research/optimizer_dynamics_followup_20260715"
            ".md T1 + equation inr_weight_norm_radial_ode_v1): Muon-group norms did NOT random-"
            "walk-grow — they SHRANK 0.9-28.7% across the mod32cap finisher (film -28.7%), "
            "GRADIENT-driven (radial:diffusion:wd = 7:1:0.05; mean inward cosine 0.0085). "
            "Consequence: decoupled shrink-to-zero wd is the WRONG control (adds force in the "
            "measured drift direction); promotion path is SUPERSEDED by the --muon-lr eta_rel-"
            "pin / restoring-decay ticket below. This ticket stays open ONLY for the corrected-"
            "lambda law's bookkeeping value if a decoupled path is ever built for other reasons"
        ),
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--muon-lr",
        current_value="2e-3 flat (mod32cap lineage) / --muon-lr-final-frac default 1.0",
        poison_evidence=(
            "MEASURED ($0, existing stage checkpoints; memo .omx/research/"
            "optimizer_dynamics_followup_20260715.md T1, equation inr_weight_norm_radial_ode_v1 "
            "anchor muon_finisher_norm_shrink_hidden_lr_increase_measured_20260715): the NS "
            "update norm is weight-independent, so eta_rel = ||u||/||W(t)|| is a STATE variable; "
            "with flat muon lr the mod32cap finisher ran a hidden per-layer LR INCREASE x1.40 "
            "(film) / x1.09-1.22 (hidden) as norms shrank — self-accelerating, and it COMPOUNDS "
            "multiplicatively with any --muon-lr-final-frac anneal (0.1 anneal x 1.4 drift = "
            "x0.14 net, not x0.10). Also the static sqrt(max(1,n/m)) RMS rule gives film 2.0-2.8x "
            "the relative step of square hidden layers — a per-layer LR ratio nobody chose. "
            "||W|| is SPECTRAL content for sin/hosc rows => the drift is also an NTK band shift."
        ),
        law=(
            "hold the CHOSEN invariant (full-pipeline co-design authority 2026-07-15): either "
            "(a) eta_rel pin — gamma_muon,t per tensor proportional to ||W(t)||/||W(t0)|| "
            "(relative step stationary), or (b) row-norm projection at ckpt cadence onto the "
            "designed spectral profile ||w||* = min(k_need, k_R)/omega (band edge at the R "
            "cutoff — any norm above is strictly dominated: pays bits AND aliases), or (c) "
            "restoring decay -lambda*(||W||-||W||*)*W-hat. Candidates ranked by the staged A/B, "
            "not by publication provenance."
        ),
        law_source=(
            "inr_weight_norm_radial_ode_v1 (src/tac/canonical_equations/"
            "inr_weight_norm_radial_ode_20260715.py) + memo T2/T2'"
        ),
        built_implementation="",
        unlock=(
            "ONE logging change first (score-neutral, defaults ON): per-tensor ||W|| live+EMA "
            "telemetry row at verdict cadence (the stream the pin/projection READS — sensors "
            "read trainer streams, no recompute); then a bounded n24 A/B pin-vs-flat at the "
            "finisher window through the governed launcher"
        ),
        joint_wallclock_axis="epochs_to_target",
    ),
    AdaptivizationTicket(
        constant="--grad-autoclip percentile/window (p=10, w=1000)",
        current_value="p10, w=1000 steps (AutoClip defaults, arXiv:2007.14469)",
        poison_evidence=(
            "MEASURED (armB telemetry, memo .omx/research/optimizer_dynamics_followup_20260715"
            ".md T4): at n24 accum8 = 3 opt steps/ep, w=1000 steps = 333 EPOCHS of memory — "
            "history never fills in-window (237/1000 at ep79), so the p10 threshold includes "
            "ep1-5 gnorms forever. Under the measured DECAYING gnorm (217.9 -> ~3), AutoClip "
            "p10 became a LAGGED NORM-TARGET (frac_clipped ~= 1.0 by construction; applied step "
            "== clip_t: 0.5 warmup -> 8.99 spike at ep5 -> floor ~2.55), holding armB's applied "
            "step 5-18x the stationary arms' — one mechanism explains BOTH the ep25 win "
            "(-10.35%) and the ep25+ reversal (overshoot/EoS near the sharpening minimum; "
            "spike-guard and tau-handoff candidates FALSIFIED from the same telemetry: 0 skips, "
            "temp/beta pinned 1.0)."
        ),
        law=(
            "window in EPOCH units: w_steps = w_ep * steps_per_ep (config-derived, not a raw "
            "step count); percentile from the pre-registered S2 sweep. Discriminators S1 "
            "(w in {100,1000,3000}: reversal epoch moves with window memory), S2 (p10 -> p2: "
            "floor drops toward fixed-clip behavior; one-knob-two-phases), S3 (post-reversal "
            "d_seg wobble variance), S4 (causal rebase: resume armB@ep75 with fixed-0.5 — "
            "descent resumes within ~10ep iff the mechanism is ongoing-step-size)"
        ),
        law_source=(
            "inr_weight_norm_radial_ode_v1 domain_of_validity.t4_preregistration + "
            "autoclip_percentile_threshold_v1 + memo T4"
        ),
        built_implementation=(
            "AutoClip law itself is built (commit c219841d8c); the epoch-unit window + "
            "percentile sweep arms are NOT wired"
        ),
        unlock=(
            "S4 rebase arm first (~$0, resume existing armB ep75 checkpoint, ~10 ep) when the "
            "507/r6 chain frees the GPU; then the >=150-ep S1/S2 arms through the governed "
            "launcher"
        ),
        joint_wallclock_axis="epochs_to_target",
    ),
)


@dataclass(frozen=True)
class AdaptivizationTicketQueue:
    """Frozen, research-only carrier (mirrors the *_policy.py containment pattern)."""

    tickets: tuple[AdaptivizationTicket, ...] = field(default=ADAPTIVIZATION_TICKETS)
    live_training_enabled: bool = False
    scorer_calls_enabled: bool = False
    paid_or_remote_dispatch_enabled: bool = False
    live_run_mutation_enabled: bool = False
    archive_or_pointer_mutation_enabled: bool = False
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if any(
            (
                self.live_training_enabled,
                self.scorer_calls_enabled,
                self.paid_or_remote_dispatch_enabled,
                self.live_run_mutation_enabled,
                self.archive_or_pointer_mutation_enabled,
            )
        ):
            raise ValueError("adaptivization tickets are data-only; no actuation authority")
        if not self.research_only or self.score_claim or self.promotion_eligible:
            raise ValueError("tickets are MEANS-only non-promotion evidence")
        names = [t.constant for t in self.tickets]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ticket constants")
        for t in self.tickets:
            if not (t.poison_evidence and t.law and t.law_source and t.unlock):
                raise ValueError(f"ticket {t.constant!r} missing a required field")
            if t.joint_wallclock_axis not in (
                "epochs_to_target", "sec_per_ep", "both", "neither(score)",
            ):
                raise ValueError(f"ticket {t.constant!r} has invalid joint_wallclock_axis")

    def compile_audit_contract(self) -> dict:
        return {
            "schema": "adaptivization_ticket_queue.v1",
            "audit_memo": ".omx/research/v9_missing_signal_constants_audit_20260715.md",
            "doctrine": "constants_are_poison_unless_optimal_across_surfaces_curriculum_20260715",
            "tickets": [asdict(t) for t in self.tickets],
        }


__all__ = ["AdaptivizationTicket", "AdaptivizationTicketQueue", "ADAPTIVIZATION_TICKETS"]
