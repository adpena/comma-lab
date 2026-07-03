# SPDX-License-Identifier: MIT
"""Canonical equations: LEVER-4 S_R reachability + Muon finishing-stage schedule (triality-sync).

Two triality-sync findings (built + DAG-recorded, now formalized so the equations leg AGREES with
the DAG FEEDs 03n-03q and the DSL gauge charts MarginSaliencyGauge / MuonMomentumGauge / MuonLRGauge):

  1. ``margin_saliency_reachability_replaces_texture_proxy_v1`` (f99a3863a; FEED-03n/03p) --
     LEVER-4's UNIWARD texture multiplier ``1/(1+beta*tex)`` is MEASURED INERT (texture is orthogonal
     to through-R detector reachability: Pearson -0.033 vs S_R, top-5% Jaccard 0.024 ~= chance 0.026;
     mildly MISDIRECTS). The exact through-R fragility-weighted margin-Jacobian
     ``S_R = |d(sum_p w_p*margin_p)/dx|``, ``w = exp(-margin/tau)`` REPLACES it ->
     ``sal = exp(-margin/tau) * S_R_norm`` (3.0x concentrated on the fragile margin band;
     theta-INDEPENDENT -> cached alongside ``margins``).

  2. ``muon_finisher_schedule_warmstart_and_lr_anneal_v1`` (cba2e4375; FEED-03o/03q) -- the
     AdamW->Muon transition levers. COLD momentum start (fresh optim.Muon zero buffer) MEASURED a
     +0.000357 d_seg SPIKE at the boundary (sister thetastar run, ep750 saddle-gate) = cold-start
     thrash; warm-start (seed Muon ``v`` from the outgoing AdamW ``m``) removes it. FLAT Muon LR
     plateaus (Newton-Schulz fixes update MAGNITUDE so it cannot self-reduce the step near a minimum,
     river-valley 2606.21514); cosine-DECAY to a floor (``--muon-lr-final-frac``) fixes it.

means != ends: both are 0-archive-byte train-time/optimizer levers whose NET n600 d_seg is #205-gated
(the byte-closed A/B is the #205 arm #270). Pointer 0.19110 UNMOVED. These equations record MEANS +
the MEASURED grounding numbers; a score claim requires the byte-closed exact row.
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

SALIENCY_EQUATION_ID = "margin_saliency_reachability_replaces_texture_proxy_v1"
MUON_EQUATION_ID = "muon_finisher_schedule_warmstart_and_lr_anneal_v1"

_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"

# The durable DAG record of both findings (FEED-03n..03q) + the muon convergence-arm ledger.
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_MUON_ARM = ".omx/research/muon_vs_adamw_from_stage4_convergence_arm_20260622.md"
_SR_TOOL = "tools/precompute_sR_reachability.py"

# --- MEASURED grounding constants (advisory; the byte-closed net d_seg is #205-gated) --------------
# texture-proxy vs S_R alignment (top-5% Jaccard) ~= statistical chance -> texture is INERT.
TEXPROX_VS_SR_JACCARD = 0.024
TEXPROX_VS_SR_JACCARD_CHANCE = 0.026
TEXPROX_VS_SR_PEARSON = -0.033
# the cold-start Muon transition d_seg SPIKE (sister thetastar run, ep750 saddle-gate).
MUON_COLD_START_DSEG_SPIKE = 0.000357


def through_r_saliency(margin: float, s_r_norm: float, tau: float = 0.5) -> float:
    """The reachability-correct LEVER-4 saliency weight ``sal = exp(-margin/tau) * S_R_norm``.

    ``margin`` is the (signed) SegNet argmax margin at the pixel; ``s_r_norm`` is the normalized
    through-R fragility-weighted margin-Jacobian magnitude at that pixel (``S_R`` from
    ``tools/precompute_sR_reachability.py``, in [0, 1]); ``tau`` is the fragility temperature. This
    REPLACES the (measured-inert) texture multiplier ``1/(1+beta*tex)`` -- the ``w = exp(-margin/tau)``
    fragility factor is unchanged, only the multiplier is swapped texture -> S_R."""

    return math.exp(-float(margin) / float(tau)) * float(s_r_norm)


def muon_cold_start_transition_spike() -> float:
    """The MEASURED d_seg SPIKE at a COLD-start AdamW->Muon transition (+0.000357; sister thetastar
    run, ep750 saddle-gate). Warm-starting Muon ``v`` from the outgoing AdamW ``m`` removes it."""

    return MUON_COLD_START_DSEG_SPIKE


def build_margin_saliency_reachability_replaces_texture_proxy_v1() -> CanonicalEquation:
    """Build the LEVER-4 S_R-reachability-replaces-texture-proxy canonical equation (FEED-03n/03p)."""

    anchor_texture_inert = EmpiricalAnchor(
        anchor_id="texture_proxy_orthogonal_to_through_r_reachability_at_chance_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"lever": "LEVER-4_margin_saliency_uniward_multiplier",
                "multiplier": "1/(1+beta*tex)", "n_pairs": "6/16/96 advisory",
                "method": "torch-CPU autograd through the REAL R + frozen contest SegNet"},
        predicted_output={"texprox_is_inert": True},
        empirical_output={
            "texprox_vs_sR_pearson": TEXPROX_VS_SR_PEARSON,
            "texprox_vs_sR_top5pct_jaccard": TEXPROX_VS_SR_JACCARD,
            "chance_jaccard": TEXPROX_VS_SR_JACCARD_CHANCE,
            "full_lever_sR_alignment": 0.214,
            "w_only_sR_alignment": 0.205,
            "texture_marginal_alignment_gain": 0.009,
            "texprox_vs_grad_margin": -0.215,
            "note": "texture is orthogonal to through-R reachability (~chance) AND mildly misdirects",
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="torch_cpu_autograd_through_R_frozen_segnet_n6_n96_advisory",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria="re-measure texprox-vs-S_R alignment at n600 through R",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_sR_replacement = EmpiricalAnchor(
        anchor_id="sR_reachability_replaces_texture_multiplier_net_dseg_205_gated_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"replacement": "sal = exp(-margin/tau) * S_R_norm",
                "S_R": "|d(sum_p w_p*margin_p)/dx|, w=exp(-margin/tau)",
                "theta_independent": True, "cached_in": "gt-cache alongside margins"},
        predicted_output={"sal": through_r_saliency(margin=0.0, s_r_norm=1.0, tau=0.5)},
        empirical_output={
            "sR_concentration_on_fragile_band": "3.0x",
            "sR_vs_grad_margin": 0.272,
            "sR_vs_margin": -0.323,
            "net_n600_dseg": "205-gated (byte-closed A/B arm #270); expect MODEST refinement",
        },
        residual=0.0,
        source_artifact=_SR_TOOL,
        measurement_method="205_gated_byte_closed_ab_arm",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SR_TOOL,
            reactivation_criteria="#205 arm: --margin-saliency-reachability ON vs texture, byte-closed n600 d_seg",
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=SALIENCY_EQUATION_ID,
        name="LEVER-4 through-R reachability S_R replaces the inert texture proxy",
        one_line_summary=(
            "texture multiplier 1/(1+b*tex) is INERT (texprox-vs-S_R Jaccard 0.024 ~= chance 0.026); "
            "S_R = |d(sum w*margin)/dx| replaces it -> sal = exp(-margin/tau)*S_R_norm"
        ),
        latex_form=(
            r"S_R = \left|\frac{\partial \sum_p w_p\, m_p}{\partial x}\right|,\ "
            r"w = e^{-m/\tau}\ \Rightarrow\ \mathrm{sal} = e^{-m/\tau}\cdot \hat S_R"
            r"\quad(\mathrm{texture}\ \perp\ \mathrm{reachability},\ J{\approx}0.024{\approx}\mathrm{chance})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.margin_saliency_reachability_and_muon_finisher_20260703:through_r_saliency"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": "LEVER-4_margin_saliency_multiplier",
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": "the texture-inert finding is measured (advisory); the net d_seg is #205-gated",
        },
        units_in={"margin": "segnet_argmax_logit_margin", "s_r_norm": "normalized_through_R_jacobian_0_1",
                  "tau": "fragility_temperature"},
        units_out={"sal": "dimensionless_saliency_weight"},
        empirical_anchors=(anchor_texture_inert, anchor_sR_replacement),
        predicted_vs_empirical_residual={
            "texprox_vs_sR_alignment_at_chance": 0.0,
        },
        last_calibration_utc="2026-07-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
        ),
        canonical_producers=(
            "tools/precompute_sR_reachability.py",
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="margin_saliency_reachability_replaces_texture_proxy.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


def build_muon_finisher_schedule_warmstart_and_lr_anneal_v1() -> CanonicalEquation:
    """Build the Muon finishing-stage schedule (warm-start + LR anneal) canonical equation
    (FEED-03o/03q; the +0.000357 cold-start transition spike is the measured grounding)."""

    anchor_cold_spike = EmpiricalAnchor(
        anchor_id="muon_cold_start_transition_dseg_spike_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"transition": "AdamW->Muon finisher boundary", "init": "cold (fresh optim.Muon zero buffer)",
                "run": "sister thetastar", "epoch": 750, "gate": "saddle-gate"},
        predicted_output={"cold_start_dseg_spike": muon_cold_start_transition_spike()},
        empirical_output={"dseg_spike": MUON_COLD_START_DSEG_SPIKE,
                          "note": "cold first orthogonalized step = wild unit-norm dir from one noisy "
                                  "gradient -> boundary thrash; warm-start (seed v from AdamW m) removes it"},
        residual=0.0,
        source_artifact=_MUON_ARM,
        measurement_method="saddle_gate_measured_dseg_at_muon_transition",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MUON_ARM,
            reactivation_criteria="re-measure the AdamW->Muon transition d_seg spike on the #205 witness at n600",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_schedule = EmpiricalAnchor(
        anchor_id="muon_finisher_schedule_warmstart_and_lr_anneal_net_dseg_205_gated_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"warm_start": "--muon-warm-start-momentum (seed v from AdamW m)",
                "lr_anneal": "--muon-lr-final-frac 0.1 (cosine 0.002->2e-4 across the Muon span)",
                "grounding": "flat LR cannot self-reduce step (NS fixes magnitude) -> plateaus (2606.21514)"},
        predicted_output={"verdict": "warm_start kills the +0.000357 spike; anneal escapes the flat-LR plateau"},
        empirical_output={"byte_identical_off": True,
                          "net_n600_dseg": "205-gated (arm #270, GO-gated); advisory until byte-closed"},
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="205_gated_byte_closed_ab_arm",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria="#205 improved-Muon arm #270: warm-start + lr-anneal ON, byte-closed n600 d_seg",
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=MUON_EQUATION_ID,
        name="Muon finishing-stage schedule: warm-start momentum + cosine LR anneal",
        one_line_summary=(
            "cold Muon start spikes d_seg +0.000357 at the transition (warm-start v from AdamW m "
            "removes it); flat Muon LR plateaus (NS fixes magnitude) -> cosine-decay to a floor"
        ),
        latex_form=(
            r"\Delta d_{seg}^{\text{cold}} = +3.57\times10^{-4}\ \text{(warm-start}\to 0)\ ;\ "
            r"\eta_{Muon}(t) = \eta_0\big[f + (1-f)\tfrac{1+\cos\pi t}{2}\big],\ f=0.1"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.margin_saliency_reachability_and_muon_finisher_20260703:muon_cold_start_transition_spike"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "stage": "AdamW->Muon finisher transition (finishing-stage schedule)",
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": "the +0.000357 cold-start spike is measured; the warm-start/anneal net d_seg is #205-gated",
        },
        units_in={},
        units_out={"cold_start_dseg_spike": "segnet_argmax_dseg"},
        empirical_anchors=(anchor_cold_spike, anchor_schedule),
        predicted_vs_empirical_residual={
            "muon_cold_start_transition_spike": 0.0,
        },
        last_calibration_utc="2026-07-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
        ),
        canonical_producers=(
            "experiments/train_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="muon_finisher_schedule_warmstart_and_lr_anneal.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


__all__ = [
    "MUON_COLD_START_DSEG_SPIKE",
    "MUON_EQUATION_ID",
    "SALIENCY_EQUATION_ID",
    "TEXPROX_VS_SR_JACCARD",
    "TEXPROX_VS_SR_JACCARD_CHANCE",
    "TEXPROX_VS_SR_PEARSON",
    "build_margin_saliency_reachability_replaces_texture_proxy_v1",
    "build_muon_finisher_schedule_warmstart_and_lr_anneal_v1",
    "muon_cold_start_transition_spike",
    "through_r_saliency",
]
