"""``tac.phase3`` — Phase 3 joint scorer-renderer-codec scaffold (DESIGN-ONLY).

This package implements the **Phase 3 joint training of (auxiliary scorer +
renderer + codec)** under the Tishby Information Bottleneck Lagrangian per
the coherence council §1 (memo: ``feedback_grand_council_portfolio_coherence_journal_grade_20260509.md``).

  L_IB(Z; θ_enc, θ_aux) = I(X; Z)  -  β_seg · I(Z; Y_seg)  -  β_pose · I(Z; Y_pose)

with:
  - **X** = the comma video (1200 frames, 600 pairs)
  - **Z** = the encoded latent + decoder weights + side-info (everything in archive.zip)
  - **Y_seg, Y_pose** = contest-scorer outputs (NEVER the contest scorer itself
    at eval time — see CLAUDE.md ``check_no_scorer_load_at_inflate``)
  - **θ_aux** = Hinton-distilled (T=2.0) auxiliary scorer (TRAINING-ONLY;
    replaced with frozen contest scorer at eval time)
  - **β_seg, β_pose** = dual variables coordinating the rate-distortion-architecture tradeoff

Phase 3 is the FULL L_IB; Phase 2 is L_IB with savings(ρ) term and W₂-distortion
in I(Z;Y_seg); Phase 1 is L_IB with the i.i.d. and frozen-scorer simplifications.

CLAUDE.md compliance — non-negotiable
-------------------------------------

1. **No score claims at module level**: all Phase 3 score-band predictions
   tagged ``[predicted; Phase 3 council; conditional on Phase 2 landing 0.142]``.
   The closed-form S_floor = 0.131 ± 0.013 is a Bayesian posterior across 7
   theorem sources (see Phase 2 floor REBASELINE memo), NOT a contest-eval claim.
2. **Strict-scorer-rule honored**: the auxiliary scorer θ_aux is TRAINING-ONLY.
   At eval time θ_aux is REPLACED with the frozen contest scorer. The Hinton
   distillation gap (Hinton 2014 §3, T=2.0) is recorded as ``distillation_gap_estimate``
   in the trainer artifacts; gap target ≤ 3%.
3. **EMA at 0.997 with snapshot+restore**: every trainer instantiates EMA via
   ``tac.training.EMA`` (decay 0.997 for renderer + aux scorer weights;
   decay 0.99 for VQ-VAE codebooks per CLAUDE.md exception).
4. **eval_roundtrip mandatory**: every loss path simulates the contest's
   inflate roundtrip (384→874→uint8→384) so the proxy-auth gap stays bounded.
5. **Never MPS authoritative**: trainer raises on ``--device mps``; any MPS
   forward pass is tagged ``[macOS-CPU advisory only]`` per CLAUDE.md
   ``MPS-NOISE`` rule.
6. **Inflate.py LOC budget ≤ 200**: per CLAUDE.md HNeRV parity discipline L4.
   Current scaffold inflate is well under (see ``tac.phase3.inflate``).
7. **Substrate-engineering exception**: Phase 3 may engineer the substrate
   jointly with the codec ONLY because the joint training is anchored on a
   frozen-eval-time auxiliary scorer that is Hinton-distilled (T=2.0) from the
   contest scorer. This is the substrate-vs-codec meta-pattern's principled
   resolution (see ``feedback_substrate_vs_codec_composition_meta_pattern_20260508.md``).

DESIGN-ONLY STATUS — gates dispatch decision
--------------------------------------------

This scaffold is **SCAFFOLD-ONLY**. It carries:
  - ``DISPATCH_READY = False``
  - ``REQUIRES_OPERATOR_APPROVAL = True``
  - ``GATED_ON_PHASE2_ANCHOR = True`` (Phase 2 must land 0.142 [contest-CUDA verified] first)
  - ``GATED_ON_AAF68FC7_VERDICT = True`` (in-flight adversarial-review subagent must clear)

The Phase 3 dispatch decision REQUIRES a fresh inner-ten council with:
  - Phase 2 [contest-CUDA] anchor verified at ≤ 0.142
  - Phase 3 distillation-gap estimate ≤ 3% (Hinton 2014 §3 verified)
  - $600-$1200 GPU budget approval
  - Operator submission policy decision tree updated

Public API
----------

.. autosummary::
   :nosignatures:

   PHASE3_VERSION
   PHASE3_PROVENANCE
   JointScorerRendererCodecConfig
   PHASE3_PREDICTED_BAND_TAG
"""
from __future__ import annotations

PHASE3_VERSION = "0.1.0-phase3-scaffold-design-only"
"""Semantic version for the package; bumped each public-API mutation.

Phase 3 scaffold (this version): DESIGN-ONLY modules + smoke tests + lane reg.
NO empirical score claim is associated with this version. Promotion to 0.2.0
requires:

  1. Phase 2 landing 0.142 [contest-CUDA verified]
  2. Hinton distillation gap ≤ 3% verified on T10 (IB co-trained scorer)
  3. Operator approval of $600-$1200 GPU budget
  4. Fresh inner-ten council dispatch-readiness review
"""

PHASE3_PREDICTED_BAND_TAG = "[predicted; Phase 3 council; multi-source aggregated; conditional on Phase 2 landing 0.142]"
"""Tag every Phase 3 score-band reference with this string. Per CLAUDE.md
``forbidden_score_claims``, no Phase 3 score number may appear in any
artifact without this tag attached.

Predicted band per Phase 2 floor REBASELINE memo + coherence council §1:
  0.115-0.130 (NeurIPS-grade target)
  median ~0.124 (coherence council §1.1.3)

These bands are JOINT-POSTERIOR estimates from 7 theorem sources (Shannon R(D),
Berger 1971, Tishby IB, Fridrich √n, Ballé EB, MacKay MDL, Volterra
super-additive). They are NOT contest-eval claims. They become contest-eval
claims ONLY when ``contest_auth_eval.py --device cuda`` produces an exit-0
artifact with a real score number.
"""

PHASE3_PROVENANCE = {
    "schema_version": 1,
    "version": PHASE3_VERSION,
    "phase": "3_scaffold_design_only",
    "tracks": [
        "phase3_joint_scorer_renderer_codec_ib_lagrangian",
    ],
    "empirical_anchor": None,
    "predicted_score_band": PHASE3_PREDICTED_BAND_TAG,
    "council_memo_refs": [
        "fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md",
        "feedback_grand_council_portfolio_coherence_journal_grade_20260509.md",
        "feedback_grand_council_fields_medal_phase2_floor_REBASELINE_with_integration_discipline_20260509.md",
        "feedback_paradigm_dezeta_phase2_architectural_launch_20260509.md",
    ],
    "compliance_tags": [
        "ib_lagrangian_tishby_1999",
        "berger_joint_source_rd_1971",
        "hinton_distillation_T_2_0",
        "ballè_hyperprior_2018",
        "ema_0p997_snapshot_restore",
        "eval_roundtrip_true",
        "no_mps_authoritative",
        "scorer_at_eval_frozen_contest",
        "substrate_engineering_exception_principled",
        "score_tag_predicted_only",
    ],
    "dispatch_readiness": {
        "DISPATCH_READY": False,
        "REQUIRES_OPERATOR_APPROVAL": True,
        "GATED_ON_PHASE2_ANCHOR": True,
        "GATED_ON_AAF68FC7_VERDICT": True,
        "GPU_BUDGET_USD": "$600-$1200",
        "RUNTIME_ESTIMATE": "2 weeks (Phase 3 dispatch wall-clock)",
    },
    "archive_grammar_fields": {
        "representation_name": "Phase 3 IB-Lagrangian joint scorer-renderer-codec",
        "target_modes": ["contest_exact_eval"],
        "source_artifact": "experiments/results/A1_canonical/ + frozen contest scorer",
        "archive_builder": "tools/build_phase3_archive.py (FUTURE — not yet landed)",
        "inflate_consumer": "src/tac/phase3/inflate.py (≤200 LOC)",
        "runtime_manifest": "submissions/phase3_robust/runtime_manifest.json (FUTURE)",
        "changed_payload_paths": [
            "latent.bin",
            "decoder.bin",
            "aux_scorer_distill_gap.json",
        ],
        "old_new_sha256s": "tracked in build_manifest.json per dispatch",
    },
    "six_hook_wireins": {
        "sensitivity_map": "phase3_joint_score_aware axis declared",
        "pareto": "phase3_aux_scorer_distillation_gap constraint",
        "bit_allocator": "phase3_bit_allocation hook",
        "cathedral_autopilot": "catalog row added (status DESIGN-ONLY)",
        "continual_learning": "posterior update on Phase 3 anchor",
        "probe_disambiguator": "Phase 3 vs Phase 2 disambiguator",
    },
}

# Lazy imports keep cold-import cheap.
from tac.phase3.joint_scorer_renderer_codec import (  # noqa: E402
    JointScorerRendererCodecConfig,
    JointScorerRendererCodecScaffold,
    Phase3DispatchGate,
    Phase3DispatchGateError,
    phase3_lagrangian_form,
    phase3_distillation_gap_estimate,
)

__all__ = [
    "PHASE3_VERSION",
    "PHASE3_PROVENANCE",
    "PHASE3_PREDICTED_BAND_TAG",
    "JointScorerRendererCodecConfig",
    "JointScorerRendererCodecScaffold",
    "Phase3DispatchGate",
    "Phase3DispatchGateError",
    "phase3_lagrangian_form",
    "phase3_distillation_gap_estimate",
]
