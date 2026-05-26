# SPDX-License-Identifier: MIT
"""tac.substrates.time_traveler_l5_z6 — Time-Traveler L5 F-asymptote node Z6.

Per the Time-Traveler L5 Z6/Z7/Z8 predictive-coding world-model scoping design memo
(``.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md``,
commit ``aa412d2db``): Z6 is the FIRST sequenced Z-variant of the F-asymptote
trajectory along the scorer-relationship class-shift axis (predictive-coding
paradigm; Rao-Ballard 1999 + Atick-Redlich 1990 cooperative-receiver + Tishby IB
+ Wyner-Ziv side-info). Recommended FIRST build per Section 22 op-routable #2
because Z6 has the LOWEST engineering risk (single-layer FiLM-conditioned next-frame
predictor; ~75K param predictor budget) while still being a genuinely distinct
class-shift from existing Z3 (Balle hyperprior) / Z4 (cooperative-receiver loss
objective) / Z5 (hierarchical predictor) substrates.

Core insight (Rao-Ballard 1999 predictive-coding + Atick-Redlich 1990 cooperative-receiver)
--------------------------------------------------------------------------------------------

For stationary-ergodic driving video, the asymptotic conditional entropy
``H(latent_pair[t] | latent_pair[t-1], ego_motion[t])`` is substantially lower
than the marginal entropy ``H(latent_pair[t])`` because the predictor exploits
frame-to-frame redundancy through the ego-motion-conditioned forecast. The
FiLM-conditioned predictor (Perez 2017 + Atick-Redlich cooperative-receiver
framing) maps the ego-motion vector to per-channel (scale, shift) modulation
of a single-layer convolutional predictor; the residual ``r[t] = latent[t]
- predictor(latent[t-1], ego_motion[t])`` is what the archive stores, NOT the
absolute latent. The predictor itself is a shared prior between encoder and
decoder; the bit budget shrinks from ``H(latent)`` to ``H(residual)``.

Distinction from Z5 (sister staircase Step 3)
---------------------------------------------

Z6 differs from Z5 by SIMPLIFYING the predictor to a single FiLM-conditioned
conv block (~75K params) vs Z5's 2-3 layer hierarchical recurrent predictor.
The simplification trades a small amount of predictive power for substantially
lower engineering risk (sister Z5 L1 scaffold pattern was sister-tested but Z5
still pends Phase 2 council approval; Z6 explicitly opts for the MINIMUM
viable predictive-coding architecture as the FIRST F-asymptote-class-shift
empirical anchor).

**Hypothesis:** the FiLM-conditioned next-frame predictor reduces residual
bytes vs Z5 baseline (and Z4 baseline) enough to test the predicted band
**[0.13, 0.16]** per the design memo Section 18 Dykstra-feasibility convex-
intersection projection. That magnitude is a planning prior; not promotable
language until paired CPU/CUDA empirical anchors land per CLAUDE.md
"Apples-to-apples evidence discipline".

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` with Z6PCWM1 magic
  prefix; predictor + encoder + decoder state_dicts + latent_init +
  per-pair residuals + ego_motion buffer + meta JSON
- ``parser_section_manifest``: Z6PCWM1 header + predictor_blob +
  encoder_blob + decoder_blob + latent_init_blob + residuals_blob +
  ego_motion_blob + meta_blob; canonical Z6PCWM1_HEADER_FMT below
- ``inflate_runtime_loc_budget``: ≤120 LOC substrate-engineering waiver
  per HNeRV parity L4 explicit waiver (sister Z5 pattern at ~165 LOC)
- ``runtime_dep_closure``: torch + brotli only (HNeRV L4 ≤ 2 deps)
- ``export_format``: Z6PCWM1 monolithic single-file ``0.bin`` (fp16 + brotli)
- ``score_aware_loss``: ``Z6PredictiveCodingScoreAwareLoss`` routes through
  ``score_pair_components`` per Catalog #164; eval-roundtrip mandatory;
  cooperative-receiver Lagrangian + Rao-Ballard residual term
  (lambda_residual_entropy)
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV L7);
  FiLM predictor + auto-decoder + residual codec is substrate engineering
- ``no_op_detector_planned``: predictor weight section MUST be consumed by
  the inflate runtime — empirical detector: mutate predictor bytes via
  ``tools/verify_distinguishing_feature_byte_mutation.py``; verify decoded
  frames change. Catalog #272 distinguishing-feature integration contract
  honored: FiLM predictor IS the substrate-distinguishing primitive.

target_modes: ``contest_one_video_replay``, ``contest_generalized``,
``research_substrate``
lane_class: ``substrate_engineering``
research_only: true (Phase 1b lift implements trainer _full_main, but
score-bearing dispatch authority remains blocked until paired CPU/CUDA evidence,
lane claim custody, and exact-eval artifacts satisfy Catalog #240 + the Z6 memo
Section 19 reactivation criteria)

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Two defensible interpretations of "what dominates ΔS":

1. **Predictor-class hypothesis** — the FiLM-conditioned predictor's
   ego-motion-conditioned forecast dominates ΔS (matches Rao-Ballard 1999 +
   Atick-Redlich 1990 cooperative-receiver; substantiates the world-model
   class-shift claim).
2. **Capacity hypothesis** — the additional ~50 KB of predictor parameters
   merely adds capacity to overfit to the contest video; the
   "predictive-coding" framing is post-hoc rationalization.

The probe trains TWO variants: (a) full FiLM predictor, (b) identity
predictor (predictor_t(z_{t-1}) := z_{t-1}, no learning) with the SAME
parameter budget repurposed into the encoder/decoder. Compares auth-eval
score; if (a) beats (b) by ΔS > 0.005, predictive-coding wins. Probe at
``tools/probe_z6_predictive_coding_vs_identity_disambiguator.py`` (planned
post Phase 2 dispatch approval).

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map** — FiLM predictor gradient norm IS the per-tensor
   importance signal (∂L_full/∂θ_predictor); register
   ``sensitivity_map.time_traveler_l5_z6_v1`` post Phase 2.
2. **Pareto constraint** — adds
   ``predictor_residual_entropy ≤ ε_residual`` to the convex feasibility
   region; register ``tac.pareto.time_traveler_l5_z6_v1`` post-smoke.
3. **Bit-allocator hook** — per-pair-residual bit allocation derives
   directly from predictor forecast uncertainty (FiLM modulation
   amplitude); register ``bit_allocator.time_traveler_l5_z6_residual_v1``
   post-smoke.
4. **Cathedral autopilot dispatch hook** — recipe registered at
   ``.omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml``;
   gated by Catalog #167 smoke-before-full; ranker v2 receives
   ``literature_anchor=Rao-Ballard1999+Atick-Redlich1990`` as
   source-basis metadata only. No class-shift reward is valid until a
   paired exact anchor exists.
5. **Continual-learning posterior** — every Z6 empirical anchor seeds the
   posterior via ``posterior_update_locked`` (Catalog #128).
6. **Probe-disambiguator** — identity-predictor ablation IS the probe;
   see above.

Cross-references
----------------

- Design memo (parent):
  ``.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md``
- Sister Z5 L1 scaffold:
  ``lane_z5_predictive_coding_world_model_step3_20260514``
  + ``src/tac/substrates/z5_predictive_coding_world_model/__init__.py``
- Sister Z4 L1 scaffold:
  ``lane_z4_cooperative_receiver_loss_step2_20260514``
- HORIZON-CLASS standing directive 2026-05-16:
  ``feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516.md``
- Rao & Ballard (1999) "Predictive coding in the visual cortex: a
  functional interpretation of some extra-classical receptive-field
  effects" Nature Neuroscience 2(1):79-87
- Atick-Redlich (1990) cooperative-receiver theorem
- Perez et al. (2017) "FiLM: Visual Reasoning with a General Conditioning
  Layer" (FiLM modulation canonical reference)

Observability surface (per Catalog #305 MAX-OBSERVABILITY directive)
--------------------------------------------------------------------

This substrate honors the 6-facet observability surface:

1. **Per-layer inspection.** FiLM MLP / predictor conv / encoder / decoder
   layers each expose forward-pass observables via canonical xray-style
   hook pattern; serialized to ``experiments/results/<lane>/observability/
   per_layer/<layer_name>.jsonl``.
2. **Per-signal decomposition.** Composite metrics (``final_score =
   seg + sqrt(10*pose) + 25*rate``) decompose into constituent contributions
   per canonical ``tac.xray.per_pair_score_decomposition`` lens. FiLM
   predictor residual entropy tracked separately. Serialized to
   ``observability/score_decomposition.json`` with axis labels per CLAUDE.md
   "Apples-to-apples evidence discipline".
3. **Run-to-run diff.** Two runs of Z6 produce byte-identical reproducible
   artifacts under same ``(seed, commit_sha, upstream_snapshot_sha256)``
   tuple per Catalog #166 Modal HEAD-parity ledger + Catalog #245
   modal_call_id_ledger.
4. **Post-hoc query interface.** Run artifacts under ``experiments/results/
   <lane>/`` serialize as structured JSON/JSONL consumable without
   re-running. Continual-learning posterior queryable per (substrate, axis,
   hardware, evidence_grade) per Catalog #128 + #131 fcntl-locked discipline.
5. **Cite-chain.** Every behavior signal anchors to canonical tuple
   ``(substrate_id=time_traveler_l5_z6, commit_sha, modal_call_id,
   config_path, random_seed, upstream_snapshot_sha256)`` via Catalog #245
   ``tac.deploy.modal.call_id_ledger.register_dispatched_call_id(...)``.
6. **Counterfactual hooks.** Byte-mutation surface per Catalog #139 +
   Catalog #272 distinguishing-feature integration contract. Per-component
   ablation switches via ``--identity-predictor`` argparse flag.

Lane: ``lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516``
"""

from __future__ import annotations

from tac.substrates.time_traveler_l5_z6.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    FilmConditionedNextFramePredictor,
    Z6PredictiveCodingConfig,
    Z6PredictiveCodingSubstrate,
)
from tac.substrates.time_traveler_l5_z6.archive import (
    Z6PCWM1_HEADER_FMT,
    Z6PCWM1_HEADER_SIZE,
    Z6PCWM1_MAGIC,
    Z6PCWM1_SCHEMA_VERSION,
    Z6PCWM1_SECTION_ROLES,
    Z6PredictiveCodingArchive,
    pack_archive,
    parse_archive,
    parse_z6pcwm1_archive_bytes,
)
from tac.substrates.time_traveler_l5_z6.score_aware_loss import (
    Z6PredictiveCodingLossWeights,
    Z6PredictiveCodingScoreAwareLoss,
)

IMPLEMENTATION_STATUS = (
    "l1_research_only_architecture_archive_loss_scaffold"
)
RESEARCH_ONLY = True

PLANNED_PUBLIC_API = (
    "FilmConditionedNextFramePredictor",
    "Z6PredictiveCodingArchive",
    "Z6PredictiveCodingConfig",
    "Z6PredictiveCodingLossWeights",
    "Z6PredictiveCodingScoreAwareLoss",
    "Z6PredictiveCodingSubstrate",
    "pack_archive",
    "parse_archive",
    "parse_z6pcwm1_archive_bytes",
)


__all__ = [
    "EVAL_HW",
    "IMPLEMENTATION_STATUS",
    "NUM_PAIRS",
    "PLANNED_PUBLIC_API",
    "RESEARCH_ONLY",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "Z6PCWM1_HEADER_FMT",
    "Z6PCWM1_HEADER_SIZE",
    "Z6PCWM1_MAGIC",
    "Z6PCWM1_SCHEMA_VERSION",
    "Z6PCWM1_SECTION_ROLES",
    "FilmConditionedNextFramePredictor",
    "Z6PredictiveCodingArchive",
    "Z6PredictiveCodingConfig",
    "Z6PredictiveCodingLossWeights",
    "Z6PredictiveCodingScoreAwareLoss",
    "Z6PredictiveCodingSubstrate",
    "pack_archive",
    "parse_archive",
    "parse_z6pcwm1_archive_bytes",
    # WAVE-1 canonical posterior emission wire-in (2026-05-26)
    "SUBSTRATE_ID",
    "ARCHITECTURE_CLASS",
    "CANONICAL_EQUATION_IDS",
    "emit_landing_posterior_anchor",
]


# ─── Canonical landing-time posterior emission (WAVE-1 wire-in 2026-05-26) ──
# Per OPTIMIZATION-TOOLING-AUDIT roadmap commit `e757bb74c` META #1 + the
# canonical helper at `tac.substrates._shared.posterior_emission_helper`:
# lifts this substrate's L1 signal into the cathedral autopilot's 62
# auto-discovered consumers via the canonical posterior surfaces.
#
# Sister coordination: L1-PROMOTION-D-Z6 (commit `8833b9db5`) JUST landed
# non-promotable markers + predicted_band_validation_status=
# post_training_mlx_50_100ep_local per Catalog #324. This wire-in extends
# the L1 posterior emission to the cathedral-consumer-discoverable surface.

SUBSTRATE_ID: str = "time_traveler_l5_z6"
ARCHITECTURE_CLASS: str = "z6_predictive_coding_film_conditioned_next_frame_l1_mlx"

# Per WAVE-3 op-routable #3 the NEW canonical equation for this paradigm
# is queued: predictive_coding_residual_capacity_v1 (B'/D/F shared per
# the audit). Until registered in tac.canonical_equations, the manifest
# row's canonical_equation_ids carries the proposed-equation token so
# audit tooling can trace the lineage per Catalog #344.
CANONICAL_EQUATION_IDS: tuple[str, ...] = (
    "predictive_coding_residual_capacity_v1_proposed_per_audit_e757bb74c_op_routable_3",
)


def emit_landing_posterior_anchor(
    *,
    archive_sha256: str | None = None,
    archive_bytes: int = 9_500,
    source_path: str | None = None,
    predicted_score: float = 0.190,
    predicted_d_seg: float | None = 0.00112,
    predicted_d_pose: float | None = 0.000027,
    notes: str = (
        "L1 MLX landing per WAVE-1 canonical posterior emission wire-in 2026-05-26 "
        "(audit commit e757bb74c META #1 closure; sister L1-PROMOTION-D-Z6 commit "
        "8833b9db5 cathedral-discoverable surface extension). Z6 FiLM-conditioned "
        "predictive-coding next-frame predictor per Ballard 2007 ego-motion sister "
        "to Atick-Redlich 1990. Non-promotable per CLAUDE.md MLX research-signal "
        "discipline + Catalog #324 predicted_band_validation_status=post_training_"
        "mlx_50_100ep_local."
    ),
    posterior_path: object | None = None,
    posterior_lock_path: object | None = None,
    manifest_path: object | None = None,
):
    """Emit canonical landing-time posterior anchor for this substrate.

    Per WAVE-1-POSTERIOR-EMISSION-CANONICAL-WIRE-IN charter 2026-05-26 +
    OPTIMIZATION-TOOLING-AUDIT META #1 CRITICAL finding closure: invokes
    the canonical helper at
    ``tac.substrates._shared.posterior_emission_helper.emit_substrate_landing_posterior_anchor``
    with this substrate's canonical identifiers + canonical equation IDs
    threaded through ``extra_manifest_fields`` for cathedral consumer
    observability.

    Lifts this substrate's signal into:
    - ``.omx/state/continual_learning_posterior.json`` (refused as
      advisory-grade per custody validator; bumps ``refused_anchor_count``)
    - ``.omx/state/mps_research_signal_manifest.jsonl`` (canonical MLX
      research-signal posterior; cathedral-queryable surface)

    Per Catalog #287/#323/#341: anchor is non-promotable by construction.
    Per Catalog #128 + #131 + #138 sister discipline: writes through
    canonical fcntl-locked helpers only.
    """
    from tac.substrates._shared.posterior_emission_helper import (
        emit_substrate_landing_posterior_anchor,
        synthesize_substrate_archive_sha256,
    )

    sha = archive_sha256 or synthesize_substrate_archive_sha256(SUBSTRATE_ID)
    src = source_path or (
        "src/tac/substrates/time_traveler_l5_z6/"
        "__init__.py:emit_landing_posterior_anchor_l1_extension"
    )

    return emit_substrate_landing_posterior_anchor(
        substrate_id=SUBSTRATE_ID,
        archive_sha256=sha,
        archive_bytes=int(archive_bytes),
        source_path=src,
        predicted_score=predicted_score,
        predicted_d_seg=predicted_d_seg,
        predicted_d_pose=predicted_d_pose,
        architecture_class=ARCHITECTURE_CLASS,
        notes=notes,
        posterior_path=posterior_path,  # type: ignore[arg-type]
        posterior_lock_path=posterior_lock_path,  # type: ignore[arg-type]
        manifest_path=manifest_path,  # type: ignore[arg-type]
        extra_manifest_fields={
            "paradigm": "predictive_coding_ego_motion_conditioned",
            "lane_class": "substrate_engineering",
            "horizon_class": "frontier_pursuit",
            "canonical_equation_ids": list(CANONICAL_EQUATION_IDS),
            "research_only": RESEARCH_ONLY,
            "implementation_status": IMPLEMENTATION_STATUS,
            "predicted_band_validation_status": (
                "post_training_mlx_50_100ep_local_per_catalog_324_sister_l1_promotion_d_z6_8833b9db5"
            ),
            "z6_archive_target_bytes_min": TOTAL_ARCHIVE_TARGET_BYTES_MIN,
            "z6_archive_target_bytes_max": TOTAL_ARCHIVE_TARGET_BYTES_MAX,
            "z6_num_pairs": NUM_PAIRS,
            "sister_l1_promotion_commit": "8833b9db5",
            "ego_motion_canonical_anchor": (
                "atick_redlich_1990_plus_ballard_2007_foe_projection"
            ),
        },
    )
