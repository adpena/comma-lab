# Substrate Contest-CUDA Chain — Sweep Audit (2026-05-15)

**Lane**: `lane_canonicalize_substrate_contest_chain_20260515`
**Mandate**: operator directive *"need to canonicalize and standardize all and implement and fix and harden and test all"* (2026-05-15).
**Anchor**: Z3 v2 smoke `fc-01KRNHWD6AEMWGDJXBEY8GM0P3` SMOKE GREEN but manifest reveals `"layout": "append_only_z3hp1_diagnostic"` because the recipe is `smoke_only: true / research_only: true` so the v2 latent-replacement code path NEVER ACTIVATES even though env→CLI wiring is intact. Sister-substrate parity audit needed.

## Premise verification (per CLAUDE.md "Prompt-premise verification before edit pattern")

Reproducer at `.omx/tmp/canonicalize_substrate_contest_chain_audit_v2.py` (run via `.venv/bin/python`). Verdict matrix persisted at `.omx/state/substrate_contest_cuda_chain_audit.json` (machine-readable, regenerable).

## Verdict matrix (post-fix 2026-05-15)

| substrate | recipe(s) | full_main | not_impl | auth_wired | status |
|---|---|---|---|---|---|
| a1_plus_lapose | a1_plus_lapose_modal_a100_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| a1_plus_wavelet_residual | a1_plus_wavelet_residual_modal_t4_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| balle_renderer | balle_renderer_modal_a100_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| block_nerv | block_nerv_modal_a100_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| c1_world_model_foveation | c1_world_model_foveation_modal_t4_smoke_dispatch | True | True | True | RESEARCH_ONLY_TRANSPARENT |
| c6_e4_mdl_ibps | c6_e4_mdl_ibps_modal_t4_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| cool_chic | cool_chic_modal_a100_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| d1_segnet_margin_polytope | d1_segnet_margin_polytope_modal_t4_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| d4_wyner_ziv_frame_0 | d4_wyner_ziv_frame_0_modal_t4_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| ds_nerv | ds_nerv_modal_a100_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| ff_nerv | ff_nerv_modal_a100_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| grayscale_lut | grayscale_lut_modal_a100_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| hi_nerv | hi_nerv_modal_a100_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| hybrid_renderer_residual | hybrid_renderer_residual_modal_a100_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| pr101_lc_v2_clone_enhanced_curriculum | pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch | True | True | True | RESEARCH_ONLY_TRANSPARENT |
| pretrained_driving_prior | pretrained_driving_prior_modal_t4_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| s2sbs_byte_stuffing | s2sbs_byte_stuffing_modal_t4_dispatch | False | False | False | RESEARCH_ONLY_TRANSPARENT |
| sabor_boundary_only_renderer | sabor_boundary_only_renderer_modal_t4_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| sane_hnerv | sane_hnerv_modal_a100_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| sar_coherent_pose_pairs | sar_coherent_pose_pairs_modal_t4_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| self_compress_nn | self_compress_nn_modal_a100_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| siren | siren_modal_a100_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| stack_of_stacks | stack_of_stacks_modal_a100_dispatch | False | False | False | RESEARCH_ONLY_TRANSPARENT |
| tc_nerv | tc_nerv_modal_a100_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| time_traveler_l5_autonomy | time_traveler_l5_autonomy_modal_a100_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| vq_vae | vq_vae_modal_a100_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| wavelet | wavelet_modal_a100_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| wyner_ziv_cooperative_receiver | wyner_ziv_cooperative_receiver_modal_a100_dispatch | True | False | True | DISPATCHABLE_CONTEST_CUDA |
| z3_balle_hyperprior_bolton | z3_balle_hyperprior_bolton_modal_t4_dispatch | True | False | True | RESEARCH_ONLY_TRANSPARENT |
| z4_cooperative_receiver_loss | z4_cooperative_receiver_loss_modal_t4_dispatch | True | True | True | **RESEARCH_ONLY_TRANSPARENT** (FIXED 2026-05-15) |
| z5_predictive_coding_world_model | z5_predictive_coding_world_model_modal_t4_dispatch | True | True | True | **RESEARCH_ONLY_TRANSPARENT** (FIXED 2026-05-15) |

## Verdict counts

- **DISPATCHABLE_CONTEST_CUDA**: 14
- **RESEARCH_ONLY_TRANSPARENT**: 17
- **PARTIAL_COUNCIL_GATED**: 0 (was 2 pre-fix; Z4 + Z5 now tagged `research_only: true`)
- **INCOMPLETE_NEEDS_FIX**: 0
- **NO_RECIPE**: 0

## Bugs found + fixed (Phase 4)

### Z4 + Z5 PARTIAL_COUNCIL_GATED

Both `experiments/train_substrate_z4_cooperative_receiver_loss.py:314` and
`experiments/train_substrate_z5_predictive_coding_world_model.py:360` have
`_full_main` raising `NotImplementedError` pending Phase 2 council approval.
Their recipes (`substrate_z4_*_modal_t4_dispatch.yaml`,
`substrate_z5_*_modal_t4_dispatch.yaml`) had NO `smoke_only: true` /
`research_only: true` / `dispatch_enabled: false` flag — meaning a Modal
dispatch through `tools/operator_authorize.py` would have reached the trainer
and crashed pre-auth-eval (Z3 v2 fc-01KRNHWD6AEMWGDJXBEY8GM0P3 bug class
recurrence at higher cost).

**Fix**: added `research_only: true` + `dispatch_blockers: [phase_2_council_approval_required_to_lift_full_main_NotImplementedError]` to both recipes with traceable comments referencing CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #220.

## Orphan recipes (Phase 1 sub-finding)

9 substrate recipes have no matching `experiments/train_substrate_*.py` trainer.
ALL are tagged `dispatch_enabled: false`:

- `substrate_cnerv_modal_a100_dispatch`
- `substrate_diffusion_renderer_modal_a100_dispatch`
- `substrate_dp_sims_renderer_modal_a100_dispatch`
- `substrate_e_nerv_modal_a100_dispatch`
- `substrate_ego_nerv_modal_a100_dispatch`
- `substrate_lane_12_v2_nerv_modal_a100_dispatch`
- `substrate_mlx_mask_renderer_local_apple_silicon_dispatch`
- `substrate_nervdc_modal_a100_dispatch`
- `substrate_quantizr_faithful_modal_a100_dispatch`

These are intentional design-stage scaffolds (recipe-first, trainer pending).
Out of scope for this sweep; they cannot fire any Modal dispatch.

## Canonical pattern (Phase 2 — separate doc)

See `.omx/research/canonical_substrate_contest_cuda_chain_pattern_20260515.md`.

## STRICT preflight gate (Phase 5)

Catalog #232 candidate `check_substrate_contest_cuda_chain_complete_or_research_only_tagged` lands warn-only-then-strict-flipped per CLAUDE.md "Strict-flip atomicity rule" — live count at landing: 0 (Z4 + Z5 fix in same commit batch).

## Local pre-deploy check (Phase 7)

`tools/local_pre_deploy_check.py` extended with 6th check `recipe_status_consistent_with_trainer_state` that surfaces the bug class at the operator-authorize 30-second harness BEFORE any Modal dispatch fires.

## 6-hook wire-in (per Catalog #125)

1. **Sensitivity-map**: N/A — no score-affecting payload change.
2. **Pareto constraint**: N/A — no R/D regime change.
3. **Bit-allocator hook**: N/A — substrate composition unchanged.
4. **Cathedral autopilot dispatch hook**: ACTIVE — the audit JSON `.omx/state/substrate_contest_cuda_chain_audit.json` is the canonical "is_dispatchable_for_contest_cuda" predicate the autopilot ranker can consume to refuse promoting a non-dispatchable substrate to a dispatch slot.
5. **Continual-learning posterior**: N/A — no empirical anchor produced.
6. **Probe-disambiguator**: N/A — no design tension to arbitrate.

## Cross-references

- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 2 (export-first design)
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable
- CLAUDE.md FORBIDDEN_PATTERNS — `forbidden representation-without-archive-grammar (the "research-substrate trap")` (8th forbidden pattern)
- Catalog #220 (`check_substrate_l1_scaffold_no_byte_addition_without_operational_score_improvement_mechanism`) — the runtime-effect sister
- Catalog #226 (`check_trainer_auth_eval_uses_canonical_helper`) — auth-eval canonical gate routing
- Catalog candidate #232 (`check_substrate_contest_cuda_chain_complete_or_research_only_tagged`) — this audit's STRICT gate
- Memory `feedback_z3_v2_smoke_green_but_v2_path_inactive_diagnostic_layout_anchor_20260515.md` (the empirical anchor)
