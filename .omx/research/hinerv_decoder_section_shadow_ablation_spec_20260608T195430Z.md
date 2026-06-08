# Next patch spec — hi_nerv_decoder_section_shadow_ablation.v1 + sidecar-scorer (the section-guilt + lowering-race measurement)

UTC: 2026-06-08T195430Z · claude (solo) · planning, no score claim. Consumes the LANDED v2 L-set certificate (`d250af859`). Pre-registered before build so the inverse-mapping risk is gated.

## Status of GPT's hardening list — ALL LANDED (d250af859, v2)
1. region_semantics + optional `originally_wrong_mask_bhw` ✓ · 2. `fakequant_argmax_won_count`/`parseback_argmax_won_count` alongside positive-margin counts ✓ · 3. `strong_margin_floor` recorded ✓ · 4. `parseback_won` + `fakequant_retained` subsets ✓ · 5. lost-margin bands (`fraction_ge_{0,0.05,strong_floor,0.25,0.5}` + p01/p05/p50) ✓ · 6. `causal_hint` advisory (`causal_hint_is_advisory:True`, wording `likely_quantization_or_export`/`likely_margin_safety`) ✓. 8 tests.

## THE KEY RISK that gates the build (why this is a careful sub-build, not a rush)

Parse-back renders a TORCH model (`tac.substrates.hi_nerv.inflate.build_model_from_archive` → torch receiver), while live/fakequant/shadow use the MLX model. The decoder-section shadow must take the LIVE MLX model and replace ONLY section s's weights with the archive-decoded values, others live, then render via MLX. The archive-decoded weights are TORCH tensors (from `parse_archive(payload).decoder_state_dict`) in torch layout; `export_state_dict` (mlx_renderer.py:7400-7442) maps MLX→torch with PER-SECTION TRANSPOSES (heads/convnext pwconv `(0,3,1,2)`; injector Linear none; grids per their own layout). The shadow needs the EXACT INVERSE per section. A wrong transpose silently yields a misleading row — the precise failure mode the L-set discipline exists to prevent.

**Safety gate (build step 1):** implement `import_state_dict` (torch-convention decoder_state_dict → MLX attributes) as the exact inverse of `export_state_dict`, and prove it with a ROUND-TRIP IDENTITY TEST: `export_state_dict(model)` → `import_state_dict(model2)` → `export_state_dict(model2)` byte-equal, AND `model2(idx)` render byte-equal to `model(idx)`. No section shadow may run until this passes.

## Build (in-loop, like the latents_fine shadow at birth_survival.py)

`measure_birth_decoder_section_shadow(model, *, scorer_teacher, parsed_archive, section, pair_indices, target_labels, live_birth_payload, fakequant_logits, parseback_logits)`:
- snapshot the MLX section attrs for `section` ∈ {head_rgb_1, fine_injector, feature_grids, head_rgb_0, mid_injector, convnext_blocks}; replace with `import`-mapped archive-decoded weights; render frame1; restore (MLX immutable snapshot pattern, already proven).
- emit a `hi_nerv_decoder_section_shadow_ablation.v1` row per section carrying the v2 L-set certificate (evaluated = this section's shadow logits; fakequant + parseback logits passed in from the same live birth), wrong_to_target, retention, exact ΔS_nonrate, support/action/archive hashes.
- if NO single section drives `causal_hint=likely_quantization_or_export` with `median_{p∈L} m_eval ≤ 0`: compute pairwise commutators (head_rgb_1∘fine_injector, ∘grid, fine_injector∘grid): `retention_comm(s,t)=ret(Qs,Qt)-ret(Qs)-ret(Qt)+ret(live)`.
- wire into runner birth-survival block (in-loop, live model present) exactly like the latents_fine shadow (`0218839bb`); one bounded smoke emits all section rows.

## Sidecar-scorer row (Rule #8 — convert the lead to proof/refutation)
Extend `build_hi_nerv_target_region_action_parseback_survival` (archive_candidate.py:206) to run SegNet on the sidecar-applied receiver frame: emit `sidecar_wrong_to_target`, `sidecar_target_margin_p10` over L, `sidecar_pose_delta`, `sidecar_exact_delta_score_total` (incl. sidecar bytes), `parseback_scorer_effect_survived` (currently null/unmeasured).

## Lowering race (the decision)
argmin `ΔS_total = 100·Δd_seg + (√(10·d_pose') − √(10·d_pose)) + 25·Δbytes/37_545_489` over {guilty-section ArchiveRoundtripMarginQAT, byte-priced sidecar, discard}, each gated on positive scorer-effect survival + Pose trust. Backend NOT privileged.

## Scoped interpretation ledger (GPT-precise wording, adopted)
latents_fine FALSIFIED **as standalone first section** (not as commutator) · simple live-vs-EMA swap FALSIFIED **for this artifact** (not all custody drift) · whole-region p10 fragility RETRACTED; **L-set fragility still open** until measured on real artifact · sidecar a LEAD (payload-only, scorer unmeasured) · backend decoder section / commutator = leading hypothesis.

## DO NOT
latents_fine QAT (falsified) · change export-selection (falsified for this artifact) · sidecar-win on payload alone (Rule #8) · whole-region p10 as causal evidence · run a section shadow before the export/import round-trip-identity test passes · use 8-bit symmetric fakequant as the section proxy (preserves 12,183, hides the section) — use the exact archive decode.
