# HiNeRV parse-back collapse — structural resolution: backend dies (scorer-measured), sidecar payload survives (scorer UNMEASURED)

UTC: 2026-06-08T180552Z · claude (solo) · `[macOS-MLX research-signal]`, NON-PROMOTABLE, no score claim. Final note of the diagnosis chain (`…164835Z` → `…165806Z` → this). Artifact `…/hinerv_archive_roundtrip_shadow_smoke_20260608T160000Z_v6/`, action_id `7961882c…`.

## The decisive structural finding (from reading the two producers)

Two distinct things are measured under the name "parse-back survival":

- **BIRTH backend** (`hi_nerv_target_region_birth_survival_candidate.v1`): the scorer-domain bootstrap that modified DECODER WEIGHTS. Re-rendered + re-scored through SegNet after archive parse-back ⇒ **wrong_to_target 12,183 → 2, retention 0.000148, scorer-effect MEASURED, COLLAPSES.**
- **ACTION sidecar** (`hi_nerv_target_region_action_parseback_survival.v1`, `build_hi_nerv_target_region_action_parseback_survival`): decodes the charged action sidecar from archive meta (`TARGET_REGION_ACTION_META_KEY`), renders receiver with/without the sidecar, verifies every support pixel is overwritten with the exact uint8 RGB action value ⇒ `survived=true`, `parseback_payload_survived=true`, `parseback_program_survived=true`, **`parseback_scorer_effect_survived=null`, `scorer_effect_survival_measured=false`.**

## What this does and does NOT establish

- ESTABLISHED: the backend decoder-weight birth does NOT survive the archive at the scorer level. The fix is not in latents_fine (falsified), not in export-selection (falsified) — it is in the decoder-weight archive quantization, OR the atom must be lowered off the backend entirely.
- NOT ESTABLISHED (CLAUDE.md Absolute Rule #8 — payload/program survival ≠ scorer-effect survival): the sidecar action's BYTES survive parse-back, but whether the sidecar produces SegNet target wins after parse-back is UNMEASURED (`scorer_effect_survival_measured=false`). So "sidecar-preferred" is a strong LEAD, NOT a proven win. The same over-interpretation trap as the retracted margin-p10 claim — guarded here.

## The two next operators (both in-loop; live model not checkpointed)

1. **Measure the sidecar action's parse-back SCORER effect** — extend `build_hi_nerv_target_region_action_parseback_survival` to run SegNet on the sidecar-applied receiver frame and emit `parseback_wrong_to_target` + L-set margins. If it holds the wins → sidecar-preferred is PROVEN and the lowering race picks sidecar (cheap, GPT's contrarian call confirmed). If it also collapses → both lowerings fail; the atom needs a decoder-section fix or discard.
2. **Decoder-section shadow ablation** (`hi_nerv_decoder_section_shadow_ablation.v1`) — per backend section (head_rgb_1 / fine_injector / feature grids) replace ONLY that section with its EXACT archive decode (others live), re-render+rescore; report wrong_to_target + L-set–conditioned margins (L = fakequant-won ∧ parseback-lost, per GPT) + exact ΔS; pairwise commutators if no single section explains it.

## Exact-score frame for the lowering race (the only authority)

`ΔS_total = 100·Δd_seg + (√(10·d_pose') − √(10·d_pose)) + 25·Δbytes/37_545_489`. Compare: backend-QAT-on-guilty-section cost vs sidecar-action byte cost, both gated on POSITIVE scorer-effect survival after parse-back. Backend is NOT privileged; pick argmin ΔS_total among {backend section QAT, byte-priced sidecar, discard}.

## Falsified / retracted this chain
latents_fine quantizer cause — FALSIFIED. export-selection swap — FALSIFIED (live+ema birth parse-back identical wt=2). birth-margin-fragility (p10=−0.556) — RETRACTED (p10 over whole region, not won pixels). sidecar-preferred — DOWNGRADED to lead (scorer-effect unmeasured per Rule #8).
