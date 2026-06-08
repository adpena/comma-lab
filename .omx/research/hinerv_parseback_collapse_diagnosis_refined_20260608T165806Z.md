# HiNeRV parse-back collapse — refined diagnosis (two hypotheses falsified, one retraction)

UTC: 2026-06-08T165806Z · claude (solo) · `[macOS-MLX research-signal]`, NON-PROMOTABLE, no score claim. Refines `hinerv_archive_roundtrip_shadow_falsifies_latents_fine_cause_20260608T164835Z.md` after a cheap artifact audit (no new run). Same artifact: `…/hinerv_archive_roundtrip_shadow_smoke_20260608T160000Z_v6/`, action_id `7961882c…`.

## Confirmed (decisive)

1. **latents_fine is INNOCENT.** archive_roundtrip_shadow (latents_fine through exact HIV1 int16, decoder weights live) preserved 12,183 wins = fakequant; delta 2.26e-5. Only full parse-back → 2.

2. **NOT an export-selection swap (GPT's 40% hypothesis FALSIFIED).** Both archive candidates were built; selection chose `ema` — but the live-vs-ema decision was a near-tie (parseback proxy 33744.069325808 live vs 33744.069325103 ema; bytes 16326 vs 16325). Critically, the **live-branch birth parse-back AND the ema-branch birth parse-back collapse IDENTICALLY** (both wt=2, retention 0.000148, ffs=`parseback_margin_floor`). The live archive loses the birth exactly as the EMA archive does ⇒ the collapse is intrinsic to the archive encode/decode of the birth backend state, independent of EMA-vs-live selection.

## Retraction (honesty)

3. **Finding 2 ("birth is margin-fragile before parse-back") is RETRACTED.** `region_margin_stats` computes `target_margin_p10` over the ENTIRE target-region mask (`flat_mask = mask > 0.0`), which includes still-unsolved pixels. With ~12,183 won of N region pixels, a negative p10 (−0.556) is EXPECTED and says nothing about whether the *won* pixels are knife-edge. Per GPT: the money metric is the margin over the set L = {fakequant-won AND parse-back-lost} pixels, which is NOT yet measured. No margin-fragility claim is warranted until subset-conditioned margins exist.

## New lead (NOT yet a claim — needs code reading)

4. The target-region **ACTION** parse-back (`hi_nerv_target_region_action_parseback_survival.v1`) reports `survived=true`, while the **BIRTH** backend candidate (`hi_nerv_target_region_birth_survival_candidate.v1`) collapses (wt=2). If the "action" surface is the sidecar/applied-action lowering and the "birth" surface is the backend decoder-weight realization, this is direct evidence that the backend birth dies in the archive while the action survives — i.e., sidecar-preferred for this atom (GPT's contrarian point). MUST read the two producers before claiming this; the surfaces may measure different things.

## Narrowed cause

The 12,183→2 collapse is in the archive encode/decode of a **non-latents_fine backend section** — decoder weights (`head_rgb_1` / `fine_injector` / feature grids) under the EXACT archive quantizer (not 8-bit symmetric fakequant, which preserved 12,183), and/or a multi-section quantization interaction (commutator). Per the original parity receipt: "head_rgb has larger raw decoder quantization deltas."

## Next operator (task #21, refined)

- DROP latents_fine QAT (falsified) AND drop export-selection-policy change (falsified for this artifact).
- Build `hi_nerv_decoder_section_shadow_ablation.v1`: per-section, replace ONLY that section with its exact archive decode (others live), re-render+rescore; report wrong_to_target + subset-conditioned margins (all-region / live-won / fakequant-won-but-parseback-lost) + exact ΔS. Sections: head_rgb_1, fine_injector, feature grids, output head; then pairwise commutators if no single section explains it.
- Read the action-vs-birth parseback producers to confirm/deny the sidecar-survives-backend-dies lead.
- Add subset-conditioned margin certificates to region_margin_stats consumers (the L-set margin).

## Probability split (updated from GPT's pre-result guess)

decoder-weight archive quantization (head_rgb_1/fine_injector) single-section ~45% · multi-section quantization interaction ~25% · sidecar-survives-so-lower-to-sidecar is the right move regardless ~20% · scorer-preprocess/render parity residual ~10%. latents_fine ~0 (falsified). export-selection swap ~0 (falsified).
