# HiNeRV shadow row FALSIFIES "latents_fine_quantizer_delta" as the parse-back collapse cause

UTC: 2026-06-08T164835Z · Author: claude (solo) · Authority: `[macOS-MLX research-signal]`, batch-local, NON-PROMOTABLE, no score/rank/promotion claim. Exact objective that ultimately arbitrates: `100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`.

## What ran

ONE bounded source-qualified short smoke (sanctioned single run; no broad readiness, no long MLX), reproducing the EXACT 20260607 retentionfix argv (recovered from `campaign_identity.argv`) into a fresh dir so all four surfaces share ONE live birth. Output: `/Volumes/VertigoDataTier/pact/hinerv_archive_roundtrip_shadow_smoke_20260608T160000Z_v6/`. The new in-loop `archive_roundtrip_shadow` surface (commit `0218839bb`) emitted its row from the LIVE birth model, then the `hi_nerv_archive_roundtrip_margin_qat.v1` aggregator (commit landing this session) ran the decision table.

## The four-surface row (all action_id `7961882c…`, same-action proof holds)

| surface | wrong_to_target | retention | survived | target_margin_p10 |
|---|---|---|---|---|
| live | 13,488 | 1.0 | — | — |
| fakequant_mlx | 12,183 | 0.9032 (HIGH) | false | (fakequant_mlx_margin_floor) |
| **archive_roundtrip_shadow** | **12,183** | **0.9032 (HIGH)** | false | **−0.5566** |
| parseback_mlx | 2 | 0.000148 (LOW) | false | — |

`section_hiv1_roundtrip_max_abs_delta.latents_fine = 2.26e-5`.

Aggregator verdict: `interpretation_case = B_latent_quantizer_not_the_cause`; `first_failed_surface = parseback_mlx`; `first_failed_section = None`; `recommended_lowering = audit_export_selection_or_decoder_sections`.

## Two findings (both redirect away from latents_fine QAT)

**Finding 1 — latents_fine int16 is NOT the collapse cause (FALSIFICATION).** Routing latents_fine through the EXACT HIV1 int16 archive decode (decoder weights live) preserves 12,183/13,488 scorer wins — IDENTICAL to fakequant — with a latent delta of 2.26e-5. Decoder-weight 8-bit fakequant alone also preserves 12,183. Only FULL parse-back collapses to 2. So the 12,183→2 loss is caused by a section the shadow did not route — the EXACT archive decoder-weight quantizer and/or the EMA-export/selection step — NOT live latents_fine int16. The original parity receipt's `first_failed_section = latents_fine_quantizer_delta` was a TENSOR-delta heuristic (the EMA-exported latents had a 0.0057 delta); the scorer-effect shadow shows that tensor delta is not the causal mechanism. Note the live birth latents (2.26e-5) are far less int16-fragile than the EMA-exported latents (0.0057) — implicating EMA-selection/export, not the live latent step.

**Finding 2 — the birth is margin-fragile before parse-back.** `target_margin_p10 = −0.556` at the shadow (and fakequant fails its margin floor too): the 12,183 wins are COUNT-preserved but their 10th-percentile margin is negative — the birth crossed the SegNet wall by epsilon. Even a perfect export fix will not make this birth launch-grade without a positive target-margin floor during training (GPT's robust-margin point, now empirically grounded).

## Consequence for the plan

- DROP "latents_fine HIV1 roundtrip QAT" as the primary fix — the artifact falsified it.
- Next diagnostic (Case B): extend the section ablation to the decoder-weight sections (`head_rgb_1`, `fine_injector`, feature grids) under the EXACT archive quantizer + the EMA-export/selection step, to name which section's parse-back encode collapses the count 12,183→2. (The shadow helper only supports latent sections; decoder-weight ablation needs the archive decoder-weight decode, not 8-bit symmetric fakequant.)
- Independently: add a robust target-margin floor to the birth so `target_margin_p10 ≥ γ > 0`; for the 4D-latent case the 16-corner cell certificate is the strong form, but Finding 1 says the cell that matters is the decoder-weight/export cell, not the latent cell.

## Custody
All rows + the aggregator receipt carry `authority=planning_control_false_authority` / PROXY_FALSE_AUTHORITY_FIELDS; no nested score/promotion keys. codex's `fakequant_vs_archive_decoded_max_abs_delta` null in archive_candidate.py was NOT touched. Receipt JSON: `…/v6/hi_nerv_mlx_training/hi_nerv_archive_roundtrip_margin_qat.json`.
