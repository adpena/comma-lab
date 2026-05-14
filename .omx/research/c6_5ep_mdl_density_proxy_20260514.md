# C6 MDL-IBPS 5ep architectural-only MDL density proxy

`[5ep-architectural-only-proxy]` `[mathematical-derivation]`

**Date**: 2026-05-14 UTC
**Subagent**: c6_next_wave_grammar_smoke_mdl_proxy_20260514
**Lane**: `lane_c6_next_wave_grammar_smoke_mdl_proxy_20260514`
**Archive under test**: `experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T144949Z__smoke__5ep_modal/harvested_artifacts/lane_substrate_c6_e4_mdl_ibps_results/output/archive.zip`
- size: 224,481 bytes
- sha256: `a27328ce02211f1c8ee0cfb4318ace29c438a62cf09a42358481d0273a204607`
- ZIP member: `0.bin` (225,622 bytes)
- magic: `b'IBPS'` schema_version=1

## IBPS1 grammar section breakdown

Per `tac.substrates.c6_e4_mdl_ibps.archive.IBPS1_HEADER_FMT`:

| Section | Role | Bytes | Entropy bits/byte |
| --- | --- | --- | --- |
| `ibps1_header` | control_or_metadata | 25 | 3.4937 |
| `encoder_blob` (brotli q=9 of fp16 encoder state_dict) | decoder_weight_stream | 65,482 | 7.9907 |
| `decoder_blob` (brotli q=9 of fp16 decoder state_dict) | decoder_weight_stream | 145,198 | 7.9930 |
| `latent_blob` (int8 quantized per-pair z, num_pairs=600 × latent_dim=24) | latent_stream | 14,400 | 7.0313 |
| `meta_blob` (sorted-keys JSON: beta_ib, decoder_channels, _lat_scale, _lat_zero_point) | control_or_metadata | 517 | 4.9551 |

- IBPS1 header parameters: latent_dim=24, num_pairs=600
- Total measurable payload: 225,622 bytes
- Aggregated entropy bits/byte (byte-weighted): **7.9235**
- MDL density (`agg_bpb / 8.0`): **0.9904**

## Verdict

`MDL_density = 0.9904` → **WITHIN-HNeRV-CLASS** (>= 0.90 saturation threshold per
`tac.cathedral_autopilot_autonomous_loop.adjust_predicted_delta_for_mdl_density`).

## Interpretation

The MDL proxy is a STRUCTURAL TIER measure: it computes the byte-level Shannon entropy
of the archive sections AS LAID OUT on disk. For an HNeRV-family substrate (encoder
state_dict + decoder state_dict + latents), brotli compression of fp16 weights at
quality 9 produces near-maximum-entropy byte streams (7.99 bpb out of 8.0) regardless
of the underlying architectural family.

In other words: **the structural-tier MDL proxy CANNOT distinguish architectural
families with brotli-compressed weight blobs**. C6 / A1 / PR101 / PR106 / sane_hnerv
would all measure within-class at this tier because the brotli output is the
saturating step, not the architecture itself.

This is consistent with the Z1 memo (`feedback_z1_mdl_ablation_landed_20260514`)
note that structural-tier MDL density "is a CONSERVATIVE proxy for true
scorer-conditional MDL (tier-A in the memo)." The scorer-conditional MDL — what
actually determines whether the substrate escapes the within-class trap — requires
running PoseNet + SegNet against the *decoded frames* (Tier B + Tier C of the canonical
ablation), which only the contest-CUDA auth eval on a TRAINED 100ep+ substrate can
deliver.

## What this proxy DOES and DOES NOT prove

- **Does prove**: IBPS1 grammar wire format is byte-saturated like every other
  brotli-compressed monolithic HNeRV-family packet. Future bolt-on byte-level
  optimizations on IBPS1 should NOT expect 8-bit-per-byte savings from
  re-compressing already-brotli'd sections.

- **Does NOT prove**: whether the architectural substrate (cooperative-receiver
  encoder + IB-bottlenecked latents + MDL-aware decoder) escapes the HNeRV family
  floor. That question is answered ONLY by `contest-CUDA T4 Linux x86_64` auth eval
  on a fully-trained (100ep+) C6 archive against `upstream/videos/0.mkv`.

- **Does NOT prove**: whether 5ep / 100ep / 1000ep C6 trained substrates produce
  similar or different MDL density. Training shifts the *content* of the encoder /
  decoder state_dict weights (changing fp16 distribution), but brotli's
  near-saturation regime is robust to that shift — MDL density at this tier should
  remain ≥ 0.95 across all training durations.

## Routing implications

Per `tac.cathedral_autopilot_autonomous_loop.adjust_predicted_delta_for_mdl_density`,
substrates with `mdl_density >= 0.90` get their predicted ΔS halved (within-class
penalty) or zeroed (>= 0.95 saturated penalty). C6's structural-tier 0.9904 lands in
the **0.95 SATURATED band** → predicted ΔS = 0.

**However**, this is the structural-tier penalty alone — it does NOT cover Tier B
(sampled byte-level scorer perturbation) or Tier C (post-decode perturbation against
PoseNet/SegNet). For substrate-class shift discrimination, the **smoke auth eval**
result (Part C) is the dispositive evidence. The MDL density proxy is informational
only — a single signal layered into the planner, not a falsification.

Per CLAUDE.md "KILL is LAST RESORT": this finding does NOT retire C6. It only
clarifies that the structural-tier MDL density is the **wrong tool** to detect
architectural-family escape from the HNeRV ceiling. The Z1 ablation memo correctly
identifies Tier B + Tier C as the disambiguators; this proxy is Tier A only.

## Recommendation

1. **Continue C6 first-anchor smoke dispatch** (Part C of this lane). The smoke score
   on contest-CUDA is the dispositive evidence for substrate-class escape.

2. **Do NOT use structural-tier MDL density to gate C6 advancement**. Use it as a
   sanity check that the grammar parser produces sensible byte-section
   measurements; that is what this proxy proved (the IBPS1 parser is correct and
   the sections add up to total inner-blob size).

3. **Future work**: extend the canonical `ScorerConditionalMDLEstimator` (and
   `tools/mdl_scorer_conditional_ablation.py`) Tier B + Tier C against a TRAINED
   100ep C6 archive once smoke confirms the architecture viability. Tier B
   (sampled byte-flip + scorer Δscore) is what would distinguish C6's
   IB-bottlenecked latents from PR101's straight HNeRV latents.

## Tags

- `[5ep-architectural-only-proxy]` — measurement is structural Tier A only, on a 5ep
  smoke archive (not a trained substrate density which only floor v3 100ep+ would
  produce)
- `[mathematical-derivation]` — byte-Shannon-entropy computation; no scorer forward
  pass involved
- `[planning_only_no_score_claim]` — does not claim a score; only computes byte
  density
- `[no_mps_authoritative]` — N/A (no scorer forward; CPU-only entropy math)
- `[no_tmp_paths]` — N/A
- `evidence_grade="mathematical-derivation"` per
  `tac.xray.mdl_scorer_conditional.ScorerConditionalMDLEstimator`

## Cross-references

- Source memo for C6 substrate: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_recovery_2_c6_finish_and_modal_harvest_landed_20260514.md`
- Z1 ablation framework: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_z1_mdl_ablation_landed_20260514.md`
- Zen floor v2 within-class-vs-across-class taxonomy: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_zen_floor_band_v2_post_z1_ablation_20260514.md`
- IBPS1 grammar definition: `src/tac/substrates/c6_e4_mdl_ibps/archive.py`
- Canonical xray primitive: `src/tac/xray/mdl_scorer_conditional.py::ScorerConditionalMDLEstimator`
- Cathedral autopilot density adjuster: `tac.cathedral_autopilot_autonomous_loop.adjust_predicted_delta_for_mdl_density`
