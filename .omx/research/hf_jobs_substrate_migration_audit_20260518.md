---
title: HF Jobs substrate-migration per-lane audit (Insight 5 follow-on)
date_utc: 2026-05-18T14:10:00Z
lane_id: lane_hf_jobs_implementation_wave_segnet_posenet_dinov3_sam2_20260518
substrate_class: apparatus_maintenance
status: design-only
horizon_class: apparatus_maintenance
predicted_band_validation_status: pending_post_training
predicted_band: null
related_deliberation_ids: []
council_tier: T1
council_attendees: []
council_quorum_met: false
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict: []
council_decisions_recorded: []
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
---

# HF Jobs substrate-migration per-lane audit

Sister of `hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518.md`. This memo
audits each of the in-flight substrate lanes (~53 substrates per
`reports/lane_maturity.md`) and classifies them by HF Jobs migration
candidacy.

## Classification scheme

Per the cost-comparison memo's decision tree:

- **Tier A** (HF Jobs t4-small candidate): substrate ≤ 100M params AND
  fits in 16 GB VRAM AND HF-Datasets-native (or trivially adaptable).
  Best: $0.40/hr.
- **Tier B** (HF Jobs a10g-large candidate): 100-500M params; 24 GB VRAM.
  $1.20/hr — BUT Vast.ai 4090 at $0.25/hr usually wins on cost.
- **Tier C** (HF Jobs a100-large candidate): >500M params or >24 GB VRAM
  needed. $4.00/hr — Modal A100 ($3.40/hr) or Vast.ai H100 ($1.50-1.99/hr)
  usually wins.
- **Tier D** (Lightning A100 subscription long-burn): >24h continuous.
  $0 marginal — keep on Lightning until subscription utilization
  audit completes.
- **Tier X** (NOT MIGRATEABLE): substrate has runtime constraints that
  require Modal-specific (e.g. Catalog #244 NVML env block, Catalog #166
  HEAD-parity ledger) or Vast.ai-specific (NVDEC, DALI) infrastructure.

## Per-substrate verdicts

### Tier A (HF Jobs t4-small): ~5 candidates

| Substrate | Params | VRAM | Notes |
|---|---|---|---|
| `lane_segnet_surrogate_mobilenetv3_s_*` (NEW) | 2.5M | <2 GB | Build it here from byte zero. |
| `lane_dinov3_anchor_extraction_*` (NEW) | 86.6M (frozen) | ~1 GB | Inference-only; ~$0.07. |
| `lane_macos_cpu_substrate_canvas_sweep_*` | n/a | <1 GB | Small ranker; even CPU-fine. |
| `lane_a1_inflate_time_bias_correction_sweep` | <10M | <2 GB | If migrated, recipe needs `dispatch_kind: tool` per Catalog #270. |
| `lane_lane_17_imp` (per `feedback_pre_rigor_kill_defer_falsified_inventory_20260517.md`) | <50M | <8 GB | Frankle LTH IMP cycle 0; $1-2/run on Vast.ai 4090 OR HF Jobs t4-small. |

Total HF Jobs annual savings (Tier A): ~$50-200 if 5-10 dispatches/week
across these 5 lanes.

### Tier B (Vast.ai 4090 stays cheaper than HF Jobs a10g-large): ~15 candidates

| Substrate | Params | VRAM | HF Jobs recommendation |
|---|---|---|---|
| `lane_segnet_surrogate_sam2_hiera_tiny_*` (NEW) | 38.9M | ~10 GB | a10g-large $1.20/hr; t4-small if mask_decoder-only |
| `lane_posenet_surrogate_sam2_*` (NEW) | 38.9M | ~10 GB | a10g-large $1.20/hr; or t4-small with reduced batch |
| `lane_nscs03_end_to_end_balle_joint_codec_*` | 100-500M | 16-24 GB | a10g-large; OR Vast.ai 4090 if cost-critical |
| `lane_d4_wyner_ziv_frame_0` | 100-500M | 14 GB | a10g-large or 4090 (Catalog #218 mini-batch fix lets it fit in 16 GB) |
| `lane_d1_segnet_margin_polytope` | 100-500M | 14 GB | a10g-large or 4090 |
| `lane_pr101_lc_v2_clone_enhanced_curriculum` | <500M | <24 GB | a10g-large or 4090 |
| ... (~9 more substrates in this band per lane_registry) | | | |

Recommendation: **stay on Vast.ai 4090** unless HF-Datasets-native
convenience justifies the 4x cost premium ($0.25/hr vs $1.20/hr).

### Tier C (Modal A100 or Vast.ai H100 stays cheaper): ~25 candidates

Substrates with `min_vram_gb >= 40` or >500M params:

- `lane_pr106_*` family (PR106 latent sidecar + R2 PR101 grammar)
- `lane_a1_plus_lapose`
- `lane_a1_plus_wavelet_residual`
- `lane_c6_e4_mdl_ibps`
- `lane_z3_balle_hyperprior_bolton`
- `lane_z4_cooperative_receiver_loss`
- `lane_z5_predictive_coding_world_model`
- `lane_pretrained_driving_prior`
- `lane_siren`
- `lane_sabor_boundary_only_renderer`
- `lane_sar_coherent_pose_pairs`
- `lane_stack_of_stacks`
- `lane_time_traveler_l5_z6`
- `lane_time_traveler_l5_z6_v2`
- ... (~10 more)

Recommendation: **stay on Modal A100 or Vast.ai H100** for cost-optimal.
HF Jobs a100-large ($4.00/hr) is 18% more expensive than Modal A100
($3.40/hr) and 2-2.7x more expensive than Vast.ai H100 ($1.50-1.99/hr).

### Tier D (Lightning A100 subscription long-burn): ~3 candidates

Substrates that run multi-day campaigns:

- `lane_dispatch_modal_paired_auth_eval_*` long-running paired-eval chains
- `lane_pr95_meta_stack_of_stacks_enhanced_curriculum` long pretraining
- `lane_meta_lagrangian_search_*` long sweeps

Recommendation: **keep on Lightning** until subscription utilization
audit completes.

### Tier X (NOT MIGRATEABLE): ~5 candidates

Substrates with Modal-specific runtime requirements:

- Any substrate that requires DALI NVDEC (Catalog #244-only) — none
  currently, since DALI is optional for our paradigm.
- Substrates with custom Modal Volume mount requirements (e.g.,
  large pretrained checkpoints not yet on HF Hub).

## Migration cost estimate

Per-substrate migration: ~1-2h editor work (refactor `experiments/train_*.py`
to use HF Datasets `load_dataset` instead of upstream/videos decode +
emit HF Hub `Trainer`-compatible artifacts).

5 Tier A migrations: ~5-10h editor → up to ~$50-200/year saved.
0 Tier B migrations recommended (cost premium not justified).

Net session investment: 5-10h editor.
Net annual savings: $50-200 (Tier A only).

ROI: ~$10-20/h editor saving → moderately positive; primarily justified by
**dispatch convenience** (HF Datasets-native + TrackIO + 30% cheaper t4)
not by raw cost.

## Operator decision points

1. **Approve Tier A migration sweep?** (5-10h editor; ~$50-200/year saved
   + HF-Datasets-native convenience)
2. **Approve canonical HF dataset build + upload?** (Insight 4, ~30-60
   min local M5 Max CPU + $0.01 HF Jobs)
3. **Fire the 4 new t4-small dispatches?** (Insights 1+2+3, ~$1-2 total)
4. **Migrate ANY existing substrate to HF Jobs?** (recommendation: no
   blanket migration; case-by-case for Tier A only)
5. **Audit Lightning A100 subscription utilization** (1-2h spreadsheet
   analysis; could free ~$100s if underutilized).

## Cross-references

- `hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518.md` (sister cost
  comparison memo).
- `reports/lane_maturity.md` (canonical lane registry; classification
  source).
- `feedback_deep_research_wave_landed_20260518.md` Insight 5.
- CLAUDE.md "GPU budget and compute resources — non-negotiable".
- CLAUDE.md "Production-hardened dispatch optimization protocol"
  (Catalog #270 scope for tool vs substrate dispatches).
