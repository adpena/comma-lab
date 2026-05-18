---
title: HF Jobs vs Modal vs Vast.ai cost-comparison (Insight 5)
date_utc: 2026-05-18T14:10:00Z
lane_id: lane_hf_jobs_implementation_wave_segnet_posenet_dinov3_sam2_20260518
substrate_class: apparatus_maintenance
status: design-only
horizon_class: apparatus_maintenance
predicted_band_validation_status: pending_post_training
predicted_band: null  # cost-comparison memo, no contest score prediction
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

# HF Jobs vs Modal vs Vast.ai cost-comparison

## Summary table

| Platform | GPU | $/hr | VRAM | Best for | Catalog #270 scope |
|---|---|---|---|---|---|
| HF Jobs `t4-small` | T4 | $0.40 | 16 GB | Sub-100M-param distillation; HF Dataset-native | tool dispatches |
| HF Jobs `a10g-large` | A10G | $1.20 | 24 GB | 100-500M-param training | substrate dispatches |
| HF Jobs `a100-large` | A100 | $4.00 | 80 GB | 500M+-param training; long-context | substrate dispatches |
| Modal `t4` | T4 | $0.59 | 16 GB | Established infra; quick deploys | substrate dispatches |
| Modal `a10g` | A10G | $1.10 | 22 GB shared | OOM-prone (lower than nominal due to shared) | substrate dispatches |
| Modal `a100` | A100 | $3.40 | 80 GB | Smoke + full-mode substrate training | substrate dispatches |
| Vast.ai `RTX 4090` | 4090 | $0.25 | 24 GB | OPTIMAL price/perf; 4-5x faster than T4 | substrate dispatches |
| Vast.ai `H100 SXM` | H100 | $1.50-1.99 | 80 GB | High-VRAM substrate training | substrate dispatches |
| Lightning `A100 24/7` | A100 | $0 subscription | 80 GB | Long-burn campaigns (eats subscription) | substrate dispatches |
| Local M5 Max MPS | MPS | $0 | 128 GB unified | Free advisory dev loop (MPS=NOISE per CLAUDE.md) | local-mps target |
| Local M5 Max CPU | CPU | $0 | 128 GB unified | macOS-CPU-advisory (non-promotable) | local-cpu target |

## Decision tree per substrate class

```
substrate has <100M params?
  YES -> HF Jobs t4-small ($0.40/hr)
  NO  -> substrate has <500M params?
            YES -> Vast.ai 4090 ($0.25/hr; OPTIMAL price/perf)
                   or HF Jobs a10g-large ($1.20/hr; if HF-Datasets-native)
            NO  -> Vast.ai H100 ($1.50-1.99/hr) for paid
                   or Lightning A100 ($0; eats subscription)
                   or HF Jobs a100-large ($4.00/hr; if HF-Datasets-native)
```

## HF Jobs vs Modal head-to-head

HF Jobs **advantages**:
- $0.40/hr t4-small is **30% cheaper** than Modal t4 ($0.59/hr).
- Direct integration with `adpena/comma-video-substrate-eval-600pairs`
  dataset (no need to mount upstream/videos via Catalog #152 dispatch
  protocol).
- Native TrackIO support for experiment tracking.
- 24h ledger TTL not a concern (HF Jobs has explicit dataset/model repo
  output, not result-cache).
- `hf_jobs("uv", {script: <inline>})` MCP tool — no need to manage Modal
  Volumes / sentinel files / Catalog #166 HEAD-parity ledger.

HF Jobs **disadvantages**:
- Cold-start time longer (~30-60 sec to provision t4-small).
- Smaller variety of GPU classes (no L40S, fewer A100 SKUs).
- Per CLAUDE.md cross-check: HF Jobs has NOT been the canonical contest
  dispatch path; switching ALL substrate dispatch to HF Jobs would require
  Catalog #270 dispatch optimization protocol refactor for the
  `dispatch_kind: tool` vs `dispatch_kind: substrate` scope distinction.

Modal **advantages**:
- Established infra; Catalog #244 NVML env block + Catalog #166 HEAD
  parity ledger + Catalog #245 call_id ledger all canonical for Modal.
- 53+ substrate trainers ALREADY wired for Modal dispatch.
- Generous A100 / H100 quota.

Modal **disadvantages**:
- 47% more expensive per GPU-hour on t4.
- `.spawn()` result-cache 24h TTL per CLAUDE.md "Modal `.spawn()` HARVEST
  OR LOSE" non-negotiable — must harvest within 24h or artifacts vanish.
- A10G is `shared` 22 GB VRAM, OOM-prone for substrates that need full
  16+ GB (lane_sc_plus_plus_v3 crashed at 21 GB allocation).

Vast.ai **advantages**:
- $0.25/hr RTX 4090 is the **cheapest GPU/hr in the market** for 24 GB
  VRAM dedicated.
- 4-5x faster than T4 for our 287K-param renderer workload (forward/back
  dominated by scorer pass on (B, 2, 3, 384, 512) tensors).
- 1:1 contest-compliant hardware for `[contest-CUDA]` (matches the
  contest's reference T4/A100/4090 substrate).

Vast.ai **disadvantages**:
- Manual instance lifecycle management (Catalog #225 `claim_lane_dispatch`
  + `vastai_active_instances.json` tracker per Catalog #131).
- Variable hardware quality (need `reliability>0.95` filter).
- Subject to CUDA driver version drift (Catalog #289 `uv pip install
  torch+cu124` pinning required).

## Per-substrate migration recommendation

Based on the 53-substrate canvas as of 2026-05-18 (`reports/lane_maturity.md`):

### Tier A: HF Jobs t4-small candidates (sub-100M params)
Migration ~1-2h editor per substrate. Cost saving: $0.19/hr per dispatch.

- `lane_segnet_surrogate_mobilenetv3_s_*` (NEW; this wave) — ~2.5M params
- `lane_dinov3_anchor_extraction_*` (NEW; this wave) — frozen anchor, 0 training
- `lane_macos_cpu_substrate_canvas_sweep_*` — small ranker
- (Substrates with `min_vram_gb <= 8` per recipe)

Estimated total saving: $0.20/dispatch × N dispatches/week × 52 weeks =
~$50-200/year if we run 5-10 small-substrate dispatches/week.

### Tier B: HF Jobs a10g-large candidates (100-500M params)
Migration ~1-2h editor per substrate. Cost saving: tradeoff vs Vast.ai
4090 (HF Jobs a10g-large is $1.20/hr; Vast.ai 4090 is $0.25/hr but 24 GB
VRAM, not 24 GB shared like Modal a10g).

- `lane_segnet_surrogate_sam2_hiera_tiny_*` (NEW; this wave) — ~38.9M params
- `lane_posenet_surrogate_sam2_*` (NEW; this wave) — ~38.9M params
- `lane_nscs03_end_to_end_balle_joint_codec_*` — moderate-size codec
- (Substrates with `min_vram_gb > 8 AND <= 24`)

Recommendation: **stay on Vast.ai 4090 for cost-optimal**, use HF Jobs
a10g-large for HF-Datasets-native convenience when not cost-critical.

### Tier C: HF Jobs a100-large candidates (>500M params)
Migration ~2-4h editor per substrate. Cost saving: WORSE than Modal A100
($4.00 vs $3.40/hr).

Recommendation: **stay on Modal A100 or Vast.ai H100** for cost-optimal.

### Tier D: Long-burn campaigns
Lightning A100 subscription dominates ($0 marginal cost). Use Lightning for
multi-day campaigns; use HF Jobs/Modal/Vast.ai for shorter dispatches.

## Catalog #270 dispatch optimization protocol compliance for HF Jobs

Per CLAUDE.md "Production-hardened dispatch optimization protocol" + the
recent (2026-05-17) `dispatch_kind: tool` scope clarification:

HF Jobs dispatches that train a **TOOL surrogate** (e.g., the 4 scripts in
`submitted_jobs/training_*_20260518T140408Z.py`) are out-of-scope for the
substrate-only Catalog #270 fields:
- Skipped: `--enable-autocast-fp16` (Catalog #172), TF32 (Catalog #178),
  `--enable-torch-compile` (Catalog #179), `gate_auth_eval_call`
  (Catalog #226), `min_smoke_gpu` GPU class (Catalog #215).
- Enforced: lane_id pattern, dispatch_enabled, cost_band, driver+trainer
  existence, native platform legality, min_vram_gb, video_input_strategy,
  pyav_decode_strategy, target_modes, canary_status, Modal NVML env block
  (Catalog #244), no_grad/inference_mode eval-time memory hygiene
  (Catalog #180).

HF Jobs dispatches that train a **SUBSTRATE** (contest-eligible) follow the
full Catalog #270 protocol. None of the 4 scripts in this wave are
substrates; they're all TOOL surrogates per the operator's framing.

## Recommended migration roadmap

1. **Q1 2026** (now): build the canonical dataset + the 4 surrogate scripts.
   Fire on HF Jobs t4-small after operator approves. Total cost: ~$1-2.

2. **Q2 2026**: audit the 53 substrate canvas; classify by Tier A/B/C/D
   per the table above. Migrate Tier A substrates to HF Jobs t4-small
   (estimated $50-200/year savings).

3. **Q3 2026**: evaluate if Vast.ai 4090 + HF Datasets co-location
   produces NET savings for Tier B substrates (mount HF dataset to Vast.ai
   instance via HF Hub API rather than rsync from local). Decision-grade
   prototype required.

4. **Q4 2026**: revisit Lightning A100 subscription utilization; if HF
   Jobs a100-large is cheaper for our actual usage pattern, drop the
   Lightning subscription.

## Cross-references

- `feedback_deep_research_wave_landed_20260518.md` Insight 5.
- CLAUDE.md "GPU budget and compute resources — non-negotiable" (canonical
  cost table).
- CLAUDE.md "Production-hardened dispatch optimization protocol" (Catalog
  #270 scope clarification).
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" (24h TTL constraint).
- CLAUDE.md "MPS auth eval is NOISE" (local M5 Max non-promotable axis).
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" (dual-eval mandate
  for shippable archives).
- `.omx/research/hf_jobs_substrate_migration_audit_20260518.md` (the 53-
  substrate per-lane migration audit, sister memo).
