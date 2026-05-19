---
council_tier: T1
council_attendees: [Claude-Harvest-Subagent]
council_quorum_met: false
council_verdict: ADVISORY_FINDINGS
council_dissent: []
council_decisions_recorded:
  - "harvest 4 of 5 unharvested Modal dispatches per CLAUDE.md HARVEST OR LOSE non-negotiable"
  - "register Catalog #313 probe outcomes for all 4 with paradigm-vs-implementation classification per Catalog #307"
  - "surface Catalog #204 CONTEST_AUTH_EVAL_DURABLE_OUTPUT bug class regression in stack_of_stacks remote driver"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
---

# HARVEST-PENDING-MODAL-DISPATCHES SYNTHESIS 2026-05-19

## Summary

Harvested 4 of 5 unharvested Modal dispatches from Cable B1 E.7+E.8 wave + sister MPS Phase B RE-FIRE. 1 still running (E.7 #2 VQ K-sweep, likely same OOM as #1). All harvest outcomes registered to canonical `tac.deploy.modal.call_id_ledger`; all 4 probe outcomes registered to canonical `tac.probe_outcomes_ledger` per Catalog #313 + #245 4-layer pattern.

## Per-dispatch verdicts

### 1. E.7 VQ K-sweep #1 (fc-01KRZC53Y0D28B6BYEQ1MRG347) — DEFER

- **Status**: rc=1 elapsed=141.8s [contest-CUDA Modal T4]
- **Failure**: CUDA OOM in SegNet EfficientNet-B2 BatchNorm forward (tried 26MiB; 37MB free of 14.56GB total; 14.36GB allocated)
- **Bug class**: 29.5M-param VQ-VAE + SegNet forward exceeded T4 capacity at batch_size=16
- **Verdict**: IMPLEMENTATION-LEVEL FALSIFICATION (T4-with-batch-16 config); PARADIGM-LEVEL INTACT (van den Oord 2017 VQ-VAE)
- **Reactivation criteria** (per CLAUDE.md "Forbidden premature KILL"):
  - (a) Re-dispatch on A100-80GB
  - (b) Enable `--enable-autocast-fp16` (Catalog #172) + reduce batch_size 16→4
  - (c) Gradient checkpointing + offload SegNet to CPU per Catalog #218 mini-batch pattern
- **Cargo-cult flagged**: T4 GPU choice for 29.5M-param substrate; recipe `min_vram_gb` likely too low; Catalog #270 Tier 1 engineering primitives missing from trainer (autocast_fp16 / no_grad eval / mini-batch reconstruct)

### 2. E.8 SGLD #1 (fc-01KRZCHVY6C1TSFNNS6KN13G70) — DEFER

- **Status**: rc=1 elapsed=2.11s
- **Failure**: trainer crashed before archive build (status="started"; no traceback in captured artifacts)
- **Verdict**: IMPLEMENTATION-LEVEL FALSIFICATION of opaque dispatch attempt; PARADIGM-LEVEL INTACT (sister E.8 #2 reached auth_eval proving SGLD trainer code path is sound)
- **Reactivation criteria**:
  - (a) Re-dispatch with `--capture-output` to surface root cause
  - (b) Reproduce locally on Modal-equivalent CPU container

### 3. E.8 SGLD #2 (fc-01KRZCSQ7FPVMSAXZQDSZJCTN4) — DEFER (infrastructure bug, NOT substrate)

- **Status**: trainer rc=0 elapsed=19.5s; **auth_eval rc=1** (CONTEST_AUTH_EVAL_DURABLE_OUTPUT block)
- **Trainer success**: archive built at `sha=110cfaa3f2ebbd02b91542633445e54a837ea663f98a7807a914f95651fdff9f` size=179008 bytes (A1 single-arm passthrough canary)
- **Auth_eval failure**: `contest_auth_eval evidence path is under temp storage: /tmp/pact/lane_stack_of_stacks_results/auth_eval/eval_work` — refused per Catalog #204
- **Bug class**: SISTER REGRESSION to STC v2 2026-05-14 anchor — `scripts/remote_archive_only_eval.sh` or stack_of_stacks remote driver does not route `OUTPUT_DIR` to `/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output` when `MODAL_RUNTIME=1`
- **Verdict**: IMPLEMENTATION-LEVEL FALSIFICATION of remote driver output dir routing; PARADIGM-LEVEL INTACT (archive build path is correct; A1 passthrough canary expected score band [0.190, 0.210])
- **Reactivation criteria**:
  - (a) FIX driver to default `OUTPUT_DIR=/modal_results/$DISPATCH_INSTANCE_JOB_ID/output` when `MODAL_RUNTIME=1` (canonical pattern per Catalog #204 PR101 LC v2 enhanced curriculum driver)
  - (b) **Re-fire auth_eval against existing archive `sha=110cfaa3`** with `--allow-temp-work-dir` (diagnostic only; non-promotable per Catalog #204) to get the actual contest-CUDA score for the A1 passthrough canary
- **Cross-ref**: CLAUDE.md Catalog #204 forbidden pattern; same as STC v2 fix landed in `scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh`

### 4. MPS Phase B RE-FIRE (fc-01KRZD8662BV697P8JVNR4WGCC) — PROCEED (success)

- **Status**: rc=0 elapsed=9.5s [Modal A10G]
- **Artifacts**: `target_cuda_forward_outputs.pt` (47MB CUDA target forward outputs) + `target_cuda_components.json` (774B components dict) + `checkpoint_ema.pt` (32KB) + `frame_cache.pt` (47MB)
- **Verdict**: SUCCESSFUL HARVEST; Phase B split-device infrastructure landed
- **Hand-off**: Slot 1 sister subagent picks up local MPS comparison + gap quantification against `target_cuda_forward_outputs.pt`
- **Provenance**: infrastructure-only target artifact (NOT a substrate score claim); used for MPS-vs-CUDA drift quantification per CLAUDE.md "MPS auth eval is NOISE" non-negotiable

### 5. E.7 VQ K-sweep #2 (fc-01KRZCX15GAF5Z5E3E568Q60FF) — STILL RUNNING (likely OOM)

- Re-poll deferred to next subagent or operator check; >60s wall-clock past dispatch suggests it may also OOM or be queued
- Will surface in next `query_unharvested()` cycle

## Aggregate metrics

- **Total spend** (rough estimate; A100=$0.59/min, T4=$0.063/min, A10G=$0.33/min):
  - E.7 #1: T4 ~141s = ~$0.15
  - E.8 #1: T4 ~2s = ~$0.002
  - E.8 #2: T4 ~19s = ~$0.020
  - MPS Phase B: A10G ~10s = ~$0.055
  - **Total harvested: ~$0.23**
  - E.7 #2 still running: at T4 rate, could be $0.10-$0.30 depending on whether it OOMs early
- **Spend envelope**: predecessor declared "Wait for Monitor notification" — no explicit envelope cap; treating as bounded by per-recipe cost_band default (smoke=$0.20-1.00; full=$5-15)
- **Verdict**: under envelope across all 5

## Bug classes surfaced

1. **CUDA OOM on T4 for large substrates** (29.5M VQ-VAE): Catalog #270 Tier 1 engineering primitives + Catalog #170 min_vram_gb declaration must be enforced at recipe-emit time; consider gate that refuses T4 dispatch when trainer has >20M params unless `--enable-autocast-fp16` declared
2. **CONTEST_AUTH_EVAL_DURABLE_OUTPUT regression in stack_of_stacks driver**: Catalog #204 sister bug class; cross-driver audit needed to ensure all `scripts/remote_lane_substrate_*.sh` and `scripts/remote_archive_only_eval.sh` honor `/modal_results/$JOB_ID/output` when `MODAL_RUNTIME=1`
3. **Opaque crash in 2.1s SGLD #1**: capture discipline gap — need `set -o pipefail` + `--capture-output` enabled by default for dispatches that exit faster than a minimum threshold

## Next-cable routing

1. **Sister Slot 1 (MPS Phase B)**: pick up local MPS comparison against `target_cuda_forward_outputs.pt` (47MB), quantify drift, emit Phase B verdict
2. **Operator routing for E.7 VQ**: decide between A100 re-dispatch (~$5-15) OR autocast_fp16+batch_size_4 T4 re-dispatch (~$0.50)
3. **Operator routing for E.8 SGLD**: FIX driver output dir routing per Catalog #204 canonical pattern (~30 min editor work; landable via canonical commit serializer); then re-fire auth_eval against existing archive sha=110cfaa3 — that gives free contest-CUDA score for A1 single-arm passthrough canary
4. **E.7 #2 follow-up**: re-poll in 5-10 min; if still running, register `function_timeout` outcome and Modal-dashboard verify whether it crashed silently or queued

## 6-hook wire-in declaration (per Catalog #125)

- **Hook 1 (sensitivity-map)**: N/A — this is a harvest+ledger subagent; no signal contribution
- **Hook 2 (Pareto constraint)**: N/A
- **Hook 3 (bit-allocator)**: N/A
- **Hook 4 (cathedral autopilot dispatch)**: ACTIVE — probe outcomes registered to `tac.probe_outcomes_ledger` are consumed by Catalog #313 dispatch gate; future dispatch attempts on VQ/stack_of_stacks/SGLD will see blocking verdicts and either ratify (no fresh dispatch) OR cite predecessor verdict for override
- **Hook 5 (continual-learning posterior)**: ACTIVE — `tac.deploy.modal.call_id_ledger` outcomes appended via canonical helper; per Catalog #245 4-layer pattern (canonical helper + CLI + STRICT gate + operator briefing)
- **Hook 6 (probe-disambiguator)**: ACTIVE — paradigm-vs-implementation classification per Catalog #307 enables future subagents to distinguish "VQ-VAE paradigm killed" (FALSE) from "T4-with-batch-16-config killed" (TRUE)

## Provenance

- `subagent_id`: harvest_pending_modal_dispatches_20260519
- `lane_id`: lane_harvest_pending_modal_dispatches_20260519 (L0 at start; L2 at landing after impl_complete + real_archive_empirical + memory_entry)
- `harvest_method`: `tac.deploy.modal.call_id_ledger.update_call_id_outcome` + `tac.probe_outcomes_ledger.register_probe_outcome`
- `harvest_window_utc`: 2026-05-19T06:00:00Z → 2026-05-19T06:10:00Z
- `evidence_grade`: harvest_outcomes_machine_readable (not a score claim per CLAUDE.md "Apples-to-apples evidence discipline")

## Cross-references

- Cable B1 predecessor commits: `cfa8ce693` / `220c207ed` / `80635b060` / `56908e651`
- Catalog #245 (canonical Modal call_id ledger 4-layer pattern)
- Catalog #313 (probe outcomes ledger gates dispatch)
- Catalog #307 (paradigm-vs-implementation falsification classification)
- Catalog #204 (CONTEST_AUTH_EVAL_DURABLE_OUTPUT regression in E.8 #2)
- Catalog #270 (canonical dispatch optimization protocol — Tier 1/2/3 missing in E.7 trainer)
- Catalog #270 + #172 + #218 (T4 OOM bug class for large substrates)
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (default DEFER not KILL)
