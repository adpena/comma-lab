# Cascade C' WAVE-4: empirical anchor harvested; IMPLEMENTATION-LEVEL falsification per Catalog #307

**Date**: 2026-05-26 (UTC 2026-05-27T01:27:20Z trainer start; harvest 2026-05-27T01:45Z)
**Lane**: `lane_cascade_c_prime_option_a_build_scaffold_20260526`
**Predecessor**: `cascade_c_prime_wave_4_paired_axis_dispatched_pending_harvest_20260526.md` (commit `7b56f51e5`; APPEND-ONLY sister per Catalog #110/#113)
**Verdict**: IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307; PARADIGM INTACT; cycle DEFERRED-pending-Catalog-#325-symposium-revision per CLAUDE.md "Forbidden premature KILL without research exhaustion"
**Mission contribution** per Catalog #300: `frontier_protecting` (empirical anchor disambiguates synthesis prediction; canonical equation #344 PROMOTION blocked pending implementation fixes)

## Empirical anchor

| Field | Value |
|---|---|
| Modal call_id | `fc-01KSKGKACS7X28HM3RKDJ1MRF8` |
| Return code | **rc=0** |
| Elapsed wall-clock | 773.76 s |
| Archive sha256 | `9d1d6a20b49455a108f076e3418cb2d49e24442e1d0118c09dd58199db09a003` |
| Archive bytes | 4653 |
| canonical_score | **89.21** (numpy-fallback research-signal) |
| auth_eval_lane_tag | `[diagnostic-auth-eval]` |
| auth_eval_score_axis | `diagnostic_cpu` |
| score_claim | `false` (canonical non-promotable per Catalog #127/#221) |
| promotion_eligible | `false` |
| ready_for_exact_eval_dispatch | `false` |
| score_pose_contribution | 38.72 (avg_posenet_dist = 149.95) |
| score_seg_contribution | 50.48 (avg_segnet_dist = 0.505) |
| score_rate_contribution | 0.0031 (rate_unscaled = 0.0001239) |
| frame_1_routing_pct | **2.33%** (synthesis predicted 25.2% — 10× off) |
| score_delta_research_signal | -0.0004967040501323647 |
| canonical_equation_proposal | `atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1` |
| canonical_equation_status | **FORMALIZATION_PENDING** (NOT promoted; per Catalog #344) |
| predicted_band_validation_status | `pending_post_training` (per Catalog #324) |
| substrate_id | `cascade_c_prime_frame_1_segnet_waterfill` |
| smoke | `false` (1-epoch full mode per env `CASCADE_C_PRIME_EPOCHS=1`) |
| seed | 20260526 |
| n_pairs | 600 |

## Comparison vs canonical frontier per Catalog #343

Per `.omx/state/canonical_frontier_pointer.json` (refreshed 2026-05-27T01:38:20Z):

- `our_local_frontier_contest_cpu.score` = 0.19202828295713675 (sha `7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe`; `linux_x86_64_cpu`)
- `our_local_frontier_contest_cuda.score` = 0.20533002902019143 (sha `9cb989cef519...`; `linux_x86_64_t4`)

Cascade C' WAVE-4 diagnostic-cpu score 89.21 is **464× worse** than the contest-CPU frontier. This is NOT a frontier-band signal; it is a 1-epoch L0 SCAFFOLD smoke producing a defective per-pair routing decision (frame-1 selected only 2.33% of the time vs synthesis 25.2% — the Lagrangian dual is collapsing toward frame-0 monopoly).

## Catalog #307 paradigm-vs-implementation classification

- **PARADIGM** (Atick-Redlich asymmetric scorer channel theory): **INTACT**. The mathematical claim that SegNet's `x[:,-1,...]` slice creates a free-cost frame-0 vs full-cost frame-1 asymmetric channel is unchanged; sister #1324 PoseNet-null measurement at 22.3% frame-1 ratio remains valid evidence.
- **IMPLEMENTATION** (L0 SCAFFOLD trainer + inflate + per-pair routing decision logic): **FALSIFIED at this configuration**. The 1-epoch L0 SCAFFOLD produced score 89.21 because (a) PoseNet response is broken (avg_posenet_dist 149.95 — orders of magnitude above frontier's ~10⁻⁵), suggesting inflate runtime or eval roundtrip is mis-rendering frame-1; (b) per-pair Lagrangian dual routing is converging to frame-0 monopoly (2.33% frame-1 vs 25.2% synthesis prediction); (c) score-seg dominance (50.48) suggests SegNet is also mis-rendering at full-frame resolution.

Per CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable: this is **DEFERRED-pending-implementation-iteration**, NOT KILLED. Default verdict for one-config failure is `DEFERRED-pending-research`. Reactivation criteria enumerated below.

## Reactivation criteria (Catalog #308 alternative probe methodologies; N=5)

1. **Inflate runtime fix**: Catalog #295/#205 sister extends — verify `submissions/cascade_c_prime/inflate.py` (vendored into submission_dir per sister codex commit `5bcb53070`) actually renders frame-1 from per-pair routing decision bytes vs renders frame-0 placeholder. Empirical signature: avg_posenet_dist drops from 149.95 toward ~10⁻⁵.
2. **Eval roundtrip discipline**: per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE": verify the trainer's per-pair routing uses `apply_eval_roundtrip_during_training` per `tac.differentiable_eval_roundtrip` before scoring; smoke artifact may have eval_roundtrip=False at the routing decision surface.
3. **Per-pair Lagrangian dual coefficient calibration**: the 2.33% frame-1 vs 25.2% synthesis suggests the Lagrangian dual variable is mis-set; recalibrate via sister `tac.findings_lagrangian.posterior_update_from_anchors` consuming sister #1324 PoseNet-null 22.3% prior + this WAVE-4 anchor (per Catalog #355).
4. **Smoke-before-full re-dispatch per Catalog #167**: route via `tools/run_modal_smoke_before_full.py --recipe substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch --smoke-only` ($0.30 cheap smoke; validate fix BEFORE full canary).
5. **Per-substrate symposium per Catalog #325**: trigger re-deliberation when an alternative configuration empirically materializes a frame-1 routing > 15% (closer to sister #1324 22.3%) + canonical_score < 1.0 (within 5× of frontier band). The symposium's PROCEED verdict is the unblock condition for canonical equation #344 PROMOTION.

## Canonical paired-axis routing (Catalog #246) is the next operator-routable step

The WAVE-4 dispatch routed `auth_eval_device: "cpu"` + `auth_eval_advisory_only: true` per recipe cost-band smoke variant. For a true paired-CUDA `[contest-CUDA]` + `[contest-CPU]` anchor needed to PROMOTE canonical equation #344 + validate Catalog #324 predicted band:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive .omx/research/cascade_a_fec10_hybrid_artifacts_20260526/.../archive.zip \
    --skip-axis-if-promotable-anchor-exists
```

PRE-CONDITION: must FIRST land an implementation iteration that resolves the 5 reactivation criteria above (frame_1_routing > 15% AND canonical_score < 1.0). Firing paired-CUDA on the current 89.21-scoring archive would burn ~$0.30-0.50 measuring a falsified implementation.

## Discipline declaration

- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: predecessor memo `cascade_c_prime_wave_4_paired_axis_dispatched_pending_harvest_20260526.md` UNCHANGED; this is a NEW sister memo
- Catalog #117/#157/#174 canonical serializer + POST-EDIT `--expected-content-sha256`
- Catalog #119 Co-Authored-By Claude trailer
- Catalog #206 final checkpoint complete
- Catalog #127 per-call-site custody routing: empirical anchor carries `score_claim=false` + `promotion_eligible=false` + `ready_for_exact_eval_dispatch=false` + `score_axis=diagnostic_cpu` (canonical non-promotable triple)
- Catalog #221 result artifact fail-closed for score claims: `auth_eval_lane_tag=[diagnostic-auth-eval]` + `auth_eval_score_axis=diagnostic_cpu` + `auth_eval_score_claim_valid=false`
- Catalog #287 placeholder rejection: zero placeholders in this memo
- Catalog #307 paradigm-vs-implementation: PARADIGM INTACT classified explicitly
- Catalog #308 N=5 alternative probe methodologies enumerated
- Catalog #313 probe-outcomes ledger: operator-routable to append `INDEPENDENT` outcome with this anchor's sha for future dispatch refusal until reactivation criteria met
- Catalog #319/#322/#323 canonical Provenance: every numeric carries axis+hardware+evidence_grade
- Catalog #324 predicted_band_validation_status: `pending_post_training` (NOT validated)
- Catalog #343 frontier comparison via canonical pointer (not hardcoded literals)
- Catalog #344 PROMOTION DEFERRED: `FORMALIZATION_PENDING` preserved
- Catalog #355 meta-Lagrangian wire-in: anchor consumable by `tac.findings_lagrangian.posterior_update_from_anchors`

## 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map: N/A (single dispatch; no per-pair sensitivity surfaced)
- Hook #2 Pareto constraint: N/A (1-epoch L0 SCAFFOLD; Pareto polytope not yet visited)
- Hook #3 bit-allocator: N/A (archive bytes 4653 fixed by ZIP-member grammar)
- Hook #4 cathedral autopilot dispatch: **ACTIVE** (Modal call_id terminal `harvested` event auto-fires per Catalog #343; autopilot ranker consumes diagnostic-CPU anchor as non-promotable observability-only signal per Catalog #341)
- Hook #5 continual-learning posterior: **ACTIVE** (post-harvest auto-refresh appends to `.omx/state/continual_learning_posterior.jsonl` per canonical frontier pointer hook; row carries `score_claim=false` so it does NOT promote)
- Hook #6 probe-disambiguator: **ACTIVE** (this WAVE-4 anchor IS the canonical disambiguator between synthesis prediction -0.058820 vs empirical implementation-level 89.21; operator-routable to add `INDEPENDENT` outcome to `.omx/state/probe_outcomes.jsonl` per Catalog #313)

## Cycle closure verdict

**DEFERRED-PENDING-IMPLEMENTATION-ITERATION + Catalog #325-SYMPOSIUM-REVISION**. PARADIGM INTACT. WAVE-5 sister subagent operator-routable after one or more of the 5 reactivation criteria empirically lands.

## Operator-routable next steps

1. **Sister subagent WAVE-5** addresses reactivation criteria #1 (inflate.py runtime per-pair routing rendering correctness) + #2 (eval_roundtrip wiring discipline) — likely the dominant signal.
2. **Cheap smoke re-dispatch** per Catalog #167 `tools/run_modal_smoke_before_full.py --smoke-only` post-fix (~$0.30) BEFORE any paired-CUDA full canary.
3. **Catalog #313 probe-outcomes ledger entry**: register `INDEPENDENT` outcome on substrate `cascade_c_prime_frame_1_segnet_waterfill` to block future dispatch on the SAME archive sha until reactivation per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable.
4. **Catalog #325 symposium re-deliberation** when an alternative configuration empirically materializes (per reactivation #5).
