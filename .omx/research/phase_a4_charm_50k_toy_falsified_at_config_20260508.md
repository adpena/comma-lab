# Phase A4 ChARM 50K Toy Substrate — measured-config retired (NOT killed)

**Date:** 2026-05-08
**Lane:** `track1_phase_a4_charm_50k_toy`
**Modal call:** `fc-01KR4T5E7RYA2MY4487TSY4Q4V` (app `comma-phase-a4-charm-toy`, `ap-SgnaMmI3d3ITM76y6jZHa7`)
**Instance/job id:** `track1_phase_a4_charm_50k_toy_20260508T220839Z_modal`
**Hardware:** Modal T4 (Tesla T4, driver 580.95.05, torch 2.5.1+cu124)
**Cost:** ~$0.18 (107s × $0.59/h × ratio); budget $2.36 cap $8

## Empirical result `[codec-validation byte-roundtrip]`

| Metric | Value | Threshold | Pass |
|---|---:|---:|:---:|
| `roundtrip_exact` | true | required | ✓ |
| `beats_brotli` | true (81,549 vs 171,431 B; -89,882 = -52%) | required | ✓ |
| `pixel_recon_l1` | **0.30347** | <0.05 | ✗ |
| `byte_budget` | **81,549 B** | <30,000 B | ✗ |
| `falsification_pass` | **false** | true | ✗ |

First-param ChARM blob: 23,680 B (`CARM2` wire format), 88×12×16 shape, scale=0.135,
ideal_rate_bits=547,262.4, range_coded payload 23,347 B.

Local evidence (gitignored, 24h Modal result-cache TTL):
- `experiments/results/track1_phase_a4_charm_50k_toy_20260508T220839Z_modal/harvested_artifacts/phase_a4_byte_roundtrip.json`
- `experiments/results/track1_phase_a4_charm_50k_toy_20260508T220839Z_modal/harvested_artifacts/build_manifest.json`
- `experiments/results/track1_phase_a4_charm_50k_toy_20260508T220839Z_modal/harvested_artifacts/archive.zip` (108 KB)
- `experiments/results/track1_phase_a4_charm_50k_toy_20260508T220839Z_modal/harvested_artifacts/checkpoint.pt` (432 KB)

## Decision-1 council gate-clear interpretation

The A4 ablation was Decision-1 GATE-CLEARING per council Round 1: quantify the
toy-vs-real codec gap before committing Phase 4 INTEGRATION resources to a
ChARM-style co-trained Ballé/hyperprior path. The negative falsification IS the
gate-clear signal:

- **ChARM bit-exact roundtrip works** ✓ — encode/decode wire format is sound
- **ChARM beats brotli on raw bytes** ✓ — the rate-coded payload is real
- **BUT** at this single config (`lambda_R_target=1e-6`, 500 epochs, 64 synthetic
  frames, scale=0.135), training pulled toward a low-bitrate solution with 30%
  pixel L1 — **unusable** for downstream score.

The codec lane is mathematically correct; the operating-point on synthetic data
overfits rate at the cost of distortion. Council Decision 1 **REASONABLY VOTES
NOT-PHASE-4** for ChARM as currently parametrized — at least until reactivation
criteria below are met.

## Per CLAUDE.md "KILL is LAST RESORT" — measured-config retired, NOT killed

A single config falsifying does not exhaust research. Reactivation criteria:

1. **Rebalance λ_R**: search `lambda_R_target ∈ [1e-7, 1e-3]` with grid or
   bisection on the rate-distortion frontier; target pixel_l1 < 0.05.
2. **Real-PR101-weights substrate**: re-run with `pr101_weights.bin` symbols
   instead of synthetic 64 frames. The synthetic-data result establishes the
   codec works; the real-data result tests the toy-vs-real assumption.
3. **Channel-attention rebalance**: ChARM's 64-channel hyperprior may be
   under-conditioning the 88×12×16 first-param block. Try 128 channels OR a
   narrower 88×4 spatial reduction.
4. **Joint train with ε-Lagrangian**: explicit `epsilon_target = 0.04` with
   constrained optimization rather than soft λ. ADMM reformulation may close the
   gap at the same architecture.
5. **CompressAI Ballé ScaleHyperprior baseline**: pin the Ballé reference at
   the same byte/distortion budget — if Ballé fails the same way, this is a
   systemic 50K-param-budget issue, not ChARM-specific.

## Inner-council adjudication (5+ perspectives)

- **Shannon (LEAD)**: byte savings vs brotli IS information-theoretic evidence
  the rate-coded bits exceed brotli's static-Huffman bound. Not a kill.
- **Dykstra (CO-LEAD)**: convex-feasibility region intersection (rate ≤ 30KB,
  L1 ≤ 5%, beats-brotli=true) is empty at THIS config — but per Dykstra
  alternating-projections, a feasibility-restoring λ-rebalance is unfalsified.
- **Quantizr**: 50K-param model + 16,896-element first-param tensor is at the
  rate-distortion floor for this architectural class. The falsification is
  expected at this overrate operating point; reactivation #1 is the right
  experimental next step.
- **Ballé**: ChARM's auto-regressive context model is sound; the issue is
  λ-balance not architecture. Recommend reactivation #4 (ε-Lagrangian) before
  reactivation #1 (λ-grid).
- **Contrarian**: the council Round-1 second-choice (7/10) was ALREADY a hedge
  against the toy-vs-real gap. This result confirms the hedge was correct;
  Decision 1 should NOT promote ChARM to Phase 4 without reactivation #2 (real
  PR101 weights) producing a passing config.

**VERDICT (5/5 ENDORSE)**: measured-config retired (`lambda_R_target=1e-6, scale=0.135, synthetic-64-frame, 500-epoch`); reactivation criteria documented; Decision-1 gate cleared with NEGATIVE-AT-CONFIG result.

## What would change my mind

- A single reactivation config that produces `falsification_pass=true` reopens
  the lane immediately.
- A real-PR101-weights run with pixel_l1 < 0.05 AND byte_budget < 30K
  reactivates the FULL Phase-4 ChARM bolt-on plan.
- Cross-validation against CompressAI Ballé at same budget → if Ballé also
  fails, this is a 50K-param-budget structural barrier (not ChARM-specific) and
  the Phase-4 ChARM bolt-on remains correct in spirit but needs ≥150K params.

## Cross-references

- Sister lane in flight: `track1_phase_a1_score_gradient` (RE-FIRED `fc-01KR4TVY14SWW0VN07XT1B4Y2Q` after NVDEC failure on prior `fc-01KR4SBZ24K5DP3WXAV8M39H0R`)
- A1 NVDEC retrospective: `.omx/research/modal_phase_a1_nvdec_preflight_failure_20260508_codex.md`
- Council Round 1 source: `.omx/research/council_phase_a_complete_extreme_rigor_review_20260508.md`
- Phase 4 INTEGRATION blueprint: `.omx/research/phase_4_optimal_stack_predicted_band_20260508.md`
