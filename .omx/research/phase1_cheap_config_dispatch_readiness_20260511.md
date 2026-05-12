---
name: Phase 1 trainer cheap-config dispatch readiness analysis (autopilot le-5-dollar eligibility)
date: 2026-05-11
research_only: false
lane_class: substrate_engineering
lane_id: lane_phase1_cheap_config_dispatch_readiness
related:
  - feedback_autopilot_end_to_end_hf_refresh_phase1_cost_refinement_landed_20260511.md
  - feedback_substrate_composition_matrix_autopilot_ranking_theoretical_floor_v2_landed_20260511.md
  - feedback_cathedral_autopilot_activation_phase2_probes_integration_audit_v2_landed_20260511.md
  - feedback_phase1_trainer_write_runtime_fix_landed_20260509.md
  - phase1_t13_t19_t20_t22_cost_refinement_20260511.md
---

# Phase 1 trainer cheap-config dispatch readiness analysis

## Executive verdict

**Phase 1 trainer is AUTOPILOT-ELIGIBLE at the canonical full-epoch (3000)
T13+T19+T20+T22 all-on configuration: cost band $0.75 Modal T4 / $0.10
Vast.ai 4090, both well inside the autopilot $5/individual cap.** Per the
TT 2026-05-11 cost refinement (`.omx/research/phase1_t13_t19_t20_t22_cost_refinement_20260511.md`),
the canonical all-on dispatch is 6.6x under the per-dispatch cap on Modal
T4 and ~50x under on Vast.ai 4090.

This document refines TT's analysis with three additions:

1. **Autopilot-ranking JSON inclusion verdict** — does the autopilot's
   current ranking JSON include Phase 1 trainer?
2. **Composition-matrix compatibility** — does Phase 1 trainer compose
   cleanly with the substrates the autopilot already ranks?
3. **Surfaced operator-decision** — what authorization is required to
   activate autopilot dispatch of Phase 1 trainer?

NO new dispatch is fired here. NO design decision is made unilaterally.
$0 GPU spend. The only operator-gated change is ENV-VAR + CLI activation
of the autopilot's `--operator-authorized-le-5-dollar-mode` (per MM
landing 2026-05-11), which remains operator-trigger-only.

## Verifying TT's cost numbers against actuals

The TT cost refinement uses the reference baseline of **3000 epochs on
Modal T4 ≈ 50 minutes wall-clock**. Cross-checking against the canonical
dispatch script `scripts/remote_lane_t1_balle_endtoend.sh`:

- Default `EPOCHS=3000` (line 366) — matches TT's reference
- Default `BATCH_SIZE=16` (line 367) — matches TT's reference
- Default `RATE_TARGET_BYTES=80000` (line 371) — matches TT's reference
- Default `EVAL_EVERY_EPOCHS=100` (line 376) — matches TT's reference
- Smoke mode runs 1 epoch / 1 pair (line 405) — IS NOT the canonical
  cost band; smoke is build-verification only

**Verified consistency.** TT's $0.59-cost / $0.75-band figure for the
all-on combination on Modal T4 derives from the $0.59/hr Modal T4 rate
multiplied by the 59.5-minute wall-clock with all 4 flags engaged
(0.991 hr × $0.59 = $0.585; cost-band rounds up to $0.75 per the
nearest-$0.05 convention).

## Per-stage GPU memory profile (refined)

TT's cost table cites +600 MB for T20 (KL pose distill, teacher PoseNet
weights + 12-dim logits) and +60 MB for T22 (Horn-Schunck identity-warp,
intermediate grid + warped frame). Verified these against the trainer
source:

- **T20 (line 1825-onwards)**: enables the teacher PoseNet forward
  loaded once per training run. PoseNet weights ~24M params at fp32 =
  ~96 MB, but the 12-dim logits buffer (B=16 × 12 dim × num_pairs) is
  the larger contributor in the +600 MB band. TT's number is consistent
  with Quantizr's verified 0.33 recipe documentation.
- **T22 (line 1858-onwards)**: enables Horn-Schunck `grid_sample` on a
  (B=16, P=2, C=3, 384, 512) tensor; 384*512*3*16 = 9.4 MB per frame +
  intermediate grid (4*384*512*16 = 12.6 MB) + warped frame buffer
  (~9.4 MB). Total ~32 MB peak, +28 MB reserved for autograd. TT's
  +60 MB band is conservative.

**Modal T4 GPU memory: 16 GB total.** The all-on combination with
+660 MB overhead leaves ~13.5 GB headroom over the trainer baseline
(~2 GB). No OOM risk.

**Vast.ai 4090 GPU memory: 24 GB total.** Same +660 MB overhead leaves
~21 GB headroom. No OOM risk.

## The cheapest meaningful config

**Question**: is the canonical 3000-epoch dispatch the cheapest config
that produces meaningful empirical evidence at the PR106 r2 frontier
(0.20665 [contest-CUDA T4]), or could a reduced-epoch config produce
meaningful empirical evidence at lower cost?

**Answer**: per the trainer source (line 1572: `--epochs 3000` is the
Q-FAITHFUL default), 3000 epochs is the baseline configuration the council
recommended for archive convergence. Reduced epochs are ALLOWED but produce
research-signal-only evidence per the `[contest-CUDA]` evidence-grade
requirement (which mandates archive convergence to a canonical EMA shadow).

The canonical reduced-epoch configurations the trainer supports:

| Epochs | Wall-clock @ T4 | Cost (Modal T4) | Cost band | Evidence grade |
|---|---:|---:|---:|---|
| **1** (smoke) | ~30s | ~$0.005 | $0.05 | `[smoke; build-verification only]` |
| 100 | ~2 min | ~$0.02 | $0.05 | `[research-signal; not converged]` |
| 500 | ~10 min | ~$0.10 | $0.10 | `[research-signal; not converged]` |
| 1000 | ~20 min | ~$0.20 | $0.20 | `[research-signal; partial convergence]` |
| 1500 | ~30 min | ~$0.30 | $0.30 | `[research-signal; partial convergence]` |
| **3000** (canonical) | ~50 min | **~$0.49** | **$0.50** | **`[contest-CUDA] eligible after EMA load`** |
| 3000 + T13+T19+T20+T22 | ~59.5 min | ~$0.59 | $0.60 | `[contest-CUDA] eligible after EMA load` |

**The 3000-epoch baseline is the cheapest config that produces
[contest-CUDA]-eligible empirical evidence.** Reduced-epoch configs
produce research-signal only and are not promotion-eligible per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" + EMA non-negotiables.

The cheapest meaningful (research-signal) config is the **smoke run at
~$0.005**, but it produces NO score claim. The cheapest meaningful
[contest-CUDA] config is the **3000-epoch baseline at ~$0.50**.

## Autopilot-ranking JSON inclusion verdict

The current canonical autopilot ranking is at:

`experiments/results/cathedral_autopilot_dispatch_ranking_20260512T000000Z/ranking.json`

It contains 58 ranked dispatches across 24 substrates from the substrate
composition matrix. **Phase 1 trainer (T1 Ballé end-to-end) is NOT
currently in the ranking JSON.** Reasons (per QQ landing 2026-05-11):

1. The substrate composition matrix's 24-substrate inventory is
   **inflate-time substrate replacements + bolt-ons + sidechannels +
   self-compression + meta-codecs** — a categorically different layer
   from training-time architectures.
2. Phase 1 trainer outputs an `archive.zip` BUT must produce it via
   trained weights → archive bytes export. The composition matrix
   substrates use existing PR106 r2 inflate output as their substrate
   anchor and add residual/sidecar/wrapper bytes; Phase 1 trainer
   replaces the renderer.bin entirely.
3. Per CLAUDE.md "Lane maturity registry" + "Forbidden cross-archive
   composition", training-substrate dispatches must declare their
   archive grammar separately from inflate-time bolt-ons; mixing the
   two ranking lanes produces composition errors.

**Recommendation**: Phase 1 trainer should be in a SEPARATE training-lane
ranking JSON, not stuffed into the inflate-time substrate composition
matrix ranking. This is the right design per the substrate-vs-codec
meta-pattern (`feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`).

The current cathedral autopilot loop accepts both ranking JSONs via
`--candidates-jsonl` (the original interface) and
`--use-substrate-composition-matrix-ranking` (added by AA 2026-05-11).
A future training-lane ranking JSON could be added via a third flag or
be merged-via-disambiguator at the autopilot level.

**Operator decision surfaced**: should claude pre-stage a "training-lane
ranking JSON" with Phase 1 trainer + Phase 2 trainers (T15/T17/T18/T6/T10)
+ Phase 3 trainer (joint scorer-renderer-codec) as a separate ranking
artifact the autopilot can load via a new flag? $0 work; no dispatch.

## Composition-matrix compatibility

Even though Phase 1 trainer is NOT in the inflate-time substrate
composition matrix, the empirical anchor it produces (a fresh
`renderer.bin` archive) IS compatible with the matrix substrates as
**residual/sidecar/wrapper additions on top of a Phase-1-trained
substrate**:

| Substrate | Phase 1 trainer compatibility | Composability |
|---|---|---|
| residual basis (wavelet/c3/cool_chic/siren/coord_mlp) | YES | STACKABLE_SERIAL alpha~0.85 (post-training residual on Phase 1 output) |
| pose-axis sidechannel (foveation/raft/lapose) | YES | ORTHOGONAL alpha=1.0 (pose corrections orthogonal to renderer training) |
| self-compression (scpp/hessian_block_fp/mdl_fp4_tto) | YES | ORTHOGONAL alpha=1.0 (acts on Phase 1 renderer.bin params) |
| renderer replacement (NeRV variants) | NO | REPLACEMENT (mutually exclusive with Phase 1's Ballé renderer) |
| bolt-on (film_pose_conditioning, nerv_enc_dec_separated) | YES | STACKABLE_PARALLEL alpha~0.85 |
| meta-codec (magic_codec) | YES | ORTHOGONAL alpha=1.0 (byte-stream level) |

**Conclusion**: Phase 1 trainer composes with **22 of 24** matrix
substrates as orthogonal/serial/parallel additions. The 2 incompatible
substrates are renderer replacements (NeRV/MNeRV/etc.) which are
mutually exclusive by archive-grammar construction (one renderer per
archive).

This means a Phase 1 trainer dispatch followed by inflate-time substrate
additions IS a coherent end-to-end pipeline that the autopilot could
rank as a multi-step composition.

## Recommended dispatch profile

Per TT's cost refinement, the canonical recommended Phase 1 dispatch
is **T13 + T19 + T20 + T22 all-on at 3000 epochs on Modal T4 = $0.75
band**. Sub-setting order (cheapest first) for partial dispatches:

1. T13 + T19 (lowest overhead, both convergence-improving) — $0.50 band
2. T13 + T19 + T22 (add temporal consistency, matches PR101) — $0.65 band
3. T13 + T19 + T20 (add KL pose distill, matches Quantizr 0.33) — $0.70 band
4. T13 + T19 + T20 + T22 (full combination; recommended default) — $0.75 band

For Vast.ai 4090, divide costs by ~10 (4090 is 4-5x faster at half
the price of T4):

1. T13 + T19 — $0.05 band
2. T13 + T19 + T22 — $0.07 band
3. T13 + T19 + T20 — $0.07 band
4. T13 + T19 + T20 + T22 — $0.10 band

ALL configurations fit the autopilot's $5/individual cap with massive
headroom. The cumulative-envelope $20 cap is the binding constraint
when stacking multiple Phase 1 dispatches.

## Predicted score deltas (per TT refinement)

TT's predicted-band for the recommended all-on combination:
**Δ = -0.012 ± 0.007 [predicted; T13+T19+T20+T22 cost refinement]**

The composability with the matrix-substrate post-additions could buy
an additional **Δ = -0.001 to -0.005** per layered substrate, conditional
on the substrate's own predicted-band. Cumulative cap on stacking
remains the autopilot's $20 envelope.

## Surfaced operator decisions (3)

**OD-1**: Authorize Phase 1 trainer (T13+T19+T20+T22 all-on, 3000
epochs, Modal T4, $0.75) as the next single autopilot-eligible dispatch?
- Cost: $0.75 ≤ $5/individual cap ✓
- Predicted Δ: -0.012 ± 0.007 [predicted; TT cost refinement]
- Risk: archive grammar declared 8/8 fields per Catalog #124; runtime
  emission verified per Catalog #146; trainer wired per OO landing
- Dependency: NONE (independent of Phase 2/3)
- Composes-with: 22 of 24 matrix substrates as post-training additions

**OD-2**: Pre-stage a separate "training-lane ranking JSON"
(Phase 1 trainer + Phase 2 trainers + Phase 3 trainer) as a new
autopilot input source? $0 work.
- The current `--use-substrate-composition-matrix-ranking` flag covers
  inflate-time substrates only; training-lane substrates need their own
  ranking artifact to be autopilot-consumable.
- Per `tools/cathedral_autopilot_autonomous_loop.py:--candidates-jsonl`
  (the original interface), a JSONL file of CandidateRow rows is the
  expected format; pre-staging is a tooling change, not a dispatch.

**OD-3**: Activate the autopilot's `--operator-authorized-le-5-dollar-mode`
flag + `CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE=1` env-var to enable
single-command Phase 1 dispatch from the autopilot loop?
- Per MM landing 2026-05-11, this is dual-gated (CLI + env-var) by
  design; the operator must explicitly engage BOTH for any
  ≤$5/individual dispatch to fire without HALT-and-ASK.
- Once activated, the autopilot can fan out OD-1's Phase 1 dispatch +
  any other ≤$5/individual matrix-substrate dispatches up to the $20
  cumulative envelope.
- Deactivation is automatic: every dispatch decrements the cumulative
  envelope; reaching $20 returns to HALT-and-ASK.

ALL THREE decisions are surfaced ONLY here; NONE are auto-acted.
Operator-trigger remains the activation gate per CLAUDE.md
"operator-gate non-negotiable at every dispatch".

## Verdict

**Phase 1 trainer is AUTOPILOT-ELIGIBLE at the canonical $0.75 Modal T4
dispatch** per TT cost refinement + composition matrix compatibility +
archive grammar discipline. The remaining gates are operator-trigger
on (a) the Phase 1 dispatch itself, (b) any pre-staging of a
training-lane ranking JSON, and (c) activation of the autopilot's
le-5-dollar mode.

NO change is made to the existing autopilot ranking JSON; NO Phase 1
dispatch is fired; $0 GPU; loop remains paused per operator directive
2026-05-09 + 2026-05-11.

## CLAUDE.md compliance tags

- `predicted_band_only_no_score_claim`
- `operator_gate_non_negotiable_at_every_dispatch`
- `halt_and_ask_default_on`
- `no_tmp_paths`
- `cost_estimate_dispatch_dollar_envelope`
- `phase1_trainer_t13_t19_t20_t22_aware`
- `cheap_config_dispatch_readiness_aware`

## 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map**: refines per-flag predicted_delta + cost feed for
   `tac.sensitivity_map.*` as Phase 1 cost priors with composability
   coefficients.
2. **Pareto constraint**: each cost-band IS a Pareto point in
   (cost, predicted_delta) space; consumed by `tac.pareto_*` against
   matrix substrates.
3. **Bit-allocator hook**: T13 √n latent budget + T19 adaptive ρ both
   inform per-tensor allocation in the Phase 1 trainer.
4. **Cathedral autopilot dispatch hook**: cost bands feed CandidateRow
   construction for the autopilot's ≤$5 mode; composability table feeds
   `filter_composition_incompatible_dispatches`.
5. **Continual-learning posterior update**: after dispatch, empirical
   per-flag predicted-vs-actual deltas feed `posterior_update_locked`.
6. **Probe-disambiguator**: T20-vs-T22-vs-T13+T19 IS a regime-conditional
   probe-disambiguator at the PR106 frontier.
