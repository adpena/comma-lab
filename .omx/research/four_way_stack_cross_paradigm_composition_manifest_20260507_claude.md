---
title: Four-way stack × cross-paradigm composition manifest
date: 2026-05-07
author: Claude (continuation per "continue with all with the four way stack and also with the paradigm and cross paradigm")
status: DESIGN — Op 1 in flight (subagent a087d8f145eb8ff66); Op 2-4 + paradigm composition queued
score_claim: false
---

## Goal

Compose the **four engineering operations** (Op 1-4 from `pr_top3_bit_level_deconstruction_20260507_claude.md`) WITH the **paradigm work** (PARADIGM-α / β / γ / δεζ from project history) into a single ordered execution plan.

## The four-way engineering stack

| # | Op | Source | Saves | Predicted | Risk |
|---|---|---|---|---|---|
| 1 | PR101 split-Brotli + byte-maps on any HNeRV substrate | PR101 codec.py port (in flight, subagent `a087d8f145eb8ff66`) | -7,963 B vs PR106 | ~0.205 on PR106 | Trivial — deterministic byte port |
| 2 | PR103 arithmetic-coded densest payloads | PR103 codec via `constriction` range coder | -290 B on top of Op 1 | ~0.1925 | Depends on `constriction` runtime + 8-tensor selection |
| 3 | Our apogee_int6 weights × PR101 schema | int6-quantized HNeRV weights → PR101 split-Brotli | -10-15KB more | ~0.189 | Basin-parity must hold for smaller weights (already PASS for int6 alone; compounding with split-Brotli unverified) |
| 4 | Stack ALL: PR101 schema + PR102 inference tuning + PR103 AC + our int6 + PR98 channel postprocess | composition | cumulative | **~0.185-0.187** | Cross-trick interaction risk |

## The cross-paradigm dimension

Per project history, four paradigms have been pursued:

| Paradigm | Domain | Status |
|---|---|---|
| **α** | Mask payload overhaul (NeRV / wavelet / VQ-VAE / grayscale-LUT) | COMPLETE Phase 1 (#304) |
| **β** | Sensitivity-aware everything (Ω-W-V3 + IMP/19/20 weighting) | COMPLETE (#305) |
| **γ** | Joint score-aware codec stack (ADMM + Ballé + arithmetic) | COMPLETE (#306); runtime wiring this session via JCSP |
| **δεζ** | Joint training + Self-Compress NN + MDL/Bayesian (sub-0.30 target) | Phase 1 scaffolded `6fe8125b`; Phase 2-5 pending GPU |

## Composition matrix: four-way × cross-paradigm

```
                       Op 1            Op 2            Op 3            Op 4 (full)
                       PR101 split-B   +PR103 AC       +int6 weights   +PR102 tuning
                       on HNeRV        on Op1          on Op2          on Op3
PARADIGM-α (NeRV mask) replace mkv     replace mkv     replace mkv     replace mkv
                       in Op1 stack    in Op2 stack    in Op3 stack    in Op4 stack
                       → -150KB mask   compose         compose         compose
                       Δ score -0.10   Δ score -0.10   Δ score -0.10   Δ score -0.10
                       NET ~0.105      NET ~0.092      NET ~0.089      NET ~0.085

PARADIGM-β (sensitivity replace flat   compose         apogee weights  full stack +
weighted)              brotli with     with AC         + sensitivity   sensitivity
                       Ω-W-V3 wf       biased weights  weighted        per-channel

PARADIGM-γ (joint      score-aware     score-aware AC  score-aware     full joint
score-aware codec)     PR101 encoding  on weights      apogee weights  optimization

PARADIGM-δεζ (joint    REPLACE PR101   REPLACE PR101   REPLACE PR101   from-scratch
training + self-       WITH δεζ-       WITH δεζ-       WITH δεζ-       δεζ pipeline
compress NN)           trained         trained + AC    trained + ε     produces
                       JFG/HNeRV       hyperprior      learned prior   smallest blob
```

## Concrete next-steps after Op 1 lands

### Wave A — pure-engineering (CPU-only, no new ML)

1. **Op 1 in flight**: PR101 split-Brotli port (subagent `a087d8f145eb8ff66`). Predicted -7,963 B on PR106 → ~0.205.
2. **Op 2**: PR103 AC on top of Op 1. Port `constriction` range-coder wrapping from PR103 codec. Predicted -290 B on top of Op 1 → ~0.205-0.0002 = ~0.2048 (note: PR103's claim was -290B vs flat brotli; on top of split-Brotli the marginal may be smaller).
3. **Op 2.5**: PR102 inference-time tuning on top of Op 1 stack (latent correction scale 0.0100→0.0095, frame-0 red −1). FREE bytes; ε > 0 score reduction.
4. **Op 3**: apogee_int6 quantized weights × PR101 schema. Take int6's 170,450-byte archive, extract 28 quantized weight tensors, run them through Op 1's encoder. Predicted -10-15KB additional → ~0.190.
5. **Op 4**: full stack composition. Predicted **~0.185-0.187**.

### Wave B — α-paradigm composition (mask payload overhaul)

PARADIGM-α replaces masks.mkv with NeRV / wavelet / VQ-VAE / grayscale-LUT alternatives. The HNeRV decoder substrate (PR101/106) doesn't have a separate masks section — its `decoder_blob + latent_blob + sidecar` IS the entire payload. **PARADIGM-α composes orthogonally only if** we adopt the SegMap-renderer-with-mask substrate (Quantizr/G-v3 lineage), not the HNeRV-decoder substrate.

Decision point: does the four-way stack apply to:
- (a) HNeRV decoder substrate (current top-3 are all this) → α doesn't apply directly
- (b) SegMap-renderer-with-mask substrate (Q-FAITHFUL, G-v3, owv3 lineage) → α replaces masks.mkv with α-encoder; Op 1's split-Brotli applies to renderer.bin

**Council recommendation (in-line micro-deliberation):**
- Shannon: HNeRV is the empirically-proven public top-3; SegMap-with-mask is our internal-frontier track. They're SUBSTRATES, not paradigms — both can host the four-way stack. ENDORSE: build Op 1-4 against HNeRV first (deterministic public-replay path), THEN port the same tricks to SegMap-with-mask substrate as a separate wave.
- Hotz: ship Op 1-4 on HNeRV first (proven substrate). 4 commits. Then re-test the same operations on SegMap-with-mask. ENDORSE.
- Contrarian: porting tricks across substrates is non-trivial — PR101's DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28) is HNeRV-specific tensor counts. SegMap has different layer count → splits would differ. CONDITIONAL ENDORSE: re-derive splits for SegMap-substrate before claiming savings.

### Wave C — β-paradigm composition (sensitivity-weighted)

PARADIGM-β (sensitivity-weighted Ω-W-V3, IMP/19/20 weighting) is a TRAINING-time technique that produces a BETTER MODEL → smaller post-quantization bytes. Composes by:
- Train a model with β-sensitivity-weighted loss (GPU)
- Apply Op 1 split-Brotli to the resulting weights
- Net effect: β reduces underlying entropy of weights, Op 1 captures the gain in fewer bytes

Currently β-paradigm is COMPLETE per #305. The next step is to actually re-train a β-weighted HNeRV/SegMap model and stack Op 1-4 on it. Multi-day GPU work.

### Wave D — γ-paradigm composition (joint score-aware codec stack)

PARADIGM-γ already has runtime infra committed this session (γ-JCSP StreamSource + JCSP score-marginals harness, #378 / #379). It enables joint optimization across multiple codec streams using ADMM + Ballé + arithmetic.

γ composition with the four-way: the four-way stack already uses arithmetic coding (Op 2) and split-Brotli (Op 1). γ adds the JOINT optimization across these streams — instead of choosing each codec independently, jointly optimize (Op 1 split-Brotli boundaries × Op 2 AC histograms × Op 3 quantization bit allocation) under a single rate-distortion loss.

Implementation: feed the four-way stack's intermediate streams into γ-JCSP's coordinator. The coordinator finds joint Pareto-optimal codec parameters. CPU-only optimization (no GPU needed once weights are given).

### Wave E — δεζ-paradigm composition (paradigm-shift level)

PARADIGM-δεζ is the from-scratch joint-training + self-compress + MDL/Bayesian path. Phase 1 scaffolded `6fe8125b`. Phase 2-5 are GPU work.

δεζ composition with the four-way: δεζ produces a trained checkpoint that's smaller-by-construction (self-compressing + MDL-optimal). Op 1-4 then apply on TOP of that checkpoint. Predicted: δεζ alone ~0.155; +Op 1-4 = ~0.150 [predicted-band, very speculative].

## Sequential execution plan (per user "sequentially fully and comprelely")

1. **Op 1** (in flight, subagent `a087d8f145eb8ff66`)
   - Delivers: PR101 codec port, validated byte-map savings, encoder + decoder + tests + CLI + lane registration
   - Adversarial review: Round 1-3 to clean
   - Empirical: bytes saved on PR106 substrate

2. **Op 2** (next, after Op 1 lands)
   - Port PR103's `constriction` AC wrapping for the 8 largest weight tensors + latent-hi
   - Tests + CLI + lane registration
   - Adversarial review: Round 1-3 to clean
   - Empirical: bytes saved on top of Op 1

3. **Op 2.5** (parallel to Op 2 — small)
   - Port PR102's inference-time tuning constants (scale 0.0100→0.0095, frame-0 red −1)
   - These are 2 numeric constants — trivial port to the apogee_intN inflate side
   - No GPU spend

4. **Op 3** (after Op 1 + Op 2 land)
   - Take apogee_int6's int6-quantized weights, run through Op 1 encoder
   - Verify basin-parity on the resulting smaller bytes
   - Empirical bytes saved

5. **Op 4** (full stack — after Op 1 + 2 + 2.5 + 3 land)
   - Composition CLI: build_full_stack_archive.py
   - Apply ALL operations in deterministic order
   - Empirical: final byte count + predicted score

6. **Cross-paradigm wave** (after Op 4 lands)
   - α: re-test stack on SegMap-with-mask substrate (Quantizr/G-v3) where masks.mkv exists
   - β: GPU-bound; pending billing
   - γ: γ-JCSP coordinator finds joint-optimal codec parameters
   - δεζ: GPU-bound; pending billing

## Deferred-dispatch playbook update

After Op 1-4 land empirically (locally, CPU), the deferred-dispatch playbook should fire:
- Tier 0: top-3 1:1 replays (already prepared, `deferred_dispatch_playbook_top3_replay_20260507.sh`)
- **Tier 0.5 (NEW)**: full-stack archive [contest-CUDA] eval
- Tier 1: apogee_int6/int7 (already prepared, `deferred_dispatch_playbook_apogee_int6_int7_20260507.sh`)

## Adversarial review gate

Per CLAUDE.md non-negotiable: each Op landing requires 3-clean-pass adversarial review. Each round:
- Council members review changed code from different angles
- CRITICAL findings fixed immediately, counter resets
- 3 consecutive clean rounds before lane is marked `three_clean_review`

## Cross-references

- Bit-level intel: `pr_top3_bit_level_deconstruction_20260507_claude.md`
- Wave-Ω blueprint: `wave_omega_stack_composition_blueprint_20260507_claude.md` (overlap with Op 1-4 — Ω-1/Ω-2/Ω-3 are different from Op 1/Op 2/Op 3)
- PARADIGM-δεζ Phase 1: `paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md`
- Auto-resume council: `feedback_grand_council_universal_auto_resume_pattern_20260507.md`
- Adapter set: `experiments/public_runtime_adapters/pr10{1,2,3}_*_adapter/inflate.sh`
