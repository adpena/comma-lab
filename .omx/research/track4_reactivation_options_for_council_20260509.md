# Track 4 Reactivation Options for Grand Council Review (2026-05-09)

<!-- generated_at: 2026-05-09T04:30:00Z, from_state_hash: af2e9d97_track4_landing -->

## Status

**Track 4 (UNIWARD + STC + Hessian-aware bit allocation on A1 archive)** = DEFERRED-pending-research per CLAUDE.md kill-as-last-resort. Best candidate `blocks4_7bit` 177,903 B scored **0.198694 `[contest-CPU GHA Linux x86_64]`** vs A1 baseline 0.192848 = **+0.005846 WORSE**. Predicted band [0.173, 0.188] FALSIFIED.

Cost: $0 (GHA free minutes only). NO GPU spend.

## Critical bug-class finding (the score-aware substrate inversion)

**The naive `mean(θ²)` Fisher proxy is ANTI-correlated with score-saliency on A1's score-gradient-trained substrate.**

Mechanism: A1 was trained with `experiments/train_score_gradient_pr101_finetune.py`, which propagates `d(seg)/d(theta) + d(pose)/d(theta)` through every parameter. After training:
- Tensors with HIGH `mean(θ²)` were pushed AWAY from zero by the gradient → score-relevant in the **gradient-aligned direction**
- Tensors with LOW `mean(θ²)` were pushed TOWARDS zero → score-relevant in the **nullspace orthogonal direction** (i.e., the gradient pushed them to zero precisely BECAUSE moving them perturbs the score)

Coarsening "low-Fisher" tensors (the v1 builder's pick) hits the **nullspace directions** — exactly the parameters the score-gradient training identified as score-relevant in the orthogonal sense.

**Cliff sharpening**: prior 3-anchor finding (`feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_20260508.md`) had a cliff at rms ≥ 0.04 on score-NAIVE substrates. On A1's score-AWARE substrate, the cliff is at **rms ≈ 3.5e-4** — orders of magnitude sharper. This is a HIGH-VALUE empirical anchor for the Phase 3 IB-Lagrangian council math.

## Reactivation criteria (4 options, ranked by Council EIG/$)

### Option 1 — Replace `mean(θ²)` with score-gradient saliency [HIGHEST EIG/$]

Use autograd hooks on `upstream/evaluate.py`'s SegNet+PoseNet to compute `|∂score/∂θ|` per parameter. This is the TRUE Fisher importance for our score axis. Use this as the per-tensor coarsening priority instead of `mean(θ²)`.

- **Cost**: $0 (CPU autograd; one forward+backward pass through SegNet+PoseNet on 600 image pairs ≈ 25 min on M5 Max)
- **Predicted savings**: -0.005 to -0.015 score (matches Track 4's original band IF the score-gradient saliency is the actual right axis)
- **EIG/$**: ∞ (no GPU, no retrain)
- **Risk**: requires gradient through DALI/PyAV decode path — may need fallback to PyAV-only
- **Implementation**: ~80 LOC modification to `tools/build_uniward_stc_hessian_a1_v1.py` adding a `--saliency-source score_gradient` flag

### Option 2 — STC on latent_blob (not decoder weights) [DIFFERENT TARGET]

The 600×28 latent stream is a much larger spatial cover than the decoder weights, and it's the natural domain for STC (Filler's syndrome trellis was designed for image cover signals, not weight tensors).

- **Cost**: $0-5 (CPU STC; if we want the syndrome length tuned via grid search, $5 GHA)
- **Predicted savings**: -0.003 to -0.012 score (rate axis only — STC reduces side-info bytes; latent compression unchanged)
- **EIG/$**: ∞ (no GPU)
- **Risk**: latent_blob is already brotli-coded near Shannon floor (per `feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md`); STC may not buy much
- **Implementation**: ~150 LOC; needs new STC encoder/decoder modules

### Option 3 — Sanity threshold gate (>1 KB savings at <1e-3 distortion required) [PROCEDURAL]

Add a hard gate to `build_uniward_stc_hessian_a1_v1.py`: refuse to emit any candidate where bytes-saved/distortion > some learned threshold. This prevents the v1's catastrophic +0.0058 from happening without operator review.

- **Cost**: $0 (50 LOC gate addition)
- **EIG/$**: ∞ (purely procedural; saves future false-positive work)
- **Risk**: too-strict threshold blocks all candidates; need empirical calibration

### Option 4 — Sanity-check on non-score-gradient substrate [VERIFY THE BUG-CLASS]

Run the Track 4 v1 builder on PR106 (which is NOT score-gradient-trained). If Track 4 v1 IMPROVES PR106's score, that confirms the bug-class diagnosis: the failure mode is specific to score-aware substrates.

- **Cost**: $0 (CPU build) + $5 (GHA CPU eval on PR106)
- **EIG/$**: 0.001 score per $1 (low absolute predicted gain on PR106; high diagnostic value)
- **Risk**: PR106 may have its own substrate-specific failure modes

## Council consideration prompts

For inner-ten members, here are the questions Track 4's failure raises:

- **Shannon**: is `mean(θ²)` ever a valid Fisher proxy? Or only on i.i.d.-substrate? What's the right closed-form proxy for a score-gradient-trained substrate?
- **Yousfi/Fridrich**: STC was designed for IMAGE cover signals, not WEIGHT tensors. Is Option 2 (latent_blob STC) the right Filler-canonical use?
- **Quantizr**: Quantizr's 88K params at 0.33 used FiLM conditioning — is the score-gradient training's nullspace structure related to FiLM's per-frame adaptation? Could Option 1's score-gradient saliency naturally bucket into FiLM-conditional vs FiLM-unconditional tensors?
- **MacKay**: MDL says the BEST description preserves `I(θ; Y)` while minimizing `I(θ; X)`. Track 4 v1 minimized `bytes(θ)` while INCREASING `I(θ; X) - I(θ; Y)` — exactly backwards. Option 1 fixes this directly.
- **Contrarian**: maybe Track 4 is FUNDAMENTALLY wrong on a score-gradient-trained substrate, and we should not reactivate it — instead pivot 100% to Track 1 (Ballé hyperprior end-to-end) which co-trains decoder + entropy.
- **Hotz**: 80 LOC for Option 1. Stop debating. Run it.
- **Ballé**: hyperprior network ALSO requires per-parameter saliency. Option 1's autograd hook generalizes to T6 (Ballé+UNIWARD cross-paradigm) — building it ONCE benefits both tracks.

## Recommended next-step for council consensus

**Run Option 1 first ($0, ~80 LOC, 30 min wall-clock).** It directly tests the bug-class hypothesis AND if it works, recovers most of Track 4's predicted gain WITHOUT GPU spend. The Phase 3 IB-Lagrangian Phase 3 council math depends on knowing whether score-gradient saliency is the right axis — Option 1 is the canonical empirical answer.

If Option 1 lands a [contest-CPU] score < 0.190, Track 4 is reactivated. If it lands ≥ 0.193, the bug-class is deeper than the Fisher-proxy mismatch and Options 2-4 should be re-considered as a research package, not as quick wins.

## References

- Track 4 landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_track_4_uniward_stc_hessian_a1_landed_20260509.md`
- Builder: `tools/build_uniward_stc_hessian_a1_v1.py`
- Promotion card: `experiments/results/track4_uniward_stc_hessian_a1_blocks4_7bit_20260509_codex/promotion_card.json`
- GHA workflow: `https://github.com/adpena/comma_video_compression_challenge/actions/runs/25597258234`
- Phase 2/3 council memo: `feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md`
- Cliff-class anchor: `feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_20260508.md`
