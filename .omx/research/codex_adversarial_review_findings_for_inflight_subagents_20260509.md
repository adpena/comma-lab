# Codex adversarial review findings for in-flight subagents (2026-05-09)

<!-- generated_at: 2026-05-09T05:00:00Z, from_state_hash: codex_review_b6uice9t6 -->

## Codex thread b6uice9t6, verdict: needs-attention

Target: current working tree (Track 4 v1 builder + Sinkhorn surrogate + score-gradient saliency + joint-source RD + IB Lagrangian aux scorer)

Three HIGH findings that overlap directly with the work several in-flight subagents are doing. Subagents reading this file should incorporate these fixes BEFORE landing.

### HIGH 1 — Score-gradient saliency is not per-sample Fisher
**File**: `src/tac/score_gradient_param_saliency.py:345-371`

Bug: code calls `loss.backward()` once per batch, squares the BATCH-AVERAGED gradient, divides by sample count. That is not `E[(dL_i/dtheta)^2]`. Per-sample gradients can cancel before squaring; saliency depends on `saliency_batch_size` instead of being batch-size-invariant. **This recreates the exact Track 4 v1 Fisher-proxy inversion** that produced the +0.0058 regression — the proxy is anti-correlated with true score saliency on score-aware substrates.

Fix (canonical): use `torch.func/vmap` for true per-sample gradient squares OR force microbatch size 1 for saliency accumulation. Normalize independent of batch size. Add batch-size-invariance regression test that asserts saliency is unchanged across `saliency_batch_size ∈ {1, 4, 16, 64}`.

**Owners**: a40c7b0d (Bug-class fix + self-protect — primary), a1a9359d (HNeRV lessons — must reference this as canonical example of the bug-class), a8c01d31 (coherence council — must include in lesson catalog)

### HIGH 2 — Cliff-zone gate misses the calibrated bad case
**File**: `tools/build_uniward_stc_hessian_a1_v1.py:150-180`

Bug: default threshold 1000; implemented ratio `(bytes_saved/1024) / rms^2`. For `blocks4_7bit` (359 B, rms 1.84e-3), ratio ≈ 103,000 — the gate returns not-in-cliff even though THIS IS THE KNOWN REGRESSION. The safety gate fails open on the +0.0058 score-loss class.

Fix: recalibrate units/threshold so the documented `blocks4_7bit` is blocked by default. Make the regression test assert the default fires (not an artificially-large override threshold).

**Owners**: a40c7b0d (Bug-class fix — primary)

### HIGH 3 — Allowed low-blur Sinkhorn can collapse real soft mismatches to zero
**File**: `src/tac/losses.py:369-397`

Bug: log-domain loop reconstructs transport plan via `torch.exp(log_plan)` before summing cost. At allowed `blur=0.01`, off-diagonal mass underflows for soft distributions. Test case: `p=[0.7,0.1,0.1,0.05,0.05]` vs `q=[0.1,0.7,0.1,0.05,0.05]` returns ~2.4e-11 even though true `1-I` transport cost ≈ 0.6. Training run with low blur sees NO SegNet loss for a large class-mass swap.

Fix: raise minimum supported blur to numerically validated range OR stable log-domain cost accumulation/upcasting. Add soft-distribution regression test at minimum allowed blur (not just one-hot cases).

**Owners**: a0be36e (T8 W₂ Sinkhorn implementation — primary), a3614731 (Phase 2 launch — must not import this in current state)

## Convergence with substrate-vs-codec meta-pattern

All three findings share the same META class: **"math that looks correct but a units / accumulation / numerical-regime mismatch with the calibration anchor produces silent score regression."** This is the same class as:

- Track 4 v1 Fisher-proxy inversion (`mean(θ²)` vs score-gradient saliency)
- Lossy_coarsening per-tensor K=0.05 contest-CUDA falsification (predicted [0.18-0.22] from byte anchor; actual 0.3517)
- AC-vs-brotli phantom result (improvised K-coarsening over-quantized to range [-2,2])
- PR97 anti-pattern (seg-for-pose trade looked locally optimal; lost 0.042 score)

The HNeRV-lessons subagent (a1a9359d) should treat this codex review as part of its evidence corpus.

## Action items

1. **a40c7b0d**: when fixing Track 4 v1 bug-class, address ALL THREE codex findings (saliency, cliff-gate, sinkhorn) — they are the same META class, fix together.
2. **a1a9359d**: cite codex finding 1 as canonical example of "calibration-anchor mismatch" lesson in the HNeRV retrospective.
3. **a0be36e**: do NOT land T8 Sinkhorn surrogate without the low-blur fix.
4. **a3614731**: do NOT use Sinkhorn surrogate from `losses.py:369-397` in Phase 2 launch until HIGH 3 is fixed.
5. **a8c01d31**: include "calibration-anchor-mismatch META class" as one of the unified lesson axes in the coherence portfolio audit.

## Codex follow-up status (2026-05-09)

- HIGH 1 fixed in `src/tac/score_gradient_param_saliency.py`: saliency now
  accumulates one backward pass per sample so it computes
  `E[(dL_i/dtheta)^2]`; regression test asserts invariance across
  `saliency_batch_size`.
- HIGH 2 fixed in `tools/build_uniward_stc_hessian_a1_v1.py`: cliff-zone
  calibration uses KB/rms units and blocks the known `blocks4_7bit` bad case
  by default.
- HIGH 3 fixed in `src/tac/losses.py`: minimum Sinkhorn blur is calibrated to
  `0.01` and low-blur calls receive a safe effective iteration floor; regression
  test asserts a soft class-mass swap remains visible at the minimum blur.

## References

- Codex thread ID: 019e0c15-c879-72b1-8ae8-c759c5a9e3ae
- Diff under review: working tree at codex review time
- META gate #113 (provenance-vs-state): `src/tac/artifact_lifecycle.py`
- Substrate-vs-codec meta: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`
- Track 4 reactivation options: `.omx/research/track4_reactivation_options_for_council_20260509.md`
