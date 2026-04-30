# Council — Lane 8 Multi-Pass Compress GPU Inner-Step Design

Date: 2026-04-30
Lane: 8 — Multi-pass compress (compress-time iteration with score-feedback)
Goal: Lift Lane 8 from Level 1 (MVP postfilter scope inside `trick_stack._stage_multi_pass`) to Level 3 (Full Production Hardened) per the user-mandated standard (`feedback_production_hardened_standard_definition_20260430.md`).
Reference: `project_phase1_dispatch_state_corrections_20260429.md` (Lane 8 current state); `project_codec_stacking_composition_canonical_orders_20260429.md` (canonical stack composition); `feedback_codex_detach_pattern_works_20260429.md` (Pattern A nohup).

## Problem statement

Today `trick_stack._stage_multi_pass` runs the postfilter CNN N times with uint8 rounding between passes. It is bound to the legacy `inflate_postfilter` path. The canonical Lane G v3 / Lane A pipeline uses `inflate_renderer.py`, which never touches `trick_stack` and therefore never gets multi-pass benefits.

The Phase 1 / Phase 1.5 scope demands multi-pass at compress time over the canonical inflate pipeline:

- Compress-time is unlimited compute per CLAUDE.md.
- Inflate at deploy time stays single-pass + scorer-free + within 30-min T4 bound (strict-scorer-rule).
- The output archive bytes are the only deliverable. Multi-pass is a compress-time optimization producing one final archive.

## The core loop

```
pass_idx = 0
prev_score = +inf
history = []
while pass_idx < MAX_PASSES:
    archive_bytes = encode_pipeline(state, params_at_pass[pass_idx])
    score = inflate_and_eval(archive_bytes, scorers)  # CUDA forward pass, the "GPU inner step"
    delta = prev_score - score
    history.append(record)
    if score < target OR |delta| < eps:
        break
    if regression_guard and delta < 0:   # score went UP, REVERT and stop
        revert_to_pass(pass_idx - 1)
        break
    params_at_pass[pass_idx + 1] = adjust_params(params_at_pass[pass_idx], history)
    prev_score = score
    pass_idx += 1
return best_archive_bytes, history
```

## Adversarial council perspectives (rotating)

### Shannon (LEAD — info theory)

When does an extra pass actually buy bits? At convergence the marginal score gain per pass is bounded above by the information gained from the previous pass's score measurement, which is `H(score | history[k]) - H(score | history[k+1])`. Once the entropy of the pose / mask / weight allocation conditional on the history saturates, additional passes are wasted compute.

For a convex score-vs-bytes objective the inner loop is unnecessary (one-pass coordinate descent at the analytic optimum). For the actual non-convex objective (hard-thresholded scorers, discrete codecs, AV1 quantization tables) the iteration buys real bits but with diminishing returns. Theoretical floor on number of passes for a well-conditioned problem: O(log(1/eps)) — for a target `eps = 0.001` against a Lipschitz `L = 1` objective, 10 passes is asymptotically enough. Below 3 passes the regression guard dominates; above 10 the cost of a CUDA forward pass dominates the marginal byte savings.

**Verdict**: MAX_PASSES = 5 is the sweet spot for the score/budget tradeoff. eps = 1e-3.

### Dykstra (CO-LEAD — convex feasibility)

The compress-time problem is "find archive bytes minimizing score subject to: rate ≤ R_max, scorer-validity ≤ V_max, codec-format constraints". Each pass projects onto one constraint set. Multi-pass IS an alternating-projections scheme, naturally Dykstra-style.

But: alternating projections only converges if each projection is non-expansive. AV1 mask CRF adjustment IS non-expansive (CRF is monotone in bits and in distortion — larger CRF → fewer bits, more distortion). Pose Q is non-expansive (coarser Q → fewer bits, more distortion). Block-FP block size is non-monotone (smaller block → fewer bits in some regimes, more in others due to header overhead). Residual gain is non-expansive in residual energy.

**Verdict**: 3 of the 4 axes are safe for alternating projections. Block-FP block size needs a guard against header-overhead ringing (don't oscillate between block-size=8 and block-size=32). Restrict block-size adjustments to monotone-decrease-only after the first pass.

### Hotz (raw engineering)

"Is there a closed-form optimum that avoids the inner loop entirely?" For a fixed renderer + fixed mask sequence + fixed pose budget, the rate-distortion frontier IS computable analytically per-stream IF you know the marginal sensitivity (`dScore/dByte`). We have the canonical waterline derivation in `project_codec_stacking_composition_canonical_orders_20260429.md`: pose dScore/dByte ≈ 0.00067 (saturates ~5KB), mask dScore/dByte ≈ 0.00067 BUT 50× more headroom. So the closed form says: spend the extra bits on masks until pose-mask waterlines equilibrate.

Multi-pass is the empirical way to find that equilibrium when the analytical sensitivity is unreliable (which it is for spatially-varying mask CRF and for renderer block-FP). Hotz: "Build the closed form. Use multi-pass only as a fallback when the closed form is misspecified."

**Verdict**: implement multi-pass but ALSO log the predicted closed-form optimum each pass for divergence diagnostics. If multi-pass and closed-form disagree by >2x in score, flag as "model-misspec" and surface to operator.

### Quantizr (adversarial — competitor reverse-engineer)

"I never multi-passed. What changed?" Quantizr ships at 0.33 with a single-pass FP4+Brotli encoder. He spent ~$15 of compute building it. Multi-pass at compress time is unlimited compute per CLAUDE.md so the playing field changed. But: if Quantizr's closed-form already nailed the optimum, multi-pass on his architecture would buy 0 bits. Multi-pass only helps when the encoder has a non-trivial parameter space that the encoder author didn't already optimize.

Lane A (88K, FP4A) HAS a non-trivial parameter space (mask CRF + pose Q + 88K-renderer block-FP layout + residual gain on K hard frames). Quantizr's space is larger (FP4+Brotli has Brotli quality + dictionary + …) but he tuned it manually.

**Verdict**: multi-pass on Lane A IS a real lever (predicted +5-15 bp gain). Multi-pass on Quantizr-clones may be a no-op until we add new parameter axes (e.g., per-frame CRF, per-block-FP-block bit allocation).

### Selfcomp (block-FP per-block adaptation)

"Block-FP per-block adaptation IS multi-pass within a single encode. What's the second pass worth?" If the encoder's first pass already runs Boyd's water-fill across blocks, the second pass can only buy bits by reconfiguring the AXES across which we water-fill (e.g., switch from per-block to per-channel, or from per-channel to per-row). That's a structural change, not an iterative refinement.

**Verdict**: respect Selfcomp. Multi-pass MUST add VALUE not REPEAT the per-encode-step optimization. The default adjustment policy should re-allocate budget ACROSS streams (mask vs pose vs renderer), not RE-RUN the same per-stream water-fill.

### Contrarian (when does multi-pass REGRESS)

Two failure modes:

1. **Scorer noise overfitting**: if the scorer's score has variance σ_score ≥ marginal per-pass gain, the multi-pass loop chases noise. Symptom: score oscillates around an optimum without converging. Mitigation: regression guard (revert to best-so-far on score-up move). Mitigation 2: average scorer over K seeds — but seeds are deterministic in our pipeline so this doesn't help.

2. **Adjustment policy underflow**: if `adjust_params` walks out of the codec's valid range (e.g., AV1 CRF beyond [10, 51], pose Q beyond [4, 16] bits), the encoder may produce a corrupt archive. Symptom: inflate fails or score = +inf. Mitigation: parameter clamping in `propose_next_params`.

**Verdict**: regression guard IS mandatory. Parameter clamping IS mandatory. eps = 1e-3 also acts as a noise floor.

### Carmack (30-minute version)

"What's the cheapest single pass that captures most of the gain?" Carmack ships in 30 min: pick the SINGLE highest-EV adjustment axis (mask CRF per the waterline analysis), do a 3-CRF probe (CRF=45, 50, 55), pick the best. That's 3 inflate-and-evals = 30 min on 4090. Bytes saved ≈ 10-30 bp. Sub-optimal vs full multi-pass but 80% of the gain at 30% of the cost.

**Verdict**: Carmack's 3-probe is exactly the MAX_PASSES=3 mode. We default to 3, allow operator to bump to 5 when the budget allows. NEVER over 10 passes.

## Council verdict (consensus)

| Parameter | Value | Justification |
|---|---|---|
| MAX_PASSES default | 3 | Carmack 80/30 + Shannon log(1/eps) + Hotz closed-form fallback |
| MAX_PASSES upper bound | 5 | Shannon log saturation + Selfcomp diminishing returns |
| eps | 1e-3 | Below scorer noise floor (per CLAUDE.md ≥0.001 threshold for [empirical] tags) |
| target_score | configurable, default = baseline_score - 0.005 | Worth ≥1 pass per CLAUDE.md score-arithmetic priority |
| regression_guard | True (mandatory) | Contrarian failure mode 1 |
| parameter clamping | True (mandatory) | Contrarian failure mode 2 |
| adjustment axes (default 4) | mask CRF, pose Q bits, block-FP block size, residual gain on K hard frames | Quantizr/Selfcomp/Hotz |
| block-FP adjustment | monotone-decrease-only after pass 1 | Dykstra non-expansiveness |
| device | CUDA-required (CPU opt-in with banner) | CLAUDE.md MPS-fallback trap |
| inflate path | strict-scorer-rule (no scorers loaded at inflate) | CLAUDE.md non-negotiable |

## Adjustment policy (default — coordinate descent)

For each pass `k`, the policy considers the score-arithmetic priority ranking from `project_codec_stacking_composition_canonical_orders_20260429.md`:

1. Mask CRF (45× headroom over pose, per Shannon waterline)
2. Pose Q bits (saturates ~5KB)
3. Block-FP block size for renderer.bin (Selfcomp domain)
4. Residual gain on K most-impactful frames (sensitivity-driven if available)

The default policy steps the highest-score-arithmetic-priority axis until that axis saturates (no improvement for 1 pass), then moves to the next axis. For every adjustment, the policy clamps to the codec's valid range and refuses to move past the regression guard's best-so-far point.

Pluggable via `AdjustmentPolicy` ABC: subclass and override `propose_next_params(state, history)`.

## Strict-scorer-rule compliance

- Multi-pass runs at COMPRESS time only. The inflate side is unchanged — multi-pass produces a single final archive that the existing `inflate_renderer.py` handles natively (no new magic bytes, no inflate-time scorer loads).
- The compress-time eval invokes `experiments/auth_eval_renderer.py` (which is allowed to load scorers) on the candidate archive. This is a compress-time forward pass through the contest scorer, NOT an inflate-time hook.
- Preflight Check 91 forbids any `MultiPassCompressor` instance being constructed inside `inflate.sh` or `inflate_renderer.py`.

## Cost / dispatch plan

- 3-pass on Lane G v3 anchor (88K renderer, ~600 frames): ~10min/pass × 3 = ~30min on Vast.ai 4090 = ~$0.13. UNDER $10 cap.
- Dispatch via Pattern A (`nohup` + `bash -c '...'` + `disown`) per CLAUDE.md.
- Synthetic 2-pass quadratic objective test runs in <1s on CPU.
- Real-archive 3-pass test on a 300-frame subset runs in ~3min on CPU (no scorer involvement; uses byte-only proxy for offline test).

## Predicted band (eyes-open)

`[empirical:reports/lane_8_multipass_real_archive.json]` (TBD post-run): `[+5, +15] bp` improvement over Lane G v3 1.05 baseline. Expected landing: 1.035-1.045 [contest-CUDA].

If the predicted band misses (multi-pass produces no improvement or regression > 5 bp), we flag the encoder's parameter space as already-optimized and demote the lane to "diminishing-returns / closed-form sufficient."

## Cross-references

- `project_phase1_dispatch_state_corrections_20260429.md`
- `project_codec_stacking_composition_canonical_orders_20260429.md`
- `feedback_production_hardened_standard_definition_20260430.md`
- `feedback_codex_detach_pattern_works_20260429.md`
- `src/tac/trick_stack.py:377` — current MVP scope
- `submissions/robust_current/inflate_renderer.py` — the canonical pipeline being wrapped
- `experiments/pipeline.py` — canonical compress entry-point
- `src/tac/multipass_compressor.py` — to be created
- `experiments/pipeline.py compress --multipass` — to be wired
