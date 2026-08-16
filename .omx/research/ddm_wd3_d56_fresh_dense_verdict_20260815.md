# ddm_wd3 D56 verdict — fresh dense w56 at matched budget: NEGATIVE-LEANING at n60, n120 confirmation OWED

Date: 2026-08-15/16 boundary · Owner: MAIN (#1070 fire chain) · Axis: [macOS-MPS train / n60 subset
advisory — NEVER a score] · Instance: `D56` (dense_d4_w56, from scorer-free birth, 65 epochs)

## The run

- Governed watched launch 64 (attempt 2 after the `PYTORCH_ENABLE_MPS_FALLBACK=0` gate refusal).
- rc=0 in **6,446 s (~99 s/epoch — ~2× faster than projected)**; done receipt clean; the repaired
  liveness watcher (#1064 fix) reported the completion correctly, no false child_dead.
- `TRAIN_RESULT.json`: `complete: true`, `instance_status:
  TRAINED_PENDING_N120_IF_NEGATIVE_AND_N600_SAME_INSTRUMENT`, `score_claim: false`,
  `promotion_eligible: false`, `all_evaluated_payloads_retained: true`,
  `all_stage_checkpoints_preserved: true`, `n120_negative_confirmation_run: false`,
  `n600_same_instrument_run: false`.

## The measured trajectory (n60 fixed subset, same instrument as W0)

| epoch | hard_d_seg | d_pose |
|---|---|---|
| 1 | 0.028182 | 0.874 |
| 5 | 0.009? → see evals | 0.20 |
| 30 | 0.00317 | (noisy band) |
| 40 | 0.002678 | (noisy band) |
| 55 | 0.002732 | **0.1643 (best)** |
| 65 | **0.002682** | **0.3105** |

Seg: 10.5× descent from ep1, then FLAT from ~ep40 (0.00268 ± 0.00005 — an asymptote, not a slope).
Pose: oscillating 0.16–0.87 across the whole run, no convergence trend.

## Comparison at matched TOTAL budget (65 ep vs W0's 60+5)

| arm | hard_d_seg ep65 | d_pose ep65 |
|---|---|---|
| W0_warm (wd2 lineage optimizer state) | 0.0010857 | 0.02294 |
| W0_reset (same weights, fresh Adam) | 0.0010628 | 0.03408 |
| **D56 (fresh init, fresh Adam)** | **0.002682 (2.5×)** | **0.3105 (~10–14×)** |

## Adjudication

1. **Pose gap: EXPECTED, not evidence.** The measured W0 law (Adam state carries pose descent ~3×
   per window) compounds over a from-scratch run; D56 additionally lacks the wd2 curriculum
   inheritance entirely. Pose non-convergence at 65 ep is what the law predicts for a fresh
   topology. NOT chargeable to the w56 dense family.
2. **Seg gap: the real signal.** The seg curve is at an ASYMPTOTE (flat ep40→65) 2.5× above the W0
   floor at matched budget. This is either (a) capacity of dense w56 vs flattened w64, or
   (b) budget (65 ep insufficient for fresh-init to reach its own floor). The flatness favors (a),
   but ep40→65 flatness after a 10× descent is also consistent with a slow second phase.
3. **Verdict: NEGATIVE-LEANING.** verdict_scope: instance (INSTANCE(D56, n60) — one config, one
   subset; NOT formulation, NOT family). Per the sealed spec law — "a negative cannot be emitted
   from n60" — NO family verdict is emitted. The instance status stands as written by the trainer.
4. **OWED: the n120 seeded-stratified confirmation run** (the trainer did not auto-run it;
   `n120_negative_confirmation_run: false`). Fire-order: at the F64 done boundary (same Metal
   instrument; F64 first per the sealed ARM_ORDER `("W0_warm","W0_reset","D56","F64")`).
5. **Chain continues regardless:** F64 (factorized_d4_w64_r19) requires only D56 COMPLETED — fired
   as launch 65 (pid 98052, live). F64-vs-D56 gives the dense-vs-factorized read at matched
   budget and matched (fresh-init) optimizer handicap — the cleaner within-fresh comparison.
6. **W96 gate unchanged:** needs all four arms completed + capacity pressure; D56's 2.5× seg gap
   IS prima facie capacity pressure — if F64 shows the same asymptote, the W96 case strengthens.

## Side receipts this boundary

- wc1 optimized n600 r2: REFUSED by the (gb1-fixed) safe_run admission gate under REAL kernel
  pressure (level 2, free pages 0.3 GiB, total process RSS only 5.7 GiB — transient file-cache
  pressure from harvest/compile I/O). Governed refusal receipt at
  `.../ddm_wc1_advisory_decode_wallclock_20260815/launchers/base_optimized_n600_r2/run.log`.
  QUEUED; single retry at the F64 done boundary.
- gb1 D1 residue LIVE: the cp-measurement clamp fired at 73.95 and 125.71 GiB (total-used-shaped,
  not named-process-shaped) — clamped/non-blocking, but the D1 wrong-object cure did not reach this
  leg. Routed to #1073 as a named residual.
- gb1 D5 VALIDATED in production: the F64 job appears ONCE in the admission decision under its
  declared 20 GiB (no per-session +25 phantom); dashboard charged at its declared 2.44 GiB.
