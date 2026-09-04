# ddm_pr1 PRE-REGISTRATION — written before the aggregate, with the arithmetic stated

Arm: `ddm_pr1_pose_resolve_on_renderer_change`. Tokens: `[no-triality] [p0-ledger-ok]`.
Written 2026-09-04 while the n600 solves are IN FLIGHT and no aggregate exists.

## What was pre-registered where, and what I had already seen when I wrote this

* **The charter's prediction** (`.omx/research/charters/ddm_pr1_pose_resolve_on_renderer_change_20260904.md`,
  committed `77ad212ad`, before any measurement): *"the re-solve recovers most of the renderer-induced
  pose damage (post-re-solve coupling < 20, i.e. > 10x recovery). Falsifier: recovery < 3x (coupling > 70)
  — then the closure stands with the re-solve measured, not assumed."*
* **What I had seen when I wrote THIS file:** ~13 of 600 per-pair solver rows and a 7-pair timing smoke.
  No n600 mean, no base control, no coupling. The per-pair rows visible were already violently
  heavy-tailed (recoveries 1.2x, 4.4x, 9.8x, 19x, 2.8e3x, 3.7e3x, 3.0e4x, 1.5e6x, 7.6e6x). Stating that
  plainly is the point: this file is not a clean pre-registration of the per-pair distribution, and I
  will not present it as one. It IS a clean pre-registration of every aggregate below.

## The arithmetic, fixed now so the result cannot be narrated either way

AFR1 receipt `[contest-CUDA T4 n600]`: `d_seg 0.00020139`, `d_pose 6.37e-06`, `180,002 B`,
`S 0.14797617125559104`, pose leg `0.00798123`. This arm buys `dB = 0`, so the whole move is paid on
distortion and the promotion condition is

    sqrt(10 * d_pose_new) < 0.00798123 + 100 * |delta d_seg|      (delta d_seg NEGATIVE = seg improved)

At a 25% seg cut (`delta d_seg = -5.0348e-05`) the payable ceiling is `d_pose <= 1.694e-05`. Writing the
post-re-solve coupling as `k_post = (d_pose_after - d_pose_base) / |delta d_seg|` and assuming the law's
own stated-not-measured direction symmetry, a 25% cut is payable **iff**

    k_post <= (1.694e-05 - 6.37e-06) / 5.0348e-05 = 0.2098

**That bar is 95x stricter than the charter's own success band of `k_post < 20`.** The two events are
NOT the same event, and I pre-commit to reporting them separately: "the operator's intuition was right"
and "the renderer door reopens" can both be true, both be false, or split.

## Numeric predictions (all aggregates unseen)

1. `d_pose_after` mean over n600 lands in **[1e-05, 1e-04]**, dominated by a small number of
   badly-recovering pairs with large absolute residuals, not by the median pair.
2. Mean-based recovery `rho = mean_before / mean_after` lands in **[30x, 1000x]** — i.e. FAR above
   jg5's transferred 8.0x, which was measured on token edits.
3. Median per-pair recovery exceeds **100x**, and the mean-based recovery is at least **10x smaller**
   than the median-based one (the mean is a mean of per-pair MSEs; the tail owns it).
4. `k_post` lands in **[0.1, 20]**.
5. **The charter's falsifier (recovery < 3x, coupling > 70) does NOT fire.**
6. The payability bar `k_post <= 0.2098` is a coin-flip: I give it under even odds, because a single
   unrecovered pair at `d_pose ~ 2e-3` contributes `3.3e-06` to the n600 mean on its own — half the
   entire AFR1 base.
7. The base-renderer re-solve CONTROL also improves on the shipped carrier (the shipping chain left
   slack), so the like-for-like both-re-solved comparison is the honest one and I will report it too.

## What would make me wrong in a way that matters

* If `d_pose_after` sits BELOW the AFR1 base `6.37e-06`, then `k_post` is negative and the coupling
  language stops meaning anything for this candidate — I would report the re-solve as a zero-byte
  pose lever in its own right and say the coupling is unmeasurable in that direction.
* If the base control recovers as much as the candidate does, the "recovery" is the shipping chain's
  own unclaimed slack rather than a repair of renderer damage, and the coupling must be re-derived
  against the re-solved base, not the shipped one.

## Solver, stated before the result (m116: never let the derivation control its own falsifier)

The solver is `ddm_jg5.refine_pair` — br1's damped Gauss-Newton on the shipped 12-dim basis and int12
lattice with jg5's DERIVED materiality stop — NOT `ddm_up2.solve_pair_realized`, which jg5 Sec 4 records
as a truncated `+-2` search radius that br1 was built to escape. Running the weaker solver would have
measured the SOLVER's ceiling and reported it as the CARRIER's, which is exactly the shape of a
falsifier controlled by the derivation that tests it. `--solver up2` exists as a labelled control.

The materiality floor is evaluated at the AFR1 operating point `6.37e-06` (`dd_threshold 5.5869e-09`),
not at the inflated pre-solve mean; evaluating it at the start mean would raise the floor by
`sqrt(start/target)` and stop the solver ~37x too early.
