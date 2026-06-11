# Capstone EMA-shadow-lag bug REVERSES the "seg-capacity wall" verdict (2026-06-11)

**Authority:** `[macOS advisory]` (MLX + local-CPU torch scorer; `score_claim=false`,
`promotion_eligible=false`). The exact pointer (`0.19109982 [contest-CPU]`) is UNMOVED — this is a
correctness fix + a verdict reversal on the LOCAL d_seg observable, not an exact-score row. Commit
`f771e6e00`.

## TL;DR — the central negative finding gating the capstone was a MEASUREMENT artifact
The "**d_seg frozen at 0.505 / seg-capacity wall**" — the result that made us doubt the small learned
basis all session — is an **EMA-shadow-LAG bug**, not a capacity limit. The capstone's `exact_d_seg`
renders the EMA *shadow* (`use_ema_for_eval=True`), and the shadow used a constant decay (0.999 in the
curriculum `StageSpec`, 0.997 elsewhere) with **no warmup**. Constant decay 0.999 ⇒ time constant
1/(1−0.999) = **1000 steps**. Our MLX runs are short (48 pairs ÷ bs 8 = **6 steps/epoch**; a curriculum
stage is ~240 steps), so the shadow stayed **~init weights** the entire run → `exact_d_seg` read a
**bit-frozen near-init value (0.505)** even though the **live weights solved seg**.

## The decisive empirical proof (`experiments/diag_curriculum_ema_lag.py`, 8 pairs, base_ch=20, stage-1 CE)
| epoch | seg_loss | d_seg(LIVE, use_ema=False) | d_seg(EMA shadow, use_ema=True) | GAP |
|---|---|---|---|---|
| 5  | 1.418 | 0.503 | 0.507 | +0.004 |
| 10 | 0.918 | 0.188 | 0.507 | +0.319 |
| 15 | 0.573 | 0.104 | 0.507 | +0.403 |
| 20 | 0.348 | 0.051 | 0.507 | +0.456 |
| 25 | 0.250 | **0.041** | **0.507** | **+0.466** |

The LIVE weights descend d_seg 0.507 → **0.041** in 25 epochs (still falling); the EMA shadow is **frozen
bit-identical at the init 0.507**. The GAP grows monotonically as the live weights pull away from the
stuck shadow. The seg LOSS descends 150× (1.42 → 0.25) — the model IS learning; only the shadow lagged.

This is exactly the curriculum daemon's signature: `exact_d_seg = 0.5053805311520895` bit-identical across
all 90 logged epochs while `seg_loss_mean` dropped 150× and `mean_d_pose` descended (the slow pose path
DID track the shadow, because its convergence timescale ~ the EMA time constant; the fast seg path did not).

## Why this poisons BOTH the verdict AND the export (a real correctness bug, not just telemetry)
The A1 fix (correctly, per the EMA non-negotiable) made the **export** byte the same EMA shadow
(`export_render_weights`/`export_stored_latents` snapshot the shadow). On a short run the shadow ≈ init →
a byte-closed archive built now would ship **near-init seg → real contest d_seg ≈ 0.5 (catastrophic)**. So
the lag silently corrupts the archive, not only the advisory. Any capstone archive exported pre-`f771e6e00`
on a short run is suspect.

## The fix — EMA warmup decay (`_CapstoneWeightEMA.effective_decay`)
`decay_t = min(self.decay, (1 + t) / (10 + t))` where `t` = #updates (timm `ModelEmaV2` / diffusion EMA).
Shadow ≈ live early (nothing to average yet), ramping to the target decay as updates accumulate. Correct
for a **weight-init** EMA (the Adam-style `/(1−decay^t)` bias correction assumes a ZERO init and is wrong
here). `StageSpec.ema_decay` cap also aligned 0.999 → **0.997** (the CLAUDE.md "Quantizr decay = 0.997"
mandate). Counter `_num_updates` persists across curriculum stages (the shadow keeps tracking; PR95 resumes
the weights between stages).

**After the fix** (same diagnostic): GAP ≤ **0.023** throughout, and at ep25 the shadow d_seg (0.085) is
slightly BELOW live (0.093) — the shadow now both *tracks* the live weights AND delivers the intended EMA
variance reduction (a lower-d_seg averaged point, which is exactly why we export the shadow).

## Self-protection (CLAUDE.md "bugs must be permanently fixed AND self-protected")
Two NO-FAKE guard tests in `test_carrier_independent_fixes.py`:
- `test_b4fix_ema_effective_decay_warms_up_from_low_to_cap` — asserts `effective_decay()` returns ~0.18 at
  update 1 (a constant-decay revert returns 0.997 → FAILS), monotone ramp, saturates at the cap.
- `test_b4fix_ema_shadow_tracks_live_weights_on_short_run` — asserts the shadow weights are much CLOSER to
  live than to init after a short run (the constant-0.999 bug inverts this). Scorer-independent (weight-space).
70/70 capstone tests pass (the one "timeout" is the slow real-scorer Muon loop at the 60s cap; passes in 140s).

## What this REVERSES (re-categorize prior verdicts)
- **"d_seg frozen 0.505 / seg-capacity wall" (curriculum) → FALSE.** Pure EMA-lag artifact. The small
  85K-param basis DESCENDS d_seg (≥ to 0.041 in 25 ep / 8 pairs, live, still falling).
- **"CE plateaus ~0.008 / seg-walled" (the LONG run) → SUSPECT, re-measure.** Those d_seg numbers were the
  EMA shadow too; the true LIVE d_seg was likely lower. The "best capstone S ≈ 1.75" used a shadow d_seg.
- **"the smaller-than-frontier basis fights the physics" → WEAKENED.** That argument rested on the d_seg
  wall being real. With the wall shown to be a measurement artifact, the small-basis thesis is reopened —
  exactly the operator's instinct ("the smaller learned is not dead").

This is the **third measurement-artifact correction** this session (after the premature "seg-walled" and
the retracted "pose crux confirmed"). The pattern the operator named — *"that optimizer and training issue
has poisoned all substrate work"* — has a concrete, now-fixed instance: the capstone's EMA-shadow eval was
systematically reading stale weights on every short run.

## Next (the honest verdict, in flight)
`experiments/results/capstone_c1prime_honest_b20_n48/` — the C1′ carrier (`stored_latent`, pose-capable) +
PR95 8-stage curriculum + the FIXED EMA, 48 pairs, marker-on-exit. This is the decisive honest joint
d_seg+d_pose trajectory the lag was hiding. If d_seg descends well + d_pose holds → scale to 600 pairs →
byte-close → paired contest CPU+CUDA exact eval (the pointer-mover).
