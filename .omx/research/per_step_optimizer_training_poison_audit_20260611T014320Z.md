# Per-step optimizer/training poison audit (subagent B, 2026-06-11)

**Source:** adversarial optimizer/training-correctness audit (subagent acc80882421373fb1) vs PR95's proven
torch `stages/common.py` loop, requested by operator ("that optimizer and training issue has poisoned all
substrate work"). **Verdict: STRONGLY SUPPORTED — the poison is LAYERED.** The gross poison (scorer-weights=0
+ skip-free + AdamW-100%-clip, #68/#75/#77) was fixed by #76 (d_seg now descends 0.50→0.01, 43×). But **3
per-step divergences remain in the SHARED MLX machinery** (`apply_pr95_mlx_optimizer_step` + both trainers +
the partition fn), capping d_seg at ~0.008–0.012 vs PR95's 5.6e-4 (a 15–20× gap = the whole difference
between sub-0.15 and a wall). These are DISTINCT from the missing curriculum (subagent A's scope).

## Findings (ranked by d_seg-poison impact) — review-only; fixes to integrate ON TOP of A's curriculum
**[CRITICAL #1] No LR schedule (constant LR, no cosine decay) — the dominant d_seg floor.**
PR95 `common.py:152-157,218-220` builds a cosine `LambdaLR` (`0.5*(1+cos(π·ep/eps))`, floored at
eta_min_ratio) and steps it per-epoch for BOTH adamw + muon. The MLX/capstone loop has NO scheduler;
`apply_pr95_mlx_optimizer_step` takes no epoch arg. The campaign ran `muon_lr=3e-2` (150× PR95's 2e-4),
CONSTANT. d_seg is an argmax-flip RATE — late training needs ever-finer boundary corrections; a constant
large LR overshoots them → d_seg dithers at ~0.008–0.012 instead of annealing into the argmax-correct basin.
**Fix:** port the cosine multiplier (scale both LRs by `max(0.5*(1+cos(π·ep/total)), eta_min_ratio)`,
per-epoch; ~15 LOC). **Test:** 200ep 48-pair, constant vs cosine, seeded — cosine should push the asymptote
materially below ~0.0106; if both plateau, the cap is capacity (honest re-classification).

**[CRITICAL #2] Capstone trains NO weight-EMA + exports LIVE weights — violates the EMA non-negotiable.**
CLAUDE.md "EMA": inference/archive bytes MUST come from `ema.state_dict()`, never live. PR95 EMA-updates
decoder+latents every step (decay 0.999), evals + ships the shadow. `CapstoneTrainer` builds NO weight-EMA
(only codebook EMA); `exact_d_seg`/`mean_d_pose` + `_export_int8_archive` all use LIVE weights. The EMA
shadow is the averaged, lower-variance, lower-d_seg point — exporting live = archiving the noisy floor.
Compounds #1 (constant-LR dither is exactly what EMA averaging suppresses). **Fix:** instantiate `_MlxEMA`,
update per step, eval+export the SHADOW (snapshot+restore), default `ema_decay=0.997`, `use_ema_for_eval=True`.
**Test:** end-of-run live-vs-shadow d_seg (shadow should be lower); regression test that exported bytes == shadow.

**[HIGH #3] FiLM (pose) weights routed to Muon — destabilizes d_pose + steals seg gradient budget.**
`partition_pr95_mlx_parameter_names` routes `pose_film0/1.fc{1,2}.weight` to Muon (ndim≥2, ends "weight").
Muon's Newton-Schulz drives update singular values to ≈1 (step magnitude grad-norm-INDEPENDENT) — far too
large for a zero-init 6→32→2·feat pose MLP. **This is the most likely driver of the d_pose 0.06↔0.34
oscillation**, and the thrashing FiLM injects pose noise into the SHARED feature the seg heads read →
couples into the d_seg floor. **Fix:** route FiLM (small MLP) to AdamW, not Muon (PR95's rule: Muon = hidden
conv weights only; AdamW = stem/heads/embeddings). **Test:** AdamW-FiLM vs Muon-FiLM — AdamW should give
monotone d_pose→tube (not oscillation) + slightly lower d_seg floor; add `test_film_weights_routed_to_adamw`.

**[HIGH #4] d_pose measured WITHOUT eval_roundtrip (clamp-only) — measurement poison, understates contest d_pose.**
`CapstoneTrainer.mean_d_pose` → module `_exact_d_pose` (capstone_trainer.py:316-336) does clamp-only, NO
roundtrip; the loss + `bridge.exact_d_pose` DO roundtrip. So reported d_pose is on a non-uint8 frame → the
"crux confirmed, d_pose 0.06–0.14" reads understate the contest d_pose (uint8 luma quant is where pose drifts).
A "hold the pose" verdict on this can send a candidate to a paid eval that misses the tube. (The d_seg eval
path DOES roundtrip via the bridge — so d_seg telemetry is honest; only d_pose is inconsistent.) **Fix:** delete
the bespoke `_exact_d_pose`; call `bridge.exact_d_pose` (one line, −20 LOC). **Test:** roundtrip d_pose ≥ clamp-only.

**[MEDIUM #5] EMA default 0.999 not the CLAUDE.md-mandated 0.997** (both configs). 0.999 lags ~1000 steps (the
team disabled EMA-eval to dodge the lag instead of fixing decay). **Fix:** default 0.997, then EMA-eval is safe.

**[MEDIUM #6] grad_clip double-bind: campaign=50 (effectively no clip), default=1.0 (the 100%-firing freeze).**
1.0 with Muon-throughout re-introduces a milder inert-step freeze (Muon post-NS norm ≈O(1) → 1.0 clip fires);
50 never fires (untuned guess). Clip + LR schedule are coupled (PR95 tuned 1.0 FOR its decaying LR). **Fix:**
land #1 first, then pick the smallest clip with `would_clip_fraction≈0` after warmup (the trainers emit it).

## What's FINE (not manufactured): Muon/AdamW partition faithful · latent_lr_mult=10 correct ·
Muon-throughout (#77) defensible + Newton-Schulz coeffs bit-match PR95 optim.py · eval_roundtrip faithful in
the LOSS path · ce_seg_loss bit-exact · AdamW betas/eps match · d_seg eval path DOES roundtrip.

## FACTUAL CORRECTION (B caught this): `why_substrate_work_was_broken_derivatives_and_the_redirect_20260610.md`
line ~63 wrongly states PR95 "runs Muon" everywhere. **PR95 is AdamW stages 1–7 + Muon stage 8 ONLY** (L15).
The CODE (Muon-throughout, our deliberate #77 deviation) is faithful to Muon mechanics; only that prose line
is wrong. (APPEND-ONLY HISTORICAL_PROVENANCE — note the correction here, don't mutate the historical memo.)

## FLEET LENS + integration plan
All 6 are in SHARED per-step machinery → poisoned EVERY score-aware run. **Any "substrate walled at d_seg
~0.008–0.012" verdict (incl. the capstone "possible seg-capacity wall" routing + #71 "frontier near-minimal")
is an OPTIMIZER+SCHEDULE artifact, NOT a capacity wall — RE-OPEN after #1+#2 land.** Highest-EV order:
**#1 (cosine LR) → #2 (EMA shadow @0.997) → #3 (FiLM→AdamW)**, then #4/#5/#6.

**INTEGRATION (this orchestrator owns):** subagent A is building the curriculum (stage schedule + C1a/σ/QAT)
and editing the trainer NOW. To avoid merge conflict, B's per-step fixes (#1–#6) integrate ON TOP of A's
landed curriculum — not in parallel. The complete fix = A's curriculum (the staged-loss machinery) + B's
per-step fixes (cosine LR + EMA shadow + FiLM→AdamW + d_pose roundtrip). Then re-run the 48-pair long train
(≥200ep) → the FIRST trustworthy d_seg capacity verdict for the smaller basis. The faithful-PR95-core +
capstone-adapted-synergy principle (operator) applies: #1/#2/#5/#6 are faithful PR95 ports; #3/#4 are
capstone-specific adaptations (FiLM + the pose path PR95 lacks).
