# Detailed per-epoch wall-clock accounting (operator 2026-07-13 "each epoch takes minutes")

**Assembled from MEASURED data across #306 (per-lever compute audit) + #455/#456 (95%-kill forward/backward)
+ run-log forensics. Research-only; wall-clock accounting, no score claim.** Headline: the corpus holds a
**forward-vs-backward-share CONTRADICTION** that this accounting surfaces and that a clean per-component
in-loop profile must resolve. Vehicle caveat: the #306 numbers are v4/mod-19 (2026-07-05); the CURRENT
vehicle is V9·CGauge — absolute s/ep may have shifted, the STRUCTURE holds.

## A. The wall-clock envelope (MEASURED n600, #306 — the in-loop AUTHORITY)

| stage | s/ep MEASURED | note |
|---|---:|---|
| CE stage (ep0-300) | **169.7 median** (158-186, 4 clean segs) | mod-19; the base cost |
| tau stage (ep300+) | **~217-227 (+26%)** | the ONLY large lever-GROUP cost found: tau-softplus form + persistence/clDice + lane-render-band activating together at ep300 |
| whole-run duty | 201.7 incl stalls | verdict wall 2189 s mean = **43-47% duty, ASYNC-HIDDEN** (CPU-torch verdict overlaps the MLX step in a background thread) |

Pure-step (n24 6.068 s/ep ×25) = 151.7 s/ep; the +12% to 169.7 = verdict CPU contention + reorient
(~0.6 s/ep amortized) + checkpoints + telemetry (negligible).

## B. Lever/stage attribution (MEASURED, #306) — the growth is NOT the accreted levers

The Assumption-Adversary verdict: "s/ep growth is caused by the levers" is **MEASURED FALSE**. n24 toggles
show every individual lever is ~free or NEGATIVE at CE stage. The minutes come from THREE places:
(a) the in-trainer common core vs the bench closure; (b) the ep300 stage-activation group (+26% n600);
(c) verdict-window CPU contention (47% duty). Per-lever marginal cost (n24, MEASURED):

| lever | marginal s/ep | verdict |
|---|---:|---|
| pose-carrier (store-nothing) | **−0.71 (NEGATIVE)** | SPEED-SAVER: warp(own-render) replaces a 2nd full INR render. NEVER "drop pose to save time." |
| wa-island loss | free-or-negative | removing it re-routes islands onto the composed render, costs MORE |
| seed co-grad (dual value_and_grad) | ≈0 | free at step level |
| self-orient (−48% d_seg basis) | ~0 step + ~0.6 amortized | cadence already optimal, do not touch |

## C. Component-WITHIN-STEP split — the GAP + the CONTRADICTION (the crux of this accounting)

The whole in-loop step (n24) = 6.068 s/ep = **253 ms/pair** = render-through-R (MLX-GPU) + teacher-SegNet
forward (CPU-torch 1-thread) + teacher backward/costate-VJP + witness backward (MLX) + loss/argmax. **The
per-component split WITHIN this step is NOT cleanly isolated on the in-loop harness** — #306 times the whole
step + per-lever toggles, never render-vs-teacher-fwd-vs-teacher-bwd-vs-witness-bwd on one harness.

What the DIAGNOSTIC harness measured (#455, heavier than in-loop — see the reconciliation gap):

| component | MEASURED (diagnostic) | share of teacher fwd+bwd |
|---|---:|---:|
| teacher SegNet forward only | **537 ms/pair** (n=9) | 18% |
| teacher costate forward+backward | **3009 ms/pair** (n=6) | 100% |
| ⇒ teacher backward/costate-VJP (derived) | **2472 ms/pair** | **82%** |

### ⚠ CONTRADICTION #1 — forward vs backward share (the finding that redirects the 95%-kill)
- The 95%-kill campaign framed the target as "**replace the forward (78%)**" (#455 task title) and the ANE
  arm (#482) used forward-share **p=0.78** in its Amdahl (4.55× cap).
- But the same-run direct measurement (#455) says the **BACKWARD (costate VJP) is 82%**, forward only 18%.
- These cannot both be true of the teacher fwd+bwd. If backward dominates, then: (a) the ANE-forward path
  is even more Amdahl-capped than reported (replacing 18% caps total at 1/0.82 = **1.22×**, not 4.55×);
  (b) the surrogate-the-forward arm (#455) and the cheaper-forward arm (#456) were attacking the **wrong
  component** — the real 95%-kill target is the **costate VJP (backward)**, not the forward.
- **This is the single most important thing a detailed accounting surfaces and it is currently UNRESOLVED.**

### ⚠ CONTRADICTION #2 — harness reconciliation (why the ms numbers don't add up)
- Diagnostic teacher fwd+bwd = 3009 ms/pair × 600 = **1805 s/ep ≈ 30 min** — but the in-loop epoch is
  **169.7 s/ep**. The diagnostic harness is **~12× heavier** than the in-loop path (fp64 / unbatched /
  isolated-profiling overhead). So the 537/3009 ABSOLUTES are NOT the in-loop cost; only their RATIO is a
  candidate for transfer — and even the ratio is unverified in-loop.
- The 1-thread standard (#456, 2.96×) and the ANE 38× are all forward-only diagnostic numbers on this
  heavier harness — none is the in-loop per-component cost.

## D. The instrument to close it (BUILT, wiring deferred)

The da_db D-A `_measure_component_decomposition` producer (#480, just built) emits per-component IN-LOOP
timers — `_r_fwd` (render-R), `_teacher_fwd`, `_teacher_bwd`, `_wit_fwd`, `_wit_bwd` — as an 8-field
monotonic timer row. Its standalone module `tac.witness_control.telemetry_producers` is COMMITTED; the
trainer wiring is DEFERRED (multi-arm hot-file contention). **This is the exact instrument that resolves both
contradictions.** To get the authoritative per-component in-loop split: wire the D-A timer (or a standalone
n24 profiler that reuses it) + run ONE profiled epoch (n24 extrapolates linearly — the accum loop is
per-pair linear — so n24×25 is a legitimate wall-clock estimate, cheap, ~minutes, no OOM risk).

## E. RECOMMENDATION (what to do with this accounting)

1. **RESOLVE the fwd/bwd contradiction FIRST** — it decides whether the 95%-kill target is the forward
   (ANE/surrogate/cheaper-forward arms) or the backward (costate VJP). Wire the D-A timer, run n24 profiled,
   get the real in-loop `_teacher_fwd` vs `_teacher_bwd` split. If backward is 82% in-loop too, PIVOT the
   95%-kill to attack the VJP (checkpointing the backward, a cheap-VJP surrogate, or the pre-SE tileable
   surrogate #484 which — if it clears — replaces the WHOLE teacher fwd+bwd over the boundary region, so it
   sidesteps the fwd/bwd question entirely).
2. **The tau-stage +26% is the only large lever-group cost** — if wall-clock is the priority, the
   cheapest win is to audit whether persistence/clDice/lane-render-band all need to be ON simultaneously at
   ep300, or can be staggered/cadence-reduced (score-neutral check owed).
3. **Verdict is already async-hidden (47% duty)** — do NOT spend effort making the verdict cheaper for
   wall-clock; it overlaps the step. (It DOES matter for memory, a separate axis.)

Provenance: #306 `per_lever_compute_audit_20260705.md` · #455 `onpolicy_surrogate_95kill_20260713.md`
(537/3009) · #456 `cheaper_exact_forward_transfer_95kill_20260713.md` (2.96× 1-thread) · #482
`ane_unlock_correction_20260713.md` (p=0.78, now suspect) · da_db `telemetry_producers.py` (the instrument).
