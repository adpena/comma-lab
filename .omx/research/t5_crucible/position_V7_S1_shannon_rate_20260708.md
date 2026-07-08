# Position V7 — SEAT S1 (Shannon, LEAD — information/rate) — 2026-07-08

STORES CONSULTED: CONVENING_T3_v7_design_symposium_20260708.md (seat contract) ·
DRAFT_v7_restart_config_synthesis_20260708.md (§1 resolutions · §2 spine · §3 pose · §4 A/B) ·
crucible_v7_authored_20260708.md (diff-vs-v6 · wiring-gap list · council_pending knobs) ·
witness_native_schedule_derivation_20260709.md (L_τ = τ·CE(φ/τ) continuous form) ·
ORCHESTRATION_LEDGER.md (folds 247–534: UNIFY-TAU bit-exact, TAIL_k, LADDER #323, LR-pin v6.4) ·
CLAUDE.md non-negotiables (THE GOAL threshold ladder; rate term 25·|archive.zip|/37_545_489) ·
MEMORY L17/L68/L69/L74 (d_seg gap, pose OPEN, task-RD floor). [no-triality] — research memo.
Own budget arithmetic recomputed this session (blob 65.6KB → rate 0.04368, verified).

## The rate-view frame (why this restart is a d_seg play, not a rate play)

At the projected v7 archive of ~65.6KB, the rate term is **0.0437** — only 23% of the 0.19110
pointer, and BELOW the measured task-RD rate floor slack. The store-nothing pose carrier adds ~0
counted bytes (verbatim block, §3), so the archive ≈ the entropy-shaped weight blob. Budget
arithmetic (MEASURED rate + banded pose → the d_seg headroom that must be earned):

| threshold | rate | pose (band) | d_seg budget | current d_seg ≈0.0045 | gap |
|---|---|---|---|---|---|
| **T_1** (<0.19110) | 0.0437 | 0.018–0.026 | **0.00121–0.00129** | 0.00456 (τ*-floor) | ~3.5–3.8× |
| **T_3** (<0.15) | 0.0437 | 0.018–0.026 | **0.00080–0.00088** | 0.00456 | ~5.4× |

Decisive observation: the T_3 d_seg budget (0.00080–0.00088) **coincides with the analytic
lane-band floor 0.00087** (L71). So T_3 is reachable ONLY if the LADDER lands the lane at its
analytic floor AND the annulus/other-class residual (~97% of d_seg in 4.7% area, L66) is driven
near-zero. Rate is not the constraint; d_seg is. This restart is correctly aimed.

## Weight-entropy × unified-L_τ — the continuous anneal HELPS the coded-bytes trajectory

`L_τ = τ·logsumexp(φ/τ) − φ_y` stays **O(1) across the whole anneal** (τ→0 ⇒ L_τ → max φ_k − φ_y,
the margin deficit — it does NOT scale to zero). Therefore the seg-loss magnitude vs the fixed
weight-entropy penalty (λ=15) stays in **roughly constant ratio** through τ:1→0.31. The v6 discrete
switch injected the ep300 loss-magnitude discontinuity (the measured 3.4× bump, FEED-ft) which
jolts that ratio and can transiently DECOMPRESS the entropy-shaped distribution. **Removing the
switch is rate-favorable**, not just d_seg-favorable: no mid-run jolt to the byte distribution.
This is an independent rate argument FOR the unify-tau BUILD (converges with S2/derivation).

## TAIL_k(2) — bounded, but the stop rule is RATE-BLIND (my central finding)

The council question: do TAIL cycles decompress the distribution the entropy penalty spent 3000
epochs shaping? Three facts bound the risk: (a) weight-entropy λ=15 carries into the tail
UNCHANGED (verified in the diff UNCHANGED list) — it keeps pulling weights toward the compressible
manifold DURING the tail; (b) TAIL LR ∝ τ_k with τ_k ≤ 0.155 (first cycle) — the warm-restart
perturbation is small in absolute LR; (c) dwell 237 re-settles. So decompression is bounded, and
k=2 caps the blast radius. **BUT** `--tail-stop-marginal-s 1e-4` watches **d_seg marginal only**.
A cycle can gain Δd_seg > 1e-4 while inflating bytes (ΔS = −100·Δd_seg + 25·Δbytes/B). The stop
rule cannot see the byte cost → a rate blind spot. This is the sister of the confound class:
a stop criterion that certifies "still improving" on the distortion axis while silently paying rate.

## Position (each council_pending knob)

1. **Event-sensor caps 726 / 500 / 450** — ACCEPT from rate view (schedule caps are rate-neutral;
   Muon whitens updates, no param-count change). **Launch WITH tagged caps, do NOT build the 3
   sensor→start wirings first**: the wiring gaps move WHEN d_seg transitions fire, not the coded
   bytes — rate-neutral. Building them speculatively is premature without run-1 trajectory data;
   the run itself generates the ν/nucleus/annulus traces that would calibrate those sensors. (Defer
   sensor-choice correctness to S2/S4-Rudin; my lane is only that they don't gate the launch on
   rate grounds.)
2. **TAIL k_max = 2** — ACCEPT (bounded blast radius). **REVISE the stop rule to be RATE-AWARE**:
   gate cycles on **net-ΔS marginal** (or byte-close each cycle-boundary checkpoint and gate on
   measured ΔS), NOT d_seg-marginal-alone. `stop-marginal-s 1e-4` as the d_seg leg is fine; add a
   rate leg so a byte-inflating cycle is stopped even if it still lowers d_seg. Cheap: the tail
   checkpoints already exist per-cycle; brotli-size them.
3. **LADDER gate thresholds** — ACCEPT builder defaults from rate view. **Flag (rule-118 boundary):
   confirm the lane curve-prior is render-time GENERATED (openpilot VP-tangent, offline-FREE) and
   NOT a stored per-frame learned residual.** A generated prior is free; any stored lane residual is
   COUNTED and re-opens the rate budget. (Curriculum-λ correctness → S5/Daubechies.)
5. **run-1 stop point** — prefer **seal-complete**, to maximize the incumbent-arm trajectory for the
   schedule A/B (run-1 is the only discrete-stage arm; more trajectory = a stronger baseline).

## Assumption tags (#363)
- rate = 0.0437 for v7: **INFERRED / projected** from the v6-ancestor blob. Param count unchanged ⇒
  the ceiling holds, but the brotli-compressed v7 size is **UNMEASURED** — must byte-close at export
  (AXIS-9). Do NOT claim a T_1/T_3 crossing from the projected rate. (DERIVED-AT-CONFIG, config-conditional.)
- pose 0.018 (ancestor 3.4e-5) / 0.026 (operator spare): **ASSUMED_AWAITING_VERIFICATION** — witness
  d_pose is OPEN/UNMEASURED (L68/L69); 0.018 is BORROWED ancestor, not a witness measurement. The
  budget table shows T_3 survives even the 0.026 pose band, so the restart is not pose-gated — but
  the crossing claim needs the byte-closed witness d_pose.
- L_τ stays O(1) ⇒ constant loss/entropy ratio ⇒ smoother byte trajectory: **DERIVED** (from the
  τ·CE(φ/τ) form; well-grounded).
- λ=15 holds distribution through TAIL; stop rule watches d_seg only: **VERIFIED_VIA_SOURCE_INSPECTION**
  (UNCHANGED diff list; tail_cycles stop semantics) — this is the finding, not an inference.
- current d_seg ≈ 0.0045; 5.4× gap to T_3: **MEASURED-ANCHOR** (through-R τ*-floor 0.00456, #205
  CE-floor 0.00496) — arithmetic gap, not opinion.

## Verdict contribution: PROCEED_WITH_REVISIONS
Rate view APPROVES the v7 restart: unify-L_τ is rate-favorable (removes the ep300 byte-jolt), the
rate budget is generous (0.0437), and the design correctly spends its effort on d_seg where the
binding gap is. **Launch with the tagged caps** (wiring gaps are rate-neutral d_seg-timing items).
Revisions, all cheap: (R1) make the TAIL stop **rate-aware** (net-ΔS / byte-close per cycle, not
d_seg-marginal-alone). (R2) **byte-close the v7 archive at the stop point** — 0.0437 is projected;
no crossing claim without measured bytes + measured witness d_pose. (R3) **confirm the LADDER lane
prior is generated-FREE (rule-118), not stored-counted.** Standing rate lever (non-blocking): each
10KB off the blob buys ΔS 0.0067 ≈ one tail cycle of d_seg headroom — worth a post-restart pass but
not a launch gate. Pointer 0.19110 UNMOVED; MEANS until the byte-closed n600 exact row.
