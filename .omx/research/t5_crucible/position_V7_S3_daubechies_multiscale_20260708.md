# Position V7 S3 — Daubechies (coarse-to-fine / multiscale) — crucible_v7 restart

- council_tier: T3 · seat: S3 Daubechies · [no-triality] · review_status: blind-position-round-1
- Authority: `[macOS advisory / pure-math + $0 read-only probe]`. Pointer 0.19110 [contest-CPU]
  UNMOVED — v7 is MEANS until its byte-closed n600 exact row. NO launch/stop/config change performed.

## STORES CONSULTED
CONVENING_T3_v7 (seat contract + my face) · DRAFT_v7_restart_config_synthesis_20260708 (§1–§6) ·
witness_native_schedule_derivation_20260709 (Ch.4 Γ/Modica–Mortola/turnpike; τ*=0.31 knee band
[0.19,0.53]; info/octave CV≈0.39; MCF Lane retention 1.00→0.13) · crucible_v7_authored_20260708
(diff-vs-v6 table + wiring-gap list) · **$0 probe:** `src/tac/witness_control/tail_cycles.py`
`next_tau` (L64–71) + TailCycleConfig `tau_end` clamp (L89–90) + trainer TAIL arm block
(train_levelset…_mlx.py L6340–6420). Not consulted: sibling position files (blind), council posterior.

## Position (concrete accept/adjust of each council_pending knob)

**Face-1 — geometric vs sealed cosine_hold at the knee (the coarse-to-fine axis).** ACCEPT
geometric. Geometric τ-decay = log-linear = **equal information per octave** = constant Fisher-Rao
velocity, MEASURED-confirmed by info/octave CV≈0.39. This is the textbook multiresolution ladder:
resolve one octave at a time with equal effort because 1/f interface statistics carry ~equal
content per octave. cosine_hold is scale-space-WRONG (rushes mid-τ where the diffuse interface
crosses Nyquist, lingers at the endpoints). DERIVED, not a preference.

**"Does geometric spend enough dwell near the knee where the finest scale (lane dash) births?" —
YES, and the mechanism is TAIL_k, not the removed hold (probe-confirmed).** The DRAFT deletes
`--tau-hold-frac 0.2`; the dwell function is not lost, it is RELOCATED to TAIL_k = the turnpike
tail (derivation §2: extra budget → tail, never longer transients). Probe: `next_tau` returns
`max(τ_{k-1}·halving, τ*)` **clamped `≥ tau_end=0.31`** — so TAIL runs warm-restart (SGDR-style,
LR-restarted) cycles descending τ0≈0.755(@ep726) → 0.377 → 0.31-floor, i.e. explicit finest-scale
refinement THROUGH the knee band and PARKED at the floor. The resolution floor is a HARD clamp: TAIL
can NEVER alias sub-grid below 0.19. This is strictly MORE finest-scale dwell than cosine-hold's
at-endpoint 0.2·3000≈600 ep, and it is dash-safe by construction. **My initial sub-grid-erasure
concern is dissolved by the clamp — a vindication, not a revision.**

**Knob-2 (TAIL k_max): ACCEPT 2.** 2 cycles × cycle_floor 387 ≈ 774 ep of fine-band warm-restart
dwell ≥ the removed 600-ep hold → adequate knee dwell; `stop-marginal-s 1e-4` self-limits below that.

**Face-2 — lane-dash survival under unified L_τ vs discrete tau_softplus (top-2 reduction): unified
is dash-FAVORABLE. ACCEPT `--seg-form-unify-tau`.** At the floor τ=0.31≈0.3, `L_τ=τ·logsumexp(φ/τ)−φ_y`
→ `max_k φ_k − φ_y`, i.e. the pure **top1−top2 (Lane-vs-Road) margin** — exactly the flip the dash
depends on. The multi-class "sub-runner-up mass" (classes Undriv/Movable/MyCar, far in logit space
for a lane pixel) **VANISHES as τ→small** (logsumexp→max) — so it neither dilutes nor competes with
the dash margin at the floor; unified L_τ ≡ discrete tau_softplus(0.3) THERE. The gain is on the way
down: no ep300 discontinuity (the measured 0.0056→0.020, 3.4× bump) to destabilize a just-nucleated
fragile dash. CE(τ=1) spreads gradient across all 5 classes (dilutes dash signal); the continuous
descent progressively CONCENTRATES on the one competitor that matters. HELP, both arcs. The DRAFT's
open question (does τ·CE reweight already-correct pixels) is answered — at the floor it converges to
the same hinge — and is A/B-settled regardless.

**Face-3 — persistence/amplify + LADDER lane curve-prior: coherent, correctly scale-targeted.**
persistence-loss operates on the birth-death diagram = the multiresolution structure itself; it is
the erasure-antidote for the LOWEST-persistence (finest-scale) features MCF destroys (measured Lane
retention 1.00→0.13). Under geometric anneal the finest scale is reached LAST (near τ*), which is
exactly when the LADDER source (λ_k area-nucleation + lane-band render prior + amplify) must be hot —
timing aligns. L_τ alone cannot nucleate the dash (Modica–Mortola minority-phase fact); the source
carries it. Keep the stack. **Knob-3 (LADDER gate thresholds): ACCEPT builder defaults** (λ-gate 0.0,
release 0.95, sigma-eff 1.5) — the DRAFT's recalibrate-from-run-1 alternative is MOOT (run-1 produced
NO trajectory); recalibrate in v7.1 from the first real trace.

**Knob-1 (event-sensor tags + caps): ACCEPT the honest FAIL_SAFE_CAP tagging; caps 726/500/450 OK.**
The authored config is honest that none are sensor-fired (fixed gates). Multiscale ranking of the 3
OWED wirings: **lane-band↔nucleation is HIGHEST value** (the dash is the finest scale; its birth
TIMING is ~the whole d_seg game — annulus is 97% of d_seg and the dash is its fragile tail).
Muon-timing is low-stakes (gentle finisher, can't-nucleate → early/late doesn't erase the dash);
chroma is a secondary sharpener. Under geometric, τ enters the dash band [0.19,0.53] only ~mid-run,
so lane-band@500 fires EARLY relative to nucleation — but as an analytic render prior (openpilot
poly) that is harmless-to-mildly-helpful (pre-shapes the manifold the dash nucleates into), not a
loss switch. So it is fine to launch on the cap.

**Face-4 — launch-now vs build wirings first: LAUNCH NOW.** Building the 3 sensor→start wirings BLIND
(no trajectory to calibrate the nucleus/annulus thresholds) risks mis-calibration; measurement-first
says take the first geometric+unified trajectory, then wire calibrated in v7.1. One pre-registration
ask (below) makes that build cheap.

## Assumption tags (#363)
- τ*=0.31 knee (band [0.19,0.53]) = **VERIFIED_VIA_EMPIRICAL_ANCHOR** (Kneedle).
- geometric = equal-info/octave (CV≈0.39) = **VERIFIED_VIA_EMPIRICAL_ANCHOR**.
- TAIL clamps τ ≥ 0.31 (dash-safe floor) = **VERIFIED_VIA_SOURCE_INSPECTION** (`next_tau` L64–71,
  `tau_end` L89–90).
- unified-L_τ dash-favorable (sub-runner-up mass vanishes at floor; ≡ tau_softplus there) =
  **INFERRED_FROM_DOMAIN_LITERATURE** (logsumexp→max is math; the in-practice dash benefit is A/B-gated).
- lane-band@500 premature-but-harmless render prior = **INFERRED** (no trajectory yet to time nucleation).

## Verdict contribution
**PROCEED_WITH_REVISIONS** (single, non-blocking pre-registration revision):

- **R-S3 (pre-registration, not a config change):** at relaunch, LOG the τ(epoch) trace and the
  nucleus-guard sensor's *would-fire* epoch (observability-only), so the HIGHEST-value OWED wiring —
  lane-band↔nucleation (dash-birth timing) — is built CALIBRATED in v7.1 rather than blind. This is
  the one thing my coarse-to-fine lens flags as leaving measurable signal on the table; it does NOT
  block the restart.

Everything my face touches is otherwise PROCEED/VINDICATE: geometric (equal-info/octave, DERIVED);
knee-dwell answered by floor-clamped TAIL warm-restarts (superior to the removed hold); unified L_τ
dash-favorable and discontinuity-removing; persistence/amplify/LADDER correctly targets finest-scale
erasure; k_max=2 adequate; LADDER builder defaults (no trajectory to recalibrate); launch-now.
