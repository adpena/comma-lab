# CE-WINDOW INTERVENTION PACKAGE — run-2 restart decision (ARMED, GO-GATED)

**Date:** 2026-07-05 · **Subagent:** CE-WINDOW-PRESTAGE · **Status: NOTHING LAUNCHED — every option
below awaits the operator's GO.** Live run `experiments/results/levelset_n600_witness_20260705T015247Z`
(pid 39999) was READ-ONLY throughout; all probes ran CPU-side on snapshot copies.
**Axis:** every number `[macOS-CPU/MLX advisory] NON-PROMOTABLE` (gradient-structure calibration,
NOT d_seg evidence; the pre-registered success criteria below are defined on the run's own n600
async verdicts). **Pointer 0.19110 UNMOVED — everything here is MEANS.**

---

## 0. HEADLINE — the situation changed while pre-staging: the run is DEADLOCKED, not flat

**MEASURED (run.log):** the run has taken **zero optimizer steps since ~ep92.** The spike guard
(loss > 5.0 × median of the last 50 *accepted* batch losses, `spike_factor` default 5.0) began
firing at **ep92 batch ~13** and has skipped **75/75 batches every epoch since** (2,313 skip rows,
ep92→122+ and counting; ep100 verdict row shows `ep_loss: 0.0`). Mechanism (source-verified,
trainer spike block): **skipped batches never append to `recent_losses`, so the median can never
adapt** — after a persistent loss-level shift (batch loss ~6–8.5 → ~58–66, i.e. ~7×), every
subsequent batch skips forever. The window math proves the shift was SHARP: a smooth ramp of 7×
over ≥50 accepted batches would drag the median along (max ratio ≈ e^{0.08} ≈ 1.08 ≪ 5); the first
skip at ep92 batch 13 therefore localizes the jump to ~1 batch after an accepted (grad-clipped)
step at ep92 batch ~12.

Consequences, measured:
- The EMA (and the deployed verdict) is **frozen at ~ep92**: the ep100 "verdict" d_seg 0.12129 is
  the ep92-state EMA. The apparent ep50→100 "flat regime" (0.1217 → 0.1275 → 0.1213) is partly
  real descent (75→92) and partly **frozen-model readout** (92→100).
- Every epoch until ep300 is 100% waste (~2.9 min/ep). At **ep300 the CE→tau
  `curriculum_transition` clears `recent_losses`** (source-verified re-treat) — i.e. left alone,
  the run would resume training **directly into the tau engagement + the known ep300 3-way
  collision, from 208-epoch-stale weights.**
- The `closed_loop` controller classified ep100 as `converging` (slope −0.0007) — it reads verdict
  d_seg only and is **structurally blind to the skip-deadlock** (no skip-rate input). Controller
  gap logged in §6.

**Live weights are NOT damaged (measured):** the witness-alone CE seg term with the LIVE
(resume-sidecar) weights is **0.928×** the EMA's (33.05 vs 35.61, 12 pairs) — the ep92 jump lives
in the composed-frame / auxiliary terms, not in the deployed seg surface. Resuming from the ep100
resume state is sound. Exact per-term attribution of the jump is **UNMEASURABLE offline** (the
seed module is not persisted in any checkpoint; accepted-batch losses are not logged per-term):
ranked hypotheses = composed-frame CE jump (seed compose path), amplify-hinge, persistence-ramp ×
clDice, eikonal-ramp. Per-term telemetry is BUILD-REQUIRED (§6) so the next jump is attributable.

**Checkpoints that exist RIGHT NOW** (the run dir keeps ONE rolling resume state — there are no
per-epoch retained intra-stage files):

| file | epoch | content | snapshot (read-only copy) |
|---|---|---|---|
| `levelset_resume_state.npz` | **100** | live params + opt state + RNG + frozen spike window | `experiments/results/bd_calib_20260705/snap/resume_state_ep100.npz` sha256 `d09a49b9…` |
| `levelset_witness_ema_BEST.npz` | 100 | EMA shadow, d_seg 0.12129 (best) | `…/snap/ema_BEST_ep100.npz` sha256 `ab06c6ae…` |
| `levelset_witness_ema_mlx.npz` | 100 | EMA latest (== BEST here; frozen since ~92) | — |

(The rolling resume file will be overwritten at ep125/150/… with the SAME frozen weights and the
SAME frozen spike window; the ep100 snapshot above is the canonical restart source.)

---

## 1. w_bd\* CALIBRATION (deliverable 1) — measured, ep100 EMA BEST, 12 gt_n24 pairs

Harness: `experiments/probe_boundary_distance_calibration.py` (new; sister of the focal probe;
same validated witness-alone reconstruction; the bd band map + term **imported from the trainer**
— build `535e142be` exact semantics; chunked, kill-durable; MLX-CPU; ~50 s/pair; run as 4+2
chunks; raw sidecars + merged JSON in `experiments/results/bd_calib_20260705/`). Tests:
`experiments/test_boundary_distance_calibration_probe.py` (11 green, incl. trainer-band geometry,
tie-at-boundary zero, gradient-only-on-band, tex-recovery exact inverse). Reconstruction validated
per pair: phi-leaf re-render matches the numpy deploy frame to **3.8e-5** max abs (pre-round).

**Surface (honest):** shares are measured on the **PHI (SDF-field) leaf — the ONLY surface common
to both terms.** The bd term reads `model.sdf` directly and has ZERO gradient w.r.t. the rendered
frame, so the focal probe's frame-surface shares structurally cannot see it (do NOT compare the
two tables' absolute shares; compare within-table reallocation).

| w_bd | island grad share | bulk-boundary | bulk-interior | bd share of (CE + w·BD) |
|---|---|---|---|---|
| 0 (current) | 0.0123 | 0.0565 | 0.9312 | 0 |
| 0.01 | 0.0123 | 0.0566 | 0.9310 | 0.50% |
| 0.05 | 0.0126 | 0.0575 | 0.9299 | 2.4% |
| 0.1 | 0.0130 | 0.0587 | 0.9282 | 4.7% |
| **0.2** | **0.0138** | **0.0614** | **0.9247** | **9.05%** |
| 0.5 | 0.0164 | 0.0699 | 0.9136 | 19.9% |
| 1.0 | 0.0208 | 0.0845 | 0.8947 | 33.2% |

Magnitudes (12-pair means): **CE term (w_seg=100 × mean(ce·hinge)) = 35.61; bd raw = 17.72**;
mixing law is exactly `ratio(w) = w·17.72/(35.61 + w·17.72)` ⟹ the 5–15%-of-total window =
**w ∈ [0.106, 0.355]**. Registered: canonical equation `boundary_distance_weight_calibration_v1`
(`tac/canonical_equations/boundary_distance_calibration_20260705.py`, registered via
`tools/register_bd_calibration_20260705.py`).

**Recommendation: w_bd\* = 0.2** — measured directly (not interpolated): 9.05% of total
(mid-window: neither vanishing at 0.1's 4.7% edge nor approaching 0.5's 19.9% dominance);
monotone bulk-boundary reallocation (+0.5pp, +9% relative — and the bd gradient is 100%
concentrated on the 2px GT-edge band, test-proven, so the per-band-pixel boost is ~×1.6 at equal
global share); **island share RISES** 0.0123→0.0138 (lane edges are inter-class edges inside the
band — no island collapse; interaction with the island levers is supportive, not competitive).
Caveat (registered in the equation's domain-of-validity): M_ce falls as training progresses, so
ratio(0.2) GROWS at fixed w — re-run this probe at any later checkpoint before re-tuning.

---

## 2. THE THREE OPTIONS (all GO-GATED; exact diffs vs the live `launch.sh` argv)

Common mechanics (verified against the real code, flags grep-verified — none invented):
- `--resume-from <snap>/resume_state_ep100.npz` resumes at **ep101** with opt state + bit-faithful
  RNG (`_resolve_resume_path` accepts an explicit npz; use the SNAPSHOT, not the rolling file).
- Changing any loss lever on resume trips the F2 drift guard ⟹ `--resume-allow-lever-drift`
  (intentional warm-start re-treatment; the drift log row records exactly what diverged).
- **`--resume-clear-spike-guard` (LANDED this pre-stage, default OFF)** discards the frozen
  50-batch median so the guard re-seeds from the first accepted batch (exposure = 1 batch, then
  re-armed at the new loss level). Without it, ANY resume of this checkpoint re-deadlocks
  instantly (the sidecar restores the frozen window — source-verified). Functional smoke executed:
  n1 resume with flag → `cleared_frozen_window_len: 2` + training continued; without flag →
  `restored_recent_losses: 2` (default path byte-identical; sibling lever byte-identity suite
  10/10 green after the edit).
- Routing: **through the governed launcher** `tools/launch_witness_run.py --config fresh_seeded
  --extra-trainer-flags "<deltas>"` (C5 passthrough validates every flag against the real argparse
  and re-runs the memory preflight on the emitted launch.sh; raw python launches FORBIDDEN).
  GO-procedure verification step: **diff the emitted `launch.sh` against the live run's** — it
  must be identical modulo `--out-dir` + the deltas listed below.

### OPTION A — lever-only restart (minimal delta, cleanest attribution)

Extra trainer flags (the ONLY diffs vs live argv):

```
--resume-from experiments/results/bd_calib_20260705/snap/resume_state_ep100.npz \
--resume-allow-lever-drift --resume-clear-spike-guard \
--boundary-distance-weight 0.2
```

- **Trunk identity:** params/opt/RNG at ep101 are byte-identical to the live run's ep100 state.
  This is NOT a #270-style inert-until-N restart: the trajectory diverges at ep101 batch 1 BY
  DESIGN (bd term active + guard re-seeded) — the honest framing is "warm-start re-treatment from
  a bit-faithful ep100 state", provenance = the two snapshot sha256s above + this memo.
- **Cost:** ~0 lost training (ep101–122+ contained zero accepted steps); kill + relaunch ≈ minutes
  of operator time; CE window to tau@300 = ~200 ep ≈ 9.7 h.
- **Keeps** tau@300 ⟹ retains the known ep300 3-way collision (tau + persistence-completion +
  seed-anneal-end) and gives the bd lever only ~200 CE epochs.

### OPTION B — lever + clock (recommended)

Option A flags **plus** the curriculum-symposium READY-NOW staggers and the tau delay
(order-preserving shifts; all flags exist — grep-verified):

```
--tau-softplus-start-epoch 400 \
--lane-band-start-epoch 450 \
--seed-anneal-epochs 275 \
--persistence-warmup-epochs 275
```

- tau 300→400 restores the ~110 CE epochs the deadlock consumed and gives the bd lever a full
  ~300-epoch CE window (Tishby window rule: loss changes engage during CE — restart@101 is
  mid-CE, the right window).
- band 350→450 PRESERVES the derived tau→band ordering and gap (+50; band@350 with tau@400 would
  silently reorder band BEFORE tau — an untested ordering, refused).
- seed 300→275 + persistence 300→275 = the symposium §C.ii item-3 stagger (completions land 125 ep
  before the tau boundary; the 3-way collision dissolves to: 275 completions → 400 tau → 450
  band). Note the run is INSIDE both anneals at ep101, so the shift causes small forward jumps at
  restart: seed weight 0.746→0.704, persistence ramp 0.34→0.37 (≈; quantified re-treats, absorbed
  by the same warm-start framing).
- Event-triggered tau (`--curriculum-event-triggered` exists) is NOT used: the symposium gates
  event mode on the boundary re-anchor BUILD (its item 2); fixed 400 is the honest interim.
- Muon@726 / l7@1001 / all other flags unchanged.

### OPTION C — no restart (watch) — DOMINATED, kept for completeness

The pre-staged "watch at ep300" premise is broken: there is nothing to watch — the model does not
train again until the ep300 `curriculum_transition` clears the guard, at which point tau + the
3-way collision hit 208-epoch-stale weights simultaneously (the worst studied configuration). If
GO is withheld anyway: watch (a) `spike_skip` rows — any epoch <75 skips = the regime broke
organically; (b) post-ep300 verdicts — the shadow controller (`src/tac/witness_control/`)
classifications, ROLLBACK on `diverging_erasing` (its measured #205 backtest fired that at the 3rd
post-tau verdict) with rollback target = the ep300 `stageTau` checkpoint (`--stage-checkpoints` is
on). Known gap: `closed_loop`/shadow classifications have NO skip-rate input and currently read
`converging`/`plateau` on a frozen model — do not trust them until §6 item 3 lands.

**RECOMMENDATION: OPTION B.** The deadlock makes SOME restart mandatory (C forfeits ≥9 h and then
executes the known-bad collision); between A and B, the deadlock has already broken run-2's clean
single-delta attribution, every B delta is independently certified (bd: §1 measured; staggers:
symposium READY-NOW $0; tau delay: restores the intended CE budget), and B is the only option that
both engages the calibrated lever inside its Tishby window at full width AND removes the measured
3.4× collision-bump risk. Attribution cost is accepted and recorded (risk R5). **GO-GATED — no
launch without the operator.**

---

## 3. PRE-REGISTERED SUCCESS CRITERIA (defined on the run's n600 async verdicts, NOT the probe)

Baseline: d_seg(ep100) = 0.12129 (frozen-EMA readout ≈ ep92 state).

- **SC1 — deadlock escape (immediate):** accepted steps in ep101 (log: `spike_skip` count < 75 for
  ep101; `resume_spike_guard … cleared_frozen_window_len ≈ 50` row present). ALARM: any epoch with
  ≥50% skips in ep101–125 ⟹ the ep92 regime is organic/recurring ⟹ STOP, escalate (do not
  iterate weights blind).
- **SC2 — lever verdict (primary):** d_seg slope over the first 50 post-restart epochs must beat
  **−0.01/25 ep**, i.e. d_seg(ep150) ≤ 0.10129. If slope is worse than −0.005/25 ep, the bd-lever
  verdict is NEGATIVE → next arm drops `--boundary-distance-weight` (Option A minus bd == pure
  deadlock-escape control) — the lever is killed by ITS pre-registered bar, not vibes.
- **SC3 — mechanism check (probe re-run):** at ep150–175, re-run the bd probe on the fresh EMA:
  bulk-boundary within-flip (0.503 @ep75) must FALL; island grad share must stay ≥0.8× its w=0
  baseline (no starvation).
- **SC4 — collateral:** d_pose stays within its 0.10–0.17 band; no `film_stiefel` residual blowup
  in `dm1_telemetry`.

## 4. RISK REGISTER

| # | risk | likelihood | mitigation |
|---|---|---|---|
| R1 | ep92 jump mechanism recurs post-restart (organic instability, not schedule) | med | SC1 alarm; guard re-armed after 1 accepted batch (median re-anchors at the new level, so only a further ≥5× jump re-trips); per-term telemetry BUILD (§6.4) makes the next event attributable |
| R2 | guard-clear removes spike protection for exactly 1 batch | low | exposure is 1 batch by construction (median defined after first append); grad-clip 1.0 still active |
| R3 | bd share grows as CE falls (fixed w, falling M_ce) | certain, slow | registered equation caveat; SC3 probe re-run re-derives w(r); acceptable drift direction (bd is a d_seg-aligned placement prior) |
| R4 | Option B mid-anneal shifts (seed 0.746→0.704, persistence 0.34→0.37) perturb the composed loss at restart | low | warm-start re-treatment framing; guard freshly re-seeded absorbs the level shift; magnitudes quantified above |
| R5 | attribution loss: B bundles 4 deltas | certain | accepted + recorded; the deadlock already broke the clean A/B; each delta independently certified; SC2/SC3 isolate the bd lever's verdict; run-3 remains the clean-curriculum vehicle |

## 5. WHAT WAS BUILT/LANDED THIS PRE-STAGE (all default-off / read-only; nothing launched)

- `experiments/probe_boundary_distance_calibration.py` — bd calibration probe (phi-surface, chunked,
  kill-durable, trainer-imported bd semantics, live-vs-EMA CE diagnostic, `--weights` override).
- `experiments/test_boundary_distance_calibration_probe.py` — 11 tests green.
- Trainer `--resume-clear-spike-guard` (default OFF = bit-faithful restore unchanged; ruff-delta 0;
  F821 clean; functional resume smoke executed both paths; sibling byte-identity suite 10/10).
- Canonical equation `boundary_distance_weight_calibration_v1` + registration tool (registered).
- Snapshots + raw calibration artifacts under `experiments/results/bd_calib_20260705/`.

## 6. BUILD-REQUIRED (named honestly; none block Options A/B)

1. **In-loop spike-guard escape hatch** (auto-clear + log after K consecutive full-batch skips,
   default K≈150=2 epochs) — the permanent fix for the deadlock CLASS per "bugs must be permanently
   fixed AND self-protected against"; the resume flag is the surgical instance. Pair with a STRICT
   preflight check (`check_spike_guard_has_deadlock_escape`) when landed.
2. **Event-triggered tau boundary re-anchor** (curriculum symposium §C.ii item 2) — still gates
   event-mode tau; Option B uses fixed 400 meanwhile.
3. **Skip-rate input to the closed-loop/shadow classifications** — both currently classify a
   0-steps/epoch run as `converging`/`plateau` (measured on this incident).
4. **Per-term loss telemetry row** (per-epoch `{"stage":"loss_components", ce_composed, pose, eik,
   length, persistence, amplify, seed…}`) — the ep92 attribution was UNMEASURABLE offline solely
   for lack of this row.

## 7. TRIALITY + self-reflection (Catalog #363)

- **DAG:** FEED-205ce appended (`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`).
- **equations:** `boundary_distance_weight_calibration_v1` REGISTERED (measured anchor). The
  deadlock mechanism itself is a code-behavior finding, not a law — carried by this memo + the
  BUILD list (# FORMALIZATION_PENDING:spike-guard-deadlock-is-a-bug-class-not-an-equation; the
  escape-hatch landing will carry its own gate).
- **DSL:** no gauge added — same discipline as the focal memo: the gauge belongs with the FIRING
  decision; if GO lands on A/B, add `SegLossGauge{BASELINE, BOUNDARY_DIST(0.2)}` with the restart.
- Verification statuses: deadlock mechanism + skip counts + window math —
  `VERIFIED_VIA_SOURCE_INSPECTION` + `VERIFIED_VIA_EMPIRICAL_ANCHOR` (run.log rows, trainer lines);
  calibration table — `VERIFIED_VIA_EMPIRICAL_ANCHOR` (executed probe, sidecars on disk); ep92
  per-term attribution — `ASSUMED_AWAITING_VERIFICATION` (ranked hypotheses; telemetry BUILD named);
  option-B mid-anneal jump magnitudes — computed from flag semantics
  (`INFERRED_FROM_DOMAIN_LITERATURE`-grade; exact shapes in code, ≈ labels kept).

**Pointer 0.19110 UNMOVED.** This package is MEANS; the END remains a byte-closed
`upstream/evaluate.py` row below it.
