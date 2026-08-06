# ddm_la1 2026-08-05 Receipt

## Scope

Arm: `ddm_la1` terminal JD1 LR-anneal lever for the TR1 joint-descent line.

This receipt is build/preregistration only. No launcher, scorer, eval, archive, or long run was
started. `tr1_jd4_cont_ep1646` / jd6 was read only; no live run directory was edited.

Pointer status: own-vehicle frontier remains
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; borrowed contest pointer unchanged.

## Recall

Reviewed before implementation:

- Witness-line Muon anneal laws: `muon_finisher_schedule_warmstart_and_lr_anneal_v1` and
  `muon_switch_conditioning_criterion_v1` establish the pattern of warm-started terminal
  anneal, but their `--muon-lr-final-frac 0.1` is vehicle-scoped and was not transferred.
- #518 / beta2 warmup geometry: optimizer variance memory uses the `c/(1-beta2)` shape; this
  lever uses `c=2` converted to epochs at the live batch geometry.
- Cross-regime constant discipline: no inherited scalar is treated as a TR1 constant. The only
  default final fraction is derived from parent telemetry for the actual boundary window.

## Derivation

Source telemetry:
`/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1526/telemetry.jsonl`

Endpoint mechanism evidence:
`/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd5_endpoint_n600_both_bases.json`

Measured endpoint deltas, `[macOS-CPU frozen-scorer advisory]`, n600 both bases:

- Live basis: `delta d_pose = +0.048132309262887904`
- EMA basis: `delta d_pose = -0.054913158711088456`
- Live-vs-EMA endpoint d_pose gap: `0.10388646797397635`

The parent telemetry did not contain a pose-itemized `loss_terms.terms.*pose*` value. It did
contain active JD1 epoch rows and later `loss_terms` rows with `seg`, `rate`, and
`delta_sparsity`, so the implemented derivation labels the fallback source as
`epoch.ep_loss[jd1_pose_finish_active]`.

Inputs:

- Boundary: `start_epoch=1526`, `end_epoch=1646`
- `steps_per_epoch=150`
- `beta2=0.999`
- Active EMA decay: `0.9997777777777778`
- Base LR: `0.002`

Derived time constants:

- beta2 memory: `ceil(2/(1-0.999)/150) = 14` epochs
- active EMA memory: `ceil(2/(1-0.9997777777777778)/150) = 60` epochs
- tail length: `max(14, 60) = 60` epochs
- onset epoch for a 1646 boundary: `1586`

Measured last-60 active JD1 epoch-loss oscillation:

- `n=60`
- mean `0.8992388238112132`
- sd `0.011274632999681105`
- half_range `0.032927233179410265`
- relative half_range `0.03661678333666233`
- sign changes `38/58`

Derived default final fraction:

`final_frac = sd / (sd + half_range) = 0.2550714251294281`

Thus for `--lr 0.002`, final LR is:

`0.002 * 0.2550714251294281 = 0.0005101428502588562`

## Edits

- `experiments/train_tr1_partition_renderer_mlx.py`
  - Added `--jd1-lr-anneal {off,derived_tail}` default `off`.
  - Added `--jd1-lr-final-frac` default `0.0`, meaning derive.
  - Added pure parent-telemetry derivation helpers.
  - Added validation to refuse inert or unresumable LR-anneal shapes.
  - Wired ON-only schedule resolution at JD1 resume/engagement and ON-only
    `optimizer.learning_rate` assignment per epoch.
  - Added ON-only telemetry fields; OFF writes no schedule row and adds no `TR1Config` field.
- `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`
  - Added `lever_jd1_lr_anneal()`, active-shape only, with runtime receipt schema and falsifier.
- `src/tac/tests/test_ddm_bp1_boundary_reset_race.py`
  - Added parser fail-closed tests, pure derivation test, and AST check that main reaches
    `optimizer.learning_rate`.
- `src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`
  - Added parser/DSL tests for the new lever.

## Default-Off Proof

Absent flags parse as:

- `jd1_lr_anneal == "off"`
- `jd1_lr_final_frac == 0.0`

OFF does not enter `TR1Config`, so config hashes and checkpoint metadata are not extended by
this arm. The derivation helper is only called from `_resolve_jd1_lr_anneal_schedule()` when
`args.jd1_lr_anneal == "derived_tail"`. Epoch rows are extended only when
`jd1_lr_schedule is not None`. The optimizer LR setter is also guarded by the active schedule.

## Preregistered Boundary A/B

At the jd7-or-Case-B boundary, run matched ON vs OFF from the same checkpoint:

- OFF: existing flat LR, no `--jd1-lr-anneal`.
- ON: `--jd1-lr-anneal derived_tail`, omit `--jd1-lr-final-frac` unless MAIN pre-supplies an
  explicit boundary override.

Both arms must use the same checkpoint, seed, epoch window, `batch_pairs`, EMA mode, and scorer
endpoint protocol. Measure n600 endpoint deltas on both live and EMA bases.

Prediction: live/EMA endpoint divergence closes if terminal LR oscillation is the cause.

Falsifier: ON endpoint EMA is no better and live divergence is unchanged; classify as not
LR-driven.

## Verification

- `python -m py_compile` on the trainer, DSL, and two test files: PASS.
- Focused pytest selection: `6 passed in 0.51s`.
  - The sandbox emitted the known MLX atexit `No Metal device available` warning after pytest
    completion; tests still exited 0.
- `tools/review_tracker.py scan`: PASS, then mark-file pass 1 on all four Python files.
- `tools/review_tracker.py scan`: PASS with `New: 0`, `Changed (stale): 0`, then mark-file
  pass 2.
- Review query after pass 2:
  - trainer: `117 reviewed`
  - bp1 tests: `50 reviewed`
  - TR1 DSL: `45 reviewed`
  - JD1 DSL tests: `12 reviewed`

No score claim. No scorer slot consumed. No live jd6 modification.

## ON-arm endpoint adjudication (MAIN, 2026-08-06 — receipt jd7on_endpoint_n600_both_bases.json)

ON window ep1766→1886 (`--jd1-lr-anneal derived_tail`) completed rc=0 (7,125 s); n600 probe
rc=0 (1,361 s). Anneal telemetry confirmed the derivation live at real window scale: LR flat
0.002 through ~ep1830, then tail-anneal to 6.76e-4 (final_frac 0.4235143475126433, source
derived_sd_over_sd_plus_half_range — the smoke's immediate-onset was the 6-ep-window artifact,
as predicted).

Endpoint vs ep1766 baseline (same-basis deltas):

| basis | d_seg (Δ) | d_pose (Δ) |
|---|---|---|
| EMA (ships) | 0.004802653 (−0.000169279) | 0.020818391 (**+0.010723722 WORSE**) |
| live | 0.004846124 (−0.000577121) | 0.028631812 (**−0.185617092 BETTER**) |

- **Mechanism prediction CONFIRMED**: live/EMA d_pose divergence closed 21.2× → 1.38×
  (live 0.214249 → 0.028632). Terminal LR oscillation IS the cause of the divergence.
- **Shipping basis DEGRADED**: EMA window ΔS = pose_term +0.138551 + seg −0.016928 =
  **+0.121623 net WORSE**. The ep1766 EMA (0.010095) was an average over an oscillating live
  trajectory; the window's flat-LR first half dragged the EMA up before the anneal engaged.
- **Pre-registered falsifier: HALF-FIRED** ("EMA no better" TRUE · "live divergence unchanged"
  FALSE). Full adjudication requires the OFF arm (control, same ep1766 ckpt, fired 2026-08-06,
  out-dir tr1_jd4_cont_ep1766_la1off).
- Case-A strict bar (d_pose ≤ 0.00144 on ema): NOT met (14.5× above — REGRESSED from 7.0×).

### Instrument finding — gate36 pose calibration is STATE-DEPENDENT (m96 sign)

Trainer a1_gate (36-pair, #970 channel) read EMA d_pose ≈ 0.00218 at ep1885 while the n600
probe measures 0.020818 — a **9.5× subset-easy bias**, vs the banked 1.002–1.146 calibration
series from jd4/jd5/jd6. Mechanism: pose error is tail-concentrated (m96 — the hard pose
pairs are excluded from the 36-pair gate subset); as the bulk error collapses, the gate
under-reads the surviving tail. Validates the #970 design decision (gate = advisory-trend
only, n600 probe = authority). DO NOT project n600 pose from gate36 at low-error states.

### Adjudication table at the OFF endpoint (pre-registered now)

- If OFF EMA d_pose also worsens (≳0.02): anneal is not the cause of EMA degradation →
  compare EMA endpoints directly; lever verdict by the better shipping-basis endpoint.
- If OFF EMA d_pose continues the jd6 trend (−0.026/window → ~0.004–0.005): the anneal HURT
  the shipping basis mid-campaign → lever stays **OFF mid-campaign**; RE-SCOPE derived_tail
  as a TERMINAL-ONLY move (candidate use: stabilize live weights before E2 train→solve
  handoff / terminal GN solves, where a converged live state is the input — the EMA shadow
  is what ships, so mid-campaign live-convergence buys nothing by itself).

Error-atlas NPZs (ema+live, packbits, per-basis) + manifest landed next to the probe receipt
→ wp1 rows 3/9/13 unlocked as $0 reads.

## OFF-arm endpoint + FINAL la1 ADJUDICATION (MAIN, 2026-08-06 — receipt jd7off_endpoint_n600_both_bases.json)

OFF control window ep1766→1886 (flat LR 0.002, single-variable diff verified) completed rc=0
(6,717 s); probe rc=0 (1,276 s). Endpoint vs ep1766 (same-basis):

| arm | EMA d_seg (Δ) | EMA d_pose (Δ) | EMA seg+pose S | live d_pose | live/EMA |
|---|---|---|---|---|---|
| ep1766 start | 0.004972 | 0.010095 | 0.8149 | 0.214249 | 21.2× |
| **OFF ep1886** | 0.004586 (−0.000386) | 0.013307 (+0.003212) | 0.8234 | 0.222644 | 16.7× |
| ON ep1886 | 0.004803 (−0.000169) | 0.020818 (+0.010724) | 0.9366 | 0.028632 | 1.38× |

**la1 VERDICT (pre-registered branch 1 — direct EMA comparison): the anneal lever is OFF
mid-campaign.** OFF beats ON on BOTH EMA axes; ON-minus-OFF endpoint EMA ΔS = +0.113130.
Mechanism fully resolved: `derived_tail` collapses the live oscillation (divergence 16.7×→1.38×)
but the EMA — a weight-average OVER the oscillation — is the better scorer point, and killing
the oscillation converges live toward a WORSE point that drags the EMA with it (SWA-class
effect: the averaged iterate beats every individual iterate; the anneal destroys the ensemble).
RE-SCOPE: terminal-only utility is CONJECTURE (a consumer needing live≈EMA convergence, e.g.
continued-training-after-solve); no current consumer — default OFF everywhere, DSL lever retained.

## PLATEAU ADJUDICATION (the boundary's larger finding)

Window-over-window EMA net ΔS (shipping basis): jd6 −0.333 → **jd7-OFF +0.0085 (control) —
the continuation bar (−0.03 S/window) FAILED for the first time.** Decomposition: seg still
pays (−0.000386 d_seg = −0.0386 S/window, above bar ALONE) but pose gives back more
(+0.0471 S/window, oscillating terminal phase; live pose flat ~0.21–0.22 across jd6→jd7 —
the EMA's low pose is a real weight-averaging gain that has SATURATED, not ongoing descent).
**Best campaign checkpoint = ep1766** (tr1_jd4_cont_ep1646/checkpoints/
stage_joint_pose_finish_final.npz, EMA seg+pose 0.8149). Both jd7 arms are dominated by
their own starting point.

**Typed exit routing (tp1 policy):** plain joint continuation (jd8-as-jd7) is REFUTED by the
control arm. The measured structure routes a COMPOSED exit: (i) seg-only descent still clears
the bar → the pg1 Q3-CONSTRAINED seg lever (#889: Q3 spend cannot create pose damage, exact
kernel) is the named continuation vehicle — its own pre-registered A/B fires at this boundary
[fire-order F2 satisfied: jd7 measured seg-window pose give-back]; BLOCKER: pg1's lever is an
unapplied COMMIT_INTENT.patch (sandbox git-block), apply-to-main owed first. (ii) The stalled
pose axis (EMA 0.0101 @ ep1766, contribution 0.3177) routes to the banked pose TERMINAL
machinery (QA43 tail-targeted solve #775 built-to-ready by su2; wp1 Muon finisher 098b98e11c
available for Case-B seg finishing). Error-atlas custody now exists for BOTH jd7 endpoints
(ON + OFF) → wp1 rows 3/9/13 fully unblocked.
