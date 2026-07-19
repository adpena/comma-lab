# Adversarial gauntlet verdicts — duty A/B ep725-fork tickets (#563, 2026-07-19)

Reviewer = FRESH EYES (did not author the revisions). Charter:
`.omx/tmp/codex_prompts/duty_ab_adversarial_gauntlet_20260719.md`. Craft basis:
`docs/operating_manual_craft_handoff.md` (§5 label MEASURED/DERIVED; fixes are unreviewed
new code; point-fix ≠ class-fix) + `.omx/research/confound_hunt_synthesis_20260705.md`
(3-layer immune system). **Pointer `0.1910828242 [contest-CPU]` UNMOVED — everything here is
MEANS.** No launch occurred. Reviewed against the MAIN-checkout committed artifacts at HEAD
`91e4bc721b` (tickets composed at `679e78ab0352`; every hash below RE-REPRODUCED at current
MAIN HEAD — no drift across the intervening #568/#570 merges). The reviewed DSL module is
byte-identical in this worktree (sha `c2723c80…`).

## Per-ticket verdict table (raw)

| ticket | lever | verdict | cleared full_dsl_compile_hash (off / on) |
|---|---|---|---|
| 02 | `HorizonWeightedMargin` | **CLEARED** (2 named verdict-time conditions) | `c49087ce7077c9abe3f8c06dafe85a909b6867da3319ac5c33a4d2ad24fbcb1d` / `8ac7b6cd816f4d21d42c477fa9abd0c21904425cbf14dccd5ac627c150565137` |
| 03 | `StepNativeActivation` | **CLEARED** (same 2 conditions + estimand-scope note) | `8c0e962064e4587741f6b24851b0d9f7ebcfd0c38553a73604260ca05dfdaad8` / `028bd07d738ab4f21a68d8c38abcbcec9f611228507e2e13a2a4b78a6aef4b56` |
| 04 | `#497 curvelet matched-bytes` | **REFUSED-for-launch** (3 named defects; wrapper repairs sound) | fresh-arms `be96e749…`/`7ed49820…` (composer pure-compile; not a fire receipt) |

MAIN may proceed to launch-prep (registration → dry-start → operator-GO) for **02 and 03
ONLY**, in that P0-SUPREME order. **04 does NOT enter the launch list.**

## What I verified (each axis MEASURED at MAIN HEAD 91e4bc721b)

**(a) dsl_compile_hash reproduction.** Ran `compile_v9c3_duty_ab_config(t,a)` →
`lwr.compile_dsl_document_for_config(...)['dsl_compile_hash']` for all 4 arms. All 4 full
hashes AND all 4 `typed_config_hash`es reproduce byte-exact vs the verdict cards. Schedule-
provenance gate re-run against the live trainer parser: rc0 / 0 violations all 4 (hwm off=6
verdicts, on=7; step off=6, on=6 — matches cards).

**(b) One-lever delta.** Diffed the full resolved argv per pair:
- step OFF→ON = exactly `{--hosc-beta-end: 4.0→8.0}` (1 flag).
- hwm OFF→ON = the 8 `--seg-horizon-margin-*`/`--seg-horizon-row-*` flags, ALL owned by the
  single `HorizonWeightedMargin` Lever (spec appends exactly one lever in the on-branch). One
  Lever = one semantic difference. `--out-dir` is a `<OUT_DIR>` placeholder in the argv, so
  custody dirs never pollute the diff. **hwm_off and step_off argv are byte-identical
  (`resolved_argv_sha256 bb2525ff…` both)** — the OFF twins are the same physics (custody-only
  split); confirmed, not asserted.

**(c) Threshold / power.** Re-derived sigma from the donor run.log (read-only) verdict series
ep750-1075 (14 pts). The plain second-difference estimator reproduces EXACTLY
(`sqrt(mean(d2²)/6) = 1.695337e-05`). h95 arithmetic is internally consistent
(`z·√(2/K)·σ`; z=1.959963985): h95(4)=2.600e-5, h95(10)=1.645e-5. Power for HWM is real —
the adverse-prior ceiling (1.2e-4–2.4e-4) is 4.7–9.4× h95(4). Dual-window sign-agreement
requirement is a sound single-window-fluke guard. **Two findings (below).**

**(d) Resume / fork correctness (#518 + resume-events law).** Events ARE re-anchored to the
resume epoch + geometry, exactly as `warm_start_resume_must_adapt_events_to_resume_epoch_and_geometry_20260718`
demands: **Muon OMITTED entirely** (the donor's absolute-ep726 Muon is the MEASURED confound
engine — removed, not left); **HWM start moved 726→753** (post-rewarmup, so the derived-live
boundary scan sees re-conditioned losses); ResumeLRWarmup 27ep (DERIVED
`ceil(2/(1−.999)/75)`) with fresh AdamW; ForkEmaClearance; #517 pre-v0 tau/beta/seg_form
positioning. Surgical loss terms left at start_epoch=700 is CORRECT (700<726 → active from
fork start in BOTH arms → reproduces the trunk's ep700-725 loss physics; re-anchoring them
would corrupt it). Bank custody re-verified byte-exact on disk (460,448 B, sha
`b0a431e9…` matches).

**(e) Confound sweep.** Bench cadence arm-identical (`--ckpt-every 25`,
`--component-wallclock-probe-every 1` both arms) → no `--ckpt-every`-drag contamination.
Spike-guard re-treat on ON-arm lever engagement (`recent_losses.clear()` at 753) is DISCLOSED
(AF/admissibility) and is the SAME discipline as margin-satisfice/temporal-screw; the immune
system (`ep_loss>0`, no `spike_deadlock`/`term_domination`/`gnorm_hijack` alarms in-window)
catches a differential freeze → inadmissible verdict. EMA-shadow lag: windows start ≥49ep past
resume (~3675 EMA updates ≫ 333 timescale) → shadow matured; `ema_warmup=false`-in-window is a
precondition. Verdict at **full n600** confirmed (`--verdict-pairs 0` → `list(range(P))`, line
8400). Positive-control sentinels present (resume_lr_rewarmup row, baseline_v0 positioned beta,
lever_engage fired row, HWM boundary receipt resolved_weight>0).

**(f) NO-FAKE.** Traced both levers to the actual loss, not markers:
- HWM: `_hz_hinge = mx.maximum(hz_target − _signed, 0)·_hz_mask`; `hz_term = mean`;
  **`L = L + hz_w·hz_term`** (train_levelset…:~7326) on the real through-R witness margin,
  stratified to the horizon band. OFF arm `hz_w=0.0` → term skipped → byte-identical. Real.
- step: **`model.hosc_beta = _beta`** (…:~13282) from `_hosc_beta_for_epoch`, consumed by
  `mx.tanh(self.hosc_beta·mx.sin(ω·u))` (…:2391). The 4.0→8.0 endpoint genuinely sharpens the
  activation. Real.

**Blocker scope — `V9_432_HOSC_BETA_END_LAWREF_RECOMPUTE_DEFECT` is HONESTLY scoped.** It
refuses ONLY the 432/taper (ticket 01) family — reproduced live: both 432 arms raise
`V9ProvenanceGateError … 'hosc_beta_end': 10.0 != 3.177`. The v9c3 duty programs (02/03)
full-compiled cleanly (I reproduced their hashes), so the defect does NOT touch the gauntlet
tickets. Ticket 01 remains correctly CANNOT-RESOLVE/re-scoped, out of this launch batch.

## Findings

**F1 (MEDIUM, verdict-time; applies to 02 + 03) — within-run σ used as a between-run paired-
noise proxy; the caveat is incomplete.** σ (h95's basis) is estimated from ONE trajectory's
*temporal* second-differences, but the verdict metric `Δ(ep)=d_seg_ON−d_seg_OFF` is a
*between-process* paired difference, and the arms run **sequentially in separate processes**
where MLX-GPU cross-process non-bit-identity is a MEASURED effect (D6). The memo's caveat only
argues pairing SHRINKS noise (shared common-mode) — it does not account for cross-process drift
BREAKING that sharing, which could make the true paired noise LARGER than `√2·σ_withinrun`,
i.e. h95 anti-conservative → risk of a false FIRED verdict. *This does not invalidate the
config* (the config is confound-free and the memo already labels the thresholds PROVISIONAL).
**Fix (pre-registerable, no post-hoc shopping):** admissibility precondition — measure the
between-run paired scatter on a **pre-treatment NULL window** and require it below h95 before
any FIRED verdict. HWM HAS such a window (verdict pts ep726/750 are before the 753 boundary →
pure cross-process noise, zero treatment). **step has NO null window** (beta differs from
resume 726) → for step, name the OFF-arm's own temporal residual as the σ FLOOR and treat h95
as a lower bound until the paired residual is measured. Verdict-time, does not block launch.

**F2 (LOW, non-blocking; 02 + 03) — robust-σ median convention.** The recorded
`sigma_per_point_robust = 1.876308e-5` uses `median|d2| = 3.10e-5` (the upper of the two
central order-statistics of 12), not the true median `3.05e-5` (mean of the two central); the
canonical estimator gives `1.846e-5`. Effect: +1.6%, which INFLATES h95 (harder to declare
FIRED) → **conservative direction**, flips no verdict, keeps the 4.7–9.4× HWM power margin.
Recorded for honesty; non-blocking. Fix if re-run: `statistics.median`.

Neither F1 nor F2 is a hard confound in the config; both are threshold-provenance refinements
downstream of the (confound-free) launch. 02/03 CLEAR with F1 folded into admissibility.

**Ticket 03 estimand-scope note (verdict_scope, already disclosed AF4/AF5):** the realized
contrast is beta-END **4.0→8.0** on THIS mod32 vehicle (NOT the mod19 "3.177→8.0" custody
surface), and the treatment repositions beta(726) to 6.0801 vs control 3.1772 — a +2.90
step-shock at resume that is PART of the "sharpen-a-trained-trunk" estimand, explicitly not
"train-under-8.0-from-ep0". No measured prior effect size → the instrument is honestly
NEUTRAL-capable (correct verdict when the true effect < h95, not a false verdict).

## Ticket 04 — REFUSED-for-launch (named defects; fixes already in its OWED list)

The wrapper fail-closed repairs are SOUND (sha `5c04b3a65aac…` verified on disk; absent-run-dir
→ not-quiescent, liveness-inspection-failure → fail-closed, `--skip-c2-gate` requires
`--operator-go`, arm mutual-exclusion). But the A/B is not fire-ready:
1. **Equal-byte completion chain is NOT gate-ENFORCED (confound).** #497 is a MATCHED-BYTES
   comparison — the d_seg/d_pose verdict is only valid at fixed rate. The equal-byte finalize
   chain is "documented but still not REFUSED-on-skip by any gate"; a skipped match silently
   confounds the rate axis. **Fix:** land the finalize equal-byte REFUSE gate before fire.
2. **No green governed dry-start receipt; arms PREPARED_NOT_FIRED.** No runnability proof.
3. **Treatment lever term-share engagement unproven beyond flag-presence** → inert-lever risk
   (a fake FIRED-NEUTRAL). **Fix:** lever_engage term-share telemetry beyond `front_end`.
verdict_scope: the contrast is the COMPOSITE curvelet lever (basis+native-orient+AA bundled),
never "curvelet basis alone" — recorded, correct. 04 is a different (fresh 3000-ep) vehicle,
correctly NOT #518-bound, and not blocked by V9_432; it simply owes 3 fire-gates.

## Standing pre-fire owed items for the CLEARED 02/03 (CONTAINMENT, launcher-enforced)

`--config v9c3_duty_*` launcher registration (3-line c2 pattern) · hash-matched GREEN
`--dry-start` receipt per arm (proves warm-fork weight-shape load via `resume_ok`; the
`V9C3_COMPOSED_BENCH_NOT_MEASURED` blocker) · bank byte-custody re-verify at fire · operator GO
per arm. These are execution gates the launcher fail-closes on — not confounds — and do not
affect the CLEAR.

## STORES CONSULTED

The charter; the revision memo `duty_ticket_revision_ep725_fork_20260719_claude.md`; all three
`revision_claude_20260719/` packages (compiled_pair.json ×2 + adjudication + verdict_cards);
`src/tac/witness_dsl/spec_v9c3_duty_ab_20260719.py`; `tools/revise_duty_tickets_ep725_fork_20260719.py`;
`tools/fire_curvelet_matched_bytes_ab_p0_497.py`; the live trainer
`experiments/train_levelset_witness_realized_through_R_mlx.py` (loss terms, `_hosc_beta_for_epoch`,
HWM resolve/receipt, verdict-pairs); the sacred run dir (read-only run.log verdict series,
levelset_best.json); the bank checkpoint (sha re-verified); memories
`warm_start_resume_must_adapt_events_to_resume_epoch_and_geometry_20260718`,
`vehicle_naming_v9c_warm_lineage_v10_reserved_capstone_20260718`; `CLAUDE.md`;
`docs/operating_manual_craft_handoff.md`; `.omx/research/confound_hunt_synthesis_20260705.md`.

## Self-review (rounds used: 2 of 5)

R1: re-derived every hash/gate at CURRENT main HEAD (not the composer's) to catch merge drift —
none found. Traced both levers to the real `L = L + …` / `model.hosc_beta` sites (not argparse)
to kill the fake-marker hypothesis. R2: hunted the noise-basis confound (within-run σ vs
between-run metric under MLX D6) — real, named F1 with a pre-registerable null-window fix, not a
config-invalidator. Confirmed the V9_432 blocker refuses only 432 by live reproduction. Clean on
the checks run.

## Scope clarification (Stop-hook repair, 2026-07-19)

The ticket-04 REFUSED-for-launch verdict is **defect-based, not magnitude-based**
(verdict_scope: instance — launch-readiness of THIS ticket revision only, not the curvelet
family or the #497 basis question). The three named defects are structural launch-safety
gaps (equal-byte chain not gate-enforced → rate-axis confound risk; no green dry receipt;
treatment engagement unproven), each with a named fix already in the ticket's OWED list.
Relative significance of the underlying lever is UNCHANGED and large: the #497 curvelet A/B
targets the measured 3.2× along-tangent frequency deficit feeding Lane d_seg, and Lane flips
are ~19% of total d_seg flip mass at the current operating point — this refusal defers the
measurement until its confound gaps close; it does not downgrade the lever.
