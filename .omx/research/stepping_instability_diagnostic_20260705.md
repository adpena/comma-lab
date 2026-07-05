---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "n24 reproduces the eikonal runaway but sweeps schedule-epochs 101-120 where the real run only swept 101-102; the low_lr STABLE verdict is a 60-step window, not a proof of asymptotic stability — the n600 relaunch must keep the SC1' every-epoch skip alarm armed."
council_assumption_adversary_verdict:
  - assumption: "the resume path is bit-faithful for self-orient runs"
    classification: CARGO-CULTED
    rationale: "dir feats are rebuilt from the ep100 EMA argmax while the training-time feats were the ep51-reorient vintage (5.6% argmax flip per bd_calib); the code itself calls this a 'fidelity envelope'. Measured consequence: restored-state total 94 vs the continuous run's accepted 6-8.5 on the same pairs."
  - assumption: "restored Adam moments are dangerous (stale)"
    classification: CARGO-CULTED
    rationale: "MEASURED inverted: fresh moments explode 25.3x vs 6.7x with restored moments — the restored second-moment preconditioner DAMPS the runaway."
council_decisions_recorded:
  - "op-routable #1 (GO-gated): n600 relaunch = v3 argv + resume LR re-warmup (treat resume as a stage boundary: floor 0.1, 20-epoch ramp) OR --lr 1e-4 flat; moments RESTORED (not reset)"
  - "op-routable #2 (GO-gated): fix v0-verdict-at-resume to apply the annealed hosc_beta/softmax_temp before rendering (kills the phantom d_pose ~15-21 resume telemetry)"
  - "op-routable #3 (GO-gated): persist the island-seed module + seed_opt state in the resume sidecar (#304 checkpoint-completeness class)"
related_deliberation_ids: [adversarial_review_ce_window_intervention_20260705]
---

# Stepping-instability diagnostic — the ep100-resume / ep92-organic runaway is an EIKONAL (SDF-slope) divergence gated by step size

**Axis discipline: every probe number below is `[n24 advisory — mechanism probe, NOT n600 evidence]`.
Frozen-scorer CPU-torch verdicts inside the arms are `[macOS-CPU advisory] NON-PROMOTABLE`.
Pointer 0.19110 UNMOVED (this whole memo is means/apparatus).**

## 1. The incident (3 strikes, n600, all runs preserved)

| run | config | observed |
|---|---|---|
| v1 `20260705T015247Z` | original | trained ep0-91 (accepted 6-8.5); ep92 first batch already ~58 (gnorm ~300) → guard median frozen at ~8.1 → 100% skip ep92-100+ (deadlock) |
| v2 `20260705T083453Z` | resume ep100 + bd 0.2 + staggers (fresh seed @0.70 compose) | ep101: 63 accepted @ ~40, 12 skipped @ ~199; ep102+: 100% skip @ ~199 |
| v3 `20260705T095728Z` | same + `--seed-anneal-epochs 101` (seed compose ≡ 0) | ep101: 75/75 accepted; ep102+: 100% skip @ 238→250 (rise on frozen weights) |

Snapshot: `experiments/results/bd_calib_20260705/snap/resume_state_ep100.npz` (weights = ep91-era;
ep92-100 had ZERO accepted steps, so weights AND Adam moments are the self-consistent ep91 pair).

## 2. The instrument (landed first, commit `1458b7b4d`)

`#304 item 4` per-term loss telemetry in `experiments/train_levelset_witness_realized_through_R_mlx.py`
(+ `terms_out` hook in `experiments/train_witness_realized_through_R_mlx.py::make_loss_fn`):
`loss_terms` JSON row with the stable 16-key schema {seg, pose, eikonal, length, boundary_distance,
lane_edge, margin_saliency, subpix, chroma_boundary, island_amplify, persistence, rankfloor,
code_spectral, thin_lane, margin_field_head, code_nuclear} + self-check (`sum_minus_total` at fp
tolerance, measured ~1e-5). Cadence: `TAC_LOSS_TERM_PROBE=1` per accum-chunk; `--loss-term-log-every N`;
default = per-epoch summary; `-1` = off. It is a NO-GRAD RECOMPUTE — **proven bitwise-identical
trajectory off-vs-on** (n1 CPU 3-epoch A/B: 90/90 resume-state keys byte-equal;
`experiments/results/stepping_instab_diag_20260705/lt_identity_{off,on}`). 15 tests
(`src/tac/tests/test_loss_term_telemetry.py`).

## 3. The probe (commit `0fdacddf9`) and the per-arm matrix

`experiments/probe_resume_stepping_instability.py`: slices the n600 sidecar per-pair to pairs 0-23
(gt_n24 == pairs 0..23 of gt_n600, verified by exact pose-row match), resumes through the trainer's
OWN resume path (subprocess on the real entry point), spike guard DISARMED (`--spike-factor 1e9`) so
the evolution is observed, per-batch per-term telemetry on, 20 epochs = 60 optimizer steps past
resume (the n600 blowups took ~63-75 steps). Schedule fidelity kept via `--anneal-epochs 1000`
(lr/beta/temp per-epoch values match the n600 runs at ep101+). `--pose-carrier-s-t 0.044` pinned =
the n600 runs' own fit (which used exactly pairs 0-23).

| arm | separates | verdict | trough→peak (ratio) | runaway term(s) |
|---|---|---|---|---|
| `baseline_v3` (bd 0.2, moments restored, lr ~9.1e-4) | reproduce? | **EXPLODES** | 19.1@ep111 → 128.9 (**6.7×**, still climbing) | **eikonal** |
| `fresh_moments` (opt state dropped) | stale moments? | **EXPLODES (worse)** | 24.1@ep107 → 610.7 (**25.3×**) | pose + **eikonal** |
| `low_lr` (lr ×0.1) | step size? | **STABLE** | 64.3@ep120 → 65.5 (**1.02×**) | none |
| `no_bd` (bd 0) | the new bd term? | **EXPLODES** | 13.5@ep111 → 110.8 (**8.2×**) | **eikonal** |
| `v1_pure` (v1 config exactly: no bd, seed anneal 300) | any intervention at all? | **EXPLODES** | 11.1@ep110 → 102.7 (**9.3×**) | **eikonal** |

Shape (baseline trace, epoch-end totals): 94 → 62 (ep103) → 20 trough (ep111) → then eikonal alone
runs away 6.7 → 24.9 → 34.0 → 49.2 → 101.8 → 114.7 (ep112-120) while seg stays flat ~8-9 and every
other term is flat/declining. Same descend-then-runaway signature in no_bd / v1_pure / fresh_moments.

## 4. Mechanism verdict

**The instability is an optimizer divergence along the SDF-slope direction, registered by the
eikonal term `eik_w·mean((|∇φ|−1)²)`, and it is STEP-SIZE-GATED.** At the resumed state, cosine lr
~9.1e-4 exceeds the stability threshold of the basin's sharpest direction; Adam steps oscillate →
diverge along |∇φ| (raw eikonal at the baseline's ep120 ≈ 115/0.0555 ≈ 2,070 = |∇φ| far off the
1-Lipschitz target), the quadratic penalty explodes, total crosses 5×median, and in the guarded
n600 runs the spike guard — CORRECT all three times — locks out (median never updates on skips).

Hypotheses falsified (implementation-level, per-arm measured):
- **Stale restored Adam moments: FALSIFIED — inverted.** Fresh moments explode 25.3× vs 6.7×; the
  restored second-moment preconditioner DAMPS the runaway. Do NOT add reset-moments-on-resume.
- **The bd term: FALSIFIED as cause** (no_bd explodes 8.2×; bd's own contribution is flat 3.5→3.5,
  exactly the calibrated 0.2×17.7 of `bd_calib_ep100_w02`).
- **Seed-compose interventions: FALSIFIED as primary** (v1_pure = the untouched v1 config explodes
  9.3×) — confirming FEED-05n's falsification of FEED-05l's seed-primary story.
- **Step size: CONFIRMED as the gate** (lr ×0.1 stable at 1.02× over the same 60 steps; Contrarian
  caveat: a 60-step window, not asymptotic proof).

**Retro-implication for the organic ep92 jump (v1):** same mechanism, no resume needed — v1_pure
explodes from the ep100 state organically. Through ep50-91 the landscape sharpens (hosc-β anneal
1.0→~1.38, EMA/live drift, seg CE descending into a sharper valley: v1 ep_loss already rising
431→641 between ep50 and ep75) while the cosine lr has barely decayed (~9.2e-4 at ep92) — the
stability threshold crossed the schedule organically. The v3 "rise on frozen weights" (238→250 over
ep102-121) is the epoch-indexed VALUE inflation (persistence warmup w=ep/275 + eik-weight/β anneal)
on top of the frozen post-runaway state — consistent, no second mechanism needed.

## 5. Side findings (each measured, each with a GO-gated fix)

1. **Resume applies NO re-treatment.** `last_boundary_epoch` is None at resume → the existing
   `--stage-transition-rewarmup-*` machinery (already on the argv!) never fires; the run takes
   full-lr steps into a resumed (and lever-drifted) landscape. The reorient-every-50 boundary
   (ep101) is likewise not in `_stage_boundary_now`. This is the missing-rewarmup half of the bug.
2. **v0-verdict-at-resume phantom d_pose.** The resume telemetry printed d_pose **20.754** (n600)
   / 15.537 (n24) while the pending-verdict recompute on the SAME ep100 weights printed **0.138**:
   the v0 verdict runs BEFORE the epoch loop applies the annealed `hosc_beta` (renders at
   construction β=1.0 instead of β(100)≈1.41). Training is unaffected (β is set per-epoch before
   stepping); the telemetry is misleading (looks like pose collapse at every resume of a
   β-annealed run). Fix: apply `_hosc_beta_for_epoch`/`_softmax_temp_for_epoch` for the resumed
   epoch before `realized_verdict()` at resume.
3. **The island seed is still not persisted** (sidecar has no seed keys) — every resume of a
   seed-composed run loses the trained seed (v2's ~40 level == bd_calib's witness-alone CE 35.6
   measured on the same snapshot). Not the runaway mechanism, but a real resume-fidelity gap
   (#304 checkpoint-completeness class, FEED-05l).
4. **Self-orient feats vintage.** Restored-state loss is 94 vs the continuous run's ~7 on the same
   pairs partly because resume rebuilds dir feats from the ep100 EMA argmax while training ran on
   the ep51-reorient vintage (5.6% argmax flip, bd_calib). A continuous run hits the same feats
   swap at ep101's scheduled reorient — with no re-treat (see fix #1's boundary list).

## 6. RECOMMENDED FIX (concrete, GO-gated — nothing launched)

**Primary (unblocks the n600 relaunch):** resume-as-stage-boundary LR re-warmup. Code delta in
`train_levelset_witness_realized_through_R_mlx.py`: on `--resume-from` (at least when
`--resume-allow-lever-drift` / any lever divergence / a reorient-rebuild occurred), set
`last_boundary_epoch = start_epoch` so the EXISTING `_stage_rewarmup_factor` path engages (v3 argv
already carries floor 0.1 / 20 epochs / cosine → post-resume lr starts at ~9.1e-5 and ramps).
Measured support: the low_lr arm (exactly the floor's 0.1×) is the only stable arm. Keep restored
moments (do NOT reset — measured worse). Zero-code-delta alternative for the next launch:
same v3 argv + `--lr 1e-4 --lr-end 1e-5` (flat 0.1×), accepting the changed long-run schedule.
Optional belt: add the scheduled reorient epochs to `_stage_boundary_now` so reorient boundaries
re-treat (rewarmup + guard clear) like every other stage transition.

**Secondary (same landing or next):** fix #2 (v0-verdict β), #3 (persist seed + seed_opt in the
sidecar), and consider raising `--eikonal-weight` only AFTER the step-size fix is measured (the
eikonal term is the canary here, not the underdog — at 0.055 it lost to divergence, but the stable
arm holds |∇φ| without a weight change).

**n600 confirmation plan (operator GO):** relaunch v3 argv + the primary fix; pre-registered gates:
ep101-125 skip-rate <10%/epoch (SC1' stays armed every epoch), per-epoch `loss_terms` row shows
eikonal ≤ its ep101 restored level and non-inflating trough-to-current by ep150; cost ≈ the same
~1h/25-epoch cadence as v3 (no new compute class). If the relaunch still deadlocks, the n24→n600
transfer is falsified and the next diagnostic is the same probe at n600 (~30-60 min, plan in
FEED-05o).

## 7. Retro check (deliverable 3): v1 log near ep91-92

One grep, answered: NO event fired at ep92 — no reorient (ep51 was the last; next ep101), no
curriculum transition, no lever engage, no lr/schedule row between the ep75 checkpoint block and
the first ep92 `spike_skip`. The jump is organic, as the v1_pure arm reproduces.

## Artifacts

- Probe + report: `experiments/probe_resume_stepping_instability.py` (commit `0fdacddf9`);
  `experiments/results/stepping_instab_diag_20260705/{step_instability_probe_report.json, arm_*.log,
  arm_*.argv.json, resume_state_ep100_n24*.npz}` (gitignored results; durable numbers above).
- Telemetry: commit `1458b7b4d` (+ identity A/B logs `lt_identity_{off,on}`).
- Forensic set (READ-ONLY, untouched): runs `015247Z / 083453Z / 095728Z` + the sha256'd snapshot.
