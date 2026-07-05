---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "The 'benign level shift' exoneration is a 12-pair advisory CE on one surface; the jump carrier is still unattributed. I accept the restart only because the discarded seed state removes the top suspect and SC2 is a real bar — but SC1's 25-epoch alarm window is theater against a mechanism that took 92 epochs to manifest the first time."
council_assumption_adversary_verdict:
  - assumption: "PHI-surface gradient shares are the right common surface for CE-vs-bd weight calibration"
    classification: HARD-EARNED
    rationale: "bd reads model.sdf directly (zero frame-gradient, test-proven gradient-only-on-band); phi is the ONLY shared leaf. Jacobian non-uniformity phi->params is unmodeled but arbitration is pre-registered on n600 verdicts (SC2/SC3)."
  - assumption: "argparse last-wins makes trailing duplicate flags the effective config"
    classification: HARD-EARNED
    rationale: "All 4 duplicated flags are plain type=int store actions (trainer:5371/5880/5921/5965); confirmed by the run's own telemetry rows (persistence 275, band 450, bd 0.2)."
  - assumption: "the ep92 jump was a benign loss-composition level shift, not organic weight divergence"
    classification: CARGO-CULTED
    rationale: "Partially measured (live CE 0.928x EMA on the deploy surface; frozen-weight skip losses rose 58->71 with the persistence ramp = schedule-mechanical component) but the carrier is UNMEASURABLE offline. The restart is justified by risk-bounding (fresh seed discards top suspect + SC gates), NOT by a closed attribution."
council_decisions_recorded:
  - "op-routable 1: extend SC1 skip-rate alarm to EVERY epoch until tau@400 (monitoring only; no restart)"
  - "op-routable 2: amend SC3 to recompute bd share with a pre-registered de-rate rule (halve w_bd if >20%)"
  - "op-routable 3: land BUILD 6.1 (in-loop skip-escape hatch + STRICT gate) before run-3"
related_deliberation_ids: [ce_window_intervention_package_20260705, council_grand_symposium_curriculum_derivation_20260705]
---

# INDEPENDENT ADVERSARIAL REVIEW — CE-window intervention (run seedfix3_bdB_v2, 20260705T083453Z)

**Reviewer:** independent (recusal-correct — did not author the diagnosis, the trainer flag, the
calibration, or the launch). **Round 1 verdict: FINDINGS (0 CRITICAL / 3 MEDIUM / 6 LOW).
NO restart required before tau@400.** Live run READ-ONLY throughout (pgrep/ps/log reads only).
All numbers `[macOS advisory]` NON-PROMOTABLE. **Pointer 0.19110 UNMOVED** — this review is MEANS.

## 0. What was verified from PRIMARY sources (not summaries)

| claim | source | verdict |
|---|---|---|
| Spike-guard absorbing state (skips never update median) | trainer:4729–4743 (`continue` before the `recent_losses.append` at :4774) | **CONFIRMED** |
| Deadlock arithmetic | snapshot `__recent_losses` (50 entries, median **8.137**) ⟹ threshold 5×8.137=40.7 < ep92 batch losses 58–66 | **CONFIRMED** |
| ep92 forensics | old run.log: 63 skips at ep92 (first ≈ batch 13), 75/75 every epoch after; **NO event rows ep76–91** (no engage/transition/reorient); log tail ep142 still skipping | **CONFIRMED** |
| `--resume-clear-spike-guard` touches ONLY the guard window | trainer:4144–4165 (clears the `recent_losses` init path; nothing else); default path restores bit-faithfully; new sidecars persist the NEW window (:3898) | **CONFIRMED** |
| 1-batch exposure | med=None ⟹ skip only on non-finite; first accepted batch defines the median | **CONFIRMED** (grad-clip 1.0 + non-finite guard still active on batch 1) |
| Effective live config (argparse last-wins) | ps argv + trainer add_argument types + the run's own boot rows (persistence 275 / band 450 / bd 0.2 / weight 0.2 band 2px) | **CONFIRMED** (tau=400 by argparse semantics; will appear in the next sidecar `__cfg` save) |
| EMA restored on resume (SC2 baseline continuity) | snapshot `emaP__*` keys present | **CONFIRMED** |
| Seed module NOT persisted (fresh at restart) | snapshot has only `__cfg_seed_*` keys, no seed params | **CONFIRMED** |
| Calibration mixing law + window | recomputed: ratio(0.2)=**0.0905**; window endpoints **[0.1058, 0.3546]** | **EXACT** |
| Package restart-state numbers | seed w: cos((101−1)/299)→**0.748** old / (274)→**0.706** new; persistence 101/300=**0.337** / 101/275=**0.367** | **REPRODUCED** |
| Launch diff old→live | token diff = EXACTLY the Option B deltas + resume/drift/clear flags + seed-anneal 300→275; hosc/tau-shape/lr flags IDENTICAL | **CONFIRMED** |

## 1. The two seeded suspicions — resolved from source

**(a) TAU-WINDOW SHRINK (400→726 = 326 ep vs sealed 426): NOT BROKEN — no anneal endpoint is
tied to the tau stage.** Verified from source:
- `_softmax_temp_for_epoch` (trainer:1479+) anneals softmax temp over `--anneal-epochs`
  (default = `--epochs` = 1000, global) — but the live config has `--softmax-temp-start 1.0
  --softmax-temp-end 1.0` ⟹ **constant temp; `--tau-anneal-shape geometric` is INERT** (shape of
  a zero-width anneal). There is no tau-stage anneal endpoint to miss.
- The tau stage is a **form switch** (`_seg_form_for_epoch`, :1096: ce → tau_softplus with fixed
  `tau=0.3`), not a schedule with an endpoint.
- Eikonal ramp (`_scheduled_eikonal_weight`, :1211–1246) is a **STEP anchored to
  `--tau-softplus-start-epoch`** (auto-adapts to 400), cosine-eased over the 20-ep rewarmup window
  ⟹ 0.05→0.1 across ep400–420. Adapts; fine.
- Muon LR anneal spans `muon_start→epochs` (:4428–4432, config-anchored, deterministic on resume);
  Muon warm-start momentum reads the then-current AdamW m at ep726 (326 post-reset epochs of
  moment history — ample). LR cosine is epoch-indexed over 1000 (:4630–4639) — resume continues
  at the correct position.
- Stage-boundary treatment (rewarmup + `--stage-transition-reset-moments`, :4554+) keys off the
  detected seg_form change ⟹ fires at 400 automatically; the curriculum transition clears the
  spike guard (:4397) — re-treat preserved.

**(b) HOSC-β: quantified.** `_hosc_beta_for_epoch` (:1456–1476): linear over `--anneal-epochs`
default = `--epochs` = **1000 (global horizon, unchanged by the tau shift)**.
β(ep)=1+4.134·(ep−1)/999: **β(300)=2.237, β(400)=2.651 (+18.5% at the new tau onset),
β(726)=4.000, β(1000)=5.134.** The measured divergence anchor (CLAUDE.md §Capstone trainer) is
FIXED β=4 from init (tanh saturation → vanishing grad → AdamW random-walk); annealed-β on trained
weights is the sanctioned regime and 2.65 is comfortably below 4. Not a restart trigger.
**Pre-existing note (NOT an intervention delta — launch diff proves hosc flags identical):** β
crosses 4.000 EXACTLY at the Muon switch (ep726) and ends at 5.134, above the sanctioned 1→4
anneal example. Flag for the ep726 watch window; out of scope for this restart decision.

**Other clock-coupled schedules swept (all clean):** seed compose anneal cosine 1→0 completes at
ep275 (`seed_compose_weight_at_epoch`, :1042 — verified full→0, clamps after); persistence ramp
linear 0→1 completes at ep275 (`persistence_anneal_weight`, persistence_topology_loss.py:245);
both BEFORE tau@400 ⟹ collision ordering 275 → 400 → 450 → 726 → (l7 never, 1001) is coherent and
each boundary re-treats the spike guard (band engage :4526–4532; curriculum :4397; muon :4461).
Reorient every 50 fired at resume (log `resume_reorient` 0.63581 ≈ old ep51 value). Verdict/ckpt
cadence 25 unchanged.

## 2. Findings (ranked)

### MEDIUM

**M1 — SC1's alarm window is too short for the measured onset timescale; no in-loop escape
exists in the live build.** The original deadlock took **92 epochs from init** to manifest; SC1
only alarms on ≥50% skips in **ep101–125**. A recurrence at any epoch >125 re-deadlocks
PERMANENTLY and silently (skips print but nothing alarms; the closed-loop controller is
skip-blind — package §6.3, confirmed at :1277–1317 which reads verdicts only) until tau@400's
curriculum re-treat — worst case ≈ **275 wasted epochs ≈ 13 h**, a repeat of the exact incident
under review. BUILD §6.1 (auto-clear after K consecutive full-batch skips) is NOT landed.
*Breaks:* the package's implicit claim that SC1 bounds recurrence risk.
*CORRECTIVE (no restart):* SC-gate amendment — extend the skip-rate alarm to EVERY epoch until
tau@400: alarm on any epoch with >10% skips (≥8/75) OR ≥3 consecutive epochs with any skips
(`grep -o '"spike_skip", "ep": N'` per-epoch counts in the monitoring cadence). Land BUILD §6.1 +
its STRICT gate before run-3.

**M2 — bd share exits its calibrated window faster than the risk register's "certain, slow".**
Fixed w=0.2 with falling CE (recomputed from the registered mixing law): share = **15.1% at
CE=20, 22.8% at CE=12** (vs 9.05% at the measured CE=35.61). Run-1-style CE descent plausibly
crosses the 15% window edge mid-CE-window, well before tau@400 — the lever's calibration basis
erodes during exactly the window that judges it (SC2).
*Breaks:* the domain-of-validity framing of `boundary_distance_weight_calibration_v1` ("re-run
before re-tuning" has no trigger).
*CORRECTIVE (no restart):* amend SC3 (ep150–175 probe re-run) to RECOMPUTE the bd share on the
fresh EMA and pre-register the de-rate rule: if share >20%, next gated restart halves w_bd. This
keeps SC2's verdict attributable (a NEGATIVE SC2 at share ~20% would otherwise be ambiguous
between "lever useless" and "lever overweighted").

**M3 — the "benign level shift" exoneration is bounded, not closed; the confirmation test as
posed by the parent needs restating.** What IS measured: (i) deploy-surface exoneration —
witness-alone CE_live = 0.928× CE_EMA (12 pairs, advisory); (ii) the shift was sharp (window
math, valid: smooth 7× over ≥50 accepted batches drags the median, max ratio ≈1.08 ≪ 5);
(iii) frozen-weight skip losses ROSE 58→71 over ep92→142 — with zero optimizer steps this rise
can only be schedule-driven (persistence ramp 0.307→0.473 of a large raw term), proving a
schedule-mechanical COMPONENT of the elevated level; (iv) no discrete event rows ep76–91 ⟹ the
jump was caused by an ACCEPTED (clipped) step at ep92 batch ~12, gnorm~300 pre-clip ⟹ a genuine
loss-surface cliff in the aux/composed terms. What is NOT closed: which term (package honestly
tags UNMEASURABLE; per-term telemetry unbuilt). Could the guard have been CORRECTLY protecting?
Partially — but a permanent freeze is never the correct protection (the correct response to a
harmful step is rollback/LR-cut, which the guard cannot do), and the deploy surface is measured
undamaged, so clearing was right EITHER way. **Risk bound:** the restart discards the top-ranked
suspect entirely (seed module not persisted ⟹ fresh seed) — which is simultaneously mitigation
and a CONFOUND for the parent's proposed test: first post-restart batch losses are NOT
comparable to the 58–66 band (fresh seed state + bd adds ~0.2·17.72≈3.5 raw + persistence ramp
0.367 vs 0.337). *Restated confirmation test:* EITHER ~58–70 first-batch losses (shift lives in
witness-weight aux terms) OR a return to ~10–16 (implicates the discarded seed state as the
carrier — itself a valuable attribution!) is consistent with the intervention being correct; the
alarm conditions are SC1 skips and SC2 slope, not the absolute level. At review time the run was
21 min in (v0 verdict computing, 234% CPU, process healthy); first training rows land ~25 min
post-launch — check then.

### LOW

**L1 — tau-window shrink: no broken endpoint** (full derivation §1a). The 326-ep tau window is a
capacity question only; nothing schedules past Muon.
**L2 — hosc β(400)=2.651 vs β(300)=2.237** (§1b): +18.5% saturation parameter at tau onset,
comfortably below the fixed-β=4 anchor; the tau-boundary rewarmup (floor 0.1, 20 ep) + moment
reset + eikonal step all cushion the transition. Monitoring only. β=4.000 at ep726 is
pre-existing sealed behavior — noted for the Muon watch window.
**L3 — closed-loop controller resumes with stale history** including the frozen ep100 readout
(0.12129, 'ce'-tagged) polluting within-stage slope estimates. Action surface is bounded
(:1331–1364: eikonal bump UP only, capped; early-stop only after sustained DIVERGING_ERASING
with budget spent) — contained. The known skip-blindness persists (M1's corrective covers it).
**L4 — post-clear median re-seeding is 1-sample-anchored:** the first accepted batch becomes the
sole median for the next few batches; an outlier batch-1 skews the early guard either way.
Bounded by grad-clip + non-finite check; self-corrects as the window fills (~25 batches). No
action.
**L5 — the `resume_spike_guard cleared_frozen_window_len=50` row prints AFTER the v0 verdict**
(code order :4053 resume → :4060 v0 probe → :4144 guard init) — absent from the log at review
time by TIMING, not by failure; snapshot verified to contain the 50-entry window so the clear
will engage. SC1 verification must include this row once the v0 verdict completes.
**L6 — calibration power + surface caveats:** 12 pairs is under-powered as a standalone weight
verdict and the phi-surface share comparison ignores the phi→params Jacobian's spatial
non-uniformity; acceptable ONLY because it is tagged advisory and arbitration is pre-registered
on n600 verdicts (SC2/SC3). The arithmetic itself is exact (§0). The mixing law's CE side was
measured on the EMA (35.61); with the LIVE CE (33.05) ratio(0.2)=9.7% — still mid-window.

## 3. Restart-before-tau recommendation

**NONE.** No CRITICAL finding; no flag change requires a restart. The correctives are SC-gate
amendments (M1, M2) + a restated confirmation test (M3) + one pre-run-3 build obligation
(BUILD §6.1 + STRICT gate). The ~15 h runway to tau@400 is best spent letting SC1/SC2 adjudicate.

## 4. Self-reflection (Catalog #363)

- Spike-guard mechanism / arithmetic / code isolation / argparse effective config / anneal
  completions / EMA-and-seed persistence — `VERIFIED_VIA_SOURCE_INSPECTION` +
  `VERIFIED_VIA_EMPIRICAL_ANCHOR` (trainer lines, snapshot npz keys, both run.logs, ps argv).
- Calibration table numbers — `VERIFIED_VIA_EMPIRICAL_ANCHOR` (recomputed from the registered
  mixing law; raw sidecars on disk under `experiments/results/bd_calib_20260705/`; the probe's
  12 executed pairs NOT re-run — power caveat L6).
- ep92 jump carrier — `ASSUMED_AWAITING_VERIFICATION` (M3; first post-restart batch losses +
  BUILD §6.4 telemetry are the verification paths).
- β-at-tau-onset risk assessment — `INFERRED_FROM_DOMAIN_LITERATURE` (the fixed-β=4 divergence
  anchor extrapolated to the annealed-on-trained-weights regime; no direct measurement at 2.65).

**Round 1: FINDINGS (parent runs the counter). Pointer 0.19110 UNMOVED.**
