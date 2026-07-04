# THROUGHPUT/MEMORY BUILDS — adversarial review round (pre-launch runway gate)

**2026-07-04. Role: ADVERSARY over the landed builds gating the fresh seeded launch (commits
dfb847f3c / 5d7751adf / 51864adb1 / 3795339fe / de86ec060 / f4150fde4). $0, READ-ONLY on the
trainer/tools; #205 (pid 29129) untouched; every finding carries a measured or mechanically-verified
failure scenario. Pointer contest-CPU 0.19110 UNMOVED — everything here is MEANS.**

## FINAL VERDICT: **PROCEED** (0 CRITICAL · 4 MED · 6 LOW)

No finding blocks the launch runway: the composed run-1 argv reproduces end-to-end, every safety
gate fires correctly under attack (bank-6 extras → rc=4 REFUSE; invented flag → rc=2; reconcile
refuses to invent ×3; band conversion correct at the boundary; OFF paths byte-identical), and all
cited test suites pass on this machine (22 CL + 76 memory + 7 cograd incl. the EXECUTED 106s
real-path proof). The 4 MEDs are claim-accuracy / model-anchor / dead-telemetry issues whose
worst-case readings still sit far inside the 108.8 GiB envelope — log-and-fix, not launch blockers.

---

## Surface 1 — M2 decide-on-previous + pending-shadow resume (dfb847f3c): **FINDINGS (1 MED, 2 LOW), core CLEAN**

**Verified clean under attack (mechanics):**
- Stale/missing-row paths: first eval of a run → `_cl_decide` returns False on empty rows
  (trainer:3502-3510); CL OFF→ON across resume → `_cl_restore_from_cfg` None → fresh state, no
  reconcile (`__cl_pend_epoch` absent), deterministic-forward (trainer:3928-3937). Sync path keeps
  current-epoch-row semantics (trainer:4570-4571). Skip-throttle can never fire when ON
  (join-before-schedule; structural test `test_async_decision_path_has_no_join_after_schedule`).
- Pending invariant: set under `_verdict_lock` at schedule (trainer:3441-3446), cleared under lock
  on row-land and worker-fail; `_cl_sidecar_snapshot` reads (state, verdicts, pending) under ONE
  lock (trainer:3485-3494) — the row-lost-between-two-reads race is closed.
- Reconcile is synchronous-before-first-decision: it runs in the resume block (trainer:3938-3969),
  before the training loop and before any `_cl_decide`.
- OFF path byte-identity: with `_cl_on` False the async eval branch is exactly the original
  `elif ep == args.epochs: _join_async_verdict()` order (trainer:4530-4533);
  `closed_loop_state=None` → zero sidecar keys (trainer:3607-3612). Source-guarded by
  `test_off_async_path_keeps_original_final_epoch_join_order` + `test_off_writes_zero_sidecar_keys...`.
  22/22 `experiments/test_closed_loop_control.py` re-run green this round.

**M2-F1 (MED) — the "bit-identical resume ⟹ post-resume decisions == continuous" claim does NOT
hold for the actual run-1 config (self-orient).** The continuous run's worker scores
`snap["dir"] = dir_feats_per_pair[pi].copy()` captured at schedule time — dir built at the LAST
reorient boundary (`--reorient-every 50`) from the shadow AT that boundary (trainer:3283, 3200-3221).
The resume reconcile rebuilds `dir` from the shadow AT THE CHECKPOINT via `recompute_self_orient`
(trainer:3776-3781) and feeds THAT to `_verdict_from_snapshot` (trainer:3959-3960). Checkpoints are
every 25 ep, reorient every 50 → half of all resume points sit mid-reorient-cycle with a shadow that
has moved since the boundary ⟹ the reconciled d_seg row is NOT bit-identical to the continuous
worker's, and post-resume classifications can differ near a slope-sign boundary. The code comment
honestly scopes this ("inherits that same fidelity envelope", trainer:3946-3951) and the class is
pre-existing (the training forward has the same resume contract), but the commit message + FEED-04o
state "bit-identical → post-resume decisions == continuous" UNCONDITIONALLY, and the 4-cut resume
test (`test_resume_reconcile_pending_equals_continuous`, test:425-463) proves only the persisted-
input round-trip — it substitutes the harness d_seg for `_verdict_from_snapshot`, never exercising
the self-orient dir rebuild. Failure scenario: crash/resume mid-flight in the tau stage → reconciled
row's d_seg differs in the last few digits → a DIVERGING_ERASING vs VOLATILE classification flip →
different bump/stop epoch vs continuous. Bounded (bounded actions, min-sustained 3), deterministic
going forward, but the claim needs the self-orient scope note. Fix: one sentence in the DAG/claims,
or persist `snap["dir"]` into the pending sidecar (P×in_feat fp32 ≈ 88×600×4·HW… large — the honest
cheap fix is the scope note).

**M2-F2 (LOW) — failed-verdict resume can resurrect a row the continuous run never had.** The
worker-fail clear (trainer:3462-3468) only helps sidecars written AFTER the failure. A sidecar
written while the verdict was in flight, followed by a worker failure, then a resume ⟹ the
reconcile recomputes and APPENDS the row (trainer:3952-3961) — the continuous run has NO row for
that epoch (comment at trainer:3439-3440 claims "continuous == resumed on the failure path too";
only the post-failure-sidecar half is true). Edge-of-edge (needs checkpoint-then-fail-then-crash in
one window); deterministic failures re-raise on resume anyway. Note-and-carry.

**M2-F3 (LOW) — the "~2× run wall (22h→44h)" benefit claim is inflated ~1.4×.** The mine's own
measurements (n205_memory_behavior_mine §2): 19 verdict windows over 28.29h ⟹ period ≈ 5,360s;
505 ep in 28.29h ⟹ 25-ep train window ≈ 5,042s; verdict wall mean 2,189s = **43% of the window,
not ≈100%**. Pre-M2 sync-join stall ⟹ ≈ +43% run wall, not ~2×. The reorder is still strictly
correct and valuable; the number in the commit/FEED is wrong. Same root as WF-F1 below.

## Surface 2 — l7 guard relax (5d7751adf): **CLEAN (1 LOW, observability-only)**

All trainer consumers of `l7_start_epoch` honor `> epochs` as "never": `_seg_form_for_epoch`
`ep < l7_start` (trainer:994; ep ≤ 1000 < 1001 ∀ep); C2 event-guard `>= epochs → never fire`
(trainer:1097-1098); resume cap-fallback `start_epoch > _lc` unreachable at 1001 (trainer:3917);
muon-placement check is WARN-only (trainer:5687-5695, fires benignly 726 < 1001); ordering guard
admits it (trainer:5604-5613); checkpoint stage naming is seg_form-derived (never reaches l7).
`witness_autoconfig` emits 1001 with the honest-dependency note. Dashboards:
`dashboard_trajectory_model._stage_segments` drops the out-of-range boundary correctly
(`0 < 1001 < 1000` False; tools/dashboard_trajectory_model.py:106-110).

**L7-F1 (LOW):** the legacy `render_levelset_dashboard._schedule_from_flags` naive chain
(tools/render_levelset_dashboard.py:305-307) emits `stages` = tau [300,1001) (end > epochs,
overlapping Muon [726,1000)) + an INVERTED l7 [1001,726) segment for the fresh config — the exact
naive-chain class `_stage_segments` was built to fix, still live on this second surface.
Observability-only (client strip may misrender); non-blocking.

## Surface 3 — #294 waterfill (51864adb1): **FINDING (1 MED), mechanics CLEAN**

**Verified clean:** imports the REAL preflight (`import witness_memory_preflight as wmp`,
tools/memory_waterfill_config.py:48; projection at tool:282-285 is the imported
`project_peak_rss_gib`, not a fork). Realized-spike rebase applied: `adjusted = net − preflight
transient + spike` = base + spike (tool:289-290); at vb=32 spike = MEASURED 12.3, floored below the
anchor, +0.11/pair labeled [modeled] above it (tool:87-95). Micro-batch honestly UNMEASURED-excluded
(B pinned 1; #261 points contended + wrong scale; tool:117-144). **EXECUTED this round:** BEST
reproduces exactly — `(B=1, vb=64) peak 68.65 / adj 77.43 ≤ 108.8, ×1.183 [modeled]`; internal
arithmetic checks out (wall(64)=1219.5 < 2062 ⟹ exposed 0; mult = 2439/2062 = 1.1828; the vb≥38
"fully hides" threshold ⟹ first grid point 64; tie-break picks 64 over 128/256 by adjusted peak).

**WF-F1 (MED) — `MEASURED_TRAIN_WINDOW_S = 2062` is NOT the train window; it is the verdict wall's
lower bound, and the mine's own duty-cycle measurement falsifies the anchor.** Measured
(mine §2, recomputed this round): 505 ep / 28.29 h ⟹ 201.7 s/ep ⟹ 25-ep train window ≈ **5,042s**
(cross-check: 19 windows / 28.29h ⟹ period 5,360s; duty 43-47% ✓). With the CORRECT window,
`exposed(vb=32) = max(0, 2439 − 5042) = 0` — the verdict wall is ALREADY fully hidden at vb=32
under M2, and `×(vb=64) = 1.000` even granting the ∝1/vb approximation. The ×1.183 gain and
"the operator's 32→64 ask emerges from arithmetic" (FEED-04r) are artifacts of anchoring the train
window at the wall's minimum (tool:57-59 cites "verdict wall 2062–2439s" and then reuses 2062 AS
the train window). Secondary: the ∝1/vb wall model itself is shape-suspect (the wall is 600 frozen
CPU-scorer forwards — compute-bound work that does not halve when the chunk count halves), but it
is labeled [modeled] and, with the corrected window, moot. **Consequences:** vb=64 remains SAFE
(adj 77.43, and even a doubled concurrency-climb keeps it < 90 ≪ 108.8) and can never be slower
than vb=32 — so the LAUNCH config stands — but the modeled ×1.183 must not be cited as expected
speedup, and the solver should re-anchor `train_window_s` to measured s/ep × eval_every (then BEST
flips to a lower-spike vb on the tie-break). Caveat the other way: the fresh run (mod-19, paint)
may train faster per epoch; if its 25-ep window drops below ~2,400s, vb=64 starts paying — measure
in the first eval windows via the now-default TAC_MEM_PROBE + verdict_async_done rows.

## Surface 4 — #294 bands (de86ec060): **CLEAN at the band boundary; FINDING (1 MED) on the sister admission path**

**Verified clean:** units conversion applied EXACTLY ONCE on the band path
(`actual = job.current_rss_gib × TRACKED_RSS_UNITS_TO_GIB`, tools/system_memory_governor.py:1185;
no other callsite — grep shows 2 hits total, the constant + this line). **EXECUTED dry
`--band-tick`:** green, `actual_rss_gib 54.06` = 56.69 units × 0.95367 ✓ ≡ the mine's tau baseline
54.0; `pause_scope_rss_gib 0.02` — the #246/F1 scope gap measured + carried in the row (and in red
reasons, gov:1088-1092), honestly NOT fixed. Red action is `pause_job` SIGSTOP only; the action
vocabulary is closed and kill-free (gov:1021-1023, tests assert it); control-plane untargetable
(`throttle_eligible` exclusion → `alert_no_target`, gov:1078-1082; select_band_run picks
own-group-leader tracked jobs only); pressure warn/critical → `defer_to_throttle` backstop-BEHIND
(gov:1072-1077). `pgid_only_rss_gib` computes true GiB from ps KiB — the efficacy gap subtracts
true−true, no unit mixing (gov:1122-1137). 76 memory-suite tests green this round.

**GB-F1 (MED) — the ADMISSION path still consumes tracked units UNCONVERTED (mixed-unit
arithmetic, anti-conservative).** `sum_active_growth_headroom_gib` = Σ max(0, projected_peak[true
GiB] − current_rss[units]) (gov:794-796 via `TrackedJob.growth_headroom_gib`) and
`sum_tracked_current_gib` feeds `baseline = used[true] − tracked[units]` (gov:496,799-800).
MEASURED in this round's dry-run output: `active-growth 10.9` = 67.61 − 56.69(units); the true
value is 67.61 − 54.06 = **13.55 GiB — the admission gate under-counts remaining growth by
2.63 GiB today** (≈4.6% of tracked RSS, growing with tracked size), i.e. LESS likely to refuse.
The band commit fixed the boundary it owns; FEED-04r's "Units bug FIXED at the boundary" reads
broader than the fix. Impact now: negligible (post preserve-and-stop the box is single-workload
and the fresh run's margin is ~40 GiB). Fix in the #246 wave: convert in
`TrackedJob.growth_headroom_gib`/`sum_tracked_current_gib` (or fix `memory_guard.group_rss_gb`
upstream once, killing every downstream ×0.9537).

## Surface 5 — #294 ledger + calibrate-rss (3795339fe): **FINDING (1 MED), gates CLEAN**

**Verified clean / EXECUTED:** projection auto-append at (b1) on real launches, correctly SKIPPED
on --dry-run (tools/launch_witness_run.py:497-504); reconcile REFUSES to invent — executed 3-way
probe on a temp ledger this round: no-projection-row → RuntimeError; projection-but-no-measured-
source → RuntimeError ("Refusing to invent a number"); placeholder override cite "TBD" → rejected;
`calibrated_margin` <3 rows → labeled 10 GiB fallback ✓. Blackbox fallback converts units once
(preflight:373,414) incl. rotated archives. Live ledger carries the #205 backfill
(projected 67.61 / actual 67.68 / residual −0.07, mine-cited source) and is gitignored (LIVE_STATE) ✓.
`--calibrate-rss` overrun gate: rc=5 on missing telemetry, outer-timeout, and >15% overrun
(launcher:300-336); `TAC_MEM_PROBE=1` IS in the emitted launch.sh (launcher:122, verified in the
dry-run artifact).

**LG-F1 (MED) — the calibration's reconcile leg is dead code: "every calibration feeds
calibrated_margin()" (launcher:275-278 docstring) is mechanically false.** `_run_rss_calibration`
calls `wmp.reconcile_run_dir(calib_dir)` (launcher:326), which needs `calib_dir/run.log` safe_run
telemetry OR blackbox tracked rows matching the run-dir NAME ("calibrate_rss"). Neither exists:
the calibration safe_run's output goes to the subprocess pipe (no run.log is written — run.log is
a spawn_durable_daemon artifact), and `tools/safe_run.py` never registers with the governor
registry (grep: zero register/registry/blackbox hits) so `list_tracked_jobs`-fed blackbox rows
never carry the `calib_*` label — and even if they did, the match key is the run-dir name, not the
label. ⟹ `reconcile_run_dir` raises every time and is swallowed as the launcher:327-329 WARNING;
the ledger gets calibration PROJECTION rows but never calibration RECONCILE rows, so
`calibrated_margin()` stays on the 10 GiB fallback longer than designed. The rc=5 overrun gate is
UNAFFECTED (it parses the captured safe_run output directly, launcher:304-312). Fix: pass the
measured `actual_gib` into `reconcile_run_dir(calib_dir, actual_override=(actual_gib,
"calibrate_rss safe_run exit peak"))` — one line, uses the value already in hand. Default-OFF
feature ⟹ non-blocking.

## Surface 6 — #293 dual-vg (f4150fde4): **CLEAN (1 LOW)**

**Verified clean / EXECUTED:** B=1 dispatch: `_use_micro_batch` False ⟹ BOTH `value_and_grad_batch`
and `_dual_vg_batch` are None ⟹ the accum loop takes the serial branch, and the serial `_dual_vg`
dispatch is untouched by the commit (diff touches only the batched branch; source-guard test 4
asserts it). The emitted run-1 launch.sh carries NO `--micro-batch-pairs` ⟹ run-1 never enters this
code. The 4 preserved fail-closes raise (trainer:2904, 2943, 2990, 3056 — msal-reach /
spike-reweight / subpix / chroma × micro-batch>1); the old seed fail-close is gone by design.
Tolerance-vs-measured relationship is HONEST: accum-step tol 2e-3 covers the sister worst-case
2.8e-4/3.4e-4 with ~6× headroom (test:454-458 documents the denominator-cancellation decomposition
+ 3 controls); per-group tol 2e-4 vs measured ≤5.9e-5 has 3.4× headroom and the cancellation class
does not apply per-group; the proof is fixed-seed MLX-CPU deterministic (not flaky-by-realization).
**All 7 tests EXECUTED this round (106s) — including the real-path slow proof (real gt_n6 + real
frozen MLX scorers + real R + real island seed) — ALL PASS on this machine.** Honest scope note:
the real-path proof runs a toy witness (in_feat 12 / hidden 8) — legitimate for an algebraic
wiring-equivalence gate, and B>1 stays waterfill-gated pending the n600 RSS re-measure (the n6
+12 GiB B=4 price is real and reported).

**DV-F1 (LOW):** `RuntimeWarning: invalid value encountered in matmul` fires in the proof's fixture
(test:349, clean float32 inputs — an Accelerate/NumPy quirk; outputs are finite or the `loss_rel
< 1e-5` assertion would fail on NaN). Silence-or-root it so a real NaN can never hide in an
expected warning.

## Surface 7 — CROSS-CUTTING composed argv: **CLEAN (2 notes)**

All EXECUTED this round:
- `launch_witness_run.py --config fresh_seeded --dry-run` (n600/ep1000): **90/90 flags exist** in
  the trainer argparse; emitted launch.sh verified to carry ALL of: `--verdict-batch 64` (baked
  into `_FRESH_SEEDED_DELTAS` — **no --extra-trainer-flags needed for vb=64**, the tasking's
  premise is stale there, no drift), `TAC_MEM_PROBE=1`, `--closed-loop-control`,
  `--lane-prior-phi1-mode paint`, `--seed-islands`, `--eikonal-weight 0.05` + `--eikonal-weight-end
  0.1`, `--l7-start-epoch 1001`, `--hosc-beta-end 5.134` (β(726) = 1 + 4.134·725/999 = 4.0002 ✓),
  mod-19 + `--film-stiefel`, band 350, rewarmup 20-cosine, geometric τ 1.0→1.0, dilate 1, **NO
  event-trigger flags, NO bank flags** (bank-4 default). Choice-valued flags validated against
  argparse choices (geometric / cosine / paint / witness / inverse_thickness all legal); the
  `--stage-transition-rewarmup-epochs 20` guard is satisfied (`--lr-schedule` default True,
  trainer:4882, guard:5703-5708).
- Preflight on the EMITTED launch.sh at `--safe-frac 0.85 --strict`: **SAFE, 68.65 GiB ≤ 108.8**,
  rc=0; in_feat derived = **88** = trainer arithmetic (bank-4 @ max-freq 64 keeps all 40 atoms ⟹
  2·40 + 4·2) ✓ matches the emitted config; **the projection ≈68.65 GiB SAFE reproduces** (also at
  the launcher's default 0.70: 68.65 ≤ 89.6).
- Adversarial extras probes: `--extra-trainer-flags "--bank-n-scales 6"` → preflight re-derives
  in_feat 176 → **111.85 GiB REFUSE, rc=4** (the C4 fix holds end-to-end through the extras path;
  111.85 = 110.81@vb32 + 1.04 verdict delta ✓); `--extra-trainer-flags "--bogus-flag 1"` →
  **rc=2 invented-flag refusal** ✓. Duplicate-flag composition is consistent (argparse last-wins ==
  preflight parser last-wins).
- Waterfill BEST=(B=1, vb=64) 68.65/77.43 ×1.183 reproduces (see WF-F1 for the anchor caveat).

**Notes (LOW):** (i) the master lever ledger §1 still reads "KEEP `--verdict-batch 32`" and
"bank-6 INCLUDE (slope-gated)" — both superseded by later artifacts (FEED-04r vb=64; review C3
bank-4) but with no in-file supersession marker; a reader of the ledger alone reconstructs the
wrong argv. Add the two supersession lines. (ii) the launcher's `--mem-preflight-safe-frac`
default remains 0.70 — stricter than the 0.85 policy (conservative direction, so fine), but the
"final preflight @0.85" step must pass the flag explicitly (verified working above).

---

## Disposition summary

| # | Sev | Finding | Blocking? |
|---|---|---|---|
| M2-F1 | MED | self-orient pending reconcile not bit-identical; claim unconditional (trainer:3283 vs 3776-3781/3959) | NO — scope the claim; behavior deterministic + bounded |
| WF-F1 | MED | waterfill train-window anchor 2062s = wall min, not the measured ~5,042s 25-ep window ⟹ ×1.183 and "32→64 from arithmetic" are anchor artifacts | NO — vb=64 stays SAFE + never-slower; re-anchor before citing speedups |
| GB-F1 | MED | admission path mixes true-GiB and tracked units (measured −2.63 GiB growth under-count, anti-conservative) | NO — margins huge post preserve-and-stop; fix with #246 |
| LG-F1 | MED | calibrate-rss reconcile leg dead (no run.log / no registry row) — "feeds calibrated_margin()" false | NO — default-OFF; rc=5 gate unaffected; 1-line fix |
| M2-F2 | LOW | failed-verdict + pre-failure sidecar resurrects a row on resume | NO |
| M2-F3 | LOW | "~2× wall" benefit claim ⟹ measured ≈ +43% | NO |
| L7-F1 | LOW | `_schedule_from_flags` naive chain emits inverted l7 [1001,726) segment | NO |
| DV-F1 | LOW | matmul RuntimeWarning in the cograd proof fixture | NO |
| XC-i | LOW | lever-ledger stale rows (vb 32 KEEP / bank-6 INCLUDE) lack supersession markers | NO |
| XC-ii | LOW | launcher safe-frac default 0.70 vs policy 0.85 (conservative; pass the flag) | NO |

**PROCEED** for the launch runway. Recommended same-wave (pre- or immediately post-launch, none
gating): scope-note M2-F1 in the DAG; re-anchor `MEASURED_TRAIN_WINDOW_S` and stop citing ×1.183;
the LG-F1 one-liner (`actual_override` pass-through); GB-F1 units fix with the #246 pause-scope
wave. HARD GATE unchanged: pointer 0.19110 UNMOVED — the runway's END is the byte-closed
`upstream/evaluate.py` n600 exact row, not these gates.
