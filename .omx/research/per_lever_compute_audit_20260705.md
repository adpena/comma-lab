---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "s/epoch growth 42s->minutes is caused by the accreted levers"
    classification: CARGO-CULTED
    rationale: "MEASURED FALSE at the per-lever surface: n24 toggles show every individual lever is ~free (or negative) at CE stage; the growth is (a) the in-trainer step common core vs the bench closure, (b) the ep300 stage-activation group (+26% n600 MEASURED), (c) verdict-window CPU contention."
  - assumption: "n24 timing ratios transfer to n600"
    classification: HARD-EARNED-WITH-SCOPE
    rationale: "cross-calibrated: n24 full-step 6.068 s/ep x25 = 151.7 s/ep pure-step vs n600 CE-stage measured 169.7-188.9 (residual = verdict contention + overhead, consistent with the mine's 47% duty). Ratios advisory; n600 absolutes from run-log forensics."
council_decisions_recorded:
  - "op-routable 1: add --cache-gt-skeleton to the relaunch argv (bit-identical; ledger KEEP<->argv DRIFT found)"
  - "op-routable 2: micro-batch deploy path = extend levelset_micro_batch_loss to carry wa-island/bd/focal/eik-stab (spec, follow-up build)"
  - "op-routable 3: NO lever removed for speed - zero measured cruft; training-time stays lexicographic secondary"
---

# PER-LEVER COMPUTE AUDIT (task #306) — where 42 s/ep went, MEASURED

**2026-07-05. $0 local. Axis: all timing rows `[macOS-MLX advisory] NON-PROMOTABLE`; n600
absolutes mined from real run logs; n24 rows are toggle RATIOS only. Pointer contest-CPU
0.19110 UNMOVED — everything here is MEANS (speed, never score-traded; the
training-time-lexicographic-secondary rule governs). Sibling EIK-STAB-BUILD ran a concurrent
n24 arbitration campaign; probes below marked CLEAN ran on an idle GPU, probes marked
CONTAMINATED are excluded (re-run queue at the end).**

## 1. The accretion timeline (n600, MEASURED from run-log verdict `ts` rows, Δts/Δep per 25-ep segment)

Method: `ts` deltas between consecutive verdict rows ÷ 25 ep. Error: segments are single
measurements; intra-run spread ±6% (min-max of CE segments); the ep0-25 segment always
carries startup (+30-60 s/ep) and is excluded from medians.

| run | config delta (argv-diffed) | s/ep MEASURED | attribution |
|---|---|---|---|
| bench closure 2026-07-02 (`compute_facet_...20260702T105220Z`) | seg-only stripped step (trunk→R→SegNet→CE→v&g), B=8, grouped-backward ON | **43.3** (577 ms/step ×75) | the "42s era" anchor — NOT a run config; a bench closure without pose/loss-stack/opt/EMA/accum machinery |
| pre-launch prediction (`n205_joint_nexus_...`) | + PoseNet fwd_bwd + R both frames | 54–59 [modeled] | the model that the real run falsified by ~3.3× |
| #205 `20260703T120444Z` (mod-32, replace-seed, vb32) | full lever stack v3 | **180.5 median CE-stage** (ep25-300, 11 segs, range 171.6–189.8) | the real common core (see §3) |
| same run, **tau stage** (ep300-500, 8 segs) | ep300 activates: tau-softplus form + persistence/clDice (warmup 300) + lane-render-band (start 300) | **227.6 median (+47 s/ep, +26%)** | THE stage-activation cost; the only large lever-group cost found anywhere |
| `20260704T174257Z` (fresh_seeded pre: mod-19, paint, stiefel, vb64, no wa) | mod-32→19 + geometry primaries | 182.1 median CE (3 clean segs) | ≈ same as v3 CE |
| `20260704T234054Z` (+ witness-alone + seed-anneal) | wa-island added | 215.2 (ep0-25 startup seg ONLY — not comparable; its startup seg is *faster* than 174257Z's 231.2) | no evidence of wa cost at n600 |
| **v4 live `20260705T015247Z`** (seed-fix; mod-19 + wa + closed-loop) | the launch-authority argv | **169.7 median CE** (4 clean segs, 158.1–185.6) | mod-19 recovered ~11-19 s/ep vs mod-32 |
| whole-#205-run duty-cycle (mine §2 / throughput review WF-F1) | — | 201.7 s/ep incl stalls; verdict wall mean 2189 s = 43-47% duty, async-hidden | the 5,042 s/25-ep `MEASURED_TRAIN_WINDOW_S` anchor (already re-anchored e7f1091d0) |

**Headline: 43 → 170 is NOT lever accretion.** It decomposes (MEASURED):
- **43 → ~152** (n24 full-step ×25 = 151.7): the *in-trainer step* vs the *bench closure* —
  loss composition (score-domain CE + margin/amplify + eikonal/length + hosc + film-stiefel),
  per-chunk accum loop with spike-guard `mx.eval` barriers, EMA/clip/opt — the COMMON CORE
  shared by every config. Per-lever toggles (§2) prove no single lever owns it. (P_min probe
  to split closure-vs-core: CONTAMINATED this session, re-run queued.)
- **~152 → 169.7** (+12%): verdict-window CPU-torch contention (47% duty) + reorient
  (amortized ~0.6 s/ep) + checkpoints + closed-loop/telemetry (negligible, §4).
- **169.7 → ~217 at ep300+** (+26-28%): the tau-stage lever-group activation (tau form +
  persistence/clDice + band). The single biggest recoverable item: `--cache-gt-skeleton`
  (#260) is bit-identical and NOT in the v4 argv (drift vs the master ledger's KEEP row).

## 2. n24 toggle probes (CLEAN set; idle GPU; `--profile-timing` t_step mean of ep2-4, sd ≤0.02 unless noted)

Base = v4 argv at n24/gt_n24, epochs 4, muon@4, verdict-pairs 2. `[macOS-MLX advisory]`, ratios only.

| probe | t_step s/ep (n24) | Δ vs full | reading |
|---|---:|---:|---|
| P0 full v4 stack | **6.068** | — | ×25 = 151.7 s/ep pure-step @n600 (CE stage) |
| P1 − pose stack (`--w-pose 0`, no carrier) | 6.780 | **+0.71 (+11.7%)** | **the pose-carrier is a SPEED-SAVER**: warp(own-render) replaces a second full INR render; dropping pose makes the step SLOWER. Never "drop pose to save time." Verdict also slower (4.84 vs 3.07 s). |
| P2 − witness-alone-island-loss | 6.326 | +0.26 (+4.3%) | wa-island is FREE-or-negative: removing it re-routes islands onto the composed render and costs MORE. Confirmed at n600 (§1, 234054Z). |
| P3 − seed-islands − wa (wa requires seed; fail-closed) | 6.040 | −0.03 (−0.5%) | seed co-grad (dual value_and_grad) ≈ free at step level |
| P4 − self-orient | 6.072 | +0.00 | the −48%-d_seg basis lever costs NOTHING in the step (in_feat width is not the binding op); reorient every-50 amortized ~0.6 s/ep n600 (prior MEASURED) |
| P5 − chroma | 6.230 (sd 0.128) | +0.16 (noise-level) | chroma ≈ free in the step |

CONTAMINATED (sibling n24 campaign concurrent; excluded, re-run queued): P_min (seg-only
in-trainer), P6 −palette, P7 −eikonal/length, P0-repeat. Pending (never ran): P8
persistence-active(+P8b `--cache-gt-skeleton`), P9 band-active@ep1, P10 +bd 0.2, P12
micro-batch-pairs 4 (needs wa OFF pair). Contamination signature for the record: identical
P0 config re-run under sibling load = 10.76 vs 6.07 s/ep (+77%) — inter-run GPU contention
dwarfs lever deltas; single-workload discipline is mandatory for timing probes.

## 3. Cost × benefit matrix (join: measured marginal s/ep × the master lever ledger's ΔS evidence)

Classes: NECESSARY (cost justified by measured score role) / TUNE (cadence or cache; with the
score-neutrality argument) / FUSE (duplicate compute) / CRUFT (no ΔS + nonzero cost).

| lever | marginal cost (measured) | ΔS evidence (ledger) | VERDICT |
|---|---|---|---|
| pose-carrier (store-nothing) | **negative** (−0.71 s/ep n24-scale vs no-carrier) | d_pose vehicle; witness d_pose OPEN/unmeasured (#238) | **NECESSARY** — also the cheaper render path |
| witness-alone-island-loss | ~0 (−0.26 n24; n600 no jump) | THE island-formation route; LIVE run descending (Lane flip 0.392→0.216) | **NECESSARY** |
| seed-islands + co-grad | ~0 | nucleation fix; lane_FN 3× (paint smoke) | **NECESSARY** |
| self-orient (+reorient 50) | ~0 step; ~0.6 s/ep amortized reorient | −48% d_seg MEASURED (decisive basis lever) | **NECESSARY** (cadence already optimal; do not touch) |
| chroma | ~0 (noise) | #227 GREEN, verdict-BLOCKING A/B owed | **NECESSARY** (cost-free; keep for the A/B) |
| palette-anchor | pending clean probe | breaks 0.51 luma-ramp plateau (DIAGNOSED) | NECESSARY (assumed ~0 cost; startup-only mechanism) |
| eikonal + length | pending clean probe (grid ops, expected small) | nucleation hold (94%-survival regime MEASURED proxy); length IS MCF erosion — keep 0.001 | **NECESSARY** (values, not cost, are the lever) |
| persistence/clDice(5)+amplify | **the ep300 +47 s/ep group** (joint with band+tau form; split pending P8/P9) | births finest-scale tail; amplify requires seed (now present) | **TUNE→FUSE: add `--cache-gt-skeleton`** — sg=soft_skeleton(GT) is epoch-invariant + gradient-free ⇒ caching is EXACTLY bit-identical (n64 A/B'd, ledger); skips ~half the clDice recompute. **DRIFT FOUND: ledger says KEEP, v4 argv LACKS it.** |
| lane-render-band | inside the same +47 group (split pending) | recall 0.5475; net-S #205-gated A/B owed | NECESSARY-provisional (band gate ep350 already deconflicted) |
| verdict (600-pair CPU-torch) | wall 2189 s mean, 47% duty — **fully async-hidden** (window 5042 s ≫ wall 2439 s); residual cost = CPU contention inside ~152→170 | authority trajectory | NECESSARY as-is; vb=64 never-slower; **no action** until train window <~2400 s |
| telemetry (mem_probe/loss_terms/spike rows) | 3-6 rows/run + spike rows only on guard trips | observability non-negotiable | NECESSARY, cost ≈0. NOTE: 401-1443 `spike_skip` rows in the bd-resume probes = whole COMPUTED steps discarded — a training-health flag (sibling's spike-guard re-treat), not a logging cost. |
| closed-loop-control | decision at eval rows only (async, decide-on-previous) | bounded controller (#292) | NECESSARY, ~0 |
| **CRUFT class** | — | — | **EMPTY. No lever with nonzero measured cost and no score role was found.** The operator's instinct that "some increase is necessary" is right in the strongest form: the increase is the common core + one stage-group, not removable bloat. |

## 4. Undeployed speed levers (deliverable 4)

1. **`--micro-batch-pairs` (#261/#293, measured ~2-4×)** — absent from the live argv because it
   **FAIL-CLOSES against the v4 lever set**: trainer raises NotImplementedError for
   `--witness-alone-island-loss` (serial-path routing), `--boundary-distance-weight>0`,
   `--seg-focal-gamma>0`, eik-stab flags, and `--margin-saliency-reachability` (batched twin
   `tac.boundary_math.levelset_micro_batch_loss` doesn't carry them). #293 DID build the
   seed-co-grad batched path (7 tests, 106 s real-path proof, throughput review Surface 6 CLEAN) —
   the seed blocker is GONE; **wa-island is the remaining binding blocker** for the live config.
   Secondary: waterfill honestly excludes B>1 (n6 +12 GiB@B=4; n600 RSS unmeasured) and it is
   trajectory-affecting (batched fp reduction — grad==serial within tol, NOT bit-identical).
   **Deploy spec (follow-up build):** extend `levelset_micro_batch_loss.LeverConfig` with the
   wa-island routing (witness-alone render leg + island terms), then a B∈{2,4} n24 A/B pair
   (wa-ON serial vs wa-ON batched: loss/grad equivalence gate per the #293 pattern) + n600 RSS
   re-measure through the waterfill before any relaunch flag. Risk: MED (trajectory-affecting →
   [A/B-owed]); payoff: up to −50-75% of the 152 s/ep step core = the ONLY order-of-magnitude
   speed lever on the table.
2. **`--cache-gt-skeleton` (#260)** — wired, bit-identical, n64 A/B'd… and **missing from the v4
   argv** (ledger-KEEP ↔ argv drift; it also no-ops under micro-batch). Zero risk. ADD.
   [measured-neutral by construction]
3. **`--fused-r-kernel` (#212)** — now wired WITH a startup grad-bit-identity gate (fails closed).
   R = 2.3% of step (probe rows), fused fwd+bwd 5.4× ⇒ ~1.8% step saving. Small but free.
   [measured-neutral by gate] Optional-ON at relaunch.
4. **`--mx-compile`** — EXCLUDE: measured argmax-flip (Δ~4.8e-3), fails closed (ledger §6).
5. **fp16 cf-feats (#296)** — NOT BUILT (no trainer flag). It is a MEMORY lever (cf cache 44.1 →
   ~22 GiB), not a speed lever (fp32 is the measured MLX sweet spot); score-affecting (feature
   quantization ⇒ not bit-identical). Deprioritize for compute; revisit only if memory headroom
   binds (it doesn't: v4 peak 68.65/108.8). [A/B-owed if ever used]

## 5. Streamlined relaunch config delta (composes with EIK-STAB; deliverable 5)

vs the v4 `fresh_seeded` argv — **additions only, no removals** (no cruft found):

| change | tag | expected effect |
|---|---|---|
| `+ --cache-gt-skeleton` | [measured-neutral: bit-identical by construction + n64 A/B] | recovers part of the +47 s/ep tau-stage group from ep275/300 on; exact share = P8b probe (queued); bounded above by the clDice slice |
| `+ --fused-r-kernel` (optional) | [measured-neutral by startup parity gate, fails closed] | ~−1.8% step (~−3 s/ep) |
| keep everything else | — | every toggled lever measured ~free or score-critical |
| micro-batch-pairs | NOT YET — blocked (§4.1); build spec filed | future −50-75% step [A/B-owed] |
| **EIK-STAB composition slot** | sibling memo `.omx/research/eikonal_stabilizer_build_20260705.md` not landed at write time | their `--eikonal-steik-weight/--eikonal-viscosity` arms also FAIL-CLOSE vs micro-batch (same twin gap) — fold their chosen stabilizer into the same relaunch argv; no timing conflict expected (grid-op scale) |

**Projected s/ep + waterfill reconcile:** CE stage ~167-170 (unchanged; nothing removable),
tau stage 227 → [180, 227] pending P8b (cache-gt-skeleton share). `MEASURED_TRAIN_WINDOW_S =
5042.0` (already re-anchored, e7f1091d0) stays the CE-stage anchor and remains ≳2× the verdict
wall, so the wall stays fully hidden and vb=64 stays never-slower; note for
`tools/memory_waterfill_config.py`: the window is STAGE-DEPENDENT (tau-stage window ≈ 227×25 ≈
5675 s — even safer). No waterfill retune required by this audit.

## 6. Honest gaps / re-run queue

- P_min / P6 / P7 / P0-repeat CONTAMINATED (sibling GPU campaign); P8/P8b/P9/P10/P12 not yet
  run. Harness is landed (`experiments/results/compute_audit_n24_20260705/run_probe.sh`) —
  each probe is ~75 s on an idle GPU; run them in the next quiet window before the relaunch.
- The 43→152 split (bench-closure simplification vs common-core loss terms) is UNRESOLVED
  until P_min lands clean; the streamline does not depend on it (no removal proposed).
- All n24 numbers are `[macOS-MLX advisory]` ratios; the n600 absolutes above are the
  authority for wall-clock planning. Nothing here is a score claim; pointer 0.19110 UNMOVED.

## Observability surface
Per-probe JSON logs (`profile_timing` rows: t_epoch/t_step/t_verdict/t_overhead + R micro-bench)
under `experiments/results/compute_audit_n24_20260705/P*/run.log`; collector
`scratchpad/collect.py` logic reproduced in §2 table; n600 forensics reproducible from the
named run.log `ts` rows (§1 method). Diff-able across runs; cite-able (run dirs + commits);
counterfactual via the toggle argv recorded in each probe's safe_run header.

## 6-hook wire-in declaration (Catalog #125)
sensitivity-map: N/A (timing, not score axes) · Pareto: ACTIVE (feeds the waterfill/launch
runway: §5 stage-dependent window note) · bit-allocator: N/A · cathedral autopilot: N/A
(advisory timing) · continual-learning: this memo + DAG FEED row · probe-disambiguator:
the re-run queue (§6) IS the disambiguator for the confounded tau-stage group.
