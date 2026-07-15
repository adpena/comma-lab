# Signal-loss recovery + coherence pass — 2026-07-15

**Arm:** `merge_coherence_20260715` (supersedes `p0_merge_to_main_20260715`, which died BLOCKED at step 3
with zero commits picked — its sandbox exposed `.git` read-only; this session had writable git).
**Charter:** operator 2026-07-15 *"comprehensive signal loss pass and coherence pass … recover and respawn
and continue with all … ensure no signal loss"* + charter extension *"prune all work trees that are not
live WIP after cherry picking and landing all signal on main"*.
**Pointer:** 0.19108 submittable / 0.18804 bank — **UNMOVED** by this pass (apparatus/means only; no score claim).

## Headline finding

Most of the "37 unmerged commits" pool was **already re-landed on main** by the earlier 3-way
restores (`fa5a671330`-era reconciliation): papers + INFRA_gate had ZERO differing files; p0_harvest_held's
entire code surface auto-merged to net-ZERO (main's 2026-07-15 merge-reconciliation hunks strictly
superseded every branch-side hunk). The true unlanded signal was: branch-only **ledger rows** (lane
registrations + audit rows), the **c4_mod19 dead-arm dirty worktree** (fail-closed receipts), and the
6 **held_entangled** branches whose real deltas live in live-arm-owned files.

## Merge table (per branch, with shas)

| branch (codexwt/…) | commits | action | sha / disposition | notes |
|---|---|---|---|---|
| papers_12140_12922 | 2 | MERGED --no-ff | `ff7507737f` | content pre-landed at `abe926e467`; merge restored 2 branch-only lane registrations (`lane_papers_2607_12140_12922_20260715`, `c1_deepmath_integration_20260715`) + 4 audit rows. Branch deleted. |
| modal_folded_bank_net | 1 | MERGED --no-ff | `84b9fadb08` | fail-closed folded-bank refusal receipt (DAG FEED + result.json, no score claim) + lane. Branch deleted. |
| p0_harvest_held | 9 | MERGED --no-ff | `f4e4a2b62c` | #406 DSL-compile-hash + harness consolidation. ALL code pre-reconciled (net code delta ZERO; 6 conflict hunks in confound_gates/v9_provenance_gates were main-side 2026-07-15 reconciliation fixes → resolved `--ours`). Ledger rows landed. Branch deleted. First attempt lost MERGE_HEAD to a concurrent live-arm commit → redone holding the commit-lock across merge+resolve+commit. |
| harness_consolidation_fixes | 4 | MERGED --no-ff (ancestry-only) | `d5068236fc` | fully subsumed by the harvest re-land (line-level verified: 0 branch-added lines missing from main; the single variant line restructured into `confound_gates.py:1958`). Branch deleted. |
| c4_mod19_local | 0→1 (harvested) | HARVESTED `47f712aca1` then MERGED | `51ece84eb4` | dead arm's dirty worktree (died at serializer `git add` rc=128, sandbox) committed verbatim on its branch, then 3-way merged: fail-closed C4 byte-close receipts (C4_BLOCKED_CHECKPOINT_CUSTODY), C4 contest-CPU-authority + C1 clean-checkpoint-producer tickets, `lane_c4_mod19_rate_byteclose_20260715`, task-status rows (AUTH-C4-MOD19 / C4-MOD19-RATE-BYTECLOSE / C1-WITNESS-CLEAN-STAGE-EMA). Branch deleted. |
| INFRA_gate | 1 | DELETED (no merge needed) | content = `cacff6c1a2` | verified 0 files AND 0 ledger rows differ branch-vs-main. `branch -D` (sha differs from re-land, content identical). |
| basis_d21a_prod | 1 | **HELD_ENTANGLED** | branch kept | touches dirty live-arm file `train_levelset_witness_realized_through_R_mlx.py`; remaining deltas: optimal_basis/basis_control/curriculum_dsl/launch_witness_run. |
| c0_optform_compute_audit | 3 | **HELD_ENTANGLED** | branch kept | touches both witness trainers (pose-blind compute gate through typed DSL). |
| derive_solver_provenance | 9 | **HELD_ENTANGLED** | branch kept | touches `spec_v9_cgauge.py` + `telemetry_producers.py` (both live-arm dirty). Most content already re-landed; remaining 8 files incl. D39 event-marks telemetry, Fisher H⁻¹ trust solver policy, witness-native Morse continuation. |
| iso_configs | 2 | **HELD_ENTANGLED** | branch kept | `v9_cgauge_432_{taper_off,horizon_iso,step_iso}` typed ISO A/B configs — these are the duty-rows' ISO arms; touches trainer + spec_v9_cgauge. |
| l7_default_failloud_budget_eventlaw | 3 | **HELD_ENTANGLED** | branch kept | l7 explicit fail-loud opt-in + event-conditional wall-clock budget law; touches trainer. |
| p0_confound_hardening | 1 | **HELD_ENTANGLED** | branch kept | confound alarms + warmup verdict telemetry; touches both trainers + curriculum_dsl. |
| curvelet_optimal_form_crux | 1 | **SKIPPED — LIVE ARM** | untouched | codex sol/ultra still RUNNING; only surviving worktree. |

Unmerged pool: **37 → 20** (19 held_entangled + 1 live). Every held branch has a durable
`held_entangled` disposition in `.omx/state/codex_landing_ledger.jsonl` naming the blocking live arms +
files; the landing gate now surfaces them as `held_entangled_unblocked` pending-review rows = the
auto-resurface mechanism.

## Prune table (charter extension)

| worktree | dirty files harvested/discarded | branch | worktree removed |
|---|---|---|---|
| dsl_hash_enforcement (17 dirty) | **DISCARDED-with-record**: 10 files identical-to-main; `preflight.py`+`v9_provenance_gates.py` identical to harvest branch; `launch_witness_run.py` older-subset of harvest branch (lacks mod19_sR support the harvest version has); DAG FEED = whitespace variant; 4 audit rows = timestamp-variant self-registrations (lane content preserved via `f4e4a2b62c`) | deleted (`-d`, 0 commits) | y |
| c1_deepmath_integration (9 dirty) | **DISCARDED-with-record**: `optimal_basis_20260714.py`+test = strictly older-subset-of-main (worktree-unique lines were the pre-genuine-frame import/override lines main replaced); 4 untracked memos identical-to-main; lane registration restored via papers merge `ff7507737f` | deleted | y |
| c1a_dsl_sR_config (8 dirty) | **DISCARDED-with-record**: `spec_v9_cgauge.py` + 3 test files identical to harvest branch; `launch_witness_run.py` older-subset (lacks dsl_compile_hash additions); sub015 FEED-cvl-consumer-leg block already on main (×2 occurrences) | deleted | y |
| c4_mod19_local (7 dirty) | **HARVESTED** → `47f712aca1` → merged `51ece84eb4` (unique task-status rows + lane + fail-closed receipts) | deleted after merge | y |
| papers / modal_folded / p0_harvest_held / harness_consolidation / INFRA_gate (clean) | n/a | merged/deleted per table above | y |
| basis_d21a / c0_optform / derive_solver / iso_configs / l7 / p0_confound (clean) | n/a | **kept** (held_entangled) | y |
| curvelet_optimal_form_crux (6 dirty, LIVE) | untouched | kept | **n — only survivor** |

End state: `.omx/tmp/codex_worktrees/` contains only `curvelet_optimal_form_crux_20260715_20260715T093444Z`.

## Test status

- `src/tac/tests/test_dsl_compile_hash_enforcement.py` + `test_confound_gates.py`: **134 passed, 1 failed** —
  `test_real_repo_live_count_bounded[check_levelset_hosc_requires_beta_end]` fails on **on-disk run dirs**
  (`experiments/results/_jbasin_smoke/*/launch.sh` with `--activation hosc` sans `--hosc-beta-end`); PRE-EXISTING,
  repo-state-dependent, unrelated to the merges (net code delta of the whole sweep = ZERO on `src/` + `tools/`).
- `tools/lane_maturity.py validate`: **OK — 1895 lanes** after all ledger unions (canonical `indent=2, sort_keys=False` format preserved; first papers-merge attempt reformatted the registry and was amended to an 87-line diff).

## Task-ledger disposition table (no TaskUpdate tool in this environment — parent applies)

In_progress rows:

| # | disposition | evidence |
|---|---|---|
| 171 | STILL-LIVE (program umbrella — the capstone itself) | v9·CGauge line is its executor |
| 194 | SUPERSEDED-BY v9·CGauge covariant trunk (+#483 twist-descent owed) | Einstein pass: pair-dependence factors through (ξ,R); recommend close-with-pointer |
| 205 | SUPERSEDED-BY #507 (C1 composed config is the new pointer-mover run; #205's risk-register #243 landed) | recommend close-with-pointer to #507 |
| 221 | **ORPHANED** (fine-tune-vs-from-scratch A/B; no owning arm) | respawn candidate, natural rider on #507's config family |
| 223 | STILL-LIVE — owner = held branch `derive_solver_provenance` (D39 eq, Morse continuation, Fisher H⁻¹) | resurfaces at merge |
| 248 | STILL-LIVE (v9 terminal joint pose-finish is the vehicle; P-B verdict landed) | |
| 287 | fold into #497 basis-cure + phase stack (dash-comb = along-tangent modulation instance) | recommend SUPERSEDED-BY #497 |
| 297 | STILL-LIVE (banked/deferred edge-tier envelope; no active arm — by design) | |
| 328 | **ORPHANED** (clip_profile Phase-2 consumer rewire; no owner) | respawn candidate |
| 336 | at-risk: SOL arm `sol_336` last checkpoint 59.9h ago mid-n600-tensor-curves, pid dead | respawn/verify — if its n600 process died, re-fire |
| 337 | STILL-LIVE umbrella (BUILD-WAVE, fires on #335) | |
| 343 | STILL-LIVE epic (dashboard; no live arm — backlog) | |
| 349 | STILL-LIVE; remaining = reduced-order model derivation (#318/#344/#180 routes per continuation memo) | |
| 380 | STILL-LIVE (gated: fires at crucible-2 P7) | |
| 381 | STILL-LIVE (standing Modal-envelope ledger row) | |
| 386 | STILL-LIVE (v8 increment-1, gated on v9 outcome) | |
| 394 | STILL-LIVE batch (unexploited sweep; partially consumed by #406/#336 lines) | |
| 395 | **COMPLETED-as-DROPPED** — operator dropped the texture trunk for the single covariant V9·CGauge trunk | MEMORY L86/L87 "TEXTURE TRUNK DROPPED #395"; mark completed w/ note |
| 396 | STILL-LIVE (MC-finisher; terminal-band trigger) | |
| 399 | **COMPLETED** — bank 0.18804 landed 07-12 via exact-score-gated PR128 2656-click splice on our PR110 base (sha 196acd18); NON-SUBMISSION borrowed bank | mark completed |
| 400 | STILL-LIVE (fires at witness terminal band) | |
| 406 | STILL-LIVE (trigger-gated batched apply-pass) | |
| 432 | SUPERSEDED-BY #507 — the composed config realizes the state-gated event curriculum (#430) natively; the deferred concurrent A/B is moot | recommend close-with-pointer |
| 434 | STILL-LIVE (SOL ultra synthetic-data; verify arm liveness) | |
| 445 | STILL-LIVE — owner: cuda_smoke_438_respawn arm line | |
| 449 | STILL-LIVE — feeds #509 burn-down (frozen-SegNet = 95% wall-clock question) | |
| 494 | STILL-LIVE — feeds #509 (authority ladder; rung builders active ≤40h ago) | |
| 497 | STILL-LIVE — owners: LIVE curvelet_optimal_form_crux arm + held basis_d21a_prod branch | |
| 507, 509 | STILL-LIVE (the live arms) | |

Pending rows (compact): trigger-/gate-parked and correctly so: 51, 65, 86, 132, 134, 137, 144, 154, 170,
182, 183, 191, 195, 198, 199, 200, 211, 213, 222, 226–228, 236, 242, 243, 252, 255, 270, 273, 295, 296,
299, 307, 319, 332, 357, 359, 366, 408, 425, 444, 448, 450, 452, 478, 485, 496, 506, 511. Notable:
**#506 (DSL-hash fail-closed)** — the mechanism LANDED via the harvest_held reconciliation (admission_guard +
launcher hash enforcement on main); recommend verifying live count then marking completed. **#366 (joint
pose-finish P0)** — launch-ready, gated on config slot; rides #507. **#425 phase-carrier** — being turned ON
by #507 per its spec. **#511 (wholefile-clobber gate)** — still OWED, two-landing debt from the incident that
shaped this pass's merge rules.

## Respawn queue (recent dying/blocked rows with live signal; <100h, latest row, pid dead)

1. `codex_frozen_segnet_gradient_p0_20260712` — BLOCKED: finished work needs git-writable serializer rerun for exactly 20 new files (base=new).
2. `codex_micro_batch_v9_unlock_20260712` — BLOCKED: FreSh DSL factories fix + exact-hunk patch regeneration in a coordinated git-writable session.
3. `codex-sfess-cached-replay-20260712` — BLOCKED: git-writable serializer rerun only; no code/measurement remaining.
4. `sol_336` — n600 sensitivity-curve run died mid-flight (17 tensor curves remaining + KKT allocation + joint receiver replay) — re-fire (#336/#406 line).
5. `custom_metal_conv` / `custom_sparse_adjoint_kernel` / `ane_unlock_followup` / `pythagorean_exact_arithmetic_bitident` / `whole_teacher_distilled_student` — all BLOCKED on a Metal-visible and/or git-writable local host command (#478/#494 line); each has its exact recorded command in its checkpoint row.
6. `third_order_descent_filler` / `governor_measured_growth_fix` / `exp_linear_reparam_warmstart` — BLOCKED on privileged serializer/operator-run command; artifacts final, landing only.
7. `margin_adaptive_perlayer_followon` / `recursive_fractal_optimal_representation_v9` / `round2_structural_no_fake_review` — V9 provenance strict-flip chain, blocked on the exclusive V9 owner (#501 warn-only→strict debt).
8. `codex_438_relaunch` — superseded by the live cuda_smoke_438_respawn arm (close).
9. `clickpolish-399` / `clickpolish-build` — #399 completed (bank 0.18804); close both with pointer to the bank row.
10. `p0_merge_to_main_20260715` — marked complete/superseded by this arm (done).

Rows older than ~100h (≈90 in_progress strata from May–June) are historical generations (v7.5/v8-era waves,
Z5/Z8/NSCS substrates, PR110 opts) — not respawn candidates; they predate the v9 pivot and their
next_actions reference retired vehicles. They remain queryable in `subagent_progress.jsonl`.

## Cross-arm coherence findings

- **507 ↔ 509 OVERLAP (real, managed):** both live arms edit `experiments/train_levelset_witness_realized_through_R_mlx.py`
  and `src/tac/witness_dsl/curriculum_dsl.py`. They serialize via the commit-lock and have been landing
  cleanly (`34e375a2e7`, `c219841d8c`), but this is the one live-live seam — broadcast note sent.
- **cuda arm** domain (remote_v9_cgauge_cuda.sh, witness_cloud_launcher, modal_train_lane, torch trainer)
  is disjoint from 507/509 EXCEPT the typed-config surface; the held `derive_solver_provenance` branch
  touches `spec_v9_cgauge.py` + `telemetry_producers.py` → flagged in broadcast.
- **Merged branches introduced ZERO overlap**: the entire merge sweep's net delta on `src/`+`tools/` is zero;
  only ledgers moved.
- **Live-arm interference incident (recorded):** the first p0_harvest_held merge lost MERGE_HEAD when a live
  arm committed between my lock windows — root cause: I held the flock only around `git merge`, not through
  resolution+commit. Fixed for all subsequent merges (single process holds flock across
  merge→resolve→commit). Rule for future merge arms: **the commit-lock must span the whole merge
  transaction.**
- **Witness check-in anomaly:** costate digest reports run=dry_start pid=14828 ALIVE but STALE (no signal
  files in run dir) — consistent with a compose-arm dry-run; watch that it does not linger as a phantom.

## Duty-row spot-check (costate top-3)

| duty row | consumed? |
|---|---|
| DsegAwareTaper (78.9%) | YES — real `Lever` owner in `spec_v9_cgauge.py` (taper ownership REFUSE-guarded, ON in ideal config); its OFF arm exists as `v9_cgauge_432_taper_off` on the held iso_configs branch |
| HorizonWeightedMargin* (47.3%) | REASONED — v9 isolation note "ISOLATE: favorable geometry but its treatment weight lacks V9 custody" + `v9_cgauge_432_horizon_iso` typed config on held branch |
| StepNativeActivation* (34.2%) | REASONED — "ISOLATE: endpoint/activation basin; never stack into the matched core A/B" + `v9_cgauge_432_step_iso` on held branch |

All three are consumed-or-reasoned; the ISO A/B arms are merge-ready on `codexwt/iso_configs_…` the moment
the live arms release the entangled files. Broadcast sent so the compose arm knows where they sit.

## Dispositions written

15 rows appended to `.omx/state/codex_landing_ledger.jsonl` (9 reviewed_committed with merge/land shas +
discard records; 6 held_entangled with named blockers). `p0_merge_to_main_20260715` checkpoint closed as
superseded. Checkpoints for this arm at steps 1–5 in `subagent_progress.jsonl`.
