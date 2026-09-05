# ddm_dk2 — APDataStore certify-and-MOVE cold-store sweep

Arm: ddm_dk2 (Opus). Date: 2026-09-04/05. Tokens: `[no-triality] [p0-ledger-ok]`.
Charter: `.omx/research/charters/ddm_dk2_apdatastore_cold_store_sweep_20260904.md`. Parent: dk1
(`.omx/research/ddm_dk1_local_disk_certify_and_move_reclaim_20260904.md`).
Binding: CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup, And Provenance" (certify or block); ALWAYS KEEP
THE PAYLOAD; `docs/operating_manual_craft_handoff.md`.

## Headline

**APDataStore 17 GiB → 55 GiB free. 39.00 GiB certify-MOVED to Vertigo cold store, zero bytes deleted,
zero `BLOCKED_*` rows. Vertigo 83 → 44 GiB, above its 40 GiB floor. ng5's storage leg goes 16.69 → 55.52
GiB against an 8.00 GiB reserve.** All MEASURED, 2026-09-04 23:55Z → 2026-09-05 01:00Z.

Objective (≥ 30 GiB) **met**; the stretch target (≥ 40) **missed by 1.00 GiB** — not because supply ran
out (140.30 GiB is certifiable on this tier) but because Vertigo's 40 GiB floor capped the move at ~43
GiB and I sized the batch to land safely inside it. Say that plainly: the binding constraint was the
destination, and I stopped one tree short of the stretch rather than shave the floor.

Two things worth carrying forward matter more than the bulk:

1. **The charter's predicted bulk class was wrong by ~7×.** Every finished burn-cell generation on this
   tier totals 4.44 GiB. The real 140.30 GiB sits in 49 `inflated/` trees — deterministic `inflate.sh`
   output whose generators are retained.
2. **A concurrent read-only census silently produced a 100%-blocked planner run** that read as a clean
   PASS. Cured by reporting the denominator, not by weakening the pin rule.

**Collateral I caused, MEASURED by the coordinator, not by me:** running 7–8 movers concurrently
saturated both tiers. `df /Volumes/APDataStore` went from milliseconds to 48–120 s; mc1's reads fell to
~37 MB/s; ng4's live Metal cell advanced **21 steps between 00:48Z and 00:58Z against a 130-steps/10-min
cadence**, its trainer sitting in state `U`. I capped concurrency to 4 on the first flag (suspending 3
movers with `SIGSTOP`, preserving their partial copies, and resuming them from an artifact-bound waiter
when slots freed). The second flag arrived at 00:59:55Z, the same minute the last move landed, so there
was nothing left to pause. After completion: 0 movers, `df` back to 0.003 s, ng4's trainer back to `SN`.
**I should have started at concurrency 2–4 on a tier with a live training cell on it, and measured the
victim's step rate myself rather than being told.** The throughput I bought was not worth 10 minutes of a
frontier cell's descent.

## The charter's PRIOR-LAW PREDICTION: mechanism confirmed, bulk class wrong

The prediction had two parts. **The mechanism half is CONFIRMED.** `tmutil listlocalsnapshots
/Volumes/APDataStore` returns an empty list — MEASURED, and the first thing I ran. APDataStore carries
**zero** Time Machine local snapshots, so unlike dk1's boot volume every retired byte returns to the free
pool immediately. dk1's binding constraint does not exist here.

**The bulk half is FALSIFIED, and by a wide margin.** The charter predicted the reclaim would come from
"retired arm stores" and "superseded burn-cell generations (QBR1 chain cells 1–6, ng1/ng2/ng3 milestone
`realized/` + `reencoded/` frame trees)". MEASURED:

| Predicted class | Predicted role | MEASURED size |
|---|---|---|
| QBR1 burn store, whole tree | the ~7 GiB/generation consumer | **6.73 GiB total** — `runs` 1.28, ng1 1.60, ng2 1.42, ng3 1.42, ng4 1.01 |
| ng1 + ng2 + ng3 (the movable, finished generations) | ≥ 30 GiB | **4.44 GiB** |

Moving every finished burn-cell generation on the tier yields **4.44 GiB** — under a seventh of the
objective. The four generations did take APDataStore 23 → 22 → 16 → 16 GiB, but the burn cells are not
where those bytes went; each generation also seals a full source tree on **Vertigo**
(`/Volumes/VertigoDataTier/pact/ddm_ngN_*/sealed_source_*`), which is why Vertigo fell in step.

The bulk is somewhere the charter did not look: **deterministic `inflate.sh` output**. A tier-wide scan
found **49 `inflated/` trees totalling 140.30 GiB**, mostly 3.52 GiB `fire_local_advisory` cells. That is
4.7× the objective in a single class, and it is dk1's Class B repeated ~40 times.

## Census — BEFORE (MEASURED 2026-09-04 ~23:55Z)

`/Volumes/APDataStore/pact` = **1094.72 GiB across 309 entries**. Top 15 by `du -sk`:

| GiB | path |
|---:|---|
| 119.77 | `cold_store` (`long_training_checkpoint_retention`) |
| 73.04 | `cold_store_experiments_20260813` |
| 65.02 | `ddm_bs3_born_small_resolved` (60.31 of it is `retained/`) |
| 62.06 | `ddm_qbflow_implicit_boundary_flow` |
| 57.89 | `ddm_wd3_scorer_aware_width_distillation` |
| 54.58 | `cold_store_external_dqs1_20260813` |
| 40.53 | `ddm_s1a_stage_a_adapter` |
| 35.17 | `pr135_joint_solve_20260810` |
| 31.32 | `ddm_wd2_width_distillation` |
| 28.82 | `ddm_wc1_advisory_decode_wallclock_20260815` |
| 27.63 | `ddm_w96a_aligned_window` |
| 27.33 | `vertigo_coldstore` |
| 17.17 | `ddm_rx2_current_mc36_label_hpac` |
| 16.93 | `ddm_d3a_analytic_lane_carrier` (all of it `retained/`) |
| 16.09 | `ddm_mc36_20260814` |

Volume free space: **APDataStore 17 GiB / 1.8 TiB (100% full)**; Vertigo 83 GiB; boot 211 GiB.

The top-of-census is a trap, and dk1 named it: at **whole-tree** granularity almost nothing here is
movable. `ddm_bs3_born_small_resolved` is 65.02 GiB of which 60.31 is `retained/` and 4.70 is `receipts/`
— both hard never-touch. `ddm_d3a_analytic_lane_carrier` is 16.93 GiB and *all* of it is `retained/`. The
honest planner run over all 309 candidates returns **190 `blocked_never_touch` by descendant scan, 8
pinned, 103 `certify_move_required`, 0 `git_reconstructible`** (no git worktrees live on this tier).

Reclaim on this tier therefore requires descending *past* the store root to a subtree that excludes the
protected descendants — which is exactly the shape dk1's Class B took, and exactly what the `inflated/`
class is.

## The class moved: `rebuildable_inflate_output`

A `fire_local_advisory` cell is laid out:

```
<cell>/work/archive.zip                 <- the GENERATOR
<cell>/work/inflated/                   <- 3.52 GiB of deterministic inflate.sh output
<cell>/work/report.txt, provenance.json, inflated_outputs_manifest.json
<cell>/contest_auth_eval.json           <- the MEASURED result
```

Only `work/inflated/` moved. The generating `archive.zip`, the manifest, and the measured
`contest_auth_eval.json` all stay on local disk, so ALWAYS-KEEP-THE-PAYLOAD holds in its strong form: the
thing that *makes* the bytes never leaves, and the thing the bytes were measured *for* never leaves. Only
the deterministic expansion is externalised.

`ddm_jf2_terminal_diagonal_harvest/scorer` is the one whole-directory move (8 advisory cells, 283 files).
It is admissible on the same rule for a different reason: all 14 of its generating `archive.zip`s live in
the sibling `ddm_jf2_terminal_diagonal_harvest/retained/<cell>/retained/candidate_runtime/` tree, which is
never-touch and stays local. That is dk1's `ap1/advisory` moved / `ap1/retained` kept, precisely.

Every one of the 8 sources was independently checked for protected descendants
(`retained` / `receipts` / `*.npz` / `*.pt` / `*.done` / `*SEAL*` / `*RECEIPT*` / `RETENTION_MANIFEST.json`)
before launch. All 8 returned NONE.

## Moves executed

Executor: `tools/vertigo_certify_move.py --source-root /Volumes/APDataStore --dest-root
/Volumes/VertigoDataTier/pact/cold_store --category rebuildable_inflate_output --min-dest-avail-gib 40
--apply --retire-source`, one detached launch per tree via `tools/launch_detached_process.py` with a
distinct `--done-receipt`. Ledger `.omx/research/ddm_dk2_disk_reclaim_certs_20260904.jsonl`.

| GiB freed | content manifest sha256 | source (under `/Volumes/APDataStore/pact/`) |
|---:|---|---|
| 14.13 | `cad5690fbab7…` | `ddm_jf2_terminal_diagonal_harvest/scorer` (8 advisory cells, 283 files) |
| 3.63 | `8c0c379e07bc…` | `ddm_rr6/advisory_ship_r2/work/inflated` |
| 3.63 | `a1a0e3a1b232…` | `ddm_rr6/advisory_native_r1/work/inflated` |
| 3.52 | `c7b7d77c8f53…` | `ddm_rc2/composed_decode_r1/inflated` |
| 3.52 | `f0158b3ace74…` | `ddm_rc2/composed_decode_r2/inflated` |
| 3.52 | `eedeaeec1902…` | `ddm_to1/advisory/attempt_0002/work/inflated` |
| 3.52 | `0500d378fae1…` | `ddm_up3/advisory/attempt_0001/work/inflated` |
| 3.52 | `80e73e5db85b…` | `ddm_w72_distortion_advisory/attempt_r1/work/inflated` |
| **39.00** | | **8 trees** |

**Ledger: 32 rows — `PLAN` 8, `COPIED` 8, `VERIFIED` 8, `MOVED_SYMLINKED` 8. No `BLOCKED_*` row, and no
byte deleted anywhere.** Every row carries the source, destination, category, reason, both `df` readings,
the content manifest sha256 and a separate metadata manifest sha256. Source retirement happens only after
an independent destination re-read, so an interruption at any point leaves the source intact.

Post-move spot check on all four early completions (`to1`, `up3`, `rr6_ship`, `rr6_native`): the source
path holds a transparent symlink that resolves (`inflated -> /Volumes/VertigoDataTier/pact/cold_store/pact/…`,
probe file `0.raw` lists), and the generating `archive.zip` is still local (176,420 B for `to1`/`up3`;
180,625 B for both `rr6` cells), alongside `contest_auth_eval.json`, `provenance.json` and
`inflated_outputs_manifest.json`.

### Free space, MEASURED

| | before (23:55Z) | after (01:00Z) |
|---|---:|---:|
| `/Volumes/APDataStore` avail | **17 GiB** (100% full) | **55 GiB** (98%) |
| `/Volumes/VertigoDataTier` avail | 83 GiB | **44 GiB** (floor 40 honored) |
| `df /Volumes/APDataStore` latency | 0.003 s → 48–120 s under 7–8 movers | **0.003 s** |
| Time Machine local snapshots on APDataStore | 0 | 0 |

The `df` delta on APDataStore is +38 GiB against 39.00 GiB retired. The ~1 GiB difference is other
writers on the volume during the window — ng4's live cell writes its milestones there. Stated rather than
rounded away.

**Census AFTER is DERIVED, not re-measured, and deliberately so.** `/Volumes/APDataStore/pact` =
1094.72 − 39.00 = **1055.72 GiB across 309 entries**; the eight source paths are now symlinks and every
other entry is byte-identical to the before-census. A second full `du` over 1.05 TiB would have put ~10
more minutes of I/O on the volume whose live training cell I had just starved. The authoritative
after-number is the `df` row above, which is MEASURED.

## The defect I hit, and the cure

**My first planner run reported all 307 candidates `blocked_never_touch`, every size `0.00`, and a
summary reading `certified-deletable 0.00 GiB in 0 tree(s); certify-MOVE owed 0.00 GiB in 0 tree(s)`.**
That is a clean-looking PASS that reclaims nothing — dk1's own bug #1 ("a fully-blocked census is a
vacuous PASS") wearing a different costume.

The cause was mine, not the tool's. I had a `du -sk /Volumes/APDataStore/pact/*` census running in the
background for the charter's own deliverable. The shell expanded that glob, so `ps -Ao command=` showed a
single live process with **all 304 child paths on its command line**, and `path_is_pinned` — correctly, by
its own rule — pinned every candidate. dk1 had already excluded the planner's *own* PIDs; a **sibling
read-only observer** is a different process and slips straight through. A second orphaned `du` from a
foreground call the harness had reaped (rc=144) was still running and doing the same thing.

I did not weaken the pin rule. A `find <tree> -delete` also has argv[0] `find`, so exempting "read-only"
binaries by name would trade a false positive for a real hazard. The cure is the one that generalises past
this particular cause: **report the denominator, and make a fully-blocked census loud.**
`tools/local_disk_reclaim.py` now prints
`census denominator: R reclaimable / P pinned / N never-touch / T candidates` on **stdout** (a plan is
routinely captured by redirecting stdout — a denominator hidden on stderr is exactly the number that goes
missing; I made that mistake in my own first draft of the fix and caught it in review pass 2), and emits
`WARNING VACUOUS-CENSUS` on stderr when `R == 0 < T`, naming the concurrent-census cause when pins
dominate. 6 regression tests; suite 45 → 51 passing. Commit `2130725f8`.

Genus: `[[m50]]` VACUITY==PASS: report the DENOMINATOR. Operationally: **do not run a census and the
planner over the same root at the same time** — and if you must, the denominator line now tells you the
result is worthless instead of letting you act on it.

## ng5 storage leg

`tools/cell_queue_driver.py plan --queue /Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective/QUEUE_SPEC.json`
(read-only, full seal verification, no `--skip-seal-verify`), MEASURED 01:00:45Z:

```
STORAGE LINE: available_bytes=59616133120 (55.52 GiB) reserve_bytes=8589934592 (8.00 GiB)
              path=/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep -> PASS=True
seal.content_identical=True  paths_rooted_in_firing_tree=True  pins=20
ready=False  live_cell_count=4  resolved_peak_gib=49.572
```

The storage leg passes with **6.9× the reserve**. Its trajectory across the sweep, all MEASURED:

| t | APDataStore avail to ng5 | vs 8.00 GiB reserve |
|---|---:|---|
| 00:15Z (before any move landed) | 16.69 GiB | PASS (2.1×) |
| 00:36Z (4 moves landed, 14.30 GiB) | 30.99 GiB | PASS (3.9×) |
| 01:00Z (8 moves landed, 39.00 GiB) | **55.52 GiB** | PASS (6.9×) |

Honest framing: **the storage leg already passed before this arm moved a byte.** ng5 was never blocked on
disk; it is blocked on `ready=False`, which is the admission gate refusing while ng4 is live
(`live_cell_count` fell 17 → 12 → 4 as my own movers exited the count). What the sweep bought ng5 is
margin, not unblocking — 55.52 GiB against a 49.572 GiB resolved peak means the cell can also hold its
measured peak, which at 16.69 GiB it could not have.

### A second, smaller observation: two tests are not saturation-safe

Running `pytest -k "reclaim or certify or disk or storage"` while my movers were live produced two
failures that both pass in isolation, so I record them as load-induced rather than as defects I
introduced — and I checked, rather than assuming:

- `test_vertigo_certify_move.py::test_df_for_path_measures_the_callers_filesystem_not_a_literal_mount`
  asserts two independent `df` readings of the same filesystem are equal. Under concurrent writers the
  boot volume's `avail_kib` moved between the two calls. It passed alone (`1 passed`) and passed again on
  the next sweep (`304 passed`). The assertion assumes a quiescent filesystem; that is a latent
  fragility in a test I did not write and did not touch.
- `test_micro_batch_bit_identity_probe.py::test_training_admission_classifies_complete_parity_but_disk_timing_never_go`
  hit the 60 s `pytest-timeout` inside a storage probe while `df` on the saturated tier was itself taking
  48–120 s. **A disk-timing admission test cannot pass while a tier is saturated** — which is arguably
  the test working.

Neither cites `local_disk_reclaim.py` or its suite. Separately, `preflight --scope dev` reports
`[lane-pre-registered] FAILED` on `experiments/ddm_mc1_motion_compensated_previous_plane.py:707,1083`
(unregistered `lane_oracle`, `lane_iou`) — mc1's live file, charter never-touch, left alone; and the
repo-wide orphan-module guard trips at 22 > 10 on `src/tac/` modules unrelated to this arm. Both
pre-existing. `test_local_disk_reclaim.py` + `test_vertigo_certify_move.py` = **67 passed** with no
movers running.

## Equations leg (`tac.canonical_equations`)

**None, and none is owed.** This arm is disk hygiene. It moves no score, measures no scorer quantity, and
produces no law of the form `S = f(...)`. No canonical equation is registered, refined, or cited. The
`freed_allocated_kib` figures in the cert ledger are allocation deltas; on this volume they also equal
availability deltas, because there are no snapshots to pin retired bytes — but that is a property of the
volume, not a measurement of the score.

## Blockers (certified as NOT reclaimable)

The charter's falsifier was "< 15 GiB certifiable → report the blocked classes with sizes and stop". It
did not fire — 140.30 GiB is certifiable in the `inflated/` class alone. The blocked classes below are
recorded anyway, because they are where the *bulk* of the tier is and the next disk arm will start here.

| Path / class | Size | Why blocked |
|---|---:|---|
| `cold_store/long_training_checkpoint_retention` | **119.77 GiB** | already cold store, and a retention tree by name and intent. Moving it to Vertigo's cold store is tier-to-tier churn that spends the headroom this arm must preserve. Not touched. |
| `cold_store_experiments_20260813/data` | 73.04 GiB | prior `ddm_sc3` custody move; `SC3_MOVE_MANIFEST.json` + source/destination hash checkpoints govern it. Same reasoning. |
| `ddm_bs3_born_small_resolved` | 65.02 GiB | 60.31 GiB is `retained/`, 4.70 GiB is `receipts/` — both hard never-touch. **0.01 GiB** of the tree is anything else. |
| `cold_store_external_dqs1_20260813/data` | 54.58 GiB | prior cold store (`dqs1` external). Its four `local_cpu_advisory_work/inflated` trees (13.64 GiB) *are* the same rebuildable class, but they already live in a cold store under an SC3 manifest; relocating them again is churn, not reclaim. |
| `ddm_d3a_analytic_lane_carrier` | 16.93 GiB | **all** of it is `retained/`. |
| `ddm_hv1_harvest_compose`, `ddm_hv1_base_advisory_n600_cpu` | 14.07 + 3.52 GiB | charter never-touch (`hv1`). The `hv1_base_advisory` tree holds a 3.52 GiB `work_r2/inflated` that is otherwise exactly the moved class. |
| `ddm_mc1_*`, `ddm_ps1/ps2_pr140_update_prep`, `ddm_fs1_frame0_selector`, `ddm_fs2_carrier_resolve*`, `ddm_gv1_governor`, `ddm_ng1..5_*`, `ddm_qbr1_born_fairform_burn_prep` | — | charter never-touch (live stores, pointer custody, governor, burn chain). |
| `ddm_fs3` | 7.11 GiB | not named by the charter, but it is adjacent to the live `fs2` pointer-custody work. Excluded by choice, not by rule — stated plainly because it was a judgement call, and the class had 3.6× more supply than the Vertigo headroom could take. |
| 190 store roots | — | `blocked_never_touch` by the planner's descendant scan: each contains a `retained/`, a `receipts/`, a `.npz`/`.pt`, or a `*.done` receipt somewhere below. Correct at whole-tree granularity; reclaim there means descending, as this arm did. |
| 8 store roots | — | pinned by a live claim or process at plan time. |

## What I did NOT do

- **The remaining ~101 GiB of the `inflated/` class.** 140.30 GiB is certifiable; 38.99 GiB was moved.
  The binding constraint is **Vertigo headroom, not source supply**: Vertigo held 83 GiB and the charter
  requires ≥ 40 GiB left for mc1's payloads, so ~43 GiB was the whole budget. The next reclaim on this
  tier does not need more analysis — it needs a destination. The full 49-tree inventory with sizes is
  committed at `.omx/research/ddm_dk2_inflated_class_inventory_20260904.txt` (KiB, paths relative to
  `/Volumes/APDataStore/pact`), so the next arm starts from a measured list rather than a re-scan.
- **The three `cold_store*` trees (247.39 GiB, 23% of the tier).** They are the largest thing here by far
  and I did not touch them. They are already cold store; moving them between tiers spends headroom
  without retiring a byte. Whether `long_training_checkpoint_retention` (119.77 GiB) is still worth its
  space is a real question and a different arm's — it is a retention decision, not a hygiene one.
- **The QBR1 / ng1 / ng2 / ng3 milestone trees the charter named.** Certifiable in principle, but they
  total 4.44 GiB and the objective is 30. Moving them would have added risk next to a live burn chain for
  11% of the target. Left alone.
- **A whole-volume top-level census.** `du -sk /Volumes/APDataStore/*` was launched and reaped twice by
  the harness (rc=144); I killed the orphan rather than restart it, because it was also the process
  polluting the planner's pin set. The `pact/` census (309 entries, 1094.72 GiB) is complete and is the
  part this arm can act on; the non-`pact` top-level directories (`Archive`, `Molt`, `RelocatedHome`,
  `teadata`, …) are the operator's own data and are out of this arm's scope either way.
- **I did not weaken the live-process pin rule** after it produced a 100%-blocked census. Exempting
  `du`/`find`/`ls` by name would trade a false positive for a real hazard (`find … -delete`). The
  denominator report is the cure; the pin rule is unchanged.
- **No APDataStore writes** beyond the source retirements themselves. Every destination byte went to
  Vertigo; all scratch (`dk2_work/`) is on Vertigo.
