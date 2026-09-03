# DDM QBR1 — born/QBT fair-form burn preparation

**Date:** 2026-09-02  
**Arm:** `ddm_qbr1_born_fairform_burn_prep`  
**Axis:** `[build/seal only; no scorer, Metal, Modal, contest-CPU, or contest-CUDA invocation]`  
**Disposition:** **BUILD-COMPLETE / SEAL-CONDITIONAL-ON-MAIN-RESUME-SMOKE / BURN-NOT-FIRED**.

## Result first

QBR1 now has a real runner, six same-start cell configs, deterministic schedules, a complete
retention/checkpoint law, a mechanical endpoint adjudicator, an executable real-B=16 interruption
smoke for MAIN, and a sequential governed launcher order. The code is landed through the serializer
at `84ca4f32a53bd57cf98c332868fe27a44859a8d3`.

This arm did **not** run a frozen scorer, claim the scorer or Metal lane, start the approximately
18.61-hour burn, invoke Modal, evaluate an archive, or move the frontier. The fire order remains
fail-closed until MAIN supplies a live scorer claim to the bounded resume smoke, the smoke proves
uninterrupted/resumed identity, MAIN consumes the independent qxr1/QXO1 realization, and MAIN binds
fresh unique scorer and Metal claims into each copied cell config.

Primary custody:

| artifact | bytes | SHA-256 | status |
|---|---:|---|---|
| build receipt | 3,176 | `d626ff3cec6bfeaf7160520244af560a4ce43a3352a1fa0cec331ed43746804c` | conditional, no burn |
| MAIN fire order | 18,486 | `7c8a7d3d0c1cd06bf3cd4a38a06856c5d9ea5ec4789b34facc3f3ac9fc5d0d6c` | sealed, blocked on smoke/claims |
| real-config preflight | 1,231 | `2537dd4c3ffb755446783936934cb69d995d1905493c1bc7c3a61f0d45090b9f` | PASS |
| adjudication schema | 876 | `e78aa8a34cc0222560ed356b8e30681caa5e154de3099e2625521dafd7c81c64` | preregistered |
| two-pass review receipt | 1,035 | `c1cf529f79799fdddd6ac55b016d73f64deea7e58a99d762f199b2916c272b4d` | PASS, no override |
| same-start r10 EMA state | 398,687 | `991a1cc653c786affb607347def53b9dc91176e6ffd043f500076b5c35bf27b0` | strict tensor-key round trip |

All files above live under
`/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/`. Every build payload was retained.

## Source and SHA recovery

| role | exact source | bytes | SHA-256 / identity |
|---|---|---:|---|
| QBT trainer | `experiments/ddm_qbt1_qbflow_trainer.py` | 179,146 | `5da466af9f64295d6bc9d1242ed427724faec586c53d450d3e358e7f1c7492c8`; byte-identical to commit `108fcd41f7` |
| CE1 exact target margin | `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py` | 12,823 | `ffdf098801863ff8bffe8bd818ce101928dd75b4937cbbffb2e225bddbc12f4b` |
| W96B registered law selector | `src/tac/witness_dsl/w96b_aligned_loss_levers_20260826.py` | 3,078 | `053bd12e198bb74a44036e497a1277d9d36638c96acdabba278a2c72f2234923` |
| r10 authorized config | APDataStore QBT root | 6,641 | `87eff6e8cc0339c8b669de9f714e8c666d13a9a8f406a396245540e774c200e9` |
| r10 retained result | `governed_n32_r10/RESULT.json` | 23,090,441 | `9d769f0dd95e76e40ed817aece5b3d608b8b72c3e4bd643793a59ccf0e31354d` |
| r10 safe-run status | `governed_n32_r10/resource_safe_run_status.json` | 1,181 | `ef65dc03210ac38e3c0a69e8264ef28d593d1567e9fe0213d58548aeb628a8cb` |
| r10 stage end | `governed_n32_r10/.../stage_03_end.pt` | 9,476,765 | `09fd416531c74f69ca7033cf3f13b23c9e0472486a97ce9973f62f2fb86c138f` |
| exact born archive comparator | `/Volumes/APDataStore/pact/ddm_br2/inputs/archive.zip` | 106,832 | `0e2ffdfaa5fe481d481dd70a9672a67f80b9aad7648f0c775fe2956dd3a4841d` |
| exact born packet comparator | `/Volumes/APDataStore/pact/ddm_br2/inputs/packet.qbf` | 106,724 | `8c26684d33313ca44f3d4f02cf3c369f0f33d6de37eeba42ae4220faed3e6d38` |

The born archive and packet are adverse comparators, not training inputs and not score components
for a QBR cell. Each QBR cell starts from the r10 endpoint EMA, and each cell's own QBF1 archive is
re-encoded and retained at every milestone. BR2 distortion is never transferred.

The aligned Seg loss is exactly
`mean(sigmoid(-(z_target - max_other)/tau))`, using the CE1 target-versus-best-other margin. Tau is
linear from 0.15 to 0.05 over exactly 5,000 updates. Exact argmax is piecewise constant and has zero
gradient almost everywhere, so the registered CE1 expected-flip law is the necessary differentiable
continuation; every milestone uses exact argmax after QBF1 RGB, bicubic camera resize, uint8 STE, and
the frozen scorer path.

## Re-derived runtime

The retained r10 safe run took 21,387.928 s. Its result contains 10,010 actual optimizer updates,
not the authorized 10,020-step cap. Therefore:

| quantity | re-derived value |
|---|---:|
| seconds per actual optimizer update | **2.1366561438561438 s** |
| six cells x 5,000 updates | **64,099.68431568432 s = 17.805467865467865 h** |
| six BR2-rate realizations x 484.769 s | **2,908.614 s = 0.8079483333333334 h** |
| total before build/launcher overhead | **18.613416198801197 h** |

This replaces the ladder's rounded 2.135 s/update and 18.600 h estimate.

## Fair-form cell design

The charter contains one label conflict: its scope line calls 100/100 the treatment and 100/0 the
control, while its falsifier, prior-law prediction, and owning bx1 memo all ask whether the
**zero-native treatment** beats control. The configs resolve this scientifically rather than leaving
implicit labels:

| arm | role | realized expected-flip weight | native-interface expected-flip weight | pose |
|---|---|---:|---:|---|
| `control_native100` | control | 100 | 100 | `sqrt(10*weighted_pose_mse)`, active from update 0 |
| `treatment_zero_native` | treatment | 100 | 0 | same pose term, active from update 0 |

Nothing else differs within a seed. Both arms use the same 398,687-byte r10 EMA state, AdamW
`lr=2e-4`, reviewed r6 `existence_majority` born gate lineage, Lane/Movable primal-dual constraints
(`0.12`, `0.009`, eta `0.11387788414126129`), and the same seed-specific two-chunk order.

Seeds are `20260902`, `20260903`, and `20260904`. Each schedule gives both fixed 16-pair chunks
exactly 2,500 updates:

| seed | schedule SHA-256 over 5,000 uint8 chunk IDs |
|---:|---|
| 20260902 | `e4f019a06de096773c12eb57bd5ca7620dca26fc8bb9e9a8a8736db7721edac5` |
| 20260903 | `bcfeea9c2f88a767d8f305851f1de5cc3991944e015835f31feb1b5c641151d0` |
| 20260904 | `5eb63e6173bc3e20c9b747562d44b0c6ba3d337108d72b6830a60c63f321cc8c` |

EMA is derived through registered equation `ema_decay_run_geometry_v1` with `U=5000` and terminal
seed fraction `0.01`: `d = 0.01**(1/5000) = 0.9990793899844618`. No fallback was used.

Periodic checkpoints occur every 16 updates, below the 1/300 crash-loss ceiling. They contain live
weights, optimizer, EMA, RNG, constraint multipliers, completed-update cursor, and a verified byte
prefix of the append-only history. Resume truncates only a post-checkpoint history tail and executes
`range(completed_steps, total_steps)`; it cannot extend the schedule. Stage end is preserved as
`stage_01_end.pt`. Milestones are exact at steps `0, 1000, 2000, 3000, 4000, 5000`, and each retains
all 32 camera pairs, fp16 logits, argmax/targets, Pose6/targets, QBF1 packet/archive/repeats, and reset
coder candidates.

The six config SHA-256 values, in fire order, are:

1. `c90fcc6aadce330a26cfb0461c724c8db2adc40e31c102d98d8566744561f937`
2. `ccd907bfa8b5f0aba3f78f13830215d268332b8092d3b78ad7b2d51a0c4aa834`
3. `e71fa3a1490e99a5d8d0ee22604c14aead7046b5988db4a77f183e56a398691f`
4. `f06744b254d36424680cddeaf0e68280af59a293c6e979fa710b96cbf49a6416`
5. `dddaa1baa7595b3c65804ac6a06ae3ca8e10da0c1cafe55d31d0193a30284863`
6. `3a4c29a7d4c5cdca5f1d2e72f8ce0f6a9d3e338c9d7ae29fba8d6de0fff3c1f8`

All six revalidated from disk. They have `launch_authorized=false` and null scorer/Metal claim IDs.

## Review receipts

`experiments/ddm_qbr1_born_fairform_burn_prep.py` and its test received two genuine review passes
after every Python edit. Final checks: five focused tests passed, Ruff passed, formatting passed,
`py_compile` passed, direct-script CLI import/help passed, diff-check passed, and both files were
marked by the AST review tracker on each pass. No review-gate override was used. The serializer
landings are `db1f0bdd93`, `74886c5026`, `b342e752b9`, and `84ca4f32a5`; the later commits are narrow
repairs discovered by actual build/handoff review, not hidden amendments.

## Real-config memory and storage preflight

No B=8 surrogate was used. The exact retained r10 parent is the real 16-pair QBF1/frozen-scorer
graph and measured peak RSS 2,572,632,064 B (`2,512,336 KiB`) against the 116 GiB host ceiling.
The 100/100 control uses the same graph; 100/0 removes one weighted loss edge without increasing
materialization. A fresh scorer smoke was not run by this arm because the scorer lane belongs to
MAIN.

At final seal time APDataStore had 39,253,704,704 B available. The conservative six-cell projection is:

| retained class | projection |
|---|---:|
| checkpoints per cell | 314, priced at the measured 9,476,765 B stage-end size |
| n32 realized payload per milestone | 121,825,988 B measured |
| coder allowance per milestone | 8 MiB |
| projected per cell | 3,756,991,786 B |
| projected all six cells | 22,541,950,716 B |
| required post-projection reserve | 8,589,934,592 B |
| verdict | **PASS** |

No artifact was deleted or moved.

## MAIN fire order

Fire order SHA is `7c8a7d3d0c1cd06bf3cd4a38a06856c5d9ea5ec4789b34facc3f3ac9fc5d0d6c`.
It first gives MAIN an executable `resume-smoke` command. That command requires a real scorer claim ID,
runs the control at CPU/B=16 for two updates uninterrupted and as one update plus resume, retains all
per-update camera/scorer payloads and both endpoint coder payloads, and requires equality of the live
state, EMA state, archive SHA, and final cursor `2`. It cannot run with the placeholder claim.

After smoke PASS and qxr1/QXO1 consumption, MAIN copies each immutable config, binds current unique
scorer and Metal claim IDs, and launches the six cells sequentially with
`tools/launch_detached_process.py --derive-resource-budgets`, a measured RSS basis, an 18,000-second
per-cell wall cap, durable PID/manifest/logs, and `DONE.json`. The order is control then treatment for
each seed. No cell may start from a prior cell's endpoint.

qxr1/QXO1 is deliberately absent from config identity. Its realized distortion is an input to MAIN's
decision whether this rank-2 row remains worth firing; it is not evidence for or against either QBR
arm and cannot alter their treatment tuple.

## Preregistered mechanical verdict

Every endpoint uses the fixed no2 stratified n32 estimator over population 600 and each cell's own
exact QBF1 archive bytes:

`S_hat = 100*d_seg_hat + sqrt(10*d_pose_hat) + 25*archive_bytes/37,545,489`.

A seed is a treatment win only when `treatment_zero_native.S_hat < control_native100.S_hat` at step
5,000. The cell's pose corner passes only when the remainder is positive and
`d_pose_hat < (0.12 - rate_exact - 100*d_seg_hat)^2/10`.

- `OPTIMIZATION_LIVE_DISTORTION_ROUTE`: treatment wins at least two seeds and its pose corner passes
  at least two seeds.
- `OPTIMIZATION_CLOSED_CHANGED_CAPACITY_OBJECT_ONLY`: treatment wins zero seeds, or its pose corner
  passes zero seeds.
- `INCONCLUSIVE_MIXED_NO_FAMILY_CLOSURE`: every other combination.

No n600 buy is authorized before the treatment sign repeats. These are n32 advisory outcomes, not
contest scores or pointer rows.

## RECALL EVIDENCE

The bounded recall covered the charter and common contract; `PROGRAM.md`; `AGENTS.md`/`CLAUDE.md`;
the operating manual; live `main_hot_state.md`; the bx1 owner memo; BR2 result/custody; qbt2b r5-r10
verdicts, configs, checkpoints, safe-run receipts, and packet schema; W96B/task-#1301 source and tests;
NO2 selection/weights; canonical equations registry/listing; the sub-0.15 DAG/index; specs/design
records; and canonical task-status/ledger scopes.

Relevant recovered facts were the r6 reviewed birth-gate revision, r10's actual 10,010-update
denominator, CE1 source SHA, the registered EMA LawRef, the fixed 32-pair selection and weights,
BR2's exact retained bytes, and qxr1's same-object realization boundary. Numeric task `#1301` was not
found in the searched canonical task-status store; its requirement was recovered from the W96B
source module, tests, and aligned verdict receipt. No broader claim of absence is made.

## NEXT_IF_RESUMED

- **Disposition `SEALED-BLOCKED-ON-MAIN-SCORER-LANE`; owner MAIN; consumer store `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/resume_smoke/`; fire trigger:** MAIN has a unique live scorer claim and has not started any six-cell burn; run the fire-order `resume-smoke` command and require PASS on cursor/live/EMA/archive identity.
- **Disposition `SEALED-AWAITING-MAIN-LIVE-CLAIMS`; owner MAIN; consumer store `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/runs/`; fire trigger:** resume smoke is PASS, MAIN has consumed qxr1/QXO1 realized distortion and still finds QBR1 non-dominated, and unique live scorer plus Metal claims are bound; fire the six detached cells sequentially in the sealed order.
- **Disposition `AWAITING-SIX-CELL-RESULTS`; owner MAIN or its named harvester; consumer store `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ADJUDICATION_RESULT.json`; fire trigger:** all six `RESULT.json` files are complete with step-5000 milestones and retained payload hashes; invoke the sealed `adjudicate` command.
- **Disposition `CONDITIONAL-N600-BUY`; owner MAIN; consumer store `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/n600/`; fire trigger:** the mechanical result is `OPTIMIZATION_LIVE_DISTORTION_ROUTE`; build a same-object retained n600 ticket rather than transferring n32 or BR2 distortion.

## LIVE-HYPOTHESES

- Removing the native-interface term may improve at least two same-start seeds because it stops an
  internal partition proxy from competing with the only score-facing realized path; this is plausible
  because 100/0 changes one gradient source while holding bytes, initialization, pose, constraints,
  and sample exposure fixed.
- Joint pose supervision may bring the changed object inside its byte-conditioned pose corner because
  pose is active through the rendered RGB/scorer path from update zero; BR2's raw-object pose failure
  did not test this finish.
- A zero-of-three treatment result would type the remaining wall as capacity/object rather than this
  optimizer coupling because the multi-seed fair-form treatment removes the known single-seed and
  proxy-loss confounds.
- qxr1/QXO1 may make this burn dominated before it starts; that is plausible because it is a fresh
  same-object realization and is therefore correctly consumed by MAIN before spending 18.61 hours.

## DEAD-ENDS

- The exact 106,832-byte BR2 archive is closed at instance scope: its measured distortion is about
  1,045.997 times its lawful sub-0.12 allowance. Its distortion cannot be transferred to QBR cells.
- Raw r5 balanced CE is closed as a start law: it caused Road-to-Lane over-paint and is replaced here
  by the reviewed existence-majority born lineage.
- One-seed or two-seed inference is closed for this discriminator: all verdict-bearing treatment
  conclusions require the three sealed seeds.
- Trusting the ladder's rounded 2.135 s/update is closed: the retained r10 result establishes
  2.1366561438561438 s per actual update and 18.613416198801197 h before overhead.
- Running a scorer smoke or Metal burn from this arm is closed by ownership: MAIN must claim and fire;
  an occupancy name was intentionally not frozen into the seal.
- A process-survival-only resume check is closed: the executable smoke requires exact final cursor,
  live-state, EMA-state, and archive equality and retains the materialized payloads.

**Own-vehicle frontier:** **NOT MOVED** — QBR1 produced no exact authority row; the current exact local
frontier remains AFR1 at `S=0.14797617125559104`, `180,002 B`, archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`
`[contest-CUDA T4 n600]`.

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `ema_decay_run_geometry_v1` — `tac.canonical_equations.ema_decay_run_geometry_20260717` (`tac.canonical_equations`). **Relation:** IN-DOMAIN (the build resolves its decay through this law).

The sealed fair-form burn takes its EMA decay from run geometry rather than the legacy 0.997 constant — the law's whole point ([[m21]] constants→laws). BUILD-COMPLETE and BURN-NOT-FIRED, so this memo consumes the law and adds no anchor to it.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
