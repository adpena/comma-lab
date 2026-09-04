# DDM NG1 — the next burn generation's first race: the TRANSITION itself

**Date:** 2026-09-04
**Arm:** `ddm_ng1_warm_transition_burn_design`
**Axis:** `[seal + bounded macOS-CPU mechanism smoke only; no Metal, no Modal, no contest eval]`
**Disposition:** **SEALED / BOUNDED-SMOKE-PASS / BURN-NOT-FIRED — MAIN fires.**

## Result first

The QBR1 cold-transition hypothesis survives, but **half of its stated premise is false at source
and the correction makes the race sharper, not weaker.**

The charter and the sibling memory both describe the QBR1 cells as entering cold on *two* axes —
a fresh optimizer **and** a transferred learning rate — recovering "with the LR anneal". Verified
at source:

* **There is no LR anneal.** Neither `experiments/ddm_qbt1_qbflow_trainer.py` nor
  `experiments/ddm_qbr1_born_fairform_burn_prep.py` constructs any scheduler. The learning rate is
  set once, at `ddm_qbr1_born_fairform_burn_prep.py:506`, and never touched again for 5,000
  updates. The only annealed quantity in the whole run is the expected-flip temperature
  `tau`, linear 0.15 → 0.05 (`ddm_qbt1_qbflow_trainer.py:622-626`).
* **The LR is not cold either.** r10's authorized config carries `learning_rate = 0.0002`
  (`AUTHORIZED_N32_R10_10020_20260829.json`), and r10's own retained optimizer records
  `param_groups[0]["lr"] = 0.0002` at step 10,010. With no scheduler anywhere, **2e-4 IS this
  object's terminal LR.** There is nothing to move it to.

So the transition into a QBR1 cell is cold in **exactly one** respect: the AdamW moment state
(`exp_avg`, `exp_avg_sq`) and its bias-correction step counter are rebuilt from zero
(`ddm_qbr1_born_fairform_burn_prep.py:498-506` loads only `initial_state["state_dict"]`, i.e.
weights, then constructs a fresh `torch.optim.AdamW`). That is a genuinely single-lever race, and
this arm seals it.

**The intervention needs no change to the sealed training loop.** The warm state is delivered as a
`completed_steps = 0` `resume_from` checkpoint, which the sealed `_load_checkpoint`
(`ddm_qbr1_born_fairform_burn_prep.py:382-418`) consumes — restoring `optimizer_state_dict` and
returning `completed = 0`, so the cell still runs the full 5,000 updates and still emits its
step-0 milestone. The sealed loop is byte-unmodified.

## Verified at source (every premise carries `path:line`)

| claim | evidence | label |
|---|---|---|
| the cell builds a FRESH AdamW; only weights come from `initial_state` | `experiments/ddm_qbr1_born_fairform_burn_prep.py:498-506` | MEASURED |
| no LR scheduler exists in either module | `ddm_qbt1_qbflow_trainer.py` / `ddm_qbr1_born_fairform_burn_prep.py`, zero matches for `lr_scheduler`/`LambdaLR`/`CosineAnnealing`/`OneCycleLR` (regression-tested) | MEASURED |
| the only anneal is tau, linear 0.15 → 0.05 | `ddm_qbt1_qbflow_trainer.py:622-626`, applied at `ddm_qbr1_born_fairform_burn_prep.py:545-550` | MEASURED |
| tau geometry is structurally frozen (any other refused) | `ddm_qbt1_qbflow_trainer.py:2316-2320` | MEASURED |
| r10's terminal LR = 2e-4 = the cell LR | `AUTHORIZED_N32_R10_10020_20260829.json` `learning_rate`; r10 `stage_03_end.pt` `optimizer_state_dict.param_groups[0].lr` | MEASURED |
| r10 retained a full AdamW state: 44 params, 40 state entries, every `step` = 10,010 | `governed_n32_r10/.../stage_03_end.pt` | MEASURED |
| the QBF1 twin has exactly 44 parameters, so the state is index-compatible | `qbt.load_initial_model(cpu)`, `len(list(model.parameters())) == 44` | MEASURED |
| r10's AdamW hyperparameters equal a freshly built one (betas/eps/weight_decay/decoupled) | seal receipt `learning_rate_is_not_a_lever` | MEASURED |
| the cell start weights are r10's EMA **shadow**, and `‖live − shadow‖/‖live‖ = 8.4488e-03` | `build_initial_state`, `ddm_qbr1_born_fairform_burn_prep.py:157-183`; distance computed by this arm | MEASURED |
| checkpoints carry `optimizer_state_dict` | `ddm_qbr1_born_fairform_burn_prep.py:354-379` | MEASURED |
| `config_identity` ignores `action/output/resume_from/launch_authorized/scorer_lane/metal_lane` | `ddm_qbr1_born_fairform_burn_prep.py:145-155` | MEASURED |
| `_run_resume_smoke_segment` forces `device="cpu"` | `ddm_qbr1_born_fairform_burn_prep.py:673-679` | MEASURED |
| `storage_preflight` refuses any output outside four AP custody roots | `ddm_qbt1_qbflow_trainer.py`, `storage_preflight` | MEASURED |
| the training loop consumes no global RNG (`schedule_for_seed` uses a local PCG64; no dropout/`torch.rand`) | `ddm_qbr1_born_fairform_burn_prep.py:118-129`; trainer has no `torch.rand*`/`Dropout` | MEASURED |

### Why a cold AdamW is violent on a converged argmax field (DERIVED, and now regression-tested)

At step 1 a fresh AdamW has `exp_avg = (1-β₁)g` and `exp_avg_sq = (1-β₂)g²`; after bias correction
`m̂ = g`, `v̂ = g²`, so the update is `lr·g/(|g|+ε) ≈ lr·sign(g)` — **a full `lr`-sized displacement
on every parameter, independent of gradient magnitude.** With `lr = 2e-4` over 44 tensors that is a
fixed jump, taken from a checkpoint that had converged into a narrow basin. A warm AdamW carries
`v` from 10,010 prior updates, so the same gradient produces a step smaller by the ratio of the
current gradient to the historical RMS. Both halves are asserted as tests
(`test_a_fresh_adamw_is_cold_and_takes_a_full_lr_sign_step`,
`test_a_warm_adamw_first_step_is_scaled_by_its_second_moment`). The bounded smoke below measures
the ratio on the real graph rather than the toy.

## The cold control of record (MEASURED, already run — no re-burn needed)

Seed 20260902 `control_native100`, the live QBR1 chain's completed cell. Read-only from
`/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902/control_native100/milestones/step_*/MILESTONE.json`.
Axis `[macOS-MPS n32 stratified advisory; not contest authority]`.

| step | S_hat | d_seg_hat | d_pose_hat | archive bytes |
|---:|---|---|---|---:|
| 0 | 0.39876797285867277 | 0.002518335978190104 | 0.0005757456120606528 | 106,714 |
| 1,000 | 0.46687521208987615 | 0.003051122029622396 | 0.0008233354187810106 | 106,667 |
| 2,000 | **0.48567677825279465** | 0.0032170613606770835 | 0.000864393511532432 | 106,626 |
| 3,000 | 0.47538291701253005 | 0.003139241536458333 | 0.0008181846911522883 | 106,637 |
| 4,000 | 0.44219037073377010 | 0.0029336293538411457 | 0.0006051119375803525 | 106,687 |
| 5,000 | 0.42514878445269977 | 0.002758916219075521 | 0.0006122744215585018 | 106,643 |

The peak is at step 2,000; recovery is monotone 2,000 → 5,000 and stops **above** the start.
Recomputed from components, not read off a summary field (the seal receipt and a test both assert
the decomposition sums back to the S_hat delta at `rel=1e-9`):

**Endpoint excess = +0.026380811594027 (+6.6156%)**, and it decomposes as

* `100·Δd_seg` = **+0.024058024088541685** — **91.20%**
* `Δ√(10·d_pose)` = +0.0023700634911569818 — 8.98%
* `Δ rate` = −0.00004727598567167417 — −0.18%

**The damage is d_seg.** The rate term moved 71 bytes across the whole run. Any story about this
excursion that is not a story about the argmax boundary is the wrong story.

Two further facts MAIN should budget with, both MEASURED from the completed control rather than
from the ladder's projections:

* realized wall clock **10,604.52 s = 2.9457 h** for one 5,000-update cell (`RESULT.json`
  `elapsed_seconds_this_process`) — below the derived 10,683.28 s and far below the 18,000 s cap;
* realized retained payload **1,343,104 KiB = 1.375 GB** per cell — versus the conservative
  3.757 GB projection the QBR1 build used. APDataStore had 24 GiB free at seal time, so one warm
  cell fits alongside the live chain's remaining cells with the 8 GiB reserve intact.

## The warm form chosen

**Carry r10's `optimizer_state_dict` verbatim.** The charter allowed a "least-cold alternative"
(momentum re-warm from a short frozen-LR window) if no optimizer-state checkpoint of r10 existed.
One does exist and it is index-compatible, so the alternative is unnecessary and was not used.

Well-posedness, MEASURED: the moments were accumulated around r10's **live** weights, but the cell
starts from r10's **EMA shadow**. The two differ by `‖live − shadow‖/‖live‖ = 8.4488e-03` — under
1% — so the second-moment scale transfers to the start point without a change of regime. The worst
per-tensor relative gap is 7.32e-02 on `params.coarse_logits_b` (5 elements).

### What is deliberately NOT carried (the one-lever discipline)

| available | carried? | why |
|---|---|---|
| `exp_avg`, `exp_avg_sq`, `step` | **yes** | the lever |
| `learning_rate` | n/a | already the object's tail (2e-4); changing it would be a second lever |
| tau geometry | n/a | structurally frozen; the trainer refuses any other |
| r10 terminal margin multipliers (`Lane` 0.005040981907324784, `Movable` 0.017331143732962344) | **no** | held at 0.0, identical to the cold control — carrying them would move the objective at step 0, not the transition |
| EMA `num_updates` | **no** | held at 0, identical to the cold control |
| r10 EMA decay/warmup (0.9995405077759483 / warmup=True) | **no** | the cell's sealed LawRef `ema_decay_run_geometry_v1` value 0.9990793899844618 / warmup=False is used unchanged |

Enforced in code: `verify_warm_seed` refuses a seed carrying non-zero multipliers or a non-zero EMA
counter, and `validate_warm_cell` refuses any config diff outside
`{cell_id, output, resume_from, warm_transition}`. Both are tested.

## Same-pins twin — and a working-tree drift MAIN should know about

The warm config is **derived from the control's sealed config by deep copy**, not recompiled. That
was not the original plan; it is a correction forced by a measurement:

> `qbt.verify_pins()` refuses in the working tree. Exactly one pinned input has drifted:
> `.omx/research/SPEC_ddm_qbflow_packet_schema_v1_20260827.md` — worktree
> `7fe5285f6bf6e0289fe4323b9c21abf337c421eed3dcfa46856d7246877ae54a` vs pinned
> `5405ccd499d14d28230874059e47d47f1f2818038519f1b27c97ed9377f132aa`. The sealed-source copy still
> matches the pin. **The working tree cannot currently compile any QBR1-lineage cell.**

Recompiling would therefore either refuse or re-pin the warm cell away from its own control — a
second lever smuggled in as housekeeping. Inheriting the control's `source_pins` and
`source_revision` (`106d0dd0a094dd4c289eba69c8d2c5124e13eb02`) verbatim keeps the pair a
same-pins twin. `verify_inherited_pins` re-verifies every inherited pin against its own recorded
path and sha on disk, which is a stronger check than a fresh compile because it validates the
**sealed tree that will actually run the cell**.

The other four pinned sources were diffed and are **byte-identical** between the working tree and
the sealed tree (`ddm_qbt1_qbflow_trainer.py` `6eda9c20…`, `ddm_qbr1_born_fairform_burn_prep.py`,
`semantic_renderer_oracle.py` `ffdf0988…`, `w96b_aligned_loss_levers_20260826.py` `053bd12e…`).

**Validator argument (DERIVED):** every check inside `qbr1.validate_config` reads only fields
outside `{cell_id, output, resume_from, warm_transition}`, and the control config passed that
validator at QBR1 seal time. Proving the diff is a subset of the allowed set therefore proves the
warm cell satisfies the sealed validator by inheritance. `validate_warm_cell` proves exactly that,
and a test asserts a `learning_rate` mutation is refused.

## Pre-registered falsifiers (fixed before the burn)

1. **PRIMARY — the warm cell must end BELOW its own warm start.**
   `S_hat(5,000) < 0.39876797285867277` **and** `S_hat` below the cold control at every milestone
   1,000 / 2,000 / 3,000 / 4,000 / 5,000 (values in the table above).
   *If it fails:* the cold optimizer transition is **not** the cause and the schedule/objective is.
   The next race is then the LR magnitude alone, holding the transition warm.

2. **SECONDARY, free read — surrogate-vs-exact decoupling under a warm transition.**
   Read `seg_expected_flip_realized` beside `d_seg_hat` at every milestone. **The surrogate is NOT
   in `MILESTONE.json`** (whose keys are `S_hat`, `d_seg_hat`, `d_pose_hat`, `rate_exact`,
   `pair_rows`, `reencode`, …); it is in the append-only `history.jsonl` under `objective`. The
   seal receipt now carries the control's values so the falsifier is executable, not aspirational:

   | step | `seg_expected_flip_realized` (loss) | `d_seg_hat` (exact argmax) |
   |---:|---|---|
   | 0 | — (history starts at update 1) | 0.002518335978190104 |
   | 1,000 | 0.005012067966163158 | 0.003051122029622396 |
   | 2,000 | 0.004330684896558523 | **0.0032170613606770835** ← exact peak |
   | 3,000 | 0.003676881780847907 | 0.003139241536458333 |
   | 4,000 | 0.0034637681674212217 | 0.0029336293538411457 |
   | 5,000 | **0.003253588918596506** | 0.002758916219075521 |

   Sharper than "decoupled": the surrogate is **monotone falling across the entire run
   (−35.1% from update 1's 0.005018)** while the exact argmax rises to a peak at 2,000 and ends
   **+9.56% above its start**. The loss never registered the excursion at all. If that shape
   repeats **in the warm cell**, the defect is the surrogate, not the transition, and vr1 rows 1/4
   become the next race.

3. **NO-OP DETECTOR (settled before the burn, not after).** A warm and a cold first update from the
   identical start must produce different weights. Identical post-update weights would prove the
   seed is inert. Measured by the bounded smoke below.

## Bounded CPU smoke — PASS, and it measures the mechanism

The Metal lane is held by the live QBR1 chain (`pgrep -f ddm_qbr1_cell_chain` → 95296/95299/95317
at seal time), so every segment ran on **CPU**. `_run_resume_smoke_segment` forces `device="cpu"`
itself, and this arm did **not** touch `.omx/state/active_lane_dispatch_claims.md`: the chain reads
that file every poll and a malformed edit would raise `CLAIMS_UNREADABLE` and refuse an 18-hour
burn. Low probability, catastrophic blast radius — so the smoke ran as a bounded local CPU probe
under the canonical `tools/launch_detached_process.py`, 0 Metal / 0 Modal / 0 contest-eval
invocations, **156.99 s wall**.

Four real-B=16 segments from the identical r10 EMA-shadow start:

| check | result |
|---|---|
| warm uninterrupted (2 updates) vs warm interrupted (1) + resumed (2) — `completed_steps` | equal, both 2 |
| — live state sha256 | **equal** (`0087cbc242c1d1d3…`) |
| — EMA state sha256 | **equal** |
| — re-encoded archive sha256 | **equal** (`c0586c1a3fdc2e81…`, 106,735 B) |
| **no-op detector:** warm step-1 vs cold step-1 live state | **DIFFERENT** (`507ec22c…` vs `27f51418…`) |
| — and their archives differ too | 106,735 B vs 106,676 B (`236b9e01…`) |

So the warm seed is genuinely consumed by the sealed loader, it changes the bytes, and a
warm-origin run still resumes bit-identically from a mid-run checkpoint.

**The mechanism receipt (MEASURED on the real graph, not the toy):** first-update displacement
`‖θ₁ − θ₀‖₂` from the shared start —

* cold (fresh AdamW): **0.055886740188786026**
* warm (r10 moments): **0.008653761825381008**
* **the cold first step is 6.4581× larger** (`warm/cold = 0.15484463391760736`)

That is the predicted `lr·sign(g)` blow-up versus a second-moment-scaled step, confirmed at the
actual QBF1 + frozen-scorer graph at B=16. It is a **mechanism** measurement, not a verdict: one
update is not d_seg, and the smoke makes no claim about where step 5,000 lands. That is what the
burn is for.

The smoke was run **twice** (once before and once after the review pass that added the
start-weights and re-seal guards). Both runs returned identical displacements to the last digit
(`0.055886740188786026` / `0.008653761825381008`), 156.99 s and 158.84 s wall — a free determinism
receipt on top of the mechanism one. The seal is likewise byte-stable: three independent `seal`
invocations produced the same config sha `4a40b0f35d4c38065d03f3b05c51943da7b50c05c271c43c369b90013a41325b`
and the same warm-seed sha `5fed6168aa393277f39e4c97a8da2213cc3d0b02f455d80d41faa13a89f4f9ad`.

All smoke payloads were retained (`warm_uninterrupted` 137,344 KiB, `warm_resumed` 137,344 KiB,
`cold_reference` 68,736 KiB) plus all three warm seeds.

## Per-milestone read plan

At each of `0, 1000, 2000, 3000, 4000, 5000`, from the warm cell's own
`milestones/step_*/MILESTONE.json`:

* `S_hat` against the control row and against `0.39876797285867277`;
* the **decomposition** `100·d_seg_hat`, `√(10·d_pose_hat)`, `25·bytes/37,545,489` — never the
  composite alone (the control's damage is 91.20% d_seg; a warm cell that fixes S_hat by moving
  bytes or pose would be a different finding, not this one);
* `seg_expected_flip_realized` beside `d_seg_hat` — from `<run>/history.jsonl` `objective.seg_expected_flip_realized`, **not** from `MILESTONE.json`, which does not carry it (falsifier 2);
* peak location: the control peaks at 2,000. A warm cell that merely *delays* the peak is a
  partial result and must be reported as such, not as a win.

## MAIN fire command

Preconditions MAIN owns: the Metal lane is free (the QBR1 chain has released it), a live scorer
claim and a live Metal claim exist, and the sealed tree is unchanged.

MAIN copies the sealed config to `authorized_configs/`, sets `launch_authorized: true`, and binds
both live claim IDs. Those three fields are in `config_identity`'s ignored set, so binding them
does **not** invalidate the warm seed (asserted by test).

```bash
/Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/sealed_source_106d0dd0_v2/.venv/bin/python \
  /Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/sealed_source_106d0dd0_v2/tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_ng1_warm_transition/launch/seed_20260902_warm_transition \
  --cwd /Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/sealed_source_106d0dd0_v2 \
  --purpose "NG1 warm-transition cell seed_20260902_warm_transition_control_native100" \
  --authority MAIN --derive-resource-budgets --measured-peak-rss-gib 2.3959503173828125 \
  --measured-thread-need 4 --walltime-cap-s 18000 --done-receipt DONE.json \
  -- /Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/sealed_source_106d0dd0_v2/.venv/bin/python \
     /Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/sealed_source_106d0dd0_v2/experiments/ddm_qbr1_born_fairform_burn_prep.py \
     run-config /Volumes/APDataStore/pact/ddm_ng1_warm_transition/authorized_configs/seed_20260902_warm_transition_control_native100.json
```

**Fire from the SEALED tree, not the working tree.** The inherited pins name sealed-tree paths, and
`validate_config` recomputes them from whichever tree runs the cell; the working tree would refuse
on the drifted packet-schema memo. That refusal is the design working, not a bug.

Cost: one cell, **~2.95 h** measured, ~1.375 GB retained. No control re-burn — the control is the
already-measured seed-20260902 row.

## Custody

| artifact | path |
|---|---|
| seal receipt | `/Volumes/APDataStore/pact/ddm_ng1_warm_transition/SEAL_RECEIPT.json` |
| warm cell config | `…/ddm_ng1_warm_transition/sealed_configs/seed_20260902_warm_transition_control_native100.json` |
| cold control of record | `…/ddm_ng1_warm_transition/sealed_configs/cold_control_of_record_seed_20260902_control_native100.json` |
| warm seed (burn, mps identity) | `…/ddm_ng1_warm_transition/warm_seeds/warm_seed_mps.pt` |
| bounded smoke result | `…/ddm_qbr1_born_fairform_burn_prep/ng1_warm_transition/resume_smoke/BOUNDED_SMOKE_RESULT.json` |
| run output root | `…/ddm_qbr1_born_fairform_burn_prep/ng1_warm_transition/runs/seed_20260902_warm_transition` |

`authorized_configs/` is **not** written by this arm. Run payloads live under
`QBR1_RETENTION_ROOT` (the original burn-prep root, dormant since 2026-09-02, no open handles) —
never under the live `ddm_wc3_qbr1_ema_law_cure` chain's `runs/`, `authorized_configs/`, or
`CHAIN_LEDGER.jsonl`, and this arm did not touch the claims ledger.

## Equations leg (`tac.canonical_equations`)

**Law cited:** `muon_finisher_schedule_warmstart_and_lr_anneal_v1` — *"Muon finishing-stage
schedule: warm-start momentum + cosine LR anneal"*, 3 anchors, producer
`experiments/train_witness_realized_through_R_mlx.py`, consumer `tac.witness_dsl.gauge`.

**Relation: SIBLING / PREDICTIVE — NOT in-domain, and NO anchor is appended.** The law's domain is
the Muon finisher on the MLX witness vehicle with a cosine LR anneal. QBR1/NG1 is AdamW on the QBF1
born vehicle with a **constant** LR and no anneal at all. Appending a QBR1 anchor here would be
exactly the cross-vehicle constant transfer the campaign has extincted (`[[m21]]` constants→laws,
`[[m143]]` cross-regime transfer, `[[L18]]` ancestor lessons-not-numbers). Its predictive value is
real and is why this arm exists — its anchor `muon_cold_start_transition_dseg_spike_20260703`
(cold first-order buffer → d_seg spike +0.000357) and
`mod32cap_cold_muon_fire_ep726_quench_275pct_20260707` (no warm start → +27.5% quench, never
re-beat the pre-transition best) are the same shape as the QBR1 excursion on a different vehicle.

**FORMALIZATION_PENDING** — the law this finding would need is an optimizer-agnostic one:
*"a stage entered with zero optimizer second-moment state on a converged piecewise-constant argmax
field takes a first step of size `lr` per parameter regardless of gradient magnitude, opening a
d_seg excursion whose recovery is bounded by the remaining schedule."* It generalizes across
AdamW and Muon and does not presuppose an LR anneal. It should be registered only once the NG1
burn returns, so it is anchored on a measurement rather than on this design.

`ema_decay_run_geometry_v1` is consumed unchanged (IN-DOMAIN): the warm seed reuses the control's
sealed decay 0.9990793899844618 and `verify_ema_executable_law` passes on load.

## Scope and limits (these travel with the numbers)

* **Axis.** Every S_hat here is `[macOS-MPS n32 stratified advisory]`. Not a contest score, not a
  pointer row, not promotable. The bounded smoke is `[macOS-CPU]` and is a mechanism check, not a
  verdict.
* **GT lineage.** The QBR1 vehicle pins `gt_n600.npz`, which is the **PyAV** lineage
  (`[[gt_n600_npz_is_pyav_lineage_train_on_dali_20260903]]`). The warm-vs-cold comparison is
  internally valid because both cells sit on the identical lineage; the **absolute** d_seg values
  are not DALI-authority numbers. Changing the GT would have been a second lever and was not done.
* **n = 1 seed.** This is a single-seed transition race, deliberately. It is the first-order fact
  (+0.0264 endpoint excess) against a second-order discriminator (Δ 0.0023), so one seed can move
  the design; it cannot close a family. Seeds 20260903 / 20260904 remain the sign-repeat check.
* **The discriminator is untouched.** `objective.native_interface_weight` stays at 100 (the
  control's value). This arm does not re-litigate QBR1's treatment; it races the substrate both
  arms sit on.

## NEXT_IF_RESUMED

* **`SEALED-AWAITING-MAIN-METAL-CLAIM`** — owner MAIN; store
  `…/ddm_ng1_warm_transition/launch/`; fire trigger: the QBR1 chain has released the Metal, live
  scorer + Metal claims exist; copy sealed → authorized, bind claims, fire the command above.
* **`AWAITING-WARM-CELL-MILESTONES`** — owner MAIN or its harvester; store the run's
  `milestones/step_*/MILESTONE.json`; fire trigger: `DONE.json` present; adjudicate against the
  pre-registered falsifiers, reading the decomposition and never the composite alone.
* **`CONDITIONAL-LR-MAGNITUDE-RACE`** — fire trigger: falsifier 1 FAILS (warm cell still ends above
  0.398768). Then the transition is exonerated and the next single lever is the LR magnitude,
  holding the transition warm.
* **`CONDITIONAL-LOSS-CALIBRATION-RACE`** — fire trigger: falsifier 2 fires (surrogate falls while
  `d_seg_hat` rises in the warm cell). Then the expected-flip surrogate, not the schedule, is the
  defect; vr1 rows 1/4 become the race.
* **`WORKING-TREE-PIN-DRIFT`** — owner MAIN; the working tree cannot compile a QBR1-lineage cell
  until `.omx/research/SPEC_ddm_qbflow_packet_schema_v1_20260827.md` is restored to
  `5405ccd4…` or the pin is deliberately re-cut. Not blocking for NG1 (which fires from the sealed
  tree) but it silently blocks any future working-tree seal of this lineage.

## DEAD-ENDS

* **"Set the LR to the object's own tail" is closed as a lever for this object** — 2e-4 already is
  the tail (r10 config + r10 optimizer both), and no scheduler exists to have moved it.
* **"The LR anneal produces the recovery" is closed as a mechanism** — there is no LR anneal. The
  recovery coincides with the tau anneal (0.15 → 0.05), the only schedule in the run. This memo
  makes no claim that tau *causes* the recovery; that is a separate, unraced question.
* **Recompiling the warm cell from the working tree is closed** — one pinned input has drifted, so
  a fresh compile would re-pin the cell away from its own control.
* **Using r10's stage-end checkpoint directly as `resume_from` is closed** — `_load_checkpoint`
  requires `config_identity` equality, and r10 differs in seed, steps, curriculum mode and output.
  The warm seed re-expresses r10's optimizer state under the *cell's* identity instead.
* **Carrying r10's margin multipliers or EMA counter is closed for this race** — both are second
  levers and both are refused in code.

**Own-vehicle frontier:** **NOT MOVED** — this arm designs and seals; it produced no exact
authority row. `afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED`.

## LIVE READ (MAIN, 2026-09-04 11:44Z) — warm cell @1k
S_hat **0.439328** (control @1k 0.466875; warm start 0.398768) · d_seg_hat 0.0029868 (control 0.0030511) · d_pose_hat
**0.0004834** (control 0.0008233; BELOW the start's 0.0005757). Below the control at the first milestone (−0.027547 S_hat)
but still +10.2% above the warm start — the excursion is smaller, not absent: the optimizer state removed part of the
first-order damage, and the pose leg improved outright. Falsifier 1's second clause (below control at EVERY milestone)
holds at 1k; the first clause (S_hat(5k) < 0.398768) is open. Next read at 2k (the control's peak).
**@2k (12:21Z):** S_hat **0.467442** (control @2k 0.485677, the control's peak; Δ −0.018235) · d_seg_hat 0.0031367
(control 0.0032171) · d_pose_hat 0.0006851 (above the start now). Below the control at both milestones so far, but
+17.2% above the warm start: the warm optimizer state DAMPS the excursion (≈ −0.02 S_hat at the peak) and does not
remove it — the transition is a contributor, not the sole cause; the objective/schedule (over-paint, τ band) carries the
rest, as sd1/gm1 said. Clause 1 of falsifier 1 (S_hat(5k) < 0.398768) is now unlikely; the paired-effect read at 5k is
the decision: warm S_hat(5k) < 0.425149 by more than 0.005 ⇒ WIN for composition (re-seal ng2/ng3 as WARM twins).
**@3k (13:16Z):** S_hat **0.461163** (control @3k 0.475383; Δ -0.014220) · d_seg_hat 0.0030496 (control 0.0031392) · d_pose_hat 0.0007258.
**@4k (13:48Z):** S_hat **0.458423** (control @4k 0.442190; Δ **+0.016233** — the warm cell is now ABOVE the control) ·
d_seg_hat 0.0030151 (control 0.0029336) · d_pose_hat 0.0007383. Falsifier 1's second clause ("below the control at every
milestone") FAILS at 4k: the warm optimizer state damps the excursion's onset (−0.027/−0.018 at 1k/2k) but the cold
control RECOVERS faster from 3k on. Read: the moments change the transient's shape, not its cause; the objective/
schedule (over-paint; τ band) owns the recovery. The 5k paired read decides the composition rule.
