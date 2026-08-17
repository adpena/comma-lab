# ddm_cl2 — the cl1 `rate_lambda` rung: ceiling priced on hv1, seal re-cut

Date: 2026-08-16 · Owner: `ddm_cl2_rate_lambda_seal` · Axis: `[macOS-CPU advisory / scorer-free
byte + source read]` · `score_claim=false` · `promotable=false` · pointer UNMOVED at
**0.15959729295498598 @ 182,759 B**. Payloads: `/Volumes/APDataStore/pact/ddm_cl2_rate_lambda_seal_20260816/`.

## Verdict

**The in-network branch survives its ceiling — but by 1.33×, not by the 26× the charter implies,
and the sealed fire order is REFUSED by its own Gate 0 today.**

Three findings, in the order they change a decision:

1. **The 3.810 in my charter is an AVERAGE, and I was asked to use it as a MARGINAL.** It is
   `51,484 / 13,515` — the return of the **whole** prior over `[0, M0]`. Because `tokens(model)` is
   convex decreasing, that average is a hard **upper bound** on the marginal, never an estimate of
   it. The only measured near-point marginal is **1.15**. Pricing the rung at 3.810 over-states the
   prize by up to **3.31×**.
2. **I measured the fixed-topology ceiling on the LIVE vehicle, at $0.** The per-channel bit depth
   is hard-capped at **8**. hv1 sits at **55.53%** of that cap. Total reachable growth is
   **+9,078.8 raw B / +6,834.9 counted B**. Clearing the 14,413.402 B bar therefore requires a
   **sustained** marginal of **2.588–3.109** — 68–82% of the theoretical maximum, held all the way
   to bit saturation, while convexity guarantees it decays. At the measured 1.15 the entire family
   is worth **9.45% of the bar**.
3. **The seal is stale. 4 of its 7 committed-content hashes have drifted.** Gate 0 says "refuse if
   the committed hashes no longer match." They do not match. Separately, VertigoDataTier — which
   every path in the fire order writes to — is at **100% capacity, 893 MiB free**, tripping the fire
   order's own 30 GB refusal.

Net: **do not close at $0** (ceiling is 1.33–1.77× the bar) and **do not fire as sealed** (three
hard blockers). §6 is the re-cut ticket, with a pre-registered kill threshold that turns the rung
into a decisive test rather than a hopeful one.

## 1. The apparatus, verified at source

Built, and **further along than the memory records.** `[[the_counted_byte_is_not_fungible…]]` and
`ddm_hm1:276` both say cl1 "never fired." Its λ=1.0 leg is in fact **complete through Gate 3**.

| element | location | content |
|---|---|---|
| the control | `tools/train_ddm_cl1_hpac_capacity.py:818` | `parser.add_argument("--rate-lambda", type=float, default=1.0)` |
| **the consumer** | `tools/train_ddm_cl1_hpac_capacity.py:1235` | `rate_loss = args.rate_lambda * math.log(2) * variable_weight_bits(model, deployed=False) / pixels` |
| positivity guard | `tools/train_ddm_cl1_hpac_capacity.py:898-899` | refuses `λ ≤ 0` |
| admission whitelist | `tools/train_ddm_cl1_hpac_capacity.py:460-466` | refuses any λ outside the preregistered set |
| the ladder | `tools/train_ddm_cl1_hpac_capacity.py:111-114` | `cl1: {1.0, 0.5, 0.25}` · `rx2_mc36: {1.0}` |
| one-factor by construction | `tools/train_ddm_cl1_hpac_capacity.py:649` | `schedule_config` excludes `rate_lambda`; every other key is pinned by `_assert_preregistered_config` |
| launch path | `tools/spawn_durable_daemon.py` → the trainer | governed; `TAC_ADMISSION_ENFORCE=1`, `PYTORCH_ENABLE_MPS_FALLBACK=0` |
| bit cap (the ceiling's source) | `…/pr130_eureka_intake_20260806/repro_repo/code/hpac_self_compress.py:53` | `max_bits = ceil(log2(2*weight_bound+1))` → **8** at `weight_bound=127` |
| depth storage | `integer_model_io._unpack_nibbles` | 4-bit nibble per channel; the training cap of 8 is tighter |

`rate_lambda` prices **weight bits only** — the topology (`channels 64 / patch 64 / delta 2 /
frame_dim 8`) is frozen. Lower λ buys deeper channels, nothing else. That is what makes the ceiling
in §4 computable.

**Banked, on SSD, real:** Gates 0–3 ran for λ=1.0 twice — a SIGKILL/resume control and an
uninterrupted twin — each with real pack, encode and decode plus owned-runner attestations.
`GATE2_EQUALITY_ADJUDICATION.json` records `resume_integrity: PROVEN bit-faithful: SIGKILL at epoch
1 -> resume -> epoch 60 == uninterrupted fresh run on every model/EMA/optimizer tensor`. The
CLAUDE.md per-stage-checkpoint and EMA-shadow non-negotiables are satisfied **and proven**, not
merely declared.

| λ=1.0 uninterrupted twin | value |
|---|---:|
| raw model bytes | 20,049 |
| compressed (counted) model bytes | 15,088 |
| token bytes (real Range) | 116,716 |
| joint | 131,804 |
| decoded raw token sha256 | `c5c7671d…` |

Only **Gate 4** — λ=0.5, and λ=0.25 conditional on the first secant clearing −1 — remains.

## 2. The control design

The estimand is one fixed point per λ: the immutable epoch-60 QAT stage checkpoint, real-packed,
real-encoded, exactly decoded. The comparison varies **exactly one thing**: `schedule_config`
(`:649`) strips `rate_lambda` and the fitter requires full run identity modulo only that key, so
seed, schedule, init, cache and topology are byte-pinned across rungs. This is not a 2×2 on the
diagonal — it is a one-factor ladder, and λ=1.0 is its own already-banked baseline.

One caveat MAIN will hit: `GATE2_EQUALITY_ADJUDICATION.json` records an instrument defect —
`_causal_state_sha256` (`tools/train_ddm_cl1_hpac_capacity.py:180`) folds in `run_identity`,
including `launch_git_sha`. Headline hashes will differ across rungs launched at different HEADs.
**The Gate-3 packed-EMA / Range / decoded-token / logit hashes are the true one-factor test.** Do
not read a headline mismatch as a failed rung.

## 3. The level error — why the charter's arithmetic over-prices the rung by 3.31×

`ddm_dc1:213` states it plainly: the prior costs *"13,515 counted bytes … and returns 51,484 bytes
— +3.81 bytes returned"*. That is `ΔT/ΔM` over the **whole interval `[0, M0]`**.

For a convex decreasing `T(M)` — diminishing returns from capacity, which every rung in `ddm_hm1`'s
ladder exhibits — the secant over `[0, M0]` is **at least as steep** as the tangent at `M0`:

```
|T'(M0)|  ≤  (T(0) − T(M0)) / M0  =  3.810
```

So 3.810 is a **ceiling on the marginal**, not a measurement of it. Applying it as a marginal is the
same class of error as `ra2`'s superseded 11.5× bar: right number, wrong level.

The bracket, all three points cited from `ddm_hm1:185-189`:

| direction | mechanism | base | slope |
|---|---|---|---:|
| shrink | frame_dim coordinate removal | PR130 | **−1.15** |
| — | shipped RCF1 correction table | hv1 | −3.355 |
| grow | next correction rung | hv1 | −0.016 |

The only point near the operating point in the **model** dimension is **−1.15**, and it is PR130 +
a coordinate mechanism, not hv1 + λ. Treat it as DERIVED, not transferred.

## 4. The ceiling, measured on hv1 itself

I read the shipped frontier archive directly — `archive.zip` sha `80d9c8c6…`, 182,759 B, single
STORED member `p` — materialized its IHS1 HPAC blob (17,952 B) and unpacked the per-channel depth
nibbles. **No forward pass, no scorer, no Metal, $0.**

| module | weight shape | out | mean bits | max bits | current B | at 8-bit cap B |
|---|---|---:|---:|---:|---:|---:|
| `frame_shift` | (64, 8) | 64 | 3.109 | 5 | 199.0 | 512.0 |
| `frame_scale` | (64, 8) | 64 | 3.141 | 6 | 201.0 | 512.0 |
| `conv_a` | (64, 7, 7, 7) | 64 | 4.891 | **8** | 6,299.1 | 10,304.0 |
| `conv_b1` | (64, 1, 5, 5) | 64 | 4.516 | 6 | 505.8 | 896.0 |
| `conv_b2` | (64, 1, 3, 3) | 64 | 4.609 | 6 | 184.4 | 320.0 |
| `conv_past` | (64, 5, 3, 3) | 64 | 4.812 | **8** | 1,732.5 | 2,880.0 |
| `spm_dw` | (64, 1, 3, 3) | 64 | 5.438 | 7 | 391.5 | 576.0 |
| `spm_pw` | (64, 64, 1, 1) | 64 | 3.125 | 6 | 1,600.0 | 4,096.0 |
| `head` | (5, 64, 1, 1) | 5 | 5.600 | 6 | 224.0 | 320.0 |
| **total** | 20,416 live weights | 517 | **4.442** | 8 | **11,337.2** | **20,416.0** |

`conv_a` and `conv_past` already carry channels **at the cap**. hv1 sits at **55.53%** saturation;
the headroom ratio is **1.8008×**.

Structural headroom: **+9,078.8 raw B**, or **+6,834.9 counted B** at the measured pack ratio
0.75284 (13,515 counted / 17,952 raw — within 0.003 of cl1's own 15,088/20,049).

**The arithmetic, in S units.** Bar 14,413.402 B; rate price `25/37,545,489 = 6.658558e-7` S/B;
required cut ΔS 9.5973e-3.

| reading | ΔM | required sustained slope | best-case Δjoint | ΔS | vs bar |
|---|---:|---:|---:|---:|---:|
| raw-equivalent (most generous) | 9,078.8 B | **s ≥ 2.588** | −25,511.3 B | **−0.016987** | **1.77×** |
| counted at the measured pack ratio | 6,834.9 B | **s ≥ 3.109** | −19,205.9 B | **−0.012788** | **1.33×** |
| **at the only measured marginal, s = 1.15** | 9,078.8 B | — | **−1,361.8 B** | **−0.0009068** | **9.45%** |

**Read the last two rows together.** With the marginal pinned at its hard theoretical maximum the
family clears the bar by 1.33×. With the marginal at the only value anyone has measured, the whole
family — every channel driven to saturation — is worth **9.45% of the bar**. For scale, `ddm_hm1`
closed the correction-table branch at **2.47%**.

**So: no $0 close.** The ceiling test that killed the table branch does not kill this one. But it
converts a "26× better placement" story into a narrow, falsifiable requirement: *the marginal must
hold at 68–82% of a maximum that convexity says must decay.*

## 5. Three blockers on the sealed fire order

**B1 — the seal is stale; Gate 0 refuses today.** Verified against `BLOCKED_RECEIPT.md`'s
committed-content table:

| path | expected | actual | |
|---|---|---|---|
| `tools/train_ddm_cl1_hpac_capacity.py` | `0c1e6464…` | `8392a9b9…` | **DRIFT** |
| `tools/fit_ddm_cl1_hpac_capacity.py` | `a66a911a…` | `c5ecafc1…` | **DRIFT** |
| `src/tac/tests/test_train_ddm_cl1_hpac_capacity.py` | `cf4b1a9b…` | `603e7c4c…` | **DRIFT** |
| `.omx/state/lane_registry.json` | `ee9128f0…` | `58cbcbe2…` | **DRIFT** |
| `src/tac/tests/test_fit_ddm_cl1_hpac_capacity.py` | `ab310ef3…` | `ab310ef3…` | match |
| `PREREGISTRATION.md` | `3581a47d…` | `3581a47d…` | match |
| `MAIN_METAL_FIRE_ORDER.md` | `266e69d3…` | `266e69d3…` | match |

Cause, traced: four `rx2` commits (`87d0709b96`, `a6014a67a8`, `e0e9ddce07`, `18fe6d3398`) added the
`rx2_mc36` profile and hardened the resume gate; `76ea0d98f0` fixed the fitter's venv-symlink
resolution. **The drift is benign in substance** — `PREREGISTERED_CONFIG` and the cl1 λ set are
byte-unchanged — but the gate is a hash gate and it fails. Re-seal, do not rebuild.

**B2 — the disk the fire order writes to is full.** VertigoDataTier: **893 MiB free, 100%
capacity**. Every path in Gates 1–4 is a VDT path, and Gate 0's own text refuses if "SSD free space
would fall below 30 GB." Measured cost per rung: **~150 MB** (149 M resume control, 147 M twin).
APDataStore has **230 GiB**. Remap the output roots; leave `--cache`, `--init` and `--intake-code`
on VDT (read-only inputs, sha-pinned).

**B3 — the ladder is pinned to the wrong vehicle.** `fit_ddm_cl1_hpac_capacity.py:81` hard-gates
`EXPECTED_RAW_TOKEN_SHA256 = c5c7671d…` — the **PR130** spatial field. The live frontier's field is
`9ba2e52b…` (`ddm_hm1_hpac_logit_replay.py:62`, matched to wc1/dc1 receipts). The two objects differ
in exactly the dimension that sets the slope:

| | PR130 / cl1 λ=1.0 | hv1 live |
|---|---:|---:|
| live weight elements | 33,024 | 20,416 (38% sparser) |
| bit saturation | **66.23%** | **55.53%** |
| headroom ratio | 1.5100× | **1.8008×** |
| counted model B | 15,088 | 13,515 |
| token B | 116,716 | 112,110 |

hv1 is **less** bit-saturated, so it sits **earlier** on the diminishing-returns curve. A slope
measured on PR130 is therefore **conservative** for hv1: a PR130 "closed" verdict leaves a
false-negative risk on hv1. This is the cross-regime transfer this lab keeps paying for. Firing cl1
as sealed answers a question about PR130 and licenses only a DERIVED read on hv1 —
`verdict_scope: VEHICLE=PR130`.

## 6. The re-cut ticket — fire-ready, with a pre-registered kill threshold

Not sealed as fireable-today: **B1 and B2 must be cleared by MAIN first** (both are chores, not
work). What follows replaces the stale parts of `MAIN_METAL_FIRE_ORDER.md`; Gates 1–3's procedure is
unchanged and its λ=1.0 leg is already banked.

**Pre-conditions (all three, in order).**
1. Metal free. Currently **held**: pid 4832, `experiments/train_tr1_partition_renderer_mlx.py`,
   15.5 GB RSS, 2 h 54 m elapsed, under supervisor 4831. One Metal fire at a time.
2. Re-seal B1: re-run `BLOCKED_RECEIPT.md`'s verification block against current HEAD, then rewrite
   its hash table with the four current values (`8392a9b9…`, `c5ecafc1…`, `603e7c4c…`,
   `58cbcbe2…`). Drop `lane_registry.json` from the gate — it is live state and will always drift.
3. Remap outputs to `/Volumes/APDataStore/pact/ddm_cl1_capacity_20260809/` (~300 MB for both rungs).

**Resource envelope — MEASURED from the banked λ=1.0 receipts, not projected.**

| stage | wall-clock | peak RSS (observed) |
|---|---:|---:|
| training, 60 epochs, MPS | 2,894.155 s | 1,673.391 MiB |
| pack | 2.225 s | 1,325.094 MiB |
| encode | 679.825 s | 1,585.719 MiB |
| decode | 688.595 s | 1,069.500 MiB |
| **per rung** | **4,264.8 s ≈ 71.1 min** | **1,673.4 MiB = 1.634 GiB** |

Ladder total: **71.1 min** (λ=0.5 only) to **142.2 min** (if λ=0.25 fires).

The fire order's DERIVED `--projected-gb 12 / --rss-cap-mb 12288` is **7.3× the measured peak**.
Replace it using the fire order's own formula, `ceil(1.5 × peak_rss_mib / 256) × 256`:

```
--projected-gb 3  --projected-peak-gib 3  --rss-cap-mb 2560  --walltime-cap-s 7200
```

**Gate 4 argv** — the canonical fresh-run block of `MAIN_METAL_FIRE_ORDER.md` §Gate 2 with exactly
this tuple substituted (no other change; every flag verified present in the trainer's argparse):

```bash
DDM_CL1_FRESH_ROOT=/Volumes/APDataStore/pact/ddm_cl1_capacity_20260809/lambda_0p5/training
DDM_CL1_FRESH_LAMBDA=0.5
DDM_CL1_FRESH_LABEL=ddm_cl1_lambda0p5
```

then the same terminal pack → encode → decode sequence, then the fitter. λ=0.25 fires **only** if
the first terminal-QAT Range secant is strictly below −1 and model bytes grow monotonically.

**PRE-REGISTERED THRESHOLD — this is what makes the rung worth a slot.** From §4, on the live
vehicle:

> **If the fitted λ-secant `s < 2.588`, the fixed-topology `rate_lambda` branch is CLOSED** — it
> cannot clear the 14,413.402 B bar even with every channel driven to the 8-bit cap, under the most
> generous byte accounting available. If `s ≥ 3.109` the branch is OPEN and a λ push toward
> saturation is the next rung. Between 2.588 and 3.109 the verdict turns on the pack ratio and needs
> the counted-byte read from the fitter's own pack report.

Prior: `s = 1.15`. **The ladder is most likely a paid negative — and that is its value.** It is the
last live rung on the rate axis after `ddm_hm1` closed the table branch, and this threshold makes
one 71-minute run decisive either way. Fire it when Metal is otherwise idle, exactly as
`ddm_hm1:276` recommends. Do not fire it ahead of anything with a live prize.

## 7. What I did NOT measure

1. **hv1's own λ-slope.** It needs training and Metal, which I may not touch. It is the whole point
   of the ticket.
2. **A widened HPAC.** Topology change, outside `rate_lambda`, and `hpac_integer.py:186` carries
   `channels * weight_bound * activation_bound + 32768 >= 2**24`, which may trip past `channels=64`.
   Untouched by this arm and by `ddm_hm1`.
3. **The convexity of `tokens(model)`.** I assert it as the standard capacity shape and it is
   consistent with every rung of `ddm_hm1`'s ladder, but I did not measure it on this axis. If
   `T` were non-convex near `M0`, the 3.810 upper bound on the marginal would not hold. Labelled
   DERIVED throughout.
4. **hv1's contest-CPU decode time.** No such row exists (`ddm_hm1:209`). Not binding here — a
   deeper bit depth changes arithmetic width, not the graph.

## 8. Receipts and payload

Workspace `/Volumes/APDataStore/pact/ddm_cl2_rate_lambda_seal_20260816/retained/`:

| artifact | sha256 | bytes |
|---|---|---:|
| `hv1_hpac_channel_bit_depths.u8` — hv1's 517 per-channel depths, unpacked from the shipped IHS1 blob | `c7fb362f33efe4194fd93e2366086e15316fe3f5dca31f56f773631f57900910` | 517 |
| `hv1_hpac_capacity_ceiling.json` — per-module table + the full arithmetic | `809cc83038fba01de98bca22ce1e11b8f9f2a8fe515cfe2baebd3ec238db7abd` | 4,581 |

Sources read: frontier `archive.zip` sha `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`
(182,759 B) via `/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/prepared/hv1_base_control`;
banked cl1 λ=1.0 artifacts under `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/`;
`ddm_hm1_model_byte_derivative_20260816.md`; `ddm_dc1_decode_budget_conditional_coding_20260816.md`;
`.omx/research/ddm_cl1_capacity_20260809/{BLOCKED_RECEIPT,MAIN_METAL_FIRE_ORDER}.md`.

## NEXT_IF_RESUMED

| # | row | owner | fire condition | cost |
|---|---|---|---|---|
| 1 | **Stop quoting 3.810 as a marginal.** It is a whole-prior average and a hard upper bound. Any rate charter that prices an in-network addition at 3.810 B/B over-states by up to 3.31×. Quote 1.15 (measured, DERIVED across vehicles) or measure. | MAIN | on read | $0 |
| 2 | **Re-seal cl1 (B1) and remap its outputs to APDataStore (B2).** Both are chores. Without them Gate 0 refuses. | MAIN | before any cl1 fire | $0 |
| 3 | **Fire Gate 4 with the §6 threshold** — 71.1 min, 2.5 GiB, one rung. `s < 2.588` closes the last live rate rung; `s ≥ 3.109` opens it. Metal-idle only. | MAIN Metal executor | Metal free + B1 + B2 | GPU, 71 min |
| 4 | **Re-aim to hv1's field (B3) if MAIN wants a non-DERIVED answer.** Needs a third profile pinned to `9ba2e52b…`, an hv1-lineage init, and a fitter gate change. Larger than #3 and only worth it if #3 lands in the 2.588–3.109 band. | rate owner | after #3 | code + GPU |
| 5 | **A $0 lower bound on hv1's own shrink slope exists.** Re-quantize the shipped hv1 HPAC down a bit depth and re-run `ddm_hm1`'s replay (~19 min CPU). Post-hoc, so it under-states the trained slope — but a post-hoc slope already ≥ 2.588 would flip #3 from "likely negative" to "probably worth it" before spending Metal. | rate owner | CPU idle | $0 |

## Own-vehicle frontier

Unchanged: **S = 0.15959729295498598 @ 182,759 B `[contest-CUDA T4, n600]`**, archive sha
`80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`. This arm produced no score,
claimed no lane, took no Metal or Modal slot, spent $0, and did not move the pointer.
