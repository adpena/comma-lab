# DDM JF1 joint field/model refit — epoch-2 byte-leg scope receipt

**Date:** 2026-08-23  
**Axis:** `[macOS-CPU advisory / scorer-free exact model-pack and RC64 measurement]`  
**Disposition:** `PARTIAL — SCOPE_REDUCTION_EPOCH_2_OF_60; BYTE-LEG COMPLETE AT NULL + FOUR DIAGONAL RUNGS; REFERENCE FITS FIRED; SCORER QUEUED-WITH-A-FIRE-ORDER`  
**Score claim:** false  
**Shipping candidate:** none

## Result first

The epoch-2 refit instrument is **7,554 stream bytes weaker than the shipped fit on the
unmodified field**. That deficit is the first result and bounds every diagonal row below. The
physical null model is 167 B smaller, so the null model+stream total is still **+7,387 B** versus
the shipped 127,292 B bar.

All four measured diagonal rows also lose on bytes. None recovers a single byte against its own
LD1 fixed-model stream; refitting adds **+2,861 to +7,103 B** to the stream. Their physically packed
models save only 140–226 B versus the shipped 13,515 B model. The best combined row is `k060000` at
130,007 B, still **+2,715 B** over the bar and therefore consumes 6.406021% of the 42,382 B demand
rather than supplying it.

This is a real, receiver-closed negative at one deliberately reduced fitting budget. It is **not**
an unconditional close of the diagonal: every fit below is epoch 2 of the sealed 60-epoch reference
schedule, the null itself has not recovered the shipped fit, and the seven full reference fits are
still running. It is also not a joint-score verdict because JF1 does not own the sole n600 scorer
lane.

## Incumbent and control pins

| Object | Physical bytes | SHA-256 | Result |
|---|---:|---|---|
| DX2 `archive.zip` | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | verified at prepare |
| shipped RC64 stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` | verified at prepare |
| unmodified decoded token field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | verified at prepare |
| shipped ep634 source checkpoint | 1,103,503 | `5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec` | copied into local custody |
| exact warm-start EMA state | 169,593 | `ff2d3e45d88a97cc6ae170864d0d6d72fc34c617cf211e94ccff439f23d2afd9` | generated and pinned |
| DALI/NVDEC n600 Seg argmax GT | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` | copied into local custody |

Before fitting, the staged shipped runtime re-encoded all 600 unmodified fields in 909.892 s and
emitted **113,777 B at the exact shipped stream SHA-256**, with all 113,777 prefix bytes matching.
This passes the mandatory incumbent reproduction and isolates the subsequent failure to the reduced
refit, not the RC64 measurement path.

The exchange rate is cited, not re-derived: `ddm_tx1_toolbox_crosswalk_20260819.md` §0 gives
`25/37,545,489 = 6.658590e-07 S/B`. The fixed comparison is 113,777 B stream + 13,515 B model =
127,292 B. The fixed-distortion demand is 42,382 B.

## Mandatory null refit

The null used the same 19-member IHS1/HPAC family, member count, architecture, seed, and sealed
training law as the reference form. Only the fitting budget was reduced to epoch 2 of 60.

| Quantity | `[macOS-CPU advisory / scorer-free exact bytes]` |
|---|---:|
| shipped-model stream on null field | 113,777 B |
| epoch-2 refit stream on null field | 121,331 B |
| **stream deficit** | **+7,554 B** |
| shipped physical model | 13,515 B |
| epoch-2 refit physical model | 13,348 B |
| model delta | −167 B |
| refit model + stream | 134,679 B |
| **combined delta vs 127,292 B** | **+7,387 B** |
| archive delta vs DX2 | +7,387 B |
| rate-only delta S | +0.00491870008671348 |
| receiver identity | PASS, decoded field SHA `cc10a7b…63efb` |

The positive control therefore **fails at this scope**. The stream deficit is 17.823604% of the
entire 42,382 B demand. No diagonal number below is interpreted as if this deficit were absent.

## Four-rung diagonal: real stream + physical model

Signs are explicit:

- `refit contribution = refit stream − fixed-model stream`; positive is worse.
- `bytes recovered = fixed-model stream − refit stream`; positive would be a refit win.
- combined deltas compare the physical refit model + real refit stream against 127,292 B.

| LD1 rung | fixed-model stream B | refit stream B | refit contribution B | bytes recovered B | model B | model+stream B | ΔB vs 127,292 | archive ΔB vs DX2 | rate-only ΔS | receiver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `k002500` | 113,973 | 120,607 | **+6,634** | −6,634 | 13,300 | 133,907 | **+6,615** | +6,615 | +0.004404656974903164 | PASS |
| `k010000` | 114,601 | 121,704 | **+7,103** | −7,103 | 13,289 | 134,993 | **+7,701** | +7,701 | +0.005127779797993842 | PASS |
| `k040000` | 114,375 | 118,917 | **+4,542** | −4,542 | 13,375 | 132,292 | **+5,000** | +5,000 | +0.003329294765610857 | PASS |
| `k060000` | 113,798 | 116,659 | **+2,861** | −2,861 | 13,348 | 130,007 | **+2,715** | +2,715 | +0.0018078070577266954 | PASS |

All denominators are n600 token fields = 600×384×512 = **117,964,800 positions**. Every stream is
a real native RC64 encode, every model is the smallest retained winner of the raw/XZ/Brotli q0–q11
race, and every row independently decoded to its intended coarsened field byte-for-byte. The model
packer also passed deterministic idempotent decode and zero deployed-logit difference.

The preregistered prediction required a refit contribution below −2,000 B at one rung and at least
one model+stream total below 127,292 B. At epoch 2, both are false: the best contribution is +2,861 B
and the best total is +2,715 B.

One non-verdict signal remains: `k060000`'s refit stream is 4,672 B below the epoch-2 null refit
stream, so the moved field is easier for this underfit model than the unmodified field in an absolute
within-run comparison. That does not overcome the 7,554 B null deficit and does not establish what
the terminal fit will do.

## Distortion and net score: NOT MEASURED

JF1 does not own the exclusive full-n600 scorer lane; `main_hot_state.md` grants it to
`ddm_ap1_residue_purchase_scorer`. No SegNet, PoseNet, Metal, Modal, or local advisory scorer was
fired by this arm. Therefore the following charter fields are honestly pending for every row:

| DALI GT class | Denominator | Realized d_seg |
|---|---:|---|
| Road, class 0 | 27,407,372 GT pixels | NOT MEASURED — queued |
| **Lane, class 1** | **690,754 GT pixels** | **NOT MEASURED — queued** |
| Undrivable, class 2 | 58,413,067 GT pixels | NOT MEASURED — queued |
| Movable, class 3 | 1,460,386 GT pixels | NOT MEASURED — queued |
| MyCar, class 4 | 29,993,221 GT pixels | NOT MEASURED — queued |
| all classes | 117,964,800 GT pixels | NOT MEASURED — queued |

Realized d_pose over the official DALI PoseNet first-six table (600×6 = 3,600 scalars) is also
**NOT MEASURED — queued**. Consequently net ΔS is **NOT MEASURED**. The table's rate-only ΔS values
are components, not score estimates and not bounds on an unknown distortion change.

The durable handoff is
`.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/SCORER_FIRE_ORDER.json`, disposition
`QUEUED-WITH-A-FIRE-ORDER`, owner `MAIN / exclusive n600 scorer-lane custodian`, consumer store
`.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/scorer`. It fires only after AP1 is
terminal, MAIN explicitly transfers the lane, no other full-n600 scorer is active, all terminal
epoch-60 candidate hashes validate, and local storage preflight passes. The ordered fire is serial,
CPU-only, through `tools/fire_local_advisory.py`, chunks ≤120, retaining raw output, terminal argmax,
five per-GT-class error masks with Lane separate, Pose6, checkpoints, and logs.

## Verdict scope and routing

**Verdict scope: FORMULATION AT 2/60-EPOCH FIT BUDGET, BYTE LEG ONLY.** At this reduced fitting
budget, joint field+model motion is byte-positive on all four measured LD1 rungs, and the fitting
instrument is itself 7,554 stream bytes behind the shipped model. This falsifies the charter's byte
prediction at epoch 2. It does **not** make the proposed six-arm closure unconditional, because the
reference-form epoch-60 fits and scorer join have not landed.

No shipping candidate was built. The retained runtimes are measurement-only candidates for the
queued scorer. The frontier and canonical pointer are unchanged.

The seven full reference-form jobs were fired under governed CPU launchers with exact cache-content
and init hashes, seed 20260716, deterministic environment, epoch-boundary checkpoints, and a 48-hour
wall cap. At this receipt they are all running and have reached epoch 6; terminal target is epoch 60.
The immutable epoch-2 checkpoints remain available independently. The first launch attempt
self-refused before training because `PYTHONHASHSEED=0` was absent; a second null-only wrapper attempt
self-refused because this sandbox cannot increase niceness. Neither produced a model, checkpoint, or
result. The third launch used the explicit deterministic environment and is the only live lineage.

## Custody

All new bytes were written to the charter's explicit local opt-in tier:

`/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/`

No write to `/Volumes/VertigoDataTier` or `/Volumes/APDataStore` was performed. Current retained
footprint is 2.7 GiB; local free space after the scope slice is about 454 GiB. Each measured row keeps
the source field/cache, epoch checkpoint, raw/XZ/all Brotli model representations, real stream,
per-frame bit ledger, candidate archive/runtime, decoded 117,964,800-byte token field, and five
14,745,600-byte class packbit masks with SHA-256 and byte counts. The full aggregate receipt is:

`.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/BYTE_DIAGONAL_SCOPE_E0002.json`

SHA-256: `ac48c6005eedf4e5459b9ab6e0228dc808b9cc295a74aa6a1fa515e9e0cf15e0`.

## Implementation and verification

- `7ceacbd126a418b4b11695977314d3a9d3e2b6aa` added the exact local joint-refit instrument, sealed
  `jf1_joint_refit` profile, local-output refusal boundary, real pack/RC64/receiver path, and tests.
- `b5c8edf63b9e8ff182043ae8f0c6f650044a9ada` added explicit epoch-bound fitting measurement while
  keeping epoch-60 output custody disjoint from reduced-scope rows.
- Every changed Python file received two genuine `review_tracker.py` passes before its serializer
  commit; targeted tests and Ruff passed. `upstream/` was not edited.

## RECALL EVIDENCE

Searches were against the full research corpus, receipts, canonical equations, research index/DAG,
design surfaces, and task/queue state, not only charter seeds.

- Queries: `hpac_mc36_joint_descent_law_v1`, `run-scoped`, `field refit`, `refit field`, `diagonal`,
  `IHS1`, `epoch_0634`, the exact archive/stream/field/checkpoint hashes, and the 127,292 B bar.
- Canonical-equation recall found the run-scoped HPAC law: refit every entry, treat estimates as
  non-authority, and price only the real CPU-packed model plus real stream. This changed the plan
  from using trainer estimates to retaining a raw/XZ/Brotli race and receiver-closing every row.
- `ddm_hv1_*`, `ddm_oa2_frontier_objective_alignment_20260817.md`, and RX2 provenance located the
  exact ep634 EMA checkpoint and shipped IHS1 lineage. This changed the plan from a cold refit to the
  same-family/member/architecture reference initialized from the exact shipped lineage.
- `ddm_ld1_lane_lossy_drop_exchange_20260822.md` supplied the retained coarsened fields and physical
  fixed-model streams; JF1 reused them byte-identically and did not duplicate LD1's fixed-model work.
- `ddm_bl1_per_position_bit_allocation_20260822.md` supplied the exact unmodified decoded field;
  `ddm_tx1_toolbox_crosswalk_20260819.md` supplied the exchange rate.
- Content search over `.omx/research/`, arm receipts, `CANONICAL_RESEARCH_INDEX*`, and
  `sub015_DAG_*` did not find a prior real joint field+same-family refit diagonal row in that searched
  scope. Nothing beyond the cited HPAC law and exact model lineage displaced the charter's diagonal.
- `main_hot_state.md` changed execution by withholding the scorer: AP1 owns the sole n600 lane, so
  JF1 retained byte-complete candidates and emitted the exact fire order instead of firing.

## Landing status

The required serializer was invoked with the memo's post-edit SHA and `base=new`. It failed before
staging because this managed sandbox cannot create a Git object temporary file:
`error: unable to create temporary file: Operation not permitted`. A post-failure check found the
memo untracked and the shared index empty, so no sibling content was staged or absorbed. The research
artifact is complete at the path above but is **not committed**; landing requires an operator or a
future environment with Git object-store write authority.

Own-vehicle frontier: **dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]**, unchanged.
