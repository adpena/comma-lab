# ddm_av2 fresh eyes — refusal audit and scorer-aware distillation reopen

Axis/budget: **$0, design and read-only audit; no launch; no new scorer measurement by AV2.** The
canonical pointer did not move. WD2's ep60 flattened d4/w64 result remains refused at **INSTANCE**
scope, but the refusal memo's causal and continuation claims needed correction. Distillation is
actively reopened as WD3 with a different, score-native mechanism.

## Bottom line

- The same-instrument hv1 base receipt is **COMPLETE AND CONSUMED**. Against it, WD2 has
  `Delta d_seg=+0.00074963`, **7.0059x** the `1.07e-4` cap; Pose MSE is **623.7625x** the base; and
  the exact recomputed `Delta S=+0.9840885005473307` despite a 17,372 B saving. The refusal survives
  with corrected same-axis numbers.
- The `160+ epochs` claim is **UNSOUND**. It maps a train-loss slope to scorer distortion without a
  measured transfer function. Contrary to the charter's seed parenthetical, `TRAIN_RESULT.json`
  holds direct decode-MSE evaluations every five epochs, not only ep1 and ep60; those measurements
  still cannot predict `d_seg` or `d_pose` from the single scorer endpoint.
- `d_pose=0.09198625` is strong enough for this **INSTANCE refusal**, despite the advisory stamp.
  The stamp says `uv_group_not_declared`; observed evaluator and wrapper package versions match.
  Known thread/batch seams are razor-tie Seg effects, not a plausible mechanism for a 0.092 Pose MSE.
- The current mirror digest is
  `fa7c4bf51d47a6140ec0f95275ebf86b0e6c3c1dc00caff03a417ee989645799`, not the birth digest
  `d5bb36a2b5a9c3b1a32105c129437f6d7311e44e071839d0afdfaba0dd8a2004`. The documented hygiene
  sweep explains the known class of change; a bounded post-sweep audit found no current
  AppleDouble, bytecode, pycache directory, symlink, or changed critical scorer input.
- RFO2's global ordering still makes sense, but its WD2 subroute skipped the cheapest causal test:
  continue the retained ep60 d4/w64 student under a scorer-aware loss before paying for a fresh or
  wider model.

## Mandate A — per-seed verdicts

| Seed | Verdict | Primary evidence | Consequence |
|---|---|---|---|
| A1. Cross-instrument delta | **SURVIVES WITH CORRECTED SAME-AXIS NUMBERS** | Base `/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json` SHA `cfdac1fd...`; WD2 receipt `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/retained/candidates/flattened_d4_w64_epoch_0060/attempt_0000/advisory_n600_cpu/contest_auth_eval.json` | Replace 8.2x/634x with 7.0059x cap / 623.7625x Pose. Relay base row to MP2 through this memo and MAIN's queue. |
| A2. Cross-quantity slope | **UNSOUND** | `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/TRAIN_RESULT.json` SHA `c4260cf03eb4cb19f1788150592bf84f5468ebc5052dc0ab8a0ee123c3577918` | Withdraw `160+ epochs` as a scorer forecast. Retain only the direct decode-MSE trajectory and conditional local slope; no epoch count to admission is derivable. |
| A3. Pose 0.092 evidence grade | **ROBUST FOR INSTANCE REFUSAL; not a promotable score** | WD2 `contest_auth_eval.json` and nested provenance; `.omx/research/ddm_et4_pair17_c2_batch_seam_diagnosis_20260806.md` | Environment bookkeeping weakens axis authority, not the order-of-magnitude mechanism conclusion. Require the same-instrument base for an exact multiplier. |
| A4. Mirror integrity | **CURRENT TREE CLEAN IN BOUNDED CENSUS; BIRTH-EQUALITY FALSE** | `tac.contest_compliance.compute_upstream_snapshot_sha256(..., upstream_subdir='.', reject_executable_artifacts=True)`; attempt-1 provenance under `work_attempt1_contamination_refusal/`; current WD2 and hv1-r2 provenance | Bind every local run to pre/post digests and forbidden-entry censuses. Correct the refusal header: attempt 2 stamps `fa7c4bf5...`, not `d5bb36a2...`. |
| A5. MP2 routing optimality | **GLOBAL ORDER PROCEED; WD SUBROUTE REORDERED** | `.omx/research/ddm_rfo2_fresh_eyes_gestalt_synergy_20260815.md`; WD2 `TRAIN_RESULT.json`; `.omx/research/ddm_gc15_fresh_vs_warm_20260731.md` | Preserve already-materialized mixed-precision/carrier work. Inside distillation, run scorer-aware W0 warm continuation first, then reset control, d4/w56, factorized d4/w64/r19, and only then w96. |

### A1 — same-instrument base gate

Both receipts are retained on the same local instrument:

| Quantity | hv1 local base | WD2 ep60 student | Student minus base |
|---|---:|---:|---:|
| Archive | 182,759 B | 165,387 B | -17,372 B |
| Archive SHA | `80d9c8c6...` | `e9c4a9ed...` | distinct retained objects |
| `d_seg` | 0.00042714 | 0.00117677 | **+0.00074963 = 7.0059x cap** |
| Seg contribution | 0.042714 | 0.117677 | +0.074963 |
| `d_pose` | 0.00014747 | 0.09198625 | **+0.09183878; student/base 623.7625x** |
| Pose contribution | 0.03840182287340017 | 0.9590946251543693 | +0.9206928022809691 |
| Rate contribution | 0.12169171641365491 | 0.11012441468001655 | -0.01156730173363836 |
| Recomputed score | 0.20280753928705508 | 1.186896039834386 | **+0.9840885005473307** |
| Axis | `[env-mismatch advisory]`, macOS CPU, n600 | same | non-promotable |

The base receipt is 23,416 B, SHA
`cfdac1fd0965095152ffd88c878d9c4b8f38c644d755e594ad028a798daf3a7f`. It binds the hv1 archive,
post-sweep mirror hash in provenance, the same package versions as the student, and a retained
3,662,409,600 B inflate with raw SHA
`e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`.

The same-axis Seg cap multiple falls from the refusal memo's cross-axis 8.2x to 7.0059x because hv1
itself is worse on local CPU than on T4 (`0.00042714` versus `0.00029611`). The decision does not
flip. The base row above is the MP2 admission baseline relayed by this memo.

### A2 — what the retained trajectory actually supports

The refusal memo used the normalized train-loss log slopes `-0.0210/epoch` at ep20–40 and
`-0.0128/epoch` at ep40–60 to forecast scorer-gap closure. The transfer is invalid because the
train loss, decode MSE, Seg argmax error, and Pose output MSE are different quantities.

Direct `decode_mse_uint8` was in fact evaluated 13 times: ep1 1581.98; ep5 348.63; ep10 151.46;
ep15 94.34; ep20 95.98; ep25 77.75; ep30 70.63; ep35 70.93; ep40 60.90; ep45 63.29; ep50 60.19;
ep55 53.31; ep60 50.672823. The curve is nonmonotone at ep15–20, ep30–35, and ep40–45.

What is derivable, and no more:

- ep60 RMS camera error is 7.1185 uint8 levels;
- the direct ep40–60 decode-MSE log slope is `-0.009193/epoch`, a conditional 75.4-epoch halving
  time if that local slope persisted;
- the normalized train-loss ep40–60 slope is faster at `-0.012836/epoch`, demonstrating that the
  two slopes are not interchangeable;
- with only one n600 scorer endpoint, there is no empirical map from either curve to future
  `d_seg`, `d_pose`, or the admission score. No honest `epochs-to-bar` number is available.

### A3 — why environment seams do not rescue this instance

The harness's mismatch object reports `reason=uv_group_not_declared`. Evaluator and wrapper both
used Python 3.13.12 with Torch 2.12.1, torchvision 0.27.1, timm 1.0.27, and NumPy 1.26.4. The
upstream lock query returned null package references because no UV group was declared. This proves
the environment was not lock-verified; it does **not** prove a package mismatch occurred.

Known mechanisms are far smaller and of the wrong kind:

- ET4 found batch-1 versus batch-16 oneDNN behavior changing one Seg argmax pixel in one tested
  pair, `1/196,608`; it did not find a Pose-scale failure.
- Existing thread-parity receipts find scorer tensors bit-identical across the standard nt2/4/6
  settings, with nt1 or razor-tie Seg margins as the known seam.
- These reduction/order effects can matter for exact last-bit authority. They are not a plausible
  explanation for an absolute Pose MSE of 0.09198625 from the student's rendered frames.

Therefore the result remains a valid reason not to continue the decode-MSE-only ep60 instance.
The completed same-axis multiplier is 623.7625x, replacing the earlier `~634x` expectation.

### A4 — mirror snapshot and mutation model

The source-only canonical digest recomputed post-sweep is
`fa7c4bf51d47a6140ec0f95275ebf86b0e6c3c1dc00caff03a417ee989645799`. Attempt-1 provenance
stamped the birth digest `d5bb36a2...`; current WD2 attempt-2 and hv1-r2 provenance stamp
`fa7c4bf5...`. The two are unequal. The current mirror census found zero `._*`, `__pycache__`,
`*.pyc`, `*.pyo`, or symlinks. Critical files match the checked-in upstream bytes: `evaluate.py`,
`frame_utils.py`, `modules.py`, both scorer weights, both public-test manifests, and `videos/0.mkv`.
A bounded mtime check found no mirror file newer than the documented sweep boundary.

The known digest change is consistent with deleting AppleDouble and runtime bytecode during the
hygiene sweep. The bounded audit did not find another current content drift; it does not prove a
historical file-by-file delta because no pre-sweep manifest of every path was retained.

Fail-closed mutation classes for future mirror runs:

1. Python import writes `__pycache__`/bytecode; set `PYTHONDONTWRITEBYTECODE=1` and reject bytecode.
2. ExFAT/macOS creates `._*` AppleDouble sidecars; census names before and after every run.
3. Relative outputs, model/download caches, temp files, reports, or virtual environments land
   beneath the mirror; force all work/output/cache roots outside it.
4. A tool edits, deletes, renames, hardlinks, or atomically replaces evaluator source, weights,
   manifests, or video bytes; compare pre/post canonical hashes and critical-file manifests.
5. A symlink redirects an evaluator input; the canonical helper must continue rejecting symlinks.
6. Mode bits, xattrs/quarantine, and mtimes can change while the current path/size/content digest
   remains equal; record a separate metadata manifest where execution semantics can depend on them.
7. Concurrent mutation during a one-pass hash can create a mixed snapshot; require matching digest
   repeats around the run and fail if the tree changes during either pass.

Read-only or immutable mount semantics are preferable. A writable mirror is apparatus debt even
when pre/post hashes match.

### A5 — route correction

Mixed precision and carrier rank/refit remain ahead of new multi-hour training because their
candidate bytes already exist or their pool is localized. The WD2 result does not justify moving a
fresh w96 run ahead of them. It does, however, reveal a skipped cheap rung inside distillation:

1. W0: resume the retained flattened d4/w64 ep60 state under the scorer-aware objective;
2. W0 reset control: same weights, fresh optimizer only with a magnitude-matched ramp;
3. dense d4/w56;
4. factorized d4/w64/r19;
5. factorized/flattened w96 only after a measured capacity failure.

#816 makes “warm or fresh” a false binary. Optimizer moments, EMA, weights, and effective step size
are separable. A naive zero-moment restart produces a measured/derived 3.16x–6.57x step excursion,
so it cannot serve as the fresh control.

## Beyond-seed findings, ranked

1. **The refusal memo mislabels attempt 2's mirror.** Its header says `mirror d5bb36a2`, while the
   primary attempt-2 receipt stamps `fa7c4bf5`. This does not change the student metrics, but it is
   a provenance defect and must not propagate.
2. **The “only ep1+ep60 decode-MSE endpoints” premise is false.** Thirteen direct endpoints exist.
   They strengthen the trajectory description while still refuting any scorer forecast.
3. **Quantizr-style KL must stay auxiliary.** The full corpus records pure T=2 KL at 0.005515
   (`+125%` worse), KD-dominant KD+CE at 0.002741 (`+12%`), and CE-anchored light KD at 0.002423
   (`-1%`) on another vehicle. WD3 consumes T=2 and the dense teacher signal, but does not transfer
   `kd_w=0.3`; it uses an adaptive preservation constraint plus score-native terms.
4. **Smooth proxy validation is insufficient.** FD2 records proxy improvement without the desired
   realized argmax behavior. WD3 therefore evaluates through the real receiver, uint8, and frozen
   scorers during training.
5. **One strided subset cannot bank a negative.** M88/M96/NA2 measured opposing prefix bias signs and
   serial effective N 40.22/600. WD3 uses strided n60 for control, seeded stratified-random n120 for
   negative confirmation, and n600 for admission.
6. **Width is not yet the causal diagnosis.** The only measured student optimized the wrong loss.
   W0 scorer-aware continuation must precede claims that d4/w64 lacks capacity.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus, not just charter seeds, with content queries for
`distill`, `KD`, `kl_on_logits`, `T=2`, `decode_mse`, `scorer-aware`, `fresh warm`, `reset optimizer`,
`prefix bias`, `STRIDED`, `fd2`, `thread`, `batch seam`, `mirror`, `AppleDouble`, and `__pycache__`.
Also queried the canonical equation registry with
`.venv/bin/python tools/list_canonical_equations.py --json`, the canonical research index,
`sub015_DAG_*` FEED blocks, the current task ledger, and WD2 trainer/receiver sources.

Beyond the charter seeds, the search found the measured CE-anchor/light-KD versus pure-KL result,
#816's optimizer-reset effect size, FD2's proxy/realized split, and NA2's prefix-bias sign inversion.
Those findings changed WD3 from a fixed weighted KL sketch into a score-native constrained design,
put W0 preserved-state continuation first, added a magnitude-matched reset control, and separated
the strided controller subset from the n120 negative-confirmation subset.

## Mandate B — sealed WD3 design

The successor artifacts are:

- `.omx/research/charters/ddm_wd3_scorer_aware_width_distillation_20260815.md`
- `.omx/research/ddm_wd3_scorer_aware_width_distillation_20260815.json`

They are deliberately **design-sealed, code-not-built, no-launch**. Calling the JSON an executable
launch config would be fake: the WD2 trainer does not yet implement the scorer-aware objective or
teacher-scorer cache. The charter makes that build an explicit G2 prerequisite.

WD3's objective prices a stage-frozen, hard-`d_seg`-calibrated soft disagreement with coefficient
100 and Pose with its exact nonlinear score term. Its teacher-margin constraint, decode-MSE anchor
(`<=50.6728233448345`), and teacher-KL preservation use adaptive nonnegative duals, not guessed fixed
weights. Selection uses only realized hard scorer components and exact archive bytes.

### Rate-prize erosion

`25/37,545,489 = 6.658589531221714e-7 S/B`. Only the flattened ep60 full archive is measured.
Other full sizes use fixed remainder 148,739 B plus the measured primary compression ratio
`16,648/19,465`; they are **projections**. The three w96 uncompressed packet counts are structural
derivations from `wd2_receiver.serialized_bytes_for_spec`; no w96 payload was materialized.

| Arm | Full bytes | Status | Rate prize vs 182,759 B | Pose-held maximum `Delta d_seg` after net gate | Effective cap |
|---|---:|---|---:|---:|---:|
| dense d4/w56 | 164,908 | PROJECTED | 0.01188625 | 1.18827e-4 | **1.07e-4** |
| factorized d4/w64/r19 | 165,428 | PROJECTED | 0.01154000 | 1.15365e-4 | **1.07e-4** |
| flattened d4/w64 ep60 | 165,387 | **MEASURED** | 0.01156730 | 1.15638e-4 | **1.07e-4** |
| factorized d4/w96/r20 | 174,255 | PROJECTED | 0.00566246 | 5.65896e-5 | 5.65896e-5 |
| flattened d4/w96 | 179,236 | PROJECTED | 0.00234582 | 2.34232e-5 | 2.34232e-5 |
| dense d4/w96 | 183,177 | PROJECTED | -0.00027833 | none | projected priced out |

Every candidate must satisfy
`100*Delta d_seg + Delta sqrt(10*d_pose) + (25/37,545,489)*Delta bytes < -3.5e-6`, with
`Delta d_seg <=1.07e-4`. Pose consumes the rate prize through the nonlinear term; “pose held” is
only the table's comparison case, not an assumption granted to a run.

## Adjudication queue for MAIN

| Disposition | Owner | Consumer store | Fire trigger | Action |
|---|---|---|---|---|
| **FOLDED** | AV2 audit; MP2/MAIN consumer | `/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json` and this memo | Complete | Same-axis base and deltas computed above; MP2 consumes hv1 `d_seg=0.00042714`, `d_pose=0.00014747`, `S=0.20280753928705508` on this advisory instrument. |
| **FOLDED** | MAIN routing | `.omx/research/ddm_rfo2_fresh_eyes_gestalt_synergy_20260815.md` plus WD3 charter | Immediate | Keep materialized MP2/carrier work ahead globally; use W0-first ordering inside WD3. |
| **QUEUED-WITH-A-FIRE-ORDER** | MAIN | `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/` | A1 complete; r5 PID 63183 exited; scorer and Metal lanes claimed; WD3 real code/cache/resume/retention/dry-run gates landed | Build/cache -> W0 preserved-state -> W0 matched-reset -> D56 -> F64 -> conditional W96 -> n600 same-axis -> exact contest axis only if admitted. |

## LIVE-HYPOTHESES

- **Scorer-aware W0 may recover most of the 17,372 B prize without more capacity.** This is plausible
  because the ep60 failure optimized raw camera MSE, while the admission target is a tiny set of
  Seg/Pose decision quantities; capacity has not been tested under the correct objective.
- **Dense d4/w56 may dominate factorized d4/w64/r19.** It changes the inherited computation least,
  has the smallest measured raw packet, and its projected rate prize permits the full `1.07e-4`
  Seg cap.
- **Factorized w96 may be the only wider arm worth paying for.** Its projected 8,504 B saving leaves
  a nonzero 5.66e-5 Seg allowance, whereas dense w96 is projected larger than hv1.
- **Preserved optimizer/EMA state may beat a fresh optimizer under the changed loss.** #816 shows
  the apparent reset benefit is confounded by a large step excursion; a matched reset may reveal
  that basin preservation, not reset disorder, carries the useful state.
- **A constrained W0 continuation may need very little camera-MSE improvement.** The same-axis row
  proves the failure is scorer allocation, not merely the local base shift; the retained ep60 basin
  may still contain cheap task-space directions unavailable to blind MSE continuation.

## DEAD-ENDS

- **Continue ep60 for “160+ epochs.”** Closed: no measured mapping from decode/train loss to scorer
  admission, and the direct decode curve is nonmonotone.
- **Explain `d_pose=0.09198625` with the current advisory stamp.** Closed for this instance: the
  stamp is missing lock verification, not an observed version mismatch, and known numerical seams
  are far too small.
- **Treat dense w96 as the first capacity cure.** Closed by current projection: it erases the rate
  prize before distortion is paid. Reopen only if a real coder disproves the projection.
- **Use pure or fixed-weight KL as WD3.** Closed by prior measured evidence and no-transfer
  discipline: pure/KL-dominant forms were worse on the precedent vehicle; `kd_w=0.3` is not a WD3
  constant.
- **Bank a verdict on a contiguous prefix or the strided controller set.** Closed: prefix bias has
  opposite signs on Seg and Pose, and n60 strided is control evidence only.
- **Claim the post-sweep mirror equals its birth snapshot.** Closed: `fa7c4bf5... != d5bb36a2...`.
- **Repeat the cross-axis 8.2x/634x figures.** Closed: the completed local denominator gives 7.0059x
  the Seg cap and 623.7625x Pose; the qualitative refusal survives, but the old numbers do not.

Vehicle frontier UNCHANGED: **S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**, archive SHA `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.
