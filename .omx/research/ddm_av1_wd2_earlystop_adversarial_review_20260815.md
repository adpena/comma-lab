# DDM AV1 — adversarial review of the WD2 / early-stop / HV1 chain

**Round:** recursive adversarial review round 1, finding round.  
**Authority:** source and receipt inspection plus scorer-free byte measurements. No scorer,
no Modal, no paid dispatch, and no score claim.  
**Pinned objects:** WD2 `706a8f9d9680989aa5d0c1ff67d2950ffa88df12`, selector
`5624ef8bdc`, e480b archive `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`.

## Outcome

The chain is not clean enough to seal. Three load-bearing claims fail adversarial review:

1. The teacher-cache launch did not execute the CPU pair-0 identity gate, and its retained
   MPS pair 0 differs from the CPU proof in 73 of 3,052,008 channel values, all by one.
2. WD2's EMA formula is one legal branch of the canonical law only after silently setting
   the warmup fraction to `phi=1`; it is not a resolved LawRef and leaves 13.53% of the
   initial shadow in the deployment EMA after the full 60 epochs.
3. The early-stop “two instruments converged” story is one estimated-byte stream viewed
   twice. The fitted floor was forced to the then-observed minimum, and ep634 later crossed
   that floor by 482 B.

The selector proxy is also the wrong objective for a lossless HPAC recode, and WD2's design
receipt compares an uncompressed packet to a compressed-stream ceiling. Those are real
contract defects, but neither establishes scorer harm. H7 checkpoint readiness and H8
optimizer-update geometry survive review.

At the last read-only observation, the WD2 safe-run receipt remained `status=running`
(`child_pid=28836`; the charter's launcher pid and all watchers remained untouched). Its
latest retained epoch-15 row was a byte-closed 165,662 B archive with a 16,923 B Brotli-q11
semantic stream and `top1_error=0.9287222385` `[Darwin-MPS scorer-free fidelity telemetry]`.
That is neither a contest score nor a promotion result.

## Findings

| ID | Severity | Verdict | Concrete failure scenario | Receipt |
|---|---|---|---|---|
| F1 | **HIGH** | CONFIRMED | WD2 trains against an MPS teacher cache that is not byte-identical to the CPU receiver proof. A boundary pixel can therefore train toward a different uint8 target than the shipping CPU receiver produces. Scorer effect is unmeasured. | `prepare_teacher_cache` never calls the gate at `experiments/ddm_wd2_width_distillation_build.py:472-503`; launch argv is `prepare-teacher-cache`; measured pair-0 hashes `fe863d6f…` vs `55835465…`, 73 values differ, max abs delta 1. |
| F2 | **HIGH** | CONFIRMED | A sluggish deployment EMA can make a useful live student look bad, or select an unnecessarily early/stale deployment state. At 60 epochs WD2 retains 13.53% of initialization, not the canonical default's 1%. | `experiments/ddm_wd2_width_distillation_build.py:1089-1114`; canonical identities in `src/tac/canonical_equations/ema_decay_run_geometry_20260717.py`; exact `U=4500` arithmetic below. |
| F3 | **HIGH** | CONFIRMED | A non-monotone noisy byte trace is declared asymptotic because the fit floor is pinned to the current minimum; future checkpoints are then discarded even though the same run later beats that “floor.” | Fit receipt SHA `9918be52…`: `y_inf=130875`, RMS 376.775 B, `tau=68,179,629`, n=46; selector later chooses 130,393 B. Fit bounds are visible at `tools/fit_hpac_descent_law.py:87-118`. |
| F4 | **MEDIUM** | CONFIRMED | The lossless checkpoint selector can prefer a worse exact archive because token-classification error is priced as if it were contest `d_seg`, even though decoded-token identity should make scorer distortion invariant. | `tools/select_hpac_checkpoint.py@5624ef8bdc:2-7,118-174`; trainer defines token argmax misses at `tools/train_ddm_cl1_hpac_capacity.py:1056-1083`. |
| F5 | **MEDIUM** | CONFIRMED | Widths can be excluded by comparing raw WD2S bytes to a ceiling derived from the post-Brotli semantic stream. The check is conservative for these rows, but it is not a same-coder capacity optimum. | `experiments/ddm_wd2_width_distillation_build.py:278-304`; exact live rows retain both 19,465 B raw and 17,894/17,536/17,069/16,923 B q11 streams. |
| F6 | **MEDIUM** | CONFIRMED, beyond seed | The canonical EMA law is present in source and the P0 ledger says it was registered, but the current canonical registry query returns `[]` for `ema_decay_run_geometry_v1`. A new consumer can therefore invent a local formula while believing no registered law exists. | `.venv/bin/python tools/list_canonical_equations.py --equation-id ema_decay_run_geometry_v1 --json` returned `[]`; source is `src/tac/canonical_equations/ema_decay_run_geometry_20260717.py`; `.omx/state/operator_p0_ledger.jsonl:181`. |

No finding above is an exact-score result. F1 and F2 are causal-risk findings that require
named measurements before any claim about final Seg/Pose impact.

## Assumption challenge

The shared assumption is that telemetry-space and device-space surrogates preserve the
ordering of the receiver-closed object: token top-1 preserves checkpoint quality, an MPS
cache stands for CPU receiver bytes, and a fitted estimated-byte floor predicts future
serialized bytes. That assumption does not survive this pass.

Violating it changes the chain's verdict in two different ways. For HPAC, exact decoded-token
and raw-output identity would make distortion identical and eliminate the need for the
top-1 proxy entirely; only exact archive bytes should rank candidates. For WD2, cross-device
identity is not automatic: the measured pair-0 mismatch means the cache-to-shipping-receiver
transfer remains open until parity or score-equivalence is measured.

## H1 — selector proxy validity: CONFIRMED defect

The selector openly computes

`(25 / 37,545,489) * estimated_joint_bytes + 100 * token_top1_error`.

The second term is not SegNet `d_seg`; it is HPAC token-argmax mismatch. No source or receipt
states or measures a monotonic relation to contest SegNet. More strongly, HPAC is a lossless
entropy coder: once the candidate proves decoded-token identity and full raw-output identity,
`d_seg` and `d_pose` are exactly invariant across checkpoints. A real scorer cannot invert
ep508 versus ep634 under that identity premise; an inversion would mean the identity/export
premise failed, not that token top-1 was a useful scorer proxy.

The proxy is material in general. From ep508 to ep634:

- estimated bytes improve 482 B, contributing `0.0003209440` proxy units;
- top-1 improves `2.0599365e-6`, contributing `0.0002059937`, equivalent to 309.37 B;
- total proxy improvement is `0.0005269377`.

For these two rows, ep634 is separately best on both telemetry bytes (130,393 vs 130,875)
and token top-1 (`0.0018945397` vs `0.0018965997`), so deleting the top-1 term would not
reverse their telemetry ranking. It could reverse other neighbors, and all telemetry bytes
remain advisory estimates.

**Cheapest resolving measurement:** byte-close ep508, ep634, and the closest retained
neighbors through the same RX2/HV1 packer; prove decoded-token and full raw-output identity;
then rank by exact complete-archive bytes. No scorer is needed if identity passes. If any
identity gate fails, the named resolver is one n600 exact scorer comparison on those exact
archives, owned by MAIN's scorer queue.

## H2 — teacher-cache 84 s and identity: REFUTED as a composite claim

The complete/fresh portion is confirmed. The launch manifest (SHA `0c5db16b…`) records the
target root as `state=absent`, invokes `prepare-teacher-cache --device mps --batch-size 1`,
and pins both WD2 source hashes. The code creates a full 1,831,204,800 B memmap, starts at
frame zero when no progress exists, calls the frozen teacher plus `receiver.camera_uint8`
for each index, flushes each batch, and writes preserved 10-frame stages
(`experiments/ddm_wd2_width_distillation_build.py:577-695`). Sixty stage receipts reach
frame 600. The terminal payload directly re-hashes to
`695023d4ca56e14f53f1e90b56134821c3c0a0c66f9b07f6aa6bd6ffdf9f4ebd`.

The 84 s is plausible: the safe-run elapsed time is 83.574 s, or 7.18 pairs/s, 139 ms/pair,
and 20.9 MiB/s of retained uint8 output. The inner frame-600 stage reports 75.456 s.

The identity portion is false. The pair-0 gate lives only in `verify_build` at lines
472-503; `prepare_teacher_cache` does not call it, and this launch's argv did not run it.
The old proof compared two CPU topology implementations to each other. It did not compare
the MPS cache to CPU. Reading the first retained cache item as NCHW and comparing it to the
CPU NHWC proof gives:

| object | SHA-256 |
|---|---|
| MPS cache pair 0, transposed to NHWC | `fe863d6f5e606341ffd03cb2aea9f210d5cdc3992654a15b403592f428eb6a3f` |
| CPU pair-0 receiver proof | `558354654d49c62edbc6a4a8e9f2231d20d1fb34af143ceba813f6fceb11b088` |

There are 73 differing channel values and 73 differing RGB pixels out of 3,052,008 values;
40 deltas are -1, 33 are +1, and max absolute delta is 1. This is a measured parity defect,
not a measured scorer regression. The resolving measurement is a full-n600 CPU-versus-MPS
teacher receiver comparison with per-pair mismatch counts, followed by score-equivalence only
if exact parity remains false.

## H3 — EMA decay: CONFIRMED provenance and calibration defect

WD2 has 600 batch-1 microbatches, accumulation 8, hence 75 optimizer/EMA updates per epoch
and `U=60*75=4500`. It sets

`d = 1 - 2/U = 0.9995555555555555`.

The canonical law offers two calibrated inversions: `d=eps**(1/U)` for a declared terminal
seed fraction, or `d=1-2/(phi*U)` for a declared warmup fraction. WD2's formula is therefore
not algebraically invented; it silently fixes `phi=1.0`. Neither `phi=1` nor a target seed
fraction appears in the run config, and no LawRef declaration/resolution is present. Calling
the value “derived” hides the missing calibration choice.

Exact consequences:

| point | WD2 `phi=1` seed mass | canonical default `eps=0.01` seed mass |
|---|---:|---:|
| epoch 5 | 0.84645 | 0.68129 |
| epoch 10 | 0.71648 | 0.46416 |
| epoch 15 | 0.60646 | 0.31623 |
| epoch 30 | 0.36780 | 0.10000 |
| epoch 60 | **0.13528** | **0.01000** |

WD2's registered warmup completes exactly at update 4500. The donor did this correctly:
ep634 embeds a resolved `ema_decay_run_geometry_v1` declaration with `eps=0.01`, no fallback,
and decay `0.9998720868` for its 36,000-update geometry.

The named resolver is not to mutate the live run. At a retained WD2 checkpoint, byte-close
both the deployment EMA and live weights through the same CPU receiver and compare fidelity
and complete-archive bytes. Before any successor training launch, MAIN must choose and record
the calibration target through a restored LawRef/DSL path.

## H4 — clamp / STE saturation: REFUTED as stated

The student head is `sigmoid(...) * 255` (`experiments/ddm_wd2_student_receiver.py:170-191`).
For finite activations it is inside `[0,255]`; bilinear interpolation is a convex combination,
so the pre-round clamp at `experiments/ddm_wd2_width_distillation_build.py:857-871` is not an
out-of-range gradient killer. Teacher endpoints do not activate the student's clamp.

The full retained teacher cache contains 493,197 endpoint values out of 1,831,204,800:
482,248 zeros and 10,949 values of 255, or **0.0269329%** total. Thus a broad clamp-dead
population is refuted.

A narrower numerical-sigmoid concern remains **INDETERMINATE**: very large logits can make
sigmoid derivatives negligible without leaving `[0,255]`. The named resolver is retained
eval telemetry for pre-sigmoid logit quantiles and the fraction of pixels with near-zero
sigmoid derivative. It is not a reason to stop the live run absent that measurement.

## H5 — packet arithmetic: CONFIRMED axis-label defect

The design receipt measures `exact_uncompressed_packet_bytes=19,465` and subtracts it from a
19,606 B ceiling derived by removing 15,157 B from the incumbent's **post-Brotli** 34,763 B
semantic stream. The field says “uncompressed”; the selection-law prose calls the ceiling a
“semantic packet ceiling.” These are not the same coder axis.

The live rows make the distinction concrete: the raw packet remains 19,465 B while q11 is
17,894 B at epoch 1, 17,536 B at epoch 5, 17,069 B at epoch 10, and 16,923 B at epoch 15.
The raw ceiling is conservative for these measured objects, but it cannot establish the
maximum feasible width or rank under compressed rate.

The 2,051 B MZ2 alternative is measured on the complete-archive axis: 181,451 B versus
183,502 B (`.omx/research/ddm_mz2_frozen_section_representation_attack_20260815.md:14-24`).
Therefore the family falsifier is same-axis only when it compares MZ2's exact archive saving
to a WD2 exact complete-archive saving after fidelity holds. The live WD2 candidate receipts
already provide that exact rate axis; the design-width derivation does not.

## H6 — “two instruments converged”: CONFIRMED numerology

Both receipts consume the same
`.../full_e480b_e960/launcher/run.log`. The fit used an earlier snapshot SHA
`135120903b53…`; the selector used its later 42,132 B extension SHA `fb885131dda6…`.
They are not independent instruments.

The fit is especially weak for an early-stop floor. Over 46 discrete-QAT points from epochs
482-572 it reports RMS 376.775 B and an exponential time constant of 68,179,629 epochs. Its
optimizer bound requires `y_inf <= observed_min`; the solution lands at exactly the then-minimum
130,875 B (ep508). At ep572 the observed value was 131,357 B, making the headline gap to that
floor 482 B. Later ep634 is 130,393 B—another 482 B, but now **below** the supposed floor.
The matching magnitudes are opposite sides of the same fitted minimum, not corroboration.

The early-stop's opportunity-cost argument—roughly `3e-4 S` projected residual versus WD2's
roughly `0.0115 S` target—can stand as a governance choice. The fit cannot stand as an
independent convergence measurement. The named resolver for future stops is a held-out,
rolling byte-closed forecast with a noise-aware non-monotone model; the stopping statistic
must not fit and validate on the same estimated-byte trace.

## H7 — ep634 byte-close readiness: CONFIRMED

The retained checkpoint is 1,103,503 B and re-hashes to
`5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec`.
It carries schema `ddm_cl1_hpac_capacity_checkpoint.v2`, epoch 634, phase `discrete_qat`,
run identity/config and source hashes, optimizer and scheduler state, RNG, resume lineage,
history, causal hash, live weights, EMA state, and deployment state.

All 37 tensors in `checkpoint["state_dict"]` are byte-equal to `ema.shadow`; all 37 differ
from `live_state_dict`. The history row that the selector consumed says
`evaluated_weights=ema_shadow`, and the RX2 packer loads exactly `checkpoint["state_dict"]`
at `experiments/ddm_rx2_mc36_identity_race.py:263-275`. The selector and export therefore
refer to the same tensor state. Its estimated bytes remain advisory until serialization,
which HV1 is responsible for proving; this readiness verdict does not promote the estimate.

## H8 — accumulation geometry: CONFIRMED

The donor's embedded raw argv and run config both say batch size 8. Its trainer performs one
optimizer and EMA update per eight-sample batch (`tools/train_ddm_cl1_hpac_capacity.py:1219-1242`),
giving 75 updates per 600-frame epoch. WD2 uses batch 1, divides each loss by accumulation 8,
clips and steps only after eight microbatches, and refuses partial groups
(`experiments/ddm_wd2_width_distillation_build.py:1089-1094,1178-1204`), also giving 75 updates.

The student's GroupNorm is per sample, so no batch-statistics state changes between physical
batch 8 and eight batch-1 microbatches. The effective sample/update geometry is preserved in
exact arithmetic. MPS floating-point accumulation order can differ, so bit identity is not
claimed; no evidence makes that rounding difference a mechanism verdict.

## Recursive review status

This was the mandatory finding round and included the assumption-challenge axis. It found
HIGH defects, so it does not advance the compose/freeze clean-pass counter. Status remains
**0/3 clean passes**. A successor must adjudicate the findings before a clean round can begin.

## Ranked MAIN adjudication queue

1. **QUEUED-WITH-A-FIRE-ORDER — CPU/MPS teacher parity.** Owner: MAIN local-Metal executor.
   Consumer: `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/teacher_cache_e480b/` plus
   the WD2 decision memo. Fire trigger: the live WD2 run reaches a governed terminal or
   early-stop boundary and releases the lane. Produce per-pair full-n600 byte differences;
   score only if byte parity remains false and MAIN grants the single scorer slot.
2. **FOLDED — ep508/ep634 exact selection.** Owner: the live HV1 compose arm. Consumer:
   `/Volumes/{APDataStore,VertigoDataTier}/pact/ddm_hv1_harvest_compose/`. Fire trigger already
   occurred when HV1 retargeted to ep634. It must finish exact archive and identity A/B before
   the selector ordering is treated as resolved; AV1 must not duplicate its store or process.
3. **QUEUED-WITH-A-FIRE-ORDER — EMA calibration adjudication.** Owner: MAIN WD2 maintainer.
   Consumer: `p0_ema_calibration_20260717`, the canonical-equation registry, and the next WD2
   launch ticket. Fire trigger: a stable retained WD2 checkpoint at the live run's governed
   endpoint. Byte-close EMA versus live weights, then declare `eps` or `phi` through LawRef.
4. **QUEUED-WITH-A-FIRE-ORDER — stop-law repair.** Owner: MAIN endpoint-governance maintainer.
   Consumer: the task-1058 endpoint selection/early-stop store. Fire trigger: before the next
   HPAC continuation burn. Replace same-trace floor matching with held-out byte-closed forecast
   evidence or an explicit opportunity-cost decision that makes no convergence claim.
5. **QUEUED-WITH-A-FIRE-ORDER — WD2 rate-axis naming/selection repair.** Owner: MAIN WD2
   maintainer. Consumer: the next WD2 design/build receipt. Fire trigger: after the live run
   terminates and before choosing any successor width. Rank capacity using realized compressed
   complete archives; keep raw packet size only as a separately named serializer bound.
6. **QUEUED-WITH-A-FIRE-ORDER — numerical sigmoid-saturation telemetry.** Owner: MAIN WD2
   telemetry maintainer. Consumer: the next WD2 eval schema and decision memo. Fire trigger:
   only if a successor WD2 training configuration is prepared after the live run terminates.
   Record pre-sigmoid logit quantiles and near-zero sigmoid-derivative mass; do not disturb the
   current run merely to collect it.

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus by content for `estimated_joint_bytes`,
`checkpoint_selection`, `top1_error`, `EMA`, `teacher cache`, `width distillation`, `HPAC`,
`accumulation`, `flattened`, `MZ2`, and the e480b/ep634 SHAs; the canonical research indexes
and `sub015_DAG_*` FEED blocks; `.omx/state/canonical_task_status.jsonl`, the hot-state and P0
ledgers; the canonical-equation registry; the WD2/RX2/HV1 sources; and the retained SSD receipts.

Beyond the charter seeds, the search found:

- the P0 EMA source/ledger history and the current registry's missing equation row, changing
  H3 from a formula comparison into a provenance/discoverability finding;
- the actual donor checkpoint's resolved EMA LawRef and batch-8 argv, which confirmed H7/H8;
- MZ2's exact complete-archive 2,051 B row, which preserved the final WD2 falsifier while
  refuting the raw-packet derivation as same-axis;
- the MPS cache's pair-0 mismatch against the CPU proof, changing H2 from “plausible and full”
  to a composite refutation;
- live WD2 epoch-15 retained bytes, used only to verify the raw-versus-Brotli distinction and
  liveness, never as scorer evidence.

I did not find, in those searched scopes, a measured monotonic relation from HPAC token top-1
to contest SegNet `d_seg`, or a current-WD2 CPU/MPS full-n600 parity receipt.

## Measurement and custody boundaries

- **Measured now:** receipt/source identities; selection and fit arithmetic; teacher payload
  completeness/hash; pair-0 cross-device byte difference; full-cache endpoint mass; EMA closed
  forms; ep634 checkpoint state mapping; donor/WD2 update counts; exact retained WD2 archive,
  raw packet, and Brotli sizes through epoch 15.
- **Not measured:** WD2 SegNet/PoseNet components, CPU/MPS score equivalence, final WD2 quality,
  ep508-versus-ep634 final exact archive winner, or any contest score.
- **Untouched:** live WD2 launcher/child/watchers, HV1 stores/processes, staged index, protected
  files, `upstream/`, and all retained payloads. No payload was materialized or discarded.
- **Authority:** all new numerical findings are scorer-free `[macOS-CPU/MPS advisory]`; the
  own-vehicle contest pointer remains inherited and unmoved.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN local-Metal executor; consumer store:
  `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/teacher_cache_e480b/`; fire trigger:
  governed WD2 terminal/early-stop plus lane release; run the full CPU/MPS teacher parity census.
- **FOLDED** — owner: live HV1 compose arm; consumer store:
  `/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/` with Vertigo mirrors; fire trigger:
  already fired by ep634 retarget; finish ep508/ep634 byte-close identity and exact-byte ranking.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN WD2 maintainer; consumer store:
  `p0_ema_calibration_20260717` plus the next WD2 ticket; fire trigger: stable terminal WD2
  checkpoint; byte-close EMA/live and resolve a declared `eps` or `phi` through LawRef.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN endpoint-governance maintainer; consumer store:
  task-1058 endpoint closure; fire trigger: before any next HPAC burn; replace the same-stream
  convergence claim with held-out byte-closed forecasting or explicit opportunity-cost policy.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN WD2 maintainer; consumer store: the next WD2
  design/build receipt; fire trigger: after live-run termination and before choosing a successor
  width; rank capacity on compressed complete archives and label raw packet bounds separately.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN WD2 telemetry maintainer; consumer store: the next
  WD2 eval schema and decision memo; fire trigger: preparation of a post-terminal successor WD2
  config; record logit quantiles and near-zero sigmoid-derivative mass without touching this run.

## LIVE-HYPOTHESES

- The 73 one-level pair-0 differences are boundary rounding drift and may be scorer-neutral.
  This is plausible because every measured delta is exactly one and only 0.00239% of channel
  values differ; full-n600 parity and, if needed, score-equivalence can decide it.
- WD2's live weights may be materially better than its deployment EMA during the 60-epoch
  envelope. This is plausible because the current decay still carries 60.65% initialization
  at epoch 15 while retained mismatch is descending; a same-checkpoint byte-close A/B decides.
- Ep634 may remain the exact HPAC archive winner even after removing token top-1 from selection.
  It is plausible because it is separately the telemetry minimum in bytes and top-1, and HV1's
  preliminary model-section pack is smaller; only complete archive A/B plus identity decides.
- The WD2 width envelope may admit a wider compressed winner than raw-packet arithmetic allowed.
  This is plausible because every retained trained 19,465 B raw packet compresses below 17,894 B,
  leaving over 1.7 KiB beyond the nominal 19,606 B stream ceiling; a post-terminal compressed
  iso-archive sweep decides without transferring distortion constants.

## DEAD-ENDS

- Treating token top-1 as measured contest `d_seg` is closed: it is an entropy-model argmax
  miss rate, and lossless raw identity would make scorer distortion invariant.
- Calling the fit and selector “two independent instruments” is closed: both read the same log,
  and ep634 crossed the fitted floor by 482 B.
- Claiming the pair-0 identity gate ran in the teacher-cache launch is closed: it exists only in
  `verify_build`, while the launch invoked `prepare-teacher-cache` and produced different bytes.
- Treating the WD2 pre-round clamp as a broad gradient-dead region is closed for this architecture:
  sigmoid plus bilinear interpolation stays in range, and teacher endpoint mass is 0.02693%.
- Comparing 19,465 B raw directly to the 19,606 B compressed-stream ceiling as a same-coder optimum
  is closed; complete compressed archive bytes are the deciding rate surface.
- Mutating or stopping the live WD2/HV1 work to test these findings is closed by the charter; all
  remedies remain governed post-boundary actions owned by MAIN.

Vehicle frontier unchanged: S=0.1600920261571558 @ 183,502 B `[contest-CUDA T4, n600]` (e480b SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`).
