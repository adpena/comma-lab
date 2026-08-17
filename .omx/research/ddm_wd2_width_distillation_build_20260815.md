# DDM WD2 width-distillation build — receiver-closed student ready at the Metal boundary

## Outcome

WD2 is built and launch-ready, but it was not launched. No scorer ran, no Modal job ran, and the live e960 process, watchers, closer, and output tree were not touched.

| Surface | Result | Authority / boundary |
|---|---:|---|
| Exact current archive | 183,502 B, SHA-256 `e3e6f440…` | `[contest-CUDA T4, n600]` inherited e480b authority pointer |
| Current semantic + carrier + wrapper pool | 34,763 + 22,161 + 14 = 56,938 B | exact RX1 header/stream bytes |
| Charter −15,153 B rung | semantic stream ≤19,610 B | rate-only projection is S=0.1500022654, not sub-0.15 |
| Strict rate-only sub-0.15 ceiling | save ≥15,157 B; semantic stream ≤19,606 B; archive ≤168,345 B | assumes zero exact-distortion change |
| Derived primary student | flattened, depth 4, width 64; exact raw packet 19,465 B | exact serializer accounting; trained Brotli stream not yet measured |
| Random-init apparatus archive | 166,169 B; semantic stream 17,430 B; Δarchive −17,333 B | `[byte-only scorer-free apparatus]`; explicitly not a candidate and no distortion evidence |
| Real-shape memory probe | 2,455,371,776 B peak RSS, batch 1 | `[macOS-CPU apparatus]`; one full QAT forward/backward, zero optimizer steps |
| Inactive receiver proof | 7/7 parsed fields identical; pair-0 camera uint8 identical | exact e480b parse-back plus one real semantic realization, not full-video inflate |
| Verification | 6 targeted tests pass; Ruff clean; both governed launch commands dry-run rc=0 | no training or score claim |

The exact build receipt is `/Volumes/VertigoDataTier/pact/ddm_wd2_width_distillation/build_v4/BUILD_RECEIPT.json` (13,170 B, SHA-256 `f78e99d1…`). Its retention inventory covers 53 files / 7,322,810 B at SHA-256 `0c7282be…`. The current-source memory receipt is under `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/memory_probe_flattened_d4_w64_cpu_v4/`; the complete random-init container apparatus remains retained under the preceding `memory_probe_flattened_d4_w64_cpu_v3/` receipt because the only intervening production edit makes scalar reporting consume its already-persisted payload record. Every materialized RGB payload, student packet, compressed stream, model, member, archive, repeat archive, and patched runtime was retained.

## What was built

`experiments/ddm_wd2_student_receiver.py` defines three counted receiver forms:

- `dense`: inherited depthwise + full pointwise block and per-block FiLM;
- `factorized`: inherited depthwise + learned low-rank down/up pointwise block and per-block FiLM;
- `flattened`: full pointwise blocks with one frame FiLM at the trunk input.

The `WD2S` packet stores learned content inside `archive.zip`: fp16 vectors and signed-int4 matrices with fp16 per-output-row or per-embedding-column scales. The receiver patch is additive. Untagged semantic data stays on the original WANS1 branch; tagged data is parsed into the student and rendered by the unchanged F26 video path. Candidate export copies the exact e480b runtime, binds the exact candidate archive hash/size, parses the complete RX1/Brotli container back, and retains a deterministic repeat archive.

`experiments/ddm_wd2_width_distillation_build.py` provides five real stages:

1. `verify-build`: derives the topology envelope and proves inactive receiver identity.
2. `prepare-teacher-cache`: materializes all 600 frozen teacher masters through the actual 384×512 renderer, 874×1164 bilinear receiver resize, clamp, and uint8; it is disk-resumable and checkpoints distinct frame boundaries.
3. `memory-probe`: executes the exact full-resolution QAT graph without an optimizer step and retains both outputs plus the packet.
4. `train`: seeded CPU/MPS training with fallback refused, int4/fp16 QAT from update zero, receiver+uint8 MSE, warm-started EMA, optimizer/scheduler/RNG state, distinct periodic and stage-end checkpoints, crash-safe evaluation attempts, and retained n600 candidate renders/packets/archives.
5. `inventory`: hashes every retained payload.

The eval JSON uses `top1_error` as the exact teacher-vs-student uint8-byte mismatch fraction and says so in every row. `estimated_joint_bytes` is kept only for `fit_hpac_descent_law.py` compatibility; its value is the exact retained archive length, not an estimate. `top1_regression_ratio` compares the current mismatch fraction with the best prior retained eval, allowing the static watcher to enforce a real relative-best top1 band from the first run.

Determinism is bounded honestly. RNGs and algorithms are seeded, MPS fallback is prohibited, every causal state is checkpointed, and CPU packet parse-back is serialization authority. No MPS bit-repeat was measured, so none is claimed; the retained checkpoint and exact packet/archive hashes are the reproducibility anchors.

## Derived design and budget

The capacity decision is an exact iso-payload envelope, not a transferred distortion law. HM1's full-n600 HPAC D8→D7 result got 60 B worse, so raw parameter reduction was not treated as a monotone rate law. QA83/#516/#550/#662 establish factorized and flattened forms as legal first-class mechanisms but supply no current-renderer distortion constant. The build therefore enumerates each form at the maximum GroupNorm-compatible width or rank below the strict 19,606 B packet ceiling.

The retained design has 13 candidates. The important depth-4 bracket is:

| Form | Width / rank | Exact raw packet |
|---|---:|---:|
| dense control | w56 | 18,905 B |
| factorized control | w64 / r19 | 19,513 B |
| flattened primary | w64 | 19,465 B |

Flattened d4/w64 is first because it preserves all four inherited receptive blocks and full-rank pointwise mixing while buying eight channels by paying frame conditioning once. Factorized d4/w64/r19 is the conditioning-rich matched control; dense d4/w56 is the no-mechanism control. This is a fire order, not a family verdict.

The random-init full container measured 166,169 B, 2,176 B below the strict 168,345 B rate-only archive ceiling. This proves the tagged container/runtime has enough byte capacity for this particular untrained entropy state. It does not prove trained weights retain that Brotli length, and catastrophic random-init distortion makes it non-promotable.

Admission remains:

`ΔS = 25 × (candidate_archive_bytes − 183,502) / 37,545,489 + unknown exact-distortion Δ`.

Decode MSE and top1 mismatch are watched realization guards, not contest-score proxies. The exact distortion term remains unknown until the one n600 authority slot is explicitly claimed. The pre-registered family falsifier is unchanged: if the smallest family-optimal student that holds exact distortion saves no more than 2,051 exact archive bytes, WD2 is priced out by mz2's retained structural candidate.

## Receiver and custody proof

The build retained both the original and patched parse-back of the exact e480b archive. All seven byte surfaces matched: semantic WANS1, carrier, HPAC, token stream, residual payload, compressed model section, and compensation. The rebuilt teacher topology and the current cpr1 renderer also produced byte-identical pair-0 camera uint8 output: 3,052,008 B, SHA-256 `55835465…` on both paths.

Scope is explicit: this proves exact current-container parse-back and one realized semantic frame. It does not claim a full 1,200-frame inactive output comparison. The tagged branch separately passed a full-container unit proof: a student packet went through RX1 header accounting, Brotli, stored ZIP, patched residual parser, and exact packet parse-back; deterministic archive repeats matched.

Vertigo had only about 1.0 GiB free. The small charter build proof remains at the mandated Vertigo root; the storage waterfall routes future 1.831 GB teacher output plus all n600 student renders to APDataStore, which had about 448 GiB free and passed a 64 GiB floor. Nothing was deleted or symlinked.

## Sealed launch ticket

The sealed ticket is `.omx/research/ddm_wd2_width_distillation_build_20260815/launch_ticket.json`; validation is beside it. Both commands passed the governed launcher's real `--dry-run`, both child argvs parse, all watcher configs validate, and builder/receiver hashes are enforced again inside each child before materialization or training.

Because the launcher writes manifests even during `--dry-run`, those manifests were moved recoverably with SHA-256 custody receipts under `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/dry_run_validation/`. The ticket's two exact future run roots are absent again; no training payload occupied them.

Fire order:

1. Materialize and retain the exact n600 teacher receiver cache on MPS, with distinct 10-frame checkpoints.
2. After the cache receipt verifies, train `flattened_d4_w64` for the donor's 60-epoch law-fitting envelope at batch 1 / accumulation 8, evaluating every five epochs and checkpointing every epoch.

The quality watcher has `joint_regression=true` against 183,502 B and a top1 relative-best band of 1.0. NaN/garbage and stale telemetry are live. The first continuation, if justified by the retained descent law, must replace/bootstrap the direct top1-error threshold from the parent tail following the consumed continuation-composer policy. No HPAC-specific continuation adapter is invoked by WD2.

Disposition is `READY_TO_FIRE_AT_SLOT_BOUNDARY`, owner `MAIN local-Metal executor`, consumer store `/Volumes/APDataStore/pact/ddm_wd2_width_distillation`. The exact trigger is a durable e960 terminal or governed #1058 early-stop receipt plus terminal watchers/closer and explicit Metal-lane handoff. The charter forbids firing before that trigger, so no launch occurred here.

## RECALL EVIDENCE

Searches covered `.omx/research` content and receipts for `width distill`, `semantic renderer`, `factorized`, `flattened`, `teacher distillation`, `q3`, `q4`, `FiLM`, `precision waterfill`, `HM1`, and the e480b SHA; the canonical equation registry; the research index and `sub015_DAG_*` FEED blocks for #74/#455/#516/#550/#662/#1026 and QA83; the task/hot-state ledgers; the real cpr1 and F26 receiver; and the current MPS donor/composer/watcher sources.

Findings beyond the charter seeds changed the plan:

- Live hot state superseded the common contract's stale pointer: e480b is S=0.1600920261571558 at 183,502 B on `[contest-CUDA T4, n600]`.
- Exact arithmetic tightened the charter rung by four bytes: −15,153 B reaches only S=0.1500022654 rate-only; strict sub-0.15 needs −15,157 B if distortion is unchanged.
- The inherited `GroupNorm(max(1,w//8),w)` excludes nominal widths that do not divide their group count; the ladder derives only receiver-valid widths.
- HM1's D8→D7 full-n600 rate loss prevented a naive "narrower always compresses better" claim.
- The precision-waterfill equation is an initializer around measured per-layer sensitivity, not a semantic-student capacity constant; it remains available after a trained student exists and was not used to fake a width optimum.
- The live composer worktree already contains the owed `joint_regression=true` and parent-tail top1-band policy. WD2 consumed that policy read-only and implemented a first-run relative-best top1 field without absorbing the shared file.
- Vertigo's free space forced the governed SSD waterfall to APDataStore for bulky future payloads.

No direct prior WD2 student receipt or current-renderer width/depth distortion law was found in the searched corpus. That scoped absence is why the build preserves dense, factorized, and flattened forms rather than declaring a proxy winner.

## Boundaries

- Measured now: exact packet sizes, one untrained complete-container byte row, exact inactive parse-back, pair-0 current/teacher uint8 identity, real-shape CPU RSS, storage availability, tests, watcher validation, and governed dry-runs.
- Not measured: any trained student, n600 student decode fidelity, MPS repeat identity, full-video inactive output identity, Seg/Pose components, or contest score.
- No mechanism was killed. The random-init archive is an apparatus proof only. The 60-epoch primary is a law-fitting launch envelope, not a convergence claim.
- The semantic section is the only trained scope in this build. The 22,161 B carrier remains byte-identical and is stage 2 only after a semantic student holds exact distortion but misses the rate target.
- The pointer did not move. This unit built an imminent exact-row vehicle; it did not achieve the score goal.

## LIVE-HYPOTHESES

- Flattened d4/w64 is worth firing first because full-rank mixing and all four receptive blocks survive while repeated FiLM bytes are removed; its exact tagged container already has rate headroom, but training must show that one FiLM is sufficient.
- Factorized d4/w64/r19 may beat flattened if per-block temporal conditioning matters more than full-rank pointwise mixing; it is the first matched fallback, not a post-hoc reduction.
- Dense d4/w56 may hold the teacher better than either mechanism despite lower width because it changes the inherited computation least; it is the falsification control.
- Once a distortion-holding student exists, #1026 precision waterfill may remove additional bytes because sensitivity can then be measured on the actual student/receiver; applying it before training lacks the required measured inputs.
- If semantic training holds distortion but the exact archive stays above 168,345 B, stage-2 carrier ownership remains plausible because 22,161 inherited carrier bytes are still untouched.

## DEAD-ENDS

- Exact recoding/reparameterization of the frozen 38-tensor teacher remains closed by mz2: every tensor is receiver-required and the tested exact alternatives grew bytes.
- Weight-space MSE is closed for this arm because it does not realize the teacher through resize and uint8; the trainer uses only receiver-output loss.
- A single guessed width is closed: no current-renderer distortion law supports it, so the exact iso-payload form/depth envelope is retained.
- Treating −15,153 B as exact sub-0.15 is closed by arithmetic; it misses by about 0.00000227 before any distortion change.
- Treating the 166,169 B random-init archive as a candidate is closed because no trained or scorer fidelity exists.
- Writing the teacher cache or n600 renders to Vertigo is closed at current free space; the build records the APDataStore waterfall instead.

Vehicle frontier unchanged: S=0.1600920261571558 @ 183,502 B `[contest-CUDA T4, n600]` (e480b SHA-256 `e3e6f440…`).

---

## REBASE NOTE (appended 2026-08-16 by `ddm_fb1`) — APPEND-ONLY, nothing above is changed

**The body above was CORRECT WHEN WRITTEN.** Per Catalog #110/#113 HISTORICAL_PROVENANCE no line
above is rewritten; this is a superseding row.

The `-15,157 B` sub-0.15 rung at `:95` is computed off the **e480b v2** archive (183,502 B). The
frontier has since moved to **hv1 ep0634: S = 0.15959729295498598 @ 182,759 B
`[contest-CUDA T4, n600]`**, sha `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.

**Rebased requirement: -14,413.4 B** (183,502 - 182,759 = 743 B of the old rung is already spent).

**This staleness is in the SAFE direction** — the old rung is 743.6 B TOO TIGHT, so it costs a
missed admit, never a false admit. **No verdict in this file flips:** the measured 17,372 B saving
clears the stale bar by 2,215 B and the live bar by 2,959 B, and the refusal here was on
distortion (`d_seg` 7.0059x), not on rate.

**Use the stale-proof form instead.** `seg + pose` is decode-identical across
`cp135 -> MC36 -> e480b v2 -> hv1` (measured to 1e-15), so only rate moves:

```
sub-0.15  <=>  archive <= 168,345.5977 B
```

This target is identical off every base in the lineage and does not move when the pointer does.
Caveat: it is a PURE-RATE target. Any candidate that changes `d_seg` or `d_pose` — which semantic
width distillation does — must re-measure against the live pointer, not against this number.

Sister still owed to MAIN: the same stale rung appears in this arm's final message at
`.omx/research/arm_final_messages/ddm_wd2_width_distillation_build_20260815T154455Z.md:22`, inside
a LIVE fire-order. That file is untracked (another agent's uncommitted work) and was not touched.

Full derivation + repo-wide sweep: `.omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md`.
