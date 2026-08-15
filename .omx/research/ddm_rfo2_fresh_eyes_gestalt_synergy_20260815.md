# ddm_rfo2 — fresh-eyes gestalt, synergy, and recursive rate attack

**Date:** 2026-08-15
**Authority:** exact retained receipts, scorer-free local inspection, live-run read-only telemetry, and primary online sources
**Score claim:** false for this arm; no scorer, evaluator, Modal job, or new archive ran
**Verdict scope:** current e480b v2 vehicle and the named representation/coder families only

## Conclusion

Rate is the first lever, but **another lossless coder is not the rate lever**. The exact e480b
archive is 183,502 B and needs at least **15,157 B** of savings at unchanged distortion to become
strictly sub-0.15. MZ1 already closed the same-state lossless race at 0 B saved, and MZ2 showed
that all 38 semantic tensors are receiver-required and that four exact semantic
re-representations are each 340 B larger. The remaining multi-kilobyte mechanisms change learned
state or architecture: trained width reduction, carrier rank/atom reduction with re-fit, and
token-map drop/waterfill followed by a coder re-fit.

The live e960 burn has not banked a rate win. Its telemetry reports token bytes separately from
model-plus-token joint bytes; comparing the current roughly 113 KB token number with the old
roughly 131 KB joint number was a category error. Through epoch 600, the best observed advisory
joint estimate remains **130,875 B at epoch 508**, only 345 B below the e480 endpoint estimate of
131,220 B and not a serialized archive. The e960 burn, its watcher PIDs, and its armed closer were
left read-only.

The shortest plausible sub-0.15 composition is therefore:

1. select the best retained burn checkpoint by a distortion-aware joint proxy, never by recency;
2. harvest the existing sub-KB mixed-precision candidate if it survives the current receiver;
3. race a new carrier **rank/atom reduction plus re-fit**, outside PZ4A's closed absolute-code
   coarsening scope;
4. hand the local trainer slot to real nested-width distillation;
5. re-run token drop/waterfill and the coder only on whichever new state survives.

No measured combination currently supplies 15,157 B. That is the honest state: this arm improved
the routing and instruments, not the exact score.

## Verified frontier and byte anatomy

The authority row is the exact retained T4 result at
`experiments/results/modal_auth_eval/ddm_rx2_e480b_hpac_winner_v2_paired_modal_auth_20260815T125117Z_cuda/returned_artifacts/contest_auth_eval.json`.

| quantity | value | label |
|---|---:|---|
| archive | **183,502 B** | MEASURED `[contest-CUDA T4, n600]` |
| archive SHA-256 | `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3` | MEASURED |
| `d_seg` / Seg term | `0.00029611` / **0.029611** | MEASURED, evaluator-rounded component |
| `d_pose` / Pose term | `0.00000688` / **0.008294576541331089** | MEASURED, pose term recomputed by authority receipt |
| Rate term | **0.12218644961582469** | MEASURED |
| exact score | **0.1600920261571558** | MEASURED |
| score per byte | `6.658589531221714e-7` | DERIVED as `25/37,545,489` |

The ZIP has one stored member `p`: 183,402 B plus exactly 100 B of ZIP framing. MZ1's direct
parse gives this complete decomposition:

| scale | bytes | share of archive | rate term | floor/status |
|---|---:|---:|---:|---|
| token payload | **112,749** | 61.443% | 0.0750749311 | same-state coder closed; state/drop open |
| HPAC shipped section | **13,619** | 7.422% | 0.0090683331 | fixed current HPAC coder closed; q3/q4 state open |
| semantic renderer | **34,763** | 18.944% | 0.0231472548 | exact recoding closed; trained width open |
| pose carrier | **22,161** | 12.077% | 0.0147561003 | absolute-code coarsening scoped closed; rank/re-fit open |
| RX1M header + residual + ZIP | **210** | 0.114% | 0.0001398304 | too small to matter; do not optimize first |
| **total** | **183,502** | 100% | **0.1221864496** | exact |

The recursive conclusion is visible at every scale. Tokens plus HPAC are 126,368 B, so their
physical rate cannot be separated: a smaller token alphabet that needs a larger probability model
may lose at the container boundary. Semantic plus carrier are 56,924 B, but exact recoding cannot
remove their information. Tensor/symbol work must change the trained state and then be repriced as
one complete archive.

### Symbol-scale facts

- HPAC's raw 17,996 B object is 4 magic B + 259 learned depth-metadata B + 11,382 packed-weight B
  + 6,351 fixed-parameter B. The trainer's `estimated_model_bytes` is an HPAC estimate, not the
  70,557 B RX1M wrapper.
- The carrier has 27,648 basis symbols costing 12,277 B and 7,200 AR/Rice coefficient residuals
  costing 9,755 B in the FD135 census. This is the concrete reason an atom/rank treatment can be
  multi-kilobyte, while coefficient precision alone was not.
- All 38/38 semantic tensors are strict-load required. All 16 quantized matrices were numerically
  full rank at their shape bound, with no duplicate or zero rows. This is an INSTANCE diagnostic,
  not a family proof against trained smaller networks.

## Derive-first gap allocation

The exact fixed-distortion gap is `0.010092026157155792 S`. A continuous calculation gives
15,156.4023 B, but archive bytes are integral and the target is strict: **15,156 B leaves
S=0.15000026786363616; 15,157 B gives S=0.14999960200468304**, so the archive must be at most
168,345 B.

At this operating point:

- one Seg flip over 600×512×384 pixels is `8.477105e-7 S`;
- one byte is about `0.78548` Seg flips;
- the current Seg term is about 34,930.6 printed-component flips;
- a Seg-only crossing needs about 11,905.0 net flips;
- the local Pose marginal is about `602.8035 S` per unit `d_pose`.

Because the Pose term is nonlinear, fractions of `d_pose` do not save equal fractions of the Pose
term:

| reduction in current `d_pose` | Pose-term saving | rate bytes still needed for sub-0.15 |
|---:|---:|---:|
| 0% | 0 | 15,156.40 |
| 25% | 0.00111126 | 13,487.49 |
| 50% | 0.00242943 | 11,507.84 |
| 75% | 0.00414729 | 8,927.92 |
| 100% | 0.00829458 | 2,699.44 |

This is a feasible-envelope derivation, not an optimum claim: current-vehicle RD curves for carrier
rank, semantic width, and token drop do not yet exist. It changes the route in three ways.

1. It agrees with #1058 that burn harvest is worth doing, but refutes treating the burn as the
   15 KB solution; its remaining advisory runway is hundreds of bytes.
2. It agrees with the js1/#982 joint-distortion line because a real Pose reduction drastically
   relaxes the byte requirement, but it forbids booking the 0.0082946 Pose ceiling: PK4,
   PS135B, and post-hoc carrier results do not realize it.
3. It demotes old #984 byte-only selection. Any semantic/latent or receiver child must be ranked
   by complete `Seg + sqrt(Pose) + rate`, not by a smaller payload alone.

Deleting the entire 22,161 B carrier while holding Seg fixed is only a useful upper envelope: it
would put the old Pose contribution plus new rate at `S=0.1453359258`. Sub-0.15 could then tolerate
`d_pose <= 1.67927e-5`, about 2.44× the current value. No such carrier-free candidate exists. The
number licenses a rank/re-fit race; it does not license deletion.

## Movement 0 — orderable rate ladder

Every projected delta below is against the exact 183,502 B, S=0.1600920261571558 baseline unless
the row says otherwise. A rate-only delta assumes unchanged distortion and is explicitly not a
score result.

| order / disposition | mechanism and projected delta | distortion protection: who, what, where, when, why, how | falsifier | first measurement and named consumer |
|---|---|---|---|---|
| **1 — QUEUED-WITH-A-FIRE-ORDER** | **MZ2 mixed q3/q4 current-state candidate**, already materialized at 182,679 B: `−823 B`, projected `ΔS_rate=−0.0005480019`. The six FiLM sparsity cells are separate alternatives at `−130…−2,051 B`, not additive credits. | MZ2/MAIN owns protection; strict current receiver first, then stratified-random n≥32 Seg/Pose, then one T4 only after admission. The exact rounded state is decoded before rendering so quantization is the real actuator. | Any shipping-receiver mismatch or measured distortion cost ≥0.0005445 S (the registered net gate) closes the candidate. | **Now, $0 local:** add a shipping SD1M receiver and deterministic archive repeat around retained SHA `b3b38b…`; consume `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/SCORE_GATE_RESULT.json`. |
| **2 — QUEUED-WITH-A-FIRE-ORDER** | **Carrier atom/rank reduction plus coefficient re-fit.** A purely proportional section envelope is about 1,846.75 B/atom: rank 10/8/6/4 would gross roughly `3,694/7,387/11,081/14,774 B`, rate projections `−0.00246/−0.00492/−0.00738/−0.00984 S`. These are CONJECTURE, not coded rows. | Carrier owner trains/refits frame 0 against the exact retained output bank, keeps every rank payload, and measures Pose through the shipping receiver. It runs after semantic exact-state work because the carrier is the next physical section; Seg must remain byte-identical because only frame 0 is touched. | Exact complete-archive saving is <2 KB at every rank, receiver parsing changes frame 1, or the nonlinear Pose increase costs at least the rate saving. | **Now, $0 local:** materialize r10/r8/r6/r4, strict-decode and deterministic-repeat each; then queue stratified n≥32. Consumer: MZ2 `CARRIER_QUEUE.json` and a new `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/carrier_rank/` subtree. |
| **3 — FIRED-IN-CODE / endpoint action queued** | **Joint-proxy checkpoint selection** chooses `argmin[(25/37545489)*estimated_joint_bytes + 100*top1_error]`, not the last checkpoint. Through ep600 the read-only live minimum is ep508, 130,875 B joint and top1 0.00189660; this is advisory and un-serialized. | #1058/MAIN consumes the selected SHA in the CPU identity race at burn endpoint. The selector preserves all periodic checkpoints and states that Pose is absent. Identity, receiver, micro-edit Schur compensation, archive repeat, and T4 remain later gates. | No telemetry epoch joins a retained checkpoint, duplicate epoch telemetry makes the join ambiguous, selected payload hash drifts, or identity fails. | **Burn endpoint, $0 local:** `tools/select_hpac_checkpoint.py`, now wired into future `local_endpoint_close.py`; consumer is the #1058 endpoint-closure and identity-race store. The already-running legacy closer must be followed by an explicit selector invocation because its process loaded v1 before this landing. |
| **4 — QUEUED-WITH-A-FIRE-ORDER** | **Nested-width semantic distillation with QAT.** Target a real 4–12 KB archive cut (`ΔS_rate=−0.002663…−0.007990`) rather than untrained slicing. This is a required target band, not a measured expectation. | Width-distill owner trains each width from the exact e480 teacher with deterministic resume and per-stage checkpoints. MC36 top1 and estimated joint rate are in-loop; future watchers enforce joint-byte and top1 bands; endpoint selection uses the proxy; current receiver and T4 close the result. | No width saves ≥4 KB after exact serialization while staying inside the parent terminal top1 band, or receiver-closed n≥32 predicts nonnegative net S. | **After e960 releases the local trainer slot:** run the retained MZ2 `DISTILL_QUEUE.json`; consumer `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/distill/`. |
| **5 — QUEUED-WITH-A-FIRE-ORDER** | **Coder×drop token waterfill on the surviving new state.** Race per-cell drop levels and refit HPAC/coder jointly. A 4–20 KB accepted cut corresponds to `ΔS_rate=−0.002663…−0.013317`; no #869 byte value transfers. | #869/#933 successor computes the actual per-cell net-S marginal, materializes every payload, retains model+tokens together, and uses joint Seg/Pose gates. QS5 compensation may protect a sparse accepted edit, but it earns no independent credit. | At all nonzero drops, complete model+token bytes fail to fall by ≥4 KB or random n≥32 distortion consumes the rate benefit. | **After a width/carrier state exists, $0 local first:** exact drop×coder sweep on that object. Consumer: #869 token-waterfill store resolved through the current harness bridge, never the old IX2 price. |
| **6 — QUEUED-WITH-A-FIRE-ORDER** | **#978 semantic-vs-latent tokens × #982 trained receiver.** The useful target band is 8–20 KB (`ΔS_rate=−0.005327…−0.013317`) with receiver-realized distortion. This is CONJECTURE. | The #978 owner changes the representation; #982 consumes it during joint training, with no explicit edge/GT table shipped. A sealed T4 sign gate precedes any long train, and all learned/video-derived state remains counted. | The sealed current-object T4 sign is not positive, the receiver needs a counted side model that erases the saving, or no receiver-closed child beats the exact baseline. | **Sealed MAIN fire only after scorer ownership:** consume #978/#982 stores and the task-status bridge; no Modal dispatch was made here. |

### What is deliberately absent

- No new same-state Brotli/LZMA/RC64/SMEVR row: MZ1 measured 8/8 complete alternatives and the
  current split-Brotli object won.
- No wrapper/ZIP project: the entire header, residual, and ZIP budget is 210 B.
- No transferred `−113,555 B` from #869: it is an old IX2 projection with later scorer harm and
  the coder×drop surface is unmeasured on this state.
- No sum of MZ2's q3/q4 and FiLM cells: they change overlapping tensors and require one rebuilt
  archive.
- No scalar-only measurement: every future materializer above is payload-retaining by construction.

## Distortion protection is now an instrument

The live continuation's top1 band through ep600 is 0.00189660–0.00193158
`[macOS-MPS advisory telemetry]`. It is bounded and visually flat; it is not a contest Seg result.
Inspection found two real watch-time failures:

1. the generated quality config had `joint_regression=false` and used `top1_error` only as a finite
   check;
2. the live liveness watcher died while trying `os.link` on APDataStore (`OSError 45: Operation not
   supported`). The quality watcher remained alive, but its future alert path used the same
   hard-link-only publication primitive.

The code landing cures future launches:

- `tools/fire_watched_continuation.py` now forces joint-byte regression on and derives a
  `top1_error` upper band from the last `best_not_latest.min_rows` observations in the parent's
  terminal phase. It refuses to invent a band if `top1_error` is not already finite-checked.
- `tools/run_quality_poller.py` supports typed upper regression bands independently of the joint
  byte condition.
- both watcher tools now publish a complete JSON alert by same-directory atomic replace guarded by
  an atomic publish-lock directory, so no hard-link feature is required and an existing alert is
  never overwritten.
- `tools/select_hpac_checkpoint.py` binds telemetry and retained checkpoint epochs, hashes the
  selected payload, declares the exact proxy, and states that Pose/receiver/score remain owed.
- future `local_endpoint_close.py` v2 consumes that receipt before emitting the #1058 identity-race
  order. Legacy v1 receipts remain replayable, which is necessary because the sacred live closer
  already loaded v1 code.

Verification: **45 focused tests passed**, including continuation composition, independent top1
band alerting, SSD-compatible one-shot alert publication, joint-proxy selection, closer integration,
and v1 receipt compatibility. Two review-tracker passes cover every changed Python file. No live PID,
watcher config, checkpoint, or closer store was written.

Protection by place is therefore:

| place | protection | residual boundary |
|---|---|---|
| train | QAT uses MC36 top1 and rate pressure in-loop | MC36 is a proxy, Pose absent |
| watch | joint-byte + derived top1 upper bands | live process still has old disabled config |
| endpoint | joint-proxy checkpoint argmin with retained SHA | live legacy closer needs explicit selector follow-up |
| compile | QS5 Schur compensation protects Pose for compatible micro-edits | QS5 itself was a +2.52e-6 S near miss |
| lossless gate | CPU identity race requires byte-identical decode | no distortion credit until identity |
| authority | one composed retained archive on T4 | sole scorer lane and MAIN dispatch govern |

## Movement 2 — hybrid and synergy matrix

The matrix follows GC20's law: synergy is state-dependent and only a complete container can decide.

| composition / disposition | mechanism, projection, and baseline | falsifier | first measurement / consumer |
|---|---|---|---|
| **MZ2 q3/q4 × NeuroQuant-style network calibration — RACE** | Recalibrate the four selected low-bit tensors jointly rather than independently. Projection remains only the measured `−823 B`, `−0.000548 S` rate leg on e480b; no literature accuracy transfers. | Joint calibration cannot beat the existing rounded state at equal exact bytes, or receiver distortion erases 823 B. | $0 local on the retained mixed-bit archive; MZ2 `SCORE_GATE_RESULT.json`. |
| **carrier rank × exact coefficient/Pose re-fit — PRIORITY** | Rank removes basis and coefficient streams together; re-fit uses remaining atoms to preserve frame-0 Pose. The derived r6 envelope is `−11,081 B`, `−0.007378 S` before distortion. | Every rank with ≥4 KB exact saving has nonnegative full net S. | Scorer-free rank materialization now, random n≥32 later; MZ2 carrier store. |
| **nested width × QAT × coder re-fit — PRIORITY** | Width changes tensor count and token probabilities, QAT protects labels, then coder is rebuilt. Target 8 KB gives `−0.005327 S` at e480b before distortion. | Model+token complete bytes do not improve by ≥4 KB at the protected top1 band. | Local trainer after e960; MZ2 distill then #869 coder consumer. |
| **token drop × #982 trained receiver — CONDITIONAL** | Drop changes task support while receiver training can absorb missing cells. Target 8–20 KB; no old #869 number transfers. | Receiver must ship enough extra learned state to erase the saving, or exact sign is nonnegative. | #978/#982 sealed sign gate, then deterministic resumable train. |
| **sparse micro-edit × QS5 compensation — FOLDED AS CURE** | Proven in-compile Pose compensation can accompany a rate-positive sparse edit; it earns no separate delta. | The parent edit has no positive Seg+rate margin before compensation. | Consume only inside #1058/#978 candidates; do not run another qs singleton family. |
| **level-set preimage solver × trained receiver — LESSON/CONDITIONAL** | GN/CG and R-adjoint apparatus can provide a task-space teacher, while the counted receiver distills it. Projection target is ≥8 KB or a measured Seg/Pose gain of equal score value; no current number. | Solver state requires explicit per-frame carriage or receiver realization stays below its byte cost. | Only after #982 has a receiver-closed positive control; HY1/#1015 corpus is the apparatus source. |
| **f26p CPU port × exact archive — DUAL-AXIS, NOT RATE** | Native integer decode may make a candidate contest-CPU viable but cannot lower archive S by itself. Projected `ΔS=0`. | CPU decode remains outside budget or differs bytewise. | Existing f26p/RC64 CPU consumer; never add runtime as score credit. |

Antagonisms that must remain explicit:

- coder savings before state selection are not additive after width/drop;
- carrier rank and PZ4A absolute coarsening are different mechanisms, but stacking them before a
  rank-specific sensitivity map is invalid;
- semantic q3/q4 and FiLM sparsity overlap;
- post-hoc receiver overlays, explicit per-flip streams, and dense GT tables are closed or illegal
  substitutes for joint training;
- a task-space solver that ships its solved field is a rate loss, not free computation.

## Movement 3 — online sweep and public bar

Only primary papers/official proceedings are used for technical routing.

| source | disposition | exact local lesson |
|---|---|---|
| [RNeRV, 2025](https://arxiv.org/abs/2506.24127) | **RACE** | Training with weight masks supports a nested-width/suffix family instead of post-hoc channel deletion. Its PSNR/MS-SSIM numbers do not transfer to MC36 or contest S. |
| [NeuroQuant, ICLR 2025](https://arxiv.org/abs/2502.11729) | **ADOPT APPARATUS / RACE VALUES** | Network-wise calibration, mixed precision, and channel-wise quantization directly sharpen MZ2 q3/q4. The reported INT2 behavior and speed do not transfer. |
| [Rate-constrained quantization and entropy coding, 2025](https://arxiv.org/abs/2505.18758) | **RACE** | Use a local task Hessian/OBS-style quadratic with an explicit rate term for HPAC/semantic groups. The paper's 20–40% generic-network bitrate result is not a Pact projection. |
| [NVRC, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eed57814c16645298db3164829e2e45c-Abstract-Conference.html) | **LESSON-ONLY** | Optimize representation, quantization, entropy model, and side-information bytes jointly and hierarchically. It supports the complete-container objective, not a numeric transfer. |
| [Integer-Centric Neural Video Compression, ICLR 2026 submission](https://openreview.net/forum?id=KCQo0fXtFH) | **LESSON-ONLY** | Integer-from-scratch/in-loop design reinforces deterministic cross-platform QAT. RX2 already has integer receiver discipline; no claimed bitrate transfers. |

Public activity was checked on [PR135](https://github.com/commaai/comma_video_compression_challenge/pull/135),
[PR130](https://github.com/commaai/comma_video_compression_challenge/pull/130), the public PR list, and the
linked ExperimentBook repository. PR135 is closed; the maintainer wrote “added to leaderboard” on
August 8 and the last visible conversation activity is “sending you an email” on August 10. PR130
was frozen by July 19 and officially evaluated/closed July 21. I **did not find in the checked public
scope** a post-August-10 archive, score, or open challenge PR that establishes an imminent public-bar
move. The ExperimentBook page did not expose a reliable dated post-August-10 signal through the
available public view. Private work can still move the bar, so this is a bounded absence, not a claim
of inactivity.

The public page also remains cache-lagged in some views. Local routing should use the source-reported
PR135 0.162 row already in Pact's intake, not regress to a stale public 0.172 display.

## Movement 4 — recursive optimal-form audit

| pipeline element | family-optimal grade | finding and next owner/fire order |
|---|---|---|
| MC36 labels | **useful but incomplete** | Task-aligned Seg supervision is active; Pose is absent. #982 owns joint receiver training after a positive representation sign. |
| HPAC training | **partial** | QAT and rate pressure are real, but rate is an estimator and excludes physical semantic/carrier sections. Width-distill owner must select by serialized complete bytes. |
| QAT schedule | **partial, live stable** | top1 is flat in the observed band. Future continuation generator now enforces joint and top1 regression; the current launch remains legacy/read-only. |
| token emission | **not optimal** | 112,749 B dominates and no current coder×drop point exists. #869 successor fires only after a new state and retains every model/token payload. |
| coder | **optimal at current same-state scope** | MZ1 8/8 race closed. Reopen only after probabilities/symbols change. |
| pack | **optimal enough** | One stored member, 100 B ZIP, 14 B RX1M header. Packaging cannot supply the gap. |
| semantic renderer | **exact recoding closed; architecture open** | MZ2's 38/38/full-rank census kills naive exact factorization. Trained nested-width distillation remains open. |
| micro-edit compiler | **optimal as a cure, not a main lever** | QS5 proves Schur compensation but the candidate lost by 2.52e-6 S. Fold it into qualifying parents only. |
| pose carrier | **not family-optimal** | 22,161 B is large. PZ4A kills absolute sensitivity coarsening; PK4 kills linear overlays; atom/rank re-fit is a distinct TODO owned by MZ2 carrier successor. |
| archive selection | **exact at authority, weak at endpoint** | Exact T4 pointer is sound. Endpoint used “final” semantics; new selector uses the declared joint proxy and selected SHA. |
| liveness instrument | **failed on live SSD** | The live liveness watcher crashed on hard-link publication. Future watcher publication is filesystem-portable; current run is still monitored by quality telemetry/receipt/closer surfaces, not by that dead process. |
| quality instrument | **future optimal form built** | It now has joint and top1 channels. Current PID 47776 loaded the old config and remains untouched. |
| identity/worker/gate | **correct order** | CPU identity precedes QS2/RE1 recompile; one retained T4 row is authority. MAIN owns fire and sole-lane checks. |

The flatten/factorize/distill doctrine is not a request for another SVD file format. Flattening here
means pricing the actual archive→section→tensor→symbol tree; factorization means removing trained
degrees of freedom such as carrier atoms or renderer width; distillation means re-learning the
smaller receiver. MZ2 closed exact low-rank recoding, not those architecture changes. The
flattened-KKT and score-waterfill equations therefore apply only after real coder bytes and
receiver-realized distortion are available at each rung.

## RECALL EVIDENCE

Recall preceded the verdict and searched the full `.omx/research/` corpus, arm final messages,
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, main hot state, task/queue/bridge surfaces,
the actual evaluator, and `.venv/bin/python tools/list_canonical_equations.py --json`. Content query
families included:

- `e480b|RX1M|IHS1|estimated_model_bytes|model section|token payload|archive framing`;
- `mixed q3|FiLM sparsity|semantic renderer|carrier rank|CAP1|Rice|basis symbols|coefficient`;
- `#869|token waterfill|drop level|coder x drop|#933|#996|width distill|sparse learned prior`;
- `flatten|factorize|distill|SVD|low-rank|KKT|AWARE|adaptive precision|sub-int16`;
- `#978|#982|#1058|identity race|QS5|RE1|level-set|GN|CG|R-adjoint|f26p`;
- `liveness watcher|quality watcher|joint_regression|top1_error|endpoint checkpoint`.

The equation registry contributed `score_marginal_lagrange_multipliers_v1`,
`score_atomic_flip_byte_exchange_v1`, the rate/MDL waterfill law, and the weight-entropy-in-loss
lever. They supplied exchange rates and decision form, not unmeasured candidate values.

Findings beyond the charter seeds that changed the plan:

1. MZ1's byte autopsy corrected the model-attribution error and exposed the exact 112,749/13,619/
   34,763/22,161 decomposition. That removed same-state coding and wrapper work from the ladder.
2. MZ2's 38/38 receiver census and exact +340 B representation race closed naive semantic
   factorization, while its retained −823 B and FiLM cells supplied the immediate small rungs.
3. FD135's 27,648 basis-symbol and 7,200 coefficient census identified carrier atom/rank reduction
   as a new structural mechanism outside PZ4A's absolute-code negative.
4. PZ4A showed that its best gross 500 B coarsening lost 2,232 B after depth-map cost; PK2/PK4 and
   PS135B prevented relabeling another overlay or post-hoc carrier as new.
5. The live log and WC2 receipts exposed both the joint-vs-token category error and the hard-link
   liveness crash. Those changed the work from prose protection to code-level watcher/selector cures.
6. RFO1/EU4/NA7 kept QS5 as compensation-only, closed the enumerated singleton/micro-edit scopes,
   and required new token support or joint training rather than another proposal sweep.

## Measured boundaries and goal status

**Measured here:** exact existing frontier receipt and archive decomposition; exact current payload
bytes/hashes from retained receipts; current live log through ep600 read-only; watcher failure trace;
public primary-source pages; code behavior through 45 focused tests.

**Derived here:** strict 15,157 B crossing; exchange rates; nonlinear Pose/rate allocation table;
carrier per-atom envelopes; projected rate-only deltas.

**Not measured here:** any new archive, new coder payload, carrier-rank candidate, semantic-width
child, token-drop cell, current-vehicle MZ2 distortion, CPU/CUDA result, public submission, or exact
score. No scorer, Modal dispatch, training launch, process stop, live config mutation, upstream edit,
or payload deletion occurred.

The pointer is unchanged and still above the mission target. This arm produced means and queued
measurements, not goal progress.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — action: run `tools/select_hpac_checkpoint.py` explicitly on the completed e960 log/periodic directory, then feed its selected SHA to the CPU identity race; owner: MAIN #1058 endpoint adjudicator; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full_e480b_e960/endpoint_closure/` plus the RX2 identity-race store; fire trigger: `rx2_wc2_full_mps_e960.done` is a real rc=0 receipt and the live legacy closer has closed.
- **QUEUED-WITH-A-FIRE-ORDER** — action: add the shipping SD1M receiver around the retained 182,679 B q3/q4 archive and perform deterministic repeat/strict parse-back before any score request; owner: MZ2 mixed-precision successor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/`; fire trigger: the exact retained q3/q4 SHA is reverified and no scorer is needed for the first byte/receiver stage.
- **QUEUED-WITH-A-FIRE-ORDER** — action: materialize and retain carrier r10/r8/r6/r4 atom-drop plus coefficient-refit archives, recording exact bytes and strict receiver equality per candidate; owner: MZ2 carrier-structure successor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/carrier_rank/`; fire trigger: the semantic q3/q4 byte/receiver stage is terminal and the exact e480 carrier/output bank is under custody.
- **QUEUED-WITH-A-FIRE-ORDER** — action: launch deterministic nested-width distillation with per-stage EMA checkpoints and future watcher v2 protection; owner: width-distillation trainer owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/distill/`; fire trigger: e960 has terminated, released the governed local trainer slot, and storage/governor preflight passes.
- **QUEUED-WITH-A-FIRE-ORDER** — action: run actual drop-level × HPAC/coder waterfill on the first admitted new learned state, retaining model and token payloads at every cell; owner: #869/#933 token-waterfill successor; consumer store: the current harness-bridge-resolved #869 store; fire trigger: a q3/q4, carrier-rank, or width child has a receiver-valid retained archive and a scorer-lane-safe distortion plan.

## LIVE-HYPOTHESES

- Carrier atom/rank reduction with rank-specific re-fit can recover several kilobytes because the exact section spends 12,277 B on 27,648 basis symbols and 9,755 B on coefficients; PZ4A tested coefficient-value coarsening, not removal of jointly re-fit atoms.
- MZ2's mixed q3/q4 state may be a small real first rung because it already saves 823 complete-archive bytes and changes only four named tensors, but its current receiver and distortion are still unmeasured.
- Nested-width training can beat exact semantic recoding because it removes trained channels rather than attempting to represent 16 full-rank matrices losslessly; RNeRV's trained masking makes a nested family more plausible than post-hoc slicing.
- Coder×drop may become useful only after a representation change because MZ1 closes the fixed stream while #869's actual current-object drop surface is absent; joint re-fitting can move both model and token entropy.
- A large carrier-rate win can tolerate some Pose regression and still cross sub-0.15: complete carrier deletion's envelope allows `d_pose` up to about `1.679e-5`, but only a materialized rank/re-fit curve can reveal whether any intermediate point lies inside it.

## DEAD-ENDS

- Same-decoded-state lossless coder work on e480b is closed at INSTANCE/FORMULATION scope: MZ1's 8/8 complete race saved 0 B.
- The alleged 52,566 B model-serialization gap is closed as a category error: 17,991 B was HPAC-only while RX1M also contains semantic and carrier state.
- Naive exact semantic deletion, zero/one derivation, sparse rows, row dictionaries, and another pointwise low-rank/VQ sweep are closed on the tested current state by MZ2 and its predecessor censuses.
- Absolute-code sensitivity coarsening of the carrier is closed on PZ4A's tested instance: 500 B gross became 2,232 B net growth; this does not close atom/rank re-fit.
- Linear frame-0 overlays and the tested post-hoc carrier fits are closed by PK4/PK2 at their scopes; they are not substitutes for joint or rank-specific training.
- Another singleton semantic micro-edit sweep is closed in the enumerated JS6B/QS scopes; QS5 survives only as compensation machinery inside a different rate-positive parent.
- Treating the live e960 token number as an 18 KB joint/archive win is closed. The best observed joint estimate is 130,875 B and remains advisory/unserialized.
- Packaging and ZIP work cannot close the gap: all header, residual, and ZIP bytes total only 210 B.
- Public-bar panic is unsupported in the checked public scope: no post-August-10 score/archive/open-PR signal was found, though private work remains possible.

Own-vehicle frontier remains **S = 0.1600920261571558 @ 183,502 B `[contest-CUDA T4, n600]`**, archive SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`; pointer unmoved.
