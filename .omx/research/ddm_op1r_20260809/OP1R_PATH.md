# OP1R — the optimal path forward from PR130

Tags: `[no-triality] [p0-ledger-ok]`

Axis: `[source/receipt inspection; scorer-free byte and target-content measurements; macOS-CPU advisory TOY-BRACKET receiver execution]`.
`score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`,
`full_n600_scorer_forwards_run=0`, `metal_jobs_run=0`, `paid_dispatches=0`.
Fresh-measurement reconstruction config:
`.omx/research/ddm_op1r_20260809/OP1R_REPRODUCTION.json`, SHA-256
`43fa12d89a2144c343ac025d6cdbf90dc6a72ffe55dd95a24b5eb29a9f3327e6`.
The review-tracked producer is `tools/reproduce_ddm_op1r_measurements.py`,
SHA-256 `31d737b6c3ee495e9bbcef60ca75d99f414754543e50f59c5c9410d1e776e676`.
Its two required tracker passes covered all 32 entities. The consolidated
receipt `.omx/research/ddm_op1r_20260809/OP1R_REPRODUCTION_RECEIPT.json` is
`113,594 B`, SHA-256
`852286a65d91cec811f7fd1c07d8e8ec7a46c0e42ee65a077e5df8bc67062af6`,
and is `PASS`. It freshly reran the target-cache section and hash-verified the
two preserved fresh receiver stages plus both preserved fresh XZ stages. The
cold v3 XZ stages began at row zero under one exclusive writer and completed
`675 + 2,700` decoded roundtrips. An earlier overlapping-writer attempt and a
later under-specified resume-identity attempt are quarantined and explicitly
non-promotable in `OP1R_MEASUREMENTS.json`; neither contributes a result here.

The common contract's copied `qo1` pointer is an older snapshot. Its own
full-hot-state rule plus the later OP1R charter identify `tq1c` as the current
own-vehicle row; this memo therefore uses the charter-required `tq1c` ending
without editing either governing input.

## Decision

**Use PR130's exact-DALI-GT semantic vehicle as the base. Do not use VEH as the
base, and do not build a VEH/GT hybrid. First align PR130's renderer training to
the exact DALI label object it actually ships; then attack the exact same object's
rate losslessly; only then buy task-lossy rate with a joint re-solve.**

This decision is stronger than the charter's inherited arithmetic because the
charter's premise was superseded by direct source inspection:

- `[MEASURED source]` PR130's cache builder runs the official `DistortionNet` on
  the original videos and stores `segment.argmax(1)` as `uint8`; these are the
  GT SegNet labels, not an approximate semantic field
  (`repro_repo/code/build_gt_cache_official.py:31-65`).
- `[MEASURED source]` the semantic trainer loads that exact `seg` tensor, renders
  from it, and measures `pred != target`; PR130's segmentation distortion is
  therefore realization error against exact labels
  (`repro_repo/code/train_semantic_full.py:26-39,65-89,110-134`).
- `[MEASURED contest-CUDA]` the only measured CPR1/bot tuple is
  `d_seg=0.00029660`, `d_pose=0.00002331`, and `191,052 B`, which recomputes to
  `S=0.17214129749189644`. `[MEASURED external RTX 2000 Ada, non-1:1]` The
  `0.00028609/0.00001967` distortions belong to the `194,380 B` source archive.
  Combining them with CPR1's `191,052 B` gives
  `S=0.16984766243023947`, but that is a **PROJECTED mixed-object row**, not a
  measurement (`.omx/research/pr86_pr130_fullstack_intake_20260728.md:133-152`).
- `[MEASURED byte-only]` the canonical CPR1 archive is `191,052 B`, SHA-256
  `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
  Its stored member is `190,952 B`; its compressed model bundle is `73,968 B`
  and its token stream is `116,980 B`. The renderer, carrier, and HPAC raw
  sections live inside that one XZ bundle, so their raw sizes are not additive
  charged archive shares.

`[DERIVED from the measured CPR1 bot tuple]` the fixed-distortion sub-`0.15`
byte ceiling is `157,799 B`; CPR1 must lose at least `33,253 B`, or `17.4052%`
of the archive. If all of the cut came from the token stream, it would be
`28.4262%` of `116,980 B`. The rate slope is
`25/37,545,489 = 6.658589531221714e-7 S/B`; one `1e-6` change in `d_seg` is
worth `150.181956 B`. These are exchange rates, not forecasts of attainable
gain.

## The vehicle adjudication

The price comparison must use like-for-like HPAC objects, not the killed AV
epoch-54 estimate:

| object | state and axis | token bytes | separately packed HPAC-model bytes | isolated HPAC package | interpretation |
|---|---|---:|---:|---:|---|
| PR130 DALI GT | `[MEASURED byte-only n600]` | 116,980 | 15,164 | 132,144 | exact labels shipped by CPR1; model price from `hpac_selfcompress_l1_fastbits_e60.pack.json` |
| tq1c VEH | `[MEASURED byte-only n600]` | 97,928 | 14,116 | 112,044 | exact decode verified by HB2 (`ddm_hb2_20260808/HB2_FINDINGS.md:63-101`) |
| local AV GT | `[MEASURED telemetry; PROJECTED package estimate]` | 115,600 | 20,132 | 135,732 | epoch 54 of 60, terminated with `rc=143`; not a final pack and not the DALI object |

`[MEASURED isolated-package]` the DALI-GT package exceeds the VEH package by
`20,100 B`. `[PROJECTED package-equivalent]` treating that isolated delta as
archive bytes would be `0.013383764957755643 S_rate` and would break even after
`0.00013383764957755644 d_seg` of improvement. This is **not a measured archive
delta**: CPR1 jointly XZ-compresses renderer, carrier, and HPAC. `[MEASURED macOS-CPU
advisory]` tq1c's plateau is `d_seg=0.004305420`; `[MEASURED contest-CUDA]`
CPR1 realizes exact GT at `d_seg=0.00029660`. Their gap is `0.00400882 d_seg`,
or `0.400882 S_seg`, roughly thirty times the projected package-equivalent
premium. The axes are not interchangeable, so that ratio is an
existence/routing fact rather than a predicted local score.

The conclusion is unambiguous:

- **GT-primary wins.** It is the only n600 deployed existence proof here, and
  its measured HPAC package buys a renderer-accessible target rather than a
  self-consistent error ceiling.
- **VEH-primary loses and is FOLDED as a vehicle.** It remains a useful entropy
  datum only. Its cheaper stream cannot repair labels it never carries.
- **The VEH/GT hybrid loses and is FOLDED as a base.** Starting from VEH preserves
  the wrong-label plateau; adding GT corrections then recreates an exact-label
  stream less directly, with a second ownership boundary and no measured rate or
  realization advantage. A future *lossy mutation of exact DALI GT* is allowed,
  but that is an optimization of the GT base, not a hybrid vehicle.

The old “GT label error versus PR130 semantic approximation” explanation is
withdrawn. The live problem is now cleaner: **exact DALI target content,
renderer realization, and rate**.

## Scoped content mismatch: official DALI versus local AV-like caches

I compared preserved full-population caches to establish target identity and
whether the current M1 cache matches the shipped object. This is **not the
same-host #906 experiment**: official DALI was produced on Ada, while the local
AV cache was produced on macOS CPU, so decoder, platform, and pipeline are not
isolated. The detailed hashes, class counts, and instrument are in
`OP1R_MEASUREMENTS.json`.

| quantity | result | label |
|---|---:|---|
| compared sites | 117,964,800 | `[MEASURED scorer-free n600]` |
| official-DALI vs local-macOS-AV mismatches | 20,749 | `[MEASURED scorer-free n600, cross-platform/pipeline]` |
| pooled disagreement | 0.0001758914523654514 | `[MEASURED scorer-free n600]` |
| pairs with any disagreement | 600/600 | `[MEASURED scorer-free n600, cross-platform/pipeline]` |
| per-pair mismatch count | min 15; median 33; mean 34.5817; max 103 | `[MEASURED scorer-free n600]` |
| disagreements on DALI four-neighbor boundary endpoints | 20,504/20,749 = 98.8192% | `[MEASURED scorer-free n600]` |
| disagreement scale in segmentation score units | 0.01758914523654514 | `[DERIVED: 100 times measured target disagreement]` |
| share of CPR1 bot `d_seg` | 59.3026% | `[DERIVED scale comparison; not recoverable gain]` |

The official DALI segmentation tensor has raw SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.
That is exactly the deployed decoder's retained n600 token golden. The local AV
tensor has raw SHA-256
`f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557`.
The final E2E invocation requested DALI, and the official/shipped cache
`gt_cache_600_official_ada.pt.xz` (SHA-256 `233884c6...`) supplies that deployed
token identity. A separately retained earlier artifact, `gt_cache_600.pt.xz`
(SHA-256 `67fa351a...`), differs from local AV at only one of `117,964,800`
sites and from official DALI at `20,750` sites. Its exact historical consumption
role is not inferred; the measured content identity forbids treating that
earlier artifact as the DALI target but does not undo the shipped-DALI result.

The stored pose targets also differ materially:

| quantity | result | label |
|---|---:|---|
| compared pose coordinates | 3,600 | `[MEASURED scorer-free n600, cross-platform/pipeline]` |
| exact unequal coordinates | 3,600/3,600 | `[MEASURED scorer-free n600, cross-platform/pipeline]` |
| pairs with any target difference | 600/600 | `[MEASURED scorer-free n600, cross-platform/pipeline]` |
| target-to-target MSE | 0.00014004340079290477 | `[MEASURED scorer-free n600, cross-platform/pipeline]` |
| target-to-target MAE / max absolute | 0.003892889264971018 / 0.09792327880859375 | `[MEASURED scorer-free n600, cross-platform/pipeline]` |
| root-pose-term separation scale | 0.03742237309323191 | `[DERIVED: sqrt(10 times target MSE)]` |
| scale relative to CPR1 bot pose term | 2.451095245371652 | `[DERIVED scale comparison; not candidate harm or gain]` |

The official DALI pose tensor SHA-256 is
`23ae28d20aee8697d87e015c1145b248c111bc3ce61b9b66793e770d65522b2a`;
the PR130 original AV-like pose tensor SHA-256 is
`6133af44619b5b7dcc33cc21a801a9599051e1b7192886d6d96525617877f344`.
This is target-to-target separation, not a candidate `d_pose` delta: the
candidate residual direction controls the nonlinear score response.

`[MEASURED source receipt]` PR130's carrier training evidence explicitly names
`gt_cache_600_official_ada.pt` as its target
(`archive_carrier_int6_coefftail_s4k.json:2-9` and
`archive_carrier_int6_stable_s8k.json:2-9`). The base carrier is therefore
already DALI-targeted; this measurement does not reopen pose-first redesign.
It does forbid substituting AV pose targets and requires a DALI-targeted carrier
reconciliation if changed semantic master frames move PoseNet.

`[MEASURED custody]` the current M1 target
`/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt` contains
the AV hash, not the DALI hash. This is a new launch blocker in addition to the
fifteen findings on the pinned reviewed ticket in
`.omx/research/ddm_m1r4_20260808/M1R5C_REVIEW.md:42-60`. No current M1 fire
command may be reused unchanged.

`[INFERRED mechanism, bounded]` the near-total segmentation concentration on class boundaries
is consistent with decode/chroma/convention perturbations moving marginal
argmax cells. It does not establish which individual pixel differences the
renderer can recover, does not measure pose drift, and is not a candidate
CPU-to-CUDA score offset. In particular, `0.017589 S` is a scale, **not a
promised improvement**.

`[MEASURED custody]` the retained official target is materialized durably at
`/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt`,
`117,981,301 B`, SHA-256
`382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`.
The source XZ remains preserved. This gives an immediate exact shipped-target
asset but does **not** close #906. The qualifying same-host Modal job is
**QUEUED-WITH-A-FIRE-ORDER:** recover/pin PR130's challenge commit `d3f688f...`;
claim the #906 lane; land the owned, hash-pinned `<=120`-pair chunk adapter and
its `--resume-from`/atomic-output/content-ledger controls; run that adapter for AV
and DALI in one GPU container; retain both caches plus the segmentation and
per-pose-dimension diff; compare the new DALI tensor hashes to the retained
official tensors; only then mark #906 closed. The original loop-end-only
`build_gt_cache_official.py` is a read-only source reference, not the n600
launcher. The same-host job is a prerequisite to the renderer burn under the
hot-state directive, even though the retained official DALI asset already tells
us which target M1 must consume.

## MATCH execution

The correct provenance pin is source repository HEAD
`e34f31bc4969042c0051ac81aa3c56884419a231`. `LIFTED_AT_HEAD` in our lift is a
different quantity and is not used as the executed source identity. The public
intake clone was not modified. Its `inflate.py` SHA-256 is
`335369c9b3b295707f1790feb0b5b7ae288338fae350056cc4bb03aaa18f0c9e`.

`[MEASURED TOY-BRACKET, macOS-CPU advisory, scorer-free]` I ran the real source
decoder and renderer twice on the first pair. The procedure unpacked the full
n600 model/carrier bundle, loaded the full HPAC model, then bounded only the
arithmetic decode/render population to the first real field. Both runs produced
the same token SHA-256
`8f6f0d8679c5701967a782cb4bc53f057d02c0498c570fa537c40fe31ad41789`
and the same `6,104,016 B` raw output SHA-256
`88ecb9180a8555f33799446d705590245044606703b7fb78c340eb2af9bc8609`.
`[MEASURED historical one-off, argv UNDETERMINED]` wall times were `2.7531 s`
and `2.4430 s` on Python `3.14.6`, Torch `2.11.0`, four CPU threads, with
deterministic algorithms enabled. `[MEASURED fresh governed rerun]` the final
repeat measured `5.3621 s` and `4.8697 s` for the same bounded decode/render
section; the shorter pair is retained only as non-authoritative timing. This
proves the real source
dependency/device/integrated path runs locally and is repeatable at that bound.

This row is explicitly a **TOY-BRACKET**. It cannot produce a family verdict,
full MATCH, score, or timing extrapolation.

Full archive MATCH was not fired because the shipped arithmetic decoder is one
monolithic stateful pass with no persisted seek checkpoints. `[MEASURED retained
source evidence, CPU model unspecified]` full token decode alone took
`2,197.6 s`, exceeding thirty minutes on that retained CPU; contest-CPU runtime
and CPU rendering are both `UNMEASURED`. `[DERIVED from the fixed output
geometry]` a full output is `3,662,409,600 B`.
Firing it ad hoc would violate the P0 resumability rule.

**Full MATCH is QUEUED-WITH-A-FIRE-ORDER:** (1) vendor an owned, hash-pinned
minimal CPU adapter; (2) re-run the already-passed import, archive-parse, n1
source-versus-adapter, and repeatability controls; (3) add atomic per-frame
render progress and storage receipts; (4) request an explicit narrow operator
source-reproduction exception for the immutable uncheckpointable entropy seam;
denial blocks full MATCH rather than branching to an unproved decoder rewrite;
(5) decode all n600 tokens and require
shape `[600,384,512]` plus the golden `c5c767...` hash; (6) render to SSD,
require the exact raw length, same-host repeatability, and source-versus-adapter
equality. A scorer run is a later separately claimed lane. The current result is
the maximal compliant execution, not a claimed full MATCH.

## PR130-class prices already paid

Only same-object prices are credits. Everything else below remains a hypothesis.
“Promoted” here means admitted into a candidate, not merely queued as the
experiment that obtains a price. Steps 3, 5, and 6 are price-acquisition gates;
their mechanisms are not promoted until a PR130-class receipt passes its
falsifier.

| lever/object | measured object | byte result | score result | disposition |
|---|---|---:|---:|---|
| exact DALI target alignment | PR130 renderer target schema | `[DERIVED] 0 B` schema change; compressed-weight delta unmeasured | `[DERIVED scale] 0.017589 S_seg`; recoverable d_seg gain and d_pose response **UNMEASURED** | QUEUED-WITH-A-FIRE-ORDER (Step 2) price-acquisition experiment; not promoted |
| XZ filter race | exact CPR1 raw model bundle | `73,968 -> 73,960 B`; archive `191,052 -> 191,044 B` | `[DERIVED] -0.000005326871625 S_rate`; no evaluator run | FIRED; retain candidate; further generic XZ tuning FOLDED on this bundle |
| DALI-GT HPAC versus VEH HPAC | exact n600 isolated HPAC packages | `[MEASURED package-only] 132,144` versus `112,044 B`; `+20,100 B` | `[PROJECTED package-equivalent] +0.01338376496 S_rate`; full-archive delta unmeasured | GT base accepted by deployed existence proof, not this projected archive premium |
| edge-conditioned causal context sizing only | CR1 selected-support ratio applied to CPR1 token bytes | `[PROJECTED, weak cross-object basis] 19.221% * 116,980 = 22,485 token B before added model/tag cost` | `[PROJECTED sizing] -0.0149717 S_rate before overhead` | NOT PROMOTED and not a byte credit; Step 3 must obtain the same-object full-archive price |
| sub-0.15 fixed-distortion requirement | CPR1 bot distortions | `[DERIVED]` ceiling `157,799 B`; required cut `33,253 B` | `[DERIVED]` target only | routing constraint, not a lever credit |

The XZ candidate is
`/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/rate_race/cpr1_xz_bt2_d12lc0lp1pb0_n192.zip`,
`191,044 B`, SHA-256
`dc302953ff1e7f6d09210fff80cf6981cd7a36fafbf4299770bff61abdc462bd`.
Its raw model and token parse-back are exact. `[DERIVED from that exact parse-back
plus the fresh governed source-receiver repeat]` its first-pair receiver output
must be byte-identical to CPR1. The retained historical candidate output agrees,
but its original producer and argv are `UNDETERMINED`. It has **not** been exactly evaluated; the
`0.17213597062027147` value is `[DERIVED from rounded CPR1 bot distortions plus
the exact new byte count]`, not a score row.

## The ordered path

Every long local training fire below is seeded and deterministic, runs only
through its governed MLX/Metal launcher, resumes from disk, and atomically
preserves distinct end-of-stage checkpoints plus periodic intra-stage state;
missing resume/config/EMA/optimizer state blocks launch rather than becoming an
exception inferred from this plan.

### 1. Close target and bounded receiver custody before a burn

**Run.** Recover PR130's pinned challenge commit and claim the #906 lane. Before
the short same-host AV/DALI Modal job may fire, wrap the source builder in an
owned `<=120`-pair chunk adapter with `--resume-from`, atomic per-chunk outputs,
and a monotone content-hash ledger; the original loop-end-only n600 invocation is
not launchable. Run AV then DALI in that adapter and require the new DALI content
to reconcile with the retained official target. Then land the owned receiver adapter plus its bounded
source-parity controls. Cure all current M1 findings, add the DALI target's
path/bytes/SHA and the PR130 init/source hashes to one content-identity manifest,
regenerate the memory/throughput probes against that exact cache, reset the
independent-review counter, and obtain three clean passes before training. The
target materialization and real n1 source control are already FIRED. The
remaining order is
**QUEUED-WITH-A-FIRE-ORDER:** pinned challenge custody -> #906 lane claim ->
resumable chunk-adapter receipt -> same-host #906 job -> DALI hash reconciliation
-> receiver-adapter controls -> M1 cures and three clean passes.
The full entropy pass stays on its separately stated fire order and must close
before Step 4's n600 archive promotion; it does not idle Step 2 after the
same-host authority gate closes.

The pinned job runs only the owned chunk adapter. It preserves the hash-pinned
source semantics at `build_gt_cache_official.py:33-59`, but instantiates the
official AV/DALI dataset over explicit `<=120`-name slices, writes each pose/seg
chunk atomically, and assembles the final cache only from a validated monotone
ledger. Resume starts at the first missing chunk. The original source CLI's
loop-end concatenation/save at `build_gt_cache_official.py:52-67` is never used
as the n600 execution path.

**Cost.** `[MEASURED]` target materialization used `117,981,301 B` on SSD.
`[UNDETERMINED]` adapter engineering and full CPU render wall time; the missing
input is a bounded same-runtime render probe.
`[MEASURED retained evidence, CPU model unspecified]` entropy decode alone is
`2,197.6 s`; `[DERIVED]` one raw output is
`3,662,409,600 B`. `[UNDETERMINED]` the short Modal provider charge until a
quote/preflight is recorded; all long local energy cost is also unpriced.

**Produces.** Same-host AV and DALI caches plus their segmentation/per-pose
diff, a content-bound DALI M1 ticket, clean seal, owned CPU adapter, and bounded
source-parity receipts. The full golden-token/resumable-render receipt and local
PR130 base raw object labeled `[macOS-CPU receiver research-signal]` remain
required before Step 4.

**Falsifier.** Any source/archive/cache hash drift, token hash mismatch, CPU-only
call failure, non-repeatable first-pair output, failed seal pass, same-host job
failure, or unreconciled DALI target identity blocks Step 2. Missing full-decode
resume closure blocks Step 4, not the bounded n120 burn or the scorer-free HPAC
race. Scope:
receiver/custody instance, not the PR130 family.

### 2. DALI-aligned PR130 renderer fine-tune on stratified-random n120

**Run.** From PR130's actual renderer initialization, train only against the
official DALI target on a recorded stratified-random `n=120` population—never a
prefix—through the existing exact render path. Preserve live, EMA, and terminal
tail states; checkpoint atomically every governed segment and at every stage.
Use the event controller and CPU-frozen-SegNet terminal selector only after the
M1 seal is clean. Preserve the pose-carrier and exact DALI-HPAC *payload bytes*
unchanged, but measure PoseNet on every complete rendered pair: changing the
semantic frame can change `d_pose` even when carrier bytes do not. Use the
official DALI cache's pose targets. The preserved-cache comparison measured a
large cross-platform/pipeline pose-target separation but did not isolate
same-host #906, so local AV pose targets cannot be substituted. This fire is
**QUEUED-WITH-A-FIRE-ORDER** after Step 1; the current AV ticket is forbidden.

**Cost.** `[MEASURED prior same-shape n120 receipt]` `29.3217719 s/step`.
`[PROJECTED from that receipt]` the `3,250`-step safety cap would take
`26.4710 h`, split into governed resumable segments of at most eight hours.
This is not a convergence forecast. The DALI-cache probe must remeasure memory
and throughput before fire; until then the actual cost is `UNDETERMINED`.
Training is local Metal/MLX only. `[ASSUMED external-provider charge]` is `$0`;
local energy cost is unpriced.

**Produces.** Same-population live/EMA/tail `d_seg`, full-pair `d_pose`,
per-class errors, boundary and official-DALI-vs-local-AV-like-disagreement-site facets, exact packed
renderer byte deltas, and complete resumable checkpoints. All scores remain
`[macOS-CPU frozen-scorer vs DALI-target research-signal]`,
`score_claim=false`; this is not the official macOS CPU evaluator axis because
that evaluator uses AV targets.

**Executable advance condition.** Use identical recorded pair IDs and official
DALI targets for initialization and candidate. Require two bit-identical n120
receiver renders of each candidate state, identical integer mismatch counts,
identical float64 pose-MSE reductions, and exact candidate archive hashes/bytes.
Compute
`C120 = 100*errors/(120*384*512) + sqrt(10*d_pose_120) + 25*archive_bytes/37,545,489`.
Advance only if candidate `C120` is strictly lower than initialization `C120`;
missing repeat identity or any missing term blocks rather than invoking an
unowned noise allowance.

**Falsifier.** Fold this *fine-tune formulation* if no content-bound terminal
state passes that strict deterministic advance condition with pose-carrier and
HPAC payload bytes fixed but full-pair `d_pose` remeasured.
An n120 negative cannot kill the renderer family; it routes to schedule/optimizer
re-derivation once, not an automatic longer burn.

### 3. Race causal edge context inside PR130 HPAC on the exact DALI object

**Run.** Freeze the DALI labels. Port/implement the edge-conditioned causal
context as an MLX/Metal HPAC context feature, counting every context tag and
model byte, then train locally and exact-decode all n600 labels. In the same
unit, implement the paired receiver-side context reconstruction in the owned
inflate adapter; a training-side decoder is not an admission surface. Rebuild
the entire CPR1 archive with semantic and carrier state fixed, jointly
XZ-compress the complete model bundle, and parse that exact candidate through
the owned receiver twice. Race its exact archive bytes against the current
lossless floor `191,044 B`. Report `token + separately packed HPAC model` versus
`132,144 B` only as a component diagnostic. This is a new PR130 measurement;
neither CR1's percentage nor #869's byte number is imported as a credit or
acceptance threshold. The table's CR1-scaled number is explicitly only a weak
cross-object sizing projection. This step is
**QUEUED-WITH-A-FIRE-ORDER** after Step 2's receipt. It does not wait for a full
renderer burn: the exact DALI labels are already frozen, and a lossless HPAC
change decodes to byte-identical labels independent of renderer weights.

**Cost.** `[PROJECTED timing basis only]` the killed local AV run reached epoch
54 at about `57,069 s`; linear continuation to epoch 60 would be `17.6139 h`
total. This is not a DALI/context runtime estimate. The actual cost is
`UNDETERMINED` until a deterministic one-epoch DALI/context timing receipt.
All long work stays local. `[ASSUMED external-provider charge]` is `$0`; local
energy cost is unpriced.

**Produces.** Full n600 token bytes, HPAC-model diagnostics, jointly compressed
model-bundle bytes, exact full-archive bytes/SHA, owned-receiver parse-back,
exact decode equality, deterministic repeat hashes, measured bounded full
decode runtime, per-class ideal/realized bits, and checkpointed learning curves.
The receiver runtime must fit the contest's 30-minute full-evaluation budget
after preserving the non-decoder allowance measured on the same hardware; an
unbounded training-only implementation is not a winner. The admission price is
the exact rebuilt archive, not an isolated package saving.

**Falsifier.** Fold that context formulation if the owned receiver cannot parse
and reconstruct the context, either repeated receiver hash differs, exact label
decode fails, decompressed semantic/carrier state drifts, its bounded full
decode makes the 30-minute full-evaluation budget infeasible on the tested
contest-CPU or contest-CUDA authority rail, or its best exact
rebuilt archive is not below `191,044 B`. Only a receiver-closed, repeatable,
runtime-valid full-archive lossless winner is adopted immediately because then
decoded labels, renderer input, rendered frames, and pose output remain
byte-identical. If it does not supply the entire `[DERIVED] 33,245 B` remaining
cut from the `191,044 B` lossless floor to `157,799 B`, retain the positive exact
archive credit and continue to Step 4.
Runtime failure folds only the tested authority rail; a macOS or retained
CPU-model-unspecified timing cannot fold the contest-CUDA formulation.

### 4. Promote the DALI-aligned renderer to a full n600 local candidate

**Run.** Only after Step 2's preregistered advance condition passes, run the same
sealed configuration over all n600 locally with per-stage and periodic
checkpoints, composing any Step 3 lossless HPAC winner. Before archive promotion,
close the full MATCH fire order and its golden token/resumable-render receipts.
Pack the selected live/EMA/tail winner into the exact CPR1 schema, re-run
lossless XZ selection, and render/evaluate in chunks no larger than 120 under a
claimed scorer lane. This step is **QUEUED-WITH-A-FIRE-ORDER** after the n120
receipt, Step 3 rate receipt, full MATCH closure, and fresh lane/liveness checks.

**Cost.** `UNDETERMINED` until Step 2 supplies the DALI-aligned measured
throughput and stopping trajectory; no n120-to-n600 linear wall-time guess is
admissible. Long training remains local. `[ASSUMED external-provider charge]`
is `$0`; local energy cost is unpriced.

**Produces.** An exact byte-closed PR130-format archive, n600
`[macOS-CPU frozen-scorer vs DALI-target research-signal]` component tuple,
per-class/boundary facets, receiver parse-back, archive SHA/bytes, and a measured
research-signal score-per-byte delta against the local DALI-target matched base.
This is not the official macOS CPU evaluator axis because that evaluator uses
AV targets.

**Falsifier.** Fold this trained instance if full-population realized
`100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489` is not lower than its exact
local PR130 base, or if any DALI-target, decode, archive, or checkpoint identity
fails. Carrier payload identity does not waive remeasurement of PoseNet on the
changed semantic frames. No contest claim follows from a local negative or
positive. If Step 2's renderer formulation is folded after its single
re-derivation, skip this burn and carry the original PR130 renderer plus the
Step 3 lossless rate state into Step 5.

### 5. Reconcile DALI pose after master changes, then allocate tensor precision

**Run.** If Step 4 changes semantic master frames, port the carrier continuation
to local MLX/Metal, bind it to the final master cache and official DALI pose
targets, and exact-pack a reconciled carrier before any precision allocation.
If Step 4 is folded, preserve the existing source-evidenced DALI carrier. On
that exact Step 3/4 state, measure one-tensor-at-a-time quantization curves for
renderer, carrier basis/coefficients, and HPAC parameters. Use exact pack-back
and stratified-random n120 frozen-scorer-vs-DALI-target receipts; fit the
KKT/reverse-waterfill allocator only after those PR130 curves exist. Jointly
replay every selected allocation because #336's old separable prediction had
the wrong sign. This is **QUEUED-WITH-A-FIRE-ORDER** after Step 4 is either
FIRED or FOLDED by its preregistered gate.

**Cost.** `UNDETERMINED`; the missing inputs are the number of non-dominated
PR130 tensors and measured per-rung scorer/runtime cost. The first receipt must
price those before any wide sweep. `[ASSUMED external-provider charge]` is `$0`
and local energy cost is unpriced; scorer work requires a fresh lane claim.

**Produces.** When required, a DALI-targeted carrier-reconciliation receipt and
exact full-archive delta; then, for every admitted tensor rung, exact byte delta,
`d_seg` delta, `d_pose` delta, full nonlinear research-signal delta,
confidence/noise receipt, and parse-back survival. This is the first legitimate
input to #157-style allocation on PR130. Scorer components are
`[macOS-CPU frozen-scorer vs DALI-target research-signal]`, not the official CPU
evaluator axis.

**Falsifier.** Fold the carrier continuation formulation if its exact joint
research-signal delta is non-negative; preserve the incumbent DALI carrier.
Fold any precision rung whose measured joint delta is non-negative; fold the
separable allocator formulation if its composed prediction has the wrong sign
or exceeds the calibrated error bound. Do not kill quantization as a family from
a single tensor or n120 instance.

### 6. Buy task-lossy semantic rate only if the lossless stack remains above 0.15

**Run.** Mutate exact DALI labels only through a shallow, causal tolerance
schedule implemented/trained on local MLX/Metal; retrain HPAC, renderer, and any
affected pose state as one dependent object. Use the old #869 result as a
required negative control, not as a byte credit. Start with stratified-random
n120 and advance to n600 only on measured joint research-signal improvement.
This step is **QUEUED-WITH-A-FIRE-ORDER** conditional on the
`[PROJECTED research-signal]` exact-archive composite after Steps 3–5 remaining
at or above `0.15`.

**Cost.** `UNDETERMINED` until the lossless stack defines the required remaining
byte-equivalent gap and Step 2 defines local convergence cost. `[ASSUMED
external-provider charge]` is `$0` and local energy cost is unpriced; training
stays local.

**Produces.** A byte-closed candidate with exact mutated-label ownership,
retrained prior/renderer, pose confirmation, and a full
`[macOS-CPU frozen-scorer vs DALI-target research-signal]` three-term composite;
it is not an exact authority score receipt.

**Falsifier.** Fold a tolerance formulation whenever the exact joint
`100*d_seg + sqrt(10*d_pose) + rate` delta is non-negative, or if label edits
cannot be reconstructed solely from counted payload plus generic code. Any
hidden target table is forbidden.

### 7. Exact authority replay and pointer decision

**Run.** When one exact archive has a DALI-target research-signal composite with
enough margin to justify authority replay, claim the scorer/dispatch lane,
verify archive/runtime/evaluator custody, and run the same bytes on contest CPU
and contest CUDA as separate axes. The research signal is not called same-axis
with either evaluator. CUDA/Modal is used only for the short physical authority
replay, not training. This is **QUEUED-WITH-A-FIRE-ORDER** after byte closure and
a fresh lane claim.

**Cost.** `UNDETERMINED` until the final decoder runtime and provider quote are
measured; no dollar estimate is invented.

**Produces.** Exact full-precision components, archive bytes/SHA, hardware and
runtime provenance, CPU/CUDA axis labels, and the only admissible pointer
decision.

**Falsifier.** Any runtime-budget failure, archive drift, evaluator mismatch,
CPU/CUDA nondeterministic token decode, or exact score at/above the competitive
threshold prevents promotion. No local or mixed-object row substitutes for it.

## Follow-on disposition audit

| named follow-on | disposition and fire order |
|---|---|
| retained official DALI cache materialization | FIRED; proves shipped target identity but not same-host #906 isolation |
| same-host AV/DALI #906 Modal job | QUEUED-WITH-A-FIRE-ORDER after pinned-commit recovery, #906 lane claim, and an owned hash-pinned `<=120`-pair `--resume-from` chunk adapter with atomic outputs/content ledger; the original loop-end-only CLI is not launchable; gates Step 2 |
| real source receiver control | FIRED at n1 as an explicit TOY-BRACKET |
| full CPU MATCH | QUEUED-WITH-A-FIRE-ORDER: owned adapter -> bounded controls -> narrow operator exception request -> golden n600 decode -> resumable render; denial blocks |
| configured two-grid single-LZMA2 tuning on current CPR1 | FIRED and FOLDED at FORMULATION scope after `[MEASURED]` 3,375 valid roundtrips; exact `[MEASURED] -8 B` candidate retained; broader lossless/filter-chain family remains open |
| XZ rerace after renderer weights change | QUEUED-WITH-A-FIRE-ORDER inside Step 4 after exact pack-back |
| M1 DALI repin and three-pass seal | QUEUED-WITH-A-FIRE-ORDER after target identity manifest and fresh probes |
| n120 DALI renderer burn | QUEUED-WITH-A-FIRE-ORDER after same-host #906 closure, M1 seal, and live-job checks |
| one schedule/optimizer re-derivation after an n120 formulation negative | QUEUED-WITH-A-FIRE-ORDER conditional on that negative receipt; no automatic longer burn |
| causal HPAC context race | QUEUED-WITH-A-FIRE-ORDER after the n120 receipt; decoded-label identity makes it independent of the full renderer burn |
| n600 DALI renderer promotion | QUEUED-WITH-A-FIRE-ORDER after the n120 advance receipt, HPAC rate receipt, full MATCH closure, and scorer-lane claim |
| DALI pose reconciliation plus PR130 tensor precision curves/allocation | QUEUED-WITH-A-FIRE-ORDER after the renderer step is FIRED or FOLDED and the lossless HPAC race is closed |
| task-lossy exact-label tolerance | QUEUED-WITH-A-FIRE-ORDER only if the `[PROJECTED research-signal]` exact-archive composite after the measured lossless stack remains at or above `0.15` |
| paired exact authority replay | QUEUED-WITH-A-FIRE-ORDER after byte closure, evaluator custody, and a fresh lane claim |
| VEH-primary and VEH/GT-hybrid bases | FOLDED |

## Why the other levers lost priority

| option | evidence and scope | disposition |
|---|---|---|
| more tq1c/VEH menu work | `[MEASURED macOS-CPU advisory]` plateau `d_seg=0.004305420`; wrong-label target | FOLDED as primary vehicle; retain low-priority rate datum only |
| direct #869 adaptive waterfill | `[MEASURED] -113,555 B` only on IX2; later old-vehicle scorer harm | FOLDED as a transferred price; task-lossy PR130 remeasurement only in Step 6 |
| #933 literal ±1 token range | `[MEASURED live IX2 sibling]` L14 saves `24,605 B`; separately, `[MEASURED live IX2]` endpoint clamp mass is `33.296%`; neither has a PR130 label analogue or scorer price | FOLDED as a direct lever |
| cell-drop waterfill knee | `[DERIVED from MEASURED retired-object local d_seg and exact byte count] -0.0983195` seg-plus-rate versus its parent; pose and exact evaluator were not measured, contrary to the charter seed | FOLDED as a PR130 transfer; historical calibration only |
| CR1 edge support | `[MEASURED byte-only n600] 575,095 -> 464,557 B`, `-110,538 B` or `-19.221%`, but on a selected-support object larger than CPR1 | FOLDED as a byte credit; causal-context mechanism alone enters Step 3 |
| SMEVR | `[MEASURED byte-only IX2] +5,183 B` versus shipped IX2 bulk | FOLDED on bulk IX2; no PR130 promotion |
| TROT/TR2P1 | `[MEASURED selected-support object] +2,172,998 B` versus CR1 | FOLDED for that formulation |
| #311 TropNNC | `[MEASURED old witness]` dense-trunk exact-preservation result is another architecture | FOLDED as a direct PR130 transfer |
| #336/#157 allocation | `[MEASURED old witness] -26,689 B` with `+8.7205 S` joint regression | FOLDED as a transferred allocation; allocator waits for Step 5 curves |
| #140 low-rank pose | `[MEASURED old vehicle]` corrected result is about `-525 B`/`-0.00035 S_rate`, not 2.7x, on another carrier | FOLDED as a direct PR130 lever |
| #850 “add relins” | `[MEASURED source/receipt]` initializer cap is `0/600` binding; different solver caps on about 36% of a hardest-60 subset | FOLDED as a general plan |
| direct blind-coordinate reclaim | `[MEASURED CPR1] 0 B` because no camera-resolution payload exists | FOLDED |
| od9 cheap pose carriage | `[MEASURED n32 byte-only]` the stage-2-only packet is `2,157 B`; `[PROJECTED linear n600] 40,444 B`; the combined flat ship-the-solve packet projects to `1,252,219 B` | FOLDED at FORMULATION scope for flat sparse/delta-entropy ship-the-solve; shared/task-description carriage remains outside this negative |
| #366 joint/in-loop descent family | FAMILY-OPEN; `[MEASURED CPR1] d_pose=0.00002331`, and the 23,054-B carrier already uses delta/Rice `k=8–9` | QUEUED-WITH-A-FIRE-ORDER only after a same-object mutation creates measured pose debt that Step 5 cannot reconcile; not folded as a family and not ahead of semantic alignment/rate |
| generic pose-first redesign | `[MEASURED CPR1] d_pose=0.00002331` | FOLDED ahead of semantic alignment and rate; preserve pose until a mutation forces refit |
| continuing the configured two-grid single-LZMA2 search | `[MEASURED exact bundle]` only `8 B` after 3,375 valid roundtrips | FOLDED at FORMULATION scope on this exact raw bundle; broader lossless/filter-chain family remains open, and same-form rerace is QUEUED-WITH-A-FIRE-ORDER after state changes inside Step 4 |

PR86 to PR130 is the warning against rate tunnel vision: `[MEASURED]` token
bytes worsened by `3,080 B`, while renderer and pose improvements produced the
large score win (`.omx/research/pr86_pr130_fullstack_intake_20260728.md:229-249`).
The ordered path therefore preserves exact target realization and pose while it
buys rate.

## RECALL EVIDENCE

I ran these positional full-corpus recall queries with
`.venv/bin/python tools/recall_fused.py "<query>" --top N`:

1. `PR130 CPR1 semantic pose HPAC GT VEH tq1c d_seg plateau 0.004305420 #978`
2. `PR130 semantic renderer realization GT labels VEH labels CAP n32 d_seg 0.0010862 plateau full vehicle #978`
3. `PR130 HPAC GT lstars ep54 GT primary VEH primary hybrid realization break even d_seg`
4. `PR130 rate levers adaptive per-cell #869 CR1 edge conditioned SMEVR #940 same object`
5. `PR130 renderer weights TropNNC sensitivity bit allocation KKT reverse waterfill low rank pose carrier #140`
6. `PR130 CPU CUDA drift CPR1 T4 Ada 0.172141 0.0002966 0.00002331 #906`

I also searched the canonical equations registry via
`tools/list_canonical_equations.py --json`; the canonical research index and
sub-0.15 DAG/FEED surfaces; the task ledger and full hot state; direct PR130
source, scripts, evidence, and retained archives; M1 reviews/tickets; and the
receipt corpus by content. Relevant equations included
`master_gradient_locality_violation_by_codec_v1`,
`witness_measured_reverse_waterfill_v1`,
`ddm_hb1_semantic_label_incumbent_transfer_v1`,
`score_marginal_lagrange_multipliers_v1`, and
`trajectory_derived_stopping_law_v1`.

Findings beyond the charter seeds changed the plan materially:

- `[MEASURED hot state]` The common contract's copied qo1 pointer is stale; the
  full hot state names tq1c
  `S=0.7534578126155775 @ 357,837 B [macOS-CPU advisory]` as the current
  own-vehicle pointer, so all pointer honesty uses the hot-state row.
- The OP1R addendum and direct source refuted the “approximate PR130 semantic
  field” premise: shipped tokens are exact GT.
- The source/CPR1 metric mix split `0.172141` measured from `0.169848`
  projected and prevented a fake baseline.
- The preserved official cache enabled a full n600 scorer-free
  official-DALI-versus-local-AV-like content comparison and exposed the current
  M1 cache as the wrong shipped target, but cross-platform/pipeline provenance
  means it did not close same-host #906. The Modal job remains queued and gates
  the burn.
- The pose-target comparison found target-to-target MSE `0.0001400434`; direct
  carrier receipts then showed the incumbent carrier already targets official
  DALI, so the result forbids AV substitution rather than reopening pose-first
  design.
- The official cache's token hash equals the decoder golden, closing target
  identity without inferring from chroma-siting sensitivity.
- The real source first-pair run proved local mechanism viability, while the
  RangeDecoder's missing persistent checkpoints blocked a fake “resumable full
  MATCH.”
- `[MEASURED review/cache]` The pinned M1 review has fifteen findings and the
  exact M1 target cache is AV-like; both must be cured before the DALI burn.
- The local evaluator checkout does not contain PR130's pinned challenge commit,
  so exact authority custody remains queued rather than inferred from matching
  model/video hashes.
- The exact CPR1 XZ race found only `8 B`, so generic repacking lost priority.
- CR1, SMEVR, TROT, #336, #140, #850, and direct blind-coordinate receipts
  removed their inherited headline numbers from the PR130 ledger; only
  mechanisms with a new same-object measurement remain in the path.

## Boundaries and pointer honesty

No full n600 scorer job, Metal training, paid dispatch, exact evaluator run, or
full CPU receiver MATCH occurred. No score moved. The preserved-cache rows are
cross-platform/pipeline target-content comparisons and do not close same-host
#906. The first-pair receiver rows are TOY-BRACKET mechanism controls.
The XZ row is exact byte-only state preservation plus a derived rate delta.
PR130 remains borrowed external evidence; the current own-vehicle pointer remains
tq1c. `[MEASURED Git custody]` strict authority replay also requires recovery of
PR130's pinned challenge evaluator commit `d3f688f...`; the current local
`upstream/` checkout is `11ad728f...` and is not a substitute.

NEXT_IF_RESUMED: recover pinned challenge commit `d3f688f...`, claim the #906 lane, and land the owned `<=120`-pair `--resume-from` cache-builder adapter with atomic per-chunk outputs and a monotone content-hash ledger. Only then fire the short same-host AV/DALI #906 job and reconcile its DALI tensor content to seg-raw SHA `c5c7671d...` plus pose-raw SHA `23ae28d2...`; retain file SHA `382d7dfe...` only as custody identity for the preserved differently named container. Then land the owned receiver adapter's bounded source-parity controls, cure/seal M1, and fire the stratified-random n120 DALI-target research burn—never the current AV-target ticket.

Own-vehicle frontier: S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]
