# ddm_mxo1 — free decode-time online context mixing on the shipped move-44 stream

**Status:** COMPLETE, scorer-free, no candidate fired.  The best of three original
post-shipping-prior learners saved **368 realized RC64 bytes** on the full n600
stream.  That is **1.42%** of the 25,899 B rate demand, below the charter's
3,000 B receiver-delta fire threshold and also slightly below ls2's 386 B linear
screen.  No receiver tree, candidate archive, scorer, Modal job, or upstream file
was touched.

**Axis:** `[macOS-CPU advisory / scorer-free n600 exact RC64 byte measurement]`.
The byte counts are physical RC64 stream sizes.  The runtime projections are not
T4 measurements of these new learners.  This work did not measure distortion or
produce a new exact score.

| Full-n600 online replay | RC64 bytes | realized delta vs 119,749 B | demand fraction | incremental host wall, two cold runs | projected T4 incremental wall |
|---|---:|---:|---:|---:|---:|
| shipped control | 119,749 | 0 B | 0% | 3.448 / 4.149 s encoder control | not a new receiver |
| hierarchical Lane stack | **119,381** | **−368 B** | **1.4209%** | 37.544 / 41.149 s | 36.721 / 40.247 s |
| 25-state group recurrence | 119,784 | **+35 B** | −0.1351% | 25.923 / 32.096 s | 25.355 / 31.392 s |
| previous-frame 3x3 hash | 119,635 | **−114 B** | 0.4402% | 16.122 / 20.239 s | 15.768 / 19.795 s |

The two independent stream hashes match for all four rows.  The control is also
byte-identical to the shipped stream, sha256
`c499972a33dac497412c18839b632a8e1bbf75d518528039db0e2aca80c8eb13`.

## 1. The wall-clock bound replaces the charter's 540-second premise

The live move-44 `t4_direct` receipt measures three different clocks that must not
be conflated:

- token decoder: **1,142.996295691 s**;
- decode and render inside the receiver: **1,223.867718011 s**;
- inflate wrapper: **1,232.418725255 s**.

The live receiver policy ceiling is **1,260 s**, so the admissible incremental
receiver budget is

`1260 - 1232.418725255 = 27.581274745 s`.

At `600 * 384 * 512 = 117,964,800` coded symbols, that is
**233.809363 ns/symbol**.  The incumbent token decoder itself uses
**9,689.300 ns/symbol**; 233.809 ns/symbol is only the incremental allowance.

The raw 1,800 s inflate timeout leaves 567.581 s after the present inflate, and
the measured 1,295.554 s inflate-plus-evaluate path leaves 504.446 s before the
outer job wall.  Neither is permission to spend 500+ seconds in a receiver
change: the first ignores evaluation and runner overhead, and both violate the
live 1,260 s decode policy.  Therefore the charter's approximately 540 s of
receiver slack is stale.  This prototype uses **27.581 s / 233.809 ns per symbol**
as the binding budget.

The same receiver took 1,168.622529 s locally and 1,142.996296 s on T4, giving a
same-receiver `T4/local = 0.978071` projection factor.  Applying it to new Python
model time is only a projection; it is not a compiled-native T4 benchmark.

Timing evidence:

- `.omx/research/ddm_rlc5_20260910/SEAL_ddm_rlc2_counted_cure_move43_rlc5_contest_cuda_v3.json.decode_wall_clock.json`, sha256 `c708f817cacfa6f180a85560dfb77d01b5bb10e784d8eb6e816a6dbb5c17b191`;
- `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run5/MODAL_REMOTE_RESULT.json`, move-44 retained T4 receipt;
- `/Volumes/VertigoDataTier/pact/ddm_ls1/TRACE.json`, local exact shipped-receiver trace.

## 2. Originality boundary and prior art

The charter says nobody had built an online learner of this kind.  Full-corpus
recall and inspection of the shipping receiver falsify that premise.  The move-44
`NativeFreeCorrector` already compiles **23 named adaptive families**, contextual
weight sets, and online learning; its Python reference inherits the PAQ-style
fixed-point logistic mixer and updates from decoded groups.  Rebuilding that
mechanism under a new name would not be original.

MXO1 therefore tests only mechanisms not already present in the vehicle:

1. a hierarchical residual stacker over the **final** shipped probability row,
   with receiver-visible Lane contexts and online-learned post-mix weights;
2. a tiny leaky group-recurrent calibration state after the shipped prior;
3. a richer previous-frame 3x3 hashed expert after the shipped prior.

This also avoids the `ddm_lm1` failure mode: its independently written six-model
online logistic **replacement** expanded the old stream to 303,233.7 B.  MXO1 is
nested augmentation, so a cold/zero Lane bank is bit-identical to the incumbent.

## RECALL EVIDENCE

The charter seeds were treated as a floor.  The following content searches were
run over the full research corpus, equations registry, index/DAG, design docs,
and state/ledger surfaces:

- `rg -il 'online.{0,40}(context|mixer)|context.mix|logistic.mixer|adaptive.{0,30}correct' .omx/research` — **498 files**;
- `rg -il 'previous.frame|temporal.{0,30}context|bidirectional.pyramid' .omx/research` — **520 files**;
- `rg -n 'ddm_rr1|ddm_lm1|ddm_fx1|ddm_fx2|ddm_bd1|ddm_ls1|ddm_ls2|ddm_tc1|ddm_tc4' .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` — **5 indexed/DAG hits**;
- `rg -il 'online.{0,30}(context|mixer)|logistic.mixer|previous.frame' docs .omx/state` — **4 files**;
- `.venv/bin/python tools/list_canonical_equations.py --json`, then case-insensitive term census — `context=491`, `mixer=43`, `online=17`, `temporal=123`, `previous frame=0` occurrences.

Beyond the charter's named seeds, recall found:

- `ddm_rr1` measured an older-field, free, online temporal count augmentation at
  **1,598.30 B**, but richer count contexts saturated and the formulation was
  subsequently absorbed into the current corrector lineage;
- `ddm_lm1` measured the catastrophic 303,233.7 B adaptive-logistic replacement,
  closing replacement but not nested augmentation;
- the current shipping `fx1`/`fx2`/`free_corrector`/native chain already performs
  online PAQ-style mixing over 23 families, directly changing the novelty claim;
- `ddm_bd1` had already bounded its temporal-pyramid formulation at 2.8–5.7%, so
  the previous-frame learner must beat that band or close;
- `ddm_tc4` showed small fixed causal-map interactions and then a T4 timeout,
  reinforcing the need to price compute before a receiver patch.

These findings changed the plan from “build the missing online mixer” to “measure
three post-final-prior augmentations that cannot duplicate the shipped mixer.”
They also caused the recurrent arm to be kept tiny and the previous-frame arm to
carry an explicit bd1 comparison.

Key beyond-seed memos:

- `.omx/research/ddm_rr1_free_decode_model_and_rate_rung_close_20260817.md`, sha256 `538bebd48097c863077506717d006e94011113f22d7a9060ad326ded875f3c6c`;
- `.omx/research/ddm_lm1_learned_model_falsifier_20260826.md`, sha256 `fd53719a393c612210f4ded6ec2403f102e7935248b39a54ee70a313bf1966e5`;
- `.omx/research/ddm_bd1_bidirectional_pyramid_context_20260905.md`, sha256 `f5f27bc63497660077c128e83d568a5c4c37008ecfca33beb6854fd2bcbb8e27`;
- `.omx/research/ddm_tc4_causal_context_map_slate_20260910.md`, sha256 `aedac78fe92ea72705da5f19f08818b6eaf2a35420adde198cf3091d5ce4f4be`.

## 3. Three exact online learners

All learners consume LS1's exact final receiver frequency rows and symbols in
the receiver's 190-group order.  Each starts cold.  No fitted initialization,
video-derived weights, map, or selector is stored.  Counts and weights change
only after the symbols visible to a real decoder have been decoded.  The
encoder-bearing RC64 source is the pinned source used by the shipping re-encode
lineage.

### A. Hierarchical Lane residual stacker

**Mechanism.**  For each incumbent argmax class, three KT calibration experts
use previous-frame Lane distance (8 bins), receiver-visible preceding-row Lane
distance (8 bins), and their joint cell (64 bins).  The experts correct the final
shipped row in log space.  Fifteen Q5 weights are updated once per completed
frame by an online AdaGrad rule.  The mutable tables are about 32 KiB; no state
is transmitted.

**OSS anchor.**  PAQ/LPAQ contextual weight selection and cmix log-domain online
mixing: [Matt Mahoney's open-source PAQ/LPAQ archive](https://www.mattmahoney.net/dc/)
and [Byron Knoll's cmix](https://github.com/byronknoll/cmix).

**Measured.**  119,381 B, **368 B saved**.  That is 1.4209% of demand, 4.0839%
of tc1's 9,011 B cited oracle, 4.4779% of LS1's current 8,218.071 B MM oracle,
and 3.1810% of LS1's 11,568.821 B plug-in oracle.  It also fails to exceed ls2's
385.55 B linear result.

**Compute.**  Incremental host cost was 37.544/41.149 s,
318.264–348.824 ns/symbol.  Same-receiver T4 projection is 36.721–40.247 s,
which exceeds the 27.581 s allowance by 9.139–12.665 s.

**Receiver diff needed, not produced.**  Add a post-`LaneMixer.coding` stage in
`runtime/residual_archive.py`, a native 80-cell-by-class count bank beside
`f26_corrector_native.c`, and update it only in `end_frame`.  Replace prototype
float AdaGrad accumulators with fixed Q24 gradient sums plus integer square root,
then require Python/C frequency and full-stream twins.  The 368 B result and
runtime failure do not authorize this diff.

**Verdict:** `CLOSED(FORMULATION)` for this three-expert hierarchical stacker on
move 44.  This is not a family-wide kill of every nonlinear online learner.

### B. Twenty-five-state leaky group recurrence

**Mechanism.**  A `5 x 5` observed/expected state is indexed by the incumbent
argmax class and next class.  After each of 190 groups it decays by exactly
`63/64` in integer Q31 mass and ingests that decoded group's class residual.
The next group receives a square-root-damped calibration.  Mutable state is
400 B and is derived solely from prior decoded groups.

**OSS anchor.**  DeepZip couples recurrent next-symbol probability prediction to
arithmetic coding; TRACE supplies the relevant warning and alternative that
sequential history construction dominates latency:
[DeepZip](https://arxiv.org/abs/1811.08162) and
[TRACE source](https://github.com/mynotwo/A-Fast-Transformer-based-General-Purpose-LosslessCompressor).

**Measured.**  119,784 B, **35 B worse**.  Ideal and physical deltas agree to
within one byte of coder framing.

**Compute.**  Incremental host cost was 25.923/32.096 s,
219.753–272.079 ns/symbol.  The projected T4 range is 25.355–31.392 s, which
straddles the 27.581 s budget rather than clearing it robustly.

**Receiver diff needed, not produced.**  Add a 25-cell state to the existing
group callback, compute the correction before decode, update after decode, and
expose it through the existing native corrector ABI.  The probability-table
step still needs a fixed-point C twin.  Negative rate makes that work dominated.

**Verdict:** `CLOSED(INSTANCE)` for the 25-state, 63/64-decay, square-root-damped
recurrence.  It is not evidence against all recurrent predictors.

### C. Previous-frame 3x3 hashed context

**Mechanism.**  The fully decoded previous frame is padded with a sentinel and
its 3x3 class patch is folded deterministically into 4,096 bins.  A table keyed
by incumbent argmax class and that hash supplies a half-strength KT correction.
The table is updated only after the complete current frame, so causality is
strict.  Counts plus expectations occupy about 1.56 MiB and cost zero archive
bytes because they are rebuilt from the prefix.

**OSS anchor.**  TRACE's online, no-pretraining probability model provides the
adaptive arithmetic-coding pattern, while previous-decoded-frame conditioning
is the standard causal temporal-context construction.  This arm deliberately
uses a bounded hash rather than shipping motion or a learned table.

**Measured.**  119,635 B, **114 B saved**, only 0.09520% of the shipped token
stream.  That is far below bd1's 2.8–5.7% temporal band, so the charter's explicit
bd1 gate fails.

**Compute.**  Incremental host cost was 16.122/20.239 s,
136.668–171.569 ns/symbol.  The projected T4 range is 15.768–19.795 s and fits
the 27.581 s allowance, but 114 B cannot pay for a receiver change.

**Receiver diff needed, not produced.**  At frame start, hash the previous plane
once; add a 102,400-cell observed/expected table and a single five-class
fixed-point correction in the group path; update at `end_frame`.  A native
frequency twin and first-measurement custody would remain mandatory.

**Verdict:** `CLOSED(FORMULATION)` for the 12-bit 3x3 hash with half-strength KT
calibration on move 44.  It does not close motion-aligned or learned temporal
contexts, which were not measured here.

## 4. Byte closure, determinism, and retained custody

The executed instrument is `experiments/ddm_mxo1_online_context_mixing.py`.
Primary result sha256 is
`c68f52b74b131f050bb20c340af25382ce4f21c5f32fcd91e42c033f5a4f1f78`;
repeat result sha256 is
`2ea457683e3b157f5472b461d8f36b1cf23467d4d8ff145b17c446ad118eb1a7`.

The full retained root is
`/Volumes/VertigoDataTier/pact/ddm_mxo1_free_decode_time_online_context_mixing/`.
It holds 141,391,734 B of streams, complete learner states, RC64 encoder
checkpoints, build source/library, and receipts at retention-manifest sha256
`94c8f4e1b6f21c5b40f4bbe9cbd9f04adc8f8b475fe242e8c8edb32fae368729`.
No cleanup or deletion is authorized.

Controls:

1. **Exact shipped control:** 119,749 B and exact shipped sha256, not merely the
   same ideal code length.
2. **Independent cold repeat:** control and all three learner streams match in
   byte length and sha256; `DETERMINISM.json` sha256
   `6f8d17bf2e4fa0973eb39ade1b1e3e20250a311647dd260f84a679024c5ac8e4`.
3. **Causality:** Lane/patch state updates only at a complete frame; recurrence
   updates after each decoded group; encoder order is the shipping 190-group
   order.
4. **Resumability:** every 25 frames preserves a full model NPZ and all four RC64
   interval/buffer snapshots.  A terminal resume reload passed without recoding.
5. **Source custody:** the exact executed v2 source is retained at sha256
   `5d5a388a0720dd2e216652ea946c3e8a97fda7c7ed03b2eefc47629130d3ebe5`.
   The repository source differs only by lint cleanups and terminal-result
   idempotence; final bound source sha256 is
   `2341f9f8499d3b47758fd938d4289ca65e64da3b1f1706944e5365d065ce780c`.
6. **Instrument falsifier retained:** the first attempt encoded raster order,
   producing the same control length but a different sha.  It was rejected,
   preserved as `V1_INSTRUMENT_FALSIFIED.json`, and superseded by group-order v2.

The stream repeat proves same-host determinism.  It does **not** prove cross-host
frequency identity for the prototype's NumPy floating ratio table or the Lane
AdaGrad accumulator.  Any receiver adoption must replace those update surfaces
with fixed-point/integer arithmetic and pass Python/C full-stream twins.  This
unmet cross-host control is another candidate blocker, not hidden by the small
byte win.

## 5. Prototype decision and follow-on disposition

The best row saves 368 B, short of the 25,899 B demand by **25,531 B** and short
of the 3,000 B receiver-delta fire bar by **2,632 B**.  It neither crosses the
rate target nor justifies a receiver integration.  The candidate-input branch is
therefore **NOT FIRED**.  There is no archive delta and no score projection
presented as a result.

The canonical own-vehicle frontier remains composition S 0.1372449041713402 at
180,406 B `[contest-CUDA T4 n600]` (move 44), **UNCHANGED**.

The next possible charter is `ddm_mxo2_full_family_residual_stacker`, disposition
`QUEUED-WITH-A-FIRE-ORDER`, owner `MAIN`, consumer store
`/Volumes/VertigoDataTier/pact/ddm_mxo1_free_decode_time_online_context_mixing/`.
It fires only if either (a) a native dry-run can expose the shipping corrector's
23 pre-mix family predictions and prove a conservative incremental cost at or
below 200 ns/symbol, **and** an exact n600 integer-row screen predicts at least
3,000 B physical savings, or (b) a new exact T4 receipt expands strict receiver
slack to at least 60 s.  Until a trigger occurs, do not rerun the three MXO1
forms or build a receiver delta.

## NEXT_IF_RESUMED

- `ddm_mxo2_full_family_residual_stacker`; disposition=`QUEUED-WITH-A-FIRE-ORDER`;
  owner=`MAIN`; consumer store=`/Volumes/VertigoDataTier/pact/ddm_mxo1_free_decode_time_online_context_mixing/`;
  fire trigger=`shipping 23-family pre-mix outputs have an exact n600 integer-row screen at or above 3,000 B plus a native dry-run at or below 200 ns/symbol, or a new exact T4 receipt expands strict receiver slack to at least 60 s`.

## LIVE-HYPOTHESES

- A low-rank nonlinear stacker over the 23 **pre-mix** family outputs could
  recover more than the post-final-prior Lane stack because the final probability
  has already collapsed disagreement information.  LS1's 8,218 B MM gap makes
  this plausible, but no ≥3,000 B achievable lower bound exists yet.
- Motion-aligned previous-frame contexts may outperform the static 3x3 hash.
  The present negative only covers a collision-heavy, fixed-location hash; it
  does not test causal motion compensation.  It is plausible only if a generic
  native warp stays inside the 233.809 ns/symbol budget.

## DEAD-ENDS

- Rebuilding a PAQ/cmix-style online learner as if move 44 lacked one is dead:
  the shipped native corrector already learns contextual weights over 23 families.
- The three-expert Lane residual stack is closed for this formulation: 368 B is
  below ls2, 1.42% of demand, and its projected wall exceeds strict slack.
- The 25-state leaky recurrence instance is closed: it adds 35 B and its repeat
  timing does not robustly clear the wall.
- The static previous-frame 3x3 hash formulation is closed: 114 B is only
  0.0952% of stream bytes and fails bd1's 2.8–5.7% gate.
- Raster-order replay is dead as an instrument: it preserves length but not the
  shipped RC64 bytes; only 190-group order reproduced the control sha.
