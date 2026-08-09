# RR2: PR130 semantic-leg audit

**Date:** 2026-08-09

**Lane:** `ddm_rr2`

**Audit axis:** static source, checkpoint-metadata, byte, and prior-receipt audit

**No new score claim:** no trainer, SegNet, PoseNet, archive evaluator, CUDA, Metal, or MPS job was run

## Verdict

The PR130 semantic leg is mechanically coherent from QAT forward to shipped int4 receiver. The trainer
fake-quantizes every named parameter with the same fp16 scales, axes, and signed `[-7, 7]` code range
used by the packer and inflater; the generated-frame path also applies the same bilinear camera expansion,
uint8 rounding, and scorer resize as the compressed side of `upstream/evaluate.py`. SG2 already proved
that the stage-08 packed semantic blob is exactly the blob in the 191,052-byte archive. I did not find a
second quantizer, a hidden float renderer, or a post-render semantic closer in the bounded PR130 source
and receiver scope.

Three defects remain, ranked by consequence:

1. **P0 / HIGH — the retained PyTorch QAT trainer is not crash-resumable and does not preserve
   resume-complete periodic or stage state.** It keeps the selected best state only in RAM, then performs
   one non-atomic terminal `torch.save`. The two historical invocations do leave distinct terminal model
   artifacts, but neither contains the state needed to resume. A failure can lose all 12,000 stage-07
   steps or all 6,000 stage-08 steps. It also has no EMA state or EMA save path. This is an operational
   and custody defect in this trainer instance, not evidence against the semantic-renderer family.
2. **P1 / MEDIUM — the final checkpoint's top-level `config` is six transformations stale.** Current
   in-tree consumers use only architecture fields, so I did not find current runtime schedule corruption
   among 19 direct checkpoint-config read sites. Five of those 19 sites nevertheless propagate the stale
   dictionary into another checkpoint, and any provenance reader can falsely report stage-02 schedule,
   precision, paths, and seed as stage-08 facts. The live stage schedule survives only in
   `checkpoint["result"]["config"]`.
3. **P1 / MEDIUM — the throughput ledger's FLOP denominator covers the four `TokenBlock`s, not the full
   renderer.** The independent full-renderer lower bound is 20.6486 GFLOP per forward image rather than
   15.8545. Under the ledger's same `backward ~= 2 x forward` convention, batch two is 123.8914 GFLOP;
   the measured 120.5976 ms is therefore 1,027.3 GFLOP/s, or 7.10% of the measured 14,471.765 GFLOP/s
   dense ceiling. The measured 75.40% renderer share and bandwidth-bound diagnosis survive; only the
   quoted 5.5% and its scope need correction.

The semantic rate fact remains 36,580 marginal archive bytes, 19.15% of the 191,052-byte archive, and
0.0243571 score units in the existing local anatomy ledger. This audit does not remeasure that jointly
compressed marginal.

## Authority and object identity

| Object | Evidence | Status |
|---|---|---|
| Intake repository | `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo` at `e34f31bc4969042c0051ac81aa3c56884419a231` | source-verified, read-only |
| Published archive | 191,052 B, SHA-256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd` | prior measured bytes; not rebuilt here |
| Stage-07 checkpoint | `semantic_renderer_w96_b4_qat4_12k.pt`, 283,432 B, SHA-256 `1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf` | static byte census / SG2 receipt |
| Stage-08 checkpoint | `semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt`, 282,352 B, SHA-256 `3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647` | static byte census / SG2 receipt |
| Stage-08 packed semantic blob | SHA-256 `9b98360bd56918b5a414ace375c29790b7fe9f7f55cf423c0564ef4e62a39b99`; equals archive blob | SG2 measured-byte result, not remeasured here |
| Published base score | `S = 0.172141297491896447` `[contest-CUDA, DALI GT, n600]` | prior authority row; not reproduced here |
| Source hashes | QAT trainer `4bcaf8a5c581c1e5eb057ea0ef760f269e1eabfdd2fca926bcbf61f4163a248d`; renderer `2bf3a6a8621334723fec1c3e596665d1f049a55f311c5348dd5a4c588873f25b`; float trainer `2d7a3575e422dc2b5823b97e52101ad5632e5fa04a98e0af8a83d85b7c2176b8`; quant evaluator `5bbd2136174bfa2c99219d73f45103d4293f60e3f4eced5ee188e38053923962` | static SHA-256 census |

## Element-by-element semantic step audit

The denominator here is the complete 381-line retained `train_semantic_quantized.py`, including the
step-affecting span from `fake_quantize` through terminal save (`:30-376`), plus the called renderer and
curriculum definitions. Every control branch that can affect a stage-07/08 step is accounted for below.

| Element | Actual stage-07 / stage-08 behavior | Audit disposition |
|---|---|---|
| Sample population | `N = 600`; a CPU `torch.Generator` seeded from the stage argument produces a permutation. Batch size is 2; a new permutation is drawn only at an epoch boundary (`train_semantic_quantized.py:27,227-233,260-266`). | Deterministic inside an uninterrupted run. RNG/order/cursor are not checkpointed, so the original trainer cannot resume bit-faithfully. |
| Inputs and target | Input and target caches are separately addressable; both actual stages use the named GT cache. Tokens are loaded as integer class IDs (`:192-206`). | Source-correct. Cache provenance is external to the checkpoint and must remain axis-labelled. |
| Model construction | The model is reconstructed from the inherited top-level `config` architecture fields, then loaded strictly from `state_dict` (`:207-216`). Actual architecture is width 96, four blocks, frame dimension 8, phase 1x1, temporal radius 0. | Runtime-safe for these checkpoints despite stale non-architecture fields. |
| Frozen scorer | SegNet is loaded in eval mode, and every scorer parameter has gradients disabled (`:222-225`). Gradients still flow through SegNet to the rendered frame. | Correct for semantic descent. |
| Optimizer and schedule | AdamW, zero weight decay; cosine LR over the full stage with floor 1% of initial LR (`:227-230`). Stage 07 uses `2e-5` for 12,000 steps; stage 08 uses `2e-7` for 6,000 steps. | Current values live in nested `result.config`, not top-level `config`. |
| Forward selection | Float forward is used only while `step <= float_warmup_steps` (`:272-279`). Both retained stages set warmup to zero, so **0/12,000** and **0/6,000** steps use the float path; all **18,000/18,000** steps use QAT. | Closed: there is no accidental float warmup in the retained two-stage QAT leg. |
| Round trip | QAT produces a 384x512 RGB frame, bilinear-upsamples to 874x1164, applies clamp-plus-uint8-round STE, then bilinear-downsamples to 384x512 (`:57-64,273-280`). | Same generated-frame operator shape as deployment plus evaluator preprocessing. |
| Objective | Float warmup would use CE; otherwise `curriculum_loss` receives current QAT index (`:281-289`). No master cache was supplied, so distillation is **0/18,000** steps (`:290-299`). | The retained stage is scorer-only, not RGB-reconstruction-trained. |
| Stage-07 curriculum | With zero warmup, `i = 0..11999`, progress `i/11999`: CE on **6,000/12,000** steps (`i=0..5999`), softplus margin on **4,200/12,000** (`i=6000..10199`), expected-flip on **1,800/12,000** (`i=10200..11999`). | Exact source-derived counts. |
| Stage-08 curriculum | `ce_fraction=0`, `softplus_fraction=-999`: CE **0/6,000**, softplus **0/6,000**, expected-flip **6,000/6,000**. The name `fixedtau05` is approximate: source gives tau 0.0501 on the first step and 0.0500 on the last, a 0.2% span. | Naming precision issue only; no evidence of material mechanism mismatch. |
| Backward/update | Zero gradients, backward through renderer and frozen scorer, apply any fixed-zero gradient mask, clip global gradient norm at 2, AdamW step, re-zero fixed entries, scheduler step (`:300-312`). Neither actual stage enabled the fixed-zero flag, so the mask is empty for **18,000/18,000** steps. | Complete. No AMP/autocast branch exists in this trainer. |
| Selection | Baseline full-n600 quantized evaluation seeds `best_state`; every 250 steps and final step run another n600 quantized evaluation. Candidate key is d_seg only because no master cache exists (`:234-257,314-354`). | Stage 07 has 49 records (baseline + 48); stage 08 has 25 (baseline + 24). States between evaluations cannot win. |
| Terminal state | The selected full-precision master is reloaded, quantized-evaluated, written to JSON, and saved with `quant_bits`, d_seg, history, and nested current args (`:356-376`). | Deployment later quantizes this selected master. Top-level config is stale; save is terminal-only and non-atomic. |
| EMA | No EMA construction, update, shadow selection, or save site exists in the complete QAT trainer. | P0 contract gap for any future shipping run through this file; best-checkpoint selection is not EMA. |

## Fake quantization and deployed object equivalence

### QAT object

`fake_quantize` (`train_semantic_quantized.py:30-46`) applies these rules to every named parameter through
`torch.func.functional_call` (`:49-54`):

- Rank less than two: round the value to fp16 and use a straight-through replacement.
- Rank two or greater: use signed `bits=4`, `limit=7`, hence 15 reachable codes `[-7, 7]`; `-8` is
  intentionally unused.
- Embedding weights: reduce over every dimension except the last, so there is one scale per embedding
  feature column.
- Other matrices and convolutions: reduce over dimensions 1 onward, so there is one scale per output row
  or output channel.
- Clamp the absolute-max scale floor at `1e-8`, divide by 7, round the scale to fp16, round and clamp the
  normalized values, then multiply codes by the fp16 scale. Scale and rounding decisions are detached;
  gradients use the straight-through source.

### Pack and receive object

The packer implements the same axes, fp16 scales, `limit=7`, and int4 codes at
`pack_semantic_pose.py:97-123`; its unpacker restores those values at `:126-154`. The submitted receiver
does the same at `inflate.py:171-196`, constructs the same architecture, and writes renderer output as
the last frame after bilinear camera expansion and uint8 rounding at `inflate.py:610-635`.

There are no running-stat buffers to diverge at deployment: GroupNorm has no BatchNorm-style running
state, coordinates are generated, and all persistent learned values are named parameters. QAT recomputes
the same quantized values per call, whereas deployment reconstructs them once, but the represented tensor
is the same. SG2's pack/parse result closes the remaining byte identity: the stage-08 semantic blob is the
archive blob.

**Verdict scope:** source and exact-byte object equivalence for this stage-08 checkpoint, packer, and
receiver. This is not a new cross-device floating-kernel parity measurement.

## Renderer architecture and independent census

For actual width 96, blocks 4, frame dimension 8, phase 1x1, temporal radius 0:

- token embedding: `5 x 96 = 480` parameters;
- pair embedding: `600 x 8 = 4,800`;
- coordinate mix: `(96 token + 4 coordinate) -> 96`, `1x1`: `9,696` including bias;
- each of four blocks: depthwise `3x3` (96 groups), pointwise `1x1`, GroupNorm with 12 groups, and
  `8 -> 192` FiLM: `12,192`, hence `48,768` total;
- RGB head: `96 -> 3`, `3x3`: `2,595`.

Total: **66,339 parameters**, matching the prior receipt. The four block dilations are `[1, 1, 2, 4]`.
Each block executes depthwise convolution, pointwise convolution, GroupNorm, FiLM scale/shift, GELU,
and residual addition (`semantic_renderer_oracle.py:60-76`). The complete forward embeds class tokens,
adds `x`, `y`, `x^2`, and `y^2`, applies the coordinate projection, four blocks, GELU, RGB head, sigmoid,
and `x255` (`:79-161`).

## Independent FLOP derivation and correction

Convention retained from the throughput ledger: one multiply-accumulate is two FLOPs; backward is
approximated as twice forward, so forward-plus-backward is three times forward. Bias, GroupNorm, GELU,
sigmoid, elementwise FiLM/residual, coordinate construction, quantization bookkeeping, and interpolation
are excluded. The result is therefore a **conv/linear lower bound**, not a measured exact operation count.
Let `P = 384 x 512 = 196,608` pixels.

| Component | Formula | Forward FLOPs / image |
|---|---:|---:|
| Coordinate mix | `2 * P * 100 * 96` | 3,774,873,600 |
| One depthwise 3x3 | `2 * P * 96 * 9` | 339,738,624 |
| One pointwise 1x1 | `2 * P * 96 * 96` | 3,623,878,656 |
| One FiLM linear | `2 * 8 * 192` | 3,072 |
| Four blocks | `4 * (depthwise + pointwise + FiLM)` | 15,854,481,408 |
| RGB head | `2 * P * 96 * 3 * 9` | 1,019,215,872 |
| **Full renderer** | coordinate mix + four blocks + head | **20,648,570,880** |

Consequences under the same heuristic:

- full forward-plus-backward: **61.9457 GFLOP/image**;
- batch two: **123.8914 GFLOP/step**;
- prior measured renderer median: **0.1205976045 s/step** `[macOS-Metal advisory]`;
- achieved lower-bound throughput: **1,027.31 GFLOP/s**;
- prior measured dense fp32 ceiling: **14,471.765 GFLOP/s**;
- lower-bound ceiling utilization: **7.0987%**.

The existing 15.9/47.6/95 GFLOP chain is arithmetically sound **for the four TokenBlocks alone**. It
omits the 3.775-GFLOP coordinate projection and 1.019-GFLOP RGB head, so `5.5%` cannot be labelled as the
full-renderer ratio. The corrected ratio is still a lower bound because the expensive GroupNorm and
elementwise passes are deliberately assigned zero FLOPs. Nothing here changes the measured component
split: renderer 75.4009%, SegNet-by-subtraction 23.3044%, R-chain 1.2947%. Nor does it reverse the operator
receipt showing GroupNorm/GELU traffic as the dominant shape.

**Verdict scope:** reporting arithmetic in `THROUGHPUT_ROOT_CAUSE.md`; no new wall-clock measurement and
no claim that dense GEMM peak is an attainable ceiling for this bandwidth-bound graph.

## `evaluate_all` versus `evaluate_rgb`

| Surface | What it computes | Contest relationship | Actual retained-stage role |
|---|---|---|---|
| `evaluate_all` (`train_semantic_quantized.py:90-108`) | Across exactly 600 pairs, QAT-render through camera round trip, frozen SegNet argmax, then pooled mismatched pixels / target pixels. | Same compressed-frame rendering and argmax shape as the contest semantic side, but only if the target-token cache is built on the same decoder/hardware axis. It omits pose and rate and is not an archive score. | Baseline plus every-250-step selection; 49 stage-07 and 25 stage-08 records. |
| `evaluate_rgb` (`:111-131`) | Mean squared RGB error after dividing exact-path output and resized master targets by 255. | No term in `upstream/evaluate.py`; it is a distillation/selection proxy only. | Never invoked because neither actual QAT stage supplied `--master-cache`; it affects **0/74** evaluation records and **0/18,000** training steps. |

The compressed side of `upstream/evaluate.py` reads raw inflated RGB through `TensorVideoDataset`
(`upstream/evaluate.py:67-80`). The ground-truth side is decoder-dependent: DALI on the contest-CUDA
authority lane versus the local AV path otherwise. Stage 08 records d_seg 0.0002763705783420139 against
its named cache; published contest-CUDA rows differ. That difference is bounded to cache/decoder/device
axis at this audit level and is not assigned to a mechanism without a scorer run.

## Checkpoint lineage and stale-config chain

Static metadata extraction, without tensor materialization, gives:

| Checkpoint | Current stage facts in `result.config` | Top-level `config` actually stored |
|---|---|---|
| Stage 07 | int4, 12,000 steps, batch 2, eval batch 8, eval every 250, LR `2e-5`, CE 0.5, softplus 0.85, seed 20260715, init stage 06 | inherited hybrid rooted at stage 02: width 96, **blocks 4** from expansion, but 3,000 steps, LR `1e-3`, CE 0.8, softplus 0.95, `amp=true`, 12 pairs, old save/out paths |
| Stage 08 | int4, 6,000 steps, batch 2, eval batch 8, eval every 250, LR `2e-7`, CE 0, softplus -999, seed 20260716, init stage 07 | the same inherited hybrid, byte-for-byte in the audited fields |

The cause is explicit at `train_semantic_quantized.py:208,361-376`: `config` is read from the init
checkpoint, current CLI values are written under `result["config"]`, but the save writes the inherited
`config` back at top level. The chain is:

1. stage 02 creates current metadata;
2. stage 03 copies it unchanged;
3. stage 04 copies it unchanged;
4. stage 05 copies and changes only `blocks: 2 -> 4`;
5. stage 06 copies that hybrid unchanged;
6. stage 07 copies it unchanged;
7. stage 08 copies it unchanged.

Thus final top-level metadata is **six descendant transformations stale** relative to its stage-02 root,
with one intervening architecture mutation. The strict state-dict load makes the retained architecture
fields self-consistent, but schedule, seed, precision, and paths are false for stages 07/08.

### F4 blast-radius census

The bounded census searched the immutable intake `code/`, current `src/tac/pr130_lift/`, and the active
PR130 semantic experiment/profile entry points. Denominator: **19 direct reads of an original-style
checkpoint's top-level `config`**.

| Consumer class | Sites | Effect today |
|---|---:|---|
| Intake repository | 7/19 | Four architecture-only readers; three checkpoint propagators (`expand_semantic_checkpoint`, `train_semantic_full`, `train_semantic_quantized`). |
| Lifted source/pose paths | 7/19 | Five architecture-only readers; two checkpoint propagators (lifted float and QAT trainers). |
| Active mx1 / wc3 experiment paths | 5/19 | Five architecture-only readers. `MlxSemanticConfig.from_pr130_checkpoint_config` consumes only width, blocks, frame dimension, phase, and temporal radius; live training overwrites bits/LR/steps/curriculum from arguments. |
| **Total** | **19/19** | **14 architecture-only; 5 stale-config propagators; 0 current sites use stale schedule/AMP/path fields to control execution.** |

Two additional `config` accesses in mx1 (`:1605,1626`) compare and copy normalized MLX checkpoints during
tail averaging; they do not directly read the original PR130 top-level dictionary and are excluded from
the 19-site denominator. Including them, the broader scoped `config`-access census is 21 sites.

**Current consequence:** I did not find runtime schedule corruption in this 19-site scope. The live harm
is provenance falsification and recurrence: the five propagators preserve bad facts, while external
receipts or future consumers can reasonably mistake top-level `config` for the producing stage. This is
already demonstrated by the final checkpoint claiming `amp=true` even though the QAT trainer has no AMP
path.

**Falsifier/cure:** replace the ambiguous dictionary with versioned `architecture_config`,
`producing_stage_config`, and immutable `parent_checkpoint_sha256`; migrate all 19 direct readers; reject
shape/config disagreement; and demonstrate load-pack-render identity on the existing stage-08 checkpoint.
That would close F4 without retraining.

## Ranked findings, consequences, and falsifiers

### RR2-F1 — P0 HIGH: terminal-only, non-resumable training custody

**Evidence:** one `torch.save` after the loop (`train_semantic_quantized.py:356-376`); no resume argument,
periodic save, atomic temp-plus-rename, optimizer state, scheduler state, RNG generator state, permutation,
cursor, current state, best-state checkpoint, or EMA shadow is persisted. Denominators are **0/12,000**
periodic saves for stage 07 and **0/6,000** for stage 08. The historical chain preserves **2/2** distinct
terminal selected-model artifacts, but **0/2** is a resume-complete stage-state checkpoint.

**Consequence:** a crash loses the entire invocation, and the selected best can disappear even after a
successful evaluation because it lives only in RAM. Re-running from the init checkpoint is deterministic
replay, not resume. Any future launch through this retained file would violate the P0 resumability and
per-stage-checkpoint contract.

**Verdict scope:** this PyTorch QAT formulation and checkpoint schema. It does not kill QAT, PR130's
renderer, or the separate MLX port's explicit resume formulation.

**Falsifier:** a deterministic interruption test showing an atomically written checkpoint contains model,
EMA shadow, optimizer, scheduler, generator/order/cursor, step/phase, best selection, full config and input
hashes, and that `--resume-from` reaches byte-identical final packed semantic bytes versus uninterrupted
execution.

### RR2-F2 / ancestor F4 — P1 MEDIUM: six-hop stale metadata with five live propagators

**Evidence:** lineage and 19-site census above. The final top-level dictionary reports stage-02 schedule
and `amp=true`; nested stage-08 config reports the actual stage-08 schedule. Five of 19 direct readers copy
the inherited dictionary into a new checkpoint.

**Consequence:** current rendering is protected by architecture-only reads and strict state loading, but
receipts, resumption policy, or future training code can consume false producing-stage facts. The format
cannot answer which config is authoritative without source-specific knowledge.

**Verdict scope:** original-style checkpoint metadata and its 19 current direct consumers; not the tensor
payload or archived semantic output.

**Falsifier:** the typed metadata migration and 19-reader closure described in the F4 cure, plus a census
showing zero unversioned top-level reads and zero schedule facts inherited from a parent stage.

### RR2-F3 — P1 MEDIUM: full-renderer FLOPs understated by 4.7941 GFLOP/image

**Evidence:** independent component formula above. The existing 15.8545-GFLOP figure equals exactly the
four blocks and excludes coordinate mix plus head.

**Consequence:** the published 5.5% is a block-stack ratio labelled as renderer utilization. Correcting
the same lower-bound convention gives 7.10%. This does not make the training fast, invalidate the
120.60-ms measurement, or overturn the memory-traffic diagnosis.

**Verdict scope:** `THROUGHPUT_ROOT_CAUSE.md` arithmetic and wording, not its measured receipt.

**Falsifier:** a source-matched operation profiler that counts the complete renderer forward/backward and
states whether interpolation and QAT reparameterization are included. Alternatively, relabel 15.8545 as
TokenBlock-stack-only and stop using it for whole-renderer utilization.

### RR2-F4 — P2 LOW: no EMA path

**Evidence:** zero EMA sites in the complete QAT trainer; selection is a discrete every-250-step best
checkpoint.

**Consequence:** the shipped stage-08 object is not an EMA shadow. This is a binding future-launch contract
gap, but this static audit found no evidence that EMA would have improved the historical PR130 row.

**Verdict scope:** future launches through this trainer; not a retrospective rejection of the official
PR130 archive.

**Falsifier:** an EMA-equipped resume schema and matched packed-byte A/B, or an explicit governing
directive that supersedes the EMA requirement for this exact best-state selection formulation.

### RR2-F5 — P3 INFORMATIONAL: `fixedtau05` is nearly, not exactly, fixed

**Evidence:** stage-08 curriculum arithmetic gives 0.0501 to 0.0500 over 6,000 expected-flip steps.

**Consequence:** the filename is shorthand with a 0.2% tau span; no observed quality or reproducibility
claim depends on exact constancy.

**Verdict scope:** stage-08 naming only.

**Falsifier:** none needed unless a future analysis treats tau as bitwise constant; then record the exact
schedule or set both endpoints explicitly.

## What was not checked, and why

- **No new scorer or archive evaluation.** The charter permits an optimal-form static audit without Metal;
  it does not authorize a heavy launch, and official CUDA authority is unavailable on this host. All d_seg
  and score numbers are prior receipts with their axes preserved.
- **No new QAT-versus-unpacked tensor execution.** SG2 already closed the deployed-blob identity and the
  packer/source formulas are identical. Re-running that settled proof would add no decision information.
- **No cross-device bit-parity claim.** Bilinear interpolation and convolution kernels can differ by host;
  source equivalence does not prove CUDA/CPU/Metal numeric identity.
- **No exact GroupNorm/GELU FLOP count.** The throughput ledger's convention assigns them zero arithmetic;
  this audit retained that convention to isolate its denominator error. The result is explicitly a lower
  bound.
- **No semantic counterfactual rate rebuild.** The charter supplies the 36,580-byte compressed marginal;
  SG2 separately notes a 40,252-byte raw semantic section inside a joint LZMA stream. Those are different
  denominators and are not substituted for each other.

## RECALL EVIDENCE

Before deciding findings, I searched `.omx/research`, the canonical research index/DAG surfaces,
`.omx/state/probe_outcomes.jsonl`, the authoritative intake driver and code, current PR130 lift paths, and
active mx1/wc3 experiment entry points for `PR130`, `CPR1`, `semantic renderer`, `QAT`, `quantized_exact_seg`,
`fixedtau05`, `stage08`, `F4`, `config`, `resume`, `EMA`, `15.9`, `5.5`, and the archive/checkpoint hashes.
I also queried the canonical-equation registry for a PR130-specific semantic equation; none was found in
that bounded registry scope.

The material recalled surfaces were:

- `ddm_sg2_20260809T122848Z/SG2_FINDINGS.md` and `RECEIPT.json`, which already closed the false
  float-versus-QAT evaluator premise, exhaustively traced all 49 intake stages, and proved the stage-08
  packed semantic blob equals the archive blob;
- `OFF_THE_SHELF_VS_PORTED.md`, `PR130_REPRODUCED_HERE.md`, and `THROUGHPUT_ROOT_CAUSE.md`, which supplied
  the authority row, semantic marginal, and measured Metal timing receipts;
- `ddm_mx1_20260806/PARITY.md` and the current mx1/wc3 sources, which exposed current lifted config readers
  and prevented treating the stale top-level schedule as an already-live MLX schedule bug;
- `LOCAL_TRAINING_AUDIT.md`, whose unreceipted carrier result is explicitly withdrawn by the reviewed
  ledger and was not reused here.

Recall changed the work in two ways: it stopped a duplicate search for a hidden semantic closer already
closed by SG2, and it redirected the audit to the unresolved element-level obligations—training custody,
metadata lineage/consumer blast radius, evaluator meaning, and independent full-renderer arithmetic.

## Disposition and handoff

- **QAT-to-receiver equivalence: FOLDED.** Reopen only if the checkpoint, quantizer axes/code range,
  packer, receiver, or archive semantic SHA changes.
- **RR2-F1 resumability/EMA: QUEUED-WITH-A-FIRE-ORDER (CURE-REQUIRED).** Fire before any future PyTorch QAT
  launch. The historical archive remains valid; the trainer is not launch-admissible under the current
  operating contract.
- **RR2-F2 / ancestor F4 stale config: QUEUED-WITH-A-FIRE-ORDER (CURE-REQUIRED).** Fire before a new checkpoint schema is
  treated as provenance authority. Current runtime risk is bounded by the 19-reader census; the five
  propagators are the first migration targets.
- **RR2-F3 throughput denominator: QUEUED-WITH-A-FIRE-ORDER (AMEND-REQUIRED).** Fire on the next ledger touch
  and before a throughput lever is ranked using the 5.5% figure. Keep the measured wall-clock receipt and
  bandwidth conclusion.
- **RR2-F5 tau naming: FOLDED.** Reopen only if exact constancy becomes load-bearing.

This arm produced no archive, no exact evaluation, and no pointer movement. Own-vehicle frontier remains
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER / CURE-REQUIRED** — owner: MAIN semantic-port owner; consumer store:
  `.omx/state/main_hot_state.md` PR130 task row, with implementation in
  `src/tac/pr130_lift/lifted/train_semantic_quantized.py`; fire trigger:
  before any non-disposable PyTorch semantic-QAT launch. Add atomic periodic and per-stage checkpoints,
  `--resume-from`, complete optimizer/scheduler/RNG/order/cursor/best/current state, config/input hashes,
  and EMA shadow, then prove interrupted-versus-uninterrupted packed-byte identity.
- **QUEUED-WITH-A-FIRE-ORDER / CURE-REQUIRED** — owner: MAIN PR130 checkpoint-schema owner; consumer store:
  `.omx/research/ddm_pr130_reproduce_20260809/OFF_THE_SHELF_VS_PORTED.md`; fire trigger: before any new
  PR130-derived checkpoint is used as provenance authority or training input. Introduce typed architecture,
  producing-stage, and parent-lineage metadata; migrate all 19 direct readers and close the five
  propagators with a zero-unversioned-reader census.
- **QUEUED-WITH-A-FIRE-ORDER / AMEND-REQUIRED** — owner: MAIN throughput-ledger owner; consumer store:
  `.omx/research/ddm_pr130_reproduce_20260809/THROUGHPUT_ROOT_CAUSE.md`; fire trigger: before the next
  semantic-throughput optimization decision or any reuse of the 5.5% claim. Relabel 15.8545 GFLOP as the
  four-block stack or replace it with the 20.6486-GFLOP full-renderer lower bound and 7.10% ratio.

## LIVE-HYPOTHESES

- A fused renderer implementation remains plausible because prior measured operator receipts put most
  block time in GroupNorm/GELU and QAT reparameterization traffic, while the corrected arithmetic still
  reaches only a 7.10% conv/linear lower-bound fraction of dense peak. The lead requires a same-object
  argmax and packed-byte gate; it is a throughput hypothesis, not a score hypothesis.
- Separating architecture metadata from producing-stage metadata should be a zero-score, zero-render
  migration because all 14 current runtime consumers read architecture only and strict state loading
  already fixes tensor shapes. A 19-reader migration plus old-checkpoint adapter can test this without
  retraining.
- EMA may improve tail stability because stage selection observes only every 250th state, but this is
  untested on PR130 and must not be assumed to beat the historical best-state checkpoint. Only a matched
  packed-byte A/B can decide it.

## DEAD-ENDS

- Hidden post-render semantic closer: closed by SG2's 49-stage trace and exact stage-08-blob/archive match;
  no successor should search stages 09-49 again unless the named bytes or driver change.
- Float checkpoint evaluation as an explanation of deployed d_seg: closed as the wrong object; the archive
  uses int4 QAT/dequantized parameters, not the stored full-precision master forward.
- QAT-versus-packer quantization mismatch: closed in this source/checkpoint scope; both use fp16 per-axis
  scales and the same `[-7, 7]` int4 codes, and the receiver restores that object.
- `evaluate_rgb` as a contest semantic metric or retained-stage selector: closed; it is normalized RGB MSE,
  has no contest term, and was absent for all 18,000 retained QAT steps and all 74 evaluation records.
- Stale top-level config as a current runtime schedule bug: closed in the 19 direct-reader census; no site
  consumes its stale LR/steps/AMP/curriculum to control execution. Keep the provenance and propagation bug.
- The 5.5% number as a full-renderer utilization fact: closed; it prices only the four TokenBlocks. The
  corrected same-convention lower-bound ratio is 7.10%, without overturning the bandwidth diagnosis.
