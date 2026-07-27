# G78 batch-16 target-margin and V15 base-scorer cache materializer

Date: 2026-07-26  
Lane: `lane_g78_batch16_margin_base_scorer_cache_materializer_20260726`  
Authority: encoder-only macOS-CPU frozen-scorer evidence; no candidate, score,
promotion, or pointer authority

## Outcome

G78 lands the production materializer for the first two open G72 operands:

1. `G72_FRESH_BATCH16_TARGET_MARGIN_CUSTODY_OWED`;
2. `G72_FRESH_V15_CAMERA_R_BATCH16_BASE_SCORER_STAGE_CACHE_OWED`.

The launch-hardened implementation, typed config, strict loader, resume and
semantic-reseal tests, and real zero-forward custody/storage preflight are
complete. The n600 scorer materialization itself was **not launched**.
Therefore neither blocker is closed yet. They close only when the governed run
emits and strictly reopens the complete aggregate receipt.

No archive, decoder, evaluation, score, or pointer was produced. The effective
frontier moves by zero.

## Exact fields

For one exact global source batch, let the frozen five-class SegNet logits after
the upstream live resize be:

`z = SegNet(SegNet.preprocess_input(X))`.

`SegNet.preprocess_input` selects pair frame 1 and applies PyTorch bilinear
interpolation from camera resolution to `384x512` in float32. There is no
intermediate uint8 scorer-plane roundtrip.

The winner cell and winner margin are:

`c*(p,y,x) = argmax_c z[p,c,y,x]`

`m(p,y,x) = z[p,c*,y,x] - max_(c != c*) z[p,c,y,x]`.

The target pass consumes the exact source pairs from G46's source-video and
upstream closure. Its fresh `c*` must equal the already-owned G46 batch-16
target-label bank at every cell. Any mismatch refuses the whole batch before a
checkpoint is published. G78 stores only `m_target` as little-endian float32;
G46 target cells remain mmap-backed at their existing path and are not
regenerated or copied.

The described/base pass consumes:

`fresh V15 archive -> strict CarrierComposeReceiverV1.render_camera_pairs ->
SegNet live R -> frozen SegNet`.

It stores:

- `described_cells`, uint8 `[n,384,512]`;
- `described_margins`, little-endian float32 `[n,384,512]`.

The described margin uses the same winner-minus-best-other definition. The
fresh V15 camera/archive coordinate is bound independently from the target
source coordinate.

Before either a fresh or resumed batch may use a checkpoint, G78 renders the
current V15 camera bytes and requires their exact SHA-256 to equal the owned
camera-identity checkpoint for that same global pair range. It also computes
the current live-R scorer-input SHA before the lazy scorer forward. Completed
batches revalidate source RGB, target scorer input, V15 camera, and V15 live-R
input, then skip only the already-owned scorer forward. A current-camera or
live-R-input mismatch refuses before reuse.

## Batch geometry and five stages

SegNet contains batch-sensitive numerical kernels. The materializer preserves
the exact upstream sequence of 38 global scorer batches:

`[0,16), [16,32), ... [576,592), [592,600)`.

G72 independently requires five immutable stages:

`[0,120), [120,240), [240,360), [360,480), [480,600)`.

Because 120 is not divisible by 16, the implementation does not fake one
geometry as the other. It writes atomic global batch shards first, then
assembles each 120-pair stage from hash-bound fragments. For example, batch
`[112,128)` contributes `[112,120)` to stage 0 and `[120,128)` to stage 1.

Each batch checkpoint binds source RGB bytes, target scorer-input bytes, V15
camera bytes, V15 scorer-input bytes, the owned G46 target slice, and all three
new dense shard hashes. On resume, completed batches rehash source/scorer-input
custody but skip the frozen-scorer forward. Missing batches forward their
complete original 16-pair group; the final batch forwards its exact 8 pairs.

Each stage checkpoint is published with a no-replace hard-link commit plus
file/directory fsync, immutable on resume, and binds its contributing batch
receipt hashes, G46 target slice, existing G51 stage, and three stage-cache
files. The aggregate loader recursively rehashes all 38 batch receipts, all
five stages, the stage digest chain, and the sealed preflight closure. It also
rederives every cross-boundary fragment row, reopens and exactly compares the
owned G51 stage binding, and streaming-hashes the contributing batch slices to
prove that each stage dense file is the exact batch-fragment composition.
Self-consistent resealing cannot substitute unrelated stage bytes or G51
metadata.

## Reused custody

G78 does not rederive operands already owned elsewhere:

- G46 target bank: `[600,384,512] uint8`, SHA-256
  `6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85`;
- G46 compile-ready receipt file SHA-256
  `556c6b20f12ae7f5c6b8a1a2d08c6d2c2e32e7831ab552938c7866f3256dfad1`,
  sealed receipt SHA-256
  `58db7f01674c60f060a46b955fee8c4f777f31f528ebba404e871b26b17972a7`;
- G51 aggregate file SHA-256
  `ae9048dfc24947a6268315590b65da02b56549379e347cbaced25e2e6f67d915`,
  sealed aggregate SHA-256
  `4363827c2aeb613916029d8bacde8aeb4ded961c4d1ca297310a1e53e204619c`;
- G51's five `Y0`/`Y1` stages are recursively reopened and bound but never
  regenerated or copied;
- fresh V15 archive: 133,941 bytes, SHA-256
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`;
- fresh V15 typed config RFC8785 SHA-256
  `5ab13e5fd1de997d0d4018a0ce94a2e84d74c582db18ad280ffe3b805526dd30`;
- all 38 V15 camera-identity checkpoints reopen with digest-chain SHA-256
  `5d502c1eafe0bd6b3a3e8ea323b02a66573f51939e0a21ebde6e592e04141d7c`.

The preflight records 632 distinct sealed input files, including the exact
37,545,489-byte source video, frozen SegNet weights, G46 target bank, G51 stage
files, V15 archive/config/checkpoints, upstream evaluator closure, and a
conservative 558-file transitive local receiver/runtime source closure. The
closure includes `direct_description_carrier_compose.py`,
`through_r/resolution_chain.py`, `contest_eval_contract.py`, and every
statically reachable local dependency/package initializer. Strict preflight
reopen rehashes every file. The executed receiver contract is explicitly
`CarrierComposeReceiverV1.render_camera_pairs.v15`; the ancestor V15 lineage
contract remains separately preserved.

## Storage and cleanup

The repaired zero-forward production preflight passed on
`/Volumes/VertigoDataTier/pact`:

- required free bytes: `12,884,901,888` (12 GiB);
- observed free bytes: `360,839,086,080`;
- output root:
  `/Volumes/VertigoDataTier/pact/taskspace_batch16_margin_base_scorer_cache_n600_20260726_r4`;
- preflight file SHA-256:
  `3a65a8ce9cfe0eaa681b9abb875703aefb4f3b0adff7d65020153259e13134fe`;
- sealed preflight SHA-256:
  `d5a2b4d621af6b62d2fb06e0b051006e1acf02be94f5a158e68513cce2dd989e`.

Only the small preflight receipt exists. No dense batch/stage file has been
written. Completed future shards are preserved. Cold-store or deletion is
blocked until a machine-readable certificate binds original paths, byte
counts, hashes, rebuild argv/environment, destination, and reason.

The earlier `r2` and `r3` roots are preserved as superseded preflight-only
artifacts. The final `r4` preflight was regenerated after the G81 adversarial
review repairs and canonical Ruff formatting; it binds the exact source that
may later be launched. None of these roots contains scorer output.

## Candidate prohibition

Target cells, target margins, described cells, described margins, scorer
inputs, frozen weights, G51 Y0/Y1, and all checkpoints are encoder-only
evidence. They may not enter `archive.zip`, `inflate.py`, or any candidate
payload. Every batch, stage, preflight, and aggregate schema carries this false
authority explicitly. A successful aggregate closes only the two G72 operand
custody blockers; it does not create a candidate or score.

## Verification

Focused verification:

```bash
.venv/bin/ruff check \
  src/tac/witness_control/taskspace_batch16_margin_base_scorer_cache_v1.py \
  src/tac/witness_control/tests/test_taskspace_batch16_margin_base_scorer_cache_v1.py \
  tools/materialize_taskspace_batch16_margin_base_scorer_cache_n600.py \
  tools/tests/test_materialize_taskspace_batch16_margin_base_scorer_cache_n600.py

.venv/bin/pytest -q \
  src/tac/witness_control/tests/test_taskspace_batch16_margin_base_scorer_cache_v1.py \
  tools/tests/test_materialize_taskspace_batch16_margin_base_scorer_cache_n600.py
```

Result: Ruff and format checks passed; focused `13 passed`; 39 cheap adjacent
G46/G51/G72/V15-receiver tests passed (52 total); `py_compile` passed.

The tests cover:

- exact winner-minus-maximum-nonwinner margin, including a zero-margin tie;
- global scorer batches crossing 120-pair stage boundaries;
- five exact stages and strict aggregate reopen;
- resume rehash of source, target input, V15 camera, and live-R input with no
  repeated completed scorer forward;
- owned V15 camera-identity mismatch refusal on resume;
- V15 live-R scorer-input mismatch refusal on resume;
- fresh target argmax mismatch against G46 refusal;
- dense stage-file and target-bank tamper refusal;
- self-consistently resealed wrong G51 stage refusal;
- self-consistently resealed stage bytes divergent from batch shards refusal;
- transitive runtime closure inclusion for the V15 camera renderer;
- governed admission before the heavy CLI run.

## Rebuild and launch

Preflight:

```bash
.venv/bin/python \
  tools/materialize_taskspace_batch16_margin_base_scorer_cache_n600.py \
  .omx/research/configs/taskspace_batch16_margin_base_scorer_cache_n600_20260726.json \
  --preflight-only
```

The exact heavy command is intentionally not run by this landing. When the
operator admits it, use the governed system-total and process-group caps:

```bash
.venv/bin/python tools/safe_run.py \
  --rss-mb 10240 --projected-gib 10.0 \
  --timeout 3600 --label g78_batch16_margin_base_n600 -- \
  .venv/bin/python \
    tools/materialize_taskspace_batch16_margin_base_scorer_cache_n600.py \
    .omx/research/configs/taskspace_batch16_margin_base_scorer_cache_n600_20260726.json \
    --materialize
```

The same command resumes from every committed global batch and preserved
stage. The output is not production-complete until
`MarginBaseScorerCacheLoaderV1.open(...)` reopens the aggregate receipt and all
five stages.

## Triality

### DSL

Closed schemas cover typed config, sealed preflight, global scorer-batch
checkpoint, five-stage checkpoint, aggregate receipt, and strict mmap loader.

### DAG

`G46 source/video + frozen SegNet -> exact global batch16 target logits ->
G46 argmax equality -> target margins`

and

`fresh V15 archive -> strict camera renderer -> live SegNet R -> exact global
batch16 base logits -> described cells/margins`

then

`38 atomic batches -> five fragment-closed 120-pair stages -> strict aggregate
-> G72 proposal compiler`.

G51 Y0/Y1 and G46 labels enter as already-owned immutable custody, not as
recomputed outputs.

### Equations

The only new scientific quantity is the frozen-head winner margin:

`m = z_argmax - max(z_non_argmax) >= 0`.

No threshold, proxy admission, learned quotient, distortion estimate, or score
is introduced.

## Honest remaining debt

The implementation and real preflight are ready, but the aggregate is absent.
Both named G72 custody blockers remain open until the governed n600 run
finishes and the strict loader reopens it. The other four G72 blockers remain
unchanged even after that run.
