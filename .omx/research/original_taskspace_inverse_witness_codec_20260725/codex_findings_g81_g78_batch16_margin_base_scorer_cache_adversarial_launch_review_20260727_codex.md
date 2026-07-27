# G81 adversarial launch review: G78 batch-16 margin/base scorer cache

Date: 2026-07-27  
Review lane: `lane_g81_g78_batch16_margin_base_scorer_cache_adversarial_review_20260727`  
Reviewed lane: `lane_g78_batch16_margin_base_scorer_cache_materializer_20260726`  
Authority: read-only launch review; no scorer forward, n600 materialization, candidate, score, promotion, or pointer claim

## Exact verdict

**REFUSE LAUNCH.**

Do not run the governed G78 n600 materialization command and do not close either
G72 custody blocker. The implementation has one P0 scientific-custody failure
and one independently reproduced P1 strict-reopen failure. Both can allow an
aggregate to strictly reopen while the claimed V15 base coordinate or the
stage-to-batch/G51 coordinate is not the one actually proved.

The exact r3 zero-forward preflight itself still reopens:

- file:
  `/Volumes/VertigoDataTier/pact/taskspace_batch16_margin_base_scorer_cache_n600_20260726_r3/00_preflight_receipt.json`;
- bytes: `53,180`;
- file SHA-256:
  `abe70c549dea05188b1774fc0ee816857a54d920d02b8bbb1391bdc343debae2`;
- sealed preflight SHA-256:
  `a5ae7ce2a3482f628374ce3ac86298340c53bca7cf070a69dfdaf950402cbdf4`;
- current output root contains only that preflight receipt, not scorer output.

Preflight validity is necessary but not sufficient for launch: the gaps below
are in what materialization and aggregate reopen prove after the preflight.

## Severity-ranked findings

### P0 — Live V15 camera identity is recorded but never enforced, and a live renderer dependency is outside the sealed closure

The preflight binds 38 expected V15 camera identities, including the exact
`[0,16)` and `[592,600)` batches. The run renders fresh camera bytes at
`tools/materialize_taskspace_batch16_margin_base_scorer_cache_n600.py:718-720`
and records their SHA at lines 734-735. But the core write path only checks that
the supplied camera/scorer-input strings are syntactically valid SHA-256 values
(`src/tac/witness_control/taskspace_batch16_margin_base_scorer_cache_v1.py:559-563`).
Neither `_write_batch` nor `_validate_batch_checkpoint` compares
`v15_camera_sha256` with the corresponding
`preflight.semantic_custody.full_p_camera_identity.checkpoints[*].camera_sha256`.
The batch validator does not validate `v15_camera_sha256` or
`v15_scorer_input_sha256` at all (`...cache_v1.py:455-515`).

This also breaks fail-closed resume. Completed batches rehash source RGB and
target scorer input, then skip `infer()` (`...cache_v1.py:848-873`), so they do
not re-render or rehash the current V15 camera/live-R input.

Worse, `CarrierComposeReceiverV1.render_camera_pairs` lazily imports
`tac.through_r.resolution_chain` and calls
`render_grid_to_camera_uint8`
(`src/tac/optimization/direct_description_carrier_compose.py:2721-2735`).
`_runtime_sources()` seals the direct receiver file but not
`src/tac/through_r/resolution_chain.py`
(`tools/materialize_taskspace_batch16_margin_base_scorer_cache_n600.py:362-389`).
That file is absent from all 81 r3 `sealed_input_files`. The preflight merely
rehashes the old camera-identity checkpoint JSON files; it does not replay the
current receiver against their expected camera digests.

Consequently a renderer dependency can drift after the V15 identity run, G78
can materialize different camera bytes, and both the batch checkpoint and final
aggregate can accept them. That violates the named
`fresh V15 camera -> live R -> frozen SegNet` custody contract and can silently
change every described cell/margin.

Required closure before a new preflight:

1. seal the complete transitive receiver/runtime closure, including
   `resolution_chain.py`;
2. for each global batch, compare the freshly rendered camera SHA exactly with
   the preflight's matching V15 identity checkpoint before any shard is
   published;
3. bind and validate the live-R scorer-input SHA under the same batch identity;
4. on resume, re-render/re-hash camera and live-R input before skipping the
   scorer forward, or establish an equivalently strong immutable external
   identity proof;
5. add refusal tests for wrong live camera SHA, wrong V15 scorer-input SHA, and
   drift in each transitive renderer dependency.

### P1 — “Strict aggregate reopen” does not prove stage bytes or stage metadata came from the validated batch/G51 inputs

The stage writer correctly computes cross-boundary fragments
(`...cache_v1.py:640-660,741-766`) and stores `pair_ids`,
`batch_fragments`, and `g51_y0_y1_stage`
(`...cache_v1.py:770-786`). The strict stage validator does not validate any of
those three fields. It checks the stage range, target slice, flags, and the
self-reported stage-file identities only
(`...cache_v1.py:663-715`). The aggregate loader validates batch receipts and
stage receipts independently but never proves that a stage dense file equals
the specified concatenation/slices of those batch shard files
(`...cache_v1.py:1043-1103`).

Two cheap synthetic probes reproduced the gap without any scorer execution:

1. replace stage 0's `g51_y0_y1_stage` with an unrelated object, recompute the
   unkeyed stage/aggregate hashes, and reopen with the same
   `expected_sha256=sha256_file(aggregate)` pattern used by the producer:
   **accepted**, five stages returned;
2. change one valid-class byte in stage 0 `described_cells.u8`, update the
   stage-file identity and self-hashes while leaving the validated global batch
   shard unchanged:
   **accepted**, with
   `stage_differs_from_batch_prefix=true`.

This is not merely an adversarial-edit concern. The producer's final reopen
uses the just-created aggregate's own current file hash
(`...cache_v1.py:960-969`), so it is tautological as an external identity
anchor. A stage assembly bug that emits internally hash-consistent wrong bytes
would pass the same path. The existing dense-tamper test changes bytes without
updating their self-hashes; it does not exercise this semantic condition.

Required closure:

1. rederive and compare `pair_ids` and every `batch_fragments` row from the
   already validated batch chronology;
2. compare `g51_y0_y1_stage` exactly with the preflight's stage entry;
3. prove each stage dense file equals the exact fragment composition of the
   validated batch shard files, ideally by streaming comparison/hash to retain
   bounded RSS;
4. test semantic-reseal attacks for G51 metadata, fragments, and all three
   dense stage fields.

### P1 — The registered launch gate still names superseded r2, not exact r3

The G78 lane's `strict_preflight` evidence currently points to:

`/Volumes/VertigoDataTier/pact/taskspace_batch16_margin_base_scorer_cache_n600_20260726_r2/00_preflight_receipt.json`

at `.omx/state/lane_registry.json:92303-92305`. The G78 spec explicitly says r2
was superseded after Ruff changed source bytes and that r3 is the exact launch
input (`SPEC_g78_batch16_margin_base_scorer_cache_materializer_20260726.md:137-140`).
No governed launch should rely on the stale gate. After the P0/P1 code fixes,
generate a fresh root and update the lane through `lane_maturity.py`; do not
reuse r2 or r3 because source/runtime custody will have changed.

### P2 — Production source indexing is structurally sequential but not asserted at the runtime boundary

The materializer constructs a one-video `AVVideoDataset` with batch size 16 and
walks it once in the same global range loop, so current sealed code gives the
right 38-batch chronology and final 8. However, the production preparer
discards both yielded `_path` and `_batch_index` and checks only batch length
(`tools/materialize_taskspace_batch16_margin_base_scorer_cache_n600.py:698-709`).
The batch receipt then labels the bytes using caller-supplied `pair_ids`.

This is not an observed misindex under the sealed current upstream iterator, so
it is not independently launch-blocking. It is nevertheless weaker than the
claimed exact indexing proof. Assert the yielded source path and expected
one-based batch index (including batch 38/final 8), and test refusal for a
reordered or mislabeled iterator.

### P2 — Aggregate identity and top-level claims are weaker than the “strict” API name implies

`MarginBaseScorerCacheLoaderV1.open` makes `expected_sha256` optional
(`...cache_v1.py:989-1007`). The producer and `--status` pass a hash computed
from the same file immediately before opening it, not a separately pinned
digest (`...cache_v1.py:963-966`;
`tools/materialize_taskspace_batch16_margin_base_scorer_cache_n600.py:764-770`).
The loader also does not exactly rebind top-level `pair_count`, batch/stage
counts and geometry, `coverage`, `closed_blockers`, or custody copies to the
preflight. These do not alter the mmap arrays returned today, but they can let
machine-readable closure claims diverge from the recursively validated inputs.

Make a pinned aggregate digest mandatory at downstream custody boundaries and
compare all duplicated top-level fields/closure claims exactly with rederived
values.

### P2 — Immutable output has a concurrent-writer TOCTOU window

`_write_atomic` checks `path.exists()` and then calls `os.replace`
(`...cache_v1.py:184-201`). A second writer can create the destination between
those operations, after which `os.replace` overwrites it. Temporary filenames
are PID-specific, but there is no run-root lock or no-replace commit primitive.
The governed launcher limits memory admission but does not itself prove
single-writer ownership of this output root.

Add an output-root lease/lock bound to run ID and preflight SHA, or use a
no-replace atomic publication primitive followed by byte equality validation.

### P2 — Receiver contract label names the legacy surface

The r3 preflight records:

`tac.optimization.direct_description_carrier_compose.CarrierComposeReceiverV1.render_pairs.v9_v13`

while the materializer actually calls `render_camera_pairs` for V15. The full
receiver source hash is present, but the semantic contract label is not the
executed surface. Regenerate the lineage identity with an exact V15
`render_camera_pairs` contract after fixing the live camera binding.

## Checks that passed

- `CLAUDE.md` and `AGENTS.md` were fully processed and are byte-identical:
  SHA-256
  `47d4ac3a38f91a8b8e7dc3061131717d8122bd48ffb204ffb914eb58e687f0c9`.
- `PROGRAM.md`, top current-state project memory, active directives, claim
  ledger, and lane registry were consulted before the verdict.
- Current G78 implementation/config/spec hashes equal the landing receipt.
- The exact r3 `--preflight-only` reopen passed without scorer execution.
- Focused tests: `8 passed in 1.30s`.
- Ruff: `All checks passed!`.
- `git diff --check` over the G78 files passed.
- Winner-margin behavior is correct, including first-class argmax tie selection
  and exact zero margin for a tied nonwinner.
- The core loop preserves global batch size 16 and final partial 8; the
  fragmentation algorithm correctly splits batches across 120-pair boundaries.
- Fresh target cells are compared cellwise with the mmap-backed owned G46
  labels before publishing a new batch.
- The executed scorer surface is camera uint8 -> N,T,C,H,W float32 ->
  `SegNet.preprocess_input` last-frame bilinear 384x512 -> frozen SegNet.
- New margin arrays are normalized to little-endian float32, cells to uint8;
  shapes and file hashes are checked; consumer mmaps are read-only.
- Resume rehashes source RGB and target scorer input and skips already-complete
  scorer forwards.
- False-authority fences consistently remain research-only, encoder-only,
  noncandidate, nonscore, nonpromotion, pointer-unmoved.
- Required free storage is 12 GiB versus about 1.98 GiB of duplicated
  batch-plus-stage dense payload. SSD enforcement and dynamic free-space
  recheck pass. Static batch/stage allocations fit comfortably inside the
  governed 10 GiB RSS cap; no empirical RSS claim was made.
- The documented launch uses `safe_run.py` with system admission, a 10 GiB
  process-group cap, and a one-hour wall cap. No launch was performed in this
  review.

## Launch re-entry gate

Launch can be reconsidered only after all of the following:

1. P0 V15 current-camera and live-R input equality is fail-closed per batch,
   including resume;
2. the complete receiver/runtime closure is sealed;
3. strict stage reopen semantically binds G51, fragments, and stage bytes to
   validated batches;
4. focused adversarial tests cover both failures;
5. a new SSD preflight root is generated from the corrected exact sources;
6. the G78 lane's strict-preflight evidence names that new receipt;
7. a new read-only adversarial review returns `LAUNCH`.

Pointer delta: zero. Both G72 custody blockers remain open.
