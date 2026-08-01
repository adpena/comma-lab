# G22 — ep725 lossless-xcodec frozen-receiver n600 equality replay

Date: 2026-07-26  
Lane: `lane_g22_ep725_xcodec_n600_equality_replay_20260726`  
Mode: `research_only=true`; local build and bounded exact-byte proof only  
Authority: `[macOS-CPU frozen receiver structural proof]`  
Status: **READY FOR ROOT-REVIEWED FULL n600 REPLAY; FULL REPLAY NOT LAUNCHED**

## Purpose and claim boundary

G20 proved an encoder-only lossless xcodec rewrite of the exact ep725 packet:
the selected archive is 81,027 bytes instead of 83,838 bytes and its complete
decoded quantized state is exactly equal to the source state. G17 correctly
kept `G17_EP725_XCODEC_FULL_N600_REPLAY_OWED` open because G20 executed only
pair 0 through the frozen receiver.

G22 supplies the production replay needed to compare the exact frozen receiver
output for all ordered pair IDs `0..599`. It is not a scorer, candidate
materializer, promotion action, or frontier mutation. A successful full replay
can close only the receiver-equality blocker. Contest CPU/CUDA evaluation on
the same archive bytes remains separately owed.

The only lifecycle accepted by this harness is:

```text
counted ArchiveArtifact
  -> free generic decoder runtime
  -> immutable per-chunk DecodeReceipt
  -> realized uint8 output witness
```

No caller-provided digest is accepted as artifact identity. Every digest below
is recomputed after reopening the exact bytes through a descriptor-bound,
non-symlink read.

## Frozen contract

| Typed object | Bytes | SHA-256 | Rate role |
|---|---:|---|---|
| G17 specification | — | `f315c8c0ad3708394e96cbbf40de9bb6af7d6072989bb28ea38a226f5354953b` | evidence only |
| G20 specification | — | `5388f47daaa0b9dfa7510c12ae56a73f704375068fc1d9cf29410fa746b1d5ca` | evidence only |
| G20 receipt | 5,680 | `02ccb8a6209c79651b64fa93b15aa1ed6155b03d9709f5f18b4ff98edfe25c8c` | encoder-only evidence |
| source `archive.zip` | 83,838 | `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3` | counted source control |
| source reopened `0.bin` | 84,536 | `f0c3e648f00f52e48c7be98997fb7dd57c2e5a607ed385846931af68f88cc78c` | counted video-specific state |
| selected `archive.zip` | **81,027** | `8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8` | **counted selected artifact** |
| selected reopened `0.bin` | 81,738 | `4789bf6b5f15272cc5f8a573f25137a9daf7e21755e81aa48a8fba84947b5634` | counted video-specific state |
| decoded state, both arms | — | `5485d0d94c5c834e059837e74ae5320fe9d2b526604c47008a6bfdb74144adf6` | encoder-only equality proof |
| frozen `inflate.py` | 56,814 | `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224` | generic decoder; **0 counted rate bytes**, custody-critical |
| competitive-target recomposition module | — | `502b2c77d37bce0767fb3e764aad57ac154b5bafd8bc5ca5ec12a3eb690c1994` | generic authority-context code; 0 counted rate bytes |
| G20 xcodec module | 23,743 | `7a54d13fc1fc98916997b655007ef7c5e66085f1cba3bfa3d0de28978c1b45de` | encoder-only evidence |
| G20 materializer | 11,701 | `b1b31baeef79f662ae4108379282f3a5fe8ebb76c380996aaf68506d47b16e86` | encoder-only evidence |

The runtime is free under the contest source boundary because it is generic
decoder code. That does not make it optional or weak evidence: its exact bytes
are runtime-critical and are revalidated at run start, in every worker, and at
run end. The selected archive/member/state are video-specific and remain the
counted payload. The G20 search/module/materializer are encoder-only replay
evidence and never become decoder payload. The raw output is a realized witness,
not an archive or rate-bearing object.

## Executable DAG and invariants

```text
explicit reviewed command
  -> exact frozen-contract reopen/hash gate
  -> reopen exact canonical pointer bytes; recompute target from constituents
  -> preserve content-addressed historical pointer artifact + semantic target
  -> SSD waterfall + free-space reserve gate
  -> write-once custody copies of runtime/source member/selected member
  -> private receiver inspection on both members
  -> assert identical metadata and n_pairs == 600
  -> real preallocation of both raw witnesses
  -> for each immutable ordered chunk:
       source private _setup/_render_pair
       selected private _setup/_render_pair
       fsync both ranges
       direct byte comparison of the entire range
       observe/preserve pointer artifact and recomputed semantic target
       atomic write-once range checkpoint
  -> revalidate every completed checkpoint and full raw hashes
  -> revalidate frozen contract + harness + custody copies
  -> preserve end pointer artifact; classify artifact vs semantic target change
  -> set rebase_required_before_admission on semantic change only
     (never invalidate output-byte equality)
  -> write immutable pre-cleanup DecodeReceipt
  -> write cleanup certificate
  -> re-hash each surviving certified raw immediately before unlink
  -> remove success-only raw scratch
  -> write cleanup-complete successor
  -> write durable final receipt
```

For pair `p`, receiver frame IDs are `(2p, 2p+1)`. With
`F = 874*1164*3 = 3,052,008` uint8 bytes per frame, chunk pair interval
`[a,b)` binds exactly:

```text
pair_ids   = [a, a+1, ..., b-1]
frame_ids  = [2a, 2a+1, ..., 2b-1]
byte_range = [2aF, 2bF)
```

Each checkpoint records these exact IDs and offsets, both range SHA-256 values,
the direct byte-equality verdict, worker receipts, elapsed time, and resource
usage. A resume accepts only a contiguous checkpoint prefix, re-hashes and
directly re-compares every completed range, and starts at the first uncommitted
chunk. A crash while rendering a chunk cannot mark that chunk complete.

The full PairPopulation is exactly the ordered list `0..599`, producing 1,200
frames and **3,662,409,600 bytes per arm**. There is no subset-to-n600
extrapolation.

## Fail-closed gates

The harness refuses:

- execution without `--execute-reviewed`;
- a 600-pair launch without the additional `--confirm-full-n600`;
- any nonzero bounded start (the frozen receiver writes global pair offsets);
- any full population other than exact ordered IDs `0..599`;
- a non-SSD run directory outside the configured storage waterfall;
- insufficient free bytes for both missing raw allocations plus the declared
  reserve;
- sparse truncate-only allocation on hosts without `posix_fallocate` (macOS
  receives explicit zero-write physical allocation);
- symlinks, special files, short reads, archive/member shape changes, any frozen
  hash drift, FP32 receiver mode, or worker thread-environment drift;
- receiver metadata disagreement, partial files, wrong-size output files,
  checkpoint gaps, checkpoint/config/hash mismatches, or any output byte
  mismatch;
- harness/runtime/member/module/materializer/spec/receipt/archive drift during
  the run;
- cleanup before a durable, machine-readable rebuild certificate exists.

Pointer artifact movement is evidence, not an equality failure. A metadata-only
refresh sets `pointer_artifact_changed=true` and
`competitive_target_changed=false`. A score/source/submission/axis/selection
change additionally sets `competitive_target_changed=true` and
`rebase_required_before_admission=true`. Neither case changes
`decode_equality_invalidated=false`; admission work must rebase separately.
Completed receipts validate their preserved historical start/end bytes and do
not become invalid merely because the later live pointer refreshes.

There is no score, evaluator, candidate, submission, dispatch, or pointer-write
option in the CLI.

## Resumability, storage, and cleanup

`--resume-from` is the mandatory run-state root. The run manifest, exact input
copies, inspection plans, worker plans, per-chunk checkpoints, pre-cleanup
receipt, cleanup certificate, and cleanup-complete successor are write-once.
All large raw outputs live under the SSD run root. Both raw files are physically
preallocated before rendering; the source and selected workers write only
disjoint pair ranges through the frozen `_render_pair` implementation.

Raw witnesses are deleted only after every range and both whole-file hashes
agree and after the pre-cleanup receipt plus cleanup certificate have been
fsynced. The certificate preserves original paths, bytes, SHA-256 values,
rebuild argv, tool/runtime/archive/member/state custody, run-manifest hash, and
all chunk-receipt hashes. All checkpoints and small custody inputs remain on
SSD. Any failure before exact equality leaves raw bytes in place for diagnosis.

## Implementation and tests

- Harness: `tools/replay_ep725_xcodec_n600_equality.py`
  - final tested SHA-256: `9eb414eeed9313dbd49fffd7b3c7b78ad7b07f7e9b740022e86338bbd54337a2`
- Focused tests: `tools/tests/test_replay_ep725_xcodec_n600_equality.py`
  - SHA-256: `67bdc6a656935bf57049c5e5ce42e1c30d5195ca11dc73d824e35cb97475d0b3`
  - result: `32 passed in 0.15s`
- Static checks: `ruff check` clean; `py_compile` clean.

The unit tests are structural only. They prove write-once behavior, exact ZIP
member reopening, range geometry, direct mismatch localization, physical
allocation, SSD/refusal gates, checkpoint-prefix and checkpoint-byte validation,
n600 confirmation, certified cleanup order, metadata-only pointer refresh,
semantic-target rebase signaling, completed-receipt independence from later
live pointer bytes, and lack of mutation options. They
do not stand in for real receiver execution.

## Real bounded exact-byte proof

The final-harness pair-0 replay was executed at:

```text
/Volumes/VertigoDataTier/pact/g22_ep725_xcodec_n600_equality_replay_20260726/bounded_pair0_v4
```

Durable final receipt:

```text
.omx/research/original_taskspace_inverse_witness_codec_20260725/g22_ep725_xcodec_n600_equality_replay_20260726/bounded_pair0_decode_receipt_v4.json
```

Receipt SHA-256: `a7e107a08ac854a33650bd84f24c186676b22e165cd7c2dd59c0ee2395db4ca5`.

Measured receipt facts:

- exact source and selected archives/members/runtime were reopened and matched
  the frozen contract;
- private `_setup` and `_render_pair` executed on both real members;
- 2 frames / 6,104,016 bytes per arm were directly compared;
- both raw hashes were exactly
  `22b994567d3db018df29a95c597606053a115c631d98e89b72ec7eeba93666b3`,
  matching G20's independently measured pair-0 receipt;
- wall time was 12.295 seconds; the receipt preserves native resource counters;
- storage preflight observed 385,559,310,336 free bytes, physically allocated
  12,208,032 raw bytes, and preserved a 1 GiB reserve;
- the start/end pointer artifacts and recomputed `0.172` PR130 competitive
  target were preserved separately; neither artifact nor semantic target
  changed during this bounded interval, and the receipt explicitly records
  `decode_equality_invalidated=false`;
- success-only cleanup was certified, both raw scratch files were removed, and
  the immutable run manifest plus chunk checkpoint remain;
- completed-run resume validation passed without recomputing the chunk.

Earlier development smoke receipts remain append-only as v1-v3 evidence. They
are superseded for G22 handoff by the v4 receipt above, which includes final
completed-run drift validation and G14b semantic-frontier continuity.

This is still bounded evidence. `full_n600_receiver_replay_owed=true` remains
the honest state.

## Exact root-reviewed n600 command — not launched

```bash
.venv/bin/python tools/replay_ep725_xcodec_n600_equality.py \
  --resume-from /Volumes/VertigoDataTier/pact/g22_ep725_xcodec_n600_equality_replay_20260726/full_n600 \
  --receipt .omx/research/original_taskspace_inverse_witness_codec_20260725/g22_ep725_xcodec_n600_equality_replay_20260726/full_n600_decode_receipt.json \
  --pair-start 0 \
  --pair-count 600 \
  --chunk-pairs 12 \
  --workers 4 \
  --reserve-bytes 8589934592 \
  --execute-reviewed \
  --confirm-full-n600
```

Expected immutable stage population: 50 chunks, exact ordered pair IDs
`0..599`, 1,200 frames, and 7,324,819,200 physically allocated raw scratch
bytes before the 8 GiB reserve. The command is crash-resumable from the same
`--resume-from` directory and does not recompute committed chunks.

## Triality and pointer-delta honesty

- **DSL leg:** the receipt keeps generic decoder source, counted
  archive/member/state, encoder-only rewrite evidence, DecodeReceipt, output
  witness, cleanup state, and authority status as distinct typed sections.
- **DAG leg:** the executable state machine above is enforced by write-once
  artifacts and per-chunk prefix validation.
- **Equation leg:** exact pair/frame/byte interval equations bind every stage;
  equality is direct over all realized uint8 bytes, never a proxy digest-only
  assertion.

Pointer delta: **none**. Exact score: **not measured**. Candidate claim:
**false**. Promotion eligibility: **false**. Goal progress: the exact frontier
is unmoved, so G22 has not lowered the contest score. The remaining actionable
blocker is the root-reviewed full-n600 replay above, followed separately by
same-byte contest CPU/CUDA evaluation if the full replay is exact.
n600 confirmation, certified cleanup order, metadata-only pointer refresh,
semantic-target rebase signaling, completed-receipt independence from later
live pointer bytes, and lack of mutation options. They
